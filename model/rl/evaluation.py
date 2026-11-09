"""Evaluation of the cross-embodiment FB-CPR agent and the motion priorities it produces.

Tracking errors follow BFM-Zero: a motion's priority is an exponential function of the earth
mover's distance (EMD) between the joint positions of a rollout and of the motion.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def earth_movers_distance(rollout: np.ndarray, target: np.ndarray) -> float:
    """Return the EMD between equally weighted frame sets ``[T, J]`` of the same length.

    With equal weights and sizes, optimal transport reduces to an assignment (Birkhoff's theorem).
    """
    if rollout.shape != target.shape:
        raise ValueError("rollout and target must have the same shape")
    cost = np.linalg.norm(rollout[:, None] - target[None], axis=-1)
    rows, columns = linear_sum_assignment(cost)
    return float(cost[rows, columns].mean())


def motion_priorities(emd: np.ndarray, minimum: float = 0.5, maximum: float = 2.0, scale: float = 2.0) -> np.ndarray:
    """Return BFM-Zero's sampling weights: ``2 ** (scale * clip(EMD, minimum, maximum))``."""
    return 2.0 ** (scale * np.clip(emd, minimum, maximum))


@torch.no_grad()
def run_tasks(agent, rollouts: list, group, tasks: list[list[dict]]) -> list[list[dict]]:
    """Run per-robot tasks in lockstep with mean actions, without pushes or noise.

    Args:
        agent: The agent.
        rollouts (list): Each robot's ``Rollout``.
        group: Steps the shared simulation.
        tasks (list[list[dict]]): For each robot, tasks with ``frame`` (start), ``latents``
            ``[T, Z]`` and ``steps``.

    Returns:
        list[list[dict]]: For each robot and task, ``q`` ``[T, J]`` and ``fallen``: down at
            the end, or diverged on the way (a diverged world is reset, so it must not
            count as upright).
    """
    results: list[list[dict | None]] = [[None] * len(robot_tasks) for robot_tasks in tasks]
    cursors = [0] * len(tasks)
    for rollout in rollouts:
        rollout.set_evaluating(True)
    while any(cursor < len(robot_tasks) for cursor, robot_tasks in zip(cursors, tasks)):
        batches = []
        for robot, (rollout, robot_tasks) in enumerate(zip(rollouts, tasks)):
            size = rollout.env.num_envs
            chosen = list(range(cursors[robot], min(cursors[robot] + size, len(robot_tasks))))
            cursors[robot] += len(chosen)
            batches.append(chosen)
            frames = [robot_tasks[index]["frame"] for index in chosen]
            filler = frames[-1] if frames else None
            rollout.start(frames + [filler] * (size - len(frames)))
        horizon = max((tasks[robot][index]["steps"] for robot, chosen in enumerate(batches) for index in chosen), default=0)
        logs: list[list[np.ndarray]] = [[] for _ in rollouts]
        failed = [np.zeros(rollout.env.num_envs, dtype=bool) for rollout in rollouts]
        for step in range(horizon):
            for robot, (rollout, chosen) in enumerate(zip(rollouts, batches)):
                count, joints = rollout.env.num_envs, rollout.env.joints
                action = np.zeros((count, joints))
                if chosen:
                    times = [min(step, tasks[robot][index]["steps"] - 1) for index in chosen]
                    z = torch.stack([tasks[robot][index]["latents"][time] for index, time in zip(chosen, times)])
                    action[: len(chosen)] = agent.act(robot, rollout.observation.select(torch.arange(len(chosen))), z, mean=True).cpu().numpy()
                rollout.env.prepare(action)
            group.advance()
            for robot, rollout in enumerate(rollouts):
                next_observation, result = rollout.finish()
                rollout.observation = next_observation
                logs[robot].append(result.state.q.numpy())
                failed[robot] |= result.terminated.numpy()
        for robot, (rollout, chosen) in enumerate(zip(rollouts, batches)):
            if not chosen:
                continue
            fallen = rollout.env.fallen(rollout.env.read_state()) | failed[robot]
            for position, index in enumerate(chosen):
                steps = tasks[robot][index]["steps"]
                q = np.stack([logs[robot][time][position] for time in range(steps)])
                results[robot][index] = {"q": q, "fallen": bool(fallen[position])}
    for rollout in rollouts:
        rollout.set_evaluating(False)
    return results


def evaluate(agent, rollouts: list, group, motions: list, max_motions: int | None = None, horizon: int = 250) -> tuple[dict[str, float], list[np.ndarray]]:
    """Evaluate tracking, goal reaching and getting up on every robot.

    Args:
        agent: The agent.
        rollouts (list): Each robot's ``Rollout``.
        group: Steps the shared simulation.
        motions (list): Each robot's ``ExpertMotions``.
        max_motions (int | None): Motions per robot; all by default.
        horizon (int): Steps of the goal and getting-up tasks.

    Returns:
        tuple: Metrics by name, and each robot's per-motion EMD for prioritization.
    """
    tracking, goals, recovery = [], [], []
    for robot, (rollout, expert) in enumerate(zip(rollouts, motions)):
        chosen = range(min(max_motions or len(expert), len(expert)))
        robot_tracking, robot_goals, robot_recovery = [], [], []
        for motion in chosen:
            frames, view = expert.frames[motion], expert.states[motion]
            frame_count = len(frames["q"])
            first = {key: value[0] for key, value in frames.items()}
            robot_tracking.append({"frame": first, "latents": agent.tracking_latents(robot, view).cpu(), "steps": frame_count - 1, "motion": motion})
            goal = agent.goal_latent(robot, view.select(torch.tensor([frame_count - 1]))).cpu().expand(horizon, -1)
            robot_goals.append({"frame": first, "latents": goal, "steps": horizon, "motion": motion})
            if rollout.env.floating:
                lying = rollout.env._lie_down(first)
                robot_recovery.append({"frame": lying, "latents": goal, "steps": horizon, "motion": motion})
        tracking.append(robot_tracking)
        goals.append(robot_goals)
        recovery.append(robot_recovery)
    tracking_results = run_tasks(agent, rollouts, group, tracking)
    goal_results = run_tasks(agent, rollouts, group, goals)
    recovery_results = run_tasks(agent, rollouts, group, recovery)
    metrics: dict[str, float] = {}
    distances: list[np.ndarray] = []
    for robot, (rollout, expert) in enumerate(zip(rollouts, motions)):
        name = rollout.env.record.name
        emd, error = np.ones(len(expert)), []
        for task, result in zip(tracking[robot], tracking_results[robot]):
            reference = expert.frames[task["motion"]]["q"][1 : result["q"].shape[0] + 1]
            error.append(float(np.linalg.norm(result["q"] - reference, axis=-1).mean()))
            emd[task["motion"]] = earth_movers_distance(result["q"], reference)
        distances.append(emd)
        metrics[f"eval/{name}/tracking_mpjpe"] = float(np.mean(error))
        metrics[f"eval/{name}/tracking_emd"] = float(np.mean([emd[task["motion"]] for task in tracking[robot]]))
        metrics[f"eval/{name}/tracking_fallen"] = float(np.mean([result["fallen"] for result in tracking_results[robot]]))
        goal_error = []
        for task, result in zip(goals[robot], goal_results[robot]):
            target = expert.frames[task["motion"]]["q"][-1]
            goal_error.append(float(np.linalg.norm(result["q"] - target, axis=-1).mean()))
        metrics[f"eval/{name}/goal_mpjpe"] = float(np.mean(goal_error))
        if recovery[robot]:
            metrics[f"eval/{name}/getting_up_rate"] = float(np.mean([not result["fallen"] for result in recovery_results[robot]]))
    return metrics, distances
