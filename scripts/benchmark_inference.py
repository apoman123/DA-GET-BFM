"""Measure the latency of every part of the DA-GET BFM on one robot.

One 50 Hz control step must finish in 20 ms. The first table times each step's work at
batch 1 and checks the totals against that budget:

* ``observation``: policy tokens, dynamics features and mass matrix from the robot state
  (``EmbodimentObserver``, float64 on the CPU, where the sensor readings arrive).
* ``act (eager)``: the training path, ``DaGetFbCprAgent.act``.
* ``act (policy)``: ``DaGetBfmPolicy.act``, the deployment path: one copy each way,
  cached physics memory, and on a GPU a CUDA Graph replay of encoder and actor.
* ``reference state``: the state view of a reference frame, for tracking a motion
  streamed live (a motion known ahead is encoded once, see the second table).
* ``B (eager)`` and ``B (policy)``: B of that frame.
* ``control step``: observation + act (policy).
* ``+ live tracking``: control step + reference state + B (policy).

The second table times what does not run every control step: each network alone at
batch 1 (F, critics and discriminator run only in training), and the one-off prompts
(goal, reward over a buffer, tracking latents of a whole motion).

Run on the deployment GPU::

    uv run python scripts/benchmark_inference.py --device cuda --robots g1 h1 go1 \\
        --encoder_sizes small medium large

Networks are randomly initialized: latency does not depend on the weights. The exit
status is 1 when a ``control step`` row is over budget at its 99th percentile.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import BfmConfig, EmbodimentGraph, JointFeatureConfig, actuator_specs  # noqa: E402
from model.rl import DaGetBfmPolicy, DaGetFbCprAgent, EmbodimentObserver, FbCprConfig, Observation, RobotState  # noqa: E402
from model.rl.trainer import build_encoder  # noqa: E402
from model.training.episodes import EmbodimentRecord  # noqa: E402
from model.training.ground_sim import ground_specs  # noqa: E402

URDFS = {name: ROOT / "assets" / "urdf" / f"{file}.urdf" for name, file in
         (("g1", "g1_29dof"), ("h1", "h1"), ("go1", "go1"), ("anymal", "anymal_d"))}
TOTALS = ("control step", "+ live tracking")


def build_observer(robot: str, history: int) -> EmbodimentObserver:
    """Return the observer of a floating-base robot from its URDF."""
    graph = EmbodimentGraph.from_urdf(URDFS[robot], JointFeatureConfig.da_get_zero(), floating_base=True)
    record = EmbodimentRecord.build(robot, graph, ground_specs(actuator_specs(graph)))
    return EmbodimentObserver(record, history)


def random_state(joints: int, generator: torch.Generator, count: int = 1) -> RobotState:
    """Return random upright states."""
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64).expand(count, 3)
    return RobotState(0.3 * torch.randn(count, joints, generator=generator, dtype=torch.float64),
                      torch.randn(count, joints, generator=generator, dtype=torch.float64),
                      gravity, 0.1 * torch.randn(count, 3, generator=generator, dtype=torch.float64),
                      0.1 * torch.randn(count, 3, generator=generator, dtype=torch.float64),
                      torch.full((count,), 0.8, dtype=torch.float64))


def timed(function: Callable[[], object], device: torch.device, warmup: int, steps: int) -> list[float]:
    """Return the wall-clock milliseconds of ``steps`` calls after ``warmup`` calls."""
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        function()
    synchronize()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        function()
        synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return times


def summary(times: list[float]) -> tuple[float, float, float]:
    """Return the median, 99th percentile and maximum."""
    ordered = sorted(times)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], ordered[-1]


def benchmark(robot: str, size: str, device: torch.device, arguments: argparse.Namespace) -> tuple[dict, dict]:
    """Time every part for one robot and encoder size.

    Returns:
        tuple[dict, dict]: Per-step parts and other parts, each ``{name: times}``.
    """
    history, warmup, steps = arguments.history_size, arguments.warmup, arguments.steps
    observer = build_observer(robot, history)
    pretrained = build_encoder(None, observer.token_size, size)
    config = BfmConfig.from_sizes(embedding_size=pretrained.encoder.config.hidden_size, state_token_size=observer.state_token_size)
    agent = DaGetFbCprAgent(config, FbCprConfig(), pretrained, [observer], str(device))
    model = agent.model.eval()
    policy = DaGetBfmPolicy(model, observer, device)
    generator = torch.Generator().manual_seed(0)
    joints = observer.graph.num_joints
    state, previous, reference = (random_state(joints, generator) for _ in range(3))
    target = previous.q.clone()
    torque = torch.zeros(1, joints, dtype=torch.float64)
    local_history = torch.zeros(1, history, joints, 5, dtype=torch.float64)
    global_history = torch.zeros(1, history, 11, dtype=torch.float64)
    z = model.sample_z(1).cpu()
    view = observer.state_view(state)

    def observe() -> Observation:
        """Build what the policy reads; the state view is for B, D and critics, so it is left out."""
        local, global_features = observer.policy_step(state, target, previous.q, previous.qd)
        local_history[:, :-1], local_history[:, -1] = local_history[:, 1:].clone(), local
        global_history[:, :-1], global_history[:, -1] = global_history[:, 1:].clone(), global_features
        dynamics, mass = observer.policy_physics(state, torque)
        return Observation(local_history.float(), global_history.float(), dynamics, mass, view)

    observation = observe()

    def policy_act(item: Observation) -> torch.Tensor:
        return policy.act(item.local, item.global_features, item.dynamics, item.mass, z)

    def control_step() -> torch.Tensor:
        return policy_act(observe())

    def live_tracking() -> torch.Tensor:
        control_step()
        return policy.backward(observer.state_view(reference))

    per_step = {
        "observation": timed(observe, device, warmup, steps),
        "act (eager)": timed(lambda: agent.act(0, observation, z, mean=True).cpu(), device, warmup, steps),
        "act (policy)": timed(lambda: policy_act(observation), device, warmup, steps),
        "reference state": timed(lambda: observer.state_view(reference), device, warmup, steps),
        "B (eager)": timed(lambda: agent._backward(0, view).cpu(), device, warmup, steps),
        "B (policy)": timed(lambda: policy.backward(view), device, warmup, steps),
        "control step": timed(control_step, device, warmup, steps),
        "+ live tracking": timed(live_tracking, device, warmup, steps),
    }

    with torch.no_grad():
        mask = policy.structure["joint_mask"]
        zd = z.to(device)
        embeddings = agent._embed(model.encoder, 0, observation)
        state_inputs = agent._state_inputs(0, view)
        state_embeddings = model.encode_state(state_inputs)
        action = model.actor(embeddings, zd, config.actor_std, mask).mean
        heads = {"context_states": state_inputs["input_values"], "context_mask": mask}
        motion = observer.state_view(random_state(joints, generator, arguments.motion_frames + 1))
        buffer = observer.state_view(random_state(joints, generator, arguments.reward_states))
        buffer_inputs = agent._state_inputs(0, buffer)
        rewards = torch.rand(arguments.reward_states, 1, device=device)
        others = {
            "policy encoder": timed(lambda: agent._embed(model.encoder, 0, observation), device, warmup, steps),
            "actor head": timed(lambda: model.actor(embeddings, zd, config.actor_std, mask).mean, device, warmup, steps),
            "state encoder": timed(lambda: model.encode_state(agent._state_inputs(0, view)), device, warmup, steps),
            "B head": timed(lambda: model.backward_map(state_embeddings, mask), device, warmup, steps),
            "F head (train)": timed(lambda: model.forward_map(embeddings, zd, action, mask, **heads), device, warmup, steps),
            "Q_D head (train)": timed(lambda: model.critic(embeddings, zd, action, mask, **heads), device, warmup, steps),
            "Q_R head (train)": timed(lambda: model.aux_critic(embeddings, zd, action, mask, **heads), device, warmup, steps),
            "D head (train)": timed(lambda: model.discriminator.compute_logits(state_embeddings, zd, mask), device, warmup, steps),
            "goal z": timed(lambda: agent.goal_latent(0, observer.state_view(reference)).cpu(), device, warmup, steps),
            f"reward z ({arguments.reward_states} states)": timed(
                lambda: model.reward_inference(buffer_inputs, rewards).cpu(), device, 1, arguments.long_steps
            ),
            f"tracking z ({arguments.motion_frames} frames)": timed(
                lambda: agent.tracking_latents(0, motion).cpu(), device, 1, arguments.long_steps
            ),
        }
    return per_step, others


def verdict(part: str, p99: float, worst: float, budget: float) -> str:
    """Return the budget check of a total, empty for a component."""
    if part not in TOTALS:
        return ""
    if p99 > budget:
        return "OVER BUDGET"
    return "OK" if worst <= budget else "OK (max over)"


def main() -> None:
    """Parse arguments, run the benchmarks and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--robots", nargs="+", default=["g1", "h1", "go1"], choices=sorted(URDFS))
    parser.add_argument("--encoder_sizes", nargs="+", default=["small"], choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--long_steps", type=int, default=5, help="Repetitions of the reward and tracking prompts.")
    parser.add_argument("--reward_states", type=int, default=1000, help="Buffer states of the reward prompt.")
    parser.add_argument("--motion_frames", type=int, default=500, help="Frames of the tracked motion (10 s at 50 Hz).")
    parser.add_argument("--torch_threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.torch_threads)
    device = torch.device(arguments.device)
    budget = 1000 / arguments.control_hz
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU, {arguments.torch_threads} threads"
    graphs = "CUDA Graphs on" if device.type == "cuda" else "no CUDA Graphs on a CPU"
    print(f"device: {name}; budget {budget:.1f} ms per step at {arguments.control_hz:g} Hz; batch 1; policy: {graphs}")
    header = f"{'robot':6} {'DoF':>3} {'encoder':7} {'part':28} {'median ms':>9} {'p99 ms':>7} {'max ms':>7}"
    rows = []
    for robot in arguments.robots:
        joints = build_observer(robot, arguments.history_size).graph.num_joints
        for size in arguments.encoder_sizes:
            rows.append((robot, joints, size, *benchmark(robot, size, device, arguments)))
    failed = False
    print("\nEvery control step")
    print(header)
    for robot, joints, size, per_step, _ in rows:
        for part, times in per_step.items():
            median, p99, worst = summary(times)
            mark = verdict(part, p99, worst, budget)
            failed |= part == "control step" and mark == "OVER BUDGET"
            print(f"{robot:6} {joints:>3} {size:7} {part:28} {median:9.2f} {p99:7.2f} {worst:7.2f}  {mark}".rstrip())
    print("\nNot every control step: each network alone (train = training only), and one-off prompts")
    print(header)
    for robot, joints, size, _, others in rows:
        for part, times in others.items():
            median, p99, worst = summary(times)
            print(f"{robot:6} {joints:>3} {size:7} {part:28} {median:9.2f} {p99:7.2f} {worst:7.2f}")
    print("\ncontrol step = observation + act (policy); + live tracking adds reference state + B (policy).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
