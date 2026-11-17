"""Measure the per-step latency of the policy at deployment (one robot, one environment).

One 50 Hz control step must finish in 20 ms. A step is:

* ``observation``: policy tokens, dynamics features and mass matrix from the robot state
  (``EmbodimentObserver``, float64 on the CPU, where the sensor readings arrive); the
  state view, which only B, D and the critics read during training, is left out;
* ``act``: copy to the device, DA-GET encoder, then actor (``DaGetFbCprAgent.act``);
* ``tracking``: B of one new reference frame, for tracking a motion online (optional:
  goal and reward prompts compute z once, not every step).

Run on the deployment GPU::

    uv run python scripts/benchmark_inference.py --device cuda --robots g1 h1 go1 \\
        --encoder_sizes small medium large

Networks are randomly initialized: latency does not depend on the weights.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import BfmConfig, EmbodimentGraph, JointFeatureConfig, actuator_specs  # noqa: E402
from model.rl import DaGetFbCprAgent, EmbodimentObserver, FbCprConfig, Observation, RobotState  # noqa: E402
from model.rl.trainer import build_encoder  # noqa: E402
from model.training.episodes import EmbodimentRecord  # noqa: E402
from model.training.ground_sim import ground_specs  # noqa: E402

URDFS = {name: ROOT / "assets" / "urdf" / f"{file}.urdf" for name, file in
         (("g1", "g1_29dof"), ("h1", "h1"), ("go1", "go1"), ("anymal", "anymal_d"))}


def build_observer(robot: str, history: int) -> EmbodimentObserver:
    """Return the observer of a floating-base robot from its URDF."""
    graph = EmbodimentGraph.from_urdf(URDFS[robot], JointFeatureConfig.da_get_zero(), floating_base=True)
    record = EmbodimentRecord.build(robot, graph, ground_specs(actuator_specs(graph)))
    return EmbodimentObserver(record, history)


def random_state(joints: int, generator: torch.Generator) -> RobotState:
    """Return one random standing state."""
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    return RobotState(0.3 * torch.randn(1, joints, generator=generator, dtype=torch.float64),
                      torch.randn(1, joints, generator=generator, dtype=torch.float64),
                      gravity, torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
                      torch.full((1,), 0.8, dtype=torch.float64))


def timed(function, device: torch.device, warmup: int, steps: int) -> list[float]:
    """Return the wall-clock milliseconds of ``steps`` calls after ``warmup`` calls."""
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(1000 * (time.perf_counter() - start))
    return times


def summary(times: list[float]) -> tuple[float, float, float]:
    """Return the median, 99th percentile and maximum."""
    ordered = sorted(times)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], ordered[-1]


def benchmark(robot: str, size: str, device: torch.device, history: int, warmup: int, steps: int) -> dict[str, list[float]]:
    """Time each part of one control step for one robot and encoder size."""
    observer = build_observer(robot, history)
    pretrained = build_encoder(None, observer.token_size, size)
    config = BfmConfig.from_sizes(embedding_size=pretrained.encoder.config.hidden_size, state_token_size=observer.state_token_size)
    agent = DaGetFbCprAgent(config, FbCprConfig(), pretrained, [observer], str(device))
    generator = torch.Generator().manual_seed(0)
    joints = observer.graph.num_joints
    state = random_state(joints, generator)
    previous = random_state(joints, generator)
    target = previous.q.clone()
    z = agent.model.sample_z(1, device)

    torque = torch.zeros(1, joints, dtype=torch.float64)
    local_history = torch.zeros(1, history, joints, 5, dtype=torch.float64)
    global_history = torch.zeros(1, history, 11, dtype=torch.float64)
    view = observer.state_view(state)

    def observe():
        """Build what the policy reads; the state view is for B, D and critics, so it is left out."""
        local, global_features = observer.policy_step(state, target, previous.q, previous.qd)
        local_history[:, :-1], local_history[:, -1] = local_history[:, 1:].clone(), local
        global_history[:, :-1], global_history[:, -1] = global_history[:, 1:].clone(), global_features
        dynamics, mass = observer.policy_physics(state, torque)
        return Observation(local_history.float(), global_history.float(), dynamics, mass, view)

    observation = observe()

    def act():
        return agent.act(0, observation, z, mean=True).cpu()

    def control_step():
        nonlocal observation
        observation = observe()
        return act()

    return {
        "observation": timed(observe, device, warmup, steps),
        "act": timed(act, device, warmup, steps),
        "tracking": timed(lambda: agent._backward(0, view), device, warmup, steps),
        "control step": timed(control_step, device, warmup, steps),
    }


def main() -> None:
    """Parse arguments, run the benchmarks and print a table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--robots", nargs="+", default=["g1", "h1", "go1"], choices=sorted(URDFS))
    parser.add_argument("--encoder_sizes", nargs="+", default=["small"], choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--torch_threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.torch_threads)
    device = torch.device(arguments.device)
    budget = 1000 / arguments.control_hz
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU, {arguments.torch_threads} threads"
    print(f"device: {name}; budget {budget:.1f} ms per step at {arguments.control_hz:g} Hz; batch 1")
    print(f"{'robot':6} {'DoF':>3} {'encoder':7} {'part':13} {'median ms':>9} {'p99 ms':>7} {'max ms':>7}")
    failed = False
    for robot in arguments.robots:
        for size in arguments.encoder_sizes:
            results = benchmark(robot, size, device, arguments.history_size, arguments.warmup, arguments.steps)
            joints = build_observer(robot, arguments.history_size).graph.num_joints
            for part, times in results.items():
                median, p99, worst = summary(times)
                mark = ""
                if part == "control step":
                    ok = p99 < budget
                    failed |= not ok
                    mark = "  OK" if ok else "  OVER BUDGET"
                print(f"{robot:6} {joints:>3} {size:7} {part:13} {median:9.2f} {p99:7.2f} {worst:7.2f}{mark}")
    print("control step = observation + act; add tracking when following a motion online.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
