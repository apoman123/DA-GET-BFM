"""Per-robot replay and expert-motion buffers of the cross-embodiment FB-CPR agent."""

from __future__ import annotations

import torch

from model.rl.observation import EmbodimentObserver, Observation, RobotState, trajectory_state
from model.training.episodes import Trajectory


def rotation_from_gravity(direction: torch.Tensor) -> torch.Tensor:
    """Return base-to-world rotations ``[N, 3, 3]`` with zero yaw from gravity directions in the base frame."""
    z_row = -direction
    helper = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype).expand_as(z_row)
    y_row = torch.linalg.cross(z_row, helper, dim=-1)
    y_row = y_row / y_row.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x_row = torch.linalg.cross(y_row, z_row, dim=-1)
    return torch.stack((x_row, y_row, z_row), dim=-2)


def _resample(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Interpolate ``[T, ...]`` frames linearly at fractional frame indices ``times``."""
    lower = times.floor().long().clamp_max(values.shape[0] - 1)
    upper = (lower + 1).clamp_max(values.shape[0] - 1)
    weight = (times - lower).to(values.dtype).reshape(-1, *([1] * (values.ndim - 1)))
    return values[lower] * (1 - weight) + values[upper] * weight


def resample_state(state: RobotState, rotation: torch.Tensor, dt: float, control_dt: float) -> tuple[RobotState, torch.Tensor]:
    """Resample motion frames from period ``dt`` to the control period; rotations use the nearest frame."""
    if abs(dt - control_dt) < 1e-9:
        return state, rotation
    duration = (state.q.shape[0] - 1) * dt
    times = torch.arange(int(duration / control_dt) + 1, dtype=torch.float64) * control_dt / dt
    fields = {name: _resample(getattr(state, name), times) for name in state.__dataclass_fields__}
    fields["gravity_direction"] = torch.nn.functional.normalize(fields["gravity_direction"], dim=-1)
    return RobotState(**fields), rotation[times.round().long().clamp_max(rotation.shape[0] - 1)]


class ExpertMotions:
    """Action-free motions of one robot: flat states for D and z, and frames for resets.

    Args:
        observer (EmbodimentObserver): Observer of the robot.
        trajectories (list[Trajectory]): Motions of at least two frames.
        control_dt (float | None): Resample to this period, so that frames match
            environment steps; keeps the recorded rate if None.
    """

    def __init__(self, observer: EmbodimentObserver, trajectories: list[Trajectory], control_dt: float | None = None) -> None:
        self.flat: list[torch.Tensor] = []
        self.frames: list[dict] = []
        for trajectory in trajectories:
            if trajectory.positions.shape[0] < 2:
                continue
            state, _, _ = trajectory_state(observer.record, trajectory)
            rotation = trajectory.base_orientation
            rotation = rotation_from_gravity(state.gravity_direction) if rotation is None else rotation.to(torch.float64)
            if control_dt is not None:
                state, rotation = resample_state(state, rotation, trajectory.dt, control_dt)
            if state.q.shape[0] < 2:
                continue
            self.flat.append(observer.flat_state(state))
            self.frames.append(
                {
                    "q": state.q.numpy(),
                    "qd": state.qd.numpy(),
                    "rotation": rotation.numpy(),
                    "linear": state.linear_velocity.numpy(),
                    "angular": state.angular_velocity.numpy(),
                }
            )
        if not self.flat:
            raise ValueError(f"{observer.record.name}: no motion with at least two frames")
        self.lengths = torch.tensor([len(flat) for flat in self.flat])

    def sample_windows(self, windows: int, length: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample consecutive frames, with motions weighted by length.

        Args:
            windows (int): Number of windows W.
            length (int): Frames per window L; short motions repeat their last frame.
            generator (torch.Generator | None): Random source.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: States and next states, each ``[W, L, S]``.
        """
        motions = torch.multinomial(self.lengths.float(), windows, replacement=True, generator=generator)
        states, next_states = [], []
        for motion in motions.tolist():
            flat = self.flat[motion]
            start = int(torch.randint(0, len(flat) - 1, (1,), generator=generator))
            frames = torch.arange(start, start + length).clamp_max(len(flat) - 2)
            states.append(flat[frames])
            next_states.append(flat[frames + 1])
        return torch.stack(states), torch.stack(next_states)


class ReplayBuffer:
    """Ring buffer of one robot's transitions, stored unpadded.

    Args:
        capacity (int): Maximum transitions.
        reward_names (tuple[str, ...]): Auxiliary reward keys, in storage order.
    """

    def __init__(self, capacity: int, reward_names: tuple[str, ...]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.reward_names = reward_names
        self.storage: dict[str, torch.Tensor] | None = None
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.size

    def add(
        self,
        observation: Observation,
        action: torch.Tensor,
        next_observation: Observation,
        z: torch.Tensor,
        terminated: torch.Tensor,
        aux_rewards: dict[str, torch.Tensor],
    ) -> None:
        """Append a batch of ``N`` transitions."""
        values = {
            "tokens": observation.tokens,
            "dynamics": observation.dynamics,
            "mass": observation.mass,
            "flat": observation.flat,
            "next_tokens": next_observation.tokens,
            "next_dynamics": next_observation.dynamics,
            "next_mass": next_observation.mass,
            "next_flat": next_observation.flat,
            "action": action.float(),
            "z": z.float(),
            "terminated": terminated.bool(),
            "aux_rewards": torch.stack([aux_rewards[name].float() for name in self.reward_names], dim=-1),
        }
        values = {key: value.detach().cpu() for key, value in values.items()}
        if self.storage is None:
            self.storage = {key: torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype) for key, value in values.items()}
        count = values["z"].shape[0]
        indices = (self.position + torch.arange(count)) % self.capacity
        for key, value in values.items():
            self.storage[key][indices] = value
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Return ``count`` uniformly sampled transitions."""
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = torch.randint(0, self.size, (count,), generator=generator)
        return {key: value[indices] for key, value in self.storage.items()}
