"""Per-robot replay and expert-motion buffers of the cross-embodiment FB-CPR agent."""

from __future__ import annotations

import torch

from model.rl.observation import EmbodimentObserver, Observation, RobotState, StateView, trajectory_state
from model.training.episodes import Trajectory

CHUNK_SECONDS = 10.0  # BFM-Zero splits motions into 10 s chunks.


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
    """Action-free motion chunks of one robot: state views for D and z, frames for resets, priorities.

    Args:
        observer (EmbodimentObserver): Observer of the robot.
        trajectories (list[Trajectory]): Motions.
        control_dt (float | None): Resample to this period, so that frames match
            environment steps; keeps the recorded rate if None.
        chunk_seconds (float | None): Split motions into chunks of this length.
    """

    def __init__(self, observer: EmbodimentObserver, trajectories: list[Trajectory], control_dt: float | None = None,
                 chunk_seconds: float | None = CHUNK_SECONDS) -> None:
        self.observer = observer
        self.states: list[StateView] = []
        self.frames: list[dict] = []
        for trajectory in trajectories:
            state, _, _ = trajectory_state(observer.record, trajectory)
            rotation = trajectory.base_orientation
            rotation = rotation_from_gravity(state.gravity_direction) if rotation is None else rotation.to(torch.float64)
            dt = trajectory.dt
            if control_dt is not None:
                state, rotation = resample_state(state, rotation, dt, control_dt)
                dt = control_dt
            total = state.q.shape[0]
            size = total if chunk_seconds is None else max(int(round(chunk_seconds / dt)), 2)
            for start in range(0, total, size):
                indices = torch.arange(start, min(start + size, total))
                if indices.shape[0] < 2:
                    continue
                chunk = state.select(indices)
                self.states.append(observer.state_view(chunk))
                self.frames.append(
                    {
                        "q": chunk.q.numpy(),
                        "qd": chunk.qd.numpy(),
                        "rotation": rotation[indices].numpy(),
                        "linear": chunk.linear_velocity.numpy(),
                        "angular": chunk.angular_velocity.numpy(),
                    }
                )
        if not self.states:
            raise ValueError(f"{observer.record.name}: no motion with at least two frames")
        self.priorities = torch.ones(len(self.states), dtype=torch.float64)

    def __len__(self) -> int:
        """Return the number of motion chunks."""
        return len(self.states)

    def set_priorities(self, priorities: torch.Tensor) -> None:
        """Set each motion's sampling weight."""
        if priorities.shape != self.priorities.shape or not bool((priorities > 0).all()):
            raise ValueError("priorities must be positive, one per motion")
        self.priorities = priorities.to(torch.float64)

    def sample_windows(self, windows: int, length: int, generator: torch.Generator | None = None) -> tuple[StateView, StateView]:
        """Sample consecutive frames, with motions weighted by priority and starts uniform.

        Args:
            windows (int): Number of windows W.
            length (int): Frames per window L; short motions repeat their last frame.
            generator (torch.Generator | None): Random source.

        Returns:
            tuple[StateView, StateView]: States and next states, ``[W, L, ...]``.
        """
        motions = torch.multinomial(self.priorities, windows, replacement=True, generator=generator)
        current, following = [], []
        for motion in motions.tolist():
            view = self.states[motion]
            frames_available = view.local.shape[0]
            start = int(torch.randint(0, frames_available - 1, (1,), generator=generator))
            indices = torch.arange(start, start + length).clamp_max(frames_available - 2)
            current.append(view.select(indices))
            following.append(view.select(indices + 1))

        def stack(views: list[StateView]) -> StateView:
            return StateView(*(torch.stack([getattr(view, name) for view in views]) for name in ("local", "global_features", "dynamics")))

        return stack(current), stack(following)


def _pack_mass(mass: torch.Tensor) -> torch.Tensor:
    """Keep the upper triangle of symmetric matrices ``[N, J, J]``."""
    joints = mass.shape[-1]
    rows, columns = torch.triu_indices(joints, joints, device=mass.device)
    return mass[:, rows, columns]


def _unpack_mass(packed: torch.Tensor, joints: int) -> torch.Tensor:
    """Rebuild symmetric matrices from ``_pack_mass``."""
    rows, columns = torch.triu_indices(joints, joints, device=packed.device)
    mass = torch.zeros((packed.shape[0], joints, joints), dtype=packed.dtype, device=packed.device)
    mass[:, rows, columns] = packed
    mass[:, columns, rows] = packed
    return mass


class ReplayBuffer:
    """Ring buffer of one robot's transitions, unpadded and in half precision to save memory.

    Args:
        capacity (int): Maximum transitions.
        reward_names (tuple[str, ...]): Auxiliary reward keys, in storage order.
        device (torch.device | str): Storage device.
    """

    def __init__(self, capacity: int, reward_names: tuple[str, ...], device: torch.device | str = "cpu") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.reward_names = reward_names
        self.device = torch.device(device)
        self.storage: dict[str, torch.Tensor] | None = None
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.size

    def _observation_fields(self, prefix: str, observation: Observation) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}local": observation.local.half(),
            f"{prefix}global": observation.global_features.half(),
            f"{prefix}dynamics": observation.dynamics.half(),
            f"{prefix}mass": _pack_mass(observation.mass).half(),
            f"{prefix}state_local": observation.state.local.half(),
            f"{prefix}state_global": observation.state.global_features.half(),
            f"{prefix}state_dynamics": observation.state.dynamics.half(),
        }

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
            **self._observation_fields("", observation),
            **self._observation_fields("next_", next_observation),
            "action": action.float(),
            "z": z.float(),
            "terminated": terminated.bool(),
            "aux_rewards": torch.stack([aux_rewards[name].float() for name in self.reward_names], dim=-1),
        }
        values = {key: value.detach().to(self.device) for key, value in values.items()}
        if self.storage is None:
            self.storage = {key: torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.device) for key, value in values.items()}
        count = values["z"].shape[0]
        indices = (self.position + torch.arange(count, device=self.device)) % self.capacity
        for key, value in values.items():
            self.storage[key][indices] = value
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def _observation(self, sample: dict[str, torch.Tensor], prefix: str, joints: int, device: torch.device) -> Observation:
        def read(key: str) -> torch.Tensor:
            return sample[f"{prefix}{key}"].to(device).float()

        return Observation(
            read("local"), read("global"), read("dynamics"), _unpack_mass(read("mass"), joints),
            StateView(read("state_local"), read("state_global"), read("state_dynamics")),
        )

    def sample(self, count: int, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> dict:
        """Return ``count`` uniformly sampled transitions as ``float32`` on a device."""
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        device = torch.device(device)
        indices = torch.randint(0, self.size, (count,), generator=generator, device=self.device if generator is None else "cpu").to(self.device)
        sample = {key: value[indices] for key, value in self.storage.items()}
        joints = sample["action"].shape[1]
        return {
            "observation": self._observation(sample, "", joints, device),
            "next_observation": self._observation(sample, "next_", joints, device),
            "action": sample["action"].to(device),
            "z": sample["z"].to(device),
            "terminated": sample["terminated"].to(device),
            "aux_rewards": sample["aux_rewards"].to(device),
        }

    def transition_bytes(self) -> int:
        """Return the memory used per transition, once the buffer holds data."""
        if self.storage is None:
            return 0
        return sum(value[0].numel() * value.element_size() for value in self.storage.values())
