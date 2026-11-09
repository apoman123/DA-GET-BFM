"""Observations of the cross-embodiment FB-CPR agent, built the same way from simulation and data.

Each robot state gives two per-DoF views, matching BFM-Zero's asymmetric inputs:

* **Policy view:** ``DaGetEncoder`` inputs (tokens with history and last actions, dynamics
  features with the last torque, mass matrix), identical to what physics pretraining
  feeds the encoder. It carries sensor noise and, as in BFM-Zero, no base linear velocity.
* **State view:** tokens for B, D and the critics' privileged context. Each joint token has
  the normalized position and velocity and the joint-frame position in the base frame,
  and repeats the base state (gravity direction, angular velocity, linear velocity /
  size, height / size). It has no history, action or noise, because motion data has none:
  otherwise D could spot simulated states by those signals alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.dynamics import STANDARD_GRAVITY
from model.embodiment import collate_embodiments
from model.physics import EPSILON, assemble_dynamics_features, configuration_terms, velocity_terms
from model.tokenization_da_get import DaGetTokenizer
from model.training.episodes import (
    BASE_FEATURE_SIZE,
    EmbodimentRecord,
    Trajectory,
    _difference,
    _requested,
    action_features,
    base_gravity,
    reconstruct_base,
)

VARIABLE_LOCAL_SIZE = 5
STATE_LOCAL_SIZE = 5
STATE_GLOBAL_SIZE = 10


def pad_joints(values: torch.Tensor, max_joints: int, dim: int = 1) -> torch.Tensor:
    """Zero-pad a joint dimension to ``max_joints``."""
    missing = max_joints - values.shape[dim]
    if missing < 0:
        raise ValueError(f"robot has {values.shape[dim]} joints but max_joints is {max_joints}")
    if missing == 0:
        return values
    shape = list(values.shape)
    shape[dim] = missing
    return torch.cat((values, values.new_zeros(shape)), dim=dim)


@dataclass
class RobotState:
    """Batched robot state in the base frame (``float64``, ``N`` environments or frames).

    Attributes:
        q (torch.Tensor): ``[N, J]`` joint positions.
        qd (torch.Tensor): ``[N, J]`` joint velocities.
        gravity_direction (torch.Tensor): ``[N, 3]`` unit gravity vector.
        angular_velocity (torch.Tensor): ``[N, 3]`` base angular velocity.
        linear_velocity (torch.Tensor): ``[N, 3]`` base linear velocity.
        height (torch.Tensor): ``[N]`` base height above the floor.
    """

    q: torch.Tensor
    qd: torch.Tensor
    gravity_direction: torch.Tensor
    angular_velocity: torch.Tensor
    linear_velocity: torch.Tensor
    height: torch.Tensor

    def select(self, indices: torch.Tensor) -> RobotState:
        """Return the state of some environments."""
        return RobotState(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))


@dataclass
class StateView:
    """Action-free state tokens' variable parts (``float32``).

    Attributes:
        local (torch.Tensor): ``[..., J, 5]`` per-joint features.
        global_features (torch.Tensor): ``[..., 10]`` base features.
        dynamics (torch.Tensor): ``[..., J, 7]`` dynamics features without a last torque.
    """

    local: torch.Tensor
    global_features: torch.Tensor
    dynamics: torch.Tensor

    def select(self, indices: torch.Tensor) -> StateView:
        """Return some entries along the first dimension."""
        return StateView(self.local[indices], self.global_features[indices], self.dynamics[indices])


@dataclass
class Observation:
    """Everything stored per step (``float32``).

    Attributes:
        local (torch.Tensor): ``[N, H, J, 5]`` policy joint history.
        global_features (torch.Tensor): ``[N, H, 11]`` policy base history.
        dynamics (torch.Tensor): ``[N, J, 7]`` policy dynamics features.
        mass (torch.Tensor): ``[N, J, J]`` mass matrix at the sensed configuration.
        state (StateView): True state, for B, D and the critics.
    """

    local: torch.Tensor
    global_features: torch.Tensor
    dynamics: torch.Tensor
    mass: torch.Tensor
    state: StateView

    def select(self, indices: torch.Tensor) -> Observation:
        """Return the observation of some environments."""
        return Observation(self.local[indices], self.global_features[indices], self.dynamics[indices], self.mass[indices], self.state.select(indices))

    def assign(self, indices: torch.Tensor, other: Observation) -> None:
        """Overwrite some environments in place."""
        self.local[indices] = other.local
        self.global_features[indices] = other.global_features
        self.dynamics[indices] = other.dynamics
        self.mass[indices] = other.mass
        self.state.local[indices] = other.state.local
        self.state.global_features[indices] = other.state.global_features
        self.state.dynamics[indices] = other.state.dynamics


class EmbodimentObserver:
    """Build policy and state views of one robot.

    Args:
        record (EmbodimentRecord): Robot, with position-mode actuator specs.
        history_size (int): Observation history length H of the policy tokens.
        observe_base_linear_velocity (bool): Show the base linear velocity to the policy.
        use_height (bool): Fill the height in the state view (off if the motions lack it).
        use_linear_velocity (bool): Fill the linear velocity in the state view (same).
    """

    def __init__(
        self,
        record: EmbodimentRecord,
        history_size: int,
        observe_base_linear_velocity: bool = False,
        use_height: bool = True,
        use_linear_velocity: bool = True,
    ) -> None:
        self.record = record
        self.graph = record.graph
        self.history_size = history_size
        self.observe_base_linear_velocity = observe_base_linear_velocity
        self.use_height = use_height
        self.use_linear_velocity = use_linear_velocity
        self.actuator = record.actuator
        self.floating = bool(self.graph.floating_base)
        self.length = float(self.graph.characteristic_length)
        self.structure = collate_embodiments([self.graph])
        features = record.joint_features.shape[1]
        self.tokenizer = DaGetTokenizer(fixed_local_size=features, variable_local_size=VARIABLE_LOCAL_SIZE,
                                        variable_global_size=BASE_FEATURE_SIZE, history_size=history_size)
        self.state_tokenizer = DaGetTokenizer(fixed_local_size=features, variable_local_size=STATE_LOCAL_SIZE,
                                              variable_global_size=STATE_GLOBAL_SIZE, history_size=1)
        speed = self.actuator["max_velocity"]
        self.speed_scale = torch.where(speed > 0, speed, torch.full_like(speed, 10.0))
        self._cached: dict[str, dict] = {}

    @property
    def token_size(self) -> int:
        """Return the policy encoder's input width."""
        return self.tokenizer.token_size

    @property
    def state_token_size(self) -> int:
        """Return the width of the state tokens."""
        return self.state_tokenizer.token_size

    def state_view(self, state: RobotState) -> StateView:
        """Return the view of the true state read by B, D and the critics."""
        linear = state.linear_velocity if self.use_linear_velocity else torch.zeros_like(state.linear_velocity)
        height = state.height if self.use_height else torch.zeros_like(state.height)
        positions = self.graph.forward_kinematics(state.q)[..., :3, 3] / self.length
        local = torch.cat(
            (
                self.graph.normalize_joint_positions(state.q)[..., None],
                (state.qd / self.speed_scale)[..., None],
                positions,
            ),
            dim=-1,
        )
        global_features = torch.cat((state.gravity_direction, state.angular_velocity, linear / self.length, height[:, None] / self.length), dim=-1)
        return StateView(local.float(), global_features.float(), self._dynamics(state, linear, None, 0.0).float())

    def _dynamics(self, state: RobotState, linear: torch.Tensor, previous_torque: torch.Tensor | None, source: float) -> torch.Tensor:
        """Return dynamics features; ``source`` is the torque-source flag (0 means unknown)."""
        q, qd = state.q, state.qd
        mass_diagonal, gravity_term = configuration_terms(self.graph, q, STANDARD_GRAVITY * state.gravity_direction)
        coriolis, friction = velocity_terms(
            self.graph, q, qd, linear if self.floating else None, state.angular_velocity if self.floating else None, self.actuator
        )
        previous = torch.zeros_like(q) if previous_torque is None else previous_torque
        return assemble_dynamics_features(mass_diagonal, coriolis, gravity_term, friction, previous, self.actuator["continuous_torque"], source)

    def policy_step(self, state: RobotState, previous_target: torch.Tensor | None, previous_q: torch.Tensor | None,
                    previous_qd: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one history step of policy tokens: local ``[N, J, 5]`` and global ``[N, 11]``.

        ``previous_*`` describe the command at t-1 with the state it was issued in;
        ``None`` (first frame of an episode) gives zero action features, as in pretraining.
        """
        count, joints = state.q.shape
        if previous_target is None:
            action = torch.zeros((count, joints, 3), dtype=state.q.dtype)
        else:
            requested = _requested(self.record, previous_target, "position", previous_q, previous_qd)
            action = action_features(self.record, previous_target, "position", requested)
        local = torch.cat(
            (
                self.graph.normalize_joint_positions(state.q)[..., None],
                (state.qd / self.speed_scale)[..., None],
                action,
            ),
            dim=-1,
        )
        linear = state.linear_velocity if self.observe_base_linear_velocity else torch.zeros_like(state.linear_velocity)
        flag = torch.full((count, 1), 1.0 if self.floating else 0.0, dtype=state.q.dtype)
        return local, torch.cat((state.gravity_direction, state.angular_velocity, linear / self.length, flag, flag), dim=-1)

    def observation(
        self,
        local_history: torch.Tensor,
        global_history: torch.Tensor,
        state: RobotState,
        policy_state: RobotState,
        previous_torque: torch.Tensor | None,
    ) -> Observation:
        """Assemble the stored observation.

        Args:
            local_history (torch.Tensor): ``[N, H, J, 5]``, oldest first.
            global_history (torch.Tensor): ``[N, H, 11]``, oldest first.
            state (RobotState): True state, for the state view.
            policy_state (RobotState): State the policy senses.
            previous_torque (torch.Tensor | None): ``[N, J]`` torque applied at t-1.

        Returns:
            Observation: Stored tensors.
        """
        linear = policy_state.linear_velocity if self.observe_base_linear_velocity else torch.zeros_like(policy_state.linear_velocity)
        dynamics = self._dynamics(policy_state, linear, previous_torque, 1.0)
        return Observation(
            local_history.float(), global_history.float(), dynamics.float(),
            self.graph.mass_matrix(policy_state.q).float(), self.state_view(state),
        )

    def _static(self, device: torch.device) -> dict:
        """Return the robot's per-sample encoder inputs on a device, with a batch of one."""
        key = str(device)
        if key not in self._cached:
            structure = self.structure
            self._cached[key] = {
                "parent_indices": structure.parent_indices.to(device),
                "joint_mask": structure.joint_mask.to(device),
                "adjacency_matrix": structure.adjacency_matrix.to(device),
                "graph_distances": tuple(item.to(device) for item in structure.graph_distances),
                "rigid_body_features": self.record.rigid_body_features.float().to(device)[None],
                "actuator_features": self.record.actuator_features.float().to(device)[None],
                "joint_features": self.record.joint_features.float().to(device)[None],
            }
        return self._cached[key]

    def structure_inputs(self, count: int, device: torch.device) -> dict:
        """Return the graph and hardware inputs of the encoder for ``count`` samples."""
        static = self._static(device)

        def expand(values: torch.Tensor) -> torch.Tensor:
            return values.expand(count, *values.shape[1:])

        return {
            "parent_indices": expand(static["parent_indices"]),
            "joint_mask": expand(static["joint_mask"]),
            "adjacency_matrix": expand(static["adjacency_matrix"]),
            "graph_distances": tuple(expand(item) for item in static["graph_distances"]),
            "rigid_body_features": expand(static["rigid_body_features"]),
            "actuator_features": expand(static["actuator_features"]),
        }

    def _joint_features(self, count: int, device: torch.device) -> torch.Tensor:
        return self._static(device)["joint_features"].expand(count, -1, -1)

    def policy_tokens(self, local: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """Return policy tokens ``[N, J, F]``, as physics pretraining builds them."""
        count, _, joints, _ = local.shape
        mask = torch.ones((count, joints), dtype=torch.bool, device=local.device)
        return self.tokenizer(self._joint_features(count, local.device), mask, variable_local=local, variable_global=global_features)

    def state_inputs(self, view: StateView) -> dict:
        """Return keyword inputs of a state encoder for ``N`` state views."""
        count, joints, _ = view.local.shape
        device = view.local.device
        mask = torch.ones((count, joints), dtype=torch.bool, device=device)
        tokens = self.state_tokenizer(
            self._joint_features(count, device), mask, variable_local=view.local[:, None], variable_global=view.global_features[:, None]
        )
        return {"input_values": tokens, "dynamics_features": view.dynamics, **self.structure_inputs(count, device)}


def trajectory_state(record: EmbodimentRecord, trajectory: Trajectory) -> tuple[RobotState, bool, bool]:
    """Read a recorded motion as ``RobotState`` frames.

    Args:
        record (EmbodimentRecord): Robot of the motion.
        trajectory (Trajectory): Motion; missing velocities are finite-differenced.

    Returns:
        tuple: State frames, whether the base height is known, whether the base
            linear velocity is known.
    """
    graph = record.graph
    q = trajectory.positions.to(torch.float64)
    steps = q.shape[0]
    continuous = torch.tensor([kind == "continuous" for kind in graph.joint_types])
    qd = trajectory.velocities.to(torch.float64) if trajectory.velocities is not None else _difference(q, trajectory.dt, continuous)
    gravity = base_gravity(trajectory, steps)
    direction = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
    base = reconstruct_base(trajectory, steps, gravity)
    zeros = torch.zeros((steps, 3), dtype=torch.float64)
    if not graph.floating_base:
        return RobotState(q, qd, direction, zeros, zeros, torch.zeros(steps, dtype=torch.float64)), True, True
    height = base.position[:, 2] if base.position is not None else torch.zeros(steps, dtype=torch.float64)
    state = RobotState(q, qd, direction, base.angular_velocity, base.linear_velocity, height)
    return state, base.position is not None, base.linear_known
