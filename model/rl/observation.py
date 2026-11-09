"""Observations of the cross-embodiment FB-CPR agent, built the same way from simulation and data.

Each robot state gives two views, matching BFM-Zero's asymmetric inputs:

* **Policy view:** ``DaGetEncoder`` inputs (tokens with history, dynamics features,
  mass matrix), identical to what physics pretraining feeds the encoder. As in
  BFM-Zero, the policy does not see the base linear velocity by default.
* **Flat state:** one fixed-size vector for B, D and the critics' privileged token,
  padded to ``max_joints``: gravity direction (3), base angular velocity (3), base
  linear velocity / size (3), base height / size (1), normalized joint positions,
  velocities and valid-joint mask (3 x J), and joint-frame positions in the base
  frame / size (3 x J, the counterpart of BFM-Zero's body positions).
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

FLAT_GLOBAL_SIZE = 10
VARIABLE_LOCAL_SIZE = 5


def flat_state_size(max_joints: int) -> int:
    """Return the flat-state width for robots of up to ``max_joints`` DoF."""
    return FLAT_GLOBAL_SIZE + 6 * max_joints


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
class Observation:
    """Everything stored per step (``float32``).

    Attributes:
        tokens (torch.Tensor): ``[N, J, F]`` encoder tokens.
        dynamics (torch.Tensor): ``[N, J, 7]`` dynamics features.
        mass (torch.Tensor): ``[N, J, J]`` mass matrix.
        flat (torch.Tensor): ``[N, S]`` flat state.
    """

    tokens: torch.Tensor
    dynamics: torch.Tensor
    mass: torch.Tensor
    flat: torch.Tensor

    def select(self, indices: torch.Tensor) -> Observation:
        """Return the observation of some environments."""
        return Observation(self.tokens[indices], self.dynamics[indices], self.mass[indices], self.flat[indices])


class EmbodimentObserver:
    """Build tokens, physics features and flat states of one robot.

    Args:
        record (EmbodimentRecord): Robot, with position-mode actuator specs.
        history_size (int): Observation history length H of the tokens.
        max_joints (int): Padded joint count of the flat state.
        observe_base_linear_velocity (bool): Show the base linear velocity to the policy.
        use_height (bool): Fill the height in the flat state (off if the motions lack it).
        use_linear_velocity (bool): Fill the linear velocity in the flat state (same).
    """

    def __init__(
        self,
        record: EmbodimentRecord,
        history_size: int,
        max_joints: int,
        observe_base_linear_velocity: bool = False,
        use_height: bool = True,
        use_linear_velocity: bool = True,
    ) -> None:
        self.record = record
        self.graph = record.graph
        self.history_size = history_size
        self.max_joints = max_joints
        self.observe_base_linear_velocity = observe_base_linear_velocity
        self.use_height = use_height
        self.use_linear_velocity = use_linear_velocity
        self.actuator = record.actuator
        self.floating = bool(self.graph.floating_base)
        self.length = float(self.graph.characteristic_length)
        self.structure = collate_embodiments([self.graph])
        self.tokenizer = DaGetTokenizer(
            fixed_local_size=record.joint_features.shape[1],
            variable_local_size=VARIABLE_LOCAL_SIZE,
            variable_global_size=BASE_FEATURE_SIZE,
            history_size=history_size,
        )
        speed = self.actuator["max_velocity"]
        self.speed_scale = torch.where(speed > 0, speed, torch.full_like(speed, 10.0))

    @property
    def token_size(self) -> int:
        """Return the encoder input width."""
        return self.tokenizer.token_size

    def flat_state(self, state: RobotState) -> torch.Tensor:
        """Return the padded ``[N, S]`` flat state."""
        count = state.q.shape[0]
        linear = state.linear_velocity if self.use_linear_velocity else torch.zeros_like(state.linear_velocity)
        height = state.height if self.use_height else torch.zeros_like(state.height)
        positions = self.graph.forward_kinematics(state.q)[..., :3, 3] / self.length
        mask = torch.ones_like(state.q)
        parts = (
            state.gravity_direction,
            state.angular_velocity,
            linear / self.length,
            height[:, None] / self.length,
            pad_joints(self.graph.normalize_joint_positions(state.q), self.max_joints),
            pad_joints(state.qd / self.speed_scale, self.max_joints),
            pad_joints(mask, self.max_joints),
            pad_joints(positions, self.max_joints).reshape(count, -1),
        )
        return torch.cat(parts, dim=-1).float()

    def policy_step(self, state: RobotState, previous_target: torch.Tensor | None, previous_q: torch.Tensor | None,
                    previous_qd: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one history step of tokens: local ``[N, J, 5]`` and global ``[N, 11]``.

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
        global_features = torch.cat(
            (state.gravity_direction, state.angular_velocity, linear / self.length, flag, flag), dim=-1
        )
        return local, global_features

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
            state (RobotState): True state, for the flat view.
            policy_state (RobotState): State the policy perceives (with sensor noise).
            previous_torque (torch.Tensor | None): ``[N, J]`` torque applied at t-1.

        Returns:
            Observation: Stored tensors.
        """
        count = state.q.shape[0]
        q, qd = policy_state.q, policy_state.qd
        gravity = STANDARD_GRAVITY * policy_state.gravity_direction
        mass_diagonal, gravity_term = configuration_terms(self.graph, q, gravity)
        floating = self.floating
        linear = policy_state.linear_velocity if self.observe_base_linear_velocity else torch.zeros_like(q[:, :3])
        coriolis, friction = velocity_terms(
            self.graph, q, qd, linear if floating else None,
            policy_state.angular_velocity if floating else None, self.actuator,
        )
        dynamics = assemble_dynamics_features(
            mass_diagonal, coriolis, gravity_term, friction,
            torch.zeros_like(q) if previous_torque is None else previous_torque,
            self.actuator["continuous_torque"], 1.0,
        )
        tokens = self.tokenizer(
            self.record.joint_features.float().expand(count, -1, -1),
            torch.ones((count, q.shape[1]), dtype=torch.bool),
            variable_local=local_history.float(),
            variable_global=global_history.float(),
        )
        return Observation(tokens, dynamics.float(), self.graph.mass_matrix(q).float(), self.flat_state(state))


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
