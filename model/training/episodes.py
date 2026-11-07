"""Turn raw trajectories into per-frame physics features and auxiliary labels.

A trajectory is a recorded or simulated sequence of joint states for one
embodiment. ``process_episode`` computes, once per frame, everything training
needs: the dynamics features, the mass matrix, FK targets, and the labels of
every auxiliary task. The result is cached on disk by the dataset builder, so
the analytic dynamics run once per dataset rather than once per epoch.

Each frame can carry several *action variants*. Variant 0 is what was executed.
Synthetic data adds probe variants: other torques applied at the same state,
which test action conditioning at a fixed pose (task B) and give infeasible
proposals for the motor head (task D).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from model.dynamics import forward_dynamics, frame_velocities, inverse_dynamics
from model.embodiment import EmbodimentGraph
from model.physics import (
    EPSILON,
    ActuatorSpec,
    actuator_features,
    actuator_tensors,
    configuration_terms,
    motor_labels,
    requested_torque,
    rigid_body_features,
    torque_limit,
    velocity_terms,
    assemble_dynamics_features,
)
from model.physics_schema import ACTUATOR_MODES

ACTION_TYPES = ("torque", "position", "velocity")


@dataclass
class EmbodimentRecord:
    """Everything about one embodiment that does not change over time.

    Attributes:
        name (str): Label for logs.
        graph (EmbodimentGraph): Robot graph.
        specs (tuple[ActuatorSpec, ...]): One actuator spec per joint.
        rigid_body_features (torch.Tensor): ``[J, 25]``.
        actuator_features (torch.Tensor): ``[J, 11]``.
        joint_features (torch.Tensor): ``[J, F]`` GET-Zero hardware features.
        effect_mask (torch.Tensor): Bool ``[J]`` end-effector joints for task E.
        acceleration_scale (torch.Tensor): ``[J]`` sigma^a, set from data.
        effect_scales (torch.Tensor): ``[3]`` RMS of the normalized end-effector
            displacement, linear velocity, and angular velocity, set from data.
            Effect targets are divided by them so translation and rotation
            terms have comparable magnitude.
    """

    name: str
    graph: EmbodimentGraph
    specs: tuple[ActuatorSpec, ...]
    rigid_body_features: torch.Tensor
    actuator_features: torch.Tensor
    joint_features: torch.Tensor
    effect_mask: torch.Tensor
    acceleration_scale: torch.Tensor = field(default=None)
    effect_scales: torch.Tensor = field(default=None)

    @classmethod
    def build(
        cls,
        name: str,
        graph: EmbodimentGraph,
        specs: tuple[ActuatorSpec, ...],
        end_effectors: list[str] | None = None,
    ) -> EmbodimentRecord:
        """Compute static features.

        Args:
            name (str): Label.
            graph (EmbodimentGraph): Robot.
            specs (tuple[ActuatorSpec, ...]): Actuator specs.
            end_effectors (list[str] | None): Joint names whose frames are end
                effectors; defaults to the leaf joints.

        Returns:
            EmbodimentRecord: Record without acceleration statistics.
        """
        if end_effectors:
            unknown = sorted(set(end_effectors) - set(graph.joint_names))
            if unknown:
                raise ValueError(f"unknown end-effector joints {unknown}")
            effect_mask = torch.tensor([joint in end_effectors for joint in graph.joint_names])
        else:
            effect_mask = graph.leaf_mask.clone()
        return cls(
            name=name,
            graph=graph,
            specs=tuple(specs),
            rigid_body_features=rigid_body_features(graph),
            actuator_features=actuator_features(specs),
            joint_features=graph.joint_features,
            effect_mask=effect_mask,
        )

    @property
    def actuator(self) -> dict[str, torch.Tensor]:
        """Return actuator spec tensors (float64)."""
        return actuator_tensors(self.specs)

    @property
    def actuated(self) -> torch.Tensor:
        """Return a bool ``[J]`` mask of joints with an actuator."""
        return self.actuator["mode"] != ACTUATOR_MODES.index("passive")


@dataclass
class Trajectory:
    """One episode of joint-space data for one embodiment, in SI units.

    Attributes:
        embodiment (int): Index into the embodiment record list.
        dt (float): Time between frames in s.
        positions (torch.Tensor): ``[T, J]`` q.
        velocities (torch.Tensor | None): ``[T, J]`` qd; finite-differenced
            from positions when None.
        accelerations (torch.Tensor | None): ``[T, J]`` qdd; finite-differenced
            from velocities when None.
        applied_torque (torch.Tensor | None): ``[T, J]`` measured efforts; the
            inverse-dynamics estimate is used when None.
        actions (torch.Tensor | None): ``[T, J]`` commands sent at each frame.
        action_type (str): ``torque``, ``position``, or ``velocity``.
        gravity (torch.Tensor | None): ``[T, 3]`` or ``[3]`` gravity in the root
            frame; -z when None.
        probe_torques (torch.Tensor | None): ``[T, K, J]`` extra torque
            proposals at each state (synthetic data).
        transition_valid (torch.Tensor | None): Bool ``[T]``; False where the
            transition to the next frame is not explained by the dynamics (for
            example a joint-limit impact).
    """

    embodiment: int
    dt: float
    positions: torch.Tensor
    velocities: torch.Tensor | None = None
    accelerations: torch.Tensor | None = None
    applied_torque: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    action_type: str = "position"
    gravity: torch.Tensor | None = None
    probe_torques: torch.Tensor | None = None
    transition_valid: torch.Tensor | None = None


def _difference(values: torch.Tensor, dt: float, wrap: torch.Tensor | None = None) -> torch.Tensor:
    """Central finite difference along time, one-sided at the ends."""
    if values.shape[0] < 2:
        return torch.zeros_like(values)
    forward = values[1:] - values[:-1]
    if wrap is not None:
        wrapped = torch.remainder(forward + math.pi, 2 * math.pi) - math.pi
        forward = torch.where(wrap, wrapped, forward)
    derivative = torch.empty_like(values)
    derivative[0] = forward[0] / dt
    derivative[-1] = forward[-1] / dt
    derivative[1:-1] = (forward[1:] + forward[:-1]) / (2 * dt)
    return derivative


def action_features(
    record: EmbodimentRecord,
    commands: torch.Tensor | None,
    action_type: str,
    requested: torch.Tensor,
) -> torch.Tensor:
    """Encode an action proposal per joint for the motor and effect heads.

    Features: requested effort over peak torque, the normalized position target
    (position commands only), and the velocity target over the speed limit
    (velocity commands only).

    Args:
        record (EmbodimentRecord): Embodiment.
        commands (torch.Tensor | None): ``[..., J]`` raw commands.
        action_type (str): Command type.
        requested (torch.Tensor): ``[..., J]`` requested efforts.

    Returns:
        torch.Tensor: ``[..., J, 3]``.
    """
    actuator = record.actuator
    peak = actuator["peak_torque"].to(requested) + EPSILON
    torque = (requested / peak).clamp(-5, 5)
    zero = torch.zeros_like(requested)
    position = zero
    velocity = zero
    if commands is not None and action_type == "position":
        position = record.graph.normalize_joint_positions(commands.to(requested))
    if commands is not None and action_type == "velocity":
        velocity = commands.to(requested) / (actuator["max_velocity"].to(requested) + EPSILON)
    return torch.stack((torque, position, velocity), dim=-1)


def _requested(
    record: EmbodimentRecord,
    commands: torch.Tensor,
    action_type: str,
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Return the effort a command requests, using the command's type."""
    tensors = dict(record.actuator)
    mode = ACTION_TYPES.index(action_type)
    tensors["mode"] = torch.where(record.actuated, torch.tensor(mode), tensors["mode"])
    return requested_torque(tensors, commands, positions, velocities)


def process_episode(
    record: EmbodimentRecord,
    trajectory: Trajectory,
    feature_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Compute features and labels for every frame and action variant.

    Args:
        record (EmbodimentRecord): Embodiment of the trajectory.
        trajectory (Trajectory): Raw data.
        feature_dtype (torch.dtype): Storage dtype.

    Returns:
        dict[str, torch.Tensor]: Per-frame ``[T, ...]`` and per-variant
            ``[T, V, ...]`` tensors; see the keys below.

    Raises:
        ValueError: Shapes are inconsistent.
    """
    graph = record.graph
    joints = graph.num_joints
    q = trajectory.positions.to(torch.float64)
    if q.ndim != 2 or q.shape[1] != joints:
        raise ValueError(f"positions must be [T, {joints}] for {record.name}")
    steps = q.shape[0]
    dt = trajectory.dt
    continuous = torch.tensor([kind == "continuous" for kind in graph.joint_types])
    qd = (
        trajectory.velocities.to(torch.float64)
        if trajectory.velocities is not None
        else _difference(q, dt, continuous)
    )
    qdd = (
        trajectory.accelerations.to(torch.float64)
        if trajectory.accelerations is not None
        else _difference(qd, dt)
    )
    gravity = trajectory.gravity
    if gravity is not None:
        gravity = gravity.to(torch.float64).expand(steps, 3)
    measured = trajectory.applied_torque is not None
    applied = (
        trajectory.applied_torque.to(torch.float64)
        if measured
        else inverse_dynamics(graph, q, qd, qdd, gravity)
    )
    actuator = record.actuator
    mass_diagonal, gravity_term = configuration_terms(graph, q, gravity)
    coriolis, friction = velocity_terms(graph, q, qd)
    previous = None
    if measured:
        previous = torch.cat((torch.zeros_like(applied[:1]), applied[:-1]))
    dynamics = assemble_dynamics_features(
        mass_diagonal, coriolis, gravity_term, friction, previous, actuator["continuous_torque"]
    )
    mass_matrix = graph.mass_matrix(q)
    poses = graph.forward_kinematics(q)
    linear, angular = frame_velocities(graph, q, qd)
    length = graph.characteristic_length

    # Variant 0: what was executed.
    if trajectory.actions is not None:
        commands0 = trajectory.actions.to(torch.float64)
        requested0 = _requested(record, commands0, trajectory.action_type, q, qd)
        action_type0 = trajectory.action_type
    else:
        commands0, requested0, action_type0 = None, applied, "torque"
    variant_torque = [applied]
    variant_acceleration = [qdd]
    variant_requested = [requested0]
    variant_features = [action_features(record, commands0, action_type0, requested0)]
    next_positions = [q[1:]]
    next_velocities = [qd[1:]]
    valid = torch.ones(steps, dtype=torch.bool)
    valid[-1] = False
    if trajectory.transition_valid is not None:
        valid &= trajectory.transition_valid
    variant_valid = [valid]
    # Probe variants: other torques at the same state, simulated one step.
    if trajectory.probe_torques is not None:
        limit = torque_limit(actuator, qd)
        for probe in trajectory.probe_torques.to(torch.float64).unbind(1):
            executed = torch.maximum(torch.minimum(probe, limit), -limit)
            acceleration = forward_dynamics(graph, q, qd, executed, gravity)
            velocity_next = qd + dt * acceleration
            variant_torque.append(executed)
            variant_acceleration.append(acceleration)
            variant_requested.append(probe)
            variant_features.append(action_features(record, probe, "torque", probe))
            next_velocities.append(velocity_next[:-1])
            next_positions.append((q + dt * velocity_next)[:-1])
            variant_valid.append(torch.ones(steps, dtype=torch.bool))
    torque = torch.stack(variant_torque, dim=1)
    acceleration = torch.stack(variant_acceleration, dim=1)
    requested = torch.stack(variant_requested, dim=1)
    q_variants = q[:, None].expand_as(torque)
    qd_variants = qd[:, None].expand_as(torque)
    gravity_variants = None if gravity is None else gravity[:, None].expand(*torque.shape[:2], 3)
    required = inverse_dynamics(graph, q_variants, qd_variants, acceleration, gravity_variants)
    flags, margins, margin_mask = motor_labels(graph, actuator, q_variants, qd_variants, requested)

    # Effect labels: the next frame of each variant (last frame has none).
    variants = torque.shape[1]
    effect_delta = torch.zeros((steps, variants, joints, 3), dtype=torch.float64)
    effect_rotation = torch.eye(3, dtype=torch.float64).expand(steps, variants, joints, 3, 3).clone()
    effect_linear = torch.zeros_like(effect_delta)
    effect_angular = torch.zeros_like(effect_delta)
    for index, (positions_next, velocities_next) in enumerate(zip(next_positions, next_velocities)):
        if positions_next.shape[0] == 0:
            continue
        poses_next = graph.forward_kinematics(positions_next)
        linear_next, angular_next = frame_velocities(graph, positions_next, velocities_next)
        effect_delta[:-1, index] = (poses_next[..., :3, 3] - poses[:-1, :, :3, 3]) / length
        effect_rotation[:-1, index] = poses_next[..., :3, :3]
        effect_linear[:-1, index] = linear_next / length
        effect_angular[:-1, index] = angular_next
    sigma_source = acceleration
    effect_valid = torch.stack(variant_valid, dim=1)[..., None] & record.effect_mask
    effect_squares = torch.stack(
        [
            (value.pow(2).sum(-1) * effect_valid).sum()
            for value in (effect_delta, effect_linear, effect_angular)
        ]
    )

    def store(value: torch.Tensor) -> torch.Tensor:
        return value.to(feature_dtype) if value.is_floating_point() else value

    gravity_direction = (
        torch.tensor((0.0, 0.0, -1.0), dtype=torch.float64).expand(steps, 3)
        if gravity is None
        else gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
    )
    speed_scale = torch.where(
        actuator["max_velocity"] > 0, actuator["max_velocity"], torch.full((joints,), 10.0, dtype=torch.float64)
    )
    return {
        # Per frame [T, ...]
        "positions_normalized": store(graph.normalize_joint_positions(q)),
        "velocities_normalized": store(qd / speed_scale),
        "gravity_direction": store(gravity_direction),
        "dynamics_features": store(dynamics),
        "mass_matrix": store(mass_matrix),
        "coriolis": store(coriolis),
        "gravity_torque": store(gravity_term),
        "friction_torque": store(friction),
        "fk_position": store(poses[..., :3, 3] / length),
        "fk_rotation": store(poses[..., :3, :3]),
        "torque_measured": torch.full((steps,), measured),
        # Per variant [T, V, ...]
        "applied_torque": store(torque),
        "acceleration": store(acceleration),
        "required_torque": store(required),
        "action_features": store(torch.stack(variant_features, dim=1)),
        "motor_flags": store(flags),
        "motor_margins": store(margins),
        "motor_margin_mask": margin_mask,
        "effect_delta_position": store(effect_delta),
        "effect_rotation": store(effect_rotation),
        "effect_linear_velocity": store(effect_linear),
        "effect_angular_velocity": store(effect_angular),
        "effect_valid": torch.stack(variant_valid, dim=1),
        "_acceleration_rms": (sigma_source.pow(2).mean(dim=(0, 1))).sqrt(),
        "_acceleration_count": torch.tensor(sigma_source.shape[0] * sigma_source.shape[1]),
        "_effect_squares": effect_squares,
        "_effect_count": effect_valid.sum(),
    }


def finalize_acceleration_scales(
    records: list[EmbodimentRecord], episodes: list[tuple[int, dict[str, torch.Tensor]]]
) -> None:
    """Set each record's sigma^a and effect scales from its episodes.

    sigma^a is the RMS joint acceleration over all frames and variants. The
    scale is floored at 1% of the largest joint RMS of the robot (and at
    1e-3) so a joint that barely moves does not blow up the FD loss.

    Args:
        records (list[EmbodimentRecord]): Records to update in place.
        episodes (list): ``(embodiment index, processed episode)`` pairs.
    """
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    effect_sums: dict[int, torch.Tensor] = {}
    effect_counts: dict[int, int] = {}
    for index, episode in episodes:
        count = int(episode["_acceleration_count"])
        mean_square = episode["_acceleration_rms"].pow(2) * count
        sums[index] = sums.get(index, 0) + mean_square
        counts[index] = counts.get(index, 0) + count
        effect_sums[index] = effect_sums.get(index, 0) + episode["_effect_squares"]
        effect_counts[index] = effect_counts.get(index, 0) + int(episode["_effect_count"])
    for index, record in enumerate(records):
        record.effect_scales = torch.ones(3)
        if index not in sums:
            record.acceleration_scale = torch.ones(record.graph.num_joints)
            continue
        rms = (sums[index] / counts[index]).sqrt()
        floor = max(float(rms.max()) * 0.01, 1e-3)
        record.acceleration_scale = rms.clamp_min(floor).to(torch.float32)
        if effect_counts[index]:
            effect_rms = (effect_sums[index] / effect_counts[index]).sqrt()
            record.effect_scales = effect_rms.clamp_min(1e-4).to(torch.float32)


