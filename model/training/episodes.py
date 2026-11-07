"""Turn raw trajectories into per-frame physics features and auxiliary labels.

A trajectory is a recorded or simulated sequence of joint states for one
embodiment, optionally with the state of a floating base. ``process_episode``
computes, once per frame, everything training needs: the dynamics features, the
mass matrix, FK targets, and the labels of every auxiliary task. The dataset
builder caches the result on disk, so the analytic dynamics run once per
dataset rather than once per epoch.

Missing signals are reconstructed:

* joint velocities and accelerations by finite differences;
* the base twist from the base pose, and base accelerations from the twist;
* gravity in the base frame from the base orientation or projected gravity;
* foot contacts from foot height along gravity and foot speed;
* joint torques by inverse dynamics, fixed base or floating base with contact
  forces (``model.dynamics.estimate_joint_efforts``).

Each frame can carry several *action variants*. Variant 0 is what was executed.
Synthetic data adds probe variants: other torques applied at the same state,
which test action conditioning at a fixed pose (task B) and give infeasible
proposals for the motor head (task D).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from model.dynamics import (
    STANDARD_GRAVITY,
    BaseMotion,
    estimate_joint_efforts,
    floating_forward_dynamics,
    forward_dynamics,
    frame_velocities,
    rotor_wrenches,
)
from model.embodiment import EmbodimentGraph
from model.physics import (
    EPSILON,
    ActuatorSpec,
    actuator_features,
    actuator_tensors,
    assemble_dynamics_features,
    configuration_terms,
    motor_labels,
    requested_torque,
    rigid_body_features,
    torque_limit,
    velocity_terms,
)
from model.physics_schema import ACTUATOR_MODES

ACTION_TYPES = ("torque", "position", "velocity")
BASE_FEATURE_SIZE = 11


@dataclass
class EmbodimentRecord:
    """Everything about one embodiment that does not change over time.

    Attributes:
        name (str): Label for logs.
        graph (EmbodimentGraph): Robot graph.
        specs (tuple[ActuatorSpec, ...]): One actuator spec per joint.
        rigid_body_features (torch.Tensor): ``[J, 25]``.
        actuator_features (torch.Tensor): ``[J, 14]``.
        joint_features (torch.Tensor): ``[J, F]`` GET-Zero hardware features.
        effect_mask (torch.Tensor): Bool ``[J]`` end-effector joints for task E.
        contact_links (tuple[str, ...]): Links that can touch the ground.
        surface_contacts (bool | None): Contacts carry moments (flat feet);
            None decides from the number of contact links.
        acceleration_scale (torch.Tensor): ``[J]`` sigma^a, set from data.
        base_acceleration_scale (torch.Tensor): ``[6]`` RMS base linear and
            angular acceleration, set from data (floating bases).
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
    contact_links: tuple[str, ...] = ()
    surface_contacts: bool | None = None
    acceleration_scale: torch.Tensor = field(default=None)
    base_acceleration_scale: torch.Tensor = field(default=None)
    effect_scales: torch.Tensor = field(default=None)

    @classmethod
    def build(
        cls,
        name: str,
        graph: EmbodimentGraph,
        specs: tuple[ActuatorSpec, ...],
        end_effectors: list[str] | None = None,
        contact_links: list[str] | None = None,
        surface_contacts: bool | None = None,
    ) -> EmbodimentRecord:
        """Compute static features.

        Args:
            name (str): Label.
            graph (EmbodimentGraph): Robot.
            specs (tuple[ActuatorSpec, ...]): Actuator specs.
            end_effectors (list[str] | None): Joint names whose frames are end
                effectors; defaults to the leaf joints.
            contact_links (list[str] | None): URDF links that touch the ground.
                For a floating base without a list, links whose name contains
                "foot" are used when there are any.
            surface_contacts (bool | None): See the attribute.

        Returns:
            EmbodimentRecord: Record without data statistics.
        """
        if end_effectors:
            unknown = sorted(set(end_effectors) - set(graph.joint_names))
            if unknown:
                raise ValueError(f"unknown end-effector joints {unknown}")
            effect_mask = torch.tensor([joint in end_effectors for joint in graph.joint_names])
        else:
            effect_mask = graph.leaf_mask.clone()
        if contact_links is None and graph.floating_base:
            contact_links = [link for link in graph.link_names if "foot" in link.lower()]
        if contact_links:
            graph.link_index(contact_links)  # validates the names
        return cls(
            name=name,
            graph=graph,
            specs=tuple(specs),
            rigid_body_features=rigid_body_features(graph),
            actuator_features=actuator_features(specs),
            joint_features=graph.joint_features,
            effect_mask=effect_mask,
            contact_links=tuple(contact_links or ()),
            surface_contacts=surface_contacts,
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
    """One episode of data for one embodiment, in SI units.

    Joint signals are ``[T, J]``. Base signals are optional and only used for a
    floating base; vectors are in the base frame unless noted.

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
        gravity (torch.Tensor | None): ``[T, 3]`` or ``[3]`` gravity in the base
            frame. When None it comes from the base orientation, then from
            projected gravity, then defaults to -z.
        base_orientation (torch.Tensor | None): ``[T, 3, 3]`` world-from-base.
        base_position (torch.Tensor | None): ``[T, 3]`` base origin in world.
        base_linear_velocity (torch.Tensor | None): ``[T, 3]`` base frame.
        base_angular_velocity (torch.Tensor | None): ``[T, 3]`` base frame.
        base_linear_acceleration (torch.Tensor | None): ``[T, 3]`` classical,
            base frame (exact in simulation; else finite-differenced).
        base_angular_acceleration (torch.Tensor | None): ``[T, 3]``.
        projected_gravity (torch.Tensor | None): ``[T, 3]`` unit gravity
            direction in the base frame (an IMU's gravity estimate).
        imu_specific_force (torch.Tensor | None): ``[T, 3]`` accelerometer
            reading in the base frame (m/s^2, gravity included, about 9.81 up
            at rest). Gives the base linear acceleration as ``f + g_base``
            when no acceleration is recorded; the IMU is assumed at the base
            origin.
        contacts (torch.Tensor | None): Bool ``[T, C]`` per contact link;
            detected from kinematics when None.
        probe_torques (torch.Tensor | None): ``[T, K, J]`` extra torque
            proposals at each state (synthetic data).
        transition_valid (torch.Tensor | None): Bool ``[T]``; False where the
            transition to the next frame is not explained by the dynamics.
        base_external_wrench (torch.Tensor | None): ``[T, 6]`` known external
            wrench on the base (synthetic harness), for exact probe labels.
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
    base_orientation: torch.Tensor | None = None
    base_position: torch.Tensor | None = None
    base_linear_velocity: torch.Tensor | None = None
    base_angular_velocity: torch.Tensor | None = None
    base_linear_acceleration: torch.Tensor | None = None
    base_angular_acceleration: torch.Tensor | None = None
    projected_gravity: torch.Tensor | None = None
    imu_specific_force: torch.Tensor | None = None
    contacts: torch.Tensor | None = None
    probe_torques: torch.Tensor | None = None
    transition_valid: torch.Tensor | None = None
    base_external_wrench: torch.Tensor | None = None


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


def _vee_log(rotation: torch.Tensor) -> torch.Tensor:
    """Return the rotation vector (axis times angle) of rotation matrices."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-9, 1 - 1e-9))
    skew = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    factor = torch.where(angle > 1e-6, angle / (2 * torch.sin(angle)), torch.full_like(angle, 0.5))
    return skew * factor[..., None]


def exp_rotation(vector: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrix of a rotation vector (Rodrigues)."""
    angle = vector.norm(dim=-1, keepdim=True)
    axis = vector / angle.clamp_min(1e-12)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    angle = angle[..., None]
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)


@dataclass
class _BaseTrack:
    """Reconstructed floating-base signals of one episode (all in the base frame)."""

    known: bool
    linear_known: bool
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    linear_acceleration: torch.Tensor
    angular_acceleration: torch.Tensor
    pose_known: bool
    orientation: torch.Tensor | None
    position: torch.Tensor | None


def reconstruct_base(
    trajectory: Trajectory, steps: int, gravity: torch.Tensor | None = None
) -> _BaseTrack:
    """Reconstruct the base twist and accelerations from what was recorded.

    Priority: recorded twist, else finite differences of the recorded pose.
    Accelerations use the identity ``a = d v_b/dt + w x v_b`` for the classical
    acceleration expressed in the base frame.

    Args:
        trajectory (Trajectory): Raw data.
        steps (int): Number of frames.
        gravity (torch.Tensor | None): ``[T, 3]`` gravity in the base frame,
            needed to turn accelerometer readings into accelerations.

    Returns:
        _BaseTrack: Base signals; zeros with ``known=False`` if nothing is known.
    """
    dt = trajectory.dt
    zeros = torch.zeros((steps, 3), dtype=torch.float64)
    orientation = trajectory.base_orientation
    position = trajectory.base_position
    if orientation is not None:
        orientation = orientation.to(torch.float64)
    if position is not None:
        position = position.to(torch.float64)
    omega = trajectory.base_angular_velocity
    if omega is None and orientation is not None and steps > 1:
        relative = orientation[:-1].transpose(-1, -2) @ orientation[1:]
        forward = _vee_log(relative) / dt
        omega = torch.cat((forward[:1], 0.5 * (forward[1:] + forward[:-1]), forward[-1:]))
    velocity = trajectory.base_linear_velocity
    if velocity is None and position is not None and orientation is not None:
        world = _difference(position, dt)
        velocity = (orientation.transpose(-1, -2) @ world[..., None]).squeeze(-1)
    known = omega is not None or velocity is not None
    linear_known = velocity is not None
    omega = zeros if omega is None else omega.to(torch.float64)
    velocity = zeros if velocity is None else velocity.to(torch.float64)
    alpha = trajectory.base_angular_acceleration
    alpha = _difference(omega, dt) if alpha is None else alpha.to(torch.float64)
    acceleration = trajectory.base_linear_acceleration
    if acceleration is None and trajectory.imu_specific_force is not None and gravity is not None:
        acceleration = trajectory.imu_specific_force.to(torch.float64) + gravity
        known = True
    if acceleration is None:
        acceleration = _difference(velocity, dt) + torch.linalg.cross(omega, velocity, dim=-1)
    else:
        acceleration = acceleration.to(torch.float64)
    return _BaseTrack(
        known,
        linear_known,
        velocity,
        omega,
        acceleration,
        alpha,
        orientation is not None and position is not None,
        orientation,
        position,
    )


def base_gravity(trajectory: Trajectory, steps: int) -> torch.Tensor:
    """Return ``[T, 3]`` gravity in the base frame from the best available signal."""
    if trajectory.gravity is not None:
        return trajectory.gravity.to(torch.float64).expand(steps, 3)
    world = torch.tensor((0.0, 0.0, -STANDARD_GRAVITY), dtype=torch.float64)
    if trajectory.base_orientation is not None:
        rotation = trajectory.base_orientation.to(torch.float64)
        return (rotation.transpose(-1, -2) @ world[:, None]).squeeze(-1)
    if trajectory.projected_gravity is not None:
        direction = trajectory.projected_gravity.to(torch.float64)
        return STANDARD_GRAVITY * direction / direction.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
    return world.expand(steps, 3)


def detect_contacts(
    record: EmbodimentRecord,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    gravity: torch.Tensor,
    base: _BaseTrack,
    height_tolerance: float = 0.03,
    speed_tolerance: float = 0.5,
) -> torch.Tensor:
    """Guess which contact links touch the ground.

    A foot is in contact when it is within ``height_tolerance`` (m, measured
    along gravity) of the lowest foot and, if the base twist is known, moves
    slower than ``speed_tolerance`` (m/s). The speed test needs the base linear
    velocity and is skipped without it. Flight phases with no slow foot have
    no contacts.

    Args:
        record (EmbodimentRecord): Embodiment with contact links.
        positions (torch.Tensor): ``[T, J]`` q.
        velocities (torch.Tensor): ``[T, J]`` qd.
        gravity (torch.Tensor): ``[T, 3]`` gravity in the base frame.
        base (_BaseTrack): Base signals.
        height_tolerance (float): Height band above the lowest foot.
        speed_tolerance (float): Speed below which a foot can be in stance.

    Returns:
        torch.Tensor: Bool ``[T, C]``.
    """
    graph = record.graph
    links = list(record.contact_links)
    poses = graph.link_poses(positions, links)
    points = poses[..., :3, 3]
    down = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
    depth = (points * down[:, None]).sum(-1)
    lowest = depth.max(dim=-1, keepdim=True).values
    contact = depth >= lowest - height_tolerance
    if base.linear_known:
        linear, angular = frame_velocities(
            graph, positions, velocities, base.linear_velocity, base.angular_velocity
        )
        bodies = graph.link_bodies[graph.link_index(links)].tolist()
        frames = graph.forward_kinematics(positions)
        speeds = []
        for index, body in enumerate(bodies):
            point = points[:, index]
            if body < 0:
                velocity = base.linear_velocity + torch.linalg.cross(base.angular_velocity, point, dim=-1)
            else:
                lever = point - frames[:, body, :3, 3]
                velocity = linear[:, body] + torch.linalg.cross(angular[:, body], lever, dim=-1)
            speeds.append(velocity.norm(dim=-1))
        contact &= torch.stack(speeds, dim=-1) < speed_tolerance
    return contact


def action_features(
    record: EmbodimentRecord,
    commands: torch.Tensor | None,
    action_type: str,
    requested: torch.Tensor,
) -> torch.Tensor:
    """Encode an action proposal per joint for the motor and effect heads.

    Features: requested effort over peak torque, the normalized position target
    (position commands only), and the velocity target over the speed limit
    (velocity and rotor-speed commands).

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
    """Return the effort a command requests, using the command's type.

    Rotor (thrust) joints keep their own mode: their commands are speeds.
    """
    tensors = dict(record.actuator)
    mode = ACTION_TYPES.index(action_type)
    keep = (tensors["mode"] == ACTUATOR_MODES.index("passive")) | (
        tensors["mode"] == ACTUATOR_MODES.index("thrust")
    )
    tensors["mode"] = torch.where(keep, tensors["mode"], torch.tensor(mode))
    return requested_torque(tensors, commands, positions, velocities)


def _effect_targets(
    graph: EmbodimentGraph,
    poses: torch.Tensor,
    positions_next: torch.Tensor,
    velocities_next: torch.Tensor,
    base_next: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express next-step joint frames in the base frame at time t.

    Args:
        graph (EmbodimentGraph): Robot.
        poses (torch.Tensor): ``[N, J, 4, 4]`` frames at t.
        positions_next (torch.Tensor): ``[N, J]`` q at t+1.
        velocities_next (torch.Tensor): ``[N, J]`` qd at t+1.
        base_next (tuple | None): Relative base rotation ``R_t^T R_{t+1}``
            ``[N, 3, 3]``, displacement ``R_t^T (x_{t+1} - x_t)`` ``[N, 3]``, and
            the base twist at t+1 in its own frame; None for a still base.

    Returns:
        tuple: Displacement, rotation at t+1, linear and angular velocity at
            t+1, all in the base frame at t (unnormalized).
    """
    linear_twist = None if base_next is None else base_next[2]
    angular_twist = None if base_next is None else base_next[3]
    poses_next = graph.forward_kinematics(positions_next)
    linear, angular = frame_velocities(graph, positions_next, velocities_next, linear_twist, angular_twist)
    position_next = poses_next[..., :3, 3]
    rotation_next = poses_next[..., :3, :3]
    if base_next is not None:
        relative, shift = base_next[0], base_next[1]
        position_next = (relative[:, None] @ position_next[..., None]).squeeze(-1) + shift[:, None]
        rotation_next = relative[:, None] @ rotation_next
        linear = (relative[:, None] @ linear[..., None]).squeeze(-1)
        angular = (relative[:, None] @ angular[..., None]).squeeze(-1)
    return position_next - poses[..., :3, 3], rotation_next, linear, angular


def process_episode(
    record: EmbodimentRecord,
    trajectory: Trajectory,
    feature_dtype: torch.dtype = torch.float32,
    use_estimated_previous_torque: bool = True,
    contact_height_tolerance: float = 0.03,
    contact_speed_tolerance: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute features and labels for every frame and action variant.

    Args:
        record (EmbodimentRecord): Embodiment of the trajectory.
        trajectory (Trajectory): Raw data.
        feature_dtype (torch.dtype): Storage dtype.
        use_estimated_previous_torque (bool): Put the inverse-dynamics
            estimate of ``tau_{t-1}`` in the dynamics features (flag 0.5) when
            no torque is measured; otherwise it is zero with flag 0.
        contact_height_tolerance (float): See ``detect_contacts``.
        contact_speed_tolerance (float): See ``detect_contacts``.

    Returns:
        dict[str, torch.Tensor]: Per-frame ``[T, ...]`` and per-variant
            ``[T, V, ...]`` tensors.

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
    gravity = base_gravity(trajectory, steps)
    floating = graph.floating_base
    base = reconstruct_base(trajectory, steps, gravity)
    if not floating:
        zeros = torch.zeros((steps, 3), dtype=torch.float64)
        base = _BaseTrack(False, False, zeros, zeros, zeros, zeros, False, None, None)
    motion = BaseMotion(
        base.linear_velocity, base.angular_velocity, base.linear_acceleration, base.angular_acceleration
    )
    actuator = record.actuator
    contacts = None
    if floating and record.contact_links:
        contacts = (
            trajectory.contacts.to(torch.bool)
            if trajectory.contacts is not None
            else detect_contacts(
                record, q, qd, gravity, base, contact_height_tolerance, contact_speed_tolerance
            )
        )

    def estimate(accelerations: torch.Tensor, base_motion: BaseMotion):
        return estimate_joint_efforts(
            graph,
            q,
            qd,
            accelerations,
            gravity,
            base_motion,
            record.contact_links if contacts is not None else None,
            contacts,
            actuator,
            surface_contacts=record.surface_contacts,
            base_external_wrench=trajectory.base_external_wrench,
        )

    measured = trajectory.applied_torque is not None
    first_estimate = estimate(qdd, motion)
    applied = trajectory.applied_torque.to(torch.float64) if measured else first_estimate.efforts
    residual = first_estimate.residual_wrench.norm(dim=-1)
    has_thrust = bool((actuator["mode"] == ACTUATOR_MODES.index("thrust")).any())
    if not record.contact_links and not has_thrust:
        # No contact model: the base is held by something not modelled (a
        # stand, a harness, legs the dataset does not record). The joint
        # efforts are still right; the base wrench is simply that support.
        residual = torch.zeros_like(residual)
    mass_diagonal, gravity_term = configuration_terms(graph, q, gravity)
    coriolis, friction = velocity_terms(
        graph, q, qd, base.linear_velocity if floating else None,
        base.angular_velocity if floating else None, actuator,
    )
    previous, source = None, 1.0
    if measured or use_estimated_previous_torque:
        previous = torch.cat((torch.zeros_like(applied[:1]), applied[:-1]))
        source = 1.0 if measured else 0.5
    dynamics = assemble_dynamics_features(
        mass_diagonal, coriolis, gravity_term, friction, previous, actuator["continuous_torque"], source
    )
    mass_matrix = graph.mass_matrix(q)
    poses = graph.forward_kinematics(q)
    length = graph.characteristic_length

    # Variant 0: what was executed.
    if trajectory.actions is not None:
        commands0 = trajectory.actions.to(torch.float64)
        requested0 = _requested(record, commands0, trajectory.action_type, q, qd)
        action_type0 = trajectory.action_type
    else:
        commands0, requested0, action_type0 = None, applied, "torque"
    base_acceleration0 = torch.cat((base.linear_acceleration, base.angular_acceleration), dim=-1)
    variant_torque = [applied]
    variant_acceleration = [qdd]
    variant_base_acceleration = [base_acceleration0]
    variant_requested = [requested0]
    variant_features = [action_features(record, commands0, action_type0, requested0)]
    valid = torch.ones(steps, dtype=torch.bool)
    valid[-1] = False
    if trajectory.transition_valid is not None:
        valid &= trajectory.transition_valid
    variant_valid = [valid]
    next_states = []  # per variant: (q_next, qd_next, base_next) for frames 0..T-2
    if floating and base.known:
        if base.pose_known:
            relative = base.orientation[:-1].transpose(-1, -2) @ base.orientation[1:]
            shift = (base.orientation[:-1].transpose(-1, -2) @ (base.position[1:] - base.position[:-1])[..., None]).squeeze(-1)
        else:
            relative = exp_rotation(base.angular_velocity[:-1] * dt)
            shift = base.linear_velocity[:-1] * dt
        base_next0 = (relative, shift, base.linear_velocity[1:], base.angular_velocity[1:])
    else:
        base_next0 = None
    next_states.append((q[1:], qd[1:], base_next0))
    # Probe variants: other torques at the same state, simulated one step.
    if trajectory.probe_torques is not None:
        limit = torque_limit(actuator, qd)
        wrenches = rotor_wrenches(graph, actuator, q, qd)
        for probe in trajectory.probe_torques.to(torch.float64).unbind(1):
            executed = torch.maximum(torch.minimum(probe, limit), -limit)
            if floating:
                linear_acc, angular_acc, acceleration = floating_forward_dynamics(
                    graph, q, qd, executed, gravity, base.linear_velocity, base.angular_velocity,
                    wrenches, trajectory.base_external_wrench,
                )
                omega_next = base.angular_velocity + dt * angular_acc
                velocity_next_base = base.linear_velocity + dt * (
                    linear_acc - torch.linalg.cross(base.angular_velocity, base.linear_velocity, dim=-1)
                )
                relative = exp_rotation(omega_next * dt)
                shift = velocity_next_base * dt
                base_next = (relative[:-1], shift[:-1], velocity_next_base[:-1], omega_next[:-1])
                base_acc = torch.cat((linear_acc, angular_acc), dim=-1)
            else:
                acceleration = forward_dynamics(graph, q, qd, executed, gravity, body_wrenches=wrenches)
                base_next, base_acc = None, torch.zeros_like(base_acceleration0)
            velocity_next = qd + dt * acceleration
            variant_torque.append(executed)
            variant_acceleration.append(acceleration)
            variant_base_acceleration.append(base_acc)
            variant_requested.append(probe)
            variant_features.append(action_features(record, probe, "torque", probe))
            next_states.append(((q + dt * velocity_next)[:-1], velocity_next[:-1], base_next))
            variant_valid.append(torch.ones(steps, dtype=torch.bool))
    torque = torch.stack(variant_torque, dim=1)
    acceleration = torch.stack(variant_acceleration, dim=1)
    base_acceleration = torch.stack(variant_base_acceleration, dim=1)
    requested = torch.stack(variant_requested, dim=1)
    variants = torque.shape[1]
    q_variants = q[:, None].expand_as(torque)
    qd_variants = qd[:, None].expand_as(torque)
    # Required torque (task C) for each variant's joint and base acceleration.
    required = []
    for index in range(variants):
        variant_motion = BaseMotion(
            base.linear_velocity, base.angular_velocity,
            base_acceleration[:, index, :3], base_acceleration[:, index, 3:],
        )
        required.append(estimate(acceleration[:, index], variant_motion).efforts)
    required = torch.stack(required, dim=1)
    flags, margins, margin_mask = motor_labels(graph, actuator, q_variants, qd_variants, requested)

    # Effect labels: next frame of each variant, in the base frame at t.
    effect_delta = torch.zeros((steps, variants, joints, 3), dtype=torch.float64)
    effect_rotation = torch.eye(3, dtype=torch.float64).expand(steps, variants, joints, 3, 3).clone()
    effect_linear = torch.zeros_like(effect_delta)
    effect_angular = torch.zeros_like(effect_delta)
    for index, (positions_next, velocities_next, base_next) in enumerate(next_states):
        if positions_next.shape[0] == 0:
            continue
        delta, rotation_next, linear_next, angular_next = _effect_targets(
            graph, poses[:-1], positions_next, velocities_next, base_next
        )
        effect_delta[:-1, index] = delta / length
        effect_rotation[:-1, index] = rotation_next
        effect_linear[:-1, index] = linear_next / length
        effect_angular[:-1, index] = angular_next
    effect_valid = torch.stack(variant_valid, dim=1)[..., None] & record.effect_mask
    effect_squares = torch.stack(
        [
            (value.pow(2).sum(-1) * effect_valid).sum()
            for value in (effect_delta, effect_linear, effect_angular)
        ]
    )

    def store(value: torch.Tensor) -> torch.Tensor:
        return value.to(feature_dtype) if value.is_floating_point() else value

    gravity_direction = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
    speed_scale = torch.where(
        actuator["max_velocity"] > 0, actuator["max_velocity"], torch.full((joints,), 10.0, dtype=torch.float64)
    )
    base_features = torch.cat(
        (
            gravity_direction,
            base.angular_velocity,
            base.linear_velocity / length,
            torch.full((steps, 1), 1.0 if base.known else 0.0, dtype=torch.float64),
            torch.full((steps, 1), 1.0 if floating else 0.0, dtype=torch.float64),
        ),
        dim=-1,
    )
    base_known = torch.full((steps, variants), bool(floating and base.known))
    if trajectory.probe_torques is not None and floating:
        base_known[:, 1:] = True
    return {
        # Per frame [T, ...]
        "positions_normalized": store(graph.normalize_joint_positions(q)),
        "velocities_normalized": store(qd / speed_scale),
        "base_features": store(base_features),
        "dynamics_features": store(dynamics),
        "mass_matrix": store(mass_matrix),
        "coriolis": store(coriolis),
        "gravity_torque": store(gravity_term),
        "friction_torque": store(friction),
        "fk_position": store(poses[..., :3, 3] / length),
        "fk_rotation": store(poses[..., :3, :3]),
        "torque_measured": torch.full((steps,), measured),
        "estimate_residual": store(residual),
        # Per variant [T, V, ...]
        "applied_torque": store(torque),
        "acceleration": store(acceleration),
        "base_acceleration": store(base_acceleration),
        "base_acceleration_known": base_known,
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
        "_acceleration_rms": (acceleration.pow(2).mean(dim=(0, 1))).sqrt(),
        "_acceleration_count": torch.tensor(steps * variants),
        "_base_acceleration_squares": (base_acceleration.pow(2) * base_known[..., None]).sum(dim=(0, 1)),
        "_base_acceleration_count": base_known.sum(),
        "_effect_squares": effect_squares,
        "_effect_count": effect_valid.sum(),
    }


def finalize_acceleration_scales(
    records: list[EmbodimentRecord], episodes: list[tuple[int, dict[str, torch.Tensor]]]
) -> None:
    """Set each record's sigma^a, base acceleration, and effect scales from data.

    sigma^a is the RMS joint acceleration over all frames and variants, floored
    at 1% of the robot's largest joint RMS (and at 1e-3) so a joint that barely
    moves does not blow up the FD loss.

    Args:
        records (list[EmbodimentRecord]): Records to update in place.
        episodes (list): ``(embodiment index, processed episode)`` pairs.
    """
    totals: dict[int, dict[str, torch.Tensor | int]] = {}
    for index, episode in episodes:
        entry = totals.setdefault(index, {"acc": 0, "acc_n": 0, "base": 0, "base_n": 0, "eff": 0, "eff_n": 0})
        count = int(episode["_acceleration_count"])
        entry["acc"] = entry["acc"] + episode["_acceleration_rms"].pow(2) * count
        entry["acc_n"] += count
        entry["base"] = entry["base"] + episode["_base_acceleration_squares"]
        entry["base_n"] += int(episode["_base_acceleration_count"])
        entry["eff"] = entry["eff"] + episode["_effect_squares"]
        entry["eff_n"] += int(episode["_effect_count"])
    for index, record in enumerate(records):
        record.effect_scales = torch.ones(3)
        record.base_acceleration_scale = torch.ones(6)
        entry = totals.get(index)
        if entry is None:
            record.acceleration_scale = torch.ones(record.graph.num_joints)
            continue
        rms = (entry["acc"] / max(entry["acc_n"], 1)).sqrt()
        floor = max(float(rms.max()) * 0.01, 1e-3)
        record.acceleration_scale = rms.clamp_min(floor).to(torch.float32)
        if entry["base_n"]:
            base_rms = (entry["base"] / entry["base_n"]).sqrt()
            record.base_acceleration_scale = base_rms.clamp_min(1e-2).to(torch.float32)
        if entry["eff_n"]:
            effect_rms = (entry["eff"] / entry["eff_n"]).sqrt()
            record.effect_scales = effect_rms.clamp_min(1e-4).to(torch.float32)
