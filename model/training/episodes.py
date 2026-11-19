"""Turn raw trajectories into per-frame physics features and auxiliary labels.

A trajectory is a recorded or simulated sequence of joint states for one
embodiment, optionally with the state of a floating base. ``process_episode``
computes, once per frame, everything training needs: the dynamics features, the
mass matrix, FK targets, the contact-constrained response, and the labels of every
auxiliary task. The dataset builder caches the result on disk, so the analytic
dynamics run once per dataset rather than once per epoch.

Missing signals are reconstructed:

* joint velocities and accelerations by finite differences;
* the base twist from the base pose, and base accelerations from the twist;
* gravity in the base frame from the base orientation or projected gravity;
* foot contacts from foot height along gravity and foot speed.

Joint torques are never estimated for training. Inverse-dynamics estimates from
kinematics are 43-97% off on real robots (``docs/dynamics.md``), so a dataset
without torque sensors gets no torque labels and no torque inputs. Its dynamics
supervision is the contact-constrained response ``G(q)``, which is exact for the
model and needs no torque. ``estimate_episode_efforts`` keeps the estimator for
analysis.

Each frame can carry several *action variants*. Variant 0 is what was executed.
Simulated data adds probe variants: other torques applied at the same state, which
test action conditioning at a fixed pose (task B) and give infeasible proposals for
the motor head (task D). Other data adds proposal variants for the motor head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from model.dynamics import (
    STANDARD_GRAVITY,
    BaseMotion,
    constrained_response,
    determinate_torque_projector,
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
# Where a variant's torque comes from. Only "measured" and "exact" torques enter
# training; nothing is estimated.
TORQUE_SOURCES = ("none", "measured", "exact")


@dataclass
class EmbodimentRecord:
    """Everything about one embodiment that does not change over time.

    Attributes:
        name (str): Label for logs.
        graph (EmbodimentGraph): Robot graph.
        specs (tuple[ActuatorSpec, ...]): One actuator spec per joint.
        rigid_body_features (torch.Tensor): ``[J, 26]``.
        actuator_features (torch.Tensor): ``[J, 6]``.
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
        response_scale (torch.Tensor): ``[6 + J]`` RMS acceleration change from a
            full-peak torque change (base rows, then joints), set from data. The
            response targets are divided by it.
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
    response_scale: torch.Tensor = field(default=None)

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
        applied_torque (torch.Tensor | None): ``[T, J]`` efforts from a torque
            sensor, or the exact applied torque of a simulator (with
            ``exact_dynamics``). None means unknown: the episode then has no torque
            labels and no torque inputs (nothing is estimated).
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
            wrench on the base (synthetic harness, pushes), for exact labels.
        exact_dynamics (bool): The data come from a simulator whose applied
            torques produce the recorded accelerations exactly (with contacts),
            so the torques are exact labels (torque source ``exact``).
        probe_executed (torch.Tensor | None): ``[T, K, J]`` executed probe
            torques computed by a simulator (with contacts).
        probe_accelerations (torch.Tensor | None): ``[T, K, J]`` joint
            accelerations of the probes from the simulator.
        probe_base_accelerations (torch.Tensor | None): ``[T, K, 6]`` base
            linear and angular accelerations of the probes (base frame).
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
    exact_dynamics: bool = False
    probe_executed: torch.Tensor | None = None
    probe_accelerations: torch.Tensor | None = None
    probe_base_accelerations: torch.Tensor | None = None


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
        speeds = contact_point_speeds(graph, positions, velocities, base.linear_velocity, base.angular_velocity, links)
        contact &= speeds < speed_tolerance
    return contact


def contact_point_speeds(
    graph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    links: list[str],
) -> torch.Tensor:
    """Return the speed in the world of each contact link's frame origin, ``[T, C]`` in m/s.

    The origin is the point the contact constraints hold still (``model.dynamics.contact_maps``),
    so its speed measures how far a frame is from the "contacts hold" assumption of the
    response labels.

    Args:
        graph (EmbodimentGraph): Robot with a floating base.
        positions (torch.Tensor): ``[T, J]`` q.
        velocities (torch.Tensor): ``[T, J]`` qd.
        base_linear_velocity (torch.Tensor): ``[T, 3]`` base frame.
        base_angular_velocity (torch.Tensor): ``[T, 3]`` base frame.
        links (list[str]): Contact links.

    Returns:
        torch.Tensor: ``[T, C]`` speeds.
    """
    return contact_point_velocities(graph, positions, velocities, base_linear_velocity, base_angular_velocity, links).norm(dim=-1)


def contact_point_velocities(
    graph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    links: list[str],
) -> torch.Tensor:
    """Return the world velocity of each contact link's frame origin in the base frame, ``[T, C, 3]``.

    With a zero base linear velocity this is the velocity relative to the base origin's
    motion, which is the same for every foot that stands still in the world.
    """
    points = graph.link_poses(positions, links)[..., :3, 3]
    linear, angular = frame_velocities(graph, positions, velocities, base_linear_velocity, base_angular_velocity)
    bodies = graph.link_bodies[graph.link_index(links)].tolist()
    frames = graph.forward_kinematics(positions)
    result = []
    for index, body in enumerate(bodies):
        point = points[:, index]
        if body < 0:
            velocity = base_linear_velocity + torch.linalg.cross(base_angular_velocity, point, dim=-1)
        else:
            lever = point - frames[:, body, :3, 3]
            velocity = linear[:, body] + torch.linalg.cross(angular[:, body], lever, dim=-1)
        result.append(velocity)
    return torch.stack(result, dim=-2)


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


def _contact_state(
    record: EmbodimentRecord,
    trajectory: Trajectory,
    q: torch.Tensor,
    qd: torch.Tensor,
    gravity: torch.Tensor,
    base: _BaseTrack,
    height_tolerance: float,
    speed_tolerance: float,
    method: str = "height",
    accelerations: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, bool, torch.Tensor | None, torch.Tensor | None]:
    """Return the contact flags ``[T, C]`` (None without contact links), whether they were
    given, and, for the ``dynamics`` method, the unexplained base wrench in N ``[T]`` and the
    selection confidence ``[T]`` (else None for both)."""
    from model.training.contacts import CONTACT_DETECTION_METHODS, select_contacts

    if method not in CONTACT_DETECTION_METHODS:
        raise ValueError(f"contact_detection must be one of {CONTACT_DETECTION_METHODS}, not {method!r}")
    if not (record.graph.floating_base and record.contact_links):
        return None, False, None, None
    if trajectory.contacts is not None:
        return trajectory.contacts.to(torch.bool), True, None, None
    if method == "dynamics" and len(record.contact_links) <= 8:
        contacts, relative, confidence = select_contacts(
            record, q, qd, accelerations, gravity, base, record.actuator, trajectory.base_external_wrench,
            height_tolerance=height_tolerance,
        )
        weight = record.graph.total_mass * gravity.norm(dim=-1)
        return contacts, False, relative * weight, confidence
    # More than eight feet would mean too many subsets: fall back to the height rule.
    return detect_contacts(record, q, qd, gravity, base, height_tolerance, speed_tolerance), False, None, None


def _low_feet(record: EmbodimentRecord, positions: torch.Tensor, gravity: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Return bool ``[T, C]``: feet within ``tolerance`` of the lowest foot along gravity."""
    points = record.graph.link_poses(positions, list(record.contact_links))[..., :3, 3]
    down = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
    depth = (points * down[:, None]).sum(-1)
    return depth >= depth.max(dim=-1, keepdim=True).values - tolerance


def _stable_contacts(contacts: torch.Tensor, margin: int) -> torch.Tensor:
    """Return bool ``[T]``: no contact changes within ``margin`` frames of each frame."""
    steps = contacts.shape[0]
    stable = torch.ones(steps, dtype=torch.bool)
    if steps < 2 or margin <= 0:
        return stable
    changes = torch.nonzero((contacts[1:] != contacts[:-1]).any(-1)).squeeze(-1)
    for change in changes.tolist():
        # The change happens between frames ``change`` and ``change + 1``.
        stable[max(change - margin + 1, 0) : min(change + margin + 1, steps)] = False
    return stable


def estimate_episode_efforts(
    record: EmbodimentRecord,
    trajectory: Trajectory,
    contact_height_tolerance: float = 0.03,
    contact_speed_tolerance: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Estimate joint torques of an episode by inverse dynamics, for analysis only.

    Training never uses these estimates (see ``docs/dynamics.md``): they are 43-97%
    off on real robots. This function exists to measure that error against datasets
    with torque sensors (``scripts/validate_torque_estimation.py``) and to test the
    estimator.

    Args:
        record (EmbodimentRecord): Robot.
        trajectory (Trajectory): Episode.
        contact_height_tolerance (float): See ``detect_contacts``.
        contact_speed_tolerance (float): See ``detect_contacts``.

    Returns:
        dict[str, torch.Tensor]: ``efforts`` ``[T, J]``, ``projection`` ``[T, J, J]``
            (the motion-determined part, ``determinate_torque_projector``),
            ``residual`` ``[T]`` (unexplained base wrench in N), and ``contacts``
            ``[T, C]`` (empty without contact links).
    """
    graph = record.graph
    q = trajectory.positions.to(torch.float64)
    steps, joints = q.shape
    dt = trajectory.dt
    continuous = torch.tensor([kind == "continuous" for kind in graph.joint_types])
    qd = trajectory.velocities.to(torch.float64) if trajectory.velocities is not None else _difference(q, dt, continuous)
    qdd = trajectory.accelerations.to(torch.float64) if trajectory.accelerations is not None else _difference(qd, dt)
    gravity = base_gravity(trajectory, steps)
    base = reconstruct_base(trajectory, steps, gravity)
    if not graph.floating_base:
        zeros = torch.zeros((steps, 3), dtype=torch.float64)
        base = _BaseTrack(False, False, zeros, zeros, zeros, zeros, False, None, None)
    contacts, _, _, _ = _contact_state(record, trajectory, q, qd, gravity, base, contact_height_tolerance, contact_speed_tolerance)
    estimate = estimate_joint_efforts(
        graph, q, qd, qdd, gravity,
        BaseMotion(base.linear_velocity, base.angular_velocity, base.linear_acceleration, base.angular_acceleration),
        record.contact_links if contacts is not None else None, contacts, record.actuator,
        surface_contacts=record.surface_contacts, base_external_wrench=trajectory.base_external_wrench,
    )
    projection = torch.eye(joints, dtype=torch.float64).expand(steps, joints, joints)
    if contacts is not None and bool(contacts.any()):
        surface = record.surface_contacts if record.surface_contacts is not None else len(record.contact_links) <= 2
        projection, _ = determinate_torque_projector(graph, q, list(record.contact_links), contacts, surface)
    return {
        "efforts": estimate.efforts,
        "projection": projection,
        "residual": estimate.residual_wrench.norm(dim=-1),
        "contacts": contacts if contacts is not None else torch.zeros((steps, 0), dtype=torch.bool),
    }


def _proposals(record: EmbodimentRecord, steps: int, count: int, seed: int) -> torch.Tensor:
    """Return ``[T, K, J]`` random torque proposals: random sign, log-uniform 2%-200% of peak."""
    generator = torch.Generator().manual_seed(seed)
    peak = record.actuator["peak_torque"]
    shape = (steps, count, record.graph.num_joints)
    fraction = torch.exp(torch.empty(shape, dtype=torch.float64).uniform_(math.log(0.02), math.log(2.0), generator=generator))
    sign = torch.randint(0, 2, shape, generator=generator).to(torch.float64) * 2 - 1
    return sign * fraction * peak


def process_episode(
    record: EmbodimentRecord,
    trajectory: Trajectory,
    feature_dtype: torch.dtype = torch.float32,
    contact_height_tolerance: float = 0.03,
    contact_speed_tolerance: float = 0.5,
    proposal_variants: int = 2,
    contact_margin_steps: int = 2,
    seed: int = 0,
    max_contact_slip: float | None = 0.1,
    contact_detection: str = "dynamics",
    detected_contact_response: bool = True,
    min_contact_confidence: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Compute features and labels for every frame and action variant, without torque estimates.

    Torque enters only where it is known exactly or measured: a simulator's applied
    torque (``exact_dynamics``), a dataset's torque sensor (``applied_torque``), or a
    command converted by the known control law. Datasets without either get no
    torque labels and no torque inputs; their dynamics supervision is the
    contact-constrained response ``G(q)`` (``model.dynamics.constrained_response``),
    which needs no torque.

    Variants of each frame:

    * variant 0: what was executed. Torque source ``exact`` (simulation),
      ``measured`` (torque sensor) or ``none``.
    * probe variants (simulated data): other torques at the same state, simulated one
      step, all ``exact``.
    * proposal variants (data without probes): random torque proposals at the
      recorded state, for the motor-feasibility head only.

    Args:
        record (EmbodimentRecord): Embodiment of the trajectory.
        trajectory (Trajectory): Raw data.
        feature_dtype (torch.dtype): Storage dtype.
        contact_height_tolerance (float): See ``detect_contacts``.
        contact_speed_tolerance (float): See ``detect_contacts``.
        proposal_variants (int): Proposal variants per frame of data without probes.
        contact_margin_steps (int): Frames before and after a contact change whose
            response labels are dropped (the contact set is uncertain there).
        seed (int): Seed of the proposals.
        max_contact_slip (float | None): Response labels are dropped in frames where a
            foot in contact moves faster than this (m/s): the contact does not hold there,
            and against MuJoCo the rigid response's median error grows from about 2%
            below 0.1 m/s to 9% at 0.1-0.2 m/s and over 20% above. Needs the base linear
            velocity; None turns the check off.
        contact_detection (str): How contacts are inferred when the data has none:
            ``dynamics`` (default) or ``height`` (``model.training.contacts``).
        detected_contact_response (bool): Give response labels on frames whose contacts
            were inferred (False: only sensor or simulator contacts give them).
        min_contact_confidence (float): With ``dynamics`` detection, response labels are
            kept only where the chosen foot set beats the next best by this cost margin
            (``select_contacts``).

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
    if trajectory.exact_dynamics and trajectory.applied_torque is None:
        raise ValueError(f"{record.name}: exact_dynamics needs the applied torque")
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
    actuator = record.actuator
    has_thrust = bool((actuator["mode"] == ACTUATOR_MODES.index("thrust")).any())
    contacts, contacts_given, selection_residual, confidence = _contact_state(
        record, trajectory, q, qd, gravity, base, contact_height_tolerance, contact_speed_tolerance,
        contact_detection, qdd,
    )
    surface = record.surface_contacts if record.surface_contacts is not None else len(record.contact_links) <= 2
    links = list(record.contact_links)

    # Torque source of what was executed.
    if trajectory.exact_dynamics:
        source0 = TORQUE_SOURCES.index("exact")
    elif trajectory.applied_torque is not None:
        source0 = TORQUE_SOURCES.index("measured")
    else:
        source0 = TORQUE_SOURCES.index("none")
    known0 = source0 != TORQUE_SOURCES.index("none")
    applied = trajectory.applied_torque.to(torch.float64) if known0 else torch.zeros_like(q)

    # Contact-set checks. Contacts from a sensor or a simulator are trusted; detected
    # contacts are checked by how well they explain the base wrench the motion needs.
    # This check uses the inverse-dynamics machinery, but only its residual: no
    # estimated torque is kept.
    residual = torch.zeros(steps, dtype=torch.float64)
    if selection_residual is not None:
        residual = selection_residual  # pushing-only forces in friction pyramids
    elif contacts is not None and not contacts_given:
        residual = estimate_joint_efforts(
            graph, q, qd, qdd, gravity,
            BaseMotion(base.linear_velocity, base.angular_velocity, base.linear_acceleration, base.angular_acceleration),
            links, contacts, actuator, surface_contacts=record.surface_contacts,
            base_external_wrench=trajectory.base_external_wrench,
        ).residual_wrench.norm(dim=-1)

    mass_diagonal, gravity_term = configuration_terms(graph, q, gravity)
    coriolis, friction = velocity_terms(
        graph, q, qd, base.linear_velocity if floating else None,
        base.angular_velocity if floating else None, actuator,
    )
    dynamics = assemble_dynamics_features(mass_diagonal, coriolis, gravity_term, friction, actuator["continuous_torque"])
    mass_matrix = graph.mass_matrix(q)
    poses = graph.forward_kinematics(q)
    length = graph.characteristic_length

    # Contact-constrained response G(q) and the torque directions it determines.
    response = constrained_response(graph, q, links if contacts is not None else (), contacts, surface)
    projection = torch.eye(joints, dtype=torch.float64).expand(steps, joints, joints)
    if contacts is not None and bool(contacts.any()):
        projection, _ = determinate_torque_projector(graph, q, links, contacts, surface)
    # The response needs to know what holds the base. A floating robot without contact
    # links, rotors or a recorded support wrench hangs from something unmodelled.
    support_known = (
        not floating or has_thrust or bool(links) or trajectory.base_external_wrench is not None
    )
    response_valid = torch.full((steps,), support_known)
    slipping = torch.zeros(steps, dtype=torch.bool)
    if contacts is not None:
        response_valid &= _stable_contacts(contacts, contact_margin_steps)
        if not contacts_given and not detected_contact_response:
            response_valid[:] = False
        if confidence is not None:
            response_valid &= confidence > min_contact_confidence
        if max_contact_slip is not None and floating and base.linear_known:
            speeds = contact_point_speeds(graph, q, qd, base.linear_velocity, base.angular_velocity, links)
            # Inferred contacts may leave out a sliding foot; every foot near the ground counts.
            checked = contacts if contacts_given else contacts | _low_feet(record, q, gravity, contact_height_tolerance)
            slipping = (checked & (speeds > max_contact_slip)).any(dim=-1)
            response_valid &= ~slipping

    # Variant 0: what was executed.
    if trajectory.actions is not None:
        commands0 = trajectory.actions.to(torch.float64)
        requested0 = _requested(record, commands0, trajectory.action_type, q, qd)
        features0 = action_features(record, commands0, trajectory.action_type, requested0)
    elif known0:
        requested0 = applied
        features0 = action_features(record, None, "torque", applied)
    else:
        # Nothing about the action is known: zero action features, as at the first
        # frame of an episode. The effect head then predicts the next frame from the
        # state alone, and the motor head skips this variant.
        requested0 = torch.zeros_like(q)
        features0 = torch.zeros((steps, joints, 3), dtype=torch.float64)
    base_acceleration0 = torch.cat((base.linear_acceleration, base.angular_acceleration), dim=-1)
    valid = torch.ones(steps, dtype=torch.bool)
    valid[-1] = False
    if trajectory.transition_valid is not None:
        valid &= trajectory.transition_valid
    variant_torque = [applied]
    variant_acceleration = [qdd]
    variant_base_acceleration = [base_acceleration0]
    variant_requested = [requested0]
    variant_features = [features0]
    variant_valid = [valid]
    variant_source = [torch.full((steps,), source0, dtype=torch.int8)]
    variant_motor = [torch.full((steps,), trajectory.actions is not None or known0)]
    variant_base_known = [torch.full((steps,), bool(floating and base.known))]
    next_states: list = []  # per variant: (q_next, qd_next, base_next) for frames 0..T-2, or None
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
    # Probe variants: other torques at the same state, simulated one step (exact).
    if trajectory.probe_torques is not None:
        limit = torque_limit(actuator, qd)
        wrenches = rotor_wrenches(graph, actuator, q, qd)
        for probe_index, probe in enumerate(trajectory.probe_torques.to(torch.float64).unbind(1)):
            executed = torch.maximum(torch.minimum(probe, limit), -limit)
            if trajectory.probe_accelerations is not None:
                # Simulator probes (with contacts): use its executed torque and accelerations.
                executed = trajectory.probe_executed[:, probe_index].to(torch.float64)
                acceleration = trajectory.probe_accelerations[:, probe_index].to(torch.float64)
                base_acc = (
                    trajectory.probe_base_accelerations[:, probe_index].to(torch.float64)
                    if trajectory.probe_base_accelerations is not None
                    else torch.zeros_like(base_acceleration0)
                )
                linear_acc, angular_acc = base_acc[:, :3], base_acc[:, 3:]
                if floating:
                    omega_next = base.angular_velocity + dt * angular_acc
                    velocity_next_base = base.linear_velocity + dt * (
                        linear_acc - torch.linalg.cross(base.angular_velocity, base.linear_velocity, dim=-1)
                    )
                    base_next = (exp_rotation(omega_next * dt)[:-1], (velocity_next_base * dt)[:-1],
                                 velocity_next_base[:-1], omega_next[:-1])
                else:
                    base_next = None
            elif floating:
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
            variant_source.append(torch.full((steps,), TORQUE_SOURCES.index("exact"), dtype=torch.int8))
            variant_motor.append(torch.ones(steps, dtype=torch.bool))
            variant_base_known.append(torch.full((steps,), bool(floating)))
    elif proposal_variants > 0:
        # Proposal variants: feasibility of random torques at the recorded state.
        for proposal in _proposals(record, steps, proposal_variants, seed).unbind(1):
            variant_torque.append(torch.zeros_like(q))
            variant_acceleration.append(torch.zeros_like(q))
            variant_base_acceleration.append(torch.zeros_like(base_acceleration0))
            variant_requested.append(proposal)
            variant_features.append(action_features(record, proposal, "torque", proposal))
            next_states.append(None)
            variant_valid.append(torch.zeros(steps, dtype=torch.bool))
            variant_source.append(torch.full((steps,), TORQUE_SOURCES.index("none"), dtype=torch.int8))
            variant_motor.append(torch.ones(steps, dtype=torch.bool))
            variant_base_known.append(torch.zeros(steps, dtype=torch.bool))
    torque = torch.stack(variant_torque, dim=1)
    acceleration = torch.stack(variant_acceleration, dim=1)
    base_acceleration = torch.stack(variant_base_acceleration, dim=1)
    requested = torch.stack(variant_requested, dim=1)
    source = torch.stack(variant_source, dim=1)
    labeled = source != TORQUE_SOURCES.index("none")
    variants = torque.shape[1]
    q_variants = q[:, None].expand_as(torque)
    qd_variants = qd[:, None].expand_as(torque)
    # Inverse-dynamics labels (task C): the torque that produced each labeled
    # variant's acceleration. With contacts, the loss keeps only its
    # motion-determined part (``contact_projection``).
    required = torch.where(labeled[..., None], torque, torch.zeros_like(torque))
    flags, margins, margin_mask = motor_labels(graph, actuator, q_variants, qd_variants, requested)

    # Effect labels: next frame of each variant, in the base frame at t.
    effect_delta = torch.zeros((steps, variants, joints, 3), dtype=torch.float64)
    effect_rotation = torch.eye(3, dtype=torch.float64).expand(steps, variants, joints, 3, 3).clone()
    effect_linear = torch.zeros_like(effect_delta)
    effect_angular = torch.zeros_like(effect_delta)
    for index, entry in enumerate(next_states):
        if entry is None or entry[0].shape[0] == 0:
            continue
        positions_next, velocities_next, base_next = entry
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
    base_known = torch.stack(variant_base_known, dim=1)
    peak = torch.where(record.actuated, actuator["peak_torque"], torch.zeros_like(actuator["peak_torque"]))
    response_gain = (response.pow(2) * peak.pow(2)).sum(-1)  # [T, 6 + J]
    labeled_count = labeled.sum()

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
    return {
        # Per frame [T, ...]
        "positions_normalized": store(graph.normalize_joint_positions(q)),
        "velocities_normalized": store(qd / speed_scale),
        "base_features": store(base_features),
        "dynamics_features": store(dynamics),
        "mass_matrix": store(mass_matrix),
        "fk_position": store(poses[..., :3, 3] / length),
        "fk_rotation": store(poses[..., :3, :3]),
        "contact_projection": store(projection),
        "response_operator": store(response),
        "response_valid": response_valid,
        "contact_residual": store(residual),
        "contacts_detected": torch.full((steps,), contacts is not None and not contacts_given),
        "contact_slipping": slipping,
        "contact_flags": contacts if contacts is not None else torch.zeros((steps, 0), dtype=torch.bool),
        "floating_base": torch.full((steps,), floating),
        # Per variant [T, V, ...]
        "torque_source": source,
        "applied_torque": store(torque),
        "acceleration": store(acceleration),
        "base_acceleration": store(base_acceleration),
        "base_acceleration_known": base_known,
        "required_torque": store(required),
        "action_features": store(torch.stack(variant_features, dim=1)),
        "motor_valid": torch.stack(variant_motor, dim=1),
        "motor_flags": store(flags),
        "motor_margins": store(margins),
        "motor_margin_mask": margin_mask,
        "effect_delta_position": store(effect_delta),
        "effect_rotation": store(effect_rotation),
        "effect_linear_velocity": store(effect_linear),
        "effect_angular_velocity": store(effect_angular),
        "effect_valid": torch.stack(variant_valid, dim=1),
        # Statistics for finalize_acceleration_scales.
        "_acceleration_squares": (acceleration.pow(2) * labeled[..., None]).sum(dim=(0, 1)),
        "_acceleration_count": labeled_count,
        "_base_acceleration_squares": (base_acceleration.pow(2) * (base_known & labeled)[..., None]).sum(dim=(0, 1)),
        "_base_acceleration_count": (base_known & labeled).sum(),
        "_effect_squares": effect_squares,
        "_effect_count": effect_valid.sum(),
        "_response_squares": (response_gain * response_valid[:, None]).sum(0),
        "_response_count": response_valid.sum(),
    }


def finalize_acceleration_scales(
    records: list[EmbodimentRecord], episodes: list[tuple[int, dict[str, torch.Tensor]]]
) -> None:
    """Set each record's acceleration, base acceleration, effect, and response scales from data.

    * ``acceleration_scale`` sigma^a: RMS joint acceleration over variants with torque
      labels, floored at 1% of the robot's largest joint RMS (and at 1e-3) so a joint
      that barely moves does not blow up the FD loss; ones without torque labels.
    * ``response_scale``: RMS of ``sqrt(sum_k G_rk^2 tau_k^peak^2)`` over valid
      frames, the typical acceleration change from a full-peak torque change on every
      joint. It normalizes the response targets (6 base rows, then joints).

    Args:
        records (list[EmbodimentRecord]): Records to update in place.
        episodes (list): ``(embodiment index, processed episode)`` pairs.
    """
    totals: dict[int, dict[str, torch.Tensor | int]] = {}
    for index, episode in episodes:
        entry = totals.setdefault(
            index, {"acc": 0, "acc_n": 0, "base": 0, "base_n": 0, "eff": 0, "eff_n": 0, "res": 0, "res_n": 0}
        )
        entry["acc"] = entry["acc"] + episode["_acceleration_squares"]
        entry["acc_n"] += int(episode["_acceleration_count"])
        entry["base"] = entry["base"] + episode["_base_acceleration_squares"]
        entry["base_n"] += int(episode["_base_acceleration_count"])
        entry["eff"] = entry["eff"] + episode["_effect_squares"]
        entry["eff_n"] += int(episode["_effect_count"])
        entry["res"] = entry["res"] + episode["_response_squares"]
        entry["res_n"] += int(episode["_response_count"])
    for index, record in enumerate(records):
        joints = record.graph.num_joints
        record.effect_scales = torch.ones(3)
        record.base_acceleration_scale = torch.ones(6)
        record.acceleration_scale = torch.ones(joints)
        record.response_scale = torch.ones(6 + joints)
        entry = totals.get(index)
        if entry is None:
            continue
        if entry["acc_n"]:
            rms = (entry["acc"] / entry["acc_n"]).sqrt()
            floor = max(float(rms.max()) * 0.01, 1e-3)
            record.acceleration_scale = rms.clamp_min(floor).to(torch.float32)
        if entry["base_n"]:
            base_rms = (entry["base"] / entry["base_n"]).sqrt()
            record.base_acceleration_scale = base_rms.clamp_min(1e-2).to(torch.float32)
        if entry["eff_n"]:
            effect_rms = (entry["eff"] / entry["eff_n"]).sqrt()
            record.effect_scales = effect_rms.clamp_min(1e-4).to(torch.float32)
        if entry["res_n"]:
            response_rms = (entry["res"] / entry["res_n"]).sqrt()
            joint_rms = response_rms[6:]
            joint_floor = max(float(joint_rms.max()) * 0.01, 1e-3) if joint_rms.numel() else 1e-3
            base_rms = response_rms[:6].clamp_min(1e-2)
            record.response_scale = torch.cat((base_rms, joint_rms.clamp_min(joint_floor))).to(torch.float32)
