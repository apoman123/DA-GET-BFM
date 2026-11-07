"""Rigid-body dynamics of an embodiment graph, for fixed and floating bases.

All quantities use SI units and are expressed in the root (base) link frame:

* Joint positions are rad (revolute, continuous) or m (prismatic); velocities
  rad/s or m/s; accelerations rad/s^2 or m/s^2.
* Joint efforts are N m for revolute joints and N for prismatic joints.
* ``gravity`` is the gravitational acceleration in the base frame, in m/s^2.
  For a robot standing upright it is ``(0, 0, -9.81)``; for a floating base use
  ``R_world_base^T g_world`` (the IMU's projected gravity times 9.81).
* ``BaseMotion`` holds the base twist and acceleration in the base frame: the
  absolute linear velocity and classical (not spatial) linear acceleration of
  the base origin, and the absolute angular velocity and acceleration. All
  zero gives the fixed-base case.

Equations of motion. With ``nu = (v_base, omega_base, qd)``:

    M_full(q) nu_dot + h(q, nu) = [W_ext; tau] + sum_c J_c^T f_c

``recursive_newton_euler`` returns both blocks of the left-hand side minus any
known external body wrenches: the joint efforts ``tau`` and the base wrench
``W`` the environment must supply (contacts, a mount, thrust that was not
modelled). For a fixed base, ``tau`` is the familiar
``M(q) qdd + c(q, qd) + g(q)``. ``estimate_joint_efforts`` handles floating
bases with ground contact: it distributes ``W`` over the active contact points
by least squares and removes the contact forces from the joint efforts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from model.embodiment import EmbodimentGraph

STANDARD_GRAVITY = 9.80665
FRICTION_VELOCITY_SCALE = 1e-2


@dataclass(frozen=True)
class BaseMotion:
    """Twist and acceleration of a floating base, in the base frame.

    Attributes:
        linear_velocity (torch.Tensor | None): ``[..., 3]`` m/s of the base origin.
        angular_velocity (torch.Tensor | None): ``[..., 3]`` rad/s.
        linear_acceleration (torch.Tensor | None): ``[..., 3]`` m/s^2, the
            classical acceleration of the base origin (rotated world-frame
            ``d^2 p / dt^2``), without gravity.
        angular_acceleration (torch.Tensor | None): ``[..., 3]`` rad/s^2.
    """

    linear_velocity: torch.Tensor | None = None
    angular_velocity: torch.Tensor | None = None
    linear_acceleration: torch.Tensor | None = None
    angular_acceleration: torch.Tensor | None = None


@dataclass(frozen=True)
class RneaResult:
    """Output of the recursive Newton-Euler algorithm.

    Attributes:
        efforts (torch.Tensor): ``[..., J]`` joint efforts.
        base_wrench (torch.Tensor): ``[..., 6]`` force and moment about the base
            origin, in the base frame, that the environment must apply to the
            base to produce the given motion.
        linear_velocity (torch.Tensor): ``[..., J, 3]`` velocity of each joint
            frame origin.
        angular_velocity (torch.Tensor): ``[..., J, 3]`` angular velocity of
            each joint body.
    """

    efforts: torch.Tensor
    base_wrench: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor


def default_gravity(dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    """Return gravity along -z of the base frame.

    Args:
        dtype (torch.dtype): Floating dtype.
        device (torch.device | str | None): Device.

    Returns:
        torch.Tensor: ``[3]`` gravity vector in m/s^2.
    """
    return torch.tensor((0.0, 0.0, -STANDARD_GRAVITY), dtype=dtype, device=device)


def _check_joint_tensor(name: str, value: torch.Tensor, positions: torch.Tensor) -> None:
    """Require a joint-space tensor with the shape of positions."""
    if value.shape != positions.shape:
        raise ValueError(f"{name} must have the same shape as positions")


def _vector(value: torch.Tensor | None, positions: torch.Tensor) -> torch.Tensor:
    """Broadcast an optional 3-vector to the batch shape of positions."""
    batch_shape = positions.shape[:-1]
    if value is None:
        return torch.zeros((*batch_shape, 3), dtype=positions.dtype, device=positions.device)
    value = value.to(dtype=positions.dtype, device=positions.device)
    if value.shape[-1:] != (3,):
        raise ValueError("base and gravity vectors must end with dimension 3")
    return value.expand(*batch_shape, 3)


def _gravity(gravity: torch.Tensor | None, positions: torch.Tensor) -> torch.Tensor:
    """Broadcast gravity to the batch shape of positions."""
    if gravity is None:
        gravity = default_gravity(positions.dtype, positions.device)
    return _vector(gravity, positions)


def recursive_newton_euler(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    gravity: torch.Tensor | None = None,
    base: BaseMotion | None = None,
    body_wrenches: torch.Tensor | None = None,
    base_external_wrench: torch.Tensor | None = None,
) -> RneaResult:
    """Run the recursive Newton-Euler algorithm on the tree.

    Friction is not included (see ``friction_efforts``).

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        accelerations (torch.Tensor): ``[..., J]`` joint accelerations.
        gravity (torch.Tensor | None): ``[3]`` or ``[..., 3]`` gravity in the base
            frame; defaults to ``(0, 0, -9.81)``. Pass zeros to drop gravity.
        base (BaseMotion | None): Base twist and acceleration; zero (fixed base)
            when None.
        body_wrenches (torch.Tensor | None): ``[..., J, 6]`` known external
            force and moment (about the body's joint origin) acting on each
            joint body, in the base frame, such as rotor thrust or contacts.
        base_external_wrench (torch.Tensor | None): ``[..., 6]`` known external
            wrench on the base body about the base origin.

    Returns:
        RneaResult: Joint efforts, required base wrench, and frame velocities.

    Raises:
        ValueError: Shapes do not match.
    """
    _check_joint_tensor("velocities", velocities, positions)
    _check_joint_tensor("accelerations", accelerations, positions)
    frames, _, axes = graph._kinematics(positions)
    velocities = velocities.to(positions)
    accelerations = accelerations.to(positions)
    dtype, device = positions.dtype, positions.device
    base = base or BaseMotion()
    gravity = _gravity(gravity, positions)
    base_omega = _vector(base.angular_velocity, positions)
    base_alpha = _vector(base.angular_acceleration, positions)
    base_velocity = _vector(base.linear_velocity, positions)
    base_acceleration = _vector(base.linear_acceleration, positions) - gravity
    zero = torch.zeros_like(base_omega)
    masses = graph.body_mass.to(dtype=dtype, device=device)
    coms = graph.body_com.to(dtype=dtype, device=device)
    inertias = graph.body_inertia.to(dtype=dtype, device=device)
    parents = graph.parent_indices.tolist()
    order = graph._depth_order()
    prismatic = [kind == "prismatic" for kind in graph.joint_types]
    count = graph.num_joints
    omega: list[torch.Tensor] = [zero] * count
    alpha: list[torch.Tensor] = [zero] * count
    linear_velocity: list[torch.Tensor] = [zero] * count
    linear_acceleration: list[torch.Tensor] = [zero] * count
    force: list[torch.Tensor] = [zero] * count
    moment: list[torch.Tensor] = [zero] * count
    origins = frames[..., :3, 3]
    cross = torch.linalg.cross
    if body_wrenches is not None:
        body_wrenches = body_wrenches.to(positions)
    for index in order:
        parent = parents[index]
        if parent < 0:
            parent_omega, parent_alpha = base_omega, base_alpha
            parent_velocity, parent_acceleration = base_velocity, base_acceleration
            parent_origin = zero
        else:
            parent_omega, parent_alpha = omega[parent], alpha[parent]
            parent_velocity = linear_velocity[parent]
            parent_acceleration = linear_acceleration[parent]
            parent_origin = origins[..., parent, :]
        lever = origins[..., index, :] - parent_origin
        axis = axes[..., index, :]
        speed = velocities[..., index, None]
        rate = accelerations[..., index, None]
        velocity = parent_velocity + cross(parent_omega, lever, dim=-1)
        acceleration = (
            parent_acceleration
            + cross(parent_alpha, lever, dim=-1)
            + cross(parent_omega, cross(parent_omega, lever, dim=-1), dim=-1)
        )
        if prismatic[index]:
            velocity = velocity + axis * speed
            acceleration = (
                acceleration
                + axis * rate
                + 2 * cross(parent_omega, axis * speed, dim=-1)
            )
            omega[index], alpha[index] = parent_omega, parent_alpha
        else:
            omega[index] = parent_omega + axis * speed
            alpha[index] = (
                parent_alpha + axis * rate + cross(parent_omega, axis * speed, dim=-1)
            )
        linear_velocity[index] = velocity
        linear_acceleration[index] = acceleration
        rotation = frames[..., index, :3, :3]
        offset = (rotation @ coms[index, :, None]).squeeze(-1)
        inertia = rotation @ inertias[index] @ rotation.transpose(-1, -2)
        body_force, body_moment = _body_wrench(
            masses[index], offset, inertia, acceleration, omega[index], alpha[index]
        )
        if body_wrenches is not None:
            body_force = body_force - body_wrenches[..., index, :3]
            body_moment = body_moment - body_wrenches[..., index, 3:]
        force[index], moment[index] = body_force, body_moment
    efforts: list[torch.Tensor] = [zero[..., 0]] * count
    base_force, base_moment = _body_wrench(
        torch.tensor(graph.base_mass, dtype=dtype, device=device),
        graph.base_com.to(dtype=dtype, device=device).expand_as(zero),
        graph.base_inertia.to(dtype=dtype, device=device),
        base_acceleration,
        base_omega,
        base_alpha,
    )
    for index in reversed(order):
        axis = axes[..., index, :]
        efforts[index] = (
            (axis * force[index]).sum(-1)
            if prismatic[index]
            else (axis * moment[index]).sum(-1)
        )
        parent = parents[index]
        if parent >= 0:
            lever = origins[..., index, :] - origins[..., parent, :]
            force[parent] = force[parent] + force[index]
            moment[parent] = (
                moment[parent] + moment[index] + cross(lever, force[index], dim=-1)
            )
        else:
            lever = origins[..., index, :]
            base_force = base_force + force[index]
            base_moment = base_moment + moment[index] + cross(lever, force[index], dim=-1)
    base_wrench = torch.cat((base_force, base_moment), dim=-1)
    if base_external_wrench is not None:
        base_wrench = base_wrench - base_external_wrench.to(positions)
    return RneaResult(
        torch.stack(efforts, dim=-1),
        base_wrench,
        torch.stack(linear_velocity, dim=-2),
        torch.stack(omega, dim=-2),
    )


def _body_wrench(mass, offset, inertia, acceleration, omega, alpha):
    """Newton-Euler wrench of one body about its frame origin."""
    cross = torch.linalg.cross
    com_acceleration = (
        acceleration + cross(alpha, offset, dim=-1) + cross(omega, cross(omega, offset, dim=-1), dim=-1)
    )
    force = mass * com_acceleration
    spin = (inertia @ omega[..., None]).squeeze(-1)
    moment = (
        (inertia @ alpha[..., None]).squeeze(-1)
        + cross(omega, spin, dim=-1)
        + cross(offset, force, dim=-1)
    )
    return force, moment


def gravity_efforts(
    graph: EmbodimentGraph, positions: torch.Tensor, gravity: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute ``g(q)``, the joint efforts that hold the limbs against gravity.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        gravity (torch.Tensor | None): Gravity in the base frame.

    Returns:
        torch.Tensor: ``[..., J]`` efforts.
    """
    zero = torch.zeros_like(positions)
    return recursive_newton_euler(graph, positions, zero, zero, gravity).efforts


def coriolis_efforts(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``c(q, qd)``, the Coriolis and centrifugal joint efforts.

    For a floating base, pass the base twist; the result then includes the
    velocity-product terms caused by base rotation.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        base_linear_velocity (torch.Tensor | None): ``[..., 3]`` base frame.
        base_angular_velocity (torch.Tensor | None): ``[..., 3]`` base frame.

    Returns:
        torch.Tensor: ``[..., J]`` efforts, zero when every velocity is zero.
    """
    no_gravity = torch.zeros(3, dtype=positions.dtype, device=positions.device)
    base = BaseMotion(base_linear_velocity, base_angular_velocity)
    return recursive_newton_euler(
        graph, positions, velocities, torch.zeros_like(positions), no_gravity, base
    ).efforts


def friction_efforts(
    graph: EmbodimentGraph,
    velocities: torch.Tensor,
    velocity_scale: float = FRICTION_VELOCITY_SCALE,
) -> torch.Tensor:
    """Estimate joint friction from URDF damping and Coulomb friction.

    ``tau_f = damping * qd + friction * tanh(qd / velocity_scale)``; the tanh is
    a smooth sign function, so the estimate is differentiable at rest.

    Args:
        graph (EmbodimentGraph): Robot.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        velocity_scale (float): Velocity at which Coulomb friction saturates.

    Returns:
        torch.Tensor: ``[..., J]`` friction efforts opposing motion.
    """
    damping = graph.damping.to(velocities)
    friction = graph.friction.to(velocities)
    return damping * velocities + friction * torch.tanh(velocities / velocity_scale)


def rotor_wrenches(
    graph: EmbodimentGraph,
    actuator: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor | None:
    """Aerodynamic wrenches of propellers on thrust-mode joints.

    A rotor spinning at ``w`` produces thrust ``k_f w^2`` along its joint axis
    (``k_f`` is signed: negative pushes along -axis) and a drag moment
    ``-k_m w |w|`` about the axis that opposes the spin.

    Args:
        graph (EmbodimentGraph): Robot.
        actuator (Mapping[str, torch.Tensor]): ``model.physics.actuator_tensors``.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities (rotor speeds).

    Returns:
        torch.Tensor | None: ``[..., J, 6]`` body wrenches in the base frame, or
            None if the robot has no thrust actuators.
    """
    from model.physics_schema import ACTUATOR_MODES

    thrust_mode = actuator["mode"].to(positions.device) == ACTUATOR_MODES.index("thrust")
    if not bool(thrust_mode.any()):
        return None
    _, _, axes = graph._kinematics(positions)
    speed = velocities.to(positions)
    thrust = actuator["thrust_coefficient"].to(positions) * speed.pow(2)
    drag = -actuator["torque_coefficient"].to(positions) * speed * speed.abs()
    thrust = torch.where(thrust_mode, thrust, torch.zeros_like(thrust))
    drag = torch.where(thrust_mode, drag, torch.zeros_like(drag))
    return torch.cat((axes * thrust[..., None], axes * drag[..., None]), dim=-1)


def _joint_accelerations(positions: torch.Tensor, accelerations: torch.Tensor | None) -> torch.Tensor:
    return torch.zeros_like(positions) if accelerations is None else accelerations.to(positions)


def floating_mass_matrix(graph: EmbodimentGraph, positions: torch.Tensor) -> torch.Tensor:
    """Compute the floating-base mass matrix ``M_full(q)``.

    Coordinates are ``(base linear acceleration, base angular acceleration,
    qdd)`` in the base frame, so the matrix is ``(6 + J) x (6 + J)``. Its
    joint block equals ``graph.mass_matrix(q)``.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.

    Returns:
        torch.Tensor: Symmetric positive-definite ``[..., 6 + J, 6 + J]``.
    """
    joints = graph.num_joints
    size = 6 + joints
    batch_shape = positions.shape[:-1]
    eye = torch.eye(size, dtype=positions.dtype, device=positions.device)
    eye = eye.expand(*batch_shape, size, size)
    q = positions[..., None, :].expand(*batch_shape, size, joints)
    zero = torch.zeros_like(q)
    result = recursive_newton_euler(
        graph,
        q,
        zero,
        eye[..., 6:],
        torch.zeros(3, dtype=positions.dtype, device=positions.device),
        BaseMotion(linear_acceleration=eye[..., :3], angular_acceleration=eye[..., 3:6]),
    )
    matrix = torch.cat((result.base_wrench, result.efforts), dim=-1).transpose(-1, -2)
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def floating_forward_dynamics(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    efforts: torch.Tensor,
    gravity: torch.Tensor | None = None,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
    body_wrenches: torch.Tensor | None = None,
    base_external_wrench: torch.Tensor | None = None,
    include_friction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the floating-base equations of motion for all accelerations.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` q.
        velocities (torch.Tensor): ``[..., J]`` qd.
        efforts (torch.Tensor): ``[..., J]`` applied joint efforts.
        gravity (torch.Tensor | None): Gravity in the base frame.
        base_linear_velocity (torch.Tensor | None): ``[..., 3]`` base frame.
        base_angular_velocity (torch.Tensor | None): ``[..., 3]`` base frame.
        body_wrenches (torch.Tensor | None): Known external body wrenches
            (thrust, contacts), see ``recursive_newton_euler``.
        base_external_wrench (torch.Tensor | None): Known wrench on the base.
        include_friction (bool): Subtract joint friction.

    Returns:
        tuple: Base linear acceleration ``[..., 3]`` (classical, base frame),
            base angular acceleration ``[..., 3]``, and ``qdd`` ``[..., J]``.
    """
    _check_joint_tensor("efforts", efforts, positions)
    bias = recursive_newton_euler(
        graph,
        positions,
        velocities,
        torch.zeros_like(positions),
        gravity,
        BaseMotion(base_linear_velocity, base_angular_velocity),
        body_wrenches,
        base_external_wrench,
    )
    joint_bias = bias.efforts
    if include_friction:
        joint_bias = joint_bias + friction_efforts(graph, velocities.to(positions))
    rhs = torch.cat((-bias.base_wrench, efforts.to(positions) - joint_bias), dim=-1)
    solution = torch.linalg.solve(floating_mass_matrix(graph, positions), rhs[..., None]).squeeze(-1)
    return solution[..., :3], solution[..., 3:6], solution[..., 6:]


def distribute_contact_forces(
    wrench: torch.Tensor,
    points: torch.Tensor,
    active: torch.Tensor,
    surface: bool | torch.Tensor = False,
    moment_scale: float = 0.1,
    normal: torch.Tensor | None = None,
    tangential_weight: float = 10.0,
    iterations: int = 3,
    regularization: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find contact forces (and moments) that best produce a wrench.

    Solves the weighted least-norm problem

        min sum_c f_n,c^2 + w_t |f_t,c|^2 + |m_c / s|^2
        s.t. sum f_c = F,  sum p_c x f_c + m_c = N

    where ``f_n`` is the force along the ground normal and ``f_t`` the
    tangential part. Penalizing tangential forces (``w_t > 1``) gives the
    mostly vertical ground reactions of real feet instead of large sideways
    forces. Contacts that would pull on the ground (negative normal force) are
    dropped and the problem is solved again (``iterations`` active-set
    passes). Point contacts (quadruped feet) carry only forces; surface
    contacts (flat humanoid feet) also carry a moment ``m_c``. Friction cones
    are not enforced, so the result is an approximation.

    Args:
        wrench (torch.Tensor): ``[..., 6]`` required force and moment about the
            base origin.
        points (torch.Tensor): ``[..., C, 3]`` contact points in the base frame.
        active (torch.Tensor): Bool ``[..., C]`` contacts in use.
        surface (bool | torch.Tensor): Whether each contact can carry a moment;
            a bool for all contacts or a bool ``[C]`` tensor.
        moment_scale (float): Length in m that trades contact moments against
            forces in the objective (about half a foot length).
        normal (torch.Tensor | None): ``[..., 3]`` unit ground normal in the base
            frame (opposite to gravity); without it all directions are weighted
            equally and no contact is dropped.
        tangential_weight (float): ``w_t``.
        iterations (int): Active-set passes that drop pulling contacts.
        regularization (float): Tikhonov term relative to the matrix scale.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Forces and moments ``[..., C, 3]``;
            zero for inactive contacts (moments are zero for point contacts).
    """
    count = points.shape[-2]
    dtype, device = points.dtype, points.device
    eye = torch.eye(3, dtype=dtype, device=device)
    x, y, z = points.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    surface = torch.as_tensor(surface, device=device).expand(count)
    if normal is None:
        inverse_weight = eye.expand_as(skew)
    else:
        n = normal.to(points) / normal.to(points).norm(dim=-1, keepdim=True).clamp_min(1e-12)
        outer = n[..., None, :, None] * n[..., None, None, :]
        inverse_weight = outer + (eye - outer) / tangential_weight
        inverse_weight = inverse_weight.expand_as(skew)
    force_blocks = torch.cat((eye.expand_as(skew), skew), dim=-2)  # [..., C, 6, 3]
    moment_blocks = torch.cat((torch.zeros_like(skew), moment_scale * eye.expand_as(skew)), dim=-2)
    batch_dims = force_blocks.ndim - 3
    current = active.clone()
    for _ in range(max(iterations, 1)):
        weight = current[..., None, None].to(dtype)
        blocks = torch.cat(
            (
                force_blocks * weight,
                moment_blocks * weight * surface[:, None, None].to(dtype),
            ),
            dim=-1,
        )  # [..., C, 6, 6]
        # Inverse objective weight per contact: tangential forces are costlier.
        metric = torch.zeros((*inverse_weight.shape[:-2], 6, 6), dtype=dtype, device=device)
        metric[..., :3, :3] = inverse_weight
        metric[..., 3:, 3:] = eye
        matrix = blocks.permute(*range(batch_dims), batch_dims + 1, batch_dims, batch_dims + 2).reshape(
            *points.shape[:-2], 6, 6 * count
        )
        weighted = (blocks @ metric).permute(*range(batch_dims), batch_dims + 1, batch_dims, batch_dims + 2).reshape(
            *points.shape[:-2], 6, 6 * count
        )
        gram = weighted @ matrix.transpose(-1, -2)
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12)
        gram = gram + regularization * scale[..., None] * torch.eye(6, dtype=dtype, device=device)
        solution = weighted.transpose(-1, -2) @ torch.linalg.solve(gram, wrench.to(points)[..., None])
        solution = solution.squeeze(-1).reshape(*points.shape[:-2], count, 6)
        forces = solution[..., :3] * current[..., None].to(dtype)
        if normal is None:
            break
        pulling = (forces * n[..., None, :]).sum(-1) < 0
        if not bool((pulling & current).any()):
            break
        still = current & ~pulling
        # Keep at least the original set where every contact would pull.
        current = torch.where(still.any(dim=-1, keepdim=True), still, current)
    moments = moment_scale * solution[..., 3:] * (current[..., None] & surface[:, None]).to(dtype)
    return forces, moments


@dataclass(frozen=True)
class EffortEstimate:
    """Joint efforts estimated by inverse dynamics.

    Attributes:
        efforts (torch.Tensor): ``[..., J]`` joint efforts, including friction.
        contact_forces (torch.Tensor | None): ``[..., C, 3]`` estimated forces.
        contact_moments (torch.Tensor | None): ``[..., C, 3]`` estimated moments
            of surface contacts.
        residual_wrench (torch.Tensor): ``[..., 6]`` part of the required base
            wrench that the contacts (or thrust) could not explain. Large values
            flag frames where the estimate is unreliable.
    """

    efforts: torch.Tensor
    contact_forces: torch.Tensor | None
    contact_moments: torch.Tensor | None
    residual_wrench: torch.Tensor


def estimate_joint_efforts(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    gravity: torch.Tensor | None = None,
    base: BaseMotion | None = None,
    contact_links: Sequence[str] | None = None,
    contact_active: torch.Tensor | None = None,
    actuator: Mapping[str, torch.Tensor] | None = None,
    include_friction: bool = True,
    surface_contacts: bool | None = None,
    base_external_wrench: torch.Tensor | None = None,
) -> EffortEstimate:
    """Estimate the joint efforts that produced a motion (no torque sensors).

    * Fixed base: inverse dynamics ``M qdd + c + g + tau_f``.
    * Floating base: run RNEA with the base motion, giving the base wrench the
      environment must supply. Known rotor thrust is applied as an external
      force. The remaining wrench is distributed over the active contact
      points (minimum-norm least squares), and the contact forces are removed
      from the joint efforts. With no active contact (flight), whatever the
      thrust cannot explain is reported as the residual.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` q.
        velocities (torch.Tensor): ``[..., J]`` qd.
        accelerations (torch.Tensor): ``[..., J]`` qdd.
        gravity (torch.Tensor | None): Gravity in the base frame.
        base (BaseMotion | None): Base twist and acceleration. For a floating
            base without base measurements, None assumes a quasi-static base.
        contact_links (Sequence[str] | None): URDF links that can touch the
            ground (feet). Their frame origins are the contact points.
        contact_active (torch.Tensor | None): Bool ``[..., C]`` contact state;
            all contacts are active when None.
        actuator (Mapping[str, torch.Tensor] | None): Actuator tensors, needed
            for rotor thrust.
        include_friction (bool): Add joint friction.
        surface_contacts (bool | None): Whether contacts carry moments (flat
            feet). None uses surface contacts when there are at most two
            contact links (bipeds), since two points cannot balance a moment
            about the line between them, and point contacts otherwise.
        base_external_wrench (torch.Tensor | None): ``[..., 6]`` known wrench
            on the base besides contacts (for example a harness).

    Returns:
        EffortEstimate: Efforts, contact forces and moments, and the
            unexplained wrench.
    """
    wrenches = None if actuator is None else rotor_wrenches(graph, actuator, positions, velocities)
    first = recursive_newton_euler(
        graph, positions, velocities, accelerations, gravity, base, wrenches, base_external_wrench
    )
    efforts, residual, forces, moments = first.efforts, first.base_wrench, None, None
    if graph.floating_base and contact_links:
        poses = graph.link_poses(positions, contact_links)
        points = poses[..., :3, 3]
        if contact_active is None:
            contact_active = torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
        if surface_contacts is None:
            surface_contacts = len(contact_links) <= 2
        down = _gravity(gravity, positions)
        normal = -down / down.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        forces, moments = distribute_contact_forces(
            first.base_wrench, points, contact_active.to(points.device), surface_contacts, normal=normal
        )
        frames = graph.forward_kinematics(positions)
        body_wrenches = torch.zeros((*positions.shape, 6), dtype=positions.dtype, device=positions.device)
        if wrenches is not None:
            body_wrenches = body_wrenches + wrenches
        base_contact = torch.zeros((*positions.shape[:-1], 6), dtype=positions.dtype, device=positions.device)
        if base_external_wrench is not None:
            base_contact = base_contact + base_external_wrench.to(positions)
        bodies = graph.link_bodies[graph.link_index(contact_links)].tolist()
        for contact, body in enumerate(bodies):
            force = forces[..., contact, :]
            moment = moments[..., contact, :]
            point = points[..., contact, :]
            if body < 0:
                base_contact = base_contact + torch.cat(
                    (force, torch.linalg.cross(point, force, dim=-1) + moment), dim=-1
                )
            else:
                lever = point - frames[..., body, :3, 3]
                body_wrenches[..., body, :] += torch.cat(
                    (force, torch.linalg.cross(lever, force, dim=-1) + moment), dim=-1
                )
        second = recursive_newton_euler(
            graph, positions, velocities, accelerations, gravity, base, body_wrenches, base_contact
        )
        efforts, residual = second.efforts, second.base_wrench
    if include_friction:
        efforts = efforts + friction_efforts(graph, velocities.to(positions))
    if not graph.floating_base:
        residual = torch.zeros_like(residual)
    return EffortEstimate(efforts, forces, moments, residual)


def inverse_dynamics(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    gravity: torch.Tensor | None = None,
    include_friction: bool = True,
    base: BaseMotion | None = None,
) -> torch.Tensor:
    """Compute joint efforts ``M qdd + c + g + tau_f`` for a known base motion.

    This ignores contacts; use ``estimate_joint_efforts`` for legged robots.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        accelerations (torch.Tensor): ``[..., J]`` joint accelerations.
        gravity (torch.Tensor | None): Gravity in the base frame.
        include_friction (bool): Add the friction estimate.
        base (BaseMotion | None): Base twist and acceleration (fixed when None).

    Returns:
        torch.Tensor: ``[..., J]`` required efforts.
    """
    efforts = recursive_newton_euler(
        graph, positions, velocities, accelerations, gravity, base
    ).efforts
    if include_friction:
        efforts = efforts + friction_efforts(graph, velocities.to(positions))
    return efforts


def forward_dynamics(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    efforts: torch.Tensor,
    gravity: torch.Tensor | None = None,
    include_friction: bool = True,
    body_wrenches: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve ``M(q) qdd = tau - c - g - tau_f + J^T f_ext`` with the base held fixed.

    Args:
        graph (EmbodimentGraph): Robot. Every joint must move a body with mass.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        efforts (torch.Tensor): ``[..., J]`` applied joint efforts.
        gravity (torch.Tensor | None): Gravity in the base frame.
        include_friction (bool): Subtract the friction estimate.
        body_wrenches (torch.Tensor | None): Known external body wrenches.

    Returns:
        torch.Tensor: ``[..., J]`` joint accelerations.
    """
    _check_joint_tensor("efforts", efforts, positions)
    bias = recursive_newton_euler(
        graph, positions, velocities, torch.zeros_like(positions), gravity, None, body_wrenches
    ).efforts
    if include_friction:
        bias = bias + friction_efforts(graph, velocities.to(positions))
    matrix = graph.mass_matrix(positions)
    rhs = (efforts.to(positions) - bias)[..., None]
    return torch.linalg.solve(matrix, rhs).squeeze(-1)


def frame_velocities(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute each joint frame's linear and angular velocity in the base frame.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        base_linear_velocity (torch.Tensor | None): ``[..., 3]`` base twist; the
            velocities are relative to a still base when None.
        base_angular_velocity (torch.Tensor | None): ``[..., 3]``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Linear velocity in m/s and angular
            velocity in rad/s, each ``[..., J, 3]``, of the moving frame origins
            returned by ``forward_kinematics``.
    """
    zero = torch.zeros(3, dtype=positions.dtype, device=positions.device)
    result = recursive_newton_euler(
        graph,
        positions,
        velocities,
        torch.zeros_like(positions),
        zero,
        BaseMotion(base_linear_velocity, base_angular_velocity),
    )
    return result.linear_velocity, result.angular_velocity


def quaternion_to_matrix(quaternion: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    """Convert unit quaternions to rotation matrices.

    Args:
        quaternion (torch.Tensor): ``[..., 4]``.
        order (str): ``"wxyz"`` or ``"xyzw"``.

    Returns:
        torch.Tensor: ``[..., 3, 3]``.
    """
    if order == "xyzw":
        quaternion = quaternion[..., [3, 0, 1, 2]]
    elif order != "wxyz":
        raise ValueError("quaternion order must be 'wxyz' or 'xyzw'")
    w, x, y, z = (quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)).unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )
