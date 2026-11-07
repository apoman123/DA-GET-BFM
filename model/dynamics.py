"""Rigid-body dynamics of an embodiment graph: RNEA, forward dynamics, friction.

All quantities use SI units and the root link frame:

* Joint positions are rad (revolute, continuous) or m (prismatic); velocities
  rad/s or m/s; accelerations rad/s^2 or m/s^2.
* Joint efforts are N m for revolute joints and N for prismatic joints.
* ``gravity`` is the gravitational acceleration vector in the root frame,
  in m/s^2. For a fixed-base arm it is usually ``(0, 0, -9.81)``. For a robot
  whose floating base is treated as a mount, pass the gravity direction
  measured in the base frame (for example the projected gravity from an IMU,
  times 9.81) so that ``g(q)`` follows the base orientation.

The equation of motion is

    tau_applied = M(q) qdd + c(q, qd) + g(q) + tau_f(qd)

with ``c`` the Coriolis and centrifugal efforts, ``g`` the gravity efforts, and
``tau_f`` the friction the actuators must overcome. The base is fixed: for a
floating-base robot this is the actuated-joint dynamics with the base held
still, which ignores base-joint coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from model.embodiment import EmbodimentGraph

STANDARD_GRAVITY = 9.80665
FRICTION_VELOCITY_SCALE = 1e-2


def default_gravity(dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    """Return gravity along -z of the root frame.

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


def _gravity(
    gravity: torch.Tensor | None, positions: torch.Tensor
) -> torch.Tensor:
    """Broadcast gravity to the batch shape of positions."""
    batch_shape = positions.shape[:-1]
    if gravity is None:
        gravity = default_gravity(positions.dtype, positions.device)
    gravity = gravity.to(dtype=positions.dtype, device=positions.device)
    if gravity.shape[-1:] != (3,):
        raise ValueError("gravity must end with dimension 3")
    return gravity.expand(*batch_shape, 3)


def recursive_newton_euler(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    gravity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute joint efforts with the recursive Newton-Euler algorithm.

    Returns ``M(q) qdd + c(q, qd) + g(q)``; friction is not included.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        accelerations (torch.Tensor): ``[..., J]`` joint accelerations.
        gravity (torch.Tensor | None): ``[3]`` or ``[..., 3]`` gravity in the
            root frame; defaults to ``(0, 0, -9.81)``. Pass zeros to drop g(q).

    Returns:
        tuple: Efforts ``[..., J]``, and the linear ``[..., J, 3]`` and angular
            ``[..., J, 3]`` velocity of each joint's moving frame origin in the
            root frame.

    Raises:
        ValueError: Shapes do not match.
    """
    _check_joint_tensor("velocities", velocities, positions)
    _check_joint_tensor("accelerations", accelerations, positions)
    frames, _, axes = graph._kinematics(positions)
    velocities = velocities.to(positions)
    accelerations = accelerations.to(positions)
    dtype, device = positions.dtype, positions.device
    batch_shape = positions.shape[:-1]
    gravity = _gravity(gravity, positions)
    zero = torch.zeros((*batch_shape, 3), dtype=dtype, device=device)
    masses = graph.body_mass.to(dtype=dtype, device=device)
    coms = graph.body_com.to(dtype=dtype, device=device)
    inertias = graph.body_inertia.to(dtype=dtype, device=device)
    parents = graph.parent_indices.tolist()
    order = torch.argsort(graph.ancestor_mask.sum(dim=1), stable=True).tolist()
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
    for index in order:
        parent = parents[index]
        if parent < 0:
            parent_omega, parent_alpha, parent_velocity = zero, zero, zero
            parent_acceleration = -gravity
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
        body_omega, body_alpha = omega[index], alpha[index]
        com_acceleration = (
            acceleration
            + cross(body_alpha, offset, dim=-1)
            + cross(body_omega, cross(body_omega, offset, dim=-1), dim=-1)
        )
        body_force = masses[index] * com_acceleration
        spin = (inertia @ body_omega[..., None]).squeeze(-1)
        body_moment = (
            (inertia @ body_alpha[..., None]).squeeze(-1)
            + cross(body_omega, spin, dim=-1)
            + cross(offset, body_force, dim=-1)
        )
        force[index], moment[index] = body_force, body_moment
    efforts: list[torch.Tensor] = [zero[..., 0]] * count
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
    return (
        torch.stack(efforts, dim=-1),
        torch.stack(linear_velocity, dim=-2),
        torch.stack(omega, dim=-2),
    )


def gravity_efforts(
    graph: EmbodimentGraph, positions: torch.Tensor, gravity: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute ``g(q)``, the efforts that hold the robot still against gravity.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        gravity (torch.Tensor | None): Gravity in the root frame.

    Returns:
        torch.Tensor: ``[..., J]`` efforts.
    """
    zero = torch.zeros_like(positions)
    return recursive_newton_euler(graph, positions, zero, zero, gravity)[0]


def coriolis_efforts(
    graph: EmbodimentGraph, positions: torch.Tensor, velocities: torch.Tensor
) -> torch.Tensor:
    """Compute ``c(q, qd)``, the Coriolis and centrifugal efforts.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.

    Returns:
        torch.Tensor: ``[..., J]`` efforts, zero when ``qd`` is zero.
    """
    no_gravity = torch.zeros(3, dtype=positions.dtype, device=positions.device)
    return recursive_newton_euler(
        graph, positions, velocities, torch.zeros_like(positions), no_gravity
    )[0]


def friction_efforts(
    graph: EmbodimentGraph,
    velocities: torch.Tensor,
    velocity_scale: float = FRICTION_VELOCITY_SCALE,
) -> torch.Tensor:
    """Estimate friction efforts from URDF damping and Coulomb friction.

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


def inverse_dynamics(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    gravity: torch.Tensor | None = None,
    include_friction: bool = True,
) -> torch.Tensor:
    """Compute ``tau = M(q) qdd + c(q, qd) + g(q) + tau_f(qd)``.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        accelerations (torch.Tensor): ``[..., J]`` joint accelerations.
        gravity (torch.Tensor | None): Gravity in the root frame.
        include_friction (bool): Add the friction estimate.

    Returns:
        torch.Tensor: ``[..., J]`` required efforts.
    """
    efforts = recursive_newton_euler(graph, positions, velocities, accelerations, gravity)[0]
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
) -> torch.Tensor:
    """Solve ``M(q) qdd = tau - c(q, qd) - g(q) - tau_f(qd)`` for ``qdd``.

    Args:
        graph (EmbodimentGraph): Robot. Every joint must move a body with mass.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        efforts (torch.Tensor): ``[..., J]`` applied joint efforts.
        gravity (torch.Tensor | None): Gravity in the root frame.
        include_friction (bool): Subtract the friction estimate.

    Returns:
        torch.Tensor: ``[..., J]`` joint accelerations.
    """
    _check_joint_tensor("efforts", efforts, positions)
    bias = inverse_dynamics(
        graph,
        positions,
        velocities,
        torch.zeros_like(positions),
        gravity,
        include_friction,
    )
    matrix = graph.mass_matrix(positions)
    rhs = (efforts.to(positions) - bias)[..., None]
    return torch.linalg.solve(matrix, rhs).squeeze(-1)


def frame_velocities(
    graph: EmbodimentGraph, positions: torch.Tensor, velocities: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute each joint frame's linear and angular velocity in the root frame.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Linear velocity in m/s and angular
            velocity in rad/s, each ``[..., J, 3]``, of the moving frame origins
            returned by ``forward_kinematics``.
    """
    zero = torch.zeros(3, dtype=positions.dtype, device=positions.device)
    _, linear, angular = recursive_newton_euler(
        graph, positions, velocities, torch.zeros_like(positions), zero
    )
    return linear, angular
