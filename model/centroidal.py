"""Centroidal quantities of a robot: center of mass, momentum, composite inertia and their rates.

The centroidal momentum ``h_G = (p, k_G)`` is the robot's linear momentum and its
angular momentum about the center of mass G. It is a linear map of the generalized
velocity (the centroidal momentum matrix, Orin & Goswami, IROS 2008, DOI
10.1109/IROS.2008.4650772; Orin, Goswami & Lee, Autonomous Robots 35:161-176, 2013,
DOI 10.1007/s10514-013-9341-4), and its rate equals the net external wrench, contact
forces plus gravity (Dai, Valenzuela & Tedrake, Humanoids 2014, DOI
10.1109/HUMANOIDS.2014.7041375). The same quantities hold for any robot, which is why
they are proposed as embodiment-independent auxiliary targets (proposal C3,
``docs/zero_shot.md``; using them as encoder targets is an unverified hypothesis).

Everything is expressed in the base frame and made dimensionless with the robot's
total mass ``m``, characteristic length ``L`` and standard gravity ``g`` (dynamic
similarity: Alexander & Jayes, J. Zool. 201:135-152, 1983, DOI
10.1111/j.1469-7998.1983.tb04266.x; ``g = 9.80665 m/s^2``, 3rd CGPM 1901):

=====================  ========  ===========================
quantity               columns   unit
=====================  ========  ===========================
CoM position           0-2       ``L``
linear momentum        3-5       ``m sqrt(g L)``
angular momentum (G)   6-8       ``m L sqrt(g L)``
composite inertia (G)  9-14      ``m L^2`` (xx, xy, xz, yy, yz, zz)
linear momentum rate   15-17     ``m g``
angular momentum rate  18-20     ``m g L``
=====================  ========  ===========================

The composite inertia about G uses Steiner's (parallel-axis) theorem (Lynch & Park,
*Modern Robotics*, Cambridge University Press 2017, DOI 10.1017/9781316661239, Eq. (8.27)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.dynamics import STANDARD_GRAVITY, BaseMotion, recursive_newton_euler
from model.embodiment import EmbodimentGraph

CENTROIDAL_SIZE = 21
CENTROIDAL_GROUPS = ("geometry", "momentum", "rate")
CENTROIDAL_COLUMNS = {
    "geometry": tuple(range(0, 3)) + tuple(range(9, 15)),
    "momentum": tuple(range(3, 9)),
    "rate": tuple(range(15, 21)),
}


@dataclass
class CentroidalState:
    """Centroidal quantities in SI units, base frame.

    Attributes:
        com (torch.Tensor): ``[..., 3]`` center of mass.
        linear_momentum (torch.Tensor): ``[..., 3]``.
        angular_momentum (torch.Tensor): ``[..., 3]`` about the center of mass.
        inertia (torch.Tensor): ``[..., 3, 3]`` composite rigid-body inertia about it.
    """

    com: torch.Tensor
    linear_momentum: torch.Tensor
    angular_momentum: torch.Tensor
    inertia: torch.Tensor


def _cross_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Return ``|v|^2 I - v v^T`` (the parallel-axis term per unit mass, Steiner's theorem)."""
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return (vector * vector).sum(-1)[..., None, None] * eye - vector[..., :, None] * vector[..., None, :]


def centroidal_state(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
) -> CentroidalState:
    """Compute the center of mass, centroidal momentum and composite inertia.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        base_linear_velocity (torch.Tensor | None): ``[..., 3]`` base twist in the
            base frame; a still base when None.
        base_angular_velocity (torch.Tensor | None): ``[..., 3]``.

    Returns:
        CentroidalState: Quantities in the base frame.
    """
    dtype, device = positions.dtype, positions.device
    frames = graph.forward_kinematics(positions)
    zero = torch.zeros(3, dtype=dtype, device=device)
    result = recursive_newton_euler(
        graph, positions, velocities, torch.zeros_like(positions), zero,
        BaseMotion(base_linear_velocity, base_angular_velocity),
    )
    rotation = frames[..., :3, :3]
    offset = (rotation @ graph.body_com.to(dtype=dtype, device=device)[..., None]).squeeze(-1)
    centers = frames[..., :3, 3] + offset
    omega = result.angular_velocity
    center_velocity = result.linear_velocity + torch.linalg.cross(omega, offset, dim=-1)
    mass = graph.body_mass.to(dtype=dtype, device=device)
    base_mass = float(graph.base_mass)
    base_com = (graph.base_com if graph.base_com is not None else torch.zeros(3, dtype=torch.float64)).to(dtype=dtype, device=device)
    base_inertia = (
        graph.base_inertia if graph.base_inertia is not None else torch.zeros((3, 3), dtype=torch.float64)
    ).to(dtype=dtype, device=device)
    batch_shape = positions.shape[:-1]
    base_linear = (
        torch.zeros(*batch_shape, 3, dtype=dtype, device=device) if base_linear_velocity is None else base_linear_velocity
    )
    base_angular = (
        torch.zeros(*batch_shape, 3, dtype=dtype, device=device) if base_angular_velocity is None else base_angular_velocity
    )
    total = base_mass + float(mass.sum())
    com = (base_mass * base_com + (mass[:, None] * centers).sum(-2)) / max(total, 1e-12)
    base_velocity = base_linear + torch.linalg.cross(base_angular, base_com.expand_as(base_angular), dim=-1)
    linear = base_mass * base_velocity + (mass[:, None] * center_velocity).sum(-2)
    body_inertia = rotation @ graph.body_inertia.to(dtype=dtype, device=device) @ rotation.transpose(-1, -2)
    relative = centers - com[..., None, :]
    base_relative = base_com - com
    angular = (
        (body_inertia @ omega[..., None]).squeeze(-1)
        + mass[:, None] * torch.linalg.cross(relative, center_velocity, dim=-1)
    ).sum(-2)
    angular = angular + (base_inertia @ base_angular[..., None]).squeeze(-1)
    angular = angular + base_mass * torch.linalg.cross(base_relative, base_velocity, dim=-1)
    inertia = (body_inertia + mass[:, None, None] * _cross_matrix(relative)).sum(-3)
    inertia = inertia + base_inertia + base_mass * _cross_matrix(base_relative)
    return CentroidalState(com, linear, angular, inertia)


def centroidal_targets(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    dt: float,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
    base_relative_rotation: torch.Tensor | None = None,
    momentum_known: bool = True,
    transition_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dimensionless centroidal targets ``[T, 21]`` and validity ``[T, 3]`` of a trajectory.

    The momentum rate is a forward difference: ``(R_t^T R_{t+1} h_{t+1} - h_t) / dt``,
    the change of the momentum vector in an inertial frame expressed in the base frame
    at ``t``.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[T, J]``.
        velocities (torch.Tensor): ``[T, J]``.
        dt (float): Frame period in s.
        base_linear_velocity (torch.Tensor | None): ``[T, 3]`` base frame.
        base_angular_velocity (torch.Tensor | None): ``[T, 3]``.
        base_relative_rotation (torch.Tensor | None): ``[T - 1, 3, 3]``
            ``R_t^T R_{t+1}``; identity when None (still base).
        momentum_known (bool): The base twist is known (or the base is fixed), so
            momenta are valid.
        transition_valid (torch.Tensor | None): Bool ``[T]``; False where frame
            ``t + 1`` does not follow from ``t``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Targets ``[T, 21]`` (float64) and bool
            validity ``[T, 3]`` of the geometry, momentum and rate groups.
    """
    steps = positions.shape[0]
    state = centroidal_state(graph, positions, velocities, base_linear_velocity, base_angular_velocity)
    mass = max(graph.total_mass, 1e-12)
    length = max(graph.characteristic_length, 1e-6)
    gravity = STANDARD_GRAVITY
    speed = (gravity * length) ** 0.5
    linear, angular = state.linear_momentum, state.angular_momentum
    rate_linear = torch.zeros_like(linear)
    rate_angular = torch.zeros_like(angular)
    if steps > 1:
        relative = (
            torch.eye(3, dtype=positions.dtype).expand(steps - 1, 3, 3)
            if base_relative_rotation is None
            else base_relative_rotation
        )
        rate_linear[:-1] = ((relative @ linear[1:, :, None]).squeeze(-1) - linear[:-1]) / dt
        rate_angular[:-1] = ((relative @ angular[1:, :, None]).squeeze(-1) - angular[:-1]) / dt
    inertia = state.inertia[..., [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    targets = torch.cat(
        (
            state.com / length,
            linear / (mass * speed),
            angular / (mass * length * speed),
            inertia / (mass * length**2),
            rate_linear / (mass * gravity),
            rate_angular / (mass * gravity * length),
        ),
        dim=-1,
    )
    valid = torch.zeros((steps, 3), dtype=torch.bool)
    valid[:, 0] = True
    valid[:, 1] = momentum_known
    if steps > 1 and momentum_known:
        rate = torch.ones(steps, dtype=torch.bool)
        rate[-1] = False
        if transition_valid is not None:
            rate &= transition_valid.to(torch.bool)
        valid[:, 2] = rate
    return targets, valid
