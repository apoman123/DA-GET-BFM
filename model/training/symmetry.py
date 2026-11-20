"""Left-right (sagittal) mirror symmetry of a robot and of its trajectories.

A robot is mirror-symmetric when reflecting it through its sagittal (x-z) plane,
``S = diag(1, -1, 1)`` in the base frame, gives the same robot with left and right
exchanged. Its states then come in mirror pairs: if a motion is possible, so is its
reflection. Mittal, Rudin, Klemm, Allshire and Hutter ("Symmetry Considerations for
Learning Task Symmetric Robot Policies", ICRA 2024, pp. 7433-7439,
DOI 10.1109/ICRA57147.2024.10611493, arXiv:2403.04359) use this to augment training
data with mirrored samples ("augmenting the collected trajectories with their
symmetrical copies", their Sec. III-B). Ordoñez-Apraez, Turrisi, Kostic, Martin, Agudo,
Moreno-Noguer, Pontil, Semini and Mastalli ("Morphological Symmetries in Robotics",
IJRR 44(10-11):1743-1766, 2025, DOI 10.1177/02783649241282422, arXiv:2402.15552) show
that such morphological symmetries act on joint space as signed permutations (a
permutation of the roles of identical branches and diagonal matrices of 1 and -1 for
revolute and prismatic joints, their Sec. 5.3.1) and that the equations of motion are
equivariant under them (their Proposition 1).

``mirror_map`` finds the reflection from geometry alone, with no name heuristics:

* every joint's origin at q = 0 (``graph.forward_kinematics``) must reflect onto a
  partner joint's origin, ``(x, y, z) -> (x, -y, z)``, with the parent joints paired
  the same way;
* revolute axes are pseudo-vectors, so a reflected axis is ``(-a_x, a_y, -a_z)``;
  the partner's axis must equal plus or minus that, and the sign maps joint angles.
  Prismatic axes are ordinary vectors, ``(a_x, -a_y, a_z)``;
* URDF links (feet in particular) are paired by their positions on paired bodies;
* by default the dynamics must match too: each body's mass, reflected center of mass
  and reflected inertia ``S I S``, the base body's, and the joints' limits (mapped by
  the sign), actuator limits, damping, friction, gear ratio and rotor inertia.
  Mirrored data from a robot that is symmetric in shape but not in mass would carry
  wrong dynamics labels.

``mirror_trajectory`` applies the reflection to a ``Trajectory``: joint signals are
permuted and signed; base linear quantities (velocity, acceleration, gravity,
position, forces) map as ``S v``, angular ones (angular velocity and acceleration,
moments) as ``-S w = (-x, y, -z)``; the orientation becomes ``S R S``; contacts are
permuted by the paired feet.

The default tolerances are numerical, not physical: positions to 1e-6 of the robot's
characteristic length and relative mass properties to 1e-6, so a robot written
symmetrically (as ``model.morphology`` does) passes and any modelling asymmetry is
reported. [H] Real CAD-exported URDFs are often asymmetric by a few millimetres
(Go1's trunk center of mass is at y = 2 mm); whether mirroring their data with a
looser tolerance helps or hurts is untested. Test: train with and without mirrored
data of such robots and compare the dynamics losses on held-out real data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from model.embodiment import EmbodimentGraph
from model.training.episodes import EmbodimentRecord, Trajectory

REFLECTION = (1.0, -1.0, 1.0)  # S = diag(1, -1, 1), the sagittal reflection


@dataclass(frozen=True)
class MirrorMap:
    """The sagittal mirror symmetry of one robot.

    Attributes:
        joint_permutation (torch.Tensor): Long ``[J]``; joint ``j``'s mirror partner
            (itself for joints on the sagittal plane). It is an involution.
        joint_signs (torch.Tensor): Float64 ``[J]`` of +1 or -1; the mirrored
            position of joint ``j`` is ``joint_signs[j] * q[joint_permutation[j]]``.
            Partners share a sign.
        link_partners (dict[str, str]): URDF link to its mirror link, for every link
            that has one (feet included).
        joint_names (tuple[str, ...]): The graph's joint names, in token order.
        error (float): Largest geometric mismatch found, in m (positions) or as a
            vector difference (unit axes).
    """

    joint_permutation: torch.Tensor
    joint_signs: torch.Tensor
    link_partners: dict[str, str]
    joint_names: tuple[str, ...]
    error: float = 0.0

    def mirror_joints(self, values: torch.Tensor) -> torch.Tensor:
        """Mirror a joint-space signal ``[..., J]`` (positions, velocities, torques).

        Args:
            values (torch.Tensor): Joint values with joints last.

        Returns:
            torch.Tensor: ``values[..., perm] * signs``.
        """
        return values[..., self.joint_permutation.to(values.device)] * self.joint_signs.to(values)

    def contact_permutation(self, contact_links: Sequence[str]) -> torch.Tensor:
        """Return the permutation of a list of contact links.

        Args:
            contact_links (Sequence[str]): Contact links, for example a record's feet.

        Returns:
            torch.Tensor: Long ``[C]``; contact ``c`` of the mirrored data is contact
                ``perm[c]`` of the original.

        Raises:
            ValueError: A contact link has no mirror partner in the list.
        """
        names = list(contact_links)
        try:
            return torch.tensor([names.index(self.link_partners[name]) for name in names], dtype=torch.long)
        except (KeyError, ValueError) as error:
            raise ValueError(f"contact links {names} are not closed under the mirror map") from error


def _reflect(vectors: torch.Tensor) -> torch.Tensor:
    return vectors * vectors.new_tensor(REFLECTION)


def _reflect_axis(axes: torch.Tensor, prismatic: torch.Tensor) -> torch.Tensor:
    """Reflect joint axes: pseudo-vectors (revolute) as ``-S a``, vectors (prismatic) as ``S a``."""
    reflected = _reflect(axes)
    return torch.where(prismatic[:, None], reflected, -reflected)


def _close(a: torch.Tensor, b: torch.Tensor, tolerance: float) -> bool:
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-12)
    return float((a - b).abs().max()) <= tolerance * scale


def _dynamics_match(graph: EmbodimentGraph, permutation: list[int], signs: list[float], frames, tolerance: float) -> bool:
    """Check that paired bodies and joints have mirrored dynamics within a relative tolerance."""
    reflection = torch.diag(torch.tensor(REFLECTION, dtype=torch.float64))
    rotation, origin = frames[:, :3, :3], frames[:, :3, 3]
    centers = (rotation @ graph.body_com[:, :, None]).squeeze(-1) + origin
    inertias = rotation @ graph.body_inertia @ rotation.transpose(-1, -2)
    length = graph.characteristic_length
    for joint, partner in enumerate(permutation):
        if not _close(graph.body_mass[joint], graph.body_mass[partner], tolerance):
            return False
        if float((_reflect(centers[joint]) - centers[partner]).abs().max()) > tolerance * length:
            return False
        if not _close(reflection @ inertias[joint] @ reflection, inertias[partner], tolerance):
            return False
        lower, upper = float(graph.lower[joint]), float(graph.upper[joint])
        mapped = (lower, upper) if signs[joint] > 0 else (-upper, -lower)
        target = torch.tensor([float(graph.lower[partner]), float(graph.upper[partner])], dtype=torch.float64)
        if not _close(torch.tensor(mapped, dtype=torch.float64), target, tolerance):
            return False
        for values in (graph.effort_limits, graph.velocity_limits, graph.damping, graph.friction,
                       graph.gear_ratios, graph.joint_armature):
            if not _close(values[joint], values[partner], tolerance):
                return False
    if float(graph.base_com[1].abs()) > tolerance * length:
        return False
    return _close(reflection @ graph.base_inertia @ reflection, graph.base_inertia, tolerance)


def mirror_map(
    graph: EmbodimentGraph,
    tolerance: float = 1e-6,
    check_dynamics: bool = True,
) -> MirrorMap | None:
    """Derive a robot's sagittal mirror symmetry from its geometry.

    Args:
        graph (EmbodimentGraph): Robot.
        tolerance (float): Relative tolerance: positions to ``tolerance x
            characteristic_length``, unit axes to ``tolerance``, and (with
            ``check_dynamics``) masses, inertias and limits to ``tolerance`` of their
            magnitude.
        check_dynamics (bool): Also require mirrored mass properties, joint limits
            and actuator values (see the module docstring).

    Returns:
        MirrorMap | None: The symmetry, or None if the robot is not mirror-symmetric
            within the tolerance.
    """
    q0 = torch.zeros(graph.num_joints, dtype=torch.float64)
    frames, origins, axes = graph._kinematics(q0)
    prismatic = torch.tensor([kind == "prismatic" for kind in graph.joint_types])
    reflected_origins = _reflect(origins)
    reflected_axes = _reflect_axis(axes, prismatic)
    position_tolerance = tolerance * graph.characteristic_length
    parents = graph.parent_indices.tolist()
    permutation = [-1] * graph.num_joints
    signs = [0.0] * graph.num_joints
    worst = 0.0
    # Parents first, so a joint's partner must hang from its parent's partner.
    for joint in graph._depth_order():
        if permutation[joint] >= 0:
            continue
        parent = parents[joint]
        mirrored_parent = -1 if parent < 0 else permutation[parent]
        best = None
        for candidate in range(graph.num_joints):
            if parents[candidate] != mirrored_parent or graph.joint_types[candidate] != graph.joint_types[joint]:
                continue
            if permutation[candidate] >= 0 and candidate != joint:
                continue
            distance = float((origins[candidate] - reflected_origins[joint]).norm())
            if distance > position_tolerance:
                continue
            for sign in (1.0, -1.0):
                deviation = float((axes[candidate] - sign * reflected_axes[joint]).norm())
                if deviation <= tolerance:
                    score = (max(distance / max(graph.characteristic_length, 1e-12), deviation), candidate != joint)
                    if best is None or score < best[0]:
                        best = (score, candidate, sign, max(distance, deviation))
        if best is None:
            return None
        _, partner, sign, error = best
        permutation[joint], permutation[partner] = partner, joint
        signs[joint] = signs[partner] = sign
        worst = max(worst, error)
    if check_dynamics and not _dynamics_match(graph, permutation, signs, frames, tolerance):
        return None
    # Links: pair every link with the closest reflected link on the partner body.
    link_poses = graph.link_poses(q0, list(graph.link_names))[:, :3, 3]
    bodies = graph.link_bodies.tolist()
    partners: dict[str, str] = {}
    for index, name in enumerate(graph.link_names):
        body = bodies[index]
        mirrored_body = -1 if body < 0 else permutation[body]
        target = _reflect(link_poses[index])
        best_link = None
        for candidate, other in enumerate(graph.link_names):
            if bodies[candidate] != mirrored_body:
                continue
            distance = float((link_poses[candidate] - target).norm())
            if distance <= position_tolerance:
                score = (distance, candidate != index)
                if best_link is None or score < best_link[0]:
                    best_link = (score, other)
        if best_link is not None:
            partners[name] = best_link[1]
    return MirrorMap(
        joint_permutation=torch.tensor(permutation, dtype=torch.long),
        joint_signs=torch.tensor(signs, dtype=torch.float64),
        link_partners=partners,
        joint_names=tuple(graph.joint_names),
        error=worst,
    )


def mirror_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """Mirror world-from-base rotations ``[..., 3, 3]`` as ``S R S``."""
    reflection = rotation.new_tensor(REFLECTION)
    return rotation * reflection[:, None] * reflection[None, :]


def mirror_linear(vectors: torch.Tensor) -> torch.Tensor:
    """Mirror vectors ``[..., 3]`` (velocities, accelerations, forces, gravity): ``S v``."""
    return vectors * vectors.new_tensor(REFLECTION)


def mirror_angular(vectors: torch.Tensor) -> torch.Tensor:
    """Mirror pseudo-vectors ``[..., 3]`` (angular velocity, moments): ``-S w``."""
    return -mirror_linear(vectors)


def mirror_wrench(wrenches: torch.Tensor) -> torch.Tensor:
    """Mirror force-moment pairs ``[..., 6]`` (also linear-angular accelerations)."""
    return torch.cat((mirror_linear(wrenches[..., :3]), mirror_angular(wrenches[..., 3:])), dim=-1)


def mirror_trajectory(record: EmbodimentRecord, trajectory: Trajectory, mirror: MirrorMap) -> Trajectory:
    """Return the mirror image of a trajectory of a mirror-symmetric robot.

    Args:
        record (EmbodimentRecord): The trajectory's embodiment (for its contact links).
        trajectory (Trajectory): Data in the base frame (see ``Trajectory``).
        mirror (MirrorMap): ``mirror_map(record.graph)``.

    Returns:
        Trajectory: The mirrored trajectory of the same embodiment. Joint signals,
            probes and probe results are permuted and signed; base vectors,
            orientation, external wrench and contacts are reflected; ``dt``,
            ``action_type``, ``transition_valid`` and ``exact_dynamics`` are kept.

    Raises:
        ValueError: The map belongs to another robot, or the contact links are not
            closed under it.
    """
    if mirror.joint_names != tuple(record.graph.joint_names):
        raise ValueError("the mirror map was built for a different joint order")

    def joints(values):
        return None if values is None else mirror.mirror_joints(values)

    def linear(values):
        return None if values is None else mirror_linear(values)

    def angular(values):
        return None if values is None else mirror_angular(values)

    def wrench(values):
        return None if values is None else mirror_wrench(values)

    contacts = trajectory.contacts
    if contacts is not None:
        contacts = contacts[..., mirror.contact_permutation(record.contact_links).to(contacts.device)]
    return replace(
        trajectory,
        positions=joints(trajectory.positions),
        velocities=joints(trajectory.velocities),
        accelerations=joints(trajectory.accelerations),
        applied_torque=joints(trajectory.applied_torque),
        actions=joints(trajectory.actions),
        gravity=linear(trajectory.gravity),
        base_orientation=None if trajectory.base_orientation is None else mirror_rotation(trajectory.base_orientation),
        base_position=linear(trajectory.base_position),
        base_linear_velocity=linear(trajectory.base_linear_velocity),
        base_angular_velocity=angular(trajectory.base_angular_velocity),
        base_linear_acceleration=linear(trajectory.base_linear_acceleration),
        base_angular_acceleration=angular(trajectory.base_angular_acceleration),
        projected_gravity=linear(trajectory.projected_gravity),
        imu_specific_force=linear(trajectory.imu_specific_force),
        contacts=contacts,
        probe_torques=joints(trajectory.probe_torques),
        base_external_wrench=wrench(trajectory.base_external_wrench),
        probe_executed=joints(trajectory.probe_executed),
        probe_accelerations=joints(trajectory.probe_accelerations),
        probe_base_accelerations=wrench(trajectory.probe_base_accelerations),
    )
