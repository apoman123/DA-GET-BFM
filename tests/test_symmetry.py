"""Test the sagittal mirror symmetry of robots and trajectories."""

import unittest
from dataclasses import fields, replace
from pathlib import Path

import torch

from model import EmbodimentGraph, actuator_specs, parse_urdf
from model.dynamics import BaseMotion, recursive_newton_euler
from model.morphology import generate_robot
from model.training.episodes import EmbodimentRecord, Trajectory, exp_rotation
from model.training.symmetry import (
    mirror_angular,
    mirror_linear,
    mirror_map,
    mirror_rotation,
    mirror_trajectory,
    mirror_wrench,
)

# The robot URDFs in assets/urdf are downloaded, not committed (scripts/download_urdfs.py).
NEEDS_URDFS = unittest.skipUnless(
    all((Path(__file__).resolve().parents[1] / "assets" / "urdf" / name).exists() for name in ("go1.urdf", "h1.urdf", "anymal_d.urdf")),
    "needs the robot URDFs: uv run python scripts/download_urdfs.py",
)


SYMMETRIC = {"knee_limit_scale": [1.0]}  # no single-leg knee restriction
S = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)


def symmetric_robot(family: str = "quadruped", seed: int = 3, ranges: dict | None = None):
    """Return a mirror-symmetric generated robot, its floating-base graph, and its mirror map."""
    robot = generate_robot(family, seed, {**SYMMETRIC, "knee_joints": [1, 2], **(ranges or {})})
    graph = EmbodimentGraph.from_urdf(robot.urdf, floating_base=True)
    return robot, graph, mirror_map(graph)


@NEEDS_URDFS
class MirrorMapTest(unittest.TestCase):
    """The mirror map is found from geometry and respects kinematics and dynamics."""

    def setUp(self) -> None:
        self.robot, self.graph, self.mirror = symmetric_robot()
        generator = torch.Generator().manual_seed(0)
        self.q = torch.rand((5, self.graph.num_joints), generator=generator, dtype=torch.float64) * 2 - 1

    def test_pairs_left_and_right(self) -> None:
        """The map is an involution that pairs left with right; feet pair too."""
        self.assertIsNotNone(self.mirror)
        permutation, signs = self.mirror.joint_permutation, self.mirror.joint_signs
        self.assertTrue(torch.equal(permutation[permutation], torch.arange(self.graph.num_joints)))
        self.assertTrue(torch.equal(signs, signs[permutation]))
        names = self.graph.joint_names
        partner = {names[j]: names[int(permutation[j])] for j in range(len(names))}
        self.assertEqual(partner["FL_hip_joint"], "FR_hip_joint")
        self.assertEqual(partner["RR_knee_joint"], "RL_knee_joint")
        self.assertEqual(float(signs[names.index("FL_hip_joint")]), -1.0)  # abduction about x flips
        self.assertEqual(float(signs[names.index("FL_thigh_joint")]), 1.0)  # pitch about y keeps its sign
        self.assertEqual(self.mirror.link_partners["FL_foot"], "FR_foot")
        humanoid = symmetric_robot("humanoid", 1)[2]
        torso = humanoid.joint_names.index("torso_joint")
        self.assertEqual(int(humanoid.joint_permutation[torso]), torso)  # on the plane: maps to itself
        self.assertEqual(float(humanoid.joint_signs[torso]), -1.0)  # yaw flips
        for family in ("hexapod", "humanoid"):
            self.assertIsNotNone(symmetric_robot(family, 2)[2])

    def test_mirrored_kinematics_are_reflected(self) -> None:
        """FK of the mirrored joint angles is the reflection of FK, joint by joint."""
        mirrored = self.mirror.mirror_joints(self.q)
        _, origins, axes = self.graph._kinematics(self.q)
        _, origins_m, axes_m = self.graph._kinematics(mirrored)
        permutation, signs = self.mirror.joint_permutation, self.mirror.joint_signs
        self.assertTrue(torch.allclose(origins_m[:, permutation], origins * S, atol=1e-12))
        # Revolute axes are pseudo-vectors: reflected as -S a, times the joint's sign.
        self.assertTrue(torch.allclose(axes_m[:, permutation], -axes * S * signs[:, None], atol=1e-12))
        feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        partners = [self.mirror.link_partners[name] for name in feet]
        self.assertTrue(torch.allclose(
            self.graph.link_poses(mirrored, partners)[..., :3, 3], self.graph.link_poses(self.q, feet)[..., :3, 3] * S,
            atol=1e-12,
        ))

    def test_mass_matrix_is_invariant(self) -> None:
        """M(q') equals the signed permutation of M(q)."""
        mirrored = self.mirror.mirror_joints(self.q)
        permutation, signs = self.mirror.joint_permutation, self.mirror.joint_signs
        matrix = self.graph.mass_matrix(self.q)
        expected = matrix[:, permutation][:, :, permutation] * signs[:, None] * signs[None, :]
        self.assertTrue(torch.allclose(self.graph.mass_matrix(mirrored), expected, atol=1e-12))

    def test_inverse_dynamics_is_equivariant(self) -> None:
        """Floating-base RNEA of mirrored data is the mirror of RNEA of the data."""
        generator = torch.Generator().manual_seed(1)

        def noise(*shape):
            return torch.randn(shape, generator=generator, dtype=torch.float64)

        count, joints = self.q.shape
        qd, qdd = noise(count, joints), noise(count, joints)
        gravity = noise(count, 3) * 9.81
        base = BaseMotion(noise(count, 3), noise(count, 3), noise(count, 3), noise(count, 3))
        external = noise(count, 6)
        result = recursive_newton_euler(self.graph, self.q, qd, qdd, gravity, base, base_external_wrench=external)
        mirrored_base = BaseMotion(
            mirror_linear(base.linear_velocity), mirror_angular(base.angular_velocity),
            mirror_linear(base.linear_acceleration), mirror_angular(base.angular_acceleration),
        )
        mirror = self.mirror.mirror_joints
        mirrored = recursive_newton_euler(
            self.graph, mirror(self.q), mirror(qd), mirror(qdd), mirror_linear(gravity), mirrored_base,
            base_external_wrench=mirror_wrench(external),
        )
        self.assertTrue(torch.allclose(mirrored.efforts, mirror(result.efforts), atol=1e-10))
        self.assertTrue(torch.allclose(mirrored.base_wrench, mirror_wrench(result.base_wrench), atol=1e-10))

    def test_asymmetric_robots_have_no_map(self) -> None:
        """A shifted hip, unpaired masses, or one restricted knee break the symmetry."""
        parsed = parse_urdf(self.robot.urdf)
        joint = parsed.joints["FL_hip_joint"]
        origin = joint.origin.clone()
        origin[1, 3] += 0.01
        shifted = replace(parsed, joints={**parsed.joints, "FL_hip_joint": replace(joint, origin=origin)})
        self.assertIsNone(mirror_map(EmbodimentGraph.from_urdf(shifted, floating_base=True)))
        uneven = symmetric_robot(ranges={"symmetric_mass": False})[1]
        self.assertIsNone(mirror_map(uneven))
        self.assertIsNotNone(mirror_map(uneven, check_dynamics=False))  # the shape is still symmetric
        restricted = generate_robot("quadruped", 3, {"knee_joints": [1], "knee_limit_scale": [0.2]})
        self.assertEqual(len(restricted.metadata["params"]["knee_limit_legs"]), 1)
        self.assertIsNone(mirror_map(EmbodimentGraph.from_urdf(restricted.urdf, floating_base=True)))

    def test_go1_is_symmetric_in_shape_only(self) -> None:
        """Go1's joints and feet mirror exactly, but its trunk's center of mass is at y = 2 mm."""
        graph = EmbodimentGraph.from_urdf(Path(__file__).resolve().parents[1] / "assets/urdf/go1.urdf", floating_base=True)
        shape = mirror_map(graph, check_dynamics=False)
        self.assertIsNotNone(shape)
        self.assertEqual(shape.link_partners["FR_foot"], "FL_foot")
        self.assertIsNone(mirror_map(graph))
        self.assertIsNotNone(mirror_map(graph, tolerance=1e-2))


@NEEDS_URDFS
class MirrorTrajectoryTest(unittest.TestCase):
    """Mirroring a trajectory maps every field and is its own inverse."""

    def setUp(self) -> None:
        robot, graph, self.mirror = symmetric_robot()
        self.record = EmbodimentRecord.build("robot", graph, actuator_specs(graph),
                                             contact_links=robot.metadata["contact_links"])
        generator = torch.Generator().manual_seed(2)
        steps, joints, probes = 6, graph.num_joints, 2

        def noise(*shape):
            return torch.randn(shape, generator=generator, dtype=torch.float64)

        rotation = exp_rotation(noise(steps, 3))
        self.trajectory = Trajectory(
            embodiment=0, dt=0.02, positions=noise(steps, joints), velocities=noise(steps, joints),
            accelerations=noise(steps, joints), applied_torque=noise(steps, joints), actions=noise(steps, joints),
            action_type="position", gravity=noise(steps, 3), base_orientation=rotation, base_position=noise(steps, 3),
            base_linear_velocity=noise(steps, 3), base_angular_velocity=noise(steps, 3),
            base_linear_acceleration=noise(steps, 3), base_angular_acceleration=noise(steps, 3),
            projected_gravity=noise(steps, 3), imu_specific_force=noise(steps, 3),
            contacts=noise(steps, 4) > 0, probe_torques=noise(steps, probes, joints),
            transition_valid=noise(steps) > 0, base_external_wrench=noise(steps, 6), exact_dynamics=True,
            probe_executed=noise(steps, probes, joints), probe_accelerations=noise(steps, probes, joints),
            probe_base_accelerations=noise(steps, probes, 6),
        )

    def test_mirroring_twice_is_the_identity(self) -> None:
        """Every field of a twice-mirrored trajectory equals the original."""
        twice = mirror_trajectory(self.record, mirror_trajectory(self.record, self.trajectory, self.mirror), self.mirror)
        for item in fields(Trajectory):
            original, value = getattr(self.trajectory, item.name), getattr(twice, item.name)
            if isinstance(original, torch.Tensor):
                self.assertTrue(torch.equal(original, value), item.name)
            else:
                self.assertEqual(original, value, item.name)

    def test_fields_are_reflected(self) -> None:
        """Joints are permuted and signed, base vectors reflected, contacts swapped."""
        mirrored = mirror_trajectory(self.record, self.trajectory, self.mirror)
        permutation, signs = self.mirror.joint_permutation, self.mirror.joint_signs
        self.assertTrue(torch.equal(mirrored.applied_torque, self.trajectory.applied_torque[:, permutation] * signs))
        self.assertTrue(torch.equal(mirrored.probe_torques, self.trajectory.probe_torques[..., permutation] * signs))
        self.assertTrue(torch.equal(mirrored.base_linear_velocity, self.trajectory.base_linear_velocity * S))
        self.assertTrue(torch.equal(mirrored.base_angular_velocity, -self.trajectory.base_angular_velocity * S))
        self.assertTrue(torch.equal(mirrored.base_orientation, mirror_rotation(self.trajectory.base_orientation)))
        self.assertTrue(torch.allclose(torch.linalg.det(mirrored.base_orientation), torch.ones(6, dtype=torch.float64)))
        wrench = self.trajectory.base_external_wrench
        self.assertTrue(torch.equal(mirrored.base_external_wrench[:, :3], wrench[:, :3] * S))
        self.assertTrue(torch.equal(mirrored.base_external_wrench[:, 3:], -wrench[:, 3:] * S))
        feet = list(self.record.contact_links)
        swap = [feet.index(self.mirror.link_partners[name]) for name in feet]
        self.assertTrue(torch.equal(mirrored.contacts, self.trajectory.contacts[:, swap]))
        self.assertTrue(torch.equal(mirrored.transition_valid, self.trajectory.transition_valid))
        self.assertEqual(mirrored.action_type, "position")
        broken = replace(self.record, contact_links=("FL_foot",))
        with self.assertRaises(ValueError):
            mirror_trajectory(broken, self.trajectory, self.mirror)


if __name__ == "__main__":
    unittest.main()
