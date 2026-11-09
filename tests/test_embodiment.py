"""Test URDF parsing, per-DoF embodiment graphs, kinematics, and tokenization."""

import math
import unittest

import torch

from model import (
    EmbodimentGraph,
    DaGetConfig,
    DaGetEncoder,
    DaGetTokenizer,
    JointFeatureConfig,
    collate_embodiments,
    parse_urdf,
)
from model.urdf import matrix_to_rpy, rpy_to_matrix


def link(name: str, mass: float = 0.0, com: str = "0 0 0", inertia: float = 0.0) -> str:
    """Return a URDF link with an optional diagonal inertial block."""
    if mass == 0.0:
        return f'<link name="{name}"/>'
    return (
        f'<link name="{name}"><inertial><origin xyz="{com}"/>'
        f'<mass value="{mass}"/><inertia ixx="{inertia}" iyy="{inertia}" '
        f'izz="{inertia}" ixy="0" ixz="0" iyz="0"/></inertial></link>'
    )


def joint(
    name: str,
    parent: str,
    child: str,
    kind: str = "revolute",
    xyz: str = "0 0 0",
    rpy: str = "0 0 0",
    axis: str = "0 1 0",
    extra: str = "",
) -> str:
    """Return a URDF joint with limits for revolute and prismatic joints."""
    limit = '<limit lower="-1.5" upper="1.5" effort="10" velocity="5"/>'
    return (
        f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
        f'<child link="{child}"/><origin xyz="{xyz}" rpy="{rpy}"/>'
        f'<axis xyz="{axis}"/>{limit if kind in ("revolute", "prismatic") else ""}'
        f"{extra}</joint>"
    )


def robot(name: str, *elements: str) -> str:
    """Wrap URDF elements in a robot tag."""
    return f'<robot name="{name}">{"".join(elements)}</robot>'


def double_pendulum(l1: float = 0.7, l2: float = 0.4, m1: float = 2.0, m2: float = 1.5) -> str:
    """Two revolute joints about y with point masses at the link ends."""
    return robot(
        "pendulum",
        link("base"),
        link("upper", m1, f"0 0 {-l1}"),
        link("lower", m2, f"0 0 {-l2}"),
        joint("shoulder", "base", "upper", "continuous"),
        joint("elbow", "upper", "lower", "continuous", xyz=f"0 0 {-l1}"),
    )


def quadruped() -> str:
    """A trunk with four 3-DoF legs, fixed feet, and a fixed IMU link."""
    elements = [link("trunk", 6.0, inertia=0.1), link("imu", 0.01, inertia=1e-5)]
    elements.append(joint("imu_joint", "trunk", "imu", "fixed", xyz="0 0 0.05"))
    for leg, (x, y) in {"FL": (0.2, 0.1), "FR": (0.2, -0.1), "RL": (-0.2, 0.1), "RR": (-0.2, -0.1)}.items():
        elements += [
            link(f"{leg}_hip", 0.6, inertia=1e-3),
            link(f"{leg}_thigh", 0.8, "0 0 -0.1", 1e-3),
            link(f"{leg}_calf", 0.2, "0 0 -0.1", 1e-4),
            link(f"{leg}_foot", 0.05, inertia=1e-5),
            joint(f"{leg}_hip_joint", "trunk", f"{leg}_hip", xyz=f"{x} {y} 0", axis="1 0 0"),
            joint(f"{leg}_thigh_joint", f"{leg}_hip", f"{leg}_thigh", xyz=f"0 {y / 2} 0"),
            joint(f"{leg}_calf_joint", f"{leg}_thigh", f"{leg}_calf", xyz="0 0 -0.2"),
            joint(f"{leg}_foot_fixed", f"{leg}_calf", f"{leg}_foot", "fixed", xyz="0 0 -0.2"),
        ]
    return robot("quadruped", *elements)


def humanoid() -> str:
    """A floating pelvis with legs, a waist, arms, and a mimic gripper."""
    elements = [
        link("world"),
        link("pelvis", 5.0, inertia=0.05),
        link("torso", 8.0, "0 0 0.2", 0.1),
        joint("root", "world", "pelvis", "floating"),
        joint("waist_yaw", "pelvis", "torso", axis="0 0 1", xyz="0 0 0.1"),
    ]
    for side, y in (("left", 0.1), ("right", -0.1)):
        elements += [
            link(f"{side}_hip_yaw_link", 0.5, inertia=1e-3),
            link(f"{side}_hip_roll_link", 0.5, inertia=1e-3),
            link(f"{side}_thigh", 2.0, "0 0 -0.2", 0.02),
            link(f"{side}_shin", 1.5, "0 0 -0.2", 0.01),
            link(f"{side}_ankle", 0.5, inertia=1e-3),
            joint(f"{side}_hip_yaw", "pelvis", f"{side}_hip_yaw_link", xyz=f"0 {y} -0.05", axis="0 0 1"),
            joint(f"{side}_hip_roll", f"{side}_hip_yaw_link", f"{side}_hip_roll_link", axis="1 0 0"),
            joint(f"{side}_hip_pitch", f"{side}_hip_roll_link", f"{side}_thigh"),
            joint(f"{side}_knee", f"{side}_thigh", f"{side}_shin", xyz="0 0 -0.4"),
            joint(f"{side}_ankle_pitch", f"{side}_shin", f"{side}_ankle", xyz="0 0 -0.4"),
            link(f"{side}_upper_arm", 1.0, "0 0 -0.1", 5e-3),
            link(f"{side}_forearm", 0.8, "0 0 -0.1", 3e-3),
            joint(f"{side}_shoulder", "torso", f"{side}_upper_arm", xyz=f"0 {y * 2} 0.3"),
            joint(f"{side}_elbow", f"{side}_upper_arm", f"{side}_forearm", xyz="0 0 -0.25"),
        ]
    elements += [
        link("finger_a", 0.05, inertia=1e-5),
        link("finger_b", 0.05, inertia=1e-5),
        joint("gripper", "right_forearm", "finger_a", "prismatic", xyz="0 0 -0.25", axis="0 1 0"),
        joint(
            "gripper_mimic",
            "right_forearm",
            "finger_b",
            "prismatic",
            xyz="0 0 -0.25",
            axis="0 -1 0",
            extra='<mimic joint="gripper" multiplier="1" offset="0"/>',
        ),
    ]
    return robot("humanoid", *elements)


class UrdfTest(unittest.TestCase):
    """Check URDF parsing and rotation conventions."""

    def test_rpy_round_trip(self) -> None:
        """Convert random angles to matrices and back."""
        torch.manual_seed(0)
        rpy = (torch.rand(64, 3, dtype=torch.float64) - 0.5) * torch.tensor(
            [2 * math.pi, 0.99 * math.pi, 2 * math.pi], dtype=torch.float64
        )
        torch.testing.assert_close(matrix_to_rpy(rpy_to_matrix(rpy)), rpy)
        rotation = rpy_to_matrix(torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(
            rotation[1:, 1:],
            torch.tensor([[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]], dtype=torch.float64),
        )

    def test_invalid_trees_are_rejected(self) -> None:
        """Reject two roots, two parents, and unknown joint types."""
        with self.assertRaises(ValueError):
            parse_urdf(robot("r", link("a"), link("b")))
        with self.assertRaises(ValueError):
            parse_urdf(
                robot(
                    "r",
                    link("a"),
                    link("b"),
                    link("c"),
                    joint("j1", "a", "c"),
                    joint("j2", "b", "c"),
                )
            )
        with self.assertRaises(ValueError):
            parse_urdf(robot("r", link("a"), link("b"), joint("j", "a", "b", "ball")))


class EmbodimentGraphTest(unittest.TestCase):
    """Check per-DoF graphs for different robot morphologies."""

    def test_quadruped_graph(self) -> None:
        """Fixed links vanish and the four hips are joined as chain roots."""
        graph = EmbodimentGraph.from_urdf(quadruped())
        self.assertEqual(graph.num_joints, 12)
        self.assertEqual(graph.joint_names[:3], ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"))
        self.assertEqual(graph.parent_indices[:6].tolist(), [-1, 0, 1, -1, 3, 4])
        spatial, ancestors, descendants = graph.graph_distances
        self.assertEqual(spatial[0, 3].item(), 1)
        self.assertEqual(spatial[2, 5].item(), 5)
        self.assertEqual(ancestors[2, 0].item(), 2)
        self.assertEqual(descendants[0, 2].item(), 2)
        self.assertEqual(graph.feature_size, 6)
        torch.testing.assert_close(graph.joint_features[3, :3], torch.tensor([0.2, -0.1, 0.0]))
        torch.testing.assert_close(graph.joint_features[5, :3], torch.tensor([0.0, 0.0, -0.2]))
        # The fixed foot link is merged into the calf body.
        torch.testing.assert_close(graph.body_mass[2], torch.tensor(0.25, dtype=torch.float64))

    def test_humanoid_mimic_and_floating_base(self) -> None:
        """Mimic joints are frozen and floating joints are skipped or expanded."""
        graph = EmbodimentGraph.from_urdf(humanoid())
        self.assertEqual(graph.num_joints, 16)
        self.assertNotIn("gripper_mimic", graph.joint_names)
        index = graph.joint_names.index
        self.assertEqual(graph.parent_indices[index("left_hip_yaw")].item(), -1)
        self.assertEqual(graph.parent_indices[index("left_shoulder")].item(), index("waist_yaw"))
        self.assertTrue(graph.adjacency_matrix[index("left_hip_yaw"), index("waist_yaw")])
        self.assertTrue(graph.adjacency_matrix[index("left_shoulder"), index("right_shoulder")])
        self.assertFalse(graph.adjacency_matrix[index("left_knee"), index("right_knee")])
        with_mimic = EmbodimentGraph.from_urdf(humanoid(), include_mimic_joints=True)
        self.assertEqual(with_mimic.num_joints, 17)
        expanded = EmbodimentGraph.from_urdf(humanoid(), multi_dof_joints="expand")
        self.assertEqual(expanded.num_joints, 22)
        self.assertEqual(expanded.joint_names[:6], tuple(f"root_{s}" for s in ("tx", "ty", "tz", "rx", "ry", "rz")))
        self.assertEqual(expanded.parent_indices[:6].tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(expanded.parent_indices[expanded.joint_names.index("waist_yaw")].item(), 5)

    def test_joint_order_permutes_everything(self) -> None:
        """A simulator joint order permutes features, parents, and dynamics."""
        default = EmbodimentGraph.from_urdf(quadruped(), JointFeatureConfig.cross_embodiment())
        order = sorted(default.joint_names)
        permuted = EmbodimentGraph.from_urdf(
            quadruped(), JointFeatureConfig.cross_embodiment(), joint_order=order
        )
        perm = torch.tensor([default.joint_names.index(name) for name in order])
        inverse = torch.argsort(perm)
        torch.testing.assert_close(permuted.joint_features, default.joint_features[perm])
        expected_parents = torch.where(
            default.parent_indices[perm] < 0, -1, inverse[default.parent_indices[perm]]
        )
        self.assertTrue(torch.equal(permuted.parent_indices, expected_parents))
        q = torch.randn(12, dtype=torch.float64)
        torch.testing.assert_close(
            permuted.mass_matrix(q[perm]), default.mass_matrix(q)[perm][:, perm]
        )
        with self.assertRaises(ValueError):
            EmbodimentGraph.from_urdf(quadruped(), joint_order=order[:-1])

    def test_cross_embodiment_features(self) -> None:
        """The rich preset adds axis, type, limits, actuator, and inertia data."""
        config = JointFeatureConfig.cross_embodiment()
        graph = EmbodimentGraph.from_urdf(humanoid(), config, include_mimic_joints=True)
        self.assertEqual(graph.feature_size, config.size)
        self.assertEqual(config.size, 31)
        names = graph.feature_names
        row = graph.joint_features[graph.joint_names.index("gripper")]
        self.assertEqual(row[names.index("type_prismatic")].item(), 1.0)
        self.assertEqual(row[names.index("axis_y")].item(), 1.0)
        self.assertEqual(row[names.index("limit_effort")].item(), 10.0)
        torch.testing.assert_close(row[names.index("mass")], torch.tensor(0.05))


class KinematicsTest(unittest.TestCase):
    """Check forward kinematics and the joint-space mass matrix."""

    def test_double_pendulum_matches_analytic_dynamics(self) -> None:
        """Point-mass pendulum mass matrix and elbow position match closed form."""
        l1, l2, m1, m2 = 0.7, 0.4, 2.0, 1.5
        graph = EmbodimentGraph.from_urdf(double_pendulum(l1, l2, m1, m2))
        q = torch.tensor([[0.3, -0.8], [1.1, 0.4]], dtype=torch.float64)
        mass = graph.mass_matrix(q)
        cos2 = q[:, 1].cos()
        m11 = m1 * l1**2 + m2 * (l1**2 + l2**2 + 2 * l1 * l2 * cos2)
        m12 = m2 * (l2**2 + l1 * l2 * cos2)
        expected = torch.stack(
            (torch.stack((m11, m12), -1), torch.stack((m12, torch.full_like(m12, m2 * l2**2)), -1)),
            -2,
        )
        torch.testing.assert_close(mass, expected)
        frames = graph.forward_kinematics(q)
        # Rotation about +y by q maps -z to (-sin q, 0, -cos q).
        elbow = frames[:, 1, :3, 3]
        torch.testing.assert_close(
            elbow,
            torch.stack((-l1 * q[:, 0].sin(), torch.zeros(2, dtype=torch.float64), -l1 * q[:, 0].cos()), -1),
        )

    def test_fixed_links_merge_into_equivalent_inertia(self) -> None:
        """Splitting a body across a fixed joint leaves M(q) unchanged."""
        merged = robot(
            "merged",
            link("base"),
            link("arm", 3.0, "0 0 -0.25", 0.0),
            joint("j", "base", "arm", "continuous"),
        )
        split = robot(
            "split",
            link("base"),
            link("arm", 1.0, "0 0 -0.1", 0.0),
            link("tip", 2.0, "0 0 0", 0.0),
            joint("j", "base", "arm", "continuous"),
            joint("weld", "arm", "tip", "fixed", xyz="0 0 -0.3", rpy="0.4 0.2 0.1"),
        )
        q = torch.tensor([[0.7]], dtype=torch.float64)
        a = EmbodimentGraph.from_urdf(merged).mass_matrix(q)
        b = EmbodimentGraph.from_urdf(split).mass_matrix(q)
        # Parallel axis: 1*0.1^2 + 2*0.3^2 = 0.19, and 3*0.25^2 = 0.1875.
        torch.testing.assert_close(b, torch.tensor([[[0.19]]], dtype=torch.float64))
        self.assertAlmostEqual(a.item(), 0.1875)

    def test_mass_matrix_is_positive_definite(self) -> None:
        """Robot mass matrices are symmetric with positive eigenvalues."""
        for source in (quadruped(), humanoid()):
            graph = EmbodimentGraph.from_urdf(source)
            q = torch.randn(4, graph.num_joints, dtype=torch.float64)
            mass = graph.mass_matrix(q)
            torch.testing.assert_close(mass, mass.transpose(1, 2))
            self.assertTrue(torch.all(torch.linalg.eigvalsh(mass) > 0))

    def test_normalize_joint_positions(self) -> None:
        """Limited joints map to [-1, 1]; continuous joints wrap by pi."""
        graph = EmbodimentGraph.from_urdf(humanoid(), include_mimic_joints=True)
        q = torch.zeros(graph.num_joints)
        q[0] = 1.5
        torch.testing.assert_close(graph.normalize_joint_positions(q)[0], torch.tensor(1.0))
        pendulum = EmbodimentGraph.from_urdf(double_pendulum())
        torch.testing.assert_close(
            pendulum.normalize_joint_positions(torch.tensor([1.5 * math.pi, 0.5 * math.pi])),
            torch.tensor([-0.5, 0.5]),
        )


class TokenizationAndEncoderTest(unittest.TestCase):
    """Run several robot morphologies through one encoder."""

    def test_tokenizer_layout(self) -> None:
        """Tokens follow [o_vg, o_fg, o_vl, o_fl] with feature-major history."""
        tokenizer = DaGetTokenizer(
            fixed_local_size=1, variable_local_size=2, variable_global_size=1, fixed_global_size=1, history_size=2
        )
        self.assertEqual(tokenizer.token_size, 8)
        variable_local = torch.tensor([[[[1.0, 10.0]], [[2.0, 20.0]]]])
        tokens = tokenizer(
            torch.tensor([[[7.0]]]),
            variable_local=variable_local,
            variable_global=torch.tensor([[[5.0], [6.0]]]),
            fixed_global=torch.tensor([[9.0]]),
        )
        torch.testing.assert_close(
            tokens, torch.tensor([[[5.0, 6.0, 9.0, 1.0, 2.0, 10.0, 20.0, 7.0]]])
        )
        with self.assertRaises(ValueError):
            tokenizer(torch.tensor([[[7.0]]]))

    def test_mixed_robot_batch_through_encoder(self) -> None:
        """A hand-like arm, quadruped, and humanoid share one set of weights."""
        torch.manual_seed(3)
        config = JointFeatureConfig.cross_embodiment()
        graphs = [
            EmbodimentGraph.from_urdf(double_pendulum(), config),
            EmbodimentGraph.from_urdf(quadruped(), config),
            EmbodimentGraph.from_urdf(humanoid(), config),
        ]
        batch = collate_embodiments(graphs)
        joints = batch.joint_mask.shape[1]
        history = 3
        tokenizer = DaGetTokenizer(
            fixed_local_size=config.size, variable_local_size=2, variable_global_size=6, history_size=history
        )
        q = torch.randn(3, history, joints) * batch.joint_mask[:, None]
        variable_local = torch.stack(
            (batch.normalize_joint_positions(q), torch.zeros_like(q)), dim=-1
        )
        tokens = tokenizer(
            batch.joint_features,
            batch.joint_mask,
            variable_local=variable_local,
            variable_global=torch.randn(3, history, 6),
        )
        encoder = DaGetEncoder(
            DaGetConfig(
                input_size=tokenizer.token_size,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=32,
            )
        ).eval()
        mass = batch.mass_matrix(q[:, -1])
        output = encoder(tokens, mass_matrix=mass, **batch.encoder_inputs())
        self.assertEqual(output.shape, (3, joints, 16))
        self.assertTrue(torch.all(output[0, graphs[0].num_joints :] == 0))
        recomputed = encoder(
            tokens,
            batch.parent_indices,
            mass,
            batch.joint_mask,
            batch.adjacency_matrix,
        )
        torch.testing.assert_close(output, recomputed)
        alone = collate_embodiments(graphs[:1])
        count = graphs[0].num_joints
        single = encoder(
            tokens[:1, :count], mass_matrix=mass[:1, :count, :count], **alone.encoder_inputs()
        )
        torch.testing.assert_close(single, output[:1, :count])

    def test_padding_does_not_change_unreachable_bucket(self) -> None:
        """Disconnected chains encode the same whether or not a batch is padded."""
        torch.manual_seed(5)
        graph = EmbodimentGraph.from_urdf(quadruped(), sibling_edges=False)
        encoder = DaGetEncoder(
            DaGetConfig(
                input_size=graph.feature_size,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                use_mass_encoding=False,
            )
        ).eval()
        with torch.no_grad():
            encoder.layers[0].graph_bias.spatial_embedding.weight.normal_()
        tight = collate_embodiments([graph])
        padded = collate_embodiments([graph], max_joints=20)
        a = encoder(tight.joint_features, **tight.encoder_inputs())
        b = encoder(padded.joint_features, **padded.encoder_inputs())
        torch.testing.assert_close(a, b[:, :12])
        c = encoder(padded.joint_features, padded.parent_indices, joint_mask=padded.joint_mask)
        torch.testing.assert_close(a, c[:, :12])

    def test_get_zero_config(self) -> None:
        """The paper preset uses the published architecture sizes."""
        config = DaGetConfig.get_zero(input_size=20)
        self.assertEqual(
            (config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.intermediate_size),
            (512, 8, 16, 1024),
        )


if __name__ == "__main__":
    unittest.main()
