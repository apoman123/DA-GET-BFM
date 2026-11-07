"""Test rigid-body dynamics, physics features, the feature cache, and Physics Cross Attention."""

import math
import unittest

import torch

from model import EmbodimentGraph, GetConfig, GetEncoder, collate_embodiments
from model.dynamics import (
    coriolis_efforts,
    forward_dynamics,
    frame_velocities,
    gravity_efforts,
    inverse_dynamics,
    recursive_newton_euler,
)
from model.physics import (
    ActuatorDefaults,
    ActuatorSpec,
    PhysicsFeatureCache,
    actuator_features,
    actuator_specs,
    actuator_tensors,
    dynamics_features,
    motor_labels,
    requested_torque,
    rigid_body_features,
    torque_limit,
)
from model.physics_schema import (
    ACTUATOR_FEATURE_SIZE,
    DYNAMICS_FEATURE_SIZE,
    FEATURE_UPDATE_TIERS,
    PHYSICS_FEATURE_GROUPS,
    RIGID_BODY_FEATURE_SIZE,
)
from model.urdf import parse_urdf
from tests.test_embodiment import double_pendulum, humanoid, joint, link, quadruped, robot

GRAVITY = 9.80665


def random_state(graph: EmbodimentGraph, batch: int = 4, seed: int = 0):
    """Return random positions, velocities, and accelerations in float64."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, graph.num_joints)
    return tuple(torch.randn(shape, generator=generator, dtype=torch.float64) for _ in range(3))


class DynamicsTest(unittest.TestCase):
    """Check RNEA against closed forms and identities of rigid-body dynamics."""

    def test_double_pendulum_closed_form(self) -> None:
        """Gravity and Coriolis efforts match the textbook double pendulum."""
        l1, l2, m1, m2 = 0.7, 0.4, 2.0, 1.5
        graph = EmbodimentGraph.from_urdf(double_pendulum(l1, l2, m1, m2))
        q = torch.tensor([[0.3, -0.8], [1.2, 0.5]], dtype=torch.float64)
        qd = torch.tensor([[0.7, -1.1], [-0.4, 2.0]], dtype=torch.float64)
        q1, q2 = q.unbind(-1)
        expected_gravity = GRAVITY * torch.stack(
            ((m1 + m2) * l1 * q1.sin() + m2 * l2 * (q1 + q2).sin(), m2 * l2 * (q1 + q2).sin()), dim=-1
        )
        torch.testing.assert_close(gravity_efforts(graph, q), expected_gravity)
        h = -m2 * l1 * l2 * q2.sin()
        qd1, qd2 = qd.unbind(-1)
        expected_coriolis = torch.stack((h * (2 * qd1 * qd2 + qd2**2), -h * qd1**2), dim=-1)
        torch.testing.assert_close(coriolis_efforts(graph, q, qd), expected_coriolis)

    def test_rnea_identities_on_branching_robots(self) -> None:
        """RNEA matches M(q) qdd, energy conservation, and the potential gradient."""
        for urdf in (quadruped(), humanoid()):
            graph = EmbodimentGraph.from_urdf(urdf, multi_dof_joints="expand")
            q, qd, qdd = random_state(graph)
            no_gravity = torch.zeros(3, dtype=torch.float64)
            inertial = recursive_newton_euler(graph, q, torch.zeros_like(q), qdd, no_gravity)[0]
            torch.testing.assert_close(inertial, (graph.mass_matrix(q) @ qdd[..., None]).squeeze(-1))
            # qd^T c(q, qd) = 1/2 qd^T dM/dt qd (the Coriolis matrix is skew w.r.t. Mdot).
            step = 1e-6
            mass_rate = (graph.mass_matrix(q + step * qd) - graph.mass_matrix(q - step * qd)) / (2 * step)
            power = (qd * coriolis_efforts(graph, q, qd)).sum(-1)
            expected = 0.5 * (qd[..., None, :] @ mass_rate @ qd[..., None]).squeeze(-1).squeeze(-1)
            torch.testing.assert_close(power, expected, rtol=1e-5, atol=1e-6)

            def potential(positions: torch.Tensor) -> torch.Tensor:
                frames = graph.forward_kinematics(positions)
                centers = (frames[..., :3, :3] @ graph.body_com[:, :, None]).squeeze(-1) + frames[..., :3, 3]
                return GRAVITY * (graph.body_mass * centers[..., 2]).sum(-1)

            leaf = q.clone().requires_grad_()
            gradient = torch.autograd.grad(potential(leaf).sum(), leaf)[0]
            torch.testing.assert_close(gravity_efforts(graph, q), gradient)

    def test_forward_dynamics_inverts_inverse_dynamics(self) -> None:
        """FD(q, qd, ID(q, qd, qdd)) returns qdd, with friction and tilted gravity."""
        graph = EmbodimentGraph.from_urdf(quadruped())
        q, qd, qdd = random_state(graph, seed=1)
        gravity = torch.tensor([[1.0, -2.0, -9.0]] * 4, dtype=torch.float64)
        efforts = inverse_dynamics(graph, q, qd, qdd, gravity)
        torch.testing.assert_close(forward_dynamics(graph, q, qd, efforts, gravity), qdd)

    def test_frame_velocities_match_kinematics(self) -> None:
        """Frame linear velocities equal the time derivative of FK positions."""
        graph = EmbodimentGraph.from_urdf(humanoid(), include_mimic_joints=True)
        q, qd, _ = random_state(graph, seed=2)
        linear, angular = frame_velocities(graph, q, qd)
        step = 1e-6
        positions = lambda value: graph.forward_kinematics(value)[..., :3, 3]  # noqa: E731
        torch.testing.assert_close(
            linear, (positions(q + step * qd) - positions(q - step * qd)) / (2 * step), rtol=1e-5, atol=1e-6
        )
        rotation = graph.forward_kinematics(q)[..., :3, :3]
        rate = (
            graph.forward_kinematics(q + step * qd)[..., :3, :3]
            - graph.forward_kinematics(q - step * qd)[..., :3, :3]
        ) / (2 * step)
        skew = rate @ rotation.transpose(-1, -2)
        expected = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)
        torch.testing.assert_close(angular, expected, rtol=1e-5, atol=1e-6)


class UrdfExtensionsTest(unittest.TestCase):
    """Check transmissions and frozen joints."""

    def test_gear_ratio_from_transmission(self) -> None:
        """mechanicalReduction becomes the joint gear ratio."""
        transmission = (
            '<transmission name="t"><type>transmission_interface/SimpleTransmission</type>'
            '<joint name="elbow"/><actuator name="m"><mechanicalReduction>6.33</mechanicalReduction>'
            "</actuator></transmission>"
        )
        urdf = double_pendulum().replace("</robot>", transmission + "</robot>")
        graph = EmbodimentGraph.from_urdf(urdf)
        self.assertEqual(parse_urdf(urdf).gear_ratios, {"elbow": 6.33})
        torch.testing.assert_close(graph.gear_ratios, torch.tensor([1.0, 6.33], dtype=torch.float64))

    def test_frozen_joints_merge_bodies(self) -> None:
        """Freezing the elbow makes one rigid body equal to a single pendulum."""
        l1, l2, m1, m2 = 0.7, 0.4, 2.0, 1.5
        graph = EmbodimentGraph.from_urdf(double_pendulum(l1, l2, m1, m2), frozen_joints={"elbow": 0.0})
        self.assertEqual(graph.joint_names, ("shoulder",))
        expected = m1 * l1**2 + m2 * (l1 + l2) ** 2
        torch.testing.assert_close(
            graph.mass_matrix(torch.zeros(1, dtype=torch.float64)),
            torch.tensor([[expected]], dtype=torch.float64),
        )
        bent = EmbodimentGraph.from_urdf(double_pendulum(l1, l2, m1, m2), frozen_joints={"elbow": math.pi / 2})
        self.assertLess(float(bent.mass_matrix(torch.zeros(1, dtype=torch.float64))), expected)
        with self.assertRaises(ValueError):
            EmbodimentGraph.from_urdf(double_pendulum(), frozen_joints={"missing": 0.0})


class PhysicsFeatureTest(unittest.TestCase):
    """Check feature layouts, units, and actuator modelling."""

    def test_schema_matches_feature_sizes(self) -> None:
        """Every feature has a unit and frame, and sizes match the extractors."""
        graph = EmbodimentGraph.from_urdf(quadruped())
        specs = actuator_specs(graph)
        self.assertEqual(rigid_body_features(graph).shape, (12, RIGID_BODY_FEATURE_SIZE))
        self.assertEqual(actuator_features(specs).shape, (12, ACTUATOR_FEATURE_SIZE))
        q = torch.zeros(12, dtype=torch.float64)
        self.assertEqual(dynamics_features(graph, specs, q, q).shape, (12, DYNAMICS_FEATURE_SIZE))
        for features in PHYSICS_FEATURE_GROUPS.values():
            for feature in features:
                self.assertTrue(feature.unit and feature.frame and feature.symbol)
        self.assertEqual(set(FEATURE_UPDATE_TIERS), {f.name for g in PHYSICS_FEATURE_GROUPS.values() for f in g})

    def test_rigid_body_features_are_body_frame_quantities(self) -> None:
        """Mass, vech(I), CoM, axis, and T^parent are read in body frame B_i."""
        urdf = robot(
            "arm",
            link("base"),
            '<link name="upper"><inertial><origin xyz="0.1 0 -0.2" rpy="0 0 0"/><mass value="2"/>'
            '<inertia ixx="0.03" ixy="0.001" ixz="0.002" iyy="0.04" iyz="0.003" izz="0.05"/></inertial></link>',
            joint("shoulder", "base", "upper", xyz="0.5 0 1.0", rpy="0 0 1.5707963267948966", axis="1 0 0"),
        )
        features = rigid_body_features(EmbodimentGraph.from_urdf(urdf))[0]
        torch.testing.assert_close(features[0], torch.tensor(2.0))
        torch.testing.assert_close(features[1:7], torch.tensor([0.03, 0.001, 0.002, 0.04, 0.003, 0.05]))
        torch.testing.assert_close(features[7:10], torch.tensor([0.1, 0.0, -0.2]))
        torch.testing.assert_close(features[10:13], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(features[13:16], torch.tensor([0.5, 0.0, 1.0]))
        # First rotation column of Rz(pi/2) is (0, 1, 0).
        torch.testing.assert_close(features[16:19], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(features[22:25], torch.tensor([1.0, 0.0, 0.0]))

    def test_actuator_specs_defaults_overrides_and_errors(self) -> None:
        """URDF limits fill specs, glob and exact overrides apply in order."""
        graph = EmbodimentGraph.from_urdf(quadruped())
        specs = actuator_specs(
            graph,
            overrides={"*_calf_joint": {"peak_torque": 30.0, "kp": 20.0}, "FL_calf_joint": {"kp": 5.0}},
            defaults=ActuatorDefaults(continuous_ratio=0.4, mode="position"),
        )
        by_name = dict(zip(graph.joint_names, specs))
        self.assertEqual(by_name["FL_hip_joint"].peak_torque, 10.0)
        self.assertAlmostEqual(by_name["FL_hip_joint"].continuous_torque, 4.0)
        self.assertEqual(by_name["RR_calf_joint"].peak_torque, 30.0)
        self.assertEqual(by_name["RR_calf_joint"].kp, 20.0)
        self.assertEqual(by_name["FL_calf_joint"].kp, 5.0)
        self.assertEqual(by_name["FL_hip_joint"].mode, "position")
        with self.assertRaises(ValueError):
            actuator_specs(EmbodimentGraph.from_urdf(double_pendulum()))
        expanded = EmbodimentGraph.from_urdf(humanoid(), multi_dof_joints="expand")
        modes = [spec.mode for spec in actuator_specs(expanded)]
        self.assertEqual(modes[:6], ["passive"] * 6)
        with self.assertRaises(ValueError):
            ActuatorSpec(peak_torque=1.0, continuous_torque=2.0, max_velocity=1.0, max_power=1.0)

    def test_torque_envelope_and_motor_labels(self) -> None:
        """The envelope caps torque by power and speed; flags separate peak and continuous."""
        spec = ActuatorSpec(peak_torque=10.0, continuous_torque=4.0, max_velocity=5.0, max_power=20.0)
        tensors = actuator_tensors([spec])
        speeds = torch.tensor([[0.0], [1.0], [4.0], [6.0]], dtype=torch.float64)
        torch.testing.assert_close(
            torque_limit(tensors, speeds), torch.tensor([[10.0], [10.0], [5.0], [0.0]], dtype=torch.float64)
        )
        urdf = robot("one", link("base"), link("body", 1.0, inertia=0.01), joint("j", "base", "body"))
        graph = EmbodimentGraph.from_urdf(urdf)
        q = torch.zeros((3, 1), dtype=torch.float64)
        qd = torch.zeros_like(q)
        torque = torch.tensor([[3.0], [6.0], [12.0]], dtype=torch.float64)
        flags, margins, mask = motor_labels(graph, tensors, q, qd, torque)
        torch.testing.assert_close(flags[:, 0], torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64))
        torch.testing.assert_close(margins[0, 0, :2], torch.tensor([0.7, 0.25], dtype=torch.float64), atol=1e-5, rtol=0)
        self.assertTrue(bool(mask.all()))
        pd = ActuatorSpec(10.0, 4.0, 5.0, 20.0, kp=50.0, kd=2.0, mode="position")
        request = requested_torque(
            actuator_tensors([pd]),
            torch.tensor([[0.1]], dtype=torch.float64),
            torch.tensor([[0.0]], dtype=torch.float64),
            torch.tensor([[1.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(request, torch.tensor([[3.0]], dtype=torch.float64))


class PhysicsCacheTest(unittest.TestCase):
    """Check tiered recomputation of dynamics features."""

    def test_cache_matches_direct_computation_and_skips_refreshes(self) -> None:
        """Interval 1 equals direct features; longer intervals reuse configuration terms."""
        graphs = [EmbodimentGraph.from_urdf(quadruped()), EmbodimentGraph.from_urdf(humanoid())]
        specs = [actuator_specs(graph) for graph in graphs]
        cache = PhysicsFeatureCache(graphs, specs)
        joints = cache.joint_mask.shape[1]
        q, qd, torque = random_state(graphs[1], batch=2, seed=3)
        q, qd, torque = (value[:, :joints].float() for value in (q, qd, torque))
        features = cache.update(q, qd, torque)
        for row, graph in enumerate(graphs):
            count = graph.num_joints
            expected = dynamics_features(graph, specs[row], q[row, :count], qd[row, :count], torque[row, :count])
            torch.testing.assert_close(features[row, :count], expected, rtol=1e-4, atol=1e-5)
            self.assertTrue(torch.all(features[row, count:] == 0))
        slow = PhysicsFeatureCache(graphs, specs, configuration_interval=4, configuration_tolerance=0.5)
        for step in range(8):
            slow.update(q + 0.01 * step, qd, torque)
        self.assertEqual(slow.refresh_counts["configuration"], 2 * 2)
        slow.update(q + 1.0, qd, torque)
        self.assertEqual(slow.refresh_counts["configuration"], 2 * 3)


class PhysicsCrossAttentionTest(unittest.TestCase):
    """Check the encoder's Physics Cross Attention and its memory cache."""

    def setUp(self) -> None:
        """Build a padded batch of two robots with physics features."""
        torch.manual_seed(0)
        self.graphs = [EmbodimentGraph.from_urdf(quadruped()), EmbodimentGraph.from_urdf(humanoid())]
        self.batch = collate_embodiments(self.graphs)
        specs = [actuator_specs(graph) for graph in self.graphs]
        self.cache = PhysicsFeatureCache(self.graphs, specs)
        joints = self.batch.joint_mask.shape[1]
        self.q = torch.zeros(2, joints)
        self.dynamics = self.cache.update(self.q, torch.zeros_like(self.q))
        config = GetConfig(
            input_size=self.batch.joint_features.shape[-1],
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=3,
            intermediate_size=64,
            use_physics_cross_attention=True,
            physics_cross_attention_layers=(0, 2),
        )
        self.encoder = GetEncoder(config).eval()

    def inputs(self) -> dict:
        """Return encoder keyword arguments except physics."""
        return {
            "mass_matrix": self.batch.mass_matrix(self.q.double()).float(),
            **self.batch.encoder_inputs(),
        }

    def test_memory_equals_recomputation_and_detects_staleness(self) -> None:
        """Cached static keys and values give identical outputs until weights change."""
        tokens = self.batch.joint_features
        direct = self.encoder(
            tokens,
            rigid_body_features=self.cache.rigid_body_features,
            actuator_features=self.cache.actuator_features,
            dynamics_features=self.dynamics,
            **self.inputs(),
        )
        memory = self.encoder.build_physics_memory(
            self.cache.rigid_body_features, self.cache.actuator_features, self.batch.joint_mask
        )
        self.assertEqual(len(memory.keys), 2)
        cached = self.encoder(tokens, dynamics_features=self.dynamics, physics_memory=memory, **self.inputs())
        torch.testing.assert_close(direct, cached)
        with torch.no_grad():
            self.encoder.physics_tokenizers["actuator"].group_embedding.add_(1.0)
        with self.assertRaises(ValueError):
            self.encoder(tokens, dynamics_features=self.dynamics, physics_memory=memory, **self.inputs())

    def test_physics_changes_output_and_padding_is_invariant(self) -> None:
        """Physics features matter, and the quadruped encodes the same alone or padded."""
        tokens = self.batch.joint_features
        kwargs = {
            "rigid_body_features": self.cache.rigid_body_features,
            "actuator_features": self.cache.actuator_features,
        }
        base = self.encoder(tokens, dynamics_features=self.dynamics, **kwargs, **self.inputs())
        heavier = self.encoder(tokens, dynamics_features=self.dynamics * 2, **kwargs, **self.inputs())
        self.assertGreater(float((base - heavier).abs().max().detach()), 1e-4)
        single = collate_embodiments(self.graphs[:1])
        count = self.graphs[0].num_joints
        alone = self.encoder(
            single.joint_features,
            mass_matrix=single.mass_matrix(torch.zeros(1, count, dtype=torch.float64)).float(),
            rigid_body_features=self.cache.rigid_body_features[:1, :count],
            actuator_features=self.cache.actuator_features[:1, :count],
            dynamics_features=self.dynamics[:1, :count],
            **single.encoder_inputs(),
        )
        torch.testing.assert_close(alone[0], base[0, :count], rtol=1e-4, atol=1e-5)

    def test_physics_inputs_are_validated(self) -> None:
        """Missing dynamics raise; physics inputs to a non-physics encoder raise."""
        with self.assertRaises(ValueError):
            self.encoder(self.batch.joint_features, **self.inputs())
        plain = GetEncoder(GetConfig(input_size=self.batch.joint_features.shape[-1], hidden_size=32))
        with self.assertRaises(ValueError):
            plain(self.batch.joint_features, dynamics_features=self.dynamics, **self.inputs())


if __name__ == "__main__":
    unittest.main()
