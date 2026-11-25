"""Test the options of the zero-shot proposal (docs/zero_shot.md): A1-A3, B1-B2, C1-C3, D2, D4, E1-E3, G1, R6."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from model import DaGetConfig, DaGetEncoder
from model.centroidal import CENTROIDAL_SIZE, centroidal_state, centroidal_targets
from model.configuration_bfm import BfmConfig
from model.dynamics import floating_forward_dynamics
from model.embodiment import EmbodimentGraph, JointFeatureConfig, collate_embodiments
from model.modeling_bfm import DaGetBfmModel
from model.modeling_da_get import physics_units
from model.physics import (
    actuator_features,
    actuator_specs,
    dynamics_features,
    froude_scale_base_features,
    rigid_body_features,
    root_features,
)
from model.physics_schema import (
    REFERENCE_LENGTH,
    REFERENCE_MASS,
    ROOT_FEATURE_SIZE,
    dimensionless_scales,
    feature_scales,
    reference_unit,
)
from model.pretraining import (
    ALL_LOSS_NAMES,
    DaGetForPhysicsPretraining,
    DaGetPhysicsPretrainingConfig,
    gauss_metric_error,
    interface_units,
)
from model.rl.observation import StateView
from model.rl.trainer import action_scale
from model.training import PhysicsPretrainingTrainer, build_physics_datasets
from model.training.dataset import SPEC_NOISE_KEYS, PhysicsDataCollator, PhysicsTrajectoryDataset
from model.training.episodes import EmbodimentRecord, Trajectory, exp_rotation
from model.training.ground_sim import ground_specs
from model.training.resampling import resample_trajectory
from model.training.sampling import sample_groups, temperature_weights
from model.urdf import parse_urdf
from tests.test_embodiment import quadruped

# The robot URDFs in assets/urdf are downloaded, not committed (scripts/download_urdfs.py).
NEEDS_URDFS = unittest.skipUnless(
    all((Path(__file__).resolve().parents[1] / "assets" / "urdf" / name).exists() for name in ("go1.urdf", "h1.urdf", "anymal_d.urdf")),
    "needs the robot URDFs: uv run python scripts/download_urdfs.py",
)


ROOT = Path(__file__).resolve().parents[1]
GO1 = ROOT / "assets" / "urdf" / "go1.urdf"


def go1_graph(floating: bool = True) -> EmbodimentGraph:
    """Return the Go1 graph."""
    return EmbodimentGraph.from_urdf(parse_urdf(GO1), JointFeatureConfig.da_get_zero(), floating_base=floating)


def tiny_encoder_config(input_size: int, **overrides) -> DaGetConfig:
    """Return a tiny encoder configuration with Physics Cross Attention."""
    return DaGetConfig.from_size(input_size, "tiny", use_physics_cross_attention=True, **overrides)


def encoder_inputs(graph: EmbodimentGraph, batch: int = 2, input_size: int = 12, mass_scale: float = 1.0) -> dict:
    """Return encoder inputs of ``batch`` copies of a robot whose masses and torques are scaled."""
    torch.manual_seed(0)
    specs = actuator_specs(graph)
    structure = collate_embodiments([graph] * batch)
    q = torch.zeros(graph.num_joints, dtype=torch.float64)
    rigid = rigid_body_features(graph).clone()
    rigid[:, 0] *= mass_scale
    rigid[:, 1:7] *= mass_scale
    rigid[:, 25] *= mass_scale
    actuator = actuator_features(specs).clone()
    actuator[:, :4] *= mass_scale
    dynamics = dynamics_features(graph, specs, q, q).clone()
    dynamics[:, :4] *= mass_scale
    root = root_features(graph).clone()
    root[[0, 4, 5, 6, 7, 8, 9, 10]] *= mass_scale
    generator = torch.Generator().manual_seed(1)
    return {
        "input_values": torch.randn(batch, graph.num_joints, input_size, generator=generator),
        "parent_indices": structure.parent_indices,
        "joint_mask": structure.joint_mask,
        "graph_distances": structure.graph_distances,
        "mass_matrix": (graph.mass_matrix(q) * mass_scale).float()[None].expand(batch, -1, -1),
        "rigid_body_features": rigid[None].expand(batch, -1, -1),
        "actuator_features": actuator[None].expand(batch, -1, -1),
        "dynamics_features": dynamics[None].expand(batch, -1, -1).float(),
        "root_features": root[None].expand(batch, -1),
    }


@NEEDS_URDFS
class SchemaAndEncoderTest(unittest.TestCase):
    """A3, C1, D4 and E3 in the encoder."""

    def test_reference_robot_keeps_si_tokenization(self) -> None:
        """Dimensionless scales equal SI scales divided by the reference robot's units."""
        for group in ("rigid_body", "actuator", "dynamics"):
            for si, dimensionless, feature_unit in zip(
                feature_scales(group), dimensionless_scales(group),
                [reference_unit(item) for item in __import__("model.physics_schema", fromlist=["x"]).dimension_exponents(group)[0]],
            ):
                if si:
                    self.assertAlmostEqual(dimensionless * feature_unit, si, places=12)
        self.assertGreater(REFERENCE_MASS, 0)
        self.assertGreater(REFERENCE_LENGTH, 0)

    def test_root_features(self) -> None:
        """Root features hold the base body, total mass and size."""
        graph = go1_graph()
        features = root_features(graph)
        self.assertEqual(features.shape, (ROOT_FEATURE_SIZE,))
        self.assertAlmostEqual(float(features[10]), graph.total_mass, places=3)
        self.assertAlmostEqual(float(features[11]), graph.characteristic_length, places=5)
        self.assertEqual(float(features[12]), 1.0)

    def test_units_of_prismatic_joints(self) -> None:
        """Torque units are m g L for revolute and m g for prismatic joints."""
        mass, length = torch.tensor([10.0]), torch.tensor([0.5])
        units = physics_units("actuator", mass, length, torch.tensor([[False, True]]))
        self.assertAlmostEqual(float(units[0, 0, 0]), 10.0 * 9.80665 * 0.5, places=4)
        self.assertAlmostEqual(float(units[0, 1, 0]), 10.0 * 9.80665, places=4)

    def test_dimensionless_encoder_is_mass_scale_invariant(self) -> None:
        """Keep joint physics dimensionless while exposing mass in the root token.

        Returns:
            None: Asserts non-root physics tokens are invariant under mass
            scaling and the absolute-scale root token can change the output.
        """
        graph = go1_graph()
        encoder = DaGetEncoder(tiny_encoder_config(12, dimensionless_physics=True)).eval()
        first_inputs = encoder_inputs(graph, mass_scale=1.0)
        second_inputs = encoder_inputs(graph, mass_scale=3.0)
        with torch.no_grad():
            first_memory = encoder.build_physics_memory(
                first_inputs["rigid_body_features"], first_inputs["actuator_features"],
                first_inputs["joint_mask"], first_inputs["root_features"],
            )
            second_memory = encoder.build_physics_memory(
                second_inputs["rigid_body_features"], second_inputs["actuator_features"],
                second_inputs["joint_mask"], second_inputs["root_features"],
            )
            first = encoder(**first_inputs)
            second = encoder(**second_inputs)
        torch.testing.assert_close(first_memory.keys[0][:, :, :-1], second_memory.keys[0][:, :, :-1], atol=2e-4, rtol=1e-4)
        torch.testing.assert_close(first_memory.values[0][:, :, :-1], second_memory.values[0][:, :, :-1], atol=2e-4, rtol=1e-4)
        self.assertFalse(torch.allclose(first_memory.keys[0][:, :, -1], second_memory.keys[0][:, :, -1]))
        self.assertFalse(torch.allclose(first, second))

    def test_root_token_extends_memory(self) -> None:
        """Require the root token and its features in physics memory.

        Returns:
            None: Asserts static memory width and root-feature sensitivity.
        """
        graph = go1_graph()
        encoder = DaGetEncoder(tiny_encoder_config(12)).eval()
        inputs = encoder_inputs(graph)
        memory = encoder.build_physics_memory(inputs["rigid_body_features"], inputs["actuator_features"],
                                              inputs["joint_mask"], inputs["root_features"])
        self.assertEqual(memory.keys[0].shape[2], 2 * graph.num_joints + 1)
        with torch.no_grad():
            output = encoder(**inputs)
            changed = dict(inputs, root_features=inputs["root_features"] * 2)
            self.assertFalse(torch.allclose(output, encoder(**changed)))
        with self.assertRaises(ValueError):
            encoder(**{key: value for key, value in inputs.items() if key != "root_features"})

    def test_modulation_starts_as_identity(self) -> None:
        """Contextual modulation starts at zero, so it equals the plain encoder with the same weights."""
        graph = go1_graph()
        plain = DaGetEncoder(tiny_encoder_config(12)).eval()
        modulated = DaGetEncoder(tiny_encoder_config(12, use_morphology_modulation=True)).eval()
        missing = modulated.load_state_dict(plain.state_dict(), strict=False).missing_keys
        self.assertTrue(all(key.startswith("modulation.") for key in missing))
        inputs = encoder_inputs(graph)
        with torch.no_grad():
            torch.testing.assert_close(plain(**inputs), modulated(**inputs))

    def test_masked_physics_tokens_change_the_output(self) -> None:
        """Masked rigid-body tokens are replaced by the mask embedding."""
        graph = go1_graph()
        encoder = DaGetEncoder(tiny_encoder_config(12, physics_masking=True)).eval()
        with torch.no_grad():
            encoder.physics_mask_embeddings["rigid_body"].fill_(1.0)
            inputs = encoder_inputs(graph)
            mask = torch.zeros_like(inputs["joint_mask"])
            mask[:, :3] = True
            self.assertFalse(torch.allclose(encoder(**inputs), encoder(**inputs, physics_token_mask={"rigid_body": mask})))

    def test_options_need_physics_cross_attention(self) -> None:
        """Reject physics-specific options without Physics Cross Attention.

        Returns:
            None: Asserts that incompatible encoder options are rejected.
        """
        for option in ("dimensionless_physics", "use_morphology_modulation", "physics_masking"):
            with self.assertRaises(ValueError):
                DaGetConfig.from_size(12, "tiny", **{option: True})

    def test_legacy_checkpoint_without_root_token_is_rejected(self) -> None:
        """Accept old root-enabled settings and reject root-free checkpoints.

        Returns:
            None: Asserts compatibility with root-enabled checkpoints only.
        """
        legacy = {
            "input_size": 12,
            "use_physics_cross_attention": True,
            "use_root_token": True,
        }
        self.assertTrue(DaGetPhysicsPretrainingConfig(encoder=legacy).encoder_config.needs_root_features)
        with self.assertRaisesRegex(ValueError, "without root token"):
            DaGetPhysicsPretrainingConfig(
                encoder={**legacy, "use_root_token": False}
            ).encoder_config


@NEEDS_URDFS
class CentroidalTest(unittest.TestCase):
    """C3 labels."""

    def test_momentum_rate_in_free_flight_is_gravity(self) -> None:
        """Without contacts, dp/dt = m g and the angular momentum about the CoM is conserved."""
        torch.manual_seed(0)
        graph = go1_graph()
        q = torch.randn(1, 12, dtype=torch.float64) * 0.3
        qd = torch.randn(1, 12, dtype=torch.float64)
        v, w = torch.randn(1, 3, dtype=torch.float64), torch.randn(1, 3, dtype=torch.float64)
        gravity = torch.tensor([[0.0, 0.0, -9.80665]], dtype=torch.float64)
        linear, angular, qdd = floating_forward_dynamics(graph, q, qd, torch.zeros_like(q), gravity, v, w, include_friction=False)
        dt = 1e-6
        start = centroidal_state(graph, q, qd, v, w)
        w_next = w + dt * angular
        v_next = v + dt * (linear - torch.linalg.cross(w, v, dim=-1))
        end = centroidal_state(graph, q + dt * qd, qd + dt * qdd, v_next, w_next)
        rotation = exp_rotation(w * dt)
        rate = ((rotation @ end.linear_momentum[..., None]).squeeze(-1) - start.linear_momentum) / dt
        spin = ((rotation @ end.angular_momentum[..., None]).squeeze(-1) - start.angular_momentum) / dt
        torch.testing.assert_close(rate, graph.total_mass * gravity, atol=1e-2, rtol=1e-4)
        self.assertLess(float(spin.norm()), 1e-2)

    def test_targets_are_dimensionless(self) -> None:
        """Targets have 21 columns; a still robot has zero momentum and its CoM within a few L."""
        graph = go1_graph(floating=False)
        q = torch.zeros(5, 12, dtype=torch.float64)
        targets, valid = centroidal_targets(graph, q, torch.zeros_like(q), 0.02)
        self.assertEqual(targets.shape, (5, CENTROIDAL_SIZE))
        self.assertTrue(bool(valid[:, 0].all()))
        self.assertLess(float(targets[:, 3:9].abs().max()), 1e-12)
        self.assertLess(float(targets[:, :3].norm(dim=-1).max()), 3.0)


class DataPipelineTest(unittest.TestCase):
    """A1, A2, A3, C2, D2 and E2 in the data pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a tiny synthetic floating-base dataset at a 50 Hz control period."""
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        (root / "quadruped.urdf").write_text(quadruped())
        cls.config = {
            "history_size": 2,
            "eval_fraction": 0.0,
            "control_dt": 0.02,
            "sources": [
                {
                    "type": "synthetic",
                    "name": "quadruped",
                    "family": "quadruped",
                    "urdf": "quadruped.urdf",
                    "randomized_embodiments": 2,
                    "base_modes": ["fixed", "floating"],
                    "episodes_per_embodiment": 1,
                    "steps": 8,
                    "probe_torques": 1,
                }
            ],
        }
        (root / "data.json").write_text(json.dumps(cls.config))
        cls.root = root
        cls.datasets = build_physics_datasets(root / "data.json", cache_dir=root / "cache")

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove temporary files."""
        cls.directory.cleanup()

    def collate(self, items, **settings):
        """Collate dataset items with modified collator settings."""
        collator = PhysicsDataCollator(self.datasets.records, self.datasets.history_size, replace(self.datasets.settings, **settings))
        return collator(items)

    def test_control_dt_reaches_the_simulator(self) -> None:
        """Synthetic sources simulate at the control period."""
        for _, episode in self.datasets.train.episodes:
            self.assertAlmostEqual(float(episode["dt"]), 0.02)
        self.assertEqual(self.datasets.records[0].family, "quadruped")

    def test_physics_normalization_uses_inputs_only(self) -> None:
        """FD targets are qdd M_ii / tau^peak, base targets over g, effect in Froude units."""
        items = [self.datasets.train[index] for index in range(4)]
        batch = self.collate(items, target_normalization="physics")
        diagonal = batch["mass_matrix"].diagonal(dim1=1, dim2=2)
        record = self.datasets.records[items[0]["embodiment"]]
        peak = record.actuator["peak_torque"].float()
        expected = items[0]["acceleration"].float() * diagonal[0] / peak
        torch.testing.assert_close(batch["fd_target"][0], expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(batch["base_acceleration_scale"][0, :3], torch.full((3,), 9.80665))
        frequency = math.sqrt(9.80665 / record.graph.characteristic_length)
        self.assertAlmostEqual(float(batch["effect_scales"][0, 1]), frequency, places=4)

    def test_dimensionless_tokens(self) -> None:
        """Base angular velocity and linear velocity / L are multiplied by sqrt(L / g)."""
        features = torch.ones(2, 11)
        scaled = froude_scale_base_features(features, torch.tensor([1.0, 4.0]))
        self.assertAlmostEqual(float(scaled[1, 4]), math.sqrt(4.0 / 9.80665), places=6)
        self.assertEqual(float(scaled[1, 0]), 1.0)
        moving = [index for index in range(len(self.datasets.train))
                  if float(self.datasets.train[index]["variable_global"][:, 3:9].abs().sum()) > 0]
        self.assertTrue(moving)
        items = [self.datasets.train[index] for index in moving[:2]]
        plain, scaled = self.collate(items), self.collate(items, dimensionless_inputs=True)
        self.assertFalse(torch.allclose(plain["input_values"], scaled["input_values"]))

    def test_spec_noise_labels(self) -> None:
        """The given spec is perturbed and sysid labels are log(true / given)."""
        items = [self.datasets.train[index] for index in range(3)]
        torch.manual_seed(0)
        noise = {key: (0.8, 1.2) for key in SPEC_NOISE_KEYS}
        clean, noisy = self.collate(items), self.collate(items, spec_noise=noise)
        ratio = clean["actuator_features"][..., 0] / noisy["actuator_features"][..., 0].clamp_min(1e-12)
        valid = noisy["sysid_mask"][..., 1]
        torch.testing.assert_close(ratio.log()[valid], noisy["sysid_target"][..., 1][valid], rtol=1e-4, atol=1e-5)
        self.assertTrue(bool(valid.any()))
        torch.testing.assert_close(clean["fd_target"], noisy["fd_target"])

    def test_grouped_variants_match_ungrouped_losses(self) -> None:
        """Encoding each state once gives the same per-variant losses as one sample per variant."""
        records = self.datasets.records
        episodes = self.datasets.train.episodes[:1]
        flat = PhysicsTrajectoryDataset(records, episodes, self.datasets.history_size)
        grouped = PhysicsTrajectoryDataset(records, episodes, self.datasets.history_size, group_variants=True)
        self.assertLess(len(grouped), len(flat))
        config = DaGetPhysicsPretrainingConfig.from_encoder_config(
            tiny_encoder_config(self.datasets.token_size)
        )
        model = DaGetForPhysicsPretraining(config).eval()
        collator = self.datasets.collator()
        with torch.no_grad():
            first = model(**collator([flat[index] for index in range(len(flat))]))
            second = model(**collator([grouped[index] for index in range(len(grouped))]))
        for name in ("fd", "id", "motor", "effect", "fk"):
            torch.testing.assert_close(first.loss_components[name], second.loss_components[name], rtol=1e-4, atol=1e-6)
        self.assertEqual(second.predictions["fd_acceleration"].shape[0], len(flat))

    def test_temperature_weights(self) -> None:
        """alpha = 1 keeps the data distribution; alpha = 0 is uniform over groups."""
        groups = ["a"] * 6 + ["b"] * 2
        uniform = temperature_weights(groups, 1.0)
        torch.testing.assert_close(uniform, torch.full((8,), 1 / 8, dtype=torch.float64))
        balanced = temperature_weights(groups, 0.0)
        self.assertAlmostEqual(float(balanced[:6].sum()), 0.5)
        labels = sample_groups(self.datasets.train, self.datasets.records, "family")
        self.assertEqual(set(labels), {"quadruped"})
        trainer_weights = temperature_weights(labels, 0.3)
        self.assertEqual(trainer_weights.shape[0], len(self.datasets.train))

    def test_weighted_sampler(self) -> None:
        """The trainer draws with the given weights."""
        from torch.utils.data import WeightedRandomSampler
        from transformers import TrainingArguments

        config = DaGetPhysicsPretrainingConfig.from_encoder_config(tiny_encoder_config(self.datasets.token_size))
        with tempfile.TemporaryDirectory() as output:
            trainer = PhysicsPretrainingTrainer(
                model=DaGetForPhysicsPretraining(config),
                args=TrainingArguments(output_dir=output, report_to="none", remove_unused_columns=False, use_cpu=True),
                train_dataset=self.datasets.train,
                data_collator=self.datasets.collator(),
                sample_weights=torch.ones(len(self.datasets.train), dtype=torch.float64),
            )
            self.assertIsInstance(trainer._get_train_sampler(), WeightedRandomSampler)

    def test_all_heads_train(self) -> None:
        """Train all auxiliary heads with the root token present.

        Returns:
            None: Asserts finite, differentiable losses with all heads enabled.
        """
        encoder = tiny_encoder_config(
            self.datasets.token_size, dimensionless_physics=True,
            use_morphology_modulation=True, physics_masking=True,
        )
        config = DaGetPhysicsPretrainingConfig.from_encoder_config(
            encoder, head_type="linear_residual", acceleration_metric="mass", effect_mode="rollout",
            use_contact_head=True, use_centroidal_head=True, use_sysid_head=True, physics_mask_ratio=0.5,
        )
        model = DaGetForPhysicsPretraining(config).train()
        items = [self.datasets.train[index] for index in range(len(self.datasets.train))]
        torch.manual_seed(0)
        batch = self.collate(items, include_graphs=True, spec_noise={key: (0.8, 1.2) for key in SPEC_NOISE_KEYS},
                             target_normalization="physics")
        output = model(**batch)
        expected = {"operator", "consistency", "effect_rollout", "centroidal", "mask", "sysid", "fd", "response"}
        self.assertTrue(expected <= set(output.loss_components))
        self.assertTrue(set(output.loss_components) <= set(ALL_LOSS_NAMES))
        for name, value in output.loss_components.items():
            self.assertTrue(torch.isfinite(value), name)
        output.loss.backward()
        self.assertIsNotNone(model.interface.joint_factor.weight.grad)

    def test_linear_interface_reproduces_true_operator(self) -> None:
        """With the true G and its pseudo-inverse, R and R^-1 readouts match their targets (unit check)."""
        config = DaGetPhysicsPretrainingConfig.from_encoder_config(
            tiny_encoder_config(self.datasets.token_size), head_type="linear"
        )
        model = DaGetForPhysicsPretraining(config).eval()
        items = [self.datasets.train[index] for index in range(len(self.datasets.train))]
        batch = self.collate(items)
        references, scale = interface_units(batch, config.epsilon)
        operator = batch["response_operator"] * references[:, :, None] * references[:, None, 6:] / scale[:, None, None]
        interface = {"operator": operator, "inverse": torch.linalg.pinv(operator), "drift": torch.zeros_like(references)}
        joint, base = model._interface_response(batch, interface, (references, scale))
        mask = batch["response_mask"]
        torch.testing.assert_close(joint[mask], batch["response_target"][mask], rtol=1e-3, atol=1e-4)
        rows = batch["response_base_mask"]
        torch.testing.assert_close(base[rows], batch["response_base_target"][rows], rtol=1e-3, atol=1e-4)
        change = model._interface_inverse_response(batch, interface, (references, scale))
        projection = batch["response_projection"]
        peak = batch["response_torque_scale"]
        error = (projection @ ((change - batch["inverse_response_target"]) * peak)[..., None]).squeeze(-1)
        self.assertLess(float(error[batch["inverse_response_mask"]].abs().max()), 1e-2 * float(peak.max()))

    def test_gauss_metric_with_identity_is_mean_square(self) -> None:
        """With an identity mass matrix the Gauss metric is the mean square error."""
        error = torch.tensor([[1.0, 2.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        value = gauss_metric_error(error, torch.eye(3)[None], mask, 1e-8)
        self.assertAlmostEqual(float(value), 2.5, places=5)


@NEEDS_URDFS
class ContactLabelTest(unittest.TestCase):
    """C2 labels from per-link flags."""

    def test_contact_flags_map_to_joints(self) -> None:
        """A contact link's flag lands on the joint carrying it; only trusted frames are labeled."""
        graph = go1_graph()
        record = EmbodimentRecord.build("go1", graph, actuator_specs(graph), contact_links=["FR_foot", "FL_foot", "RR_foot", "RL_foot"])
        owners = record.contact_joints
        self.assertTrue(bool((owners >= 0).all()))
        flags = torch.tensor([[True, False, True, False], [True, True, True, True]])
        episode = {"contact_flags": flags, "contact_label_valid": torch.tensor([True, False])}
        dataset = PhysicsTrajectoryDataset([record], [], 1)
        target, mask = dataset._contact_labels(record, episode, 0)
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(float(target[owners[0]]), 1.0)
        self.assertEqual(float(target[owners[1]]), 0.0)
        _, untrusted = dataset._contact_labels(record, episode, 1)
        self.assertFalse(bool(untrusted.any()))


class ResamplingTest(unittest.TestCase):
    """A2 resampling of recorded data."""

    def test_resample_30_to_50_hz(self) -> None:
        """Linear signals are interpolated exactly; orientations stay rotations; commands are held."""
        steps = 31
        time = torch.arange(steps, dtype=torch.float64) / 30
        positions = torch.stack((time, 2 * time), dim=-1)
        orientation = exp_rotation(torch.stack((torch.zeros_like(time), torch.zeros_like(time), time), dim=-1))
        trajectory = Trajectory(0, 1 / 30, positions, actions=positions.clone(), base_orientation=orientation)
        resampled = resample_trajectory(trajectory, 0.02)
        self.assertEqual(resampled.positions.shape[0], 51)
        new_time = torch.arange(51, dtype=torch.float64) * 0.02
        torch.testing.assert_close(resampled.positions[:, 1], 2 * new_time)
        expected = exp_rotation(torch.stack((torch.zeros_like(new_time), torch.zeros_like(new_time), new_time), dim=-1))
        torch.testing.assert_close(resampled.base_orientation, expected, atol=1e-9, rtol=1e-9)
        self.assertTrue(bool((resampled.actions[:, 0] <= new_time + 1e-12).all()))

    def test_simulated_data_is_not_resampled(self) -> None:
        """Exact simulator labels belong to their period."""
        trajectory = Trajectory(0, 0.01, torch.zeros(3, 2), applied_torque=torch.zeros(3, 2), exact_dynamics=True)
        with self.assertRaises(ValueError):
            resample_trajectory(trajectory, 0.02)


@NEEDS_URDFS
class BfmSideTest(unittest.TestCase):
    """R6, B2 and G1."""

    def setUp(self) -> None:
        """Build a tiny pretrained model with the physics interface and a BFM around it."""
        self.graph = go1_graph()
        encoder = tiny_encoder_config(12)
        self.pretrained = DaGetForPhysicsPretraining(
            DaGetPhysicsPretrainingConfig.from_encoder_config(encoder, head_type="linear")
        )

    def test_state_path_uses_the_full_mass_matrix(self) -> None:
        """A state's full mass matrix changes the state embedding (the diagonal is normalized away)."""
        config = BfmConfig.from_sizes(embedding_size=64, state_token_size=10, actor_size="tiny", forward_size="tiny")
        model = DaGetBfmModel(config, self.pretrained.encoder).eval()
        inputs = encoder_inputs(self.graph, input_size=10)
        state = {key: value for key, value in inputs.items() if key != "mass_matrix"}
        with torch.no_grad():
            diagonal = model.encode_state(state)
            full = model.encode_state({**state, "mass_matrix": inputs["mass_matrix"]})
        self.assertFalse(torch.allclose(diagonal, full))
        view = StateView(torch.zeros(2, 12, 5), torch.zeros(2, 10), torch.zeros(2, 12, 5), torch.eye(12).expand(2, -1, -1))
        self.assertEqual(view.flatten(0, 0).mass.shape, (2, 12, 12))
        self.assertEqual(StateView.stack([view, view]).mass.shape, (2, 2, 12, 12))

    def test_interface_bias_changes_actions(self) -> None:
        """The actor reads the interface operator as an attention bias."""
        config = BfmConfig.from_sizes(embedding_size=64, state_token_size=10, actor_size="tiny", forward_size="tiny",
                                      use_interface_bias=True)
        model = DaGetBfmModel(config, self.pretrained.encoder, physics_interface=self.pretrained.interface).eval()
        embeddings = torch.randn(2, 12, 64)
        mask = torch.ones(2, 12, dtype=torch.bool)
        z = model.sample_z(2)
        operator = model.interface_operator(embeddings, mask)
        torch.testing.assert_close(operator.diagonal(dim1=1, dim2=2), torch.ones(2, 12), atol=1e-5, rtol=1e-5)
        with torch.no_grad():
            for projection in model.actor.interface_projections:
                projection.weight.fill_(1.0)
            with_bias = model.actor(embeddings, z, 0.05, mask).mean
            model.actor.operator_source = None
            without = model.actor(embeddings, z, 0.05, mask).mean
        self.assertTrue(bool(torch.isfinite(with_bias).all()))
        self.assertFalse(torch.allclose(with_bias, without))
        with self.assertRaises(ValueError):
            DaGetBfmModel(config, self.pretrained.encoder)

    def test_gain_modes_and_action_scale(self) -> None:
        """Mass-proportional gains (GenLoco) and per-family action scales."""
        specs = actuator_specs(self.graph)
        scaled = ground_specs(specs, gain_mode="mass", total_mass=10.0, kp_per_kg=2.0)
        self.assertTrue(all(spec.kp == 20.0 for spec in scaled if spec.mode != "passive"))
        self.assertTrue(all(abs(spec.kd - 0.6) < 1e-12 for spec in scaled if spec.mode != "passive"))
        with self.assertRaises(ValueError):
            ground_specs(specs, gain_mode="mass")
        self.assertEqual(action_scale({"humanoid": 0.5, "default": 1.0}, "humanoid"), 0.5)
        self.assertEqual(action_scale({"humanoid": 0.5, "default": 1.0}, "hexapod"), 1.0)
        self.assertEqual(action_scale(1.25, "quadruped"), 1.25)


if __name__ == "__main__":
    unittest.main()


class ZeroShotConfigTest(unittest.TestCase):
    """The shipped zero-shot configs keep every held-out robot out of training."""

    def test_configs_respect_the_split_manifest(self) -> None:
        """No L1-L4 robot (H1, Go2, Talos, hexapods, L2 topologies) is a training source."""
        from model.evaluation.splits import SplitManifest

        manifest = SplitManifest.load(ROOT / "configs" / "splits" / "zero_shot.json")
        for name in ("configs/data/zero_shot.json", "configs/rl/zero_shot_procedural.json"):
            self.assertEqual(manifest.check_training_config(ROOT / name), [], name)
        self.assertTrue(manifest.check_training_config(ROOT / "configs" / "data" / "ground_contact.json"))

    def test_pretraining_arguments_parse(self) -> None:
        """Parse shipped arguments with the mandatory root token.

        Returns:
            None: Asserts valid model and data arguments without a root toggle.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("train_physics", ROOT / "scripts" / "train_physics_da_get.py")
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from transformers import HfArgumentParser

        parser = HfArgumentParser((module.ModelArguments, module.DataArguments, module.PretrainingArguments))
        model_args, data_args, _ = parser.parse_json_file(str(ROOT / "configs" / "pretraining" / "zero_shot.json"))
        self.assertEqual(model_args.head_type, "linear_residual")
        self.assertEqual(data_args.data_config, "configs/data/zero_shot.json")
        encoder = module.encoder_config(model_args, 64)
        self.assertTrue(encoder.needs_root_features and encoder.dimensionless_physics and encoder.physics_masking)
