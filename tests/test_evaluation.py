"""Test the zero-shot evaluation harness: statistics, splits, probes, metrics, counterfactuals, scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import stats

from model import DaGetConfig
from model.evaluation.counterfactual import counterfactual_mass_response, denormalize_positions, perturb_body_mass
from model.evaluation.physics_metrics import (
    contact_f1,
    exact_label_dynamics_errors,
    fk_error_over_length,
    operator_from_response_head,
    relative_error,
    relative_frobenius_error,
    true_response_operators,
)
from model.evaluation.probes import (
    HeadTrainingSettings,
    MlpSettings,
    capture_layer_embeddings,
    collect_probe_data,
    control_targets,
    endless_batches,
    fit_ridge_probe,
    fixed_batches,
    head_capacity_test,
    head_reinit_transfer,
    layerwise_probes,
    r2_score,
    random_like,
    run_probe,
    split_groups,
)
from model.evaluation.splits import (
    L1_CATEGORIES,
    METAMORPH_VARIATIONS,
    RobotEntry,
    SplitManifest,
    make_heldout_split,
    name_matches,
)
from model.evaluation.statistics import (
    interquartile_mean,
    mean_score,
    performance_gap,
    robot_bootstrap_ci,
    stratified_bootstrap_ci,
)
from model.pretraining import DaGetForPhysicsPretraining, DaGetPhysicsPretrainingConfig
from model.training import build_physics_datasets
from tests.test_embodiment import humanoid, quadruped

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "splits" / "zero_shot.json"
_THREADS = torch.get_num_threads()


def setUpModule() -> None:
    """Use one CPU thread: the tensors are tiny and threads only add overhead."""
    torch.set_num_threads(1)


def tearDownModule() -> None:
    """Restore the thread count."""
    torch.set_num_threads(_THREADS)


def load_script(name: str):
    """Import a script of ``scripts/`` as a module."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_tiny_data(root: Path) -> Path:
    """Write a tiny synthetic data config: a fixed and a floating quadruped, and a humanoid."""
    (root / "quadruped.urdf").write_text(quadruped())
    (root / "humanoid.urdf").write_text(humanoid())
    config = {
        "history_size": 2,
        "eval_fraction": 0.25,
        "sources": [
            {"type": "synthetic", "name": "quadruped", "urdf": "quadruped.urdf", "randomized_embodiments": 2,
             "episodes_per_embodiment": 2, "steps": 12, "probe_torques": 1, "base_modes": ["fixed", "floating"]},
            {"type": "synthetic", "name": "humanoid", "urdf": "humanoid.urdf", "randomized_embodiments": 1,
             "episodes_per_embodiment": 2, "steps": 12, "probe_torques": 1},
        ],
    }
    path = root / "data.json"
    path.write_text(json.dumps(config))
    return path


class StatisticsTest(unittest.TestCase):
    """IQM and bootstrap intervals (Agarwal et al., 2021)."""

    def test_interquartile_mean(self) -> None:
        """The IQM drops the bottom and top quarter of the pooled runs, as scipy's trim_mean."""
        self.assertAlmostEqual(interquartile_mean(np.arange(1.0, 9.0)), 4.5)
        scores = np.random.default_rng(0).normal(size=(7, 5))
        self.assertAlmostEqual(interquartile_mean(scores), float(stats.trim_mean(scores, 0.25, axis=None)))
        ragged = {"a": [1.0, 2.0, 100.0], "b": [3.0, 4.0, -50.0, 5.0, 6.0]}
        self.assertAlmostEqual(interquartile_mean(ragged), float(stats.trim_mean(np.array([1, 2, 100, 3, 4, -50, 5, 6.0]), 0.25)))
        with self.assertRaises(ValueError):
            interquartile_mean([])

    def test_stratified_interval_matches_normal_theory(self) -> None:
        """For the mean of normal runs the 95% interval is close to +-1.96 sigma / sqrt(N)."""
        scores = np.random.default_rng(1).normal(2.0, 1.0, size=(40, 5))
        interval = stratified_bootstrap_ci(scores, mean_score, seed=0)
        self.assertEqual(interval.repetitions, 50_000)
        self.assertEqual(interval.confidence, 0.95)
        self.assertLess(interval.lower, interval.estimate)
        self.assertLess(interval.estimate, interval.upper)
        half_width = (interval.upper - interval.lower) / 2
        # Stratification removes the between-task part of the variance (zero here).
        expected = 1.96 * scores.std(axis=0, ddof=0).mean() / np.sqrt(scores.size)
        self.assertAlmostEqual(half_width / expected, 1.0, delta=0.15)
        again = stratified_bootstrap_ci(scores, mean_score, seed=0)
        self.assertEqual((interval.lower, interval.upper), (again.lower, again.upper))

    def test_constant_runs_give_a_point_interval(self) -> None:
        """Every replicate is the same when each task's runs are equal."""
        scores = {"a": [1.0, 1.0, 1.0], "b": [3.0, 3.0]}
        interval = stratified_bootstrap_ci(scores, interquartile_mean, repetitions=500)
        self.assertAlmostEqual(interval.lower, interval.upper)
        self.assertAlmostEqual(interval.estimate, interval.lower)

    def test_robot_bootstrap_covers_between_robot_spread(self) -> None:
        """Resampling robots widens the interval when robots differ; a custom statistic works too."""
        generator = np.random.default_rng(2)
        scores = {f"robot{index}": list(center + 0.1 * generator.normal(size=3)) for index, center in enumerate(range(8))}
        by_run = stratified_bootstrap_ci(scores, interquartile_mean, repetitions=4000)
        by_robot = robot_bootstrap_ci(scores, interquartile_mean, repetitions=4000)
        self.assertEqual(by_robot.unit, "robot")
        self.assertGreater(by_robot.upper - by_robot.lower, 5 * (by_run.upper - by_run.lower))
        median = robot_bootstrap_ci(scores, lambda values: float(np.median(np.concatenate(list(values.values())))), repetitions=300)
        self.assertLessEqual(median.lower, median.upper)

    def test_performance_gap(self) -> None:
        """Lk - L0 with an interval that excludes zero for a clear gap."""
        generator = np.random.default_rng(3)
        levels = {
            "L0": {f"r{index}": list(1 + 0.1 * generator.normal(size=2)) for index in range(6)},
            "L1": {f"s{index}": list(3 + 0.1 * generator.normal(size=2)) for index in range(6)},
        }
        gaps = performance_gap(levels, repetitions=3000)
        self.assertEqual(set(gaps), {"L1"})
        self.assertAlmostEqual(gaps["L1"].estimate, 2.0, delta=0.2)
        self.assertGreater(gaps["L1"].lower, 0)
        with self.assertRaises(KeyError):
            performance_gap({"L1": levels["L1"]})


class SplitManifestTest(unittest.TestCase):
    """The L0-L4 manifest, its validation, and held-out splits."""

    def test_shipped_manifest(self) -> None:
        """configs/splits/zero_shot.json is valid and records the H1 conflict."""
        manifest = SplitManifest.load(MANIFEST)
        training = {entry.name for entry in manifest.training_robots}
        self.assertEqual(training, {"go1", "anymal_d", "g1_29dof", "quadrotor_x", "g1_arms"})
        self.assertEqual(manifest.level_of("h1"), "L4")
        self.assertIn("CONFLICT", next(entry for entry in manifest.levels["L4"] if entry.name == "h1").notes)
        self.assertTrue(any("h1" in note for note in manifest.notes))
        categories = {entry.generator["category"] for entry in manifest.levels["L1"]}
        self.assertEqual(categories, set(L1_CATEGORIES))
        self.assertEqual(set(METAMORPH_VARIATIONS), {"armature", "density", "damping", "gear", "module_shape", "joint_angle"})
        self.assertEqual({entry.family for entry in manifest.levels["L3"]}, {"hexapod"})
        for level in ("L1", "L2", "L3"):
            self.assertEqual({entry.split for entry in manifest.levels[level]}, {"val", "test"}, level)
        go2 = next(entry for entry in manifest.levels["L4"] if entry.name == "go2")
        self.assertIn("unitree_ros", go2.download["url"])
        # The current Unitree configs train on H1, so they are not zero-shot for L4.
        violations = manifest.check_training_config(ROOT / "configs" / "data" / "synthetic_unitree.json")
        self.assertEqual(len(violations), 1)
        self.assertIn("h1", violations[0])
        self.assertEqual(manifest.check_training_config(ROOT / "configs" / "data" / "quadrotor_flight.json"), [])
        # Procedural training sources must not reach L2 topologies or the L3 family.
        procedural = {"type": "procedural", "name": "gen", "family": "quadruped"}
        restricted = {**procedural, "ranges": {"knee_joints": [1], "foot_scale": [1.0]}}
        self.assertIn("can generate", manifest.check_training_config({"sources": [procedural]})[0])
        self.assertEqual(manifest.check_training_config({"sources": [restricted]}), [])
        hexapods = {"type": "procedural", "name": "gen", "families": ["quadruped", "hexapod"]}
        self.assertIn("L3 family", manifest.check_training_config({"sources": [hexapods]})[0])

    def test_heldout_split(self) -> None:
        """20% of the robots go to test, deterministically and independent of order."""
        names = [f"robot{index}" for index in range(10)]
        split = make_heldout_split(names, seed=4)
        self.assertEqual(sum(value == "test" for value in split.values()), 2)
        self.assertEqual(split, make_heldout_split(list(reversed(names)), seed=4))
        self.assertEqual(sorted(make_heldout_split(["a", "b", "c"]).values()), ["test", "val", "val"])
        self.assertEqual(make_heldout_split(["only"]), {"only": "test"})
        self.assertTrue(name_matches("go1", "go1/0-fixed"))
        self.assertTrue(name_matches("go1/density", "go1/density/1-floating"))
        self.assertFalse(name_matches("go1/density", "go1/0-fixed"))
        self.assertFalse(name_matches("go1", "go10"))
        with self.assertRaises(ValueError):
            make_heldout_split(names, fraction=1.0)
        with self.assertRaises(ValueError):
            make_heldout_split(["a", "a"])

    def test_validation_rules(self) -> None:
        """Robots in two levels, training robots in L1-L4, and bad entries are rejected; JSON round-trips."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training = [RobotEntry("go1", "quadruped", urdf="go1.urdf")]
            good = SplitManifest(
                training_robots=training,
                levels={
                    "L0": [RobotEntry("go1", "quadruped", urdf="go1.urdf")],
                    "L1": [RobotEntry("go1/density", "quadruped", "val", generator={"type": "parameter_variation", "base_robot": "go1", "category": "density"})],
                    "L4": [RobotEntry("go2", "quadruped", urdf="go2.urdf", extra={"future_key": 1})],
                },
                base_dir=root,
            )
            self.assertEqual(good.problems(), [])
            good.save(root / "manifest.json")
            loaded = SplitManifest.load(root / "manifest.json")
            self.assertEqual(loaded.to_dict(), good.to_dict())
            self.assertEqual(loaded.levels["L4"][0].extra, {"future_key": 1})
            cases = {
                "is in L1 and L4": {"L4": [RobotEntry("go1/density", "quadruped", urdf="x.urdf")]},
                "is a training robot": {"L2": [RobotEntry("go1", "quadruped", urdf="other.urdf")]},
                "its URDF is the training robot": {"L3": [RobotEntry("copy", "quadruped", urdf="./go1.urdf")]},
                "L0 robots must be training robots": {"L0": [RobotEntry("h1", "humanoid", urdf="h1.urdf")]},
                "exactly one of urdf and generator": {"L3": [RobotEntry("nothing", "hexapod")]},
                "split must be one of": {"L3": [RobotEntry("bad", "hexapod", "train", urdf="b.urdf")]},
                "category must be one of": {"L1": [RobotEntry("go1/x", "quadruped", generator={"type": "parameter_variation", "base_robot": "go1", "category": "color"})]},
                "unknown level": {"L9": [RobotEntry("z", "quadruped", urdf="z.urdf")]},
            }
            for message, levels in cases.items():
                manifest = SplitManifest(training_robots=training, levels={**good.levels, **levels}, base_dir=root)
                with self.assertRaisesRegex(ValueError, message):
                    manifest.validate()


class SyntheticProbeTest(unittest.TestCase):
    """Probes on synthetic embeddings with a known linear target."""

    @classmethod
    def setUpClass(cls) -> None:
        """Make features, a linear target, types, examples and splits."""
        generator = torch.Generator().manual_seed(0)
        cls.types = torch.arange(600) % 40
        cls.examples = torch.arange(600) // 6
        cls.features = torch.randn(600, 12, generator=generator, dtype=torch.float64)
        cls.weights = torch.randn(12, 2, generator=generator, dtype=torch.float64)
        cls.targets = cls.features @ cls.weights + 0.05 * torch.randn(600, 2, generator=generator, dtype=torch.float64)
        cls.split = split_groups(cls.examples, seed=0)

    def test_split_keeps_groups_together(self) -> None:
        """Every group lands in one split; about 60/20/20 by groups."""
        for group in torch.unique(self.examples):
            self.assertEqual(len(torch.unique(self.split[self.examples == group])), 1)
        counts = torch.bincount(self.split[:: 6], minlength=3)
        self.assertEqual(counts.tolist(), [60, 20, 20])

    def test_ridge_recovers_linear_map(self) -> None:
        """Closed-form ridge recovers the weights and selects a penalty from the grid."""
        train, val = self.split == 0, self.split == 1
        probe = fit_ridge_probe(self.features[train], self.targets[train], self.features[val], self.targets[val])
        torch.testing.assert_close(probe.weight, self.weights, atol=0.03, rtol=0)
        self.assertIn(probe.penalty, (0.1, 1.0, 10.0))
        self.assertGreater(r2_score(self.targets, probe.predict(self.features)), 0.99)
        self.assertAlmostEqual(r2_score(self.targets, self.targets), 1.0)

    def test_control_targets_are_per_type(self) -> None:
        """Control targets are constant within a type and drawn from training targets."""
        train = self.split == 0
        control = control_targets(self.types, self.targets, train, seed=1)
        for kind in torch.unique(self.types):
            rows = control[self.types == kind]
            self.assertTrue(torch.equal(rows, rows[:1].expand_as(rows)))
        pool = {tuple(row.tolist()) for row in self.targets[train]}
        self.assertTrue(all(tuple(row.tolist()) in pool for row in control))

    def test_selectivity_is_lower_on_control(self) -> None:
        """Linear and MLP probes fit the task and not its control task."""
        linear = run_probe(self.features, self.targets, self.types, self.split, self.examples, "linear")
        self.assertGreater(linear["task_r2"], 0.95)
        self.assertLess(linear["control_r2"], linear["task_r2"] - 0.5)
        self.assertGreater(linear["selectivity"], 0.5)
        mlp = run_probe(
            self.features, self.targets, self.types, self.split, self.examples, "mlp",
            mlp=MlpSettings(hidden_size=64, max_epochs=40, learning_rate=1e-2),
        )
        self.assertGreater(mlp["task_r2"], 0.8)
        self.assertGreater(mlp["selectivity"], 0.4)
        with self.assertRaises(ValueError):
            run_probe(self.features, self.targets, self.types, self.split, kind="tree")


class PhysicsMetricTest(unittest.TestCase):
    """Dimensionless metrics on known values."""

    def test_relative_errors(self) -> None:
        """Frobenius and vector relative errors, with masks and zero targets."""
        target = torch.eye(3).expand(2, 3, 3).clone()
        predicted = 1.1 * target
        torch.testing.assert_close(relative_frobenius_error(predicted, target), torch.full((2,), 0.1))
        mask = torch.tensor([[True, True, False], [True, False, False]])
        predicted[:, 2, 2] = 100.0
        torch.testing.assert_close(relative_frobenius_error(predicted, target, mask, mask), torch.full((2,), 0.1))
        self.assertTrue(torch.isnan(relative_frobenius_error(target, torch.zeros_like(target))).all())
        error = relative_error(torch.tensor([[1.0, 2.0], [0.0, 0.0]]), torch.tensor([[1.0, 1.0], [3.0, 4.0]]))
        torch.testing.assert_close(error, torch.tensor([1 / 2**0.5, 1.0]))
        fk = fk_error_over_length(torch.zeros(1, 2, 3), torch.tensor([[[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]]]), torch.tensor([[True, False]]))
        torch.testing.assert_close(fk, torch.tensor([5.0], dtype=torch.float64))

    def test_contact_f1(self) -> None:
        """Precision, recall and F1 from logits."""
        logits = torch.tensor([3.0, 3.0, -3.0, -3.0])
        labels = torch.tensor([1, 0, 1, 1])
        result = contact_f1(logits, labels)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["f1"], 0.4)
        self.assertEqual(contact_f1(logits, labels, torch.tensor([True, False, False, False]))["f1"], 1.0)

    def test_exact_label_dynamics_errors(self) -> None:
        """Only rows with torque_label_weight == 1 count."""
        batch = {
            "torque_label_weight": torch.tensor([1.0, 0.5]),
            "fd_target": torch.ones(2, 2),
            "fd_mask": torch.ones(2, 2, dtype=torch.bool),
            "id_target": torch.ones(2, 2),
            "id_mask": torch.ones(2, 2, dtype=torch.bool),
        }
        predictions = {"fd_acceleration": torch.tensor([[2.0, 2.0], [9.0, 9.0]]), "id_torque": torch.ones(2, 2)}
        errors = exact_label_dynamics_errors(predictions, batch)
        self.assertAlmostEqual(errors["fd_mse"], 1.0)
        self.assertAlmostEqual(errors["fd_relative_mse"], 1.0)
        self.assertEqual(errors["fd_samples"], 1)
        self.assertAlmostEqual(errors["id_mse"], 0.0)

    def test_operator_of_a_linear_head(self) -> None:
        """A head y = A x + b gives back A, column by column, on valid joints."""
        generator = torch.Generator().manual_seed(0)
        operator = torch.randn(2, 4, 4, generator=generator)
        base_operator = torch.randn(2, 6, 4, generator=generator)
        offset = torch.randn(2, 4, generator=generator)
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

        class LinearHead(torch.nn.Module):
            def forward(self, **batch):
                torque = batch["response_torque"]
                self.keys = set(batch)
                predictions = {"response_acceleration": torch.einsum("bij,bj->bi", operator, torque) + offset}
                if bool(batch["response_base_mask"].any()):
                    predictions["response_base_acceleration"] = torch.einsum("bij,bj->bi", base_operator, torque)
                return SimpleNamespace(predictions=predictions)

        head = LinearHead()
        batch = {"input_values": torch.zeros(2, 4, 3), "joint_mask": mask, "fk_position": torch.zeros(2, 4, 3), "new_optional_key": 1}
        estimate = operator_from_response_head(head, batch, amplitude=0.5, include_base=True)
        expected = operator * mask[:, :, None] * mask[:, None, :]
        torch.testing.assert_close(estimate.joint, expected)
        torch.testing.assert_close(estimate.base, base_operator * mask[:, None, :])
        self.assertNotIn("fk_position", head.keys)
        self.assertIn("new_optional_key", head.keys)


class ModelDiagnosticsTest(unittest.TestCase):
    """Layer capture, probes, head tests and counterfactuals on a tiny model and dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a tiny dataset and a tiny model."""
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.config_path = write_tiny_data(cls.root)
        cls.datasets = build_physics_datasets(cls.config_path, cache_dir=cls.root / "cache")
        cls.collator = cls.datasets.collator()
        encoder = DaGetConfig.from_size(cls.datasets.token_size, "tiny", use_physics_cross_attention=True)
        torch.manual_seed(0)
        cls.model = DaGetForPhysicsPretraining(
            DaGetPhysicsPretrainingConfig.from_encoder_config(encoder, token_layout=cls.datasets.token_layout())
        ).eval()
        cls.samples = [cls.datasets.train[index] for index in range(0, len(cls.datasets.train), 2)]

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove temporary files."""
        cls.directory.cleanup()

    def test_layer_capture(self) -> None:
        """The hooks return the input projection, every layer and the encoder output."""
        batch = self.collator(self.samples[:6])
        layers = capture_layer_embeddings(self.model, batch)
        count = len(self.model.encoder.layers)
        self.assertEqual(list(layers), ["input_projection"] + [f"layer_{index}" for index in range(count)] + ["final"])
        with torch.no_grad():
            torch.testing.assert_close(layers["final"], self.model.encode(**batch))
        for value in layers.values():
            self.assertTrue(bool((value[~batch["joint_mask"]] == 0).all()))
        self.assertFalse(torch.allclose(layers["layer_0"], layers["final"]))

    def test_true_operator_from_batch_and_records_agree(self) -> None:
        """The normalized operator is the same from batch keys and from the samples' records."""
        samples = self.samples[:6]
        batch = self.collator(samples)
        from_records, columns = true_response_operators(
            {key: value for key, value in batch.items() if key not in ("response_operator", "response_scale")}, samples, self.datasets.records
        )
        from_batch, _ = true_response_operators(batch, samples, self.datasets.records)
        torch.testing.assert_close(from_batch, from_records, rtol=1e-4, atol=1e-6)
        self.assertTrue(bool(columns.any()))
        # The response target is the normalized operator times the normalized torque change.
        target = (from_records @ batch["response_torque"].double()[..., None]).squeeze(-1)
        rows = batch["response_mask"]
        torch.testing.assert_close(target[rows], batch["response_target"].double()[rows], rtol=1e-3, atol=1e-5)

    def test_random_encoder_probes(self) -> None:
        """Probe data from a fresh random encoder gives finite R² for every layer and target."""
        random_model = random_like(self.model, seed=5)
        self.assertFalse(torch.equal(random_model.encoder.input_projection.weight, self.model.encoder.input_projection.weight))
        groups = list(range(len(self.samples)))
        data = collect_probe_data(random_model, self.samples, self.collator, self.datasets.records, groups, batch_size=16)
        self.assertEqual(set(data.joint_targets), {"fk_position", "mass_diagonal", "gravity_torque", "response_diagonal"})
        report = layerwise_probes(data, kinds=("linear", "mlp"), mlp=MlpSettings(hidden_size=16, max_epochs=2), seed=0)
        self.assertIn("final", report)
        for kinds in report["final"].values():
            if "skipped" in kinds:
                continue
            for result in kinds.values():
                self.assertTrue(np.isfinite(result["task_r2"]))
                self.assertTrue(np.isfinite(result["selectivity"]))

    def test_head_tests_run(self) -> None:
        """The head-capacity and re-initialization tests train only the heads."""
        settings = HeadTrainingSettings(steps=2, warmup_steps=0, batch_size=8)
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        eval_batches = fixed_batches(self.samples[:8], self.collator, 8, seed=0)
        capacity = head_capacity_test(self.model, lambda: endless_batches(self.samples, self.collator, 8), eval_batches, settings)
        self.assertIn("loss", capacity["loss_ratio"])
        transfer = head_reinit_transfer(
            self.model, lambda: endless_batches(self.samples, self.collator, 8), {"heldout_robots": eval_batches, "empty": []}, settings
        )
        self.assertEqual(set(transfer["heldout_robots"]), {"original", "trained_encoder_new_heads", "random_encoder_new_heads"})
        self.assertIn("skipped", transfer["empty"])
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))

    def test_counterfactual_mass_change(self) -> None:
        """The analytic Delta G is nonzero and finite, and the state is reconstructed exactly."""
        dataset = self.datasets.train
        floating = next(index for index, (episode, _, _) in enumerate(dataset.index) if dataset.records[dataset.episodes[episode][0]].graph.floating_base)
        for item in (0, floating):
            result = counterfactual_mass_response(self.model, dataset, item, self.collator)
            self.assertGreater(result["delta_norm"], 0)
            for key in ("cosine", "relative_error", "magnitude_ratio", "operator_relative_error", "delta_norm"):
                self.assertTrue(np.isfinite(result[key]), key)
            for key, value in result["reconstruction"].items():
                self.assertLess(value, 1e-4, key)
        graph = dataset.records[0].graph
        heavier = perturb_body_mass(graph, 1, 2.0)
        self.assertAlmostEqual(heavier.total_mass - graph.total_mass, float(graph.body_mass[1]))
        torch.testing.assert_close(heavier.body_inertia[1], 2 * graph.body_inertia[1])
        q = torch.tensor([0.1, -0.2, 0.3] * 4, dtype=torch.float64)
        torch.testing.assert_close(denormalize_positions(graph, graph.normalize_joint_positions(q)), q)


class ScriptTest(unittest.TestCase):
    """The diagnose script end to end, and the zero-shot aggregation."""

    def test_diagnose_script_end_to_end(self) -> None:
        """A random tiny model on tiny data produces every section of the report."""
        script = load_script("diagnose_physics_location")
        self.addCleanup(sys.modules.pop, "diagnose_physics_location")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = write_tiny_data(root)
            output = root / "report.json"
            script.main([
                "--random-init", "--config", "tiny", "--data-config", str(config), "--cache-dir", str(root / "cache"),
                "--output", str(output), "--max-samples", "48", "--mlp-hidden-size", "8", "--mlp-max-epochs", "2",
                "--head-steps", "2", "--head-warmup-steps", "0", "--batch-size", "16", "--operator-samples", "6",
                "--counterfactuals", "2", "--split-manifest", str(MANIFEST),
            ])
            report = json.loads(output.read_text())
        for key in ("probes", "physics_metrics", "head_capacity", "head_reinit_transfer", "counterfactual", "settings"):
            self.assertIn(key, report)
        self.assertEqual(set(report["probes"]), {"model", "random_encoder"})
        self.assertIn("final", report["probes"]["model"])
        self.assertEqual(report["heldout_robots"]["records"], ["humanoid/0-fixed"])
        self.assertEqual(report["settings"]["ridge_penalties"]["value"], [0.1, 1.0, 10.0])
        self.assertEqual(report["counterfactual"]["model"]["summary"]["cosine"]["count"], 2)
        self.assertGreater(report["model"]["parameters"]["heads"], 0)

    def test_zero_shot_aggregation(self) -> None:
        """Per-robot metrics become IQMs with robot-level intervals and gaps to L0."""
        script = load_script("evaluate_zero_shot")
        self.addCleanup(sys.modules.pop, "evaluate_zero_shot")
        parsed = script.parse_metrics({"eval/quad/0-ground/tracking_mpjpe": 0.5, "eval/biped/goal_mpjpe": 1.0, "train/x": 3})
        self.assertEqual(parsed, {"quad/0-ground": {"tracking_mpjpe": 0.5}, "biped": {"goal_mpjpe": 1.0}})
        self.assertEqual(script.parse_level_configs(["L2=a.json"]), {"L2": Path("a.json")})
        with self.assertRaises(ValueError):
            script.parse_level_configs(["L7=a.json"])
        scores = {"tracking_mpjpe": {"L0": {"a": [0.1, 0.12], "b": [0.2, 0.18]}, "L4": {"go2": [0.5, 0.6], "talos": [0.9, 0.8]}}}
        report = script.aggregate(scores, repetitions=500)
        levels = report["tracking_mpjpe"]["levels"]
        self.assertAlmostEqual(levels["L0"]["iqm"], interquartile_mean(scores["tracking_mpjpe"]["L0"]))
        gap = report["tracking_mpjpe"]["gap_to_L0"]["L4"]
        self.assertGreater(gap["gap"], 0)
        manifest = SplitManifest.load(MANIFEST)
        entry = script.match_entry(manifest, "L4", "go2_ground", (ROOT / "assets" / "urdf" / "go2.urdf"))
        self.assertEqual(entry.name, "go2")
        self.assertIsNone(script.match_entry(manifest, "L4", "go1", ROOT / "assets" / "urdf" / "go1.urdf"))
        missing = script.missing_entries(manifest, "L3", set(), "all")
        self.assertTrue(all("generator" in item["reason"] for item in missing))

    @unittest.skipUnless(importlib.util.find_spec("mujoco") is not None, "needs mujoco (uv sync --extra sim)")
    def test_zero_shot_script_end_to_end(self) -> None:
        """An untrained tiny BFM is evaluated per level in MuJoCo; unmatched robots are reported missing."""
        from model.configuration_bfm import BfmConfig
        from model.rl.agent import DaGetFbCprAgent
        from model.rl.configuration_fb_cpr import FbCprConfig
        from model.rl.trainer import TrainingArguments, build_encoder, load_embodiments

        script = load_script("evaluate_zero_shot")
        self.addCleanup(sys.modules.pop, "evaluate_zero_shot")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "quad.urdf").write_text(quadruped())
            (root / "biped.urdf").write_text(humanoid())
            feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            stand = {f"{leg}_{part}_joint": value for leg in ("FL", "FR", "RL", "RR") for part, value in (("thigh", 0.3), ("calf", -0.6))}
            common = {"randomized_embodiments": 0, "episodes_per_embodiment": 1, "steps": 20, "probe_torques": 0, "push_probability": 0.0, "dt": 0.02}
            sources = {
                "L0": {"type": "mujoco", "name": "quad", "urdf": str(root / "quad.urdf"), "contact_links": feet, "nominal_pose": stand, "gain_scale": 3.0, **common},
                "L4": {"type": "mujoco", "name": "biped", "urdf": str(root / "biped.urdf"), "contact_links": ["left_ankle", "right_ankle"], **common},
            }
            for level, source in sources.items():
                (root / f"{level}.json").write_text(json.dumps({"history_size": 2, "sources": [source]}))
            embodiments, _ = load_embodiments(root / "L0.json", TrainingArguments(data_config="", output_dir=""))
            observers = [embodiment.observer for embodiment in embodiments]
            torch.manual_seed(0)
            pretrained = build_encoder(None, observers[0].token_size, "tiny")
            model_config = BfmConfig.from_sizes(
                embedding_size=pretrained.encoder.config.hidden_size, state_token_size=observers[0].state_token_size,
                actor_size="tiny", forward_size="tiny", backward_size="tiny", z_dim=8, seq_length=4,
            )
            agent = DaGetFbCprAgent(model_config, FbCprConfig(batch_size=16), pretrained, observers, "cpu")
            run = root / "run"
            run.mkdir()
            torch.save(agent.state_dict(), run / "checkpoint.pt")
            agent.pretrained.save_pretrained(run / "encoder")
            manifest = SplitManifest(
                training_robots=[RobotEntry("quad", "quadruped", urdf="quad.urdf")],
                levels={
                    "L0": [RobotEntry("quad", "quadruped", urdf="quad.urdf")],
                    "L4": [RobotEntry("biped", "humanoid", urdf="biped.urdf"), RobotEntry("go2", "quadruped", urdf="go2.urdf")],
                },
                base_dir=root,
            )
            manifest.save(root / "manifest.json")
            report = script.main([
                "--checkpoint", str(run), "--checkpoint", str(run), "--split-manifest", str(root / "manifest.json"),
                "--level-data-config", f"L0={root / 'L0.json'}", "--level-data-config", f"L4={root / 'L4.json'}",
                "--training-data-config", str(root / "L4.json"), "--split", "all", "--envs-per-robot", "1",
                "--max-motions", "1", "--horizon", "5", "--repetitions", "200", "--output", str(root / "report.json"),
            ])
        mpjpe = report["metrics"]["tracking_mpjpe"]
        self.assertEqual(set(mpjpe["levels"]), {"L0", "L4"})
        self.assertEqual(list(mpjpe["levels"]["L4"]["robots"]), ["biped/0-ground"])
        self.assertEqual(len(mpjpe["levels"]["L0"]["robots"]["quad/0-ground"]), 2)
        self.assertIn("L4", mpjpe["gap_to_L0"])
        self.assertEqual([item["robot"] for item in report["missing"]["L4"]], ["go2"])
        self.assertEqual(len(report["training_config_violations"]), 1)


if __name__ == "__main__":
    unittest.main()
