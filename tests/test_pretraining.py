"""Test the physics pretraining data pipeline, heads, losses, and Trainer."""

import copy
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainingArguments

from model import DaGetConfig
from model.pretraining import (
    LOSS_NAMES,
    DaGetForPhysicsPretraining,
    DaGetPhysicsPretrainingConfig,
    geodesic_distance,
    masked_joint_mean,
    rotation_from_6d,
)
from model.training import PhysicsPretrainingTrainer, PhysicsTrajectoryDataset, build_physics_datasets
from model.training.episodes import TORQUE_SOURCES
from tests.test_embodiment import humanoid, quadruped

ARM_JOINTS = ["left_shoulder", "left_elbow", "right_shoulder", "right_elbow"]


def write_robots(folder: Path) -> None:
    """Write the test quadruped and humanoid URDFs."""
    (folder / "quadruped.urdf").write_text(quadruped())
    (folder / "humanoid.urdf").write_text(humanoid())


def write_lerobot_dataset(folder: Path, episodes: int = 3, frames: int = 40) -> None:
    """Write a LeRobot-style parquet dataset of the humanoid's arm joints."""
    rows = {"episode_index": [], "frame_index": [], "observation.state": [], "action": []}
    generator = np.random.default_rng(0)
    for episode in range(episodes):
        phase = generator.uniform(0, np.pi)
        for frame in range(frames):
            time = frame / 30
            state = [0.4 * np.sin(2 * time + phase + k) for k in range(4)] + [0.0, 0.0]
            rows["episode_index"].append(episode)
            rows["frame_index"].append(frame)
            rows["observation.state"].append(state)
            rows["action"].append([value + 0.05 for value in state])
    target = folder / "data" / "chunk-000"
    target.mkdir(parents=True)
    Dataset.from_dict(rows).to_parquet(str(target / "file-000.parquet"))


class PretrainingPipelineTest(unittest.TestCase):
    """Build data from both source types and train the model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create URDFs, a local LeRobot dataset, and processed datasets."""
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        write_robots(root)
        write_lerobot_dataset(root / "arm_dataset")
        cls.config = {
            "history_size": 2,
            "eval_fraction": 0.25,
            "sources": [
                {
                    "type": "synthetic",
                    "name": "quadruped",
                    "urdf": "quadruped.urdf",
                    "randomized_embodiments": 2,
                    "episodes_per_embodiment": 2,
                    "steps": 12,
                    "probe_torques": 1,
                },
                {
                    "type": "huggingface",
                    "name": "humanoid_arms",
                    "path": "arm_dataset",
                    "urdf": "humanoid.urdf",
                    "joint_names": ARM_JOINTS,
                    "state_indices": [0, 1, 2, 3],
                    "action_key": "action",
                    "action_indices": [0, 1, 2, 3],
                    "action_type": "position",
                    "fps": 30,
                    "end_effectors": ["left_elbow", "right_elbow"],
                    "actuator_defaults": {"kp": 30.0, "kd": 1.0},
                    # Upper-body data: the torso is held, so the arms see a fixed base.
                    "floating_base": False,
                },
            ],
        }
        config_path = root / "data.json"
        config_path.write_text(json.dumps(cls.config))
        cls.config_path = config_path
        cls.datasets = build_physics_datasets(config_path, cache_dir=root / "cache")

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove temporary files."""
        cls.directory.cleanup()

    def model(self, **overrides) -> DaGetForPhysicsPretraining:
        """Return a small pretraining model for the test data."""
        encoder = DaGetConfig(
            input_size=self.datasets.token_size,
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=64,
            use_physics_cross_attention=True,
        )
        torch.manual_seed(0)
        config = DaGetPhysicsPretrainingConfig.from_encoder_config(
            encoder, token_layout=self.datasets.token_layout(), **overrides
        )
        return DaGetForPhysicsPretraining(config)

    def batch(self, count: int = 12) -> dict:
        """Collate samples from both embodiments."""
        dataset = self.datasets.train
        step = max(1, len(dataset) // count)
        return self.datasets.collator()([dataset[index] for index in range(0, len(dataset), step)][:count])

    def test_sources_records_and_cache(self) -> None:
        """Both sources load; recorded joints follow joint_names; the cache is reused."""
        names = [record.name for record in self.datasets.records]
        self.assertEqual(names, ["quadruped/0-fixed", "quadruped/1-fixed", "humanoid_arms"])
        arm = self.datasets.records[2]
        self.assertEqual(arm.graph.joint_names, tuple(ARM_JOINTS))
        self.assertEqual(arm.effect_mask.tolist(), [False, True, False, True])
        self.assertTrue(torch.all(arm.acceleration_scale > 0))
        _, episode = next(item for item in self.datasets.train.episodes if item[0] == 2)
        # No torque sensor: no torque anywhere, the executed variant plus two proposals.
        self.assertEqual(episode["torque_source"].unique().tolist(), [TORQUE_SOURCES.index("none")])
        self.assertEqual(float(episode["applied_torque"].abs().max()), 0.0)
        self.assertEqual(episode["applied_torque"].shape[1], 3)
        quadruped_episode = next(item for item in self.datasets.train.episodes if item[0] == 0)[1]
        self.assertEqual(quadruped_episode["applied_torque"].shape[1], 2)
        self.assertEqual(quadruped_episode["torque_source"].unique().tolist(), [TORQUE_SOURCES.index("exact")])
        root = Path(self.directory.name)
        entries = [path for path in (root / "cache").iterdir() if path.is_dir()]
        self.assertEqual(len(entries), 1)
        self.assertTrue((entries[0] / "complete").exists())
        marker_time = (entries[0] / "complete").stat().st_mtime_ns
        # Every shard is a Hugging Face dataset with one row per frame.
        from datasets import Dataset

        shards = sorted(entries[0].glob("frames-*.arrow"))
        frames = sum(len(Dataset.from_file(str(path))) for path in shards)
        store = self.datasets.train.episodes
        evaluation = self.datasets.eval.episodes if self.datasets.eval else []
        total = sum(store.steps(index) for index in range(len(store))) + sum(evaluation.steps(index) for index in range(len(evaluation)))
        self.assertEqual(frames, total)
        again = build_physics_datasets(self.config_path, cache_dir=root / "cache")
        self.assertEqual(len(again.train), len(self.datasets.train))
        self.assertEqual((entries[0] / "complete").stat().st_mtime_ns, marker_time)

    def test_preprocessing_reports_source_progress_and_cache_reuse(self) -> None:
        """Report episode completion during a build and identify a later cache hit.

        Returns:
            None: Asserts that progress reaches the completed source and that
            the next load reports reuse of the completed cache.
        """
        root = Path(self.directory.name)
        config_path = root / "progress_data.json"
        config_path.write_text(json.dumps({"sources": [self.config["sources"][1]]}))
        cache_dir = root / "progress_cache"
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            build_physics_datasets(config_path, cache_dir=cache_dir, workers=1, progress=True)
        self.assertIn("0/1 sources", output.getvalue())
        self.assertIn("loading", output.getvalue())
        self.assertIn("0/3 episodes", output.getvalue())
        self.assertIn("1/1 sources", output.getvalue())
        self.assertIn("3/3 episodes", output.getvalue())
        self.assertIn("Processed-data cache ready:", output.getvalue())

        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            build_physics_datasets(config_path, cache_dir=cache_dir, progress=True)
        self.assertIn("Using processed-data cache:", output.getvalue())

    def test_local_preprocess_entry_point_reuses_training_cache(self) -> None:
        """Build and reuse the cache named by a pretraining JSON file.

        Returns:
            None: Checks that the CLI writes one complete cache and reuses it
            when launched again with the same training configuration.
        """
        root = Path(self.directory.name)
        cache_dir = root / "local_preprocess_cache"
        training_config = root / "local_preprocess.json"
        training_config.write_text(
            json.dumps(
                {
                    "data_config": str(self.config_path),
                    "cache_dir": str(cache_dir),
                    "preprocess_workers": 1,
                }
            ),
            encoding="utf-8",
        )
        project_root = Path(__file__).resolve().parents[1]
        command = [
            sys.executable,
            str(project_root / "scripts/preprocess_physics_data.py"),
            str(training_config),
        ]
        first = subprocess.run(
            command, cwd=project_root, capture_output=True, text=True, check=True
        )
        self.assertIn("Preprocessing ready:", first.stdout)
        self.assertIn("Processed-data cache ready:", first.stderr)
        completed = list(cache_dir.glob("*/complete"))
        self.assertEqual(len(completed), 1)
        marker_time = completed[0].stat().st_mtime_ns

        second = subprocess.run(
            command, cwd=project_root, capture_output=True, text=True, check=True
        )
        self.assertIn("Using processed-data cache:", second.stderr)
        self.assertEqual(completed[0].stat().st_mtime_ns, marker_time)

    def test_preprocessing_resumes_completed_sources_from_configured_cache(self) -> None:
        """Reuse the first source after the next source fails to load.

        Returns:
            None: Asserts that the configured cache survives interruption,
            skips the completed source, and becomes a complete dataset.
        """
        from model.training import dataset as data_module

        root = Path(self.directory.name)
        config = copy.deepcopy(self.config)
        config["sources"][1]["path"] = "future_arm_dataset"
        cache_dir = root / "resume_cache"
        config["cache_dir"] = str(cache_dir)
        config_path = root / "resume_data.json"
        config_path.write_text(json.dumps(config))

        with self.assertRaises(FileNotFoundError):
            build_physics_datasets(config_path, workers=1)
        partials = list(cache_dir.glob("*.incomplete"))
        self.assertEqual(len(partials), 1)
        checkpoint = json.loads((partials[0] / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["completed_sources"], 1)
        metadata = json.loads((partials[0] / "checkpoint-0001-episodes.json").read_text())
        orphan_name = f"frames-{metadata['shards'] + 1:05d}.arrow"
        (partials[0] / orphan_name).write_bytes(b"interrupted shard")

        write_lerobot_dataset(root / "future_arm_dataset")
        with mock.patch.object(data_module, "load_source", wraps=data_module.load_source) as loader:
            resumed = build_physics_datasets(config_path, workers=1)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[0]["name"], "humanoid_arms")
        self.assertEqual(len(resumed.records), len(self.datasets.records))
        self.assertEqual(len(resumed.train), len(self.datasets.train))
        self.assertFalse(list(cache_dir.glob("*.incomplete")))
        self.assertEqual(len(list(cache_dir.glob("*/complete"))), 1)
        self.assertFalse(list(cache_dir.glob(f"*/{orphan_name}")))

    def test_adding_cache_directory_reuses_existing_processed_data(self) -> None:
        """Reuse a complete entry when JSON gains a cache directory setting.

        Returns:
            None: Asserts that cache location does not change the content key
            or trigger source loading.
        """
        from model.training import dataset as data_module

        root = Path(self.directory.name)
        config_path = root / "data_with_cache.json"
        config_path.write_text(
            json.dumps({**self.config, "cache_dir": str(root / "cache")})
        )
        with mock.patch.object(
            data_module, "load_source", side_effect=AssertionError("cache miss")
        ):
            reused = build_physics_datasets(config_path)
        self.assertEqual(len(reused.train), len(self.datasets.train))

    def test_parallel_preprocessing_gives_the_same_cache(self) -> None:
        """Worker processes write the same episodes, in the same order, as the main process."""
        from model.training import dataset as data_module

        root = Path(self.directory.name)
        with mock.patch.object(data_module, "PARALLEL_MIN_FRAMES", 0):
            parallel = build_physics_datasets(self.config_path, cache_dir=root / "parallel", workers=2)
        serial = build_physics_datasets(self.config_path, cache_dir=root / "serial", workers=1)
        self.assertEqual((len(parallel.train), len(parallel.eval)), (len(serial.train), len(serial.eval)))
        for index in range(len(serial.train)):
            left, right = serial.train[index], parallel.train[index]
            self.assertEqual(left.keys(), right.keys())
            for key, value in left.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(right[key], value, rtol=1e-6, atol=1e-6, msg=key)
                else:
                    self.assertEqual(right[key], value, key)
        for left, right in zip(serial.records, parallel.records):
            torch.testing.assert_close(right.acceleration_scale, left.acceleration_scale)

    def test_arrow_samples_match_samples_in_memory(self) -> None:
        """Reading frames from the Arrow cache gives the same samples as the episodes in memory."""
        store = self.datasets.train.episodes
        memory = [(store.embodiment(index), {key: store[index][1][key] for key in store[index][1]}) for index in range(len(store))]
        for group in (False, True):
            arrow = PhysicsTrajectoryDataset(self.datasets.records, store, self.datasets.history_size, group_variants=group)
            in_memory = PhysicsTrajectoryDataset(self.datasets.records, memory, self.datasets.history_size, group_variants=group)
            self.assertEqual(len(arrow), len(in_memory))
            self.assertEqual(list(arrow.index), list(in_memory.index))
            # A worker started with spawn gets a pickled copy that maps the files again.
            import pickle

            payload = pickle.dumps(arrow)
            self.assertLess(len(payload), 1_000_000)
            copy = pickle.loads(payload)
            torch.testing.assert_close(copy[len(copy) - 1]["mass_matrix"], arrow[len(arrow) - 1]["mass_matrix"])
            for item in sorted({0, 1, len(arrow) // 2, len(arrow) - 1}):
                first, second = arrow[item], in_memory[item]
                self.assertEqual(sorted(first), sorted(second))
                for key, value in first.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, second[key], rtol=0, atol=0, msg=key)
                    else:
                        self.assertEqual(value, second[key], key)

    def test_collated_batch_is_finite_and_masked(self) -> None:
        """Padding, masks, and normalized targets are consistent."""
        batch = self.batch()
        mask = batch["joint_mask"]
        self.assertEqual(batch["input_values"].shape[:2], mask.shape)
        self.assertEqual(batch["input_values"].shape[-1], self.datasets.token_size)
        for name, value in batch.items():
            self.assertTrue(torch.isfinite(value.float()).all(), name)
        for name in ("fd_mask", "id_mask", "motor_mask", "effect_mask", "response_mask", "inverse_response_mask"):
            self.assertFalse(bool((batch[name] & ~mask).any()), name)
        self.assertTrue(torch.all(batch["input_values"][~mask] == 0))
        self.assertTrue(bool(batch["response_mask"].any()))

    def test_forward_losses_and_training_reduce_loss(self) -> None:
        """All seven losses are computed and a few steps of Adam reduce the total."""
        model = self.model()
        batch = self.batch()
        output = model(**batch)
        self.assertEqual(set(output.loss_components), set(LOSS_NAMES))
        self.assertEqual(output.joint_embeddings.shape[:2], batch["joint_mask"].shape)
        self.assertEqual(output.pooled_embedding.shape, (batch["joint_mask"].shape[0], 32))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = float(output.loss.detach())
        for _ in range(15):
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        self.assertLess(float(model(**batch).loss.detach()), 0.7 * first)

    def test_data_without_torque_gives_no_torque_labels_or_inputs(self) -> None:
        """Samples of a dataset without torque sensors train R, R^-1, FK, D and E, never FD or ID."""
        dataset = self.datasets.train
        arm = [index for index, (episode, _, _) in enumerate(dataset.index) if dataset.episodes[episode][0] == 2]
        samples = [dataset[index] for index in arm[:24]]
        batch = self.datasets.collator()(samples)
        self.assertFalse(bool(batch["fd_mask"].any()))
        self.assertFalse(bool(batch["id_mask"].any()))
        self.assertEqual(float(batch["torque_label_weight"].max()), 0.0)
        self.assertTrue(bool(batch["response_mask"].any()))
        self.assertTrue(bool(batch["inverse_response_mask"].any()))
        self.assertTrue(bool(batch["motor_mask"].any()))  # commands and proposals
        # The previous-action token features come from the recorded commands, never from a torque estimate.
        self.assertEqual(float(batch["fd_torque"].abs().max()), 0.0)

    def test_response_targets_are_the_exact_response(self) -> None:
        """The collator's response target is G(q) dtau, normalized, for the drawn torque change."""
        torch.manual_seed(3)
        dataset = self.datasets.train
        sample = dataset[0]
        record = self.datasets.records[sample["embodiment"]]
        batch = self.datasets.collator()([sample])
        peak = record.actuator["peak_torque"].float()
        change = batch["response_torque"][0] * peak
        expected = sample["response_operator"].float() @ change
        torch.testing.assert_close(batch["response_target"][0] * record.response_scale[6:], expected[6:], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(batch["inverse_response_target"][0], batch["response_torque"][0])
        fractions = batch["response_torque"][0].abs()
        self.assertTrue(bool(((fractions >= 0.02 - 1e-6) & (fractions <= 1.0 + 1e-6)).all()))

    def test_removed_config_keys_are_rejected(self) -> None:
        """Keys of the estimate-based pipeline fail loudly instead of being ignored."""
        for key in ("use_estimated_previous_torque", "estimated_torque_weight", "allow_estimated_torque_for_fd", "max_estimate_residual"):
            with self.assertRaisesRegex(ValueError, key):
                build_physics_datasets({**self.config, key: 0.5})
        with self.assertRaisesRegex(ValueError, "rbd"):
            DaGetPhysicsPretrainingConfig(loss_weights={"rbd": 0.1})

    def test_floating_base_without_known_support_has_no_response(self) -> None:
        """A floating robot with no feet, rotors or recorded support wrench gets no response labels."""
        config = json.loads(json.dumps(self.config))
        config["sources"] = [{**config["sources"][1], "floating_base": True}]
        root = Path(self.directory.name)
        path = root / "floating.json"
        path.write_text(json.dumps({**config, "eval_fraction": 0.0}))
        datasets = build_physics_datasets(path, cache_dir=root / "cache_floating")
        for _, episode in datasets.train.episodes:
            self.assertFalse(bool(episode["response_valid"].any()))

    def test_measured_torques_are_off_by_default(self) -> None:
        """Torque-sensor labels need measured_torque_weight; simulated ones always count."""
        from model.training.dataset import CollatorSettings, PhysicsDataCollator

        dataset = self.datasets.train
        index = next(i for i, (episode, _, variant) in enumerate(dataset.index) if dataset.episodes[episode][0] == 0 and variant == 0)
        sample = dict(dataset[index])
        measured = {**sample, "torque_source": TORQUE_SOURCES.index("measured")}
        records = self.datasets.records
        default = PhysicsDataCollator(records, 2)([sample, measured])
        self.assertEqual(default["torque_label_weight"].tolist(), [1.0, 0.0])
        self.assertFalse(bool(default["fd_mask"][1].any()))
        enabled = PhysicsDataCollator(records, 2, CollatorSettings(measured_torque_weight=0.5))([sample, measured])
        self.assertEqual(enabled["torque_label_weight"].tolist(), [1.0, 0.5])
        self.assertTrue(bool(enabled["fd_mask"][1].any()))

    def test_position_only_fk_and_disabled_tasks(self) -> None:
        """fk_orientation=False drops the rotation term; weight 0 removes a task."""
        batch = self.batch()
        full = self.model()(**batch).loss_components["fk"]
        position_only = self.model(fk_orientation=False)(**batch).loss_components["fk"]
        self.assertLess(float(position_only), float(full))
        partial = {key: value for key, value in batch.items() if not key.startswith("effect_")}
        components = self.model()(**partial).loss_components
        self.assertNotIn("effect", components)

    def test_ablations_accept_the_same_batch(self) -> None:
        """Without Physics Cross Attention or mass encoding, the model ignores those inputs."""
        batch = self.batch()
        for overrides in ({"use_physics_cross_attention": False}, {"use_mass_encoding": False}):
            encoder = DaGetConfig(
                input_size=self.datasets.token_size,
                hidden_size=32,
                num_attention_heads=4,
                num_hidden_layers=2,
                intermediate_size=64,
                **{"use_physics_cross_attention": True, **overrides},
            )
            model = DaGetForPhysicsPretraining(DaGetPhysicsPretrainingConfig.from_encoder_config(encoder))
            self.assertTrue(torch.isfinite(model(**batch).loss))

    def test_save_and_reload(self) -> None:
        """save_pretrained and from_pretrained round-trip the weights and config."""
        model = self.model().eval()
        batch = self.batch()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            loaded = DaGetForPhysicsPretraining.from_pretrained(folder).eval()
        with torch.no_grad():
            torch.testing.assert_close(model(**batch).joint_embeddings, loaded(**batch).joint_embeddings)
        self.assertEqual(loaded.config.encoder_config, model.config.encoder_config)

    def test_script_records_curves_in_tensorboard(self) -> None:
        """The training script writes train and eval task losses to TensorBoard by default."""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        path = Path(__file__).resolve().parents[1] / "scripts" / "train_physics_da_get.py"
        spec = importlib.util.spec_from_file_location("train_physics_da_get", path)
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as folder:
            defaults = script.PretrainingArguments(output_dir=folder)
            self.assertEqual(list(defaults.report_to), ["tensorboard"])
            self.assertEqual(defaults.logging_steps, 50)
            self.assertEqual(str(defaults.eval_strategy.value), "steps")
            arguments = [
                str(path), "--data_config", str(self.config_path), "--cache_dir", str(Path(folder, "cache")),
                "--output_dir", folder, "--preset", "small", "--hidden_size", "32", "--num_hidden_layers", "1",
                "--intermediate_size", "64", "--max_steps", "4", "--logging_steps", "1", "--eval_steps", "2",
                "--per_device_train_batch_size", "8", "--per_device_eval_batch_size", "8",
                "--save_strategy", "no", "--use_cpu", "true",
            ]
            with mock.patch.object(sys, "argv", arguments):
                script.main()
            folders = {event.parent for event in Path(folder, "runs").rglob("events.out.tfevents.*")}
            self.assertEqual(len(folders), 1)
            events = EventAccumulator(str(folders.pop()))
            events.Reload()
            tags = set(events.Tags()["scalars"])
            for tag in ("train/loss", "train/loss_fk", "train/loss_fd", "train/learning_rate", "eval/loss", "eval/loss_fk"):
                self.assertIn(tag, tags)
            steps = [event.step for event in events.Scalars("train/loss_fk")]
            self.assertEqual(steps, [1, 2, 3, 4])
            self.assertEqual({event.step for event in events.Scalars("eval/loss_fk")}, {2, 4})

    def test_trainer_runs_and_logs_task_losses(self) -> None:
        """The HF Trainer trains, evaluates, and logs per-task losses."""
        model = self.model()
        with tempfile.TemporaryDirectory() as folder:
            arguments = TrainingArguments(
                output_dir=folder,
                max_steps=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                logging_steps=1,
                report_to=[],
                save_strategy="no",
                remove_unused_columns=False,
                prediction_loss_only=True,
                label_names=["fk_position"],
                use_cpu=True,
            )
            trainer = PhysicsPretrainingTrainer(
                model=model,
                args=arguments,
                train_dataset=self.datasets.train,
                eval_dataset=self.datasets.eval,
                data_collator=self.datasets.collator(),
            )
            trainer.train()
            metrics = trainer.evaluate()
        logged = [entry for entry in trainer.state.log_history if "loss" in entry]
        self.assertIn("loss_fk", logged[0])
        self.assertIn("eval_loss_motor", metrics)
        self.assertTrue(np.isfinite(metrics["eval_loss"]))


    def test_gradient_accumulation_steps_on_the_mean_of_the_micro_batches(self) -> None:
        """With ``gradient_accumulation_steps`` 2, one SGD step uses the mean gradient of both micro-batches.

        The Trainer divides each micro-batch loss by the number of steps only while the batch
        has no ``labels`` for it to count; this test fails if that changes.
        """
        model = self.model()
        start = copy.deepcopy(model.state_dict())
        collator = self.datasets.collator()
        seen = []

        def collate(samples):
            batch = collator(samples)
            seen.append({key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()})
            return batch

        with tempfile.TemporaryDirectory() as folder:
            arguments = TrainingArguments(
                output_dir=folder, max_steps=1, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                optim="sgd", learning_rate=0.1, lr_scheduler_type="constant", weight_decay=0.0, max_grad_norm=0.0,
                dataloader_num_workers=0, report_to=[], save_strategy="no", eval_strategy="no",
                remove_unused_columns=False, prediction_loss_only=True, label_names=["fk_position"], use_cpu=True,
            )
            PhysicsPretrainingTrainer(model=model, args=arguments, train_dataset=self.datasets.train, data_collator=collate).train()
        reference = self.model()
        reference.load_state_dict(start)
        reference.train()
        for batch in seen[:2]:
            (reference(**batch).loss / 2).backward()
        for (name, trained), parameter in zip(model.named_parameters(), reference.parameters()):
            expected = parameter.detach() if parameter.grad is None else parameter.detach() - 0.1 * parameter.grad
            torch.testing.assert_close(trained.detach(), expected, rtol=1e-5, atol=1e-6, msg=name)
        self.assertTrue(any(parameter.grad is not None for parameter in reference.parameters()))

class ExampleConfigTest(unittest.TestCase):
    """The shipped data configs use only current keys."""

    def test_example_configs_have_no_removed_keys(self) -> None:
        """Every config under configs/ passes check_data_config."""
        from model.training.dataset import check_data_config

        root = Path(__file__).resolve().parents[1] / "configs"
        paths = sorted(root.rglob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            check_data_config(json.loads(path.read_text()))


class LossHelpersTest(unittest.TestCase):
    """Check rotation and averaging helpers."""

    def test_rotation_helpers(self) -> None:
        """6D maps to a rotation and the geodesic distance recovers the angle."""
        values = torch.randn(10, 6)
        rotation = rotation_from_6d(values)
        torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), torch.eye(3).expand(10, 3, 3), atol=1e-5, rtol=0)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(10))
        angle = torch.tensor(0.7)
        about_z = torch.tensor([[angle.cos(), -angle.sin(), 0], [angle.sin(), angle.cos(), 0], [0, 0, 1.0]])
        torch.testing.assert_close(geodesic_distance(about_z, torch.eye(3)), angle, atol=1e-4, rtol=0)

    def test_masked_joint_mean_weights_embodiments_equally(self) -> None:
        """Each sample counts once regardless of its joint count."""
        values = torch.tensor([[1.0, 1.0, 1.0, 1.0], [3.0, 0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
        torch.testing.assert_close(masked_joint_mean(values, mask), torch.tensor(2.0))
        empty = masked_joint_mean(values, torch.zeros_like(mask))
        self.assertEqual(float(empty), 0.0)


if __name__ == "__main__":
    unittest.main()
