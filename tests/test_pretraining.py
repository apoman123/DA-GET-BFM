"""Test the physics pretraining data pipeline, heads, losses, and Trainer."""

import importlib.util
import json
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
from model.training import PhysicsPretrainingTrainer, build_physics_datasets
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
            encoder, loss_weights={"rbd": 0.1}, token_layout=self.datasets.token_layout(), **overrides
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
        self.assertFalse(bool(episode["torque_measured"][0]))
        self.assertEqual(episode["applied_torque"].shape[1], 1)
        quadruped_episode = next(item for item in self.datasets.train.episodes if item[0] == 0)[1]
        self.assertEqual(quadruped_episode["applied_torque"].shape[1], 2)
        root = Path(self.directory.name)
        self.assertEqual(len(list((root / "cache").glob("*.pt"))), 1)
        again = build_physics_datasets(self.config_path, cache_dir=root / "cache")
        self.assertEqual(len(again.train), len(self.datasets.train))

    def test_collated_batch_is_finite_and_masked(self) -> None:
        """Padding, masks, and normalized targets are consistent."""
        batch = self.batch()
        mask = batch["joint_mask"]
        self.assertEqual(batch["input_values"].shape[:2], mask.shape)
        self.assertEqual(batch["input_values"].shape[-1], self.datasets.token_size)
        for name, value in batch.items():
            self.assertTrue(torch.isfinite(value.float()).all(), name)
        for name in ("fd_mask", "id_mask", "motor_mask", "effect_mask"):
            self.assertFalse(bool((batch[name] & ~mask).any()), name)
        self.assertTrue(torch.all(batch["input_values"][~mask] == 0))

    def test_forward_losses_and_training_reduce_loss(self) -> None:
        """All six losses are computed and a few steps of Adam reduce the total."""
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
