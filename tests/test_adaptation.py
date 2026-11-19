"""Test LoRA, the L2-SP and LwF regularizers, and the adaptation script (proposal F1)."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from model import DaGetConfig
from model.adaptation import (
    LoRALinear,
    LoRAMultiheadAttention,
    inject_lora,
    l2_sp_penalty,
    lora_parameters,
    lwf_loss,
    merge_lora,
)
from model.embodiment import JointFeatureConfig
from model.pretraining import DaGetForPhysicsPretraining, DaGetPhysicsPretrainingConfig
from model.training import build_physics_datasets

# The robot URDFs in assets/urdf are downloaded, not committed (scripts/download_urdfs.py).
NEEDS_URDFS = unittest.skipUnless(
    all((Path(__file__).resolve().parents[1] / "assets" / "urdf" / name).exists() for name in ("go1.urdf", "h1.urdf", "anymal_d.urdf")),
    "needs the robot URDFs: uv run python scripts/download_urdfs.py",
)


ROOT = Path(__file__).resolve().parents[1]
GO1 = ROOT / "assets" / "urdf" / "go1.urdf"
# Tiny synthetic Go1 data: the nominal robot, fixed base, two short episodes.
TINY_SOURCE = {"randomized_embodiments": 0, "episodes_per_embodiment": 2, "steps": 12, "probe_torques": 1}


def load_script():
    """Import scripts/adapt_da_get.py as a module."""
    path = ROOT / "scripts" / "adapt_da_get.py"
    spec = importlib.util.spec_from_file_location("adapt_da_get", path)
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    return script


@NEEDS_URDFS
class AdaptationTest(unittest.TestCase):
    """LoRA on a tiny physics model of Go1, the regularizers, and the script end to end."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build tiny Go1 data with the script's own config writer, and a tiny checkpoint."""
        cls.script = load_script()
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        layout = {"history_size": 2, "fixed_local": list(JointFeatureConfig.da_get_zero().feature_names())}
        cls.config = cls.script.robot_data_config(GO1, layout, overrides=TINY_SOURCE)
        cls.datasets = build_physics_datasets(cls.config, cache_dir=cls.root / "cache")
        cls.checkpoint = cls.root / "checkpoint"
        cls.make_model().save_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove temporary files."""
        sys.modules.pop("adapt_da_get", None)
        cls.directory.cleanup()

    @classmethod
    def make_model(cls) -> DaGetForPhysicsPretraining:
        """Return a tiny pretraining model whose tokens match the data."""
        encoder = DaGetConfig(
            input_size=cls.datasets.token_size,
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=64,
            use_physics_cross_attention=True,
        )
        torch.manual_seed(0)
        config = DaGetPhysicsPretrainingConfig.from_encoder_config(encoder, token_layout=cls.datasets.token_layout())
        return DaGetForPhysicsPretraining(config)

    def batch(self, count: int = 8) -> dict:
        """Collate a fixed batch, with fixed response draws."""
        torch.manual_seed(1)
        dataset = self.datasets.train
        return self.datasets.collator()([dataset[index] for index in range(min(count, len(dataset)))])

    def test_lora_starts_as_an_exact_no_op(self) -> None:
        """With B = 0 every output equals the pretrained model's, bit for bit.

        The comparison runs without autograd: with autograd on, merely freezing parameters
        changes which attention kernel PyTorch picks, which moves outputs by about 1e-6
        with or without LoRA.
        """
        batch = self.batch()
        model = self.make_model().train()
        with torch.no_grad():
            expected = model(**batch)
            wrapped = inject_lora(model, rank=2, alpha=2.0)
            output = model(**batch)
        self.assertTrue(any(isinstance(model.get_submodule(name), LoRAMultiheadAttention) for name in wrapped))
        self.assertTrue(any(isinstance(model.get_submodule(name), LoRALinear) for name in wrapped))
        self.assertTrue(all(name.startswith("encoder.layers.") for name in wrapped))
        self.assertTrue(torch.equal(output.loss, expected.loss))
        for key, value in expected.predictions.items():
            self.assertTrue(torch.equal(output.predictions[key], value), key)
        # With autograd and every parameter trainable, the outputs are identical too.
        model = self.make_model().train()
        expected = model(**batch)
        inject_lora(model, rank=2, alpha=2.0, freeze_base=False)
        self.assertTrue(torch.equal(model(**batch).loss, expected.loss))

    def test_only_lora_parameters_get_gradients(self) -> None:
        """The base model is frozen; every LoRA parameter receives a gradient."""
        model = self.make_model().train()
        inject_lora(model, rank=2, alpha=2.0)
        trainable = {id(parameter) for parameter in lora_parameters(model)}
        model(**self.batch()).loss.backward()
        for name, parameter in model.named_parameters():
            if id(parameter) in trainable:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        # B starts at zero, so only B gets a nonzero gradient at the first step.
        self.assertTrue(any(bool(p.grad.abs().sum() > 0) for n, p in model.named_parameters() if "lora_B" in n))

    def test_merge_equals_unmerged_output(self) -> None:
        """Folding B A into the weights gives the same outputs and the original parameter names."""
        model = self.make_model().eval()
        names = set(model.state_dict())
        inject_lora(model, rank=3, alpha=3.0)
        torch.manual_seed(2)
        with torch.no_grad():
            for parameter in lora_parameters(model):
                parameter.normal_(std=0.2)
        batch = self.batch()
        with torch.no_grad():
            unmerged = model(**batch)
            merge_lora(model)
            merged = model(**batch)
        self.assertEqual(set(model.state_dict()), names)
        self.assertGreater(float((unmerged.loss - self.make_model().eval()(**batch).loss.detach()).abs()), 1e-4)
        torch.testing.assert_close(merged.loss, unmerged.loss, rtol=1e-5, atol=1e-5)
        for key, value in unmerged.predictions.items():
            torch.testing.assert_close(merged.predictions[key], value, rtol=1e-4, atol=1e-5)

    def test_lora_linear_and_attention_formulas(self) -> None:
        """LoRALinear adds (alpha/r) B A x; the attention wrapper rejects out_proj targets."""
        torch.manual_seed(3)
        linear = nn.Linear(5, 4)
        layer = LoRALinear(copy.deepcopy(linear), rank=2, alpha=6.0)
        with torch.no_grad():
            layer.lora_B.normal_()
        inputs = torch.randn(7, 5)
        expected = linear(inputs) + 3.0 * inputs @ (layer.lora_B @ layer.lora_A).T
        torch.testing.assert_close(layer(inputs), expected)
        holder = nn.Module()
        holder.attention = nn.MultiheadAttention(8, 2, batch_first=True)
        with self.assertRaisesRegex(ValueError, "out_proj"):
            inject_lora(holder, 1, 1.0, target=r"attention\.out_proj")
        with self.assertRaisesRegex(ValueError, "no module matched"):
            inject_lora(holder, 1, 1.0, target="nothing")

    def test_l2_sp_is_zero_at_the_reference(self) -> None:
        """The penalty vanishes at the pretrained weights and matches alpha/2 ||dW||^2 away from them."""
        model = self.make_model()
        reference = {key: value.clone() for key, value in model.state_dict().items()}
        inject_lora(model, rank=2, alpha=2.0)
        self.assertEqual(l2_sp_penalty(model, reference, alpha=0.1).item(), 0.0)
        with torch.no_grad():
            for parameter in lora_parameters(model):
                parameter.normal_(std=0.1)
        deltas = 0.0
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    deltas += float(module.delta_weight().pow(2).sum())
                elif isinstance(module, LoRAMultiheadAttention):
                    deltas += sum(float(value.pow(2).sum()) for value in module.delta_weights().values())
        torch.testing.assert_close(l2_sp_penalty(model, reference, alpha=0.1).item(), 0.05 * deltas, rtol=1e-4, atol=0.0)
        # A trainable parameter unknown to the reference is a new one and gets beta/2 ||w||^2.
        model.extra = nn.Parameter(torch.full((3,), 2.0))
        with_new = l2_sp_penalty(model, reference, alpha=0.1, beta=0.01)
        torch.testing.assert_close(with_new.item(), 0.05 * deltas + 0.005 * 12.0, rtol=1e-4, atol=0.0)

    def test_lwf_is_zero_for_identical_models(self) -> None:
        """Equal predictions give zero; a changed model gives a positive loss in every term."""
        model = self.make_model().eval()
        teacher = copy.deepcopy(model)
        batch = self.batch()
        with torch.no_grad():
            student_output, teacher_output = model(**batch), teacher(**batch)
        self.assertEqual(float(lwf_loss(student_output, teacher_output, batch["joint_mask"])), 0.0)
        logits = {"motor_logits": teacher_output.predictions["motor_logits"]}
        shifted = {"motor_logits": logits["motor_logits"] + 1.0}
        self.assertGreater(float(lwf_loss(shifted, logits, batch["joint_mask"])), 0.0)
        inject_lora(model, rank=2, alpha=2.0)
        with torch.no_grad():
            for parameter in lora_parameters(model):
                parameter.normal_(std=0.2)
            self.assertGreater(float(lwf_loss(model(**batch), teacher_output, batch["joint_mask"])), 0.0)

    def test_script_adapts_end_to_end(self) -> None:
        """The script adapts Go1 with each regularizer, early-stops, and saves a loadable merged model."""
        for regularizer in ("l2sp", "lwf"):
            with self.subTest(regularizer=regularizer), tempfile.TemporaryDirectory() as folder:
                arguments = [
                    "--checkpoint", str(self.checkpoint), "--urdf", str(GO1), "--output-dir", folder,
                    "--cache-dir", str(self.root / "cache"), "--lora-rank", "2", "--learning-rate", "1e-3",
                    "--max-steps", "4", "--eval-every", "2", "--patience", "2", "--batch-size", "8",
                    "--regularizer", regularizer, "--l2sp-alpha", "0.01", "--device", "cpu",
                ]
                for key, value in TINY_SOURCE.items():
                    arguments += [f"--{key.replace('_', '-')}", str(value)]
                log = self.script.main(arguments)
                written = json.loads(Path(folder, "adaptation_log.json").read_text())
                self.assertEqual(written["eval_after"], log["eval_after"])
                self.assertTrue(torch.isfinite(torch.tensor(log["eval_after"]["loss"])))
                self.assertTrue(torch.isfinite(torch.tensor(log["eval_before"]["loss"])))
                self.assertLessEqual(log["eval_after"]["loss"], log["eval_before"]["loss"] + 1e-6)
                self.assertEqual(json.loads(Path(folder, "data_config.json").read_text()), self.config)
                loaded = DaGetForPhysicsPretraining.from_pretrained(folder)
                self.assertFalse(any("lora" in key or "base_layer" in key for key in loaded.state_dict()))
                self.assertTrue(torch.isfinite(loaded.eval()(**self.batch()).loss))


if __name__ == "__main__":
    unittest.main()
