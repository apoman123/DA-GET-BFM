"""Train the cross-embodiment behavior foundation model with online off-policy FB-CPR.

Each source of the data config is one robot: its motions are the expert data, and a
MuJoCo environment built from its URDF collects the replay data. Defaults follow
BFM-Zero. Example::

    uv run python scripts/train_bfm.py \\
        --data_config configs/rl/go1_h1_ground.json \\
        --encoder_checkpoint runs/ground --output_dir runs/bfm

Watch the run with ``uv run tensorboard --logdir runs``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.configuration_bfm import BfmConfig  # noqa: E402
from model.rl.configuration_fb_cpr import FbCprConfig  # noqa: E402
from model.rl.trainer import FbCprTrainer, TrainingArguments  # noqa: E402

MODEL_FIELDS = (
    "z_dim", "hidden_size", "num_attention_heads", "intermediate_size", "num_latent_tokens", "actor_layers",
    "forward_layers", "num_parallel", "actor_std", "seq_length", "backward_hidden_size", "backward_hidden_layers",
    "discriminator_hidden_size", "discriminator_hidden_layers",
)


def to_bool(text: str) -> bool:
    """Parse a boolean flag value."""
    if text.lower() in ("1", "true", "yes"):
        return True
    if text.lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def add_fields(parser: argparse._ArgumentGroup, values: dict) -> None:
    """Add one optional flag per default value, using the value's type."""
    for name, default in values.items():
        kind = to_bool if isinstance(default, bool) else type(default) if default is not None else str
        parser.add_argument(f"--{name}", type=kind, default=default, help=f"default: {default}")


def parse_arguments(argv: list[str] | None = None) -> tuple[TrainingArguments, dict, FbCprConfig]:
    """Parse the command line into run, model and agent settings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--output_dir", required=True)
    add_fields(
        parser.add_argument_group("run"),
        {
            item.name: item.default
            for item in dataclasses.fields(TrainingArguments)
            if item.name not in ("data_config", "output_dir")
        },
    )
    defaults = {item.name: item.default for item in dataclasses.fields(BfmConfig)}
    add_fields(parser.add_argument_group("model"), {name: defaults[name] for name in MODEL_FIELDS})
    add_fields(
        parser.add_argument_group("agent"),
        {item.name: item.default for item in dataclasses.fields(FbCprConfig) if item.name != "aux_reward_scales"},
    )
    values = vars(parser.parse_args(argv))
    run = TrainingArguments(**{item.name: values[item.name] for item in dataclasses.fields(TrainingArguments)})
    model = {name: values[name] for name in MODEL_FIELDS}
    agent = FbCprConfig(**{item.name: values[item.name] for item in dataclasses.fields(FbCprConfig) if item.name != "aux_reward_scales"})
    return run, model, agent


def main(argv: list[str] | None = None) -> None:
    """Train and print where the results are."""
    run, model, agent = parse_arguments(argv)
    trainer = FbCprTrainer(run, model, agent)
    names = ", ".join(f"{e.record.name} ({e.record.graph.num_joints} DoF)" for e in trainer.embodiments)
    print(f"robots: {names}")
    print(f"networks: {sum(p.numel() for p in trainer.agent.model.parameters()) / 1e6:.2f}M parameters on {run.device}")
    print(f"TensorBoard: uv run tensorboard --logdir {run.output_dir}")
    trainer.train()
    print(f"checkpoint: {Path(run.output_dir) / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
