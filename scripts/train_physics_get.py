"""Pretrain the GET encoder with the auxiliary physics tasks using the HF Trainer.

Examples::

    # Synthetic, domain-randomized data for three Unitree robots
    uv run python scripts/download_urdfs.py
    uv run python scripts/train_physics_get.py \\
        --data_config configs/data/synthetic_unitree.json \\
        --output_dir runs/synthetic --max_steps 2000 --per_device_train_batch_size 64

    # Recorded Unitree G1 teleoperation data from the Hugging Face Hub, mixed
    # with synthetic data
    uv run python scripts/train_physics_get.py \\
        --data_config configs/data/g1_hub_plus_synthetic.json --output_dir runs/g1

Every ``transformers.TrainingArguments`` option works (``--learning_rate``,
``--report_to``, ``--push_to_hub``, ...). A single JSON file with all arguments
can be passed instead of flags.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import HfArgumentParser, TrainingArguments, set_seed  # noqa: E402

from model.configuration_get import DaGetConfig  # noqa: E402
from model.pretraining import (  # noqa: E402
    GetForPhysicsPretraining,
    GetPhysicsPretrainingConfig,
)
from model.training import PhysicsPretrainingTrainer, build_physics_datasets  # noqa: E402


@dataclass
class ModelArguments:
    """Encoder and head settings."""

    preset: str = field(
        default="small",
        metadata={"help": "'small' (128 wide, 4 layers) or 'get_zero' (paper size, 512 wide, 8 layers)."},
    )
    hidden_size: int | None = field(default=None, metadata={"help": "Override the preset width."})
    num_hidden_layers: int | None = field(default=None, metadata={"help": "Override the preset depth."})
    num_attention_heads: int | None = field(default=None, metadata={"help": "Override the preset heads."})
    intermediate_size: int | None = field(default=None, metadata={"help": "Override the feed-forward width."})
    dropout: float | None = field(default=None, metadata={"help": "Override the dropout."})
    physics_cross_attention: bool = field(default=True, metadata={"help": "Enable Physics Cross Attention."})
    physics_layers: str | None = field(
        default=None, metadata={"help": "Comma-separated layer indices with Physics Cross Attention; all by default."}
    )
    mass_encoding: bool = field(default=True, metadata={"help": "Enable the mass-matrix attention bias."})
    mixing_layers: int = field(default=1, metadata={"help": "Graph action-mixing layers in the FD, ID, effect heads."})
    head_hidden_size: int | None = field(default=None, metadata={"help": "Head MLP width."})
    fk_orientation: bool = field(default=True, metadata={"help": "False: position-only FK (first pass)."})
    fk_weight: float = 1.0
    fd_weight: float = 1.0
    id_weight: float = 1.0
    motor_weight: float = 1.0
    effect_weight: float = 1.0
    rbd_weight: float = field(default=0.0, metadata={"help": "Optional task F; 0 disables it."})
    init_from: str | None = field(default=None, metadata={"help": "Pretrained checkpoint to continue from."})


@dataclass
class DataArguments:
    """Data settings."""

    data_config: str = field(metadata={"help": "JSON data config (see configs/data)."})
    cache_dir: str | None = field(default=None, metadata={"help": "Processed-data cache directory."})
    rebuild_cache: bool = field(default=False, metadata={"help": "Recompute features and labels."})


def encoder_config(arguments: ModelArguments, input_size: int) -> DaGetConfig:
    """Build the encoder configuration from the preset and overrides."""
    if arguments.preset == "get_zero":
        values = {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 16,
                  "intermediate_size": 1024, "hidden_dropout_prob": 0.1}
    elif arguments.preset == "small":
        values = {"hidden_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
                  "intermediate_size": 512, "hidden_dropout_prob": 0.0}
    else:
        raise ValueError("preset must be 'small' or 'get_zero'")
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"):
        if getattr(arguments, name) is not None:
            values[name] = getattr(arguments, name)
    if arguments.dropout is not None:
        values["hidden_dropout_prob"] = arguments.dropout
    layers = None
    if arguments.physics_layers:
        layers = tuple(int(item) for item in arguments.physics_layers.split(","))
    return DaGetConfig(
        input_size=input_size,
        use_mass_encoding=arguments.mass_encoding,
        use_physics_cross_attention=arguments.physics_cross_attention,
        physics_cross_attention_layers=layers,
        **values,
    )


def main() -> None:
    """Parse arguments, build data and model, and train."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # The batch is a dict of padded tensors from our collator; keep every key,
    # and report only losses during evaluation.
    training_args.remove_unused_columns = False
    training_args.prediction_loss_only = True
    training_args.label_names = ["fk_position"]
    set_seed(training_args.seed)

    datasets = build_physics_datasets(data_args.data_config, data_args.cache_dir, data_args.rebuild_cache)
    print(
        f"embodiments: {len(datasets.records)}, train samples: {len(datasets.train)}, "
        f"eval samples: {len(datasets.eval) if datasets.eval else 0}, token size: {datasets.token_size}"
    )
    if model_args.init_from:
        model = GetForPhysicsPretraining.from_pretrained(model_args.init_from)
        if model.config.encoder["input_size"] != datasets.token_size:
            raise ValueError("checkpoint token size does not match the data config")
    else:
        config = GetPhysicsPretrainingConfig.from_encoder_config(
            encoder_config(model_args, datasets.token_size),
            head_hidden_size=model_args.head_hidden_size,
            mixing_layers=model_args.mixing_layers,
            fk_orientation=model_args.fk_orientation,
            loss_weights={
                "fk": model_args.fk_weight,
                "fd": model_args.fd_weight,
                "id": model_args.id_weight,
                "motor": model_args.motor_weight,
                "effect": model_args.effect_weight,
                "rbd": model_args.rbd_weight,
            },
            token_layout=datasets.token_layout(),
        )
        model = GetForPhysicsPretraining(config)
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.2f}M")
    trainer = PhysicsPretrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train,
        eval_dataset=datasets.eval,
        data_collator=datasets.collator(),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    output = Path(training_args.output_dir)
    (output / "embodiments.json").write_text(
        json.dumps(
            [
                {"name": record.name, "joints": list(record.graph.joint_names)}
                for record in datasets.records
            ],
            indent=2,
        )
    )
    if datasets.eval is not None:
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
