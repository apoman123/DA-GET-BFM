"""Pretrain the GET encoder with the auxiliary physics tasks using the HF Trainer.

Examples::

    # Synthetic, domain-randomized data for three Unitree robots
    uv run python scripts/download_urdfs.py
    uv run python scripts/train_physics_da_get.py \\
        --data_config configs/data/synthetic_unitree.json \\
        --output_dir runs/synthetic --max_steps 2000 --per_device_train_batch_size 64

    # Recorded Unitree G1 teleoperation data from the Hugging Face Hub, mixed
    # with synthetic data
    uv run python scripts/train_physics_da_get.py \\
        --data_config configs/data/g1_hub_plus_synthetic.json --output_dir runs/g1

Every ``transformers.TrainingArguments`` option works (``--learning_rate``,
``--push_to_hub``, ...). A single JSON file with all arguments can be passed
instead of flags.

Runs are recorded in TensorBoard under ``<output_dir>/runs/``. Watch them with
``uv run tensorboard --logdir runs``, or turn this off with ``--report_to none``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import HfArgumentParser, IntervalStrategy, TrainingArguments, set_seed  # noqa: E402

from model.configuration_da_get import DaGetConfig  # noqa: E402
from model.pretraining import (  # noqa: E402
    DaGetForPhysicsPretraining,
    DaGetPhysicsPretrainingConfig,
)
from model.training import PhysicsPretrainingTrainer, build_physics_datasets  # noqa: E402
from model.training.sampling import sample_groups, temperature_weights  # noqa: E402


@dataclass
class ModelArguments:
    """Encoder and head settings."""

    preset: str = field(
        default="small",
        metadata={"help": "Encoder size: tiny, small, medium, large, or da_get_zero (paper architecture)."},
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
    response_weight: float = field(default=1.0, metadata={"help": "Task R: contact-constrained response to a torque change."})
    inverse_response_weight: float = field(default=1.0, metadata={"help": "Task R^-1: torque change from its response."})
    motor_weight: float = 1.0
    effect_weight: float = 1.0
    init_from: str | None = field(default=None, metadata={"help": "Pretrained checkpoint to continue from."})
    # Options of the zero-shot proposal (docs/zero_shot.md); defaults keep the original model.
    root_token: bool = field(default=False, metadata={"help": "C1: root-body token in Physics Cross Attention."})
    dimensionless_physics: bool = field(default=False, metadata={"help": "A3: physics features in units of m, L, sqrt(L/g)."})
    topology_bias: str = field(
        default="get", metadata={"help": "D4: 'get' (GET-Zero structural bias), 'none', or 'modulation' (ModuMorph-style)."}
    )
    modulation_rank: int | None = field(default=None, metadata={"help": "D4: hypernetwork rank; heads by default."})
    head_type: str = field(default="mixer", metadata={"help": "B1: 'mixer', 'linear', or 'linear_residual'."})
    interface_rank: int = field(default=64, metadata={"help": "B1: columns of the operator factors."})
    acceleration_metric: str = field(default="euclidean", metadata={"help": "A1: 'euclidean' or 'mass' (Gauss metric)."})
    effect_mode: str = field(default="head", metadata={"help": "B1: 'head' or 'rollout' (adds the rollout loss)."})
    contact_head: bool = field(default=False, metadata={"help": "C2: contact-probability head."})
    centroidal_head: bool = field(default=False, metadata={"help": "C3: centroidal head."})
    sysid_head: bool = field(default=False, metadata={"help": "E1: spec-deviation head (needs spec_noise in the data config)."})
    physics_mask_ratio: float = field(default=0.0, metadata={"help": "E3: masked physics modeling ratio (BERT 0.15, MAE 0.75)."})
    operator_weight: float = 1.0
    consistency_weight: float = 1.0
    effect_rollout_weight: float = 1.0
    contact_weight: float = 1.0
    centroidal_weight: float = 1.0
    mask_weight: float = 1.0
    sysid_weight: float = 1.0


@dataclass
class DataArguments:
    """Data settings."""

    data_config: str = field(metadata={"help": "JSON data config (see configs/data)."})
    cache_dir: str | None = field(default=None, metadata={"help": "Processed-data cache directory."})
    rebuild_cache: bool = field(default=False, metadata={"help": "Recompute features and labels."})


@dataclass
class PretrainingArguments(TrainingArguments):
    """``TrainingArguments`` whose defaults record train and eval curves in TensorBoard."""

    report_to: None | str | list[str] = field(
        default="tensorboard", metadata={"help": "Where to log; 'none' turns logging integrations off."}
    )
    logging_steps: float = field(default=50, metadata={"help": "Log the training losses every X steps."})
    eval_strategy: IntervalStrategy | str = field(
        default="steps", metadata={"help": "When to evaluate: 'no', 'steps', or 'epoch'."}
    )
    eval_steps: float | None = field(default=500, metadata={"help": "Evaluate every X steps."})


def encoder_config(arguments: ModelArguments, input_size: int) -> DaGetConfig:
    """Build the encoder configuration from the preset and overrides."""
    values = {}
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"):
        if getattr(arguments, name) is not None:
            values[name] = getattr(arguments, name)
    if arguments.dropout is not None:
        values["hidden_dropout_prob"] = arguments.dropout
    layers = None
    if arguments.physics_layers:
        layers = tuple(int(item) for item in arguments.physics_layers.split(","))
    if arguments.topology_bias not in ("get", "none", "modulation"):
        raise ValueError("topology_bias must be 'get', 'none' or 'modulation'")
    values.update(use_mass_encoding=arguments.mass_encoding,
                  use_physics_cross_attention=arguments.physics_cross_attention,
                  physics_cross_attention_layers=layers,
                  use_root_token=arguments.root_token,
                  dimensionless_physics=arguments.dimensionless_physics,
                  use_spatial_encoding=arguments.topology_bias != "none",
                  use_parent_child_encoding=arguments.topology_bias != "none",
                  use_morphology_modulation=arguments.topology_bias == "modulation",
                  modulation_rank=arguments.modulation_rank,
                  physics_masking=arguments.physics_mask_ratio > 0)
    if arguments.preset == "da_get_zero":
        return DaGetConfig.da_get_zero(input_size, **values)
    return DaGetConfig.from_size(input_size, arguments.preset, **values)


def main() -> None:
    """Parse arguments, build data and model, and train."""
    parser = HfArgumentParser((ModelArguments, DataArguments, PretrainingArguments))
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
    if datasets.eval is None:
        training_args.eval_strategy = IntervalStrategy.NO
    print(
        f"embodiments: {len(datasets.records)}, train samples: {len(datasets.train)}, "
        f"eval samples: {len(datasets.eval) if datasets.eval else 0}, token size: {datasets.token_size}"
    )
    if model_args.init_from:
        model = DaGetForPhysicsPretraining.from_pretrained(model_args.init_from)
        if model.config.encoder["input_size"] != datasets.token_size:
            raise ValueError("checkpoint token size does not match the data config")
    else:
        config = DaGetPhysicsPretrainingConfig.from_encoder_config(
            encoder_config(model_args, datasets.token_size),
            head_hidden_size=model_args.head_hidden_size,
            mixing_layers=model_args.mixing_layers,
            fk_orientation=model_args.fk_orientation,
            loss_weights={
                "fk": model_args.fk_weight,
                "fd": model_args.fd_weight,
                "id": model_args.id_weight,
                "response": model_args.response_weight,
                "inverse_response": model_args.inverse_response_weight,
                "motor": model_args.motor_weight,
                "effect": model_args.effect_weight,
                "operator": model_args.operator_weight,
                "consistency": model_args.consistency_weight,
                "effect_rollout": model_args.effect_rollout_weight,
                "contact": model_args.contact_weight,
                "centroidal": model_args.centroidal_weight,
                "mask": model_args.mask_weight,
                "sysid": model_args.sysid_weight,
            },
            token_layout=datasets.token_layout(),
            head_type=model_args.head_type,
            interface_rank=model_args.interface_rank,
            acceleration_metric=model_args.acceleration_metric,
            effect_mode=model_args.effect_mode,
            use_contact_head=model_args.contact_head,
            use_centroidal_head=model_args.centroidal_head,
            use_sysid_head=model_args.sysid_head,
            physics_mask_ratio=model_args.physics_mask_ratio,
        )
        model = DaGetForPhysicsPretraining(config)
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.2f}M")
    if model.config.effect_mode == "rollout":
        datasets.settings.include_graphs = True
    weights = None
    if datasets.sampling.get("sampling_alpha") is not None:
        groups = sample_groups(datasets.train, datasets.records, datasets.sampling["sampling_group"])
        weights = temperature_weights(groups, datasets.sampling["sampling_alpha"])
        print(f"temperature sampling: alpha {datasets.sampling['sampling_alpha']} over {len(set(groups))} groups")
    trainer = PhysicsPretrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train,
        eval_dataset=datasets.eval,
        data_collator=datasets.collator(),
        sample_weights=weights,
    )
    if "tensorboard" in training_args.report_to:
        print(f"TensorBoard: uv run tensorboard --logdir {training_args.output_dir}")
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
