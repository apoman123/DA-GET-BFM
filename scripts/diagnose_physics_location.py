"""Measure whether the physics lives in the encoder or in the heads (proposal v4, B0).

Runs, layer by layer, on a pretrained ``DaGetForPhysicsPretraining`` checkpoint (or a
randomly initialized model) and the same model with a fresh random encoder:

1. linear (ridge) and MLP probes, one per layer (Alain and Bengio, ICLR 2017 workshop,
   arXiv:1610.01644), with control tasks and selectivity (Hewitt and Liang,
   EMNLP-IJCNLP 2019, DOI 10.18653/v1/D19-1275), on the trained and on a random
   encoder (the untrained-encoder baselines of Zhang and Bowman, BlackboxNLP 2018,
   DOI 10.18653/v1/W18-5448, and Conneau, Kruszewski, Lample, Barrault and Baroni,
   ACL 2018, DOI 10.18653/v1/P18-1198) (``model.evaluation.probes``, which gives the
   details and the [H] regression adaptation);
2. the dimensionless physics metrics of the heads (``model.evaluation.physics_metrics``);
3. the head-capacity test (heads trained on a frozen random encoder; proposal B0);
4. the head re-initialization transfer test (new heads on the frozen trained encoder,
   trained on training robots, evaluated on held-out robots);
5. counterfactual one-link mass changes against the analytic ``Delta G``
   (``model.evaluation.counterfactual``).

Examples::

    uv run python scripts/diagnose_physics_location.py --checkpoint runs/synthetic \\
        --data-config configs/data/synthetic_unitree.json \\
        --split-manifest configs/splits/zero_shot.json --output report.json

    # A random model of the checkpoint's architecture (or of a preset size)
    uv run python scripts/diagnose_physics_location.py --random-init --config runs/synthetic/config.json \\
        --data-config configs/data/synthetic_unitree.json --output random.json

Held-out robots of the transfer test are the data config's robots that the split
manifest puts in L1–L4 (matched by name, ``model.evaluation.splits.name_matches``).
Without a manifest, or when none matches, 20% of the robots are held out with
``make_heldout_split`` (Ai et al., CoRL 2025, arXiv:2505.05753: "randomly holding out
20% of the generated embodiments"). The test is only meaningful if the
checkpoint was not pretrained on those robots; the report says which robots were used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from model.configuration_da_get import DaGetConfig  # noqa: E402
from model.evaluation.counterfactual import (  # noqa: E402
    DEFAULT_MASS_SCALE,
    counterfactual_mass_response,
    summarize_counterfactuals,
)
from model.evaluation.physics_metrics import (  # noqa: E402
    contact_f1,
    exact_label_dynamics_errors,
    fk_error_over_length,
    move_batch,
    operator_from_response_head,
    relative_error,
    relative_frobenius_error,
    summarize,
    true_response_operators,
)
from model.evaluation.probes import (  # noqa: E402
    PROBE_TEST_FRACTION,
    PROBE_VAL_FRACTION,
    RIDGE_PENALTIES,
    HeadTrainingSettings,
    MlpSettings,
    collect_probe_data,
    endless_batches,
    fixed_batches,
    head_capacity_test,
    head_reinit_transfer,
    layerwise_probes,
    parameter_counts,
    probe_settings_report,
    random_like,
)
from model.evaluation.splits import HELDOUT_FRACTION, SplitManifest, make_heldout_split, name_matches, robot_key  # noqa: E402
from model.pretraining import DaGetForPhysicsPretraining, DaGetPhysicsPretrainingConfig  # noqa: E402
from model.training import build_physics_datasets  # noqa: E402

# [H] compute budgets, not statistical choices: probe samples, operator batches, counterfactual frames.
DEFAULT_MAX_SAMPLES = 2048
DEFAULT_OPERATOR_SAMPLES = 64
DEFAULT_COUNTERFACTUALS = 8


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="save_pretrained directory of DaGetForPhysicsPretraining")
    source.add_argument("--random-init", action="store_true", help="probe a randomly initialized model")
    parser.add_argument("--config", help="with --random-init: a model config.json, or a preset (tiny, small, medium, large)")
    parser.add_argument("--data-config", required=True, help="data config JSON (configs/data)")
    parser.add_argument("--cache-dir", default=None, help="processed-data cache")
    parser.add_argument("--split-manifest", default=None, help="split manifest JSON (configs/splits/zero_shot.json)")
    parser.add_argument("--output", required=True, help="report JSON")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help="[H] probe and head-evaluation samples (compute budget)")
    parser.add_argument("--batch-size", type=int, default=None, help="default: the checkpoint's training batch size, else 64")
    parser.add_argument("--probe-kinds", default="linear,mlp", help="comma-separated: linear, mlp")
    parser.add_argument("--mlp-hidden-size", type=int, default=MlpSettings.hidden_size)
    parser.add_argument("--mlp-max-epochs", type=int, default=MlpSettings.max_epochs)
    parser.add_argument("--head-steps", type=int, default=None, help="default: the checkpoint's max_steps, else 2000")
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--head-warmup-steps", type=int, default=None)
    parser.add_argument("--skip-head-tests", action="store_true")
    parser.add_argument("--operator-samples", type=int, default=DEFAULT_OPERATOR_SAMPLES, help="[H] samples for operator metrics")
    parser.add_argument("--counterfactuals", type=int, default=DEFAULT_COUNTERFACTUALS, help="[H] frames for the mass counterfactual")
    parser.add_argument("--mass-scale", type=float, default=DEFAULT_MASS_SCALE)
    parser.add_argument("--contact-prediction-key", default="contact_logits")
    return parser


def load_model(arguments: argparse.Namespace, token_size: int, token_layout: dict) -> DaGetForPhysicsPretraining:
    """Load the checkpoint or build a random model.

    Args:
        arguments (argparse.Namespace): Parsed arguments.
        token_size (int): Observation token width of the data.
        token_layout (dict): Token layout of the data.

    Returns:
        DaGetForPhysicsPretraining: Model in eval mode.

    Raises:
        ValueError: The token size does not match the data.
    """
    if arguments.checkpoint:
        model = DaGetForPhysicsPretraining.from_pretrained(arguments.checkpoint)
    else:
        torch.manual_seed(arguments.seed)
        name = arguments.config or "small"
        if Path(name).is_file():
            config = DaGetPhysicsPretrainingConfig.from_json_file(name)
        else:
            encoder = DaGetConfig.from_size(token_size, name, use_physics_cross_attention=True)
            config = DaGetPhysicsPretrainingConfig.from_encoder_config(encoder, token_layout=token_layout)
        model = DaGetForPhysicsPretraining(config)
    if model.config.encoder["input_size"] != token_size:
        raise ValueError(f"the model's token size {model.config.encoder['input_size']} differs from the data's {token_size}")
    return model.to(arguments.device).eval()


def sample_indices(count: int, limit: int, seed: int) -> list[int]:
    """Return up to ``limit`` sorted random indices of ``range(count)``.

    Args:
        count (int): Population size.
        limit (int): Maximum number.
        seed (int): Seed.

    Returns:
        list[int]: Indices.
    """
    indices = list(range(count))
    if count > limit:
        indices = sorted(random.Random(seed).sample(indices, limit))
    return indices


def embodiment_of(dataset, index: int) -> int:
    """Return the embodiment of a dataset sample without building the sample.

    Args:
        dataset (PhysicsTrajectoryDataset): Dataset.
        index (int): Sample index.

    Returns:
        int: Record index.
    """
    return int(dataset.episodes[dataset.index[index][0]][0])


def heldout_robots(records: list, manifest: SplitManifest | None, seed: int) -> tuple[set[int], str]:
    """Choose the held-out records of the transfer test.

    Records that the manifest lists in L1–L4 (``name_matches``) are held out. Without
    such records, 20% of the robots (record names cut at the first ``/``) are held out
    with ``make_heldout_split``.

    Args:
        records (list): Embodiment records of the data.
        manifest (SplitManifest | None): Split manifest.
        seed (int): Seed of the fallback split.

    Returns:
        tuple[set[int], str]: Held-out record indices and how they were chosen.
    """
    if manifest is not None:
        listed = [entry.name for level in manifest.levels if level != "L0" for entry in manifest.levels[level]]
        chosen = {index for index, record in enumerate(records) if any(name_matches(name, record.name) for name in listed)}
        if chosen and len(chosen) < len(records):
            return chosen, "split manifest (L1-L4 robots present in the data config)"
    robots = sorted({robot_key(record.name) for record in records})
    if len(robots) < 2:
        return set(), "only one robot in the data config"
    split = make_heldout_split(robots, HELDOUT_FRACTION, seed)
    chosen = {index for index, record in enumerate(records) if split[robot_key(record.name)] == "test"}
    return chosen, f"make_heldout_split(fraction={HELDOUT_FRACTION}, seed={seed}) over robots"


@torch.no_grad()
def physics_metrics(model: torch.nn.Module, samples: list, collator, records: list, contact_key: str, seed: int) -> dict[str, Any]:
    """Return the dimensionless physics metrics of a model on samples.

    Args:
        model (torch.nn.Module): Model.
        samples (list): Dataset items.
        collator: Collator.
        records (list): Records.
        contact_key (str): Prediction key of contact logits.
        seed (int): Seed of the response torque draws.

    Returns:
        dict[str, Any]: Metric summaries.
    """
    device = next(model.parameters()).device
    batches = fixed_batches(samples, collator, 16, seed)
    operator_errors, response_errors, fk_errors = [], [], []
    dynamics: dict[str, list[float]] = {}
    contact_logits, contact_labels, contact_masks = [], [], []
    start = 0
    for batch in batches:
        chunk = samples[start : start + batch["joint_mask"].shape[0]]
        start += len(chunk)
        moved = move_batch(batch, device)
        output = model(**moved)
        predictions = {key: value.cpu() for key, value in output.predictions.items()}
        if "response_acceleration" in predictions:
            response_errors.append(relative_error(predictions["response_acceleration"], batch["response_target"], batch["response_mask"]))
            truth, columns = true_response_operators(batch, chunk, records)
            estimate = operator_from_response_head(model, moved).joint.cpu()
            rows = batch["response_mask"]
            operator_errors.append(relative_frobenius_error(estimate.double(), truth, rows, columns & rows.any(1, keepdim=True)))
        if "fk_position" in predictions:
            fk_errors.append(fk_error_over_length(predictions["fk_position"], batch["fk_position"], batch["joint_mask"]))
        for key, value in exact_label_dynamics_errors(predictions, batch).items():
            dynamics.setdefault(key, []).append(value)
        if contact_key in predictions and "contact_target" in batch:
            contact_logits.append(predictions[contact_key].reshape(-1))
            contact_labels.append(batch["contact_target"].reshape(-1))
            mask = batch.get("contact_mask", batch["joint_mask"])
            contact_masks.append(mask.reshape(-1))
    report: dict[str, Any] = {
        "response_operator_relative_frobenius": summarize(torch.cat(operator_errors)) if operator_errors else "unavailable",
        "response_relative_error": summarize(torch.cat(response_errors)) if response_errors else "unavailable",
        "fk_error_over_length": summarize(torch.cat(fk_errors)) if fk_errors else "unavailable",
    }
    for key in ("fd_mse", "fd_relative_mse", "id_mse", "id_relative_mse"):
        values = [value for value in dynamics.get(key, []) if math.isfinite(value)]
        report[f"exact_{key}"] = sum(values) / len(values) if values else float("nan")
    report["exact_label_samples"] = {"fd": sum(dynamics.get("fd_samples", [])), "id": sum(dynamics.get("id_samples", []))}
    if contact_logits:
        report["contact"] = contact_f1(torch.cat(contact_logits), torch.cat(contact_labels), torch.cat(contact_masks))
    else:
        report["contact"] = f"unavailable: no {contact_key!r} prediction or no contact_target in the batch"
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run every diagnostic and write the report.

    Args:
        argv (list[str] | None): Command-line arguments; ``sys.argv`` if None.

    Returns:
        dict[str, Any]: The report.
    """
    arguments = build_parser().parse_args(argv)
    started = time.time()
    torch.manual_seed(arguments.seed)
    datasets = build_physics_datasets(arguments.data_config, arguments.cache_dir)
    records = datasets.records
    collator = datasets.collator()
    model = load_model(arguments, datasets.token_size, datasets.token_layout())
    manifest = SplitManifest.load(arguments.split_manifest) if arguments.split_manifest else None

    head_settings = HeadTrainingSettings.from_checkpoint(arguments.checkpoint) if arguments.checkpoint else HeadTrainingSettings()
    overrides = {
        "steps": arguments.head_steps,
        "learning_rate": arguments.head_learning_rate,
        "warmup_steps": arguments.head_warmup_steps,
        "batch_size": arguments.batch_size,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(head_settings, name, value)
    batch_size = head_settings.batch_size
    mlp = MlpSettings(hidden_size=arguments.mlp_hidden_size, max_epochs=arguments.mlp_max_epochs)
    kinds = [kind.strip() for kind in arguments.probe_kinds.split(",") if kind.strip()]

    # Samples of every split, grouped by episode so frames of one episode stay together.
    pool: list[tuple[str, int]] = [("train", index) for index in range(len(datasets.train))]
    if datasets.eval is not None:
        pool += [("eval", index) for index in range(len(datasets.eval))]
    chosen = [pool[index] for index in sample_indices(len(pool), arguments.max_samples, arguments.seed)]
    sets = {"train": datasets.train, "eval": datasets.eval}
    samples = [sets[name][index] for name, index in chosen]
    episode_ids: dict[tuple[str, int], int] = {}
    groups = [episode_ids.setdefault((name, sets[name].index[index][0]), len(episode_ids)) for name, index in chosen]

    random_model = random_like(model, arguments.seed + 1)
    report: dict[str, Any] = {
        "model": {
            "checkpoint": arguments.checkpoint,
            "random_init": bool(arguments.random_init),
            "parameters": parameter_counts(model),
        },
        "data": {
            "data_config": arguments.data_config,
            "records": [record.name for record in records],
            "probe_samples": len(samples),
            "probe_groups": len(episode_ids),
        },
        "settings": {
            **probe_settings_report(RIDGE_PENALTIES, mlp, PROBE_VAL_FRACTION, PROBE_TEST_FRACTION),
            "head_training": head_settings.to_dict(),
            "max_samples": {"value": arguments.max_samples, "source": "[H] compute budget"},
            "mass_scale": {"value": arguments.mass_scale, "source": "DEFAULT_RANDOMIZATION mass upper bound; MetaMorph density range"},
            "seed": arguments.seed,
        },
    }

    # 1. Probes on the trained (or loaded) model and on a fresh random encoder.
    report["probes"] = {}
    for name, candidate in (("model", model), ("random_encoder", random_model)):
        data = collect_probe_data(candidate, samples, collator, records, groups, batch_size)
        report["probes"][name] = layerwise_probes(data, kinds, mlp=mlp, seed=arguments.seed)

    # 2. Physics metrics of the heads.
    operator_samples = [samples[index] for index in sample_indices(len(samples), arguments.operator_samples, arguments.seed + 2)]
    report["physics_metrics"] = {
        name: physics_metrics(candidate, operator_samples, collator, records, arguments.contact_prediction_key, arguments.seed)
        for name, candidate in (("model", model), ("random_encoder", random_model))
    }

    # 3 and 4. Head tests.
    heldout_ids, how = heldout_robots(records, manifest, arguments.seed)
    report["heldout_robots"] = {"records": sorted(records[index].name for index in heldout_ids), "chosen_by": how}
    if not arguments.skip_head_tests:
        train_held = [embodiment_of(datasets.train, index) in heldout_ids for index in range(len(datasets.train))]
        train_robots = Subset(datasets.train, [index for index, held in enumerate(train_held) if not held] or list(range(len(datasets.train))))
        evaluation = datasets.eval if datasets.eval is not None else datasets.train
        eval_indices = sample_indices(len(evaluation), arguments.max_samples, arguments.seed + 3)
        eval_train = [evaluation[index] for index in eval_indices if embodiment_of(evaluation, index) not in heldout_ids]
        heldout_pool = [("eval", index) for index in range(len(evaluation)) if embodiment_of(evaluation, index) in heldout_ids]
        if evaluation is not datasets.train:
            heldout_pool += [("train", index) for index, held in enumerate(train_held) if held]
        heldout_choice = [heldout_pool[index] for index in sample_indices(len(heldout_pool), arguments.max_samples, arguments.seed + 4)]
        eval_heldout = [(evaluation if name == "eval" else datasets.train)[index] for name, index in heldout_choice]
        eval_all = [evaluation[index] for index in eval_indices]
        report["head_capacity"] = head_capacity_test(
            model,
            lambda: endless_batches(datasets.train, collator, batch_size, arguments.seed),
            fixed_batches(eval_all, collator, batch_size, arguments.seed),
            head_settings,
            arguments.seed,
        )
        report["head_reinit_transfer"] = head_reinit_transfer(
            model,
            lambda: endless_batches(train_robots, collator, batch_size, arguments.seed),
            {
                "train_robots": fixed_batches(eval_train, collator, batch_size, arguments.seed),
                "heldout_robots": fixed_batches(eval_heldout, collator, batch_size, arguments.seed),
            },
            head_settings,
            arguments.seed,
        )
        report["head_reinit_transfer"]["note"] = (
            "Held-out robots must be absent from the checkpoint's pretraining data for a zero-shot reading; "
            "their evaluation samples include training-split episodes, which the new heads never saw."
        )

    # 5. Counterfactual mass changes on frames with response labels.
    dataset = datasets.eval if datasets.eval is not None else datasets.train
    candidates = [
        index for index, (episode, step, *_) in enumerate(dataset.index) if bool(dataset.episodes[episode][1]["response_valid"][step])
    ]
    frames = sample_indices(len(candidates), arguments.counterfactuals, arguments.seed)
    report["counterfactual"] = {}
    for name, candidate in (("model", model), ("random_encoder", random_model)):
        results = [
            counterfactual_mass_response(candidate, dataset, candidates[index], collator, scale=arguments.mass_scale)
            for index in frames
        ]
        report["counterfactual"][name] = {"summary": summarize_counterfactuals(results), "frames": results}
    report["runtime_seconds"] = time.time() - started
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"report: {output}")
    return report


if __name__ == "__main__":
    main()
