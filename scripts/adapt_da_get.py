"""Adapt a pretrained DA-GET physics model to a new robot from its URDF (proposal F1).

The new robot's data come from the repository's simulators (``synthetic`` or
``mujoco`` source), which give exact labels for every physics task without a torque
sensor. The pretrained model is frozen, LoRA is injected into the encoder's attention
(``model.adaptation``), and the LoRA weights are trained on the physics losses plus an
optional regularizer toward the pretrained model:

* ``none`` (default): only the self-supervised physics losses, as PAD (Hansen et al.,
  ICLR 2021) and TTT (Sun et al., ICML 2020) adapt at test time;
* ``l2sp``: L2-SP (Li, Grandvalet and Davoine, ICML 2018); ``--l2sp-alpha`` is required
  and ``--l2sp-beta`` defaults to their 0.01;
* ``lwf``: Learning without Forgetting (Li and Hoiem, TPAMI 2018) against the frozen
  original model on the new robot's data, with their T = 2 and lambda_o = 1.

Training stops early on the new robot's held-out episodes (the evaluation split of
``build_physics_datasets``): the evaluation loss is computed every ``--eval-every``
steps, the best LoRA weights are kept, and training ends after ``--patience``
evaluations without improvement (the early-stopping meta-algorithm of Goodfellow,
Bengio and Courville, *Deep Learning*, MIT Press 2016, Sec. 7.8, Algorithm 7.1, with
``n = --eval-every`` and ``p = --patience``; no patience means training for
``--max-steps`` and keeping the best). Unlike Algorithm 7.1, whose first validation
error always counts as an improvement, the unadapted model is a candidate too, so the
result is never worse on the held-out episodes than the pretrained model. The merged model is saved
with ``save_pretrained``, so ``scripts/train_bfm.py --encoder_checkpoint`` and
``DaGetForPhysicsPretraining.from_pretrained`` read it like any checkpoint, together
with ``adaptation_log.json`` (losses before and after, the curve, the settings) and the
data config used.

Defaults: the batch size 64 is the one of the documented pretraining runs
(``docs/development.md``, ``scripts/train_physics_da_get.py`` example); AdamW with
weight decay 0 is the Hugging Face ``Trainer`` default that pretraining uses. The LoRA
rank, the learning rate, the step budget and the evaluation interval have no cited
values and must be given; ``--lora-alpha`` defaults to the rank (Hu et al. "set alpha
to the first r we try and do not tune it").

Example::

    uv run python scripts/adapt_da_get.py --checkpoint runs/synthetic \\
        --urdf assets/urdf/anymal_d.urdf --floating-base \\
        --contact-links LF_FOOT RF_FOOT LH_FOOT RH_FOOT \\
        --lora-rank 4 --learning-rate 1e-4 --max-steps 2000 --eval-every 100 \\
        --patience 5 --regularizer l2sp --l2sp-alpha 0.01 --output-dir runs/anymal_adapted
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from model.adaptation import (  # noqa: E402
    DEFAULT_LORA_TARGET,
    L2SP_BETA,
    LWF_TEMPERATURE,
    LWF_WEIGHT,
    inject_lora,
    l2_sp_penalty,
    lora_parameters,
    lwf_loss,
    merge_lora,
)
from model.embodiment import JointFeatureConfig  # noqa: E402
from model.pretraining import DaGetForPhysicsPretraining  # noqa: E402
from model.training import build_physics_datasets  # noqa: E402

# Batch size of the documented pretraining runs (docs/development.md: "600 steps, batch 64").
DEFAULT_BATCH_SIZE = 64
SOURCE_OPTIONS = ("randomized_embodiments", "episodes_per_embodiment", "steps", "probe_torques")


def _json_argument(value: str | None) -> Any:
    """Parse a JSON string or the content of a JSON file."""
    if value is None:
        return None
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.suffix == ".json" and path.exists() else value
    return json.loads(text)


def joint_features_name(token_layout: dict[str, Any]) -> str:
    """Return the data-config ``joint_features`` preset whose features a checkpoint was trained on.

    Args:
        token_layout (dict[str, Any]): ``config.token_layout`` of the checkpoint.

    Returns:
        str: ``"da_get_zero"`` or ``"cross_embodiment"``.

    Raises:
        ValueError: The checkpoint has no token layout or uses another feature set.
    """
    names = list(token_layout.get("fixed_local") or [])
    for preset, config in (("da_get_zero", JointFeatureConfig.da_get_zero()), ("cross_embodiment", JointFeatureConfig.cross_embodiment())):
        if names == list(config.feature_names()):
            return preset
    raise ValueError("the checkpoint's joint features match no preset; pass --data-config with matching joint_features")


def robot_data_config(
    urdf: str | Path,
    token_layout: dict[str, Any],
    source_type: str = "synthetic",
    name: str | None = None,
    floating_base: bool = False,
    contact_links: list[str] | None = None,
    nominal_pose: dict[str, float] | None = None,
    joint_names: list[str] | None = None,
    seed: int = 0,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the data config of one new robot, mirroring the example configs.

    ``synthetic`` follows ``configs/data/synthetic_unitree.json`` (``base_modes``
    ``["fixed", "floating"]`` for a floating robot, otherwise ``["fixed"]``) and
    ``mujoco`` follows ``configs/data/ground_contact.json``. Keys not given keep the
    sources' own defaults (``model/training/sources.py``). ``history_size`` and
    ``joint_features`` come from the checkpoint, so its tokens match.

    Args:
        urdf (str | Path): Robot URDF.
        token_layout (dict[str, Any]): ``config.token_layout`` of the checkpoint.
        source_type (str): ``synthetic`` or ``mujoco``.
        name (str | None): Source name; the URDF file stem by default.
        floating_base (bool): The robot's base moves freely.
        contact_links (list[str] | None): Feet.
        nominal_pose (dict[str, float] | None): Standing pose for ``mujoco``.
        joint_names (list[str] | None): Joints to keep; others are frozen.
        seed (int): Source and split seed.
        overrides (dict[str, Any] | None): Other source keys, such as ``steps``.

    Returns:
        dict[str, Any]: Data config.

    Raises:
        ValueError: The source type is unknown or the checkpoint has no history size.
    """
    if source_type not in ("synthetic", "mujoco"):
        raise ValueError("source must be 'synthetic' or 'mujoco'")
    if "history_size" not in token_layout:
        raise ValueError("the checkpoint has no token_layout.history_size; pass --data-config")
    path = Path(urdf).expanduser().resolve()
    source: dict[str, Any] = {"type": source_type, "name": name or path.stem, "urdf": str(path), "seed": seed}
    if source_type == "synthetic":
        source["base_modes"] = ["fixed", "floating"] if floating_base else ["fixed"]
    if contact_links:
        source["contact_links"] = list(contact_links)
    if nominal_pose:
        source["nominal_pose"] = dict(nominal_pose)
    if joint_names:
        source["joint_names"] = list(joint_names)
    source.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return {
        "history_size": int(token_layout["history_size"]),
        "joint_features": joint_features_name(token_layout),
        "seed": seed,
        "sources": [source],
    }


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move every tensor of a batch to a device."""
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, seed: int) -> dict[str, float]:
    """Return the sample-weighted mean loss and per-task losses on a dataset.

    The collator draws the response tasks' torque changes from the global generator, so
    it is reseeded (inside a forked RNG state) to give every evaluation the same draws.

    Args:
        model (torch.nn.Module): Physics model.
        loader (DataLoader): Batches in a fixed order.
        device (torch.device): Device.
        seed (int): Seed of the collator's draws.

    Returns:
        dict[str, float]: ``loss`` and ``loss_<task>``.
    """
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        for batch in loader:
            output = model(**_move(batch, device))
            size = int(batch["joint_mask"].shape[0])
            values = {"loss": output.loss, **{f"loss_{name}": value for name, value in output.loss_components.items()}}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            count += size
    model.train(was_training)
    return {key: value / max(count, 1) for key, value in totals.items()}


def _cycle(loader: DataLoader):
    """Yield batches forever, reshuffling at every pass."""
    while True:
        yield from loader


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (list[str] | None): Arguments; ``sys.argv`` by default.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Pretrained DaGetForPhysicsPretraining directory.")
    parser.add_argument("--output-dir", required=True, help="Where the adapted model and the log are written.")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data-config", help="A data config for the new robot (its own eval split is used).")
    data.add_argument("--urdf", help="URDF of the new robot; a data config is written for it.")
    parser.add_argument("--source", choices=("synthetic", "mujoco"), default="synthetic", help="Simulator of the written config.")
    parser.add_argument("--name", help="Source name (URDF stem by default).")
    parser.add_argument("--floating-base", action="store_true", help="The robot's base is free (legged robots).")
    parser.add_argument("--contact-links", nargs="+", help="Feet links.")
    parser.add_argument("--nominal-pose", help="JSON object or .json file: joint name to standing position (mujoco).")
    parser.add_argument("--joint-names", nargs="+", help="Joints to keep; other joints are frozen.")
    for option in SOURCE_OPTIONS:
        parser.add_argument(f"--{option.replace('_', '-')}", type=int, help=f"Source key {option}; the source default if omitted.")
    parser.add_argument("--cache-dir", help="Processed-data cache; .cache/physics_data by default.")
    parser.add_argument("--lora-rank", type=int, required=True, help="LoRA rank r (Hu et al. sweep 1, 2, 4, 8, 64).")
    parser.add_argument("--lora-alpha", type=float, help="LoRA alpha; the rank if omitted (Hu et al.).")
    parser.add_argument("--lora-target", default=DEFAULT_LORA_TARGET, help="Regular expression of adapted module names.")
    parser.add_argument("--regularizer", choices=("none", "l2sp", "lwf"), default="none")
    parser.add_argument("--l2sp-alpha", type=float, help="L2-SP alpha (required with l2sp; chosen per task by Li et al.).")
    parser.add_argument("--l2sp-beta", type=float, default=L2SP_BETA, help="L2-SP beta on new parameters (Li et al.: 0.01).")
    parser.add_argument("--lwf-weight", type=float, default=LWF_WEIGHT, help="LwF lambda_o (Li and Hoiem: 1).")
    parser.add_argument("--lwf-temperature", type=float, default=LWF_TEMPERATURE, help="LwF temperature (Li and Hoiem: 2).")
    parser.add_argument("--learning-rate", type=float, required=True, help="AdamW learning rate of the LoRA weights.")
    parser.add_argument("--max-steps", type=int, required=True, help="Largest number of optimizer steps.")
    parser.add_argument("--eval-every", type=int, required=True, help="Steps between evaluations on the held-out episodes.")
    parser.add_argument("--patience", type=int, help="Evaluations without improvement before stopping; none trains all steps.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size (repo runs: 64).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the data config, the batches and the evaluation draws.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args(argv)
    if arguments.regularizer == "l2sp" and arguments.l2sp_alpha is None:
        parser.error("--regularizer l2sp needs --l2sp-alpha (Li et al. choose it per task)")
    for name in ("lora_rank", "max_steps", "eval_every", "batch_size"):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.patience is not None and arguments.patience < 1:
        parser.error("--patience must be positive")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return arguments


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Adapt, evaluate, merge and save.

    Args:
        argv (list[str] | None): Command-line arguments; ``sys.argv`` by default.

    Returns:
        dict[str, Any]: The adaptation log that is also written to disk.

    Raises:
        ValueError: The data config has no evaluation episodes or its tokens do not
            match the checkpoint.
    """
    arguments = parse_arguments(argv)
    device = torch.device(arguments.device)
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)

    model = DaGetForPhysicsPretraining.from_pretrained(arguments.checkpoint)
    if arguments.data_config:
        data_config: Any = arguments.data_config
    else:
        overrides = {option: getattr(arguments, option) for option in SOURCE_OPTIONS}
        data_config = robot_data_config(
            arguments.urdf,
            dict(model.config.token_layout or {}),
            arguments.source,
            arguments.name,
            arguments.floating_base,
            arguments.contact_links,
            _json_argument(arguments.nominal_pose),
            arguments.joint_names,
            arguments.seed,
            overrides,
        )
        (output / "data_config.json").write_text(json.dumps(data_config, indent=2))
    datasets = build_physics_datasets(data_config, arguments.cache_dir)
    if datasets.eval is None:
        raise ValueError("early stopping needs evaluation episodes; use eval_fraction > 0 and at least two episodes")
    if model.config.encoder["input_size"] != datasets.token_size:
        raise ValueError("checkpoint token size does not match the data config")
    print(f"embodiments: {len(datasets.records)}, train samples: {len(datasets.train)}, eval samples: {len(datasets.eval)}")

    model.to(device)
    collator = datasets.collator()
    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(datasets.train, arguments.batch_size, shuffle=True, collate_fn=collator, generator=generator)
    eval_loader = DataLoader(datasets.eval, arguments.batch_size, shuffle=False, collate_fn=collator)
    before = evaluate(model, eval_loader, device, arguments.seed)
    print(f"before: eval loss {before['loss']:.4f}")

    teacher = None
    reference_state = None
    if arguments.regularizer == "lwf":
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
    elif arguments.regularizer == "l2sp":
        reference_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    alpha = arguments.lora_alpha if arguments.lora_alpha is not None else float(arguments.lora_rank)
    wrapped = inject_lora(model, arguments.lora_rank, alpha, arguments.lora_target)
    parameters = lora_parameters(model)
    print(f"LoRA on {len(wrapped)} modules, {sum(p.numel() for p in parameters)} trainable parameters")
    # The Hugging Face Trainer default used by pretraining: AdamW, betas (0.9, 0.999), eps 1e-8, no weight decay.
    optimizer = torch.optim.AdamW(parameters, lr=arguments.learning_rate, weight_decay=0.0)

    history: list[dict[str, Any]] = []
    best = {"step": 0, "loss": before["loss"], "state": [parameter.detach().clone() for parameter in parameters]}
    stale = 0
    model.train()
    batches = _cycle(train_loader)
    for step in range(1, arguments.max_steps + 1):
        batch = _move(next(batches), device)
        output_ = model(**batch)
        loss = output_.loss
        regularizer = loss.new_zeros(())
        if arguments.regularizer == "l2sp":
            regularizer = l2_sp_penalty(model, reference_state, arguments.l2sp_alpha, arguments.l2sp_beta)
        elif arguments.regularizer == "lwf":
            with torch.no_grad():
                recorded = teacher(**batch)
            regularizer = arguments.lwf_weight * lwf_loss(output_, recorded, batch["joint_mask"], arguments.lwf_temperature)
        optimizer.zero_grad(set_to_none=True)
        (loss + regularizer).backward()
        optimizer.step()
        entry: dict[str, Any] = {"step": step, "train_loss": float(loss.detach()), "regularizer": float(regularizer.detach())}
        if step % arguments.eval_every == 0 or step == arguments.max_steps:
            metrics = evaluate(model, eval_loader, device, arguments.seed)
            entry["eval"] = metrics
            if math.isfinite(metrics["loss"]) and metrics["loss"] < best["loss"]:
                best = {"step": step, "loss": metrics["loss"], "state": [parameter.detach().clone() for parameter in parameters]}
                stale = 0
            else:
                stale += 1
            print(f"step {step}: train {entry['train_loss']:.4f}, eval {metrics['loss']:.4f} (best {best['loss']:.4f} at {best['step']})")
            history.append(entry)
            if arguments.patience is not None and stale >= arguments.patience:
                break
        else:
            history.append(entry)

    with torch.no_grad():
        for parameter, value in zip(parameters, best["state"]):
            parameter.copy_(value)
    merge_lora(model)
    model.requires_grad_(True)
    after = evaluate(model, eval_loader, device, arguments.seed)
    print(f"after: eval loss {after['loss']:.4f} (best step {best['step']})")
    model.save_pretrained(output)
    log = {
        "checkpoint": str(arguments.checkpoint),
        "data_config": data_config if isinstance(data_config, dict) else str(data_config),
        "settings": {key: value for key, value in vars(arguments).items()},
        "lora_alpha": alpha,
        "lora_modules": wrapped,
        "best_step": best["step"],
        "steps_run": history[-1]["step"] if history else 0,
        "eval_before": before,
        "eval_after": after,
        "history": history,
    }
    (output / "adaptation_log.json").write_text(json.dumps(log, indent=2))
    return log


if __name__ == "__main__":
    main()
