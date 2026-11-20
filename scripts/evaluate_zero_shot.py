"""Evaluate a behavior foundation model per zero-shot level, with IQM and robot-level CIs.

For every split level (L0–L4 of ``configs/splits/zero_shot.json``) the script runs the
repository's own BFM evaluation (``model.rl.evaluation.evaluate``: tracking MPJPE and
EMD, the fallen fraction during tracking, goal MPJPE, and the getting-up rate of
floating robots), then aggregates every metric over robots:

* the interquartile mean (IQM) over robots and seeds (Agarwal, Schwarzer, Castro,
  Courville and Bellemare, NeurIPS 2021, arXiv:2108.13264);
* a percentile bootstrap interval (Efron, Ann. Statist. 7(1):1-26, 1979, DOI
  10.1214/aos/1176344552) with the robot as the unit (robots drawn with replacement,
  then each robot's seeds; rliable's ``task_bootstrap=True``,
  https://github.com/google-research/rliable, Apache-2.0), with rliable's defaults of
  50000 replicates and 95%;
* the gap ``Lk - L0`` of every level, with its interval.

Runs are seeds: pass ``--checkpoint`` once per BFM training run (Henderson, Islam,
Bachman, Pineau, Precup and Meger, AAAI 2018, DOI 10.1609/aaai.v32i1.11694: results
vary across seeds). Each run directory holds
``checkpoint.pt`` and ``encoder/`` as written by ``scripts/train_bfm.py``.

Robots of a level come from a data config of that level (``--level-data-config
L1=path.json``): its expert sources give each robot's motions and worlds, exactly as for
training (``model.rl.trainer.load_embodiments``). A robot of the data config belongs to
a level when its source's URDF is that of a manifest entry, or its name is an entry's
name or continues it after a ``/`` (``model.evaluation.splits.name_matches``). Manifest robots without a matching data-config robot
are listed under ``missing`` with the reason; nothing is filled in for them.

Example (MuJoCo, CPU)::

    uv run python scripts/evaluate_zero_shot.py --checkpoint runs/bfm_seed0 --checkpoint runs/bfm_seed1 \\
        --split-manifest configs/splits/zero_shot.json \\
        --level-data-config L0=configs/rl/l0_heldout_motions.json \\
        --level-data-config L4=configs/rl/l4_go2.json \\
        --training-data-config configs/rl/zero_shot_training.json --split val --output zero_shot.json

What is not supported (reported, not faked):

* Isaac Lab: the script uses MuJoCo worlds; Isaac Lab needs its app launcher
  (``scripts/train_bfm.py`` shows how).
* L0 "held-out episodes": the RL trainer has no held-out motion split, so the L0 data
  config must contain motions that were not used in training.
* Robots without expert motions (every generated L1–L3 robot, and Go2 and Talos until
  motions are retargeted to them) cannot be evaluated: all three tasks start from, or
  are prompted by, a motion of the robot itself.
* MPJPE and EMD are Euclidean norms over all joints (``||q - q_ref||_2`` per frame,
  averaged over frames: BFM-Zero's ``E_mpjpe``, Li et al., ICLR 2026, arXiv:2511.04131,
  Appendix C; the EMD of Rubner, Tomasi and Guibas, IJCV 40(2):99-121, 2000, DOI
  10.1023/A:1026543900054, as BFM-Zero uses it), so they grow with the joint count;
  IQMs across robots with different joint counts mix scales.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from model.evaluation.splits import LEVELS, SplitManifest, name_matches  # noqa: E402
from model.evaluation.statistics import (  # noqa: E402
    RLIABLE_CONFIDENCE,
    RLIABLE_REPETITIONS,
    interquartile_mean,
    performance_gap,
    robot_bootstrap_ci,
)

METRICS = ("tracking_mpjpe", "tracking_emd", "tracking_fallen", "goal_mpjpe", "getting_up_rate")
# EnvConfig.num_envs default and model.rl.evaluation.evaluate's horizon default.
DEFAULT_ENVS_PER_ROBOT = 16
DEFAULT_HORIZON = 250


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", required=True, help="BFM run directory (repeat once per seed)")
    parser.add_argument("--split-manifest", default=str(ROOT / "configs" / "splits" / "zero_shot.json"))
    parser.add_argument("--level-data-config", action="append", default=[], help="LEVEL=path.json with the level's expert sources")
    parser.add_argument("--training-data-config", default=None, help="the BFM's training data config, to check it excludes L1-L4 robots")
    parser.add_argument("--split", choices=("val", "test", "all"), default="val", help="split of L1-L4 robots; L0 is always evaluated")
    parser.add_argument("--simulator", choices=("mujoco",), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--envs-per-robot", type=int, default=DEFAULT_ENVS_PER_ROBOT)
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--repetitions", type=int, default=RLIABLE_REPETITIONS)
    parser.add_argument("--confidence", type=float, default=RLIABLE_CONFIDENCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


def parse_level_configs(items: Sequence[str]) -> dict[str, Path]:
    """Parse ``LEVEL=path`` arguments.

    Args:
        items (Sequence[str]): Arguments.

    Returns:
        dict[str, Path]: Level to data config.

    Raises:
        ValueError: An item is malformed or names an unknown level.
    """
    configs: dict[str, Path] = {}
    for item in items:
        level, separator, path = item.partition("=")
        if not separator or level not in LEVELS:
            raise ValueError(f"expected LEVEL=path with LEVEL in {LEVELS}, got {item!r}")
        configs[level] = Path(path)
    return configs


def match_entry(manifest: SplitManifest, level: str, name: str, urdf: Path | None) -> Any | None:
    """Return the manifest entry of a level that a data-config robot is, or None.

    Args:
        manifest (SplitManifest): Manifest.
        level (str): Level.
        name (str): Robot name in the data config.
        urdf (Path | None): Resolved URDF of its source.

    Returns:
        RobotEntry | None: Matching entry.
    """
    for entry in manifest.levels.get(level, []):
        if urdf is not None and entry.urdf and manifest.resolve(entry.urdf) == urdf.resolve():
            return entry
        if name_matches(entry.name, name):
            return entry
    return None


def wanted(entry: Any, level: str, split: str) -> bool:
    """Return whether an entry is evaluated for the chosen split (L0 always).

    Args:
        entry (RobotEntry): Entry.
        level (str): Its level.
        split (str): ``val``, ``test`` or ``all``.

    Returns:
        bool: Evaluate it.
    """
    return level == "L0" or split == "all" or entry.split == split


def parse_metrics(metrics: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """Group ``eval/<robot>/<metric>`` values by robot.

    Args:
        metrics (Mapping[str, float]): Output of ``model.rl.evaluation.evaluate``.

    Returns:
        dict[str, dict[str, float]]: Robot to metric to value.
    """
    grouped: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if not key.startswith("eval/"):
            continue
        robot, metric = key[len("eval/") :].rsplit("/", 1)
        grouped.setdefault(robot, {})[metric] = float(value)
    return grouped


def aggregate(
    scores: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    reference: str = "L0",
    repetitions: int = RLIABLE_REPETITIONS,
    confidence: float = RLIABLE_CONFIDENCE,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate per-robot, per-seed scores into IQMs, robot-level CIs and gaps to L0.

    Args:
        scores (Mapping): ``{metric: {level: {robot: [value per seed]}}}``.
        reference (str): Level of the gaps.
        repetitions (int): Bootstrap replicates.
        confidence (float): Interval size.
        seed (int): Bootstrap seed.

    Returns:
        dict[str, Any]: ``{metric: {"levels": {level: {"iqm", "ci", "robots"}}, "gap_to_<ref>": {...}}}``.
    """
    report: dict[str, Any] = {}
    for metric, levels in scores.items():
        present: dict[str, dict[str, list[float]]] = {}
        dropped: list[str] = []
        for level, robots in levels.items():
            for robot, values in robots.items():
                finite = [float(value) for value in values if math.isfinite(float(value))]
                if len(finite) < len(values):
                    dropped.append(f"{level}/{robot}: {len(values) - len(finite)} nonfinite value(s)")
                if finite:
                    present.setdefault(level, {})[robot] = finite
        entry: dict[str, Any] = {"levels": {}}
        if dropped:
            entry["dropped"] = dropped
        for level, robots in present.items():
            interval = robot_bootstrap_ci(robots, interquartile_mean, repetitions, confidence, seed)
            entry["levels"][level] = {
                "iqm": interval.estimate,
                "ci": [interval.lower, interval.upper],
                "robots": {robot: list(values) for robot, values in robots.items()},
            }
        if reference in present and len(present) > 1:
            gaps = performance_gap(present, reference, interquartile_mean, "robot", repetitions, confidence, seed)
            entry[f"gap_to_{reference}"] = {level: {"gap": gap.estimate, "ci": [gap.lower, gap.upper]} for level, gap in gaps.items()}
        report[metric] = entry
    return report


def load_agent(run_dir: Path, embodiments: list, device: str):
    """Build the agent for a level's robots and load a run's trained networks.

    Only robot-independent state is loaded (``model`` and ``pretrained``); the
    optimizers, z buffers and per-robot reward weights are training state.

    Args:
        run_dir (Path): BFM run directory.
        embodiments (list): ``Embodiment`` objects of the level.
        device (str): Device.

    Returns:
        DaGetFbCprAgent: Agent ready for ``evaluate``.

    Raises:
        ValueError: The encoder's token size differs from the robots'.
    """
    from model.configuration_bfm import BfmConfig
    from model.pretraining import DaGetForPhysicsPretraining
    from model.rl.agent import DaGetFbCprAgent
    from model.rl.configuration_fb_cpr import FbCprConfig

    state = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model_config = BfmConfig(**state["model_config"])
    config = FbCprConfig(**state["config"])
    # The agent checks that an update can hold one expert window per robot; evaluation
    # does no update, so the batch is only widened to pass that check.
    needed = model_config.seq_length * len(embodiments)
    if config.batch_size < needed:
        config = dataclasses.replace(config, batch_size=needed)
    pretrained = DaGetForPhysicsPretraining.from_pretrained(run_dir / "encoder")
    observers = [embodiment.observer for embodiment in embodiments]
    if pretrained.config.encoder["input_size"] != observers[0].token_size:
        raise ValueError("the encoder's token size differs from the level's data config (joint_features, history_size)")
    agent = DaGetFbCprAgent(model_config, config, pretrained, observers, device)
    agent.model.load_state_dict(state["model"])
    agent.pretrained.load_state_dict(state["pretrained"])
    return agent


def evaluate_level(run_dir: Path, embodiments: list, arguments: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Run ``model.rl.evaluation.evaluate`` on one level's robots in MuJoCo.

    Args:
        run_dir (Path): BFM run directory.
        embodiments (list): The level's ``Embodiment`` objects.
        arguments (argparse.Namespace): Parsed arguments.

    Returns:
        dict[str, dict[str, float]]: Robot to metric to value.
    """
    from model.rl.evaluation import evaluate
    from model.rl.trainer import ObservationNoise, Rollout, TrainingArguments, build_group

    agent = load_agent(run_dir, embodiments, arguments.device)
    for embodiment in embodiments:
        embodiment.env_config.num_envs = arguments.envs_per_robot
        embodiment.env_config.label_worlds = 0
    settings = TrainingArguments(data_config="", output_dir="", simulator=arguments.simulator, device=arguments.device)
    group = build_group(settings, embodiments, None)
    generator = torch.Generator().manual_seed(arguments.seed)
    rollouts = []
    for embodiment, env in zip(embodiments, group.envs):
        env.set_motions(embodiment.motions.frames)
        rollouts.append(Rollout(env, embodiment.observer, ObservationNoise(enabled=False), generator))
    metrics, _ = evaluate(agent, rollouts, group, [embodiment.motions for embodiment in embodiments], arguments.max_motions, arguments.horizon)
    return parse_metrics(metrics)


def level_robots(manifest: SplitManifest, level: str, path: Path, arguments: argparse.Namespace) -> tuple[list, list[dict]]:
    """Load a level's data config and keep the robots the manifest puts in the level.

    Args:
        manifest (SplitManifest): Manifest.
        level (str): Level.
        path (Path): Data config.
        arguments (argparse.Namespace): Parsed arguments.

    Returns:
        tuple[list, list[dict]]: Embodiments to evaluate, and skipped robots with reasons.
    """
    from model.rl.trainer import TrainingArguments, load_embodiments
    from model.training.sources import resolve_path

    settings = TrainingArguments(
        data_config=str(path), output_dir=tempfile.gettempdir(), simulator=arguments.simulator,
        domain_randomization=False, observation_noise=False, device=arguments.device,
    )
    embodiments, _ = load_embodiments(path, settings)
    kept, skipped = [], []
    for embodiment in embodiments:
        urdf = resolve_path(embodiment.source["urdf"], embodiment.base) if "urdf" in embodiment.source else None
        entry = match_entry(manifest, level, embodiment.record.name, urdf)
        if entry is None:
            other = next((name for name in manifest.levels if name != level and match_entry(manifest, name, embodiment.record.name, urdf)), None)
            reason = f"belongs to {other}" if other else "not in the manifest"
            skipped.append({"robot": embodiment.record.name, "reason": reason})
        elif not wanted(entry, level, arguments.split):
            skipped.append({"robot": embodiment.record.name, "reason": f"split {entry.split} not selected"})
        else:
            kept.append(embodiment)
    return kept, skipped


def missing_entries(manifest: SplitManifest, level: str, evaluated: set[str], split: str) -> list[dict]:
    """List manifest robots of a level that were not evaluated, with the reason.

    Args:
        manifest (SplitManifest): Manifest.
        level (str): Level.
        evaluated (set[str]): Names of the evaluated robots (``name_matches`` decides).
        split (str): Selected split.

    Returns:
        list[dict]: ``{"robot", "reason"}`` per missing robot.
    """
    missing = []
    for entry in manifest.levels.get(level, []):
        if not wanted(entry, level, split) or any(name_matches(entry.name, name) for name in evaluated):
            continue
        if entry.generator is not None:
            reason = "generator specification: no URDF and no expert motions generated yet"
        elif entry.urdf and not manifest.resolve(entry.urdf).exists():
            reason = f"URDF {entry.urdf} is not on disk" + (f" (download: {entry.download['url']})" if entry.download else "")
        else:
            reason = "no expert-motion source for it in the level's data config"
        missing.append({"robot": entry.name, "reason": reason})
    return missing


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Evaluate every level and write the report.

    Args:
        argv (list[str] | None): Command-line arguments; ``sys.argv`` if None.

    Returns:
        dict[str, Any]: The report.
    """
    arguments = build_parser().parse_args(argv)
    manifest = SplitManifest.load(arguments.split_manifest)
    level_configs = parse_level_configs(arguments.level_data_config)
    report: dict[str, Any] = {
        "checkpoints": arguments.checkpoint,
        "split": arguments.split,
        "statistics": {"aggregate": "IQM", "interval": "percentile bootstrap, robot as unit", "repetitions": arguments.repetitions, "confidence": arguments.confidence},
        "warnings": [],
        "missing": {},
        "skipped": {},
    }
    if arguments.training_data_config:
        violations = manifest.check_training_config(arguments.training_data_config)
        report["training_config_violations"] = violations
        if violations:
            report["warnings"].append("the training data config contains held-out robots; their levels are not zero-shot")
    else:
        report["warnings"].append("no --training-data-config: the check that the BFM never trained on L1-L4 robots was skipped")
    if "L0" not in level_configs:
        report["warnings"].append("no L0 data config: gaps to L0 cannot be computed")
    scores: dict[str, dict[str, dict[str, list[float]]]] = {metric: {} for metric in METRICS}
    for level in LEVELS:
        if level not in level_configs:
            report["missing"][level] = missing_entries(manifest, level, set(), arguments.split) or "no robots"
            continue
        embodiments, skipped = level_robots(manifest, level, level_configs[level], arguments)
        report["skipped"][level] = skipped
        evaluated = {embodiment.record.name for embodiment in embodiments}
        report["missing"][level] = missing_entries(manifest, level, evaluated, arguments.split)
        if not embodiments:
            continue
        for run in arguments.checkpoint:
            per_robot = evaluate_level(Path(run), embodiments, arguments)
            for robot, values in per_robot.items():
                for metric, value in values.items():
                    if metric in scores:
                        scores[metric].setdefault(level, {}).setdefault(robot, []).append(value)
    report["metrics"] = aggregate(scores, "L0", arguments.repetitions, arguments.confidence, arguments.seed)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {output}")
    return report


if __name__ == "__main__":
    main()
