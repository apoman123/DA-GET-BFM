"""Measure how well contacts are inferred from motion, against simulator or foot-sensor contacts.

Response labels (tasks R and R^-1) of data without contact sensors depend on inferred
contacts (``model.training.contacts``). This script hides the reference contacts of
episodes that have them, infers them with each method, and prints:

* per-foot accuracy, precision, recall and F1, and the share of frames whose whole set
  is right (``set``);
* ``usable``: the share of frames that would get response labels (no contact change
  within ``contact_margin_steps``, no foot slipping, unexplained base wrench at most
  ``max_contact_residual`` of the weight), and ``usable set``: the share of those with
  the whole set right. A wrong set there is a wrong response label.

Sources:

* ``--data_config``: every source of a data config whose episodes carry contacts
  (``mujoco`` ground simulation, or recorded data with ``contact_key``);
* ``--mipo``: MIPO Unitree Go1 sequences (``Open-IO/mipo-go1-odometry``), whose foot-force
  sensors are the reference. The raw readings are bimodal (in the air near a per-sensor
  offset, on the ground 40 to 1200), and the offset drifts between sensors and sequences
  (one stand-trot sensor reads 24 to 39 in the air), so a foot is down above its own
  ``offset + mipo_relative_threshold * range`` (5th percentile plus 10% of the 5th-95th
  percentile range by default; ``--mipo_threshold`` sets an absolute value instead). The
  earlier absolute threshold of 200 misses light contacts: a quarter of its "flight"
  frames still need about half the weight from the ground. MIPO has no base velocity, so
  the foot-speed evidence and the slip check are unavailable there.

Usage::

    uv run python scripts/evaluate_contact_detection.py --data_config configs/data/ground_contact.json
    uv run --extra validation python scripts/evaluate_contact_detection.py \\
        --mipo indoor/20230625_gait/230625-risqh-flytrot-04-032-33square.h5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.training.contacts import CONTACT_DETECTION_METHODS, evaluate_contacts  # noqa: E402
from model.training.episodes import EmbodimentRecord, Trajectory, process_episode  # noqa: E402


def episode_metrics(
    record: EmbodimentRecord, trajectory: Trajectory, method: str, margin: int, max_residual: float, max_slip: float | None,
    min_confidence: float = 3.0,
) -> tuple[dict[str, float], int]:
    """Infer the contacts of one episode with ``method`` and compare with its own contacts."""
    reference = trajectory.contacts.bool()
    hidden = replace(trajectory, contacts=None, probe_torques=None, probe_executed=None,
                     probe_accelerations=None, probe_base_accelerations=None)
    episode = process_episode(
        record, hidden, proposal_variants=0, contact_margin_steps=margin, max_contact_slip=max_slip, contact_detection=method,
        min_contact_confidence=min_confidence,
    )
    gravity = episode["base_features"][:, :3].double()  # unit direction; the weight uses g
    weight = record.graph.total_mass * 9.80665 * torch.ones(gravity.shape[0], dtype=torch.float64)
    usable = episode["response_valid"] & (episode["contact_residual"].double() <= max_residual * weight)
    return evaluate_contacts(episode["contact_flags"], reference, usable), reference.shape[0]


def report(name: str, items: list[tuple[EmbodimentRecord, Trajectory]], arguments: argparse.Namespace) -> None:
    """Print one row per method, averaged over the episodes' frames."""
    for method in arguments.methods:
        totals: dict[str, float] = {}
        frames = 0
        for record, trajectory in items:
            metrics, count = episode_metrics(
                record, trajectory, method, arguments.contact_margin_steps, arguments.max_contact_residual, arguments.max_contact_slip,
                arguments.min_contact_confidence,
            )
            for key, value in metrics.items():
                if value == value:  # skip NaN (no usable frame)
                    totals[key] = totals.get(key, 0.0) + value * count
            frames += count
        values = {key: value / max(frames, 1) for key, value in totals.items()}
        print(
            f"{name:<34} {method:<9} foot {values['foot_accuracy']:.3f}  precision {values['precision']:.3f}  "
            f"recall {values['recall']:.3f}  F1 {values['f1']:.3f}  set {values['set_accuracy']:.3f}  "
            f"usable {values.get('usable_fraction', 0):.3f}  usable set {values.get('usable_set_accuracy', float('nan')):.3f}",
            flush=True,
        )


def config_items(path: str) -> dict[str, list[tuple[EmbodimentRecord, Trajectory]]]:
    """Load a data config's sources and keep episodes that have contacts, grouped by source."""
    from model.training.dataset import joint_feature_config, load_data_config
    from model.training.sources import SOURCE_TYPES

    config, base = load_data_config(path)
    feature_config = joint_feature_config(config.get("joint_features"))
    groups: dict[str, list[tuple[EmbodimentRecord, Trajectory]]] = {}
    for source in config["sources"]:
        records, trajectories = SOURCE_TYPES[source.get("type", "huggingface")](source, base, feature_config)
        for trajectory in trajectories:
            if trajectory.contacts is not None and records[trajectory.embodiment].contact_links:
                groups.setdefault(source.get("name", source.get("type", "source")), []).append((records[trajectory.embodiment], trajectory))
    return groups


def sensor_contacts(force: torch.Tensor, relative: float = 0.1, threshold: float | None = None) -> torch.Tensor:
    """Threshold foot sensor readings ``[T, C]`` per sensor (offset plus a share of its range), or absolutely."""
    if threshold is not None:
        return force > threshold
    low, high = force.quantile(0.05, dim=0), force.quantile(0.95, dim=0)
    return force > low + relative * (high - low)


def mipo_items(
    sequences: list[str], window: int, stride: int, relative: float = 0.1, threshold: float | None = None
) -> dict[str, list[tuple[EmbodimentRecord, Trajectory]]]:
    """Load MIPO sequences with the Go1 URDF; foot sensors give the reference contacts."""
    from huggingface_hub import hf_hub_download

    from model import EmbodimentGraph, JointFeatureConfig, actuator_specs

    spec = importlib.util.spec_from_file_location("validate_torque_estimation", ROOT / "scripts" / "validate_torque_estimation.py")
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    names = [f"{leg}_{part}_joint" for leg in validation.LEGS for part in validation.PARTS]
    graph = EmbodimentGraph.from_urdf(ROOT / "assets" / "urdf" / "go1.urdf", JointFeatureConfig(), joint_order=names, floating_base=True)
    record = EmbodimentRecord.build("go1", graph, actuator_specs(graph), contact_links=validation.FEET)
    groups = {}
    for sequence in sequences:
        data = validation.load(Path(hf_hub_download(validation.REPOSITORY, sequence, repo_type="dataset")), window, stride)
        trajectory = Trajectory(
            embodiment=0, dt=data["dt"], positions=data["q"], velocities=data["qd"], gravity=data["gravity"],
            base_orientation=data["rotation"], base_angular_velocity=data["gyro"], imu_specific_force=data["specific_force"],
            contacts=sensor_contacts(data["foot_force"], relative, threshold),
        )
        groups[f"MIPO {Path(sequence).stem}"] = [(record, trajectory)]
    return groups


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_config", action="append", default=[], help="Data config whose sources carry contacts")
    parser.add_argument("--mipo", nargs="*", default=[], help="MIPO sequence paths in the Hub repository")
    parser.add_argument("--methods", nargs="+", default=list(CONTACT_DETECTION_METHODS), choices=CONTACT_DETECTION_METHODS)
    parser.add_argument("--contact_margin_steps", type=int, default=2)
    parser.add_argument("--max_contact_residual", type=float, default=0.5)
    parser.add_argument("--max_contact_slip", type=float, default=0.1)
    parser.add_argument("--min_contact_confidence", type=float, default=3.0, help="Dynamics method: cost margin of the chosen set")
    parser.add_argument("--mipo_relative_threshold", type=float, default=0.1, help="MIPO: share of each sensor's range above its offset")
    parser.add_argument("--mipo_threshold", type=float, default=None, help="MIPO: absolute sensor reading instead")
    parser.add_argument("--window", type=int, default=8, help="MIPO smoothing window (400 Hz samples)")
    parser.add_argument("--stride", type=int, default=8, help="MIPO subsampling (8: 50 Hz)")
    arguments = parser.parse_args(argv)
    groups: dict[str, list] = {}
    for path in arguments.data_config:
        groups.update(config_items(path))
    if arguments.mipo:
        groups.update(mipo_items(arguments.mipo, arguments.window, arguments.stride, arguments.mipo_relative_threshold, arguments.mipo_threshold))
    if not groups:
        parser.error("give --data_config or --mipo")
    for name, items in groups.items():
        report(name, items, arguments)


if __name__ == "__main__":
    main()
