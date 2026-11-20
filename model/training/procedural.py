"""A training source of procedurally generated robots (``model.morphology``).

The source generates robots, writes their URDFs, and hands each one to an existing
simulator source (``synthetic_source`` or ``mujoco_source`` of
``model.training.sources``; MuJoCo is Todorov, Erez and Tassa, "MuJoCo: A physics
engine for model-based control", IROS 2012, pp. 5026-5033, DOI
10.1109/IROS.2012.6386109). The generator follows GenBot-1K (Ai et al., CoRL 2025,
arXiv:2505.05753) and GenLoco (Feng et al., CoRL 2022, arXiv:2209.05309); see
``model.morphology`` for every value. The results are concatenated, so it returns what every
source returns: embodiment records and trajectories whose ``embodiment`` indices
point into those records.

Source keys:

* ``family`` (one of ``quadruped``, ``hexapod``, ``humanoid``) or ``families`` (a
  list, cycled over the robots);
* ``count`` (number of robots, default 1) and ``seed`` (default 0). Robot ``i`` is
  ``generate_robot(families[i % n], suite_seeds(seed, count)[i], ranges)``, the same
  robots ``scripts/generate_morphologies.py`` writes for that seed;
* ``ranges``: overrides of ``model.morphology.DEFAULT_RANGES`` (all families);
* ``simulator``: ``synthetic`` (default; floating base on a center-of-mass harness,
  ``base_modes`` ``["floating"]`` unless given) or ``mujoco`` (standing on the ground);
* ``output_dir``: where URDFs are written (default ``.cache/procedural_urdf``,
  relative to the working directory). File names carry a hash of the URDF text, so
  sources with different settings never overwrite each other's robots;
* ``name``: prefix of the record names (default ``procedural``);
* every other key (``episodes_per_embodiment``, ``steps``, ``dt``,
  ``randomized_embodiments``, ``probe_torques``, ``randomization``, ...) is passed to
  the simulator source unchanged.

Each simulator call gets ``urdf``, ``name``, ``seed`` (the robot's seed),
``contact_links`` and ``end_effectors`` from the generator, and for MuJoCo also
``nominal_pose`` and ``contact_geometry`` (unless the source sets them). Each record
gets ``family`` and ``metadata = {"env": {...}, "generator": {...}}``: the per-robot
overrides an RL loader needs (nominal pose, feet, foot geometry) and the generator's
description of the robot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from model.embodiment import JointFeatureConfig
from model.morphology import GENERATOR_VERSION, generate_robot, suite_families, suite_seeds
from model.training.episodes import EmbodimentRecord, Trajectory

SIMULATORS = ("synthetic", "mujoco")
PROCEDURAL_KEYS = ("type", "family", "families", "count", "seed", "simulator", "ranges", "output_dir", "name")
DEFAULT_OUTPUT_DIR = ".cache/procedural_urdf"


def _families(source: dict[str, Any]) -> list[str] | str:
    if "families" in source and "family" in source:
        raise ValueError("give either 'family' or 'families', not both")
    families = source.get("families", source.get("family"))
    if families is None:
        raise ValueError("a procedural source needs 'family' or 'families'")
    return families


def generated_robots(source: dict[str, Any]) -> list[tuple[str, int, Any]]:
    """Generate a procedural source's robots without writing or simulating them.

    Args:
        source (dict): Procedural source config (see the module docstring).

    Returns:
        list[tuple[str, int, GeneratedRobot]]: ``(family, seed, robot)`` per robot.
    """
    count = int(source.get("count", 1))
    if count < 1:
        raise ValueError("count must be at least 1")
    families = suite_families(_families(source), count)
    seeds = suite_seeds(int(source.get("seed", 0)), count)
    ranges = source.get("ranges")
    return [(family, seed, generate_robot(family, seed, ranges)) for family, seed in zip(families, seeds)]


def source_fingerprint(source: dict[str, Any]) -> bytes:
    """Return bytes that change whenever a procedural source's data would change.

    Meant for dataset cache keys: the config itself plus the generated URDF texts,
    so a change of the generator code invalidates the cache too.

    Args:
        source (dict): Procedural source config.

    Returns:
        bytes: Deterministic fingerprint.
    """
    digest = hashlib.sha256(json.dumps(source, sort_keys=True, default=str).encode())
    digest.update(str(GENERATOR_VERSION).encode())
    for _, _, robot in generated_robots(source):
        digest.update(robot.urdf.encode())
    return digest.digest()


def procedural_source(
    source: dict[str, Any], base: Path | None, feature_config: JointFeatureConfig
) -> tuple[list[EmbodimentRecord], list[Trajectory]]:
    """Generate robots and simulate them with the synthetic or MuJoCo source.

    Args:
        source (dict): Source config (see the module docstring).
        base (Path | None): Directory for relative paths of the delegated source.
        feature_config (JointFeatureConfig): Observation hardware features.

    Returns:
        tuple: Embodiment records and trajectories; trajectory ``embodiment``
            indices point into the returned records.

    Raises:
        ValueError: The family, count or simulator is invalid.
    """
    from model.training.sources import mujoco_source, synthetic_source

    simulator = source.get("simulator", "synthetic")
    if simulator not in SIMULATORS:
        raise ValueError(f"simulator must be one of {SIMULATORS}, not {simulator!r}")
    output = Path(source.get("output_dir", DEFAULT_OUTPUT_DIR)).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    passthrough = {key: value for key, value in source.items() if key not in PROCEDURAL_KEYS}
    prefix = source.get("name", "procedural")
    records: list[EmbodimentRecord] = []
    trajectories: list[Trajectory] = []
    for index, (family, seed, robot) in enumerate(generated_robots(source)):
        metadata = robot.metadata
        digest = hashlib.sha1(robot.urdf.encode()).hexdigest()[:12]
        path = output / f"{family}_{digest}.urdf"
        if not path.exists() or path.read_text(encoding="utf-8") != robot.urdf:
            path.write_text(robot.urdf, encoding="utf-8")
        delegated = {
            **passthrough,
            "type": simulator,
            "urdf": str(path.resolve()),
            "name": f"{prefix}/{family}_{index:04d}",
            "seed": seed,
            "contact_links": list(metadata["contact_links"]),
        }
        delegated.setdefault("end_effectors", list(metadata["end_effectors"]))
        if simulator == "synthetic":
            delegated.setdefault("base_modes", ["floating"])
            run = synthetic_source
        else:
            delegated.setdefault("nominal_pose", dict(metadata["nominal_pose"]))
            delegated.setdefault("contact_geometry", dict(metadata["contact_geometry"]))
            run = mujoco_source
        new_records, new_trajectories = run(delegated, base, feature_config)
        env = {
            "nominal_pose": dict(metadata["nominal_pose"]),
            "contact_links": list(metadata["contact_links"]),
            "contact_geometry": json.loads(json.dumps(metadata["contact_geometry"])),
        }
        generator = {
            "family": family,
            "params": json.loads(json.dumps(metadata["params"])),
            "seed": seed,
            "total_mass": metadata["total_mass"],
            "characteristic_length": metadata["characteristic_length"],
            "torque_ratio": metadata["torque_ratio"],
            "urdf": str(path.resolve()),
        }
        offset = len(records)
        records += [
            replace(record, family=family, metadata={"env": dict(env), "generator": dict(generator)})
            for record in new_records
        ]
        trajectories += [replace(trajectory, embodiment=offset + trajectory.embodiment) for trajectory in new_trajectories]
    return records, trajectories
