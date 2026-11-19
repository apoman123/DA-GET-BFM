"""Zero-shot split manifest: which robots belong to which generalization level.

Levels (proposal v4, Sec. 5.1):

* **L0**: the training robots, on held-out episodes (the data config's eval split).
* **L1**: parameter changes of training robots. The categories are MetaMorph's six
  zero-shot variations [1] ("variation in dynamics parameters (armature, density,
  damping, gear) and kinematics parameters (module shape, joint angle)", Fig. 6; ranges
  in Table 2), plus size extrapolation.
* **L2**: new topology within a training family, such as the number of knee joints per
  leg and the foot size (the topology parameters of Ai et al. [2], Table 6), from a
  procedural generator.
* **L3**: a whole family held out, for example hexapods.
* **L4**: real robot models.

L1–L3 robots are split into ``val`` (model selection) and ``test`` (used once at the
end) by ``make_heldout_split``; its 20% held-out fraction is Ai et al.'s [2]: "we fix a
constant test set by randomly holding out 20% of the generated embodiments". Here the
held-out 20% become ``test`` and the rest ``val`` (our mapping: Ai et al. hold robots
out of training, while every L1–L4 robot is already out of training).

A manifest is JSON::

    {
      "version": 1,
      "notes": ["..."],
      "training_robots": [{"name": "go1", "family": "quadruped", "urdf": "...", "split": "test"}],
      "levels": {
        "L0": [...],
        "L1": [{"name": "go1/density", "family": "quadruped", "split": "val",
                "generator": {"type": "parameter_variation", "base_robot": "go1", "category": "density"}}],
        "L4": [{"name": "go2", "family": "quadruped", "split": "test", "urdf": "...",
                "download": {"url": "...", "license": "BSD-3-Clause"}}]
      }
    }

Each robot has a name, a family, a split, and either a URDF path (relative to the
manifest) or generator parameters. Procedural entries name the generator call
(``{"type": "procedural", "module": "model.morphology", "function": "generate_robot",
"family", "seed", "ranges"}``); this module stores them and does not import the
generator. Unknown keys are kept, so later tools can add fields.

References:
    [1] A. Gupta, L. Fan, S. Ganguli, L. Fei-Fei. MetaMorph: Learning Universal
        Controllers with Transformers. ICLR 2022. arXiv:2203.11931.
    [2] B. Ai et al. Towards Embodiment Scaling Laws in Robot Locomotion. CoRL 2025.
        arXiv:2505.05753.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LEVELS = ("L0", "L1", "L2", "L3", "L4")
SPLITS = ("val", "test")
LEVEL_DESCRIPTIONS = {
    "L0": "training robots, held-out episodes",
    "L1": "parameter changes of training robots (MetaMorph's six categories) and size extrapolation",
    "L2": "new topology within a training family (knee count, foot size)",
    "L3": "a whole family held out",
    "L4": "real robot models",
}
# MetaMorph's zero-shot variation categories (Gupta et al., ICLR 2022, Fig. 6 and Table 2).
METAMORPH_VARIATIONS = ("armature", "density", "damping", "gear", "module_shape", "joint_angle")
L1_CATEGORIES = METAMORPH_VARIATIONS + ("size",)
# Held-out fraction of Ai et al. (CoRL 2025): "randomly holding out 20% of the generated embodiments".
HELDOUT_FRACTION = 0.2
GENERATOR_TYPES = ("parameter_variation", "procedural")
_ENTRY_KEYS = ("name", "family", "split", "urdf", "generator", "sources", "download", "notes")


@dataclass
class RobotEntry:
    """One robot of a split level.

    Attributes:
        name (str): Unique robot name across the manifest.
        family (str): Morphology family, e.g. ``quadruped``, ``humanoid``, ``hexapod``.
        split (str): ``val`` or ``test``.
        urdf (str | None): URDF path, relative to the manifest file.
        generator (dict | None): Generator parameters instead of a URDF:
            ``{"type": "parameter_variation", "base_robot", "category", ...}`` for L1, or
            ``{"type": "procedural", "module", "function", "family", "seed", "ranges"}``.
        sources (list[dict]): Data-config sources of this robot with expert motions, for
            the BFM evaluation (paths relative to the manifest).
        download (dict | None): Where to fetch the URDF: ``url``, ``license``, ...
        notes (str): Free text, such as known problems.
        extra (dict): Unknown keys, kept on save.
    """

    name: str
    family: str
    split: str = "test"
    urdf: str | None = None
    generator: dict | None = None
    sources: list[dict] = field(default_factory=list)
    download: dict | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> RobotEntry:
        """Build an entry from JSON values, keeping unknown keys in ``extra``.

        Args:
            values (Mapping[str, Any]): Parsed JSON object.

        Returns:
            RobotEntry: The entry.

        Raises:
            ValueError: ``name`` or ``family`` is missing.
        """
        if "name" not in values or "family" not in values:
            raise ValueError(f"robot entries need a name and a family: {dict(values)}")
        known = {key: values[key] for key in _ENTRY_KEYS if key in values}
        known["sources"] = list(known.get("sources") or [])
        known["notes"] = known.get("notes") or ""
        return cls(**known, extra={key: value for key, value in values.items() if key not in _ENTRY_KEYS})

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON object of the entry, without empty optional fields.

        Returns:
            dict[str, Any]: JSON-ready values.
        """
        values: dict[str, Any] = {"name": self.name, "family": self.family, "split": self.split}
        for key in ("urdf", "generator", "download"):
            if getattr(self, key) is not None:
                values[key] = getattr(self, key)
        if self.sources:
            values["sources"] = self.sources
        if self.notes:
            values["notes"] = self.notes
        values.update(self.extra)
        return values


def robot_key(name: str) -> str:
    """Return the robot of an embodiment record name.

    Synthetic sources name their randomized variants ``"<source>/<index>-<mode>"``
    (``model.training.sources.synthetic_source``); they are variants of one robot.

    Args:
        name (str): Record or source name.

    Returns:
        str: The part before the first ``/``.
    """
    return name.split("/", 1)[0]


def name_matches(entry_name: str, record_name: str) -> bool:
    """Return whether a record or source name is a manifest robot.

    It is when the names are equal or the record name continues the entry name after a
    ``/``: ``"go1/0-fixed"`` is ``go1``, and ``"go1/density/0-fixed"`` is
    ``go1/density`` but not ``go1/armature``.

    Args:
        entry_name (str): Manifest robot name.
        record_name (str): Embodiment record or source name.

    Returns:
        bool: The record belongs to the entry.
    """
    return record_name == entry_name or record_name.startswith(entry_name + "/")


def make_heldout_split(robots: Iterable[str | RobotEntry], fraction: float = HELDOUT_FRACTION, seed: int = 0) -> dict[str, str]:
    """Assign robots of one level to ``val`` and ``test`` at random.

    ``round(fraction * n)`` robots go to ``test`` (Ai et al.'s 20% held out by default).
    With two or more robots each split gets at least one; a single robot goes to
    ``test``. The order of ``robots`` does not matter: names are sorted before the
    seeded shuffle.

    Args:
        robots (Iterable[str | RobotEntry]): Names or entries of one level.
        fraction (float): Fraction assigned to ``test``, in (0, 1).
        seed (int): Shuffle seed.

    Returns:
        dict[str, str]: Robot name to ``val`` or ``test``.

    Raises:
        ValueError: The fraction is outside (0, 1) or names repeat.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    names = [item.name if isinstance(item, RobotEntry) else str(item) for item in robots]
    if len(set(names)) != len(names):
        raise ValueError("robot names must be unique")
    order = sorted(names)
    random.Random(seed).shuffle(order)
    count = len(order)
    held = round(fraction * count)
    if count >= 2:
        held = min(max(held, 1), count - 1)
    else:
        held = count
    return {name: ("test" if index < held else "val") for index, name in enumerate(order)}


@dataclass
class SplitManifest:
    """Robots of every generalization level and the training robots they must avoid.

    Attributes:
        training_robots (list[RobotEntry]): Robots the zero-shot models may train on.
        levels (dict[str, list[RobotEntry]]): Level name (``L0`` to ``L4``) to robots.
        notes (list[str]): Known conflicts and missing pieces.
        version (int): Format version.
        base_dir (Path | None): Directory relative paths start from (not saved).
    """

    training_robots: list[RobotEntry] = field(default_factory=list)
    levels: dict[str, list[RobotEntry]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    version: int = 1
    base_dir: Path | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], base_dir: Path | None = None) -> SplitManifest:
        """Build a manifest from parsed JSON.

        Args:
            values (Mapping[str, Any]): Parsed JSON.
            base_dir (Path | None): Directory of the manifest file.

        Returns:
            SplitManifest: Manifest (not yet validated).
        """
        return cls(
            training_robots=[RobotEntry.from_dict(item) for item in values.get("training_robots", [])],
            levels={str(level): [RobotEntry.from_dict(item) for item in items] for level, items in values.get("levels", {}).items()},
            notes=[str(note) for note in values.get("notes", [])],
            version=int(values.get("version", 1)),
            base_dir=base_dir,
        )

    @classmethod
    def load(cls, path: str | Path, validate: bool = True) -> SplitManifest:
        """Read and (by default) validate a manifest file.

        Args:
            path (str | Path): JSON file.
            validate (bool): Raise on an invalid manifest.

        Returns:
            SplitManifest: The manifest, with ``base_dir`` set to the file's folder.

        Raises:
            ValueError: The manifest is invalid.
        """
        path = Path(path)
        manifest = cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path.parent)
        if validate:
            manifest.validate()
        return manifest

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON object of the manifest.

        Returns:
            dict[str, Any]: JSON-ready values, levels in ``LEVELS`` order.
        """
        order = [level for level in LEVELS if level in self.levels] + [level for level in self.levels if level not in LEVELS]
        return {
            "version": self.version,
            "notes": list(self.notes),
            "levels_described": {level: LEVEL_DESCRIPTIONS[level] for level in LEVELS},
            "training_robots": [entry.to_dict() for entry in self.training_robots],
            "levels": {level: [entry.to_dict() for entry in self.levels[level]] for level in order},
        }

    def save(self, path: str | Path, validate: bool = True) -> None:
        """Write the manifest as JSON.

        Args:
            path (str | Path): Output file.
            validate (bool): Refuse to write an invalid manifest.

        Raises:
            ValueError: The manifest is invalid.
        """
        if validate:
            self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def resolve(self, path: str) -> Path:
        """Resolve a path relative to the manifest file.

        Args:
            path (str): Relative or absolute path.

        Returns:
            Path: Absolute, normalized path.
        """
        candidate = Path(path)
        if not candidate.is_absolute() and self.base_dir is not None:
            candidate = self.base_dir / candidate
        return candidate.resolve()

    def entries(self, level: str | None = None, split: str | None = None) -> list[RobotEntry]:
        """Return the robots of one level (or all levels), optionally of one split.

        Args:
            level (str | None): Level name; all levels if None.
            split (str | None): ``val`` or ``test``; both if None.

        Returns:
            list[RobotEntry]: Matching robots.
        """
        levels = [level] if level is not None else list(self.levels)
        return [entry for name in levels for entry in self.levels.get(name, []) if split is None or entry.split == split]

    def level_of(self, name: str) -> str | None:
        """Return the level of a robot name, or None.

        Args:
            name (str): Robot name.

        Returns:
            str | None: Level name.
        """
        for level, entries in self.levels.items():
            if any(entry.name == name for entry in entries):
                return level
        return None

    def problems(self) -> list[str]:
        """List every rule the manifest breaks.

        Rules: known level names; each robot has a family, a split in ``SPLITS``, and
        exactly one of a URDF or generator parameters; generator types and L1
        categories are known; L1 variations start from a training robot; no robot name
        is in two levels or twice in one; L0 robots are training robots; no L1–L4 robot
        is a training robot, by name or by URDF file.

        Returns:
            list[str]: Human-readable problems; empty when valid.
        """
        problems: list[str] = []
        training_names = [entry.name for entry in self.training_robots]
        if len(set(training_names)) != len(training_names):
            problems.append("training robot names repeat")
        training_urdfs = {self.resolve(entry.urdf): entry.name for entry in self.training_robots if entry.urdf}
        seen: dict[str, str] = {}
        for level, entries in self.levels.items():
            if level not in LEVELS:
                problems.append(f"unknown level {level!r}; use {LEVELS}")
            for entry in entries:
                where = f"{level}/{entry.name}"
                if not entry.family:
                    problems.append(f"{where}: missing family")
                if entry.split not in SPLITS:
                    problems.append(f"{where}: split must be one of {SPLITS}, got {entry.split!r}")
                if (entry.urdf is None) == (entry.generator is None):
                    problems.append(f"{where}: give exactly one of urdf and generator")
                if entry.generator is not None:
                    kind = entry.generator.get("type")
                    if kind not in GENERATOR_TYPES:
                        problems.append(f"{where}: generator type must be one of {GENERATOR_TYPES}")
                    if kind == "parameter_variation":
                        if entry.generator.get("category") not in L1_CATEGORIES:
                            problems.append(f"{where}: category must be one of {L1_CATEGORIES}")
                        if entry.generator.get("base_robot") not in training_names:
                            problems.append(f"{where}: base_robot must be a training robot")
                if entry.name in seen:
                    problems.append(f"{entry.name} is in {seen[entry.name]} and {level}")
                seen[entry.name] = level
                if level == "L0":
                    if entry.name not in training_names:
                        problems.append(f"{where}: L0 robots must be training robots")
                    continue
                if entry.name in training_names:
                    problems.append(f"{where}: {entry.name} is a training robot")
                if entry.urdf and self.resolve(entry.urdf) in training_urdfs:
                    problems.append(f"{where}: its URDF is the training robot {training_urdfs[self.resolve(entry.urdf)]}'s")
        return problems

    def validate(self) -> None:
        """Raise if the manifest breaks a rule (see ``problems``).

        Raises:
            ValueError: With every problem listed.
        """
        problems = self.problems()
        if problems:
            raise ValueError("invalid split manifest:\n  " + "\n  ".join(problems))

    def held_out_urdfs(self) -> dict[Path, str]:
        """Return the URDF file of every L1–L4 robot that has one.

        Returns:
            dict[Path, str]: Resolved path to ``"<level>/<name>"``.
        """
        return {
            self.resolve(entry.urdf): f"{level}/{entry.name}"
            for level, entries in self.levels.items()
            if level != "L0"
            for entry in entries
            if entry.urdf
        }

    def check_training_config(self, data_config: str | Path | Mapping[str, Any], base_dir: Path | None = None) -> list[str]:
        """List sources of a training data config that are L1–L4 robots.

        A source matches by URDF file or by name (``robot``, else ``name``; see
        ``name_matches``). A procedural source (``type`` ``procedural``) matches when
        it generates an L3 family, or when its ranges include an L1–L4 procedural
        entry (a range it leaves at the generator's defaults counts as including it).
        Use it before training a zero-shot model.

        Args:
            data_config (str | Path | Mapping): Data config path or parsed dict.
            base_dir (Path | None): Directory relative URDF paths of a dict start from.

        Returns:
            list[str]: One message per offending source; empty when clean.
        """
        if isinstance(data_config, Mapping):
            config = dict(data_config)
        else:
            path = Path(data_config)
            config = json.loads(path.read_text(encoding="utf-8"))
            base_dir = path.parent
        urdfs = self.held_out_urdfs()
        names = [(entry.name, level) for level, entries in self.levels.items() if level != "L0" for entry in entries]
        held_families = {entry.family for entry in self.levels.get("L3", [])}
        procedural = [
            (entry, level)
            for level, entries in self.levels.items()
            if level != "L0"
            for entry in entries
            if entry.generator is not None and entry.generator.get("type") == "procedural"
        ]
        violations = []
        for source in config.get("sources", []):
            label = source.get("robot", source.get("name", "?"))
            # Only procedural sources generate robots; other sources may carry a ``family``
            # label (for sampling and reports) and are checked by URDF and name below.
            families = source.get("families", source.get("family")) if source.get("type") == "procedural" else None
            if families is not None:
                families = {families} if isinstance(families, str) else set(families)
                if families & held_families:
                    violations.append(f"source {label!r} generates the held-out L3 family {sorted(families & held_families)}")
                    continue
                hit = next((f"{level}/{entry.name}" for entry, level in procedural if _may_generate(source, families, entry)), None)
                if hit is not None:
                    violations.append(f"source {label!r} can generate held-out robot {hit} (restrict its ranges)")
                    continue
            urdf = source.get("urdf")
            if urdf is not None:
                resolved = Path(urdf) if Path(urdf).is_absolute() or base_dir is None else base_dir / urdf
                hit = urdfs.get(resolved.resolve())
                if hit is not None:
                    violations.append(f"source {label!r} uses the URDF of held-out robot {hit}")
                    continue
            hit = next(((name, level) for name, level in names if name_matches(name, str(label))), None)
            if hit is not None:
                violations.append(f"source {label!r} is held-out robot {hit[1]}/{hit[0]}")
        return violations


def _may_generate(source: Mapping[str, Any], families: set[str], entry: RobotEntry) -> bool:
    """Return whether a procedural source's ranges include a procedural entry's robot.

    The entry is reachable when the families agree and, for every list-valued range of
    the entry, the source's candidates overlap it. A range the source leaves out takes
    the generator's defaults, which are assumed to overlap (conservative).
    """
    if entry.generator.get("family") not in families:
        return False
    source_ranges = source.get("ranges") or {}
    for key, values in (entry.generator.get("ranges") or {}).items():
        if not isinstance(values, (list, tuple)):
            continue
        allowed = source_ranges.get(key)
        if isinstance(allowed, (list, tuple)) and not set(allowed) & set(values):
            return False
    return True


def assign_splits(entries: Sequence[RobotEntry], fraction: float = HELDOUT_FRACTION, seed: int = 0) -> list[RobotEntry]:
    """Set ``split`` of every entry of one level with ``make_heldout_split``.

    Args:
        entries (Sequence[RobotEntry]): Robots of one level.
        fraction (float): Fraction assigned to ``test``.
        seed (int): Shuffle seed.

    Returns:
        list[RobotEntry]: The same entries, updated in place.
    """
    assignment = make_heldout_split(entries, fraction, seed)
    for entry in entries:
        entry.split = assignment[entry.name]
    return list(entries)
