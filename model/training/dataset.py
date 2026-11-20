"""Datasets and collator for physics pretraining of the GET encoder.

A JSON data config lists trajectory sources and the observation layout::

    {
      "history_size": 3,
      "joint_features": "da_get_zero",
      "eval_fraction": 0.05,
      "seed": 0,
      "measured_torque_weight": 0.0,
      "max_contact_residual": 0.5,
      "contact_margin_steps": 2,
      "max_contact_slip": 0.1,
      "contact_detection": "dynamics",
      "min_contact_confidence": 3.0,
      "proposal_variants": 2,
      "response_range": [0.02, 1.0],
      "sources": [
        {"type": "synthetic", "urdf": "../assets/urdf/go1.urdf", "name": "go1"},
        {"type": "huggingface", "repo_id": "...", "urdf": "...", "joint_names": [...], "fps": 30}
      ]
    }

Options of the zero-shot proposal (``docs/zero_shot.md``), all off by default:

* ``control_dt`` (A2): one control period for every source. Simulated sources run
  at it (their ``dt``), recorded ones are resampled (``model.training.resampling``).
* ``target_normalization`` (A1): ``"data_rms"`` (default, per-record RMS from the
  data) or ``"physics"``: scales computed from the inputs only, ``tau^peak_i /
  M_ii(q)`` for joint accelerations, ``g`` and ``g / L`` for the base, Froude units
  for the action effect.
* ``dimensionless_inputs`` (A3): base angular velocity and base linear velocity / L
  in the tokens are multiplied by ``sqrt(L / g)``.
* ``spec_noise`` (E2): ranges of multiplicative noise on the specification given to
  the network (link mass and inertia, peak and continuous torque, rotor inertia);
  the labels stay those of the simulated robot, and ``sysid_*`` labels give
  ``log(true / given)`` for the deviation head (E1).
* ``group_variants`` (D2): a sample is a state with all its action variants, so the
  encoder runs once per state; per-variant tasks read it through ``variant_rows``.
* ``mirror`` (per source, D3): add left-right mirrored copies of every trajectory
  (``model.training.symmetry``).

``build_physics_datasets`` loads every source, computes features and labels
once per frame (``model.training.episodes``), caches the result on disk under a
hash of the config and the URDF files, and splits episodes into train and
evaluation sets. ``PhysicsDataCollator`` pads samples of different robots into
one batch and builds the GET-Zero observation tokens:

* variable local ``o_vl`` per joint and step (5): normalized position, velocity
  over the speed limit, and the previous executed action features (zero when the
  dataset has no commands and no measured torque);
* variable global ``o_vg`` per step (11): gravity direction in the base frame,
  base angular velocity, base linear velocity over the robot size, a flag for
  a known base twist, and a floating-base flag;
* fixed local ``o_fl``: the GET-Zero hardware features of each joint.

Torque labels come only from simulators (weight 1) and, if enabled, from torque
sensors (``measured_torque_weight``). Nothing is estimated. Every frame, of every
dataset, also trains the response tasks R and R^-1 on a torque change drawn by the
collator, whose labels ``G(q) delta_tau`` are exact for the model.

References:
    * Observation tokens ``o_vl``, ``o_vg``, ``o_fl`` with an ``H``-step history:
      GET-Zero, Patel & Song, ICRA 2025, pp. 14262-14269, Sec. III, DOI
      10.1109/ICRA55743.2025.11127922, arXiv:2407.15002.
    * Froude units (``dimensionless_inputs``, physics normalization of the effect):
      dynamic similarity, Alexander & Jayes, J. Zool. 201(1):135-152, 1983, DOI
      10.1111/j.1469-7998.1983.tb04266.x.
    * ``spec_noise`` randomizes the given specification in the manner of dynamics
      randomization (Peng, Andrychowicz, Zaremba & Abbeel, ICRA 2018, DOI
      10.1109/ICRA.2018.8460528); the ``sysid_*`` deviation labels resemble the
      extrinsics that RMA estimates from history (Kumar, Fu, Pathak & Malik, RSS 2021,
      DOI 10.15607/RSS.2021.XVII.011).
    * ``STANDARD_GRAVITY`` (9.80665 m/s^2) is the CODATA standard acceleration of gravity.
    * Datasets: ``docs/data.md#references``; the config keys: ``docs/training.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from model.centroidal import CENTROIDAL_COLUMNS, CENTROIDAL_GROUPS, CENTROIDAL_SIZE
from model.dynamics import STANDARD_GRAVITY
from model.embodiment import JointFeatureConfig, collate_embodiments
from model.physics import EPSILON, froude_scale_base_features
from model.physics_schema import feature_names
from model.tokenization_da_get import DaGetTokenizer
from model.training.episodes import (
    BASE_FEATURE_SIZE,
    TORQUE_SOURCES,
    EmbodimentRecord,
    finalize_acceleration_scales,
    process_episode,
)
from model.training.sources import SOURCE_TYPES, resolve_path

CACHE_VERSION = 7
VARIABLE_LOCAL_SIZE = 5
VARIABLE_GLOBAL_SIZE = BASE_FEATURE_SIZE
FD_TORQUE_CLIP = 5.0
TARGET_NORMALIZATIONS = ("data_rms", "physics")
SPEC_NOISE_KEYS = ("mass", "torque", "continuous", "armature")
SYSID_SIZE = len(SPEC_NOISE_KEYS)
SIMULATED_SOURCES = ("synthetic", "mujoco", "procedural")
# Default integration step of the synthetic simulator: dt 0.01 s with 2 substeps.
SYNTHETIC_INTEGRATION_STEP = 0.005
_RIGID = feature_names("rigid_body")
_ACTUATOR = feature_names("actuator")
_ROOT = feature_names("root")
RIGID_MASS_COLUMNS = [_RIGID.index("mass")] + [_RIGID.index(f"inertia_{c}") for c in ("xx", "xy", "xz", "yy", "yz", "zz")]
RIGID_ARMATURE_COLUMN = _RIGID.index("armature")
ACTUATOR_PEAK_COLUMN = _ACTUATOR.index("peak_torque")
ACTUATOR_CONTINUOUS_COLUMN = _ACTUATOR.index("continuous_torque")
ROOT_TOTAL_MASS_COLUMN = _ROOT.index("total_mass")
ROOT_BASE_MASS_COLUMN = _ROOT.index("root_mass")
# Keys of the earlier, estimate-based pipeline, with what replaces them.
REMOVED_KEYS = {
    "allow_estimated_torque_for_fd": "torque estimates are no longer used; FD trains on exact and measured torques",
    "use_estimated_previous_torque": "the dynamics features no longer contain a previous torque",
    "estimated_torque_weight": "torque estimates are no longer used; see measured_torque_weight",
    "max_estimate_residual": "renamed to max_contact_residual",
}


def check_data_config(config: dict[str, Any]) -> None:
    """Reject keys of the estimate-based pipeline, which would otherwise be ignored silently.

    Args:
        config (dict[str, Any]): Parsed data config.

    Raises:
        ValueError: The config uses a removed key or an unknown normalization.
    """
    removed = {key: reason for key, reason in REMOVED_KEYS.items() if key in config}
    if removed:
        details = "; ".join(f"{key}: {reason}" for key, reason in removed.items())
        raise ValueError(f"data config uses removed keys ({details})")
    normalization = config.get("target_normalization", "data_rms")
    if normalization not in TARGET_NORMALIZATIONS:
        raise ValueError(f"target_normalization must be one of {TARGET_NORMALIZATIONS}, not {normalization!r}")


def joint_feature_config(value: str | dict | None) -> JointFeatureConfig:
    """Parse the ``joint_features`` entry of a data config.

    Args:
        value (str | dict | None): ``"da_get_zero"``, ``"cross_embodiment"``, or
            ``JointFeatureConfig`` fields.

    Returns:
        JointFeatureConfig: Hardware features of observation tokens.
    """
    if value in (None, "da_get_zero"):
        return JointFeatureConfig.da_get_zero()
    if value == "cross_embodiment":
        return JointFeatureConfig.cross_embodiment()
    if isinstance(value, dict):
        return JointFeatureConfig(**value)
    raise ValueError(f"unknown joint_features {value!r}")


def load_data_config(source: str | Path | dict) -> tuple[dict[str, Any], Path | None]:
    """Read a data config and the directory its relative paths start from.

    Args:
        source (str | Path | dict): JSON path or an already parsed config.

    Returns:
        tuple: Config dict and base directory.
    """
    if isinstance(source, dict):
        return source, None
    path = Path(source)
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def _optional_float(value: Any) -> float | None:
    """Return ``float(value)``, or None for None."""
    return None if value is None else float(value)


def _cache_key(config: dict[str, Any], base: Path | None) -> str:
    """Hash the config and the contents of every URDF it references.

    Procedural sources have no URDF file; their generated URDF texts are hashed instead.
    """
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(str(CACHE_VERSION).encode())
    for source in config["sources"]:
        if "urdf" in source:
            digest.update(resolve_path(source["urdf"], base).read_bytes())
        elif source.get("type") == "procedural":
            from model.training.procedural import source_fingerprint

            # The generated URDF texts, so a change of the generator code invalidates the cache.
            digest.update(source_fingerprint(source))
    return digest.hexdigest()[:16]


def source_with_control_dt(source: dict[str, Any], control_dt: float | None) -> dict[str, Any]:
    """Return a source entry that simulates at ``control_dt`` unless it sets its own ``dt``.

    The synthetic simulator keeps its default integration step (0.005 s) by scaling
    its substeps with the period.

    Args:
        source (dict): Source entry.
        control_dt (float | None): Control period, or None to keep the source as is.

    Returns:
        dict: Possibly updated copy.
    """
    if control_dt is None or source.get("type") not in SIMULATED_SOURCES or "dt" in source:
        return source
    updated = {**source, "dt": float(control_dt)}
    if source.get("type") in ("synthetic", "procedural") and "substeps" not in source:
        updated["substeps"] = max(1, int(round(control_dt / SYNTHETIC_INTEGRATION_STEP)))
    return updated


def load_source(
    source: dict[str, Any], base: Path | None, feature_config: JointFeatureConfig, control_dt: float | None = None
) -> tuple[list[EmbodimentRecord], list]:
    """Load one source at the config's control period, with optional mirrored copies.

    Args:
        source (dict): Source entry.
        base (Path | None): Directory for relative paths.
        feature_config (JointFeatureConfig): Observation hardware features.
        control_dt (float | None): Control period (``control_dt`` of the config).

    Returns:
        tuple: Records and trajectories (indices local to this source).

    Raises:
        ValueError: The source type is unknown, or mirroring was requested for a
            robot that is not left-right symmetric.
    """
    kind = source.get("type")
    if kind not in SOURCE_TYPES:
        raise ValueError(f"unknown source type {kind!r}; use one of {sorted(SOURCE_TYPES)}")
    source = source_with_control_dt(source, control_dt)
    records, trajectories = SOURCE_TYPES[kind](source, base, feature_config)
    if control_dt is not None:
        from model.training.resampling import resample_trajectory

        trajectories = [resample_trajectory(trajectory, control_dt) for trajectory in trajectories]
    if source.get("mirror"):
        from model.training.symmetry import mirror_map, mirror_trajectory

        maps = {}
        mirrored = []
        for trajectory in trajectories:
            index = trajectory.embodiment
            if index not in maps:
                maps[index] = mirror_map(records[index].graph)
                if maps[index] is None:
                    raise ValueError(f"{records[index].name}: mirror requested but the robot is not left-right symmetric")
            mirrored.append(mirror_trajectory(records[index], trajectory, maps[index]))
        trajectories = trajectories + mirrored
    return records, trajectories


@dataclass
class PhysicsDatasets:
    """Train and evaluation datasets that share embodiment records.

    Attributes:
        train (PhysicsTrajectoryDataset): Training samples.
        eval (PhysicsTrajectoryDataset | None): Held-out episodes.
        records (list[EmbodimentRecord]): Embodiments.
        history_size (int): Observation history length.
        feature_config (JointFeatureConfig): Observation hardware features.
        settings (CollatorSettings): Label weights and response sampling.
        sampling (dict): ``sampling_alpha`` and ``sampling_group`` of the config.
    """

    train: PhysicsTrajectoryDataset
    eval: PhysicsTrajectoryDataset | None
    records: list[EmbodimentRecord]
    history_size: int
    feature_config: JointFeatureConfig
    settings: CollatorSettings = None
    sampling: dict = field(default_factory=dict)

    def collator(self) -> PhysicsDataCollator:
        """Return a collator for these datasets."""
        return PhysicsDataCollator(self.records, self.history_size, self.settings)

    @property
    def token_size(self) -> int:
        """Return the observation token width (the encoder input size)."""
        return self.collator().token_builder.token_size

    def token_layout(self) -> dict[str, Any]:
        """Describe the observation token layout for the model config."""
        settings = self.settings or CollatorSettings()
        return {
            "history_size": self.history_size,
            "variable_local": ["position_normalized", "velocity_normalized", "prev_action_torque", "prev_action_position", "prev_action_velocity"],
            "variable_global": [
                "gravity_direction_x", "gravity_direction_y", "gravity_direction_z",
                "base_angular_velocity_x", "base_angular_velocity_y", "base_angular_velocity_z",
                "base_linear_velocity_over_length_x", "base_linear_velocity_over_length_y",
                "base_linear_velocity_over_length_z", "base_twist_known", "floating_base",
            ],
            "fixed_local": list(self.feature_config.feature_names()),
            "dimensionless_inputs": settings.dimensionless_inputs,
            "target_normalization": settings.target_normalization,
            "control_dt": settings.control_dt,
        }


def build_physics_datasets(
    config_source: str | Path | dict,
    cache_dir: str | Path | None = None,
    rebuild: bool = False,
) -> PhysicsDatasets:
    """Load, process, cache, and split every source of a data config.

    Args:
        config_source (str | Path | dict): Data config path or dict.
        cache_dir (str | Path | None): Where processed data is cached; the
            config's ``cache_dir`` or ``.cache/physics_data`` by default.
        rebuild (bool): Ignore an existing cache.

    Returns:
        PhysicsDatasets: Datasets and records.

    Raises:
        ValueError: A source type is unknown or the config has no sources.
    """
    config, base = load_data_config(config_source)
    if not config.get("sources"):
        raise ValueError("data config needs at least one source")
    check_data_config(config)
    feature_config = joint_feature_config(config.get("joint_features"))
    history = int(config.get("history_size", 3))
    control_dt = _optional_float(config.get("control_dt"))
    cache_root = Path(cache_dir or config.get("cache_dir") or ".cache/physics_data")
    cache_file = cache_root / f"{_cache_key(config, base)}.pt"
    if cache_file.exists() and not rebuild:
        payload = torch.load(cache_file, weights_only=False)
        records, episodes = payload["records"], payload["episodes"]
    else:
        records, episodes = [], []
        for source in config["sources"]:
            source_records, trajectories = load_source(source, base, feature_config, control_dt)
            offset = len(records)
            records += source_records
            for trajectory in trajectories:
                index = offset + trajectory.embodiment
                episodes.append(
                    (
                        index,
                        process_episode(
                            records[index],
                            trajectory,
                            contact_height_tolerance=float(source.get("contact_height_tolerance", 0.03)),
                            contact_speed_tolerance=float(source.get("contact_speed_tolerance", 0.5)),
                            proposal_variants=int(config.get("proposal_variants", 2)),
                            contact_margin_steps=int(config.get("contact_margin_steps", 2)),
                            seed=int(config.get("seed", 0)) * 1_000_003 + len(episodes),
                            max_contact_slip=_optional_float(config.get("max_contact_slip", 0.1)),
                            contact_detection=str(source.get("contact_detection", config.get("contact_detection", "dynamics"))),
                            detected_contact_response=bool(source.get("detected_contact_response", True)),
                            min_contact_confidence=float(config.get("min_contact_confidence", 3.0)),
                        ),
                    )
                )
        finalize_acceleration_scales(records, episodes)
        cache_root.mkdir(parents=True, exist_ok=True)
        torch.save({"records": records, "episodes": episodes}, cache_file)
    order = list(range(len(episodes)))
    random.Random(int(config.get("seed", 0))).shuffle(order)
    eval_count = int(round(float(config.get("eval_fraction", 0.05)) * len(order)))
    if eval_count == 0 and float(config.get("eval_fraction", 0.05)) > 0 and len(order) > 1:
        eval_count = 1
    eval_ids, train_ids = order[:eval_count], order[eval_count:]
    include_probes = bool(config.get("include_probe_variants", True))
    group = bool(config.get("group_variants", False))
    train = PhysicsTrajectoryDataset(records, [episodes[i] for i in train_ids], history, include_probes, group)
    evaluation = (
        PhysicsTrajectoryDataset(records, [episodes[i] for i in eval_ids], history, include_probes, group)
        if eval_ids
        else None
    )
    sampling = {
        "sampling_alpha": _optional_float(config.get("sampling_alpha")),
        "sampling_group": str(config.get("sampling_group", "embodiment")),
    }
    return PhysicsDatasets(
        train, evaluation, records, history, feature_config, CollatorSettings.from_config(config), sampling
    )


PER_FRAME_KEYS = (
    "contact_projection",
    "dynamics_features",
    "mass_matrix",
    "fk_position",
    "fk_rotation",
    "response_operator",
)
OPTIONAL_FRAME_KEYS = ("centroidal", "centroidal_valid", "positions", "velocities")
PER_VARIANT_KEYS = (
    "applied_torque",
    "acceleration",
    "base_acceleration",
    "required_torque",
    "action_features",
    "motor_flags",
    "motor_margins",
    "motor_margin_mask",
    "effect_delta_position",
    "effect_rotation",
    "effect_linear_velocity",
    "effect_angular_velocity",
)
PER_VARIANT_FLAGS = ("torque_source", "base_acceleration_known", "effect_valid", "motor_valid")


class PhysicsTrajectoryDataset(Dataset):
    """Index every (episode, frame, action variant) of processed episodes.

    Each sample holds the observation history ending at frame t (the first frame
    is repeated before the episode starts) and the labels of one action variant
    at t. Probe variants share the state and history of the executed variant.
    With ``group_variants`` a sample is one (episode, frame) with the labels of
    all its variants stacked, so the encoder runs once per state (proposal D2).
    """

    def __init__(
        self,
        records: Sequence[EmbodimentRecord],
        episodes: Sequence[tuple[int, dict[str, torch.Tensor]]],
        history_size: int,
        include_probes: bool = True,
        group_variants: bool = False,
    ) -> None:
        """Build the sample index.

        Args:
            records (Sequence[EmbodimentRecord]): Embodiments.
            episodes (Sequence): ``(embodiment, processed episode)`` pairs.
            history_size (int): Observation history length H.
            include_probes (bool): Also index probe variants.
            group_variants (bool): One sample per state with all its variants.
        """
        self.records = list(records)
        self.episodes = list(episodes)
        self.history_size = history_size
        self.include_probes = include_probes
        self.group_variants = group_variants
        self.index: list[tuple[int, int, int]] = []
        for episode_index, (_, episode) in enumerate(self.episodes):
            steps, variants = episode["applied_torque"].shape[:2]
            for step in range(steps):
                if group_variants:
                    self.index.append((episode_index, step, -1))
                    continue
                for variant in range(variants if include_probes else 1):
                    self.index.append((episode_index, step, variant))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.index)

    def embodiment_of(self, item: int) -> int:
        """Return the embodiment index of a sample (for balanced sampling)."""
        return self.episodes[self.index[item][0]][0]

    def _contact_labels(self, record: EmbodimentRecord, episode: dict[str, torch.Tensor], step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-joint contact targets and masks ``[J]`` from per-link flags."""
        joints = record.graph.num_joints
        target = torch.zeros(joints)
        mask = torch.zeros(joints, dtype=torch.bool)
        flags = episode.get("contact_flags")
        if flags is None or flags.shape[-1] == 0 or "contact_label_valid" not in episode:
            return target, mask
        if not bool(episode["contact_label_valid"][step]):
            return target, mask
        owners = record.contact_joints
        for link, owner in enumerate(owners.tolist()):
            if owner < 0:
                continue
            mask[owner] = True
            if bool(flags[step, link]):
                target[owner] = 1.0
        return target, mask

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return one unpadded sample."""
        episode_index, step, variant = self.index[item]
        embodiment, episode = self.episodes[episode_index]
        frames = torch.arange(step - self.history_size + 1, step + 1).clamp_min(0)
        previous = (frames - 1).clamp_min(0)
        previous_action = episode["action_features"][previous, 0]
        previous_action = torch.where((frames > 0)[:, None, None], previous_action, torch.zeros_like(previous_action))
        sample = {
            "embodiment": embodiment,
            "variable_local": torch.cat(
                (
                    episode["positions_normalized"][frames, :, None],
                    episode["velocities_normalized"][frames, :, None],
                    previous_action,
                ),
                dim=-1,
            ),
            "variable_global": episode["base_features"][frames],
            "contact_residual": float(episode["contact_residual"][step]),
            "contacts_detected": bool(episode["contacts_detected"][step]),
            "response_valid": bool(episode["response_valid"][step]),
            "floating_base": bool(episode["floating_base"][step]),
            "dt": float(episode["dt"]) if "dt" in episode else 0.0,
            "grouped": variant < 0,
        }
        for key in PER_FRAME_KEYS:
            sample[key] = episode[key][step]
        for key in OPTIONAL_FRAME_KEYS:
            if key in episode:
                sample[key] = episode[key][step]
        sample["contact_target"], sample["contact_mask"] = self._contact_labels(self.records[embodiment], episode, step)
        if variant < 0:
            count = episode["applied_torque"].shape[1] if self.include_probes else 1
            for key in PER_VARIANT_KEYS:
                sample[key] = episode[key][step, :count]
            for key in PER_VARIANT_FLAGS:
                sample[key] = episode[key][step, :count].clone()
        else:
            for key in PER_VARIANT_KEYS:
                sample[key] = episode[key][step, variant]
            for key in PER_VARIANT_FLAGS:
                sample[key] = episode[key][step, variant]
        return sample


def _pad(values: Sequence[torch.Tensor], joint_count: int, dims: int = 1, fill: float = 0.0) -> torch.Tensor:
    """Stack tensors whose first ``dims`` dimensions are joints, padding to joint_count."""
    first = values[0]
    shape = [len(values)] + [joint_count] * dims + list(first.shape[dims:])
    result = torch.full(shape, fill, dtype=first.dtype)
    for row, value in enumerate(values):
        result[(row,) + tuple(slice(0, size) for size in value.shape[:dims])] = value
    return result


@dataclass
class CollatorSettings:
    """Label weights, response sampling and the zero-shot options of ``PhysicsDataCollator``.

    Attributes:
        measured_torque_weight (float): Loss weight of FD and ID labels from torque
            sensors. 0 (default) keeps them out of training: real torque sensors
            (Unitree's estimated output torque ``tauEst``, for example) are themselves
            approximate. Simulated torques always have weight 1.
        max_contact_residual (float): Drop response labels of frames whose detected
            contacts leave more than this fraction of the robot's weight in the base
            wrench unexplained (a missed or false contact). Contacts from sensors or
            simulators are trusted.
        response_range (tuple[float, float]): Torque changes of the response tasks
            are log-uniform in this range, as fractions of each joint's peak torque,
            with a random sign per joint.
        target_normalization (str): ``data_rms`` or ``physics`` (proposal A1).
        dimensionless_inputs (bool): Froude-scale the base velocities in the tokens
            (proposal A3).
        spec_noise (dict[str, tuple[float, float]] | None): Uniform ranges of the
            multiplicative noise on the given specification, keys in
            ``SPEC_NOISE_KEYS`` (proposal E2).
        include_graphs (bool): Add the padded ``EmbodimentBatch`` as ``structure``
            (needed by the differentiable-rollout effect head).
        control_dt (float | None): The config's control period (for the layout).
    """

    measured_torque_weight: float = 0.0
    max_contact_residual: float = 0.5
    response_range: tuple[float, float] = (0.02, 1.0)
    target_normalization: str = "data_rms"
    dimensionless_inputs: bool = False
    spec_noise: dict[str, tuple[float, float]] | None = None
    include_graphs: bool = False
    control_dt: float | None = None

    def __post_init__(self) -> None:
        """Validate values."""
        low, high = self.response_range
        if not 0 < low <= high:
            raise ValueError("response_range must satisfy 0 < low <= high")
        if self.measured_torque_weight < 0:
            raise ValueError("measured_torque_weight must be nonnegative")
        if self.target_normalization not in TARGET_NORMALIZATIONS:
            raise ValueError(f"target_normalization must be one of {TARGET_NORMALIZATIONS}")
        for key, bounds in (self.spec_noise or {}).items():
            if key not in SPEC_NOISE_KEYS:
                raise ValueError(f"unknown spec_noise key {key!r}; use {SPEC_NOISE_KEYS}")
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"spec_noise[{key!r}] must satisfy 0 < low <= high")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CollatorSettings:
        """Read the settings from a data config."""
        noise = config.get("spec_noise")
        return cls(
            float(config.get("measured_torque_weight", 0.0)),
            float(config.get("max_contact_residual", 0.5)),
            tuple(float(value) for value in config.get("response_range", (0.02, 1.0))),
            str(config.get("target_normalization", "data_rms")),
            bool(config.get("dimensionless_inputs", False)),
            None if not noise else {key: (float(value[0]), float(value[1])) for key, value in noise.items()},
            bool(config.get("include_graphs", False)),
            _optional_float(config.get("control_dt")),
        )


class PhysicsDataCollator:
    """Pad samples of different robots into one batch for ``DaGetForPhysicsPretraining``.

    The output holds encoder inputs (tokens, graph structure, mass matrix,
    physics features) and, for each task, normalized targets and masks. The
    response tasks draw a new torque change for every sample from the global
    PyTorch generator (seed it for reproducible batches); so does the spec noise.

    Per-variant tasks (FD, ID, motor, effect) have one row per action variant;
    ``variant_rows`` gives the state row of each. Without grouped samples it is
    ``arange(batch)``.
    """

    def __init__(
        self,
        records: Sequence[EmbodimentRecord],
        history_size: int,
        settings: CollatorSettings | None = None,
    ) -> None:
        """Precompute per-embodiment scales.

        Args:
            records (Sequence[EmbodimentRecord]): Embodiments.
            history_size (int): Observation history length.
            settings (CollatorSettings | None): Label weights and response
                sampling; defaults if None.
        """
        self.records = list(records)
        self.history_size = history_size
        self.settings = settings or CollatorSettings()
        feature_size = self.records[0].joint_features.shape[1]
        self.token_builder = DaGetTokenizer(
            fixed_local_size=feature_size,
            variable_local_size=VARIABLE_LOCAL_SIZE,
            variable_global_size=VARIABLE_GLOBAL_SIZE,
            history_size=history_size,
        )
        self._scales = []
        for record in self.records:
            actuator = record.actuator
            actuated = record.actuated
            ones = torch.ones(record.graph.num_joints, dtype=torch.float64)
            response = record.response_scale if record.response_scale is not None else torch.ones(6 + len(ones))
            peak = torch.where(actuated, actuator["peak_torque"], ones)
            # Torque reference of passive joints for physics normalization: the robot's
            # median actuated peak torque.
            reference = float(actuator["peak_torque"][actuated].median()) if bool(actuated.any()) else 1.0
            acceleration = record.acceleration_scale if record.acceleration_scale is not None else ones
            self._scales.append(
                {
                    "actuated": actuated,
                    "peak": peak.float(),
                    "torque_reference": torch.where(actuated, actuator["peak_torque"], torch.full_like(ones, reference)).float(),
                    "continuous": torch.where(actuated, actuator["continuous_torque"], ones).float(),
                    "acceleration": acceleration.float(),
                    "response": response[6:].float(),
                    "response_base": response[:6].float(),
                    "effect": record.effect_mask,
                }
            )

    def _torque_changes(self, actuated: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
        """Draw ``[B, J]`` torque changes: random sign, log-uniform fraction of peak, zero on passive joints."""
        low, high = self.settings.response_range
        fraction = torch.exp(torch.empty(peak.shape).uniform_(math.log(low), math.log(high)))
        sign = torch.randint(0, 2, peak.shape).float() * 2 - 1
        return torch.where(actuated, sign * fraction * peak, torch.zeros_like(peak))

    @staticmethod
    def _variants(samples: Sequence[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
        """Flatten per-variant entries of (possibly grouped) samples into rows."""
        rows, entries = [], []
        for index, sample in enumerate(samples):
            if sample.get("grouped"):
                count = sample["applied_torque"].shape[0]
                for variant in range(count):
                    entry = {key: sample[key][variant] for key in PER_VARIANT_KEYS}
                    entry.update({key: sample[key][variant].item() for key in PER_VARIANT_FLAGS})
                    rows.append(index)
                    entries.append(entry)
            else:
                entry = {key: sample[key] for key in PER_VARIANT_KEYS}
                entry.update({key: sample[key] if not torch.is_tensor(sample[key]) else sample[key].item() for key in PER_VARIANT_FLAGS})
                rows.append(index)
                entries.append(entry)
        return rows, entries

    def _spec_noise(
        self, rigid: torch.Tensor, actuator: torch.Tensor, dynamics: torch.Tensor, root: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perturb the given specification and return ``log(true / given)`` labels (E1, E2).

        Dynamics features keep the true model's values except those that depend on
        the perturbed rotor inertia (added to ``M_ii``) and continuous torque (the
        gravity load); link-mass noise does not change them, which is an [H]
        simplification (``docs/zero_shot.md``).
        """
        noise = self.settings.spec_noise or {}
        batch, joints = mask.shape
        factors = torch.ones((batch, joints, SYSID_SIZE))
        for index, key in enumerate(SPEC_NOISE_KEYS):
            if key in noise:
                low, high = noise[key]
                factors[..., index] = torch.empty((batch, joints)).uniform_(low, high)
        rigid, actuator, dynamics, root = rigid.clone(), actuator.clone(), dynamics.clone(), root.clone()
        true_mass = rigid[..., RIGID_MASS_COLUMNS[0]].clone()
        true_armature = rigid[..., RIGID_ARMATURE_COLUMN].clone()
        rigid[..., RIGID_MASS_COLUMNS] = rigid[..., RIGID_MASS_COLUMNS] * factors[..., :1]
        rigid[..., RIGID_ARMATURE_COLUMN] = true_armature * factors[..., 3]
        actuator[..., ACTUATOR_PEAK_COLUMN] = actuator[..., ACTUATOR_PEAK_COLUMN] * factors[..., 1]
        actuator[..., ACTUATOR_CONTINUOUS_COLUMN] = actuator[..., ACTUATOR_CONTINUOUS_COLUMN] * factors[..., 2]
        dynamics[..., 0] = dynamics[..., 0] + true_armature * (factors[..., 3] - 1)
        given_continuous = actuator[..., ACTUATOR_CONTINUOUS_COLUMN]
        load = dynamics[..., 2].abs() / (given_continuous + EPSILON)
        dynamics[..., 4] = torch.where(given_continuous > 0, load, torch.zeros_like(load))
        given_mass = (rigid[..., RIGID_MASS_COLUMNS[0]] * mask).sum(-1)
        root[:, ROOT_TOTAL_MASS_COLUMN] = root[:, ROOT_BASE_MASS_COLUMN] + given_mass
        labels = -factors.log()
        valid = mask[..., None].expand(-1, -1, SYSID_SIZE).clone()
        valid[..., 0] &= true_mass > 0
        valid[..., 1] &= actuator[..., ACTUATOR_PEAK_COLUMN] > 0
        valid[..., 2] &= actuator[..., ACTUATOR_CONTINUOUS_COLUMN] > 0
        valid[..., 3] &= true_armature > 0
        for index, key in enumerate(SPEC_NOISE_KEYS):
            if key not in noise:
                valid[..., index] = False
        return rigid, actuator, dynamics, root, labels, valid

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate samples.

        Args:
            samples (Sequence[dict]): Dataset items.

        Returns:
            dict[str, torch.Tensor]: Batch for ``DaGetForPhysicsPretraining.forward``.
        """
        settings = self.settings
        embodiments = [sample["embodiment"] for sample in samples]
        records = [self.records[index] for index in embodiments]
        graphs = [record.graph for record in records]
        structure = collate_embodiments(graphs)
        mask = structure.joint_mask
        joints = mask.shape[1]
        batch = len(samples)
        scales = [self._scales[index] for index in embodiments]
        physics_scaling = settings.target_normalization == "physics"
        lengths = torch.tensor([graph.characteristic_length for graph in graphs], dtype=torch.float32)
        masses = torch.tensor([graph.total_mass for graph in graphs], dtype=torch.float32)
        dts = torch.tensor([float(sample.get("dt", 0.0)) for sample in samples], dtype=torch.float32)

        def stack(key: str, dims: int = 1, fill: float = 0.0) -> torch.Tensor:
            return _pad([sample[key] for sample in samples], joints, dims, fill)

        def per_record(key: str, fill: float) -> torch.Tensor:
            return _pad([scale[key] for scale in scales], joints, 1, fill)

        def flags(key: str) -> torch.Tensor:
            return torch.tensor([sample[key] for sample in samples])

        variable_local = _pad(
            [sample["variable_local"].transpose(0, 1) for sample in samples], joints
        ).transpose(1, 2)
        variable_global = torch.stack([sample["variable_global"] for sample in samples]).float()
        if settings.dimensionless_inputs:
            variable_global = froude_scale_base_features(variable_global, lengths[:, None])
        tokens = self.token_builder(
            structure.joint_features,
            mask,
            variable_local=variable_local.float(),
            variable_global=variable_global,
        )
        actuated = per_record("actuated", False) & mask
        peak = per_record("peak", 1.0)
        continuous = per_record("continuous", 1.0)
        mass_matrix = stack("mass_matrix", dims=2).float()
        floating = flags("floating_base")
        gravity = STANDARD_GRAVITY
        if physics_scaling:
            # Scales from the inputs only (proposal A1): tau^peak / M_ii, g, g / L, Froude units.
            diagonal = mass_matrix.diagonal(dim1=1, dim2=2).clamp_min(1e-9)
            acceleration_scale = torch.where(mask, per_record("torque_reference", 1.0) / diagonal, torch.ones_like(diagonal))
            base_scale = torch.cat(
                (torch.full((batch, 3), gravity), (gravity / lengths)[:, None].expand(-1, 3)), dim=-1
            )
            response_scale, response_base_scale = acceleration_scale, base_scale
            frequency = torch.sqrt(gravity / lengths)
            effect_scales = torch.stack((dts.clamp_min(1e-6) * frequency, frequency, frequency), dim=-1)
        else:
            acceleration_scale = per_record("acceleration", 1.0)
            base_scale = torch.stack([record.base_acceleration_scale for record in records]).float()
            response_scale = per_record("response", 1.0)
            response_base_scale = torch.stack([scale["response_base"] for scale in scales])
            effect_scales = torch.stack([record.effect_scales for record in records]).float()
        spatial, ancestors, descendants = structure.graph_distances
        rigid = _pad([record.rigid_body_features for record in records], joints)
        actuator_features = _pad([record.actuator_features for record in records], joints)
        dynamics = stack("dynamics_features").float()
        root = torch.stack([record.root_features for record in records]).float()
        sysid = None
        if settings.spec_noise:
            rigid, actuator_features, dynamics, root, sysid_target, sysid_mask = self._spec_noise(
                rigid, actuator_features, dynamics, root, mask
            )
            sysid = (sysid_target, sysid_mask)

        # Per-variant rows (FD, ID, motor, effect).
        rows, entries = self._variants(samples)
        index = torch.tensor(rows, dtype=torch.long)

        def variant_stack(key: str, fill: float = 0.0) -> torch.Tensor:
            return _pad([entry[key] for entry in entries], joints, 1, fill)

        def variant_flags(key: str) -> torch.Tensor:
            return torch.tensor([entry[key] for entry in entries])

        source = variant_flags("torque_source")
        label_weight = torch.where(
            source == TORQUE_SOURCES.index("exact"),
            1.0,
            torch.where(source == TORQUE_SOURCES.index("measured"), settings.measured_torque_weight, 0.0),
        )
        labeled = label_weight > 0
        effect_valid = variant_flags("effect_valid")
        base_known = variant_flags("base_acceleration_known")
        base_acceleration = torch.stack([entry["base_acceleration"] for entry in entries]).float() / base_scale[index]
        base_acceleration = torch.where(base_known[:, None], base_acceleration, torch.zeros_like(base_acceleration))
        applied = variant_stack("applied_torque").float()
        acceleration = variant_stack("acceleration").float()
        row_actuated, row_peak, row_continuous = actuated[index], peak[index], continuous[index]
        row_acceleration_scale = acceleration_scale[index]
        effect_scale_rows = effect_scales[index]

        # Response tasks: a fresh torque change per state, and its exact effect G dtau.
        weight = torch.tensor([graph.total_mass * 9.80665 for graph in graphs])  # standard gravity (CODATA)
        residual = flags("contact_residual")
        trusted = ~flags("contacts_detected") | (residual <= settings.max_contact_residual * weight)
        response_rows = flags("response_valid") & trusted
        change = self._torque_changes(actuated, peak)
        operators = [sample["response_operator"].float() for sample in samples]
        operator = torch.zeros((batch, 6 + joints, joints))
        response_joint = torch.zeros((batch, joints))
        response_base = torch.zeros((batch, 6))
        for row, (sample_operator, sample_change) in enumerate(zip(operators, change)):
            count = sample_operator.shape[-1]
            operator[row, :6, :count] = sample_operator[:6]
            operator[row, 6 : 6 + count, :count] = sample_operator[6:]
            delta = sample_operator @ sample_change[:count]
            response_base[row] = delta[:6]
            response_joint[row, :count] = delta[6:]
        response_target = response_joint / response_scale
        response_base_target = response_base / response_base_scale
        response_base_rows = response_rows & floating
        inverse_input = torch.cat(
            (
                response_target[..., None],
                torch.where(response_base_rows[:, None], response_base_target, torch.zeros_like(response_base_target))[:, None, :].expand(-1, joints, -1),
            ),
            dim=-1,
        )
        contact_projection = stack("contact_projection", dims=2).float()
        batch_out = {
            "input_values": tokens,
            "parent_indices": structure.parent_indices,
            "joint_mask": mask,
            "adjacency_matrix": structure.adjacency_matrix,
            "spatial_distances": spatial,
            "ancestor_distances": ancestors,
            "descendant_distances": descendants,
            "mass_matrix": mass_matrix,
            "rigid_body_features": rigid,
            "actuator_features": actuator_features,
            "dynamics_features": dynamics,
            "root_features": root,
            "total_mass": masses,
            "characteristic_length": lengths,
            "frame_dt": dts,
            "variant_rows": index,
            # A. forward kinematics
            "fk_position": stack("fk_position").float(),
            "fk_rotation": stack("fk_rotation", fill=0.0).float(),
            # B. forward dynamics (exact or measured torques only); one row per variant
            "fd_torque": (applied / (row_peak + EPSILON)).clamp(-FD_TORQUE_CLIP, FD_TORQUE_CLIP),
            "fd_target": acceleration / row_acceleration_scale,
            "fd_mask": row_actuated & labeled[:, None],
            "fd_base_target": base_acceleration,
            "fd_base_mask": base_known & labeled,
            "acceleration_scale": row_acceleration_scale,
            "base_acceleration_scale": base_scale[index],
            # C. inverse dynamics (same rows), projected onto the motion-determined part
            "id_acceleration": acceleration / row_acceleration_scale,
            "id_base_acceleration": base_acceleration,
            "id_target": variant_stack("required_torque").float() / (row_continuous + EPSILON),
            # The projection needs the right contact set: rows with doubtful detected contacts are dropped.
            "id_mask": row_actuated & (labeled & trusted[index])[:, None],
            "contact_projection": contact_projection[index],
            "torque_label_weight": label_weight,
            "torque_scale": row_continuous,
            "variant_peak_torque": row_peak,
            # R. contact-constrained response to a torque change (every dataset); one row per state
            "response_torque": change / peak,
            "response_target": response_target,
            "response_mask": mask & response_rows[:, None],
            "response_base_target": response_base_target,
            "response_base_mask": response_base_rows,
            "response_scale": response_scale,
            "response_base_scale": response_base_scale,
            "response_operator": operator,
            "response_projection": contact_projection,
            # R^-1. torque change from its response, projected like ID
            "inverse_response_input": inverse_input,
            "inverse_response_target": change / peak,
            "inverse_response_mask": actuated & response_rows[:, None],
            "response_torque_scale": peak,
            # D. motor feasibility
            "motor_input": variant_stack("action_features").float(),
            "motor_flags": variant_stack("motor_flags").float(),
            "motor_margins": variant_stack("motor_margins").float(),
            "motor_margin_mask": variant_stack("motor_margin_mask", fill=False),
            "motor_mask": row_actuated & variant_flags("motor_valid")[:, None],
            # E. action effect
            "effect_input": variant_stack("action_features").float(),
            "effect_delta_position": variant_stack("effect_delta_position").float() / effect_scale_rows[:, None, None, 0],
            "effect_rotation": variant_stack("effect_rotation").float(),
            "effect_linear_velocity": variant_stack("effect_linear_velocity").float() / effect_scale_rows[:, None, None, 1],
            "effect_angular_velocity": variant_stack("effect_angular_velocity").float() / effect_scale_rows[:, None, None, 2],
            "effect_mask": per_record("effect", False)[index] & mask[index] & effect_valid[:, None],
            "effect_scales": effect_scale_rows,
            # C2. contact probability (exact or confidently inferred contacts)
            "contact_target": stack("contact_target").float(),
            "contact_mask": stack("contact_mask", fill=False) & mask,
        }
        if all("centroidal" in sample for sample in samples):
            # C3. centroidal quantities, dimensionless
            groups = torch.stack([sample["centroidal_valid"] for sample in samples])
            centroidal_mask = torch.zeros((batch, CENTROIDAL_SIZE), dtype=torch.bool)
            for group_index, group in enumerate(CENTROIDAL_GROUPS):
                centroidal_mask[:, list(CENTROIDAL_COLUMNS[group])] = groups[:, group_index : group_index + 1]
            batch_out["centroidal_target"] = torch.stack([sample["centroidal"] for sample in samples]).float()
            batch_out["centroidal_mask"] = centroidal_mask
        if all("positions" in sample for sample in samples):
            batch_out["joint_positions"] = stack("positions").float()
            batch_out["joint_velocities"] = stack("velocities").float()
            # Base twist at t in SI units (angular, then linear), zero when unknown.
            current = torch.stack([sample["variable_global"][-1] for sample in samples]).float()
            known = current[:, 9:10] > 0.5
            twist = torch.cat((current[:, 3:6], current[:, 6:9] * lengths[:, None]), dim=-1)
            batch_out["base_twist"] = torch.where(known, twist, torch.zeros_like(twist))
        if sysid is not None:
            batch_out["sysid_target"], batch_out["sysid_mask"] = sysid
        if settings.include_graphs:
            batch_out["structure"] = structure
        return batch_out
