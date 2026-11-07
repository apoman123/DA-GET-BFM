"""Datasets and collator for physics pretraining of the GET encoder.

A JSON data config lists trajectory sources and the observation layout::

    {
      "history_size": 3,
      "joint_features": "get_zero",
      "eval_fraction": 0.05,
      "seed": 0,
      "allow_estimated_torque_for_fd": true,
      "use_estimated_previous_torque": true,
      "max_estimate_residual": 0.5,
      "estimated_torque_weight": 0.3,
      "sources": [
        {"type": "synthetic", "urdf": "../assets/urdf/go1.urdf", "name": "go1"},
        {"type": "huggingface", "repo_id": "...", "urdf": "...", "joint_names": [...], "fps": 30}
      ]
    }

``build_physics_datasets`` loads every source, computes features and labels
once per frame (``model.training.episodes``), caches the result on disk under a
hash of the config and the URDF files, and splits episodes into train and
evaluation sets. ``PhysicsDataCollator`` pads samples of different robots into
one batch and builds the GET-Zero observation tokens:

* variable local ``o_vl`` per joint and step (5): normalized position, velocity
  over the speed limit, and the previous executed action features;
* variable global ``o_vg`` per step (11): gravity direction in the base frame,
  base angular velocity, base linear velocity over the robot size, a flag for
  a known base twist, and a floating-base flag;
* fixed local ``o_fl``: the GET-Zero hardware features of each joint.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from model.embodiment import JointFeatureConfig, collate_embodiments
from model.physics import EPSILON
from model.tokenization_get import GetTokenizer
from model.training.episodes import (
    BASE_FEATURE_SIZE,
    EmbodimentRecord,
    finalize_acceleration_scales,
    process_episode,
)
from model.training.sources import SOURCE_TYPES, resolve_path

CACHE_VERSION = 4
VARIABLE_LOCAL_SIZE = 5
VARIABLE_GLOBAL_SIZE = BASE_FEATURE_SIZE
FD_TORQUE_CLIP = 5.0


def joint_feature_config(value: str | dict | None) -> JointFeatureConfig:
    """Parse the ``joint_features`` entry of a data config.

    Args:
        value (str | dict | None): ``"get_zero"``, ``"cross_embodiment"``, or
            ``JointFeatureConfig`` fields.

    Returns:
        JointFeatureConfig: Hardware features of observation tokens.
    """
    if value in (None, "get_zero"):
        return JointFeatureConfig.get_zero()
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


def _cache_key(config: dict[str, Any], base: Path | None) -> str:
    """Hash the config and the contents of every URDF it references."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(str(CACHE_VERSION).encode())
    for source in config["sources"]:
        digest.update(resolve_path(source["urdf"], base).read_bytes())
    return digest.hexdigest()[:16]


@dataclass
class PhysicsDatasets:
    """Train and evaluation datasets that share embodiment records.

    Attributes:
        train (PhysicsTrajectoryDataset): Training samples.
        eval (PhysicsTrajectoryDataset | None): Held-out episodes.
        records (list[EmbodimentRecord]): Embodiments.
        history_size (int): Observation history length.
        feature_config (JointFeatureConfig): Observation hardware features.
        allow_estimated_torque_for_fd (bool): Train FD on ID-estimated torques.
        max_estimate_residual (float): See ``PhysicsDataCollator``.
        estimated_torque_weight (float): See ``PhysicsDataCollator``.
    """

    train: PhysicsTrajectoryDataset
    eval: PhysicsTrajectoryDataset | None
    records: list[EmbodimentRecord]
    history_size: int
    feature_config: JointFeatureConfig
    allow_estimated_torque_for_fd: bool
    max_estimate_residual: float = 0.5
    estimated_torque_weight: float = 0.3

    def collator(self) -> PhysicsDataCollator:
        """Return a collator for these datasets."""
        return PhysicsDataCollator(
            self.records,
            self.history_size,
            self.allow_estimated_torque_for_fd,
            self.max_estimate_residual,
            self.estimated_torque_weight,
        )

    @property
    def token_size(self) -> int:
        """Return the observation token width (the encoder input size)."""
        return self.collator().token_builder.token_size

    def token_layout(self) -> dict[str, Any]:
        """Describe the observation token layout for the model config."""
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
    feature_config = joint_feature_config(config.get("joint_features"))
    history = int(config.get("history_size", 3))
    cache_root = Path(cache_dir or config.get("cache_dir") or ".cache/physics_data")
    cache_file = cache_root / f"{_cache_key(config, base)}.pt"
    if cache_file.exists() and not rebuild:
        payload = torch.load(cache_file, weights_only=False)
        records, episodes = payload["records"], payload["episodes"]
    else:
        records, episodes = [], []
        for source in config["sources"]:
            kind = source.get("type")
            if kind not in SOURCE_TYPES:
                raise ValueError(f"unknown source type {kind!r}; use one of {sorted(SOURCE_TYPES)}")
            source_records, trajectories = SOURCE_TYPES[kind](source, base, feature_config)
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
                            use_estimated_previous_torque=bool(
                                config.get("use_estimated_previous_torque", True)
                            ),
                            contact_height_tolerance=float(source.get("contact_height_tolerance", 0.03)),
                            contact_speed_tolerance=float(source.get("contact_speed_tolerance", 0.5)),
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
    train = PhysicsTrajectoryDataset(records, [episodes[i] for i in train_ids], history, include_probes)
    evaluation = (
        PhysicsTrajectoryDataset(records, [episodes[i] for i in eval_ids], history, include_probes)
        if eval_ids
        else None
    )
    return PhysicsDatasets(
        train,
        evaluation,
        records,
        history,
        feature_config,
        bool(config.get("allow_estimated_torque_for_fd", True)),
        float(config.get("max_estimate_residual", 0.5)),
        float(config.get("estimated_torque_weight", 0.3)),
    )


PER_FRAME_KEYS = (
    "id_projection",
    "dynamics_features",
    "mass_matrix",
    "coriolis",
    "gravity_torque",
    "friction_torque",
    "fk_position",
    "fk_rotation",
)
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


class PhysicsTrajectoryDataset(Dataset):
    """Index every (episode, frame, action variant) of processed episodes.

    Each sample holds the observation history ending at frame t (the first frame
    is repeated before the episode starts) and the labels of one action variant
    at t. Probe variants share the state and history of the executed variant.
    """

    def __init__(
        self,
        records: Sequence[EmbodimentRecord],
        episodes: Sequence[tuple[int, dict[str, torch.Tensor]]],
        history_size: int,
        include_probes: bool = True,
    ) -> None:
        """Build the sample index.

        Args:
            records (Sequence[EmbodimentRecord]): Embodiments.
            episodes (Sequence): ``(embodiment, processed episode)`` pairs.
            history_size (int): Observation history length H.
            include_probes (bool): Also index probe variants.
        """
        self.records = list(records)
        self.episodes = list(episodes)
        self.history_size = history_size
        self.index: list[tuple[int, int, int]] = []
        for episode_index, (_, episode) in enumerate(self.episodes):
            steps, variants = episode["applied_torque"].shape[:2]
            for step in range(steps):
                for variant in range(variants if include_probes else 1):
                    self.index.append((episode_index, step, variant))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.index)

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
            "torque_measured": bool(episode["torque_measured"][step]),
            "estimate_residual": float(episode["estimate_residual"][step]),
            "base_acceleration_known": bool(episode["base_acceleration_known"][step, variant]),
            "floating_base": bool(episode["floating_base"][step]),
            "effect_valid": bool(episode["effect_valid"][step, variant]),
        }
        for key in PER_FRAME_KEYS:
            sample[key] = episode[key][step]
        for key in PER_VARIANT_KEYS:
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


class PhysicsDataCollator:
    """Pad samples of different robots into one batch for ``GetForPhysicsPretraining``.

    The output holds encoder inputs (tokens, graph structure, mass matrix,
    physics features) and, for each task, normalized targets and masks.
    """

    def __init__(
        self,
        records: Sequence[EmbodimentRecord],
        history_size: int,
        allow_estimated_torque_for_fd: bool = True,
        max_estimate_residual: float = 0.5,
        estimated_torque_weight: float = 0.3,
    ) -> None:
        """Precompute per-embodiment scales.

        Args:
            records (Sequence[EmbodimentRecord]): Embodiments.
            history_size (int): Observation history length.
            allow_estimated_torque_for_fd (bool): Keep FD labels of samples whose
                torque is an inverse-dynamics estimate.
            max_estimate_residual (float): Drop FD and ID labels of estimated
                frames whose unexplained base wrench exceeds this fraction of
                the robot's weight.
            estimated_torque_weight (float): Loss weight of FD and ID labels
                that rest on estimated torques (measured and simulated
                torques have weight 1).
        """
        self.records = list(records)
        self.history_size = history_size
        self.allow_estimated_torque_for_fd = allow_estimated_torque_for_fd
        self.max_estimate_residual = max_estimate_residual
        self.estimated_torque_weight = estimated_torque_weight
        feature_size = self.records[0].joint_features.shape[1]
        self.token_builder = GetTokenizer(
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
            self._scales.append(
                {
                    "actuated": actuated,
                    "peak": torch.where(actuated, actuator["peak_torque"], ones).float(),
                    "continuous": torch.where(actuated, actuator["continuous_torque"], ones).float(),
                    "acceleration": record.acceleration_scale.float(),
                    "effect": record.effect_mask,
                }
            )

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate samples.

        Args:
            samples (Sequence[dict]): Dataset items.

        Returns:
            dict[str, torch.Tensor]: Batch for ``GetForPhysicsPretraining.forward``.
        """
        embodiments = [sample["embodiment"] for sample in samples]
        graphs = [self.records[index].graph for index in embodiments]
        structure = collate_embodiments(graphs)
        mask = structure.joint_mask
        joints = mask.shape[1]
        scales = [self._scales[index] for index in embodiments]
        effect_scales = torch.stack([self.records[index].effect_scales for index in embodiments])
        effect_position_scale = effect_scales[:, None, None, 0]
        effect_linear_scale = effect_scales[:, None, None, 1]
        effect_angular_scale = effect_scales[:, None, None, 2]

        def stack(key: str, dims: int = 1, fill: float = 0.0) -> torch.Tensor:
            return _pad([sample[key] for sample in samples], joints, dims, fill)

        def per_record(key: str, fill: float) -> torch.Tensor:
            return _pad([scale[key] for scale in scales], joints, 1, fill)

        variable_local = _pad(
            [sample["variable_local"].transpose(0, 1) for sample in samples], joints
        ).transpose(1, 2)
        variable_global = torch.stack([sample["variable_global"] for sample in samples]).float()
        tokens = self.token_builder(
            structure.joint_features,
            mask,
            variable_local=variable_local.float(),
            variable_global=variable_global,
        )
        actuated = per_record("actuated", False) & mask
        peak = per_record("peak", 1.0)
        continuous = per_record("continuous", 1.0)
        acceleration_scale = per_record("acceleration", 1.0)
        measured = torch.tensor([sample["torque_measured"] for sample in samples])
        effect_valid = torch.tensor([sample["effect_valid"] for sample in samples])
        # Inverse-dynamics labels are unreliable where the contacts could not
        # explain the base wrench (for example a missed contact).
        weight = torch.tensor([self.records[index].graph.total_mass * 9.80665 for index in embodiments])
        residual = torch.tensor([sample["estimate_residual"] for sample in samples])
        reliable = measured | (residual <= self.max_estimate_residual * weight)
        fd_rows = (measured | self.allow_estimated_torque_for_fd) & reliable
        base_scale = torch.stack([self.records[index].base_acceleration_scale for index in embodiments])
        base_acceleration = torch.stack([sample["base_acceleration"] for sample in samples]).float() / base_scale
        base_known = torch.tensor([sample["base_acceleration_known"] for sample in samples])
        base_acceleration = torch.where(base_known[:, None], base_acceleration, torch.zeros_like(base_acceleration))
        label_weight = torch.where(measured, 1.0, self.estimated_torque_weight)
        floating = torch.tensor([sample["floating_base"] for sample in samples])
        applied = stack("applied_torque").float()
        acceleration = stack("acceleration").float()
        spatial, ancestors, descendants = structure.graph_distances
        mass_matrix = stack("mass_matrix", dims=2).float()
        return {
            "input_values": tokens,
            "parent_indices": structure.parent_indices,
            "joint_mask": mask,
            "adjacency_matrix": structure.adjacency_matrix,
            "spatial_distances": spatial,
            "ancestor_distances": ancestors,
            "descendant_distances": descendants,
            "mass_matrix": mass_matrix,
            "rigid_body_features": _pad([self.records[i].rigid_body_features for i in embodiments], joints),
            "actuator_features": _pad([self.records[i].actuator_features for i in embodiments], joints),
            "dynamics_features": stack("dynamics_features").float(),
            # A. forward kinematics
            "fk_position": stack("fk_position").float(),
            "fk_rotation": stack("fk_rotation", fill=0.0).float(),
            # B. forward dynamics
            "fd_torque": (applied / (peak + EPSILON)).clamp(-FD_TORQUE_CLIP, FD_TORQUE_CLIP),
            "fd_target": acceleration / acceleration_scale,
            "fd_mask": actuated & fd_rows[:, None],
            "fd_base_target": base_acceleration,
            "fd_base_mask": base_known & fd_rows,
            # C. inverse dynamics
            "id_acceleration": acceleration / acceleration_scale,
            "id_base_acceleration": base_acceleration,
            "id_target": stack("required_torque").float() / (continuous + EPSILON),
            "id_mask": actuated & reliable[:, None],
            "id_projection": stack("id_projection", dims=2).float(),
            "torque_label_weight": label_weight,
            # D. motor feasibility
            "motor_input": stack("action_features").float(),
            "motor_flags": stack("motor_flags").float(),
            "motor_margins": stack("motor_margins").float(),
            "motor_margin_mask": stack("motor_margin_mask", fill=False),
            "motor_mask": actuated,
            # E. action effect
            "effect_input": stack("action_features").float(),
            "effect_delta_position": stack("effect_delta_position").float() / effect_position_scale,
            "effect_rotation": stack("effect_rotation").float(),
            "effect_linear_velocity": stack("effect_linear_velocity").float() / effect_linear_scale,
            "effect_angular_velocity": stack("effect_angular_velocity").float() / effect_angular_scale,
            "effect_mask": per_record("effect", False) & mask & effect_valid[:, None],
            # F. rigid-body consistency
            "coriolis": stack("coriolis").float(),
            "gravity_torque": stack("gravity_torque").float(),
            "friction_torque": stack("friction_torque").float(),
            "applied_torque": applied,
            "acceleration_scale": acceleration_scale,
            "torque_scale": continuous,
            "rbd_mask": actuated & fd_rows[:, None] & ~floating[:, None],
        }
