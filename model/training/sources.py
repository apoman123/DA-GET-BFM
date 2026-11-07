"""Trajectory sources: domain-randomized simulation and Hugging Face datasets.

Two source types are supported in the data config (see ``model.training.dataset``):

``"synthetic"``
    Simulates a URDF robot with the analytic dynamics of ``model.dynamics``.
    Each randomized embodiment perturbs link masses and inertias, link lengths,
    friction and damping, and the torque and speed limits; each episode tilts
    gravity. A PD controller with gravity feed-forward tracks random targets;
    commands are clipped to the torque-speed envelope before they are applied,
    so labels use the *executed* torque, never the unexecuted proposal. Probe
    torques at each state give several action-conditioned samples per pose.

``"huggingface"``
    Loads recorded trajectories from the Hugging Face Hub or a local folder
    (LeRobot parquet layout or any table with per-frame vector columns). A
    joint name list maps state columns to URDF joints; joints the dataset does
    not record are frozen. Velocities and accelerations are finite-differenced
    when not recorded, and the applied torque is the inverse-dynamics estimate
    when the dataset has no effort column.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.dynamics import forward_dynamics, gravity_efforts
from model.embodiment import EmbodimentGraph, JointFeatureConfig
from model.physics import ActuatorDefaults, actuator_specs, actuator_tensors, torque_limit
from model.training.episodes import EmbodimentRecord, Trajectory
from model.urdf import UrdfInertial, UrdfRobot, parse_urdf

DEFAULT_RANDOMIZATION = {
    "mass": [0.8, 1.2],
    "length": [0.9, 1.1],
    "damping": [0.0, 0.05],
    "friction": [0.0, 0.05],
    "torque": [0.8, 1.2],
    "velocity": [0.8, 1.2],
    "gravity_tilt_degrees": 20.0,
}


def resolve_path(path: str, base: Path | None) -> Path:
    """Resolve a config path relative to the config file."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate


def _graph_kwargs(source: dict[str, Any], robot: UrdfRobot) -> dict[str, Any]:
    """Collect EmbodimentGraph.from_urdf options from a source config."""
    frozen = dict(source.get("frozen_joints", {}))
    if source.get("joint_names") and source.get("freeze_unlisted_joints", True):
        listed = set(source["joint_names"])
        for name, joint in robot.joints.items():
            if joint.joint_type in ("revolute", "continuous", "prismatic") and name not in listed:
                frozen.setdefault(name, 0.0)
    return {
        "joint_order": source.get("joint_names") or source.get("joint_order"),
        "include_mimic_joints": source.get("include_mimic_joints", False),
        "multi_dof_joints": source.get("multi_dof_joints", "skip"),
        "frozen_joints": frozen,
    }


def _actuator_defaults(source: dict[str, Any], mode: str) -> ActuatorDefaults:
    values = dict(source.get("actuator_defaults", {}))
    values.setdefault("mode", mode)
    return ActuatorDefaults(**values)


def randomize_robot(robot: UrdfRobot, randomization: dict[str, Any], generator: torch.Generator) -> UrdfRobot:
    """Return a copy of a robot with perturbed inertia, lengths, and friction.

    Args:
        robot (UrdfRobot): Nominal robot.
        randomization (dict): Ranges ``mass``, ``length`` (scale factors) and
            ``damping``, ``friction`` (added values).
        generator (torch.Generator): Random source.

    Returns:
        UrdfRobot: Perturbed robot.
    """

    def uniform(bounds) -> float:
        low, high = bounds
        return float(low + (high - low) * torch.rand((), generator=generator, dtype=torch.float64))

    length = uniform(randomization.get("length", [1.0, 1.0]))
    links = {}
    for name, link in robot.links.items():
        inertial = link.inertial
        if inertial is not None:
            scale = uniform(randomization.get("mass", [1.0, 1.0]))
            origin = inertial.origin.clone()
            origin[:3, 3] *= length
            inertial = UrdfInertial(
                mass=inertial.mass * scale,
                origin=origin,
                inertia=inertial.inertia * scale * length**2,
            )
        links[name] = replace(link, inertial=inertial)
    joints = {}
    for name, joint in robot.joints.items():
        origin = joint.origin.clone()
        origin[:3, 3] *= length
        joints[name] = replace(
            joint,
            origin=origin,
            damping=joint.damping + uniform(randomization.get("damping", [0.0, 0.0])),
            friction=joint.friction + uniform(randomization.get("friction", [0.0, 0.0])),
        )
    return replace(robot, links=links, joints=joints)


def _tilted_gravity(count: int, degrees: float, generator: torch.Generator) -> torch.Tensor:
    """Sample gravity vectors within a cone around -z."""
    tilt = torch.rand(count, generator=generator, dtype=torch.float64) * math.radians(degrees)
    heading = torch.rand(count, generator=generator, dtype=torch.float64) * 2 * math.pi
    direction = torch.stack(
        (tilt.sin() * heading.cos(), tilt.sin() * heading.sin(), -tilt.cos()), dim=-1
    )
    return direction * 9.80665


def synthetic_source(
    source: dict[str, Any], base: Path | None, feature_config: JointFeatureConfig
) -> tuple[list[EmbodimentRecord], list[Trajectory]]:
    """Simulate domain-randomized trajectories of one URDF robot.

    Source keys: ``urdf`` (required), ``name``, ``randomized_embodiments`` (4;
    0 keeps only the nominal robot), ``episodes_per_embodiment`` (8), ``steps``
    (200), ``dt`` (0.01 s), ``substeps`` (2), ``probe_torques`` (2),
    ``target_interval`` (25 steps), ``natural_frequency`` ([4, 10] rad/s),
    ``command_noise`` (0.05 rad of equivalent position error),
    ``probe_range`` and ``probe_minimum`` (probe torques are log-uniform between
    0.02 and 2.0 times peak torque, with random sign), ``randomization`` (see ``DEFAULT_RANDOMIZATION``), ``seed``, plus
    the graph and actuator keys shared with other sources.

    Args:
        source (dict): Source config.
        base (Path | None): Directory for relative paths.
        feature_config (JointFeatureConfig): Observation hardware features.

    Returns:
        tuple: Embodiment records and trajectories (indices local to this source).
    """
    robot = parse_urdf(resolve_path(source["urdf"], base))
    name = source.get("name", robot.name)
    generator = torch.Generator().manual_seed(int(source.get("seed", 0)))
    randomization = {**DEFAULT_RANDOMIZATION, **source.get("randomization", {})}
    variants = int(source.get("randomized_embodiments", 4))
    episodes = int(source.get("episodes_per_embodiment", 8))
    steps = int(source.get("steps", 200))
    dt = float(source.get("dt", 0.01))
    substeps = int(source.get("substeps", 2))
    probes = int(source.get("probe_torques", 2))
    interval = int(source.get("target_interval", 25))
    frequency = source.get("natural_frequency", [4.0, 10.0])
    noise = float(source.get("command_noise", 0.05))
    probe_range = float(source.get("probe_range", 2.0))
    probe_minimum = float(source.get("probe_minimum", 0.02))
    records, trajectories = [], []
    robots = [robot] if variants == 0 else [randomize_robot(robot, randomization, generator) for _ in range(variants)]
    for variant, variant_robot in enumerate(robots):
        graph = EmbodimentGraph.from_urdf(variant_robot, feature_config, **_graph_kwargs(source, robot))
        specs = actuator_specs(graph, source.get("actuator_overrides"), _actuator_defaults(source, "torque"))
        if variants:
            torque_scale = torch.empty(graph.num_joints, dtype=torch.float64).uniform_(
                *randomization["torque"], generator=generator
            )
            speed_scale = torch.empty(graph.num_joints, dtype=torch.float64).uniform_(
                *randomization["velocity"], generator=generator
            )
            specs = tuple(
                spec
                if spec.mode == "passive"
                else replace(
                    spec,
                    peak_torque=spec.peak_torque * float(torque_scale[index]),
                    continuous_torque=spec.continuous_torque * float(torque_scale[index]),
                    max_velocity=spec.max_velocity * float(speed_scale[index]),
                    max_power=spec.max_power * float(torque_scale[index] * speed_scale[index]),
                )
                for index, spec in enumerate(specs)
            )
        record = EmbodimentRecord.build(f"{name}/{variant}", graph, specs, source.get("end_effectors"))
        records.append(record)
        trajectories += _simulate(
            len(records) - 1, record, episodes, steps, dt, substeps, probes, interval,
            frequency, noise, probe_range, probe_minimum,
            float(randomization["gravity_tilt_degrees"]), generator,
        )
    return records, trajectories


def _simulate(
    embodiment: int,
    record: EmbodimentRecord,
    episodes: int,
    steps: int,
    dt: float,
    substeps: int,
    probes: int,
    interval: int,
    frequency: list[float],
    noise: float,
    probe_range: float,
    probe_minimum: float,
    tilt_degrees: float,
    generator: torch.Generator,
) -> list[Trajectory]:
    """Simulate a batch of episodes of one embodiment with a PD controller."""
    graph = record.graph
    joints = graph.num_joints
    actuator = actuator_tensors(record.specs)
    peak = actuator["peak_torque"]
    lower, upper = graph.lower, graph.upper
    limited = torch.tensor([kind != "continuous" for kind in graph.joint_types]) & (upper > lower)
    center = torch.where(limited, (lower + upper) / 2, torch.zeros_like(lower))
    half = torch.where(limited, (upper - lower) / 2, torch.full_like(lower, math.pi))

    def sample_targets() -> torch.Tensor:
        unit = torch.rand((episodes, joints), generator=generator, dtype=torch.float64) * 2 - 1
        return center + 0.85 * half * unit

    gravity = _tilted_gravity(episodes, tilt_degrees, generator)
    q = sample_targets()
    qd = torch.zeros_like(q)
    omega = torch.empty((episodes, 1), dtype=torch.float64).uniform_(*frequency, generator=generator)
    zeta = torch.empty((episodes, 1), dtype=torch.float64).uniform_(0.6, 1.0, generator=generator)
    target = sample_targets()
    record_q, record_qd, record_qdd, record_tau, record_cmd = [], [], [], [], []
    record_probe, record_valid = [], []
    for step in range(steps):
        if step and step % interval == 0:
            target = sample_targets()
        error = target - q
        if not bool(limited.all()):
            error = torch.where(limited, error, torch.remainder(error + math.pi, 2 * math.pi) - math.pi)
        diagonal = graph.mass_matrix(q).diagonal(dim1=-2, dim2=-1)
        # Exploration noise is a position error of ``noise`` rad (scaled by the
        # joint's own inertia), so light distal joints are not overdriven.
        jitter = noise * torch.randn(q.shape, generator=generator, dtype=torch.float64)
        command = gravity_efforts(graph, q, gravity) + diagonal * (
            omega**2 * (error + jitter) - 2 * zeta * omega * qd
        )
        limit = torque_limit(actuator, qd)
        applied = torch.maximum(torch.minimum(command, limit), -limit)
        acceleration = forward_dynamics(graph, q, qd, applied, gravity)
        # Probe torques span small to over-limit magnitudes: a random sign and a
        # log-uniform fraction in [probe_minimum, probe_range] of peak torque.
        fraction = torch.exp(
            torch.empty((episodes, probes, joints), dtype=torch.float64).uniform_(
                math.log(probe_minimum), math.log(probe_range), generator=generator
            )
        )
        sign = torch.randint(0, 2, fraction.shape, generator=generator).to(torch.float64) * 2 - 1
        record_probe.append(sign * fraction * peak)
        record_q.append(q)
        record_qd.append(qd)
        record_qdd.append(acceleration)
        record_tau.append(applied)
        record_cmd.append(command)
        valid = torch.ones(episodes, dtype=torch.bool)
        sub_dt = dt / substeps
        for substep in range(substeps):
            if substep:
                acceleration = forward_dynamics(graph, q, qd, applied, gravity)
            qd = qd + sub_dt * acceleration
            q = q + sub_dt * qd
            outside = limited & ((q < lower) | (q > upper))
            if bool(outside.any()):
                valid &= ~outside.any(dim=1)
                q = torch.where(limited, torch.clamp(q, lower, upper), q)
                qd = torch.where(outside, torch.zeros_like(qd), qd)
        record_valid.append(valid)
    stack = lambda values: torch.stack(values, dim=1)  # noqa: E731 - [E, T, ...]
    positions, velocities, accelerations = stack(record_q), stack(record_qd), stack(record_qdd)
    torques, commands, probes_all = stack(record_tau), stack(record_cmd), stack(record_probe)
    valid_all = stack(record_valid)
    trajectories = []
    for episode in range(episodes):
        trajectories.append(
            Trajectory(
                embodiment=embodiment,
                dt=dt,
                positions=positions[episode],
                velocities=velocities[episode],
                accelerations=accelerations[episode],
                applied_torque=torques[episode],
                actions=commands[episode],
                action_type="torque",
                gravity=gravity[episode],
                probe_torques=probes_all[episode] if probes else None,
                transition_valid=valid_all[episode],
            )
        )
    return trajectories


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth ``[T, J]`` values along time with edge padding."""
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack(
        [np.convolve(padded[:, joint], kernel, mode="valid") for joint in range(values.shape[1])],
        axis=1,
    )[: values.shape[0]]


def _column(table: dict[str, np.ndarray], key: str | None, indices: list[int] | None) -> np.ndarray | None:
    """Read a vector column and select indices."""
    if key is None:
        return None
    if key not in table:
        raise KeyError(f"dataset has no column {key!r}; columns are {sorted(table)}")
    values = np.stack([np.asarray(row, dtype=np.float64).reshape(-1) for row in table[key]])
    return values if indices is None else values[:, indices]


def huggingface_source(
    source: dict[str, Any], base: Path | None, feature_config: JointFeatureConfig
) -> tuple[list[EmbodimentRecord], list[Trajectory]]:
    """Load recorded trajectories from the Hugging Face Hub or a local folder.

    Source keys:

    * ``repo_id`` (Hub dataset) or ``path`` (local folder or file);
      ``data_files`` (default ``data/*/*.parquet``, the LeRobot layout),
      ``split`` ("train"), ``revision``.
    * ``urdf`` and ``joint_names``: URDF joint for each state entry, in order.
      Joints not listed are frozen at ``frozen_joints`` positions (default 0)
      unless ``freeze_unlisted_joints`` is false.
    * ``state_key`` ("observation.state") and ``state_indices``; optional
      ``velocity_key``/``velocity_indices``, ``torque_key``/``torque_indices``,
      ``action_key``/``action_indices`` with ``action_type`` ("position").
    * ``state_scale`` and ``state_offset``: per-joint ``q_urdf = scale * q +
      offset`` to fix sign or zero conventions; applied to actions too.
    * ``fps`` (frames per second), ``episode_key`` ("episode_index"),
      ``frame_key`` ("frame_index"), ``max_episodes``, ``min_episode_length``
      (10), ``gravity`` (3-vector in the root frame), ``smoothing_window``
      (odd frame count of a moving average applied to positions before finite
      differencing; 1 disables it).
    * Actuator keys: ``actuator_overrides`` and ``actuator_defaults`` (set
      ``kp``/``kd`` for position-controlled robots so requested torques are
      meaningful).

    Args:
        source (dict): Source config.
        base (Path | None): Directory for relative paths.
        feature_config (JointFeatureConfig): Observation hardware features.

    Returns:
        tuple: One embodiment record and its trajectories.
    """
    from datasets import load_dataset

    robot = parse_urdf(resolve_path(source["urdf"], base))
    graph = EmbodimentGraph.from_urdf(robot, feature_config, **_graph_kwargs(source, robot))
    action_type = source.get("action_type", "position")
    specs = actuator_specs(graph, source.get("actuator_overrides"), _actuator_defaults(source, action_type))
    record = EmbodimentRecord.build(source.get("name", robot.name), graph, specs, source.get("end_effectors"))
    if "repo_id" in source:
        dataset = load_dataset(
            source["repo_id"],
            data_files=source.get("data_files", "data/*/*.parquet"),
            split=source.get("split", "train"),
            revision=source.get("revision"),
        )
    else:
        path = resolve_path(source["path"], base)
        files = source.get("data_files") or (
            str(path / "data" / "*" / "*.parquet") if path.is_dir() else str(path)
        )
        if isinstance(files, str) and path.is_dir() and not Path(files).is_absolute():
            files = str(path / files)
        dataset = load_dataset("parquet", data_files=files, split="train")
    episode_key = source.get("episode_key", "episode_index")
    frame_key = source.get("frame_key", "frame_index")
    state_key = source.get("state_key", "observation.state")
    keys = [
        key
        for key in (
            episode_key,
            frame_key,
            state_key,
            source.get("velocity_key"),
            source.get("torque_key"),
            source.get("action_key"),
        )
        if key
    ]
    dataset = dataset.select_columns(sorted(set(keys)))
    episode_ids = np.asarray(dataset[episode_key]).reshape(-1)
    unique = np.unique(episode_ids)
    if source.get("max_episodes"):
        unique = unique[: int(source["max_episodes"])]
        rows = np.nonzero(np.isin(episode_ids, unique))[0]
        dataset = dataset.select(rows.tolist())
        episode_ids = episode_ids[rows]
    table = {key: dataset[key] for key in dataset.column_names}
    joint_count = graph.num_joints
    state_indices = source.get("state_indices") or list(range(joint_count))
    if len(state_indices) != joint_count:
        raise ValueError(f"{record.name}: state_indices must list one entry per joint")
    states = _column(table, state_key, state_indices)
    velocities = _column(
        table, source.get("velocity_key"), source.get("velocity_indices", state_indices)
    )
    torques = _column(table, source.get("torque_key"), source.get("torque_indices", state_indices))
    actions = _column(table, source.get("action_key"), source.get("action_indices", state_indices))
    scale = np.asarray(source.get("state_scale", [1.0] * joint_count), dtype=np.float64)
    offset = np.asarray(source.get("state_offset", [0.0] * joint_count), dtype=np.float64)
    states = states * scale + offset
    if velocities is not None:
        velocities = velocities * scale
    if actions is not None and action_type == "position":
        actions = actions * scale + offset
    elif actions is not None:
        actions = actions * scale
    if torques is not None:
        torques = torques * scale
    frames = np.asarray(table[frame_key]).reshape(-1) if frame_key in table else np.arange(len(episode_ids))
    gravity = torch.tensor(source["gravity"], dtype=torch.float64) if "gravity" in source else None
    dt = 1.0 / float(source["fps"])
    minimum = int(source.get("min_episode_length", 10))
    window = int(source.get("smoothing_window", 1))
    trajectories = []
    for episode in unique:
        rows = np.nonzero(episode_ids == episode)[0]
        rows = rows[np.argsort(frames[rows], kind="stable")]
        if len(rows) < minimum:
            continue

        def pick(values, smooth: bool = False):
            if values is None:
                return None
            selected = values[rows]
            if smooth and window > 1:
                selected = _moving_average(selected, window)
            return torch.from_numpy(selected)

        trajectories.append(
            Trajectory(
                embodiment=0,
                dt=dt,
                positions=pick(states, smooth=True),
                velocities=pick(velocities),
                applied_torque=pick(torques),
                actions=pick(actions),
                action_type=action_type,
                gravity=gravity,
            )
        )
    if not trajectories:
        raise ValueError(f"{record.name}: no episodes with at least {minimum} frames")
    return [record], trajectories


SOURCE_TYPES = {"synthetic": synthetic_source, "huggingface": huggingface_source}
