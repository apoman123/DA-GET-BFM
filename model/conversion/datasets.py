"""Converters from public robot datasets to the training table format.

Each ``convert_*`` function downloads (or reads) the source data, converts it to
``model.conversion.common`` episodes, writes them, and returns a data-config
source that the training pipeline can use directly. Conventions were checked
against the actual files on 2026-11-07:

* **MIPO-Go1** (``Open-IO/mipo-go1-odometry``, HDF5, AGPL-3.0): 400 Hz joints,
  measured torques, body IMU (orientation x, y, z, w; gyro; accelerometer
  with gravity), raw foot-force counts. Joint ``FL0, FL1, FL2`` are hip,
  thigh, calf.
* **GrandTour** (``leggedrobotics/grand_tour_dataset``, zarr in tar): 400 Hz
  ANYmal D state estimator with measured joint efforts, pose (x, y, z, w
  quaternion in ``odom``), twist in the base frame, and per-foot contact state
  (0 open, 1 contact, 2 slipping).
* **NeuroBEM** (UZH, CSV inside ``processed_data.zip``): 400 Hz Vicon state,
  body-frame twist and accelerations, and four motor speeds. No geometry or
  propeller coefficients are published, so they are identified from the data
  and a URDF is generated.
* **LocoMuJoCo** (``robfiras/loco-mujoco-datasets``, npz, CC BY-NC-ND 4.0):
  MuJoCo ``qpos``/``qvel`` with a free root (position, w-x-y-z quaternion,
  world linear velocity, body angular velocity) and joint names.
* **AMASS retargeted to G1** (``ember-lab-berkeley/AMASS_Retargeted_for_G1``,
  npz): named joint positions; body 0 is the pelvis with w-x-y-z quaternions.
  Stored joint velocities do not match the positions, so velocities are left
  to finite differencing.
* **LAFAN1 retargeted to Unitree** (``lvhaidong/LAFAN1_Retargeting_Dataset``,
  CSV, 30 Hz): root x, y, z and an x, y, z, w quaternion, then joints in the
  order of the dataset README.
"""

from __future__ import annotations

import fnmatch
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from model.conversion.common import (
    Episode,
    HttpRangeFile,
    chunk_episode,
    quaternion_wxyz_to_xyzw,
    resample,
    resample_quaternion,
    rotation_from_xyzw,
    source_config,
    split_segments,
    to_base_frame,
    uniform_time,
    write_dataset,
)

GRAVITY = 9.80665


def _hub_files(repo: str, prefix: str = "") -> list[str]:
    """List files of a Hub dataset under a prefix."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(name for name in files if name.startswith(prefix))


def _hub_download(repo: str, name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, name, repo_type="dataset"))


def _select(names: list[str], patterns: list[str] | None, limit: int | None) -> list[str]:
    if patterns:
        names = [name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]
    return names[:limit] if limit else names


# --------------------------------------------------------------------------- MIPO

MIPO_REPOSITORY = "Open-IO/mipo-go1-odometry"
MIPO_JOINTS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
GO1_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def convert_mipo(
    output: str | Path,
    patterns: list[str] | None = None,
    max_files: int | None = None,
    rate: float = 200.0,
    contact_threshold: float = 200.0,
    urdf: str = "assets/urdf/go1.urdf",
    files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Convert MIPO Unitree Go1 recordings (measured torques, IMU, foot forces).

    Args:
        output (str | Path): Output folder.
        patterns (list[str] | None): Globs over Hub paths, e.g. ``["indoor/*trot*"]``.
        max_files (int | None): Limit on sequences.
        rate (float): Output rate in Hz (data are 400 Hz).
        contact_threshold (float): Foot-force count above which a foot is in
            contact.
        urdf (str): URDF path written into the source.
        files (list | None): Local ``.h5`` files instead of downloading.

    Returns:
        dict: Data-config source.
    """
    import h5py

    if files is None:
        names = _select([n for n in _hub_files(MIPO_REPOSITORY) if n.endswith(".h5")], patterns, max_files)
        files = [_hub_download(MIPO_REPOSITORY, name) for name in names]
    episodes = []
    for path in files:
        with h5py.File(path) as handle:
            joints = handle["unitree_hardware/joint_foot"]
            imu = handle["unitree_hardware/imu"]
            time = joints["time"][:]
            raw = {
                "state": joints["position"][:, :12],
                "velocity": joints["velocity"][:, :12],
                "effort": joints["effort"][:, :12],
                "force": joints["effort"][:, 12:16],
            }
            imu_time = imu["time"][:]
            orientation = imu["orientation"][:]
            gyro = imu["angular_velocity"][:]
            accelerometer = imu["linear_acceleration"][:]
        order = np.argsort(time)
        time = time[order]
        for start, stop in split_segments(time, 0.1, 2.0):
            grid = uniform_time(time[start], time[stop - 1], rate)
            segment = time[start:stop]
            signals = {key: resample(segment, value[order][start:stop], grid) for key, value in raw.items()}
            force = signals.pop("force")
            signals["contact"] = (force > contact_threshold).astype(np.float64)
            signals["base_angular_velocity"] = resample(imu_time, gyro, grid)
            signals["imu_acceleration"] = resample(imu_time, accelerometer, grid)
            if np.abs(orientation).sum() > 0:
                signals["base_orientation"] = resample_quaternion(imu_time, orientation, grid)
            else:
                specific = signals["imu_acceleration"]
                signals["projected_gravity"] = -specific / np.linalg.norm(specific, axis=1, keepdims=True)
            episodes.append(Episode(signals, {"file": Path(path).name}))
    if not episodes:
        raise ValueError("no MIPO sequences selected")
    keys = set.intersection(*(set(episode.signals) for episode in episodes))
    episodes = [Episode({k: v for k, v in e.signals.items() if k in keys}, e.meta) for e in episodes]
    info = write_dataset(
        output, episodes, rate, MIPO_JOINTS, GO1_FEET,
        {"source": f"https://huggingface.co/datasets/{MIPO_REPOSITORY}", "license": "AGPL-3.0",
         "robot": "Unitree Go1"},
    )
    return source_config(output, info, urdf, "mipo_go1", True, GO1_FEET, {"smoothing_window": 5})


# ------------------------------------------------------------------------ GrandTour

GRANDTOUR_REPOSITORY = "leggedrobotics/grand_tour_dataset"
ANYMAL_JOINTS = [f"{leg}_{joint}" for leg in ("LF", "RF", "LH", "RH") for joint in ("HAA", "HFE", "KFE")]
ANYMAL_FEET = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]


def convert_grandtour(
    output: str | Path,
    missions: list[str] | None = None,
    max_missions: int | None = 1,
    rate: float = 200.0,
    episode_seconds: float = 20.0,
    urdf: str = "assets/urdf/anymal_d.urdf",
    archives: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Convert GrandTour ANYmal D missions (measured torques, contacts, base state).

    Only ``data/anymal_state_state_estimator.tar`` (about 100 MB per mission)
    is downloaded.

    Args:
        output (str | Path): Output folder.
        missions (list[str] | None): Mission folder names or globs; all when None.
        max_missions (int | None): Limit on missions.
        rate (float): Output rate in Hz (data are 400 Hz).
        episode_seconds (float): Long missions are cut into episodes this long.
        urdf (str): URDF path written into the source.
        archives (list | None): Local state-estimator tar files instead.

    Returns:
        dict: Data-config source.
    """
    from model.conversion.zarr_lite import ZarrGroup

    if archives is None:
        names = sorted({name.split("/")[0] for name in _hub_files(GRANDTOUR_REPOSITORY) if name.endswith("anymal_state_state_estimator.tar")})
        names = _select(names, missions, max_missions)
        archives = [_hub_download(GRANDTOUR_REPOSITORY, f"{name}/data/anymal_state_state_estimator.tar") for name in names]
    episodes = []
    for archive in archives:
        group = ZarrGroup(archive)
        time = group["timestamp"]
        raw = {
            "state": group["joint_positions"],
            "velocity": group["joint_velocities"],
            "effort": group["joint_efforts"],
            "base_position": group["pose_pos"],
            "base_linear_velocity": group["twist_lin"],
            "base_angular_velocity": group["twist_ang"],
            "contact": np.stack([(group[f"{foot}_state"] >= 1).astype(np.float64) for foot in ANYMAL_FEET], 1),
            "contact_force": np.stack([np.linalg.norm(group[f"{foot}_wrench_force"], axis=1) for foot in ANYMAL_FEET], 1),
        }
        orientation = group["pose_orien"]
        for start, stop in split_segments(time, 0.05, 2.0):
            grid = uniform_time(time[start], time[stop - 1], rate)
            segment = time[start:stop]
            signals = {key: resample(segment, value[start:stop], grid) for key, value in raw.items()}
            signals["contact"] = (signals["contact"] > 0.5).astype(np.float64)
            signals["base_orientation"] = resample_quaternion(segment, orientation[start:stop], grid)
            episodes += chunk_episode(Episode(signals, {"archive": Path(archive).parent.parent.name}), int(episode_seconds * rate))
    info = write_dataset(
        output, episodes, rate, ANYMAL_JOINTS, ANYMAL_FEET,
        {"source": f"https://huggingface.co/datasets/{GRANDTOUR_REPOSITORY}",
         "license": "see the dataset card (MIT) and website (CC BY-SA 4.0)", "robot": "ANYmal D"},
    )
    return source_config(output, info, urdf, "grandtour_anymal_d", True, ANYMAL_FEET)


# ------------------------------------------------------------------------ NeuroBEM

NEUROBEM_URL = "https://download.ifi.uzh.ch/rpg/NeuroBEM/processed_data.zip"
NEUROBEM_MASS = 0.772
NEUROBEM_INERTIA = (0.0025, 0.0021, 0.0043)
NEUROBEM_ROTORS = ("back_right", "front_right", "back_left", "front_left")  # mot 1..4
ROTOR_SIGNS = {"back_right": (-1, -1), "front_right": (1, -1), "back_left": (-1, 1), "front_left": (1, 1)}


def _neurobem_tables(source: str | Path | None, patterns: list[str] | None, limit: int | None) -> list[tuple[str, np.ndarray]]:
    """Read NeuroBEM CSV tables from the remote zip, a local zip, or a folder."""
    if source is not None and Path(source).is_dir():
        paths = sorted(Path(source).rglob("*.csv"))
        names = _select([str(path) for path in paths], patterns, limit)
        return [(name, np.loadtxt(name, delimiter=",", skiprows=1)) for name in names]
    handle = HttpRangeFile(NEUROBEM_URL) if source is None else open(source, "rb")
    with zipfile.ZipFile(handle) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        tables = []
        for name in _select(members, patterns, limit):
            with archive.open(name) as member:
                tables.append((name, np.loadtxt(io.TextIOWrapper(member), delimiter=",", skiprows=1)))
    return tables


def identify_quadrotor(
    data: list[np.ndarray], mass: float, inertia: tuple[float, float, float], specific_force: np.ndarray
) -> dict[str, Any]:
    """Identify thrust and drag coefficients and the rotor layout from flight data.

    Model: thrust ``k_f w_i^2`` along body z; roll and pitch moments from rotor
    positions ``(s_x a, s_y b)``; yaw moment ``sum spin_i k_m w_i^2``.

    Args:
        data (list[np.ndarray]): Per-flight arrays with columns as in NeuroBEM:
            angular acceleration (1:4), angular velocity (4:7), motor speeds
            (20:24).
        mass (float): Vehicle mass in kg.
        inertia (tuple): Diagonal inertia in kg m^2.
        specific_force (np.ndarray): Concatenated body-frame specific force.

    Returns:
        dict: ``k_f``, ``k_m``, ``arm_x``, ``arm_y``, ``spin`` (per rotor), fit
            quality ``thrust_r2``.
    """
    stack = np.concatenate(data)
    alpha, omega, speeds = stack[:, 1:4], stack[:, 4:7], stack[:, 20:24]
    squares = speeds**2
    thrust = mass * specific_force[:, 2]
    k_f = float(np.dot(squares.sum(1), thrust) / np.dot(squares.sum(1), squares.sum(1)))
    prediction = k_f * squares.sum(1)
    thrust_r2 = 1 - np.var(thrust - prediction) / np.var(thrust)
    inertia = np.diag(inertia)
    moments = alpha @ inertia + np.cross(omega, omega @ inertia)
    sx = np.array([ROTOR_SIGNS[name][0] for name in NEUROBEM_ROTORS], dtype=np.float64)
    sy = np.array([ROTOR_SIGNS[name][1] for name in NEUROBEM_ROTORS], dtype=np.float64)
    # M_x = k_f sum(y_i w_i^2) with y_i = sy_i * b; M_y = -k_f sum(x_i w_i^2) with x_i = sx_i * a.
    def fit(basis, target):
        gain = float(np.dot(basis, target) / max(np.dot(basis, basis), 1e-30))
        spread = float(np.var(target))
        return gain, float(1 - np.var(target - gain * basis) / spread) if spread > 0 else 0.0

    arm_y, roll_r2 = fit(k_f * (squares @ sy), moments[:, 0])
    arm_x, pitch_r2 = fit(-k_f * (squares @ sx), moments[:, 1])
    best = None
    for spin in ([1, -1, -1, 1], [1, 1, -1, -1], [1, -1, 1, -1]):
        k_m, yaw_r2 = fit(squares @ np.asarray(spin, dtype=np.float64), moments[:, 2])
        if best is None or yaw_r2 > best[1]:
            best = (k_m, yaw_r2, spin)
    k_m, yaw_r2, spin = best
    if k_m < 0:
        k_m, spin = -k_m, [-value for value in spin]
    return {
        "k_f": k_f, "k_m": k_m, "arm_x": abs(arm_x), "arm_y": abs(arm_y), "spin": spin,
        "thrust_r2": float(thrust_r2), "roll_r2": roll_r2, "pitch_r2": pitch_r2, "yaw_r2": yaw_r2,
    }


def quadrotor_urdf(
    name: str,
    mass: float,
    inertia: tuple[float, float, float],
    rotors: list[tuple[str, float, float]],
    rotor_mass: float = 0.01,
    rotor_inertia: float = 2e-5,
    effort: float = 0.2,
    velocity: float = 3000.0,
) -> str:
    """Write a URDF for a multirotor: one base link and continuous rotor joints about +z."""
    parts = [
        '<?xml version="1.0"?>',
        f'<robot name="{name}">',
        '  <link name="base_link">',
        f'    <inertial><origin xyz="0 0 0"/><mass value="{mass - rotor_mass * len(rotors):.6f}"/>',
        f'      <inertia ixx="{inertia[0]}" iyy="{inertia[1]}" izz="{inertia[2]}" ixy="0" ixz="0" iyz="0"/></inertial>',
        "  </link>",
    ]
    for rotor, x, y in rotors:
        parts += [
            f'  <link name="rotor_{rotor}">',
            f'    <inertial><origin xyz="0 0 0"/><mass value="{rotor_mass}"/>',
            f'      <inertia ixx="{rotor_inertia / 2}" iyy="{rotor_inertia / 2}" izz="{rotor_inertia}" ixy="0" ixz="0" iyz="0"/></inertial>',
            "  </link>",
            f'  <joint name="rotor_{rotor}_joint" type="continuous"><parent link="base_link"/><child link="rotor_{rotor}"/>',
            f'    <origin xyz="{x:.5f} {y:.5f} 0"/><axis xyz="0 0 1"/><limit effort="{effort:.5f}" velocity="{velocity:.1f}"/></joint>',
        ]
    parts.append("</robot>")
    return "\n".join(parts) + "\n"


def convert_neurobem(
    output: str | Path,
    patterns: list[str] | None = None,
    max_flights: int | None = None,
    rate: float = 200.0,
    source: str | Path | None = None,
    rotor_inertia: float = 2e-5,
    motor_time_constant: float = 0.03,
    arm_length: float = 0.13,
    drag_ratio: float = 0.016,
    min_fit_r2: float = 0.5,
) -> dict[str, Any]:
    """Convert NeuroBEM quadrotor flights and identify the vehicle's propeller model.

    The thrust coefficient is fitted from the measured specific force (the fit
    is good, R^2 about 0.96). The rotor positions and the drag coefficient are
    also fitted from the angular dynamics, but on NeuroBEM these fits are poor
    (R^2 below 0.2), so the published frame size is used instead: an Armattan
    Chameleon 6-inch frame, ``arm_length`` from the center to each motor in an X
    layout, and ``k_m = drag_ratio * k_f``. Fit qualities are saved in
    ``meta/info.json``.

    Args:
        output (str | Path): Output folder; ``neurobem_quadrotor.urdf`` is written
            there too.
        patterns (list[str] | None): Globs over CSV member names.
        max_flights (int | None): Limit on CSV files (flight segments).
        rate (float): Output rate in Hz (data are 400 Hz).
        source (str | Path | None): Local ``processed_data.zip`` or folder;
            the remote zip is read with range requests when None.
        rotor_inertia (float): Rotor inertia about its axis (not published).
        motor_time_constant (float): Sets the rotor speed gain ``kd``.
        arm_length (float): Center-to-motor distance in m used when the
            moment fits are poor.
        drag_ratio (float): ``k_m / k_f`` in m used when the yaw fit is poor.
        min_fit_r2 (float): Fits below this R^2 fall back to the defaults.

    Returns:
        dict: Data-config source with the identified actuator model.
    """
    tables = _neurobem_tables(source, patterns, max_flights)
    if not tables:
        raise ValueError("no NeuroBEM flights selected")
    # Decide whether the body acceleration column is kinematic or specific force.
    kinematic_error, specific_error = [], []
    for _, table in tables:
        time = table[:, 0]
        velocity, omega = table[:, 14:17], table[:, 4:7]
        derivative = np.gradient(velocity, time, axis=0) + np.cross(omega, velocity)
        gravity = to_base_frame(table[:, 7:11], np.tile([0.0, 0.0, -GRAVITY], (len(table), 1)))
        acceleration = table[:, 11:14]
        kinematic_error.append(np.abs(acceleration - derivative).mean())
        specific_error.append(np.abs(acceleration - (derivative - gravity)).mean())
    specific = float(np.mean(specific_error)) < float(np.mean(kinematic_error))
    forces = []
    for _, table in tables:
        gravity = to_base_frame(table[:, 7:11], np.tile([0.0, 0.0, -GRAVITY], (len(table), 1)))
        forces.append(table[:, 11:14] if specific else table[:, 11:14] - gravity)
    model = identify_quadrotor([table for _, table in tables], NEUROBEM_MASS, NEUROBEM_INERTIA, np.concatenate(forces))
    model["used"] = {}
    if min(model["roll_r2"], model["pitch_r2"]) < min_fit_r2:
        model["arm_x"] = model["arm_y"] = arm_length / np.sqrt(2)
        model["used"]["geometry"] = f"default X layout, arm {arm_length} m (fit R^2 too low)"
    if model["yaw_r2"] < min_fit_r2:
        model["k_m"] = drag_ratio * model["k_f"]
        model["used"]["k_m"] = f"k_f * {drag_ratio} m (fit R^2 too low)"
    rotors = [(name, ROTOR_SIGNS[name][0] * model["arm_x"], ROTOR_SIGNS[name][1] * model["arm_y"]) for name in NEUROBEM_ROTORS]
    # The paper gives about 33 N of static thrust in total.
    max_speed = max(
        float(max(np.abs(table[:, 20:24]).max() for _, table in tables)) * 1.05,
        float(np.sqrt(33.0 / 4 / model["k_f"])),
    )
    effort = 3 * model["k_m"] * max_speed**2
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    urdf_path = output / "neurobem_quadrotor.urdf"
    urdf_path.write_text(
        quadrotor_urdf("neurobem_quadrotor", NEUROBEM_MASS, NEUROBEM_INERTIA, rotors,
                       rotor_inertia=rotor_inertia, effort=effort, velocity=max_speed)
    )
    spin = np.asarray(model["spin"], dtype=np.float64)
    episodes = []
    for (name, table), force in zip(tables, forces):
        time = table[:, 0] - table[0, 0]
        grid = uniform_time(0.0, time[-1], rate)
        speeds = table[:, 20:24] * spin
        rotor_angle = np.cumsum(np.concatenate([[np.zeros(4)], 0.5 * (speeds[1:] + speeds[:-1]) * np.diff(time)[:, None]]), 0)
        signals = {
            "state": resample(time, rotor_angle, grid),
            "velocity": resample(time, speeds, grid),
            "base_position": resample(time, table[:, 17:20], grid),
            "base_orientation": resample_quaternion(time, table[:, 7:11], grid),
            "base_linear_velocity": resample(time, table[:, 14:17], grid),
            "base_angular_velocity": resample(time, table[:, 4:7], grid),
            "imu_acceleration": resample(time, force, grid),
        }
        episodes.append(Episode(signals, {"file": name}))
    joints = [f"rotor_{name}_joint" for name in NEUROBEM_ROTORS]
    info = write_dataset(
        output, episodes, rate, joints, None,
        {"source": "https://rpg.ifi.uzh.ch/NeuroBEM.html", "robot": "NeuroBEM quadrotor (0.772 kg)",
         "identified_model": model, "acceleration_column": "specific force" if specific else "kinematic"},
    )
    overrides = {
        "rotor_*": {
            "mode": "thrust",
            "thrust_coefficient": model["k_f"],
            "torque_coefficient": model["k_m"],
            "kd": rotor_inertia / motor_time_constant,
            "peak_torque": effort,
            "continuous_torque": 0.7 * effort,
            "max_velocity": max_speed,
            "max_power": effort * max_speed,
        }
    }
    return source_config(
        output, info, str(urdf_path.resolve()), "neurobem_quadrotor", True, None,
        {"actuator_overrides": overrides, "action_type": "velocity"},
    )


# ---------------------------------------------------------------------- LocoMuJoCo

LOCOMUJOCO_REPOSITORY = "robfiras/loco-mujoco-datasets"


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load every non-object array of an npz file (pickled entries are skipped)."""
    arrays = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            try:
                arrays[key] = data[key]
            except ValueError:
                continue
    return arrays


def convert_locomujoco(
    output: str | Path,
    robot: str = "UnitreeG1",
    collection: str = "DefaultDatasets",
    patterns: list[str] | None = None,
    max_files: int | None = None,
    urdf: str = "assets/urdf/g1_29dof.urdf",
    rename: dict[str, str] | None = None,
    files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Convert LocoMuJoCo motions (floating-base qpos/qvel with joint names).

    Args:
        output (str | Path): Output folder.
        robot (str): Robot folder, e.g. ``UnitreeG1``, ``UnitreeH1``, ``Talos``.
        collection (str): ``DefaultDatasets`` or ``Lafan1``.
        patterns (list[str] | None): Globs over file names, e.g. ``["*walk*"]``.
        max_files (int | None): Limit on motion files.
        urdf (str): URDF path written into the source.
        rename (dict | None): Map from LocoMuJoCo joint names to URDF names.
        files (list | None): Local npz files instead of downloading.

    Returns:
        dict: Data-config source. License: CC BY-NC-ND 4.0.
    """
    if files is None:
        prefix = f"{collection}/mocap/{robot}/"
        names = _select([n for n in _hub_files(LOCOMUJOCO_REPOSITORY, prefix) if n.endswith(".npz")], patterns, max_files)
        files = [_hub_download(LOCOMUJOCO_REPOSITORY, name) for name in names]
    episodes, joints = [], None
    for path in files:
        data = _load_npz(path)
        names = [str(name) for name in data["joint_names"]]
        if names[0] != "root":
            raise ValueError(f"{path}: expected a free root joint first")
        names = [(rename or {}).get(name, name) for name in names[1:]]
        if joints is None:
            joints = names
        elif names != joints:
            raise ValueError("all motions must have the same joints")
        rate = float(np.asarray(data["frequency"]).reshape(-1)[0])
        qpos, qvel = data["qpos"].astype(np.float64), data["qvel"].astype(np.float64)
        splits = data.get("split_points", np.array([0, len(qpos)]))
        for start, stop in zip(splits[:-1], splits[1:]):
            quaternion = quaternion_wxyz_to_xyzw(qpos[start:stop, 3:7])
            signals = {
                "state": qpos[start:stop, 7:],
                "velocity": qvel[start:stop, 6:],
                "base_position": qpos[start:stop, :3],
                "base_orientation": quaternion,
                "base_linear_velocity": to_base_frame(quaternion, qvel[start:stop, :3]),
                "base_angular_velocity": qvel[start:stop, 3:6],
            }
            episodes.append(Episode(signals, {"file": Path(path).name}))
    info = write_dataset(
        output, episodes, rate, joints, None,
        {"source": f"https://huggingface.co/datasets/{LOCOMUJOCO_REPOSITORY}", "license": "CC BY-NC-ND 4.0",
         "robot": robot},
    )
    return source_config(output, info, urdf, f"locomujoco_{robot.lower()}", True)


# --------------------------------------------------------------------- AMASS -> G1

AMASS_REPOSITORY = "ember-lab-berkeley/AMASS_Retargeted_for_G1"
G1_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def convert_amass_g1(
    output: str | Path,
    patterns: list[str] | None = None,
    max_files: int | None = 50,
    urdf: str = "assets/urdf/g1_29dof.urdf",
    files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Convert AMASS motions retargeted to the Unitree G1 (kinematic, 29 DoF).

    Args:
        output (str | Path): Output folder.
        patterns (list[str] | None): Globs over Hub paths, e.g. ``["*Running*"]``.
        max_files (int | None): Limit on motion files.
        urdf (str): URDF path written into the source.
        files (list | None): Local npz files instead of downloading.

    Returns:
        dict: Data-config source.
    """
    if files is None:
        names = _select([n for n in _hub_files(AMASS_REPOSITORY, "g1/") if n.endswith(".npz")], patterns, max_files)
        files = [_hub_download(AMASS_REPOSITORY, name) for name in names]
    episodes, joints, rate = [], None, None
    for path in files:
        data = _load_npz(path)
        names = [str(name) for name in data["dof_names"]]
        joints = joints or names
        if names != joints:
            raise ValueError("all motions must have the same joints")
        rate = float(np.asarray(data["fps"]).reshape(-1)[0])
        signals = {
            "state": data["dof_positions"].astype(np.float64),
            "base_position": data["body_positions"][:, 0].astype(np.float64),
            "base_orientation": quaternion_wxyz_to_xyzw(data["body_rotations"][:, 0].astype(np.float64)),
        }
        episodes.append(Episode(signals, {"file": Path(path).name}))
    info = write_dataset(
        output, episodes, rate, joints, None,
        {"source": f"https://huggingface.co/datasets/{AMASS_REPOSITORY}",
         "license": "CC BY 4.0 on the card; AMASS terms apply", "robot": "Unitree G1"},
    )
    return source_config(output, info, urdf, "amass_g1", True, G1_FEET, {"smoothing_window": 3})


# ------------------------------------------------------------------------- LAFAN1

LAFAN1_REPOSITORY = "lvhaidong/LAFAN1_Retargeting_Dataset"
_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"]
_WRIST = ["wrist_roll", "wrist_pitch", "wrist_yaw"]
LAFAN1_JOINTS = {
    "g1": [f"{side}_{joint}_joint" for side in ("left", "right") for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{side}_{joint}_joint" for side in ("left", "right") for joint in _ARM + _WRIST],
    "h1_2": [f"{side}_{joint}_joint" for side in ("left", "right") for joint in ("hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")]
    + ["torso_joint"]
    + [f"{side}_{joint}_joint" for side in ("left", "right") for joint in _ARM + _WRIST],
    "h1": [f"{side}_{joint}_joint" for side in ("left", "right") for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
    + ["torso_joint"]
    + [f"{side}_{joint}_joint" for side in ("left", "right") for joint in _ARM],
}
LAFAN1_FEET = {"g1": G1_FEET, "h1": ["left_ankle_link", "right_ankle_link"], "h1_2": ["left_ankle_roll_link", "right_ankle_roll_link"]}
LAFAN1_URDF = {"g1": "assets/urdf/g1_29dof.urdf", "h1": "assets/urdf/h1.urdf", "h1_2": "assets/urdf/h1_2.urdf"}


def convert_lafan1(
    output: str | Path,
    robot: str = "g1",
    patterns: list[str] | None = None,
    max_files: int | None = None,
    urdf: str | None = None,
    files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Convert LAFAN1 motions retargeted to Unitree G1, H1, or H1_2 (30 Hz CSV).

    Args:
        output (str | Path): Output folder.
        robot (str): ``g1``, ``h1``, or ``h1_2``.
        patterns (list[str] | None): Globs over file names, e.g. ``["*walk*"]``.
        max_files (int | None): Limit on CSV files.
        urdf (str | None): URDF path; defaults per robot.
        files (list | None): Local CSV files instead of downloading.

    Returns:
        dict: Data-config source. LAFAN1 motion data is CC BY-NC-ND 4.0.
    """
    joints = LAFAN1_JOINTS[robot]
    if files is None:
        names = _select([n for n in _hub_files(LAFAN1_REPOSITORY, f"{robot}/") if n.endswith(".csv")], patterns, max_files)
        files = [_hub_download(LAFAN1_REPOSITORY, name) for name in names]
    episodes = []
    for path in files:
        table = np.loadtxt(path, delimiter=",", ndmin=2)
        if table.shape[1] != 7 + len(joints):
            raise ValueError(f"{path}: expected {7 + len(joints)} columns for {robot}")
        signals = {"state": table[:, 7:], "base_position": table[:, :3], "base_orientation": table[:, 3:7]}
        episodes.append(Episode(signals, {"file": Path(path).name}))
    info = write_dataset(
        output, episodes, 30.0, joints, None,
        {"source": f"https://huggingface.co/datasets/{LAFAN1_REPOSITORY}",
         "license": "LAFAN1 motions CC BY-NC-ND 4.0; retargeting BSD-3", "robot": f"Unitree {robot.upper()}"},
    )
    return source_config(output, info, urdf or LAFAN1_URDF[robot], f"lafan1_{robot}", True, LAFAN1_FEET[robot], {"smoothing_window": 3})


CONVERTERS = {
    "mipo": convert_mipo,
    "grandtour": convert_grandtour,
    "neurobem": convert_neurobem,
    "locomujoco": convert_locomujoco,
    "amass_g1": convert_amass_g1,
    "lafan1": convert_lafan1,
}
