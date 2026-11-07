"""Check inverse-dynamics torque estimates against measured joint torques.

Uses the MIPO Unitree Go1 recordings (``Open-IO/mipo-go1-odometry`` on the
Hugging Face Hub, AGPL-3.0, from Cerberus2.0). They have 400 Hz joint
positions, velocities, and measured torques (Unitree's current-based
estimate), a body IMU, and foot-force sensors. The script estimates the
torques from kinematics alone, the way the training pipeline does for
datasets without torque, and compares them with the measured torques.

Usage::

    uv run python scripts/download_urdfs.py
    uv run --extra validation python scripts/validate_torque_estimation.py \\
        --sequence indoor/20230625_gait/230625-risqh-flytrot-04-032-33square.h5

It prints, per estimate, the RMSE in N m, the RMSE relative to the RMS of the
measured torque (by joint type), and the mean correlation:

* fixed-base inverse dynamics (what you get if you ignore the floating base);
* floating base with contacts from the foot-force sensors;
* floating base with contacts detected from kinematics (no sensors);
* floating base with sensor contacts and a quasi-static base (no IMU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import EmbodimentGraph, JointFeatureConfig, actuator_specs  # noqa: E402
from model.dynamics import STANDARD_GRAVITY, quaternion_to_matrix  # noqa: E402
from model.training.episodes import EmbodimentRecord, Trajectory, process_episode  # noqa: E402

REPOSITORY = "Open-IO/mipo-go1-odometry"
LEGS = ("FL", "FR", "RL", "RR")
PARTS = ("hip", "thigh", "calf")
FEET = [f"{leg}_foot" for leg in LEGS]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average along time with edge padding."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])], 1)


def load(path: Path, window: int, stride: int) -> dict[str, torch.Tensor | float]:
    """Read one MIPO sequence and align the IMU to the joint timestamps."""
    import h5py

    with h5py.File(path) as handle:
        joints = handle["unitree_hardware/joint_foot"]
        imu = handle["unitree_hardware/imu"]
        time = joints["time"][:]
        position = joints["position"][:, :12]
        velocity = joints["velocity"][:, :12]
        effort = joints["effort"][:, :12]
        foot = joints["effort"][:, 12:]
        imu_time = imu["time"][:]

        def align(values):
            return np.stack([np.interp(time, imu_time, values[:, k]) for k in range(values.shape[1])], 1)

        quaternion = align(imu["orientation"][:])
        gyro = align(imu["angular_velocity"][:])
        accelerometer = align(imu["linear_acceleration"][:])
    position, velocity = smooth(position, window), smooth(velocity, window)
    gyro, accelerometer = smooth(gyro, window), smooth(accelerometer, window)

    def tensor(values):
        return torch.from_numpy(np.ascontiguousarray(values[::stride])).double()

    rotation = quaternion_to_matrix(tensor(quaternion), "xyzw")
    gravity = (rotation.transpose(-1, -2) @ torch.tensor([0, 0, -STANDARD_GRAVITY], dtype=torch.float64)[:, None]).squeeze(-1)
    return {
        "dt": float(np.mean(np.diff(time))) * stride,
        "rate": 1 / float(np.mean(np.diff(time))),
        "q": tensor(position),
        "qd": tensor(velocity),
        "tau": tensor(effort),
        "contacts": tensor(foot) > 200.0,
        "rotation": rotation,
        "gravity": gravity,
        "gyro": tensor(gyro),
        "specific_force": tensor(accelerometer),
    }


def evaluate(name: str, data, urdf: Path, floating: bool, contacts, use_imu: bool) -> None:
    """Estimate torques for one setting and print error statistics."""
    names = [f"{leg}_{part}_joint" for leg in LEGS for part in PARTS]
    graph = EmbodimentGraph.from_urdf(urdf, JointFeatureConfig(), joint_order=names, floating_base=floating)
    record = EmbodimentRecord.build("go1", graph, actuator_specs(graph), contact_links=FEET if floating else [])
    trajectory = Trajectory(
        embodiment=0,
        dt=data["dt"],
        positions=data["q"],
        velocities=data["qd"],
        gravity=data["gravity"],
        base_orientation=data["rotation"] if use_imu else None,
        base_angular_velocity=data["gyro"] if use_imu else None,
        imu_specific_force=data["specific_force"] if use_imu else None,
        contacts=contacts,
    )
    estimate = process_episode(record, trajectory)["applied_torque"][:, 0].double()
    measured = data["tau"]
    keep = slice(10, -10)
    error = (estimate - measured)[keep]
    rmse = error.pow(2).mean(0).sqrt()
    rms = measured[keep].pow(2).mean(0).sqrt()
    total = float(rmse.pow(2).mean().sqrt())
    relative = total / float(rms.pow(2).mean().sqrt())
    parts = "  ".join(
        f"{part} {100 * float((rmse[i::3].pow(2).mean() / rms[i::3].pow(2).mean()).sqrt()):5.1f}%"
        for i, part in enumerate(PARTS)
    )
    correlation = np.mean(
        [np.corrcoef(estimate[keep, j], measured[keep, j])[0, 1] for j in range(measured.shape[1])]
    )
    print(f"{name:46s} RMSE {total:5.2f} N m ({100 * relative:5.1f}%)  {parts}  corr {correlation:.2f}")


def main() -> None:
    """Download a sequence and compare the estimates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequence", default="indoor/20230625_gait/230625-risqh-flytrot-04-032-33square.h5")
    parser.add_argument("--urdf", default=str(ROOT / "assets" / "urdf" / "go1.urdf"))
    parser.add_argument("--window", type=int, default=15, help="Moving-average length in samples (400 Hz).")
    parser.add_argument("--stride", type=int, default=2, help="Keep every n-th sample.")
    arguments = parser.parse_args()
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPOSITORY, arguments.sequence, repo_type="dataset"))
    data = load(path, arguments.window, arguments.stride)
    print(
        f"{path.name}: {data['q'].shape[0]} frames at {data['rate'] / arguments.stride:.0f} Hz, "
        f"RMS measured torque {float(data['tau'].pow(2).mean().sqrt()):.2f} N m"
    )
    urdf = Path(arguments.urdf)
    evaluate("fixed-base inverse dynamics", data, urdf, False, None, False)
    evaluate("floating base, sensor contacts, IMU", data, urdf, True, data["contacts"], True)
    evaluate("floating base, detected contacts, IMU", data, urdf, True, None, True)
    evaluate("floating base, sensor contacts, quasi-static", data, urdf, True, data["contacts"], False)


if __name__ == "__main__":
    main()
