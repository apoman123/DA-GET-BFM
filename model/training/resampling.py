"""Resample recorded trajectories to one control period (proposal A2).

Physics pretraining mixes simulators (0.01 s by default), recorded datasets (``1 / fps``,
for example 30 Hz) and RL rollouts (0.02 s). A data config with ``control_dt`` puts every
source on one period: simulated sources are run at that period, and recorded ones are
interpolated here. BFM-Zero controls at 50 Hz (Li et al., arXiv:2511.04131, Sec. 3), and
its motion library blends motion frames linearly at the control step (``step_dt``,
default 1/50 s, ``humanoidverse/utils/motion_lib/motion_lib_base.py`` of
https://github.com/LeCAR-Lab/BFM-Zero; ``docs/bfm.md``), which is why the zero-shot
configs use ``control_dt = 0.02``.

Interpolation is linear for joint and base signals (joint positions are unwrapped first,
so a continuous joint crossing +-pi interpolates the short way), spherical for the base
orientation, zero-order hold for commands, and nearest-frame for contact flags.
Simulated trajectories with exact labels are not resampled: their accelerations and
probe effects belong to the period they were simulated at, so such sources must be
simulated at ``control_dt`` instead.

References:
    * Spherical interpolation ``R_0 exp(u log(R_0^T R_1))`` is Shoemake's slerp
      ``q_1 (q_1^-1 q_2)^u`` in matrix form: Shoemake, "Animating rotation with
      quaternion curves", SIGGRAPH 1985, Computer Graphics 19(3):245-254, DOI
      10.1145/325165.325242.
    * ``_unwrap`` equals ``numpy.unwrap`` (NumPy: Harris et al., Nature 585:357-362,
      2020, DOI 10.1038/s41586-020-2649-2).
"""

from __future__ import annotations

import math
from dataclasses import fields, replace

import torch

from model.training.episodes import Trajectory, _vee_log, exp_rotation

LINEAR_FIELDS = (
    "velocities",
    "accelerations",
    "applied_torque",
    "base_position",
    "base_linear_velocity",
    "base_angular_velocity",
    "base_linear_acceleration",
    "base_angular_acceleration",
    "imu_specific_force",
    "base_external_wrench",
)


def _unwrap(positions: torch.Tensor) -> torch.Tensor:
    """Remove jumps of more than pi between consecutive frames (``numpy.unwrap``)."""
    if positions.shape[0] < 2:
        return positions
    steps = positions.diff(dim=0)
    wrapped = torch.remainder(steps + math.pi, 2 * math.pi) - math.pi
    correction = torch.where(steps.abs() > math.pi, wrapped - steps, torch.zeros_like(steps))
    return positions + torch.cat((torch.zeros_like(positions[:1]), correction.cumsum(0)))


def _linear(values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Interpolate ``[T, ...]`` between frames ``lower`` and ``upper``."""
    shaped = weight.to(values.dtype).reshape(-1, *([1] * (values.ndim - 1)))
    return values[lower] * (1 - shaped) + values[upper] * shaped


def resample_trajectory(trajectory: Trajectory, dt: float) -> Trajectory:
    """Return the trajectory at period ``dt`` (unchanged if it already has it).

    Args:
        trajectory (Trajectory): Recorded trajectory.
        dt (float): Target period in s.

    Returns:
        Trajectory: Resampled copy covering the same time span.

    Raises:
        ValueError: The trajectory has exact simulator labels or probes, which
            cannot be resampled; simulate it at ``dt`` instead.
    """
    if abs(trajectory.dt - dt) < 1e-9:
        return trajectory
    if trajectory.exact_dynamics or trajectory.probe_torques is not None:
        raise ValueError(
            f"a simulated trajectory at dt={trajectory.dt} cannot be resampled to {dt}; "
            "simulate the source at control_dt instead"
        )
    steps = trajectory.positions.shape[0]
    duration = (steps - 1) * trajectory.dt
    count = int(math.floor(duration / dt + 1e-9)) + 1
    times = torch.arange(count, dtype=torch.float64) * (dt / trajectory.dt)
    lower = times.floor().long().clamp(0, steps - 1)
    upper = (lower + 1).clamp_max(steps - 1)
    weight = times - lower.to(torch.float64)
    nearest = times.round().long().clamp(0, steps - 1)
    changes: dict[str, object] = {"dt": float(dt)}
    positions = _unwrap(trajectory.positions.to(torch.float64))
    changes["positions"] = _linear(positions, lower, upper, weight).to(trajectory.positions.dtype)
    for name in LINEAR_FIELDS:
        value = getattr(trajectory, name)
        if value is not None:
            changes[name] = _linear(value, lower, upper, weight)
    if trajectory.gravity is not None and trajectory.gravity.ndim == 2:
        changes["gravity"] = _linear(trajectory.gravity, lower, upper, weight)
    if trajectory.projected_gravity is not None:
        direction = _linear(trajectory.projected_gravity, lower, upper, weight)
        changes["projected_gravity"] = torch.nn.functional.normalize(direction, dim=-1)
    if trajectory.base_orientation is not None:
        start = trajectory.base_orientation[lower].to(torch.float64)
        end = trajectory.base_orientation[upper].to(torch.float64)
        step = _vee_log(start.transpose(-1, -2) @ end)
        changes["base_orientation"] = (start @ exp_rotation(step * weight[:, None])).to(trajectory.base_orientation.dtype)
    if trajectory.actions is not None:
        changes["actions"] = trajectory.actions[lower]
    if trajectory.contacts is not None:
        changes["contacts"] = trajectory.contacts[nearest]
    if trajectory.transition_valid is not None:
        valid = trajectory.transition_valid.to(torch.bool)
        changes["transition_valid"] = valid[lower] & valid[upper]
    known = {item.name for item in fields(Trajectory)}
    return replace(trajectory, **{key: value for key, value in changes.items() if key in known})
