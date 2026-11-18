"""Turn policy actions into joint targets, and measure torque-limit use, in simulation and on a robot.

The actor outputs normalized actions in [-1, 1]. ``ActionMap`` turns them into PD position
targets the way BFM-Zero does (``nominal + action_scale * action``, clipped to the joint
limits) and then, by default, moves each target so that the effort the drive's PD law
requests at the current state stays inside the actuator's torque-speed envelope
``tau^max(qd)`` (``model.physics.envelope_targets``). The training environments and the
deployed robot call the same code, so the policy is trained on exactly the targets the
robot will receive.

``TorqueMonitor`` computes the torque auxiliary rewards and metrics, normalized by each
robot's own limits so one weight fits every robot:

* ``penalty_torque_limit``: ``sum_j relu(|tau_req| - tau^max(qd)) / tau^peak`` of the
  actor's raw targets, how far it asks beyond the envelope (before clipping);
* ``penalty_torque_continuous``: ``sum_j relu(EMA[(tau / tau^cont)^2] - 1)``, sustained load
  above the continuous rating, a thermal proxy with time constant ``thermal_time_constant``;
* ``penalty_power``: ``sum_j relu(|tau qd| - P^max) / P^max``.

``tau`` is the simulator's applied torque, which is exact there. A real robot only needs
``ActionMap``; it never needs a torque measurement.
"""

from __future__ import annotations

import numpy as np
import torch

from model.physics import EPSILON, envelope_targets, requested_torque, torque_limit, torque_limit_excess
from model.physics_schema import ACTUATOR_MODES
from model.training.episodes import EmbodimentRecord

TORQUE_REWARDS = ("penalty_torque_limit", "penalty_torque_continuous", "penalty_power")


class ActionMap:
    """Map normalized actions to PD position targets, optionally inside the torque envelope.

    Args:
        record (EmbodimentRecord): Robot with position-mode actuator specs (gains, limits).
        nominal (np.ndarray): ``[J]`` or ``[N, J]`` nominal pose (targets at zero action).
        action_scale (float): Target offset in rad at ``|action| = 1``.
        clip_to_envelope (bool): Move targets so the requested effort is within ``tau^max(qd)``.
    """

    def __init__(self, record: EmbodimentRecord, nominal: np.ndarray, action_scale: float, clip_to_envelope: bool = True) -> None:
        graph = record.graph
        self.tensors = record.actuator
        self.nominal = np.asarray(nominal, dtype=np.float64)
        self.action_scale = float(action_scale)
        self.clip_to_envelope = clip_to_envelope
        self.lower, self.upper = graph.lower.numpy(), graph.upper.numpy()
        self.limited = np.array([kind != "continuous" for kind in graph.joint_types]) & (self.upper > self.lower)

    def raw_targets(self, actions: np.ndarray) -> np.ndarray:
        """Return ``nominal + action_scale * clip(action)``, clipped to the joint limits."""
        actions = np.clip(np.nan_to_num(np.asarray(actions, dtype=np.float64)), -1.0, 1.0)
        targets = self.nominal + self.action_scale * actions
        return np.where(self.limited, np.clip(targets, self.lower, self.upper), targets)

    def __call__(self, actions: np.ndarray, q: np.ndarray, qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the targets to send and the raw targets before envelope clipping.

        Args:
            actions (np.ndarray): ``[N, J]`` normalized actions.
            q (np.ndarray): ``[N, J]`` joint positions when the targets are sent.
            qd (np.ndarray): ``[N, J]`` joint velocities at that time.

        Returns:
            tuple[np.ndarray, np.ndarray]: Targets ``[N, J]`` and raw targets ``[N, J]``.
        """
        raw = self.raw_targets(actions)
        if not self.clip_to_envelope:
            return raw, raw
        clipped = envelope_targets(self.tensors, torch.from_numpy(raw), torch.as_tensor(q), torch.as_tensor(qd)).numpy()
        clipped = np.where(self.limited, np.clip(clipped, self.lower, self.upper), clipped)
        return clipped, raw


class TorqueMonitor:
    """Torque auxiliary rewards and metrics of one robot's worlds.

    Args:
        record (EmbodimentRecord): Robot.
        num_envs (int): Worlds.
        control_dt (float): Control period in s.
        thermal_time_constant (float): Time constant in s of the load average.
    """

    def __init__(self, record: EmbodimentRecord, num_envs: int, control_dt: float, thermal_time_constant: float = 1.0) -> None:
        self.tensors = record.actuator
        self.actuated = (self.tensors["mode"] != ACTUATOR_MODES.index("passive")).numpy()
        self.rate = min(control_dt / max(thermal_time_constant, EPSILON), 1.0)
        self.load = np.zeros((num_envs, record.graph.num_joints))
        self.requested_excess = np.zeros(num_envs)
        self.saturated = np.zeros(num_envs)

    def reset(self, indices: np.ndarray) -> None:
        """Forget the load history of reset worlds."""
        self.load[indices] = 0.0

    def before_step(self, raw_targets: np.ndarray, q: np.ndarray, qd: np.ndarray) -> None:
        """Record how far the actor's raw targets ask beyond the envelope at the current state."""
        q_t, qd_t = torch.as_tensor(q), torch.as_tensor(qd)
        request = requested_torque(self.tensors, torch.as_tensor(raw_targets), q_t, qd_t)
        excess = torque_limit_excess(self.tensors, request, qd_t).numpy()
        self.requested_excess = excess.sum(axis=1)
        self.saturated = ((excess > 0) & self.actuated).sum(axis=1) / max(int(self.actuated.sum()), 1)

    def after_step(self, torque: np.ndarray, qd: np.ndarray) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Update the load average with the applied torque and return rewards and metrics.

        Args:
            torque (np.ndarray): ``[N, J]`` applied joint torques of the step.
            qd (np.ndarray): ``[N, J]`` joint velocities after the step.

        Returns:
            tuple: Unscaled auxiliary rewards ``{name: [N]}`` and metrics ``{name: [N]}``.
        """
        continuous = self.tensors["continuous_torque"].numpy()
        ratio = np.where(self.actuated, torque / (continuous + EPSILON), 0.0)
        self.load += self.rate * (ratio**2 - self.load)
        power_limit = self.tensors["max_power"].numpy()
        power = np.where(self.actuated, np.clip(np.abs(torque * qd) - power_limit, 0.0, None) / (power_limit + EPSILON), 0.0)
        envelope = torque_limit(self.tensors, torch.as_tensor(qd)).numpy()
        at_limit = (np.abs(torque) >= 0.99 * envelope) & self.actuated & (envelope > 0)
        rewards = {
            "penalty_torque_limit": torch.from_numpy(self.requested_excess.copy()),
            "penalty_torque_continuous": torch.from_numpy(np.clip(self.load - 1.0, 0.0, None).sum(axis=1)),
            "penalty_power": torch.from_numpy(power.sum(axis=1)),
        }
        count = max(int(self.actuated.sum()), 1)
        metrics = {
            "torque_request_over_envelope": torch.from_numpy(self.saturated.copy()),
            "torque_at_envelope": torch.from_numpy(at_limit.sum(axis=1) / count),
            "torque_rms_over_continuous": torch.from_numpy(np.sqrt((ratio**2).sum(axis=1) / count)),
            "torque_load_over_continuous": torch.from_numpy(self.load.max(axis=1)),
        }
        return rewards, metrics
