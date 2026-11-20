"""Torque penalties as constraints: per-robot weights adapted by Lagrange multipliers.

Fixed weights on the torque penalties (``model.rl.actions``) must be tuned: too weak and
the policy rides the motors' limits and overheats them; too strong and it moves timidly
and cannot get up. Each penalty instead gets a target from the actuator data sheet on a
measured rollout metric, and its weight is scaled per robot by a multiplier ``lambda``
updated by multiplicative dual ascent before every update round:

    lambda <- clip(lambda * exp(rate * (metric - target) / target), low, high)

The weight grows while a robot violates its target and decays while it satisfies it, so
the constraint ends up just active, as in constrained RL with Lagrangian relaxation
(Altman, *Constrained Markov Decision Processes*, Chapman & Hall/CRC, 1999, DOI
10.1201/9781315140223; Tessler, Mankowitz & Mannor, "Reward Constrained Policy
Optimization", ICLR 2019, arXiv:1805.11074; Stooke, Achiam & Abbeel, "Responsive Safety
in Reinforcement Learning by PID Lagrangian Methods", ICML 2020, PMLR 119:9133-9143,
arXiv:2007.03964). The update in log space and the targets are this repo's. The targets are physical and normalized by
each robot's own limits, so the same targets fit every robot, and robots of different
strength get different weights.

| Penalty | Metric (``TorqueMonitor``) | Default target |
|---|---|---|
| ``penalty_torque_limit`` | ``torque_request_over_envelope``: share of joints whose request exceeds ``tau^max(qd)`` | 0.05 |
| ``penalty_torque_continuous`` | ``torque_load_over_continuous``: largest thermal load, ``EMA[(tau / tau^cont)^2]`` | 1.0 |
| ``penalty_power`` | ``power_over_limit``: share of joints and physics steps above ``P^max`` | 0.02 |
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

DEFAULT_TORQUE_TARGETS = {
    "penalty_torque_limit": ("torque_request_over_envelope", 0.05),
    "penalty_torque_continuous": ("torque_load_over_continuous", 1.0),
    "penalty_power": ("power_over_limit", 0.02),
}


class PenaltyMultipliers:
    """Per-robot multipliers of constrained auxiliary rewards.

    Args:
        robots (int): Number of robots.
        targets (Mapping[str, tuple[str, float]]): Reward name to (metric name, target).
        rate (float): Step size of the multiplicative update.
        bounds (tuple[float, float]): Smallest and largest multiplier.
    """

    def __init__(
        self,
        robots: int,
        targets: Mapping[str, tuple[str, float]] | None = None,
        rate: float = 0.05,
        bounds: tuple[float, float] = (0.1, 10.0),
    ) -> None:
        self.targets = dict(DEFAULT_TORQUE_TARGETS if targets is None else targets)
        for name, (_, target) in self.targets.items():
            if not target > 0:
                raise ValueError(f"the target of {name} must be positive")
        if not 0 < bounds[0] <= 1 <= bounds[1]:
            raise ValueError("bounds must satisfy 0 < low <= 1 <= high")
        self.rewards = list(self.targets)
        self.rate = rate
        self.bounds = bounds
        self.values = np.ones((robots, len(self.rewards)))
        self._sums = np.zeros((robots, len(self.rewards)))
        self._counts = np.zeros((robots, len(self.rewards)))

    def observe(self, robot: int, metrics: Mapping[str, torch.Tensor]) -> None:
        """Accumulate one step's metrics ``{name: [N]}`` of a robot."""
        for index, name in enumerate(self.rewards):
            metric = self.targets[name][0]
            if metric in metrics:
                values = torch.as_tensor(metrics[metric]).double()
                self._sums[robot, index] += float(values.sum())
                self._counts[robot, index] += values.numel()

    def update(self) -> np.ndarray:
        """Apply one dual step from the metrics observed since the last one; return ``[R, K]`` multipliers."""
        low, high = self.bounds
        for robot in range(self.values.shape[0]):
            for index, name in enumerate(self.rewards):
                if self._counts[robot, index] == 0:
                    continue
                mean = self._sums[robot, index] / self._counts[robot, index]
                target = self.targets[name][1]
                step = math.exp(self.rate * (mean - target) / target)
                self.values[robot, index] = float(np.clip(self.values[robot, index] * step, low, high))
        self._sums[:] = 0.0
        self._counts[:] = 0.0
        return self.values.copy()

    def scales(self, base: Mapping[str, float], names: Sequence[str]) -> torch.Tensor:
        """Return ``[R, K]`` reward weights: ``base`` times each robot's multiplier where constrained."""
        matrix = torch.tensor([[float(base[name]) for name in names]] * self.values.shape[0], dtype=torch.float32)
        for index, name in enumerate(self.rewards):
            if name in names:
                matrix[:, list(names).index(name)] *= torch.from_numpy(self.values[:, index]).float()
        return matrix

    def state_dict(self) -> dict:
        """Return the multipliers, for checkpoints."""
        return {"rewards": list(self.rewards), "values": self.values.copy()}

    def load_state_dict(self, state: Mapping) -> None:
        """Restore multipliers saved by ``state_dict`` (rewards matched by name)."""
        for index, name in enumerate(state["rewards"]):
            if name in self.rewards:
                self.values[:, self.rewards.index(name)] = np.asarray(state["values"])[:, index]
