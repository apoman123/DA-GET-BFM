"""Hugging Face ``Trainer`` that logs every auxiliary loss separately."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from transformers import Trainer


class PhysicsPretrainingTrainer(Trainer):
    """Trainer for ``DaGetForPhysicsPretraining``.

    Besides the weighted total ``loss``, it logs the running mean of each task
    loss as ``loss_fk``, ``loss_fd``, ``loss_id``, ``loss_motor``,
    ``loss_effect``, and ``loss_rbd`` (and ``eval_loss_*`` during evaluation).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create the trainer; arguments are those of ``transformers.Trainer``."""
        super().__init__(*args, **kwargs)
        self._component_sums: dict[str, dict[str, float]] = {
            "train": defaultdict(float),
            "eval": defaultdict(float),
        }
        self._component_counts = {"train": 0, "eval": 0}

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        """Run the model and record its per-task losses."""
        outputs = model(**inputs)
        phase = "train" if model.training else "eval"
        for name, value in (outputs.loss_components or {}).items():
            self._component_sums[phase][name] += float(value)
        self._component_counts[phase] += 1
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Add the mean task losses since the last log, then log."""
        phase, prefix = ("eval", "eval_loss_") if "eval_loss" in logs else ("train", "loss_")
        if phase == "train" and "loss" not in logs:
            return super().log(logs, start_time)
        count = self._component_counts[phase]
        if count:
            for name, total in self._component_sums[phase].items():
                logs[f"{prefix}{name}"] = round(total / count, 6)
        self._component_sums[phase] = defaultdict(float)
        self._component_counts[phase] = 0
        return super().log(logs, start_time)
