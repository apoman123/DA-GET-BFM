"""Hugging Face ``Trainer`` that logs every auxiliary loss separately.

References:
    * ``transformers.Trainer``: Wolf et al., "Transformers: State-of-the-Art Natural
      Language Processing", EMNLP 2020 System Demonstrations, pp. 38-45, DOI
      10.18653/v1/2020.emnlp-demos.6. The subclass overrides ``_get_train_sampler``,
      ``compute_loss`` and ``log``; no Trainer code is copied.
    * ``torch.utils.data.WeightedRandomSampler``: PyTorch, Paszke et al., NeurIPS 2019,
      arXiv:1912.01703.
    * Temperature sampling of the weights (``model.training.sampling``): Conneau et al.,
      ACL 2020, DOI 10.18653/v1/2020.acl-main.747 (alpha = 0.3); Arivazhagan et al.,
      arXiv:1907.05019, 2019 (T = 5).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer


class PhysicsPretrainingTrainer(Trainer):
    """Trainer for ``DaGetForPhysicsPretraining``.

    Besides the weighted total ``loss``, it logs the running mean of each task
    loss as ``loss_fk``, ``loss_fd``, ``loss_id``, ``loss_response``,
    ``loss_inverse_response``, ``loss_motor``, and ``loss_effect`` (and
    ``eval_loss_*`` during evaluation).

    With ``sample_weights`` (``model.training.sampling.temperature_weights``) the
    training samples are drawn with replacement in proportion to them instead of
    being shuffled uniformly (proposal D2).
    """

    def __init__(self, *args: Any, sample_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Create the trainer; arguments are those of ``transformers.Trainer``.

        Args:
            *args (Any): ``transformers.Trainer`` arguments.
            sample_weights (torch.Tensor | None): Per-sample weights of the training set.
            **kwargs (Any): ``transformers.Trainer`` keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
        self._component_sums: dict[str, dict[str, float]] = {
            "train": defaultdict(float),
            "eval": defaultdict(float),
        }
        self._component_counts = {"train": 0, "eval": 0}

    def _get_train_sampler(self, *args: Any, **kwargs: Any):
        """Return a weighted sampler when sample weights are set, else the default."""
        if self.sample_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        generator = torch.Generator().manual_seed(int(self.args.seed))
        return WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights), replacement=True, generator=generator)

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
