"""Temperature sampling over robots or families (proposal D2).

With ``n_g`` samples in group ``g`` (a robot, a morphology family, or a source),
temperature sampling draws group ``g`` with probability ``q_g = p_g^alpha / sum_h
p_h^alpha`` where ``p_g = n_g / sum_h n_h`` (Lample & Conneau, arXiv:1901.07291).
``alpha = 1`` is the data distribution and ``alpha = 0`` is uniform over groups.
XLM-R found ``alpha = 0.3`` best (Conneau et al., ACL 2020, DOI
10.18653/v1/2020.acl-main.747); Arivazhagan et al. (arXiv:1907.05019) use ``T = 5``,
that is ``alpha = 1 / T = 0.2``. The zero-shot configs start from these two values
and tune alpha on the validation split; balancing by family rather than by robot is
an unverified hypothesis motivated by the unsaturated humanoid results of Ai et al.
(CoRL 2025, arXiv:2505.05753).
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch

SAMPLING_GROUPS = ("embodiment", "family", "source")


def temperature_weights(groups: Sequence[object], alpha: float) -> torch.Tensor:
    """Return per-sample weights whose group totals are ``p_g^alpha`` (normalized).

    Args:
        groups (Sequence[object]): Group label of every sample.
        alpha (float): Temperature exponent in [0, 1].

    Returns:
        torch.Tensor: Float64 ``[N]`` weights summing to one.

    Raises:
        ValueError: alpha is outside [0, 1] or there are no samples.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not groups:
        raise ValueError("no samples to weight")
    counts = Counter(groups)
    total = sum(counts.values())
    mass = {group: (count / total) ** alpha for group, count in counts.items()}
    normalizer = sum(mass.values())
    weights = torch.tensor([mass[group] / normalizer / counts[group] for group in groups], dtype=torch.float64)
    return weights / weights.sum()


def sample_groups(dataset, records: Sequence, group: str) -> list[object]:
    """Return the group label of every sample of a ``PhysicsTrajectoryDataset``.

    Args:
        dataset (PhysicsTrajectoryDataset): Training samples.
        records (Sequence[EmbodimentRecord]): Records the dataset indexes.
        group (str): ``embodiment``, ``family`` (record family, else the record
            name), or ``source`` (record name before the first ``/``).

    Returns:
        list[object]: One label per sample.

    Raises:
        ValueError: The group kind is unknown.
    """
    if group not in SAMPLING_GROUPS:
        raise ValueError(f"sampling_group must be one of {SAMPLING_GROUPS}")
    labels = []
    for item in range(len(dataset)):
        record = records[dataset.embodiment_of(item)]
        if group == "embodiment":
            labels.append(record.name)
        elif group == "family":
            labels.append(record.family or record.name)
        else:
            labels.append(record.name.split("/")[0])
    return labels
