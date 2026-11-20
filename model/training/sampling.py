"""Temperature sampling over robots or families (proposal D2).

With ``n_g`` samples in group ``g`` (a robot, a morphology family, or a source),
temperature sampling draws group ``g`` with probability ``q_g = p_g^alpha / sum_h
p_h^alpha`` where ``p_g = n_g / sum_h n_h`` (Lample and Conneau, "Cross-lingual
Language Model Pretraining", NeurIPS 2019, arXiv:1901.07291, Sec. 3.1).
``alpha = 1`` is the data distribution and ``alpha = 0`` is uniform over groups.
XLM-R found ``alpha = 0.3`` best ("we found 0.3 to be an optimal value for alpha";
Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer and
Stoyanov, "Unsupervised Cross-lingual Representation Learning at Scale", ACL 2020,
pp. 8440-8451, DOI 10.18653/v1/2020.acl-main.747); Arivazhagan, Bapna, Firat,
Lepikhin, Johnson, Krikun, Chen, Cao, Foster, Cherry, Macherey, Chen and Wu
("Massively Multilingual Neural Machine Translation in the Wild: Findings and
Challenges", arXiv:1907.05019, 2019) sample with ``p^(1/T)`` and use ``T = 5``, that is
``alpha = 1 / T = 0.2``. The zero-shot configs start from these two values and tune
alpha on the validation split; balancing by family rather than by robot is an
unverified hypothesis motivated by the unsaturated humanoid results of Ai et al.
(CoRL 2025, PMLR 305:3483-3515, arXiv:2505.05753: "for quadrupeds and hexapods,
performance saturates around 100 training embodiments, while for humanoids, it
continues to improve steadily").
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
