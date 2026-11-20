"""Data pipeline and trainer for physics pretraining of the GET encoder.

The encoder is the Graph Embodiment Transformer of GET-Zero (Patel & Song, ICRA 2025,
pp. 14262-14269, DOI 10.1109/ICRA55743.2025.11127922, arXiv:2407.15002). Sources and
datasets are cited in the submodules, ``docs/training.md`` and ``docs/data.md``.
"""

from model.training.dataset import (
    CollatorSettings,
    PhysicsDataCollator,
    PhysicsDatasets,
    PhysicsTrajectoryDataset,
    build_physics_datasets,
)
from model.training.episodes import (
    TORQUE_SOURCES,
    EmbodimentRecord,
    Trajectory,
    estimate_episode_efforts,
    process_episode,
)
from model.training.trainer import PhysicsPretrainingTrainer

__all__ = [
    "TORQUE_SOURCES",
    "CollatorSettings",
    "EmbodimentRecord",
    "PhysicsDataCollator",
    "PhysicsDatasets",
    "PhysicsPretrainingTrainer",
    "PhysicsTrajectoryDataset",
    "Trajectory",
    "build_physics_datasets",
    "estimate_episode_efforts",
    "process_episode",
]
