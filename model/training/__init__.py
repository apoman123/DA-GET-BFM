"""Data pipeline and trainer for physics pretraining of the GET encoder."""

from model.training.dataset import (
    PhysicsDataCollator,
    PhysicsDatasets,
    PhysicsTrajectoryDataset,
    build_physics_datasets,
)
from model.training.episodes import EmbodimentRecord, Trajectory, process_episode
from model.training.trainer import PhysicsPretrainingTrainer

__all__ = [
    "EmbodimentRecord",
    "PhysicsDataCollator",
    "PhysicsDatasets",
    "PhysicsPretrainingTrainer",
    "PhysicsTrajectoryDataset",
    "Trajectory",
    "build_physics_datasets",
    "process_episode",
]
