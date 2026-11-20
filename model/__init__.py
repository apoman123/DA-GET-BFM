"""Reusable transformer components and graph embodiment encoder models.

The encoder implements the Graph Embodiment Transformer of GET-Zero (Patel & Song, ICRA
2025, arXiv:2407.15002); the behavior foundation model networks follow BFM-Zero (Li et
al., ICLR 2026, arXiv:2511.04131) and FB-CPR (Tirinzoni et al., ICLR 2025, arXiv:2504.11054).
Each module's docstring cites the sources it uses.
"""

from model.configuration_bfm import BfmConfig
from model.configuration_da_get import DaGetConfig
from model.embodiment import (
    EmbodimentBatch,
    EmbodimentGraph,
    JointFeatureConfig,
    collate_embodiments,
)
from model.modeling_bfm import (
    DaGetBackwardMap,
    DaGetBfmActor,
    DaGetBfmModel,
    DaGetDiscriminator,
    DaGetForwardMap,
    TruncatedNormal,
)
from model.modeling_da_get import DaGetEncoder, PhysicsMemory
from model.physics import (
    ActuatorDefaults,
    ActuatorSpec,
    PhysicsFeatureCache,
    actuator_features,
    actuator_specs,
    dynamics_features,
    envelope_clip,
    envelope_targets,
    pd_envelope_torque,
    rigid_body_features,
    torque_limit,
)
from model.dynamics import constrained_response
from model.tokenization_da_get import DaGetTokenizer
from model.urdf import UrdfRobot, parse_urdf

__all__ = [
    "ActuatorDefaults",
    "ActuatorSpec",
    "BfmConfig",
    "DaGetBackwardMap",
    "DaGetBfmActor",
    "DaGetBfmModel",
    "DaGetConfig",
    "DaGetDiscriminator",
    "DaGetEncoder",
    "DaGetForwardMap",
    "DaGetTokenizer",
    "EmbodimentBatch",
    "EmbodimentGraph",
    "JointFeatureConfig",
    "PhysicsFeatureCache",
    "PhysicsMemory",
    "UrdfRobot",
    "actuator_features",
    "actuator_specs",
    "collate_embodiments",
    "constrained_response",
    "dynamics_features",
    "envelope_clip",
    "envelope_targets",
    "parse_urdf",
    "pd_envelope_torque",
    "rigid_body_features",
    "torque_limit",
    "TruncatedNormal",
]
