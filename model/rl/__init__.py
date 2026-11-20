"""Online off-policy training of the cross-embodiment behavior foundation model.

The algorithm is FB-CPR (Tirinzoni et al., ICLR 2025, arXiv:2504.11054) with auxiliary
rewards as in BFM-Zero (Li et al., ICLR 2026, arXiv:2511.04131); parts of the code are
adapted from https://github.com/LeCAR-Lab/BFM-Zero (CC BY-NC 4.0, see each module and
``THIRD_PARTY_NOTICES.md``).
"""

from model.rl.actions import ActionMap, TorqueMonitor
from model.rl.agent import DaGetFbCprAgent
from model.rl.buffers import ExpertMotions, ReplayBuffer
from model.rl.configuration_fb_cpr import FbCprConfig
from model.rl.deployment import DaGetBfmPolicy
from model.rl.observation import EmbodimentObserver, Observation, RobotState, StateView
from model.rl.physics_replay import PhysicsReplay, PhysicsReplaySettings

__all__ = [
    "ActionMap",
    "DaGetBfmPolicy",
    "DaGetFbCprAgent",
    "EmbodimentObserver",
    "ExpertMotions",
    "FbCprConfig",
    "Observation",
    "PhysicsReplay",
    "PhysicsReplaySettings",
    "ReplayBuffer",
    "RobotState",
    "StateView",
    "TorqueMonitor",
]
