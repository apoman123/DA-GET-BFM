"""Parallel MuJoCo environments of one robot, built from its embodiment graph.

Follows BFM-Zero's environment: 200 Hz physics and 50 Hz PD position control around a
nominal pose, 500-step episodes that start from random motion frames, domain
randomization (friction, link mass, base center of mass, nominal pose, pushes), and
embodiment-independent auxiliary rewards. Only the feet collide in the generated
model, so a falling robot ends its episode instead of recovering from the ground.
"""

from __future__ import annotations

import copy
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from model.physics_schema import ACTUATOR_MODES
from model.rl.observation import RobotState
from model.training.episodes import EmbodimentRecord
from model.training.ground_sim import ContactGeometry, build_mjcf, foot_points

UP = np.array([0.0, 0.0, 1.0])
SOFT_LIMIT = 0.95  # BFM-Zero's soft_dof_pos_limit


@dataclass
class DomainRandomization:
    """BFM-Zero's randomization ranges (paper Table 4); ``enabled=False`` turns all of them off."""

    enabled: bool = True
    friction: tuple[float, float] = (0.5, 1.25)
    link_mass_scale: tuple[float, float] = (0.95, 1.05)
    base_com_offset: float = 0.02
    nominal_offset: float = 0.02
    push_interval: tuple[float, float] = (1.0, 3.0)
    push_linear_velocity: float = 0.5
    push_angular_velocity: float = 0.5


@dataclass
class EnvConfig:
    """Settings of ``MujocoEmbodimentEnv``.

    Attributes:
        num_envs (int): Parallel worlds.
        control_dt (float): Control period in s (BFM-Zero: 50 Hz).
        physics_dt (float): Physics step in s (BFM-Zero: 200 Hz).
        max_episode_steps (int): Episode length in control steps.
        action_scale (float): Target offset in rad at ``|action| = 1`` (BFM-Zero: 0.25 x 5).
        min_height_ratio (float): Episode ends below this fraction of the nominal height.
        max_tilt_degrees (float): Episode ends above this base tilt.
        nominal_pose (dict[str, float]): Joint name to nominal position; others are 0.
        geometry (ContactGeometry): Foot collision geometry.
        randomization (DomainRandomization): Domain randomization.
        seed (int): Random seed.
    """

    num_envs: int = 16
    control_dt: float = 0.02
    physics_dt: float = 0.005
    max_episode_steps: int = 500
    action_scale: float = 1.25
    min_height_ratio: float = 0.4
    max_tilt_degrees: float = 70.0
    nominal_pose: dict[str, float] = field(default_factory=dict)
    geometry: ContactGeometry = field(default_factory=ContactGeometry)
    randomization: DomainRandomization = field(default_factory=DomainRandomization)
    seed: int = 0


@dataclass
class EnvStep:
    """Result of one control step, before any reset.

    Attributes:
        state (RobotState): Next state of every world.
        torque (torch.Tensor): ``[N, J]`` applied joint torques.
        target (torch.Tensor): ``[N, J]`` joint targets that were commanded.
        terminated (torch.Tensor): Bool ``[N]``: fell or diverged.
        truncated (torch.Tensor): Bool ``[N]``: reached the time limit.
        aux_rewards (dict[str, torch.Tensor]): ``[N]`` unscaled auxiliary rewards.
    """

    state: RobotState
    torque: torch.Tensor
    target: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    aux_rewards: dict[str, torch.Tensor]


class MujocoEmbodimentEnv:
    """Batch of independent MuJoCo worlds of one robot, stepped by a thread pool.

    Args:
        record (EmbodimentRecord): Robot with position-mode actuators.
        config (EnvConfig): Settings.
        pool (ThreadPoolExecutor | None): Shared workers; worlds step serially if None.
    """

    def __init__(self, record: EmbodimentRecord, config: EnvConfig, pool: ThreadPoolExecutor | None = None) -> None:
        import mujoco

        self.mujoco = mujoco
        self.record = record
        self.config = config
        self.pool = pool
        self.graph = record.graph
        self.num_envs = config.num_envs
        self.joints = self.graph.num_joints
        self.floating = bool(self.graph.floating_base)
        self.rng = np.random.default_rng(config.seed)
        actuator = record.actuator
        if bool((actuator["mode"] == ACTUATOR_MODES.index("thrust")).any()):
            raise NotImplementedError(f"{record.name}: rotor (thrust) robots are not supported by the MuJoCo RL environment yet")
        self.substeps = max(1, int(round(config.control_dt / config.physics_dt)))
        self.lower, self.upper = self.graph.lower.numpy(), self.graph.upper.numpy()
        self.limited = np.array([kind != "continuous" for kind in self.graph.joint_types]) & (self.upper > self.lower)
        nominal = np.zeros(self.joints)
        for name, value in config.nominal_pose.items():
            nominal[self.graph.joint_names.index(name)] = value
        self.nominal = np.where(self.limited, np.clip(nominal, self.lower, self.upper), nominal)
        surface = record.surface_contacts if record.surface_contacts is not None else len(record.contact_links) <= 2
        self.feet = foot_points(record, config.geometry, surface) if record.contact_links else []
        xml = build_mjcf(record, actuator["kp"].numpy(), actuator["kd"].numpy(), config.geometry, 0.9, config.physics_dt)
        self.base_model = mujoco.MjModel.from_xml_string(xml)
        self._resolve_indices()
        self.models = [self._randomized_model() for _ in range(self.num_envs)]
        self.datas = [mujoco.MjData(model) for model in self.models]
        for model, data in zip(self.models, self.datas):
            mujoco.mj_setConst(model, data)
        self.world_nominal = np.stack([self._world_nominal() for _ in range(self.num_envs)])
        self.previous_action = np.zeros((self.num_envs, self.joints))
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.next_push = np.zeros(self.num_envs)
        self.reference_height = self._nominal_height()
        self.motions: list[dict[str, np.ndarray]] = []

    def _resolve_indices(self) -> None:
        """Look up the MuJoCo addresses of joints, actuators and feet."""
        mujoco, model, names = self.mujoco, self.base_model, self.graph.joint_names
        self.qpos_index = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
        self.dof_index = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
        self.actuator_joint = np.array(
            [names.index(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)[2:]) for a in range(model.nu)], dtype=np.int64
        )
        self.foot_geoms: dict[int, int] = {}
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if name.startswith("foot"):
                self.foot_geoms[geom] = int(model.geom_bodyid[geom])

    def _randomized_model(self):
        """Return a copy of the model with randomized friction, link masses and base center of mass."""
        model = copy.copy(self.base_model)
        settings = self.config.randomization
        if not settings.enabled:
            return model
        model.geom_friction[:, 0] = self.rng.uniform(*settings.friction)
        scale = self.rng.uniform(*settings.link_mass_scale, size=model.nbody)
        scale[0] = 1.0
        model.body_mass[:] = model.body_mass * scale
        model.body_inertia[:] = model.body_inertia * scale[:, None]
        if self.floating:
            model.body_ipos[1] += self.rng.uniform(-settings.base_com_offset, settings.base_com_offset, 3)
        return model

    def _world_nominal(self) -> np.ndarray:
        """Return one world's nominal pose, with its random joint offset."""
        settings = self.config.randomization
        if not settings.enabled:
            return self.nominal.copy()
        return self.nominal + self.rng.uniform(-settings.nominal_offset, settings.nominal_offset, self.joints)

    def _foot_clearance(self, model, data) -> float:
        """Return the base height that puts the lowest foot sphere on the floor."""
        self.mujoco.mj_kinematics(model, data)
        lowest = min(
            (data.geom_xpos[geom][2] for geom in self.foot_geoms),
            default=None,
        )
        if lowest is None:
            return float(data.qpos[2])
        return float(data.qpos[2]) - float(lowest) + self.config.geometry.radius + 0.002

    def _nominal_height(self) -> float:
        """Return the base height of the nominal pose standing on the floor."""
        if not self.floating:
            return 0.0
        data = self.mujoco.MjData(self.base_model)
        data.qpos[self.qpos_index] = self.nominal
        return self._foot_clearance(self.base_model, data)

    def set_motions(self, motions: list[dict[str, np.ndarray]]) -> None:
        """Set the motions episodes start from: dicts of ``q``, ``qd``, ``rotation``, ``linear``, ``angular``."""
        self.motions = motions

    def _set_state(self, index: int, frame: dict[str, np.ndarray] | None) -> None:
        """Put a world in a motion frame (or the nominal pose) with its feet on the floor."""
        model, data = self.models[index], self.datas[index]
        self.mujoco.mj_resetData(model, data)
        if frame is None:
            data.qpos[self.qpos_index] = self.world_nominal[index]
        else:
            data.qpos[self.qpos_index] = frame["q"]
            data.qvel[self.dof_index] = frame["qd"]
        if self.floating:
            rotation = np.eye(3) if frame is None else frame["rotation"]
            quaternion = np.zeros(4)
            self.mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
            data.qpos[3:7] = quaternion
            data.qpos[:3] = 0.0
            data.qpos[2] = self._foot_clearance(model, data)
            if frame is not None:
                data.qvel[:3] = rotation @ frame["linear"]
                data.qvel[3:6] = frame["angular"]
        data.ctrl[:] = data.qpos[self.qpos_index][self.actuator_joint]
        self.mujoco.mj_forward(model, data)
        self.previous_action[index] = 0.0
        self.steps[index] = 0
        low, high = self.config.randomization.push_interval
        self.next_push[index] = self.rng.uniform(low, high)

    def reset(self, indices: np.ndarray | None = None, frames: list[dict[str, np.ndarray]] | None = None) -> None:
        """Reset worlds to given frames, or to random motion frames (nominal pose without motions)."""
        indices = np.arange(self.num_envs) if indices is None else indices
        for position, index in enumerate(indices):
            frame = None
            if frames is not None:
                frame = frames[position]
            elif self.motions:
                motion = self.motions[int(self.rng.integers(len(self.motions)))]
                time = int(self.rng.integers(len(motion["q"])))
                frame = {key: value[time] for key, value in motion.items()}
            self._set_state(int(index), frame)

    def _read_state(self, index: int) -> tuple[np.ndarray, ...]:
        """Return one world's q, qd, gravity direction, angular velocity, linear velocity and height."""
        data = self.datas[index]
        q, qd = data.qpos[self.qpos_index].copy(), data.qvel[self.dof_index].copy()
        if not self.floating:
            return q, qd, np.array([0.0, 0.0, -1.0]), np.zeros(3), np.zeros(3), 0.0
        rotation = data.xmat[1].reshape(3, 3)
        return q, qd, -rotation[2], data.qvel[3:6].copy(), rotation.T @ data.qvel[:3], float(data.qpos[2])

    def state(self) -> RobotState:
        """Return the current state of every world."""
        rows = [self._read_state(index) for index in range(self.num_envs)]
        return RobotState(*(torch.from_numpy(np.stack([row[k] for row in rows])) for k in range(6)))

    def _physics(self, index: int, target: np.ndarray) -> None:
        """Advance one world by one control step."""
        model, data = self.models[index], self.datas[index]
        data.ctrl[:] = target[self.actuator_joint]
        for _ in range(self.substeps):
            self.mujoco.mj_step(model, data)

    def _apply_pushes(self) -> None:
        """Give due worlds a random base velocity change (in the main thread, for the random generator)."""
        settings = self.config.randomization
        if not (self.floating and settings.enabled):
            return
        elapsed = self.steps * self.config.control_dt
        for index in np.nonzero(elapsed >= self.next_push)[0]:
            data = self.datas[index]
            data.qvel[:2] += self.rng.uniform(-settings.push_linear_velocity, settings.push_linear_velocity, 2)
            data.qvel[3:6] += self.rng.uniform(-settings.push_angular_velocity, settings.push_angular_velocity, 3)
            self.next_push[index] = elapsed[index] + self.rng.uniform(*settings.push_interval)

    def _diverged(self, index: int) -> bool:
        """Return whether a world's physics blew up."""
        data = self.datas[index]
        return not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()) or float(np.abs(data.qvel).max()) > 1e3

    def _fell(self, index: int) -> bool:
        """Return whether a floating robot fell over."""
        if not self.floating:
            return False
        data = self.datas[index]
        tilt = math.degrees(math.acos(float(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1, 1))))
        return float(data.qpos[2]) < self.config.min_height_ratio * self.reference_height or tilt > self.config.max_tilt_degrees

    def _slippage(self, index: int) -> float:
        """Return the summed speed of feet in contact (BFM-Zero's ``penalty_slippage``)."""
        model, data = self.models[index], self.datas[index]
        bodies = set()
        for contact in range(data.ncon):
            item = data.contact[contact]
            for geom in (item.geom1, item.geom2):
                if geom in self.foot_geoms:
                    bodies.add(self.foot_geoms[geom])
        velocity = np.zeros(6)
        total = 0.0
        for body in bodies:
            self.mujoco.mj_objectVelocity(model, data, self.mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            total += float(np.linalg.norm(velocity[3:]))
        return total

    def step(self, actions: np.ndarray) -> EnvStep:
        """Apply normalized actions ``[N, J]`` in [-1, 1] as PD targets around the nominal pose."""
        actions = np.clip(np.nan_to_num(actions), -1.0, 1.0)
        targets = self.world_nominal + self.config.action_scale * actions
        targets = np.where(self.limited, np.clip(targets, self.lower, self.upper), targets)
        self._apply_pushes()
        if self.pool is None:
            for index in range(self.num_envs):
                self._physics(index, targets[index])
        else:
            list(self.pool.map(lambda index: self._physics(index, targets[index]), range(self.num_envs)))
        self.steps += 1
        terminated = np.array([self._diverged(index) or self._fell(index) for index in range(self.num_envs)])
        for index in np.nonzero(terminated)[0]:
            if self._diverged(int(index)):
                # Keep the buffer finite: report the reset state of a world that blew up.
                self._set_state(int(index), None)
        truncated = (self.steps >= self.config.max_episode_steps) & ~terminated
        state = self.state()
        torque = torch.from_numpy(np.stack([self.datas[i].qfrc_actuator[self.dof_index].copy() for i in range(self.num_envs)]))
        span = np.where(self.limited, self.upper - self.lower, np.inf)
        middle = 0.5 * (self.upper + self.lower)
        soft_lower = np.where(self.limited, middle - 0.5 * span * SOFT_LIMIT, -np.inf)
        soft_upper = np.where(self.limited, middle + 0.5 * span * SOFT_LIMIT, np.inf)
        q = state.q.numpy()
        outside = np.clip(soft_lower - q, 0.0, None) + np.clip(q - soft_upper, 0.0, None)
        aux_rewards = {
            "limits_dof_pos": torch.from_numpy(outside.sum(axis=1)),
            "penalty_action_rate": torch.from_numpy(((actions - self.previous_action) ** 2).sum(axis=1)),
            "penalty_slippage": torch.tensor([self._slippage(index) for index in range(self.num_envs)], dtype=torch.float64),
        }
        self.previous_action = actions.copy()
        return EnvStep(state, torque, torch.from_numpy(targets), torch.from_numpy(terminated), torch.from_numpy(truncated), aux_rewards)
