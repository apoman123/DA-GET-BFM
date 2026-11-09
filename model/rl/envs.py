"""Parallel environments of one robot, with one implementation per simulator.

``EmbodimentEnv`` holds what BFM-Zero's environment defines independently of the simulator:
PD position control around a nominal pose at 50 Hz, 500-step episodes without fall
termination, starts from prioritized motion frames with 30% lying-down starts, domain
randomization (nominal pose, pushes; simulators add friction, mass, center of mass), and
the auxiliary rewards. ``MujocoEmbodimentEnv`` runs on CPU from an embodiment graph, for
tests and small checks. Training at scale uses ``model.rl.isaac_env``.
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
from model.training.ground_sim import ContactGeometry, build_mjcf

SOFT_LIMIT = 0.95  # BFM-Zero's soft_dof_pos_limit
QUARTER_TURN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


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
    """Settings of an environment.

    Attributes:
        num_envs (int): Parallel worlds.
        control_dt (float): Control period in s (BFM-Zero: 50 Hz).
        physics_dt (float): Physics step in s (BFM-Zero: 200 Hz).
        max_episode_steps (int): Episode length in control steps.
        action_scale (float): Target offset in rad at ``|action| = 1`` (BFM-Zero: 0.25 x 5).
        fall_start_probability (float): Episodes that start lying down (BFM-Zero: 0.3).
        terminate_on_fall (bool): End episodes when fallen (BFM-Zero: no, it must get up).
        min_height_ratio (float): Fallen below this fraction of the nominal height.
        max_tilt_degrees (float): Fallen above this base tilt.
        armature (float): Rotor inertia of every joint in kg m^2, which URDFs lack. Without
            it, light links with stiff gains make MuJoCo diverge (G1 wrists at 200 Hz).
        nominal_pose (dict[str, float]): Joint name to nominal position; others are 0.
        geometry (ContactGeometry): Foot collision geometry (MuJoCo only).
        randomization (DomainRandomization): Domain randomization.
        seed (int): Random seed.
    """

    num_envs: int = 16
    control_dt: float = 0.02
    physics_dt: float = 0.005
    max_episode_steps: int = 500
    action_scale: float = 1.25
    fall_start_probability: float = 0.3
    terminate_on_fall: bool = False
    min_height_ratio: float = 0.4
    max_tilt_degrees: float = 70.0
    armature: float = 0.01
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
        terminated (torch.Tensor): Bool ``[N]``: diverged, or fell if that ends episodes.
        truncated (torch.Tensor): Bool ``[N]``: reached the time limit.
        aux_rewards (dict[str, torch.Tensor]): ``[N]`` unscaled auxiliary rewards.
    """

    state: RobotState
    torque: torch.Tensor
    target: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    aux_rewards: dict[str, torch.Tensor]


class EmbodimentEnv:
    """Simulator-independent logic of one robot's worlds.

    Simulators implement ``_write_state``, ``_apply_targets``, ``_push``, ``read_state``,
    ``_torque``, ``_diverged``, ``_slippage``, ``_undesired_contact`` and ``_feet_orientation``.
    Stepping has three phases so that several robots can share one simulation:
    ``prepare`` (all robots), one physics advance, ``collect`` (all robots).

    Args:
        record (EmbodimentRecord): Robot with position-mode actuators.
        config (EnvConfig): Settings.
    """

    def __init__(self, record: EmbodimentRecord, config: EnvConfig) -> None:
        self.record = record
        self.config = config
        self.graph = record.graph
        self.num_envs = config.num_envs
        self.joints = self.graph.num_joints
        self.floating = bool(self.graph.floating_base)
        self.rng = np.random.default_rng(config.seed)
        if bool((record.actuator["mode"] == ACTUATOR_MODES.index("thrust")).any()):
            raise NotImplementedError(f"{record.name}: rotor (thrust) robots are not supported by the RL environments")
        self.lower, self.upper = self.graph.lower.numpy(), self.graph.upper.numpy()
        self.limited = np.array([kind != "continuous" for kind in self.graph.joint_types]) & (self.upper > self.lower)
        nominal = np.zeros(self.joints)
        for name, value in config.nominal_pose.items():
            nominal[self.graph.joint_names.index(name)] = value
        self.nominal = np.where(self.limited, np.clip(nominal, self.lower, self.upper), nominal)
        self.world_nominal = np.stack([self._world_nominal() for _ in range(self.num_envs)])
        self.previous_action = np.zeros((self.num_envs, self.joints))
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.next_push = np.zeros(self.num_envs)
        self.evaluating = False
        self.motions: list[dict[str, np.ndarray]] = []
        self.motion_weights: np.ndarray | None = None
        self.reference_height = 0.0
        self._targets = np.zeros((self.num_envs, self.joints))
        self._actions = np.zeros((self.num_envs, self.joints))

    def _world_nominal(self) -> np.ndarray:
        """Return one world's nominal pose, with its random joint offset."""
        settings = self.config.randomization
        if not settings.enabled:
            return self.nominal.copy()
        return self.nominal + self.rng.uniform(-settings.nominal_offset, settings.nominal_offset, self.joints)

    def set_motions(self, motions: list[dict[str, np.ndarray]], weights: np.ndarray | None = None) -> None:
        """Set the motions episodes start from, with optional sampling weights."""
        self.motions = motions
        self.set_motion_weights(weights)

    def set_motion_weights(self, weights: np.ndarray | None) -> None:
        """Set how often each motion starts an episode (prioritization by tracking error)."""
        if weights is None:
            self.motion_weights = None
            return
        weights = np.asarray(weights, dtype=np.float64)
        self.motion_weights = weights / weights.sum()

    def _lie_down(self, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Turn a motion frame by a quarter turn about x, onto the back or the front (BFM-Zero's start)."""
        sign = 1.0 if self.rng.uniform() < 0.5 else -1.0
        turn = QUARTER_TURN.copy()
        turn[1, 2] *= sign
        turn[2, 1] *= sign
        return {**frame, "rotation": turn @ frame["rotation"], "lying": True}

    def sample_frame(self) -> dict[str, np.ndarray] | None:
        """Sample a start frame: a prioritized motion, a uniform time, maybe lying down."""
        if not self.motions:
            return None
        motion = self.motions[int(self.rng.choice(len(self.motions), p=self.motion_weights))]
        time = int(self.rng.integers(len(motion["q"])))
        frame = {key: value[time] for key, value in motion.items()}
        if self.floating and self.rng.uniform() < self.config.fall_start_probability:
            frame = self._lie_down(frame)
        return frame

    def reset(self, indices: np.ndarray | None = None, frames: list[dict | None] | None = None) -> None:
        """Reset worlds to given frames, or to sampled ones."""
        indices = np.arange(self.num_envs) if indices is None else indices
        for position, index in enumerate(indices):
            frame = frames[position] if frames is not None else self.sample_frame()
            self._write_state(int(index), frame)
            self.previous_action[index] = 0.0
            self.steps[index] = 0
            low, high = self.config.randomization.push_interval
            self.next_push[index] = self.rng.uniform(low, high)
        self._after_reset()

    def _after_reset(self) -> None:
        """Let the simulator refresh its state after writes."""

    def prepare(self, actions: np.ndarray) -> None:
        """Turn normalized actions ``[N, J]`` in [-1, 1] into PD targets and apply pushes."""
        actions = np.clip(np.nan_to_num(actions), -1.0, 1.0)
        targets = self.world_nominal + self.config.action_scale * actions
        self._targets = np.where(self.limited, np.clip(targets, self.lower, self.upper), targets)
        self._actions = actions
        self._apply_targets(self._targets)
        settings = self.config.randomization
        if self.floating and settings.enabled and not self.evaluating:
            elapsed = self.steps * self.config.control_dt
            for index in np.nonzero(elapsed >= self.next_push)[0]:
                linear = self.rng.uniform(-settings.push_linear_velocity, settings.push_linear_velocity, 2)
                angular = self.rng.uniform(-settings.push_angular_velocity, settings.push_angular_velocity, 3)
                self._push(int(index), linear, angular)
                self.next_push[index] = elapsed[index] + self.rng.uniform(*settings.push_interval)

    def fallen(self, state: RobotState) -> np.ndarray:
        """Return which worlds' base is down or tilted."""
        if not self.floating:
            return np.zeros(self.num_envs, dtype=bool)
        minimum_up = math.cos(math.radians(self.config.max_tilt_degrees))
        low = state.height.numpy() < self.config.min_height_ratio * self.reference_height
        return low | (-state.gravity_direction[:, 2].numpy() < minimum_up)

    def collect(self) -> EnvStep:
        """Read the results of the physics advance."""
        self.steps += 1
        diverged = self._diverged()
        for index in np.nonzero(diverged)[0]:
            self._write_state(int(index), None)  # Keep the buffer finite.
        if diverged.any():
            self._after_reset()
        state = self.read_state()
        terminated = diverged | (self.fallen(state) if self.config.terminate_on_fall else False)
        truncated = (self.steps >= self.config.max_episode_steps) & ~terminated
        span = np.where(self.limited, self.upper - self.lower, np.inf)
        middle = np.where(self.limited, 0.5 * (self.upper + self.lower), 0.0)
        soft_lower = np.where(self.limited, middle - 0.5 * span * SOFT_LIMIT, -np.inf)
        soft_upper = np.where(self.limited, middle + 0.5 * span * SOFT_LIMIT, np.inf)
        q = state.q.numpy()
        outside = np.clip(soft_lower - q, 0.0, None) + np.clip(q - soft_upper, 0.0, None)
        aux_rewards = {
            "limits_dof_pos": torch.from_numpy(outside.sum(axis=1)),
            "penalty_action_rate": torch.from_numpy(((self._actions - self.previous_action) ** 2).sum(axis=1)),
            "penalty_slippage": torch.from_numpy(self._slippage()),
            "penalty_undesired_contact": torch.from_numpy(self._undesired_contact().astype(np.float64)),
            "penalty_feet_ori": torch.from_numpy(self._feet_orientation()),
        }
        self.previous_action = self._actions.copy()
        return EnvStep(state, self._torque(), torch.from_numpy(self._targets.copy()), torch.from_numpy(terminated), torch.from_numpy(truncated), aux_rewards)

    def _write_state(self, index: int, frame: dict[str, np.ndarray] | None) -> None:
        raise NotImplementedError

    def _apply_targets(self, targets: np.ndarray) -> None:
        raise NotImplementedError

    def _push(self, index: int, linear: np.ndarray, angular: np.ndarray) -> None:
        raise NotImplementedError

    def read_state(self) -> RobotState:
        """Return the current state of every world."""
        raise NotImplementedError

    def _torque(self) -> torch.Tensor:
        raise NotImplementedError

    def _diverged(self) -> np.ndarray:
        raise NotImplementedError

    def _slippage(self) -> np.ndarray:
        raise NotImplementedError

    def _undesired_contact(self) -> np.ndarray:
        raise NotImplementedError

    def _feet_orientation(self) -> np.ndarray:
        raise NotImplementedError


class MujocoEmbodimentEnv(EmbodimentEnv):
    """Worlds in MuJoCo, built from the embodiment graph.

    Links collide with the floor through one sphere each, and feet through their contact
    geometry. Meshes are not used and links do not collide with each other, so self-contact
    needs ``model.rl.isaac_env``.

    Args:
        record (EmbodimentRecord): Robot with position-mode actuators.
        config (EnvConfig): Settings.
        pool (ThreadPoolExecutor | None): Shared workers; worlds step serially if None.
    """

    def __init__(self, record: EmbodimentRecord, config: EnvConfig, pool: ThreadPoolExecutor | None = None) -> None:
        import mujoco

        super().__init__(record, config)
        self.mujoco = mujoco
        self.pool = pool
        self.substeps = max(1, int(round(config.control_dt / config.physics_dt)))
        actuator = record.actuator
        radius = max(0.02, 0.05 * float(self.graph.characteristic_length))
        xml = build_mjcf(record, actuator["kp"].numpy(), actuator["kd"].numpy(), config.geometry, 0.9, config.physics_dt, body_collision_radius=radius, armature=config.armature)
        self.base_model = mujoco.MjModel.from_xml_string(xml)
        self._resolve_indices()
        self.models = [self._randomized_model() for _ in range(self.num_envs)]
        self.datas = [mujoco.MjData(model) for model in self.models]
        for model, data in zip(self.models, self.datas):
            mujoco.mj_setConst(model, data)
        self.reference_height = self._nominal_height()

    def _resolve_indices(self) -> None:
        """Look up the MuJoCo addresses of joints, actuators and collision geoms."""
        mujoco, model, names = self.mujoco, self.base_model, self.graph.joint_names
        self.qpos_index = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
        self.dof_index = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
        self.actuator_joint = np.array(
            [names.index(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)[2:]) for a in range(model.nu)], dtype=np.int64
        )
        self.foot_geoms: dict[int, int] = {}
        self.link_geoms: set[int] = set()
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if name.startswith("foot"):
                self.foot_geoms[geom] = int(model.geom_bodyid[geom])
            elif name.startswith("link"):
                self.link_geoms.add(geom)
        self.foot_bodies = sorted(set(self.foot_geoms.values()))
        surface = self.record.surface_contacts if self.record.surface_contacts is not None else len(self.record.contact_links) <= 2
        self.flat_feet = bool(surface and self.record.contact_links)

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

    def _clearance(self, model, data, geoms) -> float:
        """Return the base height that puts the lowest of some geoms on the floor."""
        self.mujoco.mj_kinematics(model, data)
        lowest = min((float(data.geom_xpos[geom][2] - model.geom_size[geom, 0]) for geom in geoms), default=None)
        return float(data.qpos[2]) if lowest is None else float(data.qpos[2]) - lowest + 0.002

    def _nominal_height(self) -> float:
        """Return the base height of the nominal pose standing on the floor."""
        if not self.floating:
            return 0.0
        data = self.mujoco.MjData(self.base_model)
        data.qpos[self.qpos_index] = self.nominal
        return self._clearance(self.base_model, data, self.foot_geoms or self.link_geoms)

    def _write_state(self, index: int, frame: dict[str, np.ndarray] | None) -> None:
        """Put a world in a frame (the nominal pose if None), resting on the floor."""
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
            lying = frame is not None and frame.get("lying", False)
            data.qpos[2] = self._clearance(model, data, set(self.foot_geoms) | self.link_geoms if lying else self.foot_geoms or self.link_geoms)
            if frame is not None:
                data.qvel[:3] = rotation @ frame["linear"]
                data.qvel[3:6] = frame["angular"]
        data.ctrl[:] = data.qpos[self.qpos_index][self.actuator_joint]
        self.mujoco.mj_forward(model, data)

    def _apply_targets(self, targets: np.ndarray) -> None:
        for index, data in enumerate(self.datas):
            data.ctrl[:] = targets[index, self.actuator_joint]

    def _push(self, index: int, linear: np.ndarray, angular: np.ndarray) -> None:
        data = self.datas[index]
        data.qvel[:2] += linear
        data.qvel[3:6] += angular

    def _physics(self, index: int) -> None:
        for _ in range(self.substeps):
            self.mujoco.mj_step(self.models[index], self.datas[index])

    def advance(self) -> None:
        """Run the physics of one control step."""
        if self.pool is None:
            for index in range(self.num_envs):
                self._physics(index)
        else:
            list(self.pool.map(self._physics, range(self.num_envs)))

    def step(self, actions: np.ndarray) -> EnvStep:
        """Apply actions and advance one control step."""
        self.prepare(actions)
        self.advance()
        return self.collect()

    def read_state(self) -> RobotState:
        rows = [self._read_world(index) for index in range(self.num_envs)]
        return RobotState(*(torch.from_numpy(np.stack([row[k] for row in rows])) for k in range(6)))

    def _read_world(self, index: int) -> tuple[np.ndarray, ...]:
        data = self.datas[index]
        q, qd = data.qpos[self.qpos_index].copy(), data.qvel[self.dof_index].copy()
        if not self.floating:
            return q, qd, np.array([0.0, 0.0, -1.0]), np.zeros(3), np.zeros(3), 0.0
        rotation = data.xmat[1].reshape(3, 3)
        return q, qd, -rotation[2], data.qvel[3:6].copy(), rotation.T @ data.qvel[:3], float(data.qpos[2])

    def _torque(self) -> torch.Tensor:
        return torch.from_numpy(np.stack([data.qfrc_actuator[self.dof_index].copy() for data in self.datas]))

    def _diverged(self) -> np.ndarray:
        return np.array(
            [not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()) or float(np.abs(data.qvel).max()) > 1e3 for data in self.datas]
        )

    def _contact_geoms(self, data) -> set[int]:
        return {geom for contact in range(data.ncon) for geom in (data.contact[contact].geom1, data.contact[contact].geom2)}

    def _slippage(self) -> np.ndarray:
        """Return the summed speed of feet in contact (BFM-Zero's ``penalty_slippage``)."""
        totals = np.zeros(self.num_envs)
        velocity = np.zeros(6)
        for index, (model, data) in enumerate(zip(self.models, self.datas)):
            bodies = {self.foot_geoms[geom] for geom in self._contact_geoms(data) if geom in self.foot_geoms}
            for body in bodies:
                self.mujoco.mj_objectVelocity(model, data, self.mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
                totals[index] += float(np.linalg.norm(velocity[3:]))
        return totals

    def _undesired_contact(self) -> np.ndarray:
        """Return which worlds have a non-foot link on the floor."""
        return np.array([bool(self._contact_geoms(data) & self.link_geoms) for data in self.datas])

    def _feet_orientation(self) -> np.ndarray:
        """Return the summed ``|g_xy|`` in the frame of each flat foot in contact (BFM-Zero's ``penalty_feet_ori``)."""
        if not self.flat_feet:
            return np.zeros(self.num_envs)
        totals = np.zeros(self.num_envs)
        for index, data in enumerate(self.datas):
            bodies = {self.foot_geoms[geom] for geom in self._contact_geoms(data) if geom in self.foot_geoms}
            for body in bodies:
                totals[index] += float(np.linalg.norm(data.xmat[body].reshape(3, 3)[2, :2]))
        return totals


class MujocoGroup:
    """Step several robots' MuJoCo worlds; each advances on its own."""

    def __init__(self, envs: list[MujocoEmbodimentEnv]) -> None:
        self.envs = envs

    def advance(self) -> None:
        """Run the physics of one control step for every robot."""
        for env in self.envs:
            env.advance()
