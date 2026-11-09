"""Worlds in Isaac Lab: several robots in one GPU simulation, with body and self collisions.

Written against Isaac Lab 3.0.0 (commit 4cfdfcf) with its PhysX backend, and tested only
through a fake of its API (``tests/test_isaac_env.py``), because Isaac Lab needs an NVIDIA GPU.

One simulation holds every robot. Each robot's worlds live under ``/World/robot<r>_<i>`` on
a shared grid, one ``Articulation`` per robot, as in Isaac Lab's articulation tutorial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from model.rl.envs import EmbodimentEnv, EnvConfig
from model.rl.observation import RobotState
from model.training.episodes import EmbodimentRecord


def _warp():
    """Import Warp, which Isaac Lab's PhysX views use for arrays."""
    import warp

    return warp


def matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices ``[N, 3, 3]`` to Isaac Lab 3's x, y, z, w quaternions ``[N, 4]``."""
    m = rotation
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    candidates = np.stack(
        (
            np.stack((1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 2, 1] - m[:, 1, 2]), axis=-1),
            np.stack((m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1], m[:, 0, 2] - m[:, 2, 0]), axis=-1),
            np.stack((m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], m[:, 1, 0] - m[:, 0, 1]), axis=-1),
            np.stack((m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1], 1 + trace), axis=-1),
        ),
        axis=1,
    )
    best = np.argmax(np.stack((m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], trace), axis=-1), axis=-1)
    quaternion = candidates[np.arange(m.shape[0]), best]
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion * np.where(quaternion[:, 3:] < 0, -1.0, 1.0)


def rotate_inverse_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate ``vector`` from the world into the frame of x, y, z, w quaternions ``[..., 4]``."""
    axis, w = quaternion[..., :3], quaternion[..., 3:]
    cross = torch.linalg.cross(axis, vector.expand_as(axis), dim=-1)
    return vector + 2 * torch.linalg.cross(axis, cross - w * vector.expand_as(axis), dim=-1)


@dataclass
class IsaacRobotConfig:
    """How one robot enters the simulation.

    Attributes:
        urdf (str): URDF with resolvable collision meshes.
        ros_package_paths (list[dict[str, str]]): ``{"name", "path"}`` for ``package://`` meshes.
        undesired_contact_links (list[str]): Regexes of links whose contact is penalized; by
            default every link but the feet (BFM-Zero's G1: pelvis, shoulders, hips).
        self_collisions (bool): Enable self-collisions.
    """

    urdf: str
    ros_package_paths: list[dict[str, str]] = field(default_factory=list)
    undesired_contact_links: list[str] | None = None
    self_collisions: bool = True


class IsaacEmbodimentEnv(EmbodimentEnv):
    """One robot's worlds inside a shared Isaac Lab simulation.

    Args:
        record (EmbodimentRecord): Robot with position-mode actuators.
        config (EnvConfig): Settings; ``geometry`` is unused, because meshes define collisions.
        articulation (Any): The robot's ``isaaclab.assets.Articulation``.
        contact_sensor (Any): ``isaaclab.sensors.ContactSensor`` on all of its links.
        origins (torch.Tensor): ``[N, 3]`` world origins.
        undesired_contact_links (list[str] | None): See ``IsaacRobotConfig``.
        refresh (Any): Callback that recomputes simulator state after writes.
    """

    def __init__(self, record: EmbodimentRecord, config: EnvConfig, articulation: Any, contact_sensor: Any, origins: torch.Tensor,
                 undesired_contact_links: list[str] | None = None, refresh: Any = None) -> None:
        super().__init__(record, config)
        self.articulation, self.contact_sensor = articulation, contact_sensor
        self.origins = origins
        self.device = origins.device
        self.refresh = refresh or (lambda: None)
        sim_joints = list(articulation.joint_names)
        missing = [name for name in self.graph.joint_names if name not in sim_joints]
        if missing:
            raise ValueError(f"{record.name}: joints missing from the simulated robot: {missing}")
        self.joint_map = torch.tensor([sim_joints.index(name) for name in self.graph.joint_names], device=self.device)
        extra = [name for name in sim_joints if name not in self.graph.joint_names]
        if extra:
            print(f"WARNING: {record.name}: joints without a token are not actuated and may move freely: {extra}")
        self.default_joints = articulation.data.default_joint_pos.torch.clone()
        sim_bodies = list(articulation.body_names)
        sensor_bodies = list(contact_sensor.body_names)
        feet = list(record.contact_links)
        unknown = [name for name in feet if name not in sim_bodies]
        if unknown:
            raise ValueError(f"{record.name}: contact links missing from the simulated robot: {unknown}; keep fixed joints when importing")
        self.foot_bodies = torch.tensor([sim_bodies.index(name) for name in feet], dtype=torch.long, device=self.device)
        self.foot_sensors = torch.tensor([sensor_bodies.index(name) for name in feet], dtype=torch.long, device=self.device)
        if undesired_contact_links is None:
            names = [name for name in sensor_bodies if name not in feet]
        else:
            names = [name for name in sensor_bodies if any(re.fullmatch(pattern, name) for pattern in undesired_contact_links)]
        self.undesired_sensors = torch.tensor([sensor_bodies.index(name) for name in names], dtype=torch.long, device=self.device)
        surface = record.surface_contacts if record.surface_contacts is not None else len(feet) <= 2
        self.flat_feet = bool(surface and feet)
        self._pending: list[tuple[int, dict | None]] = []
        self._pushes: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.reference_height = self._start_heights(self.nominal[None], np.eye(3)[None], np.zeros(1, dtype=bool))[0] if self.floating else 0.0
        self._randomize()

    def _start_heights(self, q: np.ndarray, rotation: np.ndarray, lying: np.ndarray) -> np.ndarray:
        """Return base heights that rest the feet on the floor, or drop a lying robot from just above it."""
        positions = torch.as_tensor(q, dtype=torch.float64)
        frames = self.graph.forward_kinematics(positions)[..., :3, 3].numpy()
        if self.record.contact_links:
            feet = self.graph.link_poses(positions, list(self.record.contact_links))[..., :3, 3].numpy()
        else:
            feet = frames
        turned_frames = np.einsum("nij,nkj->nki", rotation, frames)
        turned_feet = np.einsum("nij,nkj->nki", rotation, feet)
        standing = -turned_feet[..., 2].min(axis=1) + 0.03
        lying_height = -np.minimum(turned_frames[..., 2].min(axis=1), 0.0) + 0.1
        return np.where(lying, lying_height, standing)

    def _randomize(self) -> None:
        """Apply friction, link mass and base center-of-mass randomization to each world."""
        settings = self.config.randomization
        if not settings.enabled:
            return
        count, articulation = self.num_envs, self.articulation
        ids = torch.arange(count, device=self.device)
        masses = articulation.data.body_mass.torch.clone()
        scale = torch.as_tensor(self.rng.uniform(*settings.link_mass_scale, size=masses.shape), dtype=masses.dtype, device=masses.device)
        inertias = articulation.data.body_inertia.torch.clone()
        articulation.set_masses_index(masses=masses * scale, env_ids=ids)
        articulation.set_inertias_index(inertias=inertias * scale[..., None], env_ids=ids)
        if self.floating:
            coms = articulation.data.body_com_pose_b.torch.clone()
            offsets = self.rng.uniform(-settings.base_com_offset, settings.base_com_offset, (count, 3))
            coms[:, 0, :3] += torch.as_tensor(offsets, dtype=coms.dtype, device=coms.device)
            articulation.set_coms_index(coms=coms, env_ids=ids)
        view = articulation.root_view
        raw = view.get_material_properties()
        materials = raw if isinstance(raw, torch.Tensor) else _warp().to_torch(raw)
        friction = torch.as_tensor(self.rng.uniform(*settings.friction, size=count), dtype=materials.dtype, device=materials.device)
        materials[..., 0] = friction[:, None]
        materials[..., 1] = friction[:, None]
        indices = ids.cpu().to(torch.int32)
        if isinstance(raw, torch.Tensor):
            view.set_material_properties(materials, indices)
        else:
            warp = _warp()
            view.set_material_properties(warp.from_torch(materials, dtype=warp.float32), warp.from_torch(indices, dtype=warp.int32))

    def _write_state(self, index: int, frame: dict[str, np.ndarray] | None) -> None:
        """Queue a world's reset; ``_after_reset`` writes the batch."""
        self._pending.append((index, frame))

    def _after_reset(self) -> None:
        """Write the queued resets to the simulator in one batch."""
        if not self._pending:
            return
        indices = np.array([index for index, _ in self._pending])
        q = np.stack([self.world_nominal[index] if frame is None else frame["q"] for index, frame in self._pending])
        qd = np.stack([np.zeros(self.joints) if frame is None else frame["qd"] for _, frame in self._pending])
        rotation = np.stack([np.eye(3) if frame is None else frame["rotation"] for _, frame in self._pending])
        lying = np.array([frame is not None and bool(frame.get("lying", False)) for _, frame in self._pending])
        linear = np.stack([np.zeros(3) if frame is None else frame["linear"] for _, frame in self._pending])
        angular = np.stack([np.zeros(3) if frame is None else frame["angular"] for _, frame in self._pending])
        self._pending = []
        ids = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        articulation = self.articulation
        joint_position = self.default_joints[ids].clone()
        joint_position[:, self.joint_map] = torch.as_tensor(q, dtype=joint_position.dtype, device=self.device)
        joint_velocity = torch.zeros_like(joint_position)
        joint_velocity[:, self.joint_map] = torch.as_tensor(qd, dtype=joint_velocity.dtype, device=self.device)
        articulation.reset(env_ids=ids)
        if self.floating:
            heights = self._start_heights(q, rotation, lying)
            pose = torch.zeros((len(ids), 7), dtype=joint_position.dtype, device=self.device)
            pose[:, :3] = self.origins[ids]
            pose[:, 2] += torch.as_tensor(heights, dtype=pose.dtype, device=self.device)
            pose[:, 3:] = torch.as_tensor(matrix_to_xyzw(rotation), dtype=pose.dtype, device=self.device)
            velocity = np.concatenate((np.einsum("nij,nj->ni", rotation, linear), np.einsum("nij,nj->ni", rotation, angular)), axis=-1)
            articulation.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
            articulation.write_root_link_velocity_to_sim_index(root_velocity=torch.as_tensor(velocity, dtype=pose.dtype, device=self.device), env_ids=ids)
        articulation.write_joint_position_to_sim_index(position=joint_position, env_ids=ids)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=ids)
        articulation.actuators.target_command.set_position_index(value=joint_position, env_ids=ids)
        self.contact_sensor.reset(env_ids=ids)
        self.refresh()

    def _apply_targets(self, targets: np.ndarray) -> None:
        full = self.default_joints.clone()
        full[:, self.joint_map] = torch.as_tensor(targets, dtype=full.dtype, device=self.device)
        self.articulation.actuators.target_command.set_position_index(value=full)

    def _push(self, index: int, linear: np.ndarray, angular: np.ndarray) -> None:
        self._pushes.append((index, linear, angular))

    def prepare(self, actions: np.ndarray) -> None:
        """Apply targets, then write this step's pushes as one batch of velocity changes."""
        super().prepare(actions)
        if not self._pushes:
            return
        ids = torch.as_tensor([item[0] for item in self._pushes], dtype=torch.long, device=self.device)
        data = self.articulation.data
        velocity = torch.cat((data.root_link_lin_vel_w.torch[ids], data.root_link_ang_vel_w.torch[ids]), dim=-1).clone()
        velocity[:, :2] += torch.as_tensor(np.stack([item[1] for item in self._pushes]), dtype=velocity.dtype, device=self.device)
        velocity[:, 3:6] += torch.as_tensor(np.stack([item[2] for item in self._pushes]), dtype=velocity.dtype, device=self.device)
        self._pushes = []
        self.articulation.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)

    def read_state(self) -> RobotState:
        data = self.articulation.data

        def read(values: torch.Tensor) -> torch.Tensor:
            return values.detach().cpu().double()

        q, qd = read(data.joint_pos.torch[:, self.joint_map]), read(data.joint_vel.torch[:, self.joint_map])
        if not self.floating:
            zeros = torch.zeros((self.num_envs, 3), dtype=torch.float64)
            direction = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64).expand(self.num_envs, 3)
            return RobotState(q, qd, direction, zeros, zeros, torch.zeros(self.num_envs, dtype=torch.float64))
        return RobotState(
            q, qd, read(data.projected_gravity_b.torch), read(data.root_link_ang_vel_b.torch),
            read(data.root_link_lin_vel_b.torch), read(data.root_link_pos_w.torch[:, 2]),
        )

    def _torque(self) -> torch.Tensor:
        return self.articulation.data.applied_torque.torch[:, self.joint_map].detach().cpu().double()

    def _diverged(self) -> np.ndarray:
        data = self.articulation.data
        position, velocity = data.joint_pos.torch, data.joint_vel.torch
        bad = ~torch.isfinite(position).all(dim=1) | ~torch.isfinite(velocity).all(dim=1) | (velocity.abs().max(dim=1).values > 1e3)
        return bad.detach().cpu().numpy()

    def _net_forces(self) -> torch.Tensor:
        return self.contact_sensor.data.net_forces_w.torch

    def _foot_contacts(self) -> torch.Tensor:
        return self._net_forces()[:, self.foot_sensors].norm(dim=-1) > 1.0

    def _slippage(self) -> np.ndarray:
        """Return the summed speed of feet in contact (BFM-Zero's ``penalty_slippage``)."""
        if self.foot_bodies.numel() == 0:
            return np.zeros(self.num_envs)
        speed = self.articulation.data.body_link_lin_vel_w.torch[:, self.foot_bodies].norm(dim=-1)
        return (speed * self._foot_contacts()).sum(dim=1).detach().cpu().double().numpy()

    def _undesired_contact(self) -> np.ndarray:
        """Return which worlds have a contact above 1 N on a penalized link, with the floor or with itself."""
        if self.undesired_sensors.numel() == 0:
            return np.zeros(self.num_envs, dtype=bool)
        forces = self._net_forces()[:, self.undesired_sensors]
        return (forces.abs() > 1.0).any(dim=-1).any(dim=-1).detach().cpu().numpy()

    def _feet_orientation(self) -> np.ndarray:
        """Return the summed ``|g_xy|`` in the frame of each flat foot in contact (BFM-Zero's ``penalty_feet_ori``)."""
        if not self.flat_feet:
            return np.zeros(self.num_envs)
        quaternion = self.articulation.data.body_link_quat_w.torch[:, self.foot_bodies]
        down = torch.tensor([0.0, 0.0, -1.0], dtype=quaternion.dtype, device=quaternion.device)
        tilt = rotate_inverse_xyzw(quaternion, down)[..., :2].norm(dim=-1)
        return (tilt * self._foot_contacts()).sum(dim=1).detach().cpu().double().numpy()


class IsaacGroup:
    """Step every robot of one Isaac Lab simulation together.

    Args:
        simulation (Any): ``isaaclab.sim.SimulationContext``.
        envs (list[IsaacEmbodimentEnv]): Every robot's environment.
        substeps (int): Physics steps per control step.
    """

    def __init__(self, simulation: Any, envs: list[IsaacEmbodimentEnv], substeps: int) -> None:
        self.simulation, self.envs, self.substeps = simulation, envs, substeps
        self.physics_dt = float(simulation.get_physics_dt())

    def _update(self, dt: float) -> None:
        for env in self.envs:
            env.articulation.update(dt)
            env.contact_sensor.update(dt)

    def refresh(self) -> None:
        """Recompute simulator state after writing resets or pushes."""
        self.simulation.forward()
        self._update(0.0)

    def advance(self) -> None:
        """Run the physics of one control step for every robot."""
        for _ in range(self.substeps):
            for env in self.envs:
                env.articulation.write_data_to_sim()
            self.simulation.step(render=False)
            self._update(self.physics_dt)


def build_isaac_group(robots: list[tuple[EmbodimentRecord, EnvConfig, IsaacRobotConfig]], simulation: Any, spacing: float = 5.0) -> IsaacGroup:
    """Create the scene for every robot and return its environments.

    Call this inside ``isaaclab.app.launch_simulation``.

    Args:
        robots (list[tuple]): Robot, environment settings and simulator settings of each robot.
        simulation (Any): ``isaaclab.sim.SimulationContext``.
        spacing (float): Distance between worlds in m; robots of different worlds must not touch.

    Returns:
        IsaacGroup: The environments.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    total = sum(config.num_envs for _, config, _ in robots)
    side = int(np.ceil(np.sqrt(total)))
    grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), dim=-1).reshape(-1, 2)[:total].float() * spacing
    origins = torch.cat((grid, torch.zeros(total, 1)), dim=-1)
    cursor = 0
    created = []
    for robot_index, (record, config, settings) in enumerate(robots):
        names = list(record.graph.joint_names)
        actuator = record.actuator
        joint_gain = {name: float(actuator["kp"][i]) for i, name in enumerate(names)}
        joint_damping = {name: float(actuator["kd"][i]) for i, name in enumerate(names)}
        joint_effort = {name: float(actuator["peak_torque"][i]) for i, name in enumerate(names)}
        robot_origins = origins[cursor : cursor + config.num_envs]
        cursor += config.num_envs
        for index, origin in enumerate(robot_origins.tolist()):
            sim_utils.create_prim(f"/World/robot{robot_index}_{index}", "Xform", translation=origin)
        spawn = sim_utils.UrdfFileCfg(
            asset_path=settings.urdf,
            fix_base=not record.graph.floating_base,
            merge_fixed_joints=False,
            self_collision=settings.self_collisions,
            ros_package_paths=settings.ros_package_paths,
            activate_contact_sensors=True,
            articulation_props=[
                PhysxArticulationCfg(enabled_self_collisions=settings.self_collisions, solver_position_iteration_count=4, solver_velocity_iteration_count=0)
            ],
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
        )
        articulation = Articulation(
            ArticulationCfg(
                prim_path=f"/World/robot{robot_index}_.*/Robot",
                spawn=spawn,
                actuators={
                    "all": ImplicitActuatorCfg(
                        joint_names_expr=[".*"], stiffness=joint_gain, damping=joint_damping, actuator_effort_limit=joint_effort, armature=config.armature
                    )
                },
            )
        )
        sensor = ContactSensor(ContactSensorCfg(prim_path=f"/World/robot{robot_index}_.*/Robot/.*", update_period=0.0, history_length=0))
        created.append((record, config, settings, articulation, sensor, robot_origins))
    simulation.reset()
    device = torch.device(simulation.device)
    envs: list[IsaacEmbodimentEnv] = []
    group = IsaacGroup(simulation, envs, max(1, int(round(robots[0][1].control_dt / robots[0][1].physics_dt))))
    for record, config, settings, articulation, sensor, robot_origins in created:
        envs.append(
            IsaacEmbodimentEnv(record, config, articulation, sensor, robot_origins.to(device), settings.undesired_contact_links, group.refresh)
        )
    return group
