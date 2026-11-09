"""Ground-contact simulation with MuJoCo, built from an embodiment graph.

``build_mjcf`` turns an ``EmbodimentGraph`` into MJCF: one MuJoCo body per
token with the graph's merged inertia, the root body with a free joint, one
hinge or slide joint per DoF, position actuators (PD inside the simulator, so
stiff gains stay stable), and collision spheres at the contact links (one
sphere for point feet, four corner spheres for flat feet) above a floor plane.
Only the feet collide, so contact is foot-ground only.

``simulate_ground`` drives the robot around a standing pose: smoothly varying
joint targets (crouching, leaning, shifting weight), random base pushes, and
randomized floor friction. A plain PD controller cannot balance a biped, so
bipeds get a "spotter" by default: a soft spring-damper on the base that pulls
it upright and over its start point, like a person steadying the robot. The
feet still carry the weight, and the spotter's wrench is recorded as a known
external wrench, so the labels stay exact. Everything recorded is exact for the simulated
system: applied joint torques, joint and base accelerations, contact flags
and forces, the push wrench, and the accelerations of probe torques computed
with the contacts in place. Episodes end early if the robot falls.

Requires ``mujoco`` (``uv sync --extra sim``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from model.embodiment import EmbodimentGraph
from model.physics import ActuatorSpec, actuator_tensors
from model.training.episodes import EmbodimentRecord, Trajectory

GRAVITY = 9.80665


@dataclass
class ContactGeometry:
    """Foot collision geometry, in each contact link's frame.

    Attributes:
        radius (float): Sphere radius in m.
        sole_offset (tuple): Offset from the link frame to the sole center in m.
        foot_size (tuple): Length and width of a flat foot in m.
    """

    radius: float = 0.02
    sole_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    foot_size: tuple[float, float] = (0.16, 0.06)


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to a w, x, y, z quaternion."""
    trace = np.trace(rotation)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return np.array([0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s, (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s])
    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        return np.array([(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s])
    if index == 1:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        return np.array([(rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s])
    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
    return np.array([(rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s])


def _vector(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _inertial(mass: float, com, inertia) -> str:
    inertia = np.asarray(inertia, dtype=np.float64)
    inertia = 0.5 * (inertia + inertia.T) + 1e-8 * np.eye(3)
    full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return f'<inertial pos="{_vector(com)}" mass="{max(float(mass), 1e-4):.9g}" fullinertia="{_vector(full)}"/>'


def foot_points(record: EmbodimentRecord, geometry: ContactGeometry, surface: bool) -> list[tuple[int, int, np.ndarray]]:
    """Return ``(contact index, carrying body, point in body frame)`` sphere centers."""
    graph = record.graph
    points = []
    offsets = [np.zeros(3)]
    if surface:
        half_length, half_width = geometry.foot_size[0] / 2, geometry.foot_size[1] / 2
        offsets = [np.array([sx * half_length, sy * half_width, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
    for contact, link in enumerate(record.contact_links):
        index = graph.link_names.index(link)
        body = int(graph.link_bodies[index])
        transform = graph.link_transforms[index].numpy()
        for offset in offsets:
            point = transform[:3, 3] + transform[:3, :3] @ (np.asarray(geometry.sole_offset) + offset)
            points.append((contact, body, point))
    return points


def build_mjcf(
    record: EmbodimentRecord,
    kp: np.ndarray,
    kd: np.ndarray,
    geometry: ContactGeometry,
    friction: float = 0.9,
    timestep: float = 0.002,
) -> str:
    """Write MJCF for a robot with foot contacts and position actuators.

    The root gets a free joint only if the graph has a floating base.

    Args:
        record (EmbodimentRecord): Robot, with ``contact_links`` if it has feet.
        kp (np.ndarray): ``[J]`` position gains.
        kd (np.ndarray): ``[J]`` velocity gains.
        geometry (ContactGeometry): Foot geometry.
        friction (float): Floor sliding friction.
        timestep (float): Physics step in s.

    Returns:
        str: MJCF XML.
    """
    graph = record.graph
    surface = record.surface_contacts if record.surface_contacts is not None else len(record.contact_links) <= 2
    actuator = actuator_tensors(record.specs)
    spheres: dict[int, list[str]] = {}
    for contact, body, point in foot_points(record, geometry, surface):
        spheres.setdefault(body, []).append(
            f'<geom name="foot{contact}_{len(spheres.get(body, []))}" type="sphere" size="{geometry.radius}" '
            f'pos="{_vector(point)}" friction="{friction} 0.02 0.002" condim="3"/>'
        )
    children: dict[int, list[int]] = {}
    for index, parent in enumerate(graph.parent_indices.tolist()):
        children.setdefault(parent, []).append(index)

    def body_xml(index: int, indent: str) -> str:
        transform = graph.tree_transforms[index].numpy()
        name = graph.joint_names[index]
        kind = "slide" if graph.joint_types[index] == "prismatic" else "hinge"
        limited = graph.joint_types[index] != "continuous" and float(graph.upper[index]) > float(graph.lower[index])
        limits = f' range="{float(graph.lower[index]):.6g} {float(graph.upper[index]):.6g}" limited="true"' if limited else ' limited="false"'
        lines = [
            f'{indent}<body name="b_{name}" pos="{_vector(transform[:3, 3])}" quat="{_vector(_quaternion(transform[:3, :3]))}">',
            f'{indent}  <joint name="{name}" type="{kind}" axis="{_vector(graph.axes[index].numpy())}"{limits} '
            f'damping="{float(graph.damping[index]):.6g}" frictionloss="{float(graph.friction[index]):.6g}" armature="1e-5"/>',
            f"{indent}  " + _inertial(float(graph.body_mass[index]), graph.body_com[index].numpy(), graph.body_inertia[index].numpy()),
        ]
        lines += [f"{indent}  {geom}" for geom in spheres.get(index, [])]
        for child in children.get(index, []):
            lines.append(body_xml(child, indent + "  "))
        lines.append(f"{indent}</body>")
        return "\n".join(lines)

    base_lines = ['    <body name="base" pos="0 0 1">']
    if graph.floating_base:
        base_lines.append("      <freejoint/>")
    base_lines.append("      " + _inertial(max(graph.base_mass, 1e-3), graph.base_com.numpy(), graph.base_inertia.numpy()))
    base_lines += [f"      {geom}" for geom in spheres.get(-1, [])]
    base_lines += [body_xml(child, "      ") for child in children.get(-1, [])]
    base_lines.append("    </body>")
    actuators = []
    for index, name in enumerate(graph.joint_names):
        peak = float(actuator["peak_torque"][index])
        if peak <= 0:
            continue
        actuators.append(
            f'    <position name="a_{name}" joint="{name}" kp="{kp[index]:.6g}" kv="{kd[index]:.6g}" '
            f'forcelimited="true" forcerange="{-peak:.6g} {peak:.6g}" ctrllimited="false"/>'
        )
    return "\n".join(
        [
            f'<mujoco model="{graph.name}">',
            '  <compiler angle="radian"/>',
            f'  <option timestep="{timestep}" gravity="0 0 {-GRAVITY}" integrator="implicitfast"/>',
            '  <default><geom contype="1" conaffinity="0"/></default>',
            "  <worldbody>",
            f'    <geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="1" friction="{friction} 0.02 0.002"/>',
            *base_lines,
            "  </worldbody>",
            "  <actuator>",
            *actuators,
            "  </actuator>",
            "</mujoco>",
        ]
    )


def _rotate_to_base(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation.T @ vector, rotation


def simulate_ground(
    embodiment: int,
    record: EmbodimentRecord,
    settings: dict[str, Any],
    generator: torch.Generator,
) -> list[Trajectory]:
    """Simulate a legged robot standing and moving on flat ground.

    Settings keys: ``episodes``, ``steps``, ``dt`` (control period),
    ``timestep`` (physics), ``probes``, ``interval`` (target change period),
    ``target_amplitude`` (fraction of joint range), ``gain_scale``,
    ``push_probability`` and ``push_force`` (fraction of weight),
    ``friction`` ([low, high]), ``nominal_pose`` (joint name to position),
    ``geometry`` (``ContactGeometry``), ``probe_range``, ``probe_minimum``,
    ``assist`` (spotter stiffness as a fraction of ``m g h`` per rad of tilt;
    0.6 for two feet or fewer, 0 otherwise).

    Args:
        embodiment (int): Record index.
        record (EmbodimentRecord): Robot (floating base, with contact links).
        settings (dict): See above.
        generator (torch.Generator): Random source.

    Returns:
        list[Trajectory]: Episodes with exact labels.
    """
    import mujoco

    graph = record.graph
    if not record.contact_links:
        raise ValueError(f"{record.name}: ground simulation needs contact_links")
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
    actuator = actuator_tensors(record.specs)
    joints = graph.num_joints
    peak = actuator["peak_torque"].numpy()
    kp = settings.get("gain_scale", 1.0) * 2.0 * peak  # peak torque at 0.5 rad of error
    kd = 0.03 * kp
    lower, upper = graph.lower.numpy(), graph.upper.numpy()
    limited = np.array([kind != "continuous" for kind in graph.joint_types]) & (upper > lower)
    nominal = np.zeros(joints)
    for name, value in (settings.get("nominal_pose") or {}).items():
        nominal[graph.joint_names.index(name)] = value
    nominal = np.where(limited, np.clip(nominal, lower, upper), nominal)
    # Target offsets scale with each joint's range, capped at 1 rad so joints
    # with very wide ranges do not swing wildly.
    half = np.minimum(np.where(limited, (upper - lower) / 2, 0.5), 1.0)
    geometry = settings.get("geometry") or ContactGeometry()
    dt, timestep = settings["dt"], settings.get("timestep", 0.002)
    substeps = max(1, int(round(dt / timestep)))
    surface = record.surface_contacts if record.surface_contacts is not None else len(record.contact_links) <= 2
    feet = foot_points(record, geometry, surface)
    trajectories = []
    for episode in range(settings["episodes"]):
        friction = float(rng.uniform(*settings.get("friction", [0.5, 1.2])))
        model = mujoco.MjModel.from_xml_string(build_mjcf(record, kp, kd, geometry, friction, timestep))
        data = mujoco.MjData(model)
        probe_data = mujoco.MjData(model)
        foot_geoms = {}
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if name.startswith("foot"):
                foot_geoms[geom] = int(name[4:].split("_")[0])
        joint_qpos = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in graph.joint_names])
        joint_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in graph.joint_names])
        actuator_joint = np.array(
            [graph.joint_names.index(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)[2:]) for a in range(model.nu)]
        )
        start = nominal + np.where(limited, 0.05 * half * rng.uniform(-1, 1, joints), 0.0)
        data.qpos[joint_qpos] = start
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[2] = 0.0
        mujoco.mj_kinematics(model, data)
        lowest = min(
            (data.xpos[model.geom_bodyid[geom]] + data.xmat[model.geom_bodyid[geom]].reshape(3, 3) @ model.geom_pos[geom])[2]
            for geom in foot_geoms
        )
        data.qpos[2] = -lowest + geometry.radius + 0.005
        yaw = rng.uniform(-math.pi, math.pi)
        data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        target = start.copy()
        height0 = float(data.qpos[2])
        anchor = data.qpos[0:3].copy()
        assist = settings.get("assist")
        if assist is None:
            assist = 0.6 if len(record.contact_links) <= 2 else 0.0
        mgh = graph.total_mass * GRAVITY * max(height0, 0.1)
        omega_n = math.sqrt(GRAVITY / max(height0, 0.1))
        push = np.zeros(3)
        push_left = 0
        log: dict[str, list] = {key: [] for key in ("q", "qd", "qdd", "tau", "cmd", "R", "x", "v", "w", "a", "alpha", "contact", "ext", "probe", "probe_exec", "probe_acc", "probe_base")}
        settle = int(0.3 / dt)
        for step in range(settings["steps"] + settle):
            if step >= settle and (step - settle) % settings["interval"] == 0:
                offset = settings.get("target_amplitude", 0.15) * half * rng.uniform(-1, 1, joints)
                target = np.where(limited, np.clip(nominal + offset, lower, upper), nominal + offset)
            if push_left == 0 and step >= settle and rng.uniform() < settings.get("push_probability", 0.02):
                direction = rng.normal(size=3)
                direction[2] *= 0.2
                push = settings.get("push_force", 0.3) * graph.total_mass * GRAVITY * direction / np.linalg.norm(direction)
                push_left = int(0.1 / dt)
            push_left = max(push_left - 1, 0)
            current_push = push if push_left > 0 else np.zeros(3)

            def apply_external() -> None:
                data.xfrc_applied[1, :3] = current_push
                data.xfrc_applied[1, 3:] = 0.0
                if assist > 0:
                    # Spotter: torque toward upright, horizontal spring to the start point.
                    rotation = data.xmat[1].reshape(3, 3)
                    tilt = np.cross(rotation[:, 2], np.array([0.0, 0.0, 1.0]))
                    angular = rotation @ data.qvel[3:6]
                    angular[2] = 0.0
                    data.xfrc_applied[1, 3:] = assist * mgh * (tilt - 0.15 / omega_n * angular)
                    horizontal = data.qpos[0:3] - anchor
                    velocity = data.qvel[0:3].copy()
                    horizontal[2] = velocity[2] = 0.0
                    data.xfrc_applied[1, :3] += -assist * graph.total_mass * omega_n**2 * (horizontal + 0.5 / omega_n * velocity)

            apply_external()
            command = target
            data.ctrl[:] = command[actuator_joint]
            mujoco.mj_forward(model, data)
            if step >= settle:
                quaternion = data.qpos[3:7].copy()
                linear, rotation = _rotate_to_base(quaternion, data.qvel[0:3])
                acceleration, _ = _rotate_to_base(quaternion, data.qacc[0:3])
                contact = np.zeros(len(record.contact_links), dtype=bool)
                for index in range(data.ncon):
                    item = data.contact[index]
                    for geom in (item.geom1, item.geom2):
                        if geom in foot_geoms:
                            contact[foot_geoms[geom]] = True
                external_force, _ = _rotate_to_base(quaternion, data.xfrc_applied[1, :3])
                external_torque, _ = _rotate_to_base(quaternion, data.xfrc_applied[1, 3:])
                center = data.xipos[1] - data.xpos[1]
                center_base, _ = _rotate_to_base(quaternion, center)
                log["q"].append(data.qpos[joint_qpos].copy())
                log["qd"].append(data.qvel[joint_dof].copy())
                log["qdd"].append(data.qacc[joint_dof].copy())
                log["tau"].append(data.qfrc_actuator[joint_dof].copy())
                log["cmd"].append(command.copy())
                log["R"].append(rotation)
                log["x"].append(data.qpos[0:3].copy())
                log["v"].append(linear)
                log["w"].append(data.qvel[3:6].copy())
                log["a"].append(acceleration)
                log["alpha"].append(data.qacc[3:6].copy())
                log["contact"].append(contact)
                log["ext"].append(np.concatenate([external_force, np.cross(center_base, external_force) + external_torque]))
                probes, executed, accelerations, base_accelerations = [], [], [], []
                for _ in range(settings.get("probes", 0)):
                    fraction = math.exp(rng.uniform(math.log(settings.get("probe_minimum", 0.02)), math.log(settings.get("probe_range", 2.0))))
                    torque = rng.choice([-1.0, 1.0], joints) * fraction * peak * rng.uniform(0, 1, joints)
                    probe_data.qpos[:] = data.qpos
                    probe_data.qvel[:] = data.qvel
                    probe_data.xfrc_applied[:] = data.xfrc_applied
                    q_now, qd_now = data.qpos[joint_qpos], data.qvel[joint_dof]
                    probe_data.ctrl[:] = (q_now + (torque + kd * qd_now) / np.maximum(kp, 1e-9))[actuator_joint]
                    mujoco.mj_forward(model, probe_data)
                    probe_acc, _ = _rotate_to_base(quaternion, probe_data.qacc[0:3])
                    probes.append(torque)
                    executed.append(probe_data.qfrc_actuator[joint_dof].copy())
                    accelerations.append(probe_data.qacc[joint_dof].copy())
                    base_accelerations.append(np.concatenate([probe_acc, probe_data.qacc[3:6]]))
                if probes:
                    log["probe"].append(np.stack(probes))
                    log["probe_exec"].append(np.stack(executed))
                    log["probe_acc"].append(np.stack(accelerations))
                    log["probe_base"].append(np.stack(base_accelerations))
            for _ in range(substeps):
                apply_external()
                mujoco.mj_step(model, data)
            up = data.xmat[1].reshape(3, 3)[2, 2]
            if step >= settle and (data.qpos[2] < 0.5 * height0 or up < math.cos(math.radians(50))):
                break
        if len(log["q"]) < 20:
            continue

        def tensor(key):
            return torch.from_numpy(np.asarray(log[key], dtype=np.float64))

        trajectories.append(
            Trajectory(
                embodiment=embodiment,
                dt=dt,
                positions=tensor("q"),
                velocities=tensor("qd"),
                accelerations=tensor("qdd"),
                applied_torque=tensor("tau"),
                actions=tensor("cmd"),
                action_type="position",
                base_orientation=tensor("R"),
                base_position=tensor("x"),
                base_linear_velocity=tensor("v"),
                base_angular_velocity=tensor("w"),
                base_linear_acceleration=tensor("a"),
                base_angular_acceleration=tensor("alpha"),
                contacts=torch.from_numpy(np.asarray(log["contact"])),
                base_external_wrench=tensor("ext"),
                exact_dynamics=True,
                probe_torques=tensor("probe") if log["probe"] else None,
                probe_executed=tensor("probe_exec") if log["probe"] else None,
                probe_accelerations=tensor("probe_acc") if log["probe"] else None,
                probe_base_accelerations=tensor("probe_base") if log["probe"] else None,
            )
        )
    return trajectories


def ground_specs(specs: tuple[ActuatorSpec, ...], gain_scale: float = 1.0) -> tuple[ActuatorSpec, ...]:
    """Return specs in position mode with the simulator's PD gains."""
    from dataclasses import replace

    output = []
    for spec in specs:
        if spec.mode == "passive":
            output.append(spec)
            continue
        kp = gain_scale * 2.0 * spec.peak_torque
        output.append(replace(spec, mode="position", kp=kp, kd=0.03 * kp))
    return tuple(output)
