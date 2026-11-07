"""Trajectory sources: domain-randomized simulation and recorded datasets.

Two source types are supported in the data config (see ``model.training.dataset``).

``"synthetic"``
    Simulates a URDF robot with the analytic dynamics of ``model.dynamics``.
    Each randomized embodiment perturbs link masses and inertias, link lengths,
    friction and damping, and the torque and speed limits. Three base modes
    exist:

    * ``fixed``: the base is bolted down; gravity is tilted per episode.
    * ``floating``: the base moves freely, hanging from a harness that holds the
      whole-body center of mass against gravity. The harness applies no moment,
      so base rotation and base-limb coupling are fully dynamic.
    * ``flight``: a multirotor flies under its own rotor thrust (every
      actuated joint must be a ``thrust`` rotor), tracking random waypoints
      with a geometric attitude controller.

    Commands are clipped to the torque-speed envelope before they are applied,
    so labels use the executed torque, never the unexecuted proposal. Probe
    torques at each state give several action-conditioned samples per pose.

``"huggingface"``
    Loads recorded trajectories from the Hugging Face Hub or a local folder or
    file (LeRobot parquet layout, or any parquet, CSV, or JSON table with
    per-frame vector columns). A joint name list maps state columns to URDF
    joints; joints the dataset does not record are frozen. Optional base
    columns (orientation, position, twist, projected gravity) and contact
    columns support floating-base robots. Missing velocities and accelerations
    are finite-differenced, and missing torques are estimated by inverse
    dynamics (floating base with contact forces when the base is floating).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.dynamics import (
    STANDARD_GRAVITY,
    floating_forward_dynamics,
    floating_mass_matrix,
    forward_dynamics,
    gravity_efforts,
    quaternion_to_matrix,
    rotor_wrenches,
)
from model.embodiment import EmbodimentGraph, JointFeatureConfig
from model.physics import ActuatorDefaults, actuator_specs, actuator_tensors, torque_limit
from model.physics_schema import ACTUATOR_MODES
from model.training.episodes import EmbodimentRecord, Trajectory, exp_rotation
from model.urdf import UrdfInertial, UrdfRobot, parse_urdf, rpy_to_matrix

DEFAULT_RANDOMIZATION = {
    "mass": [0.8, 1.2],
    "length": [0.9, 1.1],
    "damping": [0.0, 0.02],
    "friction": [0.0, 0.02],
    "torque": [0.8, 1.2],
    "velocity": [0.8, 1.2],
    "gravity_tilt_degrees": 20.0,
}
BASE_MODES = ("fixed", "floating", "flight")


def resolve_path(path: str, base: Path | None) -> Path:
    """Resolve a config path relative to the config file."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate


def _graph_kwargs(source: dict[str, Any], robot: UrdfRobot) -> dict[str, Any]:
    """Collect EmbodimentGraph.from_urdf options from a source config."""
    frozen = dict(source.get("frozen_joints", {}))
    if source.get("joint_names") and source.get("freeze_unlisted_joints", True):
        listed = set(source["joint_names"])
        for name, joint in robot.joints.items():
            if joint.joint_type in ("revolute", "continuous", "prismatic") and name not in listed:
                frozen.setdefault(name, 0.0)
    return {
        "joint_order": source.get("joint_names") or source.get("joint_order"),
        "include_mimic_joints": source.get("include_mimic_joints", False),
        "multi_dof_joints": source.get("multi_dof_joints", "skip"),
        "frozen_joints": frozen,
        "floating_base": source.get("floating_base"),
    }


def _actuator_defaults(source: dict[str, Any], mode: str) -> ActuatorDefaults:
    values = dict(source.get("actuator_defaults", {}))
    values.setdefault("mode", mode)
    return ActuatorDefaults(**values)


def _record(name: str, graph: EmbodimentGraph, specs, source: dict[str, Any]) -> EmbodimentRecord:
    """Build an embodiment record with the source's task-E and contact settings."""
    return EmbodimentRecord.build(
        name,
        graph,
        specs,
        source.get("end_effectors"),
        source.get("contact_links"),
        source.get("surface_contacts"),
    )


def randomize_robot(robot: UrdfRobot, randomization: dict[str, Any], generator: torch.Generator) -> UrdfRobot:
    """Return a copy of a robot with perturbed inertia, lengths, and friction.

    Damping and friction are added relative to each joint's own limits, so the
    same ranges suit a finger, a knee, and a propeller: ``damping`` is a
    fraction of ``effort / velocity`` (the damping that would use the whole
    torque at full speed) and ``friction`` a fraction of ``effort``. Joints
    without both limits are left unchanged.

    Args:
        robot (UrdfRobot): Nominal robot.
        randomization (dict): Ranges ``mass``, ``length`` (scale factors) and
            ``damping``, ``friction`` (fractions, see above).
        generator (torch.Generator): Random source.

    Returns:
        UrdfRobot: Perturbed robot.
    """

    def uniform(bounds) -> float:
        low, high = bounds
        return float(low + (high - low) * torch.rand((), generator=generator, dtype=torch.float64))

    length = uniform(randomization.get("length", [1.0, 1.0]))
    links = {}
    for name, link in robot.links.items():
        inertial = link.inertial
        if inertial is not None:
            scale = uniform(randomization.get("mass", [1.0, 1.0]))
            origin = inertial.origin.clone()
            origin[:3, 3] *= length
            inertial = UrdfInertial(
                mass=inertial.mass * scale,
                origin=origin,
                inertia=inertial.inertia * scale * length**2,
            )
        links[name] = replace(link, inertial=inertial)
    joints = {}
    for name, joint in robot.joints.items():
        origin = joint.origin.clone()
        origin[:3, 3] *= length
        effort, speed = joint.effort or 0.0, joint.velocity or 0.0
        damping, friction = joint.damping, joint.friction
        if effort > 0 and speed > 0:
            damping += uniform(randomization.get("damping", [0.0, 0.0])) * effort / speed
            friction += uniform(randomization.get("friction", [0.0, 0.0])) * effort
        joints[name] = replace(joint, origin=origin, damping=damping, friction=friction)
    return replace(robot, links=links, joints=joints)


def _tilted_rotation(count: int, degrees: float, generator: torch.Generator) -> torch.Tensor:
    """Sample rotations that tilt the z axis by up to ``degrees``."""
    tilt = torch.rand(count, generator=generator, dtype=torch.float64) * math.radians(degrees)
    heading = torch.rand(count, generator=generator, dtype=torch.float64) * 2 * math.pi
    axis = torch.stack((-heading.sin(), heading.cos(), torch.zeros_like(heading)), dim=-1)
    return exp_rotation(axis * tilt[:, None])


def center_of_mass(graph: EmbodimentGraph, positions: torch.Tensor) -> torch.Tensor:
    """Return the whole-body center of mass ``[..., 3]`` in the base frame."""
    frames = graph.forward_kinematics(positions)
    centers = (frames[..., :3, :3] @ graph.body_com[:, :, None].to(positions)).squeeze(-1) + frames[..., :3, 3]
    moment = (graph.body_mass.to(positions)[:, None] * centers).sum(-2)
    moment = moment + graph.base_mass * graph.base_com.to(positions)
    return moment / max(graph.total_mass, 1e-9)


def synthetic_source(
    source: dict[str, Any], base: Path | None, feature_config: JointFeatureConfig
) -> tuple[list[EmbodimentRecord], list[Trajectory]]:
    """Simulate domain-randomized trajectories of one URDF robot.

    Source keys: ``urdf`` (required), ``name``, ``randomized_embodiments`` (4;
    0 keeps only the nominal robot), ``episodes_per_embodiment`` (8), ``steps``
    (200), ``dt`` (0.01 s), ``substeps`` (2; flight uses at least 10),
    ``probe_torques`` (2), ``target_interval`` (25 steps),
    ``natural_frequency`` ([4, 10] rad/s), ``command_noise`` (0.05 rad of
    equivalent position error), ``probe_range`` and ``probe_minimum`` (probe
    torques are log-uniform between 0.02 and 2.0 times peak torque, with random
    sign), ``randomization`` (see ``DEFAULT_RANDOMIZATION``), ``seed``, and
    ``base_modes``: a list cycled over the randomized embodiments, from
    ``fixed``, ``floating``, ``flight``. The default is ``["fixed"]``, or
    ``["flight"]`` for robots whose actuators are all ``thrust`` rotors. The
    graph and actuator keys shared with other sources also apply.

    Args:
        source (dict): Source config.
        base (Path | None): Directory for relative paths.
        feature_config (JointFeatureConfig): Observation hardware features.

    Returns:
        tuple: Embodiment records and trajectories (indices local to this source).
    """
    robot = parse_urdf(resolve_path(source["urdf"], base))
    name = source.get("name", robot.name)
    generator = torch.Generator().manual_seed(int(source.get("seed", 0)))
    randomization = {**DEFAULT_RANDOMIZATION, **source.get("randomization", {})}
    variants = int(source.get("randomized_embodiments", 4))
    settings = {
        "episodes": int(source.get("episodes_per_embodiment", 8)),
        "steps": int(source.get("steps", 200)),
        "dt": float(source.get("dt", 0.01)),
        "substeps": int(source.get("substeps", 2)),
        "probes": int(source.get("probe_torques", 2)),
        "interval": int(source.get("target_interval", 25)),
        "frequency": source.get("natural_frequency", [4.0, 10.0]),
        "noise": float(source.get("command_noise", 0.05)),
        "probe_range": float(source.get("probe_range", 2.0)),
        "probe_minimum": float(source.get("probe_minimum", 0.02)),
        "tilt": float(randomization["gravity_tilt_degrees"]),
    }
    records, trajectories = [], []
    robots = [robot] if variants == 0 else [randomize_robot(robot, randomization, generator) for _ in range(variants)]
    modes = source.get("base_modes")
    for variant, variant_robot in enumerate(robots):
        kwargs = _graph_kwargs(source, robot)
        probe_graph = EmbodimentGraph.from_urdf(variant_robot, feature_config, **{**kwargs, "floating_base": False})
        specs = actuator_specs(probe_graph, source.get("actuator_overrides"), _actuator_defaults(source, "torque"))
        thrust = [spec.mode == "thrust" for spec in specs]
        mode_list = modes or (["flight"] if all(thrust) else ["fixed"])
        mode = mode_list[variant % len(mode_list)]
        if mode not in BASE_MODES:
            raise ValueError(f"base mode must be one of {BASE_MODES}")
        if mode == "flight" and not all(spec.mode in ("thrust", "passive") for spec in specs):
            raise ValueError(f"{name}: flight mode needs every actuated joint to be a thrust rotor")
        kwargs["floating_base"] = mode != "fixed"
        graph = EmbodimentGraph.from_urdf(variant_robot, feature_config, **kwargs)
        if variants:
            torque_scale = torch.empty(graph.num_joints, dtype=torch.float64).uniform_(
                *randomization["torque"], generator=generator
            )
            speed_scale = torch.empty(graph.num_joints, dtype=torch.float64).uniform_(
                *randomization["velocity"], generator=generator
            )
            specs = tuple(
                spec
                if spec.mode in ("passive", "thrust")
                else replace(
                    spec,
                    peak_torque=spec.peak_torque * float(torque_scale[index]),
                    continuous_torque=spec.continuous_torque * float(torque_scale[index]),
                    max_velocity=spec.max_velocity * float(speed_scale[index]),
                    max_power=spec.max_power * float(torque_scale[index] * speed_scale[index]),
                )
                for index, spec in enumerate(specs)
            )
        record = _record(f"{name}/{variant}-{mode}", graph, specs, source)
        records.append(record)
        simulate = {"fixed": _simulate_fixed, "floating": _simulate_floating, "flight": _simulate_flight}[mode]
        trajectories += simulate(len(records) - 1, record, settings, generator)
    return records, trajectories


def _joint_targets(graph: EmbodimentGraph, episodes: int, generator: torch.Generator, spread: float = 0.85):
    """Return a sampler of random joint targets and the limited-joint mask."""
    lower, upper = graph.lower, graph.upper
    limited = torch.tensor([kind != "continuous" for kind in graph.joint_types]) & (upper > lower)
    center = torch.where(limited, (lower + upper) / 2, torch.zeros_like(lower))
    half = torch.where(limited, (upper - lower) / 2, torch.full_like(lower, math.pi))

    def sample() -> torch.Tensor:
        unit = torch.rand((episodes, graph.num_joints), generator=generator, dtype=torch.float64) * 2 - 1
        return center + spread * half * unit

    return sample, limited


def _probe_torques(peak: torch.Tensor, shape, settings, generator) -> torch.Tensor:
    """Sample probe torques: random sign, log-uniform fraction of peak torque."""
    fraction = torch.exp(
        torch.empty(shape, dtype=torch.float64).uniform_(
            math.log(settings["probe_minimum"]), math.log(settings["probe_range"]), generator=generator
        )
    )
    sign = torch.randint(0, 2, shape, generator=generator).to(torch.float64) * 2 - 1
    return sign * fraction * peak


def effective_joint_inertia(graph: EmbodimentGraph, positions: torch.Tensor) -> torch.Tensor:
    """Diagonal of the joint inertia with the base free (Schur complement).

    For a floating base each joint drives less inertia than when the base is
    bolted down, ``M_jj - M_jb M_bb^-1 M_bj``; using the fixed-base diagonal
    for PD gains would make the free-floating system too stiff and unstable.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` q.

    Returns:
        torch.Tensor: ``[..., J]`` effective inertia.
    """
    full = floating_mass_matrix(graph, positions)
    base, coupling, joint = full[..., :6, :6], full[..., :6, 6:], full[..., 6:, 6:]
    effective = joint - coupling.transpose(-1, -2) @ torch.linalg.solve(base, coupling)
    return effective.diagonal(dim1=-2, dim2=-1).clamp_min(1e-9)


def _pd_command(graph, q, qd, target, limited, omega, zeta, noise, gravity, generator, floating=False):
    """PD toward targets, scaled by each joint's inertia, with gravity feed-forward."""
    error = target - q
    if not bool(limited.all()):
        error = torch.where(limited, error, torch.remainder(error + math.pi, 2 * math.pi) - math.pi)
    diagonal = (
        effective_joint_inertia(graph, q) if floating else graph.mass_matrix(q).diagonal(dim1=-2, dim2=-1)
    )
    # Exploration noise is a position error of ``noise`` rad (scaled by the
    # joint's own inertia), so light distal joints are not overdriven.
    jitter = noise * torch.randn(q.shape, generator=generator, dtype=torch.float64)
    return gravity_efforts(graph, q, gravity) + diagonal * (omega**2 * (error + jitter) - 2 * zeta * omega * qd)


def _clamp_limits(q, qd, limited, lower, upper, valid):
    """Stop joints at their limits and mark the transition invalid."""
    outside = limited & ((q < lower) | (q > upper))
    if bool(outside.any()):
        valid = valid & ~outside.any(dim=1)
        q = torch.where(limited, torch.clamp(q, lower, upper), q)
        qd = torch.where(outside, torch.zeros_like(qd), qd)
    return q, qd, valid


def _simulate_fixed(embodiment: int, record: EmbodimentRecord, settings, generator) -> list[Trajectory]:
    """Simulate a bolted-down robot tracking random joint targets."""
    graph = record.graph
    episodes, dt, substeps = settings["episodes"], settings["dt"], settings["substeps"]
    actuator = actuator_tensors(record.specs)
    sample, limited = _joint_targets(graph, episodes, generator)
    rotation = _tilted_rotation(episodes, settings["tilt"], generator)
    gravity = (rotation.transpose(-1, -2) @ torch.tensor([0.0, 0.0, -STANDARD_GRAVITY], dtype=torch.float64)[:, None]).squeeze(-1)
    q, qd, target = sample(), None, sample()
    qd = torch.zeros_like(q)
    omega = torch.empty((episodes, 1), dtype=torch.float64).uniform_(*settings["frequency"], generator=generator)
    zeta = torch.empty((episodes, 1), dtype=torch.float64).uniform_(0.6, 1.0, generator=generator)
    log: dict[str, list[torch.Tensor]] = {key: [] for key in ("q", "qd", "qdd", "tau", "cmd", "probe", "valid")}
    for step in range(settings["steps"]):
        if step and step % settings["interval"] == 0:
            target = sample()
        command = _pd_command(graph, q, qd, target, limited, omega, zeta, settings["noise"], gravity, generator)
        limit = torque_limit(actuator, qd)
        applied = torch.maximum(torch.minimum(command, limit), -limit)
        acceleration = forward_dynamics(graph, q, qd, applied, gravity)
        log["probe"].append(_probe_torques(actuator["peak_torque"], (episodes, settings["probes"], graph.num_joints), settings, generator))
        for key, value in (("q", q), ("qd", qd), ("qdd", acceleration), ("tau", applied), ("cmd", command)):
            log[key].append(value)
        valid = torch.ones(episodes, dtype=torch.bool)
        for substep in range(substeps):
            if substep:
                acceleration = forward_dynamics(graph, q, qd, applied, gravity)
            qd = qd + dt / substeps * acceleration
            q = q + dt / substeps * qd
            q, qd, valid = _clamp_limits(q, qd, limited, graph.lower, graph.upper, valid)
        log["valid"].append(valid)
    stacked = {key: torch.stack(values, dim=1) for key, values in log.items()}
    return [
        Trajectory(
            embodiment=embodiment,
            dt=dt,
            positions=stacked["q"][episode],
            velocities=stacked["qd"][episode],
            accelerations=stacked["qdd"][episode],
            applied_torque=stacked["tau"][episode],
            actions=stacked["cmd"][episode],
            action_type="torque",
            gravity=gravity[episode],
            probe_torques=stacked["probe"][episode] if settings["probes"] else None,
            transition_valid=stacked["valid"][episode],
        )
        for episode in range(episodes)
    ]


def _floating_loop(record: EmbodimentRecord, settings, generator, controller, harness: bool):
    """Integrate a floating-base robot; ``controller`` returns joint commands."""
    graph = record.graph
    episodes, dt = settings["episodes"], settings["dt"]
    substeps = settings["substeps"]
    actuator = actuator_tensors(record.specs)
    world_gravity = torch.tensor([0.0, 0.0, -STANDARD_GRAVITY], dtype=torch.float64)
    state = controller.initial_state()
    rotation = state["rotation"]
    position = torch.zeros((episodes, 3), dtype=torch.float64)
    linear = 0.05 * torch.randn((episodes, 3), generator=generator, dtype=torch.float64)
    angular = 0.2 * torch.randn((episodes, 3), generator=generator, dtype=torch.float64)
    q, qd = state["q"], state["qd"]
    log: dict[str, list[torch.Tensor]] = {
        key: []
        for key in ("q", "qd", "qdd", "tau", "cmd", "probe", "valid", "R", "x", "v", "w", "a", "alpha", "ext")
    }

    def accelerations(q, qd, rotation, linear, angular, applied):
        gravity = (rotation.transpose(-1, -2) @ world_gravity[:, None]).squeeze(-1)
        external = None
        if harness:
            force = -graph.total_mass * gravity
            external = torch.cat((force, torch.linalg.cross(center_of_mass(graph, q), force, dim=-1)), dim=-1)
        result = floating_forward_dynamics(
            graph, q, qd, applied, gravity, linear, angular,
            rotor_wrenches(graph, actuator, q, qd), external,
        )
        return result, external

    for step in range(settings["steps"]):
        command, logged = controller(step, q, qd, rotation, position, linear, angular)
        limit = torque_limit(actuator, qd)
        applied = torch.maximum(torch.minimum(command, limit), -limit)
        (linear_acc, angular_acc, acceleration), external = accelerations(q, qd, rotation, linear, angular, applied)
        log["probe"].append(_probe_torques(actuator["peak_torque"], (episodes, settings["probes"], graph.num_joints), settings, generator))
        for key, value in (
            ("q", q), ("qd", qd), ("qdd", acceleration), ("tau", applied), ("cmd", logged),
            ("R", rotation), ("x", position), ("v", linear), ("w", angular),
            ("a", linear_acc), ("alpha", angular_acc),
            ("ext", torch.zeros((episodes, 6), dtype=torch.float64) if external is None else external),
        ):
            log[key].append(value)
        valid = torch.ones(episodes, dtype=torch.bool)
        sub = dt / substeps
        for substep in range(substeps):
            if substep:
                (linear_acc, angular_acc, acceleration), _ = accelerations(q, qd, rotation, linear, angular, applied)
            angular = angular + sub * angular_acc
            linear = linear + sub * (linear_acc - torch.linalg.cross(angular, linear, dim=-1))
            position = position + sub * (rotation @ linear[..., None]).squeeze(-1)
            rotation = rotation @ exp_rotation(sub * angular)
            qd = qd + sub * acceleration
            q = q + sub * qd
            q, qd, valid = _clamp_limits(q, qd, controller.limited, graph.lower, graph.upper, valid)
        log["valid"].append(valid)
    return {key: torch.stack(values, dim=1) for key, values in log.items()}


def _floating_trajectories(embodiment, record, settings, stacked, action_type) -> list[Trajectory]:
    contacts = None
    if record.contact_links:
        contacts = torch.zeros((settings["steps"], len(record.contact_links)), dtype=torch.bool)
    return [
        Trajectory(
            embodiment=embodiment,
            dt=settings["dt"],
            positions=stacked["q"][episode],
            velocities=stacked["qd"][episode],
            accelerations=stacked["qdd"][episode],
            applied_torque=stacked["tau"][episode],
            actions=stacked["cmd"][episode],
            action_type=action_type,
            base_orientation=stacked["R"][episode],
            base_position=stacked["x"][episode],
            base_linear_velocity=stacked["v"][episode],
            base_angular_velocity=stacked["w"][episode],
            base_linear_acceleration=stacked["a"][episode],
            base_angular_acceleration=stacked["alpha"][episode],
            contacts=contacts,
            probe_torques=stacked["probe"][episode] if settings["probes"] else None,
            transition_valid=stacked["valid"][episode],
            base_external_wrench=stacked["ext"][episode],
        )
        for episode in range(settings["episodes"])
    ]


class _JointController:
    """PD joint controller for a floating robot on a center-of-mass harness."""

    def __init__(self, record, settings, generator):
        self.graph = record.graph
        self.settings = settings
        self.generator = generator
        self.sample, self.limited = _joint_targets(self.graph, settings["episodes"], generator, 0.5)
        episodes = settings["episodes"]
        # A free base reacts to every joint motion, so track targets more gently.
        low, high = settings["frequency"]
        self.omega = torch.empty((episodes, 1), dtype=torch.float64).uniform_(low / 2, high / 2, generator=generator)
        self.zeta = torch.empty((episodes, 1), dtype=torch.float64).uniform_(0.6, 1.0, generator=generator)
        self.target = self.sample()

    def initial_state(self):
        q = self.sample()
        return {
            "q": q,
            "qd": torch.zeros_like(q),
            "rotation": _tilted_rotation(self.settings["episodes"], self.settings["tilt"], self.generator),
        }

    def __call__(self, step, q, qd, rotation, position, linear, angular):
        if step and step % self.settings["interval"] == 0:
            self.target = self.sample()
        world = torch.tensor([0.0, 0.0, -STANDARD_GRAVITY], dtype=torch.float64)
        gravity = (rotation.transpose(-1, -2) @ world[:, None]).squeeze(-1)
        command = _pd_command(
            self.graph, q, qd, self.target, self.limited, self.omega, self.zeta,
            self.settings["noise"], gravity, self.generator, floating=True,
        )
        return command, command


def _simulate_floating(embodiment, record, settings, generator) -> list[Trajectory]:
    """Simulate a free-floating robot held by a center-of-mass harness."""
    settings = {**settings, "substeps": max(settings["substeps"], 4)}
    controller = _JointController(record, settings, generator)
    stacked = _floating_loop(record, settings, generator, controller, harness=True)
    return _floating_trajectories(embodiment, record, settings, stacked, "torque")


class _FlightController:
    """Geometric multirotor controller that tracks random waypoints.

    Desired thrust and body moments come from position and attitude PD laws.
    They are mapped to rotor speeds through the mixing matrix of the actual
    rotor layout, so any multirotor (quad, hexa, coaxial) works. Rotor spin
    directions alternate around the base to make yaw controllable.
    """

    def __init__(self, record, settings, generator):
        self.graph = record.graph
        self.settings = settings
        self.generator = generator
        self.actuator = actuator_tensors(record.specs)
        self.rotors = self.actuator["mode"] == ACTUATOR_MODES.index("thrust")
        self.limited = torch.zeros(self.graph.num_joints, dtype=torch.bool)
        q0 = torch.zeros(self.graph.num_joints, dtype=torch.float64)
        frames, _, axes = self.graph._kinematics(q0)
        points = frames[:, :3, 3]
        angle = torch.atan2(points[:, 1], points[:, 0])
        order = torch.argsort(torch.where(self.rotors, angle, torch.full_like(angle, 10.0)))
        spin = torch.zeros(self.graph.num_joints, dtype=torch.float64)
        sign = 1.0
        for index in order.tolist():
            if bool(self.rotors[index]):
                spin[index] = sign
                sign = -sign
        self.spin = spin
        k_f = self.actuator["thrust_coefficient"]
        k_m = self.actuator["torque_coefficient"]
        force = axes * k_f[:, None]
        moment = torch.linalg.cross(points, force, dim=-1) - (spin * k_m)[:, None] * axes
        wrench = torch.cat((force, moment), dim=-1).T  # [6, J]
        self.mixer = torch.linalg.pinv(wrench[2:])  # thrust z and three moments
        self.max_square = self.actuator["max_velocity"].pow(2)
        full = floating_mass_matrix(self.graph, q0)
        self.mass = self.graph.total_mass
        self.inertia = full[3:6, 3:6]
        episodes = settings["episodes"]
        self.waypoint = torch.zeros((episodes, 3), dtype=torch.float64)
        self.yaw = torch.zeros(episodes, dtype=torch.float64)

    def initial_state(self):
        episodes = self.settings["episodes"]
        hover = self.mixer @ torch.tensor([self.mass * STANDARD_GRAVITY, 0.0, 0.0, 0.0], dtype=torch.float64)
        speed = self.spin * hover.clamp_min(0).sqrt()
        return {
            "q": torch.zeros((episodes, self.graph.num_joints), dtype=torch.float64),
            "qd": speed.expand(episodes, -1).clone(),
            "rotation": _tilted_rotation(episodes, 10.0, self.generator),
        }

    def __call__(self, step, q, qd, rotation, position, linear, angular):
        episodes = self.settings["episodes"]
        if step % self.settings["interval"] == 0:
            self.waypoint = (torch.rand((episodes, 3), generator=self.generator, dtype=torch.float64) * 2 - 1)
            heading = rotation[..., 0, :2]
            current = torch.atan2(heading[..., 1], heading[..., 0])
            turn = (torch.rand(episodes, generator=self.generator, dtype=torch.float64) * 2 - 1) * 0.5
            self.yaw = current + turn
        velocity_world = (rotation @ linear[..., None]).squeeze(-1)
        gravity_world = torch.tensor([0.0, 0.0, -STANDARD_GRAVITY], dtype=torch.float64)
        desired = 4.0 * (self.waypoint - position) - 3.0 * velocity_world - gravity_world
        force = self.mass * desired
        z_axis = rotation[..., :, 2]
        thrust = (force * z_axis).sum(-1)
        z_desired = force / force.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        heading = torch.stack((self.yaw.cos(), self.yaw.sin(), torch.zeros_like(self.yaw)), dim=-1)
        y_desired = torch.linalg.cross(z_desired, heading, dim=-1)
        y_desired = y_desired / y_desired.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x_desired = torch.linalg.cross(y_desired, z_desired, dim=-1)
        target = torch.stack((x_desired, y_desired, z_desired), dim=-1)
        error_matrix = 0.5 * (target.transpose(-1, -2) @ rotation - rotation.transpose(-1, -2) @ target)
        error = torch.stack((error_matrix[..., 2, 1], error_matrix[..., 0, 2], error_matrix[..., 1, 0]), dim=-1)
        spin = (self.inertia @ angular[..., None]).squeeze(-1)
        gains = torch.tensor([60.0, 60.0, 8.0], dtype=torch.float64)
        damping = torch.tensor([12.0, 12.0, 4.0], dtype=torch.float64)
        moment = (self.inertia @ (-gains * error - damping * angular)[..., None]).squeeze(-1)
        moment = moment + torch.linalg.cross(angular, spin, dim=-1)
        # Thrust and roll/pitch first; yaw uses whatever rotor headroom is left,
        # because propeller drag gives far less yaw authority.
        primary = (self.mixer[:, :3] @ torch.cat((thrust[:, None], moment[:, :2]), dim=-1)[..., None]).squeeze(-1)
        primary = torch.minimum(primary.clamp_min(0), self.max_square)
        yaw = self.mixer[:, 3] * moment[:, 2:3]
        room_up = (self.max_square - primary) / yaw.clamp_min(1e-12)
        room_down = primary / (-yaw).clamp_min(1e-12)
        scale = torch.where(yaw > 0, room_up, torch.where(yaw < 0, room_down, torch.full_like(yaw, 1.0)))
        scale = scale.min(dim=-1, keepdim=True).values.clamp(0.0, 1.0)
        squares = torch.minimum((primary + scale * yaw).clamp_min(0), self.max_square)
        speed_command = self.spin * squares.sqrt()
        kd = self.actuator["kd"]
        k_m = self.actuator["torque_coefficient"]
        command = kd * (speed_command - qd) + k_m * speed_command * speed_command.abs()
        command = torch.where(self.rotors, command, torch.zeros_like(command))
        return command, torch.where(self.rotors, speed_command, torch.zeros_like(speed_command))


def _simulate_flight(embodiment, record, settings, generator) -> list[Trajectory]:
    """Simulate a multirotor flying between random waypoints."""
    flight_settings = {**settings, "substeps": max(settings["substeps"], 10)}
    controller = _FlightController(record, flight_settings, generator)
    stacked = _floating_loop(record, flight_settings, generator, controller, harness=False)
    return _floating_trajectories(embodiment, record, flight_settings, stacked, "velocity")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth ``[T, D]`` values along time with edge padding."""
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack(
        [np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])],
        axis=1,
    )[: values.shape[0]]


def _keys(key: str | list[str] | None) -> list[str]:
    """Return the column names of a key that may list several columns."""
    if key is None:
        return []
    return [key] if isinstance(key, str) else list(key)


def _column(table: dict[str, Any], key: str | list[str] | None, indices: list | None) -> np.ndarray | None:
    """Read vector columns (concatenated in order) and select entries.

    ``None`` entries in ``indices`` become NaN columns.
    """
    if key is None:
        return None
    parts = []
    for name in _keys(key):
        if name not in table:
            raise KeyError(f"dataset has no column {name!r}; columns are {sorted(table)}")
        parts.append(np.stack([np.asarray(row, dtype=np.float64).reshape(-1) for row in table[name]]))
    values = np.concatenate(parts, axis=1)
    if indices is None:
        return values
    output = np.full((values.shape[0], len(indices)), np.nan)
    for target, index in enumerate(indices):
        if index is not None:
            output[:, target] = values[:, index]
    return output


def _load_table(source: dict[str, Any], base: Path | None):
    """Load a Hub dataset or local parquet, CSV, or JSON files."""
    from datasets import load_dataset

    if "repo_id" in source:
        return load_dataset(
            source["repo_id"],
            data_files=source.get("data_files", "data/*/*.parquet"),
            split=source.get("split", "train"),
            revision=source.get("revision"),
        )
    path = resolve_path(source["path"], base)
    files = source.get("data_files") or (str(path / "data" / "*" / "*.parquet") if path.is_dir() else str(path))
    if isinstance(files, str) and path.is_dir() and not Path(files).is_absolute():
        files = str(path / files)
    first = files if isinstance(files, str) else files[0]
    builder = {"csv": "csv", "json": "json", "jsonl": "json"}.get(first.rsplit(".", 1)[-1], "parquet")
    return load_dataset(builder, data_files=files, split="train")


def _orientation(values: np.ndarray, kind: str) -> torch.Tensor:
    """Convert recorded orientations to world-from-base rotation matrices."""
    tensor = torch.from_numpy(values)
    if kind in ("wxyz", "xyzw"):
        return quaternion_to_matrix(tensor, kind)
    if kind == "rpy":
        return rpy_to_matrix(tensor)
    if kind == "matrix":
        return tensor.reshape(-1, 3, 3)
    raise ValueError("base_orientation_format must be wxyz, xyzw, rpy, or matrix")


def huggingface_source(
    source: dict[str, Any], base: Path | None, feature_config: JointFeatureConfig
) -> tuple[list[EmbodimentRecord], list[Trajectory]]:
    """Load recorded trajectories from the Hugging Face Hub or local files.

    Joint keys:

    * ``repo_id`` (Hub dataset) or ``path`` (local folder or file);
      ``data_files`` (default ``data/*/*.parquet``, the LeRobot layout),
      ``split`` ("train"), ``revision``.
    * ``urdf`` and ``joint_names``: URDF joint for each state entry, in order.
      Joints not listed are frozen at ``frozen_joints`` positions (default 0)
      unless ``freeze_unlisted_joints`` is false.
    * ``state_key`` ("observation.state") and ``state_indices``. Any key may be
      a list of columns, concatenated in order (for example
      ``["observation.state.lower_body", "observation.state.left_arm"]``). An
      index may be null for a joint whose position is not recorded (a rotor);
      its position is then integrated from its velocity.
    * Optional ``velocity_key``/``velocity_indices``, ``torque_key``/
      ``torque_indices``, ``action_key``/``action_indices`` with
      ``action_type`` ("position", "velocity", or "torque").
    * ``state_scale`` and ``state_offset``: per-joint ``q_urdf = scale * q +
      offset`` to fix sign or zero conventions; applied to actions too.
    * ``fps``, ``episode_key`` ("episode_index"), ``frame_key``
      ("frame_index"), ``max_episodes``, ``min_episode_length`` (10),
      ``smoothing_window`` (odd moving-average length before finite
      differencing; 1 disables it), ``end_effectors``.

    Floating-base keys (``floating_base: true``):

    * ``base_orientation_key`` with ``base_orientation_format`` (``wxyz``,
      ``xyzw``, ``rpy``, or ``matrix``) and optional ``base_orientation_indices``;
    * ``base_position_key`` (world frame);
    * ``base_linear_velocity_key`` and ``base_angular_velocity_key``, with
      ``base_linear_velocity_frame``/``base_angular_velocity_frame`` set to
      ``base`` (default, as IMUs and estimators report) or ``world``;
    * ``projected_gravity_key``: unit gravity direction in the base frame;
    * ``imu_acceleration_key``: accelerometer specific force in the base frame
      (gravity included), used for the base acceleration;
    * ``contact_links`` (URDF feet), ``contact_key``/``contact_indices``
      (per-foot contact flags or forces, thresholded at ``contact_threshold``,
      default 0.5), ``surface_contacts``.
    * ``gravity``: a constant 3-vector in the base frame, overriding the rest.

    Actuator keys: ``actuator_overrides`` and ``actuator_defaults`` (set
    ``kp``/``kd`` for position-controlled robots so requested torques are
    meaningful; set ``mode: thrust`` and propeller coefficients for rotors).

    Args:
        source (dict): Source config.
        base (Path | None): Directory for relative paths.
        feature_config (JointFeatureConfig): Observation hardware features.

    Returns:
        tuple: One embodiment record and its trajectories.
    """
    robot = parse_urdf(resolve_path(source["urdf"], base))
    graph = EmbodimentGraph.from_urdf(robot, feature_config, **_graph_kwargs(source, robot))
    action_type = source.get("action_type", "position")
    specs = actuator_specs(graph, source.get("actuator_overrides"), _actuator_defaults(source, action_type))
    record = _record(source.get("name", robot.name), graph, specs, source)
    dataset = _load_table(source, base)
    episode_key = source.get("episode_key", "episode_index")
    frame_key = source.get("frame_key", "frame_index")
    state_key = source.get("state_key", "observation.state")
    vector_keys = {
        "velocity": source.get("velocity_key"),
        "torque": source.get("torque_key"),
        "action": source.get("action_key"),
        "orientation": source.get("base_orientation_key"),
        "position": source.get("base_position_key"),
        "linear": source.get("base_linear_velocity_key"),
        "angular": source.get("base_angular_velocity_key"),
        "gravity": source.get("projected_gravity_key"),
        "imu": source.get("imu_acceleration_key"),
        "contact": source.get("contact_key"),
    }
    keys = {episode_key, *_keys(state_key)}
    for key in vector_keys.values():
        keys.update(_keys(key))
    if frame_key in dataset.column_names:
        keys.add(frame_key)
    dataset = dataset.select_columns(sorted(keys))
    episode_ids = np.asarray(dataset[episode_key]).reshape(-1)
    unique = np.unique(episode_ids)
    if source.get("max_episodes"):
        unique = unique[: int(source["max_episodes"])]
        rows = np.nonzero(np.isin(episode_ids, unique))[0]
        dataset = dataset.select(rows.tolist())
        episode_ids = episode_ids[rows]
    table = {key: dataset[key] for key in dataset.column_names}
    joint_count = graph.num_joints
    state_indices = source.get("state_indices") or list(range(joint_count))
    if len(state_indices) != joint_count:
        raise ValueError(f"{record.name}: state_indices must list one entry per joint")
    states = _column(table, state_key, state_indices)
    velocities = _column(table, vector_keys["velocity"], source.get("velocity_indices", state_indices))
    torques = _column(table, vector_keys["torque"], source.get("torque_indices", state_indices))
    actions = _column(table, vector_keys["action"], source.get("action_indices", state_indices))
    missing_positions = np.isnan(states).all(axis=0)
    if missing_positions.any() and (velocities is None or np.isnan(velocities[:, missing_positions]).any()):
        raise ValueError(f"{record.name}: joints without positions need recorded velocities")
    scale = np.asarray(source.get("state_scale", [1.0] * joint_count), dtype=np.float64)
    offset = np.asarray(source.get("state_offset", [0.0] * joint_count), dtype=np.float64)
    states = states * scale + offset
    if velocities is not None:
        velocities = velocities * scale
    if actions is not None and action_type == "position":
        actions = actions * scale + offset
    elif actions is not None:
        actions = actions * scale
    if torques is not None:
        torques = torques * scale
    base_columns = {
        "orientation": _column(table, vector_keys["orientation"], source.get("base_orientation_indices")),
        "position": _column(table, vector_keys["position"], source.get("base_position_indices")),
        "linear": _column(table, vector_keys["linear"], source.get("base_linear_velocity_indices")),
        "angular": _column(table, vector_keys["angular"], source.get("base_angular_velocity_indices")),
        "gravity": _column(table, vector_keys["gravity"], source.get("projected_gravity_indices")),
        "imu": _column(table, vector_keys["imu"], source.get("imu_acceleration_indices")),
        "contact": _column(table, vector_keys["contact"], source.get("contact_indices")),
    }
    if base_columns["contact"] is not None and len(record.contact_links) != base_columns["contact"].shape[1]:
        raise ValueError(f"{record.name}: contact columns must match contact_links")
    frames = np.asarray(table[frame_key]).reshape(-1) if frame_key in table else np.arange(len(episode_ids))
    gravity = torch.tensor(source["gravity"], dtype=torch.float64) if "gravity" in source else None
    dt = 1.0 / float(source["fps"])
    minimum = int(source.get("min_episode_length", 10))
    window = int(source.get("smoothing_window", 1))
    orientation_format = source.get("base_orientation_format", "wxyz")
    threshold = float(source.get("contact_threshold", 0.5))
    trajectories = []
    for episode in unique:
        rows = np.nonzero(episode_ids == episode)[0]
        rows = rows[np.argsort(frames[rows], kind="stable")]
        if len(rows) < minimum:
            continue

        def pick(values, smooth: bool = False):
            if values is None:
                return None
            selected = values[rows]
            if smooth and window > 1:
                selected = _moving_average(selected, window)
            return torch.from_numpy(np.ascontiguousarray(selected))

        positions = pick(states, smooth=True)
        joint_velocities = pick(velocities)
        if missing_positions.any():
            integrated = torch.cumsum(joint_velocities * dt, dim=0)
            positions[:, torch.from_numpy(missing_positions)] = integrated[:, torch.from_numpy(missing_positions)]
        orientation = None
        if base_columns["orientation"] is not None:
            orientation = _orientation(base_columns["orientation"][rows], orientation_format)
        linear = pick(base_columns["linear"])
        angular = pick(base_columns["angular"])
        if orientation is not None and source.get("base_linear_velocity_frame", "base") == "world" and linear is not None:
            linear = (orientation.transpose(-1, -2) @ linear[..., None]).squeeze(-1)
        if orientation is not None and source.get("base_angular_velocity_frame", "base") == "world" and angular is not None:
            angular = (orientation.transpose(-1, -2) @ angular[..., None]).squeeze(-1)
        contacts = pick(base_columns["contact"])
        trajectories.append(
            Trajectory(
                embodiment=0,
                dt=dt,
                positions=positions,
                velocities=joint_velocities,
                applied_torque=pick(torques),
                actions=pick(actions),
                action_type=action_type,
                gravity=gravity,
                base_orientation=orientation,
                base_position=pick(base_columns["position"]),
                base_linear_velocity=linear,
                base_angular_velocity=angular,
                projected_gravity=pick(base_columns["gravity"]),
                imu_specific_force=pick(base_columns["imu"]),
                contacts=None if contacts is None else contacts > threshold,
            )
        )
    if not trajectories:
        raise ValueError(f"{record.name}: no episodes with at least {minimum} frames")
    return [record], trajectories


def mujoco_source(
    source: dict[str, Any], base: Path | None, feature_config: JointFeatureConfig
) -> tuple[list[EmbodimentRecord], list[Trajectory]]:
    """Simulate legged robots on the ground with MuJoCo (exact contact labels).

    Source keys: ``urdf``, ``name``, ``contact_links`` (required unless the
    robot analysis finds feet), ``surface_contacts``, ``nominal_pose`` (joint
    name to standing position; zero clamped into the limits by default),
    ``randomized_embodiments`` (4), ``episodes_per_embodiment`` (4), ``steps``
    (300), ``dt`` (0.01 s control period), ``timestep`` (0.002 s physics),
    ``probe_torques`` (2), ``target_interval`` (50 steps),
    ``target_amplitude`` (0.15 of each joint's half range; 0.1 for bipeds),
    ``gain_scale`` (1.0; PD gains put peak torque at 0.5 rad of error),
    ``push_probability`` (0.02 per step), ``push_force`` (0.3 of the weight;
    0.15 for bipeds), ``assist`` (spotter stiffness; 0.6 for bipeds, 0 for
    robots with more feet),
    ``friction`` ([0.5, 1.2]), ``contact_geometry`` (``radius``,
    ``sole_offset``, ``foot_size``), ``randomization``, ``seed``, plus the graph
    and actuator keys shared with other sources.

    Args:
        source (dict): Source config.
        base (Path | None): Directory for relative paths.
        feature_config (JointFeatureConfig): Observation hardware features.

    Returns:
        tuple: Embodiment records and trajectories.
    """
    from model.robot_analysis import analyze_robot
    from model.training.ground_sim import ContactGeometry, ground_specs, simulate_ground

    robot = parse_urdf(resolve_path(source["urdf"], base))
    name = source.get("name", robot.name)
    contact_links = source.get("contact_links")
    if not contact_links:
        contact_links = analyze_robot(robot).contact_links
    if not contact_links:
        raise ValueError(f"{name}: no contact links given or found; ground simulation needs feet")
    source = {**source, "contact_links": contact_links, "floating_base": True}
    generator = torch.Generator().manual_seed(int(source.get("seed", 0)))
    randomization = {**DEFAULT_RANDOMIZATION, **source.get("randomization", {})}
    variants = int(source.get("randomized_embodiments", 4))
    geometry = ContactGeometry(**source.get("contact_geometry", {}))
    settings = {
        "episodes": int(source.get("episodes_per_embodiment", 4)),
        "steps": int(source.get("steps", 300)),
        "dt": float(source.get("dt", 0.01)),
        "timestep": float(source.get("timestep", 0.002)),
        "probes": int(source.get("probe_torques", 2)),
        "interval": int(source.get("target_interval", 50)),
        "target_amplitude": float(source.get("target_amplitude", 0.1 if len(contact_links) <= 2 else 0.15)),
        "gain_scale": float(source.get("gain_scale", 1.0)),
        "push_probability": float(source.get("push_probability", 0.02)),
        "push_force": float(source.get("push_force", 0.15 if len(contact_links) <= 2 else 0.3)),
        "assist": source.get("assist"),
        "friction": source.get("friction", [0.5, 1.2]),
        "nominal_pose": source.get("nominal_pose", {}),
        "geometry": geometry,
        "probe_range": float(source.get("probe_range", 2.0)),
        "probe_minimum": float(source.get("probe_minimum", 0.02)),
    }
    robots = [robot] if variants == 0 else [randomize_robot(robot, randomization, generator) for _ in range(variants)]
    records, trajectories = [], []
    for variant, variant_robot in enumerate(robots):
        graph = EmbodimentGraph.from_urdf(variant_robot, feature_config, **_graph_kwargs(source, robot))
        specs = actuator_specs(graph, source.get("actuator_overrides"), _actuator_defaults(source, "position"))
        specs = ground_specs(specs, settings["gain_scale"])
        record = _record(f"{name}/{variant}-ground", graph, specs, source)
        records.append(record)
        trajectories += simulate_ground(len(records) - 1, record, settings, generator)
    return records, trajectories


SOURCE_TYPES = {"synthetic": synthetic_source, "huggingface": huggingface_source, "mujoco": mujoco_source}
