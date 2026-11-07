"""Infer data-config settings from a URDF: base type, feet, rotors, and actuators.

``analyze_robot`` looks at the kinematic tree at its rest pose and decides:

* **Multirotor**: at least three ``continuous`` leaf joints with parallel
  axes, spread around the base, and no other joint chain longer than one.
  Rotors get ``thrust`` actuators with propeller coefficients sized from the
  robot's mass, an assumed thrust-to-weight ratio, and the rotor speed limit.
* **Legged** (quadruped, biped, humanoid, hexapod): at least two independent
  chains of two or more joints that end in the lowest links of the robot,
  well below the base. Those lowest links are the contact links (names with
  foot, toe, sole, or ankle are preferred). Two feet or fewer are flat
  surface contacts. Legged robots are floating-base.
* **Fixed base** otherwise (arms, hands, humanoid upper bodies).

A URDF with a ``floating`` joint is always floating-base. The result is a
starting point: check the printed report, and override anything that is wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from model.embodiment import EmbodimentGraph
from model.urdf import UrdfRobot, parse_urdf

FOOT_WORDS = ("foot", "toe", "sole", "ankle", "feet")
ROTOR_WORDS = ("rotor", "prop", "motor", "rotar")


@dataclass
class RobotProfile:
    """Settings inferred from a URDF.

    Attributes:
        kind (str): ``multirotor``, ``quadruped``, ``biped``, ``legged``, or
            ``fixed``.
        floating_base (bool): Whether the base is free.
        contact_links (list[str]): Feet.
        surface_contacts (bool | None): Flat feet (two or fewer feet).
        end_effectors (list[str]): Joints whose frames task E supervises.
        rotor_joints (list[str]): Thrust joints of a multirotor.
        actuator_overrides (dict): Suggested ``actuator_overrides``.
        nominal_pose (dict[str, float]): Rest joint positions within limits.
        notes (list[str]): Explanations of each decision.
    """

    kind: str
    floating_base: bool
    contact_links: list[str] = field(default_factory=list)
    surface_contacts: bool | None = None
    end_effectors: list[str] = field(default_factory=list)
    rotor_joints: list[str] = field(default_factory=list)
    actuator_overrides: dict[str, Any] = field(default_factory=dict)
    nominal_pose: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def source_settings(self) -> dict[str, Any]:
        """Return the data-config keys this profile implies."""
        settings: dict[str, Any] = {"floating_base": self.floating_base}
        if self.contact_links:
            settings["contact_links"] = self.contact_links
            settings["surface_contacts"] = self.surface_contacts
        if self.end_effectors:
            settings["end_effectors"] = self.end_effectors
        if self.actuator_overrides:
            settings["actuator_overrides"] = self.actuator_overrides
        return settings


def _words(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower())


def _rest_pose(graph: EmbodimentGraph) -> torch.Tensor:
    """Zero pose clamped into the joint limits."""
    zero = torch.zeros(graph.num_joints, dtype=torch.float64)
    limited = torch.tensor([kind != "continuous" for kind in graph.joint_types]) & (graph.upper > graph.lower)
    return torch.where(limited, zero.clamp(graph.lower, graph.upper), zero)


def analyze_robot(
    source: str | Path | UrdfRobot,
    thrust_to_weight: float = 2.5,
    max_rotor_speed: float | None = None,
    drag_ratio: float = 0.016,
    motor_time_constant: float = 0.03,
) -> RobotProfile:
    """Infer base type, feet, rotors, and suggested actuator settings.

    Args:
        source (str | Path | UrdfRobot): URDF path, text, or parsed robot.
        thrust_to_weight (float): Multirotor total thrust at full speed over weight.
        max_rotor_speed (float | None): Rotor speed limit in rad/s; the URDF
            velocity limit, or 1000 rad/s, when None.
        drag_ratio (float): Propeller ``k_m / k_f`` in m.
        motor_time_constant (float): Rotor speed loop time constant in s.

    Returns:
        RobotProfile: Inferred settings and the reasoning behind them.
    """
    robot = source if isinstance(source, UrdfRobot) else parse_urdf(source)
    graph = EmbodimentGraph.from_urdf(robot, floating_base=False)
    has_floating = any(joint.joint_type == "floating" for joint in robot.joints.values())
    pose = _rest_pose(graph)
    notes: list[str] = []
    if has_floating:
        notes.append("URDF has a floating joint: floating base")
    profile = RobotProfile(kind="fixed", floating_base=has_floating, nominal_pose={
        name: float(value) for name, value in zip(graph.joint_names, pose)
    })

    frames = graph.forward_kinematics(pose)
    origins = frames[:, :3, 3]
    _, _, axes = graph._kinematics(pose)
    leaf_joints = [index for index in range(graph.num_joints) if bool(graph.leaf_mask[index])]
    depth = graph.ancestor_mask.sum(dim=1)

    # Multirotor: parallel continuous leaf joints spread around the base.
    rotors = [
        index for index in leaf_joints
        if graph.joint_types[index] == "continuous" and int(depth[index]) == 1
    ]
    if len(rotors) >= 3:
        mean_axis = axes[rotors].mean(0)
        mean_axis = mean_axis / mean_axis.norm().clamp_min(1e-9)
        parallel = all(abs(float(axes[index] @ mean_axis)) > 0.95 for index in rotors)
        spread = all(float((origins[index] - (origins[index] @ mean_axis) * mean_axis).norm()) > 0.03 for index in rotors)
        named = sum(any(word in _words(graph.joint_names[index]) for word in ROTOR_WORDS) for index in rotors)
        others = [index for index in range(graph.num_joints) if index not in rotors]
        if parallel and spread and (named >= len(rotors) // 2 or len(others) <= 2):
            profile.kind = "multirotor"
            profile.floating_base = True
            profile.rotor_joints = [graph.joint_names[index] for index in rotors]
            notes.append(f"{len(rotors)} parallel continuous leaf joints around the base: multirotor")
            profile.actuator_overrides = _rotor_overrides(
                graph, rotors, thrust_to_weight, max_rotor_speed, drag_ratio, motor_time_constant, notes
            )
            profile.end_effectors = profile.rotor_joints
            profile.notes = notes
            return profile

    # Legged: independent chains ending in the lowest links, well below the base.
    link_frames = graph.link_poses(pose[None], list(graph.link_names))[0]
    heights = link_frames[:, 2, 3]
    children = {name: robot.child_joints[name] for name in robot.links}
    leaf_links = [name for name in graph.link_names if not children.get(name)]
    span = float(heights.max() - heights.min()) or 1.0
    lowest = float(heights.min())
    first_joint: dict[int, int] = {}
    for index in range(graph.num_joints):
        ancestors = [j for j in range(graph.num_joints) if bool(graph.ancestor_mask[index, j])]
        first_joint[index] = min(ancestors, key=lambda j: int(depth[j]))
    candidates: dict[int, tuple[float, str]] = {}
    for name in leaf_links:
        position = graph.link_names.index(name)
        body = int(graph.link_bodies[position])
        height = float(heights[position])
        if body < 0 or height > lowest + 0.15 * span or int(depth[body]) < 2:
            continue
        chain = first_joint[body]
        preferred = any(word in _words(name) for word in FOOT_WORDS)
        score = height - (1.0 if preferred else 0.0)
        if chain not in candidates or score < candidates[chain][0]:
            candidates[chain] = (score, name)
    feet = [name for _, name in sorted(candidates.values(), key=lambda item: item[1])]
    if any(any(word in _words(name) for word in FOOT_WORDS) for name in feet):
        feet = [name for name in feet if any(word in _words(name) for word in FOOT_WORDS)] or feet
    below = lowest < float(heights[0]) - 0.3 * span if len(heights) else False
    if len(feet) >= 2 and below:
        profile.floating_base = True
        profile.contact_links = feet
        profile.surface_contacts = len(feet) <= 2
        profile.kind = {2: "biped", 4: "quadruped"}.get(len(feet), "legged")
        notes.append(f"{len(feet)} leg chains end at the lowest links {feet}: {profile.kind}, floating base")
        if profile.surface_contacts:
            notes.append("two feet: flat surface contacts that carry moments")
    elif not has_floating:
        notes.append("no legs or rotors found: fixed base")
    foot_joints = {
        graph.joint_names[int(graph.link_bodies[graph.link_names.index(name)])] for name in profile.contact_links
    }
    profile.end_effectors = sorted(foot_joints | {graph.joint_names[index] for index in leaf_joints})
    profile.notes = notes
    return profile


def _rotor_overrides(graph, rotors, thrust_to_weight, max_rotor_speed, drag_ratio, time_constant, notes):
    """Size propeller coefficients from mass, thrust-to-weight, and rotor speed."""
    mass = graph.total_mass
    limits = [float(graph.velocity_limits[index]) for index in rotors]
    speed = max_rotor_speed or (min(limits) if min(limits) > 0 else 1000.0)
    k_f = thrust_to_weight * mass * 9.80665 / (len(rotors) * speed**2)
    k_m = drag_ratio * k_f
    inertia = []
    for index in rotors:
        axis = graph.axes[index]
        inertia.append(float(axis @ graph.body_inertia[index] @ axis) + 1e-7)
    rotor_inertia = max(inertia)
    peak = 2.0 * k_m * speed**2
    notes.append(
        f"propellers sized for thrust-to-weight {thrust_to_weight} at {speed:.0f} rad/s: "
        f"k_f={k_f:.3g} N s^2, k_m={k_m:.3g} N m s^2 (replace with measured values when known)"
    )
    pattern = "|".join(graph.joint_names[index] for index in rotors)
    return {
        name: {
            "mode": "thrust",
            "thrust_coefficient": k_f,
            "torque_coefficient": k_m,
            "kd": rotor_inertia / time_constant,
            "peak_torque": peak,
            "continuous_torque": 0.7 * peak,
            "max_velocity": speed * 1.05,
            "max_power": peak * speed,
        }
        for name in pattern.split("|")
    }


def suggested_position_gains(graph: EmbodimentGraph, stiffness_angle: float = 1.0, damping_ratio: float = 0.025) -> dict[str, dict[str, float]]:
    """Suggest PD gains for position-controlled joints from their effort limits.

    ``K_p = effort / stiffness_angle`` (peak torque at ``stiffness_angle`` rad of
    error) and ``K_d = damping_ratio * K_p``, close to the gains Unitree uses
    on its legged robots. Replace them with the robot's real gains when known.

    Args:
        graph (EmbodimentGraph): Robot.
        stiffness_angle (float): Error in rad that saturates the joint.
        damping_ratio (float): ``K_d / K_p`` in s.

    Returns:
        dict: ``actuator_overrides`` entries per joint.
    """
    gains = {}
    for name, effort in zip(graph.joint_names, graph.effort_limits.tolist()):
        if effort > 0:
            kp = effort / stiffness_angle
            gains[name] = {"kp": round(kp, 3), "kd": round(damping_ratio * kp, 4)}
    return gains


def describe(profile: RobotProfile) -> str:
    """Return a readable report of a profile."""
    lines = [f"kind: {profile.kind}", f"floating_base: {profile.floating_base}"]
    if profile.contact_links:
        lines.append(f"contact_links: {profile.contact_links} (surface contacts: {profile.surface_contacts})")
    if profile.rotor_joints:
        lines.append(f"rotor_joints: {profile.rotor_joints}")
    if profile.end_effectors:
        lines.append(f"end_effectors: {profile.end_effectors}")
    lines += [f"note: {note}" for note in profile.notes]
    return "\n".join(lines)
