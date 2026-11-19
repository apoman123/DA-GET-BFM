"""Procedurally generate legged robots (quadrupeds, hexapods, humanoids) as URDF text.

The generator emits plain URDF, so every consumer of this repository works
unchanged: ``parse_urdf``, ``EmbodimentGraph.from_urdf``, the ``synthetic`` and
``mujoco`` training sources, and MuJoCo's own URDF importer. Feet are links named
``*_foot`` (``EmbodimentRecord.build`` and ``model.robot_analysis`` look for them).

Where every number comes from
=============================

**Topology, geometry and kinematic variation: GenBot-1K.** Ai, Dai, Bohlinger, Li,
Mu, Wu, Fay, Christensen, Peters and Su, "Towards Embodiment Scaling Laws in Robot
Locomotion", CoRL 2025, PMLR 305:3483-3515, arXiv:2505.05753 (v2 checked). Table 6
lists the candidate values, used here unchanged as discrete sets:

* number of knee joints per leg {0, 1, 2, 3} (zero knees: the calf is omitted and
  the thigh carries the foot, Appendix B.2);
* scaling factor for all link sizes {0.8, 1.0, 1.2};
* thigh and calf link length factors {0.4, 0.8, 1.0, 1.2, 1.6};
* foot link size factor {1.0, 2.0} (sphere radius for quadrupeds and hexapods);
* torso link size factor {0.4, 0.8, 1.0, 1.2, 1.6} (humanoids only);
* knee joint-limit scaling factor {0.2, 0.6, 1.0}.

Tables 4, 5 and 7 give the base links, joint limits and nominal angles. The authors'
generator (github.com/BoAi01/embodiment-scaling-laws, BSD-3-Clause, commit 62e43ce,
``generation/embodiment_generation/main_color.py`` and ``gen_urdf_color.py``) fixes
the details the paper leaves open, and the joint offsets, link shapes, link masses
and variation schedule below are taken from it:

* every leg has the same knee count;
* the knee-limit scaling hits the first knee of one leg (quadruped: any leg;
  hexapod: the four corner legs; humanoid: left, right or both legs), and extra knees
  have limits of +-1.57 rad and a nominal angle of 0;
* GenBot-1K is a grid in which **one** geometry factor differs from 1 at a time
  (``geometry_sampling="one_at_a_time"``, the default here).

Where the code and the paper differ, this module follows the code (the code produced
the dataset) and says so:

* the paper scales the humanoid foot "by length"; the code scales all three box sides;
* Table 4 lists a quadruped hip mass of 1.152 kg; the code uses 0.678 kg;
* the paper scales joint limits about the nominal angle; the code scales them about the
  range center and then moves the nominal angle onto the limit if it falls outside.
  Here the paper's definition is used, so the nominal angle always stays inside.

Two simplifications are deliberate. "Scaling factor for all link sizes" scales every
length uniformly, including joint offsets that GenBot-1K's code hard-codes (as
GenLoco's size factor does). The quadruped's lower calf (three fixed links in the
code) is lumped into one straight segment of the same span, 0.0907 m.

**Mass variation: GenLoco.** Feng, Zhang, Li, Peng, Basireddy, Yue, Song, Yang, Liu,
Sreenath and Levine, "GenLoco: Generalized Locomotion Controllers for Quadrupedal
Robots", CoRL 2022, PMLR 205:1893-1903 (arXiv:2209.05309). Table 1 (checked): size
factor in [0.8, 1.2], base density in [400, 1200] kg/m^3, link masses in
[0.5, 1.5] x nominal, thigh length = calf length x [0.75, 1.25], and PD gains scaled
by robot mass / nominal mass (12.458 kg, the A1, Appendix A). GenBot-1K holds masses
fixed within a class (its Limitations section), so the link-mass factor is GenLoco's.
GenLoco's base density is A1's 798 kg/m^3 x [0.5, 1.5] (``a1_density +-
a1_density / 2`` in ``motion_imitation/utilities/quadruped_generator.py`` of
github.com/HybridRobotics/GenLoco). Here the same relative rule applies to every link:
mass = nominal density x volume x factor in [0.5, 1.5]. The nominal density of each
link is its GenBot-1K mass over its nominal shape volume. The absolute [400, 1200]
kg/m^3 range is available through ``ranges["base_density"]``. It is not the default
because GenLoco pairs it with its own base box, and on GenBot-1K's Go2-sized trunk it
gives 1.6-4.8 kg instead of the nominal 6.9 kg.

GenLoco couples thigh and calf lengths, while GenBot-1K varies them independently.
GenBot-1K's sets are the default; set ``ranges["thigh_calf_ratio"]`` to
``{"uniform": [0.75, 1.25]}`` to use GenLoco's coupling instead (the thigh length
factor is then the calf factor times the ratio).

**Inertia from geometry.** Every link is a solid box, cylinder or sphere of uniform
density, so its inertia follows from its shape and mass. ``check_physical_consistency``
verifies the result with the pseudo-inertia test of Wensing, Kim and Slotine, "Linear
Matrix Inequalities for Physically Consistent Inertial Parameter Identification: A
Statistical Perspective on the Mass Distribution", IEEE RA-L 3(1):60-67, 2018,
DOI 10.1109/LRA.2017.2729659 (Theorem 3: the parameters are physically consistent if
and only if the 4x4 pseudo-inertia is positive definite; Proposition 1: equivalently
m > 0 and the principal moments are positive and satisfy the triangle inequalities).

**Peak torque: a measurement in this repository.** No paper gives a rule (GenBot-1K
fixes torques per class, Table 5). The static standing torque of each joint is
computed at the nominal pose with all feet in contact (``estimate_joint_efforts``
without friction, surface contacts for two feet), and peak torque = ratio x standing
torque. The ratio is drawn per robot from [2.3, 13.3]: the measured peak/standing
ratio over loaded joints of the real robots in ``assets/urdf`` has medians 3.4
(ANYmal D), 7.8 (H1), 12.6 (Go1), 13.3 (G1) and a minimum of 2.3 (ANYmal D).

**Joint speed and the other actuator fields: the nearest real robot, rescaled.**
Go1 (quadrupeds, hexapods) and H1 (humanoids) from ``assets/urdf`` give the speed
limit of each joint role, and damping and friction (zero in both URDFs). Speeds are
rescaled by Froude dynamic similarity, Alexander and Jayes, "A dynamic similarity
hypothesis for the gaits of quadrupedal mammals", J. Zool. 201(1):135-152, 1983,
DOI 10.1111/j.1469-7998.1983.tb04266.x: at equal Froude number angular velocities
scale as sqrt(g / L), so omega = omega_template x sqrt(L_template / L), with L the
``EmbodimentGraph.characteristic_length``.

Ranges
======

``DEFAULT_RANGES[family]`` holds the cited defaults; ``ranges`` arguments override
them key by key. A range is a list of candidates (drawn uniformly) or
``{"uniform": [low, high]}``. Keys: ``knee_joints``, ``link_scale``, ``thigh_scale``,
``calf_scale``, ``foot_scale``, ``torso_scale`` (humanoid), ``knee_limit_scale``,
``knee_limit_targets`` (lists of legs), ``geometry_sampling`` (``one_at_a_time`` or
``independent``), ``thigh_calf_ratio`` (None or a range), ``arms`` (humanoid, a list
of booleans), ``link_mass_scale``, ``symmetric_mass``, ``base_density`` (quadruped
and hexapod; None or a range in kg/m^3) and ``torque_ratio``.

Unverified hypotheses ([H])
===========================

* [H] **Torque of lightly loaded joints.** Joints in the same role (for example
  all knees) share one limit, the largest standing torque of the role. A role whose
  standing torque is small gets at least the template's peak-torque ratio of that
  role to its strongest role, times the robot's most loaded role (Go1: hip 23.7 / knee
  35.55 N m; H1: elbow 18 / knee 300 N m). Robots without knees stand on straight
  legs that need almost no torque, so for them the most loaded role is at least the
  template's largest standing torque scaled by dynamic similarity,
  (m L) / (m_template L_template) (torques scale as weight x length when Froude
  numbers match). Test: compare the resulting peak/standing ratios with more real
  robots, and check in simulation that robots with straight legs can still move.
* [H] **Distribution of the torque ratio.** Uniform on [2.3, 13.3]; only the
  bounds are measured. Test: fit the distribution over more real robots.
* [H] **Mirror-symmetric masses.** GenLoco draws each link's mass factor
  independently; by default here mirror-image links (left and right) share one
  factor, so the robot keeps the sagittal symmetry real legged robots have
  (Ordonez-Apraez et al., IJRR 2025, DOI 10.1177/02783649241282422) and mirrored data
  stay valid (``model.training.symmetry``). Set ``ranges["symmetric_mass"] = False``
  for GenLoco's independent draws. Test: whether the symmetric or independent prior
  generalizes better to held-out real robots.
* [H] **Nominal pose with feet under the hips.** The first knee takes GenBot-1K's
  nominal angle (quadruped -1.5, humanoid 0.8 rad, Table 7), extra knees 0, and the
  thigh (hip pitch) angle is solved so the foot is straight below the thigh joint;
  the humanoid ankle keeps the sole flat. For equal thigh and calf the result is
  GenBot-1K's own pose (humanoid hip -0.40 rad), for other proportions it is a
  choice. Hexapods keep GenBot-1K's sprawled pose (thigh and knee 0.79 rad).
  Test: whether the generated robots stand in ``mujoco_source``.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

FAMILIES = ("quadruped", "hexapod", "humanoid")
GENERATOR_VERSION = 1
ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "urdf"

# GenBot-1K Table 6 candidate sets.
_KNEES = [0, 1, 2, 3]
_LINK_SCALE = [0.8, 1.0, 1.2]
_SEGMENT_SCALE = [0.4, 0.8, 1.0, 1.2, 1.6]
_FOOT_SCALE = [1.0, 2.0]
_KNEE_LIMIT_SCALE = [0.2, 0.6, 1.0]

_COMMON_RANGES: dict[str, Any] = {
    "knee_joints": _KNEES,
    "link_scale": _LINK_SCALE,
    "thigh_scale": _SEGMENT_SCALE,
    "calf_scale": _SEGMENT_SCALE,
    "foot_scale": _FOOT_SCALE,
    "knee_limit_scale": _KNEE_LIMIT_SCALE,
    "geometry_sampling": "one_at_a_time",
    "thigh_calf_ratio": None,
    "link_mass_scale": {"uniform": [0.5, 1.5]},
    "symmetric_mass": True,
    "torque_ratio": {"uniform": [2.3, 13.3]},
}
# Legs whose first knee GenBot-1K's code restricts (one variant per entry and factor).
_KNEE_LIMIT_TARGETS = {
    "quadruped": [["FL"], ["FR"], ["RL"], ["RR"]],
    "hexapod": [["FL"], ["RL"], ["RR"], ["FR"]],
    "humanoid": [["left"], ["right"], ["left", "right"]],
}
DEFAULT_RANGES: dict[str, dict[str, Any]] = {
    "quadruped": {**_COMMON_RANGES, "knee_limit_targets": _KNEE_LIMIT_TARGETS["quadruped"], "base_density": None},
    "hexapod": {**_COMMON_RANGES, "knee_limit_targets": _KNEE_LIMIT_TARGETS["hexapod"], "base_density": None},
    "humanoid": {
        **_COMMON_RANGES,
        "knee_limit_targets": _KNEE_LIMIT_TARGETS["humanoid"],
        "torso_scale": _SEGMENT_SCALE,
        "arms": [True],
    },
}

# GenBot-1K Appendix A.1 PD gains (scaled by mass as GenLoco does, see ``pd_gains``).
_FAMILY_GAINS = {"quadruped": (20.0, 0.5), "hexapod": (25.0, 0.5), "humanoid": (60.0, 2.0)}

# Real robots that set joint speeds and the torque floor of each role. Contacts and
# standing poses are those of ``configs/data/ground_contact.json``.
TEMPLATES: dict[str, dict[str, Any]] = {
    "go1": {
        "urdf": "go1.urdf",
        "roles": {"hip": "FR_hip_joint", "thigh": "FR_thigh_joint", "knee": "FR_calf_joint"},
        "contact_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
        "nominal_pose": {
            **{f"{leg}_thigh_joint": 0.9 for leg in ("FR", "FL", "RR", "RL")},
            **{f"{leg}_calf_joint": -1.8 for leg in ("FR", "FL", "RR", "RL")},
        },
    },
    "h1": {
        "urdf": "h1.urdf",
        "roles": {
            "hip_yaw": "left_hip_yaw_joint",
            "hip_roll": "left_hip_roll_joint",
            "hip_pitch": "left_hip_pitch_joint",
            "knee": "left_knee_joint",
            "ankle": "left_ankle_joint",
            "torso": "torso_joint",
            "shoulder_pitch": "left_shoulder_pitch_joint",
            "shoulder_roll": "left_shoulder_roll_joint",
            "shoulder_yaw": "left_shoulder_yaw_joint",
            "elbow": "left_elbow_joint",
        },
        "contact_links": ["left_ankle_link", "right_ankle_link"],
        "nominal_pose": {
            "left_hip_pitch_joint": -0.4, "right_hip_pitch_joint": -0.4,
            "left_knee_joint": 0.8, "right_knee_joint": 0.8,
            "left_ankle_joint": -0.4, "right_ankle_joint": -0.4,
        },
    },
}
FAMILY_TEMPLATE = {"quadruped": "go1", "hexapod": "go1", "humanoid": "h1"}

# Quadruped base unit: GenBot-1K's Go2-based ``dog_base_parameters`` and joint data.
_QUADRUPED = {
    "body": (0.3762, 0.0935, 0.114), "body_mass": 6.921,
    "hip_length": 0.04, "hip_radius": 0.046, "hip_center_y": 0.08, "hip_mass": 0.678,
    "thigh_offset_y": 0.0955,
    "thigh_length": 0.213, "thigh_width": 0.0245, "thigh_height": 0.034, "thigh_mass": 1.152,
    "calf_length": 0.12, "calf_radius": 0.013, "calf_mass": 0.154,
    "lower_calf": 0.0907,  # span of the code's fixed calflower links at factor 1
    "foot_radius": 0.022, "foot_mass": 0.04,
    "hip_limits": (-1.05, 1.05),
    "front_thigh_limits": (-1.57, 3.49), "rear_thigh_limits": (-0.52, 4.53),
    "knee_limits": (-2.72, -0.84), "extra_knee_limits": (-1.57, 1.57),
    "knee_nominal": -1.5,
}
# Hexapod base unit: GenBot-1K's ``hexapod_base_parameters`` and joint data.
_HEXAPOD = {
    "body": (0.8, 0.5, 0.1), "body_mass": 6.921, "hip_spacing": 0.3,
    "hip_radius": 0.05, "hip_mass": 0.678,
    "thigh_length": 0.22, "thigh_radius": 0.03, "thigh_mass": 1.152,
    "calf_length": 0.22, "calf_radius": 0.025, "calf_mass": 0.154,
    "foot_radius": 0.03, "foot_mass": 0.04,
    "limits": (-1.57, 1.57), "thigh_nominal": 0.79, "knee_nominal": 0.79,
    # leg name and angle around the trunk in degrees (legs 1-6 at 30..330 deg)
    "legs": (("FL", 30.0), ("ML", 90.0), ("RL", 150.0), ("RR", -150.0), ("MR", -90.0), ("FR", -30.0)),
}
# Humanoid base unit: GenBot-1K's H1-based ``humanoid_base_parameters`` and joint data.
_HUMANOID = {
    "pelvis_radius": 0.05, "pelvis_mass": 5.39,
    "torso": (0.08, 0.26, 0.18), "torso_gap": 0.05, "torso_mass": 17.789,
    "shoulder": (0.0055, 0.15535, 0.42999), "shoulder_rpy": 0.43633,
    "shoulder_pitch_shape": (0.04, 0.03), "shoulder_pitch_center_y": 0.05, "shoulder_pitch_mass": 1.033,
    "shoulder_roll": (-0.0055, 0.0565, -0.0165), "shoulder_roll_shape": (0.01, 0.04),
    "shoulder_roll_center_z": 0.04, "shoulder_roll_mass": 0.793,
    "shoulder_yaw_z": 0.1343, "shoulder_yaw_shape": (0.15, 0.04), "shoulder_yaw_center_z": 0.11,
    "shoulder_yaw_mass": 0.839,
    "elbow": (0.0185, 0.0, -0.198), "lower_arm": 0.1, "arm_radius": 0.04, "elbow_center_x": 0.135,
    "elbow_mass": 0.723,
    "hip": (0.0, 0.13, -0.1742), "hip_yaw_shape": (0.02, 0.01), "hip_yaw_center_x": 0.02, "hip_yaw_mass": 2.244,
    "hip_roll": (0.039468, 0.0, 0.0), "hip_roll_shape": (0.01, 0.02), "hip_roll_center_y": 0.06,
    "hip_roll_mass": 2.232,
    "hip_pitch_y": 0.1, "joint_gap": 0.2,
    "thigh_length": 0.2, "leg_radius": 0.05, "thigh_mass": 4.152,
    "calf_length": 0.2, "calf_mass": 1.721,
    "foot": (0.28, 0.03, 0.024), "foot_mass": 0.474,
    "limits": {
        "torso": (-2.35, 2.35), "shoulder_pitch": (-2.87, 2.87), "shoulder_roll": (-0.34, 3.11),
        "shoulder_yaw": (-1.30, 4.45), "elbow": (-1.25, 2.61), "hip_yaw": (-0.43, 0.43),
        "hip_roll": (-0.43, 0.43), "hip_pitch": (-3.10, 2.50), "knee": (-0.26, 2.00),
        "extra_knee": (-1.57, 1.57), "ankle": (-0.87, 0.52),
    },
    "knee_nominal": 0.8,
}


@dataclass(frozen=True)
class GeneratedRobot:
    """Hold one generated robot.

    Attributes:
        urdf (str): URDF XML text.
        metadata (dict): JSON-serializable description: ``family``, ``seed``,
            ``params``, ``contact_links``, ``contact_geometry`` (``ContactGeometry``
            keys), ``surface_contacts``, ``nominal_pose``, ``end_effectors``,
            ``total_mass``, ``characteristic_length``, ``torque_ratio``,
            ``standing_torque`` and ``standing_residual`` (base wrench the feet cannot
            supply at the nominal pose), ``peak_torque``, ``max_velocity``,
            ``pd_gains`` (GenBot-1K's class gains times total mass over the class's
            nominal mass, as GenLoco scales them), ``name``, ``template`` and
            ``generator_version``.
    """

    urdf: str
    metadata: dict = field(default_factory=dict)


@dataclass
class _Shape:
    """A solid of uniform density in its link frame."""

    kind: str  # "box", "cylinder", or "sphere"
    size: tuple[float, ...]  # box extents (x, y, z); cylinder (length, radius); sphere (radius,)
    nominal_size: tuple[float, ...]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "z"  # cylinder axis in the link frame


@dataclass
class _Link:
    name: str
    shape: _Shape | None = None
    nominal_mass: float = 0.0
    mass_key: str | None = None
    mass: float = 0.0


@dataclass
class _Joint:
    name: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kind: str = "revolute"
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    lower: float = 0.0
    upper: float = 0.0
    role: str = ""
    nominal: float = 0.0


@dataclass
class _Body:
    """A family builder's output before masses and actuator limits are set."""

    links: list[_Link]
    joints: list[_Joint]
    contact_links: list[str]
    contact_geometry: dict[str, Any]


def _volume(shape: _Shape, size: Sequence[float]) -> float:
    if shape.kind == "box":
        return size[0] * size[1] * size[2]
    if shape.kind == "cylinder":
        return math.pi * size[1] ** 2 * size[0]
    return 4.0 / 3.0 * math.pi * size[0] ** 3


def shape_inertia(kind: str, size: Sequence[float], mass: float, axis: str = "z") -> torch.Tensor:
    """Return the inertia of a uniform solid about its center, in its own axes.

    Args:
        kind (str): ``box`` (extents x, y, z), ``cylinder`` (length, radius), or
            ``sphere`` (radius).
        size (Sequence[float]): Dimensions in m.
        mass (float): Mass in kg.
        axis (str): Cylinder axis, ``x``, ``y`` or ``z``.

    Returns:
        torch.Tensor: Float64 diagonal ``[3, 3]`` inertia in kg m^2.

    Raises:
        ValueError: The kind or axis is unknown.
    """
    if kind == "box":
        x, y, z = size
        moments = (mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12)
    elif kind == "cylinder":
        length, radius = size
        along = mass * radius * radius / 2
        across = mass * (3 * radius * radius + length * length) / 12
        if axis not in "xyz" or len(axis) != 1:
            raise ValueError(f"cylinder axis must be x, y or z, not {axis!r}")
        moments = tuple(along if name == axis else across for name in "xyz")
    elif kind == "sphere":
        moments = (2.0 / 5.0 * mass * size[0] ** 2,) * 3
    else:
        raise ValueError(f"unknown shape {kind!r}")
    return torch.diag(torch.tensor(moments, dtype=torch.float64))


def pseudo_inertia(mass: float, com: Sequence[float] | torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    """Return the 4x4 pseudo-inertia of a rigid body (Wensing, Kim and Slotine, 2018).

    ``J = [[Sigma + m c c^T, m c], [m c^T, m]]`` with the density-weighted covariance
    ``Sigma = tr(I_C) / 2 * 1 - I_C`` about the center of mass.

    Args:
        mass (float): Mass in kg.
        com (Sequence[float] | torch.Tensor): Center of mass ``[3]`` in m.
        inertia (torch.Tensor): ``[3, 3]`` inertia about the center of mass.

    Returns:
        torch.Tensor: Float64 symmetric ``[4, 4]`` pseudo-inertia.
    """
    inertia = torch.as_tensor(inertia, dtype=torch.float64)
    center = torch.as_tensor(com, dtype=torch.float64)
    covariance = 0.5 * torch.trace(inertia) * torch.eye(3, dtype=torch.float64) - inertia
    matrix = torch.zeros((4, 4), dtype=torch.float64)
    matrix[:3, :3] = covariance + mass * torch.outer(center, center)
    matrix[:3, 3] = matrix[3, :3] = mass * center
    matrix[3, 3] = mass
    return 0.5 * (matrix + matrix.T)


def check_physical_consistency(
    mass: float, com: Sequence[float] | torch.Tensor, inertia: torch.Tensor
) -> bool:
    """Check that inertial parameters can come from a real mass distribution.

    Wensing, Kim and Slotine (IEEE RA-L 3(1):60-67, 2018, Theorem 3): the parameters
    are physically consistent if and only if the pseudo-inertia is positive definite.
    Equivalently (their Proposition 1), the mass is positive, the inertia about the
    center of mass is symmetric positive definite, and its principal moments satisfy
    the triangle inequalities ``J1 + J2 > J3`` (and permutations).

    Args:
        mass (float): Mass in kg.
        com (Sequence[float] | torch.Tensor): Center of mass ``[3]`` in m, in any frame.
        inertia (torch.Tensor): ``[3, 3]`` inertia about the center of mass.

    Returns:
        bool: True when all conditions hold.
    """
    inertia = torch.as_tensor(inertia, dtype=torch.float64)
    if not math.isfinite(float(mass)) or mass <= 0 or not bool(torch.isfinite(inertia).all()):
        return False
    scale = max(float(inertia.abs().max()), 1e-300)
    if float((inertia - inertia.T).abs().max()) > 1e-9 * scale:
        return False
    moments = torch.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    if float(moments.min()) <= 0:
        return False
    first, second, third = moments.tolist()
    if not (first + second > third and second + third > first and first + third > second):
        return False
    return bool(float(torch.linalg.eigvalsh(pseudo_inertia(mass, com, inertia)).min()) > 0)


def resolve_ranges(family: str, ranges: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge range overrides into a family's cited defaults.

    A range is either a list of candidates (drawn uniformly) or
    ``{"uniform": [low, high]}``. ``geometry_sampling`` is ``one_at_a_time``
    (GenBot-1K: one geometry factor differs from 1 per robot; needs list ranges) or
    ``independent`` (every factor drawn on its own).

    Args:
        family (str): One of ``FAMILIES``.
        ranges (Mapping[str, Any] | None): Overrides of ``DEFAULT_RANGES[family]``.

    Returns:
        dict[str, Any]: The merged ranges.

    Raises:
        ValueError: The family or a key is unknown, or a value is malformed.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, not {family!r}")
    merged = copy.deepcopy(DEFAULT_RANGES[family])
    unknown = sorted(set(ranges or {}) - set(merged))
    if unknown:
        raise ValueError(f"unknown {family} range keys {unknown}; valid keys are {sorted(merged)}")
    merged.update(copy.deepcopy(dict(ranges or {})))
    for key, value in merged.items():
        if key in ("geometry_sampling", "symmetric_mass", "knee_limit_targets", "arms") or value is None:
            continue
        if isinstance(value, Mapping):
            bounds = value.get("uniform")
            if set(value) != {"uniform"} or len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
                raise ValueError(f"range {key!r} must be a list or {{'uniform': [low, high]}}, not {value!r}")
        elif not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"range {key!r} must be a nonempty list or {{'uniform': [low, high]}}")
    if merged["geometry_sampling"] not in ("one_at_a_time", "independent"):
        raise ValueError("geometry_sampling must be 'one_at_a_time' or 'independent'")
    return merged


def _draw(spec: Any, generator: torch.Generator) -> Any:
    """Draw from a list of candidates or a ``{"uniform": [low, high]}`` range."""
    if isinstance(spec, Mapping):
        low, high = (float(value) for value in spec["uniform"])
        return low + (high - low) * float(torch.rand((), generator=generator, dtype=torch.float64))
    choices = list(spec)
    return choices[int(torch.randint(len(choices), (1,), generator=generator))]


_GEOMETRY_KEYS = ("link_scale", "thigh_scale", "calf_scale", "foot_scale", "torso_scale")


def _legs(family: str) -> list[str]:
    if family == "quadruped":
        return ["FL", "FR", "RL", "RR"]
    if family == "hexapod":
        return [name for name, _ in _HEXAPOD["legs"]]
    return ["left", "right"]


def _mass_group(family: str, leg: str, symmetric: bool) -> str:
    """Return the mass-factor group of a leg: mirror partners share it when symmetric."""
    if not symmetric:
        return leg
    if family == "humanoid":
        return "both"
    return leg[0]  # front, middle, or rear pair


def _mass_keys(family: str, knees: int, arms: bool, symmetric: bool) -> list[str]:
    """List the mass-factor keys of a robot, in a fixed order."""
    keys = ["pelvis", "torso"] if family == "humanoid" else ["trunk"]
    parts = ["hip", "thigh"] + [f"calf{index + 1}" for index in range(knees)] + ["foot"]
    if family == "humanoid":
        parts = ["hip_yaw", "hip_roll", "thigh"] + [f"calf{index + 1}" for index in range(knees)] + ["ankle"]
        if arms:
            parts += ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"]
    for leg in _legs(family):
        group = _mass_group(family, leg, symmetric)
        for part in parts:
            key = f"{group}_{part}"
            if key not in keys:
                keys.append(key)
    return keys


def sample_parameters(
    family: str, generator: torch.Generator, ranges: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Draw the parameters of one robot.

    Args:
        family (str): ``quadruped``, ``hexapod`` or ``humanoid``.
        generator (torch.Generator): Random source.
        ranges (Mapping[str, Any] | None): Overrides of ``DEFAULT_RANGES[family]``
            (see ``resolve_ranges``).

    Returns:
        dict[str, Any]: JSON-serializable parameters: ``family``, ``knee_joints``,
            the geometry factors, ``knee_limit_scale`` and ``knee_limit_legs``,
            ``arms`` (humanoids), ``mass_scale`` (one factor per mass key),
            ``base_density`` (or None) and ``torque_ratio``.

    Raises:
        ValueError: The ranges are invalid.
    """
    ranges = resolve_ranges(family, ranges)
    knees = int(_draw(ranges["knee_joints"], generator))
    if knees < 0:
        raise ValueError("knee_joints must be nonnegative")
    # Knee-limit variant: none, or one factor on the first knee of the target legs.
    targets = [list(target) for target in ranges["knee_limit_targets"]]
    limit_spec = ranges["knee_limit_scale"]
    if knees == 0:
        limit_scale, limit_legs = 1.0, []
    elif isinstance(limit_spec, Mapping):
        limit_scale, limit_legs = float(_draw(limit_spec, generator)), _draw(targets, generator)
    else:
        variants = [(1.0, [])] if any(float(value) == 1.0 for value in limit_spec) else []
        variants += [(float(value), target) for target in targets for value in limit_spec if float(value) != 1.0]
        limit_scale, limit_legs = _draw(variants, generator)
    geometry_keys = [key for key in _GEOMETRY_KEYS if key in ranges]
    geometry = {key: 1.0 for key in geometry_keys}
    if ranges["geometry_sampling"] == "one_at_a_time":
        if any(isinstance(ranges[key], Mapping) for key in geometry_keys):
            raise ValueError("one_at_a_time geometry sampling needs list ranges; use 'independent'")
        variants = [("link_scale", float(value)) for value in ranges["link_scale"]]
        for key in geometry_keys[1:]:
            if key == "calf_scale" and knees == 0:
                continue  # GenBot-1K skips calf variants of robots without calves
            variants += [(key, float(value)) for value in ranges[key] if float(value) != 1.0]
        key, value = _draw(variants, generator)
        geometry[key] = value
    else:
        geometry = {key: float(_draw(ranges[key], generator)) for key in geometry_keys}
    if ranges["thigh_calf_ratio"] is not None:
        geometry["thigh_scale"] = geometry["calf_scale"] * float(_draw(ranges["thigh_calf_ratio"], generator))
    arms = bool(_draw(ranges["arms"], generator)) if family == "humanoid" else False
    symmetric = bool(ranges["symmetric_mass"])
    mass_scale = {
        key: float(_draw(ranges["link_mass_scale"], generator)) for key in _mass_keys(family, knees, arms, symmetric)
    }
    density = None if ranges.get("base_density") is None else float(_draw(ranges["base_density"], generator))
    return {
        "family": family,
        "knee_joints": knees,
        **geometry,
        "knee_limit_scale": float(limit_scale),
        "knee_limit_legs": list(limit_legs),
        **({"arms": arms} if family == "humanoid" else {}),
        "symmetric_mass": symmetric,
        "mass_scale": mass_scale,
        "base_density": density,
        "torque_ratio": float(_draw(ranges["torque_ratio"], generator)),
    }


def _scaled_limits(limits: tuple[float, float], nominal: float, factor: float) -> tuple[float, float]:
    """Scale a joint range about the nominal angle (GenBot-1K Appendix B.2)."""
    lower, upper = limits
    return nominal + factor * (lower - nominal), nominal + factor * (upper - nominal)


def _thigh_angle(segments: Sequence[tuple[float, float]]) -> float:
    """Solve the thigh angle that puts the end of a planar leg straight below its joint.

    Args:
        segments (Sequence[tuple[float, float]]): ``(length, joint angle before the
            segment)`` from the thigh down; segments hang along -z at zero angle and
            joints rotate about +y.

    Returns:
        float: The thigh angle in rad.
    """
    x = z = angle = 0.0
    for length, joint in segments:
        angle += joint
        x -= length * math.sin(angle)
        z -= length * math.cos(angle)
    return math.atan2(x, -z)


def _quadruped(params: Mapping[str, Any]) -> _Body:
    base, s = _QUADRUPED, params["link_scale"]
    thigh_length = base["thigh_length"] * params["thigh_scale"] * s
    calf_length = base["calf_length"] * params["calf_scale"] * s
    knees, symmetric = params["knee_joints"], params["symmetric_mass"]
    body = tuple(value * s for value in base["body"])
    links = [_Link("trunk", _Shape("box", body, base["body"]), base["body_mass"], "trunk")]
    joints: list[_Joint] = []
    contacts = []
    for leg, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        group = _mass_group("quadruped", leg, symmetric)
        hip_size = (base["hip_length"] * s, base["hip_radius"] * s)
        links.append(_Link(f"{leg}_hip", _Shape("cylinder", hip_size, (base["hip_length"], base["hip_radius"]),
                                               (0.0, sy * base["hip_center_y"] * s, 0.0), "y"),
                           base["hip_mass"], f"{group}_hip"))
        joints.append(_Joint(f"{leg}_hip_joint", "trunk", f"{leg}_hip", (sx * body[0] / 2, sy * body[1] / 2, 0.0),
                             axis=(1.0, 0.0, 0.0), lower=base["hip_limits"][0], upper=base["hip_limits"][1], role="hip"))
        thigh_size = (base["thigh_height"] * s, base["thigh_width"] * s, thigh_length)
        links.append(_Link(f"{leg}_thigh", _Shape("box", thigh_size,
                                                 (base["thigh_height"], base["thigh_width"], base["thigh_length"]),
                                                 (0.0, 0.0, -thigh_length / 2)),
                           base["thigh_mass"], f"{group}_thigh"))
        limits = base["front_thigh_limits"] if sx > 0 else base["rear_thigh_limits"]
        thigh = _Joint(f"{leg}_thigh_joint", f"{leg}_hip", f"{leg}_thigh", (0.0, sy * base["thigh_offset_y"] * s, 0.0),
                       lower=limits[0], upper=limits[1], role="thigh")
        joints.append(thigh)
        parent, drop, segments = f"{leg}_thigh", thigh_length, [(thigh_length, 0.0)]
        for index in range(knees):
            suffix = "" if index == 0 else str(index + 1)
            child = f"{leg}_calf{suffix}"
            nominal = base["knee_nominal"] if index == 0 else 0.0
            limits = base["knee_limits"] if index == 0 else base["extra_knee_limits"]
            if index == 0 and leg in params["knee_limit_legs"]:
                limits = _scaled_limits(limits, nominal, params["knee_limit_scale"])
            joints.append(_Joint(f"{leg}_knee{suffix}_joint", parent, child, (0.0, 0.0, -drop),
                                 lower=limits[0], upper=limits[1], role="knee", nominal=nominal))
            links.append(_Link(child, _Shape("cylinder", (calf_length, base["calf_radius"] * s),
                                             (base["calf_length"], base["calf_radius"]), (0.0, 0.0, -calf_length / 2)),
                               base["calf_mass"], f"{group}_calf{index + 1}"))
            parent, drop = child, calf_length
            segments.append((calf_length, nominal))
        lower = base["lower_calf"] * s
        segments.append((lower, 0.0))
        radius = base["foot_radius"] * params["foot_scale"] * s
        links.append(_Link(f"{leg}_foot", _Shape("sphere", (radius,), (base["foot_radius"],)),
                           base["foot_mass"], f"{group}_foot"))
        joints.append(_Joint(f"{leg}_foot_joint", parent, f"{leg}_foot", (0.0, 0.0, -(drop + lower)), kind="fixed"))
        thigh.nominal = min(max(_thigh_angle(segments) if knees else 0.0, thigh.lower), thigh.upper)
        contacts.append(f"{leg}_foot")
    radius = base["foot_radius"] * params["foot_scale"] * s
    geometry = {"radius": radius, "sole_offset": [0.0, 0.0, 0.0], "foot_size": [2 * radius, 2 * radius]}
    return _Body(links, joints, contacts, geometry)


def _hexapod(params: Mapping[str, Any]) -> _Body:
    base, s = _HEXAPOD, params["link_scale"]
    thigh_length = base["thigh_length"] * params["thigh_scale"] * s
    calf_length = base["calf_length"] * params["calf_scale"] * s
    knees, symmetric = params["knee_joints"], params["symmetric_mass"]
    body = tuple(value * s for value in base["body"])
    links = [_Link("trunk", _Shape("box", body, base["body"]), base["body_mass"], "trunk")]
    joints: list[_Joint] = []
    contacts = []
    reach = (max(base["body"][0], base["body"][1]) + base["hip_spacing"]) / 2 * s
    low, high = base["limits"]
    for leg, degrees in base["legs"]:
        group = _mass_group("hexapod", leg, symmetric)
        angle = math.radians(degrees)
        links.append(_Link(f"{leg}_hip", _Shape("sphere", (base["hip_radius"] * s,), (base["hip_radius"],)),
                           base["hip_mass"], f"{group}_hip"))
        joints.append(_Joint(f"{leg}_hip_joint", "trunk", f"{leg}_hip", (reach * math.cos(angle), reach * math.sin(angle), 0.0),
                             rpy=(0.0, 0.0, angle), axis=(0.0, 0.0, 1.0), lower=low, upper=high, role="hip"))
        links.append(_Link(f"{leg}_thigh", _Shape("cylinder", (thigh_length, base["thigh_radius"] * s),
                                                 (base["thigh_length"], base["thigh_radius"]), (thigh_length / 2, 0.0, 0.0), "x"),
                           base["thigh_mass"], f"{group}_thigh"))
        joints.append(_Joint(f"{leg}_thigh_joint", f"{leg}_hip", f"{leg}_thigh", (0.0, 0.0, 0.0),
                             lower=low, upper=high, role="thigh", nominal=base["thigh_nominal"]))
        parent, reach_along = f"{leg}_thigh", thigh_length
        for index in range(knees):
            suffix = "" if index == 0 else str(index + 1)
            child = f"{leg}_calf{suffix}"
            nominal = base["knee_nominal"] if index == 0 else 0.0
            limits = (low, high)
            if index == 0 and leg in params["knee_limit_legs"]:
                limits = _scaled_limits(limits, nominal, params["knee_limit_scale"])
            joints.append(_Joint(f"{leg}_knee{suffix}_joint", parent, child, (reach_along, 0.0, 0.0),
                                 lower=limits[0], upper=limits[1], role="knee", nominal=nominal))
            links.append(_Link(child, _Shape("cylinder", (calf_length, base["calf_radius"] * s),
                                             (base["calf_length"], base["calf_radius"]), (calf_length / 2, 0.0, 0.0), "x"),
                               base["calf_mass"], f"{group}_calf{index + 1}"))
            parent, reach_along = child, calf_length
        radius = base["foot_radius"] * params["foot_scale"] * s
        links.append(_Link(f"{leg}_foot", _Shape("sphere", (radius,), (base["foot_radius"],)),
                           base["foot_mass"], f"{group}_foot"))
        joints.append(_Joint(f"{leg}_foot_joint", parent, f"{leg}_foot", (reach_along, 0.0, 0.0), kind="fixed"))
        contacts.append(f"{leg}_foot")
    radius = base["foot_radius"] * params["foot_scale"] * s
    geometry = {"radius": radius, "sole_offset": [0.0, 0.0, 0.0], "foot_size": [2 * radius, 2 * radius]}
    return _Body(links, joints, contacts, geometry)


def _humanoid(params: Mapping[str, Any]) -> _Body:
    base, s, torso_scale = _HUMANOID, params["link_scale"], params["torso_scale"]
    knees, symmetric, limits = params["knee_joints"], params["symmetric_mass"], base["limits"]
    pelvis = base["pelvis_radius"] * s
    links = [_Link("pelvis", _Shape("sphere", (pelvis,), (base["pelvis_radius"],)), base["pelvis_mass"], "pelvis")]
    torso = tuple(value * s * torso_scale for value in base["torso"])
    torso_center = (0.0, 0.0, pelvis + torso[2] / 2 + base["torso_gap"] * s)
    links.append(_Link("torso_link", _Shape("box", torso, base["torso"], torso_center), base["torso_mass"], "torso"))
    joints = [_Joint("torso_joint", "pelvis", "torso_link", (0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                     lower=limits["torso"][0], upper=limits["torso"][1], role="torso")]
    thigh_drop = (base["thigh_length"] * params["thigh_scale"] + base["joint_gap"]) * s
    calf_drop = (base["calf_length"] * params["calf_scale"] + base["joint_gap"]) * s
    foot = tuple(value * params["foot_scale"] * s for value in base["foot"])
    contacts = []
    for side, sy in (("left", 1.0), ("right", -1.0)):
        group = _mass_group("humanoid", side, symmetric)

        def add(name: str, parent: str, xyz, role: str, shape: _Shape, mass: float, key: str, axis=(0.0, 1.0, 0.0),
                rpy=(0.0, 0.0, 0.0), bounds=None, nominal: float = 0.0, side: str = side, group: str = group) -> _Joint:
            bounds = bounds or limits[role]
            joint = _Joint(f"{side}_{name}_joint", parent, f"{side}_{name}_link", tuple(xyz), tuple(rpy),
                           axis=axis, lower=bounds[0], upper=bounds[1], role=role, nominal=nominal)
            links.append(_Link(f"{side}_{name}_link", shape, mass, f"{group}_{key}"))
            joints.append(joint)
            return joint

        if params["arms"]:
            x, y, z = base["shoulder"]
            if torso[1] / 2 > y * s:  # GenBot-1K moves the shoulders out and up for wide torsos
                y, z = torso[1] / 2 / s + (base["shoulder"][1] - base["torso"][1] / 2), z + 0.05
            roll = base["shoulder_rpy"]
            pitch_shape = tuple(value * s for value in base["shoulder_pitch_shape"])
            add("shoulder_pitch", "torso_link", (x * s, sy * y * s, z * s), "shoulder_pitch",
                _Shape("cylinder", pitch_shape, base["shoulder_pitch_shape"], (0.0, -sy * base["shoulder_pitch_center_y"] * s, 0.0)),
                base["shoulder_pitch_mass"], "shoulder_pitch", rpy=(sy * roll, 0.0, 0.0))
            rx, ry, rz = base["shoulder_roll"]
            roll_bounds = limits["shoulder_roll"] if sy > 0 else (-limits["shoulder_roll"][1], -limits["shoulder_roll"][0])
            add("shoulder_roll", f"{side}_shoulder_pitch_link", (rx * s, sy * ry * s, rz * s), "shoulder_roll",
                _Shape("cylinder", tuple(value * s for value in base["shoulder_roll_shape"]), base["shoulder_roll_shape"],
                       (0.0, 0.0, -base["shoulder_roll_center_z"] * s)),
                base["shoulder_roll_mass"], "shoulder_roll", axis=(1.0, 0.0, 0.0), rpy=(-sy * roll, 0.0, 0.0),
                bounds=roll_bounds)
            yaw_bounds = limits["shoulder_yaw"] if sy > 0 else (-limits["shoulder_yaw"][1], -limits["shoulder_yaw"][0])
            add("shoulder_yaw", f"{side}_shoulder_roll_link", (0.0, 0.0, -base["shoulder_yaw_z"] * s), "shoulder_yaw",
                _Shape("cylinder", tuple(value * s for value in base["shoulder_yaw_shape"]), base["shoulder_yaw_shape"],
                       (0.0, 0.0, -base["shoulder_yaw_center_z"] * s)),
                base["shoulder_yaw_mass"], "shoulder_yaw", axis=(0.0, 0.0, 1.0), bounds=yaw_bounds)
            add("elbow", f"{side}_shoulder_yaw_link", tuple(value * s for value in base["elbow"]), "elbow",
                _Shape("cylinder", (base["lower_arm"] * s, base["arm_radius"] * s), (base["lower_arm"], base["arm_radius"]),
                       (base["elbow_center_x"] * s, 0.0, 0.0), "x"),
                base["elbow_mass"], "elbow")
        hx, hy, hz = base["hip"]
        add("hip_yaw", "pelvis", (hx * s, sy * hy * s, hz * s), "hip_yaw",
            _Shape("cylinder", tuple(value * s for value in base["hip_yaw_shape"]), base["hip_yaw_shape"],
                   (base["hip_yaw_center_x"] * s, 0.0, 0.0), "x"),
            base["hip_yaw_mass"], "hip_yaw", axis=(0.0, 0.0, 1.0))
        add("hip_roll", f"{side}_hip_yaw_link", tuple(value * s for value in base["hip_roll"]), "hip_roll",
            _Shape("cylinder", tuple(value * s for value in base["hip_roll_shape"]), base["hip_roll_shape"],
                   (0.0, sy * base["hip_roll_center_y"] * s, 0.0)),
            base["hip_roll_mass"], "hip_roll", axis=(1.0, 0.0, 0.0))
        thigh_length = base["thigh_length"] * params["thigh_scale"] * s
        hip_pitch = add("hip_pitch", f"{side}_hip_roll_link", (0.0, sy * base["hip_pitch_y"] * s, 0.0), "hip_pitch",
                        _Shape("cylinder", (thigh_length, base["leg_radius"] * s), (base["thigh_length"], base["leg_radius"]),
                               (0.0, 0.0, -thigh_drop / 2)),
                        base["thigh_mass"], "thigh")
        parent, drop, segments, bent = f"{side}_hip_pitch_link", thigh_drop, [(thigh_drop, 0.0)], 0.0
        calf_length = base["calf_length"] * params["calf_scale"] * s
        for index in range(knees):
            suffix = "" if index == 0 else str(index + 1)
            nominal = base["knee_nominal"] if index == 0 else 0.0
            bounds = limits["knee"] if index == 0 else limits["extra_knee"]
            if index == 0 and side in params["knee_limit_legs"]:
                bounds = _scaled_limits(bounds, nominal, params["knee_limit_scale"])
            add(f"knee{suffix}", parent, (0.0, 0.0, -drop), "knee",
                _Shape("cylinder", (calf_length, base["leg_radius"] * s), (base["calf_length"], base["leg_radius"]),
                       (0.0, 0.0, -calf_drop / 2)),
                base["calf_mass"], f"calf{index + 1}", bounds=bounds, nominal=nominal)
            parent, drop = f"{side}_knee{suffix}_link", calf_drop
            segments.append((calf_drop, nominal))
            bent += nominal
        hip_pitch.nominal = min(max(_thigh_angle(segments) if knees else 0.0, hip_pitch.lower), hip_pitch.upper)
        sole = (foot[0] / 4, 0.0, -foot[0] / 4)  # GenBot-1K's foot box center below the ankle
        ankle = add("ankle", parent, (0.0, 0.0, -drop), "ankle", _Shape("box", foot, base["foot"], sole),
                    base["foot_mass"], "ankle")
        ankle.nominal = min(max(-(hip_pitch.nominal + bent), ankle.lower), ankle.upper)
        links.append(_Link(f"{side}_foot"))
        joints.append(_Joint(f"{side}_foot_joint", f"{side}_ankle_link", f"{side}_foot",
                             (sole[0], 0.0, sole[2] - foot[2] / 2), kind="fixed"))
        contacts.append(f"{side}_foot")
    radius = foot[2] / 2
    geometry = {"radius": radius, "sole_offset": [0.0, 0.0, radius], "foot_size": [foot[0], foot[1]]}
    return _Body(links, joints, contacts, geometry)


_BUILDERS = {"quadruped": _quadruped, "hexapod": _hexapod, "humanoid": _humanoid}


def _assign_masses(body: _Body, params: Mapping[str, Any]) -> None:
    """Set each link's mass: nominal density x volume x its GenLoco mass factor."""
    for link in body.links:
        if link.shape is None:
            continue
        ratio = _volume(link.shape, link.shape.size) / _volume(link.shape, link.shape.nominal_size)
        if link.mass_key == "trunk" and params.get("base_density") is not None:
            link.mass = float(params["base_density"]) * _volume(link.shape, link.shape.size)
        else:
            link.mass = link.nominal_mass * ratio * float(params["mass_scale"][link.mass_key])
        inertia = shape_inertia(link.shape.kind, link.shape.size, link.mass, link.shape.axis)
        if not check_physical_consistency(link.mass, link.shape.center, inertia):
            raise ValueError(f"link {link.name} is not physically consistent (mass {link.mass})")


def _number(value: float) -> str:
    return repr(float(value))  # shortest text that reads back as the same float


def _vector(values: Sequence[float]) -> str:
    return " ".join(_number(value) for value in values)


def _geometry_xml(shape: _Shape) -> str:
    rpy = {"x": "0 1.5707963267948966 0", "y": "1.5707963267948966 0 0", "z": "0 0 0"}[shape.axis]
    if shape.kind == "box":
        geometry, rpy = f'<box size="{_vector(shape.size)}"/>', "0 0 0"
    elif shape.kind == "cylinder":
        geometry = f'<cylinder length="{_number(shape.size[0])}" radius="{_number(shape.size[1])}"/>'
    else:
        geometry, rpy = f'<sphere radius="{_number(shape.size[0])}"/>', "0 0 0"
    origin = f'<origin xyz="{_vector(shape.center)}" rpy="{rpy}"/>'
    return f"<visual>{origin}<geometry>{geometry}</geometry></visual><collision>{origin}<geometry>{geometry}</geometry></collision>"


def _urdf(name: str, body: _Body, efforts: Mapping[str, float], speeds: Mapping[str, float]) -> str:
    """Write URDF text with the given effort and speed limit per joint."""
    lines = ['<?xml version="1.0"?>', f'<robot name="{name}">']
    for link in body.links:
        if link.shape is None or link.mass <= 0:
            lines.append(f'  <link name="{link.name}"/>')
            continue
        inertia = shape_inertia(link.shape.kind, link.shape.size, link.mass, link.shape.axis)
        values = (inertia[0, 0], inertia[0, 1], inertia[0, 2], inertia[1, 1], inertia[1, 2], inertia[2, 2])
        keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        tensor = " ".join(f'{key}="{_number(value)}"' for key, value in zip(keys, values))
        lines.append(
            f'  <link name="{link.name}"><inertial><origin xyz="{_vector(link.shape.center)}" rpy="0 0 0"/>'
            f'<mass value="{_number(link.mass)}"/><inertia {tensor}/></inertial>{_geometry_xml(link.shape)}</link>'
        )
    for joint in body.joints:
        parts = [
            f'  <joint name="{joint.name}" type="{joint.kind}">',
            f'<origin xyz="{_vector(joint.xyz)}" rpy="{_vector(joint.rpy)}"/>',
            f'<parent link="{joint.parent}"/><child link="{joint.child}"/>',
        ]
        if joint.kind == "revolute":
            parts.append(f'<axis xyz="{_vector(joint.axis)}"/>')
            parts.append(
                f'<limit lower="{_number(joint.lower)}" upper="{_number(joint.upper)}" '
                f'effort="{_number(efforts[joint.name])}" velocity="{_number(speeds[joint.name])}"/>'
            )
            parts.append('<dynamics damping="0" friction="0"/>')
        parts.append("</joint>")
        lines.append("".join(parts))
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def _graph(urdf: str):
    from model.embodiment import EmbodimentGraph

    return EmbodimentGraph.from_urdf(urdf, floating_base=True)


def standing_torques(
    graph, nominal_pose: Mapping[str, float], contact_links: Sequence[str], return_residual: bool = False
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Return the static joint torques that hold a robot standing on all its feet.

    The pattern of this repository's actuator-margin measurement: inverse dynamics at
    rest with gravity along -z, contact forces distributed over every foot by
    ``estimate_joint_efforts`` (minimum norm, pulling contacts dropped), no friction,
    and surface contacts for two feet or fewer.

    Args:
        graph (EmbodimentGraph): Floating-base robot.
        nominal_pose (Mapping[str, float]): Joint name to angle; missing joints are 0.
        contact_links (Sequence[str]): Feet.
        return_residual (bool): Also return the norm of the base wrench the feet
            could not supply (N and N m mixed; near zero when the robot can stand).

    Returns:
        torch.Tensor | tuple[torch.Tensor, float]: Float64 ``[J]`` signed joint
            torques in N m, and the residual when requested.
    """
    from model.dynamics import STANDARD_GRAVITY, BaseMotion, estimate_joint_efforts

    q = torch.zeros(1, graph.num_joints, dtype=torch.float64)
    for name, value in nominal_pose.items():
        q[0, graph.joint_names.index(name)] = float(value)
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    gravity = torch.tensor([[0.0, 0.0, -STANDARD_GRAVITY]], dtype=torch.float64)
    estimate = estimate_joint_efforts(
        graph, q, torch.zeros_like(q), torch.zeros_like(q), gravity, BaseMotion(zeros, zeros, zeros, zeros),
        list(contact_links), torch.ones(1, len(contact_links), dtype=torch.bool), None,
        include_friction=False, surface_contacts=len(contact_links) <= 2,
    )
    if return_residual:
        return estimate.efforts[0], float(estimate.residual_wrench[0].norm())
    return estimate.efforts[0]


@functools.cache
def template_summary(name: str) -> dict[str, Any]:
    """Measure a template robot of ``assets/urdf``: limits per role, size, standing torque.

    Args:
        name (str): A key of ``TEMPLATES`` (``go1`` or ``h1``).

    Returns:
        dict[str, Any]: ``effort`` and ``velocity`` per role (URDF limits),
            ``characteristic_length`` (m), ``total_mass`` (kg), and
            ``standing_torque`` (largest absolute standing torque, N m).
    """
    from model.embodiment import EmbodimentGraph

    template = TEMPLATES[name]
    path = ASSET_DIRECTORY / template["urdf"]
    if not path.exists():
        raise FileNotFoundError(
            f"template robot {path} is missing; the generator needs the Go1 and H1 URDFs: "
            "uv run python scripts/download_urdfs.py"
        )
    graph = EmbodimentGraph.from_urdf(path, floating_base=True)
    roles = template["roles"]
    efforts = {role: float(graph.effort_limits[graph.joint_names.index(joint)]) for role, joint in roles.items()}
    speeds = {role: float(graph.velocity_limits[graph.joint_names.index(joint)]) for role, joint in roles.items()}
    torque = standing_torques(graph, template["nominal_pose"], template["contact_links"])
    return {
        "effort": efforts,
        "velocity": speeds,
        "characteristic_length": graph.characteristic_length,
        "total_mass": graph.total_mass,
        "standing_torque": float(torque.abs().max()),
    }


def _actuator_limits(family: str, body: _Body, params: Mapping[str, Any], graph, standing: torch.Tensor):
    """Return peak torques and speed limits per joint (see the module docstring)."""
    template = template_summary(FAMILY_TEMPLATE[family])
    joints = [joint for joint in body.joints if joint.kind == "revolute"]
    roles: dict[str, list[str]] = {}
    for joint in joints:
        roles.setdefault(joint.role, []).append(joint.name)
    load = {role: max(float(standing[graph.joint_names.index(name)].abs()) for name in names) for role, names in roles.items()}
    strongest = max(template["effort"].values())
    most = max(load.values())
    if params["knee_joints"] == 0:
        # Straight legs hold the robot with almost no torque; use dynamic similarity.
        most = max(most, template["standing_torque"] * (graph.total_mass * graph.characteristic_length) / (
            template["total_mass"] * template["characteristic_length"]
        ))
    ratio = float(params["torque_ratio"])
    peak = {role: ratio * max(load[role], template["effort"][role] / strongest * most) for role in roles}
    froude = math.sqrt(template["characteristic_length"] / graph.characteristic_length)
    efforts = {joint.name: peak[joint.role] for joint in joints}
    speeds = {joint.name: template["velocity"][joint.role] * froude for joint in joints}
    return efforts, speeds


def _end_effectors(graph, contact_links: Sequence[str]) -> list[str]:
    feet = {graph.joint_names[int(graph.link_bodies[graph.link_names.index(link)])] for link in contact_links}
    leaves = {name for name, leaf in zip(graph.joint_names, graph.leaf_mask.tolist()) if leaf}
    return [name for name in graph.joint_names if name in feet | leaves]


@functools.cache
def _nominal_mass(family: str) -> float:
    """Total mass of a family's 1.0x robot (one knee, every factor 1)."""
    params = {
        "family": family, "knee_joints": 1, **{key: 1.0 for key in _GEOMETRY_KEYS}, "knee_limit_scale": 1.0,
        "knee_limit_legs": [], "arms": True, "symmetric_mass": True, "base_density": None, "torque_ratio": 1.0,
        "mass_scale": {key: 1.0 for key in _mass_keys(family, 1, True, True)},
    }
    body = _BUILDERS[family](params)
    _assign_masses(body, params)
    return sum(link.mass for link in body.links)


def _assemble(family: str, params: Mapping[str, Any], name: str | None = None) -> GeneratedRobot:
    if family not in FAMILIES or params.get("family", family) != family:
        raise ValueError(f"family must be one of {FAMILIES} and match params['family']")
    params = {**params, "family": family}
    if family == "humanoid":
        params.setdefault("arms", True)
        params.setdefault("torso_scale", 1.0)
    body = _BUILDERS[family](params)
    _assign_masses(body, params)
    revolute = [joint.name for joint in body.joints if joint.kind == "revolute"]
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    name = name or f"procedural_{family}_{digest}"
    draft = _urdf(name, body, dict.fromkeys(revolute, 1.0), dict.fromkeys(revolute, 1.0))
    graph = _graph(draft)
    nominal = {joint.name: float(joint.nominal) for joint in body.joints if joint.kind == "revolute"}
    standing, residual = standing_torques(graph, nominal, body.contact_links, return_residual=True)
    efforts, speeds = _actuator_limits(family, body, params, graph, standing)
    urdf = _urdf(name, body, efforts, speeds)
    scale = graph.total_mass / _nominal_mass(family)
    kp, kd = _FAMILY_GAINS[family]
    metadata = {
        "name": name,
        "family": family,
        "params": copy.deepcopy(dict(params)),
        "contact_links": list(body.contact_links),
        "contact_geometry": copy.deepcopy(body.contact_geometry),
        "surface_contacts": len(body.contact_links) <= 2,
        "nominal_pose": nominal,
        "end_effectors": _end_effectors(graph, body.contact_links),
        "total_mass": graph.total_mass,
        "characteristic_length": graph.characteristic_length,
        "torque_ratio": float(params["torque_ratio"]),
        "standing_torque": {joint: float(standing[graph.joint_names.index(joint)]) for joint in revolute},
        "standing_residual": residual,
        "peak_torque": efforts,
        "max_velocity": speeds,
        "pd_gains": {"kp": kp * scale, "kd": kd * scale, "mass_ratio": scale},
        "template": FAMILY_TEMPLATE[family],
        "generator_version": GENERATOR_VERSION,
    }
    return GeneratedRobot(urdf, metadata)


def build_urdf(family: str, params: Mapping[str, Any]) -> str:
    """Build the URDF text of a robot from its parameters.

    Args:
        family (str): ``quadruped``, ``hexapod`` or ``humanoid``.
        params (Mapping[str, Any]): Output of ``sample_parameters``.

    Returns:
        str: URDF XML text.

    Raises:
        ValueError: The family is unknown or a link is not physically consistent.
    """
    return _assemble(family, params).urdf


def generate_robot(family: str, seed: int, ranges: Mapping[str, Any] | None = None) -> GeneratedRobot:
    """Generate one robot deterministically from a seed.

    Args:
        family (str): ``quadruped``, ``hexapod`` or ``humanoid``.
        seed (int): Random seed; the same seed and ranges give the same URDF.
        ranges (Mapping[str, Any] | None): Overrides of ``DEFAULT_RANGES[family]``.

    Returns:
        GeneratedRobot: URDF text and metadata.
    """
    generator = torch.Generator().manual_seed(int(seed))
    params = sample_parameters(family, generator, ranges)
    robot = _assemble(family, params, name=f"procedural_{family}_{int(seed)}")
    robot.metadata["seed"] = int(seed)
    return robot


def suite_seeds(seed: int, count: int) -> list[int]:
    """Return ``count`` per-robot seeds drawn deterministically from one suite seed.

    Args:
        seed (int): Suite seed.
        count (int): Number of robots.

    Returns:
        list[int]: Seeds in [0, 2^31 - 1).
    """
    generator = torch.Generator().manual_seed(int(seed))
    return [int(value) for value in torch.randint(0, 2**31 - 1, (int(count),), generator=generator).tolist()]


def suite_families(family: str | Sequence[str], count: int) -> list[str]:
    """Cycle one or several families over ``count`` robots."""
    families = [family] if isinstance(family, str) else list(family)
    if not families or any(name not in FAMILIES for name in families):
        raise ValueError(f"families must be drawn from {FAMILIES}, not {families}")
    return [families[index % len(families)] for index in range(int(count))]


def write_robot_suite(
    directory: str | Path,
    family: str | Sequence[str],
    count: int,
    seed: int,
    ranges: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write ``count`` generated robots and a ``manifest.json`` to a directory.

    Robot ``i`` is ``generate_robot(families[i % len(families)], suite_seeds(seed,
    count)[i], ranges)`` and is written to ``<family>_<i:04d>.urdf``.

    Args:
        directory (str | Path): Output directory (created if needed).
        family (str | Sequence[str]): One family or several, cycled.
        count (int): Number of robots.
        seed (int): Suite seed.
        ranges (Mapping[str, Any] | None): Range overrides for every family.

    Returns:
        list[Path]: The URDF files in order.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths, entries = [], []
    for index, (name, robot_seed) in enumerate(zip(suite_families(family, count), suite_seeds(seed, count))):
        robot = generate_robot(name, robot_seed, ranges)
        path = directory / f"{name}_{index:04d}.urdf"
        path.write_text(robot.urdf, encoding="utf-8")
        paths.append(path)
        entries.append({"file": path.name, **robot.metadata})
    manifest = {
        "generator": "model.morphology",
        "generator_version": GENERATOR_VERSION,
        "families": [family] if isinstance(family, str) else list(family),
        "count": int(count),
        "seed": int(seed),
        "ranges": copy.deepcopy(dict(ranges or {})),
        "robots": entries,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths
