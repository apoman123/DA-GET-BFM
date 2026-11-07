"""Parse URDF kinematic trees and rigid transforms without external robotics libraries.

Only the parts of URDF that describe joint-level structure are read: links with
their inertial block, and joints with type, origin, axis, limit, dynamics, and
mimic tags. Visual and collision geometry are ignored. Xacro must be expanded
before parsing (for example with ``xacro robot.urdf.xacro > robot.urdf``).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import torch

DOF_JOINT_TYPES = ("revolute", "continuous", "prismatic")
MULTI_DOF_JOINT_TYPES = ("planar", "floating")
JOINT_TYPES = DOF_JOINT_TYPES + MULTI_DOF_JOINT_TYPES + ("fixed",)


@dataclass(frozen=True)
class UrdfInertial:
    """Hold one link's inertial block in the link frame.

    Attributes:
        mass (float): Nonnegative link mass in kilograms.
        origin (torch.Tensor): Float64 ``[4, 4]`` transform from the link frame
            to the center-of-mass inertia frame.
        inertia (torch.Tensor): Float64 symmetric ``[3, 3]`` inertia about the
            center of mass, expressed in the inertia frame.
    """

    mass: float
    origin: torch.Tensor
    inertia: torch.Tensor


@dataclass(frozen=True)
class UrdfLink:
    """Hold a URDF link name and optional inertial data.

    Attributes:
        name (str): Unique link name.
        inertial (UrdfInertial | None): Inertial block, or None for massless links.
    """

    name: str
    inertial: UrdfInertial | None = None


@dataclass(frozen=True)
class UrdfJoint:
    """Hold the joint fields needed to build an embodiment graph.

    Attributes:
        name (str): Unique joint name.
        joint_type (str): One of ``JOINT_TYPES``.
        parent (str): Parent link name.
        child (str): Child link name.
        origin (torch.Tensor): Float64 ``[4, 4]`` transform from the parent link
            frame to the joint frame, which is also the child link frame at q=0.
        axis (torch.Tensor): Float64 unit axis ``[3]`` in the joint frame.
        lower (float | None): Lower position limit, if given.
        upper (float | None): Upper position limit, if given.
        effort (float | None): Effort limit, if given.
        velocity (float | None): Velocity limit, if given.
        damping (float): Viscous damping, zero when absent.
        friction (float): Coulomb friction, zero when absent.
        mimic_joint (str | None): Joint this joint mimics, if any.
        mimic_multiplier (float): Mimic multiplier, one when absent.
        mimic_offset (float): Mimic offset, zero when absent.
    """

    name: str
    joint_type: str
    parent: str
    child: str
    origin: torch.Tensor
    axis: torch.Tensor
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    damping: float = 0.0
    friction: float = 0.0
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass(frozen=True)
class UrdfRobot:
    """Hold a validated URDF kinematic tree.

    Attributes:
        name (str): Robot name from the ``<robot>`` tag.
        links (dict[str, UrdfLink]): Links in declaration order.
        joints (dict[str, UrdfJoint]): Joints in declaration order.
        root_link (str): The single link that is never a joint child.
        child_joints (dict[str, tuple[str, ...]]): Joints leaving each link, in
            declaration order.
        gear_ratios (dict[str, float]): Joint name to the ``mechanicalReduction``
            of a ``<transmission>`` that drives it. Joints without a
            transmission are absent.
    """

    name: str
    links: dict[str, UrdfLink]
    joints: dict[str, UrdfJoint]
    root_link: str
    child_joints: dict[str, tuple[str, ...]] = field(repr=False)
    gear_ratios: dict[str, float] = field(default_factory=dict, repr=False)


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert URDF fixed-axis roll, pitch, yaw to rotation matrices.

    Args:
        rpy (torch.Tensor): Angles ``[..., 3]`` in radians; URDF applies roll
            about x, then pitch about y, then yaw about z, all in the fixed frame.

    Returns:
        torch.Tensor: Rotation matrices ``[..., 3, 3]`` equal to Rz @ Ry @ Rx.
    """
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    rows = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def matrix_to_rpy(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to URDF roll, pitch, yaw angles.

    Args:
        rotation (torch.Tensor): Rotation matrices ``[..., 3, 3]``.

    Returns:
        torch.Tensor: Angles ``[..., 3]`` with pitch in [-pi/2, pi/2]. At gimbal
            lock, roll is set to zero and yaw absorbs the remaining rotation.
    """
    sin_pitch = (-rotation[..., 2, 0]).clamp(-1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    cos_pitch = torch.sqrt(rotation[..., 0, 0] ** 2 + rotation[..., 1, 0] ** 2)
    regular = cos_pitch > 1e-9
    roll = torch.where(
        regular,
        torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2]),
        torch.zeros_like(pitch),
    )
    yaw = torch.where(
        regular,
        torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0]),
        torch.atan2(-rotation[..., 0, 1], rotation[..., 1, 1]),
    )
    return torch.stack((roll, pitch, yaw), dim=-1)


def make_transform(
    xyz: torch.Tensor | None = None, rotation: torch.Tensor | None = None
) -> torch.Tensor:
    """Build float64 homogeneous transforms from a translation and rotation.

    Args:
        xyz (torch.Tensor | None): Translation ``[3]``; defaults to zero.
        rotation (torch.Tensor | None): Rotation ``[3, 3]``; defaults to identity.

    Returns:
        torch.Tensor: Float64 transform of shape ``[4, 4]``.
    """
    transform = torch.eye(4, dtype=torch.float64)
    if rotation is not None:
        transform[:3, :3] = rotation
    if xyz is not None:
        transform[:3, 3] = xyz
    return transform


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate about unit axes with Rodrigues' formula.

    Args:
        axis (torch.Tensor): Unit axes ``[..., 3]``.
        angle (torch.Tensor): Angles ``[...]`` in radians, broadcastable to axis.

    Returns:
        torch.Tensor: Rotation matrices ``[..., 3, 3]``.
    """
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    angle = angle[..., None, None]
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew)


def _floats(text: str | None, count: int, default: float, where: str) -> torch.Tensor:
    """Parse a whitespace-separated float vector.

    Args:
        text (str | None): Attribute text, or None to use the default.
        count (int): Required number of values.
        default (float): Fill value when text is None.
        where (str): Context for error messages.

    Returns:
        torch.Tensor: Float64 vector of length count.

    Raises:
        ValueError: The text has the wrong number of finite values.
    """
    if text is None:
        return torch.full((count,), default, dtype=torch.float64)
    try:
        values = [float(value) for value in text.split()]
    except ValueError as error:
        raise ValueError(f"{where}: expected {count} numbers, got {text!r}") from error
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{where}: expected {count} finite numbers, got {text!r}")
    return torch.tensor(values, dtype=torch.float64)


def _optional_float(element: ElementTree.Element | None, key: str) -> float | None:
    """Read one optional float attribute.

    Args:
        element (ElementTree.Element | None): Tag holding the attribute.
        key (str): Attribute name.

    Returns:
        float | None: Parsed value, or None when the tag or attribute is absent.
    """
    if element is None or element.get(key) is None:
        return None
    return float(element.get(key))


def _origin(element: ElementTree.Element | None, where: str) -> torch.Tensor:
    """Parse an optional ``<origin xyz rpy>`` tag.

    Args:
        element (ElementTree.Element | None): The origin tag, if present.
        where (str): Context for error messages.

    Returns:
        torch.Tensor: Float64 ``[4, 4]`` transform; identity when absent.
    """
    if element is None:
        return make_transform()
    xyz = _floats(element.get("xyz"), 3, 0.0, f"{where} origin xyz")
    rpy = _floats(element.get("rpy"), 3, 0.0, f"{where} origin rpy")
    return make_transform(xyz, rpy_to_matrix(rpy))


def _parse_link(element: ElementTree.Element) -> UrdfLink:
    """Parse one ``<link>`` tag.

    Args:
        element (ElementTree.Element): Link tag.

    Returns:
        UrdfLink: Parsed link with optional inertial data.

    Raises:
        ValueError: The name is missing or mass is negative.
    """
    name = element.get("name")
    if not name:
        raise ValueError("every <link> needs a name")
    inertial = element.find("inertial")
    if inertial is None:
        return UrdfLink(name)
    mass_tag = inertial.find("mass")
    mass = float(mass_tag.get("value", 0.0)) if mass_tag is not None else 0.0
    if mass < 0 or not math.isfinite(mass):
        raise ValueError(f"link {name}: mass must be finite and nonnegative")
    inertia_tag = inertial.find("inertia")
    inertia = torch.zeros((3, 3), dtype=torch.float64)
    if inertia_tag is not None:
        keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        ixx, ixy, ixz, iyy, iyz, izz = (float(inertia_tag.get(k, 0.0)) for k in keys)
        inertia = torch.tensor(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=torch.float64
        )
    origin = _origin(inertial.find("origin"), f"link {name} inertial")
    return UrdfLink(name, UrdfInertial(mass, origin, inertia))


def _parse_joint(element: ElementTree.Element) -> UrdfJoint:
    """Parse one ``<joint>`` tag.

    Args:
        element (ElementTree.Element): Joint tag.

    Returns:
        UrdfJoint: Parsed joint with a normalized axis.

    Raises:
        ValueError: A required field is missing or the type or axis is invalid.
    """
    name = element.get("name")
    joint_type = element.get("type")
    parent = element.find("parent")
    child = element.find("child")
    if not name or parent is None or child is None:
        raise ValueError("every <joint> needs a name, <parent>, and <child>")
    if joint_type not in JOINT_TYPES:
        raise ValueError(f"joint {name}: unsupported type {joint_type!r}")
    axis_tag = element.find("axis")
    axis = _floats(
        None if axis_tag is None else axis_tag.get("xyz"), 3, 0.0, f"joint {name} axis"
    )
    if axis_tag is None:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    norm = axis.norm()
    if joint_type != "fixed" and norm < 1e-12:
        raise ValueError(f"joint {name}: axis must be nonzero")
    axis = axis / norm if norm > 0 else axis
    limit = element.find("limit")
    dynamics = element.find("dynamics")
    mimic = element.find("mimic")
    lower = _optional_float(limit, "lower")
    upper = _optional_float(limit, "upper")
    if joint_type in ("revolute", "prismatic") and limit is not None:
        lower = 0.0 if lower is None else lower
        upper = 0.0 if upper is None else upper
    return UrdfJoint(
        name=name,
        joint_type=joint_type,
        parent=parent.get("link", ""),
        child=child.get("link", ""),
        origin=_origin(element.find("origin"), f"joint {name}"),
        axis=axis,
        lower=lower,
        upper=upper,
        effort=_optional_float(limit, "effort"),
        velocity=_optional_float(limit, "velocity"),
        damping=_optional_float(dynamics, "damping") or 0.0,
        friction=_optional_float(dynamics, "friction") or 0.0,
        mimic_joint=None if mimic is None else mimic.get("joint"),
        mimic_multiplier=(
            1.0
            if mimic is None or mimic.get("multiplier") is None
            else float(mimic.get("multiplier"))
        ),
        mimic_offset=_optional_float(mimic, "offset") or 0.0,
    )


def _parse_gear_ratios(
    root: ElementTree.Element, joints: dict[str, UrdfJoint]
) -> dict[str, float]:
    """Read gear ratios from ROS ``<transmission>`` blocks.

    Both the ROS 1 layout (``mechanicalReduction`` under ``<actuator>``) and the
    older layout (directly under ``<transmission>``) are read. The reduction is
    the actuator-to-joint gear ratio N, so joint torque is N times motor torque.

    Args:
        root (ElementTree.Element): The ``<robot>`` element.
        joints (dict[str, UrdfJoint]): Parsed joints, to ignore unknown names.

    Returns:
        dict[str, float]: Joint name to positive gear ratio.

    Raises:
        ValueError: A reduction is not a positive number.
    """
    ratios: dict[str, float] = {}
    for transmission in root.findall("transmission"):
        reduction = transmission.find("actuator/mechanicalReduction")
        if reduction is None:
            reduction = transmission.find("mechanicalReduction")
        if reduction is None or reduction.text is None:
            continue
        try:
            value = float(reduction.text)
        except ValueError as error:
            raise ValueError("mechanicalReduction must be a number") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError("mechanicalReduction must be positive")
        for joint in transmission.findall("joint"):
            name = joint.get("name")
            if name in joints:
                ratios[name] = value
    return ratios


def parse_urdf(source: str | Path) -> UrdfRobot:
    """Parse URDF text or a URDF file into a validated kinematic tree.

    Args:
        source (str | Path): A path to a ``.urdf`` file, or URDF XML text.

    Returns:
        UrdfRobot: The robot with a single root link and tree-structured joints.

    Raises:
        ValueError: The XML is not a URDF robot, names are duplicated, joints
            reference missing links, a link has two parent joints, the tree has
            several roots, or a mimic joint references a missing joint.
    """
    text = source
    if isinstance(source, Path) or (
        isinstance(source, str) and not source.lstrip().startswith("<")
    ):
        text = Path(source).read_text(encoding="utf-8")
    root = ElementTree.fromstring(text)
    if root.tag != "robot":
        raise ValueError("URDF root tag must be <robot>")
    links: dict[str, UrdfLink] = {}
    joints: dict[str, UrdfJoint] = {}
    for element in root.findall("link"):
        link = _parse_link(element)
        if link.name in links:
            raise ValueError(f"duplicate link name {link.name!r}")
        links[link.name] = link
    for element in root.findall("joint"):
        joint = _parse_joint(element)
        if joint.name in joints:
            raise ValueError(f"duplicate joint name {joint.name!r}")
        joints[joint.name] = joint
    child_joints: dict[str, list[str]] = {name: [] for name in links}
    parent_joint: dict[str, str] = {}
    for joint in joints.values():
        if joint.parent not in links or joint.child not in links:
            raise ValueError(f"joint {joint.name} references a missing link")
        if joint.child in parent_joint:
            raise ValueError(
                f"link {joint.child} has two parent joints; URDF must be a tree"
            )
        parent_joint[joint.child] = joint.name
        child_joints[joint.parent].append(joint.name)
        if joint.mimic_joint is not None and joint.mimic_joint not in joints:
            raise ValueError(f"joint {joint.name} mimics a missing joint")
    roots = [name for name in links if name not in parent_joint]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, found {roots}")
    reached = set()
    stack = [roots[0]]
    while stack:
        link = stack.pop()
        reached.add(link)
        stack.extend(joints[name].child for name in child_joints[link])
    if len(reached) != len(links):
        raise ValueError("URDF contains a kinematic cycle")
    return UrdfRobot(
        name=root.get("name", ""),
        links=links,
        joints=joints,
        root_link=roots[0],
        child_joints={name: tuple(value) for name, value in child_joints.items()},
        gear_ratios=_parse_gear_ratios(root, joints),
    )
