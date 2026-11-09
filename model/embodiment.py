"""Build GET-Zero embodiment graphs from URDF files, with one token per DoF.

The embodiment graph follows GET-Zero (Patel and Song, ICRA 2025): every
degree of freedom is a node, links are directed parent-to-child edges, and
undirected edges join joints that start independent serial chains on the same
body. The graph is general enough for hands, arms, quadrupeds, bipeds, and
humanoids:

* Fixed joints are collapsed. Their links are merged into the rigid body of the
  nearest moving joint above them, so sensor frames, feet, and mounting links do
  not create tokens.
* Mimic joints are not independent DoFs and are frozen at their offset unless
  ``include_mimic_joints`` is set.
* ``floating`` and ``planar`` joints are skipped (treated as a rigid mount) by
  default. With ``multi_dof_joints="expand"`` they become a serial chain of 1-DoF
  virtual joints (6 for floating, 3 for planar), so each DoF is still a token.
* Joints that share a parent body are joined by undirected sibling edges. For a
  hand these are the finger roots on the palm; for a quadruped the four hip
  joints on the trunk; for a humanoid the hips and waist on the pelvis.

Per-joint hardware features are the GET-Zero "fixed local" observations: the
joint position and rotation relative to the parent joint frame, read from URDF.
``JointFeatureConfig.cross_embodiment()`` adds the axis, joint type, limits, and
child-body inertia, which matter once robots no longer share joint axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import torch

from model.modeling_get import graph_distances
from model.urdf import (
    UrdfJoint,
    UrdfRobot,
    axis_angle_to_matrix,
    make_transform,
    matrix_to_rpy,
    parse_urdf,
)

TOKEN_JOINT_TYPES = ("revolute", "continuous", "prismatic")


@dataclass(frozen=True)
class JointFeatureConfig:
    """Select the fixed per-joint hardware features read from URDF.

    The default matches the GET-Zero LEAP Hand configuration: joint origin and
    joint rotation relative to the parent joint (``includeJointOrigin`` and
    ``includeJointRot``). Other GET-Zero options map to ``joint_axis`` and
    ``joint_limits``.

    Attributes:
        joint_origin (bool): Joint position in the parent joint frame (3).
        joint_rotation (bool): Joint rotation in the parent joint frame (3 for
            ``rpy``, 6 for ``rot6d``).
        rotation_representation (str): ``"rpy"`` as in GET-Zero, or ``"rot6d"``,
            the first two rotation-matrix columns, which has no wrap-around.
        joint_axis (bool): Unit motion axis in the joint frame (3).
        joint_type (bool): One-hot revolute, continuous, prismatic (3).
        joint_limits (bool): Lower and upper position limits (2); continuous
            joints use [-pi, pi].
        actuator_limits (bool): Effort and velocity limits (2); zero if absent.
        joint_dynamics (bool): Damping and friction (2).
        child_inertial (bool): Mass, center of mass (3), and the six unique
            inertia entries about it (6) of the rigid body moved by the joint,
            in the joint frame (10).
    """

    joint_origin: bool = True
    joint_rotation: bool = True
    rotation_representation: Literal["rpy", "rot6d"] = "rpy"
    joint_axis: bool = False
    joint_type: bool = False
    joint_limits: bool = False
    actuator_limits: bool = False
    joint_dynamics: bool = False
    child_inertial: bool = False

    def __post_init__(self) -> None:
        """Reject an unknown rotation representation.

        Returns:
            None: The configuration is valid after construction.

        Raises:
            ValueError: rotation_representation is not ``rpy`` or ``rot6d``.
        """
        if self.rotation_representation not in ("rpy", "rot6d"):
            raise ValueError("rotation_representation must be 'rpy' or 'rot6d'")

    @classmethod
    def get_zero(cls) -> JointFeatureConfig:
        """Return the paper's hardware features: joint origin and rotation.

        Returns:
            JointFeatureConfig: Configuration with six features per joint.
        """
        return cls()

    @classmethod
    def cross_embodiment(cls) -> JointFeatureConfig:
        """Return every feature, for robots with differing axes and dynamics.

        Returns:
            JointFeatureConfig: Configuration with 31 features per joint.
        """
        return cls(
            rotation_representation="rot6d",
            joint_axis=True,
            joint_type=True,
            joint_limits=True,
            actuator_limits=True,
            joint_dynamics=True,
            child_inertial=True,
        )

    def feature_names(self) -> tuple[str, ...]:
        """List feature names in token order.

        Returns:
            tuple[str, ...]: One name per scalar feature.
        """
        names: list[str] = []
        if self.joint_origin:
            names += ["origin_x", "origin_y", "origin_z"]
        if self.joint_rotation:
            if self.rotation_representation == "rpy":
                names += ["rot_roll", "rot_pitch", "rot_yaw"]
            else:
                names += [f"rot6d_{index}" for index in range(6)]
        if self.joint_axis:
            names += ["axis_x", "axis_y", "axis_z"]
        if self.joint_type:
            names += [f"type_{name}" for name in TOKEN_JOINT_TYPES]
        if self.joint_limits:
            names += ["limit_lower", "limit_upper"]
        if self.actuator_limits:
            names += ["limit_effort", "limit_velocity"]
        if self.joint_dynamics:
            names += ["damping", "friction"]
        if self.child_inertial:
            names += ["mass", "com_x", "com_y", "com_z"]
            names += ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
        return tuple(names)

    @property
    def size(self) -> int:
        """Return the number of features per joint token.

        Returns:
            int: Length of ``feature_names()``.
        """
        return len(self.feature_names())


@dataclass
class _Token:
    """Collect one DoF while traversing the URDF tree."""

    name: str
    joint_type: str
    parent: int
    tree_transform: torch.Tensor
    axis: torch.Tensor
    source: UrdfJoint
    lower: float
    upper: float
    mass: float = 0.0
    first_moment: torch.Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    origin_inertia: torch.Tensor = field(
        default_factory=lambda: torch.zeros((3, 3), dtype=torch.float64)
    )


@dataclass
class _Body:
    """Accumulate the mass properties of the root (base) body."""

    mass: float = 0.0
    first_moment: torch.Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    origin_inertia: torch.Tensor = field(
        default_factory=lambda: torch.zeros((3, 3), dtype=torch.float64)
    )


def _center_inertia(body) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the center of mass and the inertia about it of an accumulated body."""
    if body.mass <= 0:
        return torch.zeros(3, dtype=torch.float64), torch.zeros((3, 3), dtype=torch.float64)
    com = body.first_moment / body.mass
    inertia = body.origin_inertia - body.mass * (
        (com @ com) * torch.eye(3, dtype=torch.float64) - torch.outer(com, com)
    )
    return com, inertia


def _skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the cross-product matrix of a 3-vector."""
    x, y, z = vector.tolist()
    return torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=vector.dtype)


def _frozen_motion(joint: UrdfJoint, position: float | None = None) -> torch.Tensor:
    """Return the motion of a joint that is not a token.

    Joints in ``frozen_joints`` are held at their given position, mimic joints at
    their offset, and floating and planar joints at zero.
    """
    if joint.joint_type not in TOKEN_JOINT_TYPES:
        return make_transform()
    if position is None:
        if joint.mimic_joint is None:
            return make_transform()
        position = joint.mimic_offset
    value = torch.tensor(position, dtype=torch.float64)
    if joint.joint_type == "prismatic":
        return make_transform(joint.axis * value)
    return make_transform(rotation=axis_angle_to_matrix(joint.axis, value))


def _limits(joint: UrdfJoint) -> tuple[float, float]:
    """Return position limits, using [-pi, pi] for continuous joints."""
    if joint.joint_type == "continuous":
        return -math.pi, math.pi
    return joint.lower or 0.0, joint.upper or 0.0


def _virtual_tokens(joint: UrdfJoint) -> list[tuple[str, str, torch.Tensor]]:
    """Expand a floating or planar joint into 1-DoF virtual joints.

    Floating joints become x, y, z translations and then x, y, z rotations in the
    joint frame. Planar joints become two in-plane translations and a rotation
    about the plane normal, following the common 3-DoF planar convention.
    """
    eye = torch.eye(3, dtype=torch.float64)
    if joint.joint_type == "floating":
        translations = [
            (f"{joint.name}_t{axis}", "prismatic", eye[index])
            for index, axis in enumerate("xyz")
        ]
        rotations = [
            (f"{joint.name}_r{axis}", "continuous", eye[index])
            for index, axis in enumerate("xyz")
        ]
        return translations + rotations
    normal = joint.axis
    helper = eye[0] if abs(float(normal[0])) < 0.9 else eye[1]
    first = torch.linalg.cross(normal, helper)
    first = first / first.norm()
    second = torch.linalg.cross(normal, first)
    return [
        (f"{joint.name}_tu", "prismatic", first),
        (f"{joint.name}_tv", "prismatic", second),
        (f"{joint.name}_rn", "continuous", normal),
    ]


@dataclass(frozen=True)
class EmbodimentGraph:
    """Hold one robot's per-DoF embodiment graph, features, and kinematics.

    Joint ``j`` is token ``j`` everywhere: in ``joint_names``, rows of
    ``joint_features``, entries of ``parent_indices``, and joint-space tensors
    such as positions and the mass matrix.

    Attributes:
        name (str): Robot name.
        joint_names (tuple[str, ...]): Token joint names in token order.
        joint_types (tuple[str, ...]): ``revolute``, ``continuous``, or
            ``prismatic`` per token.
        parent_indices (torch.Tensor): Long ``[J]`` nearest moving ancestor, or
            -1 for joints mounted on the root body.
        adjacency_matrix (torch.Tensor): Symmetric bool ``[J, J]`` undirected
            edges: parent-child pairs and joints that share a parent body.
        joint_features (torch.Tensor): Float32 ``[J, F]`` fixed hardware features.
        feature_names (tuple[str, ...]): Names of the F feature columns.
        lower (torch.Tensor): Float64 ``[J]`` lower position limits.
        upper (torch.Tensor): Float64 ``[J]`` upper position limits.
        graph_distances (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            Precomputed ``[J, J]`` spatial, ancestor, and descendant distances.
        tree_transforms (torch.Tensor): Float64 ``[J, 4, 4]`` transform from the
            parent joint's moving frame (or the root link) to each joint frame.
        axes (torch.Tensor): Float64 ``[J, 3]`` unit axes in each joint frame.
        body_mass (torch.Tensor): Float64 ``[J]`` mass moved rigidly by each joint.
        body_com (torch.Tensor): Float64 ``[J, 3]`` center of mass in joint frame.
        body_inertia (torch.Tensor): Float64 ``[J, 3, 3]`` inertia about the center
            of mass in the joint frame.
        ancestor_mask (torch.Tensor): Bool ``[J, J]``; entry ``[k, j]`` is True if
            joint ``j`` is joint ``k`` or one of its ancestors.
        actuated (torch.Tensor): Bool ``[J]``; False for virtual DoFs of an
            expanded floating or planar joint, which have no actuator.
        effort_limits (torch.Tensor): Float64 ``[J]`` URDF effort limit in N m
            (revolute) or N (prismatic); zero when absent.
        velocity_limits (torch.Tensor): Float64 ``[J]`` URDF velocity limit in
            rad/s or m/s; zero when absent.
        damping (torch.Tensor): Float64 ``[J]`` viscous damping in N m s/rad.
        friction (torch.Tensor): Float64 ``[J]`` Coulomb friction in N m.
        gear_ratios (torch.Tensor): Float64 ``[J]`` transmission reduction N from
            URDF ``<transmission>``; one when absent.
        leaf_mask (torch.Tensor): Bool ``[J]``; True for joints with no moving
            descendant, the default end-effector frames.
        characteristic_length (float): Largest distance from the root to a joint
            origin at q = 0, in meters (at least 1 cm). Used to normalize
            positions across robots of different size.
        floating_base (bool): Whether the root body moves freely (humanoids,
            quadrupeds, multirotors) rather than being bolted down.
        base_mass (float): Mass of the root body, the root link plus every link
            rigidly attached to it, in kg.
        base_com (torch.Tensor): Float64 ``[3]`` root body center of mass in the
            root frame.
        base_inertia (torch.Tensor): Float64 ``[3, 3]`` root body inertia about
            its center of mass, in the root frame.
        link_names (tuple[str, ...]): Every URDF link reached from the root.
        link_bodies (torch.Tensor): Long ``[L]`` token whose body carries each
            link, or -1 for the root body.
        link_transforms (torch.Tensor): Float64 ``[L, 4, 4]`` link frame in the
            carrying body's frame.
    """

    name: str
    joint_names: tuple[str, ...]
    joint_types: tuple[str, ...]
    parent_indices: torch.Tensor
    adjacency_matrix: torch.Tensor
    joint_features: torch.Tensor
    feature_names: tuple[str, ...]
    lower: torch.Tensor
    upper: torch.Tensor
    graph_distances: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    tree_transforms: torch.Tensor = field(repr=False)
    axes: torch.Tensor = field(repr=False)
    body_mass: torch.Tensor = field(repr=False)
    body_com: torch.Tensor = field(repr=False)
    body_inertia: torch.Tensor = field(repr=False)
    ancestor_mask: torch.Tensor = field(repr=False)
    actuated: torch.Tensor = field(repr=False, default=None)
    effort_limits: torch.Tensor = field(repr=False, default=None)
    velocity_limits: torch.Tensor = field(repr=False, default=None)
    damping: torch.Tensor = field(repr=False, default=None)
    friction: torch.Tensor = field(repr=False, default=None)
    gear_ratios: torch.Tensor = field(repr=False, default=None)
    leaf_mask: torch.Tensor = field(repr=False, default=None)
    characteristic_length: float = 1.0
    floating_base: bool = False
    base_mass: float = 0.0
    base_com: torch.Tensor = field(repr=False, default=None)
    base_inertia: torch.Tensor = field(repr=False, default=None)
    link_names: tuple[str, ...] = field(repr=False, default=())
    link_bodies: torch.Tensor = field(repr=False, default=None)
    link_transforms: torch.Tensor = field(repr=False, default=None)

    @property
    def num_joints(self) -> int:
        """Return the number of DoF tokens.

        Returns:
            int: Token count J.
        """
        return len(self.joint_names)

    @property
    def feature_size(self) -> int:
        """Return the number of fixed hardware features per token.

        Returns:
            int: Feature count F.
        """
        return len(self.feature_names)

    @classmethod
    def from_urdf(
        cls,
        source: str | Path | UrdfRobot,
        feature_config: JointFeatureConfig | None = None,
        joint_order: Sequence[str] | None = None,
        include_mimic_joints: bool = False,
        multi_dof_joints: Literal["skip", "expand"] = "skip",
        sibling_edges: bool = True,
        frozen_joints: Mapping[str, float] | None = None,
        floating_base: bool | None = None,
    ) -> EmbodimentGraph:
        """Build an embodiment graph with one token per URDF degree of freedom.

        Args:
            source (str | Path | UrdfRobot): URDF path, URDF XML text, or a parsed
                robot.
            feature_config (JointFeatureConfig | None): Hardware features to put
                in each token; defaults to the GET-Zero joint origin and rotation.
            joint_order (Sequence[str] | None): Token joint names in the order
                used by your simulator or controller. Defaults to depth-first URDF
                order, the order used by MuJoCo and Pinocchio.
            include_mimic_joints (bool): Make mimic joints tokens instead of
                freezing them at their offset.
            multi_dof_joints (str): ``"skip"`` treats floating and planar joints
                as rigid mounts; ``"expand"`` gives each of their DoFs a token.
            sibling_edges (bool): Add undirected edges between joints that share
                a parent body, as GET-Zero does between serial chains.
            frozen_joints (Mapping[str, float] | None): 1-DoF joints to treat as
                rigid at the given position (rad or m). Use this when a dataset
                records only part of a robot, for example the arms of a humanoid
                whose legs are locked; their links are merged like fixed links.
            floating_base (bool | None): Whether the root body is free. None
                detects it: True when the URDF has a ``floating`` joint that is
                skipped (the usual ``world -> floating -> pelvis`` layout).
                Many legged URDFs have no floating joint and start at the trunk;
                pass True for them.

        Returns:
            EmbodimentGraph: The graph, features, and kinematic data.

        Raises:
            ValueError: The URDF has no DoF, the order is not a permutation of
                token names, or an argument value is invalid.
        """
        if multi_dof_joints not in ("skip", "expand"):
            raise ValueError("multi_dof_joints must be 'skip' or 'expand'")
        robot = source if isinstance(source, UrdfRobot) else parse_urdf(source)
        feature_config = feature_config or JointFeatureConfig()
        frozen = dict(frozen_joints or {})
        unknown = sorted(
            name
            for name in frozen
            if name not in robot.joints
            or robot.joints[name].joint_type not in TOKEN_JOINT_TYPES
        )
        if unknown:
            raise ValueError(f"frozen_joints must name 1-DoF URDF joints: {unknown}")
        base = _Body()
        links: dict[str, tuple[int, torch.Tensor]] = {}
        tokens = _collect_tokens(
            robot, include_mimic_joints, multi_dof_joints, frozen, base, links
        )
        if not tokens:
            raise ValueError(f"robot {robot.name!r} has no degrees of freedom")
        if joint_order is not None:
            tokens, old_to_new = _reorder(tokens, joint_order)
            links = {
                name: (body if body < 0 else old_to_new[body], transform)
                for name, (body, transform) in links.items()
            }
        if floating_base is None:
            floating_base = multi_dof_joints == "skip" and any(
                joint.joint_type == "floating" for joint in robot.joints.values()
            )
        base_com, base_inertia = _center_inertia(base)
        joint_count = len(tokens)
        parents = torch.tensor([token.parent for token in tokens], dtype=torch.long)
        adjacency = torch.zeros((joint_count, joint_count), dtype=torch.bool)
        for index, parent in enumerate(parents.tolist()):
            if parent >= 0:
                adjacency[index, parent] = adjacency[parent, index] = True
        if sibling_edges:
            same_parent = parents[:, None] == parents[None, :]
            adjacency |= same_parent & ~torch.eye(joint_count, dtype=torch.bool)
        ancestors = torch.eye(joint_count, dtype=torch.bool)
        for index in range(joint_count):
            parent = int(parents[index])
            while parent >= 0:
                ancestors[index, parent] = True
                parent = int(parents[parent])
        mass = torch.tensor([token.mass for token in tokens], dtype=torch.float64)
        com = torch.stack(
            [
                token.first_moment / token.mass
                if token.mass > 0
                else torch.zeros(3, dtype=torch.float64)
                for token in tokens
            ]
        )
        inertia = torch.stack(
            [
                token.origin_inertia
                - token.mass
                * (
                    (com[index] @ com[index]) * torch.eye(3, dtype=torch.float64)
                    - torch.outer(com[index], com[index])
                )
                for index, token in enumerate(tokens)
            ]
        )
        lower = torch.tensor([token.lower for token in tokens], dtype=torch.float64)
        upper = torch.tensor([token.upper for token in tokens], dtype=torch.float64)
        tree = torch.stack([token.tree_transform for token in tokens])
        axes = torch.stack([token.axis for token in tokens])
        features = _joint_features(
            feature_config, tokens, tree, axes, lower, upper, mass, com, inertia
        )
        distances = graph_distances(
            parents[None], torch.ones((1, joint_count), dtype=torch.bool), adjacency[None]
        )
        real = [token.source.joint_type in TOKEN_JOINT_TYPES for token in tokens]

        def joint_values(read) -> torch.Tensor:
            return torch.tensor(
                [read(token.source) if is_real else 0.0 for token, is_real in zip(tokens, real)],
                dtype=torch.float64,
            )

        leaf = torch.ones(joint_count, dtype=torch.bool)
        leaf[parents[parents >= 0]] = False
        origins = _rest_origins(parents, tree)
        return cls(
            name=robot.name,
            joint_names=tuple(token.name for token in tokens),
            joint_types=tuple(token.joint_type for token in tokens),
            parent_indices=parents,
            adjacency_matrix=adjacency,
            joint_features=features,
            feature_names=feature_config.feature_names(),
            lower=lower,
            upper=upper,
            graph_distances=tuple(distance[0] for distance in distances),
            tree_transforms=tree,
            axes=axes,
            body_mass=mass,
            body_com=com,
            body_inertia=inertia,
            ancestor_mask=ancestors,
            actuated=torch.tensor(real, dtype=torch.bool),
            effort_limits=joint_values(lambda joint: joint.effort or 0.0),
            velocity_limits=joint_values(lambda joint: joint.velocity or 0.0),
            damping=joint_values(lambda joint: joint.damping),
            friction=joint_values(lambda joint: joint.friction),
            gear_ratios=torch.tensor(
                [
                    robot.gear_ratios.get(token.source.name, 1.0) if is_real else 1.0
                    for token, is_real in zip(tokens, real)
                ],
                dtype=torch.float64,
            ),
            leaf_mask=leaf,
            characteristic_length=max(float(origins.norm(dim=1).max()), 0.01),
            floating_base=bool(floating_base),
            base_mass=float(base.mass),
            base_com=base_com,
            base_inertia=base_inertia,
            link_names=tuple(links),
            link_bodies=torch.tensor([body for body, _ in links.values()], dtype=torch.long),
            link_transforms=torch.stack([transform for _, transform in links.values()]),
        )

    def normalize_joint_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Scale joint positions to [-1, 1] by their limits, as GET-Zero does.

        Args:
            positions (torch.Tensor): Joint positions ``[..., J]``.

        Returns:
            torch.Tensor: Normalized positions. Continuous joints are wrapped to
                [-pi, pi) and divided by pi; joints without a limit range keep
                their raw value.
        """
        lower = self.lower.to(positions)
        upper = self.upper.to(positions)
        span = upper - lower
        continuous = torch.tensor(
            [kind == "continuous" for kind in self.joint_types],
            device=positions.device,
        )
        wrapped = torch.remainder(positions + math.pi, 2 * math.pi) - math.pi
        scaled = torch.where(
            span > 0,
            2 * (positions - lower) / torch.where(span > 0, span, 1) - 1,
            positions,
        )
        return torch.where(continuous, wrapped / math.pi, scaled)

    def forward_kinematics(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute each joint's moving frame in the root link frame.

        This gives the GET-Zero self-modeling target: the pose of every joint
        in the robot's local frame.

        Args:
            positions (torch.Tensor): Floating joint positions ``[..., J]``.

        Returns:
            torch.Tensor: Transforms ``[..., J, 4, 4]`` of each joint frame after
                its own motion. The translation is the joint position.

        Raises:
            ValueError: positions does not end with dimension J.
        """
        return self._kinematics(positions)[0]

    @property
    def total_mass(self) -> float:
        """Return the mass of the whole robot in kg."""
        return self.base_mass + float(self.body_mass.sum())

    def link_index(self, names: Sequence[str]) -> torch.Tensor:
        """Return indices into ``link_names``.

        Args:
            names (Sequence[str]): URDF link names.

        Returns:
            torch.Tensor: Long ``[K]`` indices.

        Raises:
            ValueError: A link is unknown.
        """
        missing = sorted(set(names) - set(self.link_names))
        if missing:
            raise ValueError(f"unknown links {missing}")
        return torch.tensor([self.link_names.index(name) for name in names], dtype=torch.long)

    def link_poses(self, positions: torch.Tensor, links: Sequence[str]) -> torch.Tensor:
        """Compute poses of URDF links in the root frame.

        Args:
            positions (torch.Tensor): Joint positions ``[..., J]``.
            links (Sequence[str]): Link names, for example feet or rotors.

        Returns:
            torch.Tensor: ``[..., K, 4, 4]`` link frames.
        """
        indices = self.link_index(links)
        frames = self.forward_kinematics(positions)
        dtype, device = positions.dtype, positions.device
        identity = torch.eye(4, dtype=dtype, device=device).expand(*positions.shape[:-1], 4, 4)
        poses = []
        for index in indices.tolist():
            body = int(self.link_bodies[index])
            carrier = identity if body < 0 else frames[..., body, :, :]
            poses.append(carrier @ self.link_transforms[index].to(dtype=dtype, device=device))
        return torch.stack(poses, dim=-3)

    def mass_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute the fixed-base joint-space inertia matrix ``M(q)``.

        For a floating-base robot with skipped floating joints, this equals the
        actuated-joint block of the floating-base mass matrix, because that block
        depends only on the bodies below each joint.

        Args:
            positions (torch.Tensor): Floating joint positions ``[..., J]``.

        Returns:
            torch.Tensor: Symmetric positive semidefinite ``[..., J, J]``.

        Raises:
            ValueError: positions does not end with dimension J.
        """
        body_frames, joint_origins, joint_axes = self._kinematics(positions)
        dtype, device = positions.dtype, positions.device
        rotation = body_frames[..., :3, :3]
        com = (
            rotation @ self.body_com.to(dtype=dtype, device=device)[..., None]
        ).squeeze(-1) + body_frames[..., :3, 3]
        inertia = (
            rotation
            @ self.body_inertia.to(dtype=dtype, device=device)
            @ rotation.transpose(-1, -2)
        )
        prismatic = torch.tensor(
            [kind == "prismatic" for kind in self.joint_types], device=device
        )
        mask = self.ancestor_mask.to(device=device, dtype=dtype)[..., None]
        lever = com[..., :, None, :] - joint_origins[..., None, :, :]
        axes = joint_axes[..., None, :, :].expand_as(lever)
        linear = torch.where(
            prismatic[:, None], axes, torch.linalg.cross(axes, lever, dim=-1)
        )
        angular = torch.where(prismatic[:, None], torch.zeros_like(axes), axes)
        linear, angular = linear * mask, angular * mask
        mass = self.body_mass.to(dtype=dtype, device=device)
        # Sum over bodies k of m_k Jv_k^T Jv_k + Jw_k^T I_k Jw_k, as two matmuls
        # over the flattened (body, xyz) dimension.
        joints = self.num_joints
        flat_linear = linear.transpose(-3, -2).reshape(*linear.shape[:-3], joints, -1)
        weighted = (linear * mass[:, None, None]).transpose(-3, -2)
        translational = flat_linear @ weighted.reshape(*linear.shape[:-3], joints, -1).transpose(-1, -2)
        spun = (angular @ inertia.transpose(-1, -2)).transpose(-3, -2)
        flat_angular = angular.transpose(-3, -2).reshape(*angular.shape[:-3], joints, -1)
        rotational = flat_angular @ spun.reshape(*angular.shape[:-3], joints, -1).transpose(-1, -2)
        matrix = translational + rotational
        return 0.5 * (matrix + matrix.transpose(-1, -2))

    def _depth_order(self) -> list[int]:
        """Return joint indices sorted so parents come before children."""
        return torch.argsort(self.ancestor_mask.sum(dim=1), stable=True).tolist()

    def _kinematics(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward kinematics for joint frames, origins, and axes.

        Args:
            positions (torch.Tensor): Floating joint positions ``[..., J]``.

        Returns:
            tuple: Moving joint frames ``[..., J, 4, 4]`` (also the frames of
                the bodies they move), joint origins ``[..., J, 3]``, and joint
                axes ``[..., J, 3]``, all in the root link frame.

        Raises:
            ValueError: positions has the wrong last dimension or dtype.
        """
        if positions.shape[-1:] != (self.num_joints,) or not positions.is_floating_point():
            raise ValueError("positions must be floating with last dimension J")
        dtype, device = positions.dtype, positions.device
        tree = self.tree_transforms.to(dtype=dtype, device=device)
        axes = self.axes.to(dtype=dtype, device=device)
        batch_shape = positions.shape[:-1]
        identity = torch.eye(4, dtype=dtype, device=device).expand(*batch_shape, 4, 4)
        # Joint motions for all joints at once: rotation about the axis for
        # revolute and continuous joints, translation along it for prismatic.
        prismatic = torch.tensor(
            [kind == "prismatic" for kind in self.joint_types], device=device
        )
        rotation = axis_angle_to_matrix(
            axes.expand(*batch_shape, -1, 3), torch.where(prismatic, 0.0, positions)
        )
        translation = axes * torch.where(prismatic, positions, 0.0)[..., None]
        motions = torch.cat(
            (
                torch.cat((rotation, translation[..., None]), dim=-1),
                identity[..., None, 3:, :].expand(*batch_shape, self.num_joints, 1, 4),
            ),
            dim=-2,
        )
        frames: list[torch.Tensor | None] = [None] * self.num_joints
        static: list[torch.Tensor | None] = [None] * self.num_joints
        parents = self.parent_indices.tolist()
        for index in self._depth_order():
            parent = parents[index]
            parent_frame = identity if parent < 0 else frames[parent]
            joint_frame = parent_frame @ tree[index]
            static[index] = joint_frame
            frames[index] = joint_frame @ motions[..., index, :, :]
        moving = torch.stack(frames, dim=-3)
        fixed = torch.stack(static, dim=-3)
        world_axes = (fixed[..., :3, :3] @ axes[..., None]).squeeze(-1)
        return moving, fixed[..., :3, 3], world_axes


def _rest_origins(parents: torch.Tensor, tree: torch.Tensor) -> torch.Tensor:
    """Return joint origins in the root frame at q = 0.

    Args:
        parents (torch.Tensor): Long ``[J]`` parent indices, parents before
            children not required.
        tree (torch.Tensor): Float64 ``[J, 4, 4]`` parent-to-joint transforms.

    Returns:
        torch.Tensor: Float64 ``[J, 3]`` origins.
    """
    frames: dict[int, torch.Tensor] = {}

    def frame(index: int) -> torch.Tensor:
        if index not in frames:
            parent = int(parents[index])
            base = make_transform() if parent < 0 else frame(parent)
            frames[index] = base @ tree[index]
        return frames[index]

    return torch.stack([frame(index)[:3, 3] for index in range(len(parents))])


def _collect_tokens(
    robot: UrdfRobot,
    include_mimic_joints: bool,
    multi_dof_joints: str,
    frozen_joints: Mapping[str, float] | None = None,
    base: _Body | None = None,
    links: dict[str, tuple[int, torch.Tensor]] | None = None,
) -> list[_Token]:
    """Traverse the URDF depth-first and create one token per DoF.

    Args:
        robot (UrdfRobot): Parsed robot.
        include_mimic_joints (bool): Whether mimic joints become tokens.
        multi_dof_joints (str): ``skip`` or ``expand``.
        frozen_joints (Mapping[str, float] | None): 1-DoF joints held rigid at
            the given positions.
        base (_Body | None): Receives the mass of links rigid with the root.
        links (dict | None): Receives each link's carrying token and transform.

    Returns:
        list[_Token]: Tokens in depth-first order with merged body inertia.
    """
    tokens: list[_Token] = []
    frozen_joints = frozen_joints or {}
    base = base if base is not None else _Body()
    links = links if links is not None else {}

    def add_link(link_name: str, parent: int, link_in_body: torch.Tensor) -> None:
        links[link_name] = (parent, link_in_body)
        inertial = robot.links[link_name].inertial
        if inertial is not None and inertial.mass > 0:
            frame = link_in_body @ inertial.origin
            rotation, center = frame[:3, :3], frame[:3, 3]
            token = tokens[parent] if parent >= 0 else base
            token.mass += inertial.mass
            token.first_moment = token.first_moment + inertial.mass * center
            skew = _skew(center)
            token.origin_inertia = (
                token.origin_inertia
                + rotation @ inertial.inertia @ rotation.T
                - inertial.mass * skew @ skew
            )
        for joint_name in robot.child_joints[link_name]:
            joint = robot.joints[joint_name]
            to_joint = link_in_body @ joint.origin
            is_dof = (
                joint.joint_type in TOKEN_JOINT_TYPES
                and joint.name not in frozen_joints
                and (joint.mimic_joint is None or include_mimic_joints)
            )
            if is_dof:
                lower, upper = _limits(joint)
                tokens.append(
                    _Token(
                        joint.name,
                        joint.joint_type,
                        parent,
                        to_joint,
                        joint.axis,
                        joint,
                        lower,
                        upper,
                    )
                )
                add_link(joint.child, len(tokens) - 1, make_transform())
            elif joint.joint_type in ("floating", "planar") and multi_dof_joints == "expand":
                current, transform = parent, to_joint
                for name, kind, axis in _virtual_tokens(joint):
                    lower, upper = (-math.pi, math.pi) if kind == "continuous" else (0.0, 0.0)
                    tokens.append(
                        _Token(name, kind, current, transform, axis, joint, lower, upper)
                    )
                    current, transform = len(tokens) - 1, make_transform()
                add_link(joint.child, current, make_transform())
            else:
                add_link(
                    joint.child,
                    parent,
                    to_joint @ _frozen_motion(joint, frozen_joints.get(joint.name)),
                )

    add_link(robot.root_link, -1, make_transform())
    return tokens


def _reorder(
    tokens: list[_Token], joint_order: Sequence[str]
) -> tuple[list[_Token], dict[int, int]]:
    """Permute tokens to a requested name order and remap parents.

    Args:
        tokens (list[_Token]): Tokens in traversal order.
        joint_order (Sequence[str]): Every token name exactly once.

    Returns:
        tuple: Tokens in the requested order and the old-to-new index map.

    Raises:
        ValueError: joint_order is not a permutation of token names.
    """
    names = [token.name for token in tokens]
    if sorted(joint_order) != sorted(names):
        missing = sorted(set(names) - set(joint_order))
        extra = sorted(set(joint_order) - set(names))
        raise ValueError(
            f"joint_order must list every DoF once; missing {missing}, unknown {extra}"
        )
    old_to_new = {names.index(name): index for index, name in enumerate(joint_order)}
    reordered = [tokens[names.index(name)] for name in joint_order]
    for token in reordered:
        token.parent = -1 if token.parent < 0 else old_to_new[token.parent]
    return reordered, old_to_new


def _joint_features(
    config: JointFeatureConfig,
    tokens: list[_Token],
    tree: torch.Tensor,
    axes: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    mass: torch.Tensor,
    com: torch.Tensor,
    inertia: torch.Tensor,
) -> torch.Tensor:
    """Assemble the fixed local hardware features for every token.

    Returns:
        torch.Tensor: Float32 ``[J, config.size]`` features in name order.
    """
    columns: list[torch.Tensor] = []
    if config.joint_origin:
        columns.append(tree[:, :3, 3])
    if config.joint_rotation:
        rotation = tree[:, :3, :3]
        if config.rotation_representation == "rpy":
            columns.append(matrix_to_rpy(rotation))
        else:
            columns.append(rotation[:, :, :2].transpose(1, 2).reshape(-1, 6))
    if config.joint_axis:
        columns.append(axes)
    if config.joint_type:
        columns.append(
            torch.tensor(
                [
                    [float(token.joint_type == kind) for kind in TOKEN_JOINT_TYPES]
                    for token in tokens
                ],
                dtype=torch.float64,
            )
        )
    if config.joint_limits:
        columns.append(torch.stack((lower, upper), dim=1))
    # Virtual DoFs of an expanded floating or planar joint have no actuator data.
    real = [token.source.joint_type in TOKEN_JOINT_TYPES for token in tokens]
    if config.actuator_limits:
        columns.append(
            torch.tensor(
                [
                    [token.source.effort or 0.0, token.source.velocity or 0.0]
                    if is_real
                    else [0.0, 0.0]
                    for token, is_real in zip(tokens, real)
                ],
                dtype=torch.float64,
            )
        )
    if config.joint_dynamics:
        columns.append(
            torch.tensor(
                [
                    [token.source.damping, token.source.friction]
                    if is_real
                    else [0.0, 0.0]
                    for token, is_real in zip(tokens, real)
                ],
                dtype=torch.float64,
            )
        )
    if config.child_inertial:
        unique = inertia[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        columns.append(torch.cat((mass[:, None], com, unique), dim=1))
    if not columns:
        return torch.zeros((len(tokens), 0), dtype=torch.float32)
    return torch.cat(columns, dim=1).to(torch.float32)


@dataclass(frozen=True)
class EmbodimentBatch:
    """Pad several embodiment graphs to one joint count for batched encoding.

    Attributes:
        graphs (tuple[EmbodimentGraph, ...]): One graph per batch item.
        parent_indices (torch.Tensor): Long ``[B, J]``; -1 for roots and padding.
        joint_mask (torch.Tensor): Bool ``[B, J]``; True for real joints.
        adjacency_matrix (torch.Tensor): Bool ``[B, J, J]`` undirected edges.
        graph_distances (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Long
            ``[B, J, J]`` spatial, ancestor, and descendant distances. Padded
            spatial entries are ``J + 1``, which the encoder treats as unreachable.
        joint_features (torch.Tensor): Float32 ``[B, J, F]``; zero on padding.
    """

    graphs: tuple[EmbodimentGraph, ...]
    parent_indices: torch.Tensor
    joint_mask: torch.Tensor
    adjacency_matrix: torch.Tensor
    graph_distances: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    joint_features: torch.Tensor

    def encoder_inputs(self) -> dict[str, object]:
        """Return the structural keyword arguments for ``DaGetEncoder.forward``.

        Returns:
            dict[str, object]: ``parent_indices``, ``joint_mask``,
                ``adjacency_matrix``, and ``graph_distances``.
        """
        return {
            "parent_indices": self.parent_indices,
            "joint_mask": self.joint_mask,
            "adjacency_matrix": self.adjacency_matrix,
            "graph_distances": self.graph_distances,
        }

    def to(self, device: torch.device | str) -> EmbodimentBatch:
        """Move all tensors to a device.

        Args:
            device (torch.device | str): Target device.

        Returns:
            EmbodimentBatch: A copy with tensors on ``device``.
        """
        return EmbodimentBatch(
            graphs=self.graphs,
            parent_indices=self.parent_indices.to(device),
            joint_mask=self.joint_mask.to(device),
            adjacency_matrix=self.adjacency_matrix.to(device),
            graph_distances=tuple(item.to(device) for item in self.graph_distances),
            joint_features=self.joint_features.to(device),
        )

    def mass_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute padded joint-space mass matrices for every batch item.

        Args:
            positions (torch.Tensor): Padded joint positions ``[B, J]``.

        Returns:
            torch.Tensor: ``[B, J, J]`` with zero rows and columns on padding.
        """
        return self._per_graph(positions, EmbodimentGraph.mass_matrix, matrix=True)

    def forward_kinematics(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute padded joint frames for every batch item.

        Args:
            positions (torch.Tensor): Padded joint positions ``[B, J]``.

        Returns:
            torch.Tensor: ``[B, J, 4, 4]`` joint frames; zero on padding.
        """
        return self._per_graph(positions, EmbodimentGraph.forward_kinematics)

    def normalize_joint_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize padded joint positions by each robot's limits.

        Args:
            positions (torch.Tensor): Padded joint positions ``[B, J, ...]``
                where the joint dimension is 1; history may follow as
                ``[B, H, J]`` when ``positions.ndim == 3``.

        Returns:
            torch.Tensor: Normalized positions with zero on padding.
        """
        if positions.ndim == 3:
            moved = positions.transpose(0, 1)
            return torch.stack(
                [self.normalize_joint_positions(step) for step in moved], dim=1
            )
        return self._per_graph(positions, EmbodimentGraph.normalize_joint_positions)

    def _per_graph(self, positions, function, matrix: bool = False) -> torch.Tensor:
        """Apply a per-graph function to unpadded rows and repad the results."""
        batch_size, joint_count = self.joint_mask.shape
        if positions.shape[:2] != (batch_size, joint_count):
            raise ValueError("positions must have shape [batch, joints]")
        outputs = []
        for index, graph in enumerate(self.graphs):
            count = graph.num_joints
            value = function(graph, positions[index, :count])
            pad = joint_count - count
            if matrix:
                value = torch.nn.functional.pad(value, (0, pad, 0, pad))
            else:
                trailing = [0, 0] * (value.ndim - 1)
                value = torch.nn.functional.pad(value, (*trailing, 0, pad))
            outputs.append(value)
        return torch.stack(outputs)


def collate_embodiments(
    graphs: Sequence[EmbodimentGraph], max_joints: int | None = None
) -> EmbodimentBatch:
    """Pad embodiment graphs with different DoF counts into one batch.

    Args:
        graphs (Sequence[EmbodimentGraph]): One graph per batch item. All must
            use the same feature names.
        max_joints (int | None): Padded joint count; defaults to the largest
            graph in the batch.

    Returns:
        EmbodimentBatch: Padded structure and features.

    Raises:
        ValueError: The batch is empty, feature layouts differ, or max_joints is
            smaller than a graph.
    """
    if not graphs:
        raise ValueError("graphs must not be empty")
    feature_names = graphs[0].feature_names
    if any(graph.feature_names != feature_names for graph in graphs):
        raise ValueError("all graphs must use the same JointFeatureConfig")
    joint_count = max_joints or max(graph.num_joints for graph in graphs)
    if joint_count < max(graph.num_joints for graph in graphs):
        raise ValueError("max_joints is smaller than a graph in the batch")
    batch_size = len(graphs)
    parents = torch.full((batch_size, joint_count), -1, dtype=torch.long)
    mask = torch.zeros((batch_size, joint_count), dtype=torch.bool)
    adjacency = torch.zeros((batch_size, joint_count, joint_count), dtype=torch.bool)
    spatial = torch.full(
        (batch_size, joint_count, joint_count), joint_count + 1, dtype=torch.long
    )
    ancestors = torch.zeros_like(spatial)
    descendants = torch.zeros_like(spatial)
    features = torch.zeros((batch_size, joint_count, len(feature_names)))
    for index, graph in enumerate(graphs):
        count = graph.num_joints
        parents[index, :count] = graph.parent_indices
        mask[index, :count] = True
        adjacency[index, :count, :count] = graph.adjacency_matrix
        graph_spatial, graph_ancestors, graph_descendants = graph.graph_distances
        spatial[index, :count, :count] = torch.where(
            graph_spatial > count, joint_count + 1, graph_spatial
        )
        ancestors[index, :count, :count] = graph_ancestors
        descendants[index, :count, :count] = graph_descendants
        features[index, :count] = graph.joint_features
    return EmbodimentBatch(
        graphs=tuple(graphs),
        parent_indices=parents,
        joint_mask=mask,
        adjacency_matrix=adjacency,
        graph_distances=(spatial, ancestors, descendants),
        joint_features=features,
    )
