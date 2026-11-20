"""Retarget a motion from one robot to another with keypoints and inverse kinematics (proposal G3).

A motion or goal prompt recorded on robot A is mapped to robot B, and BFM-Zero's
tracking and goal inference (``DaGetBfmModel.tracking_inference`` and
``goal_inference``) then run on B unchanged: ``RetargetResult.to_trajectory`` gives a
``model.training.episodes.Trajectory``, the motion format of ``ExpertMotions`` and
``trajectory_state``.

Method (Peng, Coumans, Zhang, Lee, Tan and Levine, "Learning Agile Robotic Locomotion
Skills by Imitating Animals", RSS 2020, DOI 10.15607/RSS.2020.XVI.064, arXiv:2004.00784,
Sec. IV, Eq. 1, checked against the paper): source keypoints are "paired with
corresponding target keypoints on the robot's body" ("The keypoints include the
positions of the feet and hips" in their case), and IK finds, at every frame, the pose
that tracks them (retargeting by inverse kinematics, which they take from Gleicher,
"Retargetting Motion to New Characters", SIGGRAPH 1998, pp. 33-42, DOI
10.1145/280814.280820),

    argmin_q  sum_i ||x_hat_i(t) - x_i(q_t)||^2 + (q_bar - q_t)^T W (q_bar - q_t),

where the second term keeps the pose near a default pose ``q_bar`` with a diagonal
``W``. The paper gives no values for ``W``; ``rest_weight`` (a scalar ``W = w I``)
defaults to 0, which leaves the term out. The paper itself does not scale the source
motion. Their code release (github.com/erwincoumans/motion_imitation, Apache-2.0,
"Copyright 2020 The Google Research Authors", ``retarget_motion/retarget_motion.py``)
multiplies every reference position by a hand-set constant ``REF_POS_SCALE`` (1 in
``retarget_config_laikago.py``, 0.825 in ``retarget_config_a1.py``) and places each toe
target at the robot's hip plus the source's hip-to-toe offset (height taken from the
source toe). No code is copied from it. Here the scale is computed from the
two robots instead (``scale_ratio``):

* ``"characteristic_length"`` (default): ``L_B / L_A`` with
  ``EmbodimentGraph.characteristic_length``. This is the length the repository already
  uses to compare robots of different size: the FK targets of pretraining are
  ``p / L``, and the BFM state view divides joint-frame positions, base height and base
  linear velocity by ``L`` (``model/rl/observation.py``). With this ratio the scaled
  keypoints have the same normalized coordinates on both robots, which is what B and D
  read. It needs no choice of keypoints or pose. Its limit: it measures joint origins
  only, not the last link (Go1: 0.311 m to the calf joint; the feet are 0.44 m from the
  base at the clamped zero pose).
* ``"keypoints"``: the ratio of the mean root-to-keypoint distances at each robot's
  reference pose, which is the leg length when the keypoints are the feet. It depends on
  that pose (Go1's zero pose bends the knee, which its limits require, while ANYmal D's
  is straight).
* a number: a fixed ratio, as in Peng et al.'s code.

``retarget`` evaluates ``"keypoints"`` at the target's ``initial_pose`` and the source's
zero pose clamped into its limits. Measured on Go1 to ANYmal D (the standing motion of
``tests/test_retarget.py``, ANYmal's pose from ``configs/data/ground_contact.json``): the
two ratios are 1.783 and 1.787, and both reach every scaled foot to within 1e-7 m with
all joints at least 0.42 rad inside their limits. [H] unverified hypothesis: that either
automatic ratio gives better BFM prompts than a hand-tuned one; nothing here measures
prompt quality. Scaling is about the world origin, so a floor at ``z = 0`` stays at
``z = 0``.

Solver: damped least squares (Wampler, "Manipulator Inverse Kinematic Solutions Based on
Vector Formulations and Damped Least-Squares Methods", IEEE Trans. SMC 16(1):93-101,
1986, DOI 10.1109/TSMC.1986.289285; independently Nakamura and Hanafusa, "Inverse
Kinematic Solutions With Singularity Robustness for Robot Manipulator Control", ASME
J. Dyn. Syst. Meas. Control 108(3):163-171, 1986, DOI 10.1115/1.3143764), in the
Levenberg-Marquardt form (Levenberg, Q. Appl. Math. 2(2):164-168, 1944, DOI
10.1090/qam/10666; Marquardt, J. SIAM 11(2):431-441, 1963, DOI 10.1137/0111030) with
Sugihara's damping
``W_N = E_k I + w_bar I`` ("Solvability-Unconcerned Inverse Kinematics by the
Levenberg-Marquardt Method", IEEE T-RO 27(5):984-991, 2011, DOI
10.1109/TRO.2011.2148230), where ``E_k`` is half the squared residual. The bias
``w_bar`` is between ``0.01 l^2`` and ``l^2`` for a typical link length ``l``, as Haviland
and Corke summarize Sugihara ("Manipulator Differential Kinematics: Part I: Kinematics,
Velocity, and Applications", IEEE Robotics & Automation Magazine 31(4):149-158, 2024,
DOI 10.1109/MRA.2023.3270228, arXiv:2207.01796, Eq. 51 of the arXiv version). The default is the lower end, ``0.01 l^2``, with ``l`` the mean nonzero joint
offset of the target: on the Go1 to ANYmal D motion above it needed 5.1 iterations per
frame against 54 for ``l^2``, at the same final error (measured). The Jacobian
comes from automatic differentiation of ``EmbodimentGraph.link_poses``
(``torch.func.jacrev``). Joints with limits are clamped after every step, and each
frame starts from the previous frame's solution. A floating target's base pose is
solved together with the joints; its translation unknowns are scaled by the
characteristic length and its rotation is a body-frame increment, so all unknowns move
keypoints in meters, the unit of ``E_k``. Keypoints with ``orientation_weight`` also
match their frame axes: the residual is ``(R_hat - R) L``, the error of three virtual
points one characteristic length along the axes, so it is in meters as well. This
assumes both robots' link frames share a convention (REP 103, Foote and Purvis,
"Standard Units of Measure and Coordinate Conventions", 2010,
https://www.ros.org/reps/rep-0103.html: x forward, y left, z up), as the URDFs in
``assets/urdf`` do. [H] unverified hypothesis: the stopping settings
(``MAX_ITERATIONS``, ``TOLERANCE``, ``STALL_FRACTION``) are numerical choices;
``RetargetResult`` reports the remaining keypoint error of every frame.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from model.embodiment import EmbodimentGraph
from model.training.episodes import Trajectory, exp_rotation

SCALE_METHODS = ("characteristic_length", "keypoints")
# Sugihara's bias range is 0.01 l^2 to l^2 (Haviland and Corke 2024, arXiv:2207.01796 Eq. 51); we use the lower end.
DAMPING_BIAS_FRACTION = 0.01
# [H] numerical stopping settings (see the module docstring): at most MAX_ITERATIONS per
# frame; stop when the residual norm is below TOLERANCE times the target's characteristic
# length, or when an iteration lowers the energy by less than STALL_FRACTION of it.
MAX_ITERATIONS = 100
TOLERANCE = 1e-6
STALL_FRACTION = 1e-6


@dataclass(frozen=True)
class KeypointPair:
    """One correspondence between a source link and a target link.

    Attributes:
        source (str): URDF link of the source robot.
        target (str): URDF link of the target robot.
        weight (float): Weight of the position residual.
        orientation_weight (float): Weight of the frame-axes residual; 0 matches the
            position only. Give the base pair a weight so the base orientation follows.
    """

    source: str
    target: str
    weight: float = 1.0
    orientation_weight: float = 0.0

    def __post_init__(self) -> None:
        """Validate the weights."""
        if not (self.weight >= 0 and self.orientation_weight >= 0) or not math.isfinite(self.weight + self.orientation_weight):
            raise ValueError("keypoint weights must be finite and nonnegative")
        if self.weight == 0 and self.orientation_weight == 0:
            raise ValueError(f"keypoint {self.source} -> {self.target} has no weight")


@dataclass(frozen=True)
class KeypointMap:
    """Correspondences of source links to target links (feet, hands, base, knees, ...).

    The JSON form is either ``{"keypoints": [{"source": ..., "target": ...,
    "weight": 1.0, "orientation_weight": 0.0}, ...]}`` or a plain
    ``{"source link": "target link"}`` object. Example for Go1 to ANYmal D::

        {"keypoints": [
            {"source": "trunk", "target": "base", "orientation_weight": 1.0},
            {"source": "FL_foot", "target": "LF_FOOT"}, {"source": "FR_foot", "target": "RF_FOOT"},
            {"source": "RL_foot", "target": "LH_FOOT"}, {"source": "RR_foot", "target": "RH_FOOT"}]}

    Attributes:
        pairs (tuple[KeypointPair, ...]): Correspondences.
    """

    pairs: tuple[KeypointPair, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate that the map is not empty."""
        if not self.pairs:
            raise ValueError("a keypoint map needs at least one pair")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KeypointMap:
        """Build a map from its JSON form.

        Args:
            data (Mapping[str, Any]): ``{"keypoints": [...]}`` or ``{source: target}``.

        Returns:
            KeypointMap: The map.
        """
        if "keypoints" in data:
            return cls(tuple(KeypointPair(**dict(item)) for item in data["keypoints"]))
        return cls(tuple(KeypointPair(str(source), str(target)) for source, target in data.items()))

    @classmethod
    def from_json(cls, path: str | Path) -> KeypointMap:
        """Read a map from a JSON file.

        Args:
            path (str | Path): File.

        Returns:
            KeypointMap: The map.
        """
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            dict[str, Any]: ``{"keypoints": [...]}``.
        """
        return {"keypoints": [pair.__dict__.copy() for pair in self.pairs]}

    @property
    def source_links(self) -> list[str]:
        """Return the source links in pair order."""
        return [pair.source for pair in self.pairs]

    @property
    def target_links(self) -> list[str]:
        """Return the target links in pair order."""
        return [pair.target for pair in self.pairs]

    def check(self, source_graph: EmbodimentGraph, target_graph: EmbodimentGraph) -> None:
        """Check that every link exists on its robot.

        Args:
            source_graph (EmbodimentGraph): Source robot.
            target_graph (EmbodimentGraph): Target robot.

        Raises:
            ValueError: A link is unknown.
        """
        source_graph.link_index(self.source_links)
        target_graph.link_index(self.target_links)


@dataclass
class RetargetResult:
    """A retargeted motion of the target robot.

    Attributes:
        dt (float): Time between frames, the source's.
        joint_names (tuple[str, ...]): Target joints, in token order.
        positions (torch.Tensor): Float64 ``[T, J]`` joint positions.
        base_position (torch.Tensor | None): Float64 ``[T, 3]`` base origin in the world
            (floating targets).
        base_orientation (torch.Tensor | None): Float64 ``[T, 3, 3]`` world-from-base.
        scale (float): Size ratio applied to the source keypoints.
        keypoint_error (torch.Tensor): Float64 ``[T, K]`` distance in meters between each
            target keypoint and its scaled source keypoint.
        iterations (torch.Tensor): Long ``[T]`` solver iterations per frame.
    """

    dt: float
    joint_names: tuple[str, ...]
    positions: torch.Tensor
    base_position: torch.Tensor | None
    base_orientation: torch.Tensor | None
    scale: float
    keypoint_error: torch.Tensor
    iterations: torch.Tensor

    def to_trajectory(self, embodiment: int = 0) -> Trajectory:
        """Return the motion as a ``Trajectory`` for ``ExpertMotions`` and ``trajectory_state``.

        Velocities are left out; ``trajectory_state`` finite-differences them.

        Args:
            embodiment (int): Index of the target robot's record.

        Returns:
            Trajectory: Kinematic motion with the base pose of a floating target.
        """
        return Trajectory(
            embodiment=embodiment,
            dt=self.dt,
            positions=self.positions.clone(),
            base_orientation=None if self.base_orientation is None else self.base_orientation.clone(),
            base_position=None if self.base_position is None else self.base_position.clone(),
        )


def limited_joints(graph: EmbodimentGraph) -> torch.Tensor:
    """Return the bool ``[J]`` mask of joints with a position range (not continuous).

    Args:
        graph (EmbodimentGraph): Robot.

    Returns:
        torch.Tensor: True where ``lower < upper`` and the joint is not continuous.
    """
    continuous = torch.tensor([kind == "continuous" for kind in graph.joint_types])
    return ~continuous & (graph.upper > graph.lower)


def clamp_to_limits(graph: EmbodimentGraph, positions: torch.Tensor) -> torch.Tensor:
    """Clamp limited joints into their range.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]``.

    Returns:
        torch.Tensor: Clamped positions; continuous joints unchanged.
    """
    lower = graph.lower.to(positions)
    upper = graph.upper.to(positions)
    return torch.where(limited_joints(graph).to(positions.device), positions.clamp(lower, upper), positions)


def reference_pose(graph: EmbodimentGraph, pose: Mapping[str, float] | torch.Tensor | None = None) -> torch.Tensor:
    """Return a float64 ``[J]`` pose: zero clamped into the limits, with optional overrides.

    The zero-clamped default is the repository's convention for a nominal pose (the
    ``mujoco`` source's ``nominal_pose``).

    Args:
        graph (EmbodimentGraph): Robot.
        pose (Mapping[str, float] | torch.Tensor | None): Joint name to position, or a
            full ``[J]`` pose.

    Returns:
        torch.Tensor: Pose inside the limits.

    Raises:
        ValueError: A joint name is unknown or the tensor has the wrong shape.
    """
    if isinstance(pose, torch.Tensor):
        if pose.shape != (graph.num_joints,):
            raise ValueError(f"pose must have shape [{graph.num_joints}]")
        return clamp_to_limits(graph, pose.to(torch.float64))
    values = torch.zeros(graph.num_joints, dtype=torch.float64)
    unknown = sorted(set(pose or {}) - set(graph.joint_names))
    if unknown:
        raise ValueError(f"unknown joints {unknown}")
    for name, value in (pose or {}).items():
        values[graph.joint_names.index(name)] = float(value)
    return clamp_to_limits(graph, values)


def typical_link_length(graph: EmbodimentGraph) -> float:
    """Return the mean nonzero distance between a joint and its parent joint frame, in meters.

    Args:
        graph (EmbodimentGraph): Robot.

    Returns:
        float: ``l`` of Sugihara's damping bias; the characteristic length if every
            offset is zero.
    """
    offsets = graph.tree_transforms[:, :3, 3].norm(dim=-1)
    offsets = offsets[offsets > 1e-9]
    return float(offsets.mean()) if offsets.numel() else float(graph.characteristic_length)


def scale_ratio(
    source_graph: EmbodimentGraph,
    target_graph: EmbodimentGraph,
    keypoints: KeypointMap,
    method: str | float = "characteristic_length",
    source_pose: torch.Tensor | None = None,
    target_pose: torch.Tensor | None = None,
) -> float:
    """Return the factor that scales source keypoints to the target's size.

    Args:
        source_graph (EmbodimentGraph): Source robot.
        target_graph (EmbodimentGraph): Target robot.
        keypoints (KeypointMap): Correspondences (used by ``"keypoints"``).
        method (str | float): ``"characteristic_length"``, ``"keypoints"`` or a positive
            number (see the module docstring).
        source_pose (torch.Tensor | None): Source reference pose for ``"keypoints"``;
            zero clamped into the limits by default.
        target_pose (torch.Tensor | None): Target reference pose, likewise.

    Returns:
        float: Positive scale.

    Raises:
        ValueError: The method is unknown, or the ratio is not positive.
    """
    if isinstance(method, (int, float)) and not isinstance(method, bool):
        ratio = float(method)
    elif method == "characteristic_length":
        ratio = target_graph.characteristic_length / source_graph.characteristic_length
    elif method == "keypoints":
        source_pose = reference_pose(source_graph) if source_pose is None else source_pose
        target_pose = reference_pose(target_graph) if target_pose is None else target_pose
        source = source_graph.link_poses(source_pose, keypoints.source_links)[..., :3, 3].norm(dim=-1).mean()
        target = target_graph.link_poses(target_pose, keypoints.target_links)[..., :3, 3].norm(dim=-1).mean()
        ratio = float(target / source.clamp_min(1e-9))
    else:
        raise ValueError(f"scale must be a number or one of {SCALE_METHODS}")
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("the scale must be positive")
    return ratio


def _skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the ``[3, 3]`` cross-product matrix of a 3-vector."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((torch.stack((zero, -z, y)), torch.stack((z, zero, -x)), torch.stack((-y, x, zero))))


class _FrameSolver:
    """Levenberg-Marquardt IK of one robot for one frame of keypoint targets."""

    def __init__(
        self,
        graph: EmbodimentGraph,
        keypoints: KeypointMap,
        floating: bool,
        rest: torch.Tensor,
        rest_weight: float,
        damping_bias: float,
        max_iterations: int,
        tolerance: float,
    ) -> None:
        self.graph = graph
        self.links = keypoints.target_links
        self.floating = floating
        self.length = float(graph.characteristic_length)
        self.weights = torch.tensor([pair.weight for pair in keypoints.pairs], dtype=torch.float64)
        self.orientation_weights = torch.tensor([pair.orientation_weight for pair in keypoints.pairs], dtype=torch.float64)
        self.oriented = self.orientation_weights > 0
        self.rest = rest
        self.rest_weight = rest_weight
        self.damping_bias = damping_bias
        self.max_iterations = max_iterations
        self.tolerance = tolerance * self.length
        self.limited = limited_joints(graph)
        self.joints = graph.num_joints
        self.unknowns = self.joints + (6 if floating else 0)

    def keypoints(self, q: torch.Tensor, position: torch.Tensor, rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target keypoint positions ``[K, 3]`` and frames ``[K, 3, 3]`` in the world."""
        poses = self.graph.link_poses(q, self.links)
        points, frames = poses[:, :3, 3], poses[:, :3, :3]
        if self.floating:
            points = position + points @ rotation.transpose(0, 1)
            frames = rotation @ frames
        return points, frames

    def residual(self, delta, q, position, rotation, goal_points, goal_frames):
        """Weighted residual at ``q + delta`` (linearized base rotation; exact at delta = 0)."""
        joints = q + delta[: self.joints]
        if self.floating:
            position = position + self.length * delta[self.joints : self.joints + 3]
            rotation = rotation @ (torch.eye(3, dtype=delta.dtype) + _skew(delta[self.joints + 3 :]))
        points, frames = self.keypoints(joints, position, rotation)
        parts = [(self.weights[:, None] * (goal_points - points)).reshape(-1)]
        if bool(self.oriented.any()):
            axes = self.length * self.orientation_weights[self.oriented, None, None] * (goal_frames[self.oriented] - frames[self.oriented])
            parts.append(axes.reshape(-1))
        if self.rest_weight > 0:
            parts.append(math.sqrt(self.rest_weight) * (self.rest - joints))
        return torch.cat(parts)

    def solve(self, q, position, rotation, goal_points, goal_frames):
        """Run LM from a warm start; return the best joints, base pose and iteration count."""
        best = None
        iterations = 0
        previous = math.inf
        identity = torch.eye(self.unknowns, dtype=torch.float64)
        zero = torch.zeros(self.unknowns, dtype=torch.float64)
        for iterations in range(1, self.max_iterations + 1):

            def residual_and_value(delta):
                value = self.residual(delta, q, position, rotation, goal_points, goal_frames)
                return value, value.detach()

            jacobian, value = torch.func.jacrev(residual_and_value, has_aux=True)(zero)
            energy = 0.5 * float(value @ value)
            if best is None or energy < best[0]:
                best = (energy, q, position, rotation)
            # Converged, or stalled at an unreachable target (the least-squares pose).
            stalled = math.isfinite(previous) and 0 <= previous - energy <= STALL_FRACTION * previous
            if math.sqrt(2 * energy) <= self.tolerance or stalled:
                break
            previous = energy
            hessian = jacobian.T @ jacobian + (energy + self.damping_bias) * identity
            step = -torch.linalg.solve(hessian, jacobian.T @ value)
            if float(step.norm()) < 1e-12:
                break
            q = q + step[: self.joints]
            q = torch.where(self.limited, q.clamp(self.graph.lower, self.graph.upper), q)
            if self.floating:
                position = position + self.length * step[self.joints : self.joints + 3]
                rotation = rotation @ exp_rotation(step[self.joints + 3 :])
        value = self.residual(zero, q, position, rotation, goal_points, goal_frames)
        energy = 0.5 * float(value @ value)
        if energy < best[0]:
            best = (energy, q, position, rotation)
        return best[1], best[2], best[3], iterations


def retarget(
    source_graph: EmbodimentGraph,
    source_motion: torch.Tensor | Trajectory,
    target_graph: EmbodimentGraph,
    keypoints: KeypointMap,
    dt: float | None = None,
    base_position: torch.Tensor | None = None,
    base_orientation: torch.Tensor | None = None,
    scale: str | float = "characteristic_length",
    initial_pose: Mapping[str, float] | torch.Tensor | None = None,
    rest_weight: float = 0.0,
    damping_bias: float | None = None,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> RetargetResult:
    """Retarget a motion of the source robot to the target robot.

    Source keypoints come from ``EmbodimentGraph.link_poses`` and the source base pose,
    are multiplied by the size ratio, and IK finds the target pose of each frame (see the
    module docstring). A fixed-base target matches keypoints in the source's base frame;
    a floating target matches them in the world and solves its base pose too.

    Args:
        source_graph (EmbodimentGraph): Source robot.
        source_motion (torch.Tensor | Trajectory): ``[T, J_src]`` joint positions, or a
            ``Trajectory`` that also gives ``dt`` and the base pose.
        target_graph (EmbodimentGraph): Target robot.
        keypoints (KeypointMap): Link correspondences.
        dt (float | None): Frame period in s; required with a tensor motion.
        base_position (torch.Tensor | None): ``[T, 3]`` source base origin in the world.
        base_orientation (torch.Tensor | None): ``[T, 3, 3]`` source world-from-base.
        scale (str | float): ``"characteristic_length"``, ``"keypoints"`` or a number.
        initial_pose (Mapping[str, float] | torch.Tensor | None): Target pose that starts
            the first frame and is the default pose ``q_bar`` of the rest term; zero
            clamped into the limits by default.
        rest_weight (float): ``w`` of Peng et al.'s ``(q_bar - q)^T W (q_bar - q)`` with
            ``W = w I``; 0 (no value in the paper) leaves it out.
        damping_bias (float | None): Sugihara's ``w_bar`` in m^2; ``0.01 l^2`` by default.
        max_iterations (int): Solver iterations per frame at most ([H]).
        tolerance (float): Stop a frame when the residual norm is below this fraction of
            the target's characteristic length ([H]).

    Returns:
        RetargetResult: Target motion with the source's ``dt`` and the remaining
            keypoint errors.

    Raises:
        ValueError: Shapes, links or settings are invalid.
    """
    if isinstance(source_motion, Trajectory):
        if dt is not None or base_position is not None or base_orientation is not None:
            raise ValueError("a Trajectory already gives dt and the base pose")
        dt = source_motion.dt
        base_position, base_orientation = source_motion.base_position, source_motion.base_orientation
        source_motion = source_motion.positions
    if dt is None or dt <= 0:
        raise ValueError("dt must be positive")
    if max_iterations < 1 or tolerance <= 0 or rest_weight < 0:
        raise ValueError("max_iterations and tolerance must be positive and rest_weight nonnegative")
    positions = source_motion.to(torch.float64)
    if positions.ndim != 2 or positions.shape[1] != source_graph.num_joints or positions.shape[0] < 1:
        raise ValueError(f"source motion must have shape [T, {source_graph.num_joints}]")
    frames = positions.shape[0]
    keypoints.check(source_graph, target_graph)
    floating = bool(target_graph.floating_base)
    if base_position is not None:
        base_position = base_position.to(torch.float64)
        if base_position.shape != (frames, 3):
            raise ValueError("base_position must have shape [T, 3]")
    if base_orientation is not None:
        base_orientation = base_orientation.to(torch.float64)
        if base_orientation.shape != (frames, 3, 3):
            raise ValueError("base_orientation must have shape [T, 3, 3]")
    source_rotation = base_orientation if base_orientation is not None else torch.eye(3, dtype=torch.float64).expand(frames, 3, 3)
    source_translation = base_position if base_position is not None else torch.zeros((frames, 3), dtype=torch.float64)

    rest = reference_pose(target_graph, initial_pose)
    ratio = scale_ratio(source_graph, target_graph, keypoints, scale, target_pose=rest)
    source_poses = source_graph.link_poses(positions, keypoints.source_links)
    goal_points, goal_frames = source_poses[..., :3, 3], source_poses[..., :3, :3]
    if floating:
        # World keypoints: the floating target's base is solved too.
        goal_points = source_translation[:, None, :] + goal_points @ source_rotation.transpose(-1, -2)
        goal_frames = source_rotation[:, None] @ goal_frames
    goal_points = ratio * goal_points

    if damping_bias is None:
        damping_bias = DAMPING_BIAS_FRACTION * typical_link_length(target_graph) ** 2
    solver = _FrameSolver(target_graph, keypoints, floating, rest, rest_weight, damping_bias, max_iterations, tolerance)
    q = rest
    position = ratio * source_translation[0]
    rotation = source_rotation[0]
    joints_out, positions_out, rotations_out, errors, counts = [], [], [], [], []
    for frame in range(frames):
        if frame > 0 and floating:
            # Warm start: the previous solution moved by the source base's increment.
            position = position + ratio * (source_translation[frame] - source_translation[frame - 1])
            rotation = source_rotation[frame] @ source_rotation[frame - 1].transpose(0, 1) @ rotation
        q, position, rotation, count = solver.solve(q, position, rotation, goal_points[frame], goal_frames[frame])
        points, _ = solver.keypoints(q, position, rotation)
        joints_out.append(q)
        positions_out.append(position)
        rotations_out.append(rotation)
        errors.append((points - goal_points[frame]).norm(dim=-1))
        counts.append(count)
    return RetargetResult(
        dt=float(dt),
        joint_names=tuple(target_graph.joint_names),
        positions=torch.stack(joints_out),
        base_position=torch.stack(positions_out) if floating else None,
        base_orientation=torch.stack(rotations_out) if floating else None,
        scale=ratio,
        keypoint_error=torch.stack(errors),
        iterations=torch.tensor(counts, dtype=torch.long),
    )
