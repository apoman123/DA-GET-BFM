"""Rigid-body, actuator, and dynamics features for Physics Cross Attention.

Three feature groups describe each joint ``i`` (see ``model.physics_schema`` for
symbols, units, and frames):

* Rigid body ``[m_i, vech(I_i), r_i^CoM, a_i^axis, T_i^parent, jointType_i]``,
  from URDF.
* Actuator ``[tau^peak, tau^cont, omega^max, P^max, N, K_p, K_d, mode]``, from an
  ``ActuatorSpec`` per joint (URDF values plus your overrides).
* Dynamics ``[M_ii(q), c_i(q, qd), g_i(q), tau_hat_f, tau_{t-1}^applied,
  |g_i| / (tau^cont + eps), known flag]``, computed from the state.

Rigid body and actuator features depend only on the embodiment and are
computed once per robot. Dynamics features are split into tiers by what they
depend on (``model.physics_schema.FEATURE_UPDATE_TIERS``), and
``PhysicsFeatureCache`` recomputes each tier only when needed at control time.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import torch

from model.dynamics import (
    coriolis_efforts,
    friction_efforts,
    gravity_efforts,
)
from model.embodiment import EmbodimentGraph
from model.physics_schema import (
    ACTUATOR_FEATURE_SIZE,
    ACTUATOR_MODES,
    DYNAMICS_FEATURE_SIZE,
    JOINT_TYPE_NAMES,
    RIGID_BODY_FEATURE_SIZE,
)

EPSILON = 1e-6
MARGIN_NAMES = ("torque_peak", "torque_continuous", "power", "velocity", "position")
FEASIBILITY_NAMES = ("peak_feasible", "continuous_feasible")


@dataclass(frozen=True)
class ActuatorSpec:
    """Describe the actuator that drives one joint, in joint-side SI units.

    Torques are at the joint (after the gearbox). For prismatic joints read N for
    N m and m/s for rad/s.

    Attributes:
        peak_torque (float): tau^peak, short-term maximum effort.
        continuous_torque (float): tau^cont, effort sustainable indefinitely.
        max_velocity (float): omega^max, no-load speed limit.
        max_power (float): P^max, mechanical power limit in W.
        gear_ratio (float): N, motor-to-joint reduction.
        kp (float): K_p, position gain of a PD-controlled joint.
        kd (float): K_d, velocity gain.
        mode (str): ``torque``, ``position``, ``velocity``, or ``passive``.
    """

    peak_torque: float
    continuous_torque: float
    max_velocity: float
    max_power: float
    gear_ratio: float = 1.0
    kp: float = 0.0
    kd: float = 0.0
    mode: str = "torque"

    def __post_init__(self) -> None:
        """Validate values.

        Returns:
            None: The spec is valid after construction.

        Raises:
            ValueError: A value is negative or nonfinite, the continuous limit
                exceeds the peak, or the mode is unknown.
        """
        values = [getattr(self, item.name) for item in fields(self) if item.name != "mode"]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("actuator values must be finite and nonnegative")
        if self.mode not in ACTUATOR_MODES:
            raise ValueError(f"mode must be one of {ACTUATOR_MODES}")
        if self.continuous_torque > self.peak_torque:
            raise ValueError("continuous_torque cannot exceed peak_torque")
        if self.mode != "passive" and (
            self.peak_torque <= 0 or self.max_velocity <= 0 or self.max_power <= 0
        ):
            raise ValueError("an actuated joint needs positive torque, velocity, and power limits")


@dataclass(frozen=True)
class ActuatorDefaults:
    """Fill actuator values that URDF does not provide.

    URDF gives one ``effort`` and one ``velocity`` limit; these are taken as the
    peak torque and the no-load speed. The defaults below are assumptions; set
    real motor data through overrides whenever it is known.

    Attributes:
        continuous_ratio (float): tau^cont = ratio * tau^peak.
        power_ratio (float): P^max = ratio * tau^peak * omega^max. With 0.5 the
            torque limit starts to drop above half the no-load speed, a common
            shape for electric motors.
        mode (str): Control mode of every actuated joint.
        kp (float): Default position gain.
        kd (float): Default velocity gain.
        peak_torque (float | None): Used when URDF has no positive effort limit.
        max_velocity (float | None): Used when URDF has no positive velocity limit.
    """

    continuous_ratio: float = 0.5
    power_ratio: float = 0.5
    mode: str = "torque"
    kp: float = 0.0
    kd: float = 0.0
    peak_torque: float | None = None
    max_velocity: float | None = None


def actuator_specs(
    graph: EmbodimentGraph,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
    defaults: ActuatorDefaults | None = None,
) -> tuple[ActuatorSpec, ...]:
    """Build one actuator spec per joint from URDF limits and overrides.

    Args:
        graph (EmbodimentGraph): Robot.
        overrides (Mapping | None): Joint name or glob pattern (``"*_hip_*"``)
            to ``ActuatorSpec`` field values. Later patterns win over earlier
            ones; an exact name wins over every pattern.
        defaults (ActuatorDefaults | None): Values for fields URDF lacks.

    Returns:
        tuple[ActuatorSpec, ...]: Specs in token order. Virtual joints of an
            expanded floating base are ``passive`` with zero limits.

    Raises:
        ValueError: An actuated joint has no effort or velocity limit and no
            default or override supplies one, or an override names an unknown
            field.
    """
    defaults = defaults or ActuatorDefaults()
    overrides = dict(overrides or {})
    valid_fields = {item.name for item in fields(ActuatorSpec)}
    for values in overrides.values():
        unknown = set(values) - valid_fields
        if unknown:
            raise ValueError(f"unknown actuator fields {sorted(unknown)}")
    specs = []
    missing = []
    for index, name in enumerate(graph.joint_names):
        merged: dict[str, object] = {}
        for pattern, values in overrides.items():
            if pattern != name and fnmatch.fnmatchcase(name, pattern):
                merged.update(values)
        merged.update(overrides.get(name, {}))
        if not bool(graph.actuated[index]) and merged.get("mode", "passive") == "passive":
            specs.append(ActuatorSpec(0.0, 0.0, 0.0, 0.0, mode="passive"))
            continue
        peak = float(merged.get("peak_torque") or float(graph.effort_limits[index]))
        peak = peak or (defaults.peak_torque or 0.0)
        speed = float(merged.get("max_velocity") or float(graph.velocity_limits[index]))
        speed = speed or (defaults.max_velocity or 0.0)
        if peak <= 0 or speed <= 0:
            missing.append(name)
            continue
        values = {
            "peak_torque": peak,
            "continuous_torque": defaults.continuous_ratio * peak,
            "max_velocity": speed,
            "max_power": defaults.power_ratio * peak * speed,
            "gear_ratio": float(graph.gear_ratios[index]),
            "kp": defaults.kp,
            "kd": defaults.kd,
            "mode": defaults.mode,
        }
        values.update(merged)
        specs.append(ActuatorSpec(**values))
    if missing:
        raise ValueError(
            f"{graph.name}: no effort or velocity limit for {missing}; add them to the "
            "URDF, pass overrides, or set ActuatorDefaults.peak_torque/max_velocity"
        )
    return tuple(specs)


def actuator_tensors(
    specs: Sequence[ActuatorSpec], dtype: torch.dtype = torch.float64
) -> dict[str, torch.Tensor]:
    """Stack actuator specs into ``[J]`` tensors.

    Args:
        specs (Sequence[ActuatorSpec]): One spec per joint.
        dtype (torch.dtype): Floating dtype.

    Returns:
        dict[str, torch.Tensor]: One tensor per numeric field, plus ``mode``
            (long index into ``ACTUATOR_MODES``).
    """
    tensors = {
        item.name: torch.tensor([getattr(spec, item.name) for spec in specs], dtype=dtype)
        for item in fields(ActuatorSpec)
        if item.name != "mode"
    }
    tensors["mode"] = torch.tensor([ACTUATOR_MODES.index(spec.mode) for spec in specs])
    return tensors


def rigid_body_features(graph: EmbodimentGraph) -> torch.Tensor:
    """Return ``[m, vech(I), r^CoM, a^axis, T^parent, jointType]`` per joint.

    The inertia is about the body's center of mass, and it, the center of mass,
    and the axis are expressed in body frame ``B_i``. ``T^parent`` is the
    transform from ``B_p(i)`` to ``B_i`` at ``q_i = 0``: translation in m and the
    first two rotation columns.

    Args:
        graph (EmbodimentGraph): Robot.

    Returns:
        torch.Tensor: Float32 ``[J, 25]``.
    """
    inertia = graph.body_inertia
    vech = inertia[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    rotation = graph.tree_transforms[:, :3, :3]
    joint_type = torch.tensor(
        [[float(kind == name) for name in JOINT_TYPE_NAMES] for kind in graph.joint_types],
        dtype=torch.float64,
    )
    features = torch.cat(
        (
            graph.body_mass[:, None],
            vech,
            graph.body_com,
            graph.axes,
            graph.tree_transforms[:, :3, 3],
            rotation[:, :, :2].transpose(1, 2).reshape(-1, 6),
            joint_type,
        ),
        dim=1,
    )
    assert features.shape[1] == RIGID_BODY_FEATURE_SIZE
    return features.to(torch.float32)


def actuator_features(specs: Sequence[ActuatorSpec]) -> torch.Tensor:
    """Return ``[tau^peak, tau^cont, omega^max, P^max, N, K_p, K_d, mode]``.

    Args:
        specs (Sequence[ActuatorSpec]): One spec per joint.

    Returns:
        torch.Tensor: Float32 ``[J, 11]``; ``mode`` is one-hot.
    """
    tensors = actuator_tensors(specs)
    mode = torch.nn.functional.one_hot(tensors["mode"], len(ACTUATOR_MODES))
    features = torch.cat(
        (
            torch.stack(
                [
                    tensors[name]
                    for name in (
                        "peak_torque",
                        "continuous_torque",
                        "max_velocity",
                        "max_power",
                        "gear_ratio",
                        "kp",
                        "kd",
                    )
                ],
                dim=1,
            ),
            mode.to(torch.float64),
        ),
        dim=1,
    )
    assert features.shape[1] == ACTUATOR_FEATURE_SIZE
    return features.to(torch.float32)


def configuration_terms(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    gravity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the configuration tier: ``M_ii(q)`` and ``g(q)``.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        gravity (torch.Tensor | None): Gravity in the root frame.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Mass-matrix diagonal and gravity
            efforts, each ``[..., J]``.
    """
    diagonal = graph.mass_matrix(positions).diagonal(dim1=-2, dim2=-1)
    return diagonal, gravity_efforts(graph, positions, gravity)


def velocity_terms(
    graph: EmbodimentGraph, positions: torch.Tensor, velocities: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the velocity tier: ``c(q, qd)`` and the friction estimate.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Coriolis and friction efforts.
    """
    return coriolis_efforts(graph, positions, velocities), friction_efforts(graph, velocities)


def assemble_dynamics_features(
    mass_diagonal: torch.Tensor,
    coriolis: torch.Tensor,
    gravity: torch.Tensor,
    friction: torch.Tensor,
    applied_torque_prev: torch.Tensor | None,
    continuous_torque: torch.Tensor,
) -> torch.Tensor:
    """Stack dynamics terms into the 7 dynamics features.

    Args:
        mass_diagonal (torch.Tensor): ``[..., J]`` M_ii(q).
        coriolis (torch.Tensor): ``[..., J]`` c_i(q, qd).
        gravity (torch.Tensor): ``[..., J]`` g_i(q).
        friction (torch.Tensor): ``[..., J]`` friction estimate.
        applied_torque_prev (torch.Tensor | None): ``[..., J]`` measured
            efforts of the previous step, or None when unknown (features are
            then zero with the known flag off).
        continuous_torque (torch.Tensor): ``[J]`` tau^cont.

    Returns:
        torch.Tensor: Float32 ``[..., J, 7]``.
    """
    known = torch.ones_like(mass_diagonal)
    if applied_torque_prev is None:
        applied_torque_prev = torch.zeros_like(mass_diagonal)
        known = torch.zeros_like(mass_diagonal)
    load = gravity.abs() / (continuous_torque.to(gravity) + EPSILON)
    load = torch.where(continuous_torque.to(gravity) > 0, load, torch.zeros_like(load))
    features = torch.stack(
        (
            mass_diagonal,
            coriolis,
            gravity,
            friction,
            applied_torque_prev.to(mass_diagonal),
            load,
            known,
        ),
        dim=-1,
    )
    assert features.shape[-1] == DYNAMICS_FEATURE_SIZE
    return features.to(torch.float32)


def dynamics_features(
    graph: EmbodimentGraph,
    specs: Sequence[ActuatorSpec],
    positions: torch.Tensor,
    velocities: torch.Tensor,
    applied_torque_prev: torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute all dynamics features at once (no caching).

    Args:
        graph (EmbodimentGraph): Robot.
        specs (Sequence[ActuatorSpec]): Actuator specs, for tau^cont.
        positions (torch.Tensor): ``[..., J]`` q_t.
        velocities (torch.Tensor): ``[..., J]`` qd_t.
        applied_torque_prev (torch.Tensor | None): ``[..., J]`` tau_{t-1}.
        gravity (torch.Tensor | None): Gravity in the root frame.

    Returns:
        torch.Tensor: Float32 ``[..., J, 7]``.
    """
    positions = positions.to(torch.float64)
    velocities = velocities.to(torch.float64)
    mass_diagonal, gravity_term = configuration_terms(graph, positions, gravity)
    coriolis, friction = velocity_terms(graph, positions, velocities)
    return assemble_dynamics_features(
        mass_diagonal,
        coriolis,
        gravity_term,
        friction,
        applied_torque_prev,
        actuator_tensors(specs)["continuous_torque"],
    )


def torque_limit(tensors: Mapping[str, torch.Tensor], velocities: torch.Tensor) -> torch.Tensor:
    """Evaluate the torque-speed envelope ``tau^max(qd)``.

    ``tau^max = min(tau^peak, P^max / |qd|)`` for ``|qd| <= omega^max``, and zero
    beyond the no-load speed.

    Args:
        tensors (Mapping[str, torch.Tensor]): Output of ``actuator_tensors``.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.

    Returns:
        torch.Tensor: ``[..., J]`` available effort magnitude.
    """
    speed = velocities.abs()
    peak = tensors["peak_torque"].to(velocities)
    power = tensors["max_power"].to(velocities)
    limit = torch.minimum(peak, power / speed.clamp_min(EPSILON))
    return torch.where(speed <= tensors["max_velocity"].to(velocities), limit, torch.zeros_like(limit))


def requested_torque(
    tensors: Mapping[str, torch.Tensor],
    proposal: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Convert an action proposal into the effort it requests.

    ``torque`` mode: the proposal is the effort. ``position``: a PD target,
    ``K_p (a - q) - K_d qd``. ``velocity``: a velocity target, ``K_d (a - qd)``.
    ``passive``: zero.

    Args:
        tensors (Mapping[str, torch.Tensor]): Output of ``actuator_tensors``.
        proposal (torch.Tensor): ``[..., J]`` commands.
        positions (torch.Tensor): ``[..., J]`` q.
        velocities (torch.Tensor): ``[..., J]`` qd.

    Returns:
        torch.Tensor: ``[..., J]`` requested efforts.
    """
    mode = tensors["mode"].to(proposal.device)
    kp = tensors["kp"].to(proposal)
    kd = tensors["kd"].to(proposal)
    pd = kp * (proposal - positions) - kd * velocities
    velocity = kd * (proposal - velocities)
    zero = torch.zeros_like(proposal)
    return torch.where(
        mode == 0, proposal, torch.where(mode == 1, pd, torch.where(mode == 2, velocity, zero))
    )


def motor_labels(
    graph: EmbodimentGraph,
    tensors: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    velocities: torch.Tensor,
    torque: torch.Tensor,
    margin_clip: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute feasibility flags and capability margins for requested efforts.

    Margins are 1 at no load, 0 at the limit, and negative beyond it:

    * ``torque_peak``: ``1 - |tau| / (tau^max(qd) + eps)``, the short-term envelope.
    * ``torque_continuous``: ``1 - |tau| / (tau^cont + eps)``, the sustainable limit.
    * ``power``: ``1 - |tau qd| / (P^max + eps)``.
    * ``velocity``: ``1 - |qd| / (omega^max + eps)``.
    * ``position``: distance to the nearest joint limit over half the range.

    ``peak_feasible`` requires the peak, power, velocity, and position margins to
    be nonnegative; ``continuous_feasible`` also requires the continuous margin.
    A peak-feasible command is executable now but may not be sustainable.

    Args:
        graph (EmbodimentGraph): Robot, for position limits.
        tensors (Mapping[str, torch.Tensor]): Output of ``actuator_tensors``.
        positions (torch.Tensor): ``[..., J]`` q.
        velocities (torch.Tensor): ``[..., J]`` qd.
        torque (torch.Tensor): ``[..., J]`` requested efforts.
        margin_clip (float): Margins are clamped to ``[-margin_clip, 1]``.

    Returns:
        tuple: Flags ``[..., J, 2]``, margins ``[..., J, 5]``, and a margin mask
            ``[..., J, 5]`` that is False where a margin is undefined (passive
            joints; position margin of unlimited joints).
    """
    actuated = tensors["mode"].to(torque.device) != ACTUATOR_MODES.index("passive")
    lower = graph.lower.to(positions)
    upper = graph.upper.to(positions)
    limited = torch.tensor(
        [kind != "continuous" for kind in graph.joint_types], device=positions.device
    ) & (upper > lower)
    half_range = torch.where(limited, (upper - lower) / 2, torch.ones_like(upper))
    margins = torch.stack(
        (
            1 - torque.abs() / (torque_limit(tensors, velocities) + EPSILON),
            1 - torque.abs() / (tensors["continuous_torque"].to(torque) + EPSILON),
            1 - (torque * velocities).abs() / (tensors["max_power"].to(torque) + EPSILON),
            1 - velocities.abs() / (tensors["max_velocity"].to(torque) + EPSILON),
            torch.minimum(positions - lower, upper - positions) / half_range,
        ),
        dim=-1,
    ).clamp(-margin_clip, 1.0)
    mask = actuated[..., None].expand(margins.shape).clone()
    mask[..., 4] &= limited
    nonnegative = (margins >= 0) | ~mask
    peak = nonnegative[..., [0, 2, 3, 4]].all(dim=-1)
    continuous = peak & nonnegative[..., 1]
    flags = torch.stack((peak, continuous), dim=-1).to(torque.dtype)
    return flags, margins, mask


@dataclass
class _Tier:
    """Cached values of one dynamics tier for one environment row."""

    reference: torch.Tensor
    values: tuple[torch.Tensor, ...]
    age: int = 0


class PhysicsFeatureCache:
    """Serve dynamics features at control time, recomputing tiers only when needed.

    Each environment row has its own robot. Embodiment features (rigid body and
    actuator) are computed once at construction. Dynamics features are split by
    dependency:

    * configuration tier ``M_ii(q)``, ``g(q)``, ``|g| / tau^cont``: recomputed
      every ``configuration_interval`` steps, or sooner when any joint moves more
      than ``configuration_tolerance`` (rad or m) or gravity changes direction
      by more than ``gravity_tolerance`` (cosine distance) since the last
      refresh;
    * velocity tier ``c(q, qd)``, ``tau_f(qd)``: every ``velocity_interval``
      steps, when ``qd`` changes by more than ``velocity_tolerance``, or when
      the configuration tier refreshes;
    * measurement tier ``tau_{t-1}``: every step.

    Set an interval to 1 to recompute every step. Call ``reset(rows)`` when an
    environment resets.

    Attributes:
        graphs (tuple[EmbodimentGraph, ...]): One robot per row.
        rigid_body_features (torch.Tensor): Float32 ``[B, J, 25]``, zero padded.
        actuator_features (torch.Tensor): Float32 ``[B, J, 11]``, zero padded.
        joint_mask (torch.Tensor): Bool ``[B, J]``.
        refresh_counts (dict[str, int]): Number of recomputations per tier, for
            monitoring.
    """

    def __init__(
        self,
        graphs: Sequence[EmbodimentGraph],
        specs: Sequence[Sequence[ActuatorSpec]],
        max_joints: int | None = None,
        configuration_interval: int = 1,
        configuration_tolerance: float = 0.0,
        velocity_interval: int = 1,
        velocity_tolerance: float = 0.0,
        gravity_tolerance: float = 1e-4,
    ) -> None:
        """Precompute embodiment features for every row.

        Args:
            graphs (Sequence[EmbodimentGraph]): One robot per row.
            specs (Sequence[Sequence[ActuatorSpec]]): Actuator specs per row.
            max_joints (int | None): Padded joint count; defaults to the largest.
            configuration_interval (int): Maximum age in steps of M_ii and g.
            configuration_tolerance (float): Joint motion that forces a refresh;
                0 refreshes whenever q changes at all.
            velocity_interval (int): Maximum age in steps of c and tau_f.
            velocity_tolerance (float): Velocity change that forces a refresh.
            gravity_tolerance (float): Gravity direction change that forces a
                configuration refresh.

        Raises:
            ValueError: Inputs are inconsistent.
        """
        if len(graphs) != len(specs) or not graphs:
            raise ValueError("graphs and specs must be nonempty and of equal length")
        if configuration_interval < 1 or velocity_interval < 1:
            raise ValueError("intervals must be at least 1")
        self.graphs = tuple(graphs)
        self.specs = tuple(tuple(row) for row in specs)
        joint_count = max_joints or max(graph.num_joints for graph in graphs)
        self.joint_mask = torch.zeros((len(graphs), joint_count), dtype=torch.bool)
        self.rigid_body_features = torch.zeros((len(graphs), joint_count, RIGID_BODY_FEATURE_SIZE))
        self.actuator_features = torch.zeros((len(graphs), joint_count, ACTUATOR_FEATURE_SIZE))
        self._continuous = []
        for row, (graph, row_specs) in enumerate(zip(self.graphs, self.specs)):
            if len(row_specs) != graph.num_joints:
                raise ValueError("each row needs one actuator spec per joint")
            count = graph.num_joints
            self.joint_mask[row, :count] = True
            self.rigid_body_features[row, :count] = rigid_body_features(graph)
            self.actuator_features[row, :count] = actuator_features(row_specs)
            self._continuous.append(actuator_tensors(row_specs)["continuous_torque"])
        self.configuration_interval = configuration_interval
        self.configuration_tolerance = configuration_tolerance
        self.velocity_interval = velocity_interval
        self.velocity_tolerance = velocity_tolerance
        self.gravity_tolerance = gravity_tolerance
        self._configuration: list[_Tier | None] = [None] * len(graphs)
        self._velocity: list[_Tier | None] = [None] * len(graphs)
        self._gravity: list[torch.Tensor | None] = [None] * len(graphs)
        self.refresh_counts = {"configuration": 0, "velocity": 0}

    def reset(self, rows: Sequence[int] | None = None) -> None:
        """Drop cached dynamics so the next update recomputes them.

        Args:
            rows (Sequence[int] | None): Rows to reset; all when None.
        """
        for row in range(len(self.graphs)) if rows is None else rows:
            self._configuration[row] = None
            self._velocity[row] = None
            self._gravity[row] = None

    def update(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        applied_torque_prev: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return dynamics features for the current step.

        Args:
            positions (torch.Tensor): Padded ``[B, J]`` q_t.
            velocities (torch.Tensor): Padded ``[B, J]`` qd_t.
            applied_torque_prev (torch.Tensor | None): Padded ``[B, J]`` measured
                tau_{t-1}, or None if not measured.
            gravity (torch.Tensor | None): ``[3]`` or ``[B, 3]`` gravity in each
                root frame; defaults to -z.

        Returns:
            torch.Tensor: Float32 ``[B, J, 7]``, zero on padding.

        Raises:
            ValueError: Shapes do not match the cache.
        """
        if positions.shape != self.joint_mask.shape or velocities.shape != positions.shape:
            raise ValueError("positions and velocities must be [B, J] like the cache")
        output = torch.zeros((*positions.shape, DYNAMICS_FEATURE_SIZE))
        for row, graph in enumerate(self.graphs):
            count = graph.num_joints
            q = positions[row, :count].detach().to("cpu", torch.float64)
            qd = velocities[row, :count].detach().to("cpu", torch.float64)
            row_gravity = None
            if gravity is not None:
                row_gravity = (gravity if gravity.ndim == 1 else gravity[row]).detach()
                row_gravity = row_gravity.to("cpu", torch.float64)
            configuration, moved = self._configuration_tier(row, graph, q, row_gravity)
            velocity = self._velocity_tier(row, graph, q, qd, moved)
            torque = None
            if applied_torque_prev is not None:
                torque = applied_torque_prev[row, :count].detach().to("cpu", torch.float64)
            output[row, :count] = assemble_dynamics_features(
                configuration.values[0],
                velocity.values[0],
                configuration.values[1],
                velocity.values[1],
                torque,
                self._continuous[row],
            )
        return output.to(positions.device)

    def _configuration_tier(self, row, graph, q, gravity) -> tuple[_Tier, bool]:
        """Return the configuration tier of one row and whether it was refreshed."""
        tier = self._configuration[row]
        direction = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float64)
        if gravity is not None:
            direction = gravity / gravity.norm().clamp_min(EPSILON)
        previous = self._gravity[row]
        gravity_moved = previous is not None and float(1 - previous @ direction) > self.gravity_tolerance
        if (
            tier is None
            or gravity_moved
            or tier.age + 1 >= self.configuration_interval
            or float((q - tier.reference).abs().max()) > self.configuration_tolerance
        ):
            tier = _Tier(q.clone(), configuration_terms(graph, q, gravity))
            self._configuration[row] = tier
            self._gravity[row] = direction
            self.refresh_counts["configuration"] += 1
            return tier, True
        tier.age += 1
        return tier, False

    def _velocity_tier(self, row, graph, q, qd, configuration_refreshed) -> _Tier:
        """Return the velocity tier of one row, refreshing if stale.

        c(q, qd) also depends on q, so it is refreshed with the configuration tier.
        """
        tier = self._velocity[row]
        if (
            tier is None
            or configuration_refreshed
            or tier.age + 1 >= self.velocity_interval
            or float((qd - tier.reference).abs().max()) > self.velocity_tolerance
        ):
            tier = _Tier(qd.clone(), velocity_terms(graph, q, qd))
            self._velocity[row] = tier
            self.refresh_counts["velocity"] += 1
        else:
            tier.age += 1
        return tier

