"""Rigid-body, actuator, and dynamics features for Physics Cross Attention.

Three feature groups describe each joint ``i`` (see ``model.physics_schema`` for
symbols, units, and frames):

* Rigid body ``[m_i, vech(I_i), r_i^CoM, a_i^axis, T_i^parent, jointType_i,
  I_i^rotor]``, from URDF (and the rotor inertia setting).
* Actuator ``[tau^peak, tau^cont, K_p, K_d, k_f, k_m]``, from an
  ``ActuatorSpec`` per joint (URDF values plus your overrides).
* Dynamics ``[M_ii(q), c_i(q, qd), g_i(q), tau_hat_f, |g_i| / (tau^cont + eps)]``,
  computed from the sensed state and the model only. No dynamics feature needs a
  torque measurement or estimate.

Rigid body and actuator features depend only on the embodiment and are
computed once per robot. Dynamics features are split into tiers by what they
depend on (``model.physics_schema.FEATURE_UPDATE_TIERS``), and
``PhysicsFeatureCache`` recomputes each tier only when needed at control time.

References:

* ``M``, ``c`` and ``g`` come from ``model.dynamics`` (recursive Newton-Euler; see its
  references). The joint damping and friction are the URDF ``<dynamics>`` values (ROS
  URDF joint specification, http://wiki.ros.org/urdf/XML/joint). The reflected rotor
  inertia ``I^rotor`` is MuJoCo's joint ``armature`` (Todorov, Erez & Tassa, IROS 2012,
  DOI 10.1109/IROS.2012.6386109; Wensing et al., IEEE T-RO 33(3):509-522, 2017, DOI
  10.1109/TRO.2016.2640183).
* The two rotation columns of ``T^parent`` are the 6D rotation representation of
  Zhou et al. (CVPR 2019, DOI 10.1109/CVPR.2019.00589).
* Rotor thrust ``k_f w^2`` and drag moment ``k_m w^2`` follow the motor model of RotorS
  (Furrer et al., in *Robot Operating System (ROS): The Complete Reference (Volume
  1)*, Springer 2016, pp. 595-625, DOI 10.1007/978-3-319-26054-9_23).
* The PD drive ``K_p (a - q) - K_d qd`` is recomputed and clipped at every physics
  step, as legged_gym does with its torque limits (Rudin et al., CoRL 2021, PMLR
  164:91-100, https://github.com/leggedrobotics/legged_gym). The
  speed-dependent envelope ``min(tau^peak, P^max / |qd|)``, the margins and the
  defaults of ``ActuatorDefaults`` are this repository's model, not taken from a source.
* ``froude_scale_base_features``: dynamic similarity (Alexander & Jayes, J. Zool.
  201:135-152, 1983, DOI 10.1111/j.1469-7998.1983.tb04266.x) with standard gravity
  ``g = 9.80665 m/s^2`` (3rd CGPM 1901, Declaration 2, DOI 10.59161/CGPM1901DECL2E).
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import torch

from model.dynamics import (
    BaseMotion,
    coriolis_efforts,
    default_gravity,
    friction_efforts,
    gravity_efforts,
    recursive_newton_euler,
)
from model.embodiment import EmbodimentGraph
from model.physics_schema import (
    ACTUATOR_FEATURE_SIZE,
    ACTUATOR_MODES,
    DYNAMICS_FEATURE_SIZE,
    JOINT_TYPE_NAMES,
    RIGID_BODY_FEATURE_SIZE,
    ROOT_FEATURE_SIZE,
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
        thrust_coefficient (float): k_f of a propeller on a ``thrust`` joint:
            thrust ``k_f w^2`` in N along the joint axis (negative: along -axis).
        torque_coefficient (float): k_m of a propeller: drag moment ``k_m w^2``
            in N m opposing the spin.
        mode (str): ``torque``, ``position``, ``velocity``, ``passive``, or
            ``thrust`` (a rotor whose speed is commanded and whose propeller
            pushes the base).
    """

    peak_torque: float
    continuous_torque: float
    max_velocity: float
    max_power: float
    gear_ratio: float = 1.0
    kp: float = 0.0
    kd: float = 0.0
    thrust_coefficient: float = 0.0
    torque_coefficient: float = 0.0
    mode: str = "torque"

    def __post_init__(self) -> None:
        """Validate values.

        Returns:
            None: The spec is valid after construction.

        Raises:
            ValueError: A value is negative or nonfinite, the continuous limit
                exceeds the peak, or the mode is unknown.
        """
        values = [
            getattr(self, item.name)
            for item in fields(self)
            if item.name not in ("mode", "thrust_coefficient")
        ]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("actuator values must be finite and nonnegative")
        if not math.isfinite(self.thrust_coefficient):
            raise ValueError("thrust_coefficient must be finite")
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
    """Return ``[m, vech(I), r^CoM, a^axis, T^parent, jointType, I^rotor]`` per joint.

    The inertia is about the body's center of mass, and it, the center of mass,
    and the axis are expressed in body frame ``B_i``. ``T^parent`` is the
    transform from ``B_p(i)`` to ``B_i`` at ``q_i = 0``: translation in m and the
    first two rotation columns. ``I^rotor`` is ``graph.joint_armature``.

    Args:
        graph (EmbodimentGraph): Robot.

    Returns:
        torch.Tensor: Float32 ``[J, 26]``.
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
            graph.joint_armature[:, None],
        ),
        dim=1,
    )
    assert features.shape[1] == RIGID_BODY_FEATURE_SIZE
    return features.to(torch.float32)


def actuator_features(specs: Sequence[ActuatorSpec]) -> torch.Tensor:
    """Return joint effort limits, PD gains, and optional propeller coefficients.

    Args:
        specs (Sequence[ActuatorSpec]): One spec per joint.

    Returns:
        torch.Tensor: Float32 ``[J, 6]`` in schema column order.
    """
    tensors = actuator_tensors(specs)
    features = torch.stack(
        [
            tensors[name]
            for name in (
                "peak_torque",
                "continuous_torque",
                "kp",
                "kd",
                "thrust_coefficient",
                "torque_coefficient",
            )
        ],
        dim=1,
    )
    assert features.shape[1] == ACTUATOR_FEATURE_SIZE
    return features.to(torch.float32)


def root_features(graph: EmbodimentGraph) -> torch.Tensor:
    """Return the root token's features ``[m_0, r_0^CoM, vech(I_0), m, L, floating]``.

    The root body is the base link with every link fixed to it; its inertia is about
    its own center of mass in the root frame. ``m`` and ``L`` are the robot's total
    mass and characteristic length (proposal C1 and A3, ``docs/zero_shot.md``).

    Args:
        graph (EmbodimentGraph): Robot.

    Returns:
        torch.Tensor: Float32 ``[ROOT_FEATURE_SIZE]``.
    """
    inertia = graph.base_inertia if graph.base_inertia is not None else torch.zeros((3, 3), dtype=torch.float64)
    com = graph.base_com if graph.base_com is not None else torch.zeros(3, dtype=torch.float64)
    vech = inertia[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    features = torch.cat(
        (
            torch.tensor([float(graph.base_mass)], dtype=torch.float64),
            com.to(torch.float64),
            vech.to(torch.float64),
            torch.tensor(
                [graph.total_mass, graph.characteristic_length, 1.0 if graph.floating_base else 0.0],
                dtype=torch.float64,
            ),
        )
    )
    assert features.shape[0] == ROOT_FEATURE_SIZE
    return features.to(torch.float32)


def froude_scale_base_features(features: torch.Tensor, length: torch.Tensor | float, columns: slice = slice(3, 9)) -> torch.Tensor:
    """Make base angular velocity and base linear velocity / L dimensionless.

    The base-feature layout puts the angular velocity (rad/s) and the linear velocity
    divided by the robot's size (1/s) in ``columns``; both become dimensionless when
    multiplied by ``sqrt(L / g)``, giving ``omega sqrt(L/g)`` and the Froude-like
    ``v / sqrt(g L)`` (Alexander & Jayes, J. Zool. 201:135-152, 1983; proposal A3), with
    ``g = 9.80665 m/s^2``.

    Args:
        features (torch.Tensor): ``[..., F]`` base features.
        length (torch.Tensor | float): Characteristic length, broadcastable to
            ``features[..., 0]``.
        columns (slice): Columns with units of 1/s.

    Returns:
        torch.Tensor: Scaled copy.
    """
    scaled = features.clone()
    period = torch.sqrt(torch.as_tensor(length, dtype=features.dtype, device=features.device) / 9.80665)
    scaled[..., columns] = scaled[..., columns] * period[..., None]
    return scaled


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


def loss_efforts(
    graph: EmbodimentGraph, tensors: Mapping[str, torch.Tensor] | None, velocities: torch.Tensor
) -> torch.Tensor:
    """Return joint friction plus propeller drag ``k_m w |w|`` of thrust joints.

    Args:
        graph (EmbodimentGraph): Robot.
        tensors (Mapping | None): ``actuator_tensors`` output, for rotor drag.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.

    Returns:
        torch.Tensor: ``[..., J]`` efforts that oppose motion.
    """
    efforts = friction_efforts(graph, velocities)
    if tensors is not None:
        thrust = tensors["mode"].to(velocities.device) == ACTUATOR_MODES.index("thrust")
        drag = tensors["torque_coefficient"].to(velocities) * velocities * velocities.abs()
        efforts = efforts + torch.where(thrust, drag, torch.zeros_like(drag))
    return efforts


def velocity_terms(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the velocity tier: ``c(q, qd, base twist)`` and losses.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        base_linear_velocity (torch.Tensor | None): ``[..., 3]`` floating base.
        base_angular_velocity (torch.Tensor | None): ``[..., 3]`` floating base.
        tensors (Mapping | None): Actuator tensors, for rotor drag.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Coriolis efforts and friction (plus
            rotor drag).
    """
    coriolis = coriolis_efforts(
        graph, positions, velocities, base_linear_velocity, base_angular_velocity
    )
    return coriolis, loss_efforts(graph, tensors, velocities)


@dataclass(frozen=True)
class DynamicsTerms:
    """Both tiers of dynamics terms at one state.

    Attributes:
        mass_matrix (torch.Tensor): ``[..., J, J]`` M(q).
        gravity (torch.Tensor): ``[..., J]`` g(q).
        coriolis (torch.Tensor): ``[..., J]`` c(q, qd, base twist).
        friction (torch.Tensor): ``[..., J]`` friction plus rotor drag.
    """

    mass_matrix: torch.Tensor
    gravity: torch.Tensor
    coriolis: torch.Tensor
    friction: torch.Tensor


def dynamics_terms(
    graph: EmbodimentGraph,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    gravity: torch.Tensor | None = None,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> DynamicsTerms:
    """Compute ``configuration_terms`` and ``velocity_terms`` together, for a control loop.

    Forward kinematics runs once, and g(q) and c(q, qd) come from one Newton-Euler
    pass over a batch of two problems instead of two passes.

    Args:
        graph (EmbodimentGraph): Robot.
        positions (torch.Tensor): ``[..., J]`` joint positions.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        gravity (torch.Tensor | None): Gravity in the base frame.
        base_linear_velocity (torch.Tensor | None): ``[..., 3]`` floating base.
        base_angular_velocity (torch.Tensor | None): ``[..., 3]`` floating base.
        tensors (Mapping | None): Actuator tensors, for rotor drag.

    Returns:
        DynamicsTerms: The full mass matrix (its diagonal is the configuration
            term), gravity, Coriolis and friction efforts.
    """
    kinematics = graph._kinematics(positions)
    shape = positions.shape[:-1]

    def pair(first: torch.Tensor | None, second: torch.Tensor | None, size: tuple) -> torch.Tensor:
        """Stack the gravity problem's and the Coriolis problem's input."""
        zeros = positions.new_zeros(size)
        first = zeros if first is None else first.to(positions).expand(size)
        second = zeros if second is None else second.to(positions).expand(size)
        return torch.stack((first, second))

    vector = (*shape, 3)
    gravity_vector = default_gravity(positions.dtype, positions.device) if gravity is None else gravity
    efforts = recursive_newton_euler(
        graph,
        positions.expand(2, *positions.shape),
        pair(None, velocities, positions.shape),
        positions.new_zeros((2, *positions.shape)),
        pair(gravity_vector, None, vector),
        BaseMotion(pair(None, base_linear_velocity, vector), pair(None, base_angular_velocity, vector)),
        kinematics=tuple(item.expand(2, *item.shape) for item in kinematics),
    ).efforts
    return DynamicsTerms(
        graph.mass_matrix(positions, kinematics), efforts[0], efforts[1], loss_efforts(graph, tensors, velocities)
    )


def assemble_dynamics_features(
    mass_diagonal: torch.Tensor,
    coriolis: torch.Tensor,
    gravity: torch.Tensor,
    friction: torch.Tensor,
    continuous_torque: torch.Tensor,
) -> torch.Tensor:
    """Stack dynamics terms into the 5 dynamics features.

    Args:
        mass_diagonal (torch.Tensor): ``[..., J]`` M_ii(q).
        coriolis (torch.Tensor): ``[..., J]`` c_i(q, qd).
        gravity (torch.Tensor): ``[..., J]`` g_i(q).
        friction (torch.Tensor): ``[..., J]`` friction estimate.
        continuous_torque (torch.Tensor): ``[J]`` tau^cont.

    Returns:
        torch.Tensor: Float32 ``[..., J, 5]``.
    """
    continuous = continuous_torque.to(gravity)
    load = gravity.abs() / (continuous + EPSILON)
    load = torch.where(continuous > 0, load, torch.zeros_like(load))
    features = torch.stack((mass_diagonal, coriolis, gravity, friction, load), dim=-1)
    assert features.shape[-1] == DYNAMICS_FEATURE_SIZE
    return features.to(torch.float32)


def dynamics_features(
    graph: EmbodimentGraph,
    specs: Sequence[ActuatorSpec],
    positions: torch.Tensor,
    velocities: torch.Tensor,
    gravity: torch.Tensor | None = None,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute all dynamics features at once (no caching).

    Args:
        graph (EmbodimentGraph): Robot.
        specs (Sequence[ActuatorSpec]): Actuator specs, for tau^cont and rotor drag.
        positions (torch.Tensor): ``[..., J]`` q_t.
        velocities (torch.Tensor): ``[..., J]`` qd_t.
        gravity (torch.Tensor | None): Gravity in the base frame.
        base_linear_velocity (torch.Tensor | None): ``[..., 3]`` floating base.
        base_angular_velocity (torch.Tensor | None): ``[..., 3]`` floating base.

    Returns:
        torch.Tensor: Float32 ``[..., J, 5]``.
    """
    positions = positions.to(torch.float64)
    velocities = velocities.to(torch.float64)
    tensors = actuator_tensors(specs)
    mass_diagonal, gravity_term = configuration_terms(graph, positions, gravity)
    coriolis, friction = velocity_terms(
        graph, positions, velocities, base_linear_velocity, base_angular_velocity, tensors
    )
    return assemble_dynamics_features(mass_diagonal, coriolis, gravity_term, friction, tensors["continuous_torque"])


def envelope_limit(
    velocities: torch.Tensor, peak_torque: torch.Tensor, max_power: torch.Tensor, max_velocity: torch.Tensor
) -> torch.Tensor:
    """Evaluate the torque-speed envelope from its three parameters.

    ``tau^max = min(tau^peak, P^max / |qd|)`` for ``|qd| <= omega^max``, and zero beyond
    the no-load speed. This is the one definition of the envelope: ``torque_limit``,
    ``ActionMap``, the MuJoCo environment's per-physics-step clip and the Isaac Lab
    actuator all call it.

    Args:
        velocities (torch.Tensor): ``[..., J]`` joint velocities.
        peak_torque (torch.Tensor): ``[J]`` (or broadcastable) peak effort.
        max_power (torch.Tensor): Mechanical power limit.
        max_velocity (torch.Tensor): No-load speed.

    Returns:
        torch.Tensor: ``[..., J]`` available effort magnitude.
    """
    speed = velocities.abs()
    peak = peak_torque.to(velocities)
    limit = torch.minimum(peak, max_power.to(velocities) / speed.clamp_min(EPSILON))
    return torch.where(speed <= max_velocity.to(velocities), limit, torch.zeros_like(limit))


def envelope_clip(
    torque: torch.Tensor, velocities: torch.Tensor, peak_torque: torch.Tensor, max_power: torch.Tensor, max_velocity: torch.Tensor
) -> torch.Tensor:
    """Clip efforts into ``[-tau^max(qd), tau^max(qd)]`` (``envelope_limit``)."""
    limit = envelope_limit(velocities.to(torque), peak_torque, max_power, max_velocity)
    return torch.maximum(torch.minimum(torque, limit), -limit)


def torque_limit(tensors: Mapping[str, torch.Tensor], velocities: torch.Tensor) -> torch.Tensor:
    """Evaluate the torque-speed envelope ``tau^max(qd)`` of a robot (``envelope_limit``).

    Args:
        tensors (Mapping[str, torch.Tensor]): Output of ``actuator_tensors``.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.

    Returns:
        torch.Tensor: ``[..., J]`` available effort magnitude.
    """
    return envelope_limit(velocities, tensors["peak_torque"], tensors["max_power"], tensors["max_velocity"])


def pd_envelope_torque(
    tensors: Mapping[str, torch.Tensor], targets: torch.Tensor, positions: torch.Tensor, velocities: torch.Tensor
) -> torch.Tensor:
    """Return the effort a position-mode PD drive applies, inside the envelope.

    ``clip(K_p (a - q) - K_d qd, -tau^max(qd), tau^max(qd))`` for position-mode joints with
    ``K_p > 0``, zero for the others. Simulators call it at every physics step, the way a
    motor driver limits current inside its own loop.

    Args:
        tensors (Mapping[str, torch.Tensor]): Output of ``actuator_tensors``.
        targets (torch.Tensor): ``[..., J]`` position targets.
        positions (torch.Tensor): ``[..., J]`` q now.
        velocities (torch.Tensor): ``[..., J]`` qd now.

    Returns:
        torch.Tensor: ``[..., J]`` applied efforts.
    """
    targets = targets.to(torch.float64)
    kp = tensors["kp"].to(targets)
    kd = tensors["kd"].to(targets)
    position_mode = (tensors["mode"].to(targets.device) == ACTUATOR_MODES.index("position")) & (kp > 0)
    request = kp * (targets - positions.to(targets)) - kd * velocities.to(targets)
    applied = envelope_clip(request, velocities.to(targets), tensors["peak_torque"], tensors["max_power"], tensors["max_velocity"])
    return torch.where(position_mode, applied, torch.zeros_like(applied))


def torque_limit_excess(tensors: Mapping[str, torch.Tensor], torque: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
    """Return how far efforts exceed the torque-speed envelope, over peak torque.

    ``relu(|tau| - tau^max(qd)) / tau^peak`` per joint, zero for passive joints. It is
    zero inside the envelope and dimensionless, so it can be summed over joints of
    robots of any size.

    Args:
        tensors (Mapping[str, torch.Tensor]): Output of ``actuator_tensors``.
        torque (torch.Tensor): ``[..., J]`` requested or applied efforts.
        velocities (torch.Tensor): ``[..., J]`` joint velocities.

    Returns:
        torch.Tensor: ``[..., J]`` excess, nonnegative.
    """
    actuated = tensors["mode"].to(torque.device) != ACTUATOR_MODES.index("passive")
    excess = (torque.abs() - torque_limit(tensors, velocities.to(torque))).clamp_min(0)
    excess = excess / (tensors["peak_torque"].to(torque) + EPSILON)
    return torch.where(actuated, excess, torch.zeros_like(excess))


def envelope_targets(
    tensors: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Move PD position targets so the effort they request stays inside the envelope.

    For a position-mode joint the drive requests ``K_p (a - q) - K_d qd``. Where that
    exceeds ``tau^max(qd)``, the target is moved to the one whose request equals the
    limit, ``a' = q + (clip(tau) + K_d qd) / K_p``. Other joints are unchanged. The same
    function runs in the training simulators and on the robot, so the policy is trained
    on the targets it will send.

    Args:
        tensors (Mapping[str, torch.Tensor]): Output of ``actuator_tensors``.
        targets (torch.Tensor): ``[..., J]`` position targets.
        positions (torch.Tensor): ``[..., J]`` q at which the targets are sent.
        velocities (torch.Tensor): ``[..., J]`` qd at that time.

    Returns:
        torch.Tensor: ``[..., J]`` targets whose request is within ``tau^max(qd)``.
    """
    targets = targets.to(torch.float64)
    positions = positions.to(targets)
    velocities = velocities.to(targets)
    kp = tensors["kp"].to(targets)
    kd = tensors["kd"].to(targets)
    position_mode = (tensors["mode"].to(targets.device) == ACTUATOR_MODES.index("position")) & (kp > 0)
    request = kp * (targets - positions) - kd * velocities
    limit = torque_limit(tensors, velocities)
    clipped = torch.maximum(torch.minimum(request, limit), -limit)
    moved = positions + (clipped + kd * velocities) / kp.clamp_min(EPSILON)
    return torch.where(position_mode & (clipped != request), moved, targets)


def requested_torque(
    tensors: Mapping[str, torch.Tensor],
    proposal: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Convert an action proposal into the effort it requests.

    ``torque`` mode: the proposal is the effort. ``position``: a PD target,
    ``K_p (a - q) - K_d qd``. ``velocity``: a velocity target, ``K_d (a - qd)``.
    ``thrust``: a rotor speed target, ``K_d (a - qd) + k_m a |a|`` (speed
    tracking plus the propeller drag at the target speed). ``passive``: zero.

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
    rotor = velocity + tensors["torque_coefficient"].to(proposal) * proposal * proposal.abs()
    zero = torch.zeros_like(proposal)
    return torch.where(
        mode == 0,
        proposal,
        torch.where(
            mode == 1,
            pd,
            torch.where(mode == 2, velocity, torch.where(mode == 4, rotor, zero)),
        ),
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
      the configuration tier refreshes.

    No tier reads a torque: every feature comes from the sensed state and the model.

    Set an interval to 1 to recompute every step. Call ``reset(rows)`` when an
    environment resets.

    Attributes:
        graphs (tuple[EmbodimentGraph, ...]): One robot per row.
        rigid_body_features (torch.Tensor): Float32 ``[B, J, 26]``, zero padded.
        actuator_features (torch.Tensor): Float32 ``[B, J, 6]``, zero padded.
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
        self._tensors = []
        for row, (graph, row_specs) in enumerate(zip(self.graphs, self.specs)):
            if len(row_specs) != graph.num_joints:
                raise ValueError("each row needs one actuator spec per joint")
            count = graph.num_joints
            self.joint_mask[row, :count] = True
            self.rigid_body_features[row, :count] = rigid_body_features(graph)
            self.actuator_features[row, :count] = actuator_features(row_specs)
            self._tensors.append(actuator_tensors(row_specs))
            self._continuous.append(self._tensors[-1]["continuous_torque"])
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
        gravity: torch.Tensor | None = None,
        base_linear_velocity: torch.Tensor | None = None,
        base_angular_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return dynamics features for the current step.

        Args:
            positions (torch.Tensor): Padded ``[B, J]`` q_t.
            velocities (torch.Tensor): Padded ``[B, J]`` qd_t.
            gravity (torch.Tensor | None): ``[3]`` or ``[B, 3]`` gravity in each
                base frame (projected gravity times 9.81 for a floating base);
                defaults to -z.
            base_linear_velocity (torch.Tensor | None): ``[B, 3]`` base frame.
            base_angular_velocity (torch.Tensor | None): ``[B, 3]`` base frame.

        Returns:
            torch.Tensor: Float32 ``[B, J, 5]``, zero on padding.

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
            twist = [
                None if value is None else value[row].detach().to("cpu", torch.float64)
                for value in (base_linear_velocity, base_angular_velocity)
            ]
            configuration, moved = self._configuration_tier(row, graph, q, row_gravity)
            velocity = self._velocity_tier(row, graph, q, qd, moved, twist)
            output[row, :count] = assemble_dynamics_features(
                configuration.values[0],
                velocity.values[0],
                configuration.values[1],
                velocity.values[1],
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

    def _velocity_tier(self, row, graph, q, qd, configuration_refreshed, twist) -> _Tier:
        """Return the velocity tier of one row, refreshing if stale.

        c(q, qd, base twist) also depends on q, so it is refreshed with the
        configuration tier; the base twist counts as part of the velocity.
        """
        tier = self._velocity[row]
        reference = torch.cat([qd] + [value for value in twist if value is not None])
        if (
            tier is None
            or configuration_refreshed
            or tier.age + 1 >= self.velocity_interval
            or tier.reference.shape != reference.shape
            or float((reference - tier.reference).abs().max()) > self.velocity_tolerance
        ):
            tier = _Tier(
                reference.clone(),
                velocity_terms(graph, q, qd, twist[0], twist[1], self._tensors[row]),
            )
            self._velocity[row] = tier
            self.refresh_counts["velocity"] += 1
        else:
            tier.age += 1
        return tier

