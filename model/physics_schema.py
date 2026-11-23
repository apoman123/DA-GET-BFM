"""Schema of the physics features read by Physics Cross Attention.

Every feature has a symbol, an SI unit, the frame it is expressed in, and a
scale ``s``. The physics tokenizer maps a raw value ``x`` to ``asinh(x / s)``,
which is linear for ``|x| << s`` and logarithmic above, so a 20 g finger link
and a 10 kg thigh both land in a useful range (the inverse hyperbolic sine
transformation of Burbidge, Magee & Robb, J. Am. Stat. Assoc. 83(401):123-127, 1988,
DOI 10.1080/01621459.1988.10478575; the scales ``s`` are this repository's choice).
Features with ``scale=None`` (unit vectors, rotation columns, one-hot flags) are
passed through unchanged. The rotation columns ``R6d`` are the 6D rotation
representation of Zhou et al. (CVPR 2019, DOI 10.1109/CVPR.2019.00589).

Frames:

* ``B_i``: body frame of joint ``i``. Its origin is joint ``i``'s origin and its
  axes are the joint frame's axes; it moves rigidly with the body that joint
  ``i`` moves (the child link after fixed links are merged). At ``q_i = 0`` it is
  the URDF child link frame.
* ``B_p(i)``: body frame of the parent joint of ``i``, or the root link frame
  for joints mounted on the root.
* ``joint``: a joint-space quantity of joint ``i``, no spatial frame.

Units: revolute and continuous joints use N m, rad, rad/s; prismatic joints
use N, m, m/s. Both are written as ``N m | N`` and so on.

Dimensions. Each feature also records its physical dimension as exponents of
mass, length and time (``dimensions``; ``prismatic_dimensions`` when a joint-space
quantity of a prismatic joint differs). With ``DaGetConfig.dimensionless_physics``
the encoder divides every feature by the unit ``m^a L^b T^c`` built from the
robot's total mass ``m``, characteristic length ``L`` and time ``T = sqrt(L / g)``
(dynamic similarity: Alexander & Jayes, J. Zool. 201:135-152, 1983, DOI
10.1111/j.1469-7998.1983.tb04266.x; Buckingham, Phys. Rev. 4(4):345-376, 1914, DOI
10.1103/PhysRev.4.345). ``g`` is standard gravity, ``STANDARD_GRAVITY`` = 9.80665 m/s^2
(3rd CGPM 1901, Declaration 2, DOI 10.59161/CGPM1901DECL2E). The asinh
scales of dimensionless features are the SI scales divided by the same unit of a
reference robot (``REFERENCE_MASS``, ``REFERENCE_LENGTH``), so a robot of that size
is tokenized exactly as with SI inputs.

The root group (one token per robot) describes the root body and the robot's
absolute size. It is never made dimensionless: it is where the absolute scale
lives (MetaMorph also gives the torso a token: Gupta et al., ICLR 2022,
arXiv:2203.11931).
"""

from __future__ import annotations

from dataclasses import dataclass

JOINT_TYPE_NAMES = ("revolute", "continuous", "prismatic")
ACTUATOR_MODES = ("torque", "position", "velocity", "passive", "thrust")
STANDARD_GRAVITY = 9.80665  # m/s^2, conventional value of g_n (3rd CGPM 1901, Declaration 2)
# Median total mass and characteristic length of the four robots measured at their
# standing poses (Go1 13.1 kg / 0.31 m, G1 33.3 kg / 0.77 m with g1_29dof_rev_1_0.urdf,
# ANYmal D 57.0 kg / 0.55 m, H1 59.3 kg / 1.00 m; docs/zero_shot.md, evidence R9).
REFERENCE_MASS = 45.15
REFERENCE_LENGTH = 0.66


@dataclass(frozen=True)
class PhysicsFeature:
    """Describe one scalar physics feature.

    Attributes:
        name (str): Column name.
        symbol (str): Mathematical symbol, as used in the design notes.
        unit (str): SI unit.
        frame (str): Frame the value is expressed in (see module docstring).
        scale (float | None): asinh scale, or None to pass the value through.
        dimensions (tuple[int, int, int]): Exponents of mass, length and time of
            the unit (revolute joints for joint-space quantities).
        prismatic_dimensions (tuple[int, int, int] | None): Exponents for
            prismatic joints when they differ.
    """

    name: str
    symbol: str
    unit: str
    frame: str
    scale: float | None
    dimensions: tuple[int, int, int] = (0, 0, 0)
    prismatic_dimensions: tuple[int, int, int] | None = None


def _vector(prefix: str, symbol: str, unit: str, frame: str, scale: float | None, dimensions=(0, 0, 0)):
    return tuple(
        PhysicsFeature(f"{prefix}_{axis}", f"{symbol}[{axis}]", unit, frame, scale, dimensions)
        for axis in "xyz"
    )


MASS = (1, 0, 0)
LENGTH = (0, 1, 0)
INERTIA = (1, 2, 0)
TORQUE = (1, 2, -2)
FORCE = (1, 1, -2)


RIGID_BODY_FEATURES: tuple[PhysicsFeature, ...] = (
    PhysicsFeature("mass", "m_i", "kg", "joint", 0.01, MASS),
    # vech(I_i): the 6 independent entries of the symmetric inertia tensor about
    # the body's center of mass, expressed in body frame B_i.
    PhysicsFeature("inertia_xx", "vech(I_i)[xx]", "kg m^2", "B_i @ CoM", 1e-5, INERTIA),
    PhysicsFeature("inertia_xy", "vech(I_i)[xy]", "kg m^2", "B_i @ CoM", 1e-5, INERTIA),
    PhysicsFeature("inertia_xz", "vech(I_i)[xz]", "kg m^2", "B_i @ CoM", 1e-5, INERTIA),
    PhysicsFeature("inertia_yy", "vech(I_i)[yy]", "kg m^2", "B_i @ CoM", 1e-5, INERTIA),
    PhysicsFeature("inertia_yz", "vech(I_i)[yz]", "kg m^2", "B_i @ CoM", 1e-5, INERTIA),
    PhysicsFeature("inertia_zz", "vech(I_i)[zz]", "kg m^2", "B_i @ CoM", 1e-5, INERTIA),
    *_vector("com", "r_i^CoM", "m", "B_i", 0.01, LENGTH),
    *_vector("axis", "a_i^axis", "1 (unit vector)", "B_i", None),
    *_vector("parent_translation", "T_i^parent.p", "m", "B_p(i)", 0.01, LENGTH),
    *(
        PhysicsFeature(
            f"parent_rotation_{index}",
            f"T_i^parent.R6d[{index}]",
            "1 (rotation columns)",
            "B_p(i)",
            None,
        )
        for index in range(6)
    ),
    *(
        PhysicsFeature(f"type_{name}", f"jointType_i[{name}]", "1 (one-hot)", "joint", None)
        for name in JOINT_TYPE_NAMES
    ),
    # Reflected rotor inertia of the joint's drive (EmbodimentGraph.armature).
    PhysicsFeature("armature", "I_i^rotor (reflected)", "kg m^2 | kg", "joint", 1e-5, INERTIA, MASS),
)

ACTUATOR_FEATURES: tuple[PhysicsFeature, ...] = (
    PhysicsFeature("peak_torque", "tau_i^peak", "N m | N", "joint", 1.0, TORQUE, FORCE),
    PhysicsFeature("continuous_torque", "tau_i^cont", "N m | N", "joint", 1.0, TORQUE, FORCE),
    PhysicsFeature("kp", "K_p,i", "N m/rad | N/m", "joint", 10.0, TORQUE, (1, 0, -2)),
    PhysicsFeature("kd", "K_d,i", "N m s/rad | N s/m", "joint", 1.0, (1, 2, -1), (1, 0, -1)),
    # Propeller coefficients of thrust-mode joints (multirotors); zero otherwise.
    PhysicsFeature("thrust_coefficient", "k_f,i (thrust = k_f w^2 along a_i)", "N s^2/rad^2", "B_i", 1e-6, (1, 1, 0)),
    PhysicsFeature("torque_coefficient", "k_m,i (drag = k_m w^2)", "N m s^2/rad^2", "joint", 1e-8, INERTIA),
)

# Every dynamics feature is a function of the sensed state (q, qd, gravity
# direction, base twist) and the robot model only. None of them needs a torque
# measurement or a torque estimate, so they are exact (up to the model) in
# pretraining, in simulation, and on a real robot alike.
DYNAMICS_FEATURES: tuple[PhysicsFeature, ...] = (
    PhysicsFeature("mass_diagonal", "M_ii(q_t) (with rotor inertia)", "kg m^2 | kg", "joint", 1e-3, INERTIA, MASS),
    PhysicsFeature("coriolis", "c_i(q_t, qd_t, base twist)", "N m | N", "joint", 0.1, TORQUE, FORCE),
    PhysicsFeature("gravity", "g_i(q_t) with gravity in the base frame", "N m | N", "joint", 0.1, TORQUE, FORCE),
    PhysicsFeature("friction", "tau_hat_f,i (joint friction + rotor drag)", "N m | N", "joint", 0.1, TORQUE, FORCE),
    PhysicsFeature("gravity_load", "|g_i| / (tau_i^cont + eps)", "1", "joint", 1.0),
)

# One token per robot: the root body (the base link and everything fixed to it) and
# the robot's absolute size. Proposal C1 (docs/zero_shot.md): the encoder otherwise
# never sees the base's inertia (rigid-body features describe each joint's child body).
ROOT_FEATURES: tuple[PhysicsFeature, ...] = (
    PhysicsFeature("root_mass", "m_0", "kg", "root", 0.01, MASS),
    *_vector("root_com", "r_0^CoM", "m", "root", 0.01, LENGTH),
    *(
        PhysicsFeature(f"root_inertia_{comp}", f"vech(I_0)[{comp}]", "kg m^2", "root @ CoM", 1e-5, INERTIA)
        for comp in ("xx", "xy", "xz", "yy", "yz", "zz")
    ),
    PhysicsFeature("total_mass", "m", "kg", "robot", 1.0, MASS),
    PhysicsFeature("characteristic_length", "L", "m", "robot", 0.01, LENGTH),
    PhysicsFeature("floating_base", "floating", "1 (flag)", "robot", None),
)

PHYSICS_FEATURE_GROUPS: dict[str, tuple[PhysicsFeature, ...]] = {
    "rigid_body": RIGID_BODY_FEATURES,
    "actuator": ACTUATOR_FEATURES,
    "dynamics": DYNAMICS_FEATURES,
    "root": ROOT_FEATURES,
}

# How often each feature has to be recomputed. "embodiment" features change only
# when the robot (or its randomized physical parameters) changes; the
# dynamics features are split by what they depend on.
FEATURE_UPDATE_TIERS: dict[str, str] = {
    **{feature.name: "embodiment" for feature in RIGID_BODY_FEATURES + ACTUATOR_FEATURES + ROOT_FEATURES},
    "mass_diagonal": "configuration",  # q (and nothing else)
    "gravity": "configuration",  # q and the gravity direction
    "gravity_load": "configuration",
    "coriolis": "velocity",  # q, qd, and the base twist
    "friction": "velocity",  # qd
}


def feature_names(group: str) -> tuple[str, ...]:
    """Return the column names of a feature group.

    Args:
        group (str): ``rigid_body``, ``actuator``, or ``dynamics``.

    Returns:
        tuple[str, ...]: Names in column order.
    """
    return tuple(feature.name for feature in PHYSICS_FEATURE_GROUPS[group])


def feature_scales(group: str) -> tuple[float, ...]:
    """Return asinh scales of a group, with 0 for pass-through features.

    Args:
        group (str): ``rigid_body``, ``actuator``, or ``dynamics``.

    Returns:
        tuple[float, ...]: One scale per column; 0 means pass through.
    """
    return tuple(feature.scale or 0.0 for feature in PHYSICS_FEATURE_GROUPS[group])


def dimension_exponents(group: str) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Return per-feature unit exponents of a group for revolute and prismatic joints.

    Args:
        group (str): A key of ``PHYSICS_FEATURE_GROUPS``.

    Returns:
        tuple: Revolute exponents and prismatic exponents, one ``(mass, length,
            time)`` triple per feature.
    """
    features = PHYSICS_FEATURE_GROUPS[group]
    revolute = tuple(feature.dimensions for feature in features)
    prismatic = tuple(feature.prismatic_dimensions or feature.dimensions for feature in features)
    return revolute, prismatic


def reference_unit(dimensions: tuple[int, int, int]) -> float:
    """Return ``m^a L^b T^c`` of the reference robot, with ``T = sqrt(L / g)``.

    Args:
        dimensions (tuple[int, int, int]): Mass, length and time exponents.

    Returns:
        float: Unit value in SI.
    """
    mass, length, time = dimensions
    period = (REFERENCE_LENGTH / STANDARD_GRAVITY) ** 0.5
    return REFERENCE_MASS**mass * REFERENCE_LENGTH**length * period**time


def dimensionless_scales(group: str) -> tuple[float, ...]:
    """Return asinh scales of a group's dimensionless features (0 passes through).

    Each is the SI scale divided by the reference robot's unit of that feature,
    using the revolute dimensions.

    Args:
        group (str): A key of ``PHYSICS_FEATURE_GROUPS``.

    Returns:
        tuple[float, ...]: One scale per feature.
    """
    return tuple(
        0.0 if not feature.scale else feature.scale / reference_unit(feature.dimensions)
        for feature in PHYSICS_FEATURE_GROUPS[group]
    )


RIGID_BODY_FEATURE_SIZE = len(RIGID_BODY_FEATURES)
ROOT_FEATURE_SIZE = len(ROOT_FEATURES)
ACTUATOR_FEATURE_SIZE = len(ACTUATOR_FEATURES)
DYNAMICS_FEATURE_SIZE = len(DYNAMICS_FEATURES)
