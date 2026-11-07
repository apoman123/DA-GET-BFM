"""Schema of the physics features read by Physics Cross Attention.

Every feature has a symbol, an SI unit, the frame it is expressed in, and a
scale ``s``. The physics tokenizer maps a raw value ``x`` to ``asinh(x / s)``,
which is linear for ``|x| << s`` and logarithmic above, so a 20 g finger link
and a 10 kg thigh both land in a useful range. Features with ``scale=None``
(unit vectors, rotation columns, one-hot flags) are passed through unchanged.

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
"""

from __future__ import annotations

from dataclasses import dataclass

JOINT_TYPE_NAMES = ("revolute", "continuous", "prismatic")
ACTUATOR_MODES = ("torque", "position", "velocity", "passive", "thrust")


@dataclass(frozen=True)
class PhysicsFeature:
    """Describe one scalar physics feature.

    Attributes:
        name (str): Column name.
        symbol (str): Mathematical symbol, as used in the design notes.
        unit (str): SI unit.
        frame (str): Frame the value is expressed in (see module docstring).
        scale (float | None): asinh scale, or None to pass the value through.
    """

    name: str
    symbol: str
    unit: str
    frame: str
    scale: float | None


def _vector(prefix: str, symbol: str, unit: str, frame: str, scale: float | None):
    return tuple(
        PhysicsFeature(f"{prefix}_{axis}", f"{symbol}[{axis}]", unit, frame, scale)
        for axis in "xyz"
    )


RIGID_BODY_FEATURES: tuple[PhysicsFeature, ...] = (
    PhysicsFeature("mass", "m_i", "kg", "joint", 0.01),
    # vech(I_i): the 6 independent entries of the symmetric inertia tensor about
    # the body's center of mass, expressed in body frame B_i.
    PhysicsFeature("inertia_xx", "vech(I_i)[xx]", "kg m^2", "B_i @ CoM", 1e-5),
    PhysicsFeature("inertia_xy", "vech(I_i)[xy]", "kg m^2", "B_i @ CoM", 1e-5),
    PhysicsFeature("inertia_xz", "vech(I_i)[xz]", "kg m^2", "B_i @ CoM", 1e-5),
    PhysicsFeature("inertia_yy", "vech(I_i)[yy]", "kg m^2", "B_i @ CoM", 1e-5),
    PhysicsFeature("inertia_yz", "vech(I_i)[yz]", "kg m^2", "B_i @ CoM", 1e-5),
    PhysicsFeature("inertia_zz", "vech(I_i)[zz]", "kg m^2", "B_i @ CoM", 1e-5),
    *_vector("com", "r_i^CoM", "m", "B_i", 0.01),
    *_vector("axis", "a_i^axis", "1 (unit vector)", "B_i", None),
    *_vector("parent_translation", "T_i^parent.p", "m", "B_p(i)", 0.01),
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
)

ACTUATOR_FEATURES: tuple[PhysicsFeature, ...] = (
    PhysicsFeature("peak_torque", "tau_i^peak", "N m | N", "joint", 1.0),
    PhysicsFeature("continuous_torque", "tau_i^cont", "N m | N", "joint", 1.0),
    PhysicsFeature("max_velocity", "omega_i^max", "rad/s | m/s", "joint", 1.0),
    PhysicsFeature("max_power", "P_i^max", "W", "joint", 10.0),
    PhysicsFeature("gear_ratio", "N_i", "1", "joint", 1.0),
    PhysicsFeature("kp", "K_p,i", "N m/rad | N/m", "joint", 10.0),
    PhysicsFeature("kd", "K_d,i", "N m s/rad | N s/m", "joint", 1.0),
    # Propeller coefficients of thrust-mode joints (multirotors); zero otherwise.
    PhysicsFeature("thrust_coefficient", "k_f,i (thrust = k_f w^2 along a_i)", "N s^2/rad^2", "B_i", 1e-6),
    PhysicsFeature("torque_coefficient", "k_m,i (drag = k_m w^2)", "N m s^2/rad^2", "joint", 1e-8),
    *(
        PhysicsFeature(f"mode_{mode}", f"mode_i[{mode}]", "1 (one-hot)", "joint", None)
        for mode in ACTUATOR_MODES
    ),
)

DYNAMICS_FEATURES: tuple[PhysicsFeature, ...] = (
    PhysicsFeature("mass_diagonal", "M_ii(q_t)", "kg m^2 | kg", "joint", 1e-3),
    PhysicsFeature("coriolis", "c_i(q_t, qd_t, base twist)", "N m | N", "joint", 0.1),
    PhysicsFeature("gravity", "g_i(q_t) with gravity in the base frame", "N m | N", "joint", 0.1),
    PhysicsFeature("friction", "tau_hat_f,i (joint friction + rotor drag)", "N m | N", "joint", 0.1),
    PhysicsFeature("applied_torque_prev", "tau_i,t-1^applied", "N m | N", "joint", 0.1),
    PhysicsFeature("gravity_load", "|g_i| / (tau_i^cont + eps)", "1", "joint", 1.0),
    PhysicsFeature(
        "applied_torque_known",
        "source of tau_t-1: 1 measured, 0.5 inverse-dynamics estimate, 0 unknown",
        "1 (flag)",
        "joint",
        None,
    ),
)

PHYSICS_FEATURE_GROUPS: dict[str, tuple[PhysicsFeature, ...]] = {
    "rigid_body": RIGID_BODY_FEATURES,
    "actuator": ACTUATOR_FEATURES,
    "dynamics": DYNAMICS_FEATURES,
}

# How often each feature has to be recomputed. "embodiment" features change only
# when the robot (or its randomized physical parameters) changes; the
# dynamics features are split by what they depend on.
FEATURE_UPDATE_TIERS: dict[str, str] = {
    **{feature.name: "embodiment" for feature in RIGID_BODY_FEATURES + ACTUATOR_FEATURES},
    "mass_diagonal": "configuration",  # q (and nothing else)
    "gravity": "configuration",  # q and the gravity direction
    "gravity_load": "configuration",
    "coriolis": "velocity",  # q, qd, and the base twist
    "friction": "velocity",  # qd
    "applied_torque_prev": "measurement",  # new every control step
    "applied_torque_known": "measurement",
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


RIGID_BODY_FEATURE_SIZE = len(RIGID_BODY_FEATURES)
ACTUATOR_FEATURE_SIZE = len(ACTUATOR_FEATURES)
DYNAMICS_FEATURE_SIZE = len(DYNAMICS_FEATURES)
