# Physics features: sources and calculation

[← Back to the README](../README.md)

What the encoder's three physics feature groups contain, where each value comes from, and
what a robot must supply at run time. How the groups enter the encoder is in
[Physics Cross Attention](encoder.md#6-physics-cross-attention); the dynamics behind them
are in [Dynamics and torque](dynamics.md).

## Contents

- [Overview](#overview)
- [Rigid-body features: 26 values per joint](#rigid-body-features-26-values-per-joint)
- [Actuator features: 6 values per joint](#actuator-features-6-values-per-joint)
- [Dynamics features: 5 values per joint and time step](#dynamics-features-5-values-per-joint-and-time-step)
- [What a robot needs at run time](#what-a-robot-needs-at-run-time)
- [References](#references)

## Overview

The encoder receives three physics feature groups with one row per joint, in
the same order as `graph.joint_names`. `rigid_body_features` has shape
`[B, J, 26]`, `actuator_features` has `[B, J, 6]`, and
`dynamics_features` has `[B, J, 5]`; `B` is the batch size and `J` is the
padded joint count. Padded rows are zero and excluded by `joint_mask`. The
column names, units, and normalization scales are defined in
[`model/physics_schema.py`](../model/physics_schema.py). Joint-space values use
SI units: revolute joints use rad, rad/s, and N m; prismatic joints use m,
m/s, and N.

**No feature is a torque.** Every value comes from the robot model (URDF,
actuator data) and the sensed state (joint encoders, IMU). A robot, a simulator,
and a recorded dataset therefore produce the same features, and nothing needs a
torque sensor or a torque estimate. Earlier versions also fed the previous joint
torque (`[B, J, 7]` dynamics features, `[B, J, 25]` rigid-body features); see
[why that was removed](dynamics.md#why-training-uses-no-estimated-torque).
Checkpoints of that schema are rejected with a clear error and must be
pretrained again.

## Rigid-body features: 26 values per joint

These describe the link driven by each joint. `EmbodimentGraph.from_urdf`
reads the URDF's (the ROS Unified Robot Description Format) joint origins, axes, and link `<inertial>` elements, merges
fixed links, and places the result in the joint's body frame `B_i`. Its origin
is the joint origin; the parent frame `B_p(i)` is the parent joint's body frame
or the root link frame. [`rigid_body_features(graph)`](../model/physics.py)
packs the following values once per robot:

| Columns | Count | How they are obtained or calculated |
|---|---:|---|
| `mass` | 1 | Child body's mass `m_i` in kg from URDF inertials, including merged fixed links. |
| `inertia_xx`, `inertia_xy`, `inertia_xz`, `inertia_yy`, `inertia_yz`, `inertia_zz` | 6 | Independent entries of the symmetric inertia tensor about that body's center of mass, rotated into `B_i`, in kg m². |
| `com_x`, `com_y`, `com_z` | 3 | Center-of-mass offset from the joint origin, expressed in `B_i`, in m. |
| `axis_x`, `axis_y`, `axis_z` | 3 | Unit joint axis expressed in `B_i`. |
| `parent_translation_x/y/z` | 3 | Translation from `B_p(i)` to `B_i` at joint position zero, in m. |
| `parent_rotation_0` … `parent_rotation_5` | 6 | First two columns of that zero-position rotation matrix, flattened column first: the continuous 6D rotation representation of Zhou et al. (CVPR 2019). |
| `type_revolute`, `type_continuous`, `type_prismatic` | 3 | One-hot joint type from the URDF. |
| `armature` | 1 | Reflected rotor inertia `I_i^rotor` of the joint's drive (kg m², or kg for prismatic joints), `EmbodimentGraph.armature`. URDF has none: set it with the source key `armature` (one value or `{joint glob: value}`), `from_urdf(armature=...)` or `graph.with_armature(...)`. It adds `I_i^rotor` to `M_ii` and `I_i^rotor q̈_i` to the joint effort, like MuJoCo's and Isaac Lab's `armature`. |

These values change only when the robot geometry, inertial parameters or rotor
inertia change. A URDF needs valid inertial data for dynamics calculations;
measured masses and inertias are preferable to placeholders. The rotor inertia
matters most on light, highly geared joints, where it can exceed the link's own
inertia (Wensing et al., IEEE T-RO 2017).

## Actuator features: 6 values per joint

[`actuator_specs(graph)`](../model/physics.py) reads each URDF joint's `effort`
limit. URDFs usually do not provide continuous limits, controller gains, or
propeller data; those values come from defaults or explicit `overrides`. The
default assumptions are `continuous_torque = 0.5 × peak_torque` and zero PD
gains and propeller coefficients. The actuator spec still retains speed,
power, gear ratio, and control mode for simulation and physics labels, but
these four fields are **not encoder features**. Missing positive effort or
velocity limits on an actuated joint require an override or
`ActuatorDefaults`. [`actuator_features(specs)`](../model/physics.py) packs:

| Columns | Count | Meaning and source |
|---|---:|---|
| `peak_torque`, `continuous_torque` | 2 | Short-term joint-side effort from URDF `effort` or override; sustainable effort from motor data or the default ratio. Units N m (N for prismatic joints). |
| `kp`, `kd` | 2 | Position and velocity controller gains from the actual controller, `ActuatorDefaults`, or an override. |
| `thrust_coefficient`, `torque_coefficient` | 2 | Propeller coefficients `k_f` and `k_m` from rotor specifications: thrust is `k_f ω²` along the axis, and drag magnitude is `k_m ω²` (the standard rotor model, Mahony, Kumar & Corke, IEEE RAM 2012). Zero for ordinary joints. |

Supply real motor/controller data when available, for example
`actuator_specs(graph, overrides={"*_knee_joint": {"continuous_torque": 23.7,
"kp": 40.0, "kd": 1.0, "mode": "position"}})`. Dataset configs also accept
`actuator_overrides` by joint name or glob pattern. These features are static
until actuator parameters or the control mode change.

## Dynamics features: 5 values per joint and time step

These use the current joint position `q_t`, velocity `qd_t`, gravity in the
base frame, and the base twist. The functions in
[`model/dynamics.py`](../model/dynamics.py) compute the robot's mass matrix and
inverse-dynamics terms with the rigid-body algorithms in Featherstone (2008); [`dynamics_features`](../model/physics.py) assembles them
in this order:

| Column | Calculation | Inputs and update trigger |
|---|---|---|
| `mass_diagonal` | `M_ii(q_t)`, diagonal of the joint-space mass matrix, including the rotor inertia. | Joint positions, inertial parameters or rotor inertia change. |
| `coriolis` | `c_i(q_t, qd_t, base twist)`, Coriolis and centrifugal effort. | Joint positions/velocities or floating-base velocity change. |
| `gravity` | `g_i(q_t)`, gravity effort from inverse dynamics. | Joint positions or gravity direction in the base frame change. |
| `friction` | Joint viscous/Coulomb estimate `d_i qd_i + μ_i tanh(qd_i / 0.01)`, plus `k_m ω_i\|ω_i\|` for thrust joints. | Joint velocity changes; coefficients come from the robot model. |
| `gravity_load` | `\|g_i\| / (τ_i^cont + 10⁻⁶)`; zero for passive joints. | Gravity effort or continuous effort limit changes. |

For a floating base, pass IMU projected gravity multiplied by `9.81 m/s²`
and base linear/angular velocity in the base frame (the policy does not see the
base linear velocity unless `observe_base_linear_velocity` is set, as in
BFM-Zero).

```python
import torch

from model import EmbodimentGraph, JointFeatureConfig
from model.physics import (
    PhysicsFeatureCache, actuator_features, actuator_specs,
    dynamics_features, rigid_body_features,
)

features = JointFeatureConfig.da_get_zero()
graph = EmbodimentGraph.from_urdf("assets/urdf/go1.urdf", features, floating_base=True, armature=0.01)
specs = actuator_specs(graph)
body = rigid_body_features(graph)         # [J, 26], computed once
motor = actuator_features(specs)          # [J, 6], computed once

# Replace these example values with joint encoders and IMU data. No torque sensor is needed.
q_current = torch.zeros(1, graph.num_joints)
qd_current = torch.zeros_like(q_current)
gravity_base = torch.tensor([[0.0, 0.0, -9.81]])
dynamic = dynamics_features(             # [N, J, 5], recompute as the state changes
    graph, specs, q_current, qd_current, gravity=gravity_base,
)

# For mixed-robot batches, the cache stacks and pads the static groups and
# refreshes the dynamics terms according to their dependencies.
physics = PhysicsFeatureCache([graph], [specs])
dynamic_batch = physics.update(q_current, qd_current, gravity=gravity_base)
```

Replace the example state with `q_current` and `qd_current` `[N, J]` from
joint sensors. Pass `base_linear_velocity` and `base_angular_velocity` `[N, 3]`
to include floating-base motion in `c`; `gravity_base` is `[N, 3]`. For a
mixed-robot cache, give one graph/spec sequence per row and padded `[B, J]`
state arrays. Set the cache intervals to 1 for fresh values every step; larger
intervals reuse configuration and velocity terms until their age or change
thresholds trigger a refresh. Call `physics.reset()` after an environment
reset. The physics tokenizers map dimensional columns with
`asinh(value / scale)` and pass unit vectors and one-hot flags through, then use
learned projections. See [Physics Cross Attention](encoder.md#6-physics-cross-attention)
for how these groups enter the encoder.

## What a robot needs at run time

| Signal | Source on a robot | Used for |
|---|---|---|
| Joint positions and velocities `q`, `qd` | Joint encoders | Tokens, dynamics features, mass matrix |
| Gravity direction and base angular velocity | IMU | Tokens, dynamics features |
| Previous target and the PD gains | The controller's own command and configuration | Previous-action token features: the target and the effort `K_p (a − q) − K_d qd` it requests, computed, not measured |
| Base linear velocity | State estimator (optional) | Only with `observe_base_linear_velocity` |
| Joint torque | **Not needed** | — |

The PD gains on the robot must equal the training gains (`gain_scale`, or the
`kp`/`kd` overrides); otherwise the requested-effort features mean something
else. Targets go through [`ActionMap`](../model/rl/actions.py), the same code the
simulators use, so the effort the robot's drives request stays inside each
actuator's torque-speed envelope ([details](bfm.md#actuator-torque-limits)).

## References

- URDF, the Unified Robot Description Format of ROS (Open Robotics):
  <https://wiki.ros.org/urdf/XML>.
- Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. *On the Continuity of
  Rotation Representations in Neural Networks.* CVPR 2019, pp. 5738–5746. DOI
  10.1109/CVPR.2019.00589 (6D rotation columns).
- Patrick M. Wensing, Albert Wang, Sangok Seok, David Otten, Jeffrey Lang, and Sangbae
  Kim. *Proprioceptive Actuator Design in the MIT Cheetah: Impact Mitigation and
  High-Bandwidth Physical Interaction for Dynamic Legged Robots.* IEEE T-RO
  33(3):509–522, 2017. DOI 10.1109/TRO.2016.2640183 (reflected rotor inertia).
- Roy Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008. DOI
  10.1007/978-1-4899-7560-7 (mass matrix and inverse dynamics).
- Robert Mahony, Vijay Kumar, and Peter Corke. *Multirotor Aerial Vehicles: Modeling,
  Estimation, and Control of Quadrotor.* IEEE Robotics & Automation Magazine
  19(3):20–32, 2012. DOI 10.1109/MRA.2012.2206474 (rotor thrust and drag).
- Emanuel Todorov, Tom Erez, and Yuval Tassa. *MuJoCo: A Physics Engine for Model-Based
  Control.* IROS 2012, pp. 5026–5033. DOI 10.1109/IROS.2012.6386109, and Mayank Mittal
  et al. *Orbit: A Unified Simulation Framework for Interactive Robot Learning
  Environments.* IEEE RA-L 8(6):3740–3747, 2023 (Isaac Lab): both define `armature`
  the same way.
- Yitang Li et al. *BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid
  Control Using Unsupervised Reinforcement Learning.* ICLR 2026, arXiv:2511.04131 (the
  policy does not observe the base linear velocity).

All authors of the multi-author works are listed in [References](references.md).
