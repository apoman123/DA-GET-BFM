# Dynamics and torque estimation

[← Back to the README](../README.md)

The rigid-body dynamics, how torques are estimated when a dataset has none, and how accurate those estimates are.

## Contents

- [Rigid-body dynamics](#rigid-body-dynamics)
- [Estimating joint torques without torque sensors](#estimating-joint-torques-without-torque-sensors)
  - [Why the error is large, and what training does about it](#why-the-error-is-large-and-what-training-does-about-it)

## Rigid-body dynamics

`model/dynamics.py` implements the recursive Newton-Euler algorithm (RNEA),
batched over any leading dimensions, for fixed and floating bases. With
generalized velocity `ν = (v_base, ω_base, qd)` in the base frame:

```text
M_full(q) ν̇ + h(q, ν) = [ W_ext ; τ ] + Σ_c J_cᵀ f_c + rotor thrust and drag
```

For a fixed base this reduces to `τ = M(q) qdd + c(q, qd) + g(q) + τ_f(qd)`.

| Function | Returns |
|---|---|
| `recursive_newton_euler(graph, q, qd, qdd, gravity, base, body_wrenches, base_external_wrench)` | joint efforts, the base wrench the environment must supply, and frame velocities |
| `floating_mass_matrix(graph, q)` | `[..., 6+J, 6+J]` `M_full`; its joint block is `graph.mass_matrix(q)` |
| `floating_forward_dynamics(...)` | base linear and angular acceleration and `qdd` |
| `forward_dynamics(...)` / `inverse_dynamics(...)` | fixed-base (or known base motion) versions |
| `estimate_joint_efforts(...)` | torques from motion alone, with contacts and thrust (next section) |
| `distribute_contact_forces(...)` | contact forces (and moments) that produce a wrench |
| `rotor_wrenches(graph, actuator, q, qd)` | propeller thrust `k_f ω²` and drag `-k_m ω\|ω\|` on rotor bodies |
| `gravity_efforts`, `coriolis_efforts`, `friction_efforts`, `frame_velocities` | `g(q)`, `c(q, qd, base twist)`, `τ_f(qd)`, link velocities |

`BaseMotion` carries the base twist and acceleration in the base frame. The
linear acceleration is the classical acceleration of the base origin; an
accelerometer gives it as `specific force + g_base`. `gravity` is the gravity
vector in the base frame. For a floating base, pass the measured gravity
direction (projected gravity times 9.81) so `g(q)` follows the base
orientation.

The tests check every term against closed forms or identities:
- the double pendulum's `g` and `c`;
- `RNEA(q, 0, qdd) = M qdd`;
- `qdᵀ c = ½ qdᵀ Ṁ qd`;
- `g = ∇V`;
- `FD(ID(qdd)) = qdd` for fixed and floating bases;
- free fall at `g`;
- standing contact forces summing to the weight;
- hover thrust balancing the weight;
- frame velocities against finite-differenced FK.

## Estimating joint torques without torque sensors

Most recorded datasets have no joint torques. Whenever a source has no
`torque_key`, the pipeline estimates the applied torques by inverse dynamics
from the recorded motion (`model.dynamics.estimate_joint_efforts`):

1. **Motion.** Joint velocities and accelerations come from the data or from
   finite differences of positions (`smoothing_window` smooths first). For a
   floating base, the base twist comes from recorded velocities, or from
   finite differences of the base pose. The base acceleration comes from the
   twist, or from an IMU accelerometer (`imu_acceleration_key`).
2. **Gravity in the base frame** comes from the base orientation, or from
   projected gravity, or defaults to straight down.
3. **RNEA with the base motion.** Any rotor thrust is applied as an external
   force. RNEA gives the joint torques and the base wrench `W` that the
   environment must supply.
4. **Contacts** (legged robots).
   - Each `contact_links` foot is in contact if its sensor flag says so, or by
     heuristic: within `contact_height_tolerance` (3 cm) of the lowest foot
     along gravity, and, when the base linear velocity is known, slower than
     `contact_speed_tolerance` (0.5 m/s).
   - `W` is distributed over the active feet by weighted least squares.
     Tangential forces cost 10× more than normal forces, so ground reactions
     stay mostly vertical, and feet that would pull on the ground are released.
   - Point feet (quadrupeds) carry forces only. Flat feet (`surface_contacts`,
     the default with two or fewer feet) also carry moments.
5. **Joint torques.** The contact forces are applied as external forces, RNEA
   runs again, and friction and rotor drag are added.
6. **Reliability.** What the contacts cannot explain is the residual wrench.
   FD and ID labels of frames whose residual exceeds `max_estimate_residual`
   (default 50%) of the robot's weight are masked out. The estimated torque
   also feeds the dynamics features as `τ_{t-1}`, with flag 0.5.

**How accurate is it?** `scripts/validate_torque_estimation.py` compares the
estimate with the measured joint torques of a real Unitree Go1. The data is the
[MIPO](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry) recordings:
400 Hz joints, measured torques, IMU, and foot sensors. The table gives RMSE
relative to the RMS measured torque (lower is better), with the mean
correlation in parentheses:

| Sequence | Fixed-base ID | Floating, sensor contacts, IMU | Floating, detected contacts, IMU | Floating, sensor contacts, quasi-static |
|---|---|---|---|---|
| flying trot | 103% (−0.16) | 50% (0.77) | **43% (0.80)** | 60% (0.66) |
| stand-trot | 104% (−0.02) | **43% (0.72)** | 50% (0.66) | 45% (0.69) |
| trot | 101% (0.14) | 68% (0.66) | **64% (0.67)** | 71% (0.60) |
| trot, convex MPC | 105% (0.00) | 97% (0.47) | 90% (0.48) | **77% (0.51)** |

Two more checks:

- **Real ANYmal D (GrandTour).** The first 6 episodes of one mission
  (outdoor walking) were converted with `scripts/convert_dataset.py
  grandtour`. Torques were then estimated without `torque_key` and compared
  with the measured joint torques (RMS 27 N m):

  | Setting | Error | Correlation |
  |---|---|---|
  | fixed-base ID | 100% | 0.13 |
  | floating, detected contacts | 72% | 0.60 |
  | floating, contact states from the robot | 47% (knees 32%) | 0.77 |

- **Exact dynamics, estimated contact forces (MuJoCo).** The generated MuJoCo
  model has the same dynamics as ours (mass matrix and bias forces agree to
  1e-7, tested). For a Go1 standing and shifting on the ground, the
  distributed contact forces give 60% error against MuJoCo's torques. Our
  dynamics are exact here, so this isolates the error that comes from
  distributing the contact forces alone.

Ignoring the floating base gives useless torques (about 100% error, no
correlation). With the floating base and contacts, the estimate follows the
measured torques (correlation 0.5–0.8). Typical errors are 43–70% of the RMS
torque, larger on the most dynamic gaits and on rough terrain where "lowest
foot" contact detection is weaker. The estimate is an approximation:
- contact-force distribution is not unique while two diagonal feet are down
  (the MuJoCo check shows this alone accounts for most of the error);
- Unitree's measured torque is itself estimated from motor current;
- the URDF has no rotor inertia or motor friction.

Use contact sensors when a dataset has them. Prefer datasets with measured
torques (GrandTour, MIPO, DFKI Go2) for tasks B and C, and mix in synthetic
and MuJoCo ground data, which have exact torques and contact forces.

### Why the error is large, and what training does about it

There are two separate error sources.

**1. Internal contact forces cannot be observed.** With several feet down
there are more contact-force unknowns than base equations: 12 against 6 for
four point feet. Forces that push the feet against each other produce no
motion, but they change the joint torques. `determinate_torque_projector`
computes these directions:
- the torque effect `B N` of the null space `N` of the contact-to-base-wrench
  map;
- `P = I − U Uᵀ`, which keeps only the torque components that the motion
  determines.

Four feet down leave 6 undetermined directions, two diagonal feet 1, and
flight 0. In MuJoCo, where our dynamics are exact, the error splits cleanly:

| Go1 in MuJoCo, 3 episodes | Full torque error | Error in the motion-determined part |
|---|---|---|
| estimated vs simulated torques | 43.7% | **3.7%** |

Almost all of the simulated error comes from internal forces, which no method
can recover from kinematics.

**2. Model and measurement mismatch on real robots.** On real data the
motion-determined part stays large:

| Data | Full error | Motion-determined part |
|---|---|---|
| MIPO Go1, 4 sequences | 43–97% | 40–92% |
| GrandTour ANYmal D | 47% | 37% |

So on real robots, internal forces are not the main problem; trotting has
only one undetermined direction. A cross-validated per-joint actuator model
was fitted to the measured torques: scale, offset, rotor inertia, viscous and
Coulomb friction. It removes only 1–7 points. Most of the remaining gap
therefore lies in things kinematics cannot fix:
- the "measured" torque itself, which Unitree estimates from motor current;
- unsynchronized IMU and joint clocks;
- acceleration noise from differentiation;
- inertia errors in the URDF.

**What the pipeline does:**
- **Projected ID loss.** For estimated torques of robots in contact, task C is
  supervised only on `P (τ̂ − τ)`. The model is never penalized for internal
  forces it cannot know. The FD labels need no change: with rigid contacts,
  torque components in the internal-force directions produce no
  acceleration.
- **Label confidence.** FD and ID losses of samples whose torques are
  estimated have weight `estimated_torque_weight` (default 0.3, in the data
  config). Measured and simulated torques have weight 1.
- **Residual masking.** As before, frames whose contacts cannot explain the
  base wrench are dropped.
- **Task F only on fixed bases.** The rigid-body residual has no contact or
  base-coupling terms, so floating-base samples are excluded from it.

**What you can do:**
- Train the dynamics heads mainly on measured torques (GrandTour, MIPO, DFKI
  Go2) and on exact simulation (synthetic, MuJoCo ground).
- Use kinematics-only datasets mostly for FK, effect, and motion statistics.
- Prefer datasets with contact sensors and synchronized clocks.
- `scripts/validate_torque_estimation.py` prints both the full and the
  motion-determined error, so you can check a new robot or dataset the same
  way.
