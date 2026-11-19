# Dynamics and torque

[← Back to the README](../README.md)

The rigid-body dynamics with rotor inertia, the contact-constrained torque response that
supervises dynamics without any torque label, why training never uses estimated torques,
and how inaccurate those estimates are.

## Contents

- [Rigid-body dynamics](#rigid-body-dynamics)
- [Rotor inertia (armature)](#rotor-inertia-armature)
- [The contact-constrained response](#the-contact-constrained-response)
  - [Derivation](#derivation)
  - [What it does not depend on](#what-it-does-not-depend-on)
  - [Assumptions and checks](#assumptions-and-checks)
  - [Which feet are down: inferring contacts](#which-feet-are-down-inferring-contacts)
- [Why training uses no estimated torque](#why-training-uses-no-estimated-torque)
  - [What went into the model before](#what-went-into-the-model-before)
  - [What replaces it](#what-replaces-it)
  - [Does a robot need a torque sensor?](#does-a-robot-need-a-torque-sensor)
- [Estimating joint torques without torque sensors (analysis only)](#estimating-joint-torques-without-torque-sensors-analysis-only)
  - [How accurate is it?](#how-accurate-is-it)
  - [Why the estimate is inaccurate](#why-the-estimate-is-inaccurate)

## Rigid-body dynamics

`model/dynamics.py` implements the recursive Newton-Euler algorithm (RNEA),
batched over any leading dimensions, for fixed and floating bases (Featherstone,
*Rigid Body Dynamics Algorithms*, Springer 2008). With generalized velocity
`ν = (v_base, ω_base, qd)` in the base frame:

```text
M_full(q) ν̇ + h(q, ν) = [ W_ext ; τ ] + Σ_c J_cᵀ f_c + rotor thrust and drag
```

For a fixed base this reduces to `τ = M(q) qdd + c(q, qd) + g(q) + τ_f(qd)`.

| Function | Returns |
|---|---|
| `recursive_newton_euler(graph, q, qd, qdd, gravity, base, body_wrenches, base_external_wrench)` | joint efforts (with rotor inertia), the base wrench the environment must supply, and frame velocities |
| `floating_mass_matrix(graph, q)` | `[..., 6+J, 6+J]` `M_full`; its joint block is `graph.mass_matrix(q)` |
| `floating_forward_dynamics(...)` | base linear and angular acceleration and `qdd` |
| `forward_dynamics(...)` / `inverse_dynamics(...)` | fixed-base (or known base motion) versions |
| `contact_maps(...)` | maps `A` (contact wrenches to base wrench) and `B` (contact wrenches to joint torques) |
| `contact_jacobian(graph, q, links, active, surface)` | `[..., n, 6+J]` Jacobian of the contact constraints, `[A; −B]ᵀ` |
| `constrained_response(graph, q, links, active, surface)` | `[..., 6+J, J]` map `G` from a torque change to the acceleration change ([below](#the-contact-constrained-response)) |
| `determinate_torque_projector(...)` | projector `P` onto the torque directions the motion determines, and the number of undetermined ones |
| `estimate_joint_efforts(...)` | torques from motion alone, with contacts and thrust ([analysis only](#estimating-joint-torques-without-torque-sensors-analysis-only)) |
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
- frame velocities against finite-differenced FK;
- MuJoCo's mass matrix (with per-joint armature) and bias forces against ours, to 1e-7.

## Rotor inertia (armature)

A geared motor's rotor spins `N` times faster than the joint, so its inertia
appears at the joint as `I_rotor N²`. On light, highly geared joints this
*reflected* inertia can exceed the link's own inertia (Wensing et al., *Proprioceptive
Actuator Design in the MIT Cheetah*, IEEE T-RO 2017). URDF has no field for it.

`EmbodimentGraph.armature` (`[J]`, kg m² or kg) holds it per joint, in the
joint-space model MuJoCo and Isaac Lab use:

- `graph.mass_matrix(q)` adds `armature_j` to `M_jj`;
- `recursive_newton_euler` adds `armature_j q̈_j` to the joint effort, so
  `floating_mass_matrix`, forward dynamics and the constrained response include it;
- `rigid_body_features` carries it as the last column, so the encoder sees it.

Set it with:

| Where | How |
|---|---|
| Python | `EmbodimentGraph.from_urdf(..., armature=0.01)`, or `graph.with_armature({"*_knee_joint": 0.02, "*": 0.005})` (first matching glob wins, others zero) |
| Data config source | `"armature": 0.01` or `{"glob": value}` |
| MuJoCo ground source | Defaults to `1e-5` per joint (stability), entered in the graph so labels and simulation agree |
| RL robot | The robot's first expert source's `armature` (default 0.01): the RL record's graph and both simulators use the same value |

Synthetic and MuJoCo sources scale each joint's armature by
`randomization["armature"]` (default `[0.5, 1.5]`) per randomized embodiment.

## The contact-constrained response

### Derivation

Write the equations of motion with the actuated joints selected by `S = [0 I]` and
the contacts held still:

```text
M(q) ν̇ + h(q, ν) = Sᵀ τ + Jᵀ λ          (dynamics; λ = contact forces)
J(q) ν̇ + J̇ ν = 0                        (contacts do not move)
```

Change the joint torque by `δτ` at the same state. `h`, `J̇ν`, friction and damping
depend only on the state, so they cancel from the difference:

```text
M δν̇ = Sᵀ δτ + Jᵀ δλ,     J δν̇ = 0
⟹  δν̇ = [ M⁻¹ − M⁻¹ Jᵀ (J M⁻¹ Jᵀ)⁻¹ J M⁻¹ ] Sᵀ δτ = G(q) δτ
```

`constrained_response(graph, q, contact_links, active, surface)` returns `G` as a
`[..., 6+J, J]` tensor: the first six rows are the base linear and angular
acceleration change (base frame), the rest `δq̈`. Point contacts constrain three
translational rows each, surface contacts (flat feet) also three rotational rows.
`contact_jacobian` builds `J` from `contact_maps`: a contact wrench `f` enters as
`Jᵀ f = [A f; −B f]`. Inactive contacts give zero rows, and `J M⁻¹ Jᵀ` gets a
relative regularization of `1e-9` so they and redundant constraints are harmless.

Special cases: a fixed base gives `G = [0; M⁻¹]`; a floating base without contacts
(flight, a robot on a harness whose wrench does not depend on the torque) gives the
free-floating inverse inertia.

### What it does not depend on

`G` depends only on the configuration `q`, the inertia (with rotor inertia) and which
contacts hold. It does **not** depend on:

- the applied torque `τ`, which is unknown for most datasets;
- the contact forces `λ`, which are not unique with several feet down;
- velocities and accelerations, so no finite differences enter;
- friction, damping and gravity, which do not change with `δτ`.

So `G δτ` is an exact label (for the model) at **any** recorded state, including
motion-capture and retargeted data. Torque changes that only press the feet against
each other (internal forces) produce no acceleration change: `G (I − P) = 0` with the
projector `P` of `determinate_torque_projector`. Task R⁻¹ is therefore supervised
on `P δτ` only, exactly like task C.

### Assumptions and checks

- **Contacts hold.** The response is the local, bilateral one: it assumes the
  feet neither slip nor lift under the change. The label of a frame is therefore
  dropped within `contact_margin_steps` (default 2) of a contact change, and where a
  foot in contact moves faster than `max_contact_slip` (default 0.1 m/s; needs the base
  linear velocity). The speed is that of the contact link's frame origin, the point
  the constraints hold still (`contact_point_speeds`). With inferred contacts, every
  foot near the ground is checked, since the inference may have left a sliding foot
  out. Measured against MuJoCo (RL environment, quadruped under random actions, stiff
  contacts, 0.5 N m random torque changes), the error of `G δτ` grows with the speed
  of the fastest foot in contact:

  | Fastest stance-foot speed | Frames | Median error | 90th percentile |
  |---|---|---|---|
  | < 0.02 m/s | 10 | 0.8% | 15% |
  | 0.02–0.05 m/s | 26 | 1.9% | 37% |
  | 0.05–0.1 m/s | 46 | 1.9% | 37% |
  | 0.1–0.2 m/s | 38 | 8.9% | 31% |
  | 0.2–0.5 m/s | 110 | 20% | 54% |
  | > 0.5 m/s | 192 | 23% | 69% |

  In the offline ground simulation (`configs/data/ground_contact.json`) the random
  joint targets push the feet along the floor, and this check drops 8% (H1), 44%
  (Go1), 46% (G1) and 68% (ANYmal D) of the frames with contact. Those frames still
  train every other task. A smaller `target_amplitude` gives more sticking frames.
- **Which feet are down must be known.** Contacts from a sensor or a simulator
  are trusted. Without them they are inferred
  ([below](#which-feet-are-down-inferring-contacts)), and only frames where the
  inference is confident and explains the base wrench the motion needs within
  `max_contact_residual` (default 0.5) of the weight keep response labels.
- **The support must be known.** A floating robot with no contact links, no rotors
  and no recorded support wrench (for example arms recorded on a humanoid whose legs
  are not in the data) gets no response labels. Declare such data as
  `"floating_base": false` if the torso is held still, or give `contact_links`.
- **Rigid contacts.** MuJoCo's default contacts are soft (`solimp` 0.9–0.95), which
  lets the feet sink and accelerate. The MJCF built by `build_mjcf` (offline ground
  simulation and RL environments) therefore uses stiff contacts, `solimp` 0.99–0.999
  (`STIFF_SOLIMP`, `contact_solimp=None` restores MuJoCo's default). Against a
  Go1-like quadruped standing on four feet in MuJoCo, with small torque changes:

  | MuJoCo contact model | Median error of `G δτ` against MuJoCo's `δq̈` |
  |---|---|
  | default soft contacts | ≈ 20% |
  | stiff contacts (`solimp` 0.99–0.999, **the default here**) | 3% |
  | stiff contacts (`solimp` 0.9999) | 2.7% (tested: `test_constrained_response_matches_mujoco_with_stiff_contacts`) |
  | ignoring the contacts (free floating) | > 100% |

  The rigid model is therefore exact up to contact compliance. Real feet with rubber
  pads are also slightly compliant; this is a model error of the same kind as an
  inaccurate URDF inertia, and it is deterministic, unlike torque-estimation noise.

### Which feet are down: inferring contacts

A wrong contact set gives a badly wrong response label, so data without foot sensors
needs a reliable inference and a way to know when it is unsure
([`model/training/contacts.py`](../model/training/contacts.py)). Two methods
(`contact_detection` in the data config or per source):

- **`height`**: a foot is down when it is within `contact_height_tolerance` (3 cm) of
  the lowest foot along gravity and, with a known base velocity, slower than
  `contact_speed_tolerance` (0.5 m/s). It assumes flat ground and cannot tell a
  low swing foot from a stance foot.
- **`dynamics`** (default, `select_contacts`): a maximum a posteriori choice among all
  `2^C` subsets of the feet (up to 8 feet; more fall back to `height`). The cost of a
  subset adds:
  1. a **kinematic prior** per foot, `p = exp(−((h/3 cm)² + (v/0.2 m/s)²)/2)`, from its
     height above the lowest foot and its speed (when the base velocity is known);
  2. a **dynamics likelihood**: the base wrench the motion needs (floating-base RNEA)
     must come from pushing-only forces inside friction pyramids (μ = 0.8) at the
     subset's feet (flat feet: at four sole corners), solved as nonnegative least
     squares; the cost is `(residual / 0.15 m g)² / 2`;
  3. a **minimum load**: every chosen foot must carry at least 5% of the weight. A
     foot that carries nothing is, for dynamics, a foot in the air; without this term
     every foot fits a flight phase;
  4. without a base velocity, the **velocity consistency** of the subset: feet that
     stand still in the world all move alike relative to the base (the cue of legged
     odometry), so the spread of the subset's foot velocities is penalized.

  The **confidence** of a frame is how much more the second-best subset costs. Only
  frames above `min_contact_confidence` (default 3) keep response labels. A foot on a
  higher step is still chosen when the weight needs it, so the method does not assume
  flat ground.

`scripts/evaluate_contact_detection.py` hides the contacts of episodes that have them,
infers them, and compares. "Usable" is the share of frames that keep response labels
(margin, slip, residual and, for `dynamics`, confidence), and "usable set" the share of
those whose whole contact set is right; a wrong set there is a wrong label.

Real Unitree Go1, [MIPO](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry)
(Yang et al., IROS 2023), 50 Hz, reference from the foot-force sensors (each sensor's
offset plus 10% of its range; the offsets drift, one stand-trot sensor reads 24–39 in
the air). MIPO has no base velocity, so neither the speed evidence nor the slip check
applies:

| Sequence | `height`: usable | usable set | `dynamics`: usable | usable set |
|---|---|---|---|---|
| stand-trot 04-06 | 43% | 80.2% | 52% | **93.2%** |
| trot 05-036 | 38% | 49.0% | 46% | **92.9%** |
| trot 10-036, convex MPC | 59% | 65.5% | 55% | **84.7%** |
| trot 05-036, kicked | 42% | 52.7% | 51% | **82.0%** |
| flying trot 08-040 | 61% | 47.1% | 32% | **93.0%** |
| flying trot 04-032 | 50% | 61.6% | 45% | **96.8%** |

MuJoCo ground simulation (`configs/data/ground_contact.json`, one robot per source,
three episodes each), reference from MuJoCo:

| Robot | `height`: usable | usable set | `dynamics`: usable | usable set |
|---|---|---|---|---|
| Go1 | 39% | 84.9% | 29% | **96.4%** |
| ANYmal D | 18% | 97.6% | 14% | **97.9%** |
| G1 | 54% | 100% | 35% | 100% |
| H1 | 91% | 99.9% | 80% | 100% |

On real gaits the height rule labels many swing feet as down (precision 0.39–0.83),
and about half of its usable frames carry a wrong set. The dynamics method keeps a
similar share of frames with 82–97% right sets; its low-confidence frames hold most of
the remaining errors (on flying trot 04-032 the share of right sets is 70% over all
stable frames and 98.9% above confidence 3). For dynamic gaits without contact data,
`"detected_contact_response": false` on a source turns their response labels off
entirely; FK, D and E still train on them.

```bash
uv run python scripts/evaluate_contact_detection.py --data_config configs/data/ground_contact.json
uv run --extra validation python scripts/evaluate_contact_detection.py \
    --mipo indoor/20230625_gait/230625-risqh-flytrot-04-032-33square.h5
```

The unit tests also check `J G = 0`, `G (I − P) = 0`, `J` against finite-differenced
foot velocities, `G = M⁻¹` for a fixed base, and `G δτ` against the difference of two
floating-base forward-dynamics solves.

## Why training uses no estimated torque

Most recorded datasets have no joint torques. Earlier versions estimated them by
inverse dynamics ([below](#estimating-joint-torques-without-torque-sensors-analysis-only))
and used the estimate. On real robots the estimate is 43–97% off (Go1) and 47–72%
(ANYmal D), and most of that error is not the kind that averages out: it comes from
clock skew, differentiation, unmodelled actuators and the "measured" torque itself.
For the model it was mostly noise, not a feature.

### What went into the model before

| Path | Effect |
|---|---|
| `applied_torque_prev` dynamics feature (flag 0.5) | Encoder input with 43–97% noise, for every frame of every dataset without torque sensors |
| Previous-action token features of data without commands | The token's "requested torque" column was the same estimate |
| FD / ID labels (weight 0.3, projected) | Labels with the same error; on real data the motion-determined part was still 37–92% off |
| Shortcut | Task C's target `τ̂_t` and the input `τ̂_{t−1}` came from one estimator; at 50 Hz they are nearly equal, so the head could copy the input, errors included |
| Train/deploy mismatch | Isaac Lab's `applied_torque` (computed PD effort), Unitree's current-based estimate and the inverse-dynamics estimate are three different quantities under one feature |

### What replaces it

| Before | Now |
|---|---|
| 7 dynamics features with `τ_{t−1}` and a source flag | 5 features: `M_ii`, `c`, `g`, friction, gravity load, all from state and model |
| Estimated torque as previous-action feature | Zero when nothing is known; the commanded target and its PD request when commands exist |
| FD/ID on estimated torques (weight 0.3) | FD/ID only on exact simulated torques (weight 1) and, if enabled, sensors (`measured_torque_weight`, default 0) |
| Task F (rigid-body residual) | Removed: it needed the applied torque and duplicated B on simulated data |
| (none) | Tasks R and R⁻¹: the constrained response `G(q) δτ` on every frame of every dataset |
| Motor head saw estimated torques of recorded data | Motor head sees recorded commands and random proposals with exact labels |
| `"use_estimated_previous_torque"`, `"estimated_torque_weight"`, `"allow_estimated_torque_for_fd"`, `"max_estimate_residual"` | Rejected with an explanation; `"max_contact_residual"`, `"measured_torque_weight"`, `"contact_margin_steps"`, `"proposal_variants"`, `"response_range"` |

The encoder remains dynamics-aware through exact inputs, the response tasks on all
data, exact-torque tasks on simulated data, and physics losses on the policy's own
rollouts during RL ([details](training.md#physics-pretraining)).

### Does a robot need a torque sensor?

No. At run time the policy reads joint positions and velocities, the IMU, the robot
model, and its own previous target with the effort the known PD law requests for it
(`K_p (a − q) − K_d qd`, computed, not measured). Simulation and robot compute every
input the same way. The PD gains on the robot must equal the training gains. See
[What a robot needs at run time](../README.md#what-a-robot-needs-at-run-time).

## Estimating joint torques without torque sensors (analysis only)

`estimate_joint_efforts` (and `model.training.episodes.estimate_episode_efforts` for a
whole episode) estimates the applied torques by inverse dynamics from the recorded
motion. Training never uses it; it remains to measure how wrong an estimate would be on
a new robot or dataset (`scripts/validate_torque_estimation.py`), to check contact sets
(its residual), and for the tests.

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
6. **Residual.** What the contacts cannot explain is the residual wrench. The
   pipeline uses it only to reject detected contact sets (`max_contact_residual`).

### How accurate is it?

`scripts/validate_torque_estimation.py` compares the estimate with the measured joint
torques of a real Unitree Go1. The data is the
[MIPO](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry) recordings
(Yang et al., IROS 2023): 400 Hz joints, measured torques, IMU, and foot sensors. The
table gives RMSE relative to the RMS measured torque (lower is better), with the mean
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
measured torques (correlation 0.5–0.8), but with typical errors of 43–70% of the RMS
torque, larger on the most dynamic gaits and on rough terrain where "lowest foot"
contact detection is weaker.

### Why the estimate is inaccurate

The estimate is `τ = M(q) q̈ + c + g − Jᵀλ`, and every term carries error:

| Source | Why | Evidence |
|---|---|---|
| Contact forces are not unique (statically indeterminate) | Four feet down give 12 force unknowns against 6 base equations. Forces that press the feet together change joint torques but not the motion, and no method recovers them from kinematics | In MuJoCo, where our dynamics are exact, 43.7% total error but only **3.7%** in the motion-determined part |
| Wrong contact set | The lowest-foot heuristic fails on rough terrain and around touchdown and lift-off | ANYmal D: 72% with detected contacts, 47% with the robot's contact states |
| Twice-differentiated accelerations | Noise grows like `1/dt²`; smoothing adds lag and misses impacts shorter than a sample | Real data: the motion-determined part is still 37–92% off |
| Clocks and base estimate | IMU and joint clocks are not synchronized; base orientation and acceleration carry estimator error | Same |
| URDF mismatch | CAD inertias miss batteries and cables; no rotor inertia, gearbox friction, backlash or elasticity | A cross-validated per-joint actuator model (scale, offset, rotor inertia, viscous and Coulomb friction) removes only 1–7 points (compare Hwangbo et al., Science Robotics 2019, who learn actuator dynamics because they resist modelling) |
| The "measured" torque | Unitree computes it from motor current times a torque constant, which is nonlinear and temperature dependent | — |

The error split shows it:

| Data | Full error | Motion-determined part |
|---|---|---|
| Go1 in MuJoCo, 3 episodes | 43.7% | **3.7%** |
| MIPO Go1, 4 sequences | 43–97% | 40–92% |
| GrandTour ANYmal D | 47% | 37% |

In simulation the error is almost all internal forces, which the projection removes.
On real robots it is mostly measurement and model error, which no projection or
actuator fit removes. Hence the design: never estimate, and supervise dynamics with
quantities that need no torque.
