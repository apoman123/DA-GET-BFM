# The DA-GET encoder

[← Back to the README](../README.md)

How a URDF becomes one token per DoF, how the graph and the physics enter attention, and every encoder option.

## Contents

- [Why one token per DoF](#why-one-token-per-dof)
- [How it works](#how-it-works)
  - [1. URDF to embodiment graph](#1-urdf-to-embodiment-graph)
  - [2. Fixed and floating bases](#2-fixed-and-floating-bases)
  - [3. Per-joint hardware features](#3-per-joint-hardware-features)
  - [4. Token assembly](#4-token-assembly)
  - [5. Encoder and attention bias](#5-encoder-and-attention-bias)
  - [6. Physics Cross Attention](#6-physics-cross-attention)
  - [7. What is recomputed when](#7-what-is-recomputed-when)
  - [8. Batching different robots](#8-batching-different-robots)
- [Physics pretraining auxiliary losses](#physics-pretraining-auxiliary-losses)
- [Configuration reference](#configuration-reference)

## Why one token per DoF

A policy that controls joints should see the robot the way its actions are
defined: one entry per actuated joint. GET-Zero represents a robot as "an
embodiment graph composed of joints which is readily accessible from URDF files
as a practical representation when joint-level actions matter." Each joint is
a node and a token, and the kinematic tree decides which tokens attend to each
other more strongly.

Because the token count follows the robot, the same weights handle a 12-DoF
quadruped, a 29-DoF humanoid, a 16-DoF hand, and a 4-rotor drone. Shorter
robots are padded and masked inside a batch. A per-joint output head can then
emit one action per token. The floating base is not a token: its state (gravity
direction, twist) enters every token as a global observation, and its dynamics
enter the physics features and the pretraining labels.

## How it works

```text
URDF ──parse_urdf──▶ UrdfRobot ──EmbodimentGraph.from_urdf──▶ EmbodimentGraph
                                                               │  joints, parents, edges, base body,
                                                               │  link frames (feet, rotors), inertia, limits
           actuator_specs / rigid_body_features ◀──────────────┤
                     several graphs ──collate_embodiments──▶ EmbodimentBatch
                                                               │
 joint + base observations ──DaGetTokenizer──▶ observation tokens [B, J, F]
 rigid body │ actuator │ dynamics ──physics tokenizers──▶ physics tokens [B, 3J, hidden]
                                                               │
 tokens, M(q), graph, physics ──DaGetEncoder──▶ joint embeddings H [B, J, hidden]
                                                               │
         auxiliary heads A, B, C, R, R⁻¹, D, E (pretraining only) ┘
```

### 1. URDF to embodiment graph

`model/urdf.py` parses URDF from a path or an XML string and checks that the
model is a single tree: one root link, no link with two parents, no cycles, no
missing links, and no duplicate names. It also reads `<transmission>` gear
ratios (`mechanicalReduction`). `EmbodimentGraph.from_urdf` in
`model/embodiment.py` then builds the joint graph:

| URDF element | Treatment |
|---|---|
| `revolute`, `continuous`, `prismatic` joints | One node and one token each. Rotors are `continuous` joints. |
| `fixed` joints | Collapsed. Their child links are merged into the parent body: poses are composed and inertia is combined with the parallel-axis theorem. Feet, IMU frames, sensor mounts, and flanges therefore add no tokens, but their frames stay available (`link_poses`) for contacts and end effectors. |
| `mimic` joints | Not independent DoFs, so frozen at their offset by default. `include_mimic_joints=True` keeps them as tokens. |
| `floating` and `planar` joints | The base. By default (`multi_dof_joints="skip"`) the links below them form the root body, and a skipped `floating` joint marks the graph as floating-base. With `"expand"`, they become chains of 1-DoF passive virtual joints (6 for floating, 3 for planar). |
| `frozen_joints={name: q}` | 1-DoF joints held rigid at a position and merged like fixed links. Use this when a dataset records only part of a robot, for example the arms of a humanoid whose legs are locked. |
| Joint order | Depth-first URDF order, the order used by MuJoCo and Pinocchio. Pass `joint_order=[...]` to match a simulator's or dataset's ordering. |

Edges:

- **Directed parent to child edges** follow the kinematic tree through any
  number of fixed joints.
- **Undirected sibling edges** join joints that share a parent body. This is
  GET-Zero's rule for "joints at the start of independent serial chains." For a
  hand these are the finger roots on the palm. For a quadruped they are the four
  hips on the trunk. For a humanoid they are the hips and waist on the pelvis,
  and the shoulders on the torso. For a multirotor they are the rotors. Turn
  them off with `sibling_edges=False`.

From these edges the graph precomputes three integer distance matrices: the
shortest path over undirected edges (SPD), the number of steps to each
ancestor, and the number of steps to each descendant.

### 2. Fixed and floating bases

`floating_base=True` (or auto-detection from a skipped URDF `floating` joint)
makes the root body free. Most legged URDFs (Unitree, ANYmal) start at the
trunk with no floating joint, so pass `floating_base=True` (or
`"floating_base": true` in a data config) for them. The graph also stores the
root body's mass, center of mass, and inertia (`base_mass`, `base_com`,
`base_inertia`), which a fixed-base model never needs.

What changes for a floating base:

| | Fixed base | Floating base |
|---|---|---|
| Frames | root link | base frame (moves with the robot); gravity expressed in it |
| Dynamics | $\tau = M\ddot q + c + g$ | joint rows plus 6 base rows; base twist and acceleration enter RNEA; external wrenches (contacts, thrust) close the base rows |
| Torque-free dynamics label (task R) | $G=M^{-1}$ | $G$ from $M_{\mathrm{full}}^{-1}$ with the contacts that hold ([derivation](dynamics.md#the-contact-constrained-response)) |
| Observation globals | gravity direction (constant) | gravity direction, base angular velocity, base linear velocity / robot size, flags |
| FD label (task B) | $\ddot q$ | $\ddot q$ plus base linear and angular acceleration |
| Effect label (task E) | in the root frame | in the base frame at time `t`, including base motion |

A humanoid that only moves its arms while standing still can be modelled as
fixed-base, by freezing or not recording the legs. A floating-base robot without
contact links, rotors or a recorded support wrench is held by something unmodelled;
it gets no response labels.

### 3. Per-joint hardware features

`JointFeatureConfig` chooses the fixed features `o_fl,j` inside each
observation token. All poses are relative to the parent joint.

| Preset | Size | Features |
|---|---|---|
| `JointFeatureConfig()` | 6 | Joint origin `x, y, z` and rotation `roll, pitch, yaw`. This matches GET-Zero. |
| `JointFeatureConfig.cross_embodiment()` | 31 | Origin, rotation as continuous 6D, joint axis, one-hot joint type, position limits, effort and velocity limits, damping and friction, and the mass, center of mass, and inertia tensor of the body the joint moves. |

With Physics Cross Attention enabled, the full physical description goes to the
physics tokens instead (section 6), so the GET-Zero preset is usually enough in
the observation tokens.

### 4. Token assembly

`DaGetTokenizer` builds token $j$ exactly as GET-Zero does:

$$
T_j = \bigl[o_{vg}^{t-H+1:t},\;o_{fg},\;o_{vl,j}^{t-H+1:t},\;o_{fl,j}\bigr].
$$

| Part | Shape in | Meaning | Training pipeline uses |
|---|---|---|---|
| `o_vg` | `[B, H, F_vg]` | Variable global observations with history, copied into every token | 11 per step: gravity direction in the base frame (3), base angular velocity (3), base linear velocity / robot size (3), base-twist-known flag, floating-base flag |
| `o_fg` | `[B, F_fg]` | Fixed global features | none (for example a task ID) |
| `o_vl,j` | `[B, H, J, F_vl]` | Variable per-joint observations with history | 5 per step: normalized position, velocity / speed limit, previous action (torque / peak, position target, velocity target) |
| `o_fl,j` | `[B, J, F_fl]` | Fixed per-joint hardware features | `EmbodimentGraph.joint_features` |

History is flattened feature-major, as in the GET-Zero code: every timestep of
the first feature, then every timestep of the second, and so on. Pass history
oldest first. Padded joints get all-zero tokens.

### 5. Encoder and attention bias

`DaGetEncoder` (`model/modeling_da_get.py`) runs:

1. A LayerNorm over each raw token, as in GET-Zero, because token features have
   very different scales.
2. A linear projection to `hidden_size`.
3. `num_hidden_layers` pre-norm transformer layers. Each layer runs, in order:
   - graph self-attention;
   - Physics Cross Attention (optional);
   - cross-attention to external context tokens (optional);
   - a GELU feed-forward block.
4. A final LayerNorm. Padded joints are set to zero.

Each self-attention layer adds a learned per-head bias to the attention logits:

$$
\widetilde M_{ij}(q)
= \frac{M_{ij}(q)}{\sqrt{M_{ii}(q)M_{jj}(q)}+\varepsilon_M},
$$

$$
A_{ij}
= \frac{\mathbf q_i^\top\mathbf k_j}{\sqrt d}
+ b_{\mathrm{spatial}}(\operatorname{SPD}(i,j))
+ b_{\mathrm{parent}}(P_{ij})
+ b_{\mathrm{child}}(C_{ij})
+ b_{\mathrm{mass}}(\widetilde M_{ij}(q)).
$$

The graph terms come from GET-Zero; this repository adds the mass term.

- `spatial`, `parent`, and `child` are embedding tables with one scalar per
  head, in every layer.
- `P` and `C` count ancestor and descendant steps and are zero for unrelated
  pairs.
- Unreachable pairs and distances above `max_graph_distance` share one overflow
  bucket. That bucket does not depend on how much a batch is padded, so a robot
  encodes the same way whichever robots share its batch.
- The mass term is the normalized off-diagonal coupling of `M(q)`, a value in
  [-1, 1]. Unlike the graph terms, it changes with the configuration at every
  step. For a floating base, `M(q)` is the joint block of the full matrix.

Each bias can be turned off: `use_spatial_encoding`,
`use_parent_child_encoding`, `use_mass_encoding`.

### 6. Physics Cross Attention

Three feature groups describe every joint `i`. All values are SI, and every
spatial quantity names its frame (`model/physics_schema.py` has the full
schema):

- $B_i$ is the body frame of joint $i$. Its origin is the joint origin, its axes
  are the joint frame's, and it moves with the body that joint `i` drives (the
  child link after fixed links are merged).
- $B_{p(i)}$ is the parent joint's body frame, or the base frame.
- For prismatic joints read N for N m, m for rad, and m/s for rad/s.

**Rigid body (26), from URDF, static per robot**

| Symbol | Size | Unit | Frame |
|---|---|---|---|
| $m_i$ | 1 | kg | — |
| $\operatorname{vech}(I_i)$: $I_{xx},I_{xy},I_{xz},I_{yy},I_{yz},I_{zz}$ about the CoM | 6 | kg m² | $B_i$ |
| $r_i^{\mathrm{CoM}}$ | 3 | m | $B_i$ |
| $a_i^{\mathrm{axis}}$ (unit) | 3 | 1 | $B_i$ |
| $T_i^{\mathrm{parent}}$: translation and first two rotation columns at $q_i=0$ | 3 + 6 | m, 1 | $B_{p(i)}$ |
| $\mathrm{jointType}_i$ (one-hot revolute, continuous, prismatic) | 3 | 1 | — |
| $I_i^{\mathrm{rotor}}$ reflected rotor inertia (`armature`; 0 when not set) | 1 | kg m² | joint |

**Actuator (6), from `ActuatorSpec`, static per robot**

| Symbol | Unit | Default when the URDF lacks it |
|---|---|---|
| $\tau_i^{\mathrm{peak}}$ | N m | URDF `effort` |
| $\tau_i^{\mathrm{cont}}$ | N m | $0.5\tau_i^{\mathrm{peak}}$ |
| $K_{p,i},K_{d,i}$ | N m/rad, N m s/rad | 0, or `actuator_defaults` |
| $k_{f,i}$ propeller thrust coefficient (thrust $k_{f,i}\omega_i^2$ along $a_i$, signed) | N s²/rad² | 0 (rotors only) |
| $k_{m,i}$ propeller drag coefficient (drag moment $k_{m,i}\omega_i^2$) | N m s²/rad² | 0 (rotors only) |

`k_f` and `k_m` extend the actuator features so multirotors can be described.
Every other joint has zeros there. Speed, power, gear ratio, and control mode
remain in `ActuatorSpec` for simulation and labels, but are not encoder features.
The defaults are assumptions.
Pass real motor data through
`actuator_specs(graph, overrides={"*_knee_joint": {"continuous_torque": 23.7}})`
or, in a data config, through `actuator_overrides` (exact names or glob
patterns).

**Dynamics (5), from the state at time `t`; no torque**

| Symbol | Unit | Depends on |
|---|---|---|
| $M_{ii}(q_t)$, including $I_i^{\mathrm{rotor}}$ | kg m² | $q$ |
| $c_i(q_t,\dot q_t)$ Coriolis and centrifugal, including base rotation for a floating base | N m | $q$, $\dot q$, base twist |
| $g_i(q_t)$ gravity, with gravity in the base frame | N m | $q$, gravity direction |
| $\hat\tau_{f,i}$ friction $d_i\dot q_i+\mu_i\tanh(\dot q_i/0.01)$, plus propeller drag $k_{m,i}\omega_i\lvert\omega_i\rvert$ | N m | $\dot q$ |
| $\lvert g_i\rvert/(\tau_i^{\mathrm{cont}}+\varepsilon)$ gravity load | 1 | $q$ |

Earlier versions had two more columns, the previous applied torque and its source
flag (1 measured, 0.5 estimated, 0 unknown). On datasets without torque sensors that
value was an inverse-dynamics estimate 43–97% off, and it gave inverse dynamics a
shortcut; both columns were removed
([why](dynamics.md#why-training-uses-no-estimated-torque)). Encoder configs of the
old sizes (25, 6, 7) are rejected.

Each group goes through its own tokenizer (`PhysicsFeatureTokenizer`). The
tokenizer first maps every feature to `asinh(x / s)` with a per-feature scale
`s`, which is linear for small values and logarithmic for large ones, so a
20 g finger link and a 10 kg thigh are both well conditioned. A two-layer MLP
then gives one token per joint, and a learned group embedding is added. The
three groups are concatenated into a memory of `3J` tokens:

$$
\mathcal M = [\,\mathrm{rigid\ body}_{1:J},\;
                 \mathrm{actuator}_{1:J},\;\mathrm{dynamics}_{1:J}\,],
$$

$$
\alpha_{i,g,j} = \operatorname{softmax}_{g,j}\!\left(
  \frac{\mathbf q_i^\top\mathbf k_{g,j}}{\sqrt d}
  + b_{\mathrm{spatial}}^{\mathrm{phys}}(\operatorname{SPD}(i,j))
  + b_{\mathrm{group}}(g)\right).
$$

Queries are the observation tokens. The bias lets a joint focus on its own
physics (`SPD = 0`) and on nearby joints. The bias tables are per layer and per
head, and padded joints are masked out. Physics Cross Attention is controlled by
`use_physics_cross_attention` and `physics_cross_attention_layers` (default:
every layer).

### 7. What is recomputed when

| Tier | Features | Changes when | At training time | At control time |
|---|---|---|---|---|
| embodiment | all rigid-body and actuator features | the robot or its randomized parameters change | once per robot, when the dataset is built | once per robot; their per-layer keys and values are cached in a `PhysicsMemory` |
| configuration | $M_{ii}(q)$, $g(q)$, gravity load | $q$ or the gravity direction | once per frame, cached on disk | `PhysicsFeatureCache`: every `configuration_interval` steps, or sooner if any joint moves more than `configuration_tolerance` or gravity turns |
| velocity | $c(q,\dot q,\text{base twist})$, $\hat\tau_f$ | $\dot q$, the base twist (and $q$) | once per frame, cached on disk | every `velocity_interval` steps, if $\dot q$ or the base twist moves more than `velocity_tolerance`, or with the configuration tier |

`encoder.build_physics_memory(rigid_body, actuator, joint_mask)` precomputes the
rigid-body and actuator tokens and their keys and values for every physics
layer. After that, a control step only embeds and projects the `J` dynamics
tokens. The memory records the versions of the parameters it came from. After
an optimizer step, or any other in-place weight change, the encoder refuses a
stale memory instead of silently using it.

### 8. Batching different robots

`collate_embodiments(graphs, max_joints=None)` pads every graph to the largest
DoF count and returns an `EmbodimentBatch` holding:

- `parent_indices`, padded with `-1`;
- `joint_mask`;
- `adjacency_matrix`;
- precomputed `graph_distances`;
- `joint_features`.

`batch.encoder_inputs()` returns the keyword arguments the encoder needs.
`batch.to(device)` moves everything. `batch.mass_matrix(q)`,
`batch.forward_kinematics(q)`, and `batch.normalize_joint_positions(q)` apply the
per-robot helpers to a padded batch. Fixed- and floating-base robots can share
a batch.

## Physics pretraining auxiliary losses

`DaGetForPhysicsPretraining` adds seven heads to the encoder during pretraining.
They receive its per-joint embedding $H_i$; the action-conditioned heads also
receive a proposed or applied action. The heads produce training targets for
the shared encoder, not constraints on the BFM actor's output. The current
losses are implemented in [`model/pretraining.py`](../model/pretraining.py),
with labels assembled by [`model/training/episodes.py`](../model/training/episodes.py)
and [`model/training/dataset.py`](../model/training/dataset.py).

| Task and logged name | Head input | Prediction and label | Per-joint loss |
|---|---|---|---|
| A. Forward kinematics, `loss_fk` | $H_i$ | Joint-frame position in the root frame, divided by robot length $L_e$; optionally its orientation | $\lambda_p\lVert\hat p_i-p_i\rVert^2+\lambda_R d_{SO(3)}(\hat R_i,R_i)^2$. Omit orientation if `fk_orientation=False`. |
| B. Forward dynamics, `loss_fd` | $H_i$ and applied effort $\tau_i/\tau_i^{\mathrm{peak}}$, with graph mixing | Joint acceleration $\ddot q_i/\sigma_i^a$; for a floating base, six normalized base acceleration components | Squared error in normalized joint acceleration, plus $\lambda_{\mathrm{base}}$ times mean squared base error when known. |
| C. Inverse dynamics, `loss_id` | $H_i$, desired normalized joint acceleration, and desired base acceleration, with graph mixing | Applied effort that produced it, divided by $\tau_i^{\mathrm{cont}}$ | Squared normalized effort error, after projecting the physical error onto the motion-determined torque directions when feet are down. Simulated (and, if enabled, measured) torques only. |
| R. Constrained response, `loss_response` | $H_i$ and a torque change $\delta\tau_i/\tau_i^{\mathrm{peak}}$ drawn per sample, with graph mixing | $\delta\ddot q_i/s_i^R$ and the base change, $G(q)\delta\tau$ with the contacts holding | Squared normalized error, plus $\lambda_{\mathrm{base}}$ times the base error for floating robots. Every dataset; no torque needed. |
| R⁻¹. Inverse response, `loss_inverse_response` | $H_i$, $\delta\ddot q_i/s_i^R$ and the base change, with graph mixing | $\delta\tau_i/\tau_i^{\mathrm{peak}}$ | Squared error after the same projection as C. |
| D. Motor feasibility, `loss_motor` | $H_i$ and an action proposal | Two feasibility flags and five capability margins | $\lambda_f\operatorname{BCEWithLogits}+\lambda_m\operatorname{Huber}$; mask missing margins. |
| E. Action effect, `loss_effect` | $H_i$ and action features, with graph mixing | Next-frame end-effector displacement, orientation, linear velocity, and angular velocity in the current base frame | Weighted squared errors for displacement and velocities, plus squared $SO(3)$ geodesic rotation error. Only selected end-effectors have labels. |

The motor head's flags are *peak feasible* and *continuous feasible*.
For a position target $a_i$, its requested effort is

$$
\tau_i^{\mathrm{request}}
=K_{p,i}(a_i-q_i)-K_{d,i}\dot q_i.
$$

Torque and velocity commands use their
respective control laws. The five margins describe peak torque, continuous
torque, power, speed, and joint-position limits. A margin of zero is the
limit, positive is inside it, and negative is beyond it. Peak feasibility
requires the peak-effort, power, speed, and position margins to be
nonnegative; continuous feasibility also requires the sustainable-effort
margin. Speed, power, and mode are retained in `ActuatorSpec` to generate
these labels even though they are no longer encoder actuator features.

For every per-joint task, `masked_joint_mean` averages valid joints within
each sample first, then averages samples. FD and ID supervise actuated joints
only, and only where the torque is known: simulated torques have weight 1,
torque-sensor torques `measured_torque_weight` (default 0). No torque is
estimated. The former task F (rigid-body residual) was removed with the
estimates. The total loss is

$$
\mathcal L_{\mathrm{total}}
=\sum_{k\in\{\mathrm{fk,fd,id,response,inverse\_response,motor,effect}\}}w_k\mathcal L_k,
$$

with every task weight 1 by default. See [physics pretraining](training.md#physics-pretraining)
for the label sources, training flags and dataset setup.

There is **no separate loss that regresses $K_p$ or $K_d$**. Task D uses them
to generate feasibility labels, so it can teach the embedding how gains affect
a proposed target. This is indirect supervision; it does not prove that the
BFM actor uses the gains. None of these losses alone guarantees that the
actor's target angles are actuator-feasible at runtime; the action map does
that ([actuator torque limits](bfm.md#actuator-torque-limits)).
These encoder pretraining losses are separate from the BFM's
[`aux_critic`](bfm.md), which learns the environment's auxiliary rewards.

## Configuration reference

Select an encoder architecture with `DaGetConfig.from_size(token_size, "tiny")`.
The same names work with `--preset` in `scripts/train_physics_da_get.py`.
Keyword arguments override individual preset dimensions. `da_get_zero` remains
available as the paper architecture with width 512, eight layers, and a 1024-wide
feed-forward block.

| Size | Width | Layers | Heads | Feed-forward width |
|---|---:|---:|---:|---:|
| `tiny` | 64 | 2 | 2 | 256 |
| `small` | 128 | 4 | 4 | 512 |
| `medium` | 256 | 6 | 8 | 1024 |
| `large` | 512 | 8 | 16 | 2048 |

`DaGetConfig` fields:

| Field | Default | `DaGetConfig.da_get_zero` | Meaning |
|---|---|---|---|
| `input_size` | required | required | Raw token width, `DaGetTokenizer.token_size` |
| `hidden_size` | 128 | 512 | Model width |
| `num_hidden_layers` | 4 | 8 | Transformer layers |
| `num_attention_heads` | 4 | 16 | Attention heads |
| `intermediate_size` | 512 | 1024 | Feed-forward width |
| `hidden_dropout_prob` | 0.0 | 0.1 | Dropout |
| `max_graph_distance` | 32 | 32 | Largest separately embedded distance |
| `context_size` | `None` | `None` | Context token width (defaults to `hidden_size`) |
| `cross_attention_layers` | `()` | `()` | Layers with context cross-attention |
| `mass_epsilon` | `1e-8` | `1e-8` | Stabilizer in the mass normalization |
| `use_spatial_encoding` | `True` | `True` | SPD bias |
| `use_parent_child_encoding` | `True` | `True` | Ancestor and descendant bias |
| `use_mass_encoding` | `True` | `True` | `M(q)` bias |
| `use_input_layer_norm` | `True` | `True` | LayerNorm on raw tokens |
| `use_physics_cross_attention` | `False` | `False` | Physics Cross Attention |
| `physics_cross_attention_layers` | `None` (all) | `None` | Layers with Physics Cross Attention |
| `rigid_body_feature_size`, `actuator_feature_size`, `dynamics_feature_size` | 26, 6, 5 | same | Physics feature widths; must equal `model/physics_schema.py` when Physics Cross Attention is on |
| `physics_tokenizer_hidden_size` | `None` (= `hidden_size`) | same | Physics tokenizer MLP width |
| `use_root_token` | `False` | `False` | One root-body token per robot in the Physics Cross Attention memory ([C1](zero_shot.md#c-physics-the-encoder-could-not-see)); needs `root_features` |
| `root_feature_size` | 13 | same | Root features (`model.physics.root_features`) |
| `dimensionless_physics` | `False` | `False` | Physics features in units of total mass, size and `sqrt(L / g)` ([A3](zero_shot.md#a-targets-and-inputs-that-depend-on-physics-only)); needs `root_features` |
| `use_morphology_modulation` | `False` | `False` | ModuMorph-style contextual modulation ([D4](zero_shot.md#d-coverage-of-morphologies)) |
| `modulation_rank` | `None` (= heads) | same | Rank of the modulation's input-projection correction |
| `physics_masking` | `False` | `False` | Learned mask embeddings for masked physics modeling ([E3](zero_shot.md#e-robustness-to-wrong-or-missing-specifications)) |

`EmbodimentGraph.from_urdf` options:

| Argument | Default | Meaning |
|---|---|---|
| `feature_config` | `JointFeatureConfig()` | Hardware features per joint |
| `joint_order` | `None` (depth-first) | Explicit joint name order |
| `include_mimic_joints` | `False` | Keep mimic joints as tokens |
| `multi_dof_joints` | `"skip"` | `"skip"` or `"expand"` floating and planar joints |
| `sibling_edges` | `True` | Add undirected edges between joints sharing a parent body |
| `frozen_joints` | `None` | Joints held rigid at given positions |
| `floating_base` | `None` (auto) | Root body moves freely |
