# DA-GET-BFM

A PyTorch encoder that turns any tree-structured robot, given its URDF, into
one token per degree of freedom (DoF) and encodes those tokens with a graph
transformer. It implements the Graph Embodiment Transformer (GET) from
[GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025) and
extends it to work across embodiments, with **fixed or floating bases**. Arms
and dexterous hands, quadrupeds, bipeds, humanoids, and multirotors can share
one model and one training batch.

The encoder outputs one embedding per joint, `H_{t,i}`, plus a pooled
embodiment embedding. On top of GET-Zero it adds the following. None of it is
in the GET-Zero paper:

1. **Mass-matrix attention bias.** Each self-attention layer gets a bias from
   the normalized joint-space mass matrix `M(q)`. The graph says how joints are
   connected; `M(q)` says how strongly they are dynamically coupled right now.
2. **Physics Cross Attention.** Every observation token attends to rigid-body,
   actuator, and dynamics tokens of all joints. These come from:
   - URDF inertials;
   - motor and propeller specs;
   - analytic dynamics: `M_ii(q)`, `c(q, qd, base twist)`, `g(q)`, friction,
     and the last applied torque.

   Parts that do not change every control step are cached.
3. **Fixed- and floating-base rigid-body dynamics.**
   - RNEA with base motion and the `(6+J)×(6+J)` floating-base mass matrix;
   - forward dynamics;
   - ground-contact forces;
   - propeller thrust and drag.

   When a dataset has no joint torques, they are **estimated by inverse
   dynamics**, with contact forces distributed over the feet of legged robots.
4. **Physics pretraining.** Six auxiliary heads train the encoder:
   - A: forward kinematics;
   - B: action-conditioned forward dynamics, including the base acceleration of
     floating robots;
   - C: inverse dynamics;
   - D: motor feasibility and margins;
   - E: cross-embodiment action effect;
   - F: rigid-body consistency.

   Training uses the Hugging Face `Trainer`.
5. **Training data for any robot.**
   - Synthetic domain-randomized data: fixed, harnessed floating, and flying
     robots.
   - **Legged robots standing and moving on the ground in MuJoCo**, with exact
     contact forces and torques.
   - Recorded trajectories from the Hugging Face Hub.
   - **Converters** for GrandTour, MIPO, NeuroBEM, LocoMuJoCo, AMASS→G1, and
     LAFAN1.
   - **A config generator** that reads a URDF (and a dataset) and fills in the
     floating base, feet, rotors, propeller coefficients, and column mapping.
6. **A cross-embodiment behavior foundation model.** BFM-Zero's FB-CPR
   networks, with an actor and critics that read the GET joint embeddings,
   cross-attend to the latent z, and output one action per DoF.

## Contents

- [Why one token per DoF](#why-one-token-per-dof)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
  - [1. URDF to embodiment graph](#1-urdf-to-embodiment-graph)
  - [2. Fixed and floating bases](#2-fixed-and-floating-bases)
  - [3. Per-joint hardware features](#3-per-joint-hardware-features)
  - [4. Token assembly](#4-token-assembly)
  - [5. Encoder and attention bias](#5-encoder-and-attention-bias)
  - [6. Physics Cross Attention](#6-physics-cross-attention)
  - [7. What is recomputed when](#7-what-is-recomputed-when)
  - [8. Batching different robots](#8-batching-different-robots)
- [Rigid-body dynamics](#rigid-body-dynamics)
- [Estimating joint torques without torque sensors](#estimating-joint-torques-without-torque-sensors)
- [Physics pretraining](#physics-pretraining)
- [Training with the Hugging Face Trainer](#training-with-the-hugging-face-trainer)
- [Ground contact in synthetic data (MuJoCo)](#ground-contact-in-synthetic-data-mujoco)
- [Writing data configs automatically](#writing-data-configs-automatically)
- [Converting datasets](#converting-datasets)
- [Recommended training datasets](#recommended-training-datasets)
- [Behavior foundation model (BFM-Zero on GET)](#behavior-foundation-model-bfm-zero-on-get)
- [Configuration reference](#configuration-reference)
- [Repository layout](#repository-layout)
- [Testing and validation](#testing-and-validation)
- [Limitations](#limitations)
- [Knowledge graph](#knowledge-graph)
- [References](#references)

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

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and needs Python 3.10 or
newer.

```bash
git clone git@github.com:apoman123/DA-GET-BFM.git
cd DA-GET-BFM
uv sync --all-extras                     # core + MuJoCo, dataset converters, validation tools
uv run pytest                            # runs the test suite
uv run python scripts/download_urdfs.py  # Unitree Go1, H1, G1 and ANYmal D URDFs
```

URDF parsing and the dynamics use only PyTorch and the Python standard library,
so no robotics packages are needed. `transformers`, `datasets`, and
`accelerate` are used by the training pipeline. The optional extras are:

| Extra | Packages | Needed for |
|---|---|---|
| `sim` | `mujoco` | ground-contact simulation (`"type": "mujoco"`) |
| `convert` | `numcodecs` | GrandTour (zarr) conversion |
| `validation` | `h5py` | MIPO conversion and the torque-validation script |

`uv sync` alone installs the core; tests that need an extra are skipped
without it.

## Quick start

Encoder only, for a floating-base robot, with Physics Cross Attention:

```python
import torch

from model import (
    EmbodimentGraph, GetConfig, GetEncoder, GetTokenizer,
    JointFeatureConfig, collate_embodiments,
)
from model.physics import PhysicsFeatureCache, actuator_specs

# 1. One graph per robot, built from URDF. Legged URDFs usually start at the
#    trunk without a floating joint, so say that the base is floating.
features = JointFeatureConfig.get_zero()
graphs = [
    EmbodimentGraph.from_urdf("assets/urdf/go1.urdf", features, floating_base=True),  # quadruped, 12 DoF
    EmbodimentGraph.from_urdf("assets/urdf/h1.urdf", features, floating_base=True),   # humanoid, 19 DoF
]
batch = collate_embodiments(graphs)                     # pads to 19 joints

# 2. Physics features: static ones are computed here once.
specs = [actuator_specs(graph) for graph in graphs]     # URDF limits + your overrides
physics = PhysicsFeatureCache(graphs, specs, configuration_interval=4)

# 3. Observations with history (oldest first) -> one token per joint.
H, J = 3, batch.joint_mask.shape[1]
q = torch.zeros(len(graphs), H, J)
qd = torch.zeros(len(graphs), H, J)
gravity = torch.tensor([[0.0, 0.0, -9.81]] * len(graphs))   # IMU projected gravity x 9.81
gyro = torch.zeros(len(graphs), 3)                          # base angular velocity (base frame)
tokenizer = GetTokenizer(fixed_local_size=features.size, variable_local_size=2, history_size=H)
tokens = tokenizer(
    batch.joint_features, batch.joint_mask,
    variable_local=torch.stack((batch.normalize_joint_positions(q), qd), dim=-1),
)

# 4. Encode.
encoder = GetEncoder(
    GetConfig.get_zero(input_size=tokenizer.token_size, use_physics_cross_attention=True)
).eval()
memory = encoder.build_physics_memory(               # cache rigid-body and actuator keys/values
    physics.rigid_body_features, physics.actuator_features, batch.joint_mask
)
dynamics = physics.update(                           # M_ii, c, g, friction, last torque
    q[:, -1], qd[:, -1], gravity=gravity, base_angular_velocity=gyro
)
hidden = encoder(
    tokens,
    mass_matrix=batch.mass_matrix(q[:, -1].double()).float(),
    dynamics_features=dynamics,
    physics_memory=memory,
    **batch.encoder_inputs(),                        # parents, mask, edges, precomputed distances
)                                                    # [2, 19, 512]; padded joints are zero
```

Pretraining in one command (details below):

```bash
uv run python scripts/train_physics_get.py \
    --data_config configs/data/synthetic_unitree.json \
    --output_dir runs/synthetic --max_steps 2000 --per_device_train_batch_size 64
```

## How it works

```text
URDF ──parse_urdf──▶ UrdfRobot ──EmbodimentGraph.from_urdf──▶ EmbodimentGraph
                                                               │  joints, parents, edges, base body,
                                                               │  link frames (feet, rotors), inertia, limits
           actuator_specs / rigid_body_features ◀──────────────┤
                     several graphs ──collate_embodiments──▶ EmbodimentBatch
                                                               │
 joint + base observations ──GetTokenizer──▶ observation tokens [B, J, F]
 rigid body │ actuator │ dynamics ──physics tokenizers──▶ physics tokens [B, 3J, hidden]
                                                               │
 tokens, M(q), graph, physics ──GetEncoder──▶ joint embeddings H [B, J, hidden]
                                                               │
                        auxiliary heads A–F (pretraining only) ┘
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
| Dynamics | `τ = M qdd + c + g` | joint rows plus 6 base rows; base twist and acceleration enter RNEA; external wrenches (contacts, thrust) close the base rows |
| Missing torques | inverse dynamics | inverse dynamics with contact forces distributed over the feet, or with rotor thrust |
| Observation globals | gravity direction (constant) | gravity direction, base angular velocity, base linear velocity / robot size, flags |
| FD label (task B) | `qdd` | `qdd` plus base linear and angular acceleration |
| Effect label (task E) | in the root frame | in the base frame at time `t`, including base motion |

A humanoid that only moves its arms while standing still can be modelled as
fixed-base, by freezing or not recording the legs. Without a contact model, a
floating-base robot is assumed held by an unmodelled support, and its joint
torques are the fixed-base ones.

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

`GetTokenizer` builds token `j` exactly as GET-Zero does:

```text
T_j = [ o_vg^{t-H+1..t},  o_fg,  o_vl,j^{t-H+1..t},  o_fl,j ]
```

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

`GetEncoder` (`model/modeling_get.py`) runs:

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

```text
normalized_mass[i, j] = M(q)[i, j] / (sqrt(M(q)[i, i] * M(q)[j, j]) + mass_epsilon)

A[i, j] = q_i · k_j / sqrt(d)
        + spatial[SPD(i, j)]                 # GET-Zero
        + parent[P(i, j)] + child[C(i, j)]   # GET-Zero
        + linear(normalized_mass[i, j])      # this repository
```

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

- `B_i` is the body frame of joint `i`. Its origin is the joint origin, its axes
  are the joint frame's, and it moves with the body that joint `i` drives (the
  child link after fixed links are merged).
- `B_p(i)` is the parent joint's body frame, or the base frame.
- For prismatic joints read N for N m, m for rad, and m/s for rad/s.

**Rigid body (25), from URDF, static per robot**

| Symbol | Size | Unit | Frame |
|---|---|---|---|
| `m_i` | 1 | kg | — |
| `vech(I_i)`: `Ixx, Ixy, Ixz, Iyy, Iyz, Izz` about the CoM | 6 | kg m² | `B_i` |
| `r_i^CoM` | 3 | m | `B_i` |
| `a_i^axis` (unit) | 3 | 1 | `B_i` |
| `T_i^parent`: translation and first two rotation columns at `q_i = 0` | 3 + 6 | m, 1 | `B_p(i)` |
| `jointType_i` (one-hot revolute, continuous, prismatic) | 3 | 1 | — |

**Actuator (14), from `ActuatorSpec`, static per robot**

| Symbol | Unit | Default when the URDF lacks it |
|---|---|---|
| `τ_i^peak` | N m | URDF `effort` |
| `τ_i^cont` | N m | `0.5 τ^peak` |
| `ω_i^max` | rad/s | URDF `velocity` |
| `P_i^max` | W | `0.5 τ^peak ω^max` (torque falls off above half the no-load speed) |
| `N_i` (gear ratio) | 1 | URDF `<transmission>` reduction, else 1 |
| `K_p,i`, `K_d,i` | N m/rad, N m s/rad | 0, or `actuator_defaults` |
| `k_f,i` propeller thrust coefficient (thrust `k_f ω²` along `a_i`, signed) | N s²/rad² | 0 (rotors only) |
| `k_m,i` propeller drag coefficient (drag moment `k_m ω²`) | N m s²/rad² | 0 (rotors only) |
| `mode_i` (one-hot torque, position, velocity, passive, thrust) | 1 | the data source's action type |

`k_f`, `k_m`, and the `thrust` mode extend the actuator list so multirotors can
be described. Every other joint has zeros there. The defaults are assumptions.
Pass real motor data through
`actuator_specs(graph, overrides={"*_knee_joint": {"continuous_torque": 23.7}})`
or, in a data config, through `actuator_overrides` (exact names or glob
patterns).

**Dynamics (7), from the state at time `t`**

| Symbol | Unit | Depends on |
|---|---|---|
| `M_ii(q_t)` | kg m² | `q` |
| `c_i(q_t, qd_t)` Coriolis and centrifugal, including base rotation for a floating base | N m | `q`, `qd`, base twist |
| `g_i(q_t)` gravity, with gravity in the base frame | N m | `q`, gravity direction |
| `τ̂_f,i` friction `d qd + μ tanh(qd / 0.01)`, plus propeller drag `k_m ω\|ω\|` | N m | `qd` |
| `τ_{i,t-1}^applied` | N m | measurement or estimate |
| `\|g_i\| / (τ_i^cont + ε)` gravity load | 1 | `q` |
| source of `τ_{t-1}`: 1 measured, 0.5 inverse-dynamics estimate, 0 unknown | 1 | data |

Each group goes through its own tokenizer (`PhysicsFeatureTokenizer`). The
tokenizer first maps every feature to `asinh(x / s)` with a per-feature scale
`s`, which is linear for small values and logarithmic for large ones, so a
20 g finger link and a 10 kg thigh are both well conditioned. A two-layer MLP
then gives one token per joint, and a learned group embedding is added. The
three groups are concatenated into a memory of `3J` tokens:

```text
memory = [ rigid_body_1..J | actuator_1..J | dynamics_1..J ]

PhysicsCrossAttention(i, (g, j)) = softmax_j,g( q_i · k_(g,j) / sqrt(d)
                                                + spatial_phys[SPD(i, j)] + group_bias[g] )
```

Queries are the observation tokens. The bias lets a joint focus on its own
physics (`SPD = 0`) and on nearby joints. The bias tables are per layer and per
head, and padded joints are masked out. Physics Cross Attention is controlled by
`use_physics_cross_attention` and `physics_cross_attention_layers` (default:
every layer).

### 7. What is recomputed when

| Tier | Features | Changes when | At training time | At control time |
|---|---|---|---|---|
| embodiment | all rigid-body and actuator features | the robot or its randomized parameters change | once per robot, when the dataset is built | once per robot; their per-layer keys and values are cached in a `PhysicsMemory` |
| configuration | `M_ii(q)`, `g(q)`, gravity load | `q` or the gravity direction | once per frame, cached on disk | `PhysicsFeatureCache`: every `configuration_interval` steps, or sooner if any joint moves more than `configuration_tolerance` or gravity turns |
| velocity | `c(q, qd, base twist)`, `τ̂_f` | `qd`, the base twist (and `q`) | once per frame, cached on disk | every `velocity_interval` steps, if `qd` or the base twist moves more than `velocity_tolerance`, or with the configuration tier |
| measurement | `τ_{t-1}^applied` | every step | per frame | every step |

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

## Physics pretraining

`GetForPhysicsPretraining` (`model/pretraining.py`) is a Hugging Face
`PreTrainedModel`, so `save_pretrained`, `from_pretrained`, and `push_to_hub`
all work. It wraps the encoder with six heads. `N_e` is the number of valid
joints of a sample's robot. Every per-joint loss is averaged per robot first,
so a 29-DoF humanoid does not outweigh a 4-rotor drone.

| Task | Head input | Target | Loss |
|---|---|---|---|
| **A. Forward kinematics** | `H_i` | `p_i / L_e` (`L_e` = robot size) and `R_i ∈ SO(3)` of every joint frame in the base frame | `1/N_e Σ λ_p ‖p̂−p‖² + λ_R d_SO(3)(R̂, R)²`; `fk_orientation=False` gives the position-only first pass |
| **B. Forward dynamics** | `[H_i; E_τ(τ_i^applied)]` passed through `mixing_layers` of graph action mixing | `qdd_t`; for a floating base also the base linear and angular acceleration (from the pooled mixed tokens) | `1/N_e Σ ((qdd̂ − qdd) / σ_i^a)²` + `λ_base` · mean of squared base-acceleration errors normalized per robot |
| **C. Inverse dynamics** | `[H_i; E(qdd_i^desired, base acceleration^desired)]` with graph mixing | `τ^required` = inverse dynamics, with contacts for legged robots | `1/N_e Σ ((P(τ̂ − τ))_i / (τ_i^cont + ε))²`; `P = I` for measured or simulated torques, the motion-determined projector for estimated torques in contact |
| **D. Motor feasibility** | `[H_i; E(a_i^proposal)]` | peak and continuous feasibility flags; margins for peak torque `1 − \|τ\|/τ^max(qd)`, continuous torque, power, velocity, and joint limit | `λ_f BCE + λ_m Huber` |
| **E. Action effect** | `[H_i; E(a_i)]` with graph mixing | at end-effector frames, in the base frame at `t`: `Δp`, `R_{t+1}`, `v_{t+1}`, `ω_{t+1}` | `λ_p‖Δp̂−Δp‖² + λ_R d_SO(3)² + λ_v‖v̂−v‖² + λ_ω‖ω̂−ω‖²` |
| **F. RBD consistency** (optional) | the FD prediction | `r = M qdd̂ + c + g + τ_f − τ^applied` | `1/N_e ‖W_e r‖²` with `W_e = 1/(τ^cont + ε)`; fixed-base samples only |

Notes on the design:

- **B: per-joint inputs are not enough.** Seeing only its own torque, a joint
  cannot represent `M⁻¹`'s coupling across joints, or how the base reacts. The
  FD, ID, and effect heads therefore start with `mixing_layers` (default 1) of
  graph self-attention over `[H_i; E(x_i)]`. Set it to 0 for the purely
  per-joint baseline.
- **D: requested torques and limits.** The requested effort comes from the
  proposal through the actuator mode:

  | Mode | Requested effort |
  |---|---|
  | torque | the proposal itself |
  | position | `K_p (a − q) − K_d qd` |
  | velocity | `K_d (a − qd)` |
  | thrust (rotor) | `K_d (a − qd) + k_m a\|a\|` |

  The torque-speed envelope is `τ^max(qd) = min(τ^peak, P^max/|qd|)` below
  `ω^max`, and zero above it. "Peak feasible" means executable now;
  "continuous feasible" also respects `τ^cont`. A short motor boost is
  therefore not mistaken for a sustainable command. The predictor does not
  replace hard limits in the motor interface.
- **E: normalization and frames.** `Δp` and `v` are divided by the robot size
  `L_e`. Each of `Δp`, `v`, and `ω` is then divided by its RMS over that
  robot's data, so translation and rotation terms have comparable size.
  Targets are in the base frame at time `t`, so a walking robot's or a drone's
  own motion is part of the effect. Rotations use the geodesic distance on
  SO(3). Only end-effector frames are supervised (leaf joints by default, or
  `end_effectors`).
- **F: off by default.** With analytic labels it largely duplicates `L_FD`, so
  its weight is 0 by default. Treat it as an ablation.

Weights: `--fk_weight`, `--fd_weight`, `--id_weight`, `--motor_weight`,
`--effect_weight`, `--rbd_weight`, plus the `lambda_*` fields of
`GetPhysicsPretrainingConfig` (including `lambda_base_acceleration`).

## Training with the Hugging Face Trainer

```bash
uv run python scripts/download_urdfs.py

# 1. Synthetic Go1, H1, and G1, each with fixed-base and floating-base episodes
uv run python scripts/train_physics_get.py \
    --data_config configs/data/synthetic_unitree.json \
    --output_dir runs/synthetic --max_steps 2000 \
    --per_device_train_batch_size 64 --learning_rate 5e-4 --warmup_steps 100 \
    --eval_strategy steps --eval_steps 500 --logging_steps 50

# 2. Real floating-base humanoid data (Unitree G1 whole-body teleoperation) + synthetic G1
uv run python scripts/train_physics_get.py \
    --data_config configs/data/g1_wbt_floating.json --output_dir runs/g1_wbt

# 3. Multirotor flight (synthetic quadrotor with randomized size and mass)
uv run python scripts/train_physics_get.py \
    --data_config configs/data/quadrotor_flight.json --output_dir runs/quadrotor
```

Several data configs can be merged into one by listing all their sources, for
example to pretrain on legged, humanoid, and flying robots together. Every
`TrainingArguments` flag works, including multi-GPU runs through
`accelerate launch`. The script adds model flags:
- `--preset small|get_zero`, plus width, depth, and head overrides;
- `--physics_cross_attention false` for an ablation;
- `--physics_layers 0,2`, `--mass_encoding`, `--mixing_layers`;
- `--fk_orientation false`, and the task weights.

The Trainer logs `loss_fk`, `loss_fd`, `loss_id`, `loss_motor`, `loss_effect`,
`loss_rbd`, and the matching `eval_loss_*` next to the total loss.
`runs/<name>` then holds a `from_pretrained` checkpoint and `embodiments.json`.
Load the encoder for downstream use with:

```python
from model.pretraining import GetForPhysicsPretraining
model = GetForPhysicsPretraining.from_pretrained("runs/g1_wbt")
embeddings = model.encode(**batch)          # [B, J, hidden]; or model.encoder directly
```

### Data config

A data config is JSON. Relative paths start from the config file.

```json
{
  "history_size": 3,
  "joint_features": "get_zero",
  "eval_fraction": 0.05,
  "allow_estimated_torque_for_fd": true,
  "use_estimated_previous_torque": true,
  "max_estimate_residual": 0.5,
  "estimated_torque_weight": 0.3,
  "sources": [ { "type": "synthetic", ... }, { "type": "huggingface", ... } ]
}
```

**`"type": "synthetic"`** simulates a URDF robot with the analytic dynamics.
Each of `randomized_embodiments` variants perturbs:
- link masses and inertias;
- link lengths (inertia is scaled consistently);
- damping and friction, as fractions of each joint's own limits, so the same
  ranges suit a finger, a knee, and a propeller;
- torque and speed limits.

`base_modes` is cycled over the variants:

| Mode | What is simulated |
|---|---|
| `fixed` | Base bolted down; gravity tilted per episode; PD with gravity feed-forward tracks random joint targets. |
| `floating` | The base is free and hangs from a harness that holds the whole-body center of mass against gravity, with no moment. Base rotation and base-limb coupling are fully dynamic. PD gains use the free-base effective inertia. |
| `flight` | A multirotor flies under its own rotor thrust: a geometric controller tracks random waypoints and maps thrust and moments to rotor speeds through the actual rotor layout (any rotor count). It needs every actuated joint to be a `thrust` rotor. |

Commands are clipped to the torque-speed envelope before they are applied, so
labels always use the executed torque, never an unexecuted proposal. Each state
also gets `probe_torques` extra torques (log-uniform from 2% to 200% of peak,
random sign), each simulated one step. This tests action conditioning at a
fixed pose and gives the motor head infeasible proposals. Ground contact is not
simulated; contact-rich legged data comes from recorded datasets.

**`"type": "huggingface"`** reads any Hub dataset (`repo_id`) or local folder
or file (`path`) in the LeRobot parquet layout, or any parquet, CSV, or JSON
table with one vector column per signal. Any column key can be a list of
columns that are concatenated in order.

| Key | Meaning |
|---|---|
| `urdf`, `joint_names` | URDF joint of each recorded state entry, in order. Unlisted joints are frozen (`frozen_joints` gives their positions, default 0). |
| `state_key`, `state_indices` | Position columns and which entries to use. A null index means the position is not recorded (rotors); it is integrated from the velocity. |
| `velocity_key`, `torque_key` | Optional columns. Without velocities, positions are finite-differenced. Without torques, torques are estimated (previous section). |
| `action_key`, `action_indices`, `action_type` | Commands (`position`, `velocity`, or `torque`; rotor joints keep the `thrust` mode). Set `actuator_defaults.kp`/`kd` for position control so requested torques are meaningful. |
| `state_scale`, `state_offset` | `q_urdf = scale · q + offset`, for sign, unit (degrees), or zero-offset differences. |
| `floating_base` | The base is free (legged robots, drones). |
| `base_orientation_key` (+ `_indices`, `base_orientation_format`: `wxyz`, `xyzw`, `rpy`, `matrix`), `base_position_key` | Base pose in the world. |
| `base_linear_velocity_key`, `base_angular_velocity_key` (+ `_frame`: `base` or `world`) | Base twist; IMU gyros give angular velocity in the base frame. |
| `projected_gravity_key`, `imu_acceleration_key` | Unit gravity direction and accelerometer specific force, both in the base frame. |
| `contact_links`, `contact_key`, `contact_threshold`, `surface_contacts`, `contact_height_tolerance`, `contact_speed_tolerance` | Feet, recorded contact flags or forces, and the detection heuristic. |
| `fps`, `episode_key`, `frame_key`, `max_episodes`, `smoothing_window`, `gravity`, `end_effectors` | Timing, grouping, size, smoothing, a constant gravity in the base frame, task-E frames. |

Processed features and labels are cached in `.cache/physics_data/<hash>.pt`.
The hash covers the config and the URDF contents, so the analytic dynamics run
once per dataset rather than once per epoch. Episodes, not frames, are split
between train and eval.

## Ground contact in synthetic data (MuJoCo)

The built-in `synthetic` floating mode hangs the robot from a harness, so no
foot touches the ground. Legged data with real foot-ground contact comes from
the `"type": "mujoco"` source (`model/training/ground_sim.py`, `uv sync --extra
sim`). It builds a MuJoCo model from the same embodiment graph:

- one body per token with the graph's merged inertia (so MuJoCo and our
  dynamics agree; the mass matrix and bias forces match to 1e-7 in the tests);
- a free root joint, and hinge or slide joints with the URDF limits, damping,
  and friction;
- position actuators (PD inside the simulator, which keeps stiff gains stable)
  limited to the peak torque;
- collision spheres at the contact links, one per point foot or four corners
  per flat foot (`contact_geometry`: `radius`, `sole_offset`, `foot_size`),
  and a floor with randomized friction. Only the feet collide.

The robot starts in `nominal_pose` with its feet on the floor. It then tracks
smoothly changing joint targets around that pose: crouching, leaning, and
shifting weight between feet, some lifting off. It also receives random
pushes. Episodes stop if the robot falls. PD control alone cannot balance a
biped, so robots with two feet get a **spotter**: a soft spring-damper on the
base that pulls it upright and back over its start point, like a person
steadying the robot. It carries about 5–10% of the weight in the tests and
runs; the feet carry the rest.

**Recorded labels are exact for the simulated system:**
- applied joint torques;
- joint and base accelerations;
- per-foot contact flags;
- the push and spotter wrench (passed as a known base wrench);
- probe torques solved by MuJoCo with the contacts in place, so task B covers
  action conditioning while standing on the ground.

The inverse-dynamics labels equal the applied torques, and no contact
estimation is needed.

```bash
uv sync --extra sim
uv run python scripts/train_physics_get.py --data_config configs/data/ground_contact.json --output_dir runs/ground
```

`configs/data/ground_contact.json` has standing poses and foot geometry for
Go1, ANYmal D, G1, and H1; all four stand for the whole episode under the
default perturbations. For another robot, set `nominal_pose` to a standing
pose (the default is zero clamped into the limits) and `contact_geometry` to
its feet. Mixing ground data with recorded data and harness or flight data
covers stance, contact switching, and free motion.

What it does not do: walk. The motions are standing, crouching, weight
shifting, partial foot lift, and recovering from pushes, not gaits; recorded
datasets provide locomotion.

## Writing data configs automatically

`scripts/make_data_config.py` (backed by `model/robot_analysis.py` and
`model/conversion/mapping.py`) writes a data config from a URDF, and
optionally from a dataset:

```bash
uv run python scripts/make_data_config.py --urdf assets/urdf/g1_29dof.urdf \
    --dataset unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit --synthetic --ground \
    --output configs/data/g1_auto.json
```

From the URDF it infers:

| Setting | How |
|---|---|
| multirotor | three or more parallel `continuous` leaf joints spread around the base; rotors get `thrust` actuators with `k_f = (T/W) m g / (N ω_max²)` (`--thrust_to_weight`, default 2.5; `ω_max` from the URDF or `--max_rotor_speed`) and `k_m = 0.016 m · k_f` |
| legged, `floating_base` | two or more independent chains of at least two joints ending in the lowest links, well below the base; a URDF `floating` joint always means floating |
| `contact_links` | the lowest link of each leg chain, preferring names with foot, toe, sole, or ankle |
| `surface_contacts` | two feet or fewer (flat feet that carry moments) |
| `end_effectors` | the foot joints and the other leaf joints (hands, grippers, rotors) |
| position gains | `K_p = effort / 1 rad`, `K_d = 0.025 K_p` when the dataset's actions are position targets |

From the dataset (`meta/info.json`, on the Hub or a converted folder) it
infers:
- **Joint columns**, matched to URDF joints by name:
  - `kLeftShoulderPitch` → `left_shoulder_pitch_joint`;
  - Unitree `FL0/FL1/FL2` → hip, thigh, calf;
  - columns split by body part (`observation.state.left_arm` with
    `shoulder_pitch`, …) take words from the column name;
  - joints the dataset does not record are frozen.
- **Action columns** with the same names.
- **Base pose, angular and linear velocity, projected gravity, IMU, and
  contacts**, from feature names such as `x y z qx qy qz qw` or
  `gx gy gz wx wy wz`.

It prints every decision and match. Unitree's G1 whole-body teleoperation
dataset maps all 29 joints, the base pose, gyro, and gravity without edits.
Always read the report, and edit the JSON where the guess is wrong. Propeller
coefficients and PD gains are sized from rules of thumb; use measured values
when you have them.

## Converting datasets

`scripts/convert_dataset.py` (`model/conversion/`) converts datasets that are
not LeRobot parquet into one table format. The output folder has:
- `data/chunk-000/file-000.parquet`, one row per frame, with these columns:
  - `observation.state`, `observation.velocity`, `observation.effort`;
  - base position and x-y-z-w quaternion, and the base-frame twist;
  - `observation.imu_acceleration`, `observation.contact`, and so on;
- `meta/info.json`, with named features and provenance;
- `source.json`, a ready data-config source pointing at the folder.

```bash
uv sync --all-extras
uv run python scripts/convert_dataset.py grandtour  --output data/grandtour --max_missions 2
uv run python scripts/convert_dataset.py mipo       --output data/mipo --pattern "indoor/*" --max_files 20
uv run python scripts/convert_dataset.py neurobem   --output data/neurobem --max_flights 40
uv run python scripts/convert_dataset.py locomujoco --output data/loco_g1 --robot UnitreeG1 --collection Lafan1
uv run python scripts/convert_dataset.py amass_g1   --output data/amass_g1 --pattern "*Walk*" --max_files 200
uv run python scripts/convert_dataset.py lafan1     --output data/lafan1_g1 --robot g1
# then train on it (or combine sources from several source.json files in one config)
uv run python scripts/make_data_config.py --urdf assets/urdf/anymal_d.urdf --dataset data/grandtour --output configs/data/grandtour.json
```

| Converter | Source and format | What it does |
|---|---|---|
| `grandtour` | `leggedrobotics/grand_tour_dataset`, zarr in tar | Downloads only `anymal_state_state_estimator.tar` per mission (about 100 MB). Measured joint torques; pose and quaternion (x, y, z, w); twist in the base frame; contact from each foot's state (1 contact, 2 slip, 0 open). Resamples from 400 Hz to 200 Hz and cuts missions into 20 s episodes. Reads zarr without the zarr package. Use the ANYmal D URDF; the payload pan/tilt joints are frozen. |
| `mipo` | `Open-IO/mipo-go1-odometry`, HDF5 | 400 Hz joints, measured torques, body IMU (orientation, gyro, accelerometer as `imu_acceleration`), foot contact from force counts above 200. Joints `FL0…RR2` are renamed to Go1 URDF names. Mocap is not used (its clock offset is uncalibrated). |
| `neurobem` | UZH `processed_data.zip`, CSV | Reads only the selected flights from the 627 MB remote zip with HTTP range requests. It writes the four motor speeds as rotor joints (signed by spin direction), the Vicon pose and twist, and the accelerometer specific force; the column type is detected from the data. It **identifies the propeller model**: `k_f` is fitted from thrust (R² about 0.9). The roll, pitch, and yaw fits are poor on this data (R² < 0.25), so the published frame size is used instead (6-inch frame, 0.13 m arms, `k_m = 0.016 m · k_f`, maximum speed from the paper's 33 N total thrust). It then writes `neurobem_quadrotor.urdf` and the actuator overrides into `source.json`. Fit qualities are in `meta/info.json`. |
| `locomujoco` | `robfiras/loco-mujoco-datasets`, npz | MuJoCo `qpos`/`qvel` with a free root: position, w-x-y-z quaternion → x-y-z-w, world linear velocity rotated into the base frame, body angular velocity. Joint names come from the file (G1 names match the Unitree URDF; use `--rename` for others). CC BY-NC-ND 4.0. |
| `amass_g1` | `ember-lab-berkeley/AMASS_Retargeted_for_G1`, npz | Named G1 joint positions and the pelvis pose (w-x-y-z quaternion). The stored joint velocities do not match the positions, so velocities are left to finite differencing. |
| `lafan1` | `lvhaidong/LAFAN1_Retargeting_Dataset`, CSV | G1, H1, or H1_2 at 30 Hz: root position and x-y-z-w quaternion, then joints in the order listed in the dataset README. Motions CC BY-NC-ND 4.0. |

The conventions (quaternion order, frames, units, contact codes) were checked
against downloaded files of each dataset. Kinematic-only datasets (LocoMuJoCo,
AMASS, LAFAN1) have no contacts or torques; contacts are detected and torques
estimated, and retargeted motions can exceed actuator limits.

## Recommended training datasets

The tasks need joint-level trajectories of robots with a known URDF or MJCF.
Below, ✅ means the `huggingface` source reads the dataset directly, ⚙️ means
`scripts/convert_dataset.py` converts it, and 🔧 means it needs your own
conversion to a table with `episode_index`, `frame_index`, and one vector
column per signal (see [Converting datasets](#converting-datasets)). "Torque" says whether measured joint torques are included. Without
them, torques are estimated as described above. Facts were checked against
each dataset's card, `meta/info.json`, or files on 2026-11-07. Check licenses
before use.

**Quadrupeds (floating base)**

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [GrandTour](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) (`leggedrobotics/grand_tour_dataset`) | ANYmal D, 12 | 52 missions | **yes** (joint torque, current, commands, PD gains) | estimator pose and twist, IMUs, GNSS, per-foot contact state and wrench | zarr in tar ⚙️ `grandtour` | MIT on the card, CC BY-SA 4.0 on the website | The best fit: measured torques and contacts. Joint names match [anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description) (`download_urdfs.py` fetches it). |
| [MIPO-Go1](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry) (`Open-IO/mipo-go1-odometry`) | Unitree Go1, 12 | 84 runs, about 1.6 h, 400 Hz | **yes** (current-based estimate) | body IMU with orientation, OptiTrack indoors, 4 foot-force sensors | HDF5 ⚙️ `mipo` | AGPL-3.0 | Used for the torque-estimation check above. Streams are unsynchronized; joint names `FL0`…`RR2` map to `FL_hip_joint`… |
| [DFKI Go2 state estimation](https://zenodo.org/records/19336009) | Unitree Go2, 12 | 27 runs, 1 kHz, 5.4 GB | **yes** (`JointState.effort`) | Vicon / RTK GNSS, IMU, foot contact flags and forces | ROS 2 bags 🔧 | CC BY 4.0 | Needs the dfki-quad message definitions. |
| [Leg-KILO](https://github.com/ouguangjun/legkilo-dataset) | Unitree Go1, 12 | 7 sequences, 50 Hz unique | unverified | IMU, onboard odometry, foot forces | ROS 1 bags 🔧 | CC BY 4.0 | Ground truth from offline map optimization. |
| [Kine2Go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go) (`MIMUW-Robotics/kine2go`) | Unitree Go2, 12 (simulated) | 800 rollouts, 60 Hz | no (PPO actions and PD gains) | base pose and twist | npy + pickle 🔧 | per clip: CC BY-NC 4.0 / CC BY-SA 4.0 / BSD-3 | Rebuild PD torques from actions and gains, or estimate. Open the pickles in a sandbox. |

**Humanoids and bipeds (floating base)**

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [Unitree G1 whole-body teleop](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit) (`unitreerobotics/G1_WBT_*`, many tasks) | Unitree G1, 29 + hands | e.g. 200 episodes / 168k frames per task, 30 Hz | no | base pose (xyz + quaternion xyzw), projected gravity and angular velocity | LeRobot parquet ✅ | Apache-2.0 | `configs/data/g1_wbt_floating.json`. The state is split into `lower_body`, `left_arm`, `right_arm`. The base position may be estimated. |
| [Humanoid Everyday](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday) (`USC-PSI-Lab/humanoid-everyday`) | Unitree G1 and H1_2 + hands | 8,949 episodes / 3.4M frames, 30 Hz | no (`tau_ff` is a command) | IMU and odometry in the combined repo | LeRobot v2.1 parquet ✅ | Apache-2.0 (card) | Joint vectors are unnamed; map them by the order in the project README. |
| [LocoMuJoCo datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) (`robfiras/loco-mujoco-datasets`) | 12 models: Unitree G1, H1, H1 v2, Talos, Apollo, Atlas, Booster T1, Fourier GR1T2, ToddlerBot, … | 19 default + 40 LAFAN1 motions per robot, 40 Hz | no | floating-base `qpos`/`qvel` with joint names | npz ⚙️ `locomujoco` | CC BY-NC-ND 4.0 | Many morphologies, good for cross-embodiment. Motions are kinematic, so some exceed actuator limits. |
| [LAFAN1 retargeted to Unitree](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (mirror `lvhaidong/LAFAN1_Retargeting_Dataset`) | G1, H1, H1_2 | 40 clips per robot, 30 Hz | no | root position and quaternion (xyzw) | CSV ⚙️ `lafan1` | motions CC BY-NC-ND 4.0 (LAFAN1); retargeting BSD-3 | The original `unitreerobotics` repo is no longer accessible. Kinematic only. |
| [AMASS retargeted to G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1) (`ember-lab-berkeley/AMASS_Retargeted_for_G1`) | Unitree G1, 29 | about 17.7k clips, 30 Hz | no | pelvis pose, joint positions and velocities, named | npz ⚙️ `amass_g1` | CC BY 4.0 on the card; AMASS terms likely apply | Large motion variety; kinematic only. |

**Multirotors (floating base, thrust actuators)**

| Dataset | Vehicle | Size | Rotor signal | Base | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [NeuroBEM](https://rpg.ifi.uzh.ch/NeuroBEM.html) | quadrotor, 0.772 kg | 1 h 15 min, 400 Hz | **motor speeds** (rad/s) and their derivatives | Vicon pose, velocity, acceleration, body rates | CSV ⚙️ `neurobem` | not stated | The best rotor-dynamics set. The converter identifies the thrust coefficient, generates the URDF, and writes the actuator overrides. |
| [Blackbird](https://github.com/mit-aera/Blackbird-Dataset) | quadrotor | 168 flights, >10 h; RPM about 190 Hz | **per-motor RPM** and PWM | 360 Hz mocap pose, IMU | ROS 1 bags (4.9 TB) 🔧 | repo MIT; data unverified | The official host was unreachable at check time; an Academic Torrents mirror exists. |
| [TII drone racing](https://github.com/tii-racing/drone-racing-dataset) | 5-inch FPV quad (open design) | 42 flights, 500 Hz | motor commands (0–1), no RPM | mocap pose and twist, IMU | CSV 🔧 | MIT | Commands are not speeds; map them with a thrust curve. |
| [UAV-SEAD](https://huggingface.co/datasets/aykutkabaoglu/UAV-SEAD_State_Estimation_Anomaly_Dataset) (`aykutkabaoglu/UAV-SEAD_…`) | various PX4 multirotors | 1,396 flights, about 52 h | PWM outputs | attitude, rates, local position, IMU | PX4 ulog + CSV 🔧 | CC BY 4.0 | Mixed, unidentified airframes. |
| Synthetic (`configs/data/quadrotor_flight.json`) | any multirotor URDF | unlimited | exact rotor speeds and torques | exact | built in ✅ | — | Exact thrust, drag, and base dynamics; randomized mass and arm length. |

**Legged robots on the ground (simulated)**

| Source | Robots | Labels | Notes |
|---|---|---|---|
| `"type": "mujoco"` (`configs/data/ground_contact.json`) | any legged URDF; poses provided for Go1, ANYmal D, G1, H1 | exact torques, accelerations, contacts, pushes | Standing, crouching, weight shifting, and push recovery with real foot contact; see [Ground contact](#ground-contact-in-synthetic-data-mujoco). |

**Fixed-base arms and hands (and arms of standing humanoids)**

| Dataset | Robot | Size | Torque | Format | Notes |
|---|---|---|---|---|---|
| [`unitreerobotics/G1_Dex3_*`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) | G1 arms 14 + Dex3 hands | e.g. 418 episodes / 352k frames, 30 Hz | no | LeRobot ✅ | `configs/data/g1_hub_plus_synthetic.json` (legs frozen, base fixed). |
| [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | Franka Panda, 7 | 95,658 episodes / 27.6M frames, 15 Hz | no | LeRobot ✅ | `state_key: "observation.state.joint_position"`; use `max_episodes`. |
| [`lerobot/aloha_sim_*`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | 2 × ViperX 300 | 50 episodes / 25k frames, 50 Hz | no | LeRobot ✅ | Map 12 arm joints; skip the grippers. |
| [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | SO-100 arm, 5 + gripper | 50 episodes / 19.6k frames, 30 Hz | no | LeRobot ✅ | Check position units; convert with `state_scale`/`state_offset`. |

**Robot models.** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
has `unitree_g1`, `unitree_h1`, `unitree_go1`, `unitree_go2`, `booster_t1`,
`apptronik_apollo`, `pal_talos`, `agility_cassie`, `anybotics_anymal_c`, and
`bitcraze_crazyflie_2` (MJCF; convert to URDF). Unitree's
[unitree_ros](https://github.com/unitreerobotics/unitree_ros) has URDFs for
G1, H1, H1_2, Go1, and Go2 (`scripts/download_urdfs.py` fetches three).

**A suggested mix for cross-embodiment pretraining:**
- the measured-torque quadruped sets (GrandTour, MIPO, DFKI Go2) for real
  dynamics;
- MuJoCo ground data for exact contact dynamics of every legged robot you
  care about;
- G1 WBT and LocoMuJoCo for humanoid motion;
- NeuroBEM for real rotor dynamics;
- synthetic fixed, floating, and flight data for every robot you care about
  (exact torques, probes, limits);
- held-out robots or randomized variants for evaluation.

## Behavior foundation model (BFM-Zero on GET)

This repo studies **cross-embodiment behavior foundation models**. It combines
two references:
- [BFM-Zero](https://lecar-lab.github.io/BFM-Zero/) (Li et al., 2025), a
  promptable humanoid controller trained with FB-CPR (Tirinzoni et al., ICLR
  2025) on forward-backward representations (Touati and Ollivier, NeurIPS
  2021);
- [GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025).

BFM-Zero's networks read one flat state vector of a single robot (the G1).
Here the actor and the critics read the per-DoF embeddings of `GetEncoder`.
Those embeddings already carry the graph, the hardware features, and (with
Physics Cross Attention) the dynamics. As a result, one policy outputs one
action per joint for any robot the encoder can tokenize.

```text
                 flat state s ──► BackwardMap B(s) ──► z  (goal / tracking / reward prompt)
                 flat state s ─┐                       │
                               └► Discriminator D(s,z) ◄┤
                                                       ▼
URDF + q, q̇, … ─► GetEncoder ─► H [B, J, 512] ─► LatentMemory(z, context) ─► K/V tokens
                                      │                               │
                                      ▼                               ▼
                        joint tokens (+ action embedding for F/Q) ── cross-attn → self-attn → FFN  × N
                                      │
                    ┌─────────────────┴────────────────────┐
     actor: per-joint head → tanh → a [B, J]     F / critics: readout query + mean → [P, B, z_dim or 1]
```

| Network | BFM-Zero | Here |
|---|---|---|
| Backward map `B(s)` | MLP: Linear → LayerNorm → Tanh → Linear, normalized to radius `√z_dim` | **Unchanged** (`BackwardMap`) |
| Discriminator `D(s, z)` | MLP on `[z, s]`, reward `log D − log(1−D)` | **Unchanged** (`Discriminator`) |
| Actor `π(a \| s, z)` | Residual MLP on `[s, z]` → `tanh` mean, truncated normal | `GetBfmActor`: joint tokens cross-attend to latent tokens, then a per-DoF head gives a `tanh` mean and a truncated normal |
| Forward map `F(s, a, z)` | Residual MLP, ensemble of 2, `z_dim` outputs | `GetForwardMap`: each joint token gets its own action, cross-attends to latent tokens, then the joints are pooled; ensemble of `num_parallel` |
| Critic, aux critic `Q(s, a, z)` | Same as F with one output | `GetForwardMap(config, 1)` |

Design details:
- **Latent tokens.** A single z would give cross-attention just one key: the
  softmax would then be constant and attention could not pick out anything.
  So each latent is mapped to `num_latent_tokens` tokens, each with a learned
  slot embedding.
  - A latent sequence `[B, L, z_dim]` (for example `B(s)` of upcoming
    reference frames) also gets sinusoidal positions.
  - Optional context tokens (base state, history) are appended to the same
    memory (`context_size`).
- **Padding and order.** Padded joints are masked everywhere and output zero
  actions. There are no joint positions, so permuting the joints permutes the
  actions and leaves `F` and `Q` unchanged (both are tested).
- **Unchanged from BFM-Zero:**
  - output layouts (`[num_parallel, batch, out]`);
  - `TruncatedNormal.sample(clip=…)`;
  - orthogonal initialization (`init_weights`);
  - latent sampling on the sphere;
  - goal, reward, weighted-reward, and tracking inference.

  The FB-CPR training losses can therefore be ported as they are.

```python
from model import BfmConfig, GetBfmModel

# hidden, batch from the quick start above
config = BfmConfig(embedding_size=hidden.shape[-1], backward_input_size=state_dim)
bfm = GetBfmModel(config)
bfm.init_weights()

z = bfm.goal_inference(goal_state)                       # or tracking_inference / reward_inference
action = bfm.act(hidden, z, batch.joint_mask)            # [2, 19], one command per DoF, padded = 0
dist = bfm.actor(hidden, z, config.actor_std, batch.joint_mask)
F = bfm.forward_map(hidden, z, dist.sample(clip=0.3), batch.joint_mask)   # [2, 2, 256]
q_fb = bfm.fb_q_value(F, z)                              # Fᵀz, [2, 2]
q_cpr = bfm.critic(hidden, z, action, batch.joint_mask)  # [2, 2, 1]
reward = bfm.discriminator.compute_reward(state, z)      # CPR reward
```

`BfmConfig` fields (defaults follow BFM-Zero's G1 configuration where one
exists):

| Field | Default | Meaning |
|---|---|---|
| `embedding_size` | required | `GetConfig.hidden_size` |
| `backward_input_size` | required | Flat state width read by B (and by D unless `discriminator_input_size` is set) |
| `backward_input_keys`, `discriminator_input_keys` | `("state", "privileged_state")` | Keys concatenated when the observation is a dict |
| `z_dim`, `norm_z` | 256, `True` | Latent size and its projection to the sphere |
| `hidden_size`, `num_attention_heads`, `intermediate_size` | 256, 4, 1024 | Actor and critic token width |
| `num_latent_tokens` | 8 | Tokens per latent vector |
| `actor_layers`, `forward_layers` | 4, 4 | Latent-conditioned blocks |
| `num_parallel` | 2 | Ensemble size of F and the critics |
| `action_size_per_joint` | 1 | Action entries per DoF |
| `context_size` | `None` | Width of extra context tokens |
| `backward_hidden_size`, `backward_hidden_layers`, `backward_norm` | 256, 1, `True` | B MLP (BFM-Zero) |
| `discriminator_hidden_size`, `discriminator_hidden_layers` | 1024, 3 | D MLP (BFM-Zero) |
| `actor_std`, `seq_length` | 0.05, 8 | Policy std, tracking window |

Open questions for cross-embodiment training:
- **B and D are unchanged, so they still need one flat state layout.** For
  several robots, use either an embodiment-agnostic state (for example base
  and keypoint features in a shared frame) or a padded layout. Only then do
  all robots share one z space. Replacing B with a GET-based set encoder is a
  natural next step, but here it is deliberately left as in BFM-Zero.
- **Inputs to the GET tokens.** BFM-Zero's actor sees `state`, `last_action`,
  and a proprioceptive history. Here those go into the GET tokens: joint
  history via `GetTokenizer`, and the last applied torque via the dynamics
  features. Base-level signals can go in as context tokens.
- **What is not included.**
  - The training agent (replay buffers, FB, critic, discriminator, and actor
    updates, and target networks) is not part of this module. The update rules
    are those of FB-CPR / BFM-Zero.
  - BFM-Zero's observation normalizer (`BatchNorm` with momentum 0.01 on each
    observation key) is also left out. Normalize the flat states before
    passing them to B and D.

The B, D, and truncated-normal code is adapted from BFM-Zero (Meta's code, CC
BY-NC 4.0); see `THIRD_PARTY_NOTICES.md`.

## Configuration reference

`GetConfig` fields:

| Field | Default | `GetConfig.get_zero` | Meaning |
|---|---|---|---|
| `input_size` | required | required | Raw token width, `GetTokenizer.token_size` |
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
| `rigid_body_feature_size`, `actuator_feature_size`, `dynamics_feature_size` | 25, 14, 7 | same | Physics feature widths |
| `physics_tokenizer_hidden_size` | `None` (= `hidden_size`) | same | Physics tokenizer MLP width |

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

## Repository layout

```text
DA-GET-BFM/
├── model/
│   ├── __init__.py            # public API
│   ├── urdf.py                # standard-library URDF parser, transmissions, transforms
│   ├── embodiment.py          # EmbodimentGraph (base body, link frames), JointFeatureConfig,
│   │                          # EmbodimentBatch, collate_embodiments, FK, mass matrix
│   ├── dynamics.py            # fixed/floating RNEA, M_full, forward dynamics, contacts,
│   │                          # rotor thrust, torque estimation
│   ├── physics_schema.py      # physics feature symbols, units, frames, scales, update tiers
│   ├── physics.py             # ActuatorSpec, feature extraction, motor labels, PhysicsFeatureCache
│   ├── tokenization_get.py    # GetTokenizer: GET-Zero token layout
│   ├── configuration_get.py   # GetConfig and the get_zero preset
│   ├── modeling_get.py        # GetEncoder, graph attention bias, Physics Cross Attention,
│   │                          # physics tokenizers, PhysicsMemory
│   ├── pretraining.py         # GetForPhysicsPretraining (HF PreTrainedModel), heads A–F
│   ├── configuration_bfm.py   # BfmConfig
│   ├── modeling_bfm.py        # BFM-Zero B, D, and GET-conditioned actor, F, critics; GetBfmModel
│   ├── transformer.py         # attention and cross-attention blocks
│   ├── robot_analysis.py      # infer floating base, feet, rotors, propellers from a URDF
│   ├── conversion/
│   │   ├── common.py          # table format, resampling, HTTP range reads of remote zips
│   │   ├── datasets.py        # GrandTour, MIPO, NeuroBEM, LocoMuJoCo, AMASS-G1, LAFAN1
│   │   ├── mapping.py         # match dataset columns to URDF joints and base signals
│   │   └── zarr_lite.py       # minimal zarr v2 reader
│   ├── training/
│   │   ├── sources.py         # synthetic, MuJoCo ground, and Hugging Face sources
│   │   ├── ground_sim.py      # MJCF from the graph, standing and pushes on the ground
│   │   ├── episodes.py        # per-frame features and labels, base reconstruction,
│   │   │                      # contact detection, embodiment records
│   │   ├── dataset.py         # data config, caching, dataset, collator
│   │   └── trainer.py         # Trainer that logs each task loss
│   └── README.md              # module-level technical notes
├── scripts/
│   ├── train_physics_get.py          # HF Trainer entry point
│   ├── make_data_config.py           # write a data config from a URDF and a dataset
│   ├── convert_dataset.py            # convert public datasets to the table format
│   ├── validate_torque_estimation.py # estimated vs measured torques on a real Go1
│   └── download_urdfs.py             # fetch Unitree and ANYmal D URDFs into assets/urdf
├── assets/robots/quadrotor_x.urdf    # X quadrotor for synthetic flight
├── configs/data/              # example data configs
├── tests/                     # unit tests (see below)
├── raw/                       # source material (the GET-Zero project page)
├── graphify-out/              # knowledge graph of the code and paper
├── THIRD_PARTY_NOTICES.md     # attribution for adapted code
├── pyproject.toml
└── uv.lock
```

## Testing and validation

```bash
uv sync --all-extras
uv run pytest
```

The suite has 84 tests (the MuJoCo and HDF5 tests are skipped without the
`sim` and `validation` extras):

- `test_embodiment.py`: URDF graphs, features, kinematics, mass matrix,
  batching, padding invariance.
- `test_get_encoder.py`, `test_transformer.py`: the encoder and its building
  blocks.
- `test_physics.py`:
  - RNEA against closed forms and identities;
  - transmissions and frozen joints;
  - feature schema and body-frame values;
  - actuator specs, the torque-speed envelope, and motor labels;
  - tiered cache refreshes;
  - Physics Cross Attention: cached memory, stale-memory detection, padding
    invariance.
- `test_floating.py`:
  - floating-base mass matrix structure, FD and RNEA consistency, free fall;
  - standing contact forces and torques;
  - surface contacts for bipeds, and dropping pulling contacts;
  - hover thrust;
  - harnessed floating and flight simulations whose torques are recovered
    from kinematics alone;
  - the determinate-torque projector removes exactly the internal-force
    directions, and the projected ID loss ignores them;
  - a Hub-style floating-base dataset with split state columns, through to a
    model forward pass.
- `test_ground.py` (MuJoCo):
  - the generated MJCF has our dynamics (mass matrix and bias forces);
  - a quadruped stands with exact labels and probe accelerations;
  - estimated torques differ from MuJoCo's almost only by internal contact
    forces (motion-determined error below 10%);
  - a biped is steadied by a small recorded spotter wrench.
- `test_conversion.py`:
  - every converter on small files in each dataset's real format;
  - the NeuroBEM thrust identification;
  - column-to-joint matching;
  - the URDF analysis of quadrupeds, bipeds, multirotors, and arms;
  - the config generator, end to end into a dataset.
- `test_bfm.py`:
  - shapes and BFM-Zero conventions of B, D, the actor, F, and the critics;
  - dict observations;
  - padding invariance and joint-permutation equivariance;
  - conditioning on z, latent sequences, and context tokens;
  - goal, reward, and tracking inference;
  - straight-through truncated-normal gradients;
  - an FB actor loss that reaches the GET encoder.
- `test_pretraining.py`:
  - both data sources and caching;
  - collation and masks;
  - all six losses, with a loss decrease;
  - ablations;
  - `save_pretrained`/`from_pretrained`;
  - a Trainer run with per-task logging.

Checks outside the test suite:

- **GET-Zero reference:** on all 633 LEAP Hand URDF variants shipped with
  GET-Zero, the adjacency matrix matches the reference implementation exactly.
  The joint origin and rotation features match the URDF to float32 precision.
- **Real robot models:** on Unitree Go1, H1, and G1, RNEA matches `M qdd`, and
  `FD(ID(qdd)) = qdd` to about 1e-13 for fixed and floating bases.
- **Synthetic floating and flight data:** for the harnessed Go1 and H1 and the
  flying quadrotor, torques estimated from kinematics alone match the applied
  torques to about 1e-15.
- **Converters on real downloads:** a MIPO trot sequence, a GrandTour mission
  (17 episodes of 20 s), NeuroBEM flights read remotely from the zip, and
  LocoMuJoCo, AMASS-G1, and LAFAN1 motions all convert and load.
- **MuJoCo ground data:** Go1, ANYmal D, G1, and H1 stand through 300-step
  episodes with pushes; the bipeds' spotter carries about 5–10% of the weight.
- **Real recorded data:**
  - Unitree G1 whole-body teleoperation loads as a 29-DoF floating-base robot.
    On 99.5% of frames the estimated contact forces explain the base wrench,
    and the knee torque estimates are about 5–6 N m.
  - Torque estimation on a real Go1 is compared with measured torques in the
    table above.

**Floating-base training run.** The data was 11 embodiments:
- 12 recorded G1 whole-body teleoperation episodes from the Hub (floating
  base, estimated torques);
- 4 randomized flying quadrotors;
- 4 randomized Go1 and 2 randomized H1 variants, each half fixed-base and
  half floating-base.

That is 36.7k training and 3.3k held-out evaluation samples. The `small`
preset (2.03M parameters) ran 600 steps, batch 64, with `rbd_weight` 0.1, on
CPU:

| | total | FK | FD | ID | motor | effect | RBD |
|---|---|---|---|---|---|---|---|
| train, first 100 steps | 8.145 | 2.066 | 1.403 | 0.223 | 0.458 | 3.919 | 0.740 |
| train, last 100 steps | 1.855 | 0.114 | 0.712 | 0.062 | 0.034 | 0.913 | 0.207 |
| eval, step 300 | 5.031 | 0.710 | 1.295 | 0.237 | 0.085 | 2.654 | 0.660 |
| eval, step 600 | 3.845 | 0.398 | 1.188 | 0.221 | 0.068 | 1.921 | 0.615 |

Every task loss decreases on held-out episodes, which shows that fixed,
floating, and flying robots train together end to end. The gap between train
and eval is expected for a run this short on this little data.

**Earlier ablation (fixed-base pipeline, before floating-base support).** This
used `configs/data/g1_hub_plus_synthetic.json` with 17 embodiments: 127k
training and 5,460 evaluation samples, the `small` preset, and 800 steps on
CPU. Evaluation loss with and without Physics Cross Attention:

| Eval loss | total | FK | FD | ID | motor | effect |
|---|---|---|---|---|---|---|
| with (2.01M params) | **1.992** | **0.129** | 0.325 | **0.143** | 0.0207 | **1.331** |
| without (1.68M params) | 2.155 | 0.179 | **0.307** | 0.181 | **0.0199** | 1.424 |

This was one seed and a short run with unequal parameter counts, so read it as
a pipeline check, not a conclusive ablation.

## Limitations

- **Tree-structured robots only.** Closed kinematic loops (parallel linkages)
  cannot be expressed in URDF and are rejected.
- **Xacro must be expanded first**, for example with
  `xacro robot.urdf.xacro > robot.urdf`. MJCF models must be converted.
- **Torque estimates are approximate.**
  - Contact forces are found by weighted least squares, without friction cones
    or contact dynamics, and are not unique with few feet down.
  - Rotor inertia and motor friction are not in URDFs.
  - On a real Go1 the error is 43–97% of the RMS torque (correlation 0.5–0.8),
    and 47–72% on ANYmal D.
  - Prefer measured torques when a dataset has them.
- **Simulated ground contact covers standing, not walking.** The MuJoCo source
  produces stance, crouching, weight shifting, partial foot lift, and push
  recovery, not gaits. Bipeds need the recorded spotter. Feet are spheres
  from `contact_geometry`, and only feet collide, so falls end the episode.
  Locomotion comes from recorded datasets.
- **Automatic configs are guesses.** The URDF analysis and column matching
  print their reasoning; check them, especially for unusual robots, and
  replace rule-of-thumb gains and propeller coefficients with measured ones.
- **Converted datasets inherit their sources' limits.** NeuroBEM publishes no
  geometry, so the rotor layout and drag coefficient are taken from the frame
  size when the fits are poor. MIPO streams are not synchronized. Retargeted
  motion sets are kinematic only.
- **Base state quality matters.** Base velocities from finite-differenced
  poses, accelerometer offsets from the base origin, and contact detection
  without base velocity all add error. Residual masking drops the worst
  frames.
- **Actuator defaults are assumptions.** URDF gives one effort and one velocity
  limit. `τ^cont`, `P^max`, gains, and propeller coefficients need real data
  for feasibility labels to be trustworthy.
- **The URDF must have inertials.** Joints that move massless bodies give a
  zero mass-matrix diagonal.

## Knowledge graph

`graphify-out/` holds a [graphify](https://github.com/safishamsi/graphify)
knowledge graph of the code and the GET-Zero paper page:

- `graph.html`: an interactive view.
- `GRAPH_REPORT.md`: the central nodes, communities, and surprising links.
- `graph.json`: the raw graph.

It predates the physics and floating-base modules; rebuild it after changes
with `/graphify . --update`.

## References

- Austin Patel and Shuran Song. *GET-Zero: Graph Embodiment Transformer for
  Zero-shot Embodiment Generalization.* ICRA 2025.
  [Project page](https://get-zero-paper.github.io/).
- Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea
  Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro
  Lazaric, Matteo Pirotta, and Guanya Shi. *BFM-Zero: A Promptable Behavioral
  Foundation Model for Humanoid Control Using Unsupervised Reinforcement
  Learning.* arXiv:2511.04131, 2025.
  [Project page](https://lecar-lab.github.io/BFM-Zero/),
  [code](https://github.com/LeCAR-Lab/BFM-Zero).
- Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi
  Kanervisto, Yingchen Xu, Alessandro Lazaric, and Matteo Pirotta. *Zero-Shot
  Whole-Body Humanoid Control via Behavioral Foundation Models.* ICLR 2025,
  arXiv:2504.11054 (FB-CPR, Meta Motivo).
- Ahmed Touati and Yann Ollivier. *Learning One Representation to Optimize All
  Rewards.* NeurIPS 2021 (forward-backward representations).
- Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. *Mastering
  Visual Continuous Control: Improved Data-Augmented Reinforcement Learning.*
  ICLR 2022 (truncated-normal policy).
- Scott Fujimoto, Herke van Hoof, and David Meger. *Addressing Function
  Approximation Error in Actor-Critic Methods.* ICML 2018 (clipped policy
  noise).
- Roy Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008 (RNEA,
  floating-base dynamics).
- Taeyoung Lee, Melvin Leok, and N. Harris McClamroch. *Geometric Tracking
  Control of a Quadrotor UAV on SE(3).* CDC 2010 (flight controller).
- Shuo Yang et al. *Multi-IMU Proprioceptive Odometry for Legged Robots.*
  IROS 2023 (MIPO data used for validation).
- Yi Zhou et al. *On the Continuity of Rotation Representations in Neural
  Networks.* CVPR 2019 (6D rotations).

BibTeX for the two main references:

```bibtex
@inproceedings{patel2025getzero,
  title     = {{GET-Zero}: Graph Embodiment Transformer for Zero-shot Embodiment Generalization},
  author    = {Patel, Austin and Song, Shuran},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}

@misc{li2025bfmzero,
  title         = {{BFM-Zero}: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning},
  author        = {Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
  year          = {2025},
  eprint        = {2511.04131},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2511.04131}
}
```
