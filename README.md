# DA-GET-BFM

A PyTorch encoder that turns any tree-structured robot, given its URDF, into
one token per degree of freedom (DoF) and encodes those tokens with a graph
transformer. It implements the Graph Embodiment Transformer (GET) from
[GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025) and
extends it to work across embodiments: dexterous hands, arms, quadrupeds,
bipeds, and humanoids can share one model and one training batch.

The encoder outputs one embedding per joint, `H_{t,i}`, plus a pooled
embodiment embedding. On top of GET-Zero it adds three physics mechanisms.
None of them are in the GET-Zero paper:

1. **Mass-matrix attention bias.** Each self-attention layer gets a bias from
   the normalized joint-space mass matrix `M(q)`. The graph says how joints are
   connected; `M(q)` says how strongly they are dynamically coupled right now.
2. **Physics Cross Attention.** Every observation token attends to rigid-body,
   actuator, and dynamics tokens of all joints, built from URDF inertials, motor
   specs, and analytic dynamics (`M_ii(q)`, `c(q, qd)`, `g(q)`, friction, last
   applied torque). Parts that do not change every control step are cached.
3. **Physics pretraining.** Six auxiliary heads train the encoder:
   - A: forward kinematics;
   - B: action-conditioned forward dynamics;
   - C: inverse dynamics;
   - D: motor feasibility and margins;
   - E: cross-embodiment action effect;
   - F: rigid-body consistency.

   Training uses the Hugging Face `Trainer`, on synthetic domain-randomized
   data and on recorded trajectories from the Hugging Face Hub.

## Contents

- [Why one token per DoF](#why-one-token-per-dof)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
  - [1. URDF to embodiment graph](#1-urdf-to-embodiment-graph)
  - [2. Per-joint hardware features](#2-per-joint-hardware-features)
  - [3. Token assembly](#3-token-assembly)
  - [4. Encoder and attention bias](#4-encoder-and-attention-bias)
  - [5. Physics Cross Attention](#5-physics-cross-attention)
  - [6. What is recomputed when](#6-what-is-recomputed-when)
  - [7. Batching different robots](#7-batching-different-robots)
- [Rigid-body dynamics](#rigid-body-dynamics)
- [Physics pretraining](#physics-pretraining)
- [Training with the Hugging Face Trainer](#training-with-the-hugging-face-trainer)
- [Recommended training datasets](#recommended-training-datasets)
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
quadruped, a 19-DoF humanoid, and a 16-DoF hand. Shorter robots are padded and
masked inside a batch. A per-joint output head can then emit one action per
token.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and needs Python 3.10 or
newer.

```bash
git clone git@github.com:apoman123/DA-GET-BFM.git
cd DA-GET-BFM
uv sync                                  # torch, numpy, transformers, datasets, accelerate, pytest
uv run pytest                            # runs the test suite
uv run python scripts/download_urdfs.py  # Unitree Go1, H1, G1 URDFs for the example configs
```

URDF parsing and the dynamics use only PyTorch and the Python standard library,
so no robotics packages are needed. `transformers`, `datasets`, and
`accelerate` are used by the training pipeline.

## Quick start

Encoder only, with Physics Cross Attention:

```python
import torch

from model import (
    EmbodimentGraph, GetConfig, GetEncoder, GetTokenizer,
    JointFeatureConfig, collate_embodiments,
)
from model.physics import PhysicsFeatureCache, actuator_specs

# 1. One graph per robot, built from URDF.
features = JointFeatureConfig.get_zero()               # GET-Zero hardware features
graphs = [
    EmbodimentGraph.from_urdf("assets/urdf/go1.urdf", features),       # quadruped, 12 DoF
    EmbodimentGraph.from_urdf("assets/urdf/h1.urdf", features),        # humanoid, 19 DoF
]
batch = collate_embodiments(graphs)                     # pads to 19 joints

# 2. Physics features: static ones are computed here once.
specs = [actuator_specs(graph) for graph in graphs]     # URDF limits + your overrides
physics = PhysicsFeatureCache(graphs, specs, configuration_interval=4)

# 3. Observations with history (oldest first) -> one token per joint.
H, J = 3, batch.joint_mask.shape[1]
q = torch.zeros(len(graphs), H, J)
qd = torch.zeros(len(graphs), H, J)
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
dynamics = physics.update(q[:, -1], qd[:, -1])       # M_ii, c, g, friction, last torque
hidden = encoder(
    tokens,
    mass_matrix=batch.mass_matrix(q[:, -1].double()).float(),
    dynamics_features=dynamics,
    physics_memory=memory,
    **batch.encoder_inputs(),                        # parents, mask, edges, precomputed distances
)                                                    # [2, 19, 512]; padded joints are zero
```

Without URDF you can call the encoder directly. Pass `parent_indices`
(`[batch, joints]`, `-1` for roots and padding), an optional `joint_mask`
(`True` for valid joints), an optional `adjacency_matrix`, and `mass_matrix`
when mass encoding is enabled.

Pretraining in one command (details below):

```bash
uv run python scripts/train_physics_get.py \
    --data_config configs/data/synthetic_unitree.json \
    --output_dir runs/synthetic --max_steps 2000 --per_device_train_batch_size 64
```

## How it works

```text
URDF ──parse_urdf──▶ UrdfRobot ──EmbodimentGraph.from_urdf──▶ EmbodimentGraph
                                                               │  joints, parents, edges,
                                                               │  hardware features, inertia, limits
           actuator_specs / rigid_body_features ◀──────────────┤
                     several graphs ──collate_embodiments──▶ EmbodimentBatch
                                                               │
 observations + joint_features ──GetTokenizer──▶ observation tokens [B, J, F]
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
| `revolute`, `continuous`, `prismatic` joints | One node and one token each. |
| `fixed` joints | Collapsed. Their child links are merged into the parent body: poses are composed and inertia is combined with the parallel-axis theorem. Feet, IMU frames, sensor mounts, and flanges therefore add no tokens. |
| `mimic` joints | Not independent DoFs, so frozen at their offset by default. `include_mimic_joints=True` keeps them as tokens. |
| `floating` and `planar` joints | Treated as rigid mounts by default (`multi_dof_joints="skip"`). With `"expand"`, they become chains of 1-DoF virtual joints (6 for floating, 3 for planar), so each DoF still gets one token. These virtual joints are unactuated (`passive`). |
| `frozen_joints={name: q}` | 1-DoF joints held rigid at a position and merged like fixed links. Use this when a dataset records only part of a robot, for example the arms of a humanoid whose legs are locked. |
| Joint order | Depth-first URDF order, the order used by MuJoCo and Pinocchio. Pass `joint_order=[...]` to match a simulator's or dataset's ordering. |

Edges:

- **Directed parent to child edges** follow the kinematic tree through any
  number of fixed joints.
- **Undirected sibling edges** join joints that share a parent body. This is
  GET-Zero's rule for "joints at the start of independent serial chains." For a
  hand these are the finger roots on the palm. For a quadruped they are the four
  hips on the trunk. For a humanoid they are the hips and waist on the pelvis,
  and the shoulders on the torso. Turn them off with `sibling_edges=False`.

From these edges the graph precomputes three integer distance matrices: the
shortest path over undirected edges (SPD), the number of steps to each
ancestor, and the number of steps to each descendant.

### 2. Per-joint hardware features

`JointFeatureConfig` chooses the fixed features `o_fl,j` inside each
observation token. All poses are relative to the parent joint.

| Preset | Size | Features |
|---|---|---|
| `JointFeatureConfig()` | 6 | Joint origin `x, y, z` and rotation `roll, pitch, yaw`. This matches GET-Zero. |
| `JointFeatureConfig.cross_embodiment()` | 31 | Origin, rotation as continuous 6D, joint axis, one-hot joint type, position limits, effort and velocity limits, damping and friction, and the mass, center of mass, and inertia tensor of the body the joint moves. |

With Physics Cross Attention enabled, the full physical description goes to the
physics tokens instead (section 5), so the GET-Zero preset is usually enough in
the observation tokens.

### 3. Token assembly

`GetTokenizer` builds token `j` exactly as GET-Zero does:

```text
T_j = [ o_vg^{t-H+1..t},  o_fg,  o_vl,j^{t-H+1..t},  o_fl,j ]
```

| Part | Shape in | Meaning | Example |
|---|---|---|---|
| `o_vg` | `[B, H, F_vg]` | Variable global observations with history, copied into every token | Task phase; base angular velocity, gravity, velocity command |
| `o_fg` | `[B, F_fg]` | Fixed global features | Task ID |
| `o_vl,j` | `[B, H, J, F_vl]` | Variable per-joint observations with history | Normalized joint position, PD target |
| `o_fl,j` | `[B, J, F_fl]` | Fixed per-joint hardware features | `EmbodimentGraph.joint_features` |

History is flattened feature-major, as in the GET-Zero code: every timestep of
the first feature, then every timestep of the second, and so on. Pass history
oldest first. Padded joints get all-zero tokens.

### 4. Encoder and attention bias

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
  step.

Each bias can be turned off: `use_spatial_encoding`,
`use_parent_child_encoding`, `use_mass_encoding`.

### 5. Physics Cross Attention

Three feature groups describe every joint `i`. All values are SI, and every
spatial quantity names its frame (`model/physics_schema.py` has the full
schema):

- `B_i` is the body frame of joint `i`. Its origin is the joint origin, its axes
  are the joint frame's, and it moves with the body that joint `i` drives (the
  child link after fixed links are merged).
- `B_p(i)` is the parent joint's body frame, or the root link frame.
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

**Actuator (11), from `ActuatorSpec`, static per robot**

| Symbol | Unit | Default when the URDF lacks it |
|---|---|---|
| `τ_i^peak` | N m | URDF `effort` |
| `τ_i^cont` | N m | `0.5 τ^peak` |
| `ω_i^max` | rad/s | URDF `velocity` |
| `P_i^max` | W | `0.5 τ^peak ω^max` (torque falls off above half the no-load speed) |
| `N_i` (gear ratio) | 1 | URDF `<transmission>` reduction, else 1 |
| `K_p,i`, `K_d,i` | N m/rad, N m s/rad | 0, or `actuator_defaults` |
| `mode_i` (one-hot torque, position, velocity, passive) | 1 | the data source's action type |

The defaults are assumptions. Pass real motor data through
`actuator_specs(graph, overrides={"*_knee_joint": {"continuous_torque": 23.7}})`
or, in a data config, through `actuator_overrides` (exact names or glob
patterns).

**Dynamics (7), from the state at time `t`**

| Symbol | Unit | Depends on |
|---|---|---|
| `M_ii(q_t)` | kg m² | `q` |
| `c_i(q_t, qd_t)` Coriolis and centrifugal | N m | `q`, `qd` |
| `g_i(q_t)` gravity | N m | `q`, gravity direction |
| `τ̂_f,i` friction estimate `d qd + μ tanh(qd / 0.01)` | N m | `qd` |
| `τ_{i,t-1}^applied` | N m | measurement |
| `\|g_i\| / (τ_i^cont + ε)` gravity load | 1 | `q` |
| flag: `τ_{t-1}^applied` was measured | 1 | measurement |

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

### 6. What is recomputed when

| Tier | Features | Changes when | At training time | At control time |
|---|---|---|---|---|
| embodiment | all rigid-body and actuator features | the robot or its randomized parameters change | once per robot, when the dataset is built | once per robot, and their per-layer keys and values are cached in a `PhysicsMemory` |
| configuration | `M_ii(q)`, `g(q)`, gravity load | `q` or the gravity direction | once per frame, cached on disk | `PhysicsFeatureCache`: every `configuration_interval` steps, or sooner if any joint moves more than `configuration_tolerance` or gravity turns |
| velocity | `c(q, qd)`, `τ̂_f` | `qd` (and `q`) | once per frame, cached on disk | every `velocity_interval` steps, if `qd` moves more than `velocity_tolerance`, or with the configuration tier |
| measurement | `τ_{t-1}^applied` | every step | per frame | every step |

`encoder.build_physics_memory(rigid_body, actuator, joint_mask)` precomputes the
rigid-body and actuator tokens and their keys and values for every physics
layer. After that, a control step only embeds and projects the `J` dynamics
tokens. The memory records the versions of the parameters it came from. After
an optimizer step, or any other in-place weight change, the encoder refuses a
stale memory instead of silently using it.

### 7. Batching different robots

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
per-robot helpers to a padded batch. All graphs in a batch must use the same
feature configuration.

## Rigid-body dynamics

`model/dynamics.py` implements the recursive Newton-Euler algorithm (RNEA),
batched over any leading dimensions, for the fixed-base tree:

```text
τ = M(q) qdd + c(q, qd) + g(q) + τ_f(qd)
```

| Function | Returns |
|---|---|
| `EmbodimentGraph.forward_kinematics(q)` | `[..., J, 4, 4]` joint frames in the root frame (the FK target) |
| `EmbodimentGraph.mass_matrix(q)` | `[..., J, J]` `M(q)` |
| `inverse_dynamics(graph, q, qd, qdd, gravity)` | required efforts, including friction |
| `forward_dynamics(graph, q, qd, τ, gravity)` | `qdd = M⁻¹ (τ − c − g − τ_f)` |
| `gravity_efforts`, `coriolis_efforts`, `friction_efforts` | `g(q)`, `c(q, qd)`, `τ_f(qd)` |
| `frame_velocities(graph, q, qd)` | linear and angular velocity of every joint frame |

`gravity` is the gravity vector in the root frame. For a floating-base robot,
pass the measured gravity direction (projected gravity times 9.81) so `g(q)`
follows the base orientation. The base itself is held fixed, so coupling
between the base and the joints is not modelled.

The tests check every term against closed forms or identities:
- the double pendulum's `g` and `c`;
- `RNEA(q, 0, qdd) = M qdd`;
- `qdᵀ c = ½ qdᵀ Ṁ qd`;
- `g = ∇V`;
- `FD(ID(qdd)) = qdd`;
- frame velocities against finite-differenced FK.

## Physics pretraining

`GetForPhysicsPretraining` (`model/pretraining.py`) is a Hugging Face
`PreTrainedModel`, so `save_pretrained`, `from_pretrained`, and `push_to_hub`
all work. It wraps the encoder with six heads. `N_e` is the number of valid
joints of a sample's robot. Every per-joint loss is averaged per robot first,
so a 29-DoF humanoid does not outweigh a 7-DoF arm.

| Task | Head input | Target | Loss |
|---|---|---|---|
| **A. Forward kinematics** | `H_i` | `p_i / L_e` (`L_e` = robot size) and `R_i ∈ SO(3)` of every joint frame in the root frame | `1/N_e Σ λ_p ‖p̂−p‖² + λ_R d_SO(3)(R̂, R)²`; `fk_orientation=False` gives the position-only first pass |
| **B. Forward dynamics** | `[H_i; E_τ(τ_i^applied)]` passed through `mixing_layers` of graph action mixing | `qdd_t` | `1/N_e Σ ((qdd̂ − qdd) / σ_i^a)²` |
| **C. Inverse dynamics** | `[H_i; E(qdd_i^desired)]` with graph mixing (the full desired acceleration vector, because `M qdd` couples joints) | `τ^required = M qdd + c + g + τ_f` | `1/N_e Σ ((τ̂ − τ) / (τ_i^cont + ε))²` |
| **D. Motor feasibility** | `[H_i; E(a_i^proposal)]` | peak and continuous feasibility flags; margins for peak torque `1 − \|τ\|/τ^max(qd)`, continuous torque, power, velocity, and joint limit | `λ_f BCE + λ_m Huber` |
| **E. Action effect** | `[H_i; E(a_i)]` with graph mixing | at end-effector frames: `Δp`, `R_{t+1}`, `v_{t+1}`, `ω_{t+1}` | `λ_p‖Δp̂−Δp‖² + λ_R d_SO(3)² + λ_v‖v̂−v‖² + λ_ω‖ω̂−ω‖²` |
| **F. RBD consistency** (optional) | the FD prediction | `r = M qdd̂ + c + g + τ_f − τ^applied` | `1/N_e ‖W_e r‖²` with `W_e = 1/(τ^cont + ε)` |

Notes on the design:

- **B: per-joint inputs are not enough.** Seeing only its own torque, a joint
  cannot represent `M⁻¹`'s coupling across joints. The FD, ID, and effect heads
  therefore start with `mixing_layers` (default 1) of graph self-attention over
  `[H_i; E(x_i)]`. Set it to 0 for the purely per-joint baseline.
- **D: requested torques and limits.** The requested effort comes from the
  proposal through the actuator mode: torque mode passes it through, position
  mode is `K_p (a − q) − K_d qd`, and velocity mode is `K_d (a − qd)`. The
  torque-speed envelope is `τ^max(qd) = min(τ^peak, P^max/|qd|)` below
  `ω^max`, and zero above it. "Peak feasible" means executable now;
  "continuous feasible" also respects `τ^cont`. A short motor boost is
  therefore not mistaken for a sustainable command. The predictor does not
  replace hard limits in the motor interface.
- **E: normalization.** `Δp` and `v` are divided by the robot size `L_e`. Each
  of `Δp`, `v`, and `ω` is then divided by its RMS over that robot's data, so
  translation and rotation terms have comparable size. Rotations are
  supervised with the geodesic distance on SO(3). The loss covers end-effector
  frames only (leaf joints by default, or `end_effectors` in the data config).
  It is one-step supervision; long-horizon objectives belong elsewhere.
- **F: off by default.** With analytic labels it largely duplicates `L_FD`, so
  its weight is 0 by default. Treat it as an ablation, not as evidence that a
  lower residual gives better returns.

Weights: `--fk_weight`, `--fd_weight`, `--id_weight`, `--motor_weight`,
`--effect_weight`, `--rbd_weight`, plus the `lambda_*` fields of
`GetPhysicsPretrainingConfig`.

## Training with the Hugging Face Trainer

```bash
uv run python scripts/download_urdfs.py

# 1. Synthetic, domain-randomized Go1, H1, and G1 (no download besides URDFs)
uv run python scripts/train_physics_get.py \
    --data_config configs/data/synthetic_unitree.json \
    --output_dir runs/synthetic --max_steps 2000 \
    --per_device_train_batch_size 64 --learning_rate 5e-4 --warmup_steps 100 \
    --eval_strategy steps --eval_steps 500 --logging_steps 50

# 2. Recorded Unitree G1 teleoperation data from the Hub, plus synthetic G1 arms and Go1
uv run python scripts/train_physics_get.py \
    --data_config configs/data/g1_hub_plus_synthetic.json --output_dir runs/g1 \
    --preset get_zero --max_steps 20000 --bf16 --report_to tensorboard
```

Every `TrainingArguments` flag works, including multi-GPU runs through
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
model = GetForPhysicsPretraining.from_pretrained("runs/g1")
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
  "sources": [ { "type": "synthetic", ... }, { "type": "huggingface", ... } ]
}
```

**`"type": "synthetic"`** simulates a URDF robot with the analytic dynamics.
Each of `randomized_embodiments` variants perturbs:
- link masses and inertias;
- link lengths (inertia is scaled consistently);
- damping and friction;
- torque and speed limits.

Each episode also tilts gravity by up to `gravity_tilt_degrees`. A PD
controller with gravity feed-forward tracks random targets, and its command is
clipped to the torque-speed envelope before it is applied. Labels therefore
always use the executed torque, never an unexecuted proposal. Each state also
gets `probe_torques` extra torques (log-uniform from 2% to 200% of peak, random
sign), each simulated one step. This tests action conditioning at a fixed pose
and gives the motor head infeasible proposals.

**`"type": "huggingface"`** reads any Hub dataset (`repo_id`) or local folder
(`path`) in the LeRobot parquet layout, or any table with one vector column per
frame:

| Key | Meaning |
|---|---|
| `urdf`, `joint_names` | URDF joint of each recorded state entry, in order. Unlisted joints are frozen (`frozen_joints` gives their positions, default 0). |
| `state_key`, `state_indices` | Position column and which entries to use (`observation.state`). |
| `velocity_key`, `torque_key` | Optional columns. Without velocities, positions are finite-differenced (`smoothing_window` smooths them first). Without torques, the applied torque is the inverse-dynamics estimate, and the dynamics features mark `τ_{t-1}` as unknown. |
| `action_key`, `action_indices`, `action_type` | Commands (`position`, `velocity`, or `torque`). Set `actuator_defaults.kp`/`kd` for position control so requested torques are meaningful. |
| `state_scale`, `state_offset` | `q_urdf = scale · q + offset`, for sign, unit (degrees), or zero-offset differences. |
| `fps`, `episode_key`, `frame_key`, `max_episodes`, `gravity`, `end_effectors` | Timing, grouping, size, gravity in the root frame, task E frames. |

Processed features and labels are cached in `.cache/physics_data/<hash>.pt`.
The hash covers the config and the URDF contents, so the analytic dynamics run
once per dataset rather than once per epoch. Episodes, not frames, are split
between train and eval.

## Recommended training datasets

The pretraining tasks need joint-level trajectories of robots with a known
URDF. The datasets below are on the Hugging Face Hub in LeRobot format, so the
`huggingface` source reads them directly. The details (size, rate, recorded
columns) were checked against each dataset's `meta/info.json` on 2026-11-07.
**None of these recorded datasets contains joint torques.** Torques are
therefore the inverse-dynamics estimate. For tasks B, C, and F, mix them with
the synthetic source, which has exact torques and probes.

| Dataset | Robot and DoF | Size | What is recorded | URDF | How to use |
|---|---|---|---|---|---|
| Synthetic (built in) | any URDF | unlimited | exact `q, qd, qdd, τ`, probe torques, randomized physics | yours | `configs/data/synthetic_unitree.json`. The only source with exact torque labels and infeasible proposals. |
| [`unitreerobotics/G1_Dex3_ToastedBread_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) and the other `unitreerobotics/G1_Dex3_*` tasks (for example [`G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)) | Unitree G1: 14 arm joints + 14 Dex3 hand joints | 418 episodes / 352k frames (ToastedBread); 301 / 281k (BlockStacking); 30 fps | named joint positions; position targets | `g1_29dof.urdf` from [unitree_ros](https://github.com/unitreerobotics/unitree_ros) (arms) | `configs/data/g1_hub_plus_synthetic.json` (arms; legs and waist frozen). Add the Dex3 hand URDF to also use the hand joints. |
| [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | Franka Panda, 7 | 95,658 episodes / 27.6M frames, 15 fps | `observation.state.joint_position`; `action.joint_position` and `action.joint_velocity` | `franka_description` ([franka_ros](https://github.com/frankaemika/franka_ros)) | `state_key: "observation.state.joint_position"`, `action_key: "action.joint_position"`. Use `max_episodes`, since the full set is very large. |
| [`lerobot/aloha_sim_insertion_human`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) (and other `lerobot/aloha_*`) | 2 × ViperX 300: 6 joints + gripper each | 50 episodes / 25k frames, 50 fps | named joint positions, position actions | `vx300s` from [interbotix_ros_manipulators](https://github.com/Interbotix/interbotix_ros_manipulators) | Map 12 arm joints, skip the grippers. Build one URDF holding both arms on one base, or two sources with one arm each. |
| [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | SO-100 arm: 5 joints + gripper | 50 episodes / 19.6k frames, 30 fps | named joint positions | [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Check the dataset's position units and calibration; convert with `state_scale`/`state_offset`. |
| [`robfiras/loco-mujoco-datasets`](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) | humanoids: Unitree H1, H1 v2, G1, Atlas, Talos, Apollo, Booster T1, Fourier GR1T2, ToddlerBot | default mocap (19 motions per robot) and LAFAN1 retargets (40 per robot) | joint positions and velocities, `.npz` | the LocoMuJoCo robot models | Not parquet: convert each motion to a table with `episode_index`, `frame_index`, `observation.state`, and pass `path`. The license is **CC BY-NC-ND 4.0** (non-commercial). Retargets are kinematic, so some motions exceed actuator limits. |

Several Open X-Embodiment ports on the Hub (for example
`lerobot/stanford_kuka_multimodal_dataset`, `lerobot/berkeley_autolab_ur5`,
`lerobot/toto`, `lerobot/fmb`) also record joint-sized state vectors. They name
their motors only by index, and some contain end-effector poses or grippers
rather than joint angles. Check each against its paper before mapping it to a
URDF. The `lerobot/unitreeh1_*` datasets have the same problem. A good mix for
cross-embodiment pretraining:
- synthetic data for every robot you care about (exact dynamics, probes, limits);
- recorded data for realistic motion statistics;
- held-out robots, or held-out randomized variants, for evaluation.

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
| `rigid_body_feature_size`, `actuator_feature_size`, `dynamics_feature_size` | 25, 11, 7 | same | Physics feature widths |
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

## Repository layout

```text
DA-GET-BFM/
├── model/
│   ├── __init__.py            # public API
│   ├── urdf.py                # standard-library URDF parser, transmissions, transforms
│   ├── embodiment.py          # EmbodimentGraph, JointFeatureConfig, EmbodimentBatch,
│   │                          # collate_embodiments, FK, mass matrix
│   ├── dynamics.py            # RNEA, inverse and forward dynamics, c, g, friction, velocities
│   ├── physics_schema.py      # physics feature symbols, units, frames, scales, update tiers
│   ├── physics.py             # ActuatorSpec, feature extraction, motor labels, PhysicsFeatureCache
│   ├── tokenization_get.py    # GetTokenizer: GET-Zero token layout
│   ├── configuration_get.py   # GetConfig and the get_zero preset
│   ├── modeling_get.py        # GetEncoder, graph attention bias, Physics Cross Attention,
│   │                          # physics tokenizers, PhysicsMemory
│   ├── pretraining.py         # GetForPhysicsPretraining (HF PreTrainedModel), heads A–F
│   ├── transformer.py         # attention and cross-attention blocks
│   ├── training/
│   │   ├── sources.py         # synthetic domain-randomized simulator, Hugging Face loader
│   │   ├── episodes.py        # per-frame features and labels, embodiment records
│   │   ├── dataset.py         # data config, caching, dataset, collator
│   │   └── trainer.py         # Trainer that logs each task loss
│   └── README.md              # module-level technical notes
├── scripts/
│   ├── train_physics_get.py   # HF Trainer entry point
│   └── download_urdfs.py      # fetch Unitree URDFs into assets/urdf
├── configs/data/              # example data configs
├── tests/                     # unit tests (see below)
├── raw/                       # source material (the GET-Zero project page)
├── graphify-out/              # knowledge graph of the code and paper
├── pyproject.toml
└── uv.lock
```

## Testing and validation

```bash
uv run pytest
```

The suite has 47 tests:

- `test_embodiment.py`: URDF graphs, features, kinematics, mass matrix,
  batching, padding invariance.
- `test_get_encoder.py`, `test_transformer.py`: the encoder and its building
  blocks.
- `test_physics.py`:
  - RNEA against closed forms and identities;
  - transmissions and frozen joints;
  - feature schema and body-frame values;
  - actuator specs and overrides, the torque-speed envelope, and motor labels;
  - tiered cache refreshes;
  - Physics Cross Attention, including cached memory against recomputation,
    stale-memory detection, padding invariance, and input validation.
- `test_pretraining.py`:
  - both data sources (synthetic, and a local LeRobot-style parquet dataset),
    and caching;
  - collation and masks;
  - all six losses, and that a few optimizer steps reduce the loss;
  - position-only FK;
  - `save_pretrained`/`from_pretrained`;
  - a Trainer run with per-task logging.

Checks outside the test suite:

- **Against the GET-Zero reference code:** on all 633 LEAP Hand URDF variants
  shipped with GET-Zero (5 to 16 DoF), the adjacency matrix matches the
  reference implementation exactly. The joint origin and rotation features match
  the URDF to float32 precision.
- **On real robot URDFs:** Unitree Go1 (12 DoF), H1 (19 DoF), and G1 (29 DoF)
  all parse and give positive-definite mass matrices. On all three, RNEA
  matches `M qdd` and `FD(ID(qdd)) = qdd` to about 1e-13.
- **On real recorded data:** `unitreerobotics/G1_Dex3_ToastedBread_Dataset`
  loads through the `huggingface` source. Its 14 arm joints map to the G1 URDF
  by name, and the shoulder gravity torques come out around 4 N m.


## Limitations

- **Tree-structured robots only.** Closed kinematic loops (parallel linkages)
  cannot be expressed in URDF and are rejected.
- **Xacro must be expanded first**, for example with
  `xacro robot.urdf.xacro > robot.urdf`.
- **Fixed-base dynamics.** A floating base is a rigid mount (or a chain of
  passive virtual joints). `M`, `c`, and `g` are the actuated-joint dynamics
  with the base held still. Contact forces and base coupling are not modelled,
  so for legged robots in contact the inverse-dynamics torques of recorded data
  are approximations. Start with fixed-base, fully actuated robots (`B = I`).
- **Actuator defaults are assumptions.** URDF gives one effort and one velocity
  limit. `τ^cont`, `P^max`, and the gains need real motor data for feasibility
  labels to be trustworthy.
- **Recorded datasets rarely have torques.** Inverse-dynamics estimates make FD
  and ID labels consistent with the analytic model, not with the real robot.
  Set `allow_estimated_torque_for_fd: false` to train FD only on measured or
  simulated torques.
- **Finite differences amplify noise.** Prefer datasets with velocities, use
  `smoothing_window`, and expect noisier acceleration labels at low frame
  rates.
- **The URDF must have inertials.** Joints that move massless bodies give a
  zero mass-matrix diagonal.

## Knowledge graph

`graphify-out/` holds a [graphify](https://github.com/safishamsi/graphify)
knowledge graph of the code and the GET-Zero paper page:

- `graph.html`: an interactive view.
- `GRAPH_REPORT.md`: the central nodes, communities, and surprising links.
- `graph.json`: the raw graph.

Rebuild it after changes with `/graphify . --update`.

## References

- Austin Patel and Shuran Song. *GET-Zero: Graph Embodiment Transformer for
  Zero-shot Embodiment Generalization.* ICRA 2025.
  [Project page](https://get-zero-paper.github.io/).
- Roy Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008 (RNEA).
- Yi Zhou et al. *On the Continuity of Rotation Representations in Neural
  Networks.* CVPR 2019 (6D rotations).
