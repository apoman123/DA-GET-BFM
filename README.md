# DA-GET-BFM

A PyTorch encoder that turns any tree-structured robot, given its URDF, into
one token per degree of freedom (DoF) and encodes those tokens with a graph
transformer. It implements the Graph Embodiment Transformer (GET) from
[GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025) and
extends it to work across embodiments: dexterous hands, arms, quadrupeds,
bipeds, and humanoids can share one model and one training batch.

On top of GET-Zero's structural attention bias, each attention layer also
receives a bias from the robot's joint-space mass matrix `M(q)`, computed from
the URDF inertials at the current configuration. The structural bias says how
the joints are connected. The mass bias says how strongly they are coupled
dynamically right now.

The repository provides the encoder only. It has no policy head, value head,
self-modeling head, training loop, or simulator integration. Those are meant to
be built on top of it.

## Contents

- [Why one token per DoF](#why-one-token-per-dof)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
  - [1. URDF to embodiment graph](#1-urdf-to-embodiment-graph)
  - [2. Per-joint hardware features](#2-per-joint-hardware-features)
  - [3. Token assembly](#3-token-assembly)
  - [4. Encoder and attention bias](#4-encoder-and-attention-bias)
  - [5. Batching different robots](#5-batching-different-robots)
- [Kinematics and dynamics helpers](#kinematics-and-dynamics-helpers)
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
uv sync          # creates .venv with torch, numpy, and pytest
uv run pytest    # runs the test suite
```

Runtime dependencies are `torch>=2.2` and `numpy`. URDF parsing uses only the
Python standard library, so no robotics packages are needed.

## Quick start

```python
import torch

from model import (
    EmbodimentGraph, GetConfig, GetEncoder, GetTokenizer,
    JointFeatureConfig, collate_embodiments,
)

# 1. One graph per robot, built from URDF.
features = JointFeatureConfig.cross_embodiment()     # or JointFeatureConfig() for the paper's 6 features
graphs = [
    EmbodimentGraph.from_urdf("go1.urdf", features),       # quadruped, 12 DoF
    EmbodimentGraph.from_urdf("h1.urdf", features),        # humanoid, 19 DoF
    EmbodimentGraph.from_urdf("leap_hand.urdf", features), # hand, 16 DoF
]
batch = collate_embodiments(graphs)                   # pads to 19 joints

# 2. Observations with history (oldest first).
H = 6
J = batch.joint_mask.shape[1]
q = torch.zeros(len(graphs), H, J)                    # joint positions
q_target = torch.zeros_like(q)                        # previous PD targets
base_obs = torch.zeros(len(graphs), H, 6)             # e.g. angular velocity + gravity

# 3. One token per joint.
tokenizer = GetTokenizer(
    fixed_local_size=features.size,     # o_fl: URDF hardware features
    variable_local_size=2,              # o_vl: normalized q and PD target
    variable_global_size=6,             # o_vg: shared global observations
    history_size=H,
)
tokens = tokenizer(
    batch.joint_features,
    batch.joint_mask,
    variable_local=torch.stack(
        (batch.normalize_joint_positions(q), batch.normalize_joint_positions(q_target)),
        dim=-1,
    ),
    variable_global=base_obs,
)                                       # [3, 19, tokenizer.token_size]

# 4. Encode.
encoder = GetEncoder(GetConfig.get_zero(input_size=tokenizer.token_size))
hidden = encoder(
    tokens,
    mass_matrix=batch.mass_matrix(q[:, -1]),   # M(q) at the latest configuration
    **batch.encoder_inputs(),                  # parents, mask, edges, precomputed distances
)                                              # [3, 19, 512]; padded joints are zero
```

Without URDF you can call the encoder directly. Pass `parent_indices`
(`[batch, joints]`, `-1` for roots and padding), an optional `joint_mask`
(`True` for valid joints), an optional `adjacency_matrix`, and `mass_matrix`
when mass encoding is enabled.

## How it works

```text
URDF ──parse_urdf──▶ UrdfRobot ──EmbodimentGraph.from_urdf──▶ EmbodimentGraph
                                                               │  joints, parents, edges,
                                                               │  hardware features, inertia
                     several graphs ──collate_embodiments──▶ EmbodimentBatch
                                                               │
 observations + joint_features ──GetTokenizer──▶ tokens [B, J, F]
                                                               │
 tokens, M(q), graph inputs ──GetEncoder──▶ joint embeddings [B, J, hidden]
```

### 1. URDF to embodiment graph

`model/urdf.py` parses URDF from a path or an XML string and checks that the
model is a single tree: one root link, no link with two parents, no cycles, no
missing links, and no duplicate names. `EmbodimentGraph.from_urdf` in
`model/embodiment.py` then builds the joint graph:

| URDF element | Treatment |
|---|---|
| `revolute`, `continuous`, `prismatic` joints | One node and one token each. |
| `fixed` joints | Collapsed. Their child links are merged into the parent body: poses are composed and inertia is combined with the parallel-axis theorem. Feet, IMU frames, sensor mounts, and flanges therefore add no tokens. |
| `mimic` joints | Not independent DoFs, so frozen at their offset by default. `include_mimic_joints=True` keeps them as tokens. |
| `floating` and `planar` joints | Treated as rigid mounts by default (`multi_dof_joints="skip"`). With `"expand"`, they become chains of 1-DoF virtual joints (6 for floating, 3 for planar), so each DoF still gets one token. |
| Joint order | Depth-first URDF order, the order used by MuJoCo and Pinocchio. Pass `joint_order=[...]` to match a simulator's ordering. |

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

`JointFeatureConfig` chooses the fixed features `o_fl,j` that describe each
joint's hardware. All poses are relative to the parent joint.

| Preset | Size | Features |
|---|---|---|
| `JointFeatureConfig()` | 6 | Joint origin `x, y, z` and rotation `roll, pitch, yaw`. This matches GET-Zero. |
| `JointFeatureConfig.cross_embodiment()` | 31 | Origin, rotation as continuous 6D, joint axis, one-hot joint type, position limits, effort and velocity limits, damping and friction, and the mass, center of mass, and inertia tensor of the body the joint moves. |

The paper's robots are LEAP Hand variants whose joints share the same axes, so
the 6-feature preset is enough there. Across robot families the axes, ranges,
and masses differ, and the encoder needs them to tell the robots apart. Use
`cross_embodiment()` for mixed training. Individual flags (`joint_axis`,
`joint_limits`, `child_inertial`, and so on) can be set independently.

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
3. `num_hidden_layers` pre-norm transformer layers. Each has biased
   self-attention, optional cross-attention to external context tokens, and a
   GELU feed-forward block.
4. A final LayerNorm. Padded joints are set to zero.

Each layer adds a learned per-head bias to the attention logits:

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

Cross-attention to context tokens (for example vision or language features) runs
after self-attention in the layers listed in `cross_attention_layers`, with
width `context_size` and optional `context_mask`.

### 5. Batching different robots

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

Graph distances depend only on the robot, so they are computed once when the
graph is built instead of inside every forward pass.

## Kinematics and dynamics helpers

Each `EmbodimentGraph` keeps the kinematic tree and the merged body inertials,
so it can compute:

| Method | Output | Use |
|---|---|---|
| `forward_kinematics(q)` | `[..., J, 4, 4]` pose of each joint in the root frame | The GET-Zero self-modeling target |
| `mass_matrix(q)` | `[..., J, J]` joint-space inertia `M(q)` | Input to the mass attention bias |
| `normalize_joint_positions(q)` | `q` scaled to [-1, 1] by the joint limits; continuous joints wrapped by pi | Joint observations in tokens |

The mass matrix is the sum of `m Jv^T Jv + Jw^T I Jw` over the bodies the
joints move. For a floating-base robot with the base treated as a mount, it
equals the actuated-joint block of the full floating-base mass matrix. A joint
whose moving body has no mass gives a zero diagonal entry, which the encoder
rejects.

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
| `cross_attention_layers` | `()` | `()` | Layers with cross-attention |
| `mass_epsilon` | `1e-8` | `1e-8` | Stabilizer in the mass normalization |
| `use_spatial_encoding` | `True` | `True` | SPD bias |
| `use_parent_child_encoding` | `True` | `True` | Ancestor and descendant bias |
| `use_mass_encoding` | `True` | `True` | `M(q)` bias |
| `use_input_layer_norm` | `True` | `True` | LayerNorm on raw tokens |

`GetConfig.get_zero(input_size, **overrides)` returns the paper's architecture
(about 16.9M parameters with the cross-embodiment features), with any field
overridable.

`EmbodimentGraph.from_urdf` options:

| Argument | Default | Meaning |
|---|---|---|
| `feature_config` | `JointFeatureConfig()` | Hardware features per joint |
| `joint_order` | `None` (depth-first) | Explicit joint name order |
| `include_mimic_joints` | `False` | Keep mimic joints as tokens |
| `multi_dof_joints` | `"skip"` | `"skip"` or `"expand"` floating and planar joints |
| `sibling_edges` | `True` | Add undirected edges between joints sharing a parent body |

## Repository layout

```text
DA-GET-BFM/
├── model/
│   ├── __init__.py            # public API
│   ├── urdf.py                # standard-library URDF parser and transform utilities
│   ├── embodiment.py          # EmbodimentGraph, JointFeatureConfig, EmbodimentBatch,
│   │                          # collate_embodiments, FK, mass matrix
│   ├── tokenization_get.py    # GetTokenizer: GET-Zero token layout
│   ├── configuration_get.py   # GetConfig and the get_zero preset
│   ├── modeling_get.py        # GetEncoder, encoder layer, graph attention bias,
│   │                          # graph distance computation
│   ├── transformer.py         # attention, cross-attention, numeric tokenizer,
│   │                          # sinusoidal position embedding
│   └── README.md              # module-level technical notes
├── tests/
│   ├── test_embodiment.py     # URDF graphs, features, kinematics, dynamics, batching
│   ├── test_get_encoder.py    # graph distances, mass bias, padding, permutation
│   │                          # equivariance, cross-attention
│   └── test_transformer.py    # transformer building blocks
├── raw/                       # source material (the GET-Zero project page)
├── graphify-out/              # knowledge graph of the code and paper
├── pyproject.toml
└── uv.lock
```

## Testing and validation

```bash
uv run pytest
```

The suite has 25 tests. The URDF tests build small robots in code: a double
pendulum, a 12-DoF quadruped with fixed feet and an IMU, and a humanoid with a
floating root, waist, legs, arms, and a mimic gripper. They check:

- the joint graph, edges, and distance matrices;
- mimic, floating, and planar joint handling, and custom joint orders;
- hardware features and the fixed-link inertia merge;
- forward kinematics and the mass matrix against the closed form for a double
  pendulum, and that `M(q)` is positive definite;
- joint normalization and the tokenizer layout;
- that a mixed batch encodes each robot the same as encoding it alone, and that
  padding does not change the output.

The implementation was also checked outside the test suite:

- **Against the GET-Zero reference code:** on all 633 LEAP Hand URDF variants
  shipped with GET-Zero (5 to 16 DoF), the adjacency matrix matches the
  reference implementation exactly. The joint origin and rotation features match
  the URDF to float32 precision.
- **On real robot URDFs:** Unitree Go1 (12 DoF), H1 (19 DoF), and G1 (29 DoF)
  all parse, give positive-definite mass matrices, and encode together with a
  LEAP Hand in one batch with finite outputs.

## Limitations

- **Tree-structured robots only.** Closed kinematic loops (parallel linkages)
  cannot be expressed in URDF and are rejected.
- **Xacro must be expanded first**, for example with
  `xacro robot.urdf.xacro > robot.urdf`.
- **The base is not tokenized by default.** A floating base is a rigid mount
  unless `multi_dof_joints="expand"`. Base state such as angular velocity and
  gravity should go into `o_vg`.
- **The mass matrix is fixed-base.** It covers the actuated joints and does not
  include the coupling with base motion.
- **The URDF must have inertials.** Joints that move massless bodies give a zero
  mass-matrix diagonal. Add inertials or set `use_mass_encoding=False`.
- **Encoder only.** Policy and self-modeling heads, training, and distillation
  from GET-Zero are not included.

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
