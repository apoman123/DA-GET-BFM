# GET encoder

An implementation of the Graph Embodiment Transformer (GET) encoder from
[GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025), with
one input token per degree of freedom (DoF) and embodiment graphs read from URDF.
It works for any tree-structured robot: dexterous hands, arms, quadrupeds,
bipeds, and humanoids. The encoder has no policy or self-modeling heads.

Environment (uses [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest
```

## Pipeline

```python
import torch

from model import (
    EmbodimentGraph, DaGetConfig, DaGetEncoder, DaGetTokenizer,
    JointFeatureConfig, collate_embodiments,
)

features = JointFeatureConfig.cross_embodiment()   # or JointFeatureConfig() = paper
graphs = [
    EmbodimentGraph.from_urdf("go1.urdf", features),
    EmbodimentGraph.from_urdf("h1.urdf", features),
    EmbodimentGraph.from_urdf("leap_hand.urdf", features),
]
batch = collate_embodiments(graphs)                 # pads to the largest DoF count

H = 6                                               # GET-Zero history length
tokenizer = DaGetTokenizer(
    fixed_local_size=features.size,                 # o_fl: URDF hardware features
    variable_local_size=2,                          # o_vl: q (normalized), PD target
    variable_global_size=6,                         # o_vg: e.g. base ang. vel + gravity
    history_size=H,
)
q = torch.zeros(len(graphs), H, batch.joint_mask.shape[1])          # oldest first
q_target = torch.zeros_like(q)
tokens = tokenizer(
    batch.joint_features,
    batch.joint_mask,
    variable_local=torch.stack(
        (batch.normalize_joint_positions(q), batch.normalize_joint_positions(q_target)), -1
    ),
    variable_global=torch.zeros(len(graphs), H, 6),
)

encoder = DaGetEncoder(DaGetConfig.da_get_zero(input_size=tokenizer.token_size))
hidden = encoder(
    tokens,
    mass_matrix=batch.mass_matrix(q[:, -1]),        # M(q) from URDF inertials
    **batch.encoder_inputs(),                       # parents, mask, edges, distances
)                                                   # [batch, joints, 512]
```

## Tokens: one per DoF

GET-Zero builds token `j` as

```text
T_j = [o_vg^{t-H+1..t}, o_fg, o_vl,j^{t-H+1..t}, o_fl,j]
```

| Part | Meaning | GET-Zero LEAP Hand |
|------|---------|--------------------|
| `o_vg` | variable global obs with history, shared by all tokens | task phase (sin/cos) |
| `o_fg` | fixed global obs | none |
| `o_vl,j` | variable per-joint obs with history | normalized joint angle, PD target |
| `o_fl,j` | fixed per-joint hardware properties from URDF | joint origin and rotation relative to the parent joint |

History is flattened feature-major, as in the GET-Zero code. The encoder applies a
layer norm over each raw token and then a learned linear embedding, as in the
paper. `DaGetConfig.da_get_zero` gives the paper's architecture: 8 layers, 16 heads,
width 512, feed-forward 1024, dropout 0.1.

`JointFeatureConfig()` matches the paper (6 features). Robots in the paper share
joint axes, so axes were not needed. Across humanoids, quadrupeds, and hands they
differ, so `JointFeatureConfig.cross_embodiment()` adds the axis, joint type,
limits, effort and velocity limits, damping and friction, and the mass, center
of mass, and inertia of the body each joint moves. It also uses the continuous
6D rotation representation (31 features).

## Embodiment graph from URDF

`EmbodimentGraph.from_urdf` parses URDF with the standard library (no
robotics dependencies). Expand xacro files first.

* **Nodes**: every `revolute`, `continuous`, and `prismatic` joint.
* **Directed edges**: parent to child, through any number of fixed joints.
* **Undirected edges**: between joints that share a parent body, GET-Zero's
  "joints at the start of independent serial chains". For a hand these are the
  finger roots on the palm, for a quadruped the four hips on the trunk, and for a
  humanoid the hips and waist on the pelvis. Shoulders on the torso are joined
  the same way. Disable this with `sibling_edges=False`.
* **Fixed joints** are collapsed. Their links' inertia is merged into the body
  of the nearest moving joint, and hardware poses are composed through them.
* **Mimic joints** (for example a gripper's second finger) are not independent
  DoFs. They are frozen at their offset unless `include_mimic_joints=True`.
* **Floating and planar joints** are rigid mounts by default. With
  `multi_dof_joints="expand"` they become chains of 1-DoF virtual joints (6 and
  3), so each DoF still gets one token.
* **Joint order** defaults to depth-first URDF order, the order MuJoCo and
  Pinocchio use. Pass `joint_order=[...]` to match your simulator.

On all 633 LEAP Hand variants that ship with GET-Zero, the graph's edges match
the reference implementation's adjacency matrix exactly. The joint
origin/rotation features match the URDF to float32 precision.

The graph also provides:

* `forward_kinematics(q)`: each joint's pose in the root frame, which is the
  GET-Zero self-modeling target.
* `mass_matrix(q)`: the joint-space inertia `M(q)`, used by the dynamic attention
  bias. For a floating base robot this equals the actuated block of the
  floating-base mass matrix. It checks against the closed form for a double
  pendulum. A joint whose moving body has no mass has a zero diagonal, which the
  encoder rejects.
* `normalize_joint_positions(q)`: scales to [-1, 1] by the limits; continuous
  joints wrap by pi.

## Attention

Each layer adds a learned per-head bias to the attention logits:

```text
normalized_mass[i, j] = M(q)[i, j] / (sqrt(M(q)[i, i] * M(q)[j, j]) + mass_epsilon)
A[i, j] = q_i k_j / sqrt(d) + spatial[SPD(i, j)] + parent[P(i, j)] + child[C(i, j)]
          + linear(normalized_mass[i, j])
```

`SPD` is the shortest path over undirected edges. `P` and `C` count ancestor and
descendant steps and are zero for unrelated pairs. Unreachable pairs, and
distances above `max_graph_distance`, share an overflow bucket. That bucket does
not depend on how far a batch is padded. The spatial and parent/child terms are
from GET-Zero. The mass term is an addition in this repository. It is computed
at every step from `M(q)`, so it changes with the configuration. Turn it off with
`use_mass_encoding=False`.

Graph distances depend only on the robot, so `EmbodimentGraph` precomputes them
and the encoder accepts them as `graph_distances`. Without them, the encoder
computes them from `parent_indices` and `adjacency_matrix` with a Python loop.

Low-level use without URDF still works: `parent_indices[b, j]` is joint `j`'s
parent or `-1`, and `joint_mask` and `context_mask` use `True` for valid tokens.
For padded joints, set `parent_indices` to `-1` and their mass rows and columns
to zero. Padded outputs are zero. Cross-attention to context tokens runs after
self-attention in the layers listed in `cross_attention_layers`.

## Physics Cross Attention and pretraining

With `use_physics_cross_attention=True`, every encoder layer (or those in
`physics_cross_attention_layers`) adds Physics Cross Attention after
self-attention. Observation tokens attend to `3J` physics tokens:

* rigid body (25): `m`, `vech(I)` about the CoM in body frame `B_i`, `r^CoM`,
  `a^axis`, `T^parent`, joint type;
* actuator (14): `τ^peak`, `τ^cont`, `ω^max`, `P^max`, `N`, `K_p`, `K_d`,
  propeller `k_f` and `k_m`, and the mode (torque, position, velocity,
  passive, thrust);
* dynamics (7): `M_ii(q)`, `c(q, qd, base twist)`, `g(q)` with gravity in the
  base frame, friction plus rotor drag, `τ_{t-1}`, `|g| / τ^cont`, and the
  source of `τ_{t-1}` (1 measured, 0.5 estimated, 0 unknown).

Each group has its own asinh-scaled MLP tokenizer
(`PhysicsFeatureTokenizer`). The attention bias is a per-head graph distance
term plus a per-group term.

`DaGetEncoder.build_physics_memory` caches the rigid-body and actuator keys and
values, so a control step only embeds the dynamics tokens. The encoder rejects
a memory built from parameters that have changed since.

The modules:

* `model/physics_schema.py`: the feature schema, with symbols, units, frames,
  and update tiers.
* `model/physics.py`: feature extraction, plus `PhysicsFeatureCache` for tiered
  recomputation at control time.
* `model/dynamics.py`: RNEA for fixed and floating bases, the floating-base
  mass matrix and forward dynamics, contact-force distribution, rotor thrust,
  and `estimate_joint_efforts` for datasets without torque sensors.
* `model/pretraining.py`: the auxiliary heads A to F.
* `model/training/`: the data pipeline and Trainer.

See the repository README for the full description, the training commands,
and the recommended datasets.

## Data tools

* `model/training/ground_sim.py`: MuJoCo ground-contact simulation built from
  the embodiment graph, with exact torque, acceleration, and contact labels.
* `model/robot_analysis.py`: infers floating base, feet, flat feet, rotors,
  and propeller coefficients from a URDF.
* `model/conversion/`: converters for GrandTour, MIPO, NeuroBEM, LocoMuJoCo,
  AMASS-G1, and LAFAN1, plus dataset-column-to-joint matching.

`scripts/make_data_config.py` and `scripts/convert_dataset.py` are the command
lines; the repository README documents them.
