# DA-GET-BFM: Dynamics Aware Cross Embodiment Behavior Foundation Model with Zero-Shot Capabilities

**Dynamic-Aware Graph Embodiment Transformer for Behavior Foundation Models.**

This is research code that asks whether one behavior foundation model (BFM) can
control many different robots. It builds on two references:

- [GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025)
  represents a robot as a graph with one token per joint, so one model can
  handle different bodies.
- [BFM-Zero](https://lecar-lab.github.io/BFM-Zero/) (Li et al., 2025) learns one
  policy that can be prompted for goals, motion tracking, or rewards without
  retraining, but its networks read the flat state of a single humanoid.

This repo makes the GET encoder aware of each robot's dynamics. One encoder
instance supplies joint embeddings to the BFM's actor, forward map, backward
map, discriminator, and critics. The actor outputs one action per joint, for
whichever robot it is given.

```text
                    M(q), rigid-body, actuator, dynamics
                                     │
policy tokens ──────────────────────────┐
state tokens ─► state adapter ──────────┤
                                        ▼
URDF + dynamics ─────────────────► DaGetEncoder ─► one embedding per joint
                                        │
                                        ├─► BackwardMap ─► latent z
                                        ├─► Discriminator
                                        ├─► Actor ─► one action per joint
                                        └─► ForwardMap and critics ─► F, Q
```

## Status

| Part | State |
|---|---|
| Encoder for fixed-base, floating-base, and multirotor robots | Done and tested |
| Rigid-body dynamics with rotor inertia, contact-constrained torque response | Done; response checked against MuJoCo (3% with stiff contacts) |
| Torque-free physics supervision: no estimated torque enters training or inference | Done and tested ([why](docs/dynamics.md#why-training-uses-no-estimated-torque)) |
| Physics pretraining (seven tasks, Hugging Face `Trainer`) | Done; short training runs only |
| Training data: synthetic, MuJoCo ground contact, Hub datasets, converters | Done |
| BFM networks: backward map, discriminator, actor, forward map, critics | Done and unit tested |
| Online off-policy FB-CPR training across robots | Logic done and tested on MuJoCo; **only short runs, so no policy is trained yet** |
| Actuator limits in RL: torque-speed envelope at every physics step, target clipping (same code on the robot), three torque penalties with per-robot Lagrange multipliers, metrics | Done and tested on MuJoCo; Isaac Lab actuator tested with a fake ([details](docs/bfm.md#actuator-torque-limits)) |
| Contact inference for data without foot sensors (dynamics-consistent subset selection with a confidence) | Done; measured on six real Go1 recordings and four simulated robots ([details](docs/dynamics.md#which-feet-are-down-inferring-contacts)) |
| 50 Hz deployment (`DaGetBfmPolicy`, CUDA Graphs) | Done; on an RTX 5070 Ti a G1 step takes at most 6.3 ms, 12.6 ms with live tracking, for every encoder size (budget 20 ms; robot communication not included) ([details](docs/bfm.md#inference-latency-at-50-hz)) |
| Isaac Lab simulation (self-collision, getting up, prioritized motions) | **Written against Isaac Lab 3 and tested with a fake only; never run in Isaac Lab, which needs a GPU** |
| **Prompting one robot with another's motion** | **B and D are shared, but nothing aligns behaviors across robots yet** ([details](docs/bfm.md#sharing-z-across-robots)) |

## What comes from where

| Component | Source | Added or changed here |
|---|---|---|
| Per-joint tokens, graph attention bias, `da_get_zero` preset | GET-Zero | Floating bases and multirotors |
| Mass-matrix attention bias, Physics Cross Attention, physics pretraining | This repo | |
| Contact-constrained response task (labels `G(q) δτ` without any torque) | This repo, from constrained rigid-body dynamics (Featherstone 2008) | |
| Torque-envelope target clipping, per-physics-step envelope, torque penalties | This repo | Added on top of BFM-Zero's auxiliary rewards |
| Adaptive torque-penalty weights | Lagrangian constrained RL (Altman 1999; Tessler et al. 2019; Stooke et al. 2020) | Per-robot multipliers with data-sheet targets |
| Contact inference from kinematics and dynamics | Probabilistic contact estimation on legged robots (Camurri et al. 2017; Bledt et al. 2018) | Subset MAP with friction pyramids, minimum load and a confidence filter for labels |
| Backward map and discriminator heads, truncated-normal policy, latent inference | BFM-Zero / FB-CPR | One shared DA-GET encoder with a state-token adapter; padding mask |
| Actor and forward map interfaces | BFM-Zero / FB-CPR | Rebuilt to read joint embeddings and cross-attend to z; one action per joint |
| Training losses, z mixture, rollout logic, fall starts, motion priorities, hyperparameters | BFM-Zero / FB-CPR | Per-robot worlds, buffers, and prompts; encoder tuned with physics losses |

The actor and critics are separate networks. In the published BFM-Zero training
configuration they share width and depth, while the critics use the forward-map
architecture to predict scalar values from state, latent, and action. This repo
uses per-joint attention in place of those residual MLPs; see
[the architecture comparison](docs/bfm.md#configuration).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and needs Python 3.10 or
newer.

```bash
git clone git@github.com:apoman123/DA-GET-BFM.git
cd DA-GET-BFM
uv sync --all-extras                     # core + MuJoCo, dataset converters, validation tools
uv run pytest                            # run the tests
uv run python scripts/download_urdfs.py  # Unitree Go1, H1, G1 and ANYmal D URDFs
```

URDF parsing and the dynamics need only PyTorch and the Python standard library.
The training pipeline also uses `transformers`, `datasets`,
`accelerate`, `tensorboard`, and `scipy`. Isaac Lab is installed separately.
`uv sync` alone installs the core, and tests that need an extra are skipped
without it:

| Extra | Packages | Needed for |
|---|---|---|
| `sim` | `mujoco` | Ground-contact simulation (`"type": "mujoco"`) and RL checks on a CPU |
| `convert` | `numcodecs` | GrandTour (zarr) conversion |
| `validation` | `h5py` | MIPO conversion and the torque-validation script |

## Physics features: sources and calculation

The encoder receives three physics feature groups with one row per joint, in
the same order as `graph.joint_names`. `rigid_body_features` has shape
`[B, J, 26]`, `actuator_features` has `[B, J, 6]`, and
`dynamics_features` has `[B, J, 5]`; `B` is the batch size and `J` is the
padded joint count. Padded rows are zero and excluded by `joint_mask`. The
column names, units, and normalization scales are defined in
[`model/physics_schema.py`](model/physics_schema.py). Joint-space values use
SI units: revolute joints use rad, rad/s, and N m; prismatic joints use m,
m/s, and N.

**No feature is a torque.** Every value comes from the robot model (URDF,
actuator data) and the sensed state (joint encoders, IMU). A robot, a simulator,
and a recorded dataset therefore produce the same features, and nothing needs a
torque sensor or a torque estimate. Earlier versions also fed the previous joint
torque (`[B, J, 7]` dynamics features, `[B, J, 25]` rigid-body features); see
[why that was removed](docs/dynamics.md#why-training-uses-no-estimated-torque).
Checkpoints of that schema are rejected with a clear error and must be
pretrained again.

### Rigid-body features: 26 values per joint

These describe the link driven by each joint. `EmbodimentGraph.from_urdf`
reads the URDF's joint origins, axes, and link `<inertial>` elements, merges
fixed links, and places the result in the joint's body frame `B_i`. Its origin
is the joint origin; the parent frame `B_p(i)` is the parent joint's body frame
or the root link frame. [`rigid_body_features(graph)`](model/physics.py)
packs the following values once per robot:

| Columns | Count | How they are obtained or calculated |
|---|---:|---|
| `mass` | 1 | Child body's mass `m_i` in kg from URDF inertials, including merged fixed links. |
| `inertia_xx`, `inertia_xy`, `inertia_xz`, `inertia_yy`, `inertia_yz`, `inertia_zz` | 6 | Independent entries of the symmetric inertia tensor about that body's center of mass, rotated into `B_i`, in kg m². |
| `com_x`, `com_y`, `com_z` | 3 | Center-of-mass offset from the joint origin, expressed in `B_i`, in m. |
| `axis_x`, `axis_y`, `axis_z` | 3 | Unit joint axis expressed in `B_i`. |
| `parent_translation_x/y/z` | 3 | Translation from `B_p(i)` to `B_i` at joint position zero, in m. |
| `parent_rotation_0` … `parent_rotation_5` | 6 | First two columns of that zero-position rotation matrix, flattened column first. |
| `type_revolute`, `type_continuous`, `type_prismatic` | 3 | One-hot joint type from the URDF. |
| `armature` | 1 | Reflected rotor inertia `I_i^rotor` of the joint's drive (kg m², or kg for prismatic joints), `EmbodimentGraph.armature`. URDF has none: set it with the source key `armature` (one value or `{joint glob: value}`), `from_urdf(armature=...)` or `graph.with_armature(...)`. It adds `I_i^rotor` to `M_ii` and `I_i^rotor q̈_i` to the joint effort, like MuJoCo's and Isaac Lab's `armature`. |

These values change only when the robot geometry, inertial parameters or rotor
inertia change. A URDF needs valid inertial data for dynamics calculations;
measured masses and inertias are preferable to placeholders. The rotor inertia
matters most on light, highly geared joints, where it can exceed the link's own
inertia (Wensing et al., IEEE T-RO 2017).

### Actuator features: 6 values per joint

[`actuator_specs(graph)`](model/physics.py) reads each URDF joint's `effort`
limit. URDFs usually do not provide continuous limits, controller gains, or
propeller data; those values come from defaults or explicit `overrides`. The
default assumptions are `continuous_torque = 0.5 × peak_torque` and zero PD
gains and propeller coefficients. The actuator spec still retains speed,
power, gear ratio, and control mode for simulation and physics labels, but
these four fields are **not encoder features**. Missing positive effort or
velocity limits on an actuated joint require an override or
`ActuatorDefaults`. [`actuator_features(specs)`](model/physics.py) packs:

| Columns | Count | Meaning and source |
|---|---:|---|
| `peak_torque`, `continuous_torque` | 2 | Short-term joint-side effort from URDF `effort` or override; sustainable effort from motor data or the default ratio. Units N m (N for prismatic joints). |
| `kp`, `kd` | 2 | Position and velocity controller gains from the actual controller, `ActuatorDefaults`, or an override. |
| `thrust_coefficient`, `torque_coefficient` | 2 | Propeller coefficients `k_f` and `k_m` from rotor specifications: thrust is `k_f ω²` along the axis, and drag magnitude is `k_m ω²`. Zero for ordinary joints. |

Supply real motor/controller data when available, for example
`actuator_specs(graph, overrides={"*_knee_joint": {"continuous_torque": 23.7,
"kp": 40.0, "kd": 1.0, "mode": "position"}})`. Dataset configs also accept
`actuator_overrides` by joint name or glob pattern. These features are static
until actuator parameters or the control mode change.

### Dynamics features: 5 values per joint and time step

These use the current joint position `q_t`, velocity `qd_t`, gravity in the
base frame, and the base twist. The functions in
[`model/dynamics.py`](model/dynamics.py) compute the robot's mass matrix and
inverse-dynamics terms; [`dynamics_features`](model/physics.py) assembles them
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
learned projections. See [Physics Cross Attention](docs/encoder.md#6-physics-cross-attention)
for how these groups enter the encoder.

### What a robot needs at run time

| Signal | Source on a robot | Used for |
|---|---|---|
| Joint positions and velocities `q`, `qd` | Joint encoders | Tokens, dynamics features, mass matrix |
| Gravity direction and base angular velocity | IMU | Tokens, dynamics features |
| Previous target and the PD gains | The controller's own command and configuration | Previous-action token features: the target and the effort `K_p (a − q) − K_d qd` it requests, computed, not measured |
| Base linear velocity | State estimator (optional) | Only with `observe_base_linear_velocity` |
| Joint torque | **Not needed** | — |

The PD gains on the robot must equal the training gains (`gain_scale`, or the
`kp`/`kd` overrides); otherwise the requested-effort features mean something
else. Targets go through [`ActionMap`](model/rl/actions.py), the same code the
simulators use, so the effort the robot's drives request stays inside each
actuator's torque-speed envelope ([details](docs/bfm.md#actuator-torque-limits)).

## How the encoder learns dynamics without torque labels

Joint torques cannot be estimated reliably from recorded motion: on real robots
the estimate is 43–97% off (Unitree Go1) and 47–72% off (ANYmal D), mostly from
sources no estimator removes. Training therefore never estimates a torque. The
encoder still learns each robot's own dynamics from three sources
([details](docs/training.md#physics-pretraining)):

1. **Exact inputs.** `M(q)` (attention bias and `M_ii`), `c`, `g`, friction and
   the rigid-body and actuator features are computed from the model, never
   measured.
2. **The contact-constrained response, on every dataset.** For any recorded
   state, the collator draws a torque change `δτ` and labels it with the exact
   acceleration change `δν̇ = G(q) δτ`,
   `G = [M⁻¹ − M⁻¹Jᵀ(J M⁻¹ Jᵀ)⁻¹J M⁻¹] Sᵀ` with the contacts holding
   ([`constrained_response`](model/dynamics.py)). `G` depends only on the
   configuration, the inertia and which feet are down: not on the applied
   torque, the contact forces or differentiated accelerations. Tasks R and R⁻¹
   learn it in both directions.
3. **Exact torques where they exist.** Simulated sources (synthetic, MuJoCo
   ground) record the applied torque together with the acceleration it produced;
   only they train forward and inverse dynamics with full torques. Torque-sensor
   data is off by default (`measured_torque_weight: 0`).

Kinematic tasks (forward kinematics, action effect, motor feasibility) run on
every dataset as before. During RL, the policy's own rollouts add response,
effect and feasibility labels with the simulator's exact contacts, and a few
label worlds per robot add exact forward and inverse dynamics labels on the
states the policy visits, walking included: torque and acceleration are read at
the start of each control step, so they belong to one instant
([physics replay](docs/bfm.md#physics-losses-during-rl)).

## Quick start

### 1. Encode different robots in one batch

```python
import torch

from model import (
    EmbodimentGraph, DaGetConfig, DaGetEncoder, DaGetTokenizer,
    JointFeatureConfig, collate_embodiments,
)
from model.physics import PhysicsFeatureCache, actuator_specs

# One graph per robot, built from URDF. Legged URDFs usually start at the
# trunk without a floating joint, so say that the base is floating.
features = JointFeatureConfig.da_get_zero()
graphs = [
    EmbodimentGraph.from_urdf("assets/urdf/go1.urdf", features, floating_base=True),  # quadruped, 12 DoF
    EmbodimentGraph.from_urdf("assets/urdf/h1.urdf", features, floating_base=True),   # humanoid, 19 DoF
]
batch = collate_embodiments(graphs)                     # pads to 19 joints

# Physics features: static ones are computed here once.
specs = [actuator_specs(graph) for graph in graphs]     # URDF limits + your overrides
physics = PhysicsFeatureCache(graphs, specs, configuration_interval=4)

# Observations with history (oldest first) -> one token per joint.
H, J = 3, batch.joint_mask.shape[1]
q = torch.zeros(len(graphs), H, J)
qd = torch.zeros(len(graphs), H, J)
gravity = torch.tensor([[0.0, 0.0, -9.81]] * len(graphs))   # IMU projected gravity x 9.81
gyro = torch.zeros(len(graphs), 3)                          # base angular velocity (base frame)
tokenizer = DaGetTokenizer(fixed_local_size=features.size, variable_local_size=2, history_size=H)
tokens = tokenizer(
    batch.joint_features, batch.joint_mask,
    variable_local=torch.stack((batch.normalize_joint_positions(q), qd), dim=-1),
)

# Encode.
encoder = DaGetEncoder(
    DaGetConfig.da_get_zero(input_size=tokenizer.token_size, use_physics_cross_attention=True)
).eval()
memory = encoder.build_physics_memory(               # cache rigid-body and actuator keys/values
    physics.rigid_body_features, physics.actuator_features, batch.joint_mask
)
dynamics = physics.update(                           # M_ii, c, g, friction, gravity load
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

### 2. Get one action per joint from a prompt

The networks below are untrained here, so the actions only show the shapes. The prompt is
a goal state of the Go1: B reads per-DoF state tokens, so it fits any robot.

```python
from model import BfmConfig, DaGetBfmModel
from model.rl import EmbodimentObserver, RobotState
from model.training.episodes import EmbodimentRecord
from model.training.ground_sim import ground_specs

go1 = EmbodimentRecord.build("go1", graphs[0], ground_specs(specs[0]))
observer = EmbodimentObserver(go1, history_size=H)
pose = torch.zeros(1, graphs[0].num_joints, dtype=torch.float64)    # the goal pose
still = torch.zeros(1, 3, dtype=torch.float64)
down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)        # gravity in the base frame
height = torch.tensor([0.3], dtype=torch.float64)
goal = observer.state_inputs(observer.state_view(RobotState(pose, pose, down, still, still, height)))

bfm = DaGetBfmModel(BfmConfig(embedding_size=hidden.shape[-1], state_token_size=observer.state_token_size), encoder)
bfm.init_weights()

z = bfm.goal_inference(goal).expand(2, -1)           # or tracking_inference / reward_inference
action = bfm.act(hidden, z, batch.joint_mask)        # [2, 19]; values in [-1, 1], padded joints are 0
```

### 3. Pretrain the encoder

```bash
uv run python scripts/train_physics_da_get.py \
    --data_config configs/data/synthetic_unitree.json \
    --output_dir runs/synthetic --max_steps 2000 --per_device_train_batch_size 64

uv run tensorboard --logdir runs    # watch the losses at http://localhost:6006
```

### 4. Train the behavior foundation model

This needs Isaac Lab and an NVIDIA GPU. Add `--simulator mujoco --device cpu` and small
sizes to check the pipeline on a CPU.

```bash
python scripts/train_bfm.py \
    --data_config configs/rl/g1_go1_real_and_synthetic.json \
    --encoder_checkpoint runs/encoder --output_dir runs/bfm --headless
```

Select architecture sizes with `--preset tiny|small|medium|large` during encoder
pretraining, or `--encoder_size`, `--actor_size`, `--forward_size`, and
`--backward_size` during BFM training. The four BFM flags can use different sizes;
`--encoder_size` applies when starting without an encoder checkpoint.

Sources with the same `robot` name are one robot, so real and synthetic trajectories
combine. Expert sources teach the policy; all sources tune the encoder through the physics
losses. The encoder checkpoint must use the same `joint_features` and `history_size`. See
[Training](docs/bfm.md#training-online-off-policy-fb-cpr).

## Documentation

| Read this | To learn |
|---|---|
| [The DA-GET encoder](docs/encoder.md) | URDF to tokens, attention bias, Physics Cross Attention, batching, all encoder options |
| [Dynamics and torque](docs/dynamics.md) | Rigid-body dynamics, rotor inertia, the constrained response, why torque estimates are not used |
| [Physics pretraining](docs/training.md) | The seven tasks, label sources, training commands, TensorBoard, data config format |
| [Training data](docs/data.md) | MuJoCo ground contact, automatic configs, dataset converters, recommended datasets |
| [Behavior foundation model](docs/bfm.md) | Actor, critics, B, and D; RL training; torque limits; physics replay; deployment; open questions |
| [Development notes](docs/development.md) | Full repository layout, test coverage, validation results, detailed limitations |
| [Module notes](model/README.md) | Technical notes on the encoder code |

## Repository layout

```text
model/                     library code
  modeling_da_get.py         DaGetEncoder and Physics Cross Attention
  modeling_bfm.py            backward map, discriminator, actor, forward map, critics
  rl/                        FB-CPR agent, Isaac Lab and MuJoCo worlds, action map and torque limits,
                             torque-penalty multipliers, physics replay, replay buffers, evaluation,
                             training loop, deployment
  pretraining.py             physics pretraining model (heads A, B, C, R, R^-1, D, E)
  dynamics.py                rigid-body dynamics, constrained response, torque estimation (analysis only)
  training/, conversion/     data pipeline (including contact inference) and dataset converters
scripts/                   training, data config generator, converters, validation, latency benchmark
configs/data/              example data configs
docs/                      documentation by topic
tests/                     test suite
```

## Limitations

- **Robots must be tree-structured**, with inertials in the URDF. Closed
  kinematic loops are rejected, and Xacro and MJCF must be converted first.
- **Torque estimates are not used.** They are 43–97% off on a real Go1 and
  47–72% on ANYmal D, so datasets without torque sensors supervise dynamics
  only through the constrained response.
- **The response assumes rigid contacts that stick.** The generated MuJoCo models
  use stiff contacts (3% from the rigid response, against 20% with MuJoCo's
  defaults), and frames where a foot in contact slides faster than 0.1 m/s are
  dropped. Without contact sensors, contacts are inferred
  ([how and how well](docs/dynamics.md#which-feet-are-down-inferring-contacts)):
  on six real Go1 gait recordings, 82–97% of the frames that keep response labels
  have the right contact set (47–80% with the old lowest-foot rule).
- **Torque targets are data-sheet style choices.** The simulators clip the PD
  torque to the torque-speed envelope at every physics step, and the torque
  penalty weights adapt per robot to targets (Lagrange multipliers), but the
  targets themselves (5% of joints over the envelope, thermal load ≤ 1, 2% over
  the power limit) are not measured on a robot. The envelope limits braking like
  driving torque, which is conservative.
- **The offline ground simulation stands, it does not walk.** Gaits would need a
  locomotion controller per robot. Recorded datasets give locomotion kinematics,
  and the RL physics replay gives exact torque and acceleration labels on the
  policy's own walking (MuJoCo, or Isaac Lab with explicit actuators;
  [details](docs/bfm.md#physics-losses-during-rl)).
- **The Isaac Lab adapter is unverified on a real simulator** and rotor robots are not
  supported in RL ([first run](docs/bfm.md#first-run-on-a-gpu-machine),
  [differences from BFM-Zero](docs/bfm.md#what-differs-from-bfm-zero)).
- **Automatic configs and actuator defaults are guesses.** Check them, and use
  measured values when you have them.

See [the full list](docs/development.md#limitations).

## License and attribution

The heads of the backward map and discriminator, the truncated-normal policy, latent
inference, and FB-CPR training code are adapted from [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) (Copyright
Meta Platforms, Inc. and affiliates) under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), which does not
allow commercial use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Motion datasets have their own licenses; see
[Training data](docs/data.md#recommended-training-datasets).

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
- Mayank Mittal et al. *Orbit: A Unified Simulation Framework for Interactive Robot
  Learning Environments.* IEEE RA-L 2023 ([Isaac Lab](https://isaac-sim.github.io/IsaacLab/),
  the simulator).
- Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. *Mastering
  Visual Continuous Control: Improved Data-Augmented Reinforcement Learning.*
  ICLR 2022 (truncated-normal policy).
- Scott Fujimoto, Herke van Hoof, and David Meger. *Addressing Function
  Approximation Error in Actor-Critic Methods.* ICML 2018 (clipped policy
  noise).
- Roy Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008 (RNEA,
  floating-base and contact-constrained dynamics).
- Emanuel Todorov, Tom Erez, and Yuval Tassa. *MuJoCo: A Physics Engine for
  Model-Based Control.* IROS 2012 (ground-contact simulation and response check).
- Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel.
  *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.* ICRA
  2018 (randomized inertia, center of mass, rotor inertia and motor strength).
- Patrick M. Wensing, Albert Wang, Sangok Seok, David Otten, Jeffrey Lang, and
  Sangbae Kim. *Proprioceptive Actuator Design in the MIT Cheetah: Impact
  Mitigation and High-Bandwidth Physical Interaction for Dynamic Legged Robots.*
  IEEE T-RO 2017 (reflected rotor inertia).
- Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios
  Tsounis, Vladlen Koltun, and Marco Hutter. *Learning Agile and Dynamic Motor
  Skills for Legged Robots.* Science Robotics 2019 (actuator modelling; why
  joint torque is hard to model from motion).
- Taeyoung Lee, Melvin Leok, and N. Harris McClamroch. *Geometric Tracking
  Control of a Quadrotor UAV on SE(3).* CDC 2010 (flight controller).
- Shuo Yang et al. *Multi-IMU Proprioceptive Odometry for Legged Robots.*
  IROS 2023 (MIPO data used for validation).
- Eitan Altman. *Constrained Markov Decision Processes.* Chapman & Hall/CRC, 1999;
  Chen Tessler, Daniel J. Mankowitz, and Shie Mannor. *Reward Constrained Policy
  Optimization.* ICLR 2019; Adam Stooke, Joshua Achiam, and Pieter Abbeel.
  *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods.* ICML 2020
  (Lagrange multipliers of the torque penalties).
- Marco Camurri, Maurice Fallon, Stéphane Bazeille, Andreea Radulescu, Victor
  Barasuol, Darwin G. Caldwell, and Claudio Semini. *Probabilistic Contact Estimation
  and Impact Detection for State Estimation of Quadruped Robots.* IEEE RA-L 2017;
  Gerardo Bledt, Patrick M. Wensing, Sam Ingersoll, and Sangbae Kim. *Contact Model
  Fusion for Event-Based Locomotion in Unstructured Terrains.* ICRA 2018 (contact
  inference from kinematics and dynamics).
- Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. *Learning to Walk in
  Minutes Using Massively Parallel Deep Reinforcement Learning.* CoRL 2021
  (legged_gym; explicit PD torques at every physics step).
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
