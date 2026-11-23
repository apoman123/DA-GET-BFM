# DA-GET-BFM: Dynamics Aware Cross Embodiment Behavior Foundation Model with Zero-Shot Capabilities

**Dynamic-Aware Graph Embodiment Transformer for Behavior Foundation Models.**

This is research code that asks whether one behavior foundation model (BFM) can
control many different robots. It builds on two references:

- [GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025)
  represents a robot as a graph with one token per joint, so one model can
  handle different bodies.
- [BFM-Zero](https://lecar-lab.github.io/BFM-Zero/) (Li et al., ICLR 2026) learns one
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
| **Prompting one robot with another's motion** | **B and D are shared, but nothing aligns behaviors across robots yet** ([details](docs/bfm.md#sharing-z-across-robots)); prompts can now be retargeted between robots first ([G3](docs/zero_shot.md#g-zero-shot-at-the-bfm-level)) |
| Zero-shot proposal v4 ([PDF](docs/proposal/DA-GET-BFM_proposal_v4.pdf)): physics-derived normalization, 50 Hz, dimensionless inputs, structured physics interface, root token, contact and centroidal heads, procedural robots, temperature sampling, mirroring, ModuMorph-style modulation, spec noise, masked physics modeling, test-time adaptation, retargeting, split-level evaluation with bootstrap CIs | **Implemented and unit tested, all off by default; no large training run or ablation yet** ([details](docs/zero_shot.md)) |

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and needs Python 3.10 or
newer.

```bash
git clone git@github.com:apoman123/DA-GET-BFM.git
cd DA-GET-BFM
uv sync --all-extras                     # core + MuJoCo, dataset converters, validation tools
uv run pytest                            # run the tests
uv run python scripts/download_urdfs.py  # robot URDFs by Unitree, ANYbotics and PAL Robotics
uv run python scripts/download_datasets.py --list  # public datasets to download and convert (docs/data.md)
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

## Quick start

### 1. Encode different robots in one batch

What each physics feature contains and where it comes from is in
[Physics features](docs/physics_features.md).

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
| [Physics features](docs/physics_features.md) | The rigid-body, actuator and dynamics features: sources, calculation, what a robot needs at run time |
| [The DA-GET encoder](docs/encoder.md) | URDF to tokens, attention bias, Physics Cross Attention, batching, all encoder options |
| [Dynamics and torque](docs/dynamics.md) | Rigid-body dynamics, rotor inertia, the constrained response, why torque estimates are not used |
| [Physics pretraining](docs/training.md) | How dynamics is learned without torque labels, the seven tasks, label sources, training commands, TensorBoard, data config format |
| [Training data](docs/data.md) | MuJoCo ground contact, automatic configs, dataset converters, recommended datasets |
| [Behavior foundation model](docs/bfm.md) | Actor, critics, B, and D; RL training; torque limits; physics replay; deployment; open questions |
| [Zero-shot across tasks and embodiments](docs/zero_shot.md) | The proposal v4 options, their evidence, configs, training and evaluation; the [proposal PDF](docs/proposal/DA-GET-BFM_proposal_v4.pdf) |
| [Development notes](docs/development.md) | Full repository layout, test coverage, validation results, limitations, CI |
| [References and credits](docs/references.md) | What comes from whom, the main references with all authors, BibTeX |
| [Module notes](model/README.md) | Technical notes on the encoder code |

## Limitations

The status table above marks what is untested. The full list (tree-structured robots
only, no torque labels for recorded data, rigid holding contacts, data-sheet torque
targets, the Isaac Lab adapter never run on a real simulator, and more) is in
[Development notes](docs/development.md#limitations).

## License and attribution

This repository combines the Graph Embodiment Transformer of GET-Zero (Austin Patel and
Shuran Song, ICRA 2025) with the behavioral foundation model of BFM-Zero (Yitang Li et
al., ICLR 2026), which builds on FB-CPR / Meta Motivo (Andrea Tirinzoni et al., ICLR
2025) and forward-backward representations (Ahmed Touati and Yann Ollivier, NeurIPS
2021). [References and credits](docs/references.md) says which part comes from whom and
lists the references with all authors, with BibTeX for the two main works.

The heads of the backward map and discriminator, the truncated-normal policy, latent
inference, and the FB-CPR training code are adapted from
[BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) (agent files Copyright Meta Platforms,
Inc. and affiliates) under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
which does not allow commercial use. The robot models (Unitree Robotics, ANYbotics, PAL
Robotics), the datasets, the fonts embedded in the proposal PDF, and the tools and
libraries used are credited with their licenses in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Motion datasets have their own terms;
see [Training data](docs/data.md#recommended-training-datasets).
