# DA-GET-BFM

**Dynamic-Aware Graph Embodiment Transformer for Behavior Foundation Models.**

This is research code that asks whether one behavior foundation model (BFM) can
control many different robots. It builds on two references:

- [GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025)
  represents a robot as a graph with one token per joint, so one model can
  handle different bodies.
- [BFM-Zero](https://lecar-lab.github.io/BFM-Zero/) (Li et al., 2025) learns one
  policy that can be prompted for goals, motion tracking, or rewards without
  retraining, but its networks read the flat state of a single humanoid.

This repo makes the GET encoder aware of each robot's dynamics, then has the
BFM's actor and critics read its joint embeddings. The actor outputs one action
per joint, for whichever robot it is given.

```text
                    M(q), rigid-body, actuator, dynamics
                                     │
URDF + joint state ─► DaGetTokenizer ─► DaGetEncoder ─► one embedding per joint
                                                              │
flat state ─► BackwardMap ─► latent z ─── cross-attention ────┤
                                                              ├─► DaGetBfmActor ─► one action per joint
                                                              └─► DaGetForwardMap ─► F(s, a, z), Q(s, a, z)
```

## Status

| Part | State |
|---|---|
| Encoder for fixed-base, floating-base, and multirotor robots | Done and tested |
| Rigid-body dynamics and torque estimation | Done; checked against real Go1 and ANYmal D data |
| Physics pretraining (six tasks, Hugging Face `Trainer`) | Done; short training runs only |
| Training data: synthetic, MuJoCo ground contact, Hub datasets, converters | Done |
| BFM networks: backward map, discriminator, actor, forward map, critics | Done and unit tested |
| Online off-policy FB-CPR training across robots (MuJoCo) | Done and tested; **only short runs so far, so no policy is trained yet** |
| **Prompting one robot with another's motion** | **Open design question** ([details](docs/bfm.md#open-questions-for-cross-embodiment-training)) |

## What comes from where

| Component | Source | Added or changed here |
|---|---|---|
| Per-joint tokens, graph attention bias, `da_get_zero` preset | GET-Zero | Floating bases and multirotors |
| Mass-matrix attention bias, Physics Cross Attention, physics pretraining | This repo | |
| Backward map, discriminator, truncated-normal policy, latent inference | BFM-Zero / FB-CPR | Dictionary inputs, padding mask |
| Actor and forward map interfaces | BFM-Zero / FB-CPR | Rebuilt to read joint embeddings and cross-attend to z; one action per joint |
| Training losses, z mixture, rollout logic, hyperparameters | BFM-Zero / FB-CPR | Per-robot simulators, buffers, and prompts |

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
`accelerate`, and `tensorboard`.
`uv sync` alone installs the core, and tests that need an extra are skipped
without it:

| Extra | Packages | Needed for |
|---|---|---|
| `sim` | `mujoco` | Ground-contact simulation (`"type": "mujoco"`) |
| `convert` | `numcodecs` | GrandTour (zarr) conversion |
| `validation` | `h5py` | MIPO conversion and the torque-validation script |

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

### 2. Get one action per joint from a prompt

The networks below are untrained here, so the actions only show the shapes.

```python
from model import BfmConfig, DaGetBfmModel

state_dim = 64                                       # your flat state for B and D
goal_state = torch.randn(2, state_dim)

bfm = DaGetBfmModel(BfmConfig(embedding_size=hidden.shape[-1], backward_input_size=state_dim))
bfm.init_weights()

z = bfm.goal_inference(goal_state)                   # or tracking_inference / reward_inference
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

```bash
uv run python scripts/train_bfm.py \
    --data_config configs/rl/go1_h1_ground.json \
    --encoder_checkpoint runs/synthetic --output_dir runs/bfm
```

Each `source` in the data config is one robot: its motions are the expert data, and
a MuJoCo environment built from its URDF collects the replay data. The encoder
checkpoint must use the same `joint_features` and `history_size`. See
[Training](docs/bfm.md#training-online-off-policy-fb-cpr).

## Documentation

| Read this | To learn |
|---|---|
| [The DA-GET encoder](docs/encoder.md) | URDF to tokens, attention bias, Physics Cross Attention, batching, all encoder options |
| [Dynamics and torque estimation](docs/dynamics.md) | Rigid-body dynamics, estimating missing torques, why the error is large |
| [Physics pretraining](docs/training.md) | The six tasks, training commands, TensorBoard, data config format |
| [Training data](docs/data.md) | MuJoCo ground contact, automatic configs, dataset converters, recommended datasets |
| [Behavior foundation model](docs/bfm.md) | Actor, critics, B, and D; RL training; what differs from BFM-Zero; open questions |
| [Development notes](docs/development.md) | Full repository layout, test coverage, validation results, detailed limitations |
| [Module notes](model/README.md) | Technical notes on the encoder code |

## Repository layout

```text
model/                     library code
  modeling_da_get.py         DaGetEncoder and Physics Cross Attention
  modeling_bfm.py            backward map, discriminator, actor, forward map, critics
  rl/                        FB-CPR agent, MuJoCo environments, replay buffers, training loop
  pretraining.py             physics pretraining model (heads A-F)
  dynamics.py                rigid-body dynamics and torque estimation
  training/, conversion/     data pipeline and dataset converters
scripts/                   training, data config generator, converters, validation
configs/data/              example data configs
docs/                      documentation by topic
tests/                     test suite (114 tests)
```

## Limitations

- **Robots must be tree-structured**, with inertials in the URDF. Closed
  kinematic loops are rejected, and Xacro and MJCF must be converted first.
- **Estimated torques are approximate:** 43–97% error on a real Go1 and 47–72%
  on ANYmal D. Prefer datasets with measured torques.
- **Simulated ground contact covers standing, not walking.** Locomotion comes
  from recorded datasets.
- **RL training runs MuJoCo on CPU**, without body collisions, fall starts, or
  rotor robots ([differences from BFM-Zero](docs/bfm.md#what-differs-from-bfm-zero)).
- **Automatic configs and actuator defaults are guesses.** Check them, and use
  measured values when you have them.

See [the full list](docs/development.md#limitations).

## License and attribution

The backward map, discriminator, truncated-normal policy, latent inference, and
FB-CPR training code are adapted from [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) (Copyright
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
