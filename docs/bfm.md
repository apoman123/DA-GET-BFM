# Behavior foundation model

[← Back to the README](../README.md)

The FB-CPR networks on top of the DA-GET encoder: what is kept from BFM-Zero, what changes, and what is still open.

## Contents

- [Design details](#design-details)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training (online off-policy FB-CPR)](#training-online-off-policy-fb-cpr)
  - [How one training step works](#how-one-training-step-works)
  - [Data config: one source per robot](#data-config-one-source-per-robot)
  - [What is the same as BFM-Zero](#what-is-the-same-as-bfm-zero)
  - [What differs from BFM-Zero](#what-differs-from-bfm-zero)
  - [Expected speed](#expected-speed)
  - [What TensorBoard shows](#what-tensorboard-shows)
- [Open questions for cross-embodiment training](#open-questions-for-cross-embodiment-training)
- [Attribution](#attribution)

This repo studies **cross-embodiment behavior foundation models**. It combines
two references:
- [BFM-Zero](https://lecar-lab.github.io/BFM-Zero/) (Li et al., 2025), a
  promptable humanoid controller trained with FB-CPR (Tirinzoni et al., ICLR
  2025) on forward-backward representations (Touati and Ollivier, NeurIPS
  2021);
- [GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025).

BFM-Zero's networks read one flat state vector of a single robot (the G1).
Here the actor and the critics read the per-DoF embeddings of `DaGetEncoder`.
Those embeddings already carry the graph, the hardware features, and (with
Physics Cross Attention) the dynamics. As a result, one policy outputs one
action per joint for any robot the encoder can tokenize.

```text
                 flat state s ──► BackwardMap B(s) ──► z  (goal / tracking / reward prompt)
                 flat state s ─┐                       │
                               └► Discriminator D(s,z) ◄┤
                                                       ▼
URDF + q, q̇, … ─► DaGetEncoder ─► H [B, J, 512] ─► LatentMemory(z, context) ─► K/V tokens
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
| Actor `π(a \| s, z)` | Residual MLP on `[s, z]` → `tanh` mean, truncated normal | `DaGetBfmActor`: joint tokens cross-attend to latent tokens, then a per-DoF head gives a `tanh` mean and a truncated normal |
| Forward map `F(s, a, z)` | Residual MLP, ensemble of 2, `z_dim` outputs | `DaGetForwardMap`: each joint token gets its own action, cross-attends to latent tokens, then the joints are pooled; ensemble of `num_parallel` |
| Critic, aux critic `Q(s, a, z)` | Same as F with one output | `DaGetForwardMap(config, 1)` |

## Design details

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

  The FB-CPR training losses therefore apply as they are (see [Training](#training-online-off-policy-fb-cpr)).

## Usage

```python
import torch

from model import BfmConfig, DaGetBfmModel

# hidden, batch from the README quick start; replace the states with your own (normalized)
state_dim = 64
goal_state, state = torch.randn(2, state_dim), torch.randn(2, state_dim)

config = BfmConfig(embedding_size=hidden.shape[-1], backward_input_size=state_dim)
bfm = DaGetBfmModel(config)
bfm.init_weights()

z = bfm.goal_inference(goal_state)                       # or tracking_inference / reward_inference
action = bfm.act(hidden, z, batch.joint_mask)            # [2, 19], one command per DoF, padded = 0
dist = bfm.actor(hidden, z, config.actor_std, batch.joint_mask)
F = bfm.forward_map(hidden, z, dist.sample(clip=0.3), batch.joint_mask)   # [2, 2, 256]
q_fb = bfm.fb_q_value(F, z)                              # Fᵀz, [2, 2]
q_cpr = bfm.critic(hidden, z, action, batch.joint_mask)  # [2, 2, 1]
reward = bfm.discriminator.compute_reward(state, z)      # CPR reward
```

## Configuration

`BfmConfig` fields (defaults follow BFM-Zero's G1 configuration where one
exists):

| Field | Default | Meaning |
|---|---|---|
| `embedding_size` | required | `DaGetConfig.hidden_size` |
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

## Training (online off-policy FB-CPR)

`scripts/train_bfm.py` trains the networks the way BFM-Zero does: online and
off-policy, with one change for cross-embodiment training. **Each robot has its
own simulator, replay buffer, and expert motions, and every gradient step draws
an equal share of each robot.**

```bash
uv sync --all-extras
uv run python scripts/download_urdfs.py

# 1. Pretrain the encoder (use the same joint_features and history_size as in step 2)
uv run python scripts/train_physics_da_get.py \
    --data_config configs/rl/go1_h1_ground.json --output_dir runs/encoder --max_steps 2000

# 2. Train the behavior foundation model
uv run python scripts/train_bfm.py \
    --data_config configs/rl/go1_h1_ground.json \
    --encoder_checkpoint runs/encoder --output_dir runs/bfm

uv run tensorboard --logdir runs
```

### How one training step works

```text
for each robot:   z from the rollout context ─► actor ─► MuJoCo ─► that robot's replay buffer
every update:     equal share of each robot's replay + its expert motion windows
                    │
                    ├─ frozen DA-GET encoder, per robot, then pad
                    ├─ discriminator: expert windows vs replay states
                    ├─ F and B: successor-measure Bellman loss + orthonormality of B
                    ├─ Q_D, Q_R: Bellman losses on the discriminator and auxiliary rewards
                    ├─ actor: maximize Fᵀz + λ_D Q_D + λ_R Q_R
                    └─ soft-update the target networks
```

### Data config: one source per robot

The script reads the same data config as pretraining. Each entry of `sources` is
one robot, and **its motions are the expert data**, as LAFAN1 is for BFM-Zero. Any
source type works, so several datasets of different robots go in one file:

```json
{
  "history_size": 3,
  "joint_features": "da_get_zero",
  "sources": [
    {
      "type": "huggingface", "name": "g1", "repo_id": "unitreerobotics/G1_WBT_...",
      "urdf": "../../assets/urdf/g1_29dof.urdf", "floating_base": true, "...": "...",
      "contact_links": ["left_ankle_roll_link", "right_ankle_roll_link"],
      "contact_geometry": {"radius": 0.01, "sole_offset": [0.03, 0, -0.03], "foot_size": [0.18, 0.06]},
      "nominal_pose": {"left_knee_joint": 0.3}
    },
    {
      "type": "huggingface", "name": "go1", "path": "../../data/mipo",
      "urdf": "../../assets/urdf/go1.urdf", "floating_base": true, "...": "...",
      "contact_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
      "nominal_pose": {"FR_thigh_joint": 0.8, "FR_calf_joint": -1.5}
    }
  ]
}
```

These keys configure the robot's environment:

| Key | Default | Meaning |
|---|---|---|
| `contact_links` | required for floating robots | Links that collide with the floor |
| `contact_geometry` | sphere of 2 cm | Foot radius, `sole_offset`, `foot_size` |
| `nominal_pose` | zero, clamped to the limits | Joint targets at zero action; set this to a standing pose |
| `gain_scale` | 1.0 | PD gains: `K_p = 2 · gain_scale · τ^peak`, `K_d = 0.03 K_p` |
| `action_scale` | 1.25 | Target offset in rad at `\|action\| = 1` (BFM-Zero: 0.25 × 5) |

Motions are resampled to the 50 Hz control rate. If any motion of a robot lacks
base height or linear velocity, the flat state hides that signal for that robot
in simulation too, so the discriminator cannot use it to spot simulated states.

`configs/rl/go1_h1_ground.json` is a runnable example. Its motions come from the
`mujoco` source, so they are standing and weight shifting rather than real
motion; replace them with recorded datasets for research runs.

### What is the same as BFM-Zero

Defaults come from BFM-Zero's published G1 configuration (`train.py` and Table 1
of the paper):

| Setting | Value |
|---|---|
| Learning rates | F, actor, Q_D, Q_R `3e-4`; B, D `1e-5` |
| Target rates | F and B `0.01`; critics `0.005` |
| Losses | `ortho_coef` 100, discount 0.98, `reg_coeff` 0.05, `reg_coeff_aux` 0.02, gradient penalty 10, pessimism 0.5 |
| z mixture | 20% replay goals, 60% expert windows, 20% uniform; 80% of transitions relabeled |
| Collection | 1,024 environments, 16 gradient steps per 1,024 frames, batch 1,024, 5.12M transitions |
| Latent | `z_dim` 256, window `seq_length` 8, `actor_std` 0.05 |
| Rollouts | z changes every 100 steps; half of the environments track expert motions for 250 steps |
| Environment | 200 Hz physics, 50 Hz PD control, 500-step episodes |
| Randomization | Friction, link mass, base center of mass, nominal pose, pushes, sensor noise (Table 4) |
| Asymmetry | The actor does not see the base linear velocity; critics also read the privileged flat state |

### What differs from BFM-Zero

| Difference | Why |
|---|---|
| Actor and critics read `DaGetEncoder` embeddings | The point of this project |
| Goal and expert prompts pair only with transitions of the same robot, and each robot has its own z buffer | A motion of one robot says nothing about another's states |
| **MuJoCo on CPU instead of IsaacLab** | Works from any URDF in this repo, but is far slower than GPU simulation |
| **Only feet collide; falling ends the episode** | The generated model has no body collisions, so there are no fall starts (BFM-Zero: 30%) and no recovery from the ground |
| **Uniform motion sampling** | BFM-Zero prioritizes motions by tracking error (EMD) |
| **Three auxiliary rewards: joint limits, action rate, foot slip** | The others are G1-specific or need self-collision |
| **Replay is not saved in checkpoints** | Resuming continues with an empty buffer |
| The encoder stays frozen | A choice here, not in BFM-Zero: joint training would also need a target encoder |
| Rotor robots are not supported yet | The environment has no thrust actuators; `NotImplementedError` |

### Expected speed

Training is too slow on a CPU to reach BFM-Zero's scale. On a 32-core CPU without a
GPU, Go1 and H1 with reduced networks (`hidden_size` 64, batch 256, 64 environments,
4 gradient steps per 128 frames) ran at about 30 frames per second, or about one
gradient step per second. BFM-Zero's 192M frames and 3M gradient steps would take
months at that rate. Most of the time goes to the networks, so a GPU should help
(`--device cuda`, not measured here). The default network sizes are larger and
slower than this run.

### What TensorBoard shows

| Tag | Meaning |
|---|---|
| `train/fb_loss`, `fb_diag`, `fb_offdiag`, `orth_loss` | F and B |
| `train/disc_loss`, `disc_expert_loss`, `disc_train_loss`, `mean_disc_reward` | Discriminator |
| `train/critic_loss`, `aux_critic_loss`, `Q_discriminator`, `Q_aux`, `Q_fb`, `actor_loss` | Critics and actor |
| `train/aux_rew/*` | Each unscaled auxiliary reward |
| `train/frames_per_second`, `gradient_steps` | Throughput |
| `rollout/<robot>/episode_length` | How long each robot stays up while training |
| `eval/<robot>/tracking_mpjpe`, `goal_mpjpe` | Paper's error `mean_t ‖q_t − q_ref‖`, without randomization or noise |
| `eval/<robot>/tracking_survival`, `goal_survival` | Fraction of worlds that stay up |

## Open questions for cross-embodiment training

- **B and D are unchanged, so they need one flat state layout.** Training uses
  a padded state per robot: base signals, then joint positions, velocities, valid
  mask, and joint-frame positions (`model/rl/observation.py`). So all robots share
  a z dimension and a policy, and **a prompt must come from the robot being
  controlled**: a joint slot of one robot means nothing for another. Prompting one
  robot with another's motion would need an embodiment-independent state, or a
  GET-based B; this is the main open question.
- **Inputs to the GET tokens.** BFM-Zero's actor sees `state`, `last_action`,
  and a proprioceptive history. Here those go into the GET tokens: joint
  history via `DaGetTokenizer`, and the last applied torque via the dynamics
  features. Base-level signals can go in as context tokens.
- **Freezing the encoder.** Embeddings come from the pretrained encoder and are
  not updated by RL. Fine-tuning it is untested.
- **Offline training.** Training needs a simulator, as in BFM-Zero. Learning only
  from recorded datasets, without one, is not implemented.

## Attribution

The B, D, truncated-normal, and FB-CPR training code is adapted from BFM-Zero (Meta's code, CC
BY-NC 4.0); see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
