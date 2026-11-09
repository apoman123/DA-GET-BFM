# Behavior foundation model

[← Back to the README](../README.md)

The FB-CPR networks on top of the DA-GET encoder: what is kept from BFM-Zero, what changes, and what is still open.

## Contents

- [Design details](#design-details)
- [Usage](#usage)
- [Configuration](#configuration)
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

  The FB-CPR training losses can therefore be ported as they are.

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

## Open questions for cross-embodiment training

- **B and D are unchanged, so they still need one flat state layout.** For
  several robots, use either an embodiment-agnostic state (for example base
  and keypoint features in a shared frame) or a padded layout. Only then do
  all robots share one z space. Replacing B with a GET-based set encoder is a
  natural next step, but here it is deliberately left as in BFM-Zero.
- **Inputs to the GET tokens.** BFM-Zero's actor sees `state`, `last_action`,
  and a proprioceptive history. Here those go into the GET tokens: joint
  history via `DaGetTokenizer`, and the last applied torque via the dynamics
  features. Base-level signals can go in as context tokens.
- **What is not included.**
  - The training agent (replay buffers, FB, critic, discriminator, and actor
    updates, and target networks) is not part of this module. The update rules
    are those of FB-CPR / BFM-Zero.
  - BFM-Zero's observation normalizer (`BatchNorm` with momentum 0.01 on each
    observation key) is also left out. Normalize the flat states before
    passing them to B and D.

## Attribution

The B, D, and truncated-normal code is adapted from BFM-Zero (Meta's code, CC
BY-NC 4.0); see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
