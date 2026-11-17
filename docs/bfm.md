# Behavior foundation model

[← Back to the README](../README.md)

The FB-CPR networks on the DA-GET encoder, how they train across robots in Isaac Lab, and what is still open.

## Contents

- [Design details](#design-details)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training (online off-policy FB-CPR)](#training-online-off-policy-fb-cpr)
  - [How one training step works](#how-one-training-step-works)
  - [Data config: real and synthetic trajectories](#data-config-real-and-synthetic-trajectories)
  - [What is the same as BFM-Zero](#what-is-the-same-as-bfm-zero)
  - [What differs from BFM-Zero](#what-differs-from-bfm-zero)
  - [Tuning the encoder](#tuning-the-encoder)
  - [Sharing z across robots](#sharing-z-across-robots)
  - [First run on a GPU machine](#first-run-on-a-gpu-machine)
  - [What TensorBoard shows](#what-tensorboard-shows)
  - [Inference latency at 50 Hz](#inference-latency-at-50-hz)
- [Open questions for cross-embodiment training](#open-questions-for-cross-embodiment-training)
- [Attribution](#attribution)

This repo studies **cross-embodiment behavior foundation models**. It combines
two references:
- [BFM-Zero](https://lecar-lab.github.io/BFM-Zero/) (Li et al., 2025), a
  promptable humanoid controller trained with FB-CPR (Tirinzoni et al., ICLR
  2025) on forward-backward representations (Touati and Ollivier, NeurIPS
  2021);
- [GET-Zero](https://get-zero-paper.github.io/) (Patel and Song, ICRA 2025).

BFM-Zero's networks read one flat state vector of a single robot (the G1). Here
every network reads per-DoF tokens from one shared `DaGetEncoder`. Its state-token
adapter maps expert and privileged state tokens to the encoder input width;
the actor, F, B, D, and critics then consume encoder embeddings.

```text
policy tokens ────────────────────┐
state tokens ─► state adapter ───┤
                                ▼
                         DaGetEncoder (one instance)
                                │ per-joint embeddings
              ┌─────────────────┼─────────────────────┐
              ▼                 ▼                     ▼
           B and D           actor              F and critics
       pooled heads      per-joint actions    action-conditioned readout
```

| Network | BFM-Zero | Here |
|---|---|---|
| Backward map `B(s)` | MLP: Linear → LayerNorm → Tanh → Linear, normalized to radius `√z_dim` | Shared encoder output, a joint readout, then the same MLP head |
| Discriminator `D(s, z)` | MLP on `[z, s]`, reward `log D − log(1−D)` | Shared encoder output, a joint readout, then the same MLP on `[z, pooled]` |
| Actor `π(a \| s, z)` | Residual MLP on `[s, z]` → `tanh` mean, truncated normal | `DaGetBfmActor`: joint tokens cross-attend to latent tokens, then a per-DoF head gives a `tanh` mean and a truncated normal |
| Forward map `F(s, a, z)` | Residual MLP, ensemble of 2, `z_dim` outputs | `DaGetForwardMap`: each joint token gets its own action, cross-attends to latent tokens, then the joints are pooled; ensemble of `num_parallel` |
| Critic, aux critic `Q(s, a, z)` | Same as F with one output | `DaGetForwardMap(config, 1)` |

For state-only inputs, the adapter preserves privileged features. Expert
records provide each joint's mass diagonal but not the full joint-space inertia
matrix, so state encoding uses a diagonal matrix for the shared encoder's mass
bias. Policy observations continue to use their full mass matrix.
This changes B and D parameter shapes, so RL checkpoints made with their old
separate encoders need a new BFM training run. Pretrained DA-GET encoder
checkpoints remain usable.

In BFM-Zero's published training configuration, actor, F, and both critics
use **separate** residual networks with the same 2048-wide, six-layer setting.
The critics are instances of the forward-map architecture and output one Q
value per sample; the actor outputs an action distribution. The actor sees
policy observations, while F and the critics also see privileged observations.
This project replaces those residual MLPs with per-joint attention networks so
they can process robots with different joint counts. Its transformer widths and
depths are project presets, not the paper's 2048-wide residual architecture.
See the upstream [training configuration](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/train.py)
and [network definitions](https://github.com/LeCAR-Lab/BFM-Zero/blob/main/humanoidverse/agents/nn_models.py).

## Design details

- **Latent tokens.** A single z would give cross-attention just one key: the
  softmax would then be constant and attention could not pick out anything.
  So each latent is mapped to `num_latent_tokens` tokens, each with a learned
  slot embedding.
  - A latent sequence `[B, L, z_dim]` (for example `B(s)` of upcoming
    reference frames) also gets sinusoidal positions.
  - Critics also read each joint's state token as privileged context, in the same memory.
- **State tokens.** Each joint token has its normalized position and velocity and
  its frame position in the base frame, and repeats the base state. They have no
  history, actions, or noise, because motion data has none: otherwise D could spot
  simulated states by those signals alone.
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
# hidden, batch and goal from the README quick start, with Go1 as the prompted robot
config = BfmConfig(embedding_size=hidden.shape[-1], state_token_size=observer.state_token_size)
bfm = DaGetBfmModel(config, encoder)
bfm.init_weights()

z = bfm.goal_inference(goal).expand(2, -1)               # or tracking_inference / reward_inference
action = bfm.act(hidden, z, batch.joint_mask)            # [2, 19], one command per DoF, padded = 0
dist = bfm.actor(hidden, z, config.actor_std, batch.joint_mask)
F = bfm.forward_map(hidden, z, dist.sample(clip=0.3), batch.joint_mask)   # [2, 2, 256]
q_fb = bfm.fb_q_value(F, z)                              # Fᵀz, [2, 2]
q_cpr = bfm.critic(hidden, z, action, batch.joint_mask)  # [2, 2, 1]
goal_hidden = bfm.encode_state(goal)
reward = bfm.discriminator.compute_reward(goal_hidden, z[:1], goal["joint_mask"])
```

## Configuration

Choose sizes independently for the actor, forward map (including both critics),
and backward map. The backward preset also sizes the shared state-token adapter
and B/D readouts. `small` matches the original defaults. Explicit config
fields or CLI dimension flags override the selected presets.

```python
config = BfmConfig.from_sizes(
    embedding_size=hidden.shape[-1],
    state_token_size=observer.state_token_size,
    actor_size="tiny",
    forward_size="medium",
    backward_size="small",
)
```

```bash
python scripts/train_bfm.py --data_config configs/rl/go1_h1_ground.json \
    --output_dir runs/bfm --encoder_size tiny --actor_size tiny --forward_size medium \
    --backward_size small
```

`--encoder_size` selects the policy encoder only when no `--encoder_checkpoint`
is supplied. A checkpoint retains its saved encoder architecture.

| Size | Actor/forward width | Actor/forward layers | Heads | Feed-forward width | State adapter width/layers | Backward head width/layers |
|---|---:|---:|---:|---:|---:|---:|
| `tiny` | 128 | 2 | 4 | 512 | 64 / 1 | 128 / 1 |
| `small` | 256 | 4 | 4 | 1024 | 128 / 2 | 256 / 1 |
| `medium` | 384 | 6 | 8 | 1536 | 256 / 4 | 512 / 2 |
| `large` | 512 | 8 | 16 | 2048 | 512 / 6 | 1024 / 3 |

`BfmConfig` fields (BFM-Zero values are identified where retained):

| Field | Default | Meaning |
|---|---|---|
| `embedding_size` | required | `DaGetConfig.hidden_size` |
| `state_token_size` | required | `EmbodimentObserver.state_token_size` |
| `z_dim`, `norm_z` | 256, `True` | Latent size and its projection to the sphere |
| `hidden_size`, `num_attention_heads`, `intermediate_size` | 256, 4, 1024 | Actor and critic token width |
| `num_latent_tokens` | 8 | Tokens per latent vector |
| `actor_*`, `forward_*` dimension fields | `None` | Optional role-specific width, heads, feed-forward width, and latent-token count |
| `actor_layers`, `forward_layers` | 4, 4 | Latent-conditioned blocks |
| `num_parallel` | 2 | Ensemble size of F and the critics |
| `action_size_per_joint` | 1 | Action entries per DoF |
| `state_hidden_size`, `state_num_layers`, `state_num_heads`, `state_intermediate_size` | 128, 2, 4, 256 | State adapter width/depth, B/D readout heads, and adapter inner width |
| `backward_hidden_size`, `backward_hidden_layers`, `backward_norm` | 256, 1, `True` | B head (BFM-Zero) |
| `discriminator_hidden_size`, `discriminator_hidden_layers` | 1024, 3 | D head (BFM-Zero) |
| `actor_std`, `seq_length` | 0.05, 8 | Policy std, tracking window |

## Training (online off-policy FB-CPR)

`scripts/train_bfm.py` trains the networks the way BFM-Zero does: online and
off-policy, with one change for cross-embodiment training. **Each robot has its
own simulated worlds, replay buffer, and expert motions, and every gradient step
draws an equal share of each robot.**

The default simulator is **Isaac Lab**, which needs an NVIDIA GPU. `--simulator
mujoco` runs on a CPU for small checks of the pipeline.

> **Status.** The Isaac Lab adapter (`model/rl/isaac_env.py`) has **never run in Isaac
> Lab**: no GPU was available. It was written against Isaac Lab 3.0.0 (commit
> `4cfdfcf`), its names were checked against that source, and its logic is tested
> with a fake of the API. Expect to fix details on the first real run. See
> [First run on a GPU machine](#first-run-on-a-gpu-machine).

```bash
# 1. Pretrain the encoder (same joint_features and history_size as in step 2)
uv run python scripts/train_physics_da_get.py \
    --data_config configs/rl/g1_go1_real_and_synthetic.json --output_dir runs/encoder --max_steps 2000

# 2. Train the behavior foundation model, inside Isaac Lab's Python environment
python scripts/train_bfm.py \
    --data_config configs/rl/g1_go1_real_and_synthetic.json \
    --encoder_checkpoint runs/encoder --output_dir runs/bfm --headless

tensorboard --logdir runs
```

### How one training step works

```text
for every robot:  z from the rollout context ─► actor ─► targets
one simulation:   all robots advance together (Isaac Lab)
for every robot:  transition ─► that robot's replay buffer

every update:     equal share of each robot's replay + its expert motion windows
                    │
                    ├─ discriminator: expert windows vs replay states
                    ├─ one backward pass over the encoder and:
                    │    F, B: successor-measure Bellman loss + orthonormality of B
                    │    Q_D, Q_R: Bellman losses on discriminator and auxiliary rewards
                    │    physics heads A–F: pretraining losses on a dataset batch
                    ├─ actor: maximize Fᵀz + λ_D Q_D + λ_R Q_R (encoder does not move)
                    └─ soft-update every target network, including the target encoder
```

### Data config: real and synthetic trajectories

The script reads the same data config as pretraining. Sources with the same `robot`
name are **one robot**, so real and synthetic trajectories can be combined:

| Source key | Default | Meaning |
|---|---|---|
| `robot` | the source's name | Sources with one name share a robot; they must have the same joints in the same order |
| `expert` | true, false for `synthetic` | Whether the trajectories are expert data for D and prompts |

**Every source trains the encoder through the physics auxiliary losses. Only `expert`
sources teach the policy what to do.** Synthetic joint trajectories are random PD motion,
which is good for learning dynamics but is not behavior to imitate, so they are not
expert data unless you set `"expert": true`. Motions are resampled to 50 Hz and split
into 10 s chunks, as in BFM-Zero.

`configs/rl/g1_go1_real_and_synthetic.json` is an example: a recorded G1 dataset plus
synthetic G1 data, and simulated Go1 standing data plus synthetic Go1 data. Replace the
sources with the datasets you choose; [Training data](data.md#recommended-training-datasets)
lists candidates. `configs/rl/go1_h1_ground.json` is a small example without downloads.

These keys of a robot's first expert source configure its worlds:

| Key | Default | Meaning |
|---|---|---|
| `contact_links` | required for floating robots | Feet |
| `nominal_pose` | zero, clamped to the limits | Joint targets at zero action; set this to a standing pose |
| `gain_scale` | 1.0 | PD gains: `K_p = 2 · gain_scale · τ^peak`, `K_d = 0.03 K_p` |
| `action_scale` | 1.25 | Target offset in rad at `\|action\| = 1` (BFM-Zero: 0.25 × 5) |
| `armature` | 0.01 | Rotor inertia per joint in kg m². URDFs have none, and without it G1 diverged in MuJoCo. Use real motor values where you have them |
| `simulation_urdf` | `urdf` | Isaac Lab: URDF whose collision meshes resolve |
| `ros_package_paths` | none | Isaac Lab: `[{"name": ..., "path": ...}]` for `package://` meshes |
| `undesired_contact_links` | every link but the feet | Isaac Lab: regexes of penalized links (BFM-Zero's G1: pelvis, shoulders, hips) |
| `self_collisions` | true | Isaac Lab: enable self-collisions |
| `contact_geometry` | sphere of 2 cm | MuJoCo only: foot radius, `sole_offset`, `foot_size` |

If any motion of a robot lacks base height or linear velocity, the state tokens hide
that signal for that robot in simulation too, so the discriminator cannot use it to
spot simulated states.

### What is the same as BFM-Zero

Defaults come from BFM-Zero's published G1 configuration (`train.py` and Tables 1 and 4
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
| Environment | 200 Hz physics, 50 Hz PD control, 500-step episodes, **no termination on falling** |
| Fall starts | 30% of episodes start lying on the back or front, so the policy learns to get up |
| Motion sampling | Priority `2^(2 · clip(EMD, 0.5, 2))` from each motion's tracking EMD, updated every evaluation, for episode starts and expert windows |
| Randomization | Friction, link mass, base center of mass, nominal pose, pushes, sensor noise |
| Auxiliary rewards | Joint limits −10, action rate −0.1, foot slip −2, undesired contact −1, feet orientation −0.4 |
| Self-collision | On in Isaac Lab |
| Asymmetry | The actor sees no base linear velocity and has noisy sensors; critics also read the clean state |

### What differs from BFM-Zero

| Difference | Why |
|---|---|
| Actor and critics read `DaGetEncoder` embeddings | The point of this project |
| A shared state adapter lets B and D use the same encoder as the actor and F | Expert states lack policy history and have a different token width |
| Prompts pair only within a robot, each robot has a z buffer, and F and B are compared only within a robot | See [Sharing z across robots](#sharing-z-across-robots) |
| The encoder keeps training, with a target encoder and physics losses | Requested for this project |
| Gradient penalty on D's pooled state, not its raw input | Second derivatives through attention are expensive and unsupported by PyTorch's fast CPU attention. Only D's head is penalized |
| No observation `BatchNorm` | Tokens are already scaled per feature, and the encoder and adapter start with LayerNorm |
| Ankle-roll penalty left out | It names G1 joints |
| Replay is not saved in checkpoints | Resuming continues with an empty buffer |
| Rotor robots are not supported | Not needed yet; `NotImplementedError` |

### Tuning the encoder

`DaGetEncoder` starts from the pretrained checkpoint and keeps training:

- **F, Q_D, Q_R** update it, so it learns what the values need.
- **Physics heads A–F** update it on batches of the data config, so it stays faithful
  to the dynamics.
- **The actor does not update it**, so the actor cannot move the critics' inputs.
- **A target encoder** (soft update, rate `fb_target_tau`) feeds the Bellman targets.

| Flag | Default | Meaning |
|---|---|---|
| `--train_encoder` | true | False keeps the encoder frozen |
| `--encoder_rl_gradients` | true | False leaves only the physics losses on the encoder |
| `--physics_loss_weight` | 1.0 | Weight of the physics losses; 0 turns them off |
| `--lr_encoder` | `3e-5` | Encoder learning rate |
| `--lr_physics_heads` | `1e-4` | Physics heads' learning rate |
| `--physics_batch_size` | 64 | Physics batch size |

These learning rates and weights are not from BFM-Zero, which has no encoder. They are
starting points and have not been tuned. The checkpoint folder holds the tuned encoder
in `encoder/`, loadable with `from_pretrained`.

### Sharing z across robots

B and D use one set of weights for every robot, and their inputs do not depend on joint
count. F and B are compared only within a robot (`--within_robot_pairs`, default true).
Otherwise the off-diagonal loss would push embeddings of different robots apart, since
one robot can never reach another's state, and B would split z space by robot.

This allows z to be shared, **but nothing yet makes "the same behavior" on two robots
land on the same z.** A Go1 motion and an H1 motion have no training signal tying them
together, so prompting one robot with another's motion is untested and may not work.
Testing it needs an evaluation across robots, and probably an alignment signal such as
pairs of motions retargeted to both robots.

### First run on a GPU machine

1. Install [Isaac Lab 3](https://isaac-sim.github.io/IsaacLab/), then this repo inside
   its Python environment (`pip install -e .`).
2. Get URDFs with collision meshes. `scripts/download_urdfs.py` fetches only the URDF
   files, so clone the robot's description package and point `ros_package_paths` at it
   (for Unitree robots, [unitree_ros](https://github.com/unitreerobotics/unitree_ros)).
3. Start small and check each point below before scaling up:

```bash
python scripts/train_bfm.py --data_config configs/rl/go1_h1_ground.json \
    --output_dir runs/check --headless --online_parallel_envs 16 --num_env_steps 2000 \
    --num_seed_steps 320 --batch_size 256 --log_every 320 --eval_every 1000 --eval_motions 2
```

- Robots appear with feet on the floor, and worlds do not touch.
- A standing robot with zero actions holds its pose, and PD gains suit the robot.
- Joint and link names resolve (the adapter raises a clear error otherwise). The
  importer must keep fixed joints, so that feet remain separate links.
- `undesired_contact` is 0 while standing, then 1 when lying down or in self-contact.
- `rollout/<robot>/fallen_fraction` is about 0.3 at the start of training.
- Friction randomization uses a PhysX-only call (`root_view.set_material_properties`),
  so another physics backend needs a change there.

### What TensorBoard shows

| Tag | Meaning |
|---|---|
| `train/fb_loss`, `fb_diag`, `fb_offdiag`, `orth_loss` | F and B |
| `train/disc_loss`, `disc_expert_loss`, `disc_train_loss`, `mean_disc_reward` | Discriminator |
| `train/critic_loss`, `aux_critic_loss`, `Q_discriminator`, `Q_aux`, `Q_fb`, `actor_loss` | Critics and actor |
| `train/physics/loss`, `physics/loss_fk`, … | Physics auxiliary losses of the encoder |
| `train/aux_rew/*` | Each unscaled auxiliary reward |
| `train/frames_per_second`, `gradient_steps` | Throughput |
| `rollout/<robot>/fallen_fraction`, `episode_length` | How much of the time each robot is down |
| `eval/<robot>/tracking_mpjpe`, `tracking_emd`, `tracking_fallen` | Tracking: the paper's `mean_t ‖q_t − q_ref‖`, the EMD behind priorities, and worlds fallen at the end |
| `eval/<robot>/goal_mpjpe` | Goal reaching toward each motion's last frame |
| `eval/<robot>/getting_up_rate` | Worlds upright at the end after a lying start, without pushes or noise |

### Inference latency at 50 Hz

`scripts/benchmark_inference.py` times every part of the model on one robot and checks
the per-step totals against the 20 ms budget of 50 Hz control. Run it on the deployment
machine:

```bash
uv run python scripts/benchmark_inference.py --device cuda --robots g1 h1 go1 \
    --encoder_sizes small medium large
```

The first table is the work of every control step, at batch 1:

| Part | What it runs | Where |
|---|---|---|
| `observation` | Policy tokens, dynamics features and mass matrix (`EmbodimentObserver.policy_physics`) | CPU, float64 |
| `act (eager)` | `DaGetFbCprAgent.act`, the training path | `--device` |
| `act (policy)` | `DaGetBfmPolicy.act`, the deployment path (below) | `--device` |
| `reference state` | State view of a reference frame, for a motion streamed live | CPU, float64 |
| `B (eager)`, `B (policy)` | B of that frame | `--device` |
| `control step` | `observation` + `act (policy)` | |
| `+ live tracking` | `control step` + `reference state` + `B (policy)` | |

A total is `OK` when its 99th percentile and maximum are within budget, `OK (max over)`
when only the maximum is over, and `OVER BUDGET` otherwise. A motion known ahead
needs no live tracking: encode it once with `DaGetFbCprAgent.tracking_latents`.

The second table is everything else: each network alone (the policy and state encoders,
the actor and B heads, and F, Q_D, Q_R and D, which only training runs) and the one-off
prompts (goal z, reward z over 1000 buffer states, tracking z of a 500-frame motion).

#### Deployment: `DaGetBfmPolicy`

```python
from model.rl import DaGetBfmPolicy

policy = DaGetBfmPolicy(agent.model, observer, device="cuda")
dynamics, mass = observer.policy_physics(sensed_state, previous_torque)
action = policy.act(local_history, global_history, dynamics, mass, z)  # [1, J] on the CPU
b = policy.backward(observer.state_view(reference_state))             # raw B(s), [1, z_dim]
```

At batch 1 the networks are bound by kernel launches, not arithmetic: on an RTX 5070 Ti,
G1's `act (eager)` grew only from 7.9 to 12.1 ms between the small and large encoders.
`DaGetBfmPolicy` removes most of that overhead:

- All inputs live in one device buffer, filled with one copy from pinned host memory;
  the action comes back in one copy.
- The embodiment-only part of Physics Cross Attention is computed once
  (`build_physics_memory`), not every step.
- On a GPU, the encoder and actor (and separately the state encoder and B) are recorded
  once as CUDA Graphs (`torch.cuda.CUDAGraph`) and replayed each step.
- A graph cannot record a wait for the device, so the encoder runs with
  `validate_inputs=False` and the mass matrix is checked on the CPU before the copy.
  `tests/test_deployment.py` checks that the recorded functions make no such call.

Call `policy.refresh()` after loading new weights; using a policy built before the
change raises a stale-memory error.

What keeps `observation` small on the CPU:

- Forward kinematics and Newton-Euler run one tree level at a time (G1: 10 levels
  instead of 29 joints), and `dynamics_terms` gets g(q) and c(q, qd) from one batched
  pass that shares the kinematics with M(q). G1 went from 16.6 to 5.6 ms on one CPU
  thread.
- The encoder checks and normalizes the mass matrix once per forward instead of in
  every layer: 2 device waits per forward instead of 18 in the training path.

## Open questions for cross-embodiment training

- **Prompting across robots.** B and D are shared, but nothing aligns behaviors between
  robots yet. See [Sharing z across robots](#sharing-z-across-robots).
- **Inputs to the GET tokens.** BFM-Zero's actor sees `state`, `last_action`,
  and a proprioceptive history. Here those go into the GET tokens: joint
  history via `DaGetTokenizer`, and the last applied torque via the dynamics
  features. Base-level signals can go in as context tokens.
- **Tuning.** The shared encoder's learning rate, physics weight, and state
  adapter have no reference values.
- **Offline training.** Training needs a simulator, as in BFM-Zero.

## Attribution

The B, D, truncated-normal, and FB-CPR training code is adapted from BFM-Zero (Meta's code, CC
BY-NC 4.0); see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
