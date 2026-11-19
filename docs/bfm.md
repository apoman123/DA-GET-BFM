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
  - [Actuator torque limits](#actuator-torque-limits)
  - [Tuning the encoder](#tuning-the-encoder)
  - [Physics losses during RL](#physics-losses-during-rl)
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
for every robot:  z from the rollout context ─► actor ─► targets ─► ActionMap (torque envelope)
one simulation:   all robots advance together (Isaac Lab)
for every robot:  transition ─► that robot's replay buffer

every update:     equal share of each robot's replay + its expert motion windows
                    │
                    ├─ discriminator: expert windows vs replay states
                    ├─ one backward pass over the encoder and:
                    │    F, B: successor-measure Bellman loss + orthonormality of B
                    │    Q_D, Q_R: Bellman losses on discriminator and auxiliary rewards
                    │    physics heads A, B, C, R, R⁻¹, D, E: pretraining losses on a batch
                    │      from the data config or (physics replay) from the rollouts
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
| `armature` | 0.01 | Rotor inertia per joint in kg m², one value or `{joint glob: value}`. URDFs have none, and without it G1 diverged in MuJoCo. The RL robot's graph gets the same value, so its physics tokens, the physics-replay labels and both simulators agree. Use real motor values where you have them |
| `simulation_urdf` | `urdf` | Isaac Lab: URDF whose collision meshes resolve |
| `ros_package_paths` | none | Isaac Lab: `[{"name": ..., "path": ...}]` for `package://` meshes |
| `undesired_contact_links` | every link but the feet | Isaac Lab: regexes of penalized links (BFM-Zero's G1: pelvis, shoulders, hips) |
| `self_collisions` | true | Isaac Lab: enable self-collisions |
| `explicit_actuators` | true | Isaac Lab: compute the PD torque at every physics step inside the torque-speed envelope (`envelope_actuator_cfg`, an `IdealPDActuator`), which also gives the physics replay exact torque labels ([limits](#actuator-torque-limits), [replay](#physics-losses-during-rl)); false uses PhysX's implicit PD with only a peak clip. Check stability at your physics step |
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
| Auxiliary rewards | Joint limits −10, action rate −0.1, foot slip −2, undesired contact −1, feet orientation −0.4 (three torque penalties are added here, [below](#actuator-torque-limits)) |
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
| Targets pass through `ActionMap`, which keeps the requested effort inside each actuator's torque-speed envelope | BFM-Zero relies on the simulator's constant effort limit only; see [Actuator torque limits](#actuator-torque-limits) |
| The simulators clip the PD torque to the torque-speed envelope at every physics step | A motor driver does; BFM-Zero's simulator clips at a constant effort |
| Three torque penalties (limit, continuous load, power), with per-robot weights adapted by Lagrange multipliers | Normalized by each robot's own limits, with data-sheet targets instead of tuned weights |
| No torque in any observation | BFM-Zero's G1 has no torque input either; earlier versions of this repo fed the last torque, now removed ([why](dynamics.md#why-training-uses-no-estimated-torque)) |
| Physics losses on the policy's own rollouts (physics replay) | Keeps the encoder's dynamics faithful on the states the policy visits |

### Actuator torque limits

The actor outputs normalized actions; a PD drive turns the resulting position targets
into torque. A policy that asks for far more torque than a motor has saturates it,
heats it, or behaves differently on hardware. Four layers handle this, and none needs a
torque measurement on the robot. All of them use one definition of the envelope,
`model.physics.envelope_limit`:

```text
τ^max(qd) = min(τ^peak, P^max / |qd|)   for |qd| ≤ ω^max,   0 beyond
```

**1. The simulators apply the envelope at every physics step.** A motor driver limits
current inside its own fast loop, so the torque available changes with speed within a
control step. The simulators do the same:

| Simulator | How | Setting |
|---|---|---|
| MuJoCo (`MujocoEmbodimentEnv`) | `motor` actuators; before every `mj_step` the environment computes `clip(K_p (a − q) − K_d qd, ±τ^max(qd))` with `model.physics.pd_envelope_torque` | `EnvConfig.substep_envelope` (default true; false uses MuJoCo's `position` actuators with only the constant peak clip) |
| Isaac Lab (`IsaacEmbodimentEnv`) | an `IdealPDActuator` subclass (`envelope_actuator_cfg`) whose effort clip is `model.physics.envelope_clip`, evaluated by Isaac Lab at every physics step | `"explicit_actuators"` on the source (default true; false uses `ImplicitActuatorCfg` with only the constant peak clip) |

Without this, a joint near its no-load speed still received up to peak torque within a
control step: on the test quadruped with full random actions, MuJoCo's peak clip alone
exceeded the envelope by up to 10 N m (the full peak), and the per-physics-step clip by
0 (`test_every_physics_step_stays_inside_the_envelope`). The explicit actuators also give
the physics replay its exact torque labels ([below](#physics-losses-during-rl)). Explicit
PD with stiff gains needs a small enough physics step; legged_gym (Rudin et al., CoRL 2021)
computes PD torques explicitly at 200 Hz, and the MuJoCo tests (including the light G1
wrists) are stable at 200 Hz with the default rotor inertia. Check new robots before a
long run.

**2. `ActionMap` keeps the request inside the envelope at the control rate.**
[`model/rl/actions.py`](../model/rl/actions.py) maps an action to a target as BFM-Zero
does (`nominal + action_scale · a`, clipped to the joint limits) and then moves each
position-mode target so the effort the PD law requests at the current state,
`K_p (a − q) − K_d qd`, stays within `τ^max(qd)` (`model.physics.envelope_targets`; the
same `torque_limit` defines the motor feasibility labels of task D). The moved target is
`q + (clip(τ) + K_d qd) / K_p`. The environments call it every control step
(`EnvConfig.clip_to_envelope`, `--clip_to_envelope`, default true), and **the robot must
call the same code**, so the policy is trained on exactly the targets it will send:

```python
from model.rl import ActionMap

action_map = ActionMap(record, nominal_pose, action_scale=1.25)   # the training values
targets, raw = action_map(actions, q, qd)    # numpy [N, J]; send `targets` to the drives
```

The observation's previous-action features use the sent (clipped) target, in training
and on the robot alike.

**3. Torque penalties shape the policy.** `TorqueMonitor` adds three auxiliary rewards,
each a sum over actuated joints and dimensionless, so one weight fits every robot. It
receives the applied torque of every physics step (`[N, S, J]`), and averages loads and
powers over them:

| Reward | Formula | Starting weight |
|---|---|---|
| `penalty_torque_limit` | `relu(\|τ_req\| − τ^max(qd)) / τ^peak` of the actor's raw targets, before `ActionMap` clips them: how far the actor asks beyond the envelope | −2.0 |
| `penalty_torque_continuous` | `relu(EMA[(τ / τ^cont)²] − 1)`, the load averaged over `thermal_time_constant` (1 s, `EnvConfig`): sustained torque above the continuous rating, a thermal proxy | −0.5 |
| `penalty_power` | `relu(\|τ qd\| − P^max) / P^max` | −0.5 |

`τ` is the simulator's applied torque (exact there; never a policy input). The penalties
enter Q_R with BFM-Zero's penalties through `aux_reward_scales`.

**4. The penalty weights adapt per robot (Lagrange multipliers).** Fixed weights would
need tuning: too weak and the policy rides the limits and overheats the motors, too
strong and it moves timidly and cannot get up. Instead each penalty has a target from
the data sheet on a measured metric, and its weight is the starting weight times a
per-robot multiplier `λ` ([`model/rl/constraints.py`](../model/rl/constraints.py)),
updated before every update round by multiplicative dual ascent:

```text
λ ← clip(λ · exp(rate · (metric − target) / target), 0.1, 10)
```

| Penalty | Metric (mean over the rollout since the last update) | Target |
|---|---|---|
| `penalty_torque_limit` | `torque_request_over_envelope`: share of joints whose request exceeds `τ^max(qd)` | 0.05 |
| `penalty_torque_continuous` | `torque_load_over_continuous`: the largest thermal load `EMA[(τ/τ^cont)²]` | 1.0 |
| `penalty_power` | `power_over_limit`: share of joints and physics steps above `P^max` | 0.02 |

A robot that violates a target gets a growing weight until it complies; one that
complies easily gets a shrinking weight, so it is not held back. This is the Lagrangian
relaxation of a constrained MDP (Altman 1999; Tessler et al., RCPO, ICLR 2019; Stooke et
al., PID Lagrangian, ICML 2020), with the multipliers in log space so they stay positive
and scale-free. The agent keeps one row of reward weights per robot
(`DaGetFbCprAgent.set_reward_scales`), and checkpoints keep the multipliers.

| Setting (`FbCprConfig`) | Default | Meaning |
|---|---|---|
| `adaptive_torque_penalties` (`--adaptive_torque_penalties`) | true | False keeps the starting weights fixed |
| `penalty_multiplier_rate` (`--penalty_multiplier_rate`) | 0.05 | Step size per update round, relative to the target |
| `penalty_multiplier_bounds` | (0.1, 10) | Multiplier range (Python only) |
| `torque_targets` | the table above | `{reward: [metric, target]}` (Python only) |

TensorBoard shows, per robot: `rollout/<robot>/torque_request_over_envelope` (share of
joints whose raw request exceeds the envelope), `torque_at_envelope` (share of joints and
physics steps at ≥ 99% of it), `torque_rms_over_continuous`, `torque_load_over_continuous`
(the largest load average), `power_over_limit`, and `multiplier/<penalty>` (the current
multipliers).

**Limits of this design.** The envelope is symmetric: braking torque (opposite to the
velocity) is limited like driving torque, although many drives brake up to peak torque.
This is conservative (the simulated robot brakes less than the real one), and the motor
labels of task D use the same convention. The targets are data-sheet style choices, not
measured on a robot. The motor heads of task D learn the envelope, but they do not
constrain actions.

### Tuning the encoder

`DaGetEncoder` starts from the pretrained checkpoint and keeps training:

- **F, Q_D, Q_R** update it, so it learns what the values need.
- **Physics heads** (A, B, C, R, R⁻¹, D, E) update it on batches of the data config and
  of the policy's own rollouts, so it stays faithful to the dynamics.
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

### Physics losses during RL

Offline physics batches cover the states their sources visit. The simulated ground
source only stands, crouches, shifts its weight and recovers from pushes, because
generating gaits would need a locomotion controller for every robot, and that
controller is what this model is meant to learn. The **physics replay**
([`model/rl/physics_replay.py`](../model/rl/physics_replay.py)) closes the gap with the
states the policy itself visits (walking, running, getting up), with exact labels.

**What is recorded.** The first `physics_replay_envs` environments of each robot are its
**label worlds** (`EnvConfig.label_worlds`). For them, the environment reports at the
start of every control step, after that step's push and with that step's targets applied
(`model.rl.envs.StepStart`):

- the state: joint positions and velocities, gravity in the base frame, base twist;
- the joint torque the simulator applies over the **first physics step**;
- the joint accelerations and base accelerations (classical linear, then angular, base
  frame) it produces at that same instant, contacts, joint damping, friction and limits
  included;
- the foot contacts, and whether a push changed the velocity at this step.

Torque and acceleration therefore belong to one instant. The earlier design recorded the
state at the end of each control step and the control-rate torque; over the four
physics steps of a control step the PD torque changes with the state, so that pair was
not one instant's, and B and C were left out. Taking both from the start of the step
removes the mismatch without recording every physics step.

| Simulator | How the labels are taken | Exact |
|---|---|---|
| MuJoCo (`MujocoEmbodimentEnv`) | `mj_forward` before the first `mj_step`: `qfrc_actuator` and `qacc` of the state | yes; checked against the graph's floating-base dynamics to 1e-6 in flight (`test_label_worlds_have_exact_start_labels_in_flight`) |
| Isaac Lab, `"explicit_actuators": true` (default) | the envelope actuator computes the PD torque, PhysX applies it for the first physics step; acceleration is `(v₁ − v₀)/physics_dt` of that step, base terms rotated into the start base frame | the torque is exact; the acceleration is PhysX's velocity change with that torque, equal to the instantaneous acceleration up to PhysX's integrator and solver tolerance. Unverified on a real Isaac Lab (no GPU here) |
| Isaac Lab, `"explicit_actuators": false` | none: PhysX solves the PD law internally, and Isaac Lab's reported `applied_torque` is the explicit formula kept as telemetry, not the torque PhysX applied | — (kinematic episodes; the trainer prints a note) |

Label worlds keep the **nominal link masses and base center of mass** (friction, pushes
and the nominal-pose offset stay randomized), so their labels belong to the robot their
physics tokens describe. The other worlds keep full randomization.

**Labels of a replay episode.** Each chunk of `physics_replay_length` control steps
becomes a processed episode (`process_episode`):

| Task | Label world (exact episode) | Other worlds or implicit Isaac Lab (kinematic episode) |
|---|---|---|
| A. FK | exact kinematics | exact kinematics |
| B. Forward dynamics | **simulator's `q̈` (and base acceleration) for the applied torque**, torque source `exact`, weight 1 | — |
| C. Inverse dynamics | **simulator's applied torque**, projected onto the motion-determined part when feet are down | — |
| R, R⁻¹ | `G(q) δτ` with the simulator's contact set | same |
| D | sent targets (PD request computed) and random proposals | same |
| E | next frame, conditioned on the sent target | same |

A frame of an exact episode is the start of a step, so a push at step `t+1` ends the
transition of frame `t`. A frame of a kinematic episode is the end of the previous step,
so a push at step `t` ends frame `t`'s transition. Either way the interrupted transition
gets no effect label (`transition_valid`), since the push is not part of the dynamics.
An exact chunk has no final end-of-chunk frame, because that state has no labels yet.

What remains approximate: the MuJoCo models are built with stiff contacts
(`solimp` 0.99–0.999), with which MuJoCo differs from the rigid-contact response of R by
about 3% (20% with MuJoCo's default soft contacts); B and C are MuJoCo's own forces and
accelerations and do not depend on it. Frames where a foot in contact slides faster than
`max_contact_slip` (0.1 m/s) get no response labels, as in the offline data
([details](dynamics.md#assumptions-and-checks)). Joint-limit forces enter the
accelerations without a feature that describes them, as in the offline simulated data.

A share `physics_replay_fraction` (default 0.5) of the physics batches comes from the
replay once it has an episode; up to two queued chunks are processed before each update
round, and `physics_replay_episodes` (default 32) episodes per robot are kept.

| Flag | Default | Meaning |
|---|---|---|
| `--physics_replay` | true | Record rollouts for the physics losses |
| `--physics_replay_fraction` | 0.5 | Share of physics batches from the replay |
| `--physics_replay_envs` | 4 | Environments recorded per robot; they become the label worlds |
| `--physics_replay_length` | 200 | Control steps per recorded episode |
| `--physics_replay_episodes` | 32 | Episodes kept per robot |
| `--clip_to_envelope` | true | `ActionMap` keeps the request inside the torque envelope |

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
- `rollout/<robot>/torque_request_over_envelope` falls as training proceeds, and
  `torque_load_over_continuous` stays near or below 1 for a standing robot.
- Friction randomization uses a PhysX-only call (`root_view.set_material_properties`),
  so another physics backend needs a change there.

### What TensorBoard shows

| Tag | Meaning |
|---|---|
| `train/fb_loss`, `fb_diag`, `fb_offdiag`, `orth_loss` | F and B |
| `train/disc_loss`, `disc_expert_loss`, `disc_train_loss`, `mean_disc_reward` | Discriminator |
| `train/critic_loss`, `aux_critic_loss`, `Q_discriminator`, `Q_aux`, `Q_fb`, `actor_loss` | Critics and actor |
| `train/physics/loss`, `physics/loss_fk`, `loss_response`, … | Physics auxiliary losses of the encoder |
| `train/aux_rew/*` | Each unscaled auxiliary reward, including `penalty_torque_limit`, `penalty_torque_continuous`, `penalty_power` |
| `rollout/<robot>/torque_request_over_envelope`, `torque_at_envelope`, `torque_rms_over_continuous`, `torque_load_over_continuous`, `power_over_limit` | Torque-limit use ([details](#actuator-torque-limits)) |
| `rollout/<robot>/multiplier/<penalty>` | Lagrange multiplier of each torque penalty's weight (1 = starting weight) |
| `physics_replay/episodes` | Rollout episodes available to the physics losses |
| `physics_replay/exact_episodes` | Those with exact torque labels (label worlds of MuJoCo, or Isaac Lab with explicit actuators) |
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
| `observation` | Policy tokens, dynamics features and mass matrix (`EmbodimentObserver.policy_physics`); no torque input | CPU, float64 |
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

Measured on an RTX 5070 Ti, policy path with CUDA Graphs, batch 1 (the CUDA Graph test of
`tests/test_deployment.py` passed on the same machine). These numbers predate the
torque-free schema, whose dynamics features have 5 columns instead of 7 and whose
observation no longer takes the previous torque; the work per step is the same or
slightly less, but rerun the benchmark to record current numbers:

| Robot | Encoder | `act` eager → policy (median) | `control step` median / p99 / max | `+ live tracking` median / p99 / max |
|---|---|---|---|---|
| G1 (29 DoF) | small | 7.74 → 1.20 ms | 2.77 / 5.34 / 5.37 ms | 5.28 / 10.40 / 10.43 ms |
| G1 (29 DoF) | large | 11.27 → 2.01 ms | 3.66 / 6.22 / 6.25 ms | 7.00 / 12.36 / 12.60 ms |
| H1 (19 DoF) | large | 12.04 → 1.94 ms | 3.05 / 4.76 / 4.76 ms | 5.72 / 8.92 / 9.43 ms |
| Go1 (12 DoF) | small | 7.18 → 0.94 ms | 1.81 / 3.06 / 3.09 ms | 3.28 / 6.87 / 8.82 ms |

Every robot and encoder size (small, medium, large) is within budget, the maximum
included. The large encoder costs about 0.9 ms more per step than the small one, so
latency does not limit the encoder size on this GPU. The CPU parts (`observation`,
`reference state`) are now about half of a live-tracking step, and their p99 is two to
three times their median; on a robot, pin the control thread to a core and pause
Python's garbage collector in the loop. An onboard computer (such as a Jetson) is
slower than this desktop: run the benchmark there before deploying.

#### Deployment: `DaGetBfmPolicy`

```python
from model.rl import DaGetBfmPolicy

from model.rl import ActionMap, DaGetBfmPolicy

policy = DaGetBfmPolicy(agent.model, observer, device="cuda")
action_map = ActionMap(observer.record, nominal_pose, action_scale=1.25)  # same as training

local, global_features = observer.policy_step(sensed_state, previous_target, previous_q, previous_qd)
# ... append them to local_history / global_history (oldest first) ...
dynamics, mass = observer.policy_physics(sensed_state)                 # no torque needed
action = policy.act(local_history, global_history, dynamics, mass, z)  # [1, J] on the CPU
targets, _ = action_map(action.numpy(), q, qd)                         # send these to the drives
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
  history via `DaGetTokenizer`, and the last action as its target and the PD
  effort it requests. No torque measurement enters. Base-level signals can go in
  as context tokens.
- **Contact compliance.** The response labels assume rigid, sticking contacts. The
  generated MuJoCo models use stiff contacts (3% from the rigid response instead of
  20%), and sliding frames are dropped; compliant real feet (rubber pads, soft ground)
  remain a model error of a few percent, like an inaccurate URDF inertia.
- **Tuning.** The shared encoder's learning rate, physics weight, and state
  adapter have no reference values.
- **Offline training.** Training needs a simulator, as in BFM-Zero.

## Attribution

The B, D, truncated-normal, and FB-CPR training code is adapted from BFM-Zero (Meta's code, CC
BY-NC 4.0); see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
