# Physics pretraining

[← Back to the README](../README.md)

The six auxiliary tasks that train the encoder, how to run them with the Hugging Face `Trainer`, and the data config format.

## Contents

- [Physics pretraining](#physics-pretraining)
- [Training with the Hugging Face Trainer](#training-with-the-hugging-face-trainer)
  - [Watching training in TensorBoard](#watching-training-in-tensorboard)
  - [Data config](#data-config)

## Physics pretraining

`DaGetForPhysicsPretraining` (`model/pretraining.py`) is a Hugging Face
`PreTrainedModel`, so `save_pretrained`, `from_pretrained`, and `push_to_hub`
all work. It wraps the encoder with six heads. `N_e` is the number of valid
joints of a sample's robot. Every per-joint loss is averaged per robot first,
so a 29-DoF humanoid does not outweigh a 4-rotor drone.

| Task | Head input | Target | Loss |
|---|---|---|---|
| **A. Forward kinematics** | `H_i` | `p_i / L_e` (`L_e` = robot size) and `R_i ∈ SO(3)` of every joint frame in the base frame | `1/N_e Σ λ_p ‖p̂−p‖² + λ_R d_SO(3)(R̂, R)²`; `fk_orientation=False` gives the position-only first pass |
| **B. Forward dynamics** | `[H_i; E_τ(τ_i^applied)]` passed through `mixing_layers` of graph action mixing | `qdd_t`; for a floating base also the base linear and angular acceleration (from the pooled mixed tokens) | `1/N_e Σ ((qdd̂ − qdd) / σ_i^a)²` + `λ_base` · mean of squared base-acceleration errors normalized per robot |
| **C. Inverse dynamics** | `[H_i; E(qdd_i^desired, base acceleration^desired)]` with graph mixing | `τ^required` = inverse dynamics, with contacts for legged robots | `1/N_e Σ ((P(τ̂ − τ))_i / (τ_i^cont + ε))²`; `P = I` for measured or simulated torques, the motion-determined projector for estimated torques in contact |
| **D. Motor feasibility** | `[H_i; E(a_i^proposal)]` | peak and continuous feasibility flags; margins for peak torque `1 − \|τ\|/τ^max(qd)`, continuous torque, power, velocity, and joint limit | `λ_f BCE + λ_m Huber` |
| **E. Action effect** | `[H_i; E(a_i)]` with graph mixing | at end-effector frames, in the base frame at `t`: `Δp`, `R_{t+1}`, `v_{t+1}`, `ω_{t+1}` | `λ_p‖Δp̂−Δp‖² + λ_R d_SO(3)² + λ_v‖v̂−v‖² + λ_ω‖ω̂−ω‖²` |
| **F. RBD consistency** (optional) | the FD prediction | `r = M qdd̂ + c + g + τ_f − τ^applied` | `1/N_e ‖W_e r‖²` with `W_e = 1/(τ^cont + ε)`; fixed-base samples only |

Notes on the design:

- **B: per-joint inputs are not enough.** Seeing only its own torque, a joint
  cannot represent `M⁻¹`'s coupling across joints, or how the base reacts. The
  FD, ID, and effect heads therefore start with `mixing_layers` (default 1) of
  graph self-attention over `[H_i; E(x_i)]`. Set it to 0 for the purely
  per-joint baseline.
- **D: requested torques and limits.** The requested effort comes from the
  proposal through the actuator mode:

  | Mode | Requested effort |
  |---|---|
  | torque | the proposal itself |
  | position | `K_p (a − q) − K_d qd` |
  | velocity | `K_d (a − qd)` |
  | thrust (rotor) | `K_d (a − qd) + k_m a\|a\|` |

  The torque-speed envelope is `τ^max(qd) = min(τ^peak, P^max/|qd|)` below
  `ω^max`, and zero above it. "Peak feasible" means executable now;
  "continuous feasible" also respects `τ^cont`. A short motor boost is
  therefore not mistaken for a sustainable command. The predictor does not
  replace hard limits in the motor interface.
- **E: normalization and frames.** `Δp` and `v` are divided by the robot size
  `L_e`. Each of `Δp`, `v`, and `ω` is then divided by its RMS over that
  robot's data, so translation and rotation terms have comparable size.
  Targets are in the base frame at time `t`, so a walking robot's or a drone's
  own motion is part of the effect. Rotations use the geodesic distance on
  SO(3). Only end-effector frames are supervised (leaf joints by default, or
  `end_effectors`).
- **F: off by default.** With analytic labels it largely duplicates `L_FD`, so
  its weight is 0 by default. Treat it as an ablation.

Weights: `--fk_weight`, `--fd_weight`, `--id_weight`, `--motor_weight`,
`--effect_weight`, `--rbd_weight`, plus the `lambda_*` fields of
`DaGetPhysicsPretrainingConfig` (including `lambda_base_acceleration`).

## Training with the Hugging Face Trainer

```bash
uv run python scripts/download_urdfs.py

# 1. Synthetic Go1, H1, and G1, each with fixed-base and floating-base episodes
uv run python scripts/train_physics_da_get.py \
    --data_config configs/data/synthetic_unitree.json \
    --output_dir runs/synthetic --max_steps 2000 \
    --per_device_train_batch_size 64 --learning_rate 5e-4 --warmup_steps 100

# 2. Real floating-base humanoid data (Unitree G1 whole-body teleoperation) + synthetic G1
uv run python scripts/train_physics_da_get.py \
    --data_config configs/data/g1_wbt_floating.json --output_dir runs/g1_wbt

# 3. Multirotor flight (synthetic quadrotor with randomized size and mass)
uv run python scripts/train_physics_da_get.py \
    --data_config configs/data/quadrotor_flight.json --output_dir runs/quadrotor
```

Several data configs can be merged into one by listing all their sources, for
example to pretrain on legged, humanoid, and flying robots together. Every
`TrainingArguments` flag works, including multi-GPU runs through
`accelerate launch`. The script adds model flags:
- `--preset small|da_get_zero`, plus width, depth, and head overrides;
- `--physics_cross_attention false` for an ablation;
- `--physics_layers 0,2`, `--mass_encoding`, `--mixing_layers`;
- `--fk_orientation false`, and the task weights.

The Trainer logs `loss_fk`, `loss_fd`, `loss_id`, `loss_motor`, `loss_effect`,
`loss_rbd`, and the matching `eval_loss_*` next to the total loss.
`runs/<name>` then holds a `from_pretrained` checkpoint and `embodiments.json`.
Load the encoder for downstream use with:

```python
from model.pretraining import DaGetForPhysicsPretraining
model = DaGetForPhysicsPretraining.from_pretrained("runs/g1_wbt")
embeddings = model.encode(**batch)          # [B, J, hidden]; or model.encoder directly
```

### Watching training in TensorBoard

The script records every run in TensorBoard, under
`<output_dir>/runs/<date>_<host>/`. While it trains, or afterwards:

```bash
uv run tensorboard --logdir runs    # then open http://localhost:6006
```

Pointing `--logdir` at the parent folder lets you compare several runs on one
chart, for example with and without Physics Cross Attention.

| Tag | Meaning |
|---|---|
| `train/loss`, `eval/loss` | Weighted total loss |
| `train/loss_fk`, `_fd`, `_id`, `_motor`, `_effect`, `_rbd` | Mean of each task loss since the last log |
| `eval/loss_fk`, `_fd`, `_id`, `_motor`, `_effect`, `_rbd` | Each task loss on held-out episodes |
| `train/learning_rate`, `train/grad_norm`, `train/epoch` | Optimizer state |
| `eval/runtime`, `eval/samples_per_second`, `eval/steps_per_second` | Evaluation speed |

The **Text** tab holds the training arguments and the model config of the run.

The script changes these `TrainingArguments` defaults so that the curves are
useful without extra flags:

| Flag | Script default | Why |
|---|---|---|
| `--report_to` | `tensorboard` | `none` turns it off; other integrations such as `wandb` also work |
| `--logging_steps` | 50 | The `transformers` default of 500 leaves few points on a short run |
| `--eval_strategy`, `--eval_steps` | `steps`, 500 | Gives a held-out curve during training. Evaluation is skipped when the data config has no eval split |

### Data config

A data config is JSON. Relative paths start from the config file.

```json
{
  "history_size": 3,
  "joint_features": "da_get_zero",
  "eval_fraction": 0.05,
  "allow_estimated_torque_for_fd": true,
  "use_estimated_previous_torque": true,
  "max_estimate_residual": 0.5,
  "estimated_torque_weight": 0.3,
  "sources": [ { "type": "synthetic", ... }, { "type": "huggingface", ... } ]
}
```

**`"type": "synthetic"`** simulates a URDF robot with the analytic dynamics.
Each of `randomized_embodiments` variants perturbs:
- link masses and inertias;
- link lengths (inertia is scaled consistently);
- damping and friction, as fractions of each joint's own limits, so the same
  ranges suit a finger, a knee, and a propeller;
- torque and speed limits.

`base_modes` is cycled over the variants:

| Mode | What is simulated |
|---|---|
| `fixed` | Base bolted down; gravity tilted per episode; PD with gravity feed-forward tracks random joint targets. |
| `floating` | The base is free and hangs from a harness that holds the whole-body center of mass against gravity, with no moment. Base rotation and base-limb coupling are fully dynamic. PD gains use the free-base effective inertia. |
| `flight` | A multirotor flies under its own rotor thrust: a geometric controller tracks random waypoints and maps thrust and moments to rotor speeds through the actual rotor layout (any rotor count). It needs every actuated joint to be a `thrust` rotor. |

Commands are clipped to the torque-speed envelope before they are applied, so
labels always use the executed torque, never an unexecuted proposal. Each state
also gets `probe_torques` extra torques (log-uniform from 2% to 200% of peak,
random sign), each simulated one step. This tests action conditioning at a
fixed pose and gives the motor head infeasible proposals. This source does not
simulate ground contact; use the
[`mujoco` source](data.md#ground-contact-in-synthetic-data-mujoco) or recorded
datasets for that.

**`"type": "huggingface"`** reads any Hub dataset (`repo_id`) or local folder
or file (`path`) in the LeRobot parquet layout, or any parquet, CSV, or JSON
table with one vector column per signal. Any column key can be a list of
columns that are concatenated in order.

| Key | Meaning |
|---|---|
| `urdf`, `joint_names` | URDF joint of each recorded state entry, in order. Unlisted joints are frozen (`frozen_joints` gives their positions, default 0). |
| `state_key`, `state_indices` | Position columns and which entries to use. A null index means the position is not recorded (rotors); it is integrated from the velocity. |
| `velocity_key`, `torque_key` | Optional columns. Without velocities, positions are finite-differenced. Without torques, torques are [estimated](dynamics.md#estimating-joint-torques-without-torque-sensors). |
| `action_key`, `action_indices`, `action_type` | Commands (`position`, `velocity`, or `torque`; rotor joints keep the `thrust` mode). Set `actuator_defaults.kp`/`kd` for position control so requested torques are meaningful. |
| `state_scale`, `state_offset` | `q_urdf = scale · q + offset`, for sign, unit (degrees), or zero-offset differences. |
| `floating_base` | The base is free (legged robots, drones). |
| `base_orientation_key` (+ `_indices`, `base_orientation_format`: `wxyz`, `xyzw`, `rpy`, `matrix`), `base_position_key` | Base pose in the world. |
| `base_linear_velocity_key`, `base_angular_velocity_key` (+ `_frame`: `base` or `world`) | Base twist; IMU gyros give angular velocity in the base frame. |
| `projected_gravity_key`, `imu_acceleration_key` | Unit gravity direction and accelerometer specific force, both in the base frame. |
| `contact_links`, `contact_key`, `contact_threshold`, `surface_contacts`, `contact_height_tolerance`, `contact_speed_tolerance` | Feet, recorded contact flags or forces, and the detection heuristic. |
| `fps`, `episode_key`, `frame_key`, `max_episodes`, `smoothing_window`, `gravity`, `end_effectors` | Timing, grouping, size, smoothing, a constant gravity in the base frame, task-E frames. |

Processed features and labels are cached in `.cache/physics_data/<hash>.pt`.
The hash covers the config and the URDF contents, so the analytic dynamics run
once per dataset rather than once per epoch. Episodes, not frames, are split
between train and eval.
