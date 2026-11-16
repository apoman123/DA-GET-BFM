# Physics pretraining

[← Back to the README](../README.md)

The six auxiliary tasks that train the encoder, how to run them with the Hugging Face `Trainer`, and the data config format.

## Contents

- [Physics pretraining](#physics-pretraining)
  - [Shared notation and loss reduction](#shared-notation-and-loss-reduction)
  - [A. Forward kinematics](#a-forward-kinematics)
  - [B. Forward dynamics](#b-forward-dynamics)
  - [C. Inverse dynamics](#c-inverse-dynamics)
  - [D. Motor feasibility](#d-motor-feasibility)
  - [E. Action effect](#e-action-effect)
  - [F. Rigid-body dynamics consistency](#f-rigid-body-dynamics-consistency)
- [Training with the Hugging Face Trainer](#training-with-the-hugging-face-trainer)
  - [Watching training in TensorBoard](#watching-training-in-tensorboard)
  - [Data config](#data-config)

## Physics pretraining

`DaGetForPhysicsPretraining` (`model/pretraining.py`) is a Hugging Face
`PreTrainedModel`, so `save_pretrained`, `from_pretrained`, and `push_to_hub`
all work. It wraps the encoder with six heads. $N_e$ is the number of valid
joints of a sample's robot. Every per-joint loss is averaged per robot first,
so a 29-DoF humanoid does not outweigh a 4-rotor drone.

| Task | Head input | Target | Loss |
|---|---|---|---|
| **A. Forward kinematics** | $H_i$ | $p_i/L_e$ ($L_e$ = robot size) and $R_i\in SO(3)$ of every joint frame in the base frame | $\frac{1}{N_e}\sum_i[\lambda_p\lVert\hat p_i-p_i\rVert^2+\lambda_R d_{SO(3)}(\hat R_i,R_i)^2]$; `fk_orientation=False` omits orientation |
| **B. Forward dynamics** | $[H_i;E_\tau(\tau_i^{\mathrm{applied}})]$ passed through `mixing_layers` of graph action mixing | $\ddot q_t$; for a floating base also base linear and angular acceleration | $\frac{1}{N_e}\sum_i[(\hat{\ddot q}_i-\ddot q_i)/\sigma_i^a]^2$, plus $\lambda_{\mathrm{base}}$ times normalized base-acceleration MSE |
| **C. Inverse dynamics** | $[H_i;E(\ddot q_i^{\mathrm{desired}},a_{\mathrm{base}}^{\mathrm{desired}})]$ with graph mixing | $\tau^{\mathrm{required}}$ from inverse dynamics, including contacts for legged robots | $\frac{1}{N_e}\sum_i[(P(\hat\tau-\tau))_i/(\tau_i^{\mathrm{cont}}+\varepsilon)]^2$; $P=I$ for measured or simulated torques |
| **D. Motor feasibility** | $[H_i;E(a_i^{\mathrm{proposal}})]$ | Peak and continuous feasibility flags; peak torque, continuous torque, power, speed, and joint-position margins | $\lambda_f\operatorname{BCEWithLogits}+\lambda_m\operatorname{Huber}$ |
| **E. Action effect** | $[H_i;E(a_i)]$ with graph mixing | At end-effector frames, in the base frame at $t$: $\Delta p$, $R_{t+1}$, $v_{t+1}$, $\omega_{t+1}$ | $\lambda_p\lVert\widehat{\Delta p}-\Delta p\rVert^2+\lambda_R d_{SO(3)}(\hat R,R)^2+\lambda_v\lVert\hat v-v\rVert^2+\lambda_\omega\lVert\hat\omega-\omega\rVert^2$ |
| **F. RBD consistency** (optional) | The FD prediction | $r=M\hat{\ddot q}+c+g+\tau_f-\tau^{\mathrm{applied}}$ | $\frac{1}{N_e}\sum_i[r_i/(\tau_i^{\mathrm{cont}}+\varepsilon)]^2$; fixed-base samples only |

### Shared notation and loss reduction

Each training sample $b$ contains one robot state and one action variant.
$i$ indexes a joint, $J_b$ is the set of its valid joints, and $N_b=|J_b|$.
$H_{b,i}$ is the encoder output. A hat marks a prediction; the unmarked
quantity is its training label. $q$, $\dot q$, and $\ddot q$ are joint
position, velocity, and acceleration. $a$ is a proposed command.
$E(\cdot)$ in the summary table denotes a learned projection of an action
or acceleration input before graph mixing. $SO(3)$ is the set of
three-dimensional rotation matrices.

**Actuator torque or force** means torque in N m for a revolute/continuous
joint, or linear force in N for a prismatic joint. It is distinct from a
position target in rad or m. For position control,
a target $a_i$ requests the PD torque or force
$\tau_i^{\mathrm{request}}=K_{p,i}(a_i-q_i)-K_{d,i}\dot q_i$,
where $K_{p,i}$ and $K_{d,i}$ are the position and velocity gains.
$\tau^{\mathrm{request}}$ is this pre-limit demand;
$\tau^{\mathrm{applied}}$ is the actuator output after its limits and
controller dynamics. The applied value comes from simulation or an actuator
measurement when available; for recordings without such measurements, the
data pipeline uses an approximate inverse-dynamics estimate. The requested
and applied values can differ when the actuator saturates.

$\tau^{\mathrm{peak}}$ and $\tau^{\mathrm{cont}}$ are short-term and
sustainable actuator torque or force limits. $L_e$ is the robot's
characteristic length.
$\varepsilon=10^{-6}$ stabilizes torque or force divisions in the data labels;
the pretraining configuration also exposes an epsilon for loss calculations.

For a per-joint loss $\ell_{b,i}$, let $m_{b,i}\in\{0,1\}$ select valid
joint labels and $w_b$ weight confidence in sample $b$. The implemented
reduction averages valid joints first, then nonempty samples:

$$
\bar\ell_b =
 \frac{\sum_i m_{b,i}\ell_{b,i}}{\max(1,\sum_i m_{b,i})},
\qquad
\mathcal A(\ell,m,w)=
 \frac{\sum_{b:\,\sum_i m_{b,i}>0}w_b\bar\ell_b}
 {\sum_{b:\,\sum_i m_{b,i}>0}w_b}.
$$

If no weights are supplied, $w_b=1$; an entirely empty mask or zero total
confidence weight yields zero.
FD and ID use $w_b=1$ for measured/exact simulated torque and
$w_b=0.3$ by default for an inverse-dynamics estimate. The total objective
is $\mathcal L=\sum_{k}\alpha_k\mathcal L_k$, where $k$ runs over A–F
and $\alpha_k$ is the configured task weight: 1 for A–E and 0 for F by
default. Every coefficient denoted $\lambda$ below defaults to 1.

### A. Forward kinematics

**Why this loss exists.** The observation token describes a joint relative
to its parent and includes the current joint state. The label is the joint
frame's pose relative to the root, which requires composing transforms
along the whole kinematic chain. Predicting it tests whether the encoder
combines graph structure and joint configuration. This can help downstream
heads reason about where a foot or hand is. The target is exactly computable
from the URDF and $q$, so this is a geometry supervision task, not new
sensor information; its downstream benefit must be checked by ablation.

**Meaning and target.** This makes $H_i$ encode the pose of its joint
frame. Forward kinematics on the URDF supplies position $p_i^{\mathrm{root}}$
and orientation $R_i\in SO(3)$ in the root frame. The position label
$p_i=p_i^{\mathrm{root}}/L_e$ normalizes robot size. The head predicts
$\hat p_i$ and $\hat R_i$; six output numbers are converted to the rotation
matrix $\hat R_i$.

$$
\ell_i^{\mathrm{FK}}=
 \lambda_p\lVert\hat p_i-p_i\rVert_2^2+
 \mathbf1_{\mathrm{orientation}}\lambda_R
 d_{SO(3)}(\hat R_i,R_i)^2,\qquad
d_{SO(3)}(\hat R,R)=
 \arccos\!\left(\operatorname{clip}
 \frac{\operatorname{tr}(\hat R^\top R)-1}{2}\right).
$$

$\lambda_p$ and $\lambda_R$ weight position and orientation.
$\mathbf1_{\mathrm{orientation}}$ is 0 when the orientation task is disabled,
otherwise 1. The clip keeps the arccos argument in
$[-1+10^{-6},1-10^{-6}]$.
$\mathcal L_{\mathrm{FK}}=\mathcal A(\ell^{\mathrm{FK}},m^{\mathrm{joint}},1)$,
where $m^{\mathrm{joint}}$ selects real, nonpadded joints.

### B. Forward dynamics

**Why this loss exists.** Two robots can receive the same command and
accelerate differently because their masses, inertia, coupling, and
actuator limits differ. Predicting acceleration from the *executed*
torque or force encourages the encoder to retain those physical
differences. Graph mixing matters because the inverse mass matrix couples
the action at one joint to motion at other joints; a floating base can move
as well. The loss may help the shared representation anticipate the
consequences of BFM actions. Its supervision is weaker where executed
torque or force is only estimated from recorded motion and contacts.

**Meaning and target.** This teaches how applied torque or force accelerates
the joints and, for floating robots, the base. The head receives $H_i$ and
$u_i=\operatorname{clip}(\tau_i^{\mathrm{applied}}/
(\tau_i^{\mathrm{peak}}+\varepsilon),-c,c)$, where $c=5$ is the
FD torque or force input clip. Graph mixing shares actuator commands between
joints. It predicts
$\hat y_i$ for label $y_i=\ddot q_i/\sigma_i^a$; $\sigma_i^a$ is that
robot's per-joint acceleration RMS with a positive floor.

$$
\mathcal L_{\mathrm{FD}}=
 \mathcal A((\hat y_i-y_i)^2,m^{\mathrm{FD}},w)
 +\lambda_{\mathrm{base}}\frac{1}{|B_{\mathrm{base}}|}
 \sum_{b\in B_{\mathrm{base}}}\frac{1}{6}
 \sum_{r=1}^{6}(\hat y^{\mathrm{base}}_{b,r}
                 -y^{\mathrm{base}}_{b,r})^2.
$$

$r$ indexes the three linear and three angular base-acceleration components.
$y^{\mathrm{base}}$ is each component divided by its robot-level RMS;
$\hat y^{\mathrm{base}}$ is the pooled-head prediction.
$B_{\mathrm{base}}$ contains rows with known base acceleration and eligible
applied torque or force; the second term is omitted when this set is empty.
$\lambda_{\mathrm{base}}$ controls its weight. $m^{\mathrm{FD}}$ selects
actuated joints with measured or exact torque, or estimated torque when
estimated labels are allowed and their contact residual passes the threshold.

### C. Inverse dynamics

**Why this loss exists.** Forward dynamics asks what motion follows an
applied torque or force; inverse dynamics asks what torque or force a
desired motion would require. This gives a capability-oriented signal:
the same acceleration may be easy for one robot and near an actuator
limit for another. It can discourage a representation that explains
motion without encoding how costly the motion is to produce. On real
contact-rich recordings, different unobserved contact forces can explain
the same motion. The projection and reliability mask reduce that
ambiguity but cannot make estimated torques exact.

**Meaning and target.** This teaches the torque or force required for a desired
acceleration. The head receives $H_i$, desired $\ddot q_i/\sigma_i^a$,
and six normalized desired base-acceleration values. Its prediction
$\hat z_i$ and label $z_i$ are torque or force values normalized by the
continuous limit:

$$
z_i=\frac{\tau_i^{\mathrm{required}}}
         {\tau_i^{\mathrm{cont}}+\varepsilon},\qquad
e_i=\frac{
 [P((\hat z-z)\odot(\tau^{\mathrm{cont}}+\varepsilon))]_i}
 {\tau_i^{\mathrm{cont}}+\varepsilon},\qquad
\mathcal L_{\mathrm{ID}}=\mathcal A(e_i^2,m^{\mathrm{ID}},w).
$$

$\tau^{\mathrm{required}}$ is the inverse-dynamics label.
$\odot$ denotes elementwise multiplication. $P$ is the motion-determined
torque projector for estimated torques in contact; $P=I$ for measured or
exact simulated torques. $m^{\mathrm{ID}}$ selects actuated joints with
reliable labels. Estimated frames with an unexplained base wrench above
the configured fraction of robot weight are excluded. Projection removes
unobservable internal-force directions, not measurement/model error.

### D. Motor feasibility

**Why this loss exists.** A position target inside the joint-angle limits
can still demand more torque or force than the motor can produce.
Feasibility labels connect a proposed command to the actuator's
short-term and sustainable limits. The continuous flag separates a
command that is briefly executable from one that can be sustained.
This is relevant to BFM's angle-target actions, but the pretraining
head does not clip BFM actions at runtime. Its input already contains
the requested torque or force calculated from the command and gains;
a low motor loss alone therefore does **not** prove that the encoder or
actor independently learned to use $K_p$ and $K_d$.

**Meaning and target.** Given $H_i$ and a proposed action, predict whether
the actuator can execute it. The three action features are requested torque
or force divided by its peak limit, normalized position target, and target
velocity divided by the speed limit. Unused command types have zero in the
corresponding slot.
For position control, $\tau_i^{\mathrm{request}}=
K_{p,i}(a_i-q_i)-K_{d,i}\dot q_i$. The labels are two feasibility flags
and five margins:

| Command mode | Requested torque or force |
|---|---|
| Torque | $a_i$ |
| Position | $K_{p,i}(a_i-q_i)-K_{d,i}\dot q_i$ |
| Velocity | $K_{d,i}(a_i-\dot q_i)$ |
| Thrust rotor | $K_{d,i}(a_i-\dot q_i)+k_{m,i}a_i\lvert a_i\rvert$ |
| Passive | $0$ |

Here $K_{p,i}$ and $K_{d,i}$ are position and velocity gains, and
$k_{m,i}$ is the rotor drag coefficient. The dataset's action type
chooses the mode, except passive and thrust joints retain their own
mode. The requested torque or force is calculated before actuator limiting.

$$
\begin{aligned}
m_i^{\mathrm{peak}}&=1-
 \frac{|\tau_i^{\mathrm{request}}|}
 {\tau_i^{\max}(\dot q_i)+\varepsilon},&
m_i^{\mathrm{cont}}&=1-
 \frac{|\tau_i^{\mathrm{request}}|}
 {\tau_i^{\mathrm{cont}}+\varepsilon},\\
m_i^{\mathrm{power}}&=1-
 \frac{|\tau_i^{\mathrm{request}}\dot q_i|}
 {P_i^{\max}+\varepsilon},&
m_i^{\mathrm{speed}}&=1-
 \frac{|\dot q_i|}{\omega_i^{\max}+\varepsilon},\\
m_i^{\mathrm{position}}&=
 \frac{\min(q_i-q_i^{\min},q_i^{\max}-q_i)}
 {(q_i^{\max}-q_i^{\min})/2}.
\end{aligned}
$$

Here $\tau_i^{\max}(\dot q_i)=
\min(\tau_i^{\mathrm{peak}},P_i^{\max}/|\dot q_i|)$ up to speed
$\omega_i^{\max}$, and zero above it; division at zero speed is stabilized.
$P_i^{\max}$ is motor power, $\omega_i^{\max}$ its speed limit, and
$q_i^{\min},q_i^{\max}$ joint position limits. Each margin is clipped to
$[-3,1]$. The position margin is undefined for continuous joints.
The peak flag is 1 when peak, power, speed, and any defined position
margins are nonnegative; the continuous flag additionally requires the
continuous margin to be nonnegative.

$$
f_i^{\mathrm{peak}}
=\mathbf1[m_i^{\mathrm{peak}}\ge0]\,
 \mathbf1[m_i^{\mathrm{power}}\ge0]\,
 \mathbf1[m_i^{\mathrm{speed}}\ge0]\,
 \mathbf1[\text{position margin undefined or }m_i^{\mathrm{position}}\ge0],
\qquad
f_i^{\mathrm{cont}}
=f_i^{\mathrm{peak}}\mathbf1[m_i^{\mathrm{cont}}\ge0].
$$

$$
\ell_i^{\mathrm{motor}}=
 \lambda_f\frac{1}{2}\sum_{r=1}^{2}
 \operatorname{BCEWithLogits}(\hat f_{i,r},f_{i,r})
 +\lambda_m
 \frac{\sum_{k=1}^{5}v_{i,k}
 \operatorname{Huber}_{\delta}(\hat m_{i,k}-m_{i,k})}
 {\max(1,\sum_{k=1}^{5}v_{i,k})},\quad
\mathcal L_{\mathrm{motor}}=
 \mathcal A(\ell^{\mathrm{motor}},m^{\mathrm{actuated}},1).
$$

$f_{i,r}$ is a binary label and $\hat f_{i,r}$ its predicted logit;
$m_{i,k}$ is a target margin, $\hat m_{i,k}$ a prediction, and
$v_{i,k}$ says whether that margin is defined. $\delta$ is the Huber
transition point. $\lambda_f,\lambda_m$ weight flags and margins.
$m^{\mathrm{actuated}}$ excludes passive and padded joints.
For logit $x$ and binary label $y$, the binary cross-entropy term is
$\operatorname{BCEWithLogits}(x,y)=\log(1+e^x)-yx$.
For residual $r$, the Huber term is $r^2/2$ when $|r|\le\delta$, and
$\delta(|r|-\delta/2)$ otherwise.
Power, speed, and mode remain in the actuator spec for label generation,
although they are not encoder actuator features. This loss predicts
feasibility but does not enforce a runtime action constraint.

### E. Action effect

**Why this loss exists.** The task-level consequence of a joint action is
often a foot, hand, or rotor moving in space, not simply a value of
$\ddot q_i$. This head connects an action to the next end-effector pose
and velocity across the robot's kinematic chain, including known base
motion. It can make the shared embedding useful for goal-reaching and
motion-tracking prompts that refer to body motion. Unlike the analytic FK
label at one instant, this target requires a transition or a simulated
one-step action probe. Its quality depends on the recorded action,
next-state alignment, and contact modelling.

**Meaning and target.** Predict an action's next-step effect at selected
end-effector frames. $\Delta p_i=p_{i,t+1}-p_{i,t}$ is displacement;
$R_{i,t+1}$, $v_{i,t+1}$, and $\omega_{i,t+1}$ are next orientation,
linear velocity, and angular velocity. All are expressed in the base
frame at time $t$, including known base motion. Displacement and linear
velocity are divided by $L_e$ and then by their respective robot-level
RMS; angular velocity is divided by its robot-level RMS. In this equation
the symbols mean those normalized labels and predictions:

$$
\ell_i^{\mathrm{effect}}=
 \lambda_{\Delta p}\lVert\widehat{\Delta p}_i-\Delta p_i\rVert_2^2
 +\lambda_R d_{SO(3)}(\hat R_i,R_i)^2
 +\lambda_v\lVert\hat v_i-v_i\rVert_2^2
 +\lambda_\omega\lVert\hat\omega_i-\omega_i\rVert_2^2,\qquad
\mathcal L_{\mathrm{effect}}=
 \mathcal A(\ell^{\mathrm{effect}},m^{\mathrm{effect}},1).
$$

The four $\lambda$ values weight displacement, rotation, linear velocity,
and angular velocity. $m^{\mathrm{effect}}$ selects configured end-effectors
with a valid next state; the last trajectory frame has no such label.
The head receives the same action features as D and mixes them across joints.

### F. Rigid-body dynamics consistency

**Why this loss exists.** The predicted acceleration from B should also
satisfy the fixed-base equation of motion with the applied torque or
force. This residual is a physics consistency check on the *prediction*,
rather than another head with a separate label. With accurate analytic
FD labels it mostly repeats B, so its task weight defaults to zero.
The implemented residual omits floating-base and contact terms; applying
it to those samples would penalize physically valid motion.

**Meaning and target.** Check whether predicted acceleration satisfies
the fixed-base equation of motion. This optional loss is off by default
because it largely duplicates B when analytic labels are available.

$$
\hat{\ddot q}_i=\sigma_i^a\hat y_i,\qquad
r_i=\frac{[M(q)\hat{\ddot q}]_i+c_i(q,\dot q)+g_i(q)
 +\tau_{f,i}(\dot q)-\tau_i^{\mathrm{applied}}}
 {\tau_i^{\mathrm{cont}}+\varepsilon},\qquad
\mathcal L_{\mathrm{RBD}}=\mathcal A(r_i^2,m^{\mathrm{RBD}},1).
$$

$M(q)$ is the joint-space mass matrix, $c_i$ is Coriolis/centrifugal
torque or force, $g_i$ gravity torque or force, and $\tau_{f,i}$ friction
torque or force.
$m^{\mathrm{RBD}}$ selects actuated fixed-base joints with eligible FD
labels. Floating-base contacts and base coupling are absent from this
residual, so floating-base samples are excluded.

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
- `--preset tiny|small|medium|large|da_get_zero`, plus width, depth, and head overrides;
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
| `state_scale`, `state_offset` | $q_{\mathrm{URDF}}=\mathrm{scale}\cdot q+\mathrm{offset}$, for sign, unit (degrees), or zero-offset differences. |
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
