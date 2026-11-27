# Physics pretraining

[← Back to the README](../README.md)

The seven auxiliary tasks that train the encoder, where their labels come from (never from an estimated torque), how to run them with the Hugging Face `Trainer`, and the data config format.

## Contents

- [Physics pretraining](#physics-pretraining)
  - [How the encoder learns dynamics without torque labels](#how-the-encoder-learns-dynamics-without-torque-labels)
  - [Where the labels come from](#where-the-labels-come-from)
  - [Shared notation and loss reduction](#shared-notation-and-loss-reduction)
  - [A. Forward kinematics](#a-forward-kinematics)
  - [B. Forward dynamics](#b-forward-dynamics)
  - [C. Inverse dynamics](#c-inverse-dynamics)
  - [R. Contact-constrained response](#r-contact-constrained-response)
  - [R-inv. Inverse response (R⁻¹)](#r-inv-inverse-response)
  - [D. Motor feasibility](#d-motor-feasibility)
  - [E. Action effect](#e-action-effect)
  - [Removed: F. Rigid-body dynamics consistency](#removed-f-rigid-body-dynamics-consistency)
- [Training with the Hugging Face Trainer](#training-with-the-hugging-face-trainer)
  - [Watching training in TensorBoard](#watching-training-in-tensorboard)
  - [Data config](#data-config)
  - [Values taken from papers in the shipped configs](#values-taken-from-papers-in-the-shipped-configs)
- [References](#references)

## Physics pretraining

`DaGetForPhysicsPretraining` (`model/pretraining.py`) is a Hugging Face
`PreTrainedModel` ([Transformers](#references), Wolf et al., EMNLP 2020), so
`save_pretrained`, `from_pretrained`, and `push_to_hub` all work. It wraps the GET-Zero
encoder (Patel and Song, ICRA 2025) with seven heads. $N_e$ is the number of valid
joints of a sample's robot. Every per-joint loss is averaged per robot first,
so a 29-DoF humanoid does not outweigh a 4-rotor drone. No task uses an
estimated torque ([why](dynamics.md#why-training-uses-no-estimated-torque)).

| Task | Head input | Target | Loss |
|---|---|---|---|
| **A. Forward kinematics** | $H_i$ | $p_i/L_e$ ($L_e$ = robot size) and $R_i\in SO(3)$ of every joint frame in the base frame | $\frac{1}{N_e}\sum_i[\lambda_p\lVert\hat p_i-p_i\rVert^2+\lambda_R d_{SO(3)}(\hat R_i,R_i)^2]$; `fk_orientation=False` omits orientation |
| **B. Forward dynamics** | $[H_i;E_\tau(\tau_i^{\mathrm{applied}})]$ passed through `mixing_layers` of graph action mixing | $\ddot q_t$; for a floating base also base linear and angular acceleration | $\frac{1}{N_e}\sum_i[(\hat{\ddot q}_i-\ddot q_i)/\sigma_i^a]^2$, plus $\lambda_{\mathrm{base}}$ times normalized base-acceleration MSE |
| **C. Inverse dynamics** | $[H_i;E(\ddot q_i^{\mathrm{desired}},a_{\mathrm{base}}^{\mathrm{desired}})]$ with graph mixing | The applied torque that produced that acceleration (simulated, or measured if enabled) | $\frac{1}{N_e}\sum_i[(P(\hat\tau-\tau))_i/(\tau_i^{\mathrm{cont}}+\varepsilon)]^2$; $P$ removes internal contact forces, $P=I$ without contacts |
| **R. Constrained response** | $[H_i;E(\delta\tau_i/\tau_i^{\mathrm{peak}})]$ with graph mixing | $\delta\ddot q$ and the base acceleration change, $G(q)\,\delta\tau$ with the contacts holding | $\frac{1}{N_e}\sum_i[(\widehat{\delta\ddot q}_i-\delta\ddot q_i)/s_i^R]^2$ plus $\lambda_{\mathrm{base}}$ times normalized base MSE |
| **R⁻¹. Inverse response** | $[H_i;E(\delta\ddot q_i/s_i^R,\delta a_{\mathrm{base}})]$ with graph mixing | $\delta\tau$ | $\frac{1}{N_e}\sum_i[(P(\widehat{\delta\tau}-\delta\tau))_i/\tau_i^{\mathrm{peak}}]^2$ |
| **D. Motor feasibility** | $[H_i;E(a_i^{\mathrm{proposal}})]$ | Peak and continuous feasibility flags; peak torque, continuous torque, power, speed, and joint-position margins | $\lambda_f\operatorname{BCEWithLogits}+\lambda_m\operatorname{Huber}$ |
| **E. Action effect** | $[H_i;E(a_i)]$ with graph mixing | At end-effector frames, in the base frame at $t$: $\Delta p$, $R_{t+1}$, $v_{t+1}$, $\omega_{t+1}$ | $\lambda_p\lVert\widehat{\Delta p}-\Delta p\rVert^2+\lambda_R d_{SO(3)}(\hat R,R)^2+\lambda_v\lVert\hat v-v\rVert^2+\lambda_\omega\lVert\hat\omega-\omega\rVert^2$ |

### How the encoder learns dynamics without torque labels

Joint torques cannot be estimated reliably from recorded motion: on real robots
the estimate is 43–97% off (Unitree Go1, MIPO data) and 47–72% off (ANYmal D, GrandTour
data), mostly from sources no estimator removes ([measurements](dynamics.md#how-accurate-is-it)). Training therefore never estimates a torque. The
encoder still learns each robot's own dynamics from three sources
([details](training.md#physics-pretraining)):

1. **Exact inputs.** `M(q)` (attention bias and `M_ii`), `c`, `g`, friction and
   the rigid-body and actuator features are computed from the model, never
   measured.
2. **The contact-constrained response, on every dataset.** For any recorded
   state, the collator draws a torque change `δτ` and labels it with the exact
   acceleration change `δν̇ = G(q) δτ`,
   `G = [M⁻¹ − M⁻¹Jᵀ(J M⁻¹ Jᵀ)⁻¹J M⁻¹] Sᵀ` with the contacts holding
   ([`constrained_response`](../model/dynamics.py); the constrained dynamics of Featherstone,
   2008, and Udwadia and Kalaba, 1992, see [References](#references)). `G` depends only on the
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
([physics replay](bfm.md#physics-losses-during-rl)).

### Where the labels come from

Each sample is one frame and one *action variant* of it. Variant 0 is what was
executed; simulated data adds probe variants (other torques, simulated one step);
other data adds proposal variants (random torques, for task D only). Each variant has
a torque source: `exact` (a simulator's applied torque), `measured` (a torque sensor)
or `none`.

| Data | A | B, C | R, R⁻¹ | D | E |
|---|---|---|---|---|---|
| `synthetic` source (analytic simulation; exact) | ✓ | ✓ executed and probes, weight 1 | ✓ | ✓ commands and probes | ✓ |
| `mujoco` source (ground contact; exact) | ✓ | ✓ executed and probes, weight 1 | ✓ (MuJoCo's contacts) | ✓ | ✓ |
| Recorded data with a torque sensor (`torque_key`) | ✓ | weight `measured_torque_weight` (default 0) | ✓ | ✓ | ✓ |
| Recorded data with commands (`action_key`) | ✓ | — | ✓ | ✓ commands (PD request computed) and proposals | ✓ conditioned on the command |
| Recorded motion only (mocap, retargeted) | ✓ | — | ✓ | ✓ proposals | ✓ unconditioned (next frame from the state) |
| RL rollouts, label worlds ([physics replay](bfm.md#physics-losses-during-rl)) | ✓ | ✓ start-of-step torque and acceleration, weight 1 (MuJoCo; Isaac Lab with explicit actuators) | ✓ (simulator contacts) | ✓ targets and proposals | ✓ conditioned on the target |
| RL rollouts, other worlds or implicit Isaac Lab | ✓ | — | ✓ (simulator contacts) | ✓ targets and proposals | ✓ conditioned on the target |

R and R⁻¹ need to know which feet are down, that they stick, and what holds the
base. No response labels go to frames within `contact_margin_steps` of a contact
change, frames where a foot in contact slides faster than `max_contact_slip`, frames
whose inferred contacts are not confident (`min_contact_confidence`) or leave more than
`max_contact_residual` of the weight unexplained, and floating robots without contacts,
rotors or a recorded support wrench
([details](dynamics.md#assumptions-and-checks),
[contact inference](dynamics.md#which-feet-are-down-inferring-contacts)).

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
$\tau^{\mathrm{request}}$ is this pre-limit demand, computed from the command;
$\tau^{\mathrm{applied}}$ is the actuator output after its limits and
controller dynamics. The applied value comes only from a simulator or, if
enabled, a torque sensor. Recordings without either have no applied torque, and
nothing estimates one. The requested and applied values can differ when the
actuator saturates.

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
FD and ID use $w_b=1$ for exact simulated torque, $w_b=$ `measured_torque_weight`
(default 0) for torque-sensor data, and $w_b=0$ otherwise. The total objective
is $\mathcal L=\sum_{k}\alpha_k\mathcal L_k$, where $k$ runs over A, B, C, R,
R⁻¹, D and E and $\alpha_k$ is the configured task weight, 1 by default.
Every coefficient denoted $\lambda$ below defaults to 1.

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
matrix $\hat R_i$ by Gram-Schmidt (the continuous 6D representation of Zhou et al.,
CVPR 2019). $d_{SO(3)}$ is the rotation angle of $\hat R^\top R$ (Lynch and Park, 2017,
Sec. 3.2.3.3).

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
consequences of BFM actions. It trains only where the executed torque is
known: simulated data (and torque-sensor data if enabled). Task R carries the
same physics (how a torque change accelerates the robot) to every other dataset.

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
actuated joints of variants whose torque source has a positive weight $w_b$.

### C. Inverse dynamics

**Why this loss exists.** Forward dynamics asks what motion follows an
applied torque or force; inverse dynamics asks what torque or force a
desired motion would require. This gives a capability-oriented signal:
the same acceleration may be easy for one robot and near an actuator
limit for another. It can discourage a representation that explains
motion without encoding how costly the motion is to produce. With several
feet down, different contact forces explain the same motion, so only the
motion-determined part of the torque is supervised.

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

$\tau^{\mathrm{required}}$ is the applied torque of the variant, which produced
its acceleration. $\odot$ denotes elementwise multiplication. $P$ is the
motion-determined torque projector of the frame's contact set
(`determinate_torque_projector`); $P=I$ without contacts. A simulator's
contact solver picks one of many internal-force distributions, which the state
does not reveal, so even exact simulated torques are only supervised on
$P\tau$. $m^{\mathrm{ID}}$ selects the same samples as $m^{\mathrm{FD}}$, except
frames whose *detected* contacts fail the `max_contact_residual` check, because $P$
needs the right contact set.

### R. Contact-constrained response

**Why this loss exists.** Tasks B and C need the applied torque, which most
recorded data lacks and which cannot be estimated reliably. The response to a
torque *change* needs no torque at all: at any state,
$\delta\dot\nu=G(q)\,\delta\tau$ with
$G=[M^{-1}-M^{-1}J^\top(JM^{-1}J^\top)^{-1}JM^{-1}]S^\top$, where $J$ is the
Jacobian of the contacts that hold
([derivation](dynamics.md#the-contact-constrained-response); constrained rigid-body
dynamics, Featherstone 2008). $G$ is each robot's
own constrained inverse inertia: how hard each joint is to accelerate, how the
joints and the base couple, and how the feet on the ground change that. It depends
on $q$, the inertia (with rotor inertia) and the contact set only, so it is an
exact label for every frame of every dataset, including motion-capture data.

**Meaning and target.** `process_episode` stores $G$ (`response_operator`,
$[6+J]\times J$) per frame. The collator draws, for every sample, a torque change
$\delta\tau_i=\pm f_i\,\tau_i^{\mathrm{peak}}$ with $f_i$ log-uniform in
`response_range` (default $[0.02,1]$), random sign, zero on passive joints. The
head receives $\delta\tau_i/\tau_i^{\mathrm{peak}}$, mixes it over the graph, and
predicts $\hat y_i$ for $y_i=\delta\ddot q_i/s_i^R$, plus the six base rows from
the pooled tokens for floating robots:

$$
\mathcal L_{\mathrm R}=
 \mathcal A((\hat y_i-y_i)^2,m^{\mathrm R},1)
 +\lambda_{\mathrm{base}}\frac{1}{|B_{\mathrm R}|}
 \sum_{b\in B_{\mathrm R}}\frac16\sum_{r=1}^{6}
 (\hat y^{\mathrm{base}}_{b,r}-y^{\mathrm{base}}_{b,r})^2 .
$$

$s^R_r=\operatorname{RMS}_t\sqrt{\sum_k G_{rk}^2(\tau_k^{\mathrm{peak}})^2}$ over the
robot's valid frames is the typical acceleration change from a full-peak change
on every joint (floored at 1% of the largest joint value and at $10^{-3}$; base rows
at $10^{-2}$). $m^{\mathrm R}$ selects the valid joints of frames with response
labels (see [above](#where-the-labels-come-from)); $B_{\mathrm R}$ those of
floating robots.

### R-inv. Inverse response

**Why this loss exists.** The mirror of C for torque changes: which torque change
produces a desired change of motion. It ties accelerations back to each actuator's
scale. Torque changes that only press the feet together leave the motion
unchanged ($G(I-P)=0$), so only $P\delta\tau$ is recoverable.

**Meaning and target.** The head receives $\delta\ddot q_i/s^R_i$ and the six
normalized base rows (broadcast to every joint), mixes them over the graph, and
predicts $\hat z_i$ for $z_i=\delta\tau_i/\tau_i^{\mathrm{peak}}$:

$$
e_i=\frac{[P((\hat z-z)\odot\tau^{\mathrm{peak}})]_i}{\tau_i^{\mathrm{peak}}},\qquad
\mathcal L_{\mathrm R^{-1}}=\mathcal A(e_i^2,m^{\mathrm{R^{-1}}},1),
$$

with $m^{\mathrm{R^{-1}}}$ the actuated joints of frames with response labels.

### D. Motor feasibility

**Why this loss exists.** A position target inside the joint-angle limits
can still demand more torque or force than the motor can produce.
Feasibility labels connect a proposed command to the actuator's
short-term and sustainable limits. The continuous flag separates a
command that is briefly executable from one that can be sustained.
This is relevant to BFM's angle-target actions; the runtime limit itself is
enforced by the action map ([actuator torque limits](bfm.md#actuator-torque-limits)).
Its input already contains the requested torque or force calculated from the
command and gains; a low motor loss alone therefore does **not** prove that the
encoder or actor independently learned to use $K_p$ and $K_d$.

**Which proposals.** Recorded commands and simulated probes, and, for data
without probes, `proposal_variants` (default 2) random torque proposals per frame
(log-uniform 2–200% of peak, random sign). All labels are computed from the
proposal, the state and the actuator limits, so they are exact. A recorded frame
with neither a command nor a known torque has no executed proposal; its variant 0
is skipped by this head.

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
For residual $r$, the Huber term (Huber, 1964) is $r^2/2$ when $|r|\le\delta$, and
$\delta(|r|-\delta/2)$ otherwise. Both are the PyTorch losses
(`binary_cross_entropy_with_logits`, `huber_loss`; Paszke et al., NeurIPS 2019).
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
When a recording has no command and no known torque, the action features are
zero (as at the first frame of an episode) and the head predicts the next frame
from the state and its history alone; the target is still exact kinematics.

### Removed: F. Rigid-body dynamics consistency

Task F checked whether B's predicted acceleration satisfied the fixed-base
equation of motion with the applied torque. It needed the applied torque, it
duplicated B wherever that torque is exact, and it could not handle contacts.
It was removed together with the torque estimates. A config with
`loss_weights={"rbd": ...}` is rejected with an explanation.

Weights: `--fk_weight`, `--fd_weight`, `--id_weight`, `--response_weight`,
`--inverse_response_weight`, `--motor_weight`, `--effect_weight`, plus the
`lambda_*` fields of `DaGetPhysicsPretrainingConfig` (including
`lambda_base_acceleration`, which also weights the base rows of R).

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
`accelerate launch` ([Accelerate](#references)). The script adds model flags:
- `--preset tiny|small|medium|large|da_get_zero`, plus width, depth, and head overrides;
- `--physics_cross_attention false` for an ablation;
- `--physics_layers 0,2`, `--mass_encoding`, `--mixing_layers`;
- `--fk_orientation false`, and the task weights.

**Gradient accumulation.** `--gradient_accumulation_steps k` (a `TrainingArguments`
flag) steps the optimizer once every k batches of `--per_device_train_batch_size`, so an
update sees k times more samples with the memory of one batch. The batch has no `labels`
key, so the Trainer divides each batch's loss by k and the gradient is the mean over the
k batches (checked with one SGD step in `tests/test_pretraining.py`). Each task loss is a
mean over the samples of its batch that have the task's labels (zero when none has), so
this mean equals the loss of one batch k times larger only when every batch has as many
labeled samples for each task.

The Trainer logs `loss_fk`, `loss_fd`, `loss_id`, `loss_response`,
`loss_inverse_response`, `loss_motor`, `loss_effect`, and the matching
`eval_loss_*` next to the total loss. A batch without a task's labels (for
example a recorded-only batch for FD) logs zero for that task.
`runs/<name>` then holds a `from_pretrained` checkpoint and `embodiments.json`.
Load the encoder for downstream use with:

```python
from model.pretraining import DaGetForPhysicsPretraining
model = DaGetForPhysicsPretraining.from_pretrained("runs/g1_wbt")
embeddings = model.encode(**batch)          # [B, J, hidden]; or model.encoder directly
```

### Watching training in TensorBoard

The script records every run in [TensorBoard](https://github.com/tensorflow/tensorboard)
(Apache-2.0), under
`<output_dir>/runs/<date>_<host>/`. While it trains, or afterwards:

```bash
uv run tensorboard --logdir runs    # then open http://localhost:6006
```

Pointing `--logdir` at the parent folder lets you compare several runs on one
chart, for example with and without Physics Cross Attention.

| Tag | Meaning |
|---|---|
| `train/loss`, `eval/loss` | Weighted total loss |
| `train/loss_fk`, `_fd`, `_id`, `_response`, `_inverse_response`, `_motor`, `_effect` | Mean of each task loss since the last log |
| `eval/loss_fk`, `_fd`, `_id`, `_response`, `_inverse_response`, `_motor`, `_effect` | Each task loss on held-out episodes (R and R⁻¹ use fresh torque changes each evaluation) |
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
  "seed": 0,
  "measured_torque_weight": 0.0,
  "max_contact_residual": 0.5,
  "contact_margin_steps": 2,
  "max_contact_slip": 0.1,
  "contact_detection": "dynamics",
  "min_contact_confidence": 3.0,
  "proposal_variants": 2,
  "response_range": [0.02, 1.0],
  "sources": [ { "type": "synthetic", ... }, { "type": "huggingface", ... } ]
}
```

| Key | Default | Meaning |
|---|---|---|
| `history_size` | 3 | Observation history of the tokens |
| `joint_features` | `da_get_zero` | Hardware features of the tokens (`cross_embodiment`, or `JointFeatureConfig` fields) |
| `eval_fraction`, `seed` | 0.05, 0 | Episodes held out; split and proposal seed |
| `measured_torque_weight` | 0.0 | Loss weight of FD/ID labels from torque sensors. Simulated torques always weigh 1. Off by default: sensor torques (Unitree's estimated output torque `tauEst`, for example) are themselves approximate |
| `max_contact_residual` | 0.5 | Response labels of frames whose *inferred* contacts leave more than this fraction of the weight in the base wrench unexplained are dropped |
| `contact_margin_steps` | 2 | Frames before and after a contact change without response labels |
| `max_contact_slip` | 0.1 | Frames where a foot in contact moves faster than this (m/s) get no response labels; `null` turns the check off. Needs the base linear velocity |
| `contact_detection` | `dynamics` | How contacts are inferred for data without them: `dynamics` (subset selection with kinematic prior, pushing-only forces in friction cones, minimum load, velocity consistency) or `height` (lowest-foot rule). A source can override it |
| `min_contact_confidence` | 3.0 | With `dynamics`, response labels only where the chosen foot set beats the next best by this cost margin |
| `proposal_variants` | 2 | Random torque proposals per frame (task D) for data without probes |
| `response_range` | `[0.02, 1.0]` | Torque changes of R and R⁻¹, as fractions of peak torque (log-uniform, random sign) |
| `include_probe_variants` | true | Index probe and proposal variants as samples |
| `cache_dir` | `.cache/physics_data` | Directory for processed-data cache entries. Set it in the pretraining run JSON (for example, `configs/pretraining/selected_large.json`) or the data JSON; the run setting takes precedence. Use an absolute path to keep the location independent of the launch directory. Completed sources are checkpointed and reused after an interrupted run; the source in progress is recomputed |

Keys of the earlier estimate-based pipeline (`allow_estimated_torque_for_fd`,
`use_estimated_previous_torque`, `estimated_torque_weight`,
`max_estimate_residual`) are rejected with a message naming their replacement.

Every source accepts `armature` (rotor inertia, one value or `{joint glob: value}`,
[details](dynamics.md#rotor-inertia-armature)), `actuator_overrides`, and the graph
keys (`joint_names`, `frozen_joints`, `floating_base`, ...).

**`"type": "synthetic"`** simulates a URDF robot with the analytic dynamics. Its
applied torques are exact (`exact_dynamics`). Each of `randomized_embodiments`
variants perturbs (`randomization`, defaults in brackets):
- link masses and inertias (`mass` [0.8, 1.2]);
- link centers of mass, by up to `com_offset` (0.1) of each link's radius of
  gyration per axis;
- link lengths (inertia is scaled consistently; `length` [0.9, 1.1]);
- damping and friction, as fractions of each joint's own limits, so the same
  ranges suit a finger, a knee, and a propeller (`damping`, `friction` [0, 0.02]);
- torque and speed limits (`torque`, `velocity` [0.8, 1.2]);
- each joint's rotor inertia, if the source sets one (`armature` [0.5, 1.5]).

These are true parameters of the variant: its physics tokens describe them, so
"same kinematics, different dynamics" samples teach the encoder to read the
physics tokens (dynamics randomization, Peng et al., ICRA 2018). In RL the
opposite convention holds: the simulator randomizes masses while the tokens stay
nominal, as on a real robot whose true masses are unknown.

`base_modes` is cycled over the variants:

| Mode | What is simulated |
|---|---|
| `fixed` | Base bolted down; gravity tilted per episode; PD with gravity feed-forward (Takegaki and Arimoto, 1981) tracks random joint targets. |
| `floating` | The base is free and hangs from a harness that holds the whole-body center of mass against gravity, with no moment. Base rotation and base-limb coupling are fully dynamic. PD gains use the free-base effective inertia. |
| `flight` | A multirotor flies under its own rotor thrust: a geometric controller (Lee, Leok and McClamroch, CDC 2010) tracks random waypoints and maps thrust and moments to rotor speeds through the actual rotor layout (any rotor count). It needs every actuated joint to be a `thrust` rotor. |

Commands are clipped to the torque-speed envelope before they are applied, so
labels always use the executed torque, never an unexecuted proposal. Each state
also gets `probe_torques` extra torques (log-uniform from 2% to 200% of peak,
random sign), each simulated one step. This tests action conditioning at a
fixed pose and gives the motor head infeasible proposals. The `floating` mode has
no ground contact (the harness wrench is recorded and does not depend on the
torque), so its probes and response labels are exact. This source does not
simulate ground contact; use the
[`mujoco` source](data.md#ground-contact-in-synthetic-data-mujoco) or recorded
datasets for that.

**`"type": "huggingface"`** reads any Hub dataset (`repo_id`) or local folder
or file (`path`) with Hugging Face `datasets` (Lhoest et al., EMNLP 2021) in the
LeRobot parquet layout ([LeRobot](data.md#references)), or any parquet, CSV, or JSON
table with one vector column per signal. Any column key can be a list of
columns that are concatenated in order.

| Key | Meaning |
|---|---|
| `urdf`, `joint_names` | URDF joint of each recorded state entry, in order. Unlisted joints are frozen (`frozen_joints` gives their positions, default 0). |
| `state_key`, `state_indices` | Position columns and which entries to use. A null index means the position is not recorded (rotors); it is integrated from the velocity. |
| `velocity_key`, `torque_key` | Optional columns. Without velocities, positions are finite-differenced. A torque column is a sensor (`measured`, weight `measured_torque_weight`); without one the episode has no torque labels and nothing is estimated. |
| `action_key`, `action_indices`, `action_type` | Commands (`position`, `velocity`, or `torque`; rotor joints keep the `thrust` mode). Set `actuator_defaults.kp`/`kd` for position control so requested torques are meaningful. |
| `state_scale`, `state_offset` | $q_{\mathrm{URDF}}=\mathrm{scale}\cdot q+\mathrm{offset}$, for sign, unit (degrees), or zero-offset differences. |
| `floating_base` | The base is free (legged robots, drones). Data of part of a robot whose support is not recorded (arms of a standing humanoid) should use `false`: a free base without contacts, rotors or support wrench gets no response labels. |
| `base_orientation_key` (+ `_indices`, `base_orientation_format`: `wxyz`, `xyzw`, `rpy`, `matrix`), `base_position_key` | Base pose in the world. |
| `base_linear_velocity_key`, `base_angular_velocity_key` (+ `_frame`: `base` or `world`) | Base twist; IMU gyros give angular velocity in the base frame. |
| `projected_gravity_key`, `imu_acceleration_key` | Unit gravity direction and accelerometer specific force, both in the base frame. |
| `contact_links`, `contact_key`, `contact_threshold`, `surface_contacts`, `contact_height_tolerance`, `contact_speed_tolerance` | Feet, recorded contact flags or forces, and the inference settings. |
| `contact_detection`, `detected_contact_response` | Per-source inference method (overrides the data config), and `false` to give inferred contacts no response labels at all (dynamic gaits without contact data). |
| `fps`, `episode_key`, `frame_key`, `max_episodes`, `smoothing_window`, `gravity`, `end_effectors` | Timing, grouping, size, smoothing, a constant gravity in the base frame, task-E frames. |

Processed features and labels are cached as Hugging Face `datasets` Arrow files in
`.cache/physics_data/<hash>/` (`model/training/frame_store.py`). The hash covers the config
and the URDF contents, so the analytic dynamics run once per dataset rather than once per
epoch. Episodes, not frames, are split between train and eval.

To build this cache locally before starting the trainer, run the preprocessing
entry point from the repository root with the same training JSON:

```bash
uv run python scripts/preprocess_physics_data.py configs/pretraining/selected_large.json
```

It reads `data_config`, `cache_dir`, and `preprocess_workers` from that JSON,
reports source and episode progress, and then exits without initializing a
model. The later training run reuses the completed cache. Use `--workers N` to
change only the number of preprocessing processes; this does not change the
cache key.
Keep `rebuild_cache` set to `false` in the training JSON so pretraining reuses
the cache prepared by this script.

| File | Content |
|---|---|
| `frames-00000.arrow`, … | One row per frame, one list column per per-frame tensor (flattened). Each file opens with `datasets.Dataset.from_file(path)`; `ProcessedEpisodes(path).dataset()` concatenates them. |
| `episodes.json` | Per episode: robot index, file, first row, frame count, tensor shapes, `dt` |
| `records.pt` | The robot records with the data scales |
| `complete` | Written last; an entry without it is rebuilt |

**Building the cache.** Sources with at least 20,000 frames are processed by worker
processes (`--preprocess_workers`, or `workers=` of `build_physics_datasets`; by default half
the cores, at most 16; 1 processes everything in the main process). Each worker uses one
PyTorch thread, and the main process at most four when it processes a source itself: the
tensors are small, so on 32 cores 32 threads were slower than 4. Episodes keep their order
and seeds, so the cache is the same for any number of workers up to floating-point
rounding (tested to 1e-6). Building the cache
of 107 xwbc G1 clips at 50 Hz (`control_dt` 0.02; 114,024 training samples) took 359.7 s
before the changes below; it now takes 21.6 s with 16 workers and 60.8 s with one (32
cores shared with other users, load average 7–9 at the start of each run):

- `EmbodimentGraph._kinematics` and `recursive_newton_euler` keep one tensor per tree level
  (a joint's parent is in the previous level) instead of rewriting a tensor over all
  joints at every level;
- `floating_mass_matrix` computes the kinematics once and shares it between its 6 + J
  Newton-Euler passes instead of recomputing it for each;
- the `huggingface` source reads columns from Arrow without a Python loop over rows and
  finds the rows of every episode with one sort instead of one scan per episode.

The Arrow files are memory-mapped and every sample reads only the frames it needs, so the
processed data can be larger than the RAM. Sources are processed one at a time and written
as they go, so building the cache needs memory for one source's trajectories, not for all
processed frames. The disk use is about 25 KB per frame for G1 with two probe variants
(`history_size` 25, measured with a synthetic source). Reading a sample from Arrow took
1.2 ms against 0.3 ms from memory, small next to collating a batch of 64 (about 250 ms on
one CPU thread); set `--dataloader_num_workers` to hide both. Caches of earlier versions
(`<hash>.pt`) are not read; delete them.

### Values taken from papers in the shipped configs

JSON has no comments, so the sources of the numbers in `configs/data/*.json` are listed
here. Datasets and robot models are cited in [Training data](data.md#references).

| Config | Value | Source |
|---|---|---|
| `zero_shot.json` | `control_dt` 0.02 s | BFM-Zero controls at 50 Hz (Li et al., arXiv:2511.04131, Sec. 3); its motion library blends frames at the control step (`model/training/resampling.py`) |
| `zero_shot.json` | `history_size` 25 | 0.5 s at 50 Hz, the history length of RMA ("k = 50 which corresponds to 0.5s" at 100 Hz; Kumar et al., RSS 2021) |
| `zero_shot.json` | `sampling_alpha` 0.3 (0.2 is the alternative) | XLM-R samples languages with α = 0.3 (Conneau et al., ACL 2020); Arivazhagan et al. (2019) use temperature T = 5, i.e. α = 0.2 |
| `zero_shot.json` | `spec_noise` ranges | This repository's `DEFAULT_RANDOMIZATION` ranges (`model/training/sources.py`), not a paper's |
| `zero_shot.json` | procedural `count` 100 per family | Ai et al. (CoRL 2025): "for quadrupeds and hexapods, performance saturates around 100 training embodiments" |
| `zero_shot.json` | procedural morphology ranges | GenLoco (Feng et al., CoRL 2022) and Ai et al. (CoRL 2025), see `model/morphology.py` and [the zero-shot references](zero_shot.md#references) |
| every `synthetic` and `mujoco` source | `randomization` defaults | Dynamics randomization (Peng et al., ICRA 2018); the ranges are this repository's |
| `ground_contact.json`, `zero_shot.json` | `nominal_pose`, `contact_geometry` | Set in this repository; no source is given |
| `g1_wbt_floating.json`, `g1_hub_plus_synthetic.json` | `actuator_defaults` (`kp` 80, `kd` 2; `kp` 60, `kd` 1.5) | Set in this repository; no source is given |
| `quadrotor_flight.json` | rotor coefficients and limits | Set in this repository for `assets/robots/quadrotor_x.urdf`; no source is given |

## References

Checked against the publisher, proceedings, arXiv or repository on 2026-11-20.

**Models and training software**
- Austin Patel and Shuran Song. *GET-Zero: Graph Embodiment Transformer for Zero-Shot
  Embodiment Generalization.* ICRA 2025, pp. 14262–14269. DOI
  [10.1109/ICRA55743.2025.11127922](https://doi.org/10.1109/ICRA55743.2025.11127922),
  arXiv:2407.15002 (observation tokens `o_vl`, `o_vg`, `o_fl`).
- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony
  Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer,
  Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao,
  Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. *Transformers:
  State-of-the-Art Natural Language Processing.* EMNLP 2020 System Demonstrations, pp.
  38–45. DOI [10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
  (`PreTrainedModel`, `Trainer`).
- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
  Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas
  Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy,
  Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. *PyTorch: An Imperative
  Style, High-Performance Deep Learning Library.* NeurIPS 2019 (Advances in Neural
  Information Processing Systems 32), arXiv:1912.01703 (`WeightedRandomSampler`, losses).
- Quentin Lhoest et al. *Datasets: A Community Library for Natural Language Processing.*
  EMNLP 2021 System Demonstrations, pp. 175–184. DOI
  [10.18653/v1/2021.emnlp-demo.21](https://doi.org/10.18653/v1/2021.emnlp-demo.21) (all 32
  authors in [Training data](data.md#references)).
- Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab
  Mangrulkar, Marc Sun, and Benjamin Bossan. *Accelerate: Training and inference at scale
  made simple, efficient and adaptable.* 2022. <https://github.com/huggingface/accelerate>
  (Apache-2.0).
- Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli
  Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J.
  Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett,
  Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre
  Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi,
  Christoph Gohlke, and Travis E. Oliphant. *Array programming with NumPy.* Nature
  585(7825):357–362, 2020. DOI
  [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) (finite
  differences equal `numpy.gradient`, unwrapping equals `numpy.unwrap`).

**Losses, rotations and dynamics**
- Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. *On the Continuity of
  Rotation Representations in Neural Networks.* CVPR 2019, pp. 5738–5746. DOI
  [10.1109/CVPR.2019.00589](https://doi.org/10.1109/CVPR.2019.00589).
- Kevin M. Lynch and Frank C. Park. *Modern Robotics: Mechanics, Planning, and Control.*
  Cambridge University Press, 2017. DOI
  [10.1017/9781316661239](https://doi.org/10.1017/9781316661239) (Rodrigues' formula,
  Prop. 3.11; SO(3) logarithm and rotation angle, Sec. 3.2.3.3; quaternions, App. B.3;
  friction pyramid, Sec. 12.2.1).
- Ken Shoemake. *Animating rotation with quaternion curves.* SIGGRAPH 1985, Computer
  Graphics 19(3):245–254. DOI [10.1145/325165.325242](https://doi.org/10.1145/325165.325242)
  (slerp, used when resampling base orientations).
- Peter J. Huber. *Robust Estimation of a Location Parameter.* The Annals of
  Mathematical Statistics 35(1):73–101, 1964. DOI
  [10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732).
- Roy Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008. DOI
  [10.1007/978-1-4899-7560-7](https://doi.org/10.1007/978-1-4899-7560-7).
- Firdaus E. Udwadia and Robert E. Kalaba. *A New Perspective on Constrained Motion.*
  Proceedings of the Royal Society of London A 439(1906):407–410, 1992. DOI
  [10.1098/rspa.1992.0158](https://doi.org/10.1098/rspa.1992.0158) (constrained response).
- J. Y. S. Luh, M. W. Walker, and R. P. C. Paul. *On-Line Computational Scheme for
  Mechanical Manipulators.* J. Dyn. Syst. Meas. Control 102(2):69–76, 1980. DOI
  [10.1115/1.3149599](https://doi.org/10.1115/1.3149599) (recursive Newton-Euler).
- R. McN. Alexander and A. S. Jayes. *A dynamic similarity hypothesis for the gaits of
  quadrupedal mammals.* Journal of Zoology 201(1):135–152, 1983. DOI
  [10.1111/j.1469-7998.1983.tb04266.x](https://doi.org/10.1111/j.1469-7998.1983.tb04266.x)
  (Froude scaling of `dimensionless_inputs` and the physics normalization).

**Synthetic data and contact inference**
- Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. *Sim-to-Real
  Transfer of Robotic Control with Dynamics Randomization.* ICRA 2018, pp. 3803–3810. DOI
  [10.1109/ICRA.2018.8460528](https://doi.org/10.1109/ICRA.2018.8460528).
- Morikazu Takegaki and Suguru Arimoto. *A New Feedback Method for Dynamic Control of
  Manipulators.* J. Dyn. Syst. Meas. Control 103(2):119–125, 1981. DOI
  [10.1115/1.3139651](https://doi.org/10.1115/1.3139651) (PD with gravity compensation).
- Taeyoung Lee, Melvin Leok, and N. Harris McClamroch. *Geometric tracking control of a
  quadrotor UAV on SE(3).* CDC 2010, pp. 5420–5425. DOI
  [10.1109/CDC.2010.5717652](https://doi.org/10.1109/CDC.2010.5717652).
- Robert Mahony, Vijay Kumar, and Peter Corke. *Multirotor Aerial Vehicles: Modeling,
  Estimation, and Control of Quadrotor.* IEEE Robotics & Automation Magazine
  19(3):20–32, 2012. DOI [10.1109/MRA.2012.2206474](https://doi.org/10.1109/MRA.2012.2206474)
  (rotor thrust and drag model).
- Gilbert Feng, Hongbo Zhang, Zhongyu Li, Xue Bin Peng, Bhuvan Basireddy, Linzhu Yue,
  Zhitao Song, Lizhi Yang, Yunhui Liu, Koushil Sreenath, and Sergey Levine. *GenLoco:
  Generalized Locomotion Controllers for Quadrupedal Robots.* CoRL 2022, PMLR
  205:1893–1903 (mass-proportional PD gains, Sec. 3.1).
- Marco Camurri, Maurice Fallon, Stephane Bazeille, Andreea Radulescu, Victor Barasuol,
  Darwin G. Caldwell, and Claudio Semini. *Probabilistic Contact Estimation and Impact
  Detection for State Estimation of Quadruped Robots.* IEEE RA-L 2(2):1023–1030, 2017. DOI
  [10.1109/LRA.2017.2652491](https://doi.org/10.1109/LRA.2017.2652491).
- Gerardo Bledt, Patrick M. Wensing, Sam Ingersoll, and Sangbae Kim. *Contact Model
  Fusion for Event-Based Locomotion in Unstructured Terrains.* ICRA 2018, pp. 4399–4406.
  DOI [10.1109/ICRA.2018.8460904](https://doi.org/10.1109/ICRA.2018.8460904).
- Michael Bloesch, Marco Hutter, Mark Hoepflinger, Stefan Leutenegger, Christian Gehring,
  C. David Remy, and Roland Siegwart. *State Estimation for Legged Robots - Consistent
  Fusion of Leg Kinematics and IMU.* RSS 2012. DOI
  [10.15607/RSS.2012.VIII.003](https://doi.org/10.15607/RSS.2012.VIII.003).
- Amir Beck and Marc Teboulle. *A Fast Iterative Shrinkage-Thresholding Algorithm for
  Linear Inverse Problems.* SIAM J. Imaging Sciences 2(1):183–202, 2009. DOI
  [10.1137/080716542](https://doi.org/10.1137/080716542) (FISTA, the contact-force solver).

**Values in the shipped configs**
- Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni,
  Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo
  Pirotta, and Guanya Shi. *BFM-Zero: A Promptable Behavioral Foundation Model for
  Humanoid Control Using Unsupervised Reinforcement Learning.* ICLR 2026, arXiv:2511.04131.
  Code: <https://github.com/LeCAR-Lab/BFM-Zero>.
- Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. *RMA: Rapid Motor
  Adaptation for Legged Robots.* RSS 2021. DOI
  [10.15607/RSS.2021.XVII.011](https://doi.org/10.15607/RSS.2021.XVII.011),
  arXiv:2107.04034.
- Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek,
  Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov.
  *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020, pp.
  8440–8451. DOI [10.18653/v1/2020.acl-main.747](https://doi.org/10.18653/v1/2020.acl-main.747).
- Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim
  Krikun, Mia Xu Chen, Yuan Cao, George Foster, Colin Cherry, Wolfgang Macherey, Zhifeng
  Chen, and Yonghui Wu. *Massively Multilingual Neural Machine Translation in the Wild:
  Findings and Challenges.* arXiv:1907.05019, 2019.
- Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu, Zhanxin Wu, K. Fay, Henrik I.
  Christensen, Jan Peters, and Hao Su. *Towards Embodiment Scaling Laws in Robot
  Locomotion.* CoRL 2025, PMLR 305:3483–3515, arXiv:2505.05753.
