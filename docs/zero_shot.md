# Zero-shot across tasks and embodiments

[← Back to the README](../README.md)

This page describes how the repository implements the research proposal
[DA-GET-BFM Proposal v4](proposal/DA-GET-BFM_proposal_v4.pdf) (Traditional Chinese;
Markdown source in [proposal/DA-GET-BFM_proposal_v4.md](proposal/DA-GET-BFM_proposal_v4.md)).
The goal is the README title: a **dynamics-aware cross-embodiment behavior foundation
model with zero-shot capabilities**. Zero-shot has two meanings here:

- **Task zero-shot**: goal, tracking and reward prompts without retraining (BFM-Zero).
- **Embodiment zero-shot**: control of robots that were never trained on.

Every option below is off by default, so the original model and its checkpoints are
unchanged. Each item is labelled with its proposal ID and with the evidence it rests on.
Items marked **[H]** are hypotheses without a direct precedent: they come with the
ablation that tests them and are not claimed to help until that ablation is run.

## Contents

- [Status](#status)
- [Evidence measured in this repository](#evidence-measured-in-this-repository)
- [Options by proposal item](#options-by-proposal-item)
- [Configs](#configs)
- [Training](#training)
- [Evaluation](#evaluation)
- [Unverified hypotheses](#unverified-hypotheses)
- [References](#references)
- [Corrections found while implementing](#corrections-found-while-implementing)

## Status

Everything listed is implemented and unit tested on CPU. **No large-scale training run
or ablation has been done**: there is no GPU in the development machine, so none of the
[H] items has been confirmed or refuted, and the evaluation scripts have only run on
tiny smoke-test models.

| Item | Where | Tests |
|---|---|---|
| A1 physics-derived normalization, Gauss metric | `model/training/dataset.py`, `model/pretraining.py` | `tests/test_zero_shot.py` |
| A2 one control period (50 Hz) | `model/training/resampling.py`, `dataset.py` | `tests/test_zero_shot.py` |
| A3 dimensionless inputs | `model/physics_schema.py`, `model/modeling_da_get.py`, `model/physics.py`, `model/rl/observation.py` | `tests/test_zero_shot.py` |
| B0 where the physics lives (probes, controls, random encoder) | `model/evaluation/`, `scripts/diagnose_physics_location.py` | `tests/test_evaluation.py` |
| B1 structured physics interface, operator and consistency losses, rollout effect | `model/pretraining.py` (`DaGetPhysicsInterface`) | `tests/test_zero_shot.py` |
| B2 interface bias in the actor | `model/modeling_bfm.py` | `tests/test_zero_shot.py` |
| B3 head learning rate in RL | `FbCprConfig.lr_physics_heads` (existing) | |
| C1 root token | `model/physics.py` (`root_features`), `model/modeling_da_get.py` | `tests/test_zero_shot.py` |
| C2 contact head | `model/training/episodes.py`, `dataset.py`, `model/pretraining.py` | `tests/test_zero_shot.py` |
| C3 centroidal head | `model/centroidal.py`, `model/pretraining.py` | `tests/test_zero_shot.py` |
| D1 procedural morphology generator | `model/morphology.py`, `model/training/procedural.py`, `scripts/generate_morphologies.py` | `tests/test_morphology.py` |
| D2 temperature sampling, one encoder pass per state | `model/training/sampling.py`, `model/training/trainer.py`, `dataset.py` | `tests/test_zero_shot.py` |
| D3 left-right mirroring | `model/training/symmetry.py` | `tests/test_symmetry.py` |
| D4 topology bias: GET, none, ModuMorph-style modulation | `model/modeling_da_get.py` | `tests/test_zero_shot.py` |
| E1 0.5 s history, spec-deviation head | data configs, `model/pretraining.py` | `tests/test_zero_shot.py` |
| E2 spec randomization | `dataset.py` (`spec_noise`) | `tests/test_zero_shot.py` |
| E3 masked physics modeling | `model/modeling_da_get.py`, `model/pretraining.py` | `tests/test_zero_shot.py` |
| F1 test-time adaptation (LoRA, L2-SP, LwF) | `model/adaptation.py`, `scripts/adapt_da_get.py` | `tests/test_adaptation.py` |
| G1 action scale per family, mass-proportional gains | `model/rl/trainer.py`, `model/training/ground_sim.py` | `tests/test_zero_shot.py` |
| G2 RL on procedural robots | `procedural` source in RL data configs | `tests/test_morphology.py` |
| G3 cross-embodiment prompts by retargeting | `model/rl/retarget.py`, `scripts/retarget_motion.py` | `tests/test_retarget.py` |
| R6 full mass matrix in the state path of B and D | `model/rl/observation.py`, `buffers.py`, `modeling_bfm.py`, `deployment.py` | `tests/test_zero_shot.py` |
| Zero-shot evaluation by split level, with bootstrap CIs | `model/evaluation/`, `scripts/evaluate_zero_shot.py`, `configs/splits/zero_shot.json` | `tests/test_evaluation.py` |

## Evidence measured in this repository

These measurements motivated the items above. File and line numbers refer to the code
before this change.

| ID | Finding |
|---|---|
| R1 | FD, R and E targets were divided by per-record data RMS; one robot in two sources got two scales, and physics replay recomputed them as episodes arrived. |
| R2 | Control periods differed: 0.01 s (synthetic, MuJoCo), `1 / fps` (Hub data), 0.02 s (RL), and the tokens carry no `dt`. |
| R3 | The base's mass, center of mass and inertia were in no input. |
| R4 | Contact state was neither an input nor a target, though the response labels depend on it. |
| R5 | The heads were larger than the encoder (small: 1.12M encoder, 1.45M heads; medium: 6.55M and 5.75M), and in RL they learn faster (1e-4) than the encoder (3e-5). |
| R6 | B and D read a diagonal mass matrix, which the normalization turns into the identity: the mass bias was off in the state path. |
| R7 | Each action variant was a separate sample, so each state was encoded several times. |
| R8 | Training robots were Go1, ANYmal D, G1, H1, a quadrotor and the G1 arm, with ±20% mass and ±10% length perturbations. |
| R9 | Measured at their standing poses: Go1 13.1 kg / 0.31 m, G1 35.1 kg / 0.77 m, ANYmal D 57.0 kg / 0.55 m, H1 59.3 kg / 1.00 m. Median peak / standing torque over loaded joints: ANYmal D 3.4, H1 7.8, Go1 12.6, G1 13.3 (minimum 2.3). |
| R10 | Within-robot `z` pairing only; actions `nominal + 1.25 a` and `K_p = 2 tau^peak` for every robot. |

## Options by proposal item

### A. Targets and inputs that depend on physics only

**A1. `target_normalization: "physics"`** (data config). Joint accelerations are divided by
`tau^peak_i / M_ii(q)` (the robot's median peak torque for passive joints), base
accelerations by `g` and `g / L`, and action-effect targets use Froude units
(`dt sqrt(g / L)` for displacement over `L`, `sqrt(g / L)` for velocities). Every scale is
a function of the encoder's inputs, so it exists for an unseen robot and does not drift
in physics replay. **`acceleration_metric: "mass"`** (model) replaces the squared error of
FD and R with Gauss's metric `e^T M_tilde e` (Udwadia & Kalaba 1992). Which works better is
[H]; compare `data_rms`, `physics`, and `physics` + `mass`.

**A2. `control_dt`** (data config). Simulated sources run at this period (the synthetic
simulator keeps its 0.005 s integration step by scaling its substeps); recorded sources are
resampled (`model/training/resampling.py`: linear, spherical for orientations, held
commands, nearest contacts). Simulated trajectories with exact labels are never resampled.
The zero-shot configs use 0.02 s, the 50 Hz of BFM-Zero.

**A3. `dimensionless_inputs`** (data config) and **`dimensionless_physics`** (encoder).
Base velocities in the tokens become `omega sqrt(L / g)` and `v / sqrt(g L)`; every
physics feature is divided by its unit `m^a L^b T^c` with `T = sqrt(L / g)`
(`model/physics_schema.py` records the exponents of every feature). Asinh scales are
chosen so that a robot of the measured median size (46.05 kg, 0.66 m; R9) is tokenized
exactly as before. The absolute size lives in the root token (C1). RL observers apply the
same scaling, and the RL trainer refuses an encoder pretrained with the other setting.
Whether this helps the encoder is [H].

### B. Physics in the encoder rather than the heads

**B0. Measure first.** `scripts/diagnose_physics_location.py` runs layer-wise linear and MLP
probes with control tasks (selectivity), a random-encoder baseline, a head-capacity test
(heads trained on a frozen random encoder) and head re-initialization transfer. The
decision rule of the proposal: only if heads on a random encoder come close to the trained
model is the structured interface needed.

**B1. `head_type: "linear"` or `"linear_residual"`** (model). `DaGetPhysicsInterface` reads
the joint embeddings linearly: an operator `G_tilde = W W^T + diag(softplus)` (joint
columns; six base rows from the pooled embedding), a drift `f_tilde` and an inverse
`K_tilde`. FD is `y = f + G u`, R is `dy = G du`, ID is `u = K (y - f)` and R^-1 is
`du = K dy`, in units `u = tau / a`, `y = a nu_dot / c` built from the inputs (`a`: peak
torque, `m g`, `m g L`; `c`: mean `a^2 / M_ii`). New losses: `operator` (relative Frobenius
error against the full `G` stored for every frame) and `consistency` (`G K G = G`).
`linear_residual` adds the original mixer heads as residuals that start at zero.
**`effect_mode: "rollout"`** adds `effect_rollout`: a differentiable one-step rollout of the
interface's accelerations through each robot's analytic forward kinematics ([H]).

**B2. `use_interface_bias`** (`BfmConfig`). The actor's joint self-attention gets a per-head
bias from the interface operator normalized to unit diagonal, like the encoder's mass bias
([H]).

**B3.** `FbCprConfig.lr_physics_heads` already exists; the proposal's ablation sets it to at
most `lr_encoder` and checks with the B0 probes whether the encoder keeps its physics after
RL ([H]).

### C. Physics the encoder could not see

**C1. `use_root_token`** (encoder). One more Physics Cross Attention token per robot with the
root body's mass, center of mass and inertia, the total mass, the characteristic length and
the floating-base flag (`model.physics.root_features`), as MetaMorph gives the torso a token.

**C2. `use_contact_head`** (model). Per-joint contact probability of the bodies that carry
contact links. Labels come from sensors or simulators, or from inferred contacts whose
subset selection is confident (`min_contact_confidence`).

**C3. `use_centroidal_head`** (model). Center of mass, linear and angular momentum about it,
composite inertia and the momentum rates, all dimensionless (`model/centroidal.py`;
checked: in free flight `dp/dt = m g` and the angular momentum is conserved). As an
encoder target this is [H].

### D. Coverage of morphologies

**D1. `procedural` source.** `model/morphology.py` generates quadrupeds, hexapods and
humanoids as URDF text with ranges from Ai et al. (Table 6) and GenLoco (Table 1),
geometry-based inertia checked for physical consistency (Wensing et al. 2018), velocity
limits rescaled by Froude similarity, and torque limits from the measured peak / standing
torque range of real robots (R9; [H]). `scripts/generate_morphologies.py` writes suites with
a manifest.

**D2. `sampling_alpha`, `sampling_group`, `group_variants`** (data config). Temperature
sampling `q_g ∝ p_g^alpha` over robots, families or sources (XLM-R's 0.3 and Arivazhagan
et al.'s 0.2 are the starting points; family balancing is [H]). With `group_variants` a
sample is one state with all its action variants; the encoder runs once per state and
per-variant tasks read it through `variant_rows` (checked: same losses as before).

**D3. `mirror`** (per source). Left-right mirrored copies of every trajectory, with the
mirror map derived from the geometry (joint origins and axes reflected in the sagittal
plane).

**D4. `topology_bias`** (training script): `get` (GET-Zero bias), `none`, or `modulation`:
ModuMorph-style contextual modulation with a morphology-only attention bias and a
hypernetwork correction of the input projection, both starting at zero. The literature
disagrees (Amorpheus, SWAT, MetaMorph, ModuMorph, GET-Zero), so the three are an ablation on
L2 and L3.

### E. Robustness to wrong or missing specifications

**E1.** `history_size: 25` at 50 Hz is the 0.5 s history of RMA (Kumar et al. 2021: "k = 50
which corresponds to 0.5s" at 100 Hz). **`use_sysid_head`** predicts
`log(true / given)` of link mass, peak torque, continuous torque and rotor inertia per joint.

**E2. `spec_noise`** (data config). Multiplicative noise on the specification given to the
network; labels stay those of the simulated robot. The zero-shot config uses the repo's
`DEFAULT_RANDOMIZATION` ranges. Dynamics features follow the noisy rotor inertia and
continuous torque but not the link masses ([H] simplification). Randomizing the
description instead of the simulator is itself [H].

**E3. `physics_mask_ratio`** (model, with `physics_masking` in the encoder). Rigid-body and
actuator tokens are replaced by a learned mask embedding and reconstructed from the joint
embedding (`mask` loss). BERT's 15% and MAE's 75% are the ends of the ablation. With
`joint_features: "cross_embodiment"` the observation tokens repeat some masked values, so use
the default `da_get_zero` features for this ablation.

### F. Test-time adaptation

**F1.** `scripts/adapt_da_get.py` generates exact physics labels for a new robot from its URDF
(synthetic or MuJoCo source), freezes the pretrained model, trains LoRA adapters on the
physics losses with L2-SP or LwF against drift, and stops early on the new robot's
synthetic validation split. Using PAD-style self-supervised adaptation for an embodiment
shift is [H].

### G. Zero-shot at the BFM level

**G1.** `action_scale` may map families to `sigma` (`default` for the rest).
`gain_mode: "mass"` with `kp_per_kg` sets `K_p ∝ m` as GenLoco does; take `kp_per_kg` from a
reference robot.

**G2.** RL data configs may use `procedural` sources with the `mujoco` simulator: each
generated robot carries its own standing pose and foot geometry
(`EmbodimentRecord.metadata["env"]`), which the RL loader uses as per-robot settings. FB-CPR
on hundreds of morphologies has no precedent ([H]).

**G3.** `model/rl/retarget.py` retargets a motion between robots with corresponding keypoints
and inverse kinematics as Peng et al. (RSS 2020) do, size scaling, and a
damped-least-squares solver (Wampler 1986; Nakamura & Hanafusa 1986) with Sugihara's
(2011) damping. A prompt from
robot A is retargeted to robot B and then used with BFM-Zero's tracking or goal inference
unchanged.

### R6 fix

`StateView` now carries the full mass matrix of each state, the replay buffer stores it
(packed, half precision), and `DaGetBfmModel.encode_state` uses it; it falls back to the
diagonal only when a caller gives none.

## Configs

| File | Purpose |
|---|---|
| `configs/data/zero_shot.json` | Pretraining data: 50 Hz, 0.5 s history, physics normalization, dimensionless inputs, spec noise, grouped variants, family temperature sampling, real robots plus procedural quadrupeds and humanoids. Hexapods (L3) and the L4 robots, H1 included, are left out. |
| `configs/pretraining/zero_shot.json` | Training arguments for `scripts/train_physics_da_get.py` with every proposal head on. |
| `configs/rl/zero_shot_procedural.json` | RL data: real and procedural robots on MuJoCo ground contact, matching the pretraining inputs. |
| `configs/splits/zero_shot.json` | Split manifest L0-L4. |

## Training

1. **Phase 0, diagnostics on the existing model:**
   `uv run python scripts/diagnose_physics_location.py --checkpoint <run> --data-config configs/data/zero_shot.json --output report.json`.
2. **Pretraining:** `uv run python scripts/train_physics_da_get.py configs/pretraining/zero_shot.json`.
   Learning rate, warmup and batch size start from `docs/training.md` (5e-4, 100, 64) and
   are tuned on the validation split only. Loss weights start at 1 (the repo default);
   uncertainty weighting (Kendall et al. 2018) and GradNorm (Chen et al. 2018) are the
   alternatives to compare.
3. **RL:** `uv run python scripts/train_bfm.py --data_config configs/rl/zero_shot_procedural.json --encoder_checkpoint <run>`,
   with BFM-Zero's hyperparameters (`docs/bfm.md`).
4. **Test-time adaptation** of an unseen robot: `uv run python scripts/adapt_da_get.py --checkpoint <run> --urdf <robot.urdf> ...`.

## Evaluation

- **Splits** (`configs/splits/zero_shot.json`): L0 held-out episodes of training robots; L1
  parameter changes (MetaMorph's six kinds and size extrapolation); L2 new topologies within
  a family; L3 a held-out family; L4 real robot models. 20% held out as in Ai et al.
- **Encoder physics** (`model/evaluation/physics_metrics.py`): operator and response
  relative errors, FD/ID on exact labels, centroidal errors, contact F1, FK over `L`; report
  the gap `Lk - L0`.
- **Where the physics lives** (B0): see above; counterfactual checks compare the change of
  the predicted operator after a link-mass change with the analytic change from
  `constrained_response` (`model/evaluation/counterfactual.py`).
- **BFM, both zero-shots** (`scripts/evaluate_zero_shot.py`): tracking MPJPE and EMD, goal
  MPJPE, getting-up rate per level, zero-shot and after adaptation.
- **Statistics** (`model/evaluation/statistics.py`): several seeds, IQM and stratified
  bootstrap confidence intervals with the robot as the unit (Agarwal et al. 2021). Scaling
  results are reported as trends; no power law is assumed.

## Unverified hypotheses

A1 (which normalization is better), A3 (dimensionless inputs help), B1 (rollout effect;
accuracy cost of a purely linear readout), B2 (interface bias), B3 (head learning rate),
C3 (centroidal targets), D1 (torque limits from the measured ratio range), D2 (family
balancing), E2 (randomizing the description), E3 (masking morphology fields), F1 (PAD-style
adaptation for an embodiment shift), G2 (FB-CPR on many morphologies), and the per-component
choices marked [H] in the code (the low-rank hypernetwork of the modulation, the regression
control task of the probes, the link-mass simplification of the spec noise). Each is tested
by the ablation listed with it in the proposal (section 5.6).

## References

The proposal's reference list ([PDF](proposal/DA-GET-BFM_proposal_v4.pdf), section 8) has
every source with DOI or arXiv number. Every work named on this page is listed here; the
ones used directly in code are also cited, with the section or table used, in the module
docstrings.

Behavior foundation models and embodiment transformers:

- Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta, Guanya Shi. *BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning.* ICLR 2026, arXiv:2511.04131 (50 Hz control; `E_mpjpe`).
- Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto, Yingchen Xu, Alessandro Lazaric, Matteo Pirotta. *Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models.* ICLR 2025, arXiv:2504.11054 (FB-CPR).
- Austin Patel, Shuran Song. *GET-Zero: Graph Embodiment Transformer for Zero-shot Embodiment Generalization.* ICRA 2025, arXiv:2407.15002.
- Vitaly Kurin, Maximilian Igl, Tim Rocktäschel, Wendelin Böhmer, Shimon Whiteson. *My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control.* ICLR 2021, arXiv:2010.01856 (Amorpheus).
- Sunghoon Hong, Deunsol Yoon, Kee-Eung Kim. *Structure-Aware Transformer Policy for Inhomogeneous Multi-Task Reinforcement Learning.* ICLR 2022, https://openreview.net/forum?id=fy_XRVHqly (SWAT).
- Zheng Xiong, Jacob Beck, Shimon Whiteson. *Universal Morphology Control via Contextual Modulation.* ICML 2023, PMLR 202:38286-38300, arXiv:2302.11070 (ModuMorph).
- Agrim Gupta, Linxi Fan, Surya Ganguli, Li Fei-Fei. *MetaMorph: Learning Universal Controllers with Transformers.* ICLR 2022, arXiv:2203.11931 (L1 variation categories, Fig. 6 and Table 2).
- Nico Bohlinger, Grzegorz Czechmanowski, Maciej Piotr Krupka, Piotr Kicki, Krzysztof Walas, Jan Peters, Davide Tateo. *One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion.* CoRL 2024, PMLR 270:3356-3378, arXiv:2409.06366 (URMA; no initial transfer to H1 or Talos, Appendix I).

Procedural robots and adaptation in locomotion:

- Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu, Zhanxin Wu, K. Fay, Henrik I. Christensen, Jan Peters, Hao Su. *Towards Embodiment Scaling Laws in Robot Locomotion.* CoRL 2025, PMLR 305:3483-3515, arXiv:2505.05753 (GenBot-1K, Tables 4-7, 20% held out). Generator: https://github.com/BoAi01/embodiment-scaling-laws (BSD-3-Clause, The Berkeley Humanoid Project Developers).
- Gilbert Feng, Hongbo Zhang, Zhongyu Li, Xue Bin Peng, Bhuvan Basireddy, Linzhu Yue, Zhitao Song, Lizhi Yang, Yunhui Liu, Koushil Sreenath, Sergey Levine. *GenLoco: Generalized Locomotion Controllers for Quadrupedal Robots.* CoRL 2022, PMLR 205:1893-1903, arXiv:2209.05309 (Table 1, mass-scaled gains).
- Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* RSS 2021, DOI 10.15607/RSS.2021.XVII.011, arXiv:2107.04034 (0.5 s history).
- Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan, Sergey Levine. *Learning Agile Robotic Locomotion Skills by Imitating Animals.* RSS 2020, DOI 10.15607/RSS.2020.XVI.064, arXiv:2004.00784 (keypoint retargeting, Sec. IV); code https://github.com/erwincoumans/motion_imitation (Apache-2.0).
- Michael Gleicher. *Retargetting Motion to New Characters.* SIGGRAPH 1998, pp. 33-42, DOI 10.1145/280814.280820 (retargeting by inverse kinematics, as cited by Peng et al.).
- Mayank Mittal, Nikita Rudin, Victor Klemm, Arthur Allshire, Marco Hutter. *Symmetry Considerations for Learning Task Symmetric Robot Policies.* ICRA 2024, pp. 7433-7439, DOI 10.1109/ICRA57147.2024.10611493 (mirrored-data augmentation).
- Daniel Ordoñez-Apraez, Giulio Turrisi, Vladimir Kostic, Mario Martin, Antonio Agudo, Francesc Moreno-Noguer, Massimiliano Pontil, Claudio Semini, Carlos Mastalli. *Morphological Symmetries in Robotics.* IJRR 44(10-11):1743-1766, 2025, DOI 10.1177/02783649241282422.

Mechanics, kinematics and simulation:

- R. McN. Alexander, A. S. Jayes. *A dynamic similarity hypothesis for the gaits of quadrupedal mammals.* J. Zool. 201(1):135-152, 1983, DOI 10.1111/j.1469-7998.1983.tb04266.x (Froude scaling).
- Firdaus E. Udwadia, Robert E. Kalaba. *A new perspective on constrained motion.* Proc. R. Soc. Lond. A 439(1906):407-410, 1992, DOI 10.1098/rspa.1992.0158 (Gauss's principle).
- Roy Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008, DOI 10.1007/978-1-4899-7560-7.
- David E. Orin, Ambarish Goswami. *Centroidal Momentum Matrix of a humanoid robot: Structure and properties.* IROS 2008, pp. 653-659, DOI 10.1109/IROS.2008.4650772.
- David E. Orin, Ambarish Goswami, Sung-Hee Lee. *Centroidal dynamics of a humanoid robot.* Autonomous Robots 35(2-3):161-176, 2013, DOI 10.1007/s10514-013-9341-4.
- Hongkai Dai, Andres Valenzuela, Russ Tedrake. *Whole-body motion planning with centroidal dynamics and full kinematics.* Humanoids 2014, pp. 295-302, DOI 10.1109/HUMANOIDS.2014.7041375.
- Patrick M. Wensing, Sangbae Kim, Jean-Jacques E. Slotine. *Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification: A Statistical Perspective on the Mass Distribution.* IEEE RA-L 3(1):60-67, 2018, DOI 10.1109/LRA.2017.2729659.
- Charles W. Wampler. *Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods.* IEEE Trans. SMC 16(1):93-101, 1986, DOI 10.1109/TSMC.1986.289285.
- Yoshihiko Nakamura, Hideo Hanafusa. *Inverse Kinematic Solutions With Singularity Robustness for Robot Manipulator Control.* ASME J. Dyn. Syst. Meas. Control 108(3):163-171, 1986, DOI 10.1115/1.3143764.
- Tomomichi Sugihara. *Solvability-Unconcerned Inverse Kinematics by the Levenberg-Marquardt Method.* IEEE T-RO 27(5):984-991, 2011, DOI 10.1109/TRO.2011.2148230.
- Jesse Haviland, Peter Corke. *Manipulator Differential Kinematics: Part I: Kinematics, Velocity, and Applications.* IEEE Robotics & Automation Magazine 31(4):149-158, 2024, DOI 10.1109/MRA.2023.3270228, arXiv:2207.01796 (Eq. 51 of the arXiv version).
- Emanuel Todorov, Tom Erez, Yuval Tassa. *MuJoCo: A physics engine for model-based control.* IROS 2012, pp. 5026-5033, DOI 10.1109/IROS.2012.6386109.
- Michael Lutter, Christian Ritter, Jan Peters. *Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning.* ICLR 2019, arXiv:1907.04490 (DeLaN).
- Manuel Watter, Jost Tobias Springenberg, Joschka Boedecker, Martin Riedmiller. *Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images.* NIPS 2015, arXiv:1506.07365 (E2C).

Training, sampling and adaptation:

- Guillaume Lample, Alexis Conneau. *Cross-lingual Language Model Pretraining.* NeurIPS 2019, arXiv:1901.07291 (temperature sampling, Sec. 3.1).
- Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020, pp. 8440-8451, DOI 10.18653/v1/2020.acl-main.747 (XLM-R, alpha = 0.3).
- Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen, Yuan Cao, George Foster, Colin Cherry, Wolfgang Macherey, Zhifeng Chen, Yonghui Wu. *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges.* arXiv:1907.05019, 2019 (T = 5).
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019, pp. 4171-4186, DOI 10.18653/v1/N19-1423 (15% masking).
- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022, pp. 15979-15988, DOI 10.1109/CVPR52688.2022.01553 (75% masking).
- Alex Kendall, Yarin Gal, Roberto Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018, pp. 7482-7491, DOI 10.1109/CVPR.2018.00781, arXiv:1705.07115.
- Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich. *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.* ICML 2018, PMLR 80:794-803, arXiv:1711.02257.
- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022, arXiv:2106.09685; code https://github.com/microsoft/LoRA (`loralib`, MIT, Microsoft Corporation).
- Xuhong Li, Yves Grandvalet, Franck Davoine. *Explicit Inductive Bias for Transfer Learning with Convolutional Networks.* ICML 2018, PMLR 80:2825-2834, arXiv:1802.01483 (L2-SP).
- Zhizhong Li, Derek Hoiem. *Learning without Forgetting.* ECCV 2016, pp. 614-629, DOI 10.1007/978-3-319-46493-0_37; IEEE TPAMI 40(12):2935-2947, 2018, DOI 10.1109/TPAMI.2017.2773081, arXiv:1606.09282 (LwF).
- Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenyà, Pieter Abbeel, Alexei A. Efros, Lerrel Pinto, Xiaolong Wang. *Self-Supervised Policy Adaptation during Deployment.* ICLR 2021, arXiv:2007.04309 (PAD).
- Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, Moritz Hardt. *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.* ICML 2020, PMLR 119:9229-9248, arXiv:1909.13231 (TTT, cited in `model/adaptation.py`).
- Ian Goodfellow, Yoshua Bengio, Aaron Courville. *Deep Learning.* MIT Press, 2016, Sec. 7.8, Algorithm 7.1 (early stopping in `scripts/adapt_da_get.py`).

Probing and evaluation statistics:

- Guillaume Alain, Yoshua Bengio. *Understanding intermediate layers using linear classifier probes.* ICLR 2017 Workshop, arXiv:1610.01644.
- John Hewitt, Percy Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP 2019, pp. 2733-2743, DOI 10.18653/v1/D19-1275; code https://github.com/john-hewitt/control-tasks (Apache-2.0).
- Kelly W. Zhang, Samuel R. Bowman. *Language Modeling Teaches You More than Translation Does: Lessons Learned Through Auxiliary Syntactic Task Analysis.* BlackboxNLP 2018, pp. 359-361, DOI 10.18653/v1/W18-5448 (arXiv:1809.10040).
- Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, Marco Baroni. *What you can cram into a single $&!#\* vector: Probing sentence embeddings for linguistic properties.* ACL 2018, pp. 2126-2136, DOI 10.18653/v1/P18-1198.
- Bastien Liétard, Mostafa Abdou, Anders Søgaard. *Do Language Models Know the Way to Rome?* BlackboxNLP 2021, pp. 510-517, DOI 10.18653/v1/2021.blackboxnlp-1.40 (permuted-target controls, the closest precedent for regression control tasks).
- Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021, pp. 29304-29320, arXiv:2108.13264; library https://github.com/google-research/rliable (Apache-2.0).
- B. Efron. *Bootstrap Methods: Another Look at the Jackknife.* Ann. Statist. 7(1):1-26, 1979, DOI 10.1214/aos/1176344552.
- Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger. *Deep Reinforcement Learning That Matters.* AAAI 2018, DOI 10.1609/aaai.v32i1.11694.

Robot models (downloaded by `scripts/download_urdfs.py`, not redistributed):

- Unitree Go1, Go2, G1 and H1: `unitreerobotics/unitree_ros`, BSD-3-Clause, HangZhou YuShu Technology Co., Ltd. ("Unitree Robotics").
- ANYmal D: `ANYbotics/anymal_d_simple_description`, BSD-3-Clause, ANYbotics AG.
- Talos: PAL Robotics, S.L., CC BY-NC-ND 3.0 (stated in the file), distributed in `Gepetto/example-robot-data` (BSD-3-Clause repository).

## Corrections found while implementing

Reading the sources again for the code changed a few details of the proposal:

- **LoRA ranks.** Table 6 of Hu et al. sweeps `r` in {1, 2, 4, 8, 64}, not 1-8; the rank is a
  required argument of `scripts/adapt_da_get.py`. The paper gives no standard deviation for
  the initialization of `A`; the authors' code (`loralib`) uses `kaiming_uniform_`.
- **GenBot-1K variations.** Table 6 of Ai et al. uses discrete sets (thigh, calf and torso
  factors {0.4, 0.8, 1.0, 1.2, 1.6}; knee-limit factors {0.2, 0.6, 1.0}), and their generator
  changes one geometry factor per robot. Where the paper and their code differ, the module
  docstring of `model/morphology.py` says which is followed.
- **GenLoco densities.** 400-1200 kg/m^3 is the A1's density times [0.5, 1.5]; the generator
  applies the relative rule per link by default.
- **Retargeting scale.** Peng et al. (RSS 2020) do not state a size ratio in the paper; their
  code uses hand-set constants. The automatic ratio of characteristic lengths in
  `model/rl/retarget.py` is therefore [H].
- **L2-SP.** Li et al. choose alpha per experiment by cross-validation (a required argument);
  beta = 0.01 is the value they report as consistently picked.
- **Talos.** Its URDF uses negative `mechanicalReduction` values (reversed actuators); the
  parser now takes the magnitude, so Talos can be an L4 robot.
- **Real robots are not mirror-symmetric** in their URDF masses and limits (Go1, H1 within 1%,
  G1 3%, ANYmal D 10%), so `mirror_map` returns None for them with its default checks; D3
  mirroring applies to the symmetric procedural robots.
