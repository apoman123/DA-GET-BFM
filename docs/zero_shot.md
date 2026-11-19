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

**E1.** `history_size: 25` at 50 Hz is the 0.5 s history of RMA. **`use_sysid_head`** predicts
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

**G3.** `model/rl/retarget.py` retargets a motion between robots with corresponding keypoints,
size scaling and damped-least-squares inverse kinematics (Peng et al. 2020). A prompt from
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
every source with DOI or arXiv number. The ones used directly in code are also cited in
the module docstrings:

- Alexander & Jayes, J. Zool. 201:135-152, 1983 (dynamic similarity).
- Udwadia & Kalaba, Proc. R. Soc. A 439:407-410, 1992 (Gauss's principle).
- Featherstone, Rigid Body Dynamics Algorithms, 2008.
- Orin & Goswami, IROS 2008; Orin, Goswami & Lee, Auton. Robots 35, 2013; Dai, Valenzuela & Tedrake, Humanoids 2014 (centroidal dynamics).
- Wensing, Kim & Slotine, RA-L 3(1), 2018 (physical consistency).
- Lutter, Ritter & Peters, ICLR 2019 (DeLaN); Watter et al., NIPS 2015 (E2C).
- Gupta et al., ICLR 2022 (MetaMorph); Xiong, Beck & Whiteson, ICML 2023 (ModuMorph).
- Feng et al., CoRL 2022 (GenLoco); Bohlinger et al., CoRL 2024 (URMA); Ai et al., CoRL 2025 (embodiment scaling).
- Kumar et al., RSS 2021 (RMA); Peng et al., RSS 2020 (motion retargeting).
- Lample & Conneau, 2019; Conneau et al., ACL 2020; Arivazhagan et al., 2019 (temperature sampling).
- Devlin et al., NAACL 2019 (BERT); He et al., CVPR 2022 (MAE).
- Hu et al., ICLR 2022 (LoRA); Li, Grandvalet & Davoine, ICML 2018 (L2-SP); Li & Hoiem, TPAMI 2018 (LwF); Hansen et al., ICLR 2021 (PAD).
- Alain & Bengio, 2017; Hewitt & Liang, EMNLP 2019; Zhang & Bowman, 2018; Conneau et al., ACL 2018 (probing).
- Agarwal et al., NeurIPS 2021; Henderson et al., AAAI 2018 (evaluation statistics).
- Mittal et al., ICRA 2024; Ordoñez-Apraez et al., IJRR 2025 (symmetry).
- Sugihara, IEEE T-RO 27(5), 2011, DOI 10.1109/TRO.2011.2148230; Wampler, IEEE T-SMC 16(1), 1986; Haviland & Corke, IEEE RAM 2023, DOI 10.1109/MRA.2023.3270228 (damped least-squares inverse kinematics in `model/rl/retarget.py`).
- Goodfellow, Bengio & Courville, *Deep Learning*, MIT Press 2016, Algorithm 7.1 (early stopping in `scripts/adapt_da_get.py`).
- Liétard, Abdou & Søgaard, BlackboxNLP 2021 (permuted-target controls, the closest precedent for regression control tasks).

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
