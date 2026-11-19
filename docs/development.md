# Development notes

[← Back to the README](../README.md)

The full repository layout, what the tests cover, validation results, and detailed limitations.

## Contents

- [Repository layout](#repository-layout)
- [Testing and validation](#testing-and-validation)
- [Continuous integration](#continuous-integration)
- [Limitations](#limitations)
- [Knowledge graph](#knowledge-graph)

## Repository layout

```text
DA-GET-BFM/
├── model/
│   ├── __init__.py            # public API
│   ├── urdf.py                # standard-library URDF parser, transmissions, transforms
│   ├── embodiment.py          # EmbodimentGraph (base body, link frames), JointFeatureConfig,
│   │                          # EmbodimentBatch, collate_embodiments, FK, mass matrix
│   ├── dynamics.py            # fixed/floating RNEA (with rotor inertia), M_full, forward dynamics,
│   │                          # contacts, contact Jacobian, constrained response, rotor thrust,
│   │                          # torque estimation (analysis only)
│   ├── physics_schema.py      # physics feature symbols, units, frames, scales, update tiers
│   ├── physics.py             # ActuatorSpec, feature extraction, motor labels, PhysicsFeatureCache
│   ├── tokenization_da_get.py # DaGetTokenizer: GET-Zero token layout
│   ├── configuration_da_get.py # DaGetConfig and the da_get_zero preset
│   ├── modeling_da_get.py     # DaGetEncoder, graph attention bias, Physics Cross Attention,
│   │                          # physics tokenizers, PhysicsMemory
│   ├── pretraining.py         # DaGetForPhysicsPretraining (HF PreTrainedModel), heads A, B, C, R, R⁻¹, D, E
│   ├── configuration_bfm.py   # BfmConfig
│   ├── modeling_bfm.py        # per-DoF B, D, actor, F and critics; DaGetBfmModel
│   ├── rl/
│   │   ├── configuration_fb_cpr.py # FbCprConfig: BFM-Zero's training hyperparameters
│   │   ├── observation.py     # policy and state views of one robot (no torque input)
│   │   ├── actions.py         # ActionMap (targets inside the torque envelope), TorqueMonitor
│   │   ├── constraints.py     # PenaltyMultipliers: torque-penalty weights by Lagrange multipliers
│   │   ├── envs.py            # environment logic, MuJoCo worlds (PD and envelope per physics step),
│   │   │                      # start-of-step labels of label worlds
│   │   ├── isaac_env.py       # Isaac Lab worlds; several robots in one simulation; envelope actuator
│   │   ├── buffers.py         # per-robot replay buffer and expert motions
│   │   ├── agent.py           # DaGetFbCprAgent: D, F, B, critics, encoder and actor updates
│   │   ├── evaluation.py      # tracking, goal and getting-up tasks, EMD, motion priorities
│   │   ├── physics_replay.py  # physics-loss episodes from RL rollouts (exact labels from label worlds)
│   │   ├── deployment.py      # DaGetBfmPolicy: 50 Hz runner with CUDA Graphs
│   │   └── trainer.py         # online training loop, checkpoints, TensorBoard
│   ├── transformer.py         # attention and cross-attention blocks
│   ├── robot_analysis.py      # infer floating base, feet, rotors, propellers from a URDF
│   ├── centroidal.py          # center of mass, centroidal momentum, composite inertia, rates
│   ├── morphology.py          # procedural quadrupeds, hexapods, humanoids (URDF text)
│   ├── adaptation.py          # LoRA, L2-SP, LwF for test-time adaptation
│   ├── evaluation/            # split manifests, probes, physics metrics, counterfactuals,
│   │                          # IQM and stratified bootstrap CIs
│   ├── conversion/
│   │   ├── common.py          # table format, resampling, HTTP range reads of remote zips
│   │   ├── datasets.py        # GrandTour, MIPO, NeuroBEM, LocoMuJoCo, AMASS-G1, LAFAN1
│   │   ├── mapping.py         # match dataset columns to URDF joints and base signals
│   │   └── zarr_lite.py       # minimal zarr v2 reader
│   ├── training/
│   │   ├── sources.py         # synthetic, MuJoCo ground, and Hugging Face sources
│   │   ├── ground_sim.py      # MJCF from the graph, standing and pushes on the ground
│   │   ├── episodes.py        # per-frame features and labels (torque sources, response
│   │   │                      # operator, slip check), base reconstruction, height rule, records
│   │   ├── contacts.py        # contact inference: subset selection with confidence, evaluation
│   │   ├── dataset.py         # data config, caching, dataset, collator
│   │   ├── resampling.py      # one control period: resample recorded trajectories
│   │   ├── procedural.py      # procedural-robot source (generator + simulator source)
│   │   ├── symmetry.py        # left-right mirror map and mirrored trajectories
│   │   ├── sampling.py        # temperature sampling over robots or families
│   │   └── trainer.py         # Trainer that logs each task loss (optionally weighted sampling)
│   └── README.md              # module-level technical notes
├── scripts/
│   ├── train_physics_da_get.py       # HF Trainer entry point
│   ├── train_bfm.py                  # online off-policy FB-CPR across robots
│   ├── make_data_config.py           # write a data config from a URDF and a dataset
│   ├── convert_dataset.py            # convert public datasets to the table format
│   ├── validate_torque_estimation.py # estimated vs measured torques on a real Go1 (analysis)
│   ├── evaluate_contact_detection.py # inferred vs sensor or simulator contacts
│   ├── benchmark_inference.py        # 50 Hz latency of every network
│   ├── diagnose_physics_location.py  # B0: probes, controls, random encoder, head re-init transfer
│   ├── evaluate_zero_shot.py         # BFM metrics per split level, IQM and bootstrap CIs
│   ├── adapt_da_get.py               # F1: test-time adaptation of a new robot (LoRA + L2-SP/LwF)
│   ├── retarget_motion.py            # G3: keypoint retargeting between robots
│   ├── generate_morphologies.py      # D1: write a suite of procedural robots
│   ├── build_proposal_pdf.py         # render docs/proposal/*.md to the PDF
│   └── download_urdfs.py             # fetch Unitree and ANYmal D URDFs into assets/urdf
├── assets/robots/quadrotor_x.urdf    # X quadrotor for synthetic flight
├── configs/data/              # example data configs
├── configs/rl/                # example RL training configs
├── configs/pretraining/       # training-argument files (zero-shot)
├── configs/splits/            # zero-shot split manifest L0-L4
├── tests/                     # unit tests (see below)
├── .github/workflows/ci.yml   # continuous integration
├── docs/                      # documentation by topic (docs/proposal/: the research proposal)
├── raw/                       # source material (the GET-Zero project page)
├── graphify-out/              # knowledge graph of the code and paper
├── THIRD_PARTY_NOTICES.md     # attribution for adapted code
├── pyproject.toml
└── uv.lock
```

## Testing and validation

```bash
uv sync --all-extras
uv run pytest
```

The suite has 197 tests (the MuJoCo and HDF5 tests are skipped without the
`sim` and `validation` extras, and the CUDA Graph test without a GPU):

- `test_embodiment.py`: URDF graphs, features, kinematics, mass matrix,
  batching, padding invariance.
- `test_da_get_encoder.py`, `test_transformer.py`: the encoder and its building
  blocks.
- `test_physics.py`:
  - RNEA against closed forms and identities, and `dynamics_terms` against the two tiers;
  - transmissions and frozen joints;
  - feature schema and body-frame values;
  - actuator specs, the torque-speed envelope, and motor labels;
  - tiered cache refreshes;
  - Physics Cross Attention: cached memory, stale-memory detection, padding
    invariance;
  - rotor inertia in `M`, RNEA and the rigid-body features, and its glob settings;
  - the constrained response: `M⁻¹` for a fixed base, the slope of floating-base
    forward dynamics without contacts, `J G = 0` and `G (I − P) = 0` with contacts,
    and the contact Jacobian against finite-differenced foot velocities;
  - `envelope_targets` puts requests exactly on the envelope, and
    `torque_limit_excess`;
  - the 5-column dynamics schema, and that encoder configs of the old sizes are rejected.
- `test_floating.py`:
  - floating-base mass matrix structure, FD and RNEA consistency, free fall;
  - standing contact forces and torques;
  - surface contacts for bipeds, and dropping pulling contacts;
  - hover thrust;
  - harnessed floating and flight simulations whose torques are recovered
    from kinematics alone;
  - the determinate-torque projector removes exactly the internal-force
    directions, and the projected ID and inverse-response losses ignore them;
  - kinematics-only data of a standing quadruped: no torque labels or inputs,
    response labels that keep the feet still, and the analysis estimate bearing
    the weight;
  - a Hub-style floating-base dataset with split state columns, through to a
    model forward pass, with recorded rows on R but not on FD.
- `test_ground.py` (MuJoCo):
  - the generated MJCF has our dynamics (mass matrix and bias forces);
  - a quadruped stands with exact labels and probe accelerations, and trusted
    simulator contacts; response labels are dropped exactly where contacts change or
    a foot in contact slides faster than `max_contact_slip`;
  - the generated MJCF has the stiff contact `solimp` and selectable `position` or
    `motor` actuators;
  - `G δτ` matches MuJoCo's acceleration change within 5% once its contacts are
    stiff (about 20% with the default soft contacts), while ignoring the contacts
    is off by more than 100%;
  - estimated torques differ from MuJoCo's almost only by internal contact
    forces (motion-determined error below 10%);
  - a biped is steadied by a small recorded spotter wrench.
- `test_contacts.py`:
  - the contact selector chooses all four feet of a standing quadruped confidently,
    no foot in free fall (a foot that carries nothing is not a contact), and leaves
    out a foot that moves unlike the others when the base velocity is unknown;
  - kinematic contact probabilities follow height; robots without feet are rejected;
  - the evaluation metrics;
  - (MuJoCo) frames that keep response labels have the simulator's contact set more
    than 90% of the time; the switches `detected_contact_response`,
    `min_contact_confidence` and `contact_detection` act as documented; the evaluation
    script reports both methods.
- `test_conversion.py`:
  - every converter on small files in each dataset's real format;
  - the NeuroBEM thrust identification;
  - column-to-joint matching;
  - the URDF analysis of quadrupeds, bipeds, multirotors, and arms;
  - the config generator, end to end into a dataset.
- `test_bfm.py`:
  - shapes and BFM-Zero conventions of B, D, the actor, F, and the critics;
  - dict observations;
  - padding invariance and joint-permutation equivariance;
  - conditioning on z, latent sequences, and context tokens;
  - goal, reward, and tracking inference;
  - straight-through truncated-normal gradients;
  - an FB actor loss that reaches the GET encoder.
- `test_fb_cpr.py`:
  - the FB, orthonormality, and pessimism formulas against explicit computations,
    including that states of other robots do not enter them;
  - each update changes only its own networks, RL and physics losses reach the encoder
    but the actor's does not, and targets move by exactly `tau`;
  - prompts pair only with transitions of the same robot;
  - tokens, dynamics features, and mass matrix equal the pretraining pipeline's,
    and no observation reads a torque;
  - environment resets reproduce a motion's state tokens, lying starts and fall
    behavior, motion priorities, EMD, and auxiliary rewards, including the three
    torque penalties and their formulas;
  - the targets sent stay inside the torque envelope, and `ActionMap` on the
    robot gives the same targets as the environment;
  - with the per-physics-step envelope no physics step exceeds `τ^max(qd)`, while
    MuJoCo's peak clip alone exceeds it by more than 1 N m, and the speed-dependent
    part is exercised;
  - the torque-penalty multipliers: dual step, bounds, per-robot weights in the
    agent's auxiliary reward, checkpoints;
  - the physics replay turns rollout chunks into episodes with response, effect,
    motor and FK labels and no torque labels when the simulator gives no
    start-of-step labels;
  - label worlds: in flight, MuJoCo's start-of-step torque and accelerations
    satisfy the graph's floating-base dynamics to 1e-6, and the start state is the
    previous end state; label worlds keep nominal masses; the replay builds exact
    episodes (B and C labels) from them and kinematic ones from the other worlds,
    and a push ends the right transition in each;
  - real and synthetic sources sharing a robot, the robot graph carrying the
    simulators' rotor inertia, and the training script on two robots (with the
    physics replay and the torque metrics).
- `test_deployment.py`: `DaGetBfmPolicy` gives the training path's actions and B,
  refreshes after a weight change, checks inputs on the CPU, and records no call that
  waits for the device or copies from the host (the CUDA Graph replay test runs only
  with a GPU).
- `test_isaac_env.py`: the Isaac Lab adapter against a fake of its API: joint and
  body orders, quaternions and frames, contact penalties, randomization, batched writes,
  stepping several robots, start-of-step labels from the first physics step with
  explicit actuators (and none with implicit ones), nominal masses in label
  worlds, the torque of every physics step for the monitor, and the envelope
  actuator's clip (on stand-ins of Isaac Lab's actuator classes). It does not run
  Isaac Lab.
- `test_pretraining.py`:
  - both data sources and caching;
  - collation and masks;
  - all seven losses, with a loss decrease;
  - data without torque gives no FD/ID rows and no torque in the tokens, but R,
    R⁻¹ and D rows;
  - the response target equals `G(q) δτ` for the drawn torque change;
  - measured torques are off by default and weighted when enabled;
  - a floating robot without known support gets no response labels;
  - keys of the estimate-based pipeline are rejected, in configs and in loss weights;
  - every example config under `configs/` uses current keys;
  - ablations;
  - `save_pretrained`/`from_pretrained`;
  - a Trainer run with per-task logging.
  - the training script recording train and eval curves in TensorBoard.
- `test_docs.py`: every relative link and `#anchor` in the README, `docs/` and
  `model/README.md` resolves (GitHub heading slugs).

Checks outside the test suite:

- **GET-Zero reference:** on all 633 LEAP Hand URDF variants shipped with
  GET-Zero, the adjacency matrix matches the reference implementation exactly.
  The joint origin and rotation features match the URDF to float32 precision.
- **Real robot models:** on Unitree Go1, H1, and G1, RNEA matches `M qdd`, and
  `FD(ID(qdd)) = qdd` to about 1e-13 for fixed and floating bases.
- **Synthetic floating and flight data:** for the harnessed Go1 and H1 and the
  flying quadrotor, torques estimated from kinematics alone match the applied
  torques to about 1e-15 (a check of the estimator; training does not use it).
- **Constrained response against MuJoCo:** for a standing quadruped with small
  torque changes, the median error of `G δτ` is 20% with MuJoCo's default soft
  contacts, 3% with `solimp` 0.99–0.999, and 2.7% with 0.9999; the free-floating
  response (contacts ignored) is off by about 500%.
- **Converters on real downloads:** a MIPO trot sequence, a GrandTour mission
  (17 episodes of 20 s), NeuroBEM flights read remotely from the zip, and
  LocoMuJoCo, AMASS-G1, and LAFAN1 motions all convert and load.
- **MuJoCo ground data:** Go1, ANYmal D, G1, and H1 stand through 300-step
  episodes with pushes; the bipeds' spotter carries about 5–10% of the weight.
- **Real recorded data:**
  - Unitree G1 whole-body teleoperation loads as a 29-DoF floating-base robot.
    On 99.5% of frames the detected contacts explain the base wrench (these
    frames keep their response labels).
  - Torque estimation on a real Go1 is
    [compared with measured torques](dynamics.md#how-accurate-is-it).

The two runs below used the earlier, estimate-based pipeline (7 dynamics features with
the previous torque, estimated FD/ID labels, task F). They are kept as a record of the
pipeline working end to end; they have not been rerun with the torque-free schema,
whose task set and labels differ.

**Floating-base training run.** The data was 11 embodiments:
- 12 recorded G1 whole-body teleoperation episodes from the Hub (floating
  base, estimated torques);
- 4 randomized flying quadrotors;
- 4 randomized Go1 and 2 randomized H1 variants, each half fixed-base and
  half floating-base.

That is 36.7k training and 3.3k held-out evaluation samples. The `small`
preset (2.03M parameters) ran 600 steps, batch 64, with `rbd_weight` 0.1, on
CPU:

| | total | FK | FD | ID | motor | effect | RBD |
|---|---|---|---|---|---|---|---|
| train, first 100 steps | 8.145 | 2.066 | 1.403 | 0.223 | 0.458 | 3.919 | 0.740 |
| train, last 100 steps | 1.855 | 0.114 | 0.712 | 0.062 | 0.034 | 0.913 | 0.207 |
| eval, step 300 | 5.031 | 0.710 | 1.295 | 0.237 | 0.085 | 2.654 | 0.660 |
| eval, step 600 | 3.845 | 0.398 | 1.188 | 0.221 | 0.068 | 1.921 | 0.615 |

Every task loss decreases on held-out episodes, which shows that fixed,
floating, and flying robots train together end to end. The gap between train
and eval is expected for a run this short on this little data.

**Earlier ablation (fixed-base pipeline, before floating-base support).** This
used `configs/data/g1_hub_plus_synthetic.json` with 17 embodiments: 127k
training and 5,460 evaluation samples, the `small` preset, and 800 steps on
CPU. Evaluation loss with and without Physics Cross Attention:

| Eval loss | total | FK | FD | ID | motor | effect |
|---|---|---|---|---|---|---|
| with (2.01M params) | **1.992** | **0.129** | 0.325 | **0.143** | 0.0207 | **1.331** |
| without (1.68M params) | 2.155 | 0.179 | **0.307** | 0.181 | **0.0199** | 1.424 |

This was one seed and a short run with unequal parameter counts, so read it as
a pipeline check, not a conclusive ablation.

## Limitations

- **Tree-structured robots only.** Closed kinematic loops (parallel linkages)
  cannot be expressed in URDF and are rejected.
- **Xacro must be expanded first**, for example with
  `xacro robot.urdf.xacro > robot.urdf`. MJCF models must be converted.
- **No torque labels for recorded data without sensors.** Torque estimates are
  43–97% off on a real Go1 and 47–72% on ANYmal D, so training does not use them.
  Such data supervises dynamics only through the constrained response; full
  forward and inverse dynamics come from simulated data.
- **The response is a rigid, holding-contact model.** It assumes contacts that
  neither slip nor lift under the torque change, and needs to know which feet are
  down. The generated MuJoCo models use stiff contacts (3% from the rigid response;
  20% with MuJoCo's defaults), and frames with a foot sliding faster than 0.1 m/s
  are dropped (this check needs the base linear velocity). Without contact sensors,
  the contact set is inferred with a confidence; on six real Go1 recordings 82–97%
  of the frames that keep response labels have the right set, so some label noise
  remains on recorded data without contacts.
- **Rotor inertia must be supplied.** URDFs have none; set `armature` from motor
  data. The RL default (0.01 kg m²) is a stability value, not a measurement.
- **Torque targets are not measured on a robot.** The simulators clip the PD torque
  to the envelope at every physics step, and the penalty weights adapt per robot
  through Lagrange multipliers, but the targets (5% of joints over the envelope,
  thermal load ≤ 1, 2% over the power limit) are data-sheet style choices. The
  envelope limits braking like driving torque (conservative). The multipliers have
  only run in short CPU runs.
- **The offline ground simulation stands, it does not walk.** The MuJoCo source
  produces stance, crouching, weight shifting, partial foot lift, and push
  recovery, not gaits; gaits would need a locomotion controller per robot.
  Bipeds need the recorded spotter. Feet are spheres from `contact_geometry`,
  and only feet collide, so falls end the episode. Locomotion kinematics come
  from recorded datasets; exact dynamics labels on gaits come from the RL
  physics replay's label worlds, which in Isaac Lab use the explicit envelope
  actuators (the default; unverified on a real Isaac Lab).
- **Automatic configs are guesses.** The URDF analysis and column matching
  print their reasoning; check them, especially for unusual robots, and
  replace rule-of-thumb gains and propeller coefficients with measured ones.
- **Converted datasets inherit their sources' limits.** NeuroBEM publishes no
  geometry, so the rotor layout and drag coefficient are taken from the frame
  size when the fits are poor. MIPO streams are not synchronized. Retargeted
  motion sets are kinematic only.
- **Base state quality matters.** Base velocities from finite-differenced
  poses, accelerometer offsets from the base origin, and contact detection
  without base velocity all add error. Residual masking drops the worst
  frames.
- **Actuator defaults are assumptions.** URDF gives one effort and one velocity
  limit. `τ^cont`, `P^max`, gains, and propeller coefficients need real data
  for feasibility labels to be trustworthy.
- **The URDF must have inertials.** Joints that move massless bodies give a
  zero mass-matrix diagonal.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push to `main` and
to `feature/**` branches, on pull requests, and on demand, on Ubuntu with Python 3.11:

| Step | Command | Checks |
|---|---|---|
| Lock | `uv lock --check` | `uv.lock` matches `pyproject.toml` |
| Install | `uv sync --locked --all-extras` | The locked environment installs, with MuJoCo, the converters and the validation tools |
| Compile | `uv run python -m compileall -q model scripts tests` | Every module byte-compiles |
| URDFs | `uv run python scripts/download_urdfs.py` | Go1, H1, G1, ANYmal D, Go2 and Talos for the tests and the morphology generator (`assets/urdf/` is not committed) |
| Test | `uv run pytest -q -ra` | The whole suite, including both training scripts end to end on MuJoCo, the zero-shot options and the documentation link check |

To run the same steps locally, run those five commands from the repository root. Tests
that need the downloaded URDFs are skipped, with that reason, when they are missing. Not
covered by CI: Isaac Lab (needs an NVIDIA GPU and Isaac Sim), the CUDA Graph replay test
(skipped without a GPU), the latency benchmark (`scripts/benchmark_inference.py`, which
needs the downloaded URDFs and should run on the deployment machine), and Python 3.10
(supported by `pyproject.toml`, but the lock's development environment is 3.11). There is
no deployment (CD) stage: the project publishes no package or service.

## Knowledge graph

`graphify-out/` holds a [graphify](https://github.com/safishamsi/graphify)
knowledge graph of the code and the GET-Zero paper page:

- `graph.html`: an interactive view.
- `GRAPH_REPORT.md`: the central nodes, communities, and surprising links.
- `graph.json`: the raw graph.

It predates the physics, floating-base, and BFM modules and still uses the old
class names; rebuild it with `/graphify . --update`.
