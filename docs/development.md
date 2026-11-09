# Development notes

[← Back to the README](../README.md)

The full repository layout, what the tests cover, validation results, and detailed limitations.

## Contents

- [Repository layout](#repository-layout)
- [Testing and validation](#testing-and-validation)
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
│   ├── dynamics.py            # fixed/floating RNEA, M_full, forward dynamics, contacts,
│   │                          # rotor thrust, torque estimation
│   ├── physics_schema.py      # physics feature symbols, units, frames, scales, update tiers
│   ├── physics.py             # ActuatorSpec, feature extraction, motor labels, PhysicsFeatureCache
│   ├── tokenization_da_get.py # DaGetTokenizer: GET-Zero token layout
│   ├── configuration_da_get.py # DaGetConfig and the da_get_zero preset
│   ├── modeling_da_get.py     # DaGetEncoder, graph attention bias, Physics Cross Attention,
│   │                          # physics tokenizers, PhysicsMemory
│   ├── pretraining.py         # DaGetForPhysicsPretraining (HF PreTrainedModel), heads A–F
│   ├── configuration_bfm.py   # BfmConfig
│   ├── modeling_bfm.py        # BFM-Zero B, D, and GET-conditioned actor, F, critics; DaGetBfmModel
│   ├── rl/
│   │   ├── configuration_fb_cpr.py # FbCprConfig: BFM-Zero's training hyperparameters
│   │   ├── observation.py     # tokens, physics features, and flat state of one robot
│   │   ├── envs.py            # parallel MuJoCo worlds, randomization, auxiliary rewards
│   │   ├── buffers.py         # per-robot replay buffer and expert motions
│   │   ├── agent.py           # DaGetFbCprAgent: discriminator, F, B, critics, actor updates
│   │   └── trainer.py         # online training loop, evaluation, checkpoints, TensorBoard
│   ├── transformer.py         # attention and cross-attention blocks
│   ├── robot_analysis.py      # infer floating base, feet, rotors, propellers from a URDF
│   ├── conversion/
│   │   ├── common.py          # table format, resampling, HTTP range reads of remote zips
│   │   ├── datasets.py        # GrandTour, MIPO, NeuroBEM, LocoMuJoCo, AMASS-G1, LAFAN1
│   │   ├── mapping.py         # match dataset columns to URDF joints and base signals
│   │   └── zarr_lite.py       # minimal zarr v2 reader
│   ├── training/
│   │   ├── sources.py         # synthetic, MuJoCo ground, and Hugging Face sources
│   │   ├── ground_sim.py      # MJCF from the graph, standing and pushes on the ground
│   │   ├── episodes.py        # per-frame features and labels, base reconstruction,
│   │   │                      # contact detection, embodiment records
│   │   ├── dataset.py         # data config, caching, dataset, collator
│   │   └── trainer.py         # Trainer that logs each task loss
│   └── README.md              # module-level technical notes
├── scripts/
│   ├── train_physics_da_get.py       # HF Trainer entry point
│   ├── train_bfm.py                  # online off-policy FB-CPR across robots
│   ├── make_data_config.py           # write a data config from a URDF and a dataset
│   ├── convert_dataset.py            # convert public datasets to the table format
│   ├── validate_torque_estimation.py # estimated vs measured torques on a real Go1
│   └── download_urdfs.py             # fetch Unitree and ANYmal D URDFs into assets/urdf
├── assets/robots/quadrotor_x.urdf    # X quadrotor for synthetic flight
├── configs/data/              # example data configs
├── configs/rl/                # example RL training config
├── tests/                     # unit tests (see below)
├── docs/                      # documentation by topic
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

The suite has 114 tests (the MuJoCo and HDF5 tests are skipped without the
`sim` and `validation` extras):

- `test_embodiment.py`: URDF graphs, features, kinematics, mass matrix,
  batching, padding invariance.
- `test_da_get_encoder.py`, `test_transformer.py`: the encoder and its building
  blocks.
- `test_physics.py`:
  - RNEA against closed forms and identities;
  - transmissions and frozen joints;
  - feature schema and body-frame values;
  - actuator specs, the torque-speed envelope, and motor labels;
  - tiered cache refreshes;
  - Physics Cross Attention: cached memory, stale-memory detection, padding
    invariance.
- `test_floating.py`:
  - floating-base mass matrix structure, FD and RNEA consistency, free fall;
  - standing contact forces and torques;
  - surface contacts for bipeds, and dropping pulling contacts;
  - hover thrust;
  - harnessed floating and flight simulations whose torques are recovered
    from kinematics alone;
  - the determinate-torque projector removes exactly the internal-force
    directions, and the projected ID loss ignores them;
  - a Hub-style floating-base dataset with split state columns, through to a
    model forward pass.
- `test_ground.py` (MuJoCo):
  - the generated MJCF has our dynamics (mass matrix and bias forces);
  - a quadruped stands with exact labels and probe accelerations;
  - estimated torques differ from MuJoCo's almost only by internal contact
    forces (motion-determined error below 10%);
  - a biped is steadied by a small recorded spotter wrench.
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
  - the FB, orthonormality, and pessimism formulas against explicit computations;
  - each update changes only its own networks, and targets move by exactly `tau`;
  - prompts pair only with transitions of the same robot;
  - tokens, dynamics features, and mass matrix equal the pretraining pipeline's;
  - environment resets reproduce a motion's flat state, a quadruped holds its pose,
    falls end episodes, and auxiliary rewards follow their formulas;
  - motion resampling, checkpoints, and the training script on two robots.
- `test_pretraining.py`:
  - both data sources and caching;
  - collation and masks;
  - all six losses, with a loss decrease;
  - ablations;
  - `save_pretrained`/`from_pretrained`;
  - a Trainer run with per-task logging.
  - the training script recording train and eval curves in TensorBoard.

Checks outside the test suite:

- **GET-Zero reference:** on all 633 LEAP Hand URDF variants shipped with
  GET-Zero, the adjacency matrix matches the reference implementation exactly.
  The joint origin and rotation features match the URDF to float32 precision.
- **Real robot models:** on Unitree Go1, H1, and G1, RNEA matches `M qdd`, and
  `FD(ID(qdd)) = qdd` to about 1e-13 for fixed and floating bases.
- **Synthetic floating and flight data:** for the harnessed Go1 and H1 and the
  flying quadrotor, torques estimated from kinematics alone match the applied
  torques to about 1e-15.
- **Converters on real downloads:** a MIPO trot sequence, a GrandTour mission
  (17 episodes of 20 s), NeuroBEM flights read remotely from the zip, and
  LocoMuJoCo, AMASS-G1, and LAFAN1 motions all convert and load.
- **MuJoCo ground data:** Go1, ANYmal D, G1, and H1 stand through 300-step
  episodes with pushes; the bipeds' spotter carries about 5–10% of the weight.
- **Real recorded data:**
  - Unitree G1 whole-body teleoperation loads as a 29-DoF floating-base robot.
    On 99.5% of frames the estimated contact forces explain the base wrench,
    and the knee torque estimates are about 5–6 N m.
  - Torque estimation on a real Go1 is
    [compared with measured torques](dynamics.md#estimating-joint-torques-without-torque-sensors).

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
- **Torque estimates are approximate.**
  - Contact forces are found by weighted least squares, without friction cones
    or contact dynamics, and are not unique with few feet down.
  - Rotor inertia and motor friction are not in URDFs.
  - On a real Go1 the error is 43–97% of the RMS torque (correlation 0.5–0.8),
    and 47–72% on ANYmal D.
  - Prefer measured torques when a dataset has them.
- **Simulated ground contact covers standing, not walking.** The MuJoCo source
  produces stance, crouching, weight shifting, partial foot lift, and push
  recovery, not gaits. Bipeds need the recorded spotter. Feet are spheres
  from `contact_geometry`, and only feet collide, so falls end the episode.
  Locomotion comes from recorded datasets.
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

## Knowledge graph

`graphify-out/` holds a [graphify](https://github.com/safishamsi/graphify)
knowledge graph of the code and the GET-Zero paper page:

- `graph.html`: an interactive view.
- `GRAPH_REPORT.md`: the central nodes, communities, and surprising links.
- `graph.json`: the raw graph.

It predates the physics, floating-base, and BFM modules and still uses the old
class names; rebuild it with `/graphify . --update`.
