# Training data

[← Back to the README](../README.md)

Where training data comes from: simulated ground contact, automatic configs, dataset converters, and recommended datasets. The config format itself is in [Physics pretraining](training.md#data-config).

## Contents

- [Ground contact in synthetic data (MuJoCo)](#ground-contact-in-synthetic-data-mujoco)
- [Writing data configs automatically](#writing-data-configs-automatically)
- [Converting datasets](#converting-datasets)
- [Recommended training datasets](#recommended-training-datasets)
- [References](#references)

## Ground contact in synthetic data (MuJoCo)

The built-in `synthetic` floating mode hangs the robot from a harness, so no
foot touches the ground. Legged data with real foot-ground contact comes from
the `"type": "mujoco"` source (`model/training/ground_sim.py`, `uv sync --extra
sim`). It builds a [MuJoCo](#references) model (Todorov et al., IROS 2012) from the same
embodiment graph:

- one body per token with the graph's merged inertia (so MuJoCo and our
  dynamics agree; the mass matrix and bias forces match to 1e-7 in the tests);
- a free root joint, and hinge or slide joints with the URDF limits, damping,
  friction and each joint's rotor inertia (`armature`, default 1e-5 kg m², also
  entered in the robot graph so labels and simulation agree);
- position actuators (PD inside the simulator, which keeps stiff gains stable)
  limited to the peak torque;
- collision spheres at the contact links, one per point foot or four corners
  per flat foot (`contact_geometry`: `radius`, `sole_offset`, `foot_size`),
  and a floor with randomized friction. Only the feet collide.

The robot starts in `nominal_pose` with its feet on the floor. It then tracks
smoothly changing joint targets around that pose: crouching, leaning, and
shifting weight between feet, some lifting off. It also receives random
pushes. Episodes stop if the robot falls. PD control alone cannot balance a
biped, so robots with two feet get a **spotter**: a soft spring-damper on the
base that pulls it upright and back over its start point, like a person
steadying the robot (its frequency is the linear-inverted-pendulum `sqrt(g/h)`
of Kajita et al., IROS 2001). It carries about 5–10% of the weight in the tests and
runs; the feet carry the rest.

**Recorded labels are exact for the simulated system:**
- applied joint torques;
- joint and base accelerations;
- per-foot contact flags;
- the push and spotter wrench (passed as a known base wrench);
- probe torques solved by MuJoCo with the contacts in place, so task B covers
  action conditioning while standing on the ground.

The torque source of these episodes is `exact`: task B and C labels are the applied
torques (C projected onto the motion-determined part when feet are down), and the
response labels use MuJoCo's own contact flags. The generated model has stiff contacts
(`solimp` 0.99–0.999; MuJoCo's default is `0.9 0.95 0.001 0.5 2`): with MuJoCo's default soft contacts the rigid-contact response
would differ from MuJoCo's by about 20%, with these by 3%. Frames where a foot in
contact slides faster than 0.1 m/s get no response labels; the random joint targets
make feet slide often, so this drops 8–68% of the stance frames depending on the robot
([details](dynamics.md#assumptions-and-checks)).

```bash
uv sync --extra sim
uv run python scripts/train_physics_da_get.py --data_config configs/data/ground_contact.json --output_dir runs/ground
```

`configs/data/ground_contact.json` has standing poses and foot geometry for
Go1, ANYmal D, G1, and H1; all four stand for the whole episode under the
default perturbations. For another robot, set `nominal_pose` to a standing
pose (the default is zero clamped into the limits) and `contact_geometry` to
its feet. Mixing ground data with recorded data and harness or flight data
covers stance, contact switching, and free motion.

What it does not do: walk. The motions are standing, crouching, weight
shifting, partial foot lift, and recovering from pushes, not gaits. Gaits would
need a locomotion controller for every robot, which is what the behavior
foundation model learns. Recorded datasets provide locomotion kinematics, and
during RL the [physics replay](bfm.md#physics-losses-during-rl) records exact
torque and acceleration labels on the policy's own gaits.

## Writing data configs automatically

`scripts/make_data_config.py` (backed by `model/robot_analysis.py` and
`model/conversion/mapping.py`) writes a data config from a URDF, and
optionally from a dataset:

```bash
uv run python scripts/make_data_config.py --urdf assets/urdf/g1_29dof.urdf \
    --dataset unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit --synthetic --ground \
    --output configs/data/g1_auto.json
```

From the URDF it infers:

| Setting | How |
|---|---|
| multirotor | three or more parallel `continuous` leaf joints spread around the base; rotors get `thrust` actuators with `k_f = (T/W) m g / (N ω_max²)` (`--thrust_to_weight`, default 2.5; `ω_max` from the URDF or `--max_rotor_speed`) and `k_m = 0.016 m · k_f` |
| legged, `floating_base` | two or more independent chains of at least two joints ending in the lowest links, well below the base; a URDF `floating` joint always means floating |
| `contact_links` | the lowest link of each leg chain, preferring names with foot, toe, sole, or ankle |
| `surface_contacts` | two feet or fewer (flat feet that carry moments) |
| `end_effectors` | the foot joints and the other leaf joints (hands, grippers, rotors) |
| position gains | `K_p = effort / 1 rad`, `K_d = 0.025 K_p` when the dataset's actions are position targets |

From the dataset (`meta/info.json`, on the Hub or a converted folder) it
infers:
- **Joint columns**, matched to URDF joints by name:
  - `kLeftShoulderPitch` → `left_shoulder_pitch_joint`;
  - Unitree `FL0/FL1/FL2` → hip, thigh, calf;
  - columns split by body part (`observation.state.left_arm` with
    `shoulder_pitch`, …) take words from the column name;
  - joints the dataset does not record are frozen.
- **Action columns** with the same names.
- **Base pose, angular and linear velocity, projected gravity, IMU, and
  contacts**, from feature names such as `x y z qx qy qz qw` or
  `gx gy gz wx wy wz`.

It prints every decision and match. Unitree's G1 whole-body teleoperation
dataset maps all 29 joints, the base pose, gyro, and gravity without edits.
Always read the report, and edit the JSON where the guess is wrong. Propeller
coefficients and PD gains are sized from rules of thumb; use measured values
when you have them.

## Converting datasets

`scripts/convert_dataset.py` (`model/conversion/`) converts datasets that are
not LeRobot parquet into one table format. The output folder has:
- `data/chunk-000/file-000.parquet`, one row per frame, with these columns:
  - `observation.state`, `observation.velocity`, `observation.effort`;
  - base position and x-y-z-w quaternion, and the base-frame twist;
  - `observation.imu_acceleration`, `observation.contact`, and so on;
- `meta/info.json`, with named features and provenance;
- `source.json`, a ready data-config source pointing at the folder.

```bash
uv sync --all-extras
uv run python scripts/convert_dataset.py grandtour  --output data/grandtour --max_missions 2
uv run python scripts/convert_dataset.py mipo       --output data/mipo --pattern "indoor/*" --max_files 20
uv run python scripts/convert_dataset.py neurobem   --output data/neurobem --max_flights 40
uv run python scripts/convert_dataset.py locomujoco --output data/loco_g1 --robot UnitreeG1 --collection Lafan1
uv run python scripts/convert_dataset.py amass_g1   --output data/amass_g1 --pattern "*Walk*" --max_files 200
uv run python scripts/convert_dataset.py lafan1     --output data/lafan1_g1 --robot g1
# then train on it (or combine sources from several source.json files in one config)
uv run python scripts/make_data_config.py --urdf assets/urdf/anymal_d.urdf --dataset data/grandtour --output configs/data/grandtour.json
```

| Converter | Source and format | What it does |
|---|---|---|
| `grandtour` | `leggedrobotics/grand_tour_dataset`, zarr in tar | Downloads only `anymal_state_state_estimator.tar` per mission (about 100 MB). Measured joint torques; pose and quaternion (x, y, z, w); twist in the base frame; contact from each foot's state (1 contact, 2 slip, 0 open). Resamples from 400 Hz to 200 Hz and cuts missions into 20 s episodes. Reads zarr without the zarr package. Use the ANYmal D URDF; the payload pan/tilt joints are frozen. |
| `mipo` | `Open-IO/mipo-go1-odometry`, HDF5 | 400 Hz joints, measured torques, body IMU (orientation, gyro, accelerometer as `imu_acceleration`), foot contact from force counts above 200. Joints `FL0…RR2` are renamed to Go1 URDF names. Mocap is not used (its clock offset is uncalibrated). |
| `neurobem` | UZH `processed_data.zip`, CSV | Reads only the selected flights from the 627 MB remote zip with HTTP range requests. It writes the four motor speeds as rotor joints (signed by spin direction), the Vicon pose and twist, and the accelerometer specific force; the column type is detected from the data. It **identifies the propeller model**: `k_f` is fitted from thrust (R² about 0.9). The roll, pitch, and yaw fits are poor on this data (R² < 0.25), so assumed values are used instead (0.13 m arms in an X layout and `k_m = 0.016 m · k_f`, neither published; the paper gives only the 6-inch Armattan Chameleon frame; maximum speed from the paper's 33 N maximum static thrust). It then writes `neurobem_quadrotor.urdf` and the actuator overrides into `source.json`. Fit qualities are in `meta/info.json`. |
| `locomujoco` | `robfiras/loco-mujoco-datasets`, npz | MuJoCo `qpos`/`qvel` with a free root: position, w-x-y-z quaternion → x-y-z-w, world linear velocity rotated into the base frame, body angular velocity. Joint names come from the file (G1 names match the Unitree URDF; use `--rename` for others). CC BY-NC-ND 4.0. |
| `amass_g1` | `ember-lab-berkeley/AMASS_Retargeted_for_G1`, npz | Named G1 joint positions and the pelvis pose (w-x-y-z quaternion). The stored joint velocities do not match the positions, so velocities are left to finite differencing. |
| `lafan1` | `lvhaidong/LAFAN1_Retargeting_Dataset`, CSV | G1, H1, or H1_2 at 30 Hz: root position and x-y-z-w quaternion, then joints in the order listed in the dataset README. Motions CC BY-NC-ND 4.0. |

The conventions (quaternion order, frames, units, contact codes) were checked
against downloaded files of each dataset. Kinematic-only datasets (LocoMuJoCo,
AMASS, LAFAN1) have no contacts or torques. They need **no torque estimate**: they
train FK, the action effect (unconditioned), motor feasibility on random proposals,
and the contact-constrained response, whose labels need only the configuration and
which feet are down (detected from kinematics). Retargeted motions can exceed actuator
limits.

## Recommended training datasets

The tasks need joint-level trajectories of robots with a known URDF or MJCF.
Below, ✅ means the `huggingface` source reads the dataset directly, ⚙️ means
`scripts/convert_dataset.py` converts it, and 🔧 means it needs your own
conversion to a table with `episode_index`, `frame_index`, and one vector
column per signal (see [Converting datasets](#converting-datasets)). "Torque" says whether measured joint torques are included. They are
used for FD/ID only with `measured_torque_weight > 0` (default 0) and are otherwise
useful to [check torque estimates](dynamics.md#how-accurate-is-it). Without them,
nothing is estimated: the data supervises dynamics through the
[constrained response](dynamics.md#the-contact-constrained-response). Facts were checked against
each dataset's card, `meta/info.json`, or files on 2026-11-07. Check licenses
before use.

**Quadrupeds (floating base)**

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [GrandTour](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) (`leggedrobotics/grand_tour_dataset`) | ANYmal D, 12 | 49 missions in the paper; 48 Hub mission folders have the state-estimator archive | **yes** (joint torque, current, commands, PD gains) | estimator pose and twist, IMUs, GNSS, per-foot contact state and wrench | zarr in tar ⚙️ `grandtour` | MIT on the card; on the website data CC BY-SA 4.0, software MIT | The best fit: measured torques and contacts. Joint names match [anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description) (`download_urdfs.py` fetches it). |
| [MIPO-Go1](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry) (`Open-IO/mipo-go1-odometry`) | Unitree Go1, 12 | 84 files (79 indoor, 5 outdoor), about 1.6 h summed from the card, about 400 Hz | **yes** (from the Unitree SDK) | body IMU with orientation, OptiTrack indoors, 4 foot-force sensors | HDF5 ⚙️ `mipo` | AGPL-3.0 | Recordings of Cerberus 2.0 and the MIPO paper (Yang et al., IROS 2023). Used for the [torque-estimation check](dynamics.md#how-accurate-is-it). Streams are unsynchronized; joint names `FL0`…`RR2` map to `FL_hip_joint`… |
| [DFKI Go2 state estimation](https://zenodo.org/records/19336009) | Unitree Go2, 12 | 27 runs, 1 kHz, 5.4 GB | **yes** (`JointState.effort`) | Vicon / RTK GNSS, IMU, foot contact flags and forces | ROS 2 bags 🔧 | CC BY 4.0 | Needs the dfki-quad message definitions. Popescu et al., Zenodo 2026, DOI 10.5281/zenodo.19336009. |
| [Leg-KILO](https://github.com/ouguangjun/legkilo-dataset) | Unitree Go1, 12 | 7 sequences, 50 Hz unique | unverified | IMU, onboard odometry, foot forces (and lidar) | ROS 1 bags 🔧 | CC BY 4.0 (recordings and ground truth) | Ground truth from offline map optimization. Ou et al., RA-L 2024. |
| [Kine2Go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go) (`MIMUW-Robotics/kine2go`) | Unitree Go2, 12 (simulated) | 800 rollouts, 60 Hz | no (PPO actions and PD gains) | base pose and twist | npy + pickle 🔧 | per clip: CC BY-NC 4.0 / CC BY-SA 4.0 / BSD-3 | Use the PPO actions as `action_key` with the PD gains: the effect and motor tasks then see the commanded targets (nothing is estimated). Open the pickles in a sandbox. Pałucki et al., arXiv:2606.14433. |

**Humanoids and bipeds (floating base)**

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [Unitree G1 whole-body teleop](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit) (`unitreerobotics/G1_WBT_*`, many tasks) | Unitree G1, 29 + hands | e.g. 200 episodes / 168k frames per task, 30 Hz | no | base pose (xyz + quaternion xyzw), projected gravity and angular velocity | LeRobot parquet ✅ | Apache-2.0 | `configs/data/g1_wbt_floating.json`. The state is split into `lower_body`, `left_arm`, `right_arm`. The base position may be estimated. By Unitree Robotics. |
| [Humanoid Everyday](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday) (`USC-PSI-Lab/humanoid-everyday`) | Unitree G1 and H1 (`H1_2` in the project README) + hands | 8,949 episodes / 3.4M frames, 30 Hz | no (`tau_ff` is a command) | IMU and odometry in the combined repo | LeRobot v2.1 parquet ✅ | Apache-2.0 on the card, MIT in the project README | Joint vectors are unnamed; map them by the order in the project README. Zhao et al., arXiv:2510.08807. |
| [LocoMuJoCo datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) (`robfiras/loco-mujoco-datasets`) | 12 models: 9 robots (Unitree G1, H1, H1 v2, Talos, Apollo, Atlas, Booster T1, Fourier GR1T2, ToddlerBot) and 3 human models | 19 default + 40 LAFAN1 motions per robot, 40 Hz | no | floating-base `qpos`/`qvel` with joint names | npz ⚙️ `locomujoco` | CC BY-NC-ND 4.0 (code MIT) | Many morphologies, good for cross-embodiment. Motions are kinematic, so some exceed actuator limits. Al-Hafez et al., arXiv:2311.02496. |
| [LAFAN1 retargeted to Unitree](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (mirror `lvhaidong/LAFAN1_Retargeting_Dataset`) | G1, H1, H1_2 | 40 clips per robot, 30 Hz | no | root position and quaternion (xyzw) | CSV ⚙️ `lafan1` | motions CC BY-NC-ND 4.0 (LAFAN1, Ubisoft La Forge); LICENSE file BSD-3-Clause (Unitree; the card text says MIT for code) | The original `unitreerobotics` repo is no longer publicly accessible. Kinematic only. LAFAN1: Harvey et al., ACM TOG 2020. |
| [AMASS retargeted to G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1) (`ember-lab-berkeley/AMASS_Retargeted_for_G1`) | Unitree G1, 29 | about 17.7k clips, 30 Hz | no | pelvis pose, joint positions and velocities, named | npz ⚙️ `amass_g1` | CC BY 4.0 on the card; the upstream retargeting (`fleaven/Retargeted_AMASS_for_robotics`, Kun Zhao) keeps the AMASS terms (non-commercial research only) | Large motion variety; kinematic only. AMASS: Mahmood et al., ICCV 2019. |

**Multirotors (floating base, thrust actuators)**

| Dataset | Vehicle | Size | Rotor signal | Base | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [NeuroBEM](https://rpg.ifi.uzh.ch/NeuroBEM.html) | quadrotor, 0.772 kg (dataset Readme; the paper says 752 g) | 1 h 15 min, 400 Hz | **motor speeds** (rad/s) and their derivatives | Vicon pose, velocity, acceleration, body rates | CSV ⚙️ `neurobem` | not stated | The best rotor-dynamics set. The converter identifies the thrust coefficient, generates the URDF, and writes the actuator overrides. Bauersfeld et al., RSS 2021. |
| [Blackbird](https://github.com/mit-aera/Blackbird-Dataset) | quadrotor | 168 flights, >10 h; RPM about 190 Hz | **per-motor RPM** and PWM | 360 Hz mocap pose, IMU | ROS 1 bags (4.9 TB) 🔧 | repo MIT; data unverified | The official host was unreachable at check time; an Academic Torrents mirror exists. Antonini et al., IJRR 2020. |
| [TII drone racing](https://github.com/tii-racing/drone-racing-dataset) | 5-inch FPV quad (open design) | 42 flights (18 piloted and 18 autonomous, plus 6 piloted in release v3.0.0), 500 Hz | motor commands (0–1), no RPM | mocap pose and twist, IMU | CSV and ROS 2 bags 🔧 | data CC BY 4.0; code MIT | Commands are not speeds; map them with a thrust curve. Bosello et al., RA-L 2024. |
| [UAV-SEAD](https://huggingface.co/datasets/aykutkabaoglu/UAV-SEAD_State_Estimation_Anomaly_Dataset) (`aykutkabaoglu/UAV-SEAD_…`) | PX4 quadrotors in six airframe configurations (paper) | 1,396 flights, about 52 h | PWM outputs | attitude, rates, local position, IMU | PX4 ulog + CSV 🔧 | CC BY 4.0 | Airframes not identified per flight. Kabaoglu and Sariel, arXiv:2602.13900; the paper's own card is [`aykutkabaoglu/uav-flight-anomaly-dataset`](https://huggingface.co/datasets/aykutkabaoglu/uav-flight-anomaly-dataset) (DOI 10.57967/hf/7772). |
| Synthetic (`configs/data/quadrotor_flight.json`) | any multirotor URDF | unlimited | exact rotor speeds and torques | exact | built in ✅ | — | Exact thrust, drag, and base dynamics; randomized mass and arm length. |

**Legged robots on the ground (simulated)**

| Source | Robots | Labels | Notes |
|---|---|---|---|
| `"type": "mujoco"` (`configs/data/ground_contact.json`) | any legged URDF; poses provided for Go1, ANYmal D, G1, H1 | exact torques, accelerations, contacts, pushes | Standing, crouching, weight shifting, and push recovery with real foot contact; see [Ground contact](#ground-contact-in-synthetic-data-mujoco). |

**Fixed-base arms and hands (and arms of standing humanoids)**

| Dataset | Robot | Size | Torque | Format | Notes |
|---|---|---|---|---|---|
| [`unitreerobotics/G1_Dex3_*`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) | G1 arms 14 + Dex3 hands | e.g. 418 episodes / 352k frames, 30 Hz | no | LeRobot ✅ | `configs/data/g1_hub_plus_synthetic.json` (legs frozen, base fixed). By Unitree Robotics, Apache-2.0. |
| [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | Franka Panda, 7 | 95,658 episodes / 27.6M frames, 15 Hz | no | LeRobot ✅ | `state_key: "observation.state.joint_position"`; use `max_episodes`. DROID, Khazatsky et al., RSS 2024; the paper releases the data under CC BY 4.0, the `lerobot` card says Apache-2.0. |
| [`lerobot/aloha_sim_*`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | 2 × ViperX 300 | 50 episodes / 25k frames, 50 Hz | no | LeRobot ✅ | Map 12 arm joints; skip the grippers. ALOHA, Zhao et al., RSS 2023; card MIT. |
| [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | SO-100 arm, 5 + gripper | 50 episodes / 19.6k frames, 30 Hz | no | LeRobot ✅ | Check position units; convert with `state_scale`/`state_offset`. From SmolVLA, Shukor et al., arXiv:2506.01844; card Apache-2.0. |

**Robot models.** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
has `unitree_g1`, `unitree_h1`, `unitree_go1`, `unitree_go2`, `booster_t1`,
`apptronik_apollo`, `pal_talos`, `agility_cassie`, `anybotics_anymal_c`, and
`bitcraze_crazyflie_2` (MJCF; convert to URDF; Apache-2.0 overall, each model has its
own license). Unitree's
[unitree_ros](https://github.com/unitreerobotics/unitree_ros) (BSD-3-Clause) has URDFs for
G1, H1, H1_2, Go1, and Go2 (`scripts/download_urdfs.py` fetches three). The Go1, H1 and
G1 URDFs used by `configs/data/*.json` come from it, the ANYmal D URDF from
[anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description)
(ANYbotics AG, BSD-3-Clause); `assets/robots/quadrotor_x.urdf` is this repository's.

**A suggested mix for cross-embodiment pretraining:**
- the quadruped sets with contact sensors (GrandTour, MIPO, DFKI Go2) for real
  locomotion with known contact sets (their torques can be enabled with
  `measured_torque_weight`, or kept for validation);
- MuJoCo ground data for exact contact dynamics of every legged robot you
  care about;
- G1 WBT and LocoMuJoCo for humanoid motion;
- NeuroBEM for real rotor dynamics;
- synthetic fixed, floating, and flight data for every robot you care about
  (exact torques, probes, limits, randomized inertia and rotor inertia): the only
  source of full-torque B and C labels besides MuJoCo;
- held-out robots or randomized variants for evaluation.

## References

Checked against the publisher, proceedings, arXiv, dataset card or repository on
2026-11-20. Licenses are those stated by the holders; check them before use.

**Simulation and robot models**
- Emanuel Todorov, Tom Erez, and Yuval Tassa. *MuJoCo: A physics engine for model-based
  control.* IROS 2012, pp. 5026–5033. DOI
  [10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109). Code:
  <https://github.com/google-deepmind/mujoco> (Apache-2.0); documentation (contact
  `solimp`, integrators): <https://mujoco.readthedocs.io/en/stable/modeling.html>.
- S. Kajita, F. Kanehiro, K. Kaneko, K. Yokoi, and H. Hirukawa.
  *The 3D linear inverted pendulum mode: a simple modeling for a biped walking pattern
  generation.* IROS 2001, vol. 1, pp. 239–246. DOI
  [10.1109/IROS.2001.973365](https://doi.org/10.1109/IROS.2001.973365) (spotter frequency).
- Kevin Zakka, Yuval Tassa, and MuJoCo Menagerie Contributors. *MuJoCo Menagerie: A
  collection of high-quality simulation models for MuJoCo.* 2022.
  <https://github.com/google-deepmind/mujoco_menagerie> (Apache-2.0; per-model licenses).
- Unitree Robotics. *unitree_ros* (Go1, H1, G1 URDFs).
  <https://github.com/unitreerobotics/unitree_ros> (BSD-3-Clause).
- ANYbotics AG. *anymal_d_simple_description* (ANYmal D URDF).
  <https://github.com/ANYbotics/anymal_d_simple_description> (BSD-3-Clause).

**Formats and tools**
- Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, Steven
  Palma, Pepijn Kooijmans, Michel Aractingi, Mustafa Shukor, Dana Aubakirova, Martino
  Russi, Francesco Capuano, Caroline Pascal, Jade Choghari, Khalil Meftah, Maxime
  Ellerbach, Jess Moss, and Thomas Wolf. *LeRobot: State-of-the-art Machine Learning for
  Real-World Robotics in Pytorch.* 2024. <https://github.com/huggingface/lerobot>
  (Apache-2.0); dataset format v3.0 (`data/chunk-000/file-000.parquet`, `meta/info.json`).
  Paper: Remi Cadene et al. *LeRobot: An Open-Source Library for End-to-End Robot
  Learning.* ICLR 2026, arXiv:2602.22818.
- Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick
  von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall,
  Joe Davison, Mario Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le
  Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid,
  Sylvain Gugger, Clément Delangue, Théo Matussière, Lysandre Debut, Stas Bekman,
  Pierric Cistac, Thibault Goehringer, Victor Mustar, François Lagunas, Alexander Rush,
  and Thomas Wolf. *Datasets: A Community Library for Natural Language Processing.*
  EMNLP 2021 System Demonstrations, pp. 175–184. DOI
  [10.18653/v1/2021.emnlp-demo.21](https://doi.org/10.18653/v1/2021.emnlp-demo.21).
- *Zarr Storage Specification Version 2.*
  <https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html>; chunks decoded with
  numcodecs, <https://github.com/zarr-developers/numcodecs> (MIT).
- *Apache Parquet file format.* <https://parquet.apache.org/docs/file-format/>; written
  with PyArrow, <https://arrow.apache.org/> (Apache-2.0).
- Roy T. Fielding, Mark Nottingham, and Julian Reschke (eds.). *HTTP Semantics.* RFC 9110,
  2022, Sec. 14 (range requests). DOI [10.17487/RFC9110](https://doi.org/10.17487/RFC9110).

**Quadruped datasets**
- Turcan Tuna, Jonas Frey, Frank Fu, Katharine Patterson, Tianao Xu, Maurice Fallon,
  Cesar Cadena, and Marco Hutter. *GrandTour: A Legged Robotics Dataset in the Wild for
  Multi-Modal Perception and State Estimation.* arXiv:2602.18164, 2026 (Tuna and Frey
  contributed equally). Data:
  <https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset> (card: MIT),
  <https://grand-tour.leggedrobotics.com/> (data CC BY-SA 4.0, software MIT).
- Shuo Yang, Zixin Zhang, Benjamin Bokser, and Zachary Manchester. *Multi-IMU
  Proprioceptive Odometry for Legged Robots.* IROS 2023, pp. 774–779. DOI
  [10.1109/IROS55552.2023.10342061](https://doi.org/10.1109/IROS55552.2023.10342061).
  Data: <https://huggingface.co/datasets/Open-IO/mipo-go1-odometry> (AGPL-3.0), recordings
  of Cerberus 2.0, <https://github.com/ShuoYangRobotics/Cerberus2.0> (AGPL-3.0), which
  builds on Shuo Yang, Zixin Zhang, Zhengyu Fu, and Zachary Manchester. *Cerberus:
  Low-Drift Visual-Inertial-Leg Odometry For Agile Locomotion.* ICRA 2023, pp. 4193–4199.
  DOI [10.1109/ICRA48891.2023.10160486](https://doi.org/10.1109/ICRA48891.2023.10160486).
- Mihaela Popescu, Franek Stark, Hannah Isermann, Rohit Kumar, Jonas Haack, and Shubham
  Vyas. *Unitree Go2 Quadruped State Estimation.* Zenodo, 2026. DOI
  [10.5281/zenodo.19336009](https://doi.org/10.5281/zenodo.19336009) (CC BY 4.0).
- Guangjun Ou, Dong Li, and Hanmin Li. *Leg-KILO: Robust Kinematic-Inertial-Lidar
  Odometry for Dynamic Legged Robots.* IEEE RA-L 9(10):8194–8201, 2024. DOI
  [10.1109/LRA.2024.3440730](https://doi.org/10.1109/LRA.2024.3440730). Data:
  <https://github.com/ouguangjun/legkilo-dataset> (CC BY 4.0).
- Władysław Pałucki, Paweł Siwak, Krzysztof Ciebiera, and Marek Cygan. *Kine2Go:
  Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions.*
  arXiv:2606.14433, 2026. Data: <https://huggingface.co/datasets/MIMUW-Robotics/kine2go>
  (per-clip CC BY-NC 4.0, CC BY-SA 4.0 or BSD-3-Clause).

**Humanoid datasets**
- Unitree Robotics. *G1 whole-body teleoperation and Dex3 datasets*
  (`unitreerobotics/G1_WBT_*`, `unitreerobotics/G1_Dex3_*`).
  <https://huggingface.co/unitreerobotics> (Apache-2.0 on the cards used here).
- Zhenyu Zhao, Hongyi Jing, Xiawei Liu, Jiageng Mao, Abha Jha, Hanwen Yang, Rong Xue,
  Sergey Zakharov, Vitor Guizilini, and Yue Wang. *Humanoid Everyday: A Comprehensive
  Robotic Dataset for Open-World Humanoid Manipulation.* arXiv:2510.08807, 2025. Data:
  <https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday> (card Apache-2.0,
  project README MIT).
- Firas Al-Hafez, Guoping Zhao, Jan Peters, and Davide Tateo. *LocoMuJoCo: A
  Comprehensive Imitation Learning Benchmark for Locomotion.* arXiv:2311.02496, 2023.
  Data: <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets> (CC BY-NC-ND 4.0);
  code <https://github.com/robfiras/loco-mujoco> (MIT).
- Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. *Robust Motion
  In-betweening.* ACM Transactions on Graphics 39(4), Article 60, 2020 (SIGGRAPH 2020). DOI
  [10.1145/3386569.3392480](https://doi.org/10.1145/3386569.3392480). LAFAN1 data:
  <https://github.com/ubisoft/ubisoft-laforge-animation-dataset> (Ubisoft La Forge,
  CC BY-NC-ND 4.0). Retargeted by Unitree Robotics; copy used here:
  <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset> (LICENSE file
  BSD-3-Clause).
- Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J.
  Black. *AMASS: Archive of Motion Capture as Surface Shapes.* ICCV 2019, pp. 5441–5450.
  DOI [10.1109/ICCV.2019.00554](https://doi.org/10.1109/ICCV.2019.00554). License:
  <https://amass.is.tue.mpg.de/license.html> (non-commercial scientific research). G1
  retargeting: <https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1>
  (card CC BY 4.0), taken from Kun Zhao, *Retargeted AMASS for Robotics*,
  <https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics>, which asks to
  also cite Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A.
  Osman, Dimitrios Tzionas, and Michael J. Black. *Expressive Body Capture: 3D Hands,
  Face, and Body From a Single Image.* CVPR 2019, pp. 10967–10977. DOI
  [10.1109/CVPR.2019.01123](https://doi.org/10.1109/CVPR.2019.01123).

**Multirotor datasets**
- Leonard Bauersfeld, Elia Kaufmann, Philipp Foehn, Sihao Sun, and Davide Scaramuzza.
  *NeuroBEM: Hybrid Aerodynamic Quadrotor Model.* RSS 2021. DOI
  [10.15607/RSS.2021.XVII.042](https://doi.org/10.15607/RSS.2021.XVII.042),
  arXiv:2106.08015. Data: <https://rpg.ifi.uzh.ch/NeuroBEM.html> (no license stated).
- Amado Antonini, Winter Guerra, Varun Murali, Thomas Sayre-McCord, and Sertac Karaman.
  *The Blackbird UAV dataset.* IJRR 39(10–11):1346–1364, 2020. DOI
  [10.1177/0278364920908331](https://doi.org/10.1177/0278364920908331). Code:
  <https://github.com/mit-aera/Blackbird-Dataset> (MIT).
- Michael Bosello, Davide Aguiari, Yvo Keuter, Enrico Pallotta, Sara Kiade, Gyordan
  Caminati, Flavio Pinzarrone, Junaid Halepota, Jacopo Panerati, and Giovanni Pau. *Race
  Against the Machine: A Fully-Annotated, Open-Design Dataset of Autonomous and Piloted
  High-Speed Flight.* IEEE RA-L 9(4):3799–3806, 2024. DOI
  [10.1109/LRA.2024.3371288](https://doi.org/10.1109/LRA.2024.3371288). Data:
  <https://github.com/tii-racing/drone-racing-dataset> (data CC BY 4.0, code MIT).
- Aykut Kabaoglu and Sanem Sariel. *UAV-SEAD: State Estimation Anomaly Dataset for UAVs.*
  arXiv:2602.13900, 2026 (CC BY 4.0).

**Arm datasets**
- Alexander Khazatsky, Karl Pertsch, Suraj Nair, et al. (98 authors in the proceedings).
  *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset.* RSS 2024. DOI
  [10.15607/RSS.2024.XX.120](https://doi.org/10.15607/RSS.2024.XX.120), arXiv:2403.12945.
  The paper releases the data under CC BY 4.0.
- Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. *Learning Fine-Grained
  Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. DOI
  [10.15607/RSS.2023.XIX.016](https://doi.org/10.15607/RSS.2023.XIX.016),
  arXiv:2304.13705 (ALOHA simulation data).
- Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma,
  Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon
  Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. *SmolVLA: A Vision-Language-Action
  Model for Affordable and Efficient Robotics.* arXiv:2506.01844, 2025
  (`lerobot/svla_so100_pickplace`).
