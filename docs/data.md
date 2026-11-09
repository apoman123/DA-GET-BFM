# Training data

[← Back to the README](../README.md)

Where training data comes from: simulated ground contact, automatic configs, dataset converters, and recommended datasets. The config format itself is in [Physics pretraining](training.md#data-config).

## Contents

- [Ground contact in synthetic data (MuJoCo)](#ground-contact-in-synthetic-data-mujoco)
- [Writing data configs automatically](#writing-data-configs-automatically)
- [Converting datasets](#converting-datasets)
- [Recommended training datasets](#recommended-training-datasets)

## Ground contact in synthetic data (MuJoCo)

The built-in `synthetic` floating mode hangs the robot from a harness, so no
foot touches the ground. Legged data with real foot-ground contact comes from
the `"type": "mujoco"` source (`model/training/ground_sim.py`, `uv sync --extra
sim`). It builds a MuJoCo model from the same embodiment graph:

- one body per token with the graph's merged inertia (so MuJoCo and our
  dynamics agree; the mass matrix and bias forces match to 1e-7 in the tests);
- a free root joint, and hinge or slide joints with the URDF limits, damping,
  and friction;
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
steadying the robot. It carries about 5–10% of the weight in the tests and
runs; the feet carry the rest.

**Recorded labels are exact for the simulated system:**
- applied joint torques;
- joint and base accelerations;
- per-foot contact flags;
- the push and spotter wrench (passed as a known base wrench);
- probe torques solved by MuJoCo with the contacts in place, so task B covers
  action conditioning while standing on the ground.

The inverse-dynamics labels equal the applied torques, and no contact
estimation is needed.

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
shifting, partial foot lift, and recovering from pushes, not gaits; recorded
datasets provide locomotion.

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
| `neurobem` | UZH `processed_data.zip`, CSV | Reads only the selected flights from the 627 MB remote zip with HTTP range requests. It writes the four motor speeds as rotor joints (signed by spin direction), the Vicon pose and twist, and the accelerometer specific force; the column type is detected from the data. It **identifies the propeller model**: `k_f` is fitted from thrust (R² about 0.9). The roll, pitch, and yaw fits are poor on this data (R² < 0.25), so the published frame size is used instead (6-inch frame, 0.13 m arms, `k_m = 0.016 m · k_f`, maximum speed from the paper's 33 N total thrust). It then writes `neurobem_quadrotor.urdf` and the actuator overrides into `source.json`. Fit qualities are in `meta/info.json`. |
| `locomujoco` | `robfiras/loco-mujoco-datasets`, npz | MuJoCo `qpos`/`qvel` with a free root: position, w-x-y-z quaternion → x-y-z-w, world linear velocity rotated into the base frame, body angular velocity. Joint names come from the file (G1 names match the Unitree URDF; use `--rename` for others). CC BY-NC-ND 4.0. |
| `amass_g1` | `ember-lab-berkeley/AMASS_Retargeted_for_G1`, npz | Named G1 joint positions and the pelvis pose (w-x-y-z quaternion). The stored joint velocities do not match the positions, so velocities are left to finite differencing. |
| `lafan1` | `lvhaidong/LAFAN1_Retargeting_Dataset`, CSV | G1, H1, or H1_2 at 30 Hz: root position and x-y-z-w quaternion, then joints in the order listed in the dataset README. Motions CC BY-NC-ND 4.0. |

The conventions (quaternion order, frames, units, contact codes) were checked
against downloaded files of each dataset. Kinematic-only datasets (LocoMuJoCo,
AMASS, LAFAN1) have no contacts or torques; contacts are detected and torques
estimated, and retargeted motions can exceed actuator limits.

## Recommended training datasets

The tasks need joint-level trajectories of robots with a known URDF or MJCF.
Below, ✅ means the `huggingface` source reads the dataset directly, ⚙️ means
`scripts/convert_dataset.py` converts it, and 🔧 means it needs your own
conversion to a table with `episode_index`, `frame_index`, and one vector
column per signal (see [Converting datasets](#converting-datasets)). "Torque" says whether measured joint torques are included. Without
them, torques are
[estimated](dynamics.md#estimating-joint-torques-without-torque-sensors). Facts were checked against
each dataset's card, `meta/info.json`, or files on 2026-11-07. Check licenses
before use.

**Quadrupeds (floating base)**

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [GrandTour](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) (`leggedrobotics/grand_tour_dataset`) | ANYmal D, 12 | 52 missions | **yes** (joint torque, current, commands, PD gains) | estimator pose and twist, IMUs, GNSS, per-foot contact state and wrench | zarr in tar ⚙️ `grandtour` | MIT on the card, CC BY-SA 4.0 on the website | The best fit: measured torques and contacts. Joint names match [anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description) (`download_urdfs.py` fetches it). |
| [MIPO-Go1](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry) (`Open-IO/mipo-go1-odometry`) | Unitree Go1, 12 | 84 runs, about 1.6 h, 400 Hz | **yes** (current-based estimate) | body IMU with orientation, OptiTrack indoors, 4 foot-force sensors | HDF5 ⚙️ `mipo` | AGPL-3.0 | Used for the [torque-estimation check](dynamics.md#estimating-joint-torques-without-torque-sensors). Streams are unsynchronized; joint names `FL0`…`RR2` map to `FL_hip_joint`… |
| [DFKI Go2 state estimation](https://zenodo.org/records/19336009) | Unitree Go2, 12 | 27 runs, 1 kHz, 5.4 GB | **yes** (`JointState.effort`) | Vicon / RTK GNSS, IMU, foot contact flags and forces | ROS 2 bags 🔧 | CC BY 4.0 | Needs the dfki-quad message definitions. |
| [Leg-KILO](https://github.com/ouguangjun/legkilo-dataset) | Unitree Go1, 12 | 7 sequences, 50 Hz unique | unverified | IMU, onboard odometry, foot forces | ROS 1 bags 🔧 | CC BY 4.0 | Ground truth from offline map optimization. |
| [Kine2Go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go) (`MIMUW-Robotics/kine2go`) | Unitree Go2, 12 (simulated) | 800 rollouts, 60 Hz | no (PPO actions and PD gains) | base pose and twist | npy + pickle 🔧 | per clip: CC BY-NC 4.0 / CC BY-SA 4.0 / BSD-3 | Rebuild PD torques from actions and gains, or estimate. Open the pickles in a sandbox. |

**Humanoids and bipeds (floating base)**

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [Unitree G1 whole-body teleop](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit) (`unitreerobotics/G1_WBT_*`, many tasks) | Unitree G1, 29 + hands | e.g. 200 episodes / 168k frames per task, 30 Hz | no | base pose (xyz + quaternion xyzw), projected gravity and angular velocity | LeRobot parquet ✅ | Apache-2.0 | `configs/data/g1_wbt_floating.json`. The state is split into `lower_body`, `left_arm`, `right_arm`. The base position may be estimated. |
| [Humanoid Everyday](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday) (`USC-PSI-Lab/humanoid-everyday`) | Unitree G1 and H1_2 + hands | 8,949 episodes / 3.4M frames, 30 Hz | no (`tau_ff` is a command) | IMU and odometry in the combined repo | LeRobot v2.1 parquet ✅ | Apache-2.0 (card) | Joint vectors are unnamed; map them by the order in the project README. |
| [LocoMuJoCo datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) (`robfiras/loco-mujoco-datasets`) | 12 models: Unitree G1, H1, H1 v2, Talos, Apollo, Atlas, Booster T1, Fourier GR1T2, ToddlerBot, … | 19 default + 40 LAFAN1 motions per robot, 40 Hz | no | floating-base `qpos`/`qvel` with joint names | npz ⚙️ `locomujoco` | CC BY-NC-ND 4.0 | Many morphologies, good for cross-embodiment. Motions are kinematic, so some exceed actuator limits. |
| [LAFAN1 retargeted to Unitree](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (mirror `lvhaidong/LAFAN1_Retargeting_Dataset`) | G1, H1, H1_2 | 40 clips per robot, 30 Hz | no | root position and quaternion (xyzw) | CSV ⚙️ `lafan1` | motions CC BY-NC-ND 4.0 (LAFAN1); retargeting BSD-3 | The original `unitreerobotics` repo is no longer accessible. Kinematic only. |
| [AMASS retargeted to G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1) (`ember-lab-berkeley/AMASS_Retargeted_for_G1`) | Unitree G1, 29 | about 17.7k clips, 30 Hz | no | pelvis pose, joint positions and velocities, named | npz ⚙️ `amass_g1` | CC BY 4.0 on the card; AMASS terms likely apply | Large motion variety; kinematic only. |

**Multirotors (floating base, thrust actuators)**

| Dataset | Vehicle | Size | Rotor signal | Base | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [NeuroBEM](https://rpg.ifi.uzh.ch/NeuroBEM.html) | quadrotor, 0.772 kg | 1 h 15 min, 400 Hz | **motor speeds** (rad/s) and their derivatives | Vicon pose, velocity, acceleration, body rates | CSV ⚙️ `neurobem` | not stated | The best rotor-dynamics set. The converter identifies the thrust coefficient, generates the URDF, and writes the actuator overrides. |
| [Blackbird](https://github.com/mit-aera/Blackbird-Dataset) | quadrotor | 168 flights, >10 h; RPM about 190 Hz | **per-motor RPM** and PWM | 360 Hz mocap pose, IMU | ROS 1 bags (4.9 TB) 🔧 | repo MIT; data unverified | The official host was unreachable at check time; an Academic Torrents mirror exists. |
| [TII drone racing](https://github.com/tii-racing/drone-racing-dataset) | 5-inch FPV quad (open design) | 42 flights, 500 Hz | motor commands (0–1), no RPM | mocap pose and twist, IMU | CSV 🔧 | MIT | Commands are not speeds; map them with a thrust curve. |
| [UAV-SEAD](https://huggingface.co/datasets/aykutkabaoglu/UAV-SEAD_State_Estimation_Anomaly_Dataset) (`aykutkabaoglu/UAV-SEAD_…`) | various PX4 multirotors | 1,396 flights, about 52 h | PWM outputs | attitude, rates, local position, IMU | PX4 ulog + CSV 🔧 | CC BY 4.0 | Mixed, unidentified airframes. |
| Synthetic (`configs/data/quadrotor_flight.json`) | any multirotor URDF | unlimited | exact rotor speeds and torques | exact | built in ✅ | — | Exact thrust, drag, and base dynamics; randomized mass and arm length. |

**Legged robots on the ground (simulated)**

| Source | Robots | Labels | Notes |
|---|---|---|---|
| `"type": "mujoco"` (`configs/data/ground_contact.json`) | any legged URDF; poses provided for Go1, ANYmal D, G1, H1 | exact torques, accelerations, contacts, pushes | Standing, crouching, weight shifting, and push recovery with real foot contact; see [Ground contact](#ground-contact-in-synthetic-data-mujoco). |

**Fixed-base arms and hands (and arms of standing humanoids)**

| Dataset | Robot | Size | Torque | Format | Notes |
|---|---|---|---|---|---|
| [`unitreerobotics/G1_Dex3_*`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) | G1 arms 14 + Dex3 hands | e.g. 418 episodes / 352k frames, 30 Hz | no | LeRobot ✅ | `configs/data/g1_hub_plus_synthetic.json` (legs frozen, base fixed). |
| [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | Franka Panda, 7 | 95,658 episodes / 27.6M frames, 15 Hz | no | LeRobot ✅ | `state_key: "observation.state.joint_position"`; use `max_episodes`. |
| [`lerobot/aloha_sim_*`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | 2 × ViperX 300 | 50 episodes / 25k frames, 50 Hz | no | LeRobot ✅ | Map 12 arm joints; skip the grippers. |
| [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | SO-100 arm, 5 + gripper | 50 episodes / 19.6k frames, 30 Hz | no | LeRobot ✅ | Check position units; convert with `state_scale`/`state_offset`. |

**Robot models.** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
has `unitree_g1`, `unitree_h1`, `unitree_go1`, `unitree_go2`, `booster_t1`,
`apptronik_apollo`, `pal_talos`, `agility_cassie`, `anybotics_anymal_c`, and
`bitcraze_crazyflie_2` (MJCF; convert to URDF). Unitree's
[unitree_ros](https://github.com/unitreerobotics/unitree_ros) has URDFs for
G1, H1, H1_2, Go1, and Go2 (`scripts/download_urdfs.py` fetches three).

**A suggested mix for cross-embodiment pretraining:**
- the measured-torque quadruped sets (GrandTour, MIPO, DFKI Go2) for real
  dynamics;
- MuJoCo ground data for exact contact dynamics of every legged robot you
  care about;
- G1 WBT and LocoMuJoCo for humanoid motion;
- NeuroBEM for real rotor dynamics;
- synthetic fixed, floating, and flight data for every robot you care about
  (exact torques, probes, limits);
- held-out robots or randomized variants for evaluation.
