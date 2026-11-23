# Training data

[← Back to the README](../README.md)

Where training data comes from: simulated ground contact, automatic configs, dataset converters, and a catalog of datasets to download for pretraining and RL. The config format itself is in [Physics pretraining](training.md#data-config).

## Contents

- [Ground contact in synthetic data (MuJoCo)](#ground-contact-in-synthetic-data-mujoco)
- [Writing data configs automatically](#writing-data-configs-automatically)
- [Downloading and converting datasets](#downloading-and-converting-datasets)
- [Converting datasets](#converting-datasets)
- [Recommended training datasets](#recommended-training-datasets)
  - [What to download first](#what-to-download-first)
  - [Quadrupeds: real robots](#quadrupeds-real-robots)
  - [Quadruped motions (kinematic)](#quadruped-motions-kinematic)
  - [Quadrupeds: simulated](#quadrupeds-simulated)
  - [Legged robots on the ground (simulated)](#legged-robots-on-the-ground-simulated)
  - [Humanoids: real robots](#humanoids-real-robots)
  - [Humanoid motions (retargeted)](#humanoid-motions-retargeted)
  - [Simulated: many embodiments and exact torques](#simulated-many-embodiments-and-exact-torques)
  - [Arms and hands (fixed base)](#arms-and-hands-fixed-base)
  - [Gated, unreleased, or unusable](#gated-unreleased-or-unusable)
  - [Multirotors (floating base, thrust actuators)](#multirotors-floating-base-thrust-actuators)
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
uv run python scripts/make_data_config.py --urdf assets/urdf/g1_29dof_rev_1_0.urdf \
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

## Downloading and converting datasets

`scripts/download_datasets.py` downloads the datasets of a catalog
(`model/conversion/catalog.py`) into folders you choose and converts them.
`scripts/convert_datasets.py` converts what was downloaded, without the network. Both take
the same names and folder options:

```bash
uv run python scripts/download_datasets.py --list                       # names, sizes, licenses
uv run python scripts/download_datasets.py --all --root /mnt/data/bfm --dry_run
uv run python scripts/download_datasets.py --dataset lafan1_g1,xwbc_g1_lafan,g1_wbt_smpl
uv run python scripts/download_datasets.py --dataset amass_g1 --path amass_g1=/mnt/big/amass_g1 --pattern "*Walk*" --limit 200
uv run python scripts/download_datasets.py --dataset g1_wbt --no-convert   # download now
uv run python scripts/convert_datasets.py  --dataset g1_wbt                # convert later, offline
```

| Option | Meaning |
|---|---|
| `--dataset NAME` (repeatable or comma-separated), `--all`, `--list` | Which datasets |
| `--root DIR` | Parent of the dataset folders (default `data`, which git ignores) |
| `--path NAME=DIR` | The folder of one dataset instead of `<root>/<NAME>` |
| `--raw_root DIR` | Downloads in `<DIR>/<NAME>` instead of `<folder>/raw`, e.g. on another disk |
| `--pattern [NAME=]GLOB`, `--limit N` | Part of a dataset: globs over the repository paths (repository names for the Unitree sets, archive members for xwbc, CSV members for NeuroBEM), and the first N in sorted order |
| `--convert` / `--no-convert` | Convert after downloading (download script; default on) |
| `--hf_dataset` / `--no-hf_dataset` | Also save every source as a Hugging Face dataset (default on) |
| `--force`, `--redownload` | Convert again; download again |
| `--delete_raw` | Delete the downloads after converting, unless a source reads them |
| `--dry_run`, `--workers N` | List what would be downloaded; parallel downloads (default 8) |

A dataset folder holds:

```
<folder>/
  raw/            downloads, at their paths in the source repository
  data/, meta/    the converted table, with source.json (converted datasets)
  hf/             the Hugging Face copies
  sources.json    the sources to paste into a data config (the hf/ copies when they exist)
  catalog.json    which files were converted, with which selection
```

Files already downloaded with the right size are skipped. A conversion that
`catalog.json` records for the same files and selection is not redone, so an interrupted
run can simply be started again. A download stops before it starts if it does not fit in
the free space of its disk. The download script runs `scripts/download_urdfs.py` for the
robot models.

**Hugging Face copies.** With `--hf_dataset`, every source is also written with
`datasets.Dataset.save_to_disk` (Arrow files of at most 1 GB) and keeps only the columns
the source reads. The `huggingface` source opens such a folder with
`datasets.load_from_disk`, memory-mapped. Reading the parquet tables instead makes
`datasets` build an Arrow copy of every column in its cache (`HF_DATASETS_CACHE`, under the
home directory by default). The copy in the dataset folder keeps that data where you put
the dataset. `scripts/convert_dataset.py` has the same flag.

| Name | Download | What | License |
|---|---|---|---|
| `grandtour` | 5.3 GB | ANYmal D: measured joint torques, foot contact states, base state; 48 missions | data CC BY-SA 4.0 (website), MIT on the card |
| `mipo` | 0.53 GB | Unitree Go1: measured torques, IMU, foot forces; 84 sequences | AGPL-3.0 |
| `neurobem` | 0.63 GB | Quadrotor flights; the propeller model is identified from the data | none stated |
| `locomujoco_g1` | 0.10 GB | LocoMuJoCo G1 motions, default and LAFAN1 (two sources). Its G1 has 23 joints: no waist roll and pitch, no wrist pitch and yaw, which stay frozen | CC BY-NC-ND 4.0 |
| `amass_g1` | 6.95 GB | AMASS on G1 by fleaven, 17,717 clips; wrist pitch and yaw are 0 | AMASS terms |
| `xwbc_g1_amass` | 8.10 GB | AMASS on G1 by xwbc, one archive; every joint moves | AMASS terms |
| `xwbc_g1_lafan` | 0.84 GB | LAFAN1 on G1 by xwbc, one archive | CC BY-NC-ND 4.0 |
| `lafan1_g1`, `lafan1_h1` | 0.08 GB each | LAFAN1 on G1 or H1 by Unitree, 40 clips each | CC BY-NC-ND 4.0; retargeting BSD-3-Clause |
| `g1_wbt` | 13.4 GB | 55 `unitreerobotics/G1_WBT_*` repositories with the split joint vectors; parquet and `meta/` only (the videos are 1.26 TB) | Apache-2.0 |
| `g1_wbt_q36` | 0.45 GB | 4 `G1_WBT_Inspire_*` repositories with `robot_q_current` [36] | Apache-2.0 |
| `g1_wbt_smpl` | 3.8 GB | 18 `G1_WBT_SMPL_*` repositories | Apache-2.0 |
| `g1_dex3` | 0.08 GB | `G1_Dex3_ToastedBread_Dataset`: arms on a fixed base | Apache-2.0 |

Sizes are those of 2026-11-23; `--all` downloads about 40 GB. The Unitree repositories are
sorted by the schema of their `meta/info.json`, and each gets its own source. The `g1_wbt`
source is that of `configs/data/g1_wbt_floating.json`, without its 40-episode limit; the
one repository without `state_base_rot` (`G1_WBT_Inspire_Put_Sink_Towel`) has no angular
velocity. `g1_wbt_q36` and `g1_wbt_smpl` read `robot_q_current`: root position,
quaternion (w, x, y, z), then the 29 joints in URDF order (checked on one file of each,
[below](#g1-joint-order-check)). Both have **no actions**: the SMPL repositories record no
joint commands, and in the q36 ones it is not confirmed that `robot_q_desired` is the motor
command, so it is not used ([details](#humanoids-real-robots)). `locomujoco_g1` has 23 of
the 29 joints; the other six stay frozen.

The converters stream their episodes, so memory stays bounded on whole datasets. The
training pipeline then builds its cache with worker processes
([data config](training.md#data-config)): 107 xwbc G1 clips at 50 Hz (114,024 training
samples) take 21.6 s with 16 workers, against 359.7 s before. At that rate the
about 13k clips of `xwbc_g1_amass` would take roughly 45 minutes (an extrapolation from
these 107 ACCAD clips, which average 233 frames at 30 Hz).

On 2026-11-23 a small part of every dataset was downloaded and converted with these
scripts, except `xwbc_g1_amass` (one 8.1 GB archive; `xwbc_g1_lafan` takes the same path)
and `neurobem` (download.ifi.uzh.ch did not answer from the test machine; the URL download
is tested on a local file). Every source loaded with the training loader, and the Hugging
Face copies gave the same trajectories as the tables.

## Converting datasets

`scripts/convert_dataset.py` (`model/conversion/`) converts one dataset that is not
LeRobot parquet into one table format, with every option of its converter, downloading
through the Hugging Face cache (the scripts above download into folders of your choice). The output folder has:
- `data/chunk-000/file-000.parquet`, one row per frame, with these columns:
  - `observation.state`, `observation.velocity`, `observation.effort`;
  - base position and x-y-z-w quaternion, and the base-frame twist;
  - `observation.imu_acceleration`, `observation.contact`, and so on;
- `meta/info.json`, with named features and provenance;
- `source.json`, a ready data-config source pointing at the folder;
- unless `--no-hf_dataset` is given, `hf/` (the [Hugging Face copy](#downloading-and-converting-datasets)) and
  `sources.json`, whose source reads it.

```bash
uv sync --all-extras
uv run python scripts/convert_dataset.py grandtour  --output data/grandtour --max_missions 2
uv run python scripts/convert_dataset.py mipo       --output data/mipo --pattern "indoor/*" --max_files 20
uv run python scripts/convert_dataset.py neurobem   --output data/neurobem --max_flights 40
uv run python scripts/convert_dataset.py locomujoco --output data/loco_g1 --robot UnitreeG1 --collection Lafan1
uv run python scripts/convert_dataset.py amass_g1   --output data/amass_g1 --pattern "*Walk*" --max_files 200  # 50 Hz, on the ground
uv run python scripts/convert_dataset.py xwbc       --output data/xwbc_g1 --archive data/unitree_g1_29dof.tar.gz  # every joint moves
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
| `amass_g1` | `fleaven/Retargeted_AMASS_for_robotics` at revision `275c4c4a`, `.npy` | `[T, 36]`: root position, x-y-z-w quaternion, 29 joints in URDF order. The rate comes from the file name (59–250 Hz). Each clip is resampled to `--rate` (50 Hz) and put on the ground ([below](#putting-kinematic-motions-on-the-ground)). Velocities are left to finite differencing. Wrist pitch and yaw are 0 in the data. |
| `xwbc` | `TonyZhang1999/xwbc-retarget-data` at revision `b4e169f4`, npz in one tar.gz per robot | Streams the archive (`--archive`, or downloads it) without extracting. Joints are taken by name; body 0 gives the root pose, and the world-frame twist is rotated into the base frame. Motions are put on the ground like `amass_g1`, and the base vertical velocity is corrected by the rate of change of the offset. 30 Hz. The G1 has defaults; other robots need `--urdf`, `--contact_link` and, without collision meshes, `--ground none`. The base twist lets contact detection use foot speeds, at a cost: on 5 clips, preprocessing took 48.5 s with it and 28.0 s without. To skip it, delete `base_linear_velocity_key` and `base_angular_velocity_key` from the source. |
| `lafan1` | `lvhaidong/LAFAN1_Retargeting_Dataset`, CSV | G1, H1, or H1_2 at 30 Hz: root position and x-y-z-w quaternion, then joints in the order listed in the dataset README. Motions CC BY-NC-ND 4.0. |

#### Putting kinematic motions on the ground

The AMASS root height is relative to the first frame, so `amass_g1` sets it from the
geometry (`model/conversion/geometry.py`); `xwbc` does the same, because its soles sit about
2 cm below z = 0. In every frame it computes the lowest point of
the G1's whole collision model: the convex hulls of the 24 collision meshes, the sole
spheres and the shoulder cylinders. `scripts/download_urdfs.py` fetches the meshes; the
heights match MuJoCo's to 0.000 mm. The ground offset is the lower envelope of that point:
a 1 s rolling minimum, then a 1 s moving average. It removes the drift of the retargeted
root height, which moves the per-second lowest point by up to 17 cm within a clip. No
frame can go below the ground, because each frame lies in every window averaged into its
offset. Flight phases shorter than the window, such as jumps, are kept.

On 10 clips (walk, run-and-jump at 250 Hz, crawl, lie-to-crouch, sit, yoga; 59–250 Hz),
with `g1_29dof_rev_1_0.urdf` and the right-wrist sign fix:

| Ground offset | Lowest point | Floating (largest 1 s minimum) | Vertical acceleration added |
|---|---|---|---|
| `--root_height 0.793` (upstream `visualize.py`) | the sitting, crawling and lying clips never come lower than 0.29–0.64 m | | 0 |
| one constant per clip, feet only | lying clip down to −7.1 cm, 38% of its frames below −1 cm | up to 10 cm | 0 |
| one constant per clip, whole body (`--ground_window 0`) | down to −3.7 cm, at most 2.5% of frames below −1 cm | up to 17 cm | 0 |
| 1 s envelope, whole body (default) | 0 | ≤ 4.0 cm (8 of 10 clips ≤ 1.8 cm); median of all 1 s minima ≤ 0.3 cm | ≤ 0.6 m/s², 1.4 m/s² in the jump clip |
| 2 s envelope | 0 | ≤ 7.7 cm | ≤ 0.46 m/s² |

The added acceleration is that of the removed drift, a retargeting artefact. The window
must be longer than the longest flight phase. `--ground feet` uses only the sole spheres
and needs no meshes.

The conventions (quaternion order, frames, units, contact codes) were checked
against downloaded files of each dataset. Kinematic-only datasets (LocoMuJoCo,
AMASS, LAFAN1) have no contacts or torques. They need **no torque estimate**: they
train FK, the action effect (unconditioned), motor feasibility on random proposals,
and the contact-constrained response, whose labels need only the configuration and
which feet are down (detected from kinematics). Retargeted motions can exceed actuator
limits.

## Recommended training datasets

The pretraining tasks need joint-level trajectories of a robot whose URDF or MJCF is
public. Expert motions for the behavior foundation model ([BFM](bfm.md)) need whole-body
motion of a floating-base legged robot with its base pose, like the LAFAN1 motions that
BFM-Zero tracks on G1.

**Legend.**

| Mark | Meaning |
|---|---|
| ✅ | The `huggingface` source reads the dataset directly (LeRobot parquet, or any parquet, CSV, or JSON table with per-frame vector columns) |
| ⚙️ | `scripts/convert_dataset.py` converts it; `scripts/download_datasets.py` downloads and converts those in its [catalog](#downloading-and-converting-datasets) |
| 🔧 | Needs a converter to the [table format](#converting-datasets); not written yet. The format is given so one can be written after download |
| 🔒 | Gated: login, form, or approval |
| ⚠️ | Python pickles (`.pkl`, joblib, `.npy` holding a dict): loading runs code, so open them only in a sandbox |
| P / R | Useful for pretraining / as RL expert motions |

"Torque" says whether joint torques are included and what they are. Measured torques are
used for FD/ID only with `measured_torque_weight > 0` (default 0) and are otherwise useful
to [check torque estimates](dynamics.md#how-accurate-is-it). Unitree's `tauEst` is the
driver's estimate, and a commanded torque is not a measurement. Without torques, nothing is
estimated: the data supervises dynamics through the
[constrained response](dynamics.md#the-contact-constrained-response).

**How this was checked.** The facts come from dataset cards, `meta/info.json`, READMEs,
file listings, loader code and papers, checked on 2026-11-07 (the first rows) and
2026-11-20/21 (the rest). Facts marked "(checked on a file)" were read from a sample file
that was then deleted; everything else was not checked on a file and should be confirmed
after download. "Unverified" means no primary source confirmed it. Check every license
before use: many datasets here are non-commercial, and several cards contradict the
license of the motions they were made from.

### What to download first

`scripts/download_datasets.py` fetches the ⚙️ datasets below and the Unitree G1 sets by
name ([catalog](#downloading-and-converting-datasets)).

**For pretraining.**

| Priority | Dataset | Why |
|---|---|---|
| 1 | [GrandTour](#quadrupeds-real-robots) ⚙️ | Measured joint torques, per-foot contact state, base state (ANYmal D) |
| 2 | [MIPO](#quadrupeds-real-robots) ⚙️ | Unitree torques, foot forces, IMU (Go1) |
| 3 | [Unitree G1 WBT and G1_WBT_SMPL](#humanoids-real-robots) ✅ | Real whole-body G1 with base pose, about 200 h together |
| 4 | [xwbc-retarget-data](#humanoid-motions-retargeted) ⚙️ `xwbc` | AMASS and LAFAN1 on 18 humanoids with base pose and twist: the widest humanoid morphology set (robots other than the G1 need their URDF) |
| 5 | [LocoMuJoCo](#humanoid-motions-retargeted) ⚙️ | 9 robots with named joints |
| 6 | [ergoCub logs](#humanoids-real-robots) 🔧 | Real humanoid walking with joint torques and 8 force-torque sensors |
| 7 | [Raw DROID](#arms-and-hands-fixed-base) 🔧 | Franka with measured motor torques; only the small `trajectory.h5` files are needed |
| 8 | [LeRobot arm sets with joint angles](#arms-and-hands-fixed-base) ✅ | Panda, xArm7, UR5, FANUC, Cobotta; UCSD Kitchen has joint torques |
| 9 | [MuJoCo ground and synthetic sources](#legged-robots-on-the-ground-simulated) | Built in: exact torques and contacts for every robot you have a URDF of |

**For RL expert motions.**

| Priority | Dataset | Why |
|---|---|---|
| 1 | [LAFAN1 on G1](#humanoid-motions-retargeted) ⚙️, or BFM-Zero's own LAFAN1 pickles ⚠️ | The motions BFM-Zero is trained on |
| 2 | [PHUMA](#humanoid-motions-retargeted) ⚠️ | G1 and H1-2, filtered for physical plausibility, about 70 h per robot |
| 3 | [TWIST2](#humanoid-motions-retargeted) ⚠️ | 32k G1 clips (AMASS, OMOMO, TWIST) with the 29-DoF model |
| 4 | [AMASS on G1 from xwbc](#humanoid-motions-retargeted) ⚙️ `xwbc` (`amass/unitree_g1_29dof.tar.gz`, 8.1 GB) | About 13k clips with every joint moving, wrists included; `amass_g1` (`fleaven`, file by file) has wrist pitch and yaw at 0 |
| 5 | [G1_WBT_SMPL](#humanoids-real-robots) ✅ | Real G1 whole-body motion that walks, with paired SMPL references |
| 6 | [xwbc-retarget-data](#humanoid-motions-retargeted) ⚙️ `xwbc` | The same motions on H1, H1_2, Talos, Booster T1 and others, for the other humanoids |
| 7 | [GenLoco](#quadruped-motions-kinematic), [AMP_for_hardware](#quadruped-motions-kinematic), [Tencent MAX](#quadruped-motions-kinematic) 🔧 | Dog motions on A1, ANYmal B/C, Laikago, Mini Cheetah and MAX |

A robot used for RL needs its URDF in the data config (`scripts/download_urdfs.py` fetches
G1, H1, Go1, ANYmal D and Talos). Retargeted motions can exceed joint and torque limits.

### Quadrupeds: real robots

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Use, notes |
|---|---|---|---|---|---|---|---|
| [GrandTour](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) (`leggedrobotics/grand_tour_dataset`) | ANYmal D, 12 | 49 missions in the paper; 48 Hub mission folders have the state-estimator archive | **yes** (joint torque, current, commands, PD gains) | estimator pose and twist, IMUs, GNSS, per-foot contact state and wrench | zarr in tar ⚙️ `grandtour` | MIT on the card; on the website data CC BY-SA 4.0, software MIT | P. The best fit: measured torques and contacts. Joint names match [anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description) (`download_urdfs.py` fetches it). |
| [MIPO-Go1](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry) (`Open-IO/mipo-go1-odometry`) | Unitree Go1, 12 | 84 files (79 indoor, 5 outdoor), about 1.6 h summed from the card, about 400 Hz | **yes** (from the Unitree SDK) | body IMU with orientation, OptiTrack indoors, 4 foot-force sensors | HDF5 ⚙️ `mipo` | AGPL-3.0 | P. Recordings of Cerberus 2.0 and the MIPO paper (Yang et al., IROS 2023). Used for the [torque-estimation check](dynamics.md#how-accurate-is-it). Streams are unsynchronized; joint names `FL0`…`RR2` map to `FL_hip_joint`… |
| [DFKI Go2 state estimation](https://zenodo.org/records/19336009) | Unitree Go2, 12 | 27 runs, 1 kHz, 5.4 GB | **yes** (`JointState.effort`) | Vicon / RTK GNSS, IMU, foot contact flags and forces | ROS 2 bags 🔧 | CC BY 4.0 | P. Needs the dfki-quad message definitions. Popescu et al., Zenodo 2026. |
| [ForEnt](https://zenodo.org/records/23802853) | Unitree Go2, 12 | 16 walks, about 2.5 km, over 140 annotated entrapments; 93.4 GB in 9 zips | `tau_est` (estimate) and `/lowcmd` (targets, gains, feed-forward torque) | 10 Hz pose and body velocity in `trajectory.csv` (Leg-KILO odometry and a GTSAM pose graph), IMU, foot force (raw units) | ROS 2 MCAP 🔧 (message definitions embedded and in `common.zip`) | CC BY 4.0 | P: off-nominal contact in forest. Rebuild `/lowstate` time from `tick` (bag stamps lag up to 10 s); accelerometers saturate near 4 g; use the `fall` flags. Kirdwichai et al., 2026. |
| [Go2 low state with RTK](https://zenodo.org/records/23287210) | Unitree Go2, 12 | 10 sequences, about 25 min, 1.1 km; 217 MB | `tau_est` | 5 Hz RTK position only, IMU, 4 foot forces | CSV 🔧 | CC BY 4.0 | P. Joint order unverified (SDK order FR, FL, RR, RL; the companion code assumes FL, FR, RL, RR). Girgin and Kilic, CoRL 2026. |
| [FusionPortableV2](https://fusionportable.github.io/dataset/fusionportable_v2), legged sequences | Unitree A1, 12 | 5 sequences, 1,336 s; 37.1 GB of 7z | `tauEst` in `/unitree/joint_state` effort (checked on a file) | IMU, Unitree body odometry, total-station position (on a prism, needs the extrinsic), raw foot-force counts | ROS 1 bags 🔧 | none stated for the data | P. 50 Hz joint states. Names `FR0…RL2` map to `FR_hip_joint`… Wei et al., IJRR 2025. |
| [Cerberus demo bags](https://github.com/ShuoYangRobotics/Cerberus) | Unitree A1 and Go1, 12 | 6 bags, 36.3 min, 16.3 GB | no (effort is zero; checked on a file) | Unitree IMU, raw foot counts; mocap only in a 30 s indoor bag | ROS 1 bags 🔧 | none stated for the data (code GPL-3.0) | P: kinematics and IMU only. Not the MIPO recordings (those are Cerberus 2.0). Yang et al., ICRA 2023. |
| [TAIL / TAIL-Plus](https://tailrobot.github.io/download/) | Unitree Go1, 12 | TAIL sequences 9–14 (888 s, 96.7 GB) and TAIL-Plus 7–10 | `tauEst` in `/high_state` (HighState, 500 Hz) | RTK-GPS with IMU (6-DoF), Unitree IMU (w, x, y, z), foot force | ROS 1 bags 🔧 | GPL-3.0 ("for commercial use, please contact Chen Yao") | P: walking on sand. Yao et al., RA-L 2024; Wang et al., ICRA 2024 workshop. |
| [QuadSLAM](https://github.com/EN3D-Lab/Quadruped-SLAM-dataset) | Unitree Go2, 12 | 22 bags (e.g. 1,240 s in 12.1 GB) | `/low_state` motor states at about 500 Hz | Leica TS60 total station, IMU, foot force | ROS 1 bags 🔧 | **none stated** | P, R: stairs and slips. Without a license it cannot be redistributed. Zhao et al., ISPRS J. 2026. |
| [DiTer++](https://github.com/sparolab/DiTer-plusplus) | Go1 and Go2, 12 | 9 bags, about 771 GB | HighState-style joints | 6-DoF ground truth from scan-to-map matching, IMU, foot contact | ROS 1 bags 🔧 | CC BY-NC-SA 4.0 | P, but the bags are mostly camera and lidar data. Kim et al., ICRA 2025. |
| [DLS HyQ proprioceptive dataset](https://ieee-dataport.org/open-access/proprioceptive-sensor-dataset-quadruped-robots) 🔒 (free IEEE login) | HyQ (hydraulic), 12 | 6 trotting sequences, about 59 min; 747 MB (plus a 1.2 GB endurance run) | **sensors**: torque on HAA; load cells (actuator force in N) on HFE and KFE, which need the linkage geometry | Vicon at 250 Hz on its own clock, two IMUs; no foot sensors | CSV and MAT 🔧 | CC BY 4.0 | P. 1 kHz. Model: `hyq` in [example-robot-data](https://github.com/Gepetto/example-robot-data) (BSD-3; same version unverified). Fink and Semini, CLAWAR 2020. |
| [Solo-12 recordings](https://github.com/paLeziart/solo12-recordings) | Solo-12, 12 | 2 trotting runs at 500 Hz, 9.8 MB | from motor current (`torquesFromCurrentMeasurment`) | mocap pose and velocity, IMU; `contactStatus` is the *planned* contact | npz 🔧 (`allow_pickle=False`) | BSD-2-Clause | P, small. Model: `solo12` in example-robot-data (BSD-3); signs and offsets unverified. |
| [UMich Mini Cheetah contact data](https://github.com/UMich-CURLY/deep-contact-estimator) | MIT Mini Cheetah, 12 | 15 files, 184 MB, about 16.9 min at 1 kHz (joints at 500 Hz, upsampled) | `tau_est` is the **commanded** torque (feed-forward plus PD) | IMU (quaternion w, x, y, z), contact labels, estimated foot forces; no base pose | MAT 🔧 (checked on a file) | **none stated** | P: many terrains, including walking in the air. Lin et al., CoRL 2021. |
| [Pronto ANYmal B sample](https://github.com/ori-drs/pronto_anymal_example) | ANYmal B, 12 | one bag, 7.41 GB | `/joint_states` effort at 400 Hz (measured unverified) | state-estimator pose at 400 Hz, IMU (z axis down) | ROS 1 bag 🔧 | **none for the data** (code LGPL-2.1) | P. Joint names match [anymal_b_simple_description](https://github.com/ANYbotics/anymal_b_simple_description). Camurri et al., Frontiers 2020. |
| [Leg-KILO](https://github.com/ouguangjun/legkilo-dataset) | Unitree Go1, 12 | 7 sequences, 50 Hz unique | unverified | IMU, onboard odometry, foot forces (and lidar) | ROS 1 bags 🔧 | CC BY 4.0 (recordings and ground truth) | P. Ground truth from offline map optimization. Its README gives the motor order "FR {0,1,2}, FL {3,4,5}, RR {9,10,11}, RL {6,7,8}", which differs from Unitree's usual order: check a file. Ou et al., RA-L 2024. |
| [Go1 on mud](https://zenodo.org/records/12673097) | Unitree Go1, 12 | 15 runs, 59 MB | motor position, velocity and torque only | none | netCDF 🔧 | CC BY 4.0 | P, small; the custom feet change the model. Godon et al., Bioinspir. Biomim. 2024. |
| [CEAR](https://daroslab.github.io/cear/) | MIT Mini Cheetah, 12 | over 100 sequences, 16 backflips | no (joint positions at about 100 Hz) | IMU at 400 Hz (w first, z down), mocap on some sequences | text files in zips (1.9 TB in all, mostly event data) 🔧 | **"All Rights Reserved"** (DARoS Lab): permission needed | The only agile real quadruped motions found. Zhu, Xiong and Kim, RA-L 2024. |

### Quadruped motions (kinematic)

None of these has torques, contact forces, or IMU. All are dog motion capture or derived
from it, so most inherit a non-commercial license.

| Dataset | Robot | Size | Content | Format | License | Use, notes |
|---|---|---|---|---|---|---|
| [Tencent lifelike agility and play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play) | Tencent MAX, 12 (`max.urdf` in the repo) | 33 clips plus 33 mirrored, 114,788 frames, 957 s at 120 Hz; raw BVH (Labrador) included | root position, root quaternion (x, y, z, w), joints in order FR, FL, HR, HL (checked on a file) | JSON 🔧 | LICENSE file MIT; the README says "for research purpose only" | R: walk, run, jump, play, sit. Frames 709–728 of `dog_quad_walkrun_007_ret.txt` have glitches (joint speeds up to 472 rad/s): filter. Han et al., Nature Machine Intelligence 2024. |
| [AMP_for_hardware](https://github.com/escontra/AMP_for_hardware) | Unitree A1, 12 | 6 clips (pace, trot, turns), 15.9 s, plus hopturn | root pose (x, y, z, w), joints (FR, FL, RR, RL), toe positions, base linear velocity (base frame) and angular velocity, joint velocities (checked on a file) | JSON 🔧 | repo BSD-3; motions apparently derived from MANN (inferred), so CC BY-NC 4.0 | R. The README's "4.5 seconds" does not match the files. Escontrela et al., arXiv:2203.15103. |
| [GenLoco motions](https://github.com/HybridRobotics/GenLoco) | A1, ANYmal B, ANYmal C, D'Kitty, Laikago, Mini Cheetah, Sirius | 42 clips (7 robots × 6 gaits), 25.6 s at 60 Hz | root position, quaternion (x, y, z, w), 12 joints | text 🔧 | Apache-2.0; motions CC BY-NC 4.0 via MANN | R, and cross-embodiment P. `a1_pace` is planar and the trot clips keep a constant base orientation. D'Kitty, Sirius and Mini Cheetah URDFs are not in that repo. Feng et al., arXiv:2209.05309. |
| [motion_imitation](https://github.com/erwincoumans/motion_imitation) | Laikago, 12 | 10 clips, 30.6 s (60, 24 or 33 Hz) | root position, quaternion (x, y, z, w) including Laikago's `INIT_ROT` (about (0.5, 0.5, 0.5, 0.5) when level), joints FR, FL, RR, RL | text 🔧 | code Apache-2.0; motion data CC BY-NC 4.0 | R. Its `retarget_motion.py` retargets the MANN clips to other URDFs. Peng et al., RSS 2020. |
| [AI4Animation MANN dog mocap](https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip) | dog skeleton (21 joints), not a robot | 51 BVH, 147,541 frames, about 41 min at 60 Hz; 33 MB | marker-based skeleton motion | BVH (needs retargeting) | CC BY-NC 4.0, research or education only | The source of the sets above; retarget to a URDF first. Zhang et al., ACM TOG 2018. |
| [LocoMuJoCo v0.4.1 quadruped](https://zenodo.org/records/13737593) | Unitree A1 (simulated, MPC) | 345 MB, plus 147 MB with actions | hinge root joints (`trunk_tz`, `trunk_list`, …), then `FR_hip_joint`… | npz 🔧 | CC BY-ND 4.0 (converted copies may not be redistributed); code MIT | R. LocoMuJoCo 1.x and its Hub repo have no quadrupeds. |

### Quadrupeds: simulated

| Dataset | Robots | Size | Signals | Format | License | Notes |
|---|---|---|---|---|---|---|
| [`DLS-IIT/quadruped_locomotion`](https://huggingface.co/datasets/DLS-IIT/quadruped_locomotion) | Aliengo, Mini Cheetah (gym-quadruped MJCF) | e.g. 300 episodes × 1,249 steps in 758 MB | `qpos` (quaternion w, x, y, z), `qvel`, base velocities and acceleration, projected gravity, `tau_ctrl_setpoint` (exact in a torque-controlled simulator), `contact_state`, 3-D `contact_forces`, feet positions and velocities, at 500 Hz | HDF5 🔧 | MIT on the card; gym-quadruped's own license unclear | P: every converter column can be filled. Mini Cheetah zero positions follow that MJCF (`qpos0_js`). |
| [`cagataydev/strands-isaaclab-{go2,anymald,cassie}-rough`](https://huggingface.co/datasets/cagataydev/strands-isaaclab-go2-rough) | Go2, ANYmal D, Cassie | 12 episodes × 500 frames each, 50 Hz | named joints, root position and quaternion (x, y, z, w) in `observation.state` [254] | LeRobot v3 ✅ | generated content CC BY 4.0 | Tiny; Isaac Lab PPO rollouts. |

### Legged robots on the ground (simulated)

| Source | Robots | Labels | Notes |
|---|---|---|---|
| `"type": "mujoco"` (`configs/data/ground_contact.json`) | any legged URDF; poses provided for Go1, ANYmal D, G1, H1 | exact torques, accelerations, contacts, pushes | Standing, crouching, weight shifting, and push recovery with real foot contact; see [Ground contact](#ground-contact-in-synthetic-data-mujoco). |
| `"type": "procedural"` and [GenBot-1K](https://github.com/BoAi01/embodiment-scaling-laws) URDFs | generated quadrupeds and humanoids; GenBot-1K has 1,012 URDFs (348 humanoids, 332 quadrupeds, 332 hexapods), BSD-3-Clause | the same | GenBot-1K releases robots, not trajectories (Ai et al., CoRL 2025): simulate them with the `mujoco` or `synthetic` source. |

### Humanoids: real robots

| Dataset | Robot | Size | Torque | Base and contacts | Format | License | Use, notes |
|---|---|---|---|---|---|---|---|
| [Unitree G1 whole-body teleop](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit) (`unitreerobotics/G1_WBT_*`, 55 repos) | Unitree G1, 29 + hands | 12,641 episodes / 14.3M frames (about 132 h), 30 Hz | no | base pose (xyz + quaternion xyzw), projected gravity and angular velocity | LeRobot parquet ✅ | Apache-2.0 | P. `configs/data/g1_wbt_floating.json`. The state is split into `lower_body`, `left_arm`, `right_arm`. The base position may be estimated. **The wrist names are wrong**: the arm vectors are stored as `…elbow, wrist_roll, wrist_pitch, wrist_yaw` (the URDF and motor order), while `meta/info.json` names them `wrist_yaw, wrist_roll, wrist_pitch` ([checked](#g1-joint-order-check)). The config and `make_data_config.py` use the stored order. By Unitree Robotics. |
| [`unitreerobotics/G1_WBT_SMPL_*`](https://huggingface.co/datasets/unitreerobotics/G1_WBT_SMPL_Brainco_Pick_Vegetables) (18 repos) | Unitree G1, 29 + hands | 5,337 episodes / 7.75M frames (about 72 h), 30 Hz | no | `observation.state.robot_q_current` [36]: root xyz, root quaternion (w, x, y, z), 29 joints in URDF order (checked on a file); the root moves metres | LeRobot v3 ✅ | Apache-2.0 | **R** (real whole-body motion that walks, with the tracked SMPL motion in `action.frame_axis`) and P. Use `state_indices` 7–35, base position 0–2 and orientation 3–6 with `base_orientation_format: "wxyz"`. Whether the base pose comes from an estimator or mocap is not documented. **No actions**: there are no joint commands (the action columns are the SMPL reference, the hands and the wrist reference), so the `g1_wbt_smpl` sources of `scripts/download_datasets.py` read only `robot_q_current`. |
| `unitreerobotics/G1_WBT_Inspire_*` with the q36 schema (4 repos) | Unitree G1 | 1,702 episodes / 0.75M frames | no | the same 36-vector, plus `robot_q_desired` | LeRobot ✅ | Apache-2.0 | P, R. The three `*_MainCamOnly` repos are nested one folder down: set `data_files` to `G1_WB_Dex5_<task>/data/*/*.parquet`. **No actions**: `robot_q_desired` follows `robot_q_current` (median gap 0.04 rad, median correlation 0.94 per joint, in one file of `Take_Plates_Into_Dishwasher`), but whether it is the motor command or a planner target is not documented, so the `g1_wbt_q36` sources of `scripts/download_datasets.py` read only `robot_q_current` and have no actions. |
| `unitreerobotics/G1_Dex1_*` with lower-body state (129 repos) | Unitree G1, 29 | 66,941 episodes / 50.2M frames (about 465 h), 30 Hz | no | none (no base pose, IMU or gravity); `base_command` in the actions | LeRobot v3 ✅ | Apache-2.0 | P (whole-body joints while balancing). The robot stands under its locomotion controller. The arm vectors have the same wrong wrist names as `G1_WBT_*` ([checked](#g1-joint-order-check)). |
| `unitreerobotics/G1_Dex1_*` with the `G1_ALL` schema (9 repos) | Unitree G1, 29 | 1,800 episodes / 1.53M frames | no | none | LeRobot v2.1 ✅ (`state_key: "observation.body"`, URDF order) | Apache-2.0 | P; standing. |
| [Humanoid Everyday](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday) (`USC-PSI-Lab/humanoid-everyday`) | Unitree G1 and H1 (`H1_2` in the project README) + hands | 8,949 episodes / 3.4M frames, 30 Hz | no (`tau_ff` is a command) | IMU and odometry in the combined repo | LeRobot v2.1 parquet ✅ | Apache-2.0 on the card, MIT in the project README | P. Joint vectors are unnamed; map them by the order in the project README. Zhao et al., arXiv:2510.08807. |
| [`mncai/G1_Dex3_Trash_LocoManipulation`](https://huggingface.co/datasets/mncai/G1_Dex3_Trash_LocoManipulation) | Unitree G1 + Dex3, 43 | 118 episodes / 183k frames, 50 Hz | no | root orientation (w, x, y, z), projected gravity; no base position | LeRobot v2.1 ✅ | Apache-2.0 (motion tokens under the NVIDIA Open Model License) | P, R: walk, kneel, carry, through the SONIC whole-body controller. GenON Physical AI Lab. |
| [`SII-ZhangYiFei/G1-WholeBody-PickPlaceToy`](https://huggingface.co/datasets/SII-ZhangYiFei/G1-WholeBody-PickPlaceToy) | Unitree G1, 43 | 128 episodes / 107k frames, 50 Hz | no | the same schema | LeRobot ✅ | Apache-2.0 | P. Real or simulated is not stated. |
| [`snupilab/humanoidtoolbench-teleop`](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop) | Unitree G1, 43 | `raw/`: 4,362 episode files at 50 Hz (MuJoCo); `hardware/`: 91 episodes at 20 Hz | `raw/`: simulator torque in `observation.effort` | `raw/`: base pose (w, x, y, z) and twist; `hardware/`: torso orientation and angular velocity | LeRobot v2.1 ✅ (one source per folder with `data_files`, episode indices restart) | CC BY-NC 4.0 | P (the only G1 set with effort at scale; `raw/` is simulated). Whether `hardware/` is a real robot is unverified. Jang et al., arXiv:2610.02089. |
| [`nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1) | Unitree G1, 43 | 1,095 episodes / 124k frames, 20 Hz | no | none | LeRobot ✅ (`data_files` per task folder) | CC BY 4.0 | P (upper-body teleop). |
| [ergoCub logs](https://huggingface.co/datasets/ami-iit/paper_romualdi_viceconte_2024_humanoids_dnn-mpc-walking_dataset) (`ami-iit/*`) | ergoCub, 33 | walking 4.0 GB; push recovery 78 MB; larger sets of 22–235 GB (`ami-iit/actuator_net`, `sensorless-torque-control`, `paper_Sorrentino_RAL_BalancingTorqueControl`) | **joint torques** in `joints_state/torques` (measured or estimated from the force-torque sensors: unverified), motor currents and PWM | waist IMU and 6 leg and foot IMUs (roll-pitch-yaw, gyro, accelerometer), 8 force-torque sensors; no base position | MATLAB 7.3 `.mat` (HDF5), separate timestamps per signal, about 99 Hz 🔧 | BSD-3-Clause | **P**: the best real humanoid set with torque found. Joint names match ergoCubSN001 `model.urdf` in [ergocub-software](https://github.com/icub-tech-iit/ergocub-software) (BSD-3-Clause). Units read "n.d.". Romualdi et al., Humanoids 2024. |
| [Digit fall prediction](https://huggingface.co/datasets/Gokul97/digit_fall_prediction_dataset) | Agility Digit, 20 motors + 10 passive joints | 81 hardware trajectories (standing on a gantry, pushed, some falling), plus 900 simulated per push type | **measured motor torque** at about 1 kHz | base pose, velocities and accelerations, IMU | ROS 1 bags with a custom message (`digit_msgs/Digit_Log`) 🔧; the processed `.npy` files are pickles ⚠️ | no license tag on the Hub; the GitHub repo is AGPL-3.0; the data's own license is unverified | P (excellent torque data); R poor. A public Digit model and the motor order are unverified. Mungai, Prabhakaran and Grizzle, arXiv:2309.14546. |
| [Cassie impact dataset](https://github.com/DAIRLab/cassie_impact_dataset) | Agility Cassie, 16 | 29 jump landings, about 50 ms each at 2 kHz | **measured motor torques** (10) | quaternion (w, x, y, z), base position and twist | arrays per landing 🔧 (legend files are pickles ⚠️; use the README tables) | BSD-3-Clause | P, tiny: clean impacts. Model from dairlib `cassie_v2.urdf` (path and license unverified). By DAIR Lab; related paper: Acosta, Yang and Posa, RA-L 2022 (link unverified). |
| [RoboCOIN Leju Kuavo 4](https://huggingface.co/datasets/RoboCOIN/Leju_Kuavo_4_sensor_card_swiping) 🔒 (some repos auto-gated) | Leju Kuavo 4 | e.g. 598 episodes / 425k frames, 30 Hz; about 86 Kuavo repos | no | none | LeRobot ✅ | Apache-2.0 | P. `observation.state` [52] mixes joints and end-effector poses with generic names: map them by hand. Wu et al., arXiv:2511.17441. |

#### G1 model

The repository uses Unitree's `g1_29dof_rev_1_0.urdf` (from `unitree_ros`, fetched by
`scripts/download_urdfs.py`) and not `g1_29dof.urdf`, because the G1 data matches it.
Checked on 2026-11-23 with forward kinematics:

| Data | `g1_29dof_rev_1_0.urdf` | `g1_29dof.urdf` |
|---|---|---|
| `state_torso` of four `G1_WBT_*` repos (Inspire, Brainco, two Dex1; 600 frames each, [below](#g1-joint-order-check)) | 0.000°, 0.00 mm | 0.000°, a constant offset of −10 mm along z |
| The 32 body positions of 107 xwbc G1 clips | at most 0.02 mm | up to 9.0 mm (median 7.1 mm) |

fleaven's AMASS README names the same model (`g1_29dof_rev_1_0.xml`). The two files have the
same joints, joint axes, and leg and arm links. They differ in:

- **Waist.** `waist_roll_joint` sits 9 mm higher (0.044 m instead of 0.035 m) and
  `waist_pitch_joint` at the waist roll origin instead of 19 mm above it, so the torso frame
  is 10 mm lower. The shoulder, head and IMU frames are 10 mm higher in the torso frame, so
  they stay at the same height above the pelvis.
- **Masses.** The torso link weighs 6.78 kg instead of 8.56 kg; the waist yaw and roll links
  0.214 and 0.086 kg instead of 0.244 and 0.047 kg; `waist_support_link` is gone. The robot
  weighs 33.3 kg instead of 35.1 kg.
- **Knees.** Limits of 139 N·m and 20 rad/s instead of 88 N·m and 32 rad/s.
- **Meshes.** `torso_link_rev_1_0.STL` for the torso collision (24 collision meshes in all).

Anything trained with `g1_29dof.urdf` used the older masses and knee limits. Its cache
entries are rebuilt automatically, because the URDF is part of the cache key.

#### G1 joint order check

On 2026-11-23 the joint and pose columns of five Unitree repos were read: one parquet file
each, through HTTP range requests. In the four repos that record end-effector poses, every
order of the three wrist entries was tried with forward kinematics of
`assets/urdf/g1_29dof_rev_1_0.urdf` in MuJoCo (the table is the same with `g1_29dof.urdf`). For the right order, the rotation from the URDF
`wrist_yaw_link` to the recorded `*_ee_pose_gripper_torso` frame is constant (here the
identity), and so is the position offset in the wrist frame (the gripper point, 0.11 to
0.16 m along x).

Each cell gives the median rotation error over both arms, then the position RMS after
fitting the constant offsets:

| Repo (revision), file | Frames used | `roll, pitch, yaw` | `yaw, roll, pitch` (the `names`) | the other four orders |
|---|---|---|---|---|
| `G1_WBT_Inspire_Pick_Up_Fruit` (7ca17dd2), `file-001` | 600 of 63,029, from 79 episodes | 0.000°, 0.9 mm | 36–40°, 51–70 mm | 19–44°, 36–71 mm |
| `G1_WBT_Brainco_Make_The_Bed` (9c905660), `file-001` | 600 of 25,425, from 40 episodes | 0.000°, 0.4–0.5 mm | 19–39°, 22–26 mm | 8–41°, 14–28 mm |
| `G1_WBT_Dex1_Put_Clothes_into_Washing_Machine` (9f160f07), `file-000` | 600 of 124,184, from 155 episodes | 0.000°, 0.6 mm | 17–35°, 26–43 mm | 13–33°, 23–37 mm |
| `G1_Dex1_Conveyor_Color_Classification` (996b4387), `file-000` | 600 of 38,690, from 261 episodes | 0.000°, 0.3–0.4 mm | 14–18°, 12–25 mm | 8–21°, 9–22 mm |

Two more checks:

- **Joint limits.** In the Brainco repo, arm entry 4 reaches −1.98 and +1.94 rad. Only wrist
  roll allows that: its limit is ±1.97 rad, while pitch and yaw are limited to ±1.61 rad.
- **Unitree's SDK.** `G1JointIndex` in `unitree_sdk2_python`
  (`example/g1/low_level/g1_low_level_example.py`, last changed 2024-12-31) numbers the
  motors `LeftWristRoll = 19, LeftWristPitch = 20, LeftWristYaw = 21`.

The other joints match their names:

- **Waist.** `lower_body[12:15]` equals `waist_state_joint`. In Brainco and both Dex1 repos,
  `state_torso` is the torso in the pelvis frame. There the order `yaw, roll, pitch` gives
  0.000° and a position error of 0.00 mm. The other orders give a 1.7–38° median rotation
  error (1.7–17° in Dex1 Conveyor, where the waist barely moves). In the Inspire repo, `state_torso` is the torso in the odometry frame of
  `state_base_pose`. Composing that pose with the waist gives 0.000° and 0.00 mm (also the
  maximum over 600 frames) for `yaw, roll, pitch`, and a 4–41° median error for the
  others. With `g1_29dof.urdf` the rotations are the same, but the position is off by a
  constant −10 mm along z in all four repos ([G1 model](#g1-model)).
- **Legs.** With the labelled order, in that odometry frame, the lower foot is flat (3.6°
  median tilt) at a steady height (IQR 8.6 mm). Swapping ankle pitch and roll gives 17°,
  and swapping hip pitch and roll gives 46°. Swapping knee and ankle pitch, which turn
  about the same axis, is ruled out by the limits. Entries 3 and 9 reach 1.36 rad
  (ankle pitch stops at 0.52 rad), and entry 10 reaches −0.60 rad (the knee stops at
  −0.09 rad). The arm actions use the same order as the states.
- **`G1_WBT_SMPL_Inspire_Pick_Vegetables`** (e681e780). `robot_q_current` is root xyz,
  quaternion w, x, y, z, then the 29 joints in URDF order. With that reading the lower foot
  is at 3.45 cm with a 3° tilt; reading the quaternion as x, y, z, w turns the robot upside
  down. Joint 19 reaches −1.78 rad, so it is wrist roll.
- **`G1_WBT_Inspire_Take_Plates_Into_Dishwasher`** (q36 schema; `file-000`, 181,793 frames).
  `robot_q_current` has the same layout. Read as w, x, y, z the robot stands upright (the
  x and y components have medians 0.003 and 0.009); read as x, y, z, w it is upside down.
  With the URDF leg order the lower foot tilts 4.8° (median; 95th percentile 7.1°, against
  12.6° with ankle pitch and roll swapped). Joint 19 reaches −1.63 rad, beyond the ±1.61 rad
  of wrist pitch and yaw, so it is wrist roll.

The scripts are not part of the repository; the numbers above are their output.
`model.conversion.mapping.KNOWN_NAME_FIXES` applies the correction when `robot_type`
starts with `unitree_g1`. The wrong names appear in 184 of the 262 `unitreerobotics` repos
(every `G1_WBT_*` and most `G1_Dex1_*`). Anything trained on
`configs/data/g1_wbt_floating.json` or `configs/rl/g1_go1_real_and_synthetic.json` before
this fix had three wrist joints per arm mislabelled. Its cache entry is rebuilt
automatically, because the joint list is part of the cache key.

### Humanoid motions (retargeted)

Kinematic motion capture or video motions mapped to robot joints. None has torques; few
have velocities or contacts. Unless a row says otherwise, velocities come from finite
differences and contacts from kinematics. Most come from AMASS (non-commercial research
terms) or LAFAN1 (CC BY-NC-ND 4.0), whatever their cards say.

| Dataset | Robots | Size | Content | Format | License | Use, notes |
|---|---|---|---|---|---|---|
| [LocoMuJoCo datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) (`robfiras/loco-mujoco-datasets`) | 12 models: 9 robots (Unitree G1, H1, H1 v2, Talos, Apollo, Atlas, Booster T1, Fourier GR1T2, ToddlerBot) and 3 human models | 19 default + 40 LAFAN1 motions per robot, 40 Hz | floating-base `qpos`/`qvel` with joint names | npz ⚙️ `locomujoco` | CC BY-NC-ND 4.0 (code MIT) | P, R. Many morphologies. Motions are kinematic, so some exceed actuator limits. Al-Hafez et al., arXiv:2311.02496. |
| [LAFAN1 retargeted to Unitree](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (mirror `lvhaidong/LAFAN1_Retargeting_Dataset`) | G1, H1, H1_2 | 40 clips per robot, 30 Hz | root position and quaternion (x, y, z, w), joints in the README order | CSV ⚙️ `lafan1` | motions CC BY-NC-ND 4.0 (LAFAN1, Ubisoft La Forge); LICENSE file BSD-3-Clause (Unitree; the card text says MIT for code) | R (BFM-Zero's motions), P. The original `unitreerobotics` repo is no longer publicly accessible. Same motions with named joints: `ember-lab-berkeley/LAFAN-G1`, `wty-yy/LAFAN1_g1`. LAFAN1: Harvey et al., ACM TOG 2020. |
| [Contact LAFAN1](https://huggingface.co/datasets/seu-zuxing/Contact_LAFAN1_Retargeting_Dataset) | G1, 29 | the 40 LAFAN1 clips at 50 Hz | `joint_pos`, `joint_vel` (Isaac Lab order; names not stored), body states (quaternion w, x, y, z), **foot `contact` (T, 2)** (checked on a file) | npz 🔧 | card Apache-2.0, which conflicts with LAFAN1's CC BY-NC-ND 4.0 | P: the only LAFAN1-G1 copy with foot contacts and velocities. In one clip the knees go 6 cm below the floor. |
| [BFM-Zero LAFAN1 motions](https://huggingface.co/LeCAR-Lab/BFM-Zero) (`data/` of the model repo; also in the GitHub repo) | G1, 29 | LAFAN1 in 10 s pieces, about 0.2 GB per file | joblib `{root_trans_offset, root_rot (x, y, z, w), dof, pose_aa, fps}` | joblib ⚠️ 🔧 | CC BY-NC 4.0; LAFAN1 CC BY-NC-ND 4.0 | R: exactly BFM-Zero's expert motions, to reproduce it. Li et al., ICLR 2026. |
| [Retargeted AMASS for robotics](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics) (`fleaven/Retargeted_AMASS_for_robotics`) | **G1 only**, 29 | 17,717 clips, about 59 h, native AMASS rates (59–250 Hz, from the file name); 6.95 GB | (T, 36) float64: position, quaternion (x, y, z, w), 29 joints in the README order, which is the URDF order; root position relative to the first frame (`g1/visualize.py` adds 0.793 m to z); wrist pitch and yaw are 0 in every clip (all checked on 10 clips) | `.npy` arrays ⚙️ `amass_g1` (no pickle) | card CC BY 4.0, README AMASS terms (non-commercial research) | P, R; for wrist motion use xwbc (next row). Uploaded again on 2025-04-16 with hip yaw, ankle roll and wrist roll retargeted; the converter pins that revision (`275c4c4a`). **The right wrist roll has the wrong sign**, and the converter negates it. On two shared clips it correlates −0.34 to −0.97 with xwbc, TWIST2 and any4hdmi, which agree with each other (+0.66 to +1.00); its left wrist roll, shoulders and elbows correlate +0.44 to +0.99. In walking (B3 walk1) its mean is −0.27, the same sign as the left (−0.33), while the others give +0.17 to +0.30. The Unitree models it names have the URDF's wrist axes. It resamples each clip to 50 Hz and puts the whole body on the ground ([how](#putting-kinematic-motions-on-the-ground)), because 0.793 m leaves clips that start lying (0.72 m), crawling (0.60 m) or sitting (0.29 m) in the air. By Kun Zhao; AMASS: Mahmood et al., ICCV 2019. |
| [AMASS retargeted to G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1) (`ember-lab-berkeley/AMASS_Retargeted_for_G1`), older copy | Unitree G1, 29 | 17,714 clips, 30 Hz | pelvis pose, joint positions and velocities, named | npz (no longer read) | CC BY 4.0 on the card; the AMASS terms apply | Superseded by the row above. It is the upstream's earlier upload (`822d5da9`, 2025-02-21) at every 4th frame. Hip yaw, ankle roll and all six wrist joints are 0, and the stored joint velocities do not match the positions. On the same clip (`ACCAD/Female1Walking_c3d/B3 - walk1`), its root and arm columns equal the new upload's; only the legs differ. |
| [AMASS for BXI Elf2](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_bxi_elf2) and [for Fourier N1](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_FourierN1) | BXI Elf2, 25; Fourier N1, 23 | the same 17,717 clips; 6.18 and 5.79 GB | (T, 32) and (T, 30) float64: position, quaternion (x, y, z, w), joints (checked on a file); root starts at the origin | `.npy` arrays 🔧 | card CC BY 4.0, README AMASS terms | P: two new morphologies with the same motions as G1. R only after clamping: joint limits are not enforced. Models: `elf2_dof25.xml` in [bxirobotics/robot_descriptions](https://github.com/bxirobotics/robot_descriptions) (Apache-2.0); `N1/urdf/N1_rotor.urdf` on branch `FourierN1` of [FFTAI/Wiki-GRx-Models](https://github.com/FFTAI/Wiki-GRx-Models) (Apache-2.0 on that branch). |
| [xwbc-retarget-data](https://huggingface.co/datasets/TonyZhang1999/xwbc-retarget-data) (revision `b4e169f4`) | 19 robot archives: Unitree G1 (29 and 23), H1, H1_2, H2, R1; Booster K1, T1, T1-29; Berkeley Humanoid Lite; EngineAI PM01; Fourier GR3, N1; HighTorque Hi; Kuavo S45; PAL Talos; PND Adam Lite; Tienkung; plus `human_motion` | AMASS (about 13k clips per robot per the card; unverified) 5.1–8.1 GB per robot, 124.8 GB in all; LAFAN1 0.5–0.8 GB per robot | npz at 30 Hz: `joint_pos`, `joint_vel` (= `np.gradient` of the positions), `joint_names` (Isaac Lab breadth-first order); position, quaternion (w, x, y, z), linear and angular velocity of every body, **all in the world frame**; body 0 is the pelvis. G1: all 32 bodies match forward kinematics of `g1_29dof_rev_1_0.urdf` to 0.02 mm ([G1 model](#g1-model)). **Every joint moves, wrists included**: wrist pitch −1.05 to 1.61 rad and wrist yaw ±1.61 rad over 107 clips, and 1 of the 107 is pinned at a limit. The soles sit at about −2 cm. (Checked on 107 G1 ACCAD clips read from the start of the archive.) | npz in tar.gz ⚙️ `xwbc` (read as a stream, never extracted; the whole archive must be downloaded) | card MIT, which conflicts with AMASS and LAFAN1: research only | **P** (the widest humanoid morphology set, with base twist) and **R** (the G1 AMASS set with real wrist motion). Retargeted with GMR (inferred: the joint ranges equal GMR's `g1_mocap_29dof.xml`). The converter puts motions on the ground like `amass_g1`. Robot models are not shipped: other robots need `--urdf` and `--contact_link`, and 16 of the 18 are in [GMR](https://github.com/YanjieZe/GMR)'s `assets/` (per-model licenses). On B3 walk1 it correlates 0.92–1.00 per joint (median 0.98) with TWIST2. Paper unverified. |
| [PHUMA](https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA) | G1 (29 columns, 23 retargeted, wrists 0) and H1-2 (27 columns) | 73,076 chunks per robot, about 7.53M frames and 70 h per robot, 30 Hz; one 3.42 GB zip | `root_trans` (m, Z up), `root_ori` (x, y, z, w), `dof_pos`, `fps` (checked on a file) | `.npy` holding a dict ⚠️ 🔧 | Apache-2.0 (card and repo); sources Humanoid-X and Motion-X (non-commercial research) | **R** (filtered for physical plausibility; the paper transfers a policy trained on it to a real G1), P. Joint order from the MJCF in the [PHUMA repo](https://github.com/DAVIAN-Robotics/PHUMA); its H1-2 arm names differ from unitree_ros. Chunks are short (30–134 frames). Lee et al., arXiv:2510.26236. |
| [TWIST motion dataset](https://github.com/YanjieZe/TWIST) (Google Drive link in the README, 6.3 GB) | G1, **23** (no wrists) | 19,930 clips (AMASS subsets, OMOMO 6,315, in-house 154); the zip holds the tree twice | `fps` (per clip, not an integer), `root_pos` (m), `root_rot` (x, y, z, w; checked on a file), `dof_pos` (23), `local_body_pos`, quantized to 1e-4 | pickle ⚠️ 🔧 | repo MIT; data AMASS- and OMOMO-derived, research only | R (TWIST's tracker was trained on it), P. The 23-joint order is inferred from TWIST's `assets/g1/` URDF. Ze et al., arXiv:2505.02833. |
| [TWIST2 motion dataset](https://github.com/amazon-far/TWIST2) (`TWIST2_full.zip` on Google Drive, 5.5 GB; Drive serves range requests, so single members can be read) | G1, 29 | 31,961 clips: AMASS 13,218, OMOMO 5,882, TWIST converted 12,788, in-house 73 | `fps` (float, per clip), `root_pos`, `root_rot` (x, y, z, w), `dof_pos` (29, URDF order; no names stored), `local_body_pos` (38 bodies, pelvis frame); every joint moves, within the full URDF limits (checked on 48 clips) | pickle ⚠️ 🔧 (only numpy arrays inside: a restricted unpickler reads them) | repo MIT; AMASS and OMOMO "for research purposes only" | **R**, P. **The stored `fps` is wrong for AMASS subsets not recorded at 120 Hz**: it is about 30 times (native rate / 120). For example, KIT (100 Hz) clips come out 17% short; correcting the rate raises the correlation with any4hdmi from 0.56 to 0.98. A converter must take the rate from the native AMASS rate. Retargeted with GMR (Araujo et al., arXiv:2510.02252). Ze et al., arXiv:2511.02832. |
| [ChingMu MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode) | G1, 29 | 92,559 CSVs, 148 GB; the card claims over 1000 h of 120 Hz optical mocap | header: root position (m, although the README says mm), root quaternion (w, x, y, z), 29 named joints (checked on a file); frame rate not stated | CSV per clip 🔧 | "ChingMu Dataset — Access Terms": academic, personal or prototyping use only; no redistribution | P, R: real mocap, not video. |
| [HOI-Retarget](https://huggingface.co/datasets/leggedrobotics/hoi-retarget) | G1, 29; Unitree H2, 31 | 6,952 motions per robot, 13.8 h per robot, 30 Hz | one row per clip: `root_pos`, `root_rot` (w, x, y, z), `dof_pos` with `dof_names`, **foot ground-contact flags**, per-link contacts, object pose, quality flags (checked on a file) | parquet 🔧 (one row per clip, not per frame) | data CC BY-NC-SA 4.0; code BSD-3-Clause (ETH Zurich) | P with contact labels, R. Object interactions need the objects simulated. Shin et al., arXiv:2609.34674. |
| [any4hdmi](https://huggingface.co/elijahgalahad) (34 repos, e.g. `any4hdmi-g1-amass`, `g1-100style`) | G1, H2, Roboparty RP1, A3 | g1-amass (revision `48e3fff0`) 10,385 segments of at most 1,000 frames from 7,817 motions (35.7 h, 50 Hz, 579 MB); g1-100style 3,573 clips (18.7 h); OMOMO object sets | `qpos` [T, 36]: root xyz, quaternion (w, x, y, z), 29 joints in URDF order (`manifest.json`); every joint moves (checked on 49 clips) | npz 🔧 | **none** (no card or README) | P, R only after filtering. The root and joints are float16 values (steps of about 1e-3 rad and 2–4 mm), which makes finite-difference velocities noisy. In 6 of 49 clips the right arm sits at a joint limit in 85–100% of frames. The soles are about 4 cm below z = 0. `h2-seed-100` derives from BONES-SEED and falls under its terms. By Haoyang Weng. |
| [OmniRetarget](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset) | G1, 29 | 2,119 clips (objects 1,826, terrain 145, both 148), about 4 h, 30 Hz | `qpos` = [qw, qx, qy, qz, x, y, z] (**orientation first**), 29 joints, optional object pose (checked on a file) | npz 🔧 | card MIT, no LICENSE file; the LAFAN1 part is withheld | R, P. Object and terrain clips need their URDFs (shipped); a flat-floor contact rule is wrong for them. Yang et al., arXiv:2509.26633. |
| [LeVERB-Bench](https://huggingface.co/datasets/ember-lab-berkeley/LeVERB-Bench-Dataset) | G1, 29 | 3,696 episodes / 446k frames (about 4.1 h), 30 Hz; 62.8 GB, mostly video | `observation.joint_pos` (Isaac Lab order, names in `info.json`), pelvis-first body positions and quaternions (w, x, y, z) | LeRobot v2.1 ✅ (map joints by name) | Apache-2.0 | P; R partly (short clips repeated across scenes). The pelvis height has a per-scene offset. Xue et al., arXiv:2506.13751. |
| [VideoMimic](https://github.com/hongsukchoi/VideoMimic) (`videomimic_captures.zip`, 154 MB) | G1, 23 | 123 clips with scene meshes, 15 or 30 Hz | `joints`, `root_pos`, `root_quat` (x, y, z, w), link poses, **foot contacts** (checked on a file) | HDF5 🔧 | MIT | P (small, with contacts). The floor is not at z = 0. Allshire et al., arXiv:2505.03729. |
| [G1 Dances](https://huggingface.co/datasets/Kyry10/g1-dances) | G1, 29 | 893 AIST++ clips, 169 min, 60 Hz | `qpos` (T, 36), quaternion (w, x, y, z), within joint limits (checked on a file) | npz 🔧 | AIST derived-motion conditions | R (dance only). Retargeted with PHUMA's method and checked to be trackable. |
| [ToddlerBot motions](https://github.com/hshi74/toddlerbot) (`motion/*.lz4`) | ToddlerBot (0.3 m), 30 motors | 30 skills × 2 hardware variants, 50 Hz (crawl, get-up, push-up, cartwheel, …) | `qpos` (free joint, quaternion w, x, y, z, then 44 MJCF joints), `action` (motor targets), body states (checked on a file) | LZ4-compressed joblib ⚠️ 🔧 | MIT | P (a very different morphology with lots of ground contact), R. Keyframe-authored, not retargeted. Shi et al., arXiv:2502.00893. |
| [Booster motions](https://github.com/BoosterRobotics/booster_assets) (`motions/`) | Booster K1, T1, T2 | 4 CSVs at 50 Hz | root xyz, quaternion (x, y, z, w), joints in `motions.py` order | CSV 🔧 | BSD-3-Clause | R, small. Models are in the same repo. Also `wu0712/retargeted_lafan1_for_booster_k1` (78 clips) and `whirlwind-ams/lafan_locomotion_k1` (7 clips), both without a license. |
| [KungfuBot motions](https://github.com/TeleHuman/PBHC) and [ASAP motions](https://github.com/LeCAR-Lab/ASAP) | G1, 23 | 7 and 52 clips, 30 Hz | `root_trans_offset`, `root_rot` (x, y, z, w), `dof`, `pose_aa` | joblib ⚠️ 🔧 | CC BY-NC 4.0 (KungfuBot); MIT (ASAP) | R: highly dynamic skills. Xie et al., NeurIPS 2025; He et al., RSS 2025. |
| [Humanoid-X](https://huggingface.co/datasets/USC-PSI-Lab/Humanoid-X) | Unitree H1-2 | 156,918 clips, 35.4 GB zip | dict with `root_pos`, `root_rot` (x, y, z, w), `dof_pos`, `fps` (from code) | `.npy` holding a dict ⚠️ 🔧 | **none stated**; sources are internet videos and AMASS/Motion-X | P at best: poses estimated from video float, penetrate the floor and slide (PHUMA paper). Prefer PHUMA, which re-curates the same sources. Mao et al., arXiv:2412.14172. |
| [Lite-Kimodo-Dataset](https://huggingface.co/datasets/Berkeley-Humanoids/Lite-Kimodo-Dataset) | Berkeley Humanoid Lite Pro | about 29k motions (about 182 h), 50 Hz | motions **generated** by NVIDIA's Kimodo model, then retargeted | Rerun `.rrd` 🔧 | Apache-2.0 | R only. Model: [lite_description](https://github.com/Berkeley-Humanoids/lite_description) (MIT). |

### Simulated: many embodiments and exact torques

| Dataset | Robots | Size | Signals | Format | License | Notes |
|---|---|---|---|---|---|---|
| [Minari MuJoCo](https://huggingface.co/datasets/farama-minari/mujoco) (`minari download mujoco/<env>/<level>-v0`) | Hopper, Walker2d, HalfCheetah, Ant (free root), Humanoid and HumanoidStandup (free root), Swimmer, … (Gymnasium v5 MJCF) | about 1M steps per set (Ant expert 1.9 GB, Humanoid 2.9 GB) | observations only: no root x (and y for Ant and Humanoid); torque = gear × action (Humanoid also stores `qfrc_actuator`); Hopper and Walker2d velocities clipped to ±10 (checked on a file) | HDF5 🔧, MJCF to URDF needed | dataset license unverified; code MIT | P: exact torques on 6 morphologies, expert gaits for Ant and Humanoid. Mask clipped frames. Minari, DOI 10.5281/zenodo.13767625. |
| [D4RL Gym-MuJoCo v2](http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/) | Hopper, Walker2d, HalfCheetah, Ant | 150–376 MB per file | full `infos/qpos` and `infos/qvel`; torque = gear × action | HDF5 🔧, MJCF | data CC BY 4.0, code Apache-2.0 | P: the clearest-licensed full-state option. Which gym XML version was used is unverified. Fu et al., arXiv:2004.07219. |
| [RL Unplugged](https://github.com/google-deepmind/deepmind-research/tree/master/rl_unplugged) (`humanoid_run`, …) | dm_control humanoid, walker, cheetah, CMU humanoid | `humanoid_run` 3,000 episodes, 1.21 GiB | humanoid joint angles and full velocities, torque = gear × action; no base x, y or yaw | TFDS / RLDS 🔧 | dataset unverified (code Apache-2.0) | P. Gulcehre et al., NeurIPS 2020. |
| [MoCapAct](https://huggingface.co/datasets/microsoft/mocapact-data) | CMU humanoid, 56 joints + free root | small 45.7 GB, large 559 GB | joint positions and velocities, actuator activations, gyro and accelerometer; torque = clip(kp (target − q)); no world x, y or yaw | HDF5 🔧, MJCF | CDLA-Permissive-2.0 (Microsoft) | P, and expert tracking of mocap on a high-DoF humanoid. Wagener et al., NeurIPS 2022 Datasets and Benchmarks. |
| [BEHAVIOR-1K 2025 challenge demos](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos) (also `lerobot/behavior1k-task*`) | Galaxea R1Pro (wheeled; torso 4, two 7-DoF arms) | 10,000 episodes / 119M frames, 30 Hz | joint positions, velocities and **simulated efforts** (arm efforts saturate at the URDF limits; torso and base efforts suspect), base pose and twist | LeRobot ✅ (index ranges in `PROPRIOCEPTION_INDICES`) | MIT tag | P. URDF in `behavior-1k/omnigibson-robot-assets`. Li et al., arXiv:2403.09227. |
| [GR00T X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | Panda (single, on a mobile base, bimanual), Fourier GR-1 (44 joints) | many tasks, 20 Hz | joint positions (velocities for the single Panda); actions are end-effector commands except GR-1 | LeRobot ✅ | CC BY 4.0 (task names match DexMimicGen, CC BY-NC-SA 4.0: check) | P. Robosuite MJCF. NVIDIA et al., arXiv:2503.14734. |
| [ManiSkill3 demos](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations) | mostly Panda; also ANYmal C reach | many tasks | root pose and twist, `qpos`, `qvel` per articulation; no torque (checked on a file) | HDF5 🔧 | Apache-2.0 | P. URDFs in the ManiSkill repo. Tao et al., arXiv:2410.00425. |

### Arms and hands (fixed base)

| Dataset | Robot | Size | Torque | Format | License | Notes |
|---|---|---|---|---|---|---|
| [`unitreerobotics/G1_Dex3_*`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) (13 repos) | G1 arms 14 + Dex3 hands | 3,152 episodes / 2.59M frames, 30 Hz | no | LeRobot ✅ | Apache-2.0 | `configs/data/g1_hub_plus_synthetic.json` (legs frozen, base fixed). By Unitree Robotics. `G1_Brainco` (9 repos) and `Z1_*` arms have the same `k*` joint naming. |
| [Raw DROID](https://droid-dataset.github.io/droid/the-droid-dataset) (`gs://gresearch/robotics/droid_raw/1.0.1/`, also over HTTPS) | Franka Panda, 7 | 1.4–2.5 MB of `trajectory.h5` per episode (skip the 8.7 TB of video) | **`motor_torques_measured`** and `joint_torques_computed`, with joint positions and velocities | HDF5 🔧 | CC BY 4.0 (paper) | P: the largest real arm set with torque. About 20% of raw episodes were missing at copy time (website). Units and gravity compensation unverified. Khazatsky et al., RSS 2024. |
| [`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1) | Franka Panda, 7 | 95,658 episodes / 27.6M frames, 15 Hz | no (this copy drops the torques) | LeRobot ✅ | the paper releases the data under CC BY 4.0; the `lerobot` card says Apache-2.0 | `state_key: "observation.state.joint_position"`; use `max_episodes`. |
| [FurnitureBench](https://clvrai.github.io/furniture-bench/) (RLDS `furniture_bench_dataset_converted_externally_to_rlds`) | Franka Panda, 7 | 5,100 episodes, 10 Hz, 123.5 GB | commanded torque in `state[27:34]` (Polymetis `joint_torques_computed`) | RLDS 🔧 (the raw `.pkl` version needs unpickling ⚠️) | code MIT; data license not stated | P. Joint positions in `state[13:20]`, velocities in `[20:27]`. Heo et al., RSS 2023. |
| [UCSD Kitchen](https://huggingface.co/datasets/lerobot/ucsd_kitchen_dataset) (`lerobot/ucsd_kitchen_dataset`) | UFACTORY xArm7, 7 | 150 episodes / 3,970 frames, 2 Hz (the info file says 5) | **joint torques** in `state[14:21]` (units unverified) | LeRobot ✅ | MIT | P: small and sparse but real torque. Model: [xarm_ros](https://github.com/xArm-Developer/xarm_ros) (BSD-3-Clause). |
| [REASSEMBLE](https://researchdata.tuwien.ac.at/records/0ewrv-8cb44) | Franka (Panda or FR3), 7 | 4,551 demonstrations, 781 min, 58.9 GB | `joint_efforts` (meaning unverified), wrist force-torque | HDF5 with per-stream timestamps 🔧 | data CC BY 4.0, code MIT | P: contact-rich assembly. Sliwowski et al., RSS 2025. |
| [Franka Panda excitation logs](https://github.com/marcocognetti/FrankaEmikaPandaDynModel) | Franka Panda, 7 | one 21 s trajectory at about 1 kHz | measured torques (gravity compensation unverified) | text 🔧 | GPL-3.0 | ID validation. Gaz et al., RA-L 2019. |
| [Real ALOHA](https://huggingface.co/datasets/lerobot/aloha_static_coffee) (`lerobot/aloha_static_*`, `lerobot/aloha_mobile_*`) | 2 × ViperX 300 6DOF | e.g. 50 episodes / 55k frames, 50 Hz | `observation.effort` is **motor current in mA** (13 static and 6 mobile repos), not torque (checked on a file) | LeRobot ✅ | MIT or Apache-2.0 per card | P; leave `torque_key` unset unless you model current to torque. No velocity column. Joint names match `vx300s` in interbotix descriptions (BSD). Zhao et al., RSS 2023; Fu et al., arXiv:2401.02117. |
| [`lerobot/aloha_sim_*`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | 2 × ViperX 300 | 50 episodes / 25k frames, 50 Hz | no | LeRobot ✅ | card MIT | Map 12 arm joints; skip the grippers. |
| LeRobot copies of Open X-Embodiment with joint angles (table below) | Panda, xArm7, UR5, FANUC, Cobotta | 50–1,500 episodes each | no (except UCSD Kitchen) | LeRobot ✅ (`state_indices` select the joints) | mostly MIT | P. Many other LeRobot OXE copies keep only the end-effector pose: use their RLDS originals. |
| RLDS originals with joints (table below) | KUKA iiwa, UR5, Jaco 2, Panda, xArm7 | 102–18,250 episodes | no | RLDS 🔧 | per dataset | P. |
| [RH20T](https://rh20t.github.io/) (HF mirror `tonyfanggalaxies/RH20T`) | Flexiv Rizon, UR5, Franka, KUKA iiwa | low-dimensional archives about 214 GB | listed in the paper; one Franka file held angles and velocities only (checked on a file) | `joint.npy` object arrays ⚠️ 🔧 | scenes 1–5 CC BY-SA 4.0, 6–10 CC BY-NC 4.0 | P. URDFs in `rh20t_api`. Fang et al., ICRA 2024. |
| [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | AgiBot G2 (wheeled), 14 arm joints | e.g. 159 episodes / 210k frames, 30 Hz | `joint/effort` in `observation.state[79:93]` (units undocumented) | LeRobot v2.1 ✅ (index ranges in `field_descriptions`) | CC BY-NC-SA 4.0 | P (arm effort). URDF in [genie_sim](https://github.com/AgibotTech/genie_sim) (MPL-2.0). |
| [Galaxea R1Pro pick and place](https://huggingface.co/datasets/OpenGalaxea/R1Pro_Pick_and_Place_Cube) | Galaxea R1 Pro (wheeled) | 200 files, 14.4 GB | arm effort at about 165 Hz, torso effort at about 450 Hz, two IMUs (checked on a file) | ROS 2 MCAP 🔧 | CC BY-NC-SA 4.0 | P. |
| [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace), [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) | SO-100 / SO-101, 5 + gripper | 50 episodes each, 30 Hz | no | LeRobot ✅ | Apache-2.0 | SO-101 values are in [−100, 100] normalized range: radians need per-robot calibration, which is not shipped. Model: [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `so101_new_calib.urdf` (Apache-2.0). Shukor et al., arXiv:2506.01844. |
| [Trossen WidowX AI bimanual](https://huggingface.co/TrossenRoboticsCommunity) (`TrossenRoboticsCommunity/*`) | Trossen WidowX AI, 6 + carriage | e.g. 60 episodes / 27k frames, 30 Hz | no | LeRobot ✅ (`joint_0..5` in rad, carriage in m) | Apache-2.0 | P. Model: [trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description) (BSD-3-Clause). |
| Community uploads with torque or current: [`sonnyjin/ur5e_cube_auto_torque_200_all_7225`](https://huggingface.co/datasets/sonnyjin/ur5e_cube_auto_torque_200_all_7225) (UR5e, 7,225 episodes / 1.09M frames), [`polarisai-robots/bento_v2_openarm`](https://huggingface.co/datasets/polarisai-robots/bento_v2_openarm) (OpenArm, torque estimated from current), [`fitz0401/fr3_peel_20261109`](https://huggingface.co/datasets/fitz0401/fr3_peel_20261109) (FR3, joint and external torque) | — | — | yes (units unverified; template cards) | LeRobot ✅ | Apache-2.0 tags | P after checking units. |

LeRobot copies of Open X-Embodiment datasets that keep the joint angles (checked on
files; rates from the Open X-Embodiment spreadsheet, which several `info.json` files
contradict):

| Repo | Robot | Episodes / frames, Hz | Joints in `observation.state` | License | Paper |
|---|---|---|---|---|---|
| `lerobot/toto` | Panda | 1,003 / 325,699, 30 | [0:7] (joint 6 goes below the Panda limit: check) | MIT | Zhou et al., ICRA 2023 |
| `lerobot/utaustin_mutex` | Panda | 1,500 / 361,883, 20 | [0:7] | MIT | Shah, Martín-Martín and Zhu, CoRL 2023 |
| `lerobot/iamlab_cmu_pickup_insert` | Panda | 631 / 146,241, 20 | [0:7] | Apache-2.0 | Saxena, Sharma and Kroemer, CoRL 2023 |
| `lerobot/austin_sirius_dataset` | Panda | 559 / 279,939, 20 | [0:7] (some frames all zero) | MIT | Liu et al., RSS 2023 |
| `lerobot/austin_buds_dataset` | Panda | 50 / 34,112, 20 | [0:7] | MIT | Zhu, Stone and Zhu, RA-L 2022 |
| `lerobot/berkeley_rpt` | Panda | 908 / 392,578, 30 | [0:7]; action is a joint delta | MIT | Radosavovic et al., arXiv:2306.10007 |
| `lerobot/nyu_franka_play_dataset` | Panda | 456 / 44,875, 3 | [0:7]; action [0:7] is joint velocity | MIT | Cui et al., arXiv:2210.10047 |
| `lerobot/cmu_play_fusion` | Panda | 576 / 235,922, 5 | [0:7] | MIT | Chen, Bahl and Pathak, CoRL 2023 |
| `lerobot/berkeley_mvp` | xArm7 | 480 / 45,308, 5 | [8:15]; action is a joint delta | MIT | Radosavovic et al., CoRL 2022 |
| `lerobot/berkeley_fanuc_manipulation` | FANUC LR Mate 200iD, 6 | 415 / 62,613, 10 | [0:6] | MIT (the website also says non-commercial) | Zhu et al., 2023 |
| `lerobot/asu_table_top` | UR5, 6 | 110 / 26,113, 12.5 | [0:6] | MIT | Zhou et al., CoRL 2023 |
| `lerobot/tokyo_u_lsmo` | DENSO Cobotta, 6 | 50 / 11,925, 10 | [6:12] | MIT (the Cobotta model is GPL-2.0) | Osa, IJRR 2022 |

Robot models: Panda from [franka_description](https://github.com/frankaemika/franka_description)
(Apache-2.0), xArm7 from xarm_ros (BSD-3-Clause), UR5 from
[Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
(BSD-3-Clause), FANUC from [ros-industrial/fanuc](https://github.com/ros-industrial/fanuc)
(exact variant unverified), Cobotta from denso_cobotta_ros (GPL-2.0).

RLDS originals whose LeRobot copies lost the joints (Open X-Embodiment, under
`https://storage.googleapis.com/gresearch/robotics/<name>/`): `stanford_kuka_multimodal_dataset_converted_externally_to_rlds`
(KUKA iiwa, `joint_pos`, `joint_vel`, end-effector forces, 3,000 episodes, MIT card; Lee et
al., ICRA 2019), `berkeley_autolab_ur5` (UR5, `robot_state[0:6]`, 1,000 episodes, CC BY 4.0),
`kaist_nonprehensile_converted_externally_to_rlds` (Panda, positions and velocities
interleaved in `state[0:14]`, CC BY 4.0), `jaco_play` (Kinova Jaco 2, `joint_pos` in the Kinova
convention, CC BY 4.0), `taco_play` (Panda, `robot_obs[7:14]`, CC BY 4.0),
`stanford_hydra_dataset_converted_externally_to_rlds` (Panda, joints and velocities, MIT),
`utokyo_xarm_pick_and_place_converted_externally_to_rlds` (xArm7, position, velocity and
acceleration, Apache-2.0), `austin_sailor_dataset_converted_externally_to_rlds` (Panda,
MIT), and `fmb` (Panda, joints and end-effector wrench, CC BY 4.0).

### Gated, unreleased, or unusable

| Candidate | Why it is not in the tables |
|---|---|
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) 🔒 | The largest G1 motion set (142,220 motions including mirrors, about 288 h, 120 Hz), but gated behind a form, with no redistribution and no competing datasets or generative models allowed. `Jay-Ye/SONIC_R1_retarget` (Unitree R1) falls under the same terms. |
| [HDMI](https://github.com/LeCAR-Lab/HDMI) | 16 G1 clips with velocities, but the repo has no license. |
| Fourier ActionNet 🔒, RoboMIND 🔒, Galaxea Open-World 🔒, AgiBot World Alpha/Beta 🔒 | Gated. ActionNet's leg columns were zero in the episode checked (upper body only); RoboMIND's Tien Kung data is arms only; Galaxea and AgiBot robots are wheeled. |
| H2O / OmniH2O, GMT, MimicKit | Only example motions are released; retargeting scripts only. |
| 1X world model data, `X-Humanoid/HEX-Datasets`, `lerobot/unitreeh1_*` | No public robot model, or unnamed state vectors. |
| M3ED, SubT-MRS, NeBula (Spot), DiTer (Go1) | No joint states (payload sensors and odometry only). |
| M-SEVIQ (Go2) | The only link has expired. |
| SARCOS, Barrett WAM, KUKA LWR inverse-dynamics sets | No public robot model or primary download. |
| GenBot-1K trajectories, URMA, MetaMorph/UNIMAL | Robot models only, no trajectories (GenBot-1K's URDFs are in [the simulated table](#legged-robots-on-the-ground-simulated)). |
| `lerobot/utokyo_saytap` (A1) | Joint values look offset and scaled, and timestamps step at 0.2 s while the source implies 50 Hz. |

### Multirotors (floating base, thrust actuators)

| Dataset | Vehicle | Size | Rotor signal | Base | Format | License | Notes |
|---|---|---|---|---|---|---|---|
| [NeuroBEM](https://rpg.ifi.uzh.ch/NeuroBEM.html) | quadrotor, 0.772 kg (dataset Readme; the paper says 752 g) | 1 h 15 min, 400 Hz | **motor speeds** (rad/s) and their derivatives | Vicon pose, velocity, acceleration, body rates | CSV ⚙️ `neurobem` | not stated | The best rotor-dynamics set. The converter identifies the thrust coefficient, generates the URDF, and writes the actuator overrides. Bauersfeld et al., RSS 2021. |
| [Blackbird](https://github.com/mit-aera/Blackbird-Dataset) | quadrotor | 168 flights, >10 h; RPM about 190 Hz | **per-motor RPM** and PWM | 360 Hz mocap pose, IMU | ROS 1 bags (4.9 TB) 🔧 | repo MIT; data unverified | The official host was unreachable at check time; an Academic Torrents mirror exists. Antonini et al., IJRR 2020. |
| [TII drone racing](https://github.com/tii-racing/drone-racing-dataset) | 5-inch FPV quad (open design) | 42 flights (18 piloted and 18 autonomous, plus 6 piloted in release v3.0.0), 500 Hz | motor commands (0–1), no RPM | mocap pose and twist, IMU | CSV and ROS 2 bags 🔧 | data CC BY 4.0; code MIT | Commands are not speeds; map them with a thrust curve. Bosello et al., RA-L 2024. |
| [UAV-SEAD](https://huggingface.co/datasets/aykutkabaoglu/UAV-SEAD_State_Estimation_Anomaly_Dataset) (`aykutkabaoglu/UAV-SEAD_…`) | PX4 quadrotors in six airframe configurations (paper) | 1,396 flights, about 52 h | PWM outputs | attitude, rates, local position, IMU | PX4 ulog + CSV 🔧 | CC BY 4.0 | Airframes not identified per flight. Kabaoglu and Sariel, arXiv:2602.13900; the paper's own card is [`aykutkabaoglu/uav-flight-anomaly-dataset`](https://huggingface.co/datasets/aykutkabaoglu/uav-flight-anomaly-dataset) (DOI 10.57967/hf/7772). |
| Synthetic (`configs/data/quadrotor_flight.json`) | any multirotor URDF | unlimited | exact rotor speeds and torques | exact | built in ✅ | — | Exact thrust, drag, and base dynamics; randomized mass and arm length. |

**Robot models.** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
has `unitree_g1`, `unitree_h1`, `unitree_go1`, `unitree_go2`, `booster_t1`,
`apptronik_apollo`, `pal_talos`, `agility_cassie`, `anybotics_anymal_c`, and
`bitcraze_crazyflie_2` (MJCF; convert to URDF; Apache-2.0 overall, each model has its
own license). Unitree's
[unitree_ros](https://github.com/unitreerobotics/unitree_ros) (BSD-3-Clause) has URDFs for
G1, H1, H1_2, Go1, Go2 and A1 (`scripts/download_urdfs.py` fetches three). The Go1, H1 and
G1 URDFs used by `configs/data/*.json` come from it, the ANYmal D URDF from
[anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description)
(ANYbotics AG, BSD-3-Clause); `assets/robots/quadrotor_x.urdf` is this repository's.

**A suggested mix for cross-embodiment pretraining:**
- the quadruped sets with contact sensors (GrandTour, MIPO, DFKI Go2, ForEnt) for real
  locomotion with known contact sets (their torques can be enabled with
  `measured_torque_weight`, or kept for validation);
- MuJoCo ground data for exact contact dynamics of every legged robot you care about,
  including GenBot-1K robots;
- G1 WBT, G1_WBT_SMPL, xwbc-retarget-data and LocoMuJoCo for humanoid motion on many
  morphologies, and ergoCub for real humanoid torques;
- arms with joint torques (raw DROID, UCSD Kitchen) and with joint angles (the LeRobot OXE
  copies) for fixed-base dynamics;
- NeuroBEM for real rotor dynamics;
- synthetic fixed, floating, and flight data for every robot you care about
  (exact torques, probes, limits, randomized inertia and rotor inertia): the only
  source of full-torque B and C labels besides MuJoCo;
- held-out robots or randomized variants for evaluation.

## References

Checked against the publisher, proceedings, arXiv, dataset card or repository on
2026-11-20 and 2026-11-21. Licenses are those stated by the holders; check them before use.

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

- Natapat Kirdwichai, Danesh Tarapore, Tobias Driskelle-Poole, and Andrei Sontea.
  *ForEnt: A multimodal dataset for characterising quadruped robot entrapments in forest
  environments.* Zenodo, 2026. DOI
  [10.5281/zenodo.23802853](https://doi.org/10.5281/zenodo.23802853) (CC BY 4.0).
- Emre Girgin and Cagri Kilic. *Learning Contact Representation for Leg Odometry.* CoRL
  2026, arXiv:2606.05501. Data: <https://zenodo.org/records/23287210> (CC BY 4.0).
- Hexiang Wei, Jianhao Jiao, Xiangcheng Hu, Jingwen Yu, Xupeng Xie, Jin Wu, Yilong Zhu,
  Yuxuan Liu, Lujia Wang, and Ming Liu. *FusionPortableV2: A Unified Multi-Sensor Dataset for
  Generalized SLAM Across Diverse Platforms and Scalable Environments.* International Journal
  of Robotics Research 44(7):1093–1116, 2025. DOI
  [10.1177/02783649241303525](https://doi.org/10.1177/02783649241303525), arXiv:2404.08563.
- Chen Yao, Yangtao Ge, Guowei Shi, Zirui Wang, Ningbo Yang, Zheng Zhu, Hexiang Wei, Yuntian
  Zhao, Jing Wu, and Zhenzhong Jia. *TAIL: A Terrain-Aware Multi-Modal SLAM Dataset for Robot
  Locomotion in Deformable Granular Environments.* IEEE RA-L 9(7):6696–6703, 2024. DOI
  [10.1109/LRA.2024.3407411](https://doi.org/10.1109/LRA.2024.3407411), arXiv:2403.16875.
  Zirui Wang et al. (10 authors). *Are We Ready for Planetary Exploration Robots? The
  TAIL-Plus Dataset for SLAM in Granular Environments.* ICRA 2024 workshop, arXiv:2404.13600. Data:
  <https://tailrobot.github.io/download/> (GPL-3.0).
- Xinge Zhao, Xing Zhang, Yiqing Ni, Qingquan Li, Wentao Zeng, Haixia Feng, and Yibo Zhou.
  *Gait-Aware quadruped 3D mapping in challenging environments with complex terrain.* ISPRS
  Journal of Photogrammetry and Remote Sensing 233:664–688, 2026. DOI
  [10.1016/j.isprsjprs.2026.02.010](https://doi.org/10.1016/j.isprsjprs.2026.02.010). Data:
  <https://github.com/EN3D-Lab/Quadruped-SLAM-dataset> (no license stated).
- Juwon Kim, Hogyun Kim, Seokhwan Jeong, Youngsik Shin, and Younggun Cho. *DiTer++: Diverse
  Terrain and Multi-Modal Dataset for Multi-Robot SLAM in Multi-Session Environments.* ICRA
  2025, pp. 12187–12193. DOI [10.1109/ICRA55743.2025.11128593](https://doi.org/10.1109/ICRA55743.2025.11128593),
  arXiv:2412.05839. Data: <https://github.com/sparolab/DiTer-plusplus> (CC BY-NC-SA 4.0).
- Geoff Fink and Claudio Semini. *The DLS Quadruped Proprioceptive Sensor Dataset.* CLAWAR
  2020. Data DOI [10.21227/4vxz-xw05](https://doi.org/10.21227/4vxz-xw05) (CC BY 4.0).
- paLeziart. *solo12-recordings.* 2021. <https://github.com/paLeziart/solo12-recordings>
  (BSD-2-Clause). Solo-12 model: Gepetto,
  [example-robot-data](https://github.com/Gepetto/example-robot-data) (BSD-3-Clause).
- Tzu-Yuan Lin, Ray Zhang, Justin Yu, and Maani Ghaffari. *Legged Robot State Estimation using
  Invariant Kalman Filtering and Learned Contact Events.* CoRL 2021, arXiv:2106.15713. Data:
  <https://github.com/UMich-CURLY/deep-contact-estimator> (no license stated).
- Marco Camurri, Milad Ramezani, Simona Nobili, and Maurice Fallon. *Pronto: A Multi-Sensor
  State Estimator for Legged Robots in Real-World Scenarios.* Frontiers in Robotics and AI
  7:68, 2020. DOI [10.3389/frobt.2020.00068](https://doi.org/10.3389/frobt.2020.00068).
- Shifan Zhu, Zixun Xiong, and Donghyun Kim. *CEAR: Comprehensive Event Camera Dataset for Rapid
  Perception of Agile Quadruped Robots.* IEEE RA-L 9(10):8999–9006, 2024. DOI
  [10.1109/LRA.2024.3426373](https://doi.org/10.1109/LRA.2024.3426373) (data: all rights
  reserved).
- S. Godon, A. Ristolainen, and M. Kruusmaa. *Robotic feet modeled after ungulates improve
  locomotion on soft wet grounds.* Bioinspiration & Biomimetics 19:066009, 2024.
  DOI [10.1088/1748-3190/ad839c](https://doi.org/10.1088/1748-3190/ad839c). Data:
  <https://zenodo.org/records/12673097> (CC BY 4.0).
- Daniel Ordoñez-Apraez (uploader), Dynamic Legged Systems lab, IIT. *quadruped_locomotion.*
  <https://huggingface.co/datasets/DLS-IIT/quadruped_locomotion> (MIT on the card).

**Quadruped motion datasets**
- Lei Han et al. (19 authors). *Lifelike Agility and Play in Quadrupedal Robots using
  Reinforcement Learning and Generative Pre-trained Models.* Nature Machine Intelligence
  6:787–798, 2024. DOI [10.1038/s42256-024-00861-3](https://doi.org/10.1038/s42256-024-00861-3),
  arXiv:2308.15143. Data: <https://github.com/Tencent-RoboticsX/lifelike-agility-and-play>.
- He Zhang, Sebastian Starke, Taku Komura, and Jun Saito. *Mode-Adaptive Neural Networks for
  Quadruped Motion Control.* ACM Transactions on Graphics 37(4), Article 145, 2018. DOI
  [10.1145/3197517.3201366](https://doi.org/10.1145/3197517.3201366). Mocap: AI4Animation
  (CC BY-NC 4.0).
- Alejandro Escontrela, Xue Bin Peng, Wenhao Yu, Tingnan Zhang, Atil Iscen, Ken Goldberg, and
  Pieter Abbeel. *Adversarial Motion Priors Make Good Substitutes for Complex Reward
  Functions.* arXiv:2203.15103, 2022. Code and motions:
  <https://github.com/escontra/AMP_for_hardware> (BSD-3-Clause).
- Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan, and Sergey Levine.
  *Learning Agile Robotic Locomotion Skills by Imitating Animals.* RSS 2020. DOI
  [10.15607/RSS.2020.XVI.064](https://doi.org/10.15607/RSS.2020.XVI.064), arXiv:2004.00784.
- Gilbert Feng et al. (11 authors). *GenLoco: Generalized Locomotion Controllers for
  Quadrupedal Robots.* arXiv:2209.05309, 2022. Code and motions:
  <https://github.com/HybridRobotics/GenLoco> (Apache-2.0).

**Humanoid datasets**
- Unitree Robotics. *G1 whole-body teleoperation and Dex3 datasets*
  (`unitreerobotics/G1_WBT_*`, `unitreerobotics/G1_Dex3_*`).
  <https://huggingface.co/unitreerobotics> (Apache-2.0 on the cards used here).
- Unitree Robotics. *unitree_sdk2_python* (`G1JointIndex`, the G1 motor order).
  <https://github.com/unitreerobotics/unitree_sdk2_python> (BSD-3-Clause).
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
  retargeting: Kun Zhao, *Retargeted AMASS for Robotics*,
  <https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics> (card CC BY 4.0;
  read by `amass_g1`; an earlier upload is copied at
  <https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1>), which asks to
  also cite Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A.
  Osman, Dimitrios Tzionas, and Michael J. Black. *Expressive Body Capture: 3D Hands,
  Face, and Body From a Single Image.* CVPR 2019, pp. 10967–10977. DOI
  [10.1109/CVPR.2019.01123](https://doi.org/10.1109/CVPR.2019.01123).

- Giulio Romualdi, Paolo Maria Viceconte, Lorenzo Moretti, Ines Sorrentino, Stefano Dafarra,
  Silvio Traversaro, and Daniele Pucci. *Online DNN-driven Nonlinear MPC for Stylistic
  Humanoid Robot Walking with Step Adjustment.* Humanoids 2024 (per the dataset card),
  arXiv:2410.07849. Data:
  <https://huggingface.co/ami-iit> (BSD-3-Clause). ergoCub model:
  <https://github.com/icub-tech-iit/ergocub-software> (BSD-3-Clause).
- M. Eva Mungai, Gokul Prabhakaran, and Jessy W. Grizzle. *Fall Prediction for Bipedal Robots:
  The Standing Phase.*
  arXiv:2309.14546. Data: <https://huggingface.co/datasets/Gokul97/digit_fall_prediction_dataset>,
  <https://github.com/UMich-BipedLab/Digit_Fall_Prediction_Dataset> (repository AGPL-3.0).
- DAIR Lab. *cassie_impact_dataset.* <https://github.com/DAIRLab/cassie_impact_dataset>
  (BSD-3-Clause). The README cites no paper; the lab's study of Cassie impacts is Brian
  Acosta, William Yang, and Michael Posa, *Validating Robotics Simulators on Real-World
  Impacts*, IEEE RA-L 7(3):6471–6478, 2022, DOI
  [10.1109/LRA.2022.3174367](https://doi.org/10.1109/LRA.2022.3174367) (that it uses this
  dataset is unverified).
- Kyochul Jang et al. (12 authors). *HumanoidToolBench: Benchmarking Humanoid Tool Use from
  Selection to Mobile Execution.* arXiv:2610.02089. Data:
  <https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop> (CC BY-NC 4.0).
- Shihan Wu et al. (84 authors). *RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection
  for Integrated Manipulation.* arXiv:2511.17441. Data: <https://huggingface.co/RoboCOIN>.
- GenON Physical AI Lab. *G1_Dex3_Trash_LocoManipulation.* 2026.
  <https://huggingface.co/datasets/mncai/G1_Dex3_Trash_LocoManipulation> (Apache-2.0).
- NVIDIA. *PhysicalAI-Robotics-GR00T-Teleop-G1.*
  <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1> (CC BY 4.0).

**Humanoid motion datasets**
- Kyungmin Lee, Sibeen Kim, Youngdo Lee, Minho Park, Hyunseung Kim, Dongyoon Hwang, Donghu
  Kim, Hojoon Lee, and Jaegul Choo. *PHUMA: Physically Reliable Humanoid Locomotion Dataset.*
  arXiv:2510.26236. Data: <https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA> (Apache-2.0).
- Yanjie Ze, Zixuan Chen, João Pedro Araújo, Zi-ang Cao, Xue Bin Peng, Jiajun Wu, and C. Karen
  Liu. *TWIST: Teleoperated Whole-Body Imitation System.* arXiv:2505.02833.
  <https://github.com/YanjieZe/TWIST> (MIT).
- Yanjie Ze, Siheng Zhao, Weizhuo Wang, Angjoo Kanazawa, Rocky Duan, Pieter Abbeel, Guanya
  Shi, Jiajun Wu, and C. Karen Liu. *TWIST2: Scalable, Portable, and Holistic Humanoid Data
  Collection System.* arXiv:2511.02832. <https://github.com/amazon-far/TWIST2> (MIT).
- Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, and C. Karen Liu. *Retargeting Matters:
  General Motion Retargeting for Humanoid Motion Tracking.* arXiv:2510.02252 (GMR,
  <https://github.com/YanjieZe/GMR>, MIT).
- Jiaman Li, Jiajun Wu, and C. Karen Liu. *Object Motion Guided Human Motion Synthesis.* ACM
  Transactions on Graphics 42, 2023. DOI [10.1145/3618333](https://doi.org/10.1145/3618333)
  (OMOMO, a source of the TWIST sets).
- Jihwan Shin, Adrià López Escoriza, Junzhe He, Matthias Heyrman, and Marco Hutter.
  *HOI-Retarget: Contact-Centric Retargeting for Human-Object Interaction.* arXiv:2609.34674.
  Data: <https://huggingface.co/datasets/leggedrobotics/hoi-retarget> (CC BY-NC-SA 4.0).
- Haoyang Weng. *any4hdmi* datasets. <https://huggingface.co/elijahgalahad>; HDMI,
  arXiv:2509.16757.
- TonyZhang1999. *xwbc-retarget-data.*
  <https://huggingface.co/datasets/TonyZhang1999/xwbc-retarget-data> (MIT on the card).
- Kun Zhao. *Retargeted AMASS for BXI Elf2* and *for Fourier N1.*
  <https://huggingface.co/fleaven> (CC BY 4.0 on the cards; AMASS terms apply).
- ChingMu (CMRobot). *MotionDecode.* <https://huggingface.co/datasets/CMRobot/MotionDecode>
  (ChingMu Dataset Access Terms, non-commercial).
- Haoru Xue et al. (11 authors). *LeVERB: Humanoid Whole-Body Control with Latent
  Vision-Language Instruction.* arXiv:2506.13751. Data:
  <https://huggingface.co/datasets/ember-lab-berkeley/LeVERB-Bench-Dataset> (Apache-2.0).
- Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Haoran Geng,
  Jitendra Malik, Vitor Guizilini, and Yue Wang. *Learning from Massive Human Videos for
  Universal Humanoid Pose Control.* arXiv:2412.14172 (Humanoid-X).
- Arthur Allshire, Hongsuk Choi, Junyi Zhang, David McAllister, Anthony Zhang, Chung Min Kim,
  Trevor Darrell, Pieter Abbeel, Jitendra Malik, and Angjoo Kanazawa. *Visual Imitation
  Enables Contextual Humanoid Control.* arXiv:2505.03729 (VideoMimic, MIT).
- Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza,
  C. Karen Liu, Rocky Duan, and Guanya Shi. *OmniRetarget: Interaction-Preserving Data
  Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction.*
  arXiv:2509.26633.
- Haochen Shi, Weizhuo Wang, Shuran Song, and C. Karen Liu. *ToddlerBot: Open-Source
  ML-Compatible Humanoid Platform for Loco-Manipulation.* arXiv:2502.00893
  (<https://github.com/hshi74/toddlerbot>, MIT).
- Shuhei Tsuchida, Satoru Fukayama, Masahiro Hamasaki, and Masataka Goto. *AIST Dance Video
  Database: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information
  Processing.* ISMIR 2019 (source of `Kyry10/g1-dances`).
- Weiji Xie, Jinrui Han, Jiakun Zheng, Huanyu Li, Xinzhe Liu, Jiyuan Shi, Weinan Zhang,
  Chenjia Bai, and Xuelong Li. *KungfuBot: Physics-Based Humanoid Whole-Body Control for
  Learning Highly-Dynamic Skills.* NeurIPS 2025, arXiv:2506.12851.
- Tairan He et al. (18 authors). *ASAP: Aligning Simulation and Real-World Physics for
  Learning Agile Humanoid Whole-Body Skills.* RSS 2025, arXiv:2502.01143.
- Yitang Li et al. (13 authors). *BFM-Zero: A Promptable Behavioral Foundation Model for
  Humanoid Control Using Unsupervised Reinforcement Learning.* ICLR 2026, arXiv:2511.04131
  (LAFAN1 motions in <https://github.com/LeCAR-Lab/BFM-Zero>, CC BY-NC 4.0).
- Booster Robotics. *booster_assets.* <https://github.com/BoosterRobotics/booster_assets>
  (BSD-3-Clause).
- Berkeley Humanoids. *Lite-Kimodo-Dataset.*
  <https://huggingface.co/datasets/Berkeley-Humanoids/Lite-Kimodo-Dataset> (Apache-2.0).

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

**Simulated datasets**
- Farama Foundation. *Minari.* Zenodo, DOI
  [10.5281/zenodo.13767625](https://doi.org/10.5281/zenodo.13767625);
  <https://huggingface.co/datasets/farama-minari/mujoco>.
- Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. *D4RL: Datasets for
  Deep Data-Driven Reinforcement Learning.* arXiv:2004.07219, 2020 (data CC BY 4.0).
- Caglar Gulcehre et al. (18 authors on arXiv). *RL Unplugged: A Suite of Benchmarks for Offline
  Reinforcement Learning.* NeurIPS 2020, arXiv:2006.13888.
- Nolan Wagener, Andrey Kolobov, Felipe Vieira Frujeri, Ricky Loynd, Ching-An Cheng, and
  Matthew Hausknecht. *MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control.*
  NeurIPS 2022 Datasets and Benchmarks, arXiv:2208.07363 (CDLA-Permissive-2.0).
- Chengshu Li et al. (35 authors). *BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with
  1,000 Everyday Activities and Realistic Simulation.* arXiv:2403.09227.
- NVIDIA et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*
  arXiv:2503.14734 (GR00T X-Embodiment-Sim, CC BY 4.0).
- Stone Tao et al. (23 authors). *ManiSkill3: GPU Parallelized Robotics Simulation and
  Rendering for Generalizable Embodied AI.* arXiv:2410.00425.
- Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu, Zhanxin Wu, K. Fay, Henrik I.
  Christensen, Jan Peters, and Hao Su. *Towards Embodiment Scaling Laws in Robot
  Locomotion.* CoRL 2025, arXiv:2505.05753 (GenBot-1K, BSD-3-Clause).

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
- Minho Heo, Youngwoon Lee, Doohyun Lee, and Joseph J. Lim. *FurnitureBench: Reproducible
  Real-World Benchmark for Long-Horizon Complex Manipulation.* RSS 2023, arXiv:2305.12821.
- Ge Yan, Kris Wu, and Xiaolong Wang. *UCSD Kitchen.* 2023 (`lerobot/ucsd_kitchen_dataset`,
  MIT).
- Daniel Sliwowski, Shail Jadav, Sergej Stanovcic, Jędrzej Orbik, Johannes Heidersberger, and
  Dongheui Lee. *Demonstrating REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and
  Disassembly.* RSS 2025. DOI
  [10.15607/RSS.2025.XXI.059](https://doi.org/10.15607/RSS.2025.XXI.059) (CC BY 4.0).
- Claudio Gaz, Marco Cognetti, Alexander Oliva, Paolo Robuffo Giordano, and Alessandro De Luca.
  *Dynamic Identification of the Franka Emika Panda Robot With Retrieval of Feasible
  Parameters Using Penalty-Based Optimization.* IEEE RA-L 4(4):4147–4154, 2019. DOI
  [10.1109/LRA.2019.2931248](https://doi.org/10.1109/LRA.2019.2931248).
- Zipeng Fu, Tony Z. Zhao, and Chelsea Finn. *Mobile ALOHA: Learning Bimanual Mobile
  Manipulation with Low-Cost Whole-Body Teleoperation.* arXiv:2401.02117, 2024.
- Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Chenxi Wang, Junbo Wang, Haoyi Zhu, and
  Cewu Lu. *RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot.*
  ICRA 2024, arXiv:2307.00595.
- Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X
  Models.* ICRA 2024, arXiv:2310.08864. The individual datasets in the tables are cited
  there by their first authors and venues; their papers: toto arXiv:2306.00942, MUTEX
  arXiv:2309.14320, Sirius arXiv:2211.08416, BUDS arXiv:2109.13841, RPT arXiv:2306.10007,
  NYU Franka play arXiv:2210.10047, Play Fusion arXiv:2312.04549, FMB arXiv:2401.08553.
- Michelle A. Lee, Yuke Zhu, Krishnan Srinivasan, Parth Shah, Silvio Savarese, Li Fei-Fei,
  Animesh Garg, and Jeannette Bohg. *Making Sense of Vision and Touch: Self-Supervised
  Learning of Multimodal Representations for Contact-Rich Tasks.* ICRA 2019 (Stanford KUKA
  multimodal).
- AgiBot World Contributors et al. *AgiBot World Colosseo: A Large-scale Manipulation Platform
  for Scalable and Intelligent Embodied Systems.* arXiv:2503.06669 (AgiBot World;
  2026 release CC BY-NC-SA 4.0).
- Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng,
  Jiyang Gao, Huazhe Xu, and Hang Zhao. *Galaxea Open-World Dataset and G0 Dual-System VLA
  Model.* arXiv:2509.00576.
- Trossen Robotics. *Trossen AI datasets.* <https://huggingface.co/TrossenRoboticsCommunity>
  (Apache-2.0).
