# Graph Report - DA-GET-BFM  (2026-11-17)

## Corpus Check
- 80 files · ~120,774 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2374 nodes · 5144 edges · 132 communities (113 shown, 19 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 240 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output recorded; semantic subagent usage was unavailable

## Community Hubs (Navigation)
- Dynamics
- Training Episodes
- Training Sources
- Physics
- Bfm
- RL Buffers
- Encoder Docs
- RL Trainer
- Training Ground Sim
- Da Get
- Pretraining
- Embodiment
- Pretraining
- Go1 URDF
- Da Get Encoder Tests
- Conversion Datasets
- Embodiment Tests
- Robot Analysis
- Fb Cpr Tests
- Anymal D URDF
- Pretraining Tests
- G1 29Dof URDF
- Urdf
- RL Agent
- RL Agent
- RL Envs
- Fb Cpr Tests
- Conversion Mapping
- H1 URDF
- RL Agent
- Bfm
- RL Envs
- Da Get
- RL Isaac Env
- Fb Cpr Tests
- Deployment Tests
- Conversion Common
- RL Observation
- Conversion Tests
- Isaac Env Tests
- Bfm Tests
- Conversion Zarr Lite
- Embodiment
- Bfm
- Isaac Env Tests
- Anymal D URDF
- RL Deployment
- Anymal D URDF
- RL Agent
- Anymal D URDF
- Go1 URDF
- Bfm
- Readme
- Quadrotor X URDF
- Anymal D URDF
- G1 29Dof URDF
- RL Evaluation
- Training Trainer
- Anymal D URDF
- Anymal D URDF
- RL Deployment
- Bfm Docs
- RL Isaac Env
- Training Dataset
- H1 URDF
- Go1 URDF
- RL Trainer
- Bfm Tests
- Embodiment Tests
- Conversion Common
- Conversion Common
- RL Observation
- Fb Cpr Tests
- Anymal D URDF
- Bfm Docs
- RL Envs
- Transformer
- Validate Torque Estimation
- Bfm Tests
- Fb Cpr Tests
- H1 URDF
- RL Envs
- RL Isaac Env
- Transformer
- Readme
- Isaac Env Tests
- Physics Tests
- Anymal D URDF
- Anymal D URDF
- Physics Schema
- RL Isaac Env
- Go1 URDF
- Go1 URDF
- Benchmark Inference
- Benchmark Inference
- Anymal D URDF
- G1 29Dof URDF
- Go1 URDF
- Ground Contact
- G1 Go1 Real And Synthetic
- Bfm
- Bfm
- Try
- Anymal D URDF
- G1 Hub Plus Synthetic
- Synthetic Unitree
- Configuration Bfm
- Embodiment
- RL Buffers
- Convert Dataset
- Download Urdfs
- Embodiment Tests
- Fb Cpr Tests
- Go1 H1 Ground
- Training Docs
- Tokenization Da Get
- Readme
- Isaac Env Tests
- Data Docs
- Conversion Init
- Pyproject
- Bfm Docs
- Training Docs
- Training Docs
- .Gitignore
- Pyproject
- Readme
- Readme
- Readme
- Readme
- Readme
- Readme

## God Nodes (most connected - your core abstractions)
1. `anymal` - 156 edges
2. `g1_29dof` - 87 edges
3. `go1` - 84 edges
4. `EmbodimentGraph` - 71 edges
5. `EmbodimentRecord` - 64 edges
6. `h1_description` - 52 edges
7. `DaGetFbCprAgent` - 49 edges
8. `DaGetConfig` - 48 edges
9. `quadruped()` - 46 edges
10. `EmbodimentObserver` - 45 edges

## Surprising Connections (you probably didn't know these)
- `ground_contact source 1: mujoco` --uses_urdf--> `anymal`  [EXTRACTED]
  configs/data/ground_contact.json → assets/urdf/anymal_d.urdf
- `g1_hub_plus_synthetic source 0: huggingface` --uses_urdf--> `g1_29dof`  [EXTRACTED]
  configs/data/g1_hub_plus_synthetic.json → assets/urdf/g1_29dof.urdf
- `g1_hub_plus_synthetic source 1: synthetic` --uses_urdf--> `g1_29dof`  [EXTRACTED]
  configs/data/g1_hub_plus_synthetic.json → assets/urdf/g1_29dof.urdf
- `ground_contact source 2: mujoco` --uses_urdf--> `g1_29dof`  [EXTRACTED]
  configs/data/ground_contact.json → assets/urdf/g1_29dof.urdf
- `synthetic_unitree source 2: synthetic` --uses_urdf--> `g1_29dof`  [EXTRACTED]
  configs/data/synthetic_unitree.json → assets/urdf/g1_29dof.urdf

## Import Cycles
- None detected.

## Communities (132 total, 19 thin omitted)

### Community 0 - "Dynamics"
Cohesion: 0.04
Nodes (99): BaseMotion, _body_wrench(), _check_joint_tensor(), contact_maps(), coriolis_efforts(), determinate_torque_projector(), distribute_contact_forces(), EffortEstimate (+91 more)

### Community 1 - "Training Episodes"
Cohesion: 0.04
Nodes (84): Dataset, DomainRandomization, BFM-Zero's randomization ranges (paper Table 4); ``enabled=False`` turns all of…, EmbodimentObserver, Observations of the cross-embodiment FB-CPR agent, built the same way from…, Build policy and state views of one robot. Args: record (EmbodimentRecord):…, Return the policy encoder's input width., Return the width of the state tokens. (+76 more)

### Community 2 - "Training Sources"
Cohesion: 0.04
Nodes (79): quaternion_to_matrix(), Convert unit quaternions to rotation matrices. Args: quaternion (torch.Tensor):…, JointFeatureConfig, Return every feature, for robots with differing axes and dynamics. Returns:…, Select the fixed per-joint hardware features read from URDF. The default…, Reject an unknown rotation representation. Returns: None: The configuration is…, Return the paper's hardware features: joint origin and rotation. Returns:…, actuator_specs() (+71 more)

### Community 3 - "Physics"
Cohesion: 0.06
Nodes (57): default_gravity(), device, dtype, Return gravity along -z of the base frame. Args: dtype (torch.dtype): Floating…, actuator_features(), actuator_tensors(), ActuatorDefaults, ActuatorSpec (+49 more)

### Community 4 - "Bfm"
Cohesion: 0.06
Nodes (48): BfmConfig, Configuration for the GET-conditioned behavior foundation model (FB-CPR…, Sizes of the BFM-Zero networks, all of which read per-DoF tokens instead of a…, Reusable transformer components and graph embodiment encoder models., DaGetBackwardMap, DaGetBfmActor, DaGetDiscriminator, DaGetForwardMap (+40 more)

### Community 5 - "RL Buffers"
Cohesion: 0.06
Nodes (33): Cross-embodiment FB-CPR agent: the BFM-Zero update on per-DoF networks. The…, ExpertMotions, _pack_mass(), device, Tensor, Per-robot replay and expert-motion buffers of the cross-embodiment FB-CPR agent., Keep the upper triangle of symmetric matrices ``[N, J, J]``., Rebuild symmetric matrices from ``_pack_mass``. (+25 more)

### Community 6 - "Encoder Docs"
Cohesion: 0.04
Nodes (49): Behavior foundation model, Configuration, Contents, Design details, hidden, batch and goal from the README quick start, with Go1 as the prompted robot, Training (online off-policy FB-CPR), Usage, Converting datasets (+41 more)

### Community 7 - "RL Trainer"
Cohesion: 0.07
Nodes (34): _ArgumentGroup, ArgumentParser, FbCprConfig, Reject mixture ratios that cannot be sampled., Hyperparameters of ``DaGetFbCprAgent``; defaults are BFM-Zero's G1 values.…, build_group(), FbCprTrainer, Any (+26 more)

### Community 8 - "Training Ground Sim"
Cohesion: 0.08
Nodes (33): EnvConfig, Parallel environments of one robot, with one implementation per simulator.…, Settings of an environment. Attributes: num_envs (int): Parallel worlds.…, build_isaac_group(), IsaacRobotConfig, Worlds in Isaac Lab: several robots in one GPU simulation, with body and self…, Create the scene for every robot and return its environments. Call this inside…, How one robot enters the simulation. Attributes: urdf (str): URDF with… (+25 more)

### Community 9 - "Da Get"
Cohesion: 0.08
Nodes (25): DaGetConfig, Configuration for the graph embodiment transformer encoder., Reject invalid dimensions and layer locations. Returns: None: The frozen…, Hold validated dimensions and graph encoding options for ``DaGetEncoder``.…, Return the layer indices with Physics Cross Attention. Returns: tuple[int,…, DaGetEncoderLayer, DaGetGraphAttentionBias, graph_distances() (+17 more)

### Community 10 - "Pretraining"
Cohesion: 0.08
Nodes (29): Build a validated encoder configuration from a named architecture size. Args:…, DaGetForPhysicsPretraining, DaGetPhysicsPretrainingConfig, Any, Module, Store settings; see the class attributes., Build from a ``DaGetConfig``. Args: encoder (DaGetConfig): Encoder…, GET encoder with the auxiliary physics heads A-F. ``forward`` takes the batch… (+21 more)

### Community 11 - "Embodiment"
Cohesion: 0.07
Nodes (22): EmbodimentBatch, device, Tensor, Return the structural keyword arguments for ``DaGetEncoder.forward``. Returns:…, Move all tensors to a device. Args: device (torch.device | str): Target device.…, Compute padded joint-space mass matrices for every batch item. Args: positions…, Compute padded joint frames for every batch item. Args: positions…, Normalize padded joint positions by each robot's limits. Args: positions… (+14 more)

### Community 12 - "Pretraining"
Cohesion: 0.08
Nodes (26): DaGetPhysicsPretrainingOutput, geodesic_distance(), GraphActionMixer, masked_joint_mean(), _mlp(), Tensor, Pretrain the GET encoder with auxiliary physics heads (Hugging Face…, Outputs of ``DaGetForPhysicsPretraining``. Attributes: loss (torch.Tensor |… (+18 more)

### Community 13 - "Go1 URDF"
Cohesion: 0.11
Nodes (34): camera_joint_chin, camera_joint_face, camera_joint_rearDown, camera_optical_joint_face, camera_optical_joint_rearDown, FL_hip_rotor_joint, floating_base, FR_hip_rotor_joint (+26 more)

### Community 14 - "Da Get Encoder Tests"
Cohesion: 0.08
Nodes (20): DaGetEncoder, PhysicsMemory, Cached keys and values of the embodiment-only physics tokens. Rigid-body and…, Encode variable-length robot joint tokens with graph and dynamics biases.…, Return the Physics Cross Attention blocks in layer order., Return versions of every parameter a ``PhysicsMemory`` depends on., Validate one physics feature tensor., Precompute the embodiment-only part of Physics Cross Attention. Call this once… (+12 more)

### Community 15 - "Conversion Datasets"
Cohesion: 0.15
Nodes (32): Any, Path, Return timestamps from start to stop at a fixed rate., Split a time series at gaps longer than ``max_gap`` seconds. Returns:…, Write episodes as a LeRobot-style parquet table with ``meta/info.json``. Args:…, Build and save a ``huggingface`` data-config source for a converted dataset.…, source_config(), split_segments() (+24 more)

### Community 16 - "Embodiment Tests"
Cohesion: 0.08
Nodes (20): Path, Build an embodiment graph with one token per URDF degree of freedom. Args:…, double_pendulum(), EmbodimentGraphTest, KinematicsTest, Check per-DoF graphs for different robot morphologies., Fixed links vanish and the four hips are joined as chain roots., Mimic joints are frozen and floating joints are skipped or expanded. (+12 more)

### Community 17 - "Robot Analysis"
Cohesion: 0.09
Nodes (29): analyze_robot(), describe(), Any, Path, Tensor, Infer data-config settings from a URDF: base type, feet, rotors, and actuators.…, Size propeller coefficients from mass, thrust-to-weight, and rotor speed., Suggest PD gains for position-controlled joints from their effort limits. ``K_p… (+21 more)

### Community 18 - "Fb Cpr Tests"
Cohesion: 0.10
Nodes (21): AgentTest, changed(), physics_batch(), Module, Path, Tensor, Copy a module's parameters., Return whether any parameter differs from a snapshot. (+13 more)

### Community 19 - "Anymal D URDF"
Cohesion: 0.12
Nodes (31): base_to_depth_camera_left_camera, base_to_inspection_payload_mount, depth_camera_left_camera_depth_to_depth_optical_frame, depth_camera_left_camera_parent_to_depth_frame, depth_camera_rear_upper_camera_depth_to_depth_optical_frame, depth_camera_rear_upper_camera_parent_to_depth_frame, face_rear_to_depth_camera_rear_upper_camera, inspection_payload_head_to_camera (+23 more)

### Community 20 - "Pretraining Tests"
Cohesion: 0.09
Nodes (19): PretrainingPipelineTest, Path, Remove temporary files., Return a small pretraining model for the test data., Collate samples from both embodiments., Both sources load; recorded joints follow joint_names; the cache is reused., Padding, masks, and normalized targets are consistent., All six losses are computed and a few steps of Adam reduce the total. (+11 more)

### Community 21 - "G1 29Dof URDF"
Cohesion: 0.12
Nodes (29): left_ankle_pitch_joint, left_ankle_roll_joint, left_hand_palm_joint, left_wrist_pitch_joint, left_wrist_roll_joint, left_wrist_yaw_joint, right_ankle_pitch_joint, right_ankle_roll_joint (+21 more)

### Community 22 - "Urdf"
Cohesion: 0.11
Nodes (29): Element, _frozen_motion(), Return the motion of a joint that is not a token. Joints in ``frozen_joints``…, axis_angle_to_matrix(), _floats(), make_transform(), _optional_float(), _origin() (+21 more)

### Community 23 - "RL Agent"
Cohesion: 0.10
Nodes (17): DaGetFbCprAgent, no_grad, Path, FB-CPR with auxiliary rewards, updating ``DaGetBfmModel`` and the encoder…, Build privileged inputs for the shared encoder on the agent device. Args: robot…, Return per-DoF actions ``[N, J]`` for one robot's environments., Encode one robot's state and evaluate B in bounded chunks. Args: robot (int):…, Return ``z_t`` for a motion ``[T, ...]``: the projected mean of B over its next… (+9 more)

### Community 24 - "RL Agent"
Cohesion: 0.09
Nodes (20): EmaRewardNormalizer, fb_bellman_loss(), _off_diagonal(), orthonormality_loss(), device, Ring buffer of latents used in training, sampled for rollouts (one per robot)., Append latents ``[N, Z]``, keeping the newest when ``N`` exceeds the capacity., Return ``count`` stored latents. (+12 more)

### Community 25 - "RL Envs"
Cohesion: 0.11
Nodes (12): EmbodimentEnv, ndarray, Simulator-independent logic of one robot's worlds. Simulators implement…, Return one world's nominal pose, with its random joint offset., Set the motions episodes start from, with optional sampling weights., Set how often each motion starts an episode (prioritization by tracking error)., Turn a motion frame by a quarter turn about x, onto the back or the front (BFM-…, Sample a start frame: a prioritized motion, a uniform time, maybe lying down. (+4 more)

### Community 26 - "Fb Cpr Tests"
Cohesion: 0.10
Nodes (15): EnvironmentTest, skipUnless, The MuJoCo environments and their agreement with the data pipeline., Return the environment of a standing quadruped and its robot., A world reset to an expert frame has that frame's state tokens (height aside)., Zero actions keep a standing robot up for the whole episode, then the time…, About 30% of starts lie on the back or front, above the floor, and count as…, As in BFM-Zero, only time ends an episode; a fallen robot stays above the floor… (+7 more)

### Community 27 - "Conversion Mapping"
Cohesion: 0.13
Nodes (22): _find_vector(), inspect_dataset(), load_info(), match_joints(), match_score(), _names(), Any, Path (+14 more)

### Community 28 - "H1 URDF"
Cohesion: 0.14
Nodes (25): d435_left_imager_joint, d435_rgb_module_joint, left_ankle_joint, left_elbow_joint, left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint, right_ankle_joint (+17 more)

### Community 29 - "RL Agent"
Cohesion: 0.14
Nodes (15): ensemble_pessimism(), Tensor, Update the statistics and return the scaled rewards., One training batch, with rows in robot order. Attributes: robots…, Encode one robot's policy observation with the selected shared core. Args:…, Encode each robot with one core and pad outputs to a common joint count. Args:…, Return the mean, the mean pairwise distance, and ``mean - penalty * distance``…, Sample a TD3-style clipped action. (+7 more)

### Community 30 - "Bfm"
Cohesion: 0.09
Nodes (13): device, Tensor, Return ``[batch, 2 * hidden]`` from ``[batch, joints, hidden]`` and a bool mask., Map shared-encoder embeddings to ``[batch, z_dim]``. Args: joint_embeddings…, Return logits from pooled state features ``[batch, output_size]``., Return expert logits from encoded state tokens and latent prompts. Args:…, Return expert probabilities ``[batch, 1]`` from encoded state tokens. Args:…, Build memory tokens. Args: z (torch.Tensor): ``[batch, z_dim]`` or a latent… (+5 more)

### Community 31 - "RL Envs"
Cohesion: 0.13
Nodes (9): MujocoEmbodimentEnv, Tensor, Read the results of the physics advance., Worlds in MuJoCo, built from the embodiment graph. Links collide with the floor…, Return the base height that puts the lowest of some geoms on the floor., Return the base height of the nominal pose standing on the floor., Put a world in a frame (the nominal pose if None), resting on the floor., Run the physics of one control step. (+1 more)

### Community 32 - "Da Get"
Cohesion: 0.12
Nodes (16): normalize_mass_matrix(), PhysicsCrossAttention, dtype, Tensor, Map shortest-path distances to embedding buckets. A shortest path over J joints…, Let observation joint tokens attend to the physics tokens of every joint. The…, Return the parameters that cached keys and values depend on. Returns:…, Reshape ``[B, N, hidden]`` to ``[B, heads, N, head_size]``. (+8 more)

### Community 33 - "RL Isaac Env"
Cohesion: 0.14
Nodes (11): Return the summed speed of feet in contact (BFM-Zero's ``penalty_slippage``)., Return the summed ``|g_xy|`` in the frame of each flat foot in contact (BFM-…, IsaacEmbodimentEnv, ndarray, Tensor, Queue a world's reset; ``_after_reset`` writes the batch., Apply targets, then write this step's pushes as one batch of velocity changes., Return which worlds have a contact above 1 N on a penalized link, with the… (+3 more)

### Community 34 - "Fb Cpr Tests"
Cohesion: 0.12
Nodes (14): project_to_sphere(), Scale the last dimension to norm ``sqrt(dim)`` (BFM-Zero's z projection)., build_agent(), observe(), The agent gives the FB and orthonormality losses the same-robot mask, unless…, Return the observation of a state at the start of an episode., A transition never gets a goal or a motion of another robot., An expert latent is the projected mean of B over its window, per robot. (+6 more)

### Community 35 - "Deployment Tests"
Cohesion: 0.10
Nodes (13): DeploymentTest, HostCallRecorder, skipUnless, Tensor, Asking for CUDA Graphs on a CPU is an error., A replayed graph gives the eager actions, and follows new inputs., Record calls that a CUDA Graph capture would reject, with the model line that…, The runner matches ``DaGetFbCprAgent`` and records nothing a CUDA Graph rejects. (+5 more)

### Community 36 - "Conversion Common"
Cohesion: 0.13
Nodes (19): ndarray, quaternion_wxyz_to_xyzw(), Shared tools for converting robot datasets into the training table format.…, Reorder quaternions from (w, x, y, z) to (x, y, z, w)., Return world-from-base rotation matrices of (x, y, z, w) quaternions., Rotate world-frame vectors into the base frame., Linearly interpolate ``[N, D]`` values onto new timestamps., Interpolate quaternions (any component order) with sign continuity. (+11 more)

### Community 37 - "RL Observation"
Cohesion: 0.15
Nodes (14): Resample motion frames from period ``dt`` to the control period; rotations use…, resample_state(), Tensor, Return the view of the true state read by B, D and the critics., Return the mass matrix and efforts at a state., Return dynamics features; ``source`` is the torque-source flag (0 means…, Assemble the stored observation. Args: local_history (torch.Tensor): ``[N, H,…, Return what the policy encoder reads besides its tokens, as at deployment.… (+6 more)

### Community 38 - "Conversion Tests"
Cohesion: 0.14
Nodes (12): ConverterTest, ndarray, Path, skipUnless, MuJoCo free-root qpos/qvel become base pose and base-frame twist., Kinematic humanoid motions convert with G1 joint names., MIPO joints are renamed to Go1 URDF joints and contacts thresholded., Write one uncompressed zarr v2 array. (+4 more)

### Community 39 - "Isaac Env Tests"
Cohesion: 0.13
Nodes (12): AdapterTest, build(), Return the adapter on fakes, with its robot., Bookkeeping between the graph and the simulator., Poses, velocities and joints go to the right worlds, frames and joint slots., A standing start rests the feet on the floor; a lying start clears every frame., Actions become PD targets on the right joints; joints outside the graph hold…, Joint signals return in graph order, and height comes from the root position. (+4 more)

### Community 40 - "Bfm Tests"
Cohesion: 0.11
Nodes (11): Build independently sized actor, forward/critic, and backward networks. Args:…, BfmNetworksTest, Check all size levels and explicit legacy and role-specific overrides. Returns:…, The actor gives per-DoF actions in [-1, 1]; F and critics follow BFM-Zero's…, Actor parameters stay separate from the forward and scalar critic maps.…, Extra padded joints change neither the valid actions nor F., Permuting joints permutes actions and leaves F unchanged., Different z, latent sequences and per-joint context tokens change the outputs. (+3 more)

### Community 41 - "Conversion Zarr Lite"
Cohesion: 0.14
Nodes (10): _codec(), ndarray, Path, Read zarr v2 arrays from a directory or a tar archive without the zarr package.…, Read files from a directory or from members of a tar archive., A read-only zarr v2 group. Args: path (str | Path): Directory or ``.tar`` file…, Return the group's ``.zattrs``., Return the names of arrays directly in the group. (+2 more)

### Community 42 - "Embodiment"
Cohesion: 0.13
Nodes (19): _Body, _center_inertia(), _collect_tokens(), _joint_features(), _limits(), Build GET-Zero embodiment graphs from URDF files, with one token per DoF. The…, Collect one DoF while traversing the URDF tree., Accumulate the mass properties of the root (base) body. (+11 more)

### Community 43 - "Bfm"
Cohesion: 0.18
Nodes (13): DaGetBfmModel, no_grad, FB-CPR model (BFM-Zero's ``FBcprAuxModel``) with GET-conditioned actor and…, Encode privileged state tokens with the shared encoder core. A diagonal inertia…, Project z to the sphere when ``config.norm_z`` is set., Return per-DoF actions in [-1, 1] (the mean, or a sample with ``actor_std``)., Goal reaching: ``z = B(s_goal)``., Reward optimization: ``z = sum_i w_i r_i B(s_i)`` over one robot's buffer… (+5 more)

### Community 44 - "Isaac Env Tests"
Cohesion: 0.12
Nodes (6): SimpleNamespace, FakeArticulation, proxy(), Tensor, Mimic Isaac Lab's ``ProxyArray``., Record what the adapter writes, with joints and bodies in simulator order.

### Community 45 - "Anymal D URDF"
Cohesion: 0.12
Nodes (18): base_battery, base_bottom_shell, base_hbc_receiver, base_LF_HAA_drive, base_RF_HAA_drive, base_to_base_inertia, base_to_body_top, base_to_docking_socket (+10 more)

### Community 46 - "RL Deployment"
Cohesion: 0.18
Nodes (11): DaGetBfmPolicy, no_grad, Tensor, Return the static encoder inputs: graph structure and physics memory., Encoder and actor mean on the static inputs; what a graph records., State encoder and B on the static state inputs; what a graph records., Return ``function``, or a replay of it recorded as a CUDA Graph., Validate the mass matrix on the CPU (the graph runs with checks off). (+3 more)

### Community 47 - "Anymal D URDF"
Cohesion: 0.14
Nodes (17): LF_HFE, LF_HFE_drive_LF_thigh_fixed, LF_hip_fixed_LF_HFE_output, LF_thigh_fixed_LF_KFE_drive, LF_THIGH_to_LF_HFE_drive, RF_HFE, RF_HFE_drive_RF_thigh_fixed, RF_KFE (+9 more)

### Community 48 - "RL Agent"
Cohesion: 0.12
Nodes (12): Module, Sample ``counts`` transitions of each robot and join them in robot order., Return D's logits ``[B, 1]`` of per-robot state inputs., Return the WGAN gradient penalty of D's head, between expert and replay samples…, Train D to separate expert windows from replay states, each with its z., Run one gradient step of every network, in BFM-Zero's order, then the target…, Move target parameters toward the source by ``tau``., Split ``total`` into ``groups`` near-equal integers. (+4 more)

### Community 49 - "Anymal D URDF"
Cohesion: 0.15
Nodes (16): LH_HFE, LH_HFE_drive_LH_thigh_fixed, LH_KFE, LH_thigh_fixed_LH_KFE_drive, LH_THIGH_to_LH_HFE_drive, RH_HFE, RH_HFE_drive_RH_thigh_fixed, RH_thigh_fixed_RH_KFE_drive (+8 more)

### Community 50 - "Go1 URDF"
Cohesion: 0.16
Nodes (15): FR_calf_joint, FR_calf_rotor_joint, FR_hip_joint, FR_thigh_joint, FR_thigh_rotor_joint, RR_calf_rotor_joint, RR_hip_joint, RR_thigh_rotor_joint (+7 more)

### Community 51 - "Bfm"
Cohesion: 0.13
Nodes (10): _joint_mask(), Return the CPR log-odds reward from encoded state tokens. Args:…, Validate embeddings ``[batch, joints, E]`` and default the mask to all joints., Return the action distribution. Args: joint_embeddings (torch.Tensor):…, Evaluate every ensemble member. Args: joint_embeddings (torch.Tensor):…, Normal policy clamped to [low, high] with straight-through gradients (BFM-…, TruncatedNormal, Test the per-DoF BFM-Zero networks. (+2 more)

### Community 52 - "Readme"
Cohesion: 0.13
Nodes (15): Attention, Data tools, Embodiment graph from URDF, GET encoder, Physics Cross Attention and pretraining, Pipeline, Tokens: one per DoF, behavior cloning distillation (+7 more)

### Community 53 - "Quadrotor X URDF"
Cohesion: 0.25
Nodes (13): rotor_back_left_joint, rotor_back_right_joint, rotor_front_left_joint, rotor_front_right_joint, base_link, rotor_back_left, rotor_back_right, rotor_front_left (+5 more)

### Community 54 - "Anymal D URDF"
Cohesion: 0.15
Nodes (14): base_LH_HAA_drive, LF_HAA, LH_HAA, LH_hip_fixed_LH_HFE_output, LH_HIP_LH_hip_fixed, RF_HAA, RH_HAA, RH_HIP_RH_hip_fixed (+6 more)

### Community 55 - "G1 29Dof URDF"
Cohesion: 0.14
Nodes (14): d435_joint, head_joint, imu_in_torso_joint, waist_support_joint, d435_link, head_link, imu_in_torso, waist_support_link (+6 more)

### Community 56 - "RL Evaluation"
Cohesion: 0.20
Nodes (12): earth_movers_distance(), evaluate(), motion_priorities(), ndarray, no_grad, Evaluation of the cross-embodiment FB-CPR agent and the motion priorities it…, Return the EMD between equally weighted frame sets ``[T, J]`` of the same…, Return BFM-Zero's sampling weights: ``2 ** (scale * clip(EMD, minimum,… (+4 more)

### Community 57 - "Training Trainer"
Cohesion: 0.15
Nodes (10): PhysicsPretrainingTrainer, Any, Module, Tensor, Hugging Face ``Trainer`` that logs every auxiliary loss separately., Trainer for ``DaGetForPhysicsPretraining``. Besides the weighted total…, Create the trainer; arguments are those of ``transformers.Trainer``., Run the model and record its per-task losses. (+2 more)

### Community 58 - "Anymal D URDF"
Cohesion: 0.17
Nodes (13): base_face_front, base_face_shell_front, depth_camera_front_upper_camera_depth_to_depth_optical_frame, depth_camera_front_upper_camera_parent_to_depth_frame, face_front_to_depth_camera_front_upper_camera, face_front_to_wide_angle_camera_front_camera, depth_camera_front_upper_camera, depth_camera_front_upper_camera_parent (+5 more)

### Community 59 - "Anymal D URDF"
Cohesion: 0.17
Nodes (13): base_face_rear, base_face_shell_rear, depth_camera_rear_lower_camera_depth_to_depth_optical_frame, depth_camera_rear_lower_camera_parent_to_depth_frame, face_rear_to_depth_camera_rear_lower_camera, face_rear_to_wide_angle_camera_rear_camera, depth_camera_rear_lower_camera, depth_camera_rear_lower_camera_parent (+5 more)

### Community 60 - "RL Deployment"
Cohesion: 0.18
Nodes (7): device, Run a trained DA-GET BFM on one robot at control rate. ``DaGetBfmPolicy`` holds…, Rebuild the embodiment-only physics memory and drop recorded graphs (after a…, Named views of one flat device buffer, filled through one copy from pinned host…, Copy CPU tensors into the device buffer (every input must be given)., StaticInputs, Test the deployment runner ``DaGetBfmPolicy`` against the training path.

### Community 61 - "Bfm Docs"
Cohesion: 0.17
Nodes (12): 2. Train the behavior foundation model, inside Isaac Lab's Python environment, Attribution, Data config: real and synthetic trajectories, First run on a GPU machine, How one training step works, Inference latency at 50 Hz, Open questions for cross-embodiment training, Sharing z across robots (+4 more)

### Community 62 - "RL Isaac Env"
Cohesion: 0.17
Nodes (8): matrix_to_xyzw(), Return base heights that rest the feet on the floor, or drop a lying robot from…, Write the queued resets to the simulator in one batch., Convert rotation matrices ``[N, 3, 3]`` to Isaac Lab 3's x, y, z, w quaternions…, QuaternionTest, Isaac Lab 3 uses x, y, z, w quaternions., Every branch of the conversion gives the rotation back., Rotating into a frame equals applying the transposed matrix.

### Community 63 - "Training Dataset"
Cohesion: 0.18
Nodes (8): _pad(), Any, Tensor, Describe the observation token layout for the model config., Build the sample index. Args: records (Sequence[EmbodimentRecord]):…, Return one unpadded sample., Stack tensors whose first ``dims`` dimensions are joints, padding to…, Collate samples. Args: samples (Sequence[dict]): Dataset items. Returns:…

### Community 64 - "H1 URDF"
Cohesion: 0.24
Nodes (11): imu_in_pelvis_joint, pelvis_contour_joint, imu_in_pelvis, pelvis_contour_link, left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint, left_hip_pitch_link (+3 more)

### Community 65 - "Go1 URDF"
Cohesion: 0.22
Nodes (11): FL_calf_joint, FL_calf_rotor_joint, FL_thigh_joint, FR_foot_fixed, RL_calf_rotor_joint, FL_calf, FL_calf_rotor, FL_foot (+3 more)

### Community 66 - "RL Trainer"
Cohesion: 0.18
Nodes (7): EnvStep, Result of one control step, before any reset. Attributes: state (RobotState):…, Tensor, Return the state the policy senses., Reset every environment and build the first observations., Collect the environment step; return the next observation before any reset., Set the observations to continue from, resetting finished environments.

### Community 67 - "Bfm Tests"
Cohesion: 0.18
Nodes (7): config(), EncoderToActorTest, Return a small config for fast tests., Several action entries per DoF and a single-member ensemble., Goal, reward (chunked) and tracking inference match their definitions., The actor and F train end to end through the encoder., The FB actor loss -Fᵀz can backpropagate into the encoder.

### Community 68 - "Embodiment Tests"
Cohesion: 0.20
Nodes (6): Return the GET-Zero LEAP Hand architecture for a token width. The paper's model…, Run several robot morphologies through one encoder., Tokens follow [o_vg, o_fg, o_vl, o_fl] with feature-major history., Disconnected chains encode the same whether or not a batch is padded., The paper preset uses the published architecture sizes., TokenizationAndEncoderTest

### Community 69 - "Conversion Common"
Cohesion: 0.22
Nodes (8): chunk_episode(), Episode, Cut an episode into pieces of at most ``frames`` frames., One episode of converted signals, all ``[T, ...]`` arrays at a common rate.…, Return the number of frames., The URDF analysis recognizes base types, feet, and rotors., The config script combines robot analysis, a dataset, and synthetic data., RobotAnalysisTest

### Community 71 - "RL Observation"
Cohesion: 0.27
Nodes (5): device, Return the robot's per-sample encoder inputs on a device, with a batch of one., Return the graph and hardware inputs of the encoder for ``count`` samples., Return policy tokens ``[N, J, F]``, as physics pretraining builds them., Build privileged tokens and graph inputs for the shared encoder. Args: view…

### Community 72 - "Fb Cpr Tests"
Cohesion: 0.20
Nodes (6): random_state(), Local features hold position, velocity and frame position; global ones the base…, B and D cannot see the last torque, which motion data does not have; the policy…, Noise reaches the policy's features only: B, D and critics read the true state., When motions lack height or linear velocity, simulated states drop them too., Return a random fixed-base state.

### Community 73 - "Anymal D URDF"
Cohesion: 0.22
Nodes (9): LH_shank_fixed_LH_FOOT, RF_shank_fixed_RF_FOOT, RH_KFE, RH_shank_fixed_RH_FOOT, LH_FOOT, RF_FOOT, RF_shank_fixed, RH_FOOT (+1 more)

### Community 74 - "Bfm Docs"
Cohesion: 0.44
Nodes (9): #, #, #, #, #, #, #, # (+1 more)

### Community 75 - "RL Envs"
Cohesion: 0.22
Nodes (5): MujocoGroup, ThreadPoolExecutor, Look up the MuJoCo addresses of joints, actuators and collision geoms., Return a copy of the model with randomized friction, link masses and base…, Step several robots' MuJoCo worlds; each advances on its own.

### Community 76 - "Transformer"
Cohesion: 0.22
Nodes (5): Tensor, Return sine on even dimensions and cosine on odd dimensions. Args: position_ids…, Attend to tokens in the input sequence and add the residual input. Args:…, Use query tokens to attend over context keys and values. Args: hidden_states…, Convert each numerical vector into one token embedding. Args: values…

### Community 77 - "Validate Torque Estimation"
Cohesion: 0.25
Nodes (9): load(), main(), ndarray, Path, Tensor, Download a sequence and compare the estimates., Moving average along time with edge padding., Read one MIPO sequence and align the IMU to the joint timestamps. (+1 more)

### Community 78 - "Bfm Tests"
Cohesion: 0.22
Nodes (6): Tensor, One B and one D read robots of different sizes; B stays on the sphere; rewards…, Without graph edges, reordering joints changes neither B nor D., Return random state-encoder inputs of a robot without graph edges, optionally…, Check one encoder instance feeds B, D, actor, and F without reinitialization.…, state_inputs()

### Community 79 - "Fb Cpr Tests"
Cohesion: 0.22
Nodes (6): fixed_record(), ObservationTest, State views, buffers and motions., Windows come mostly from high-priority motions, and from consecutive frames., Return a fixed-base robot in position mode., A robot without a floating base gets no free joint and follows its targets.

### Community 80 - "H1 URDF"
Cohesion: 0.36
Nodes (8): right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint, right_knee_joint, right_hip_pitch_link, right_hip_roll_link, right_hip_yaw_link, right_knee_link

### Community 81 - "RL Envs"
Cohesion: 0.25
Nodes (3): Turn normalized actions ``[N, J]`` in [-1, 1] into PD targets and apply pushes., Apply actions and advance one control step., Run the physics of one control step for every robot.

### Community 82 - "RL Isaac Env"
Cohesion: 0.32
Nodes (4): IsaacGroup, Step every robot of one Isaac Lab simulation together. Args: simulation (Any):…, Recompute simulator state after writing resets or pushes., Every physics step writes all robots, steps once without rendering, and updates…

### Community 83 - "Transformer"
Cohesion: 0.25
Nodes (4): Build a self-attention block for batch-first token sequences. Args: hidden_size…, Create a two-layer MLP that processes each sequence position separately. Args:…, Build a cross-attention block with optional distinct context width. Args:…, Store the sinusoidal frequency settings without trainable weights. Args:…

### Community 84 - "Readme"
Cohesion: 0.25
Nodes (8): Actuator features: 6 values per joint, DA-GET-BFM: Dynamics Aware Cross Embodiment Behavior Foundation Model with Zero-Shot Capabilities, Dynamics features: 7 values per joint and time step, Installation, Physics features: sources and calculation, Rigid-body features: 25 values per joint, Status, What comes from where

### Community 85 - "Isaac Env Tests"
Cohesion: 0.25
Nodes (4): FakeSensor, Regexes choose the penalized links, as BFM-Zero's pelvis, shoulders and hips., Merged feet or missing joints give a clear error instead of wrong data., Report contact forces, with bodies in sensor order.

### Community 86 - "Physics Tests"
Cohesion: 0.32
Nodes (5): PhysicsCrossAttentionTest, Check the encoder's Physics Cross Attention and its memory cache., Return encoder keyword arguments except physics., Cached static keys and values give identical outputs until weights change., Physics features matter, and the quadruped encodes the same alone or padded.

### Community 87 - "Anymal D URDF"
Cohesion: 0.29
Nodes (7): base_to_depth_camera_right_camera, depth_camera_right_camera_depth_to_depth_optical_frame, depth_camera_right_camera_parent_to_depth_frame, depth_camera_right_camera, depth_camera_right_camera_parent, depth_camera_right_depth_frame, depth_camera_right_depth_optical_frame

### Community 88 - "Anymal D URDF"
Cohesion: 0.29
Nodes (7): depth_camera_front_lower_camera_depth_to_depth_optical_frame, depth_camera_front_lower_camera_parent_to_depth_frame, face_front_to_depth_camera_front_lower_camera, depth_camera_front_lower_camera, depth_camera_front_lower_camera_parent, depth_camera_front_lower_depth_frame, depth_camera_front_lower_depth_optical_frame

### Community 89 - "Physics Schema"
Cohesion: 0.33
Nodes (6): feature_names(), PhysicsFeature, Schema of the physics features read by Physics Cross Attention. Every feature…, Return the column names of a feature group. Args: group (str): ``rigid_body``,…, Describe one scalar physics feature. Attributes: name (str): Column name.…, _vector()

### Community 90 - "RL Isaac Env"
Cohesion: 0.33
Nodes (4): Any, Apply friction, link mass and base center-of-mass randomization to each world., Import Warp, which Isaac Lab's PhysX views use for arrays., _warp()

### Community 91 - "Go1 URDF"
Cohesion: 0.33
Nodes (6): camera_joint_left, camera_laserscan_joint_left, camera_optical_joint_left, camera_laserscan_link_left, camera_left, camera_optical_left

### Community 92 - "Go1 URDF"
Cohesion: 0.40
Nodes (6): FL_hip_joint, FL_thigh_rotor_joint, RL_thigh_rotor_joint, FL_hip, FL_thigh_rotor, RL_hip

### Community 93 - "Benchmark Inference"
Cohesion: 0.40
Nodes (6): benchmark(), device, Namespace, Return the wall-clock milliseconds of ``steps`` calls after ``warmup`` calls., Time every part for one robot and encoder size. Returns: tuple[dict, dict]:…, timed()

### Community 94 - "Benchmark Inference"
Cohesion: 0.33
Nodes (6): main(), Return the budget check of a total, empty for a component., Parse arguments, run the benchmarks and print the tables., Return the median, 99th percentile and maximum., summary(), verdict()

### Community 95 - "Anymal D URDF"
Cohesion: 0.40
Nodes (5): LF_KFE, LF_shank_fixed_LF_FOOT, LF_FOOT, LF_SHANK, LF_shank_fixed

### Community 96 - "G1 29Dof URDF"
Cohesion: 0.40
Nodes (5): waist_pitch_joint, waist_roll_joint, waist_yaw_joint, waist_roll_link, waist_yaw_link

### Community 97 - "Go1 URDF"
Cohesion: 0.40
Nodes (5): camera_joint_right, camera_laserscan_joint_right, camera_laserscan_link_right, camera_optical_right, camera_right

### Community 98 - "Ground Contact"
Cohesion: 0.40
Nodes (4): ground_contact source 0: mujoco, ground_contact source 1: mujoco, ground_contact source 2: mujoco, ground_contact source 3: mujoco

### Community 99 - "G1 Go1 Real And Synthetic"
Cohesion: 0.40
Nodes (4): g1_go1_real_and_synthetic source 0: huggingface, g1_go1_real_and_synthetic source 1: synthetic, g1_go1_real_and_synthetic source 2: mujoco, g1_go1_real_and_synthetic source 3: synthetic

### Community 100 - "Bfm"
Cohesion: 0.40
Nodes (4): bfm_weight_init(), Module, Initialize BFM heads without changing pretrained encoder weights. Returns:…, Orthogonal weights and zero biases for linear layers, as BFM-Zero's…

### Community 101 - "Bfm"
Cohesion: 0.40
Nodes (3): Clamp in the forward pass, pass gradients straight through., Draw a reparameterized action; ``clip`` bounds the noise (TD3 style)., Size

### Community 102 - "Try"
Cohesion: 0.40
Nodes (5): import torch, from model import BfmConfig, DaGetBfmModel, bfm, empty cell, try.ipynb notebook

### Community 103 - "Anymal D URDF"
Cohesion: 0.50
Nodes (4): base_to_lidar_parent, lidar_parent_to_lidar, lidar, lidar_parent

### Community 104 - "G1 Hub Plus Synthetic"
Cohesion: 0.50
Nodes (3): g1_hub_plus_synthetic source 0: huggingface, g1_hub_plus_synthetic source 1: synthetic, g1_hub_plus_synthetic source 2: synthetic

### Community 105 - "Synthetic Unitree"
Cohesion: 0.50
Nodes (3): synthetic_unitree source 0: synthetic, synthetic_unitree source 1: synthetic, synthetic_unitree source 2: synthetic

### Community 108 - "RL Buffers"
Cohesion: 0.50
Nodes (3): Return base-to-world rotations ``[N, 3, 3]`` with zero yaw from gravity…, rotation_from_gravity(), The rotation maps world down to the given base-frame direction.

### Community 109 - "Convert Dataset"
Cohesion: 0.50
Nodes (3): main(), Convert a public robot dataset to the training table format. Writes…, Parse arguments and run one converter.

### Community 110 - "Download Urdfs"
Cohesion: 0.50
Nodes (3): main(), Download the robot URDFs used by the example data configs into assets/urdf. The…, Download every URDF that is not present yet.

### Community 111 - "Embodiment Tests"
Cohesion: 0.50
Nodes (3): Check URDF parsing and rotation conventions., Convert random angles to matrices and back., UrdfTest

### Community 112 - "Fb Cpr Tests"
Cohesion: 0.50
Nodes (3): as_float32(), Round a float the way a ``float32`` tensor stores it., The discount is zero exactly where the episode terminated, and gamma elsewhere.

### Community 114 - "Training Docs"
Cohesion: 0.67
Nodes (3): 3. Multirotor flight (synthetic quadrotor with randomized size and mass), Data config, Watching training in TensorBoard

### Community 116 - "Readme"
Cohesion: 0.67
Nodes (3): 1. Encode different robots in one batch, Quick start, refreshes the dynamics terms according to their dependencies.

### Community 117 - "Isaac Env Tests"
Cohesion: 0.67
Nodes (3): frame(), ndarray, Return a motion frame with distinct joint values.

## Knowledge Gaps
- **86 isolated node(s):** `DataArguments`, `da-get-bfm`, `behavior cloning distillation`, `embodiment graph structural attention bias`, `forward kinematics self modeling` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 945 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EmbodimentGraph` connect `Dynamics` to `Training Episodes`, `Training Sources`, `Physics`, `Bfm`, `Training Ground Sim`, `Embodiment`, `Embodiment`, `Embodiment Tests`, `Robot Analysis`, `Conversion Mapping`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `EmbodimentRecord` connect `Training Episodes` to `Dynamics`, `RL Isaac Env`, `Training Sources`, `Physics`, `Training Ground Sim`, `Fb Cpr Tests`, `RL Envs`, `Isaac Env Tests`, `Fb Cpr Tests`, `Isaac Env Tests`, `RL Envs`, `RL Isaac Env`, `Training Dataset`, `RL Envs`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `DaGetConfig` connect `Da Get` to `Da Get`, `Training Episodes`, `Dynamics`, `Bfm Tests`, `Embodiment Tests`, `Bfm`, `Physics`, `Pretraining`, `Pretraining`, `Da Get Encoder Tests`, `Bfm`, `Pretraining Tests`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 35 inferred relationships involving `EmbodimentGraph` (e.g. with `contact_maps()` and `coriolis_efforts()`) actually correct?**
  _`EmbodimentGraph` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `EmbodimentRecord` (e.g. with `EmbodimentEnv` and `MujocoEmbodimentEnv`) actually correct?**
  _`EmbodimentRecord` has 32 INFERRED edges - model-reasoned connections that need verification._
- **What connects `DataArguments`, `da-get-bfm`, `behavior cloning distillation` to the rest of the system?**
  _86 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Dynamics` be split into smaller, more focused modules?**
  _Cohesion score 0.03574553180852393 - nodes in this community are weakly interconnected._