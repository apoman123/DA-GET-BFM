# Third-party notices

This repository uses work by other people: code, configuration values, robot models,
datasets, fonts and tools. Each section names the authors or copyright holders, the
license, and what is used. The full bibliography, with all authors, is in
[docs/references.md](docs/references.md).

## BFM-Zero / Meta Motivo (CC BY-NC 4.0)

`model/modeling_bfm.py`, `model/configuration_bfm.py` and `model/rl/` adapt code and
configuration from [LeCAR-Lab/BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) (checked at
commit 05ad611), by Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto,
Andrea Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro
Lazaric, Matteo Pirotta and Guanya Shi, which builds on Meta's FB-CPR / Meta Motivo
implementation.

- License of the whole repository: Creative Commons Attribution-NonCommercial 4.0
  International (<https://creativecommons.org/licenses/by-nc/4.0/>). It does not permit
  commercial use of the adapted parts.
- Agent and network files, which carry "Copyright (c) Meta Platforms, Inc. and
  affiliates": `humanoidverse/agents/nn_models.py`, `humanoidverse/agents/fb/model.py`,
  `humanoidverse/agents/fb/agent.py`, `humanoidverse/agents/fb_cpr/model.py`,
  `humanoidverse/agents/fb_cpr/agent.py`, `humanoidverse/agents/fb_cpr_aux/model.py`,
  `humanoidverse/agents/fb_cpr_aux/agent.py` and `humanoidverse/agents/misc/zbuffer.py`.
- Training, environment, evaluation and configuration files, which have no copyright
  header and are covered by the repository license (the BFM-Zero authors):
  `humanoidverse/train.py`, `humanoidverse/agents/evaluations/humanoidverse_isaac.py`
  (`emd_numpy`), `humanoidverse/envs/legged_base_task/legged_robot_base.py`,
  `humanoidverse/envs/legged_robot_motions/legged_robot_motions.py`,
  `humanoidverse/utils/motion_lib/motion_lib_base.py`, and the configs
  `humanoidverse/config/{simulator/isaacsim, env/legged_motions, domain_rand/domain_rand,
  obs/bfm_zero_obs, rewards/reward_bfm_zero, robot/g1/g1_29dof_hard_waist}.yaml`.

Adapted parts:
- the MLP heads of `DaGetBackwardMap` and `DaGetDiscriminator`;
- `TruncatedNormal` (which BFM-Zero in turn takes from DrQ-v2, see below);
- the orthogonal `bfm_weight_init`;
- the sphere projection of z;
- the latent sampling and the goal, reward, and tracking inference of
  `DaGetBfmModel`;
- the default sizes in `BfmConfig` (from `humanoidverse/train.py`).

Changes:
- rewritten in this repository's style;
- dictionary observations selected by config keys;
- a padding mask on `TruncatedNormal`;
- vectorized tracking inference.

In `model/rl/`, the adapted parts are the FB, discriminator, critic, and actor
losses, the ensemble pessimism, the z mixture and rollout logic, the z buffer, the
reward normalizer, the online loop, the lying-down starts, the EMD and motion
priorities, and the hyperparameter values. The environment settings, randomization
ranges and penalty definitions follow BFM-Zero's configuration files listed above. Each
adapted function names its source file in its docstring, and
[docs/bfm.md](docs/bfm.md#attribution) has a file-by-file table.

The per-DoF networks, the per-robot training, the encoder tuning, and the MuJoCo and
Isaac Lab environments are new. Their design follows the BFM-Zero / FB-CPR interfaces.

Please cite:
- Yitang Li et al., *BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid
  Control Using Unsupervised Reinforcement Learning*, ICLR 2026, arXiv:2511.04131;
- Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto,
  Yingchen Xu, Alessandro Lazaric and Matteo Pirotta, *Zero-Shot Whole-Body Humanoid
  Control via Behavioral Foundation Models*, ICLR 2025, arXiv:2504.11054.

## DrQ-v2 (MIT)

`TruncatedNormal` in `model/modeling_bfm.py` originates in `utils.py` of
[facebookresearch/drqv2](https://github.com/facebookresearch/drqv2) ("Copyright (c)
Facebook, Inc. and its affiliates", MIT), from Denis Yarats, Rob Fergus, Alessandro
Lazaric and Lerrel Pinto, *Mastering Visual Continuous Control: Improved Data-Augmented
Reinforcement Learning*, ICLR 2022. It reaches this repository through BFM-Zero.

## GET-Zero

The encoder implements the Graph Embodiment Transformer of Austin Patel and Shuran Song,
*GET-Zero: Graph Embodiment Transformer for Zero-Shot Embodiment Generalization*, ICRA
2025, pp. 14262-14269, DOI 10.1109/ICRA55743.2025.11127922, arXiv:2407.15002
(<https://get-zero-paper.github.io/>). The authors' code,
[real-stanford/get_zero](https://github.com/real-stanford/get_zero) (checked at commit
279652d), has no license file. No code is copied; this repository reproduces its design
and these values and conventions, credited to Patel and Song:

- the `da_get_zero` sizes (width 512, 8 layers, 16 heads, feed-forward 1024, dropout 0.1,
  input LayerNorm) from `get_zero/distill/cfg/model/Base.yaml`;
- the tokenization switches and the history length 6 from `cfg/tokenization/Default.yaml`;
- the feature-major history layout from `embodiment_util.py`;
- the parent/child attention-bias tables with padding index 0 from
  `embodiment_attention.py`.

Please cite the paper when using the encoder.

## Other robotics code whose values or conventions are used

No code from these repositories is copied:

- [ethz-asl/rotors_simulator](https://github.com/ethz-asl/rotors_simulator) (RotorS,
  Apache-2.0; Fadri Furrer, Michael Burri, Mina Kamel, Janosch Nikolic and Markus
  Achtelik, ASL, ETH Zurich): the rotor model form and the `moment_constant` 0.016 m used
  as `drag_ratio` in `model/robot_analysis.py` and `model/physics.py`.
- [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) (BSD-3-Clause;
  ETH Zurich, Nikita Rudin, and NVIDIA): the pattern of computing and clipping the PD
  torque at every physics step.
- [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
  (BSD-3-Clause, Unitree Robotics): PD gains, cited for comparison.
- [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
  (BSD-3-Clause, Unitree Robotics): the G1 motor order (`G1JointIndex`), one of the checks
  behind `KNOWN_NAME_FIXES` in `model/conversion/mapping.py`.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause, the Isaac Lab Project
  Developers): `model/rl/isaac_env.py` calls its API, subclasses `IdealPDActuator`, and
  uses the solver iteration counts of its `UNITREE_GO1_CFG`.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Kevin Zakka,
  Yuval Tassa and the MuJoCo Menagerie contributors; `unitree_go1` model by Unitree
  Robotics, BSD-3-Clause): the Go1 nominal pose in `configs/rl/` equals the `home`
  keyframe of `unitree_go1/go1.xml`.

## GenBot-1K robot generator (BSD-3-Clause)

`model/morphology.py` takes the base dimensions, masses, joint offsets, PD gains and
the variation schedule (one geometry factor per robot, the discrete sets of Table 6)
of the procedural robots from the paper and the released generator of Ai et al.,
*Towards Embodiment Scaling Laws in Robot Locomotion*, CoRL 2025
(arXiv:2505.05753; authors Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu,
Zhanxin Wu, K. Fay, Henrik I. Christensen, Jan Peters and Hao Su), from
[BoAi01/embodiment-scaling-laws](https://github.com/BoAi01/embodiment-scaling-laws) at
commit 62e43ce. No code is copied; the numeric values are used under the generator's
BSD-3-Clause license, "Copyright (c) 2022-2024, The Berkeley Humanoid Project
Developers". Where the paper and the code disagree (humanoid foot scaling, quadruped
hip mass) the module docstring says which one is followed; it also documents one
deviation from the code (the humanoid calf length is not divided by the knee count).

The GenLoco ranges (Feng et al., *GenLoco: Generalized Locomotion Controllers for
Quadrupedal Robots*, CoRL 2022, PMLR 205:1893-1903; code Apache-2.0) are taken from
the paper's Table 1 and Appendix A; the code was only read to check the density rule.

## Control-task probing regimen (Apache-2.0)

`train_mlp_probe` in `model/evaluation/probes.py` adapts the probe training regimen of
[john-hewitt/control-tasks](https://github.com/john-hewitt/control-tasks)
(`regimen.py`: Adam at 1e-3, learning rate halved on a plateau, early stopping at the
fifth evaluation without improvement), from John Hewitt and Percy Liang, *Designing and
Interpreting Probes with Control Tasks*, EMNLP-IJCNLP 2019. License: Apache License 2.0
(the repository's license file leaves the copyright line blank).
The bootstrap and IQM statistics in `model/evaluation/statistics.py` re-implement the
methods of Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville and Marc
G. Bellemare, NeurIPS 2021, and the defaults of their library
[google-research/rliable](https://github.com/google-research/rliable) ("Copyright 2021
The Rliable Authors", Apache-2.0), without copying code.

## LoRA (MIT) and PEFT (Apache-2.0)

`model/adaptation.py` implements LoRA (Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang and Weizhu Chen, *LoRA: Low-Rank Adaptation of
Large Language Models*, ICLR 2022). No code is copied: `LoRALinear.forward` uses the same
one-line formula as `loralib`'s `Linear.forward`
([microsoft/LoRA](https://github.com/microsoft/LoRA), "Copyright (c) Microsoft
Corporation", MIT), the default initialization of A (`kaiming_uniform_` with
`a = sqrt(5)`) follows `loralib`, and substituting the merged weights in
`nn.MultiheadAttention` follows the design of Hugging Face
[PEFT](https://github.com/huggingface/peft) (Sourab Mangrulkar, Sylvain Gugger, Lysandre
Debut, Younes Belkada, Sayak Paul, Benjamin Bossan and Marian Tietz; "Copyright
2023-present the HuggingFace Inc. team", Apache-2.0).

## Values quoted from other repositories

- `model/rl/retarget.py` quotes the `REF_POS_SCALE` constants of
  [google-research/motion_imitation](https://github.com/google-research/motion_imitation)
  ("Copyright 2020 The Google Research Authors", Apache-2.0), the code of Xue Bin Peng,
  Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan and Sergey Levine, *Learning Agile
  Robotic Locomotion Skills by Imitating Animals*, RSS 2020. No code is copied.
- `configs/splits/zero_shot.json` and `model/evaluation/splits.py` use values from the
  MetaMorph paper (Agrim Gupta, Linxi Fan, Surya Ganguli and Li Fei-Fei, ICLR 2022); its
  code ([agrimgupta92/metamorph](https://github.com/agrimgupta92/metamorph)) has no
  license file and none of it is used.

## Datasets

The converters in `model/conversion/datasets.py`, `scripts/download_datasets.py` and the
example configs download these datasets from their publishers; none is redistributed in
this repository. Their terms
apply to anything trained on them. Full references are in
[docs/data.md](docs/data.md#references).

| Dataset | Authors / publisher | License stated by the publisher | Used by |
|---|---|---|---|
| [`unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit`](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit) and the other `unitreerobotics/G1_WBT_*` repositories (77 on 2026-11-23, `G1_WBT_SMPL_*` included), [`unitreerobotics/G1_Dex3_ToastedBread_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) | Unitree Robotics | Apache-2.0 | `configs/data/g1_*.json`, `configs/rl/g1_go1_real_and_synthetic.json`, `scripts/download_datasets.py` (`g1_wbt`, `g1_wbt_q36`, `g1_wbt_smpl`, `g1_dex3`) |
| MIPO-Go1, [`Open-IO/mipo-go1-odometry`](https://huggingface.co/datasets/Open-IO/mipo-go1-odometry) | Shuo Yang, Zixin Zhang, Benjamin Bokser, Zachary Manchester (IROS 2023); recordings from [Cerberus 2.0](https://github.com/ShuoYangRobotics/Cerberus2.0) | AGPL-3.0 (card and Cerberus 2.0) | `convert_mipo`, torque and contact validation |
| GrandTour, [`leggedrobotics/grand_tour_dataset`](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) | Turcan Tuna, Jonas Frey, Lanke Frank Tarimo Fu, Katharine Patterson, Tianao Xu, Maurice Fallon, Cesar Cadena, Marco Hutter | Card MIT; website: data CC BY-SA 4.0, software MIT | `convert_grandtour` |
| NeuroBEM | Leonard Bauersfeld, Elia Kaufmann, Philipp Foehn, Sihao Sun, Davide Scaramuzza (Robotics and Perception Group, University of Zurich) | None stated | `convert_neurobem` |
| LocoMuJoCo, [`robfiras/loco-mujoco-datasets`](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) | Firas Al-Hafez, Guoping Zhao, Jan Peters, Davide Tateo | Data CC BY-NC-ND 4.0; code MIT | `convert_locomujoco` |
| AMASS retargeted to G1, [`fleaven/Retargeted_AMASS_for_robotics`](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics) (revision `275c4c4a`) | Retargeting by Kun Zhao; AMASS by Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, Michael J. Black; the README also asks to cite SMPL-X (Pavlakos et al., CVPR 2019) | Card CC BY 4.0; the AMASS license (non-commercial research, no redistribution) applies | `convert_amass_g1` |
| xwbc-retarget-data, [`TonyZhang1999/xwbc-retarget-data`](https://huggingface.co/datasets/TonyZhang1999/xwbc-retarget-data) (revision `b4e169f4`) | Retargeting by the Hub user TonyZhang1999 (paper unverified), apparently with GMR (Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu, arXiv:2510.02252); AMASS by Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, Michael J. Black; LAFAN1 by Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, Christopher Pal (Ubisoft La Forge) | Card MIT; the AMASS license (non-commercial research) and LAFAN1's CC BY-NC-ND 4.0 apply to the motions | `convert_xwbc` |
| LAFAN1 retargeted to Unitree, [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (copy of `unitreerobotics/LAFAN1_Retargeting_Dataset`) | Retargeting by Unitree Robotics; LAFAN1 motions by Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, Christopher Pal (Ubisoft La Forge) | Retargeting BSD-3-Clause; LAFAN1 CC BY-NC-ND 4.0 | `convert_lafan1` |

## Robot models (URDF)

`scripts/download_urdfs.py` downloads these files into `assets/urdf/` (gitignored, so
they are not redistributed here) and the pipeline reads them unmodified:

| File | Author / copyright holder | Source | License |
|---|---|---|---|
| `go1.urdf`, `h1.urdf`, `g1_29dof_rev_1_0.urdf` and its 24 collision meshes (`meshes/*.STL`), `go2.urdf` | HangZhou YuShu Technology Co., Ltd. ("Unitree Robotics"), 2016-2022 | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | BSD-3-Clause |
| `anymal_d.urdf` | ANYbotics AG, 2023 | [ANYbotics/anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description) | BSD-3-Clause |
| `talos_reduced.urdf` | PAL Robotics, S.L., 2016 (stated in the file) | [Gepetto/example-robot-data](https://github.com/Gepetto/example-robot-data) (repository by LAAS-CNRS, University of Edinburgh and INRIA, BSD-3-Clause) | The file itself: Creative Commons Attribution-NonCommercial-NoDerivs 3.0 (<http://creativecommons.org/licenses/by-nc-nd/3.0/>) |

The Talos transmissions have negative `mechanicalReduction` values; `model/urdf.py` takes
their magnitude when parsing, in memory, and does not write a modified file.
`assets/robots/quadrotor_x.urdf` was written for this repository.

## GET-Zero project page text

`raw/get-zero-paper_github_io.md` is a verbatim text capture of the GET-Zero project page
(<https://get-zero-paper.github.io/>) by Austin Patel and Shuran Song, kept as reference
material for the knowledge graph. All rights remain with the authors.

## Fonts embedded in the proposal PDF

`docs/proposal/DA-GET-BFM_proposal_v4.pdf` embeds subsets of these fonts:

- **Noto Sans CJK TC** and **Noto Sans Mono CJK TC** (regular and bold): © 2014-2021 Adobe
  (<http://www.adobe.com/>); designers Ryoko Nishizuka (kana, bopomofo and ideographs),
  Paul D. Hunt (Latin, Greek and Cyrillic), and Sandoll Communications, Soo-young Jang
  and Joo-yeon Kang (hangul). Noto is a trademark of Google Inc. Licensed under the SIL
  Open Font License, Version 1.1 (<https://openfontlicense.org>). Source:
  [notofonts/noto-cjk](https://github.com/notofonts/noto-cjk).
- **DejaVu Sans**: by the DejaVu fonts team (<https://dejavu-fonts.github.io/>), based on
  Bitstream Vera. Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. Bitstream
  Vera is a trademark of Bitstream, Inc. Glyphs imported from Arev fonts are Copyright
  (c) 2006 by Tavmjong Bah. All Rights Reserved. DejaVu changes are in the public domain.
  The Bitstream Vera and Arev licenses permit reproduction and distribution provided
  their copyright notices and permission notices are kept; the full texts are in the
  font's `name` table and at <https://dejavu-fonts.github.io/License.html>.

## Tools that generated files in this repository

- The proposal PDF is built by `scripts/build_proposal_pdf.py` with
  [Python-Markdown](https://github.com/Python-Markdown/markdown) (Manfred Stienstra,
  Yuri Takhteyev and the Python Markdown Project, maintained by Waylan Limberg;
  BSD-3-Clause) and [WeasyPrint](https://github.com/Kozea/WeasyPrint) (Simon Sapin and
  contributors, CourtBouillon; BSD-3-Clause). Neither is copied into the repository.
- `graphify-out/` was generated by [graphify](https://github.com/Graphify-Labs/graphify)
  0.9.103, created by Safi (GitHub user safishamsi) and now maintained by Graphify Labs (Apache-2.0). Its `graph.html` loads
  [vis-network](https://github.com/visjs/vis-network) 9.1.6 from unpkg (Copyright (C)
  2010-2018 Almende B.V. and Contributors, Copyright (c) 2018-2021 Vis.js contributors;
  Apache-2.0 or MIT); the library is not copied into the repository.

## Software dependencies

The code calls these packages through their public APIs and copies none of their code
(except where a section above says otherwise). Authors are as stated in each package's
metadata or license file.

| Package | Authors | License |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | PyTorch Team and contributors | BSD-3-Clause (bundled components under other permissive licenses) |
| [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Accelerate](https://github.com/huggingface/accelerate), [safetensors](https://github.com/huggingface/safetensors) | Hugging Face and contributors | Apache-2.0 |
| [PyArrow](https://arrow.apache.org/) (Apache Arrow) | The Apache Software Foundation (NOTICE: "Copyright 2016-2026 The Apache Software Foundation") | Apache-2.0 (bundled components under other permissive licenses, listed in its NOTICE) |
| [NumPy](https://numpy.org) | Travis E. Oliphant et al., NumPy Developers | BSD-3-Clause |
| [SciPy](https://scipy.org) | Enthought, Inc. and the SciPy Developers | BSD-3-Clause |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | Google Inc. / TensorFlow authors | Apache-2.0 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) (extra `sim`) | Google DeepMind | Apache-2.0 |
| [numcodecs](https://github.com/zarr-developers/numcodecs) (extra `convert`) | Zarr developers | MIT |
| [h5py](https://github.com/h5py/h5py) (extra `validation`) | Andrew Collette and contributors | BSD-3-Clause |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (installed separately) | The Isaac Lab Project Developers (NVIDIA) | BSD-3-Clause |
| [pytest](https://github.com/pytest-dev/pytest) (dev) | Holger Krekel, Bruno Oliveira, Ronny Pfannschmidt, Floris Bruynooghe, Brianna Laugher, Freya Bruhin and others | MIT |
| [uv](https://github.com/astral-sh/uv) | Astral Software Inc. | Apache-2.0 or MIT |

The CI workflow (`.github/workflows/ci.yml`) uses
[actions/checkout](https://github.com/actions/checkout) (GitHub, Inc. and contributors,
MIT) and [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) (Kevin Stillhammer,
MIT).
