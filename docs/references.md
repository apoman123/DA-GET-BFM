# References and credits

[← Back to the README](../README.md)

This repository builds on other people's work. This page says which part comes from
whom and lists the main references with all their authors. Each module's docstring cites
the sources it uses at the point of use, and each topic page has its own reference list:
[physics features](physics_features.md#references), [encoder](encoder.md#references),
[dynamics](dynamics.md#references), [physics pretraining](training.md#references),
[training data](data.md#references), [behavior foundation model](bfm.md#references),
[zero-shot](zero_shot.md#references), and section 8 of the
[proposal](proposal/DA-GET-BFM_proposal_v4.pdf). Licenses and copyright holders of code,
robot models, datasets, fonts and tools are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Contents

- [What comes from where](#what-comes-from-where)
- [Main references](#main-references)
- [BibTeX](#bibtex)

## What comes from where

| Component | Source | Added or changed here |
|---|---|---|
| Per-joint tokens, graph attention bias, `da_get_zero` preset | GET-Zero (Patel and Song 2025): paper and the sizes of its released config | Floating bases and multirotors |
| Mass-matrix attention bias, Physics Cross Attention, physics pretraining | This repo | |
| Contact-constrained response task (labels `G(q) δτ` without any torque) | This repo, from constrained rigid-body dynamics (Featherstone 2008) | |
| Torque-envelope target clipping, per-physics-step envelope, torque penalties | This repo | Added on top of BFM-Zero's auxiliary rewards |
| Adaptive torque-penalty weights | Lagrangian constrained RL (Altman 1999; Tessler et al. 2019; Stooke et al. 2020) | Per-robot multipliers with data-sheet targets |
| Contact inference from kinematics and dynamics | Probabilistic contact estimation on legged robots (Camurri et al. 2017; Bledt et al. 2018) | Subset MAP with friction pyramids, minimum load and a confidence filter for labels |
| Backward map and discriminator heads, truncated-normal policy, latent inference | BFM-Zero (Li et al. 2026) / FB-CPR (Tirinzoni et al. 2025) | One shared DA-GET encoder with a state-token adapter; padding mask |
| Actor and forward map interfaces | BFM-Zero (Li et al. 2026) / FB-CPR (Tirinzoni et al. 2025) | Rebuilt to read joint embeddings and cross-attend to z; one action per joint |
| Training losses, z mixture, rollout logic, fall starts, motion priorities, hyperparameters | BFM-Zero (Li et al. 2026) / FB-CPR (Tirinzoni et al. 2025) | Per-robot worlds, buffers, and prompts; encoder tuned with physics losses |
| Physics-derived target normalization, dimensionless inputs (Froude scaling) | Dynamic similarity (Alexander and Jayes 1983), dimensional analysis (Buckingham 1914) | Per-feature unit exponents; reference robot |
| Structured physics interface (operator, consistency and rollout losses) | Structured dynamics models (Lutter et al. 2019; Watter et al. 2015); Gauss's principle (Udwadia and Kalaba 1992); Moore-Penrose condition (Penrose 1955) | Interface units, rank, losses |
| Centroidal head | Centroidal momentum (Orin and Goswami 2008; Orin, Goswami and Lee 2013; Dai, Valenzuela and Tedrake 2014) | Dimensionless targets |
| Procedural quadrupeds, hexapods and humanoids | GenBot-1K values and schedule (Ai et al. 2025); GenLoco mass and density rules (Feng et al. 2022) | URDF writer, simulator source |
| Temperature sampling over robots or families | Lample and Conneau 2019; Conneau et al. 2020; Arivazhagan et al. 2019 | Grouped variants, one encode per state |
| Left-right mirroring | Mittal et al. 2024; Ordoñez-Apraez et al. 2025 | Mirror map derived from geometry |
| Morphology modulation | ModuMorph (Xiong, Beck and Whiteson 2023); hypernetworks (Ha, Dai and Le 2017) | Zero-initialized attention bias and input correction |
| Masked physics modeling | Mask ratios of BERT (Devlin et al. 2019) and MAE (He et al. 2022) | Physics-token masking |
| History and specification-deviation head | RMA's 0.5 s history (Kumar et al. 2021); dynamics randomization (Peng et al. 2018) | Spec noise from the repo's randomization ranges |
| Test-time adaptation | LoRA (Hu et al. 2022), L2-SP (Li, Grandvalet and Davoine 2018), LwF (Li and Hoiem 2016/2018), PAD (Hansen et al. 2021) | LoRA on attention, L2-SP transferred to LoRA |
| Retargeting between robots | Gleicher 1998; damped least squares (Wampler 1986; Nakamura and Hanafusa 1986) | Keypoint scaling from robot geometry |
| Per-family action scale, mass-proportional gains | URMA (Bohlinger et al. 2024), Ai et al. 2025; GenLoco (Feng et al. 2022) | |
| Zero-shot splits, probes, statistics | MetaMorph splits (Gupta et al. 2022); probes (Alain and Bengio 2017; Hewitt and Liang 2019); IQM and bootstrap (Agarwal et al. 2021; Efron 1979) | Robot-level bootstrap |

The actor and critics are separate networks. In the published BFM-Zero training
configuration they share width and depth, while the critics use the forward-map
architecture to predict scalar values from state, latent, and action. This repo
uses per-joint attention in place of those residual MLPs; see
[the architecture comparison](bfm.md#configuration).

## Main references

**Foundations**

- Austin Patel and Shuran Song. *GET-Zero: Graph Embodiment Transformer for Zero-Shot
  Embodiment Generalization.* ICRA 2025, pp. 14262–14269. DOI
  10.1109/ICRA55743.2025.11127922, arXiv:2407.15002.
  [Project page](https://get-zero-paper.github.io/), [code](https://github.com/real-stanford/get_zero).
- Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni,
  Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo
  Pirotta, and Guanya Shi. *BFM-Zero: A Promptable Behavioral Foundation Model for
  Humanoid Control Using Unsupervised Reinforcement Learning.* ICLR 2026,
  arXiv:2511.04131. [Project page](https://lecar-lab.github.io/BFM-Zero/),
  [code](https://github.com/LeCAR-Lab/BFM-Zero).
- Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto,
  Yingchen Xu, Alessandro Lazaric, and Matteo Pirotta. *Zero-Shot Whole-Body Humanoid
  Control via Behavioral Foundation Models.* ICLR 2025, arXiv:2504.11054 (FB-CPR, Meta
  Motivo).
- Ahmed Touati and Yann Ollivier. *Learning One Representation to Optimize All Rewards.*
  NeurIPS 2021, arXiv:2103.07945 (forward-backward representations).

**Reinforcement learning and control**

- Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. *Mastering Visual
  Continuous Control: Improved Data-Augmented Reinforcement Learning.* ICLR 2022,
  arXiv:2107.09645 (truncated-normal policy).
- Scott Fujimoto, Herke van Hoof, and David Meger. *Addressing Function Approximation
  Error in Actor-Critic Methods.* ICML 2018, PMLR 80:1587–1596, arXiv:1802.09477
  (clipped policy noise).
- Eitan Altman. *Constrained Markov Decision Processes.* Chapman & Hall/CRC, 1999;
  Chen Tessler, Daniel J. Mankowitz, and Shie Mannor. *Reward Constrained Policy
  Optimization.* ICLR 2019, arXiv:1805.11074; Adam Stooke, Joshua Achiam, and Pieter
  Abbeel. *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods.* ICML
  2020, PMLR 119:9133–9143, arXiv:2007.03964 (Lagrange multipliers of the torque
  penalties).
- Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. *Sim-to-Real
  Transfer of Robotic Control with Dynamics Randomization.* ICRA 2018, pp. 3803–3810. DOI
  10.1109/ICRA.2018.8460528 (randomized inertia, center of mass, rotor inertia and motor
  strength).
- Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis,
  Vladlen Koltun, and Marco Hutter. *Learning Agile and Dynamic Motor Skills for Legged
  Robots.* Science Robotics 4(26):eaau5872, 2019. DOI 10.1126/scirobotics.aau5872
  (actuator modelling; why joint torque is hard to model from motion).
- Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. *Learning to Walk in
  Minutes Using Massively Parallel Deep Reinforcement Learning.* CoRL 2021, PMLR
  164:91–100, arXiv:2109.11978 (legged_gym; explicit PD torques at every physics step).
- Taeyoung Lee, Melvin Leok, and N. Harris McClamroch. *Geometric Tracking Control of a
  Quadrotor UAV on SE(3).* IEEE CDC 2010, pp. 5420–5425. DOI 10.1109/CDC.2010.5717652
  (flight controller).

**Dynamics, actuators and contacts**

- Roy Featherstone. *Rigid Body Dynamics Algorithms.* Springer, 2008. DOI
  10.1007/978-1-4899-7560-7 (RNEA, floating-base and contact-constrained dynamics).
- Patrick M. Wensing, Albert Wang, Sangok Seok, David Otten, Jeffrey Lang, and Sangbae
  Kim. *Proprioceptive Actuator Design in the MIT Cheetah: Impact Mitigation and
  High-Bandwidth Physical Interaction for Dynamic Legged Robots.* IEEE T-RO
  33(3):509–522, 2017. DOI 10.1109/TRO.2016.2640183 (reflected rotor inertia).
- Marco Camurri, Maurice Fallon, Stéphane Bazeille, Andreea Radulescu, Victor Barasuol,
  Darwin G. Caldwell, and Claudio Semini. *Probabilistic Contact Estimation and Impact
  Detection for State Estimation of Quadruped Robots.* IEEE RA-L 2(2):1023–1030, 2017.
  DOI 10.1109/LRA.2017.2652491; Gerardo Bledt, Patrick M. Wensing, Sam Ingersoll, and
  Sangbae Kim. *Contact Model Fusion for Event-Based Locomotion in Unstructured
  Terrains.* ICRA 2018, pp. 4399–4406. DOI 10.1109/ICRA.2018.8460904 (contact inference
  from kinematics and dynamics).
- Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. *On the Continuity of
  Rotation Representations in Neural Networks.* CVPR 2019, pp. 5738–5746. DOI
  10.1109/CVPR.2019.00589, arXiv:1812.07035 (6D rotations).

**Simulation**

- Emanuel Todorov, Tom Erez, and Yuval Tassa. *MuJoCo: A Physics Engine for Model-Based
  Control.* IROS 2012, pp. 5026–5033. DOI 10.1109/IROS.2012.6386109 (ground-contact
  simulation and response check).
- Mayank Mittal, Calvin Yu, Qinxi Yu, Jingzhou Liu, Nikita Rudin, David Hoeller, Jia Lin
  Yuan, Ritvik Singh, Yunrong Guo, Hammad Mazhar, Ajay Mandlekar, Buck Babich, Gavriel
  State, Marco Hutter, and Animesh Garg. *Orbit: A Unified Simulation Framework for
  Interactive Robot Learning Environments.* IEEE RA-L 8(6):3740–3747, 2023. DOI
  10.1109/LRA.2023.3270034, arXiv:2301.04195; and Mayank Mittal, Pascal Roth, James Tigue,
  Antoine Richard, Octi Zhang, et al. (105 authors, listed in Isaac Lab's
  [`CITATION.cff`](https://github.com/isaac-sim/IsaacLab/blob/main/CITATION.cff)). *Isaac
  Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning.*
  arXiv:2511.04831, 2025, the citation Isaac Lab asks for
  ([Isaac Lab](https://isaac-sim.github.io/IsaacLab/), the simulator).

**Datasets used for validation**

- Shuo Yang, Zixin Zhang, Benjamin Bokser, and Zachary Manchester. *Multi-IMU
  Proprioceptive Odometry for Legged Robots.* IROS 2023, pp. 774–779. DOI
  10.1109/IROS55552.2023.10342061 (MIPO data). The other datasets, with their creators
  and licenses, are in [Training data](data.md#references).

**Zero-shot extensions**

- Agrim Gupta, Linxi Fan, Surya Ganguli, and Li Fei-Fei. *MetaMorph: Learning Universal
  Controllers with Transformers.* ICLR 2022, arXiv:2203.11931.
- Zheng Xiong, Jacob Beck, and Shimon Whiteson. *Universal Morphology Control via
  Contextual Modulation.* ICML 2023, PMLR 202:38286–38300, arXiv:2302.11070 (ModuMorph).
- Gilbert Feng, Hongbo Zhang, Zhongyu Li, Xue Bin Peng, Bhuvan Basireddy, Linzhu Yue,
  Zhitao Song, Lizhi Yang, Yunhui Liu, Koushil Sreenath, and Sergey Levine. *GenLoco:
  Generalized Locomotion Controllers for Quadrupedal Robots.* CoRL 2022, PMLR
  205:1893–1903, arXiv:2209.05309.
- Nico Bohlinger, Grzegorz Czechmanowski, Maciej Piotr Krupka, Piotr Kicki, Krzysztof
  Walas, Jan Peters, and Davide Tateo. *One Policy to Run Them All: an End-to-end Learning
  Approach to Multi-Embodiment Locomotion.* CoRL 2024, PMLR 270:3356–3378,
  arXiv:2409.06366 (URMA).
- Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu, Zhanxin Wu, K. Fay, Henrik I.
  Christensen, Jan Peters, and Hao Su. *Towards Embodiment Scaling Laws in Robot
  Locomotion.* CoRL 2025, PMLR 305:3483–3515, arXiv:2505.05753 (GenBot-1K).
- Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. *RMA: Rapid Motor
  Adaptation for Legged Robots.* RSS 2021. DOI 10.15607/RSS.2021.XVII.011.
- Michael Lutter, Christian Ritter, and Jan Peters. *Deep Lagrangian Networks: Using
  Physics as Model Prior for Deep Learning.* ICLR 2019, arXiv:1907.04490 (DeLaN).
- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
  Lu Wang, and Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR
  2022, arXiv:2106.09685.
- Xuhong Li, Yves Grandvalet, and Franck Davoine. *Explicit Inductive Bias for Transfer
  Learning with Convolutional Networks.* ICML 2018, PMLR 80:2825–2834 (L2-SP).
- Zhizhong Li and Derek Hoiem. *Learning without Forgetting.* ECCV 2016, pp. 614–629;
  IEEE TPAMI 40(12):2935–2947, 2018. DOI 10.1109/TPAMI.2017.2773081 (LwF).
- Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenyà, Pieter Abbeel, Alexei A.
  Efros, Lerrel Pinto, and Xiaolong Wang. *Self-Supervised Policy Adaptation during
  Deployment.* ICLR 2021, arXiv:2007.04309 (PAD).
- John Hewitt and Percy Liang. *Designing and Interpreting Probes with Control Tasks.*
  EMNLP-IJCNLP 2019, pp. 2733–2743. DOI 10.18653/v1/D19-1275.
- Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G.
  Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice.*
  NeurIPS 2021, pp. 29304–29320, arXiv:2108.13264 (rliable).

The full list for the zero-shot work, with every other method it uses, is in
[Zero-shot across tasks and embodiments](zero_shot.md#references).

## BibTeX

The two works this repository combines:

```bibtex
@inproceedings{patel2025getzero,
  title     = {{GET-Zero}: Graph Embodiment Transformer for Zero-Shot Embodiment Generalization},
  author    = {Patel, Austin and Song, Shuran},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {14262--14269},
  year      = {2025},
  doi       = {10.1109/ICRA55743.2025.11127922}
}

@inproceedings{li2026bfmzero,
  title         = {{BFM-Zero}: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning},
  author        = {Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2026},
  eprint        = {2511.04131},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2511.04131}
}
```
