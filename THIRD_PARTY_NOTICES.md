# Third-party notices

## BFM-Zero / Meta Motivo (CC BY-NC 4.0)

`model/modeling_bfm.py` and `model/rl/` adapt code from
[LeCAR-Lab/BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero), which builds on
Meta's FB-CPR / Meta Motivo implementation:

- Source files: `humanoidverse/agents/nn_models.py`, `agents/fb/model.py`,
  `agents/fb/agent.py`, `agents/fb_cpr/agent.py`, `agents/fb_cpr_aux/agent.py`,
  `agents/misc/zbuffer.py`, and `humanoidverse/train.py`.
- Copyright (c) Meta Platforms, Inc. and affiliates.
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (<https://creativecommons.org/licenses/by-nc/4.0/>). This license does not
  permit commercial use of the adapted parts.

Adapted parts:
- the MLP heads of `DaGetBackwardMap` and `DaGetDiscriminator`;
- `TruncatedNormal`;
- the orthogonal `bfm_weight_init`;
- the sphere projection of z;
- the latent sampling and the goal, reward, and tracking inference of
  `DaGetBfmModel`.

Changes:
- rewritten in this repository's style;
- dictionary observations selected by config keys;
- a padding mask on `TruncatedNormal`;
- vectorized tracking inference.

In `model/rl/`, the adapted parts are the FB, discriminator, critic, and actor
losses, the ensemble pessimism, the z mixture and rollout logic, the z buffer, the
reward normalizer, the online loop, the lying-down starts, the EMD and motion
priorities, and the hyperparameter values. The environment settings and penalty
definitions follow BFM-Zero's configuration.

The per-DoF networks, the per-robot training, the encoder tuning, and the MuJoCo and
Isaac Lab environments are new. The Isaac Lab adapter calls Isaac Lab's API
(BSD-3-Clause, <https://github.com/isaac-sim/IsaacLab>) and does not copy its code. Their design follows the
BFM-Zero / FB-CPR interfaces.

Please cite:
- Li et al., *BFM-Zero*, arXiv:2511.04131, 2025;
- Tirinzoni et al., *Zero-Shot Whole-Body Humanoid Control via Behavioral
  Foundation Models*, ICLR 2025.

## GET-Zero

The encoder implements the Graph Embodiment Transformer from Patel and Song,
*GET-Zero: Graph Embodiment Transformer for Zero-shot Embodiment
Generalization*, ICRA 2025 (<https://get-zero-paper.github.io/>). Please cite
the paper when using the encoder.
