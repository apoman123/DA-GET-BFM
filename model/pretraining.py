"""Pretrain the GET encoder with auxiliary physics heads (Hugging Face compatible).

The encoder produces one embedding ``H_{t,i}`` per joint. Seven auxiliary tasks
train it; their weights are set in ``DaGetPhysicsPretrainingConfig``. None of them
uses an estimated torque: torque labels come only from simulators (and, if enabled,
torque sensors), and the response tasks need no torque at all.

A. Forward kinematics: ``f_FK(H_i) -> (p_i, R_i)`` of every joint frame in the
   root frame. Positions are divided by the robot's characteristic length so
   hands and humanoids share one scale.
   ``L_FK = 1/N_e sum_i [lambda_p ||p_hat - p||^2 + lambda_R d_SO3(R_hat, R)^2]``.
B. Action-conditioned forward dynamics: ``D_FD(H, tau^applied) -> qdd``, with a
   short graph action-mixing stack so that each joint sees the other joints'
   torques (``M^-1`` couples them).
   ``L_FD = 1/N_e sum_i ((qdd_hat_i - qdd_i) / sigma_i^a)^2``. For a floating
   base the generalized acceleration also contains the base linear and angular
   acceleration, predicted from the pooled mixed tokens and normalized per
   robot. Only samples with exact (simulated) or measured torques.
C. Inverse dynamics: ``D_ID(H, qdd^desired, base acc^desired) ->
   tau^required`` with the full desired acceleration vector mixed over the
   graph; the desired base acceleration is given to every joint. With contacts,
   the error is projected onto the torque directions the motion determines
   (``contact_projection``): internal contact forces cannot be observed from
   motion and are not supervised.
   ``L_ID = 1/N_e sum_i ((P (tau_hat - tau))_i / (tau_i^cont + eps))^2``. Same
   samples as B.
R. Contact-constrained response: ``D_R(H, delta_tau) -> delta_qdd`` (and the base
   acceleration change), where ``delta_nu_dot = G(q) delta_tau`` is the exact
   effect of a torque change with the contacts holding
   (``model.dynamics.constrained_response``). The collator draws ``delta_tau`` for
   every sample of every dataset, so robots recorded without any torque still
   supervise their own inertia, contact constraints, and actuator scale.
   ``L_R = 1/N_e sum_i ((delta_qdd_hat_i - delta_qdd_i) / s_i^R)^2`` plus
   ``lambda_base`` times the normalized base term.
R^-1. Inverse response: ``D_R^-1(H, delta_qdd, delta_base) -> delta_tau``, with the
   error projected like C: ``L = 1/N_e sum_i ((P (dtau_hat - dtau))_i / tau_i^peak)^2``.
D. Motor feasibility and capability margins: ``D_motor(H_i, a_i^proposal)``
   predicts peak and continuous feasibility flags and five margins.
   ``L_motor = lambda_f BCE + lambda_m Huber``.
E. Cross-embodiment action effect: ``D_effect(H, a_t)`` predicts, for each
   end-effector frame, the displacement ``Delta p``, the next orientation
   ``R_{t+1}``, and the next linear and angular velocity. Without a recorded
   command the action input is zero and the head predicts the next frame from the
   state alone.

All per-joint losses are averaged over the valid joints of each sample first
(``1/N_e``) and then over samples, so robots with many joints do not dominate.

Options of the zero-shot proposal (``docs/zero_shot.md``), all off by default:

* ``head_type`` (B1): ``mixer`` (the heads above), ``linear`` (the structured
  physics interface ``DaGetPhysicsInterface``: tasks B, C, R and R^-1 become linear
  readouts of the encoder through a predicted operator ``G_hat = U U^T + diag``,
  a drift ``f_hat`` and an inverse ``K_hat``), or ``linear_residual`` (interface plus
  the mixer heads as residuals). The interface adds the operator loss on the full
  ``G`` and the consistency loss ``G K G = G``.
* ``acceleration_metric`` (A1): ``mass`` replaces the squared error of FD and R by
  the Gauss metric ``e^T M_tilde e`` with the normalized mass matrix
  (Udwadia & Kalaba, Proc. R. Soc. Lond. A, 1992).
* ``effect_mode`` ``rollout`` (B1, [H]): adds ``effect_rollout``, the displacement
  and orientation of every joint frame after a differentiable one-step rollout of
  the interface's predicted accelerations through the analytic forward kinematics.
* ``use_contact_head`` (C2), ``use_centroidal_head`` (C3), ``use_sysid_head`` (E1)
  and ``physics_mask_ratio`` (E3) add the contact, centroidal, system-identification
  and masked-physics-modeling losses.

References:

* The encoder and the graph self-attention of ``GraphActionMixer`` are the Graph
  Embodiment Transformer of Patel & Song, "GET-Zero", ICRA 2025, arXiv:2407.15002.
* Rotations: the continuous 6D representation (Zhou et al., CVPR 2019,
  arXiv:1812.07035), the geodesic distance on SO(3) (Huynh, J. Math. Imaging Vis.
  35:155-164, 2009, DOI 10.1007/s10851-009-0161-2) and Rodrigues' formula (Lynch &
  Park, *Modern Robotics*, Cambridge University Press, 2017, Prop. 3.11, DOI
  10.1017/9781316661239).
* The margin loss is the Huber loss (Huber, Ann. Math. Statist. 35:73-101, 1964, DOI
  10.1214/aoms/1177703732).
* The mass metric is Gauss's principle in the form of Udwadia & Kalaba (Proc. R. Soc.
  Lond. A 439:407-410, 1992, DOI 10.1098/rspa.1992.0158). The structured interface uses
  the ``L L^T`` parameterization of Deep Lagrangian Networks (Lutter, Ritter & Peters,
  ICLR 2019, arXiv:1907.04490), the manipulator equation (Featherstone, *Rigid Body
  Dynamics Algorithms*, Springer, 2008, DOI 10.1007/978-1-4899-7560-7), the
  control-affine latent model of Embed to Control (Watter et al., NIPS 2015,
  arXiv:1506.07365) and the first Penrose condition of a generalized inverse (Penrose,
  Proc. Cambridge Philos. Soc. 51:406-413, 1955, DOI 10.1017/S0305004100030401).
* Masking ratios: BERT masks 15% of tokens (Devlin et al., NAACL 2019,
  DOI 10.18653/v1/N19-1423), MAE 75% of patches (He et al., CVPR 2022,
  arXiv:2111.06377).
* ``PretrainedConfig`` and ``PreTrainedModel`` are Hugging Face Transformers' (Wolf et
  al., EMNLP 2020 System Demonstrations, DOI 10.18653/v1/2020.emnlp-demos.6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from model.centroidal import CENTROIDAL_SIZE
from model.configuration_da_get import DaGetConfig
from model.modeling_da_get import (
    LENGTH_INDEX,
    MASKABLE_GROUPS,
    PRISMATIC_INDEX,
    TOTAL_MASS_INDEX,
    DaGetEncoder,
    DaGetEncoderLayer,
    normalize_mass_matrix,
    physics_units,
)
from model.physics_schema import feature_names

LOSS_NAMES = ("fk", "fd", "id", "response", "inverse_response", "motor", "effect")
EXTRA_LOSS_NAMES = ("operator", "consistency", "effect_rollout", "contact", "centroidal", "mask", "sysid")
ALL_LOSS_NAMES = LOSS_NAMES + EXTRA_LOSS_NAMES
HEAD_TYPES = ("mixer", "linear", "linear_residual")
ACCELERATION_METRICS = ("euclidean", "mass")
EFFECT_MODES = ("head", "rollout")
GRAVITY = 9.80665  # standard gravity in m/s^2, exact (NIST CODATA, https://physics.nist.gov/cgi-bin/cuu/Value?gn)
MOTOR_FLAG_COUNT = 2
MOTOR_MARGIN_COUNT = 5
ACTION_FEATURE_SIZE = 3


class DaGetPhysicsPretrainingConfig(PretrainedConfig):
    """Configure the encoder, the auxiliary heads, and the loss weights.

    Attributes:
        encoder (dict): ``DaGetConfig`` fields.
        head_hidden_size (int | None): Head MLP width; defaults to the encoder width.
        mixing_layers (int): Graph action-mixing layers in the FD, ID, response,
            inverse-response, and effect heads; 0 makes those heads per-joint.
        fk_orientation (bool): Supervise orientation in FK (False for the
            position-only first pass).
        loss_weights (dict[str, float]): Weight of each task in the total loss.
        lambda_position, lambda_rotation (float): FK term weights.
        lambda_feasibility, lambda_margin (float): Motor term weights.
        lambda_effect_position, lambda_effect_rotation, lambda_effect_velocity,
        lambda_effect_angular_velocity (float): Effect term weights.
        huber_delta (float): Huber transition point for margins.
        lambda_base_acceleration (float): Weight of the floating-base
            acceleration term inside the FD and response losses.
        epsilon (float): Stabilizer in normalizations.
        token_layout (dict): Observation token layout used by the data
            collator, saved for reproducibility.
        head_type (str): ``mixer``, ``linear`` or ``linear_residual`` (B1).
        interface_rank (int): Columns of the operator factors ``U``. 64 is above
            the largest DoF count in the data (G1: 29), so ``U U^T + diag`` can
            represent any symmetric positive semidefinite joint block.
        acceleration_metric (str): ``euclidean`` or ``mass`` (A1).
        effect_mode (str): ``head`` or ``rollout`` (B1, [H]).
        use_contact_head (bool): Contact-probability head (C2).
        use_centroidal_head (bool): Centroidal head on the pooled embedding (C3).
        use_sysid_head (bool): Spec-deviation head (E1), trained by spec noise.
        physics_mask_ratio (float): Probability of masking each rigid-body and
            actuator token in training (E3); 0 turns masked modeling off. BERT
            masks 15% of tokens (Devlin et al., NAACL 2019) and MAE 75% of patches
            (He et al., CVPR 2022): the two ends of the ablation.
    """

    model_type = "da_get_physics"

    def __init__(
        self,
        encoder: dict | None = None,
        head_hidden_size: int | None = None,
        mixing_layers: int = 1,
        fk_orientation: bool = True,
        loss_weights: dict[str, float] | None = None,
        lambda_position: float = 1.0,
        lambda_rotation: float = 1.0,
        lambda_feasibility: float = 1.0,
        lambda_margin: float = 1.0,
        lambda_effect_position: float = 1.0,
        lambda_effect_rotation: float = 1.0,
        lambda_effect_velocity: float = 1.0,
        lambda_effect_angular_velocity: float = 1.0,
        huber_delta: float = 1.0,
        epsilon: float = 1e-6,
        lambda_base_acceleration: float = 1.0,
        token_layout: dict | None = None,
        head_type: str = "mixer",
        interface_rank: int = 64,
        acceleration_metric: str = "euclidean",
        effect_mode: str = "head",
        use_contact_head: bool = False,
        use_centroidal_head: bool = False,
        use_sysid_head: bool = False,
        physics_mask_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Store settings; see the class attributes."""
        encoder = dict(encoder or {"input_size": 1})
        for key in ("cross_attention_layers", "physics_cross_attention_layers"):
            if isinstance(encoder.get(key), list):
                encoder[key] = tuple(encoder[key])
        self.encoder = encoder
        self.head_hidden_size = head_hidden_size
        self.mixing_layers = mixing_layers
        self.fk_orientation = fk_orientation
        weights = {name: 1.0 for name in ALL_LOSS_NAMES}
        weights.update(loss_weights or {})
        unknown = set(weights) - set(ALL_LOSS_NAMES)
        if unknown:
            hint = " (task F, rbd, was removed with the torque estimates)" if "rbd" in unknown else ""
            raise ValueError(f"unknown loss weights {sorted(unknown)}{hint}")
        self.loss_weights = weights
        self.lambda_position = lambda_position
        self.lambda_rotation = lambda_rotation
        self.lambda_feasibility = lambda_feasibility
        self.lambda_margin = lambda_margin
        self.lambda_effect_position = lambda_effect_position
        self.lambda_effect_rotation = lambda_effect_rotation
        self.lambda_effect_velocity = lambda_effect_velocity
        self.lambda_effect_angular_velocity = lambda_effect_angular_velocity
        self.huber_delta = huber_delta
        self.epsilon = epsilon
        self.lambda_base_acceleration = lambda_base_acceleration
        self.token_layout = token_layout or {}
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}")
        if acceleration_metric not in ACCELERATION_METRICS:
            raise ValueError(f"acceleration_metric must be one of {ACCELERATION_METRICS}")
        if effect_mode not in EFFECT_MODES:
            raise ValueError(f"effect_mode must be one of {EFFECT_MODES}")
        if effect_mode == "rollout" and head_type == "mixer":
            raise ValueError("effect_mode 'rollout' needs the physics interface (head_type linear or linear_residual)")
        if interface_rank <= 0:
            raise ValueError("interface_rank must be positive")
        if not 0 <= physics_mask_ratio < 1:
            raise ValueError("physics_mask_ratio must be in [0, 1)")
        self.head_type = head_type
        self.interface_rank = interface_rank
        self.acceleration_metric = acceleration_metric
        self.effect_mode = effect_mode
        self.use_contact_head = use_contact_head
        self.use_centroidal_head = use_centroidal_head
        self.use_sysid_head = use_sysid_head
        self.physics_mask_ratio = physics_mask_ratio
        super().__init__(**kwargs)

    @property
    def encoder_config(self) -> DaGetConfig:
        """Return the validated ``DaGetConfig``.

        Returns:
            DaGetConfig: Encoder configuration.
        """
        valid = {item.name for item in fields(DaGetConfig)}
        return DaGetConfig(**{key: value for key, value in self.encoder.items() if key in valid})

    @classmethod
    def from_encoder_config(cls, encoder: DaGetConfig, **kwargs: Any) -> DaGetPhysicsPretrainingConfig:
        """Build from a ``DaGetConfig``.

        Args:
            encoder (DaGetConfig): Encoder configuration.
            **kwargs (Any): Other fields.

        Returns:
            DaGetPhysicsPretrainingConfig: Configuration.
        """
        return cls(encoder=asdict(encoder), **kwargs)


@dataclass
class DaGetPhysicsPretrainingOutput(ModelOutput):
    """Outputs of ``DaGetForPhysicsPretraining``.

    Attributes:
        loss (torch.Tensor | None): Weighted total loss.
        joint_embeddings (torch.Tensor | None): ``[B, J, hidden]`` encoder output.
        pooled_embedding (torch.Tensor | None): ``[B, hidden]`` masked mean.
        loss_components (dict | None): Unweighted loss per task (detached).
        predictions (dict | None): Raw head outputs.
    """

    loss: torch.Tensor | None = None
    joint_embeddings: torch.Tensor | None = None
    pooled_embedding: torch.Tensor | None = None
    loss_components: dict[str, torch.Tensor] | None = None
    predictions: dict[str, torch.Tensor] | None = None


def _mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    """Return a LayerNorm-GELU MLP."""
    return nn.Sequential(
        nn.LayerNorm(input_size),
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, output_size),
    )


def rotation_from_6d(values: torch.Tensor) -> torch.Tensor:
    """Map the continuous 6D representation to a rotation matrix (Zhou et al., CVPR 2019).

    Args:
        values (torch.Tensor): ``[..., 6]``, two stacked column vectors.

    Returns:
        torch.Tensor: ``[..., 3, 3]`` rotation via Gram-Schmidt.
    """
    first = functional.normalize(values[..., :3], dim=-1)
    second = values[..., 3:]
    second = functional.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def geodesic_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the SO(3) geodesic angle between rotations (Huynh, J. Math. Imaging Vis., 2009).

    Args:
        prediction (torch.Tensor): ``[..., 3, 3]``.
        target (torch.Tensor): ``[..., 3, 3]``.

    Returns:
        torch.Tensor: ``[...]`` angle in rad.
    """
    trace = (prediction * target).sum(dim=(-1, -2))
    cosine = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine)


def masked_joint_mean(
    values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Average per-joint values over valid joints, then over samples.

    Args:
        values (torch.Tensor): ``[B, J]`` per-joint losses.
        mask (torch.Tensor): Bool ``[B, J]``.
        weights (torch.Tensor | None): ``[B]`` sample weights (label confidence).

    Returns:
        torch.Tensor: Scalar; zero (with gradient) when nothing is valid.
    """
    mask = mask.to(values.dtype)
    counts = mask.sum(dim=1)
    per_sample = (values * mask).sum(dim=1) / counts.clamp_min(1)
    valid = counts > 0
    if not torch.any(valid):
        return values.sum() * 0
    if weights is None:
        return per_sample[valid].mean()
    weights = weights.to(values.dtype)[valid]
    return (per_sample[valid] * weights).sum() / weights.sum().clamp_min(1e-12)


class GraphActionMixer(nn.Module):
    """Fuse per-joint embeddings with per-joint inputs and mix them over the graph.

    ``u_i = [H_i; E(x_i)]`` is projected to the encoder width and passed through
    a few graph self-attention layers with the structural bias, so every joint's
    prediction can depend on the inputs of the other joints.
    """

    def __init__(self, encoder_config: DaGetConfig, input_size: int, layers: int) -> None:
        """Build the mixer.

        Args:
            encoder_config (DaGetConfig): Encoder configuration (width, heads).
            input_size (int): Features per joint of x.
            layers (int): Graph self-attention layers; 0 keeps it per-joint.
        """
        super().__init__()
        hidden = encoder_config.hidden_size
        self.embed_input = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.fuse = nn.Linear(2 * hidden, hidden)
        mixing_config = replace(
            encoder_config,
            num_hidden_layers=max(layers, 1),
            cross_attention_layers=(),
            use_mass_encoding=False,
            use_physics_cross_attention=False,
            physics_cross_attention_layers=None,
            use_root_token=False,
            dimensionless_physics=False,
            use_morphology_modulation=False,
            physics_masking=False,
        )
        self.layers = nn.ModuleList(DaGetEncoderLayer(mixing_config, False) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        embeddings: torch.Tensor,
        inputs: torch.Tensor,
        distances: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mix inputs over the graph.

        Args:
            embeddings (torch.Tensor): ``[B, J, hidden]`` H.
            inputs (torch.Tensor): ``[B, J, input_size]`` x.
            distances (tuple): Spatial, ancestor, descendant distances.
            joint_mask (torch.Tensor): Bool ``[B, J]``.

        Returns:
            torch.Tensor: ``[B, J, hidden]``.
        """
        hidden = self.fuse(torch.cat((embeddings, self.embed_input(inputs)), dim=-1))
        hidden = hidden.masked_fill(~joint_mask[..., None], 0)
        for layer in self.layers:
            hidden = layer(hidden, distances, joint_mask, None, None, None)
        return self.norm(hidden)


def _masked_mean_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean of ``[B, J, H]`` over valid joints."""
    weights = mask.to(values.dtype)[..., None]
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def gauss_metric_error(error: torch.Tensor, mass_matrix: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Return per-sample ``e^T M_tilde e / N`` with the normalized mass matrix (Gauss's principle).

    The constrained acceleration minimizes ``(qdd - a)^T M (qdd - a)`` (Udwadia &
    Kalaba, Proc. R. Soc. A 439:407-410, 1992, DOI 10.1098/rspa.1992.0158). In
    per-joint normalized coordinates with unit diagonal the metric is
    ``M_ij / sqrt(M_ii M_jj)``, independent of the per-joint scales.

    Args:
        error (torch.Tensor): ``[B, J]`` normalized errors.
        mass_matrix (torch.Tensor): ``[B, J, J]`` joint mass matrix.
        mask (torch.Tensor): Bool ``[B, J]``.
        epsilon (float): Stabilizer.

    Returns:
        torch.Tensor: ``[B]`` metric per sample (0 without valid joints).
    """
    normalized = normalize_mass_matrix(mass_matrix, mask, epsilon, validate=False)
    pair = (mask[:, :, None] & mask[:, None, :]).to(error.dtype)
    masked = error * mask.to(error.dtype)
    value = (masked[:, :, None] * normalized * pair * masked[:, None, :]).sum((-1, -2))
    return value / mask.to(error.dtype).sum(-1).clamp_min(1)


class DaGetPhysicsInterface(nn.Module):
    """Linear, structured readout of the joint embeddings: operator, drift, and inverse (B1).

    In interface units, ``u_j = tau_j / a_j`` (a torque over the joint's reference
    torque) and ``y_r = a_r nu_dot_r / c`` (an acceleration times the row's reference
    force or torque, over the robot scale ``c = mean_j a_j^2 / M_jj``), the
    contact-constrained dynamics are ``y = f_tilde + G_tilde u`` with
    ``G_tilde = A G A / c``. ``G = [M^-1 - M^-1 J^T (J M^-1 J^T)^-1 J M^-1]`` restricted
    to the joint columns is the joint columns of a symmetric positive semidefinite
    matrix (``docs/dynamics.md``), so it is parameterized as the joint columns of
    ``W W^T + diag(softplus(d))``. ``W`` stacks six base rows (from the pooled
    embedding) and one row per joint (from that joint's embedding), as Deep
    Lagrangian Networks guarantee positive definiteness with ``L L^T`` (Lutter, Ritter
    & Peters, ICLR 2019, arXiv:1907.04490). The control-affine form ``y = f + G u`` is
    the manipulator equation itself (Featherstone, *Rigid Body Dynamics Algorithms*,
    2008) and the structure of Embed to Control (Watter et al., NIPS 2015,
    arXiv:1506.07365). The inverse ``K_tilde`` is parameterized the same way and trained
    toward a generalized inverse with ``G K G = G``, the first of Penrose's four
    conditions (Penrose, Proc. Cambridge Philos. Soc., 1955).

    The reference torque ``a_j`` is the joint's peak torque (the robot's median
    peak for passive joints); the base rows use ``m g`` and ``m g L``. Every scale
    comes from the encoder's inputs, so it is known for an unseen robot.
    """

    def __init__(self, hidden_size: int, rank: int) -> None:
        """Create the factor and drift projections.

        Args:
            hidden_size (int): Encoder width.
            rank (int): Columns of the factors.
        """
        super().__init__()
        self.rank = rank
        self.joint_factor = nn.Linear(hidden_size, rank)
        self.base_factor = nn.Linear(hidden_size, 6 * rank)
        self.joint_diagonal = nn.Linear(hidden_size, 1)
        self.inverse_joint_factor = nn.Linear(hidden_size, rank)
        self.inverse_base_factor = nn.Linear(hidden_size, 6 * rank)
        self.inverse_diagonal = nn.Linear(hidden_size, 1)
        self.drift = nn.Linear(hidden_size, 1)
        self.base_drift = nn.Linear(hidden_size, 6)

    def _matrix(self, joint: nn.Linear, base: nn.Linear, diagonal: nn.Linear, embeddings: torch.Tensor,
                pooled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the joint columns ``[B, 6 + J, J]`` of ``W W^T + diag``."""
        batch, joints, _ = embeddings.shape
        factors = joint(embeddings) * mask[..., None].to(embeddings.dtype) / self.rank**0.5
        base_factors = base(pooled).view(batch, 6, self.rank) / self.rank**0.5
        rows = torch.cat((base_factors, factors), dim=1)
        matrix = rows @ factors.transpose(1, 2)
        diagonal_values = functional.softplus(diagonal(embeddings).squeeze(-1)) * mask.to(embeddings.dtype)
        return matrix + torch.cat((matrix.new_zeros(batch, 6, joints), torch.diag_embed(diagonal_values)), dim=1)

    def forward(self, embeddings: torch.Tensor, pooled: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the operator, inverse and drift in interface units.

        Args:
            embeddings (torch.Tensor): ``[B, J, H]`` joint embeddings.
            pooled (torch.Tensor): ``[B, H]`` pooled embedding.
            mask (torch.Tensor): Bool ``[B, J]``.

        Returns:
            dict[str, torch.Tensor]: ``operator`` ``[B, 6 + J, J]`` (G_tilde),
                ``inverse`` ``[B, J, 6 + J]`` (K_tilde) and ``drift`` ``[B, 6 + J]``.
        """
        operator = self._matrix(self.joint_factor, self.base_factor, self.joint_diagonal, embeddings, pooled, mask)
        inverse = self._matrix(
            self.inverse_joint_factor, self.inverse_base_factor, self.inverse_diagonal, embeddings, pooled, mask
        ).transpose(1, 2)
        drift = torch.cat((self.base_drift(pooled), self.drift(embeddings).squeeze(-1) * mask.to(embeddings.dtype)), dim=1)
        return {"operator": operator, "inverse": inverse, "drift": drift}


def interface_units(batch: dict[str, torch.Tensor], epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reference forces ``a`` ``[B, 6 + J]`` and robot scales ``c`` ``[B]`` from the inputs.

    Args:
        batch (dict): Collator batch with ``actuator_features``, ``mass_matrix``,
            ``joint_mask`` and ``root_features`` (or ``total_mass`` and
            ``characteristic_length``).
        epsilon (float): Stabilizer.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``a`` and ``c``.
    """
    mask = batch["joint_mask"]
    peak = batch["actuator_features"][..., 0]
    actuated = (peak > 0) & mask
    big = torch.finfo(peak.dtype).max
    sorted_peaks = torch.where(actuated, peak, torch.full_like(peak, big)).sort(dim=1).values
    count = actuated.sum(1)
    median = sorted_peaks.gather(1, ((count - 1).clamp_min(0) // 2)[:, None]).squeeze(1)
    median = torch.where(count > 0, median, torch.ones_like(median))
    reference = torch.where(actuated, peak, median[:, None])
    reference = torch.where(mask, reference, torch.ones_like(reference))
    diagonal = batch["mass_matrix"].diagonal(dim1=1, dim2=2).clamp_min(epsilon)
    ratio = torch.where(mask, reference.pow(2) / diagonal, torch.zeros_like(diagonal))
    scale = ratio.sum(1) / mask.sum(1).clamp_min(1)
    if "root_features" in batch:
        mass = batch["root_features"][:, TOTAL_MASS_INDEX]
        length = batch["root_features"][:, LENGTH_INDEX]
    else:
        mass, length = batch["total_mass"], batch["characteristic_length"]
    force = (mass * GRAVITY)[:, None].expand(-1, 3)
    moment = (mass * GRAVITY * length)[:, None].expand(-1, 3)
    return torch.cat((force, moment, reference), dim=1), scale.clamp_min(epsilon)


class DaGetForPhysicsPretraining(PreTrainedModel):
    """GET encoder with the auxiliary physics heads A, B, C, R, R^-1, D and E.

    ``forward`` takes the batch produced by
    ``model.training.PhysicsDataCollator`` and returns the weighted loss, the
    per-task losses, and the joint embeddings. ``encode`` returns embeddings
    only, for downstream use.
    """

    config_class = DaGetPhysicsPretrainingConfig
    base_model_prefix = "encoder"
    main_input_name = "input_values"

    def __init__(self, config: DaGetPhysicsPretrainingConfig) -> None:
        """Build the encoder and heads.

        Args:
            config (DaGetPhysicsPretrainingConfig): Configuration.
        """
        super().__init__(config)
        encoder_config = config.encoder_config
        hidden = encoder_config.hidden_size
        head = config.head_hidden_size or hidden
        self.encoder = DaGetEncoder(encoder_config)
        self.fk_head = _mlp(hidden, head, 9)
        self.interface = None
        if config.head_type != "mixer":
            self.interface = DaGetPhysicsInterface(hidden, config.interface_rank)
        if config.head_type != "linear":
            self.fd_mixer = GraphActionMixer(encoder_config, 1, config.mixing_layers)
            self.fd_head = _mlp(hidden, head, 1)
            self.fd_base_head = _mlp(hidden, head, 6)
            self.id_mixer = GraphActionMixer(encoder_config, 7, config.mixing_layers)
            self.id_head = _mlp(hidden, head, 1)
            self.response_mixer = GraphActionMixer(encoder_config, 1, config.mixing_layers)
            self.response_head = _mlp(hidden, head, 1)
            self.response_base_head = _mlp(hidden, head, 6)
            self.inverse_response_mixer = GraphActionMixer(encoder_config, 7, config.mixing_layers)
            self.inverse_response_head = _mlp(hidden, head, 1)
        self.motor_mixer = GraphActionMixer(encoder_config, ACTION_FEATURE_SIZE, 0)
        self.motor_head = _mlp(hidden, head, MOTOR_FLAG_COUNT + MOTOR_MARGIN_COUNT)
        self.effect_mixer = GraphActionMixer(encoder_config, ACTION_FEATURE_SIZE, config.mixing_layers)
        self.effect_head = _mlp(hidden, head, 15)
        self.contact_head = _mlp(hidden, head, 1) if config.use_contact_head else None
        self.centroidal_head = _mlp(hidden, head, CENTROIDAL_SIZE) if config.use_centroidal_head else None
        self.sysid_head = _mlp(hidden, head, 4) if config.use_sysid_head else None
        self.mask_heads = None
        if config.physics_mask_ratio > 0:
            if not encoder_config.physics_masking:
                raise ValueError("physics_mask_ratio needs DaGetConfig.physics_masking")
            sizes = {"rigid_body": encoder_config.rigid_body_feature_size, "actuator": encoder_config.actuator_feature_size}
            self.mask_heads = nn.ModuleDict({group: nn.Linear(hidden, sizes[group]) for group in MASKABLE_GROUPS})
        self.post_init()
        if config.head_type == "linear_residual":
            # Residual heads start at zero, so the model starts as the linear interface.
            for module in (self.fd_head, self.fd_base_head, self.id_head, self.response_head,
                           self.response_base_head, self.inverse_response_head):
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)

    def _init_weights(self, module: nn.Module) -> None:
        """Keep PyTorch's default initialization of every module."""

    def encode(
        self,
        input_values: torch.Tensor,
        parent_indices: torch.Tensor,
        joint_mask: torch.Tensor,
        adjacency_matrix: torch.Tensor | None = None,
        spatial_distances: torch.Tensor | None = None,
        ancestor_distances: torch.Tensor | None = None,
        descendant_distances: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        rigid_body_features: torch.Tensor | None = None,
        actuator_features: torch.Tensor | None = None,
        dynamics_features: torch.Tensor | None = None,
        physics_memory=None,
        root_features: torch.Tensor | None = None,
        physics_token_mask: dict[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Return joint embeddings ``[B, J, hidden]``.

        Args:
            input_values (torch.Tensor): Observation tokens ``[B, J, F]``.
            parent_indices (torch.Tensor): ``[B, J]``.
            joint_mask (torch.Tensor): Bool ``[B, J]``.
            adjacency_matrix (torch.Tensor | None): ``[B, J, J]``.
            spatial_distances, ancestor_distances, descendant_distances
                (torch.Tensor | None): Precomputed graph distances.
            mass_matrix (torch.Tensor | None): ``[B, J, J]`` M(q_t).
            rigid_body_features, actuator_features, dynamics_features
                (torch.Tensor | None): Physics features.
            physics_memory (PhysicsMemory | None): Cached static physics.
            root_features (torch.Tensor | None): ``[B, 13]`` root features.
            physics_token_mask (dict | None): Masked physics tokens (training).

        Returns:
            torch.Tensor: Joint embeddings, zero on padding.
        """
        distances = None
        if spatial_distances is not None:
            distances = (spatial_distances, ancestor_distances, descendant_distances)
        if self.encoder.physics_tokenizers is None:
            # Physics Cross Attention is off (for example in an ablation): the
            # collator still provides physics features, so drop them here.
            rigid_body_features = actuator_features = dynamics_features = None
            physics_memory = root_features = physics_token_mask = None
        if not self.encoder.config.needs_root_features:
            root_features = None
        if not self.encoder.config.use_mass_encoding:
            mass_matrix = None
        return self.encoder(
            input_values,
            parent_indices,
            mass_matrix=mass_matrix,
            joint_mask=joint_mask,
            adjacency_matrix=adjacency_matrix,
            graph_distances=distances,
            rigid_body_features=rigid_body_features,
            actuator_features=actuator_features,
            dynamics_features=dynamics_features,
            physics_memory=physics_memory,
            root_features=root_features,
            physics_token_mask=physics_token_mask,
        )

    def _physics_token_mask(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Draw masked physics tokens in training (E3), never all joints of a robot."""
        if self.mask_heads is None or not self.training or "rigid_body_features" not in batch:
            return None
        mask = batch["joint_mask"]
        chosen = {}
        for group in MASKABLE_GROUPS:
            draw = (torch.rand(mask.shape, device=mask.device) < self.config.physics_mask_ratio) & mask
            # Keep at least one token of the group visible per robot.
            full = draw.sum(1) >= mask.sum(1)
            first = mask.float().argmax(1)
            draw[full, first[full]] = False
            chosen[group] = draw
        return chosen

    def _mask_targets(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the asinh-scaled features the masked tokens are reconstructed to."""
        encoder = self.encoder
        units = {group: None for group in MASKABLE_GROUPS}
        if encoder.config.dimensionless_physics:
            root = batch["root_features"]
            prismatic = batch["rigid_body_features"][..., PRISMATIC_INDEX] > 0.5
            units = {
                group: physics_units(group, root[:, TOTAL_MASS_INDEX], root[:, LENGTH_INDEX], prismatic)
                for group in MASKABLE_GROUPS
            }
        return {
            "rigid_body": encoder.physics_tokenizers["rigid_body"].scaled(batch["rigid_body_features"], units["rigid_body"]),
            "actuator": encoder.physics_tokenizers["actuator"].scaled(batch["actuator_features"], units["actuator"]),
        }

    def _acceleration_loss(self, error: torch.Tensor, mask: torch.Tensor, mass_matrix: torch.Tensor,
                           weights: torch.Tensor | None = None) -> torch.Tensor:
        """Average a normalized acceleration error with the configured metric."""
        if self.config.acceleration_metric == "euclidean":
            return masked_joint_mean(error.pow(2), mask, weights)
        per_sample = gauss_metric_error(error, mass_matrix, mask, self.config.epsilon)
        valid = mask.any(1)
        if not torch.any(valid):
            return error.sum() * 0
        if weights is None:
            return per_sample[valid].mean()
        weights = weights.to(error.dtype)[valid]
        return (per_sample[valid] * weights).sum() / weights.sum().clamp_min(1e-12)

    def forward(self, **batch: torch.Tensor) -> DaGetPhysicsPretrainingOutput:
        """Encode a batch and compute every auxiliary loss with labels present.

        Args:
            **batch (torch.Tensor): Collator output. Encoder inputs are required;
                each task's loss is computed only if its labels are present.

        Returns:
            DaGetPhysicsPretrainingOutput: Loss, components, embeddings, predictions.
        """
        config = self.config
        eps = config.epsilon
        joint_mask = batch["joint_mask"]
        token_mask = self._physics_token_mask(batch)
        embeddings = self.encode(**{**batch, "physics_token_mask": token_mask})
        distances = (
            batch["spatial_distances"],
            batch["ancestor_distances"],
            batch["descendant_distances"],
        )
        mask_float = joint_mask.to(embeddings.dtype)
        pooled = (embeddings * mask_float[..., None]).sum(1) / mask_float.sum(1, keepdim=True)
        losses: dict[str, torch.Tensor] = {}
        predictions: dict[str, torch.Tensor] = {}
        # Per-variant tasks read the state embedding of their row.
        rows = batch.get("variant_rows")
        if rows is None:
            rows = torch.arange(embeddings.shape[0], device=embeddings.device)
        row_embeddings = embeddings[rows]
        row_mask = joint_mask[rows]
        row_distances = tuple(item[rows] for item in distances)
        row_mask_float = row_mask.to(embeddings.dtype)
        mass_matrix = batch.get("mass_matrix")
        interface = units = None
        if self.interface is not None:
            interface = self.interface(embeddings, pooled, joint_mask)
            units = interface_units(batch, eps)
            predictions["interface_operator"] = interface["operator"]

        if "fk_position" in batch:
            output = self.fk_head(embeddings)
            position, rotation = output[..., :3], rotation_from_6d(output[..., 3:])
            predictions["fk_position"], predictions["fk_rotation"] = position, rotation
            per_joint = config.lambda_position * (position - batch["fk_position"]).pow(2).sum(-1)
            if config.fk_orientation:
                per_joint = per_joint + config.lambda_rotation * geodesic_distance(
                    rotation, batch["fk_rotation"]
                ).pow(2)
            losses["fk"] = masked_joint_mean(per_joint, joint_mask)

        normalized_acceleration = None
        if "fd_torque" in batch:
            joint_prediction = base_prediction = None
            if interface is not None:
                joint_prediction, base_prediction = self._interface_forward_dynamics(batch, interface, units, rows)
            if config.head_type != "linear":
                mixed = self.fd_mixer(row_embeddings, batch["fd_torque"][..., None], row_distances, row_mask)
                residual = self.fd_head(mixed).squeeze(-1)
                pooled_mixed = (mixed * row_mask_float[..., None]).sum(1) / row_mask_float.sum(1, keepdim=True)
                base_residual = self.fd_base_head(pooled_mixed)
                joint_prediction = residual if joint_prediction is None else joint_prediction + residual
                base_prediction = base_residual if base_prediction is None else base_prediction + base_residual
            normalized_acceleration = joint_prediction
            predictions["fd_acceleration"] = normalized_acceleration
            row_mass = None if mass_matrix is None else mass_matrix[rows]
            losses["fd"] = self._acceleration_loss(
                normalized_acceleration - batch["fd_target"], batch["fd_mask"], row_mass, batch.get("torque_label_weight")
            )
            if "fd_base_target" in batch and bool(batch["fd_base_mask"].any()):
                # Floating base: the generalized acceleration also has the base
                # linear and angular acceleration (normalized per robot).
                predictions["fd_base_acceleration"] = base_prediction
                base_error = (base_prediction - batch["fd_base_target"]).pow(2).mean(-1)
                base_mask = batch["fd_base_mask"].to(base_error.dtype)
                base_loss = (base_error * base_mask).sum() / base_mask.sum().clamp_min(1)
                losses["fd"] = losses["fd"] + config.lambda_base_acceleration * base_loss

        if "id_acceleration" in batch:
            base_acceleration = batch.get("id_base_acceleration")
            if base_acceleration is None:
                base_acceleration = embeddings.new_zeros((row_embeddings.shape[0], 6))
            torque = None
            if interface is not None:
                torque = self._interface_inverse_dynamics(batch, interface, units, rows, base_acceleration)
            if config.head_type != "linear":
                id_input = torch.cat(
                    (
                        batch["id_acceleration"][..., None],
                        base_acceleration[:, None, :].expand(-1, row_embeddings.shape[1], -1),
                    ),
                    dim=-1,
                )
                mixed = self.id_mixer(row_embeddings, id_input, row_distances, row_mask)
                residual = self.id_head(mixed).squeeze(-1)
                torque = residual if torque is None else torque + residual
            predictions["id_torque"] = torque
            error = torque - batch["id_target"]
            if "contact_projection" in batch:
                # Only the part of the torque that the motion determines is
                # supervised; internal contact forces are unobservable.
                scale = batch["torque_scale"] + eps
                error = (batch["contact_projection"] @ (error * scale)[..., None]).squeeze(-1) / scale
            losses["id"] = masked_joint_mean(error.pow(2), batch["id_mask"], batch.get("torque_label_weight"))

        if "response_torque" in batch:
            response = base_prediction = None
            if interface is not None:
                response, base_prediction = self._interface_response(batch, interface, units)
            if config.head_type != "linear":
                mixed = self.response_mixer(embeddings, batch["response_torque"][..., None], distances, joint_mask)
                residual = self.response_head(mixed).squeeze(-1)
                pooled_mixed = (mixed * mask_float[..., None]).sum(1) / mask_float.sum(1, keepdim=True)
                base_residual = self.response_base_head(pooled_mixed)
                response = residual if response is None else response + residual
                base_prediction = base_residual if base_prediction is None else base_prediction + base_residual
            predictions["response_acceleration"] = response
            loss = self._acceleration_loss(response - batch["response_target"], batch["response_mask"], mass_matrix)
            base_mask = batch["response_base_mask"].to(response.dtype)
            if bool(batch["response_base_mask"].any()):
                predictions["response_base_acceleration"] = base_prediction
                base_error = (base_prediction - batch["response_base_target"]).pow(2).mean(-1)
                base_loss = (base_error * base_mask).sum() / base_mask.sum().clamp_min(1)
                loss = loss + config.lambda_base_acceleration * base_loss
            losses["response"] = loss
            if interface is not None and "response_operator" in batch:
                losses["operator"], losses["consistency"] = self._operator_losses(batch, interface, units)

        if "inverse_response_input" in batch:
            change = None
            if interface is not None:
                change = self._interface_inverse_response(batch, interface, units)
            if config.head_type != "linear":
                mixed = self.inverse_response_mixer(embeddings, batch["inverse_response_input"], distances, joint_mask)
                residual = self.inverse_response_head(mixed).squeeze(-1)
                change = residual if change is None else change + residual
            predictions["inverse_response_torque"] = change
            scale = batch["response_torque_scale"] + eps
            error = change - batch["inverse_response_target"]
            projection = batch.get("response_projection", batch.get("contact_projection"))
            if projection is not None and projection.shape[0] == error.shape[0]:
                # Torque changes that only press the feet together leave the
                # accelerations unchanged, so they cannot be recovered from them.
                error = (projection @ (error * scale)[..., None]).squeeze(-1) / scale
            losses["inverse_response"] = masked_joint_mean(error.pow(2), batch["inverse_response_mask"])

        if "motor_input" in batch:
            mixed = self.motor_mixer(row_embeddings, batch["motor_input"], row_distances, row_mask)
            output = self.motor_head(mixed)
            logits, margins = output[..., :MOTOR_FLAG_COUNT], output[..., MOTOR_FLAG_COUNT:]
            predictions["motor_logits"], predictions["motor_margins"] = logits, margins
            bce = functional.binary_cross_entropy_with_logits(
                logits, batch["motor_flags"], reduction="none"
            ).mean(-1)
            margin_mask = batch["motor_margin_mask"].to(margins.dtype)
            huber = functional.huber_loss(
                margins, batch["motor_margins"], reduction="none", delta=config.huber_delta
            )
            huber = (huber * margin_mask).sum(-1) / margin_mask.sum(-1).clamp_min(1)
            losses["motor"] = masked_joint_mean(
                config.lambda_feasibility * bce + config.lambda_margin * huber, batch["motor_mask"]
            )

        if "effect_input" in batch:
            mixed = self.effect_mixer(row_embeddings, batch["effect_input"], row_distances, row_mask)
            output = self.effect_head(mixed)
            delta, rotation = output[..., :3], rotation_from_6d(output[..., 3:9])
            linear, angular = output[..., 9:12], output[..., 12:15]
            predictions.update(
                effect_delta_position=delta,
                effect_rotation=rotation,
                effect_linear_velocity=linear,
                effect_angular_velocity=angular,
            )
            per_joint = (
                config.lambda_effect_position * (delta - batch["effect_delta_position"]).pow(2).sum(-1)
                + config.lambda_effect_rotation
                * geodesic_distance(rotation, batch["effect_rotation"]).pow(2)
                + config.lambda_effect_velocity
                * (linear - batch["effect_linear_velocity"]).pow(2).sum(-1)
                + config.lambda_effect_angular_velocity
                * (angular - batch["effect_angular_velocity"]).pow(2).sum(-1)
            )
            losses["effect"] = masked_joint_mean(per_joint, batch["effect_mask"])
            if config.effect_mode == "rollout" and interface is not None and "structure" in batch:
                rollout = self._effect_rollout(batch, interface, units, rows)
                if rollout is not None:
                    losses["effect_rollout"] = rollout

        if self.contact_head is not None and "contact_target" in batch:
            logits = self.contact_head(embeddings).squeeze(-1)
            predictions["contact_logits"] = logits
            bce = functional.binary_cross_entropy_with_logits(logits, batch["contact_target"], reduction="none")
            losses["contact"] = masked_joint_mean(bce, batch["contact_mask"])

        if self.centroidal_head is not None and "centroidal_target" in batch:
            output = self.centroidal_head(pooled)
            predictions["centroidal"] = output
            centroidal_mask = batch["centroidal_mask"].to(output.dtype)
            squared = (output - batch["centroidal_target"]).pow(2) * centroidal_mask
            per_sample = squared.sum(-1) / centroidal_mask.sum(-1).clamp_min(1)
            losses["centroidal"] = per_sample.mean()

        if self.sysid_head is not None and "sysid_target" in batch:
            output = self.sysid_head(embeddings)
            predictions["sysid"] = output
            sysid_mask = batch["sysid_mask"].to(output.dtype)
            per_joint = ((output - batch["sysid_target"]).pow(2) * sysid_mask).sum(-1) / sysid_mask.sum(-1).clamp_min(1)
            losses["sysid"] = masked_joint_mean(per_joint, batch["sysid_mask"].any(-1))

        if token_mask is not None:
            targets = self._mask_targets(batch)
            total, count = embeddings.new_zeros(()), 0
            for group in MASKABLE_GROUPS:
                chosen = token_mask[group]
                if bool(chosen.any()):
                    error = (self.mask_heads[group](embeddings) - targets[group]).pow(2).mean(-1)
                    total = total + masked_joint_mean(error, chosen)
                    count += 1
            if count:
                losses["mask"] = total / count

        loss = None
        if losses:
            loss = sum(config.loss_weights.get(name, 0.0) * value for name, value in losses.items())
        return DaGetPhysicsPretrainingOutput(
            loss=loss,
            joint_embeddings=embeddings,
            pooled_embedding=pooled,
            loss_components={name: value.detach() for name, value in losses.items()},
            predictions=predictions,
        )

    # Interface readouts (B1). Units: u = tau / a, y = a nu_dot / c (see DaGetPhysicsInterface).

    @staticmethod
    def _to_interface(acceleration: torch.Tensor, references: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Map physical generalized accelerations ``[R, 6 + J]`` to interface units."""
        return acceleration * references / scale[:, None]

    @staticmethod
    def _from_interface(values: torch.Tensor, references: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Map interface accelerations ``[R, 6 + J]`` to physical units."""
        return values * scale[:, None] / references

    def _interface_forward_dynamics(self, batch, interface, units, rows) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FD predictions (normalized like the targets): ``y = f + G u``."""
        references, scale = units
        references, scale = references[rows], scale[rows]
        torque = batch["fd_torque"] * batch["variant_peak_torque"]
        u = torque / references[:, 6:]
        y = interface["drift"][rows] + (interface["operator"][rows] @ u[..., None]).squeeze(-1)
        acceleration = self._from_interface(y, references, scale)
        joint = acceleration[:, 6:] / batch["acceleration_scale"]
        base = acceleration[:, :6] / batch["base_acceleration_scale"]
        return joint, base

    def _interface_inverse_dynamics(self, batch, interface, units, rows, base_acceleration) -> torch.Tensor:
        """Return ID predictions over the continuous torque: ``u = K (y - f)``."""
        references, scale = units
        references, scale = references[rows], scale[rows]
        joint = batch["id_acceleration"] * batch["acceleration_scale"]
        base = base_acceleration * batch["base_acceleration_scale"]
        y = self._to_interface(torch.cat((base, joint), dim=1), references, scale)
        u = (interface["inverse"][rows] @ (y - interface["drift"][rows])[..., None]).squeeze(-1)
        return u * references[:, 6:] / (batch["torque_scale"] + self.config.epsilon)

    def _interface_response(self, batch, interface, units) -> tuple[torch.Tensor, torch.Tensor]:
        """Return R predictions (normalized like the targets): ``dy = G du``."""
        references, scale = units
        change = batch["response_torque"] * batch["response_torque_scale"]
        u = change / references[:, 6:]
        y = (interface["operator"] @ u[..., None]).squeeze(-1)
        acceleration = self._from_interface(y, references, scale)
        return acceleration[:, 6:] / batch["response_scale"], acceleration[:, :6] / batch["response_base_scale"]

    def _interface_inverse_response(self, batch, interface, units) -> torch.Tensor:
        """Return R^-1 predictions over the peak torque: ``du = K dy``."""
        references, scale = units
        inputs = batch["inverse_response_input"]
        joint = inputs[..., 0] * batch["response_scale"]
        base = inputs[:, 0, 1:7] * batch["response_base_scale"]
        y = self._to_interface(torch.cat((base, joint), dim=1), references, scale)
        u = (interface["inverse"] @ y[..., None]).squeeze(-1)
        return u * references[:, 6:] / (batch["response_torque_scale"] + self.config.epsilon)

    def _operator_losses(self, batch, interface, units) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the relative Frobenius errors of ``G_tilde`` and of ``G K G = G``."""
        references, scale = units
        operator = batch["response_operator"]
        target = operator * references[:, :, None] * references[:, None, 6:] / scale[:, None, None]
        joint_mask = batch["joint_mask"]
        rows_valid = torch.cat(
            (batch["response_base_mask"][:, None].expand(-1, 6), joint_mask), dim=1
        )
        entry_mask = (rows_valid[:, :, None] & joint_mask[:, None, :]).to(target.dtype)
        valid = batch["response_mask"].any(1)
        predicted = interface["operator"]
        if not torch.any(valid):
            zero = predicted.sum() * 0
            return zero, zero
        denominator = (target.pow(2) * entry_mask).sum((-1, -2)).clamp_min(self.config.epsilon)
        operator_error = ((predicted - target).pow(2) * entry_mask).sum((-1, -2)) / denominator
        product = predicted @ interface["inverse"] @ predicted
        own = (predicted.pow(2) * entry_mask).sum((-1, -2)).clamp_min(self.config.epsilon)
        consistency = ((product - predicted).pow(2) * entry_mask).sum((-1, -2)) / own
        return operator_error[valid].mean(), consistency[valid].mean()

    def _effect_rollout(self, batch, interface, units, rows) -> torch.Tensor | None:
        """Displacement and orientation of joint frames after a one-step rollout ([H], B1).

        ``qd' = qd + dt qdd_hat`` and ``q' = q + dt qd'`` with the interface's
        accelerations, then the analytic forward kinematics of each robot, plus the
        base motion from the predicted base acceleration on a floating base. Only rows
        whose torque is known (exact or measured) are used.
        """
        if "joint_positions" not in batch or "frame_dt" not in batch:
            return None
        labeled = batch["torque_label_weight"] > 0
        if not bool(labeled.any()):
            return None
        joint, base = self._interface_forward_dynamics(batch, interface, units, rows)
        acceleration = joint * batch["acceleration_scale"]
        base_acceleration = base * batch["base_acceleration_scale"]
        structure = batch["structure"]
        q = batch["joint_positions"][rows]
        qd = batch["joint_velocities"][rows]
        dt = batch["frame_dt"][rows]
        length = batch["characteristic_length"][rows]
        velocity_next = qd + dt[:, None] * acceleration
        q_next = q + dt[:, None] * velocity_next
        floating = batch.get("base_twist")
        errors = []
        for row in torch.nonzero(labeled).squeeze(-1).tolist():
            graph = structure.graphs[int(rows[row])]
            count = graph.num_joints
            dtype = q.dtype
            poses = graph.forward_kinematics(q[row, :count].to(torch.float64)).to(dtype)
            poses_next = graph.forward_kinematics(q_next[row, :count].to(torch.float64)).to(dtype)
            position_next = poses_next[..., :3, 3]
            rotation_next = poses_next[..., :3, :3]
            if graph.floating_base and floating is not None:
                twist = floating[int(rows[row])]
                omega_next = twist[:3] + dt[row] * base_acceleration[row, 3:]
                velocity_base = twist[3:] + dt[row] * (
                    base_acceleration[row, :3] - torch.linalg.cross(twist[:3], twist[3:], dim=-1)
                )
                angle = omega_next * dt[row]
                relative = _exp_so3(angle)
                position_next = position_next @ relative.T + velocity_base * dt[row]
                rotation_next = relative @ rotation_next
            delta = (position_next - poses[..., :3, 3]) / length[row]
            delta = delta / batch["effect_scales"][row, 0]
            mask = batch["effect_mask"][row, :count].to(dtype)
            target_delta = batch["effect_delta_position"][row, :count]
            target_rotation = batch["effect_rotation"][row, :count]
            per_joint = (delta - target_delta).pow(2).sum(-1) + geodesic_distance(rotation_next, target_rotation).pow(2)
            if bool(mask.any()):
                errors.append((per_joint * mask).sum() / mask.sum())
        if not errors:
            return None
        return torch.stack(errors).mean()


def _exp_so3(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula for a rotation vector ``[3]`` (differentiable; Lynch & Park 2017, Prop. 3.11)."""
    angle = vector.norm().clamp_min(1e-12)
    axis = vector / angle
    zero = vector.new_zeros(())
    skew = torch.stack(
        (
            torch.stack((zero, -axis[2], axis[1])),
            torch.stack((axis[2], zero, -axis[0])),
            torch.stack((-axis[1], axis[0], zero)),
        )
    )
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)
