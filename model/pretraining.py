"""Pretrain the GET encoder with auxiliary physics heads (Hugging Face compatible).

The encoder produces one embedding ``H_{t,i}`` per joint. Six auxiliary tasks
train it; their weights are set in ``GetPhysicsPretrainingConfig``:

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
   robot.
C. Inverse dynamics: ``D_ID(H, qdd^desired, base acc^desired) ->
   tau^required`` with the full desired acceleration vector mixed over the
   graph; the desired base acceleration is given to every joint. For
   estimated torques of robots in contact, the error is projected onto the
   torque directions the motion determines (``id_projection``): internal
   contact forces cannot be observed and are not supervised.
   ``L_ID = 1/N_e sum_i ((tau_hat_i - tau_i) / (tau_i^cont + eps))^2``.
D. Motor feasibility and capability margins: ``D_motor(H_i, a_i^proposal)``
   predicts peak and continuous feasibility flags and five margins.
   ``L_motor = lambda_f BCE + lambda_m Huber``.
E. Cross-embodiment action effect: ``D_effect(H, a_t)`` predicts, for each
   end-effector frame, the displacement ``Delta p``, the next orientation
   ``R_{t+1}``, and the next linear and angular velocity.
F. Rigid-body dynamics consistency (optional, weight 0 by default): the
   residual ``M qdd_hat + c + g + tau_f - tau^applied`` of the FD prediction,
   weighted by ``1 / (tau^cont + eps)``.

All per-joint losses are averaged over the valid joints of each sample first
(``1/N_e``) and then over samples, so robots with many joints do not dominate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from model.configuration_get import DaGetConfig
from model.modeling_get import DaGetEncoder, GetEncoderLayer

LOSS_NAMES = ("fk", "fd", "id", "motor", "effect", "rbd")
MOTOR_FLAG_COUNT = 2
MOTOR_MARGIN_COUNT = 5
ACTION_FEATURE_SIZE = 3


class GetPhysicsPretrainingConfig(PretrainedConfig):
    """Configure the encoder, the auxiliary heads, and the loss weights.

    Attributes:
        encoder (dict): ``DaGetConfig`` fields.
        head_hidden_size (int | None): Head MLP width; defaults to the encoder width.
        mixing_layers (int): Graph action-mixing layers in the FD, ID, and effect
            heads; 0 makes those heads per-joint.
        fk_orientation (bool): Supervise orientation in FK (False for the
            position-only first pass).
        loss_weights (dict[str, float]): Weight of each task in the total loss.
        lambda_position, lambda_rotation (float): FK term weights.
        lambda_feasibility, lambda_margin (float): Motor term weights.
        lambda_effect_position, lambda_effect_rotation, lambda_effect_velocity,
        lambda_effect_angular_velocity (float): Effect term weights.
        huber_delta (float): Huber transition point for margins.
        lambda_base_acceleration (float): Weight of the floating-base
            acceleration term inside the FD loss.
        epsilon (float): Stabilizer in normalizations.
        token_layout (dict): Observation token layout used by the data
            collator, saved for reproducibility.
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
        weights = {"fk": 1.0, "fd": 1.0, "id": 1.0, "motor": 1.0, "effect": 1.0, "rbd": 0.0}
        weights.update(loss_weights or {})
        unknown = set(weights) - set(LOSS_NAMES)
        if unknown:
            raise ValueError(f"unknown loss weights {sorted(unknown)}")
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
    def from_encoder_config(cls, encoder: DaGetConfig, **kwargs: Any) -> GetPhysicsPretrainingConfig:
        """Build from a ``DaGetConfig``.

        Args:
            encoder (DaGetConfig): Encoder configuration.
            **kwargs (Any): Other fields.

        Returns:
            GetPhysicsPretrainingConfig: Configuration.
        """
        return cls(encoder=asdict(encoder), **kwargs)


@dataclass
class GetPhysicsPretrainingOutput(ModelOutput):
    """Outputs of ``GetForPhysicsPretraining``.

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
    """Map the continuous 6D representation to a rotation matrix.

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
    """Return the SO(3) geodesic angle between rotations.

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
        )
        self.layers = nn.ModuleList(GetEncoderLayer(mixing_config, False) for _ in range(layers))
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


class GetForPhysicsPretraining(PreTrainedModel):
    """GET encoder with the auxiliary physics heads A-F.

    ``forward`` takes the batch produced by
    ``model.training.PhysicsDataCollator`` and returns the weighted loss, the
    per-task losses, and the joint embeddings. ``encode`` returns embeddings
    only, for downstream use.
    """

    config_class = GetPhysicsPretrainingConfig
    base_model_prefix = "encoder"
    main_input_name = "input_values"

    def __init__(self, config: GetPhysicsPretrainingConfig) -> None:
        """Build the encoder and heads.

        Args:
            config (GetPhysicsPretrainingConfig): Configuration.
        """
        super().__init__(config)
        encoder_config = config.encoder_config
        hidden = encoder_config.hidden_size
        head = config.head_hidden_size or hidden
        self.encoder = DaGetEncoder(encoder_config)
        self.fk_head = _mlp(hidden, head, 9)
        self.fd_mixer = GraphActionMixer(encoder_config, 1, config.mixing_layers)
        self.fd_head = _mlp(hidden, head, 1)
        self.fd_base_head = _mlp(hidden, head, 6)
        self.id_mixer = GraphActionMixer(encoder_config, 7, config.mixing_layers)
        self.id_head = _mlp(hidden, head, 1)
        self.motor_mixer = GraphActionMixer(encoder_config, ACTION_FEATURE_SIZE, 0)
        self.motor_head = _mlp(hidden, head, MOTOR_FLAG_COUNT + MOTOR_MARGIN_COUNT)
        self.effect_mixer = GraphActionMixer(encoder_config, ACTION_FEATURE_SIZE, config.mixing_layers)
        self.effect_head = _mlp(hidden, head, 15)
        self.post_init()

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
            physics_memory = None
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
        )

    def forward(self, **batch: torch.Tensor) -> GetPhysicsPretrainingOutput:
        """Encode a batch and compute every auxiliary loss with labels present.

        Args:
            **batch (torch.Tensor): Collator output. Encoder inputs are required;
                each task's loss is computed only if its labels are present.

        Returns:
            GetPhysicsPretrainingOutput: Loss, components, embeddings, predictions.
        """
        config = self.config
        eps = config.epsilon
        joint_mask = batch["joint_mask"]
        embeddings = self.encode(**batch)
        distances = (
            batch["spatial_distances"],
            batch["ancestor_distances"],
            batch["descendant_distances"],
        )
        mask_float = joint_mask.to(embeddings.dtype)
        pooled = (embeddings * mask_float[..., None]).sum(1) / mask_float.sum(1, keepdim=True)
        losses: dict[str, torch.Tensor] = {}
        predictions: dict[str, torch.Tensor] = {}

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
            mixed = self.fd_mixer(embeddings, batch["fd_torque"][..., None], distances, joint_mask)
            normalized_acceleration = self.fd_head(mixed).squeeze(-1)
            predictions["fd_acceleration"] = normalized_acceleration
            losses["fd"] = masked_joint_mean(
                (normalized_acceleration - batch["fd_target"]).pow(2),
                batch["fd_mask"],
                batch.get("torque_label_weight"),
            )
            if "fd_base_target" in batch and bool(batch["fd_base_mask"].any()):
                # Floating base: the generalized acceleration also has the base
                # linear and angular acceleration (normalized per robot).
                pooled_mixed = (mixed * mask_float[..., None]).sum(1) / mask_float.sum(1, keepdim=True)
                base_prediction = self.fd_base_head(pooled_mixed)
                predictions["fd_base_acceleration"] = base_prediction
                base_error = (base_prediction - batch["fd_base_target"]).pow(2).mean(-1)
                base_mask = batch["fd_base_mask"].to(base_error.dtype)
                base_loss = (base_error * base_mask).sum() / base_mask.sum().clamp_min(1)
                losses["fd"] = losses["fd"] + config.lambda_base_acceleration * base_loss

        if "id_acceleration" in batch:
            base_acceleration = batch.get("id_base_acceleration")
            if base_acceleration is None:
                base_acceleration = embeddings.new_zeros((embeddings.shape[0], 6))
            id_input = torch.cat(
                (
                    batch["id_acceleration"][..., None],
                    base_acceleration[:, None, :].expand(-1, embeddings.shape[1], -1),
                ),
                dim=-1,
            )
            mixed = self.id_mixer(embeddings, id_input, distances, joint_mask)
            torque = self.id_head(mixed).squeeze(-1)
            predictions["id_torque"] = torque
            error = torque - batch["id_target"]
            if "id_projection" in batch:
                # Only the part of the torque that the motion determines is
                # supervised; internal contact forces are unobservable.
                scale = batch["torque_scale"] + eps
                error = (batch["id_projection"] @ (error * scale)[..., None]).squeeze(-1) / scale
            losses["id"] = masked_joint_mean(error.pow(2), batch["id_mask"], batch.get("torque_label_weight"))

        if "motor_input" in batch:
            mixed = self.motor_mixer(embeddings, batch["motor_input"], distances, joint_mask)
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
            mixed = self.effect_mixer(embeddings, batch["effect_input"], distances, joint_mask)
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

        if config.loss_weights.get("rbd", 0.0) > 0 and normalized_acceleration is not None:
            acceleration = normalized_acceleration * batch["acceleration_scale"]
            residual = (
                (batch["mass_matrix"] @ acceleration[..., None]).squeeze(-1)
                + batch["coriolis"]
                + batch["gravity_torque"]
                + batch["friction_torque"]
                - batch["applied_torque"]
            ) / (batch["torque_scale"] + eps)
            # The residual has no contact or base-coupling terms, so it is only
            # valid for fixed-base samples.
            losses["rbd"] = masked_joint_mean(residual.pow(2), batch.get("rbd_mask", batch["fd_mask"]))

        loss = None
        if losses:
            loss = sum(config.loss_weights.get(name, 0.0) * value for name, value in losses.items())
        return GetPhysicsPretrainingOutput(
            loss=loss,
            joint_embeddings=embeddings,
            pooled_embedding=pooled,
            loss_components={name: value.detach() for name, value in losses.items()},
            predictions=predictions,
        )
