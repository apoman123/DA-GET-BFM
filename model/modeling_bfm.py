"""BFM-Zero (FB-CPR) networks whose actor and critics read GET joint embeddings.

"Da" stands for dynamic-aware: the GET embeddings these networks read carry the
robot's dynamics.

* ``BackwardMap`` B(s) and ``Discriminator`` D(s, z) keep the BFM-Zero MLPs.
* ``DaGetBfmActor`` π(a | s, z) and ``DaGetForwardMap`` F(s, a, z) (also used for
  the discriminator critic and the auxiliary critic, with one output) replace
  the flat-state MLPs. They take the ``[batch, joints, hidden]`` output of
  ``DaGetEncoder``, cross-attend from joint tokens to latent tokens made from z
  (and optional context tokens), mix joints with self-attention, and either
  decode one action per DoF (actor) or pool the joints (forward map, critics).

Nothing depends on the joint count or order, so one set of weights serves any
embodiment the GET encoder can tokenize.

References and attribution:

* BFM-Zero: Y. Li, Z. Luo, T. Zhang, et al. "BFM-Zero: A Promptable Behavioral
  Foundation Model for Humanoid Control Using Unsupervised Reinforcement
  Learning." arXiv:2511.04131, 2025. https://lecar-lab.github.io/BFM-Zero/
* FB-CPR / Meta Motivo: A. Tirinzoni, A. Touati, J. Farebrother, M. Guzek,
  A. Kanervisto, Y. Xu, A. Lazaric, M. Pirotta. "Zero-Shot Whole-Body Humanoid
  Control via Behavioral Foundation Models." ICLR 2025, arXiv:2504.11054.
* Forward-backward representations: A. Touati, Y. Ollivier. "Learning One
  Representation to Optimize All Rewards." NeurIPS 2021.
* GET-Zero: A. Patel, S. Song. "GET-Zero: Graph Embodiment Transformer for
  Zero-shot Embodiment Generalization." ICRA 2025.
* ``BackwardMap``, ``Discriminator``, ``TruncatedNormal``, ``bfm_weight_init``,
  the z projection and the inference methods of ``DaGetBfmModel`` are adapted
  from ``humanoidverse/agents/nn_models.py`` and ``agents/fb/model.py`` of
  https://github.com/LeCAR-Lab/BFM-Zero (Copyright (c) Meta Platforms, Inc.
  and affiliates, CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/).
  Changes: rewritten in this repo's style, dict inputs selected by config keys,
  a padding mask on ``TruncatedNormal``, vectorized ``tracking_inference``.
  The truncated normal itself follows DrQ-v2 (Yarats et al., ICLR 2022) and
  the clipped target noise follows TD3 (Fujimoto et al., ICML 2018).
* The GET-conditioned actor and forward map (latent tokens, joint-to-latent
  cross-attention, per-DoF action head, joint readout) are new in this repo.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal

from model.configuration_bfm import BfmConfig
from model.transformer import AttentionBlock, CrossAttentionBlock, NumericTokenizer, SinusoidalPositionEmbedding

Observation = torch.Tensor | Mapping[str, torch.Tensor]


def project_to_sphere(values: torch.Tensor) -> torch.Tensor:
    """Scale the last dimension to norm ``sqrt(dim)`` (BFM-Zero's z projection)."""
    return math.sqrt(values.shape[-1]) * F.normalize(values, dim=-1)


def select_inputs(observation: Observation, keys: tuple[str, ...]) -> torch.Tensor:
    """Concatenate ``keys`` of a dict observation; return a tensor unchanged."""
    if isinstance(observation, Mapping):
        return torch.cat([observation[key] for key in keys], dim=-1)
    return observation


def bfm_weight_init(module: nn.Module) -> None:
    """Orthogonal weights and zero biases for linear layers, as BFM-Zero's ``weight_init``."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SphereNorm(nn.Module):
    """Module form of ``project_to_sphere``."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Project ``values`` onto the sphere of radius ``sqrt(dim)``."""
        return project_to_sphere(values)


class TruncatedNormal(pyd.Normal):
    """Normal policy clamped to [low, high] with straight-through gradients (BFM-Zero).

    Args:
        loc (torch.Tensor): Mean action, ``[batch, joints]`` or ``[batch, joints, A]``.
        scale (torch.Tensor): Standard deviation, same shape.
        low (float): Lower action bound.
        high (float): Upper action bound.
        eps (float): Margin kept from the bounds.
        mask (torch.Tensor | None): Bool ``[batch, joints]``; padded joints sample 0.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
        mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low, self.high, self.eps = low, high, eps
        if mask is not None and mask.ndim < loc.ndim:
            mask = mask.reshape(*mask.shape, *([1] * (loc.ndim - mask.ndim))).expand_as(loc)
        self.mask = mask

    def _clamp(self, values: torch.Tensor) -> torch.Tensor:
        """Clamp in the forward pass, pass gradients straight through."""
        clamped = torch.clamp(values, self.low + self.eps, self.high - self.eps)
        return values - values.detach() + clamped.detach()

    def sample(self, clip: float | None = None, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw a reparameterized action; ``clip`` bounds the noise (TD3 style)."""
        shape = self._extended_shape(sample_shape)
        noise = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device) * self.scale
        if clip is not None:
            noise = torch.clamp(noise, -clip, clip)
        action = self._clamp(self.loc + noise)
        return action if self.mask is None else action.masked_fill(~self.mask, 0)


class BackwardMap(nn.Module):
    """Backward representation B(s) → z: BFM-Zero's MLP on the flat (privileged) state.

    Linear → LayerNorm → Tanh, then ``backward_hidden_layers - 1`` Linear → ReLU,
    then Linear to ``z_dim`` and an optional projection to the sphere.
    """

    def __init__(self, config: BfmConfig) -> None:
        super().__init__()
        self.config = config
        width = config.backward_hidden_size
        layers: list[nn.Module] = [nn.Linear(config.backward_input_size, width), nn.LayerNorm(width), nn.Tanh()]
        for _ in range(config.backward_hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, config.z_dim))
        if config.backward_norm:
            layers.append(SphereNorm())
        self.net = nn.Sequential(*layers)

    def forward(self, observation: Observation) -> torch.Tensor:
        """Embed states ``[..., backward_input_size]`` (or a dict of them) into ``[..., z_dim]``."""
        return self.net(select_inputs(observation, self.config.backward_input_keys))


class Discriminator(nn.Module):
    """FB-CPR discriminator D(s, z): BFM-Zero's MLP on the flat state and the latent.

    Linear → LayerNorm → Tanh, then ``discriminator_hidden_layers - 1`` Linear → ReLU,
    then Linear to one logit. ``compute_reward`` gives the CPR reward log D / (1 - D).
    """

    def __init__(self, config: BfmConfig) -> None:
        super().__init__()
        self.config = config
        width = config.discriminator_hidden_size
        layers: list[nn.Module] = [nn.Linear(config.discriminator_size + config.z_dim, width), nn.LayerNorm(width), nn.Tanh()]
        for _ in range(config.discriminator_hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 1))
        self.trunk = nn.Sequential(*layers)

    def compute_logits(self, observation: Observation, z: torch.Tensor) -> torch.Tensor:
        """Return logits ``[batch, 1]`` that (s, z) comes from the expert data."""
        state = select_inputs(observation, self.config.discriminator_input_keys)
        return self.trunk(torch.cat([z, state], dim=-1))

    def forward(self, observation: Observation, z: torch.Tensor) -> torch.Tensor:
        """Return the expert probability ``[batch, 1]``."""
        return torch.sigmoid(self.compute_logits(observation, z))

    def compute_reward(self, observation: Observation, z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Return the CPR reward ``log D - log(1 - D)`` of shape ``[batch, 1]``."""
        probability = self.forward(observation, z).clamp(eps, 1 - eps)
        return probability.log() - (1 - probability).log()


class LatentMemory(nn.Module):
    """Turn z (and optional context tokens) into key/value tokens for joint cross-attention.

    Each latent vector becomes ``num_latent_tokens`` tokens: a linear map to
    ``num_latent_tokens * hidden_size`` plus a learned slot embedding. A sequence
    of latents (for example B of future frames) also gets a sinusoidal position.
    """

    def __init__(self, config: BfmConfig) -> None:
        super().__init__()
        self.num_tokens, self.hidden_size = config.num_latent_tokens, config.hidden_size
        self.latent_projection = nn.Linear(config.z_dim, config.num_latent_tokens * config.hidden_size)
        self.slot_embedding = nn.Parameter(torch.randn(config.num_latent_tokens, config.hidden_size) * 0.02)
        self.position_embedding = SinusoidalPositionEmbedding(config.hidden_size)
        self.context_projection = (
            nn.Linear(config.context_size, config.hidden_size) if config.context_size is not None else None
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        z: torch.Tensor,
        latent_mask: torch.Tensor | None = None,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build memory tokens.

        Args:
            z (torch.Tensor): ``[batch, z_dim]`` or a latent sequence ``[batch, L, z_dim]``.
            latent_mask (torch.Tensor | None): Bool ``[batch, L]`` of valid latents.
            context_states (torch.Tensor | None): ``[batch, C, context_size]`` extra tokens.
            context_mask (torch.Tensor | None): Bool ``[batch, C]`` of valid context tokens.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tokens ``[batch, M, hidden]`` and bool mask ``[batch, M]``.
        """
        if z.ndim == 2:
            z = z[:, None]
        batch, length, _ = z.shape
        tokens = self.latent_projection(z).view(batch, length, self.num_tokens, self.hidden_size) + self.slot_embedding
        if length > 1:
            positions = torch.arange(length, device=z.device).expand(batch, length)
            tokens = tokens + self.position_embedding(positions).to(tokens.dtype)[:, :, None]
        tokens = tokens.reshape(batch, length * self.num_tokens, self.hidden_size)
        if latent_mask is None:
            latent_mask = torch.ones(batch, length, dtype=torch.bool, device=z.device)
        mask = latent_mask.repeat_interleave(self.num_tokens, dim=1)
        if context_states is not None:
            if self.context_projection is None:
                raise ValueError("context_states given but config.context_size is None")
            if context_mask is None:
                context_mask = torch.ones(context_states.shape[:2], dtype=torch.bool, device=z.device)
            tokens = torch.cat([tokens, self.context_projection(context_states)], dim=1)
            mask = torch.cat([mask, context_mask], dim=1)
        return self.norm(tokens), mask


class LatentConditionedBlock(nn.Module):
    """Joint tokens cross-attend to latent memory, self-attend across joints, then feed forward."""

    def __init__(self, config: BfmConfig) -> None:
        super().__init__()
        width, heads = config.hidden_size, config.num_attention_heads
        self.cross_attention = CrossAttentionBlock(width, heads, dropout=config.dropout)
        self.self_attention = AttentionBlock(width, heads, dropout=config.dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, width),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, hidden_states: torch.Tensor, joint_mask: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor
    ) -> torch.Tensor:
        """Update joint tokens ``[batch, joints, hidden]``; padded joints stay zero."""
        hidden_states = self.cross_attention(hidden_states, memory, context_padding_mask=~memory_mask)
        hidden_states = self.self_attention(hidden_states, key_padding_mask=~joint_mask)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        return hidden_states.masked_fill(~joint_mask[..., None], 0)


def _joint_mask(joint_embeddings: torch.Tensor, joint_mask: torch.Tensor | None) -> torch.Tensor:
    """Validate embeddings ``[batch, joints, E]`` and default the mask to all joints."""
    if joint_embeddings.ndim != 3:
        raise ValueError("joint_embeddings must be [batch, joints, embedding_size]")
    if joint_mask is None:
        return torch.ones(joint_embeddings.shape[:2], dtype=torch.bool, device=joint_embeddings.device)
    if joint_mask.shape != joint_embeddings.shape[:2] or joint_mask.dtype != torch.bool:
        raise ValueError("joint_mask must be bool [batch, joints]")
    return joint_mask


class DaGetBfmActor(nn.Module):
    """z-conditioned policy π(a | s, z) that outputs one action per DoF.

    GET joint embeddings are projected to ``hidden_size``, refined by
    ``actor_layers`` latent-conditioned blocks, and decoded per joint into
    ``tanh`` means of a truncated normal, as in BFM-Zero's actor.
    """

    def __init__(self, config: BfmConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.embedding_projection = nn.Sequential(nn.LayerNorm(config.embedding_size), nn.Linear(config.embedding_size, width))
        self.memory = LatentMemory(config)
        self.layers = nn.ModuleList(LatentConditionedBlock(config) for _ in range(config.actor_layers))
        self.action_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.Mish(), nn.Linear(width, config.action_size_per_joint)
        )

    def forward(
        self,
        joint_embeddings: torch.Tensor,
        z: torch.Tensor,
        std: float,
        joint_mask: torch.Tensor | None = None,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
    ) -> TruncatedNormal:
        """Return the action distribution.

        Args:
            joint_embeddings (torch.Tensor): ``DaGetEncoder`` output ``[batch, joints, embedding_size]``.
            z (torch.Tensor): Latent ``[batch, z_dim]`` or latent sequence ``[batch, L, z_dim]``.
            std (float): Policy standard deviation.
            joint_mask (torch.Tensor | None): Bool ``[batch, joints]`` of real DoFs.
            context_states (torch.Tensor | None): Optional ``[batch, C, context_size]`` tokens.
            context_mask (torch.Tensor | None): Bool ``[batch, C]``.
            latent_mask (torch.Tensor | None): Bool ``[batch, L]``.

        Returns:
            TruncatedNormal: Over ``[batch, joints]`` (or ``[batch, joints, A]``); padded joints are 0.
        """
        joint_mask = _joint_mask(joint_embeddings, joint_mask)
        memory, memory_mask = self.memory(z, latent_mask, context_states, context_mask)
        hidden_states = self.embedding_projection(joint_embeddings).masked_fill(~joint_mask[..., None], 0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, joint_mask, memory, memory_mask)
        mean = torch.tanh(self.action_head(hidden_states))
        if self.config.action_size_per_joint == 1:
            mean = mean.squeeze(-1)
        mask = joint_mask if mean.ndim == 2 else joint_mask[..., None].expand_as(mean)
        mean = mean.masked_fill(~mask, 0)
        return TruncatedNormal(mean, torch.full_like(mean, std), mask=joint_mask)


class DaGetForwardNetwork(nn.Module):
    """One ensemble member of F(s, a, z): joint tokens with their action, conditioned on z, pooled.

    Each joint token is its projected GET embedding plus an embedding of the
    action applied to that DoF. After the latent-conditioned blocks, a learned
    query attends over the joints; its output and the masked mean are mapped to
    ``output_size``.
    """

    def __init__(self, config: BfmConfig, output_size: int) -> None:
        super().__init__()
        width = config.hidden_size
        self.embedding_projection = nn.Sequential(nn.LayerNorm(config.embedding_size), nn.Linear(config.embedding_size, width))
        self.action_embedding = NumericTokenizer(config.action_size_per_joint, width, width)
        self.memory = LatentMemory(config)
        self.layers = nn.ModuleList(LatentConditionedBlock(config) for _ in range(config.forward_layers))
        self.readout_query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.readout = CrossAttentionBlock(width, config.num_attention_heads, dropout=config.dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, config.intermediate_size), nn.Mish(),
            nn.Linear(config.intermediate_size, output_size),
        )

    def forward(
        self,
        joint_embeddings: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        joint_mask: torch.Tensor,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[batch, output_size]``; see ``DaGetForwardMap.forward`` for arguments."""
        memory, memory_mask = self.memory(z, latent_mask, context_states, context_mask)
        if action.ndim == 2:
            action = action[..., None]
        hidden_states = self.embedding_projection(joint_embeddings) + self.action_embedding(action.to(joint_embeddings.dtype))
        hidden_states = hidden_states.masked_fill(~joint_mask[..., None], 0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, joint_mask, memory, memory_mask)
        query = self.readout_query.expand(hidden_states.shape[0], -1, -1)
        pooled = self.readout(query, hidden_states, context_padding_mask=~joint_mask)[:, 0]
        count = joint_mask.sum(1, keepdim=True).clamp(min=1).to(hidden_states.dtype)
        mean = hidden_states.sum(1) / count
        return self.output_head(torch.cat([pooled, mean], dim=-1))


class DaGetForwardMap(nn.Module):
    """Ensemble of ``num_parallel`` forward networks (F with ``z_dim`` outputs, critics with 1).

    Matches BFM-Zero's ``ForwardMap`` output layout: ``[num_parallel, batch, out]``
    when ``num_parallel > 1`` and ``[batch, out]`` otherwise.
    """

    def __init__(self, config: BfmConfig, output_size: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.output_size = output_size or config.z_dim
        self.members = nn.ModuleList(DaGetForwardNetwork(config, self.output_size) for _ in range(config.num_parallel))

    def forward(
        self,
        joint_embeddings: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate every ensemble member.

        Args:
            joint_embeddings (torch.Tensor): ``DaGetEncoder`` output ``[batch, joints, embedding_size]``.
            z (torch.Tensor): Latent ``[batch, z_dim]`` or ``[batch, L, z_dim]``.
            action (torch.Tensor): Per-DoF action ``[batch, joints]`` or ``[batch, joints, A]``.
            joint_mask (torch.Tensor | None): Bool ``[batch, joints]``.
            context_states (torch.Tensor | None): Optional ``[batch, C, context_size]``.
            context_mask (torch.Tensor | None): Bool ``[batch, C]``.
            latent_mask (torch.Tensor | None): Bool ``[batch, L]``.

        Returns:
            torch.Tensor: ``[num_parallel, batch, out]`` (or ``[batch, out]`` for one member).
        """
        joint_mask = _joint_mask(joint_embeddings, joint_mask)
        if action.shape[:2] != joint_embeddings.shape[:2]:
            raise ValueError("action must be [batch, joints] or [batch, joints, action_size_per_joint]")
        outputs = torch.stack(
            [
                member(joint_embeddings, z, action, joint_mask, context_states, context_mask, latent_mask)
                for member in self.members
            ]
        )
        return outputs if len(self.members) > 1 else outputs[0]


class DaGetBfmModel(nn.Module):
    """FB-CPR model (BFM-Zero's ``FBcprAuxModel``) with GET-conditioned actor and critics.

    Holds B, F, the actor, the discriminator critic, the discriminator and an
    optional auxiliary critic, plus BFM-Zero's latent sampling and inference.
    Actor and critics take ``DaGetEncoder`` embeddings; B and D take flat states.

    Attributes:
        backward_map (BackwardMap): B(s).
        forward_map (DaGetForwardMap): F(s, a, z) with ``z_dim`` outputs.
        actor (DaGetBfmActor): π(a | s, z).
        critic (DaGetForwardMap): Discriminator-reward critic Q(s, a, z).
        discriminator (Discriminator): D(s, z).
        aux_critic (DaGetForwardMap | None): Auxiliary-reward critic.
    """

    def __init__(self, config: BfmConfig, use_aux_critic: bool = True) -> None:
        super().__init__()
        self.config = config
        self.backward_map = BackwardMap(config)
        self.forward_map = DaGetForwardMap(config, config.z_dim)
        self.actor = DaGetBfmActor(config)
        self.critic = DaGetForwardMap(config, 1)
        self.discriminator = Discriminator(config)
        self.aux_critic = DaGetForwardMap(config, 1) if use_aux_critic else None

    def init_weights(self) -> None:
        """Apply BFM-Zero's orthogonal initialization to every linear layer."""
        self.apply(bfm_weight_init)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project z to the sphere when ``config.norm_z`` is set."""
        return project_to_sphere(z) if self.config.norm_z else z

    def sample_z(self, size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Sample ``[size, z_dim]`` latents uniformly on the sphere."""
        return self.project_z(torch.randn(size, self.config.z_dim, device=device))

    @staticmethod
    def fb_q_value(forward_output: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the FB value ``F(s, a, z)ᵀ z`` (keeps the ensemble dimension)."""
        return (forward_output * z).sum(-1)

    @torch.no_grad()
    def act(
        self,
        joint_embeddings: torch.Tensor,
        z: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        mean: bool = True,
        **context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return per-DoF actions in [-1, 1] (the mean, or a sample with ``actor_std``)."""
        distribution = self.actor(joint_embeddings, z, self.config.actor_std, joint_mask, **context)
        return distribution.mean if mean else distribution.sample()

    @torch.no_grad()
    def goal_inference(self, next_observation: Observation) -> torch.Tensor:
        """Goal reaching: ``z = B(s_goal)``."""
        return self.project_z(self.backward_map(next_observation))

    @torch.no_grad()
    def reward_inference(
        self, next_observation: Observation, reward: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reward optimization: ``z = Σ_i w_i r_i B(s_i)`` over buffer states, in chunks.

        Args:
            next_observation (Observation): States ``[N, ...]`` (tensor or dict).
            reward (torch.Tensor): Rewards ``[N, 1]``.
            weight (torch.Tensor | None): Optional per-sample weights ``[N, 1]``.

        Returns:
            torch.Tensor: Latent ``[1, z_dim]``.
        """
        weighted = reward if weight is None else reward * weight
        size, step = weighted.shape[0], self.config.inference_batch_size
        z = 0
        for start in range(0, size, step):
            chunk = _slice(next_observation, start, start + step)
            z = z + weighted[start:start + step].T @ self.backward_map(chunk)
        return self.project_z(z)

    def reward_wr_inference(self, next_observation: Observation, reward: torch.Tensor) -> torch.Tensor:
        """Reward inference with softmax(10 r) weights, as BFM-Zero."""
        return self.reward_inference(next_observation, reward, F.softmax(10 * reward, dim=0))

    @torch.no_grad()
    def tracking_inference(self, next_observation: Observation) -> torch.Tensor:
        """Motion tracking: ``z_t`` = mean of B over the next ``seq_length`` frames of one motion ``[T, ...]``."""
        embeddings = self.backward_map(next_observation)
        prefix = torch.cat([embeddings.new_zeros(1, embeddings.shape[-1]), embeddings.cumsum(0)])
        ends = torch.clamp(torch.arange(embeddings.shape[0], device=embeddings.device) + self.config.seq_length,
                           max=embeddings.shape[0])
        starts = torch.arange(embeddings.shape[0], device=embeddings.device)
        z = (prefix[ends] - prefix[starts]) / (ends - starts)[:, None].to(embeddings.dtype)
        return self.project_z(z)


def _slice(observation: Observation, start: int, end: int) -> Observation:
    """Slice a tensor or every entry of a dict observation along the batch."""
    if isinstance(observation, Mapping):
        return {key: value[start:end] for key, value in observation.items()}
    return observation[start:end]
