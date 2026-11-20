"""BFM-Zero (FB-CPR) networks whose actor and critics read GET joint embeddings.

"Da" stands for dynamic-aware: the GET embeddings these networks read carry the
robot's dynamics.

* One ``DaGetEncoder`` handles policy and state tokens. A shared state-token
  adapter maps privileged state tokens to its input width. ``DaGetBackwardMap``
  B(s) and ``DaGetDiscriminator`` D(s, z) pool its joint embeddings before their
  BFM-Zero-style MLP heads.
* ``DaGetBfmActor`` pi(a | s, z) and ``DaGetForwardMap`` F(s, a, z) (also used for
  the discriminator critic and the auxiliary critic, with one output) take the
  ``[batch, joints, hidden]`` output of ``DaGetEncoder``, cross-attend from joint tokens
  to latent tokens made from z (and optional context tokens), mix joints with
  self-attention, and either decode one action per DoF (actor) or pool the joints.

Nothing depends on the joint count or order, so one set of weights serves any
embodiment the GET encoder can tokenize.

References and attribution:

* BFM-Zero: Li et al., "BFM-Zero: A Promptable Behavioral Foundation Model for
  Humanoid Control Using Unsupervised Reinforcement Learning", ICLR 2026,
  arXiv:2511.04131. https://lecar-lab.github.io/BFM-Zero/
* FB-CPR / Meta Motivo: Tirinzoni et al., "Zero-Shot Whole-Body Humanoid Control via
  Behavioral Foundation Models", ICLR 2025, arXiv:2504.11054. The CPR reward
  ``log D - log(1 - D)`` of ``DaGetDiscriminator.compute_reward`` is theirs.
* Forward-backward representations (``F(s, a, z)^T z`` as Q, ``B`` as the reward
  embedding): Touati & Ollivier, "Learning One Representation to Optimize All
  Rewards", NeurIPS 2021, arXiv:2103.07945.
* GET-Zero: Patel & Song, "GET-Zero: Graph Embodiment Transformer for Zero-Shot
  Embodiment Generalization", ICRA 2025, arXiv:2407.15002, DOI
  10.1109/ICRA55743.2025.11127922.
* Adapted code (https://github.com/LeCAR-Lab/BFM-Zero, Copyright (c) Meta Platforms,
  Inc. and affiliates, CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/):
  from ``humanoidverse/agents/nn_models.py``, the MLP heads of ``BackwardMap`` and
  ``Discriminator`` (here ``DaGetBackwardMap``, ``DaGetDiscriminator``), ``TruncatedNormal``,
  ``Norm`` (``project_to_sphere``) and ``weight_init`` (``bfm_weight_init``); from
  ``humanoidverse/agents/fb/model.py``, ``FBModel.sample_z``, ``project_z``, ``act`` and the
  goal, reward, weighted-reward and tracking inference; from
  ``humanoidverse/agents/fb_cpr/model.py`` and ``fb_cpr_aux/model.py``, the set of
  networks ``FBcprAuxModel`` holds. Changes: rewritten in this repo's style, a padding
  mask on ``TruncatedNormal``, vectorized ``tracking_inference``, per-DoF inputs.
* ``TruncatedNormal`` is the class of DrQ-v2 (Yarats et al., ICLR 2022, arXiv:2107.09645;
  ``utils.py`` of https://github.com/facebookresearch/drqv2, MIT); its clipped sampling
  noise is TD3's target policy smoothing (Fujimoto, van Hoof & Meger, ICML 2018, PMLR
  80:1587-1596, arXiv:1802.09477).
* Orthogonal initialization: Saxe, McClelland & Ganguli, ICLR 2014, arXiv:1312.6120.
* Layers: attention and sinusoidal positions (Vaswani et al., NIPS 2017,
  arXiv:1706.03762), LayerNorm (Ba, Kiros & Hinton, arXiv:1607.06450), GELU
  (Hendrycks & Gimpel, arXiv:1606.08415), Mish (Misra, BMVC 2020, arXiv:1908.08681).
* The per-DoF networks (latent tokens, joint-to-latent cross-attention, per-DoF
  action head, joint readout, and shared state-token adapter) are new in this repo.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal
from torch.func import functional_call

from model.configuration_bfm import BfmConfig
from model.modeling_da_get import DaGetEncoder
from model.transformer import AttentionBlock, CrossAttentionBlock, NumericTokenizer, SinusoidalPositionEmbedding

StateInputs = Mapping[str, torch.Tensor]


def project_to_sphere(values: torch.Tensor) -> torch.Tensor:
    """Scale the last dimension to norm ``sqrt(dim)`` (BFM-Zero's ``Norm``, ``nn_models.py``)."""
    return math.sqrt(values.shape[-1]) * F.normalize(values, dim=-1)


def bfm_weight_init(module: nn.Module) -> None:
    """Orthogonal weights (Saxe et al., ICLR 2014) and zero biases, as BFM-Zero's ``weight_init`` (``nn_models.py``)."""
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
    """Normal policy clamped to [low, high] with straight-through gradients.

    Adapted from BFM-Zero's ``TruncatedNormal`` (``humanoidverse/agents/nn_models.py``),
    the class of DrQ-v2 (Yarats et al., ICLR 2022, ``utils.py``); this version adds
    ``mask``.

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
        """Draw a reparameterized action; ``clip`` bounds the noise (TD3, Fujimoto et al., ICML 2018)."""
        shape = self._extended_shape(sample_shape)
        noise = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device) * self.scale
        if clip is not None:
            noise = torch.clamp(noise, -clip, clip)
        action = self._clamp(self.loc + noise)
        return action if self.mask is None else action.masked_fill(~self.mask, 0)


class JointReadout(nn.Module):
    """Pool joint tokens: a learned query attends over them, concatenated with their masked mean."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.attention = CrossAttentionBlock(hidden_size, num_heads, dropout=dropout)

    def forward(self, hidden_states: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, 2 * hidden]`` from ``[batch, joints, hidden]`` and a bool mask."""
        query = self.query.expand(hidden_states.shape[0], -1, -1)
        pooled = self.attention(query, hidden_states, context_padding_mask=~joint_mask)[:, 0]
        count = joint_mask.sum(1, keepdim=True).clamp(min=1).to(hidden_states.dtype)
        return torch.cat((pooled, (hidden_states * joint_mask[..., None]).sum(1) / count), dim=-1)


class DaGetBackwardMap(nn.Module):
    """Backward representation B(s) -> z, shared across robots.

    A readout of the shared encoder replaces BFM-Zero's flat input. Its head is adapted
    from BFM-Zero's ``BackwardMap`` (``humanoidverse/agents/nn_models.py``):
    Linear -> LayerNorm -> Tanh, ``backward_hidden_layers - 1`` Linear -> ReLU, then Linear
    to ``z_dim`` and an optional projection to the sphere.

    Attributes:
        config (BfmConfig): Validated head dimensions and latent settings.
        readout (JointReadout): Pool shared-encoder joint embeddings.
        head (nn.Sequential): Map pooled state to the backward latent.
    """

    def __init__(self, config: BfmConfig) -> None:
        """Construct the backward head without creating an encoder.

        Args:
            config (BfmConfig): Validated embedding, readout, and latent sizes.

        Returns:
            None: Registers the readout and backward MLP.
        """
        super().__init__()
        self.config = config
        self.readout = JointReadout(config.embedding_size, math.gcd(config.embedding_size, config.state_num_heads))
        width = config.backward_hidden_size
        layers: list[nn.Module] = [nn.Linear(2 * config.embedding_size, width), nn.LayerNorm(width), nn.Tanh()]
        for _ in range(config.backward_hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, config.z_dim))
        if config.backward_norm:
            layers.append(SphereNorm())
        self.head = nn.Sequential(*layers)

    def forward(self, joint_embeddings: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
        """Map shared-encoder embeddings to ``[batch, z_dim]``.

        Args:
            joint_embeddings (torch.Tensor): Encoded state tokens ``[batch, joints, embedding_size]``.
            joint_mask (torch.Tensor): Boolean ``[batch, joints]`` valid-token mask.

        Returns:
            torch.Tensor: Backward representation ``[batch, z_dim]``.
        """
        return self.head(self.readout(joint_embeddings, joint_mask))


class DaGetDiscriminator(nn.Module):
    """FB-CPR discriminator D(s, z), shared across robots.

    A readout of the shared encoder replaces BFM-Zero's flat input. Its head, adapted from
    BFM-Zero's ``Discriminator`` (``humanoidverse/agents/nn_models.py``), follows on the
    pooled state and z. ``compute_reward`` gives the CPR reward ``log D - log(1 - D)``
    (Tirinzoni et al., ICLR 2025).

    Attributes:
        config (BfmConfig): Validated head dimensions and latent settings.
        readout (JointReadout): Pool shared-encoder joint embeddings.
        trunk (nn.Sequential): Classify pooled state and a latent prompt.
    """

    def __init__(self, config: BfmConfig) -> None:
        """Construct the discriminator head without creating an encoder.

        Args:
            config (BfmConfig): Validated embedding, readout, and latent sizes.

        Returns:
            None: Registers the readout and classification MLP.
        """
        super().__init__()
        self.config = config
        self.readout = JointReadout(config.embedding_size, math.gcd(config.embedding_size, config.state_num_heads))
        width = config.discriminator_hidden_size
        layers: list[nn.Module] = [nn.Linear(2 * config.embedding_size + config.z_dim, width), nn.LayerNorm(width), nn.Tanh()]
        for _ in range(config.discriminator_hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 1))
        self.trunk = nn.Sequential(*layers)

    def logits_from_features(self, features: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return logits from pooled state features ``[batch, output_size]``."""
        return self.trunk(torch.cat([z, features], dim=-1))

    def compute_logits(self, joint_embeddings: torch.Tensor, z: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
        """Return expert logits from encoded state tokens and latent prompts.

        Args:
            joint_embeddings (torch.Tensor): Shared-encoder output ``[batch, joints, embedding_size]``.
            z (torch.Tensor): Latent prompts ``[batch, z_dim]``.
            joint_mask (torch.Tensor): Boolean ``[batch, joints]`` valid-token mask.

        Returns:
            torch.Tensor: Expert logits ``[batch, 1]``.
        """
        return self.logits_from_features(self.readout(joint_embeddings, joint_mask), z)

    def forward(self, joint_embeddings: torch.Tensor, z: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
        """Return expert probabilities ``[batch, 1]`` from encoded state tokens.

        Args:
            joint_embeddings (torch.Tensor): Shared-encoder output ``[batch, joints, embedding_size]``.
            z (torch.Tensor): Latent prompts ``[batch, z_dim]``.
            joint_mask (torch.Tensor): Boolean ``[batch, joints]`` valid-token mask.

        Returns:
            torch.Tensor: Expert probabilities ``[batch, 1]``.
        """
        return torch.sigmoid(self.compute_logits(joint_embeddings, z, joint_mask))

    def compute_reward(self, joint_embeddings: torch.Tensor, z: torch.Tensor, joint_mask: torch.Tensor,
                       eps: float = 1e-7) -> torch.Tensor:
        """Return the CPR log-odds reward from encoded state tokens.

        Args:
            joint_embeddings (torch.Tensor): Shared-encoder output ``[batch, joints, embedding_size]``.
            z (torch.Tensor): Latent prompts ``[batch, z_dim]``.
            joint_mask (torch.Tensor): Boolean ``[batch, joints]`` valid-token mask.
            eps (float): Probability clamp margin within ``(0, 0.5)``.

        Returns:
            torch.Tensor: Log-odds rewards ``[batch, 1]``.
        """
        probability = self.forward(joint_embeddings, z, joint_mask).clamp(eps, 1 - eps)
        return probability.log() - (1 - probability).log()


class LatentMemory(nn.Module):
    """Turn z (and optional context tokens) into key/value tokens for joint cross-attention.

    Each latent vector becomes ``num_latent_tokens`` tokens: a linear map to
    ``num_latent_tokens * hidden_size`` plus a learned slot embedding. A sequence
    of latents (for example B of future frames) also gets a sinusoidal position.
    """

    def __init__(self, config: BfmConfig, role: str) -> None:
        """Create latent memory at the actor or forward network width.

        Args:
            config (BfmConfig): Validated network configuration.
            role (str): ``actor`` or ``forward``; critics use ``forward``.

        Returns:
            None: Registers projections, embeddings, and normalization.

        Raises:
            ValueError: The role is not supported.
        """
        super().__init__()
        self.hidden_size, _, _, self.num_tokens = config.network_dimensions(role)
        self.latent_projection = nn.Linear(config.z_dim, self.num_tokens * self.hidden_size)
        self.slot_embedding = nn.Parameter(torch.randn(self.num_tokens, self.hidden_size) * 0.02)
        self.position_embedding = SinusoidalPositionEmbedding(self.hidden_size)
        self.context_projection = nn.Linear(config.state_token_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

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
            context_states (torch.Tensor | None): ``[batch, C, state_token_size]`` extra tokens.
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
            if context_mask is None:
                context_mask = torch.ones(context_states.shape[:2], dtype=torch.bool, device=z.device)
            tokens = torch.cat([tokens, self.context_projection(context_states)], dim=1)
            mask = torch.cat([mask, context_mask], dim=1)
        return self.norm(tokens), mask


class LatentConditionedBlock(nn.Module):
    """Joint tokens cross-attend to latent memory, self-attend across joints, then feed forward."""

    def __init__(self, config: BfmConfig, role: str) -> None:
        """Create one latent-conditioned attention block for a network role.

        Args:
            config (BfmConfig): Validated network configuration.
            role (str): ``actor`` or ``forward``; critics use ``forward``.

        Returns:
            None: Registers cross-attention, self-attention, and feed-forward layers.

        Raises:
            ValueError: The role is not supported.
        """
        super().__init__()
        width, heads, intermediate, _ = config.network_dimensions(role)
        self.cross_attention = CrossAttentionBlock(width, heads, dropout=config.dropout)
        self.self_attention = AttentionBlock(width, heads, dropout=config.dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, intermediate),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(intermediate, width),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        joint_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update joint tokens ``[batch, joints, hidden]``; padded joints stay zero.

        ``attention_bias`` (``[batch * heads, joints, joints]``) is added to the joint
        self-attention logits (the physics-interface bias of proposal B2).
        """
        hidden_states = self.cross_attention(hidden_states, memory, context_padding_mask=~memory_mask)
        if attention_bias is None:
            hidden_states = self.self_attention(hidden_states, key_padding_mask=~joint_mask)
        else:
            bias = attention_bias.masked_fill(
                ~joint_mask.repeat_interleave(attention_bias.shape[0] // joint_mask.shape[0], dim=0)[:, None, :],
                float("-inf"),
            )
            hidden_states = self.self_attention(hidden_states, attention_mask=bias.to(hidden_states.dtype))
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
    ``tanh`` means of a truncated normal, as in BFM-Zero's ``Actor``
    (``humanoidverse/agents/nn_models.py``).
    """

    def __init__(self, config: BfmConfig) -> None:
        """Create the actor with its independently selected dimensions.

        Args:
            config (BfmConfig): Validated actor and input dimensions.

        Returns:
            None: Registers the policy projection, memory, blocks, and action head.
        """
        super().__init__()
        self.config = config
        width, _, _, _ = config.network_dimensions("actor")
        self.embedding_projection = nn.Sequential(nn.LayerNorm(config.embedding_size), nn.Linear(config.embedding_size, width))
        self.memory = LatentMemory(config, "actor")
        self.layers = nn.ModuleList(LatentConditionedBlock(config, "actor") for _ in range(config.actor_layers))
        self.action_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.Mish(), nn.Linear(width, config.action_size_per_joint)
        )
        _, heads, _, _ = config.network_dimensions("actor")
        self.num_heads = heads
        self.interface_projections = (
            nn.ModuleList(nn.Linear(1, heads, bias=False) for _ in range(config.actor_layers))
            if config.use_interface_bias
            else None
        )
        # Set by DaGetBfmModel: a function of (joint embeddings, mask) returning the
        # normalized operator, or None. A plain attribute, not a submodule: the
        # interface belongs to (and is trained with) the physics heads.
        self.operator_source = None

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
            context_states (torch.Tensor | None): Optional ``[batch, C, state_token_size]`` tokens.
            context_mask (torch.Tensor | None): Bool ``[batch, C]``.
            latent_mask (torch.Tensor | None): Bool ``[batch, L]``.

        Returns:
            TruncatedNormal: Over ``[batch, joints]`` (or ``[batch, joints, A]``); padded joints are 0.
        """
        joint_mask = _joint_mask(joint_embeddings, joint_mask)
        memory, memory_mask = self.memory(z, latent_mask, context_states, context_mask)
        hidden_states = self.embedding_projection(joint_embeddings).masked_fill(~joint_mask[..., None], 0)
        operator = None
        if self.interface_projections is not None and self.operator_source is not None:
            operator = self.operator_source(joint_embeddings, joint_mask)
        for index, layer in enumerate(self.layers):
            bias = None
            if operator is not None:
                batch, joints = joint_mask.shape
                bias = self.interface_projections[index](operator[..., None]).permute(0, 3, 1, 2)
                bias = bias.reshape(batch * self.num_heads, joints, joints)
            hidden_states = layer(hidden_states, joint_mask, memory, memory_mask, bias)
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
        """Create one forward or critic member at the selected forward size.

        Args:
            config (BfmConfig): Validated forward and input dimensions.
            output_size (int): Positive output width for F or a critic.

        Returns:
            None: Registers the action tokenizer, attention blocks, and readout.
        """
        super().__init__()
        width, heads, intermediate, _ = config.network_dimensions("forward")
        self.embedding_projection = nn.Sequential(nn.LayerNorm(config.embedding_size), nn.Linear(config.embedding_size, width))
        self.action_embedding = NumericTokenizer(config.action_size_per_joint, width, width)
        self.memory = LatentMemory(config, "forward")
        self.layers = nn.ModuleList(LatentConditionedBlock(config, "forward") for _ in range(config.forward_layers))
        self.readout = JointReadout(width, heads, config.dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, intermediate), nn.Mish(),
            nn.Linear(intermediate, output_size),
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
        return self.output_head(self.readout(hidden_states, joint_mask))


class DaGetForwardMap(nn.Module):
    """Ensemble of ``num_parallel`` forward networks (F with ``z_dim`` outputs, critics with 1).

    Matches the output layout of BFM-Zero's ``ForwardMap`` (``humanoidverse/agents/nn_models.py``):
    ``[num_parallel, batch, out]``
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
            context_states (torch.Tensor | None): Optional ``[batch, C, state_token_size]``.
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
    optional auxiliary critic (as ``FBcprAuxModel``, ``humanoidverse/agents/fb_cpr_aux/model.py``),
    plus the latent sampling and inference adapted from ``FBModel``
    (``humanoidverse/agents/fb/model.py``).
    One ``DaGetEncoder`` produces embeddings for every head. State tokens pass
    through a shared input adapter before that encoder.

    Attributes:
        config (BfmConfig): Validated dimensions for all BFM heads.
        encoder (DaGetEncoder): Shared policy and state encoder.
        state_adapter (nn.Sequential): Map state tokens to the encoder input width.
        backward_map (DaGetBackwardMap): B(s) from encoder embeddings.
        forward_map (DaGetForwardMap): F(s, a, z) with ``z_dim`` outputs.
        actor (DaGetBfmActor): π(a | s, z).
        critic (DaGetForwardMap): Discriminator-reward critic Q(s, a, z).
        discriminator (DaGetDiscriminator): D(s, z).
        aux_critic (DaGetForwardMap | None): Auxiliary-reward critic.
    """

    def __init__(
        self, config: BfmConfig, encoder: DaGetEncoder, use_aux_critic: bool = True, physics_interface: nn.Module | None = None
    ) -> None:
        """Build independent heads around a single shared DA-GET encoder.

        Args:
            config (BfmConfig): Validated BFM dimensions and head settings.
            encoder (DaGetEncoder): Policy encoder, optionally loaded from pretraining.
            use_aux_critic (bool): Whether to construct the auxiliary reward critic.
            physics_interface (nn.Module | None): ``DaGetPhysicsInterface`` of the
                pretrained model; required by ``use_interface_bias``.

        Returns:
            None: Registers the encoder, state adapter, and all BFM heads.

        Raises:
            ValueError: The encoder output width differs from ``embedding_size``.
        """
        super().__init__()
        if encoder.config.hidden_size != config.embedding_size:
            raise ValueError("encoder hidden_size must equal BfmConfig.embedding_size")
        self.config = config
        self.encoder = encoder
        adapter_layers: list[nn.Module] = [
            nn.LayerNorm(config.state_token_size),
            nn.Linear(config.state_token_size, config.state_hidden_size),
            nn.GELU(),
        ]
        for _ in range(config.state_num_layers - 1):
            adapter_layers.extend((
                nn.Linear(config.state_hidden_size, config.state_intermediate_size),
                nn.GELU(),
                nn.Linear(config.state_intermediate_size, config.state_hidden_size),
                nn.GELU(),
            ))
        adapter_layers.append(nn.Linear(config.state_hidden_size, encoder.config.input_size))
        self.state_adapter = nn.Sequential(*adapter_layers)
        self.backward_map = DaGetBackwardMap(config)
        self.forward_map = DaGetForwardMap(config, config.z_dim)
        self.actor = DaGetBfmActor(config)
        self.critic = DaGetForwardMap(config, 1)
        self.discriminator = DaGetDiscriminator(config)
        self.aux_critic = DaGetForwardMap(config, 1) if use_aux_critic else None
        if config.use_interface_bias and physics_interface is None:
            raise ValueError("use_interface_bias needs the pretrained physics interface (head_type linear or linear_residual)")
        # Not a submodule: the interface is saved and trained with the physics heads.
        object.__setattr__(self, "physics_interface", physics_interface)
        if config.use_interface_bias:
            self.actor.operator_source = self.interface_operator

    def interface_operator(self, joint_embeddings: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor | None:
        """Return the interface's joint operator normalized to unit diagonal (detached), or None.

        Args:
            joint_embeddings (torch.Tensor): ``[B, J, E]`` policy-encoder embeddings.
            joint_mask (torch.Tensor): Bool ``[B, J]``.

        Returns:
            torch.Tensor | None: ``[B, J, J]`` ``G_ij / sqrt(G_ii G_jj)``.
        """
        if self.physics_interface is None:
            return None
        from model.modeling_da_get import normalize_mass_matrix

        with torch.no_grad():
            weights = joint_mask.to(joint_embeddings.dtype)[..., None]
            pooled = (joint_embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)
            operator = self.physics_interface(joint_embeddings, pooled, joint_mask)["operator"][:, 6:, :]
            return normalize_mass_matrix(operator, joint_mask, 1e-8, validate=False)

    def encode_state(
        self,
        state: StateInputs,
        *,
        encoder: DaGetEncoder | None = None,
        adapter: nn.Module | None = None,
        train_encoder: bool = True,
        validate_inputs: bool = True,
    ) -> torch.Tensor:
        """Encode privileged state tokens with the shared encoder core.

        The state's full mass matrix (``StateView.mass``, passed as
        ``mass_matrix``) feeds the mass-matrix attention bias; without one, a
        diagonal matrix from the dynamics features' mass diagonal is used, which
        the normalization turns into the identity (evidence R6,
        ``docs/zero_shot.md``).

        Args:
            state (StateInputs): State-token and graph inputs from ``state_inputs``.
            encoder (DaGetEncoder | None): Target encoder override; defaults to the online encoder.
            adapter (nn.Module | None): Target state-adapter override; defaults to the online adapter.
            train_encoder (bool): Whether this call accumulates encoder parameter gradients.
            validate_inputs (bool): Passed to ``DaGetEncoder.forward``.

        Returns:
            torch.Tensor: Per-joint embeddings ``[batch, joints, embedding_size]``.

        Raises:
            ValueError: State inputs have invalid dimensions or physics features.
        """
        source = self.encoder if encoder is None else encoder
        projection = self.state_adapter if adapter is None else adapter
        if "input_values" not in state or "dynamics_features" not in state:
            raise ValueError("state must include input_values and dynamics_features")
        if state["input_values"].ndim != 3 or state["input_values"].shape[-1] != self.config.state_token_size:
            raise ValueError("state input_values must be [batch, joints, state_token_size]")
        inputs = dict(state, validate_inputs=validate_inputs)
        inputs["input_values"] = projection(inputs["input_values"])
        if source.config.use_mass_encoding:
            if inputs.get("mass_matrix") is None:
                diagonal = inputs["dynamics_features"][..., 0].clamp_min(1e-8)
                inputs["mass_matrix"] = torch.diag_embed(diagonal)
        else:
            inputs.pop("mass_matrix", None)
        if source.physics_tokenizers is None:
            for name in ("rigid_body_features", "actuator_features", "dynamics_features", "root_features"):
                inputs.pop(name, None)
        elif not source.config.needs_root_features:
            inputs.pop("root_features", None)
        if train_encoder:
            return source(**inputs)
        frozen_parameters = {name: parameter.detach() for name, parameter in source.named_parameters()}
        return functional_call(source, frozen_parameters, (), inputs)

    def init_weights(self) -> None:
        """Initialize BFM heads without changing pretrained encoder weights.

        Returns:
            None: Orthogonally initializes new linear layers in place.
        """
        for name, module in self.named_children():
            if name != "encoder":
                module.apply(bfm_weight_init)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project z to the sphere when ``config.norm_z`` is set."""
        return project_to_sphere(z) if self.config.norm_z else z

    def sample_z(self, size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Sample ``[size, z_dim]`` latents uniformly on the sphere (BFM-Zero's ``FBModel.sample_z``)."""
        return self.project_z(torch.randn(size, self.config.z_dim, device=device))

    @staticmethod
    def fb_q_value(forward_output: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the FB value ``F(s, a, z)ᵀ z`` (Touati & Ollivier, NeurIPS 2021; keeps the ensemble dimension)."""
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
    def goal_inference(self, goal: StateInputs) -> torch.Tensor:
        """Goal reaching: ``z = B(s_goal)`` (BFM-Zero's ``FBModel.goal_inference``)."""
        return self.project_z(self.backward_map(self.encode_state(goal), goal["joint_mask"]))

    @torch.no_grad()
    def reward_inference(self, next_state: StateInputs, reward: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """Reward optimization: ``z = sum_i w_i r_i B(s_i)`` over one robot's buffer states, in chunks.

        Adapted from BFM-Zero's ``FBModel.reward_inference``; the estimate of z for a
        reward is that of Touati & Ollivier (NeurIPS 2021).

        Args:
            next_state (StateInputs): State tokens of ``N`` frames.
            reward (torch.Tensor): Rewards ``[N, 1]``.
            weight (torch.Tensor | None): Optional per-sample weights ``[N, 1]``.

        Returns:
            torch.Tensor: Latent ``[1, z_dim]``.
        """
        weighted = reward if weight is None else reward * weight
        size, step = weighted.shape[0], self.config.inference_batch_size
        z = 0
        for start in range(0, size, step):
            chunk = _slice_inputs(next_state, start, start + step)
            z = z + weighted[start:start + step].T @ self.backward_map(self.encode_state(chunk), chunk["joint_mask"])
        return self.project_z(z)

    def reward_wr_inference(self, next_state: StateInputs, reward: torch.Tensor) -> torch.Tensor:
        """Reward inference with softmax(10 r) weights (BFM-Zero's ``FBModel.reward_wr_inference``)."""
        return self.reward_inference(next_state, reward, F.softmax(10 * reward, dim=0))

    @torch.no_grad()
    def tracking_inference(self, next_state: StateInputs) -> torch.Tensor:
        """Motion tracking: ``z_t`` = mean of B over the next ``seq_length`` frames of one motion ``[T, ...]``.

        BFM-Zero's ``FBModel.tracking_inference``, vectorized with a prefix sum.
        """
        embeddings = self.backward_map(self.encode_state(next_state), next_state["joint_mask"])
        prefix = torch.cat([embeddings.new_zeros(1, embeddings.shape[-1]), embeddings.cumsum(0)])
        starts = torch.arange(embeddings.shape[0], device=embeddings.device)
        ends = torch.clamp(starts + self.config.seq_length, max=embeddings.shape[0])
        z = (prefix[ends] - prefix[starts]) / (ends - starts)[:, None].to(embeddings.dtype)
        return self.project_z(z)


def _slice_inputs(state: StateInputs, start: int, end: int) -> dict:
    """Slice encoder keyword inputs along the batch; graph distances are a tuple of tensors."""
    return {
        key: tuple(item[start:end] for item in value) if isinstance(value, tuple) else value[start:end]
        for key, value in state.items()
    }
