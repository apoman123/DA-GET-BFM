"""Configuration for the GET-conditioned behavior foundation model (FB-CPR networks)."""

from dataclasses import dataclass

from model.configuration_da_get import DaGetConfig


@dataclass(frozen=True)
class BfmConfig:
    """Sizes of the BFM-Zero networks, all of which read per-DoF tokens instead of a flat state.

    The actor, forward map and critics read the ``[batch, joints, embedding_size]`` output
    of ``DaGetEncoder`` and cross-attend from joints to latent tokens made from z. The
    backward map and discriminator each own a ``DaGetEncoder`` over state tokens, then keep
    BFM-Zero's MLP head, so neither depends on a robot's joint count.

    Attributes:
        embedding_size (int): Width of policy joint embeddings (``DaGetConfig.hidden_size``).
        state_token_size (int): Width of the state tokens read by B and D; critics also
            read them as privileged context tokens.
        z_dim (int): Latent (task) dimension.
        norm_z (bool): Project z onto the sphere of radius ``sqrt(z_dim)``.
        hidden_size (int): Token width inside actor and critics.
        num_attention_heads (int): Heads of every attention block.
        intermediate_size (int): Feed-forward width of every block.
        num_latent_tokens (int): Tokens each latent vector is split into.
        actor_layers (int): Latent-conditioned blocks in the actor.
        forward_layers (int): Latent-conditioned blocks in each forward/critic member.
        num_parallel (int): Ensemble size of the forward map and critics.
        action_size_per_joint (int): Action entries per DoF (1 = one command per joint).
        dropout (float): Dropout in actor and critic blocks.
        state_hidden_size (int): Width of the encoders of B and D.
        state_num_layers (int): Depth of the encoders of B and D.
        state_num_heads (int): Attention heads of the encoders of B and D.
        state_intermediate_size (int): Feed-forward width of the encoders of B and D.
        backward_hidden_size (int): Backward head width (BFM-Zero: 256).
        backward_hidden_layers (int): Backward head depth (BFM-Zero: 1).
        backward_norm (bool): Normalize B(s) to radius ``sqrt(z_dim)``.
        discriminator_hidden_size (int): Discriminator head width (BFM-Zero: 1024).
        discriminator_hidden_layers (int): Discriminator head depth (BFM-Zero: 3).
        actor_std (float): Exploration std of the truncated-normal policy (BFM-Zero: 0.05).
        seq_length (int): Window averaged by ``tracking_inference`` (BFM-Zero: 8).
        inference_batch_size (int): Chunk size of ``reward_inference``.
    """

    embedding_size: int
    state_token_size: int
    z_dim: int = 256
    norm_z: bool = True
    hidden_size: int = 256
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    num_latent_tokens: int = 8
    actor_layers: int = 4
    forward_layers: int = 4
    num_parallel: int = 2
    action_size_per_joint: int = 1
    dropout: float = 0.0
    state_hidden_size: int = 128
    state_num_layers: int = 2
    state_num_heads: int = 4
    state_intermediate_size: int = 256
    backward_hidden_size: int = 256
    backward_hidden_layers: int = 1
    backward_norm: bool = True
    discriminator_hidden_size: int = 1024
    discriminator_hidden_layers: int = 3
    actor_std: float = 0.05
    seq_length: int = 8
    inference_batch_size: int = 500_000

    def __post_init__(self) -> None:
        """Reject non-positive sizes and a width the heads do not divide."""
        sizes = (
            self.embedding_size, self.state_token_size, self.z_dim, self.hidden_size,
            self.num_attention_heads, self.intermediate_size, self.num_latent_tokens,
            self.actor_layers, self.forward_layers, self.num_parallel, self.action_size_per_joint,
            self.state_hidden_size, self.state_num_layers, self.state_num_heads, self.state_intermediate_size,
            self.backward_hidden_size, self.backward_hidden_layers, self.discriminator_hidden_size,
            self.discriminator_hidden_layers, self.seq_length, self.inference_batch_size,
        )
        if any(value <= 0 for value in sizes):
            raise ValueError("all sizes must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.actor_std < 0:
            raise ValueError("actor_std must be non-negative")

    @property
    def state_encoder_config(self) -> DaGetConfig:
        """Return the encoder of B and D: physics-aware, without the mass-matrix bias."""
        return DaGetConfig(
            input_size=self.state_token_size,
            hidden_size=self.state_hidden_size,
            num_hidden_layers=self.state_num_layers,
            num_attention_heads=self.state_num_heads,
            intermediate_size=self.state_intermediate_size,
            use_mass_encoding=False,
            use_physics_cross_attention=True,
        )
