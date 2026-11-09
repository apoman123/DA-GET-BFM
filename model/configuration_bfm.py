"""Configuration for the GET-conditioned behavior foundation model (FB-CPR networks)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BfmConfig:
    """Sizes of the BFM-Zero networks whose actor and critics read GET joint embeddings.

    The backward map and discriminator keep the BFM-Zero MLPs on a flat state.
    The actor, forward map and critics read the ``[batch, joints, embedding_size]``
    output of ``GetEncoder`` and cross-attend from joints to latent tokens made from z.

    Attributes:
        embedding_size (int): Width of GET joint embeddings (``GetConfig.hidden_size``).
        backward_input_size (int): Flat state width read by the backward map.
        discriminator_input_size (int | None): Flat state width read by the
            discriminator; defaults to backward_input_size.
        backward_input_keys (tuple[str, ...]): Observation-dict keys concatenated
            into the backward input (BFM-Zero: state and privileged state).
        discriminator_input_keys (tuple[str, ...]): Same for the discriminator.
        z_dim (int): Latent (task) dimension.
        norm_z (bool): Project z onto the sphere of radius ``sqrt(z_dim)``.
        context_size (int | None): Width of optional extra context tokens (base
            state, history) that actor and critics attend to next to the latents.
        hidden_size (int): Token width inside actor and critics.
        num_attention_heads (int): Heads of every attention block.
        intermediate_size (int): Feed-forward width of every block.
        num_latent_tokens (int): Tokens each latent vector is split into.
        actor_layers (int): Latent-conditioned blocks in the actor.
        forward_layers (int): Latent-conditioned blocks in each forward/critic member.
        num_parallel (int): Ensemble size of the forward map and critics.
        action_size_per_joint (int): Action entries per DoF (1 = one command per joint).
        dropout (float): Dropout in actor and critic blocks.
        backward_hidden_size (int): Backward MLP width (BFM-Zero: 256).
        backward_hidden_layers (int): Backward MLP depth (BFM-Zero: 1).
        backward_norm (bool): Normalize B(s) to radius ``sqrt(z_dim)``.
        discriminator_hidden_size (int): Discriminator MLP width (BFM-Zero: 1024).
        discriminator_hidden_layers (int): Discriminator MLP depth (BFM-Zero: 3).
        actor_std (float): Exploration std of the truncated-normal policy (BFM-Zero: 0.05).
        seq_length (int): Window averaged by ``tracking_inference`` (BFM-Zero: 8).
        inference_batch_size (int): Chunk size of ``reward_inference``.
    """

    embedding_size: int
    backward_input_size: int
    discriminator_input_size: int | None = None
    backward_input_keys: tuple[str, ...] = ("state", "privileged_state")
    discriminator_input_keys: tuple[str, ...] = ("state", "privileged_state")
    z_dim: int = 256
    norm_z: bool = True
    context_size: int | None = None
    hidden_size: int = 256
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    num_latent_tokens: int = 8
    actor_layers: int = 4
    forward_layers: int = 4
    num_parallel: int = 2
    action_size_per_joint: int = 1
    dropout: float = 0.0
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
            self.embedding_size, self.backward_input_size, self.z_dim, self.hidden_size,
            self.num_attention_heads, self.intermediate_size, self.num_latent_tokens,
            self.actor_layers, self.forward_layers, self.num_parallel, self.action_size_per_joint,
            self.backward_hidden_size, self.backward_hidden_layers, self.discriminator_hidden_size,
            self.discriminator_hidden_layers, self.seq_length, self.inference_batch_size,
        )
        if any(value <= 0 for value in sizes):
            raise ValueError("all sizes must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        for name in ("discriminator_input_size", "context_size"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.actor_std < 0:
            raise ValueError("actor_std must be non-negative")

    @property
    def discriminator_size(self) -> int:
        """Flat state width read by the discriminator."""
        return self.discriminator_input_size or self.backward_input_size
