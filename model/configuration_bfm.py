"""Configuration for the GET-conditioned behavior foundation model (FB-CPR networks)."""

from dataclasses import dataclass

ACTOR_FORWARD_SIZES: dict[str, dict[str, int]] = {
    "tiny": {
        "hidden_size": 128, "num_attention_heads": 4, "intermediate_size": 512,
        "num_latent_tokens": 4, "layers": 2,
    },
    "small": {
        "hidden_size": 256, "num_attention_heads": 4, "intermediate_size": 1024,
        "num_latent_tokens": 8, "layers": 4,
    },
    "medium": {
        "hidden_size": 384, "num_attention_heads": 8, "intermediate_size": 1536,
        "num_latent_tokens": 8, "layers": 6,
    },
    "large": {
        "hidden_size": 512, "num_attention_heads": 16, "intermediate_size": 2048,
        "num_latent_tokens": 16, "layers": 8,
    },
}

BACKWARD_SIZES: dict[str, dict[str, int]] = {
    "tiny": {
        "state_hidden_size": 64, "state_num_layers": 1, "state_num_heads": 2,
        "state_intermediate_size": 128, "backward_hidden_size": 128, "backward_hidden_layers": 1,
    },
    "small": {
        "state_hidden_size": 128, "state_num_layers": 2, "state_num_heads": 4,
        "state_intermediate_size": 256, "backward_hidden_size": 256, "backward_hidden_layers": 1,
    },
    "medium": {
        "state_hidden_size": 256, "state_num_layers": 4, "state_num_heads": 8,
        "state_intermediate_size": 1024, "backward_hidden_size": 512, "backward_hidden_layers": 2,
    },
    "large": {
        "state_hidden_size": 512, "state_num_layers": 6, "state_num_heads": 16,
        "state_intermediate_size": 2048, "backward_hidden_size": 1024, "backward_hidden_layers": 3,
    },
}


@dataclass(frozen=True)
class BfmConfig:
    """Sizes of the BFM-Zero networks, all of which read per-DoF tokens instead of a flat state.

    The actor, forward map, backward map, discriminator, and critics read the
    ``[batch, joints, embedding_size]`` output of one shared ``DaGetEncoder``.
    State tokens first pass through a shared MLP adapter. No head depends on a
    robot's joint count.

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
        state_hidden_size (int): Hidden width of the shared state-token adapter.
        state_num_layers (int): Positive depth of the state-token adapter.
        state_num_heads (int): Maximum attention heads in B and D joint readouts.
        state_intermediate_size (int): Inner width of deeper state-adapter blocks.
        backward_hidden_size (int): Backward head width (BFM-Zero: 256).
        backward_hidden_layers (int): Backward head depth (BFM-Zero: 1).
        backward_norm (bool): Normalize B(s) to radius ``sqrt(z_dim)``.
        discriminator_hidden_size (int): Discriminator head width (BFM-Zero: 1024).
        discriminator_hidden_layers (int): Discriminator head depth (BFM-Zero: 3).
        actor_std (float): Exploration std of the truncated-normal policy (BFM-Zero: 0.05).
        seq_length (int): Window averaged by ``tracking_inference`` (BFM-Zero: 8).
        inference_batch_size (int): Chunk size of ``reward_inference``.
        actor_hidden_size (int | None): Actor width; None uses ``hidden_size``.
        actor_num_attention_heads (int | None): Actor heads; None uses shared heads.
        actor_intermediate_size (int | None): Actor feed-forward width; None uses shared width.
        actor_num_latent_tokens (int | None): Actor latent slots; None uses shared slots.
        forward_hidden_size (int | None): Forward/critic width; None uses ``hidden_size``.
        forward_num_attention_heads (int | None): Forward/critic heads; None uses shared heads.
        forward_intermediate_size (int | None): Forward/critic feed-forward width; None uses shared width.
        forward_num_latent_tokens (int | None): Forward/critic latent slots; None uses shared slots.
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
    actor_hidden_size: int | None = None
    actor_num_attention_heads: int | None = None
    actor_intermediate_size: int | None = None
    actor_num_latent_tokens: int | None = None
    forward_hidden_size: int | None = None
    forward_num_attention_heads: int | None = None
    forward_intermediate_size: int | None = None
    forward_num_latent_tokens: int | None = None

    @classmethod
    def from_sizes(
        cls,
        embedding_size: int,
        state_token_size: int,
        actor_size: str = "small",
        forward_size: str = "small",
        backward_size: str = "small",
        **overrides: object,
    ) -> "BfmConfig":
        """Build independently sized actor, forward/critic, and backward networks.

        Args:
            embedding_size (int): Positive policy-encoder output width.
            state_token_size (int): Positive feature count of each state token.
            actor_size (str): Actor preset name: tiny, small, medium, or large.
            forward_size (str): Forward-map and critic preset name.
            backward_size (str): Backward-map state adapter and MLP preset name.
            **overrides (object): BfmConfig fields replacing preset dimensions.

        Returns:
            BfmConfig: Validated configuration for all BFM networks.

        Raises:
            ValueError: A size name or resulting dimension is invalid.
            TypeError: An override is not a BfmConfig field.
        """
        for role, size, presets in (
            ("actor", actor_size, ACTOR_FORWARD_SIZES),
            ("forward", forward_size, ACTOR_FORWARD_SIZES),
            ("backward", backward_size, BACKWARD_SIZES),
        ):
            if size not in presets:
                raise ValueError(f"unknown {role} size {size!r}; choose from {tuple(presets)}")
        values = {}
        for role, size in (("actor", actor_size), ("forward", forward_size)):
            for name, value in ACTOR_FORWARD_SIZES[size].items():
                values[f"{role}_{name}"] = value
        values.update(BACKWARD_SIZES[backward_size])
        for name in ("hidden_size", "num_attention_heads", "intermediate_size", "num_latent_tokens"):
            if name in overrides:
                for role in ("actor", "forward"):
                    values[f"{role}_{name}"] = overrides[name]
        return cls(embedding_size=embedding_size, state_token_size=state_token_size, **{**values, **overrides})

    def network_dimensions(self, role: str) -> tuple[int, int, int, int]:
        """Return width, heads, feed-forward width, and latent slots for a network.

        Args:
            role (str): ``actor`` or ``forward``; critics use ``forward``.

        Returns:
            tuple[int, int, int, int]: Positive dimensions in construction order.

        Raises:
            ValueError: The role is neither actor nor forward.
        """
        if role not in ("actor", "forward"):
            raise ValueError(f"unknown network role {role!r}")
        dimensions: list[int] = []
        for name in ("hidden_size", "num_attention_heads", "intermediate_size", "num_latent_tokens"):
            selected = getattr(self, f"{role}_{name}")
            dimensions.append(getattr(self, name) if selected is None else selected)
        return dimensions[0], dimensions[1], dimensions[2], dimensions[3]

    def __post_init__(self) -> None:
        """Validate all shared and per-network dimensions after construction.

        Returns:
            None: The frozen configuration remains valid.

        Raises:
            ValueError: A size, dropout, or actor standard deviation is invalid.
        """
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
        for role in ("actor", "forward"):
            width, heads, intermediate, slots = self.network_dimensions(role)
            if min(width, heads, intermediate, slots) <= 0 or width % heads:
                raise ValueError(f"{role} dimensions must be positive and width divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.actor_std < 0:
            raise ValueError("actor_std must be non-negative")
