"""Configuration for the graph embodiment transformer encoder."""

from dataclasses import dataclass

from model.physics_schema import ACTUATOR_FEATURE_SIZE, DYNAMICS_FEATURE_SIZE, RIGID_BODY_FEATURE_SIZE

ENCODER_SIZES: dict[str, dict[str, int]] = {
    "tiny": {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 256},
    "small": {"hidden_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4, "intermediate_size": 512},
    "medium": {"hidden_size": 256, "num_hidden_layers": 6, "num_attention_heads": 8, "intermediate_size": 1024},
    "large": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 16, "intermediate_size": 2048},
}


@dataclass(frozen=True)
class DaGetConfig:
    """Hold validated dimensions and graph encoding options for ``DaGetEncoder``.

    Attributes:
        input_size (int): Positive number of features in each joint token.
        hidden_size (int): Positive encoder width, divisible by the head count.
        num_hidden_layers (int): Positive number of transformer layers.
        num_attention_heads (int): Positive self- and cross-attention head count.
        intermediate_size (int): Positive feed-forward hidden width.
        context_size (int | None): Positive context-token width when cross-attention
            is enabled; defaults to hidden_size.
        cross_attention_layers (tuple[int, ...]): Zero-based layer indices that
            apply cross-attention after self-attention.
        max_graph_distance (int): Largest separately embedded graph distance;
            greater or disconnected distances share one overflow bucket.
        hidden_dropout_prob (float): Residual and feed-forward dropout in [0, 1).
        mass_epsilon (float): Positive stabilizer added to the mass denominator.
        use_spatial_encoding (bool): Whether to embed undirected graph distance.
        use_parent_child_encoding (bool): Whether to embed directed ancestry.
        use_mass_encoding (bool): Whether to project the normalized mass matrix.
        use_input_layer_norm (bool): Whether to layer-normalize each raw joint
            token over its features before the input projection, as GET-Zero
            does because token features have very different scales.
        use_physics_cross_attention (bool): Whether observation tokens attend to
            rigid-body, actuator, and dynamics tokens ("Physics Cross
            Attention"). The encoder then requires physics features.
        physics_cross_attention_layers (tuple[int, ...] | None): Layers with
            Physics Cross Attention; None means every layer.
        rigid_body_feature_size (int): Rigid-body features per joint.
        actuator_feature_size (int): Actuator features per joint.
        dynamics_feature_size (int): Dynamics features per joint.
        physics_tokenizer_hidden_size (int | None): Hidden width of each physics
            tokenizer MLP; defaults to hidden_size.
    """

    input_size: int
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 512
    context_size: int | None = None
    cross_attention_layers: tuple[int, ...] = ()
    max_graph_distance: int = 32
    hidden_dropout_prob: float = 0.0
    mass_epsilon: float = 1e-8
    use_spatial_encoding: bool = True
    use_parent_child_encoding: bool = True
    use_mass_encoding: bool = True
    use_input_layer_norm: bool = True
    use_physics_cross_attention: bool = False
    physics_cross_attention_layers: tuple[int, ...] | None = None
    rigid_body_feature_size: int = RIGID_BODY_FEATURE_SIZE
    actuator_feature_size: int = ACTUATOR_FEATURE_SIZE
    dynamics_feature_size: int = DYNAMICS_FEATURE_SIZE
    physics_tokenizer_hidden_size: int | None = None

    @classmethod
    def from_size(cls, input_size: int, size: str = "small", **overrides: object) -> "DaGetConfig":
        """Build a validated encoder configuration from a named architecture size.

        Args:
            input_size (int): Positive raw joint-token feature count.
            size (str): One of ``tiny``, ``small``, ``medium``, or ``large``.
            **overrides (object): Configuration fields replacing preset values.

        Returns:
            DaGetConfig: Configuration with the selected dimensions and overrides.

        Raises:
            ValueError: The size name or resulting configuration is invalid.
            TypeError: An override is not a configuration field.
        """
        if size not in ENCODER_SIZES:
            raise ValueError(f"unknown encoder size {size!r}; choose from {tuple(ENCODER_SIZES)}")
        return cls(input_size=input_size, **{**ENCODER_SIZES[size], **overrides})

    @classmethod
    def da_get_zero(cls, input_size: int, **overrides: object) -> "DaGetConfig":
        """Return the GET-Zero LEAP Hand architecture for a token width.

        The paper's model uses 8 layers, 16 heads, width 512, feed-forward
        width 1024, dropout 0.1, and input layer normalization.

        Args:
            input_size (int): Raw joint token width, for example
                ``DaGetTokenizer.token_size``.
            **overrides (object): Fields to change from the paper values.

        Returns:
            DaGetConfig: Validated configuration.
        """
        values = {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "intermediate_size": 1024,
            "hidden_dropout_prob": 0.1,
            "use_input_layer_norm": True,
        }
        values.update(overrides)
        return cls(input_size=input_size, **values)

    def __post_init__(self) -> None:
        """Reject invalid dimensions and layer locations.

        Returns:
            None: The frozen configuration is valid after construction.

        Raises:
            ValueError: A dimension, probability, epsilon, or layer index is invalid.
        """
        dimensions = (
            self.input_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.max_graph_distance,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all dimensions and max_graph_distance must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.context_size is not None and self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if not 0 <= self.hidden_dropout_prob < 1:
            raise ValueError("hidden_dropout_prob must be in [0, 1)")
        if self.mass_epsilon <= 0:
            raise ValueError("mass_epsilon must be positive")
        if len(set(self.cross_attention_layers)) != len(
            self.cross_attention_layers
        ) or any(
            not 0 <= index < self.num_hidden_layers
            for index in self.cross_attention_layers
        ):
            raise ValueError("cross_attention_layers must contain unique valid indices")
        physics_layers = self.physics_cross_attention_layers
        if physics_layers is not None and (
            len(set(physics_layers)) != len(physics_layers)
            or any(not 0 <= index < self.num_hidden_layers for index in physics_layers)
        ):
            raise ValueError("physics_cross_attention_layers must contain unique valid indices")
        physics_sizes = (
            self.rigid_body_feature_size,
            self.actuator_feature_size,
            self.dynamics_feature_size,
        )
        if any(size <= 0 for size in physics_sizes) or (
            self.physics_tokenizer_hidden_size is not None
            and self.physics_tokenizer_hidden_size <= 0
        ):
            raise ValueError("physics feature and tokenizer sizes must be positive")
        schema = (RIGID_BODY_FEATURE_SIZE, ACTUATOR_FEATURE_SIZE, DYNAMICS_FEATURE_SIZE)
        if self.use_physics_cross_attention and physics_sizes != schema:
            raise ValueError(
                f"physics feature sizes {physics_sizes} differ from the schema {schema} "
                "(rigid body, actuator, dynamics). Checkpoints from before the torque-free "
                "schema (25, 6, 7) cannot be loaded; pretrain again."
            )

    @property
    def physics_layers(self) -> tuple[int, ...]:
        """Return the layer indices with Physics Cross Attention.

        Returns:
            tuple[int, ...]: Empty when the mechanism is disabled.
        """
        if not self.use_physics_cross_attention:
            return ()
        if self.physics_cross_attention_layers is None:
            return tuple(range(self.num_hidden_layers))
        return tuple(sorted(self.physics_cross_attention_layers))
