"""Configuration for the graph embodiment transformer encoder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GetConfig:
    """Hold validated dimensions and graph encoding options for ``GetEncoder``.

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
