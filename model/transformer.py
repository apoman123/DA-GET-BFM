"""Reusable blocks for numerical token sequences and Transformer attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class NumericTokenizer(nn.Module):
    """Map each numerical feature vector to a token embedding with an MLP.

    Attributes:
        input_dim (int): Number of features per token, which must be positive.
        embedding_dim (int): Output width of each token, which must be positive.
        fc1 (nn.Linear): Projection from input_dim to the MLP hidden width.
        activation (nn.GELU): Nonlinear activation between projections.
        fc2 (nn.Linear): Projection from the MLP hidden width to embedding_dim.
    """

    def __init__(
        self, input_dim: int, embedding_dim: int, mlp_hidden_dim: int | None = None
    ) -> None:
        """Create a two-layer MLP that processes each sequence position separately.

        Args:
            input_dim (int): Positive number of numerical features per position.
            embedding_dim (int): Positive output token width.
            mlp_hidden_dim (int | None): Optional positive MLP hidden width;
                defaults to four times embedding_dim.

        Returns:
            None: Initializes trainable projection parameters.

        Raises:
            ValueError: Any effective dimension is not positive.
        """
        super().__init__()
        if input_dim <= 0 or embedding_dim <= 0:
            raise ValueError("input_dim and embedding_dim must be positive")
        if mlp_hidden_dim is None:
            mlp_hidden_dim = 4 * embedding_dim
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(input_dim, mlp_hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Convert each numerical vector into one token embedding.

        Args:
            values (torch.Tensor): Float tensor of shape
                ``[batch, seq, input_dim]``. Shape ``[batch, seq]`` is also
                accepted when input_dim is one, with each scalar treated as
                a separate token.

        Returns:
            torch.Tensor: Token embeddings of shape
                ``[batch, seq, embedding_dim]``.

        Raises:
            ValueError: The input rank or feature width does not match input_dim.
        """
        if values.ndim == 2 and self.input_dim == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError("values must have shape [batch, seq, input_dim]")
        return self.fc2(self.activation(self.fc1(values)))


class SinusoidalPositionEmbedding(nn.Module):
    """Generate fixed sinusoidal embeddings for nonnegative token positions.

    Attributes:
        embedding_dim (int): Width of each position vector, which must be positive.
        base (float): Positive frequency base used by sine and cosine waves.
    """

    def __init__(self, embedding_dim: int, base: float = 10000.0) -> None:
        """Store the sinusoidal frequency settings without trainable weights.

        Args:
            embedding_dim (int): Positive output position width.
            base (float): Positive frequency base; defaults to 10000.

        Returns:
            None: Initializes a parameter-free module.

        Raises:
            ValueError: embedding_dim or base is not positive.
        """
        super().__init__()
        if embedding_dim <= 0 or base <= 0:
            raise ValueError("embedding_dim and base must be positive")
        self.embedding_dim = embedding_dim
        self.base = base

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Return sine on even dimensions and cosine on odd dimensions.

        Args:
            position_ids (torch.Tensor): Nonnegative integer positions of shape
                ``[batch, seq]``. Positions may be nonconsecutive.

        Returns:
            torch.Tensor: Float32 embeddings of shape
                ``[batch, seq, embedding_dim]`` on the input device. Add the
                result to token embeddings after matching their dtype if needed.

        Raises:
            ValueError: position_ids has the wrong rank, is not integral, or
                contains negative positions.
        """
        if (
            position_ids.ndim != 2
            or position_ids.dtype not in (torch.int32, torch.int64)
            or torch.any(position_ids < 0)
        ):
            raise ValueError(
                "position_ids must contain nonnegative integers of shape [batch, seq]"
            )
        even_dimensions = torch.arange(
            0, self.embedding_dim, 2, device=position_ids.device, dtype=torch.float32
        )
        inverse_frequencies = torch.exp(
            -math.log(self.base) * even_dimensions / self.embedding_dim
        )
        angles = position_ids.to(torch.float32).unsqueeze(-1) * inverse_frequencies
        embeddings = torch.empty(
            (*position_ids.shape, self.embedding_dim),
            device=position_ids.device,
            dtype=torch.float32,
        )
        embeddings[..., 0::2] = angles.sin()
        embeddings[..., 1::2] = angles[..., : self.embedding_dim // 2].cos()
        return embeddings


class AttentionBlock(nn.Module):
    """Apply pre-normalized multi-head self-attention with a residual connection.

    Attributes:
        hidden_size (int): Width of each input and output token.
        norm (nn.LayerNorm): Normalization applied before self-attention.
        attention (nn.MultiheadAttention): Batch-first self-attention module.
        dropout (nn.Dropout): Dropout applied to the attention output.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Build a self-attention block for batch-first token sequences.

        Args:
            hidden_size (int): Positive token width divisible by num_heads.
            num_heads (int): Positive number of attention heads.
            dropout (float): Attention and output dropout probability in [0, 1).
            bias (bool): Whether attention projections use bias.

        Returns:
            None: Initializes normalization and attention parameters.

        Raises:
            ValueError: A dimension or dropout probability is invalid.
        """
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, bias=bias, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Attend to tokens in the input sequence and add the residual input.

        Args:
            hidden_states (torch.Tensor): Float tensor of shape
                ``[batch, seq, hidden_size]`` used for queries, keys, and values.
            attention_mask (torch.Tensor | None): Optional boolean blocking mask
                or floating additive attention bias of shape ``[seq, seq]`` or
                ``[batch * num_heads, seq, seq]``.
            key_padding_mask (torch.Tensor | None): Optional boolean mask of
                shape ``[batch, seq]``; True marks an ignored key token.
            causal (bool): If True, block attention to future positions.

        Returns:
            torch.Tensor: Updated tokens of shape
                ``[batch, seq, hidden_size]``.

        Raises:
            ValueError: Input shape or mask dtype is invalid.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, seq, hidden_size]")
        if (
            attention_mask is not None
            and attention_mask.dtype != torch.bool
            and not attention_mask.is_floating_point()
        ):
            raise ValueError("attention_mask must be boolean or floating point")
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be boolean")
        if causal:
            seq_len = hidden_states.shape[1]
            future_mask = torch.ones(
                (seq_len, seq_len), device=hidden_states.device, dtype=torch.bool
            ).triu(diagonal=1)
            if attention_mask is None:
                attention_mask = future_mask
            elif attention_mask.dtype == torch.bool:
                attention_mask = attention_mask | future_mask
            else:
                attention_mask = attention_mask.masked_fill(future_mask, float("-inf"))
        normalized = self.norm(hidden_states)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return hidden_states + self.dropout(attended)


class CrossAttentionBlock(nn.Module):
    """Attend from one token sequence to a separate context sequence.

    Attributes:
        hidden_size (int): Width of query and output tokens.
        context_size (int): Width of context tokens used as keys and values.
        query_norm (nn.LayerNorm): Normalization applied to query tokens.
        context_norm (nn.LayerNorm): Normalization applied to context tokens.
        attention (nn.MultiheadAttention): Batch-first cross-attention module.
        dropout (nn.Dropout): Dropout applied to the attention output.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Build a cross-attention block with optional distinct context width.

        Args:
            hidden_size (int): Positive query width divisible by num_heads.
            num_heads (int): Positive number of attention heads.
            context_size (int | None): Optional positive context width; defaults
                to hidden_size.
            dropout (float): Attention and output dropout probability in [0, 1).
            bias (bool): Whether attention projections use bias.

        Returns:
            None: Initializes normalization and attention parameters.

        Raises:
            ValueError: A dimension or dropout probability is invalid.
        """
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if context_size is None:
            context_size = hidden_size
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(context_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            bias=bias,
            kdim=context_size,
            vdim=context_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use query tokens to attend over context keys and values.

        Args:
            hidden_states (torch.Tensor): Query tokens of shape
                ``[batch, target_seq, hidden_size]``.
            context_states (torch.Tensor): Key/value tokens of shape
                ``[batch, source_seq, context_size]``.
            attention_mask (torch.Tensor | None): Optional boolean mask of shape
                ``[target_seq, source_seq]`` or
                ``[batch * num_heads, target_seq, source_seq]``; True blocks
                a query-key pair.
            context_padding_mask (torch.Tensor | None): Optional boolean mask of
                shape ``[batch, source_seq]``; True marks an ignored context token.

        Returns:
            torch.Tensor: Updated query tokens of shape
                ``[batch, target_seq, hidden_size]``.

        Raises:
            ValueError: Input shapes or mask dtypes are invalid.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have shape [batch, target_seq, hidden_size]"
            )
        if (
            context_states.ndim != 3
            or context_states.shape[0] != hidden_states.shape[0]
            or context_states.shape[-1] != self.context_size
        ):
            raise ValueError(
                "context_states must have shape [batch, source_seq, context_size]"
            )
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean")
        if (
            context_padding_mask is not None
            and context_padding_mask.dtype != torch.bool
        ):
            raise ValueError("context_padding_mask must be boolean")
        query = self.query_norm(hidden_states)
        context = self.context_norm(context_states)
        attended, _ = self.attention(
            query,
            context,
            context,
            attn_mask=attention_mask,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        return hidden_states + self.dropout(attended)
