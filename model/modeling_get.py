"""Graph embodiment encoder with structural, dynamic, and context attention."""

from __future__ import annotations

import torch
from torch import nn

from model.configuration_get import GetConfig
from model.transformer import AttentionBlock, CrossAttentionBlock


def graph_distances(
    parent_indices: torch.Tensor,
    joint_mask: torch.Tensor,
    adjacency_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute undirected shortest paths and directed ancestry distances.

    Args:
        parent_indices (torch.Tensor): Integer ``[batch, joints]`` direct parent
            indices, with -1 for roots and padded joints. Parent chains must be
            acyclic and point to valid joints.
        joint_mask (torch.Tensor): Boolean ``[batch, joints]`` with True for real
            joints.
        adjacency_matrix (torch.Tensor | None): Optional symmetric boolean
            ``[batch, joints, joints]`` graph edges. Use it to connect roots of
            independent serial chains, as in GET-Zero. Parent edges are included
            automatically. Defaults to parent edges alone.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Integer tensors of
            shape ``[batch, joints, joints]`` for undirected shortest distance,
            query-to-key ancestor distance, and query-to-key descendant distance.
            Unreachable pairs have distance ``joints + 1``; unrelated ancestry
            pairs have distance zero.

    Raises:
        ValueError: Shapes, indices, graph symmetry, or parent chains are invalid.
    """
    if parent_indices.ndim != 2 or parent_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("parent_indices must be an integer [batch, joints] tensor")
    batch_size, joint_count = parent_indices.shape
    if (
        joint_count == 0
        or joint_mask.shape != parent_indices.shape
        or joint_mask.dtype != torch.bool
    ):
        raise ValueError("joint_mask must be boolean [batch, joints] with joints > 0")
    if torch.any((parent_indices < -1) | (parent_indices >= joint_count)):
        raise ValueError("parent_indices contains an out-of-range index")
    if adjacency_matrix is not None and (
        adjacency_matrix.shape != (batch_size, joint_count, joint_count)
        or adjacency_matrix.dtype != torch.bool
        or not torch.equal(adjacency_matrix, adjacency_matrix.transpose(1, 2))
    ):
        raise ValueError(
            "adjacency_matrix must be symmetric boolean [batch, joints, joints]"
        )
    unreachable = joint_count + 1
    spatial = torch.full(
        (batch_size, joint_count, joint_count),
        unreachable,
        dtype=torch.long,
        device=parent_indices.device,
    )
    parents = torch.zeros_like(spatial)
    for batch in range(batch_size):
        for joint in range(joint_count):
            if not joint_mask[batch, joint]:
                continue
            spatial[batch, joint, joint] = 0
            visited = {joint}
            current = joint
            distance = 0
            while True:
                parent = int(parent_indices[batch, current])
                if parent == -1:
                    break
                if not joint_mask[batch, parent] or parent in visited:
                    raise ValueError(
                        "parent chains must be acyclic and reference valid joints"
                    )
                distance += 1
                parents[batch, joint, parent] = distance
                if distance == 1:
                    spatial[batch, joint, parent] = 1
                    spatial[batch, parent, joint] = 1
                visited.add(parent)
                current = parent
        if adjacency_matrix is not None:
            valid_pairs = joint_mask[batch, :, None] & joint_mask[batch, None, :]
            if torch.any(adjacency_matrix[batch] & ~valid_pairs):
                raise ValueError("adjacency_matrix cannot connect padded joints")
            spatial[batch][adjacency_matrix[batch]] = 1
            spatial[batch].diagonal().masked_fill_(joint_mask[batch], 0)
        for intermediate in range(joint_count):
            spatial[batch] = torch.minimum(
                spatial[batch],
                spatial[batch, :, intermediate, None]
                + spatial[batch, None, intermediate, :],
            )
    return spatial.clamp(max=unreachable), parents, parents.transpose(1, 2)


class GetGraphAttentionBias(nn.Module):
    """Learn per-head spatial, parent, child, and normalized mass logit biases.

    Attributes:
        config (GetConfig): Encoder configuration controlling enabled biases.
        spatial_embedding (nn.Embedding | None): Undirected distance table.
        parent_embedding (nn.Embedding | None): Query-to-ancestor distance table.
        child_embedding (nn.Embedding | None): Query-to-descendant distance table.
        mass_projection (nn.Linear | None): Linear map from one normalized mass
            entry to one bias value per attention head.
    """

    def __init__(self, config: GetConfig) -> None:
        """Initialize independent bias parameters for one encoder layer.

        Args:
            config (GetConfig): Validated encoder configuration.

        Returns:
            None: Registers trainable graph and dynamics projections.
        """
        super().__init__()
        self.config = config
        buckets = config.max_graph_distance + 2
        heads = config.num_attention_heads
        self.spatial_embedding = (
            nn.Embedding(buckets, heads) if config.use_spatial_encoding else None
        )
        self.parent_embedding = (
            nn.Embedding(buckets, heads, padding_idx=0)
            if config.use_parent_child_encoding
            else None
        )
        self.child_embedding = (
            nn.Embedding(buckets, heads, padding_idx=0)
            if config.use_parent_child_encoding
            else None
        )
        self.mass_projection = (
            nn.Linear(1, heads, bias=False) if config.use_mass_encoding else None
        )

    def forward(
        self,
        spatial_distances: torch.Tensor,
        parent_distances: torch.Tensor,
        child_distances: torch.Tensor,
        mass_matrix: torch.Tensor | None,
        joint_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build an additive PyTorch attention mask for each batch and head.

        Args:
            spatial_distances (torch.Tensor): Integer ``[batch, joints, joints]``
                undirected shortest paths.
            parent_distances (torch.Tensor): Integer distances to ancestors.
            child_distances (torch.Tensor): Integer distances to descendants.
            mass_matrix (torch.Tensor | None): Symmetric positive-diagonal joint
                mass matrix ``[batch, joints, joints]`` at the current q. Required
                when mass encoding is enabled.
            joint_mask (torch.Tensor): Boolean ``[batch, joints]`` valid joints.
            dtype (torch.dtype): Floating dtype of attention hidden states.

        Returns:
            torch.Tensor: Additive logits of shape
                ``[batch * heads, joints, joints]``; padded keys are -infinity.

        Raises:
            ValueError: Required mass input is absent, malformed, nonfinite, or
                has nonpositive diagonal entries for valid joints.
        """
        batch_size, joint_count = joint_mask.shape
        heads = self.config.num_attention_heads
        bias = torch.zeros(
            (batch_size, joint_count, joint_count, heads),
            device=joint_mask.device,
            dtype=dtype,
        )
        max_distance = self.config.max_graph_distance + 1
        if self.spatial_embedding is not None:
            bias = bias + self.spatial_embedding(
                spatial_distances.clamp(max=max_distance)
            ).to(dtype)
        if self.parent_embedding is not None and self.child_embedding is not None:
            bias = bias + self.parent_embedding(
                parent_distances.clamp(max=max_distance)
            ).to(dtype)
            bias = bias + self.child_embedding(
                child_distances.clamp(max=max_distance)
            ).to(dtype)
        if self.mass_projection is not None:
            if mass_matrix is None or mass_matrix.shape != (
                batch_size,
                joint_count,
                joint_count,
            ):
                raise ValueError("mass_matrix must have shape [batch, joints, joints]")
            diagonal = mass_matrix.diagonal(dim1=1, dim2=2)
            if (
                not mass_matrix.is_floating_point()
                or not torch.isfinite(mass_matrix).all()
                or torch.any(diagonal[joint_mask] <= 0)
                or not torch.allclose(mass_matrix, mass_matrix.transpose(1, 2))
            ):
                raise ValueError(
                    "mass_matrix must be floating, finite, symmetric, and have positive valid diagonal"
                )
            safe_diagonal = torch.where(joint_mask, diagonal, torch.ones_like(diagonal))
            denominator = torch.sqrt(
                safe_diagonal.unsqueeze(2) * safe_diagonal.unsqueeze(1)
            )
            normalized = mass_matrix / (denominator + self.config.mass_epsilon)
            bias = bias + self.mass_projection(normalized.unsqueeze(-1)).to(dtype)
        bias = bias.masked_fill(~joint_mask[:, None, :, None], float("-inf"))
        return bias.permute(0, 3, 1, 2).reshape(
            batch_size * heads, joint_count, joint_count
        )


class GetEncoderLayer(nn.Module):
    """Apply graph self-attention, optional context attention, and an MLP.

    Attributes:
        graph_bias (GetGraphAttentionBias): Per-head logit bias module.
        self_attention (AttentionBlock): Pre-normalized joint self-attention.
        cross_attention (CrossAttentionBlock | None): Optional context attention.
        output_norm (nn.LayerNorm): Normalization before the feed-forward MLP.
        intermediate (nn.Linear): First feed-forward projection.
        activation (nn.GELU): Feed-forward nonlinearity.
        output (nn.Linear): Second feed-forward projection.
        dropout (nn.Dropout): Feed-forward residual dropout.
    """

    def __init__(self, config: GetConfig, use_cross_attention: bool) -> None:
        """Build one independently parameterized transformer layer.

        Args:
            config (GetConfig): Validated dimensions and bias options.
            use_cross_attention (bool): Whether this layer attends to context.

        Returns:
            None: Registers attention and feed-forward modules.
        """
        super().__init__()
        self.graph_bias = GetGraphAttentionBias(config)
        self.self_attention = AttentionBlock(
            config.hidden_size, config.num_attention_heads, config.hidden_dropout_prob
        )
        self.cross_attention = (
            CrossAttentionBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.context_size,
                config.hidden_dropout_prob,
            )
            if use_cross_attention
            else None
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        distances: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        joint_mask: torch.Tensor,
        mass_matrix: torch.Tensor | None,
        context_states: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Update joint states while keeping padded positions zero.

        Args:
            hidden_states (torch.Tensor): Joint states ``[batch, joints, hidden]``.
            distances (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Spatial,
                ancestor, and descendant distance matrices.
            joint_mask (torch.Tensor): Boolean valid-joint mask.
            mass_matrix (torch.Tensor | None): Joint-space mass matrix at q.
            context_states (torch.Tensor | None): Context tokens when enabled.
            context_mask (torch.Tensor | None): Boolean valid-context mask.

        Returns:
            torch.Tensor: Updated joint states of the same shape, zero on padding.

        Raises:
            ValueError: A cross-attention layer has no context tokens.
        """
        bias = self.graph_bias(*distances, mass_matrix, joint_mask, hidden_states.dtype)
        hidden_states = self.self_attention(hidden_states, attention_mask=bias)
        if self.cross_attention is not None:
            if context_states is None:
                raise ValueError("context_states is required by cross-attention layers")
            hidden_states = self.cross_attention(
                hidden_states,
                context_states,
                context_padding_mask=None if context_mask is None else ~context_mask,
            )
        hidden_states = hidden_states + self.dropout(
            self.output(
                self.activation(self.intermediate(self.output_norm(hidden_states)))
            )
        )
        return hidden_states.masked_fill(~joint_mask.unsqueeze(-1), 0)


class GetEncoder(nn.Module):
    """Encode variable-length robot joint tokens with graph and dynamics biases.

    Attributes:
        config (GetConfig): Encoder dimensions and enabled mechanisms.
        input_projection (nn.Linear): Joint feature to hidden-state projection.
        layers (nn.ModuleList): Independently parameterized transformer layers.
        final_norm (nn.LayerNorm): Output normalization.
    """

    def __init__(self, config: GetConfig) -> None:
        """Create a GET-style encoder from its configuration.

        Args:
            config (GetConfig): Validated model configuration.

        Returns:
            None: Registers projections and all encoder layers.
        """
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        self.layers = nn.ModuleList(
            GetEncoderLayer(config, index in config.cross_attention_layers)
            for index in range(config.num_hidden_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_values: torch.Tensor,
        parent_indices: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        joint_mask: torch.Tensor | None = None,
        adjacency_matrix: torch.Tensor | None = None,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode joint features at the current configuration q.

        Args:
            input_values (torch.Tensor): Float joint features of shape
                ``[batch, joints, config.input_size]``.
            parent_indices (torch.Tensor): Direct parent indices of shape
                ``[batch, joints]``; -1 denotes roots or padding.
            mass_matrix (torch.Tensor | None): Joint-space inertia matrix
                ``M(q)`` of shape ``[batch, joints, joints]``. Required when
                mass encoding is enabled.
            joint_mask (torch.Tensor | None): Boolean valid-joint mask; defaults
                to all joints valid.
            adjacency_matrix (torch.Tensor | None): Additional undirected graph
                edges, including connections among independent chain roots.
            context_states (torch.Tensor | None): Context tokens of shape
                ``[batch, context, config.context_size or hidden_size]``.
            context_mask (torch.Tensor | None): Boolean valid-context mask.

        Returns:
            torch.Tensor: Joint representations of shape
                ``[batch, joints, hidden_size]`` with padded joints set to zero.

        Raises:
            ValueError: Inputs or masks have invalid shapes or required data is
                missing, including an empty valid sequence.
        """
        if (
            input_values.ndim != 3
            or input_values.shape[-1] != self.config.input_size
            or not input_values.is_floating_point()
        ):
            raise ValueError(
                "input_values must be floating [batch, joints, input_size]"
            )
        batch_size, joint_count, _ = input_values.shape
        if (
            parent_indices.shape != (batch_size, joint_count)
            or parent_indices.device != input_values.device
        ):
            raise ValueError(
                "parent_indices must match input batch, joints, and device"
            )
        if joint_mask is None:
            joint_mask = torch.ones(
                (batch_size, joint_count), dtype=torch.bool, device=input_values.device
            )
        if (
            joint_mask.shape != (batch_size, joint_count)
            or joint_mask.dtype != torch.bool
            or not torch.all(joint_mask.any(dim=1))
        ):
            raise ValueError("each batch item must contain at least one valid joint")
        if context_states is not None:
            expected_width = self.config.context_size or self.config.hidden_size
            if (
                context_states.ndim != 3
                or context_states.shape[0] != batch_size
                or context_states.shape[2] != expected_width
                or context_states.shape[1] == 0
            ):
                raise ValueError("context_states has invalid batch, length, or width")
            if context_mask is not None and (
                context_mask.shape != context_states.shape[:2]
                or context_mask.dtype != torch.bool
                or not torch.all(context_mask.any(dim=1))
            ):
                raise ValueError("context_mask must allow at least one token per item")
        elif context_mask is not None:
            raise ValueError("context_mask requires context_states")
        distances = graph_distances(parent_indices, joint_mask, adjacency_matrix)
        hidden_states = self.input_projection(input_values)
        hidden_states = hidden_states.masked_fill(~joint_mask.unsqueeze(-1), 0)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                distances,
                joint_mask,
                mass_matrix,
                context_states,
                context_mask,
            )
        return self.final_norm(hidden_states).masked_fill(~joint_mask.unsqueeze(-1), 0)
