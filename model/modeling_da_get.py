"""Graph embodiment encoder with structural, dynamic, physics, and context attention.

Each encoder layer runs, in order:

1. graph self-attention over the observation joint tokens, with the GET-Zero
   structural bias and the mass-matrix bias;
2. Physics Cross Attention (optional): every observation token attends to the
   rigid-body, actuator, and dynamics tokens of all joints;
3. context cross-attention (optional) to external tokens;
4. a feed-forward block.

The zero-shot proposal adds a root token to every Physics Cross Attention
memory (C1). Its remaining optional mechanisms are:

* dimensionless physics features, divided by units built from the robot's total
  mass, size and ``sqrt(L / g)`` (A3);
* ModuMorph-style contextual modulation (Xiong, Beck & Whiteson, "Universal
  Morphology Control via Contextual Modulation", ICML 2023, PMLR 202:38286-38300):
  a morphology-only attention bias per layer and a hypernetwork correction of the
  input projection, both from each joint's static physics tokens (D4);
* learned mask embeddings for masked physics modeling (E3), like BERT's ``[MASK]``
  token (Devlin et al., NAACL 2019, DOI 10.18653/v1/N19-1423) and MAE's shared, learned
  mask token (He et al., CVPR 2022, DOI 10.1109/CVPR52688.2022.01553).

References:

* GET-Zero (Patel & Song, ICRA 2025, DOI 10.1109/ICRA55743.2025.11127922,
  arXiv:2407.15002): the one-token-per-joint encoder,
  the input LayerNorm and linear embedding, and the structural attention bias, a
  learned scalar per head for the undirected shortest-path distance (GET-Zero follows
  Graphormer's spatial encoding here: Ying et al., "Do Transformers Really Perform
  Badly for Graph Representation?", NeurIPS 2021, arXiv:2106.05234) plus GET-Zero's
  directed parent and child distance tables. The bias tables follow the design of
  GET-Zero's ``SpatialEncodingTransformerEncoderLayer``
  (``get_zero/distill/models/embodiment_attention.py`` in
  https://github.com/real-stanford/get_zero, commit 279652d, no license file); they are
  reimplemented here, no code is copied. The mass-matrix bias and Physics Cross
  Attention are new in this repository.
* ``graph_distances`` computes all-pairs shortest paths with the Floyd-Warshall
  recursion (Floyd, Commun. ACM 5(6):345, 1962, DOI 10.1145/367766.368168; Warshall,
  J. ACM 9(1):11-12, 1962, DOI 10.1145/321105.321107).
* Attention, pre-norm and GELU: see ``model.transformer``. The asinh feature scaling
  is the inverse hyperbolic sine transformation (Burbidge, Magee & Robb, J. Am. Stat.
  Assoc. 83(401):123-127, 1988, DOI 10.1080/01621459.1988.10478575).
* Contextual modulation: ModuMorph (above) and hypernetworks (Ha, Dai & Le, ICLR 2017,
  arXiv:1609.09106).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from model.configuration_da_get import DaGetConfig
from model.physics_schema import (
    STANDARD_GRAVITY,
    dimension_exponents,
    dimensionless_scales,
    feature_names,
    feature_scales,
)
from model.transformer import AttentionBlock, CrossAttentionBlock

PHYSICS_GROUPS = ("rigid_body", "actuator", "dynamics")
MASKABLE_GROUPS = ("rigid_body", "actuator")
TOTAL_MASS_INDEX = feature_names("root").index("total_mass")
LENGTH_INDEX = feature_names("root").index("characteristic_length")
PRISMATIC_INDEX = feature_names("rigid_body").index("type_prismatic")


def physics_units(
    group: str, total_mass: torch.Tensor, length: torch.Tensor, prismatic: torch.Tensor
) -> torch.Tensor:
    """Return each feature's unit ``m^a L^b T^c`` with ``T = sqrt(L / g)``.

    Args:
        group (str): ``rigid_body``, ``actuator``, or ``dynamics``.
        total_mass (torch.Tensor): ``[B]`` total mass in kg.
        length (torch.Tensor): ``[B]`` characteristic length in m.
        prismatic (torch.Tensor): Bool ``[B, J]`` prismatic joints.

    Returns:
        torch.Tensor: ``[B, J, F]`` positive units.
    """
    revolute, prismatic_exponents = dimension_exponents(group)
    dtype, device = total_mass.dtype, total_mass.device
    revolute = torch.tensor(revolute, dtype=dtype, device=device)
    prismatic_exponents = torch.tensor(prismatic_exponents, dtype=dtype, device=device)
    length = length.clamp_min(1e-6)
    logs = torch.stack(
        (total_mass.clamp_min(1e-9).log(), length.log(), 0.5 * (length / STANDARD_GRAVITY).log()), dim=-1
    )
    log_revolute = logs @ revolute.T
    log_prismatic = logs @ prismatic_exponents.T
    return torch.where(prismatic[..., None], log_prismatic[:, None], log_revolute[:, None]).exp()


def graph_distances(
    parent_indices: torch.Tensor,
    joint_mask: torch.Tensor,
    adjacency_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute undirected shortest paths and directed ancestry distances.

    The shortest paths use the Floyd-Warshall recursion (Floyd 1962; Warshall 1962); the
    ancestor and descendant distances are GET-Zero's parent and child matrices.

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


compute_graph_distances = graph_distances


def _spatial_buckets(spatial_distances: torch.Tensor, max_graph_distance: int) -> torch.Tensor:
    """Map shortest-path distances to embedding buckets.

    A shortest path over J joints is at most J - 1 long, so any value of at least
    J marks an unreachable pair. Those and distances above ``max_graph_distance``
    share the overflow bucket, which therefore does not depend on how far a batch
    was padded.

    Args:
        spatial_distances (torch.Tensor): Integer ``[batch, joints, joints]``.
        max_graph_distance (int): Largest separately embedded distance.

    Returns:
        torch.Tensor: Bucket indices in ``[0, max_graph_distance + 1]``.
    """
    joint_count = spatial_distances.shape[-1]
    overflow = max_graph_distance + 1
    return torch.where(
        spatial_distances >= joint_count, overflow, spatial_distances.clamp(max=overflow)
    )


class PhysicsFeatureTokenizer(nn.Module):
    """Embed one group of raw physics features into one token per joint.

    Each feature is first mapped to ``asinh(x / s)`` (the inverse hyperbolic sine
    transformation, Burbidge, Magee & Robb 1988) with the scale ``s`` from
    ``model.physics_schema`` (pass-through features keep their value), so
    quantities spanning orders of magnitude across robots stay well conditioned.
    A two-layer MLP then maps the features to ``hidden_size``, and a learned
    group embedding marks which group the token belongs to.

    Attributes:
        scales (tuple[float, ...]): Per-feature asinh scales; 0 passes through.
            Kept as constants, not parameters or buffers, so checkpoints do not
            depend on them.
        mlp (nn.Sequential): Feature MLP.
        group_embedding (nn.Parameter): ``[hidden_size]`` group marker.
    """

    def __init__(
        self, group: str, input_size: int, hidden_size: int, mlp_size: int, dimensionless: bool = False
    ) -> None:
        """Create the tokenizer for one feature group.

        Args:
            group (str): ``rigid_body``, ``actuator``, ``dynamics``, or ``root``.
                Scales are read from the schema when its size matches ``input_size``.
            input_size (int): Features per joint.
            hidden_size (int): Output token width.
            mlp_size (int): MLP hidden width.
            dimensionless (bool): Features arrive divided by their units, so use
                the schema's dimensionless scales.

        Returns:
            None: Registers parameters.
        """
        super().__init__()
        scales = dimensionless_scales(group) if dimensionless else feature_scales(group)
        if len(scales) != input_size:
            scales = (0.0,) * input_size
        self.scales = tuple(float(scale) for scale in scales)
        # Device copies of the scales, made on first use: creating the tensor in every
        # forward would copy it from the host each call, which a CUDA Graph cannot record.
        # Not a buffer, because transformers' from_pretrained leaves unsaved buffers empty.
        self._scale_tensors: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_size), nn.GELU(), nn.Linear(mlp_size, hidden_size)
        )
        self.group_embedding = nn.Parameter(torch.zeros(hidden_size))

    def scaled(self, features: torch.Tensor, units: torch.Tensor | None = None) -> torch.Tensor:
        """Return the asinh-scaled features the MLP reads (also the masked-modeling target).

        Args:
            features (torch.Tensor): ``[batch, joints, input_size]`` raw features.
            units (torch.Tensor | None): Units to divide by first (dimensionless mode).

        Returns:
            torch.Tensor: Same shape.
        """
        if units is not None:
            features = features / units
        key = (features.device, features.dtype)
        if key not in self._scale_tensors:
            self._scale_tensors[key] = torch.tensor(self.scales, dtype=features.dtype, device=features.device)
        scales = self._scale_tensors[key]
        return torch.where(scales > 0, torch.asinh(features / torch.where(scales > 0, scales, 1)), features)

    def forward(self, features: torch.Tensor, units: torch.Tensor | None = None) -> torch.Tensor:
        """Embed features.

        Args:
            features (torch.Tensor): ``[batch, joints, input_size]`` raw features.
            units (torch.Tensor | None): Units to divide by first (dimensionless mode).

        Returns:
            torch.Tensor: ``[batch, joints, hidden_size]`` tokens.
        """
        if units is not None:
            return self.mlp(self.scaled(features, units)) + self.group_embedding
        key = (features.device, features.dtype)
        if key not in self._scale_tensors:
            self._scale_tensors[key] = torch.tensor(self.scales, dtype=features.dtype, device=features.device)
        scales = self._scale_tensors[key]
        scaled = torch.where(
            scales > 0, torch.asinh(features / torch.where(scales > 0, scales, 1)), features
        )
        return self.mlp(scaled) + self.group_embedding


class PhysicsCrossAttention(nn.Module):
    """Let observation joint tokens attend to the physics tokens of every joint.

    The memory is the concatenation ``[rigid_body (J), actuator (J), dynamics
    (J)]`` of physics tokens. Attention logits get a learned per-head bias from
    the graph distance between the query joint and the joint a physics token
    describes, plus a per-head bias per feature group, so a joint can focus on
    its own physics and that of nearby joints. Keys and values of the rigid-body
    and actuator tokens depend only on the embodiment and can be cached with
    ``DaGetEncoder.build_physics_memory``.

    Attributes:
        query_norm (nn.LayerNorm): Pre-norm of observation tokens.
        memory_norm (nn.LayerNorm): Norm of physics tokens.
        query (nn.Linear): Query projection.
        key (nn.Linear): Key projection.
        value (nn.Linear): Value projection.
        output (nn.Linear): Output projection.
        spatial_embedding (nn.Embedding): Graph distance to per-head bias.
        group_bias (nn.Parameter): ``[3, heads]`` per-group bias.
        root_bias (nn.Parameter): Per-head bias for the root-body token.
    """

    def __init__(self, config: DaGetConfig) -> None:
        """Build one Physics Cross Attention block.

        Args:
            config (DaGetConfig): Encoder configuration.

        Returns:
            None: Registers parameters.
        """
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = hidden // self.num_heads
        self.max_graph_distance = config.max_graph_distance
        self.dropout_prob = config.hidden_dropout_prob
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.spatial_embedding = nn.Embedding(config.max_graph_distance + 2, self.num_heads)
        self.group_bias = nn.Parameter(torch.zeros(len(PHYSICS_GROUPS), self.num_heads))
        self.root_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def memory_parameters(self) -> list[nn.Parameter]:
        """Return the parameters that cached keys and values depend on.

        Returns:
            list[nn.Parameter]: Memory norm, key, and value parameters.
        """
        return [
            *self.memory_norm.parameters(),
            *self.key.parameters(),
            *self.value.parameters(),
        ]

    def _heads(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, N, hidden]`` to ``[B, heads, N, head_size]``."""
        batch, length, _ = tokens.shape
        return tokens.view(batch, length, self.num_heads, self.head_size).transpose(1, 2)

    def project_memory(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project physics tokens to keys and values.

        Args:
            tokens (torch.Tensor): ``[batch, tokens, hidden]`` physics tokens.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Keys and values
                ``[batch, heads, tokens, head_size]``.
        """
        normalized = self.memory_norm(tokens)
        return self._heads(self.key(normalized)), self._heads(self.value(normalized))

    def attention_bias(
        self, spatial_distances: torch.Tensor, joint_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build additive attention bias for three joint groups and root.

        Args:
            spatial_distances (torch.Tensor): Integer ``[batch, joints, joints]``.
            joint_mask (torch.Tensor): Bool ``[batch, joints]``.
            dtype (torch.dtype): Logit dtype.

        Returns:
            torch.Tensor: ``[batch, heads, joints, 3 * joints + 1]``. The root
                token sits after the actuator tokens; padded physics tokens
                are -infinity.
        """
        buckets = _spatial_buckets(spatial_distances, self.max_graph_distance)
        spatial = self.spatial_embedding(buckets).permute(0, 3, 1, 2).to(dtype)
        groups = [spatial + self.group_bias[group].to(dtype)[None, :, None, None] for group in range(3)]
        masks = [joint_mask] * 3
        batch, joints = joint_mask.shape
        root = self.root_bias.to(dtype)[None, :, None, None].expand(batch, -1, joints, 1)
        groups.insert(2, root)
        masks.insert(2, torch.ones((batch, 1), dtype=torch.bool, device=joint_mask.device))
        bias = torch.cat(groups, dim=-1)
        key_mask = torch.cat(masks, dim=1)
        return bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from observation tokens to physics tokens.

        Args:
            hidden_states (torch.Tensor): ``[batch, joints, hidden]``.
            keys (torch.Tensor): ``[batch, heads, 3 * joints + 1, head_size]``.
            values (torch.Tensor): Same shape as keys.
            bias (torch.Tensor): ``[batch, heads, joints, 3 * joints + 1]``.

        Returns:
            torch.Tensor: Updated tokens with the residual added.
        """
        batch, joints, hidden = hidden_states.shape
        query = self._heads(self.query(self.query_norm(hidden_states)))
        attended = functional.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=bias,
            dropout_p=self.dropout_prob if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, joints, hidden)
        return hidden_states + self.dropout(self.output(attended))


@dataclass(frozen=True)
class PhysicsMemory:
    """Cached keys and values of the embodiment-only physics tokens.

    Rigid-body and actuator features do not change while a robot runs, so their
    tokens and their per-layer keys and values can be computed once and reused
    every control step. Build it with ``DaGetEncoder.build_physics_memory`` and
    pass it as ``physics_memory``. It records the versions of the parameters it
    was computed from; the encoder rejects it after those parameters change
    (for example after an optimizer step), so a stale cache cannot be used by
    mistake.

    Attributes:
        keys (tuple[torch.Tensor, ...]): Per physics layer,
            ``[B, heads, 2J + 1, d]``.
        values (tuple[torch.Tensor, ...]): Per physics layer, same shape.
        joint_mask (torch.Tensor): Bool ``[B, J]`` the memory was built for.
        parameter_versions (tuple[int, ...]): Versions of the source parameters.
        dynamics_units (torch.Tensor | None): ``[B, J, F_dyn]`` units of the
            dynamics features (dimensionless mode).
        context (torch.Tensor | None): ``[B, J, hidden]`` static morphology context
            of every joint (contextual modulation).
    """

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    joint_mask: torch.Tensor
    parameter_versions: tuple[int, ...]
    dynamics_units: torch.Tensor | None = None
    context: torch.Tensor | None = None


def normalize_mass_matrix(
    mass_matrix: torch.Tensor, joint_mask: torch.Tensor, epsilon: float, validate: bool = True
) -> torch.Tensor:
    """Validate ``M(q)`` and return ``M_ij / sqrt(M_ii M_jj)``, with padded diagonals set to one.

    All value checks reduce to one boolean, so a GPU waits for the device once per call.

    Args:
        mass_matrix (torch.Tensor): ``[batch, joints, joints]`` joint mass matrix.
        joint_mask (torch.Tensor): Boolean ``[batch, joints]`` valid joints.
        epsilon (float): Stabilizer added to the denominator.
        validate (bool): Check the values (off: only the shape, with no device wait).

    Returns:
        torch.Tensor: Normalized mass matrix of the same shape.

    Raises:
        ValueError: The matrix is not floating, finite and symmetric with a positive
            valid diagonal, or has the wrong shape.
    """
    batch_size, joint_count = joint_mask.shape
    if mass_matrix is None or mass_matrix.shape != (batch_size, joint_count, joint_count):
        raise ValueError("mass_matrix must have shape [batch, joints, joints]")
    if not mass_matrix.is_floating_point():
        raise ValueError("mass_matrix must be floating, finite, symmetric, and have positive valid diagonal")
    diagonal = mass_matrix.diagonal(dim1=1, dim2=2)
    if validate:
        transposed = mass_matrix.transpose(1, 2)
        valid = (
            torch.isfinite(mass_matrix).all()
            & ((diagonal > 0) | ~joint_mask).all()
            # The tolerances of torch.allclose.
            & ((mass_matrix - transposed).abs() <= 1e-8 + 1e-5 * transposed.abs()).all()
        )
        if not valid:
            raise ValueError("mass_matrix must be floating, finite, symmetric, and have positive valid diagonal")
    safe_diagonal = torch.where(joint_mask, diagonal, torch.ones_like(diagonal))
    denominator = torch.sqrt(safe_diagonal.unsqueeze(2) * safe_diagonal.unsqueeze(1))
    return mass_matrix / (denominator + epsilon)


class DaGetGraphAttentionBias(nn.Module):
    """Learn per-head spatial, parent, child, and normalized mass logit biases.

    The spatial, parent and child tables are GET-Zero's structural bias (Patel & Song,
    ICRA 2025; the spatial table follows Graphormer, Ying et al., NeurIPS 2021); the
    mass term is this repository's.

    Attributes:
        config (DaGetConfig): Encoder configuration controlling enabled biases.
        spatial_embedding (nn.Embedding | None): Undirected distance table.
        parent_embedding (nn.Embedding | None): Query-to-ancestor distance table.
        child_embedding (nn.Embedding | None): Query-to-descendant distance table.
        mass_projection (nn.Linear | None): Linear map from one normalized mass
            entry to one bias value per attention head.
    """

    def __init__(self, config: DaGetConfig) -> None:
        """Initialize independent bias parameters for one encoder layer.

        Args:
            config (DaGetConfig): Validated encoder configuration.

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
        mass_normalized: bool = False,
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
            mass_normalized (bool): ``mass_matrix`` already comes from
                ``normalize_mass_matrix``, as the encoder passes it to every layer.

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
            buckets = _spatial_buckets(spatial_distances, self.config.max_graph_distance)
            bias = bias + self.spatial_embedding(buckets).to(dtype)
        if self.parent_embedding is not None and self.child_embedding is not None:
            bias = bias + self.parent_embedding(
                parent_distances.clamp(max=max_distance)
            ).to(dtype)
            bias = bias + self.child_embedding(
                child_distances.clamp(max=max_distance)
            ).to(dtype)
        if self.mass_projection is not None:
            normalized = (
                mass_matrix
                if mass_normalized
                else normalize_mass_matrix(mass_matrix, joint_mask, self.config.mass_epsilon)
            )
            bias = bias + self.mass_projection(normalized.unsqueeze(-1)).to(dtype)
        bias = bias.masked_fill(~joint_mask[:, None, :, None], float("-inf"))
        return bias.permute(0, 3, 1, 2).reshape(
            batch_size * heads, joint_count, joint_count
        )


class DaGetMorphologyModulation(nn.Module):
    """ModuMorph-style contextual modulation from each joint's static physics tokens.

    Xiong, Beck & Whiteson (ICML 2023, PMLR 202:38286-38300) condition a universal
    morphology controller on per-limb morphology context in two ways: a hypernetwork
    (Ha, Dai & Le, ICLR 2017) that generates the limb's input embedding (and decoder)
    weights, and "fixed attention" whose weights depend on the morphology context only.
    Here both are additive to the GET encoder,
    so the option can be ablated against the GET structural bias (proposal D4):

    * ``attention_bias``: per layer and head, ``(W_q c_i) . (W_k c_j) / sqrt(d)``
      added to the self-attention logits;
    * ``input_correction``: ``x_i A(c_i) B`` added to the input projection, a
      rank-``r`` hypernetwork correction (the paper generates the full matrix;
      the low-rank factorization is [H]).

    Both corrections start at zero (zero-initialized output weights), so a new
    model starts as the unmodulated encoder.
    """

    def __init__(self, config: DaGetConfig) -> None:
        """Create the modulation parameters.

        Args:
            config (DaGetConfig): Encoder configuration.
        """
        super().__init__()
        hidden, heads = config.hidden_size, config.num_attention_heads
        self.num_heads = heads
        self.head_size = hidden // heads
        self.rank = config.modulation_rank or heads
        self.input_size = config.input_size
        self.context_norm = nn.LayerNorm(hidden)
        self.queries = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(config.num_hidden_layers))
        self.keys = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(config.num_hidden_layers))
        for key in self.keys:
            nn.init.zeros_(key.weight)
            nn.init.zeros_(key.bias)
        self.hyper_input = nn.Linear(hidden, config.input_size * self.rank)
        self.hyper_output = nn.Linear(self.rank, hidden, bias=False)
        nn.init.zeros_(self.hyper_output.weight)

    def attention_bias(self, context: torch.Tensor, layer: int, joint_mask: torch.Tensor) -> torch.Tensor:
        """Return the morphology-only logit bias of one layer.

        Args:
            context (torch.Tensor): ``[B, J, hidden]`` morphology context.
            layer (int): Layer index.
            joint_mask (torch.Tensor): Bool ``[B, J]``.

        Returns:
            torch.Tensor: ``[B * heads, J, J]``.
        """
        batch, joints, _ = context.shape
        normalized = self.context_norm(context)
        query = self.queries[layer](normalized).view(batch, joints, self.num_heads, self.head_size).transpose(1, 2)
        key = self.keys[layer](normalized).view(batch, joints, self.num_heads, self.head_size).transpose(1, 2)
        logits = query @ key.transpose(-1, -2) / self.head_size**0.5
        logits = logits.masked_fill(~joint_mask[:, None, None, :], 0.0)
        return logits.reshape(batch * self.num_heads, joints, joints)

    def input_correction(self, context: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Return the hypernetwork correction of the input projection.

        Args:
            context (torch.Tensor): ``[B, J, hidden]``.
            inputs (torch.Tensor): ``[B, J, F]`` (normalized) joint tokens.

        Returns:
            torch.Tensor: ``[B, J, hidden]``.
        """
        batch, joints, _ = context.shape
        weights = self.hyper_input(self.context_norm(context)).view(batch, joints, self.input_size, self.rank)
        return self.hyper_output((inputs[..., None, :] @ weights).squeeze(-2))


class DaGetEncoderLayer(nn.Module):
    """Apply graph self-attention, optional context attention, and an MLP.

    Attributes:
        graph_bias (DaGetGraphAttentionBias): Per-head logit bias module.
        self_attention (AttentionBlock): Pre-normalized joint self-attention.
        physics_attention (PhysicsCrossAttention | None): Physics Cross Attention.
        cross_attention (CrossAttentionBlock | None): Optional context attention.
        output_norm (nn.LayerNorm): Normalization before the feed-forward MLP.
        intermediate (nn.Linear): First feed-forward projection.
        activation (nn.GELU): Feed-forward nonlinearity.
        output (nn.Linear): Second feed-forward projection.
        dropout (nn.Dropout): Feed-forward residual dropout.
    """

    def __init__(
        self, config: DaGetConfig, use_cross_attention: bool, use_physics_attention: bool = False
    ) -> None:
        """Build one independently parameterized transformer layer.

        Args:
            config (DaGetConfig): Validated dimensions and bias options.
            use_cross_attention (bool): Whether this layer attends to context.
            use_physics_attention (bool): Whether this layer has Physics Cross
                Attention.

        Returns:
            None: Registers attention and feed-forward modules.
        """
        super().__init__()
        self.graph_bias = DaGetGraphAttentionBias(config)
        self.self_attention = AttentionBlock(
            config.hidden_size, config.num_attention_heads, config.hidden_dropout_prob
        )
        self.physics_attention = (
            PhysicsCrossAttention(config) if use_physics_attention else None
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
        physics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        mass_normalized: bool = False,
        extra_bias: torch.Tensor | None = None,
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
            physics (tuple | None): Static keys ``[B, heads, 2J + 1, d]``, static
                values, and dynamics tokens ``[B, J, hidden]``, required when
                the layer has Physics Cross Attention.
            mass_normalized (bool): ``mass_matrix`` comes from ``normalize_mass_matrix``.
            extra_bias (torch.Tensor | None): ``[batch * heads, joints, joints]``
                logits added to the graph bias (contextual modulation).

        Returns:
            torch.Tensor: Updated joint states of the same shape, zero on padding.

        Raises:
            ValueError: A cross-attention layer has no context or physics tokens.
        """
        bias = self.graph_bias(*distances, mass_matrix, joint_mask, hidden_states.dtype, mass_normalized)
        if extra_bias is not None:
            bias = bias + extra_bias.to(bias.dtype)
        hidden_states = self.self_attention(hidden_states, attention_mask=bias)
        if self.physics_attention is not None:
            if physics is None:
                raise ValueError("physics tokens are required by Physics Cross Attention")
            static_keys, static_values, dynamics_tokens = physics
            dynamic_keys, dynamic_values = self.physics_attention.project_memory(dynamics_tokens)
            hidden_states = self.physics_attention(
                hidden_states,
                torch.cat((static_keys, dynamic_keys), dim=2),
                torch.cat((static_values, dynamic_values), dim=2),
                self.physics_attention.attention_bias(
                    distances[0], joint_mask, hidden_states.dtype
                ),
            )
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


class DaGetEncoder(nn.Module):
    """Encode variable-length robot joint tokens with graph and dynamics biases.

    Attributes:
        config (DaGetConfig): Encoder dimensions and enabled mechanisms.
        input_norm (nn.LayerNorm | None): Per-token feature normalization applied
            before the projection when ``config.use_input_layer_norm`` is set.
        input_projection (nn.Linear): Joint feature to hidden-state projection.
        layers (nn.ModuleList): Independently parameterized transformer layers.
        final_norm (nn.LayerNorm): Output normalization.
        physics_tokenizers (nn.ModuleDict | None): Physics feature tokenizers,
            including ``root`` with a root token.
        physics_mask_embeddings (nn.ParameterDict | None): Mask tokens of masked
            physics modeling.
        modulation (DaGetMorphologyModulation | None): Contextual modulation.
    """

    def __init__(self, config: DaGetConfig) -> None:
        """Create a GET-style encoder from its configuration.

        Args:
            config (DaGetConfig): Validated model configuration.

        Returns:
            None: Registers projections and all encoder layers.
        """
        super().__init__()
        self.config = config
        self.input_norm = (
            nn.LayerNorm(config.input_size) if config.use_input_layer_norm else None
        )
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        physics_layers = config.physics_layers
        self.layers = nn.ModuleList(
            DaGetEncoderLayer(
                config,
                index in config.cross_attention_layers,
                index in physics_layers,
            )
            for index in range(config.num_hidden_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.physics_tokenizers = None
        self.physics_mask_embeddings = None
        self.modulation = None
        if physics_layers:
            mlp_size = config.physics_tokenizer_hidden_size or config.hidden_size
            sizes = {
                "rigid_body": config.rigid_body_feature_size,
                "actuator": config.actuator_feature_size,
                "dynamics": config.dynamics_feature_size,
            }
            tokenizers = {
                group: PhysicsFeatureTokenizer(
                    group, sizes[group], config.hidden_size, mlp_size, config.dimensionless_physics
                )
                for group in PHYSICS_GROUPS
            }
            tokenizers["root"] = PhysicsFeatureTokenizer(
                "root", config.root_feature_size, config.hidden_size, mlp_size
            )
            self.physics_tokenizers = nn.ModuleDict(tokenizers)
            if config.physics_masking:
                self.physics_mask_embeddings = nn.ParameterDict(
                    {group: nn.Parameter(torch.zeros(config.hidden_size)) for group in MASKABLE_GROUPS}
                )
            if config.use_morphology_modulation:
                self.modulation = DaGetMorphologyModulation(config)

    def _physics_attention_layers(self) -> list[PhysicsCrossAttention]:
        """Return the Physics Cross Attention blocks in layer order."""
        return [
            layer.physics_attention
            for layer in self.layers
            if layer.physics_attention is not None
        ]

    def _static_physics_versions(self) -> tuple[int, ...]:
        """Return versions of every parameter a ``PhysicsMemory`` depends on."""
        parameters = [
            *self.physics_tokenizers["rigid_body"].parameters(),
            *self.physics_tokenizers["actuator"].parameters(),
        ]
        parameters += list(self.physics_tokenizers["root"].parameters())
        if self.physics_mask_embeddings is not None:
            parameters += list(self.physics_mask_embeddings.parameters())
        for attention in self._physics_attention_layers():
            parameters += attention.memory_parameters()
        return tuple(parameter._version for parameter in parameters)

    def _check_physics_features(
        self, name: str, features: torch.Tensor | None, size: int, batch: int, joints: int
    ) -> torch.Tensor:
        """Validate one physics feature tensor."""
        if features is None or features.shape != (batch, joints, size) or not features.is_floating_point():
            raise ValueError(f"{name} must be floating [batch, joints, {size}]")
        return features

    def _check_root_features(self, root_features: torch.Tensor | None, batch: int) -> torch.Tensor:
        """Validate root features ``[batch, F_root]``."""
        size = self.config.root_feature_size
        if root_features is None or root_features.shape != (batch, size) or not root_features.is_floating_point():
            raise ValueError(f"root_features must be floating [batch, {size}] (model.physics.root_features)")
        return root_features

    def build_physics_memory(
        self,
        rigid_body_features: torch.Tensor,
        actuator_features: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        root_features: torch.Tensor | None = None,
        physics_token_mask: dict[str, torch.Tensor] | None = None,
    ) -> PhysicsMemory:
        """Precompute the embodiment-only part of Physics Cross Attention.

        Call this once per robot (or per batch of robots) at deployment, and pass
        the result as ``physics_memory`` every control step. Only the dynamics
        tokens are then embedded and projected per step. Rebuild it after any
        parameter update; the encoder raises an error on a stale memory.

        Args:
            rigid_body_features (torch.Tensor): ``[B, J, F_rb]``.
            actuator_features (torch.Tensor): ``[B, J, F_act]``.
            joint_mask (torch.Tensor | None): Bool ``[B, J]``; defaults to all.
            root_features (torch.Tensor | None): ``[B, F_root]``; required by
                Physics Cross Attention.
            physics_token_mask (dict | None): Bool ``[B, J]`` per maskable group
                (``rigid_body``, ``actuator``): tokens replaced by the learned
                mask embedding (masked physics modeling).

        Returns:
            PhysicsMemory: Cached keys and values for every physics layer.

        Raises:
            ValueError: Physics Cross Attention is disabled or shapes are wrong.
        """
        if self.physics_tokenizers is None:
            raise ValueError("Physics Cross Attention is disabled in the config")
        batch, joints = rigid_body_features.shape[:2]
        if joint_mask is None:
            joint_mask = torch.ones(
                (batch, joints), dtype=torch.bool, device=rigid_body_features.device
            )
        self._check_physics_features(
            "rigid_body_features", rigid_body_features, self.config.rigid_body_feature_size, batch, joints
        )
        self._check_physics_features(
            "actuator_features", actuator_features, self.config.actuator_feature_size, batch, joints
        )
        units = {group: None for group in PHYSICS_GROUPS}
        root_features = self._check_root_features(root_features, batch)
        if self.config.dimensionless_physics:
            prismatic = rigid_body_features[..., PRISMATIC_INDEX] > 0.5
            mass, length = root_features[:, TOTAL_MASS_INDEX], root_features[:, LENGTH_INDEX]
            units = {group: physics_units(group, mass, length, prismatic) for group in PHYSICS_GROUPS}
        group_tokens = {
            "rigid_body": self.physics_tokenizers["rigid_body"](rigid_body_features, units["rigid_body"]),
            "actuator": self.physics_tokenizers["actuator"](actuator_features, units["actuator"]),
        }
        if physics_token_mask:
            if self.physics_mask_embeddings is None:
                raise ValueError("physics_token_mask needs DaGetConfig.physics_masking")
            for group, mask in physics_token_mask.items():
                embedding = self.physics_mask_embeddings[group].to(group_tokens[group].dtype)
                group_tokens[group] = torch.where(mask[..., None], embedding, group_tokens[group])
        pieces = [group_tokens["rigid_body"], group_tokens["actuator"]]
        pieces.append(self.physics_tokenizers["root"](root_features[:, None, :]))
        tokens = torch.cat(pieces, dim=1)
        token_mask = joint_mask.repeat(1, 2)
        token_mask = torch.cat(
            (token_mask, torch.ones((batch, 1), dtype=torch.bool, device=joint_mask.device)), dim=1
        )
        tokens = tokens.masked_fill(~token_mask[..., None], 0)
        keys, values = [], []
        for attention in self._physics_attention_layers():
            key, value = attention.project_memory(tokens)
            keys.append(key)
            values.append(value)
        context = None
        if self.modulation is not None:
            context = (group_tokens["rigid_body"] + group_tokens["actuator"]).masked_fill(~joint_mask[..., None], 0)
        return PhysicsMemory(
            tuple(keys), tuple(values), joint_mask, self._static_physics_versions(), units["dynamics"], context
        )

    def forward(
        self,
        input_values: torch.Tensor,
        parent_indices: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        joint_mask: torch.Tensor | None = None,
        adjacency_matrix: torch.Tensor | None = None,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        graph_distances: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        rigid_body_features: torch.Tensor | None = None,
        actuator_features: torch.Tensor | None = None,
        dynamics_features: torch.Tensor | None = None,
        physics_memory: PhysicsMemory | None = None,
        validate_inputs: bool = True,
        root_features: torch.Tensor | None = None,
        physics_token_mask: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode joint features at the current configuration q.

        Args:
            input_values (torch.Tensor): Float joint tokens of shape
                ``[batch, joints, config.input_size]``, one token per DoF. Build
                them with ``DaGetTokenizer`` to follow the GET-Zero layout.
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
            graph_distances (tuple | None): Precomputed integer ``[batch, joints,
                joints]`` spatial, ancestor, and descendant distances, such as
                ``EmbodimentBatch.graph_distances``. They are a function of the
                robot only, so precomputing them avoids a Python loop per step.
                When given, they replace the distances that would be computed
                from ``parent_indices`` and ``adjacency_matrix``.
            rigid_body_features (torch.Tensor | None): ``[batch, joints, 26]``
                rigid-body features (``model.physics.rigid_body_features``).
                Required by Physics Cross Attention unless ``physics_memory``
                is given.
            actuator_features (torch.Tensor | None): ``[batch, joints, 6]``
                actuator features. Same requirement as rigid_body_features.
            dynamics_features (torch.Tensor | None): ``[batch, joints, 5]``
                dynamics features at time t. Required by Physics Cross Attention.
            physics_memory (PhysicsMemory | None): Cached embodiment-only physics
                keys and values from ``build_physics_memory``. Replaces
                rigid_body_features, actuator_features and root_features.
            validate_inputs (bool): Check input values: masks with a valid
                token, a well-formed mass matrix, and the memory's mask. Shapes
                are always checked. Turn it off only for inputs checked before,
                as ``DaGetBfmPolicy`` does: each value check waits for the
                device, which a CUDA Graph cannot capture.
            root_features (torch.Tensor | None): ``[batch, 13]`` root features
                (``model.physics.root_features``); required by Physics Cross
                Attention unless ``physics_memory`` is given.
            physics_token_mask (dict | None): Masked physics tokens, see
                ``build_physics_memory``.

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
            or (validate_inputs and not torch.all(joint_mask.any(dim=1)))
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
                or (validate_inputs and not torch.all(context_mask.any(dim=1)))
            ):
                raise ValueError("context_mask must allow at least one token per item")
        elif context_mask is not None:
            raise ValueError("context_mask requires context_states")
        if graph_distances is None:
            distances = compute_graph_distances(
                parent_indices, joint_mask, adjacency_matrix
            )
        else:
            if len(graph_distances) != 3 or any(
                item.shape != (batch_size, joint_count, joint_count)
                or item.dtype not in (torch.int32, torch.int64)
                for item in graph_distances
            ):
                raise ValueError(
                    "graph_distances must hold three integer [batch, joints, joints] tensors"
                )
            distances = tuple(item.long() for item in graph_distances)
        physics = None
        if self.physics_tokenizers is not None:
            dynamics_features = self._check_physics_features(
                "dynamics_features", dynamics_features, self.config.dynamics_feature_size, batch_size, joint_count
            )
            if physics_memory is None:
                physics_memory = self.build_physics_memory(
                    rigid_body_features, actuator_features, joint_mask, root_features, physics_token_mask
                )
            elif physics_memory.parameter_versions != self._static_physics_versions():
                raise ValueError(
                    "physics_memory is stale: parameters changed since it was built; rebuild it"
                )
            elif physics_memory.joint_mask.shape != joint_mask.shape or (
                validate_inputs and not torch.equal(physics_memory.joint_mask, joint_mask)
            ):
                raise ValueError("physics_memory was built for a different joint_mask")
            dynamics_tokens = self.physics_tokenizers["dynamics"](dynamics_features, physics_memory.dynamics_units)
            dynamics_tokens = dynamics_tokens.masked_fill(~joint_mask[..., None], 0)
        elif physics_memory is not None or dynamics_features is not None:
            raise ValueError("physics inputs given but Physics Cross Attention is disabled")
        if self.config.use_mass_encoding:
            # Once per forward, not per layer: each check waits for the device.
            mass_matrix = normalize_mass_matrix(mass_matrix, joint_mask, self.config.mass_epsilon, validate_inputs)
        if self.input_norm is not None:
            input_values = self.input_norm(input_values)
        hidden_states = self.input_projection(input_values)
        context = None
        if self.modulation is not None:
            context = physics_memory.context
            hidden_states = hidden_states + self.modulation.input_correction(context, input_values)
        hidden_states = hidden_states.masked_fill(~joint_mask.unsqueeze(-1), 0)
        physics_index = 0
        for index, layer in enumerate(self.layers):
            if layer.physics_attention is not None:
                physics = (
                    physics_memory.keys[physics_index],
                    physics_memory.values[physics_index],
                    dynamics_tokens,
                )
                physics_index += 1
            hidden_states = layer(
                hidden_states,
                distances,
                joint_mask,
                mass_matrix,
                context_states,
                context_mask,
                physics if layer.physics_attention is not None else None,
                mass_normalized=True,
                extra_bias=None if context is None else self.modulation.attention_bias(context, index, joint_mask),
            )
        return self.final_norm(hidden_states).masked_fill(~joint_mask.unsqueeze(-1), 0)
