# GET encoder

`GetEncoder` maps per-joint features to per-joint hidden states. Its config and
model live in separate modules, following the layout of Hugging Face models.
Each layer applies graph-biased self-attention, optional cross-attention, and a
feed-forward network. The encoder does not include policy or self-modeling heads.

```python
import torch

from model import GetConfig, GetEncoder

config = GetConfig(
    input_size=6,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    context_size=64,
    cross_attention_layers=(1, 3),
)
encoder = GetEncoder(config)

joint_features = torch.randn(2, 3, 6)
parent_indices = torch.tensor([[-1, 0, 1], [-1, 0, 0]])
mass_matrix = torch.eye(3).expand(2, -1, -1).clone()
context = torch.randn(2, 5, 64)
hidden_states = encoder(
    joint_features,
    parent_indices,
    mass_matrix=mass_matrix,
    context_states=context,
)
assert hidden_states.shape == (2, 3, 128)
```

`parent_indices[b, j]` identifies joint `j`'s direct parent or uses `-1` for a
root. `joint_mask` and `context_mask` use `True` for valid tokens. For padded
joints, set `parent_indices` to `-1` and their mass-matrix rows and columns to
zero. The output at padded positions is zero. `adjacency_matrix` adds undirected
edges, including edges between roots of independent serial chains used by the
GET-Zero embodiment graph.

Spatial distance is the shortest path over undirected edges. Parent and child
distances count directed ancestor steps in each direction. Each attention head
in each encoder layer learns separate scalar embeddings for those distances.
Unconnected pairs and distances above `max_graph_distance` share an overflow
bucket. A linear projection of each normalized inertia entry also contributes
to the self-attention logit:

```text
normalized_mass[i, j] = M(q)[i, j] / (sqrt(M(q)[i, i] * M(q)[j, j]) + mass_epsilon)
attention_bias[i, j] = spatial[i, j] + parent[i, j] + child[i, j]
                       + linear(normalized_mass[i, j])
```

The mass matrix is supplied at each forward pass, so changes in `q` change the
dynamic bias. The module requires a symmetric, finite mass matrix with a
positive diagonal for every valid joint. Cross-attention is applied after
self-attention in the zero-based layers listed in `cross_attention_layers`.
