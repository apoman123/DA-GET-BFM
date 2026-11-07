# Graph Report - DA-GET-BFM  (2026-11-06)

## Corpus Check
- 10 files · ~13,764 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 318 nodes · 578 edges · 10 communities (6 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.9)
- Token cost: 48,165 input · 0 output

## Community Hubs (Navigation)
- URDF Parsing & Graph Build
- GET-Zero Paper & README Concepts
- Encoder Config & Package
- Embodiment Graph & Dynamics
- Transformer Building Blocks
- Batched Embodiment Ops
- Tokenizer & Collation
- Joint Feature Config
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `GetConfig` - 21 edges
2. `GetEncoder` - 15 edges
3. `EmbodimentGraph` - 13 edges
4. `GetEncoderTest` - 12 edges
5. `TransformerComponentsTest` - 12 edges
6. `parse_urdf()` - 12 edges
7. `JointFeatureConfig` - 11 edges
8. `EmbodimentBatch` - 10 edges
9. `quadruped()` - 10 edges
10. `humanoid()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Dynamic Mass Matrix Attention Bias` --semantically_similar_to--> `Embodiment Graph Structural Attention Bias`  [INFERRED] [semantically similar]
  model/README.md → raw/get-zero-paper_github_io.md
- `EmbodimentGraph` --implements--> `Embodiment Graph (joints as nodes, links as edges)`  [INFERRED]
  model/README.md → raw/get-zero-paper_github_io.md
- `GetEncoderTest` --uses--> `GetConfig`  [INFERRED]
  tests/test_get_encoder.py → model/configuration_get.py
- `GetEncoderTest` --uses--> `GetEncoder`  [INFERRED]
  tests/test_get_encoder.py → model/modeling_get.py
- `GetEncoderTest` --uses--> `GetGraphAttentionBias`  [INFERRED]
  tests/test_get_encoder.py → model/modeling_get.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GET Attention Bias Terms** — model_readme_attention_bias, model_readme_spd_encoding, model_readme_parent_child_encoding, model_readme_mass_encoding [EXTRACTED 1.00]
- **GET Per-DoF Token Composition** — model_readme_per_dof_token, model_readme_o_vg, model_readme_o_fg, model_readme_o_vl, model_readme_o_fl [EXTRACTED 1.00]
- **GET-Zero Core Mechanisms for Zero-shot Generalization** — raw_get_zero_paper_github_io_structural_attention_bias, raw_get_zero_paper_github_io_per_joint_tokens, raw_get_zero_paper_github_io_self_modeling, raw_get_zero_paper_github_io_behavior_cloning, raw_get_zero_paper_github_io_zero_shot_generalization [EXTRACTED 1.00]

## Communities (10 total, 4 thin omitted)

### Community 0 - "URDF Parsing & Graph Build"
Cohesion: 0.07
Nodes (23): _collect_tokens(), add_link(), _frozen_motion(), _joint_features(), _limits(), _reorder(), _skew(), _Token (+15 more)

### Community 1 - "GET-Zero Paper & README Concepts"
Cohesion: 0.05
Nodes (35): collate_embodiments, Cross-attention to Context Tokens, EmbodimentGraph, forward_kinematics(q), GET Encoder Implementation, GetConfig.get_zero, GetEncoder, GetTokenizer (+27 more)

### Community 2 - "Encoder Config & Package"
Cohesion: 0.07
Nodes (6): GetConfig, GetEncoder, GetEncoderLayer, GetGraphAttentionBias, graph_distances(), GetEncoderTest

### Community 3 - "Embodiment Graph & Dynamics"
Cohesion: 0.08
Nodes (9): double_pendulum(), EmbodimentGraphTest, humanoid(), joint(), KinematicsTest, link(), quadruped(), robot() (+1 more)

### Community 4 - "Transformer Building Blocks"
Cohesion: 0.07
Nodes (5): AttentionBlock, CrossAttentionBlock, NumericTokenizer, SinusoidalPositionEmbedding, TransformerComponentsTest

### Community 6 - "Tokenizer & Collation"
Cohesion: 0.11
Nodes (3): collate_embodiments(), GetTokenizer, TokenizationAndEncoderTest

## Knowledge Gaps
- **12 isolated node(s):** `da-get-bfm`, `uv (environment manager)`, `Variable Global Observation (o_vg)`, `Fixed Global Observation (o_fg)`, `Variable Local Per-joint Observation (o_vl,j)` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 144 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GetConfig` connect `Encoder Config & Package` to `Embodiment Graph & Dynamics`, `Tokenizer & Collation`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `GetConfig` (e.g. with `GetEncoder` and `GetEncoderLayer`) actually correct?**
  _`GetConfig` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `da-get-bfm`, `uv (environment manager)`, `Variable Global Observation (o_vg)` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `URDF Parsing & Graph Build` be split into smaller, more focused modules?**
  _Cohesion score 0.06663141195134849 - nodes in this community are weakly interconnected._
- **Why does `EmbodimentBatch` connect `Batched Embodiment Ops` to `URDF Parsing & Graph Build`, `Encoder Config & Package`, `Tokenizer & Collation`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GetEncoder` (e.g. with `GetConfig` and `GetEncoderTest`) actually correct?**
  _`GetEncoder` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `GET-Zero Paper & README Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.05380852550663871 - nodes in this community are weakly interconnected._