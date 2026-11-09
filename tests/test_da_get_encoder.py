"""Test GET graph distances, attention biases, cross-attention, and gradients."""

import unittest

import torch

from model.configuration_da_get import DaGetConfig
from model.modeling_da_get import DaGetEncoder, DaGetGraphAttentionBias, graph_distances


class DaGetEncoderTest(unittest.TestCase):
    """Exercise structural and dynamic encoder behavior with small robot graphs."""

    def setUp(self) -> None:
        """Initialize deterministic test inputs.

        Returns:
            None: Resets the PyTorch random seed for this test.
        """
        torch.manual_seed(11)
        self.config = DaGetConfig(
            input_size=3,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            context_size=6,
            cross_attention_layers=(1,),
            max_graph_distance=5,
        )
        self.parents = torch.tensor([[-1, 0, 1, -1]])
        self.joint_mask = torch.tensor([[True, True, True, False]])
        self.mass = torch.tensor(
            [
                [
                    [4.0, 1.0, 0.0, 0.0],
                    [1.0, 9.0, 2.0, 0.0],
                    [0.0, 2.0, 16.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )

    def test_graph_distances_and_root_edges(self) -> None:
        """Check directed ancestry, undirected paths, and optional root edges.

        Returns:
            None: Asserts distance values and cycle rejection.
        """
        spatial, parent, child = graph_distances(self.parents, self.joint_mask)
        self.assertEqual(spatial[0, 0, 2].item(), 2)
        self.assertEqual(parent[0, 2, 0].item(), 2)
        self.assertEqual(parent[0, 0, 2].item(), 0)
        self.assertEqual(child[0, 0, 2].item(), 2)
        self.assertEqual(spatial[0, 0, 3].item(), 5)
        with self.assertRaises(ValueError):
            graph_distances(torch.tensor([[1, 0]]), torch.ones(1, 2, dtype=torch.bool))
        roots = torch.tensor([[-1, -1]])
        edges = torch.tensor([[[False, True], [True, False]]])
        distance, _, _ = graph_distances(
            roots, torch.ones(1, 2, dtype=torch.bool), edges
        )
        self.assertEqual(distance[0, 0, 1].item(), 1)

    def test_mass_normalization_matches_formula(self) -> None:
        """Check that a linear mass projection sees normalized inertia entries.

        Returns:
            None: Asserts the exact dynamic bias for a known matrix.
        """
        config = DaGetConfig(
            input_size=1,
            hidden_size=2,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=4,
            use_spatial_encoding=False,
            use_parent_child_encoding=False,
            mass_epsilon=1e-8,
        )
        bias_module = DaGetGraphAttentionBias(config)
        with torch.no_grad():
            bias_module.mass_projection.weight.fill_(2.0)
        mask = torch.ones(1, 2, dtype=torch.bool)
        distances = graph_distances(torch.tensor([[-1, 0]]), mask)
        mass = torch.tensor([[[4.0, -3.0], [-3.0, 9.0]]])
        bias = bias_module(*distances, mass, mask, torch.float32)
        torch.testing.assert_close(bias[0, 0, 1], torch.tensor(-1.0))
        torch.testing.assert_close(bias[0, 0, 0], torch.tensor(2.0))
        with self.assertRaises(ValueError):
            bias_module(*distances, torch.zeros_like(mass), mask, torch.float32)

    def test_cross_attention_padding_and_gradients(self) -> None:
        """Check context sensitivity, masked joints, and trainable mass bias.

        Returns:
            None: Asserts output shape, padding, and gradient flow.
        """
        encoder = DaGetEncoder(self.config).eval()
        values = torch.randn(1, 4, 3, requires_grad=True)
        context = torch.randn(1, 3, 6)
        context_mask = torch.tensor([[True, True, False]])
        output = encoder(
            values,
            self.parents,
            self.mass,
            self.joint_mask,
            context_states=context,
            context_mask=context_mask,
        )
        self.assertEqual(output.shape, (1, 4, 8))
        torch.testing.assert_close(output[0, 3], torch.zeros(8))
        changed = context.clone()
        changed[:, 2] += 20
        torch.testing.assert_close(
            output,
            encoder(
                values,
                self.parents,
                self.mass,
                self.joint_mask,
                context_states=changed,
                context_mask=context_mask,
            ),
        )
        changed[:, 0] += torch.randn_like(changed[:, 0])
        self.assertFalse(
            torch.allclose(
                output,
                encoder(
                    values,
                    self.parents,
                    self.mass,
                    self.joint_mask,
                    context_states=changed,
                    context_mask=context_mask,
                ),
            )
        )
        output.square().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(encoder.layers[0].graph_bias.mass_projection.weight.grad)
        self.assertIsNotNone(encoder.layers[0].graph_bias.spatial_embedding.weight.grad)
        self.assertIsNotNone(
            encoder.layers[1].cross_attention.attention.q_proj_weight.grad
        )

    def test_joint_permutation_equivariance(self) -> None:
        """Check outputs follow a joint permutation with remapped graph data.

        Returns:
            None: Asserts the graph encoding does not depend on token order.
        """
        config = DaGetConfig(
            input_size=3,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        encoder = DaGetEncoder(config).eval()
        values = torch.randn(1, 3, 3)
        parents = torch.tensor([[-1, 0, 1]])
        mass = torch.eye(3).unsqueeze(0) + 0.2
        original = encoder(values, parents, mass)
        order = torch.tensor([2, 0, 1])
        inverse = torch.argsort(order)
        remapped_parents = torch.where(
            parents[:, order] < 0, -1, inverse[parents[:, order]]
        )
        permuted = encoder(
            values[:, order], remapped_parents, mass[:, order][:, :, order]
        )
        torch.testing.assert_close(permuted, original[:, order])

    def test_mass_matrix_changes_joint_representations(self) -> None:
        """Check that a changed inertia matrix changes encoder predictions.

        Returns:
            None: Asserts dynamic bias affects valid joint representations.
        """
        config = DaGetConfig(
            input_size=3,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            use_spatial_encoding=False,
            use_parent_child_encoding=False,
        )
        encoder = DaGetEncoder(config).eval()
        values = torch.randn(1, 3, 3)
        parents = torch.tensor([[-1, 0, 1]])
        mass = torch.eye(3).unsqueeze(0)
        changed_mass = mass.clone()
        changed_mass[0, 0, 1] = changed_mass[0, 1, 0] = 0.8
        original = encoder(values, parents, mass)
        changed = encoder(values, parents, changed_mass)
        self.assertFalse(torch.allclose(original, changed))

    def test_padded_mass_has_finite_gradients(self) -> None:
        """Check zero padded inertia diagonals do not create NaN gradients.

        Returns:
            None: Asserts finite input mass gradients after backpropagation.
        """
        encoder = DaGetEncoder(self.config)
        values = torch.randn(1, 4, 3)
        mass = self.mass.clone().requires_grad_()
        context = torch.randn(1, 2, 6)
        output = encoder(
            values, self.parents, mass, self.joint_mask, context_states=context
        )
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(mass.grad).all())


if __name__ == "__main__":
    unittest.main()
