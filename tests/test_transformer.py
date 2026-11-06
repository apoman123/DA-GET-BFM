"""Test numerical tokenization, position embeddings, and attention blocks."""

import unittest

import torch

from model.transformer import (
    AttentionBlock,
    CrossAttentionBlock,
    NumericTokenizer,
    SinusoidalPositionEmbedding,
)


class TransformerComponentsTest(unittest.TestCase):
    """Check shapes, masking, conditioning, and gradients."""

    def setUp(self) -> None:
        """Fix the random seed so comparisons are reproducible.

        Returns:
            None: Sets PyTorch's random state for this test.
        """
        torch.manual_seed(7)

    def test_numeric_tokenizer_maps_scalar_and_vector_inputs(self) -> None:
        """Check scalar convenience input and multivariate token shapes.

        Returns:
            None: Asserts equivalent scalar forms and the vector output shape.
        """
        scalar_tokenizer = NumericTokenizer(input_dim=1, embedding_dim=8)
        scalars = torch.randn(2, 4)
        torch.testing.assert_close(
            scalar_tokenizer(scalars), scalar_tokenizer(scalars.unsqueeze(-1))
        )
        vector_tokenizer = NumericTokenizer(input_dim=3, embedding_dim=8)
        self.assertEqual(vector_tokenizer(torch.randn(2, 4, 3)).shape, (2, 4, 8))

    def test_sinusoidal_embedding_is_fixed_and_supports_odd_width(self) -> None:
        """Check the zero-position pattern and deterministic odd-width output.

        Returns:
            None: Asserts sine and cosine values, shape, and repeatability.
        """
        embedding = SinusoidalPositionEmbedding(embedding_dim=5)
        positions = torch.tensor([[0, 1, 3], [0, 2, 4]])
        result = embedding(positions)
        self.assertEqual(result.shape, (2, 3, 5))
        torch.testing.assert_close(
            result[0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        )
        torch.testing.assert_close(result, embedding(positions))

    def test_causal_self_attention_ignores_future_tokens(self) -> None:
        """Check that later tokens cannot affect earlier causal outputs.

        Returns:
            None: Asserts invariance of earlier token outputs.
        """
        block = AttentionBlock(hidden_size=8, num_heads=2).eval()
        inputs = torch.randn(2, 4, 8)
        changed = inputs.clone()
        changed[:, 2:] += torch.randn_like(changed[:, 2:]) * 5
        original = block(inputs, causal=True)
        altered = block(changed, causal=True)
        torch.testing.assert_close(original[:, :2], altered[:, :2])

    def test_cross_attention_uses_context_and_respects_padding(self) -> None:
        """Check that visible context matters while masked context is ignored.

        Returns:
            None: Asserts context sensitivity and padding-mask invariance.
        """
        block = CrossAttentionBlock(hidden_size=8, num_heads=2, context_size=6).eval()
        queries = torch.randn(2, 3, 8)
        context = torch.randn(2, 4, 6)
        padding = torch.tensor([[False, False, False, True]] * 2)
        baseline = block(queries, context, context_padding_mask=padding)
        changed_masked = context.clone()
        changed_masked[:, -1] += 10
        torch.testing.assert_close(
            baseline,
            block(queries, changed_masked, context_padding_mask=padding),
        )
        changed_visible = context.clone()
        changed_visible[:, 0] += torch.randn_like(changed_visible[:, 0]) * 5
        self.assertFalse(
            torch.allclose(
                baseline,
                block(queries, changed_visible, context_padding_mask=padding),
            )
        )

    def test_tokenizer_to_attention_has_gradients(self) -> None:
        """Check that tokenization and both attention blocks train together.

        Returns:
            None: Asserts gradients reach input data and all three modules.
        """
        tokenizer = NumericTokenizer(input_dim=3, embedding_dim=8)
        positions = SinusoidalPositionEmbedding(embedding_dim=8)
        self_attention = AttentionBlock(hidden_size=8, num_heads=2)
        cross_attention = CrossAttentionBlock(
            hidden_size=8, num_heads=2, context_size=6
        )
        values = torch.randn(2, 4, 3, requires_grad=True)
        context = torch.randn(2, 5, 6)
        tokens = tokenizer(values)
        tokens = tokens + positions(torch.arange(4).expand(2, -1))
        output = cross_attention(self_attention(tokens), context)
        self.assertEqual(output.shape, (2, 4, 8))
        output.square().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(tokenizer.fc1.weight.grad)
        self.assertIsNotNone(self_attention.attention.in_proj_weight.grad)
        self.assertIsNotNone(cross_attention.attention.q_proj_weight.grad)
        self.assertIsNotNone(cross_attention.attention.k_proj_weight.grad)


if __name__ == "__main__":
    unittest.main()
