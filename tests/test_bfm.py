"""Test the GET-conditioned BFM-Zero networks."""

import copy
import unittest

import torch

from model import BfmConfig, DaGetBfmModel, DaGetConfig, DaGetEncoder
from model.modeling_bfm import TruncatedNormal

EMBED, STATE, Z = 16, 10, 12


def config(**overrides) -> BfmConfig:
    """Small config for fast tests."""
    values = dict(embedding_size=EMBED, backward_input_size=STATE, z_dim=Z, hidden_size=16, num_attention_heads=2,
                  intermediate_size=32, num_latent_tokens=3, actor_layers=2, forward_layers=1, context_size=5,
                  backward_hidden_size=24, discriminator_hidden_size=24)
    values.update(overrides)
    return BfmConfig(**values)


class BfmNetworksTest(unittest.TestCase):
    """Shapes, BFM-Zero conventions, padding and permutation behavior."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = DaGetBfmModel(config()).eval()
        self.model.init_weights()
        self.embeddings = torch.randn(3, 6, EMBED)
        self.mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2, [True] * 2 + [False] * 4])
        self.z = self.model.sample_z(3)

    def test_shapes_and_ranges(self) -> None:
        """Actor gives per-DoF actions in [-1, 1]; F, critics, B, D follow BFM-Zero shapes."""
        distribution = self.model.actor(self.embeddings, self.z, 0.2, self.mask)
        action = distribution.sample(clip=0.3)
        self.assertEqual(action.shape, (3, 6))
        self.assertTrue(bool((action.abs() < 1).all()))
        self.assertTrue(bool((action[~self.mask] == 0).all()))
        self.assertTrue(bool((distribution.mean[~self.mask] == 0).all()))
        forward = self.model.forward_map(self.embeddings, self.z, action, self.mask)
        self.assertEqual(forward.shape, (2, 3, Z))
        self.assertEqual(self.model.fb_q_value(forward, self.z).shape, (2, 3))
        self.assertEqual(self.model.critic(self.embeddings, self.z, action, self.mask).shape, (2, 3, 1))
        self.assertEqual(self.model.aux_critic(self.embeddings, self.z, action, self.mask).shape, (2, 3, 1))
        state = torch.randn(3, STATE)
        b = self.model.backward_map(state)
        torch.testing.assert_close(b.norm(dim=-1), torch.full((3,), Z**0.5))
        logits = self.model.discriminator.compute_logits(state, self.z)
        self.assertEqual(logits.shape, (3, 1))
        torch.testing.assert_close(self.model.discriminator.compute_reward(state, self.z), logits, atol=1e-4, rtol=1e-4)

    def test_dict_observations_select_keys(self) -> None:
        """B and D concatenate state and privileged state from a dict, as BFM-Zero's filters."""
        state, privileged = torch.randn(3, 4), torch.randn(3, 6)
        flat = self.model.backward_map(torch.cat([state, privileged], -1))
        torch.testing.assert_close(self.model.backward_map({"state": state, "privileged_state": privileged}), flat)

    def test_padding_does_not_change_outputs(self) -> None:
        """Extra padded joints change neither the valid actions nor F."""
        embeddings, mask = self.embeddings[:1, :4], self.mask[1:2, :4]
        padded = torch.cat([embeddings, torch.randn(1, 3, EMBED)], 1)
        padded_mask = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], 1)
        z = self.z[:1]
        short = self.model.actor(embeddings, z, 0.2, mask).mean
        long = self.model.actor(padded, z, 0.2, padded_mask).mean
        torch.testing.assert_close(long[:, :4], short, atol=1e-5, rtol=1e-5)
        action = torch.rand(1, 4) * 2 - 1
        padded_action = torch.cat([action, torch.randn(1, 3)], 1)
        torch.testing.assert_close(
            self.model.forward_map(padded, z, padded_action, padded_mask),
            self.model.forward_map(embeddings, z, action, mask), atol=1e-5, rtol=1e-5,
        )

    def test_joint_permutation(self) -> None:
        """Permuting joints permutes actions and leaves F unchanged."""
        order = torch.tensor([3, 0, 5, 1, 4, 2])
        embeddings, z = self.embeddings[:1], self.z[:1]
        action = self.model.act(embeddings, z)
        torch.testing.assert_close(self.model.act(embeddings[:, order], z), action[:, order], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            self.model.forward_map(embeddings[:, order], z, action[:, order]),
            self.model.forward_map(embeddings, z, action), atol=1e-5, rtol=1e-5,
        )

    def test_latent_and_context_condition_the_actor(self) -> None:
        """Different z, latent sequences and context tokens change the actions."""
        base = self.model.act(self.embeddings, self.z, self.mask)
        other = self.model.act(self.embeddings, self.model.sample_z(3), self.mask)
        self.assertGreater(float((base - other).abs().max()), 1e-3)
        sequence = torch.stack([self.z, self.model.sample_z(3)], 1)
        latent_mask = torch.tensor([[True, True], [True, False], [True, True]])
        actions = self.model.act(self.embeddings, sequence, self.mask, latent_mask=latent_mask)
        self.assertEqual(actions.shape, (3, 6))
        context = torch.randn(3, 2, 5)
        with_context = self.model.act(self.embeddings, self.z, self.mask, context_states=context)
        self.assertGreater(float((base - with_context).abs().max()), 1e-3)

    def test_multi_output_actions_and_single_member(self) -> None:
        """Several action entries per DoF and a single-member ensemble."""
        model = DaGetBfmModel(config(action_size_per_joint=2, num_parallel=1), use_aux_critic=False)
        self.assertIsNone(model.aux_critic)
        action = model.actor(self.embeddings, self.z, 0.1, self.mask).sample()
        self.assertEqual(action.shape, (3, 6, 2))
        self.assertEqual(model.forward_map(self.embeddings, self.z, action, self.mask).shape, (3, Z))

    def test_inference(self) -> None:
        """Goal, reward (chunked) and tracking inference match their definitions."""
        model = DaGetBfmModel(config(inference_batch_size=7)).eval()
        states, reward = torch.randn(20, STATE), torch.randn(20, 1)
        b = model.backward_map(states)
        torch.testing.assert_close(model.reward_inference(states, reward), model.project_z(reward.T @ b))
        self.assertEqual(model.goal_inference(states[:2]).shape, (2, Z))
        tracking = model.tracking_inference(states)
        torch.testing.assert_close(tracking[3], model.project_z(b[3:11].mean(0)))
        torch.testing.assert_close(tracking[-2], model.project_z(b[-2:].mean(0)))
        self.assertEqual(model.reward_wr_inference(states, reward).shape, (1, Z))

    def test_truncated_normal_gradient(self) -> None:
        """Clamping passes gradients straight through."""
        loc = torch.tensor([[0.99, -0.2]], requires_grad=True)
        TruncatedNormal(loc, torch.full_like(loc, 0.5)).sample().sum().backward()
        torch.testing.assert_close(loc.grad, torch.ones_like(loc))


class DaGetEncoderToActorTest(unittest.TestCase):
    """The actor and F train end to end through the GET encoder."""

    def test_td3_style_update_reaches_encoder(self) -> None:
        """The FB actor loss -Fᵀz backpropagates into the encoder."""
        torch.manual_seed(1)
        encoder = DaGetEncoder(DaGetConfig(input_size=3, hidden_size=EMBED, num_hidden_layers=1, num_attention_heads=2,
                                       intermediate_size=32, use_mass_encoding=False))
        model = DaGetBfmModel(config())
        tokens = torch.randn(2, 4, 3)
        parents = torch.tensor([[-1, 0, 1, 2], [-1, 0, -1, -1]])
        mask = torch.tensor([[True] * 4, [True, True, False, False]])
        embeddings = encoder(tokens, parents, joint_mask=mask)
        z = model.sample_z(2)
        action = model.actor(embeddings, z, 0.2, mask).sample(clip=0.3)
        target = copy.deepcopy(model.forward_map).requires_grad_(False)
        loss = -model.fb_q_value(target(embeddings, z, action, mask), z).mean()
        loss.backward()
        self.assertGreater(float(encoder.input_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.actor.action_head[-1].weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
