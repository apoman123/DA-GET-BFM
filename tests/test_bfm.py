"""Test the per-DoF BFM-Zero networks."""

import unittest

import torch

from model import BfmConfig, DaGetBfmModel, DaGetConfig, DaGetEncoder
from model.modeling_bfm import TruncatedNormal

EMBED, Z, STATE = 16, 12, 9


def config(**overrides) -> BfmConfig:
    """Return a small config for fast tests."""
    values = dict(embedding_size=EMBED, state_token_size=STATE, z_dim=Z, hidden_size=16, num_attention_heads=2, intermediate_size=32,
                  num_latent_tokens=3, actor_layers=2, forward_layers=1, state_hidden_size=16, state_num_layers=1, state_num_heads=2,
                  state_intermediate_size=32, backward_hidden_size=24, discriminator_hidden_size=24)
    values.update(overrides)
    return BfmConfig(**values)


def shared_encoder(input_size: int = STATE) -> DaGetEncoder:
    """Build the small encoder shared by every BFM head in tests.

    Args:
        input_size (int): Positive width of raw policy tokens.

    Returns:
        DaGetEncoder: Encoder producing ``EMBED``-wide joint representations.
    """
    return DaGetEncoder(DaGetConfig(
        input_size=input_size, hidden_size=EMBED, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, use_mass_encoding=False,
    ))


def state_inputs(count: int, joints: int, order: torch.Tensor | None = None, values: dict | None = None) -> dict:
    """Return random state-encoder inputs of a robot without graph edges, optionally with joints reordered."""
    values = values or {
        "input_values": torch.randn(count, joints, STATE),
        "rigid_body_features": torch.randn(count, joints, 25),
        "actuator_features": torch.randn(count, joints, 6),
        "dynamics_features": torch.randn(count, joints, 7),
    }
    inputs = {key: value if order is None else value[:, order] for key, value in values.items()}
    return {
        **inputs,
        "parent_indices": torch.full((count, joints), -1),
        "joint_mask": torch.ones(count, joints, dtype=torch.bool),
    }


class BfmNetworksTest(unittest.TestCase):
    """Shapes, BFM-Zero conventions, padding and permutation behavior."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = DaGetBfmModel(config(), shared_encoder()).eval()
        self.model.init_weights()
        self.embeddings = torch.randn(3, 6, EMBED)
        self.mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2, [True] * 2 + [False] * 4])
        self.z = self.model.sample_z(3)

    def test_one_encoder_feeds_every_head_and_keeps_pretrained_weights(self) -> None:
        """Check one encoder instance feeds B, D, actor, and F without reinitialization.

        Returns:
            None: Asserts ownership, outputs, and backward gradient flow.
        """
        encoder = shared_encoder()
        before = encoder.input_projection.weight.detach().clone()
        model = DaGetBfmModel(config(), encoder)
        model.init_weights()
        self.assertIs(model.encoder, encoder)
        self.assertEqual(sum(isinstance(module, DaGetEncoder) for module in model.modules()), 1)
        torch.testing.assert_close(encoder.input_projection.weight, before)
        state = state_inputs(2, 4)
        hidden = model.encode_state(state)
        z = model.sample_z(2)
        backward = model.backward_map(hidden, state["joint_mask"])
        self.assertEqual(model.discriminator(hidden, z, state["joint_mask"]).shape, (2, 1))
        action = model.actor(hidden, z, 0.1, state["joint_mask"]).mean
        self.assertEqual(model.forward_map(hidden, z, action, state["joint_mask"]).shape, (2, 2, Z))
        backward.square().mean().backward()
        self.assertGreater(float(encoder.input_projection.weight.grad.abs().sum()), 0)

    def test_independent_named_sizes(self) -> None:
        """Check role-specific preset dimensions and actual module construction.

        Returns:
            None: Asserts independent widths, depths, and valid outputs.
        """
        config = BfmConfig.from_sizes(EMBED, STATE, actor_size="tiny", forward_size="small", backward_size="tiny", z_dim=Z)
        self.assertEqual(config.network_dimensions("actor"), (128, 4, 512, 4))
        self.assertEqual(config.network_dimensions("forward"), (256, 4, 1024, 8))
        self.assertEqual(config.state_hidden_size, 64)
        self.assertEqual(config.backward_hidden_size, 128)
        model = DaGetBfmModel(config, shared_encoder(), use_aux_critic=False).eval()
        self.assertEqual(model.actor.embedding_projection[-1].out_features, 128)
        self.assertEqual(model.forward_map.members[0].embedding_projection[-1].out_features, 256)
        self.assertEqual(model.backward_map.head[0].out_features, 128)
        with torch.no_grad():
            action = model.act(self.embeddings[:1], self.z[:1])
            self.assertEqual(model.forward_map(self.embeddings[:1], self.z[:1], action).shape, (2, 1, Z))

    def test_preset_overrides_and_validation(self) -> None:
        """Check all size levels and explicit legacy and role-specific overrides.

        Returns:
            None: Asserts dimensions and rejection of invalid presets.
        """
        sizes = ("tiny", "small", "medium", "large")
        widths = [BfmConfig.from_sizes(EMBED, STATE, actor_size=size).network_dimensions("actor")[0] for size in sizes]
        self.assertEqual(widths, [128, 256, 384, 512])
        config = BfmConfig.from_sizes(EMBED, STATE, hidden_size=64, actor_hidden_size=96, actor_num_attention_heads=3)
        self.assertEqual(config.network_dimensions("actor")[0], 96)
        self.assertEqual(config.network_dimensions("forward")[0], 64)
        with self.assertRaisesRegex(ValueError, "unknown backward size"):
            BfmConfig.from_sizes(EMBED, STATE, backward_size="huge")
        with self.assertRaisesRegex(ValueError, "actor dimensions"):
            BfmConfig.from_sizes(EMBED, STATE, actor_hidden_size=95)

    def test_shapes_and_ranges(self) -> None:
        """The actor gives per-DoF actions in [-1, 1]; F and critics follow BFM-Zero's shapes."""
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

    def test_actor_and_critics_have_distinct_parameters_and_outputs(self) -> None:
        """Actor parameters stay separate from the forward and scalar critic maps.

        Returns:
            None: Asserts module identity, output widths, and action conditioning.
        """
        networks = (self.model.actor, self.model.forward_map, self.model.critic, self.model.aux_critic)
        parameter_ids = [{id(parameter) for parameter in network.parameters()} for network in networks]
        for index, ids in enumerate(parameter_ids):
            for other in parameter_ids[index + 1:]:
                self.assertFalse(ids & other)
        self.assertEqual(self.model.forward_map.members[0].output_head[-1].out_features, Z)
        self.assertEqual(self.model.critic.members[0].output_head[-1].out_features, 1)
        action = self.model.actor(self.embeddings, self.z, 0.2, self.mask).mean
        changed = action.clone()
        changed[self.mask] = 0.9
        with torch.no_grad():
            first = self.model.critic(self.embeddings, self.z, action, self.mask)
            second = self.model.critic(self.embeddings, self.z, changed, self.mask)
        self.assertFalse(torch.allclose(first, second))

    def test_backward_and_discriminator_fit_any_robot(self) -> None:
        """One B and one D read robots of different sizes; B stays on the sphere; rewards are logits."""
        for joints in (3, 12, 29):
            state = state_inputs(2, joints)
            hidden = self.model.encode_state(state)
            backward = self.model.backward_map(hidden, state["joint_mask"])
            self.assertEqual(backward.shape, (2, Z))
            torch.testing.assert_close(backward.norm(dim=-1), torch.full((2,), Z**0.5))
            logits = self.model.discriminator.compute_logits(hidden, self.z[:2], state["joint_mask"])
            self.assertEqual(logits.shape, (2, 1))
            torch.testing.assert_close(
                self.model.discriminator.compute_reward(hidden, self.z[:2], state["joint_mask"]), logits,
                atol=1e-4, rtol=1e-4,
            )
            self.assertEqual(self.model.goal_inference(state).shape, (2, Z))

    def test_backward_and_discriminator_ignore_joint_order(self) -> None:
        """Without graph edges, reordering joints changes neither B nor D."""
        base = state_inputs(2, 5)
        values = {key: base[key] for key in ("input_values", "rigid_body_features", "actuator_features", "dynamics_features")}
        order = torch.tensor([3, 0, 4, 1, 2])
        reordered = state_inputs(2, 5, order, values)
        torch.testing.assert_close(
            self.model.backward_map(self.model.encode_state(reordered), reordered["joint_mask"]),
            self.model.backward_map(self.model.encode_state(base), base["joint_mask"]),
            atol=1e-5, rtol=1e-5,
        )
        z = self.z[:2]
        torch.testing.assert_close(
            self.model.discriminator.compute_logits(self.model.encode_state(reordered), z, reordered["joint_mask"]),
            self.model.discriminator.compute_logits(self.model.encode_state(base), z, base["joint_mask"]),
            atol=1e-5, rtol=1e-5,
        )

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
        """Different z, latent sequences and per-joint context tokens change the outputs."""
        base = self.model.act(self.embeddings, self.z, self.mask)
        other = self.model.act(self.embeddings, self.model.sample_z(3), self.mask)
        self.assertGreater(float((base - other).abs().max().detach()), 1e-3)
        sequence = torch.stack([self.z, self.model.sample_z(3)], 1)
        latent_mask = torch.tensor([[True, True], [True, False], [True, True]])
        self.assertEqual(self.model.act(self.embeddings, sequence, self.mask, latent_mask=latent_mask).shape, (3, 6))
        context = torch.randn(3, 6, STATE)
        action = torch.zeros(3, 6)
        plain = self.model.critic(self.embeddings, self.z, action, self.mask)
        privileged = self.model.critic(self.embeddings, self.z, action, self.mask, context_states=context, context_mask=self.mask)
        self.assertGreater(float((plain - privileged).abs().max().detach()), 1e-3)
        changed_padding = context.clone()
        changed_padding[~self.mask] = 5.0
        again = self.model.critic(self.embeddings, self.z, action, self.mask, context_states=changed_padding, context_mask=self.mask)
        torch.testing.assert_close(again, privileged, atol=1e-5, rtol=1e-5)

    def test_multi_output_actions_and_single_member(self) -> None:
        """Several action entries per DoF and a single-member ensemble."""
        model = DaGetBfmModel(config(action_size_per_joint=2, num_parallel=1), shared_encoder(), use_aux_critic=False)
        self.assertIsNone(model.aux_critic)
        action = model.actor(self.embeddings, self.z, 0.1, self.mask).sample()
        self.assertEqual(action.shape, (3, 6, 2))
        self.assertEqual(model.forward_map(self.embeddings, self.z, action, self.mask).shape, (3, Z))

    def test_inference(self) -> None:
        """Goal, reward (chunked) and tracking inference match their definitions."""
        model = DaGetBfmModel(config(inference_batch_size=7), shared_encoder()).eval()
        states, reward = state_inputs(20, 4), torch.randn(20, 1)
        backward = model.backward_map(model.encode_state(states), states["joint_mask"])
        torch.testing.assert_close(model.reward_inference(states, reward), model.project_z(reward.T @ backward), atol=1e-5, rtol=1e-5)
        tracking = model.tracking_inference(states)
        torch.testing.assert_close(tracking[3], model.project_z(backward[3:11].mean(0)), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(tracking[-2], model.project_z(backward[-2:].mean(0)), atol=1e-5, rtol=1e-5)
        self.assertEqual(model.reward_wr_inference(states, reward).shape, (1, Z))

    def test_truncated_normal_gradient(self) -> None:
        """Clamping passes gradients straight through."""
        loc = torch.tensor([[0.99, -0.2]], requires_grad=True)
        TruncatedNormal(loc, torch.full_like(loc, 0.5)).sample().sum().backward()
        torch.testing.assert_close(loc.grad, torch.ones_like(loc))


class EncoderToActorTest(unittest.TestCase):
    """The actor and F train end to end through the encoder."""

    def test_td3_style_update_reaches_encoder(self) -> None:
        """The FB actor loss -Fᵀz can backpropagate into the encoder."""
        torch.manual_seed(1)
        encoder = DaGetEncoder(DaGetConfig(input_size=3, hidden_size=EMBED, num_hidden_layers=1, num_attention_heads=2,
                                           intermediate_size=32, use_mass_encoding=False))
        model = DaGetBfmModel(config(), encoder)
        tokens = torch.randn(2, 4, 3)
        parents = torch.tensor([[-1, 0, 1, 2], [-1, 0, -1, -1]])
        mask = torch.tensor([[True] * 4, [True, True, False, False]])
        embeddings = encoder(tokens, parents, joint_mask=mask)
        z = model.sample_z(2)
        action = model.actor(embeddings, z, 0.2, mask).sample(clip=0.3)
        loss = -model.fb_q_value(model.forward_map(embeddings, z, action, mask), z).mean()
        loss.backward()
        self.assertGreater(float(encoder.input_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.actor.action_head[-1].weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
