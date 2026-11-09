"""Test the cross-embodiment FB-CPR agent, its MuJoCo environments and its training script."""

import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from model import BfmConfig, DaGetConfig, EmbodimentGraph, JointFeatureConfig, actuator_specs
from model.modeling_bfm import project_to_sphere
from model.pretraining import DaGetForPhysicsPretraining, DaGetPhysicsPretrainingConfig
from model.rl import DaGetFbCprAgent, EmbodimentObserver, ExpertMotions, FbCprConfig, ReplayBuffer, RobotState, StateView
from model.rl.agent import EmaRewardNormalizer, ZBuffer, ensemble_pessimism, fb_bellman_loss, orthonormality_loss, soft_update, split_counts
from model.rl.buffers import _pack_mass, _unpack_mass, resample_state, rotation_from_gravity
from model.rl.evaluation import earth_movers_distance, motion_priorities
from model.training.dataset import build_physics_datasets
from model.training.episodes import EmbodimentRecord, Trajectory, process_episode
from model.training.ground_sim import ground_specs
from tests.test_embodiment import humanoid, quadruped

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
STAND = {f"{leg}_{part}_joint": value for leg in ("FL", "FR", "RL", "RR") for part, value in (("thigh", 0.3), ("calf", -0.6))}
HISTORY = 2
Z = 8


def fixed_record(urdf: str, name: str) -> EmbodimentRecord:
    """Return a fixed-base robot in position mode."""
    graph = EmbodimentGraph.from_urdf(urdf, JointFeatureConfig.da_get_zero())
    return EmbodimentRecord.build(name, graph, ground_specs(actuator_specs(graph)))


def random_state(record: EmbodimentRecord, count: int) -> RobotState:
    """Return a random fixed-base state."""
    joints = record.graph.num_joints
    direction = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64).expand(count, 3)
    zeros = torch.zeros((count, 3), dtype=torch.float64)
    return RobotState(0.3 * torch.randn(count, joints, dtype=torch.float64), torch.randn(count, joints, dtype=torch.float64),
                      direction, zeros, zeros, torch.zeros(count, dtype=torch.float64))


def observe(observer: EmbodimentObserver, state: RobotState):
    """Return the observation of a state at the start of an episode."""
    local, global_features = observer.policy_step(state, None, None, None)
    local = local[:, None].expand(-1, HISTORY, -1, -1)
    global_features = global_features[:, None].expand(-1, HISTORY, -1)
    return observer.observation(local, global_features, state, state, None)


def small_encoder(token_size: int) -> DaGetForPhysicsPretraining:
    """Return a tiny encoder with physics cross attention."""
    encoder = DaGetConfig(input_size=token_size, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=32, use_physics_cross_attention=True)
    return DaGetForPhysicsPretraining(DaGetPhysicsPretrainingConfig.from_encoder_config(encoder))


def small_model_config(state_token_size: int) -> BfmConfig:
    """Return tiny networks."""
    return BfmConfig(embedding_size=16, state_token_size=state_token_size, z_dim=Z, hidden_size=16, num_attention_heads=2,
                     intermediate_size=32, num_latent_tokens=2, actor_layers=1, forward_layers=1, state_hidden_size=16,
                     state_num_layers=1, state_num_heads=2, state_intermediate_size=32, backward_hidden_size=32,
                     discriminator_hidden_size=32, seq_length=4)


def build_agent(**overrides) -> tuple[DaGetFbCprAgent, list[ReplayBuffer], list[ExpertMotions]]:
    """Return an agent on two fixed-base robots of different size, with filled buffers."""
    records = [fixed_record(quadruped(), "quadruped"), fixed_record(humanoid(), "humanoid")]
    observers = [EmbodimentObserver(record, HISTORY) for record in records]
    values = {"batch_size": 16, "z_buffer_size": 32, **overrides}
    agent = DaGetFbCprAgent(small_model_config(observers[0].state_token_size), FbCprConfig(**values), small_encoder(observers[0].token_size), observers)
    replays, motions = [], []
    for record, observer in zip(records, observers):
        joints = record.graph.num_joints
        replay = ReplayBuffer(64, agent.reward_names)
        for _ in range(2):
            count = 16
            first, second = random_state(record, count), random_state(record, count)
            rewards = {name: torch.rand(count, dtype=torch.float64) for name in agent.reward_names}
            replay.add(observe(observer, first), torch.rand(count, joints) * 2 - 1, observe(observer, second),
                       project_to_sphere(torch.randn(count, Z)), torch.rand(count) < 0.2, rewards)
        replays.append(replay)
        trajectory = Trajectory(embodiment=0, dt=0.02, positions=0.2 * torch.randn(30, joints, dtype=torch.float64),
                                velocities=torch.randn(30, joints, dtype=torch.float64))
        motions.append(ExpertMotions(observer, [trajectory]))
    return agent, replays, motions


def physics_batch(root: Path):
    """Return a collated batch of the physics auxiliary losses for the quadruped."""
    (root / "quad.urdf").write_text(quadruped())
    config = {"history_size": HISTORY, "joint_features": "da_get_zero", "eval_fraction": 0.0,
              "sources": [{"type": "synthetic", "name": "quad", "urdf": str(root / "quad.urdf"), "randomized_embodiments": 1,
                           "episodes_per_embodiment": 1, "steps": 10, "probe_torques": 1}]}
    datasets = build_physics_datasets(config, cache_dir=root / "cache")
    return datasets.collator()([datasets.train[index] for index in range(8)])


def snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    """Copy a module's parameters."""
    return [parameter.detach().clone() for parameter in module.parameters()]


def changed(module: torch.nn.Module, before: list[torch.Tensor]) -> bool:
    """Return whether any parameter differs from a snapshot."""
    return any(not torch.equal(parameter, old) for parameter, old in zip(module.parameters(), before))


def as_float32(value: float) -> float:
    """Round a float the way a ``float32`` tensor stores it."""
    return torch.tensor(value, dtype=torch.float32).item()


class LossTest(unittest.TestCase):
    """The FB-CPR formulas against explicit computations."""

    def test_ensemble_pessimism(self) -> None:
        """Two members: mean - penalty * |a - b|; several: the mean pairwise distance."""
        values = torch.tensor([[1.0, 4.0], [3.0, 0.0]])
        mean, spread, pessimistic = ensemble_pessimism(values, 0.5)
        torch.testing.assert_close(mean, torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(spread, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(pessimistic, torch.tensor([1.0, 0.0]))
        three = torch.tensor([[0.0], [1.0], [3.0]])
        torch.testing.assert_close(ensemble_pessimism(three, 0.0)[1], torch.tensor([(1.0 + 3.0 + 2.0) / 3]))

    def test_fb_loss_matches_the_definition(self) -> None:
        """The matrix form equals the loop over members, states and goals, with and without robot pairs."""
        torch.manual_seed(0)
        members, batch = 2, 4
        measures, target = torch.randn(members, batch, batch), torch.randn(batch, batch)
        discount = torch.tensor([[0.98], [0.0], [0.98], [0.98]])
        robots = torch.tensor([0, 0, 1, 1])
        for pairs in (None, robots[:, None] == robots[None, :]):
            off, diagonal, compared = 0.0, 0.0, 0
            for member in range(members):
                for i in range(batch):
                    for j in range(batch):
                        residual = measures[member, i, j] - discount[i, 0] * target[i, j]
                        if i == j:
                            diagonal -= residual / batch
                        elif pairs is None or pairs[i, j]:
                            off += 0.5 * residual**2
            compared = (batch**2 - batch) if pairs is None else int(pairs.sum()) - batch
            loss, loss_diagonal, loss_off = fb_bellman_loss(measures, discount, target, pairs)
            torch.testing.assert_close(loss_off, torch.as_tensor(off / compared))
            torch.testing.assert_close(loss_diagonal, torch.as_tensor(diagonal))
            torch.testing.assert_close(loss, loss_off + loss_diagonal)

    def test_other_robots_states_do_not_enter_the_losses(self) -> None:
        """Entries between two robots do not change the FB or orthonormality losses."""
        torch.manual_seed(0)
        robots = torch.tensor([0, 0, 1, 1])
        pairs = robots[:, None] == robots[None, :]
        measures, target = torch.randn(2, 4, 4), torch.randn(4, 4)
        discount = torch.full((4, 1), 0.98)
        changed_measures = measures.clone()
        changed_measures[:, 0, 2] += 100.0
        torch.testing.assert_close(fb_bellman_loss(changed_measures, discount, target, pairs)[0], fb_bellman_loss(measures, discount, target, pairs)[0])
        self.assertGreater(float(fb_bellman_loss(changed_measures, discount, target)[0]), float(fb_bellman_loss(measures, discount, target)[0]) + 1)
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(float(orthonormality_loss(embeddings, pairs)[2]), 0.0)
        self.assertGreater(float(orthonormality_loss(embeddings)[2]), 0.0)

    def test_orthonormality_is_smallest_for_orthonormal_embeddings(self) -> None:
        """Orthogonal rows of a given norm reach the analytic minimum."""
        embeddings = torch.eye(4) * 2
        loss, diagonal, off = orthonormality_loss(embeddings)
        torch.testing.assert_close(diagonal, torch.tensor(-4.0))
        torch.testing.assert_close(off, torch.tensor(0.0))
        self.assertGreater(float(orthonormality_loss(embeddings + 0.5)[0]), float(loss))

    def test_soft_update_and_helpers(self) -> None:
        """Target update, shares, z buffer wrap-around and reward normalizer."""
        source, target = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        old = snapshot(target)
        soft_update(source, target, 0.1)
        for parameter, previous, current in zip(source.parameters(), old, target.parameters()):
            torch.testing.assert_close(current, 0.9 * previous + 0.1 * parameter)
        self.assertEqual(split_counts(10, 3), [4, 3, 3])
        buffer = ZBuffer(3, 1, torch.device("cpu"))
        buffer.add(torch.arange(5.0)[:, None])
        self.assertEqual(buffer.size, 3)
        self.assertEqual(sorted(buffer.storage[:, 0].tolist()), [2.0, 3.0, 4.0])
        normalizer = EmaRewardNormalizer()
        values = torch.tensor([[1.0], [3.0]])
        torch.testing.assert_close(normalizer(values), values / torch.sqrt(torch.tensor(5.0 - 4.0)), rtol=1e-4, atol=1e-4)

    def test_earth_movers_distance_and_priorities(self) -> None:
        """EMD equals the best matching of frames; priorities follow BFM-Zero's exponential rule."""
        rng = np.random.default_rng(0)
        rollout, target = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        best = min(np.linalg.norm(rollout - target[list(order)], axis=-1).mean() for order in itertools.permutations(range(5)))
        self.assertAlmostEqual(earth_movers_distance(rollout, target), float(best))
        self.assertAlmostEqual(earth_movers_distance(rollout, rollout[::-1]), 0.0)
        np.testing.assert_allclose(motion_priorities(np.array([0.1, 1.0, 9.0])), [2.0**1.0, 2.0**2.0, 2.0**4.0])

    def test_rotation_from_gravity(self) -> None:
        """The rotation maps world down to the given base-frame direction."""
        direction = torch.nn.functional.normalize(torch.tensor([[0.3, -0.2, -1.0]], dtype=torch.float64), dim=-1)
        rotation = rotation_from_gravity(direction)
        torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), torch.eye(3, dtype=torch.float64)[None])
        down = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
        torch.testing.assert_close((rotation.transpose(-1, -2) @ down[:, None]).squeeze(-1), direction)


class ObservationTest(unittest.TestCase):
    """State views, buffers and motions."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.record = fixed_record(quadruped(), "quadruped")
        self.observer = EmbodimentObserver(self.record, HISTORY)

    def test_state_view_layout(self) -> None:
        """Local features hold position, velocity and frame position; global ones the base state."""
        state = random_state(self.record, 2)
        view = self.observer.state_view(state)
        joints = self.record.graph.num_joints
        self.assertEqual(view.local.shape, (2, joints, 5))
        self.assertEqual(view.global_features.shape, (2, 10))
        torch.testing.assert_close(view.local[..., 0], self.record.graph.normalize_joint_positions(state.q).float())
        torch.testing.assert_close(view.global_features[:, :3], state.gravity_direction.float())
        tokens = self.observer.state_inputs(view)["input_values"]
        self.assertEqual(tokens.shape, (2, joints, self.observer.state_token_size))

    def test_state_view_has_no_action_or_torque(self) -> None:
        """B and D cannot see the last torque, which motion data does not have; the policy can."""
        state = random_state(self.record, 2)
        local, global_features = self.observer.policy_step(state, None, None, None)
        local = local[:, None].expand(-1, HISTORY, -1, -1)
        global_features = global_features[:, None].expand(-1, HISTORY, -1)
        torque = torch.ones_like(state.q)
        first = self.observer.observation(local, global_features, state, state, None)
        second = self.observer.observation(local, global_features, state, state, torque)
        self.assertGreater(float((first.dynamics - second.dynamics).abs().max()), 1e-3)
        torch.testing.assert_close(first.state.dynamics, second.state.dynamics)
        torch.testing.assert_close(first.state.dynamics[..., -1], torch.zeros_like(first.state.dynamics[..., -1]))

    def test_sensor_noise_stays_out_of_the_state_view(self) -> None:
        """Noise reaches the policy's features only: B, D and critics read the true state."""
        state = random_state(self.record, 2)
        noisy = RobotState(state.q + 0.05, state.qd + 0.5, state.gravity_direction, state.angular_velocity, state.linear_velocity, state.height)
        local, global_features = self.observer.policy_step(noisy, None, None, None)
        observation = self.observer.observation(local[:, None].expand(-1, HISTORY, -1, -1), global_features[:, None].expand(-1, HISTORY, -1), state, noisy, None)
        clean = self.observer.state_view(state)
        torch.testing.assert_close(observation.state.local, clean.local)
        torch.testing.assert_close(observation.state.dynamics, clean.dynamics)
        self.assertGreater(float((observation.dynamics - clean.dynamics).abs().max()), 1e-3)
        self.assertGreater(float((observation.local[:, 0, :, 1] - clean.local[..., 1]).abs().max()), 1e-3)

    def test_unknown_signals_are_hidden_from_the_state_view(self) -> None:
        """When motions lack height or linear velocity, simulated states drop them too."""
        state = random_state(self.record, 2)
        state.linear_velocity = state.linear_velocity + 1.0
        state.height = state.height + 1.0
        observer = EmbodimentObserver(self.record, HISTORY, use_height=False, use_linear_velocity=False)
        self.assertEqual(float(observer.state_view(state).global_features[:, 6:10].abs().sum()), 0.0)

    def test_replay_buffer_wraps_and_round_trips(self) -> None:
        """The ring keeps the newest transitions, in half precision with packed mass matrices."""
        joints = self.record.graph.num_joints
        buffer = ReplayBuffer(3, ("a",))
        for value in range(2):
            count = 2
            observation = observe(self.observer, random_state(self.record, count))
            z = torch.full((count, 1), float(value))
            buffer.add(observation, torch.zeros(count, joints), observation, z, torch.zeros(count, dtype=torch.bool), {"a": torch.zeros(count)})
        self.assertEqual(len(buffer), 3)
        self.assertEqual(sorted(buffer.storage["z"][:, 0].tolist()), [0.0, 1.0, 1.0])
        self.assertEqual(buffer.storage["local"].dtype, torch.float16)
        sample = buffer.sample(5)
        self.assertEqual(sample["observation"].local.shape[0], 5)
        self.assertEqual(sample["observation"].local.dtype, torch.float32)
        self.assertGreater(buffer.transition_bytes(), 0)
        mass = observation.mass
        torch.testing.assert_close(_unpack_mass(_pack_mass(mass), joints), mass)
        single = ReplayBuffer(1, ("a",))
        single.add(observation.select(torch.tensor([0])), torch.zeros(1, joints), observation.select(torch.tensor([0])),
                   torch.zeros(1, 1), torch.zeros(1, dtype=torch.bool), {"a": torch.zeros(1)})
        stored = single.sample(1)["observation"]
        torch.testing.assert_close(stored.mass, observation.mass[:1], rtol=2e-3, atol=1e-4)
        torch.testing.assert_close(stored.local, observation.local[:1], rtol=2e-3, atol=1e-4)

    def test_motions_are_resampled_and_chunked(self) -> None:
        """A 30 Hz motion lands on 50 Hz control steps and is split into chunks."""
        steps, dt, control_dt = 91, 1 / 30, 0.02
        time = torch.arange(steps, dtype=torch.float64) * dt
        joints = self.record.graph.num_joints
        q = torch.sin(time)[:, None].expand(steps, joints)
        qd = torch.cos(time)[:, None].expand_as(q)
        trajectory = Trajectory(embodiment=0, dt=dt, positions=q, velocities=qd)
        whole = ExpertMotions(self.observer, [trajectory], control_dt, chunk_seconds=None)
        self.assertEqual(len(whole), 1)
        frames = whole.frames[0]
        self.assertEqual(len(frames["q"]), int(3.0 / control_dt) + 1)
        np.testing.assert_allclose(frames["q"][:, 0], np.sin(np.arange(len(frames["q"])) * control_dt), atol=2e-4)
        chunks = ExpertMotions(self.observer, [trajectory], control_dt, chunk_seconds=1.0)
        self.assertEqual([len(frame["q"]) for frame in chunks.frames], [50, 50, 50])
        zeros = torch.zeros((steps, 3), dtype=torch.float64)
        state = RobotState(q, qd, zeros, zeros, zeros, zeros[:, 0])
        rotation = torch.eye(3, dtype=torch.float64).expand(steps, 3, 3)
        self.assertIs(resample_state(state, rotation, dt, dt)[0], state)

    def test_priorities_shift_expert_sampling(self) -> None:
        """Windows come mostly from high-priority motions, and from consecutive frames."""
        joints = self.record.graph.num_joints
        first = Trajectory(embodiment=0, dt=0.02, positions=torch.zeros(40, joints, dtype=torch.float64), velocities=torch.zeros(40, joints, dtype=torch.float64))
        second = Trajectory(embodiment=0, dt=0.02, positions=torch.ones(40, joints, dtype=torch.float64) * 0.5, velocities=torch.zeros(40, joints, dtype=torch.float64))
        motions = ExpertMotions(self.observer, [first, second])
        motions.set_priorities(torch.tensor([1.0, 99.0]))
        current, following = motions.sample_windows(200, 4, torch.Generator().manual_seed(0))
        high = (current.local[:, 0, 0, 0] - motions.states[1].local[0, 0, 0]).abs() < 1e-4
        self.assertGreater(float(high.float().mean()), 0.9)
        self.assertEqual(current.local.shape, (200, 4, joints, 5))
        with self.assertRaises(ValueError):
            motions.set_priorities(torch.tensor([1.0, 0.0]))
        del following


class AgentTest(unittest.TestCase):
    """The update rules."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.agent, self.replays, self.motions = build_agent()

    def prepared(self):
        """Return a batch with latents."""
        batch = self.agent._training_batch(self.replays, [8, 8])
        return batch, project_to_sphere(torch.randn(16, Z))

    def networks(self) -> dict[str, torch.nn.Module]:
        """Return every trained network by name."""
        model = self.agent.model
        names = ("forward_map", "backward_map", "actor", "critic", "aux_critic", "discriminator")
        return {
            **{name: getattr(model, name) for name in names},
            "encoder": self.agent.pretrained.encoder,
            "physics_heads": torch.nn.ModuleList([module for name, module in self.agent.pretrained.named_children() if name != "encoder"]),
        }

    def test_batch_has_an_equal_share_of_each_robot(self) -> None:
        """Robots are padded together, with zero actions and context on missing joints."""
        batch, _ = self.prepared()
        self.assertEqual(int((batch.robots == 0).sum()), 8)
        joints = [observer.record.graph.num_joints for observer in self.agent.observers]
        embeddings = self.agent._padded_embeddings(self.agent.pretrained, batch.observations, max(joints))
        for robot, count in enumerate(joints):
            rows = batch.robots == robot
            self.assertTrue(bool((batch.joint_mask[rows].sum(1) == count).all()))
            self.assertEqual(float(embeddings[rows][:, count:].abs().sum()), 0.0)
            self.assertEqual(float(batch.action[rows][:, count:].abs().sum()), 0.0)
            self.assertEqual(float(batch.context[rows][:, count:].abs().sum()), 0.0)

    def test_terminal_transitions_have_zero_discount(self) -> None:
        """The discount is zero exactly where the episode terminated, and gamma elsewhere."""
        self.replays[0].storage["terminated"][:] = True
        self.replays[1].storage["terminated"][:] = False
        batch, _ = self.prepared()
        self.assertEqual(set(batch.discount[batch.robots == 0].flatten().tolist()), {0.0})
        self.assertEqual(set(batch.discount[batch.robots == 1].flatten().tolist()), {as_float32(0.98)})

    def test_each_update_changes_only_its_networks(self) -> None:
        """Value networks and encoder move together; the actor and D move alone; targets wait."""
        agent = self.agent
        batch, z = self.prepared()
        targets = [agent.target_forward_map, agent.target_backward_map, agent.target_critic, agent.target_aux_critic, agent.target_pretrained]
        networks = self.networks()
        with tempfile.TemporaryDirectory() as folder:
            physics = physics_batch(Path(folder))
        steps = {
            ("forward_map", "backward_map", "critic", "aux_critic", "encoder", "physics_heads"): lambda: agent.update_value_networks(batch, z, physics),
            ("actor",): lambda: agent.update_actor(batch, z),
            ("discriminator",): lambda: agent.update_discriminator(batch.states, z, batch.next_states, z, batch.counts),
        }
        target_before = [snapshot(target) for target in targets]
        for expected, step in steps.items():
            before = {name: snapshot(network) for name, network in networks.items()}
            metrics = step()
            self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))
            for name, network in networks.items():
                self.assertEqual(changed(network, before[name]), name in expected, f"{expected}: {name}")
        for target, before in zip(targets, target_before):
            self.assertFalse(changed(target, before))

    def test_actor_loss_never_reaches_the_encoder(self) -> None:
        """After an actor update, the encoder has no gradient."""
        batch, z = self.prepared()
        self.agent.pretrained.zero_grad(set_to_none=True)
        self.agent.update_actor(batch, z)
        self.assertTrue(all(parameter.grad is None for parameter in self.agent.pretrained.parameters()))

    def test_encoder_gradient_sources(self) -> None:
        """RL losses and physics losses each reach the encoder, and each can be turned off."""
        batch, z = self.prepared()
        with tempfile.TemporaryDirectory() as folder:
            physics = physics_batch(Path(folder))
        rl_only, _, _ = build_agent()
        before = snapshot(rl_only.pretrained.encoder)
        rl_only.update_value_networks(rl_only._training_batch(self.replays, [8, 8]), z)
        self.assertTrue(changed(rl_only.pretrained.encoder, before))
        physics_only, _, _ = build_agent(encoder_rl_gradients=False)
        before = snapshot(physics_only.pretrained.encoder)
        physics_only.update_value_networks(physics_only._training_batch(self.replays, [8, 8]), z)
        self.assertFalse(changed(physics_only.pretrained.encoder, before))
        metrics = physics_only.update_value_networks(physics_only._training_batch(self.replays, [8, 8]), z, physics)
        self.assertTrue(changed(physics_only.pretrained.encoder, before))
        self.assertIn("physics/loss_fk", metrics)
        frozen, _, _ = build_agent(train_encoder=False)
        before = snapshot(frozen.pretrained)
        frozen.update(self.replays, self.motions, physics)
        self.assertFalse(changed(frozen.pretrained, before))
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen.pretrained.parameters()))
        del batch

    def test_physics_loss_leaves_the_encoder_in_evaluation_mode(self) -> None:
        """Dropout stays off for RL, whichever batches ran before."""
        with tempfile.TemporaryDirectory() as folder:
            physics = physics_batch(Path(folder))
        self.agent.update(self.replays, self.motions, physics)
        self.assertFalse(self.agent.pretrained.training)

    def test_full_update_moves_targets_by_tau(self) -> None:
        """All networks learn, the targets follow by their tau, and the metrics are finite."""
        agent = self.agent
        with tempfile.TemporaryDirectory() as folder:
            physics = physics_batch(Path(folder))
        forward, target = snapshot(agent.model.critic), snapshot(agent.target_critic)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(forward, target)))
        encoder_target = snapshot(agent.target_pretrained)
        metrics = agent.update(self.replays, self.motions, physics)
        self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))
        self.assertIn("aux_rew/penalty_undesired_contact", metrics)
        tau = agent.config.critic_target_tau
        for new, old, online in zip(agent.target_critic.parameters(), target, agent.model.critic.parameters()):
            torch.testing.assert_close(new, (1 - tau) * old + tau * online, rtol=1e-5, atol=1e-6)
        tau = agent.config.fb_target_tau
        self.assertTrue(changed(agent.target_pretrained, encoder_target))
        for new, old, online in zip(agent.target_pretrained.parameters(), encoder_target, agent.pretrained.parameters()):
            torch.testing.assert_close(new, (1 - tau) * old + tau * online, rtol=1e-5, atol=1e-6)

    def test_f_and_b_are_compared_only_within_a_robot(self) -> None:
        """The agent gives the FB and orthonormality losses the same-robot mask, unless told not to."""
        for within, expected in ((True, True), (False, False)):
            agent, replays, _ = build_agent(within_robot_pairs=within)
            batch = agent._training_batch(replays, [8, 8])
            z = project_to_sphere(torch.randn(16, Z))
            with mock.patch("model.rl.agent.fb_bellman_loss", wraps=fb_bellman_loss) as fb, mock.patch("model.rl.agent.orthonormality_loss", wraps=orthonormality_loss) as orth:
                agent.update_value_networks(batch, z)
            for call in (fb.call_args, orth.call_args):
                pairs = call.args[-1]
                if expected:
                    torch.testing.assert_close(pairs, batch.robots[:, None] == batch.robots[None, :])
                    self.assertFalse(bool(pairs[0, 15]))
                else:
                    self.assertIsNone(pairs)

    def test_bellman_targets_use_the_target_encoder(self) -> None:
        """s' goes through the target encoder and s through the online one, which alone gets gradients."""
        agent = self.agent
        batch, z = self.prepared()
        calls = []
        original = agent._padded_embeddings

        def spy(model, observations, joints, grad=False):
            calls.append((model, observations, grad))
            return original(model, observations, joints, grad)

        with mock.patch.object(agent, "_padded_embeddings", side_effect=spy):
            agent.update_value_networks(batch, z)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], agent.pretrained)
        self.assertIs(calls[0][1], batch.observations)
        self.assertTrue(calls[0][2])
        self.assertIs(calls[1][0], agent.target_pretrained)
        self.assertIs(calls[1][1], batch.next_observations)
        self.assertFalse(calls[1][2])

    def test_goal_and_expert_prompts_stay_within_a_robot(self) -> None:
        """A transition never gets a goal or a motion of another robot."""
        agent = self.agent
        count = 16
        robots = torch.tensor([0] * 8 + [1] * 8)
        goals = project_to_sphere(torch.randn(count, Z))
        expert_z = project_to_sphere(torch.randn(count, Z))
        expert_robots = torch.tensor([1] * 8 + [0] * 8)
        agent.config.train_goal_ratio, agent.config.expert_asm_ratio = 1.0, 0.0
        z = agent.sample_mixed_z(robots, goals, expert_z, expert_robots)
        for row in range(count):
            self.assertEqual(int(robots[torch.cdist(z[row:row + 1], goals)[0].argmin()]), int(robots[row]))
        agent.config.train_goal_ratio, agent.config.expert_asm_ratio = 0.0, 1.0
        z = agent.sample_mixed_z(robots, goals, expert_z, expert_robots)
        for row in range(count):
            self.assertEqual(int(expert_robots[torch.cdist(z[row:row + 1], expert_z)[0].argmin()]), int(robots[row]))

    def test_expert_latents_average_each_window(self) -> None:
        """An expert latent is the projected mean of B over its window, per robot."""
        agent = self.agent
        _, next_states = agent._expert_batch(self.motions, [2, 2])
        z = agent.encode_expert(next_states)
        self.assertEqual(z.shape, (16, Z))
        embeddings = torch.cat([agent.model.backward_map(inputs) for inputs in next_states]).detach()
        torch.testing.assert_close(z[0], project_to_sphere(embeddings[:4].mean(0)))
        torch.testing.assert_close(z[3], z[0])
        torch.testing.assert_close(z[8], project_to_sphere(embeddings[8:12].mean(0)))

    def test_prompts_for_one_robot(self) -> None:
        """Tracking and goal latents use the robot's own B."""
        agent = self.agent
        view = self.motions[1].states[0]
        tracking = agent.tracking_latents(1, view)
        self.assertEqual(tracking.shape, (view.local.shape[0] - 1, Z))
        torch.testing.assert_close(tracking.norm(dim=-1), torch.full((tracking.shape[0],), Z**0.5))
        goal = agent.goal_latent(1, view.select(torch.tensor([5, 6])))
        self.assertEqual(goal.shape, (2, Z))

    def test_rollout_context(self) -> None:
        """Latents change on schedule, and expert environments follow their tracking latents."""
        agent = self.agent
        agent.config.rollout_expert_trajectories_percentage = 0.5
        agent.config.update_z_every_step = 3
        count = 4
        z = agent.rollout_context(0, None, torch.zeros(count, dtype=torch.long), self.motions[0])
        self.assertEqual(z.shape, (count, Z))
        torch.testing.assert_close(z.norm(dim=-1), torch.full((count,), Z**0.5))
        environments, tracking = agent._tracking[0]
        torch.testing.assert_close(z[environments], tracking[:, 0])
        others = torch.tensor([index for index in range(count) if index not in environments.tolist()])
        kept = agent.rollout_context(0, z.clone(), torch.ones(count, dtype=torch.long), self.motions[0])
        torch.testing.assert_close(kept[others], z[others])
        torch.testing.assert_close(kept[environments], tracking[:, 1])
        changed_z = agent.rollout_context(0, z.clone(), torch.full((count,), 3, dtype=torch.long), self.motions[0])
        self.assertGreater(float((changed_z[others] - z[others]).abs().max()), 1e-3)

    def test_checkpoint_round_trip(self) -> None:
        """Reloading a checkpoint restores the same policy, encoder and targets."""
        agent = self.agent
        agent.update(self.replays, self.motions)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "agent.pt")
            agent.save(path)
            other, _, _ = build_agent()
            other.load_state_dict(torch.load(path, weights_only=False))
        observer = agent.observers[0]
        observation = observe(observer, random_state(observer.record, 2))
        z = project_to_sphere(torch.randn(2, Z))
        torch.testing.assert_close(other.act(0, observation, z, mean=True), agent.act(0, observation, z, mean=True))
        torch.testing.assert_close(other.goal_latent(0, observation.state), agent.goal_latent(0, observation.state))
        for new, old in zip(other.target_critic.parameters(), agent.target_critic.parameters()):
            torch.testing.assert_close(new, old)
        for new, old in zip(other.target_pretrained.parameters(), agent.target_pretrained.parameters()):
            torch.testing.assert_close(new, old)

    def test_discriminator_learns_to_separate_expert_states(self) -> None:
        """With separable data, the discriminator loss falls and experts score higher."""
        agent, _, _ = build_agent(lr_discriminator=3e-3)
        observer = agent.observers[0]
        joints = observer.record.graph.num_joints
        view = lambda value: StateView(torch.full((16, joints, 5), value), torch.full((16, 10), value), torch.zeros(16, joints, 7))
        expert, train = observer.state_inputs(view(1.0)), observer.state_inputs(view(-1.0))
        z = project_to_sphere(torch.randn(16, Z))
        first = float(agent.update_discriminator([expert], z, [train], z, [16])["disc_loss"])
        for _ in range(60):
            last = float(agent.update_discriminator([expert], z, [train], z, [16])["disc_loss"])
        self.assertLess(last, 0.7 * first)
        discriminator = agent.model.discriminator
        self.assertGreater(float(discriminator.compute_reward(expert, z).mean().detach()), float(discriminator.compute_reward(train, z).mean().detach()))

    def test_critic_learns_a_constant_reward(self) -> None:
        """With terminal-only transitions on a fixed batch, the value loss falls toward the reward."""
        agent, replays, _ = build_agent(lr_aux_critic=1e-3, lr_critic=1e-3, lr_f=1e-3)
        batch = agent._training_batch(replays, [8, 8])
        batch.discount = torch.zeros_like(batch.discount)
        z = project_to_sphere(torch.randn(16, Z))
        losses = [float(agent.update_value_networks(batch, z)["aux_critic_loss"]) for _ in range(120)]
        self.assertLess(np.mean(losses[-10:]), 0.3 * np.mean(losses[:10]))

    def test_rejects_invalid_configs(self) -> None:
        """Batch size and ensemble size are validated."""
        record = fixed_record(quadruped(), "quadruped")
        observer = EmbodimentObserver(record, HISTORY)
        common = dict(embedding_size=16, state_token_size=observer.state_token_size, hidden_size=16, num_attention_heads=2, seq_length=4)
        encoder = small_encoder(observer.token_size)
        with self.assertRaisesRegex(ValueError, "multiple"):
            DaGetFbCprAgent(BfmConfig(**common), FbCprConfig(batch_size=15), encoder, [observer])
        with self.assertRaisesRegex(ValueError, "num_parallel"):
            DaGetFbCprAgent(BfmConfig(num_parallel=1, **common), FbCprConfig(batch_size=16), encoder, [observer])
        with self.assertRaisesRegex(ValueError, "too small"):
            DaGetFbCprAgent(BfmConfig(**common), FbCprConfig(batch_size=4), encoder, [observer, observer])
        with self.assertRaises(ValueError):
            FbCprConfig(train_goal_ratio=0.7, expert_asm_ratio=0.6)


@unittest.skipUnless(HAS_MUJOCO, "needs mujoco (uv sync --extra sim)")
class EnvironmentTest(unittest.TestCase):
    """The MuJoCo environments and their agreement with the data pipeline."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        (root / "quad.urdf").write_text(quadruped())
        (root / "biped.urdf").write_text(humanoid())
        self.root = root
        self.quad_source = {"type": "mujoco", "name": "quad", "urdf": str(root / "quad.urdf"), "contact_links": FEET, "nominal_pose": STAND,
                            "randomized_embodiments": 0, "episodes_per_embodiment": 2, "steps": 60, "probe_torques": 0, "gain_scale": 3.0,
                            "push_probability": 0.0, "dt": 0.02}

    def env(self, randomize: bool = False, **overrides):
        """Return the environment of a standing quadruped and its robot."""
        from model.rl.envs import MujocoEmbodimentEnv
        from model.rl.trainer import TrainingArguments, load_embodiments

        arguments = TrainingArguments(data_config="", output_dir="", domain_randomization=randomize)
        embodiment = load_embodiments({"history_size": HISTORY, "sources": [self.quad_source]}, arguments)[0][0]
        for key, value in overrides.items():
            setattr(embodiment.env_config, key, value)
        return MujocoEmbodimentEnv(embodiment.record, embodiment.env_config), embodiment

    def test_observations_match_the_pretraining_pipeline(self) -> None:
        """Tokens, dynamics and mass matrix equal what pretraining feeds the encoder."""
        from model.rl.observation import trajectory_state
        from model.training.dataset import PhysicsDataCollator, PhysicsTrajectoryDataset
        from model.training.episodes import finalize_acceleration_scales
        from model.training.sources import mujoco_source

        records, trajectories = mujoco_source(self.quad_source, None, JointFeatureConfig.da_get_zero())
        record, trajectory = records[0], trajectories[0]
        episode = process_episode(record, trajectory)
        finalize_acceleration_scales(records, [(0, episode)])
        dataset = PhysicsTrajectoryDataset(records, [(0, episode)], HISTORY, include_probes=False)
        collator = PhysicsDataCollator(records, HISTORY)
        observer = EmbodimentObserver(record, HISTORY, observe_base_linear_velocity=True)
        state, height_known, linear_known = trajectory_state(record, trajectory)
        self.assertTrue(height_known and linear_known)
        for frame in (0, 1, 10):
            batch = collator([dataset[frame]])
            steps = torch.arange(frame - HISTORY + 1, frame + 1).clamp_min(0).tolist()
            local, global_features = [], []
            for step in steps:
                previous = (trajectory.actions[step - 1:step].to(torch.float64), state.q[step - 1:step], state.qd[step - 1:step]) if step > 0 else (None, None, None)
                step_local, step_global = observer.policy_step(state.select(torch.tensor([step])), *previous)
                local.append(step_local)
                global_features.append(step_global)
            previous_torque = trajectory.applied_torque[frame - 1:frame].to(torch.float64) if frame > 0 else None
            frame_state = state.select(torch.tensor([frame]))
            observation = observer.observation(torch.stack(local, dim=1), torch.stack(global_features, dim=1), frame_state, frame_state, previous_torque)
            tokens = observer.policy_tokens(observation.local, observation.global_features)
            torch.testing.assert_close(tokens, batch["input_values"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(observation.dynamics, batch["dynamics_features"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(observation.mass, batch["mass_matrix"], rtol=1e-4, atol=1e-5)

    def test_reset_to_a_motion_frame_reproduces_its_state_view(self) -> None:
        """A world reset to an expert frame has that frame's state tokens (height aside)."""
        env, embodiment = self.env(num_envs=2)
        motion = embodiment.motions.frames[0]
        frames = [{key: value[time] for key, value in motion.items()} for time in (5, 20)]
        env.reset(frames=frames)
        view = embodiment.observer.state_view(env.read_state())
        expert = embodiment.motions.states[0].select(torch.tensor([5, 20]))
        torch.testing.assert_close(view.local, expert.local, rtol=1e-3, atol=2e-3)
        torch.testing.assert_close(view.global_features[:, :9], expert.global_features[:, :9], rtol=1e-3, atol=2e-3)
        self.assertLess(float((view.global_features[:, 9] - expert.global_features[:, 9]).abs().max()), 0.15)

    def test_quadruped_holds_its_nominal_pose(self) -> None:
        """Zero actions keep a standing robot up for the whole episode, then the time limit ends it."""
        env, _ = self.env(num_envs=2, max_episode_steps=100)
        env.reset(frames=[None, None])
        for _ in range(100):
            result = env.step(np.zeros((2, env.joints)))
            self.assertFalse(bool(result.terminated.any()))
        self.assertTrue(bool(result.truncated.all()))
        self.assertFalse(env.fallen(result.state).any())

    def test_lying_starts(self) -> None:
        """About 30% of starts lie on the back or front, above the floor, and count as fallen."""
        env, embodiment = self.env(num_envs=1)
        env.set_motions(embodiment.motions.frames)
        frames = [env.sample_frame() for _ in range(600)]
        share = np.mean([frame.get("lying", False) for frame in frames])
        self.assertLess(abs(share - 0.3), 0.06)
        lying = next(frame for frame in frames if frame.get("lying", False))
        upright = next(frame for frame in frames if not frame.get("lying", False))
        np.testing.assert_allclose(lying["rotation"] @ lying["rotation"].T, np.eye(3), atol=1e-9)
        env.reset(frames=[lying])
        state = env.read_state()
        self.assertTrue(env.fallen(state).all())
        lowest = min(float(env.datas[0].geom_xpos[geom][2] - env.models[0].geom_size[geom, 0]) for geom in set(env.foot_geoms) | env.link_geoms)
        self.assertGreaterEqual(lowest, 0.0)
        env.reset(frames=[upright])
        self.assertFalse(env.fallen(env.read_state()).any())

    def test_falling_does_not_end_the_episode(self) -> None:
        """As in BFM-Zero, only time ends an episode; a fallen robot stays above the floor and is penalized."""
        env, embodiment = self.env(num_envs=1, max_episode_steps=120)
        lying = env._lie_down({key: value[0] for key, value in embodiment.motions.frames[0].items()})
        env.reset(frames=[lying])
        contact = 0.0
        for _ in range(119):
            result = env.step(np.zeros((1, env.joints)))
            self.assertFalse(bool(result.terminated.any()) or bool(result.truncated.any()))
            contact += float(result.aux_rewards["penalty_undesired_contact"][0])
        self.assertGreater(contact, 0.0)
        self.assertGreater(float(result.state.height[0]), 0.0)
        self.assertTrue(bool(env.step(np.zeros((1, env.joints))).truncated.all()))
        fall, _ = self.env(num_envs=1, terminate_on_fall=True)
        fall.reset(frames=[lying])
        self.assertTrue(bool(fall.step(np.zeros((1, fall.joints))).terminated.all()))

    def test_light_links_do_not_diverge(self) -> None:
        """A robot with light links and stiff gains stays finite under random actions."""
        from model.training.ground_sim import build_mjcf

        env, embodiment = self.env(num_envs=2, max_episode_steps=10**9)
        self.assertAlmostEqual(float(env.base_model.dof_armature[env.dof_index[0]]), 0.01)
        rng = np.random.default_rng(0)
        env.reset(frames=[None, None])
        for _ in range(100):
            self.assertFalse(bool(env.step(rng.uniform(-1, 1, (2, env.joints))).terminated.any()))
        xml = build_mjcf(embodiment.record, np.ones(env.joints), np.ones(env.joints), embodiment.env_config.geometry)
        self.assertIn('armature="1e-05"', xml)

    def test_diverged_worlds_count_as_fallen_in_evaluations(self) -> None:
        """A world that diverges is reset to a standing pose, but must not count as upright."""
        from model.rl.evaluation import run_tasks
        from model.rl.envs import MujocoGroup
        from model.rl.trainer import ObservationNoise, Rollout

        env, embodiment = self.env(num_envs=1, max_episode_steps=10**9)
        rollout = Rollout(env, embodiment.observer, ObservationNoise(enabled=False), torch.Generator())
        agent = mock.Mock()
        agent.act.side_effect = lambda robot, observation, z, mean: torch.zeros(1, env.joints)
        task = {"frame": None, "latents": torch.zeros(6, Z), "steps": 6}
        self.assertFalse(run_tasks(agent, [rollout], MujocoGroup([env]), [[task]])[0][0]["fallen"])
        original = env._diverged
        calls = []

        def diverge_once():
            calls.append(1)
            return np.array([len(calls) == 3]) | original()

        env._diverged = diverge_once
        result = run_tasks(agent, [rollout], MujocoGroup([env]), [[task]])[0][0]
        self.assertTrue(result["fallen"])
        self.assertFalse(env.fallen(env.read_state()).any())
        self.assertFalse(rollout.evaluating)

    def test_motion_weights_shift_episode_starts(self) -> None:
        """Worlds start mostly from high-priority motions."""
        env, embodiment = self.env(num_envs=1)
        first = {"q": np.zeros((5, env.joints)), "qd": np.zeros((5, env.joints)), "rotation": np.tile(np.eye(3), (5, 1, 1)),
                 "linear": np.zeros((5, 3)), "angular": np.zeros((5, 3))}
        second = {**first, "q": np.ones((5, env.joints))}
        env.set_motions([first, second], np.array([1.0, 99.0]))
        share = np.mean([float(env.sample_frame()["q"][0]) for _ in range(300)])
        self.assertGreater(share, 0.9)
        env.set_motion_weights(np.array([99.0, 1.0]))
        share = np.mean([float(env.sample_frame()["q"][0]) for _ in range(300)])
        self.assertLess(share, 0.1)
        del embodiment

    def test_auxiliary_rewards_follow_their_formulas(self) -> None:
        """Action rate and joint limits follow BFM-Zero's penalties."""
        env, _ = self.env(num_envs=2)
        env.reset(frames=[None, None])
        actions = np.stack([np.ones(env.joints), -np.ones(env.joints)])
        result = env.step(actions)
        torch.testing.assert_close(result.aux_rewards["penalty_action_rate"], torch.full((2,), float(env.joints), dtype=torch.float64))
        q = result.state.q.numpy()
        middle, span = 0.5 * (env.upper + env.lower), env.upper - env.lower
        low, high = middle - 0.475 * span, middle + 0.475 * span
        expected = (np.clip(low - q, 0, None) + np.clip(q - high, 0, None)).sum(axis=1)
        np.testing.assert_allclose(result.aux_rewards["limits_dof_pos"].numpy(), expected)

    def test_fixed_base_robot_tracks_targets(self) -> None:
        """A robot without a floating base gets no free joint and follows its targets."""
        from model.rl.envs import DomainRandomization, EnvConfig, MujocoEmbodimentEnv

        record = fixed_record(quadruped(), "arm")
        env = MujocoEmbodimentEnv(record, EnvConfig(num_envs=1, action_scale=0.2, randomization=DomainRandomization(enabled=False)))
        self.assertEqual(env.base_model.nq, record.graph.num_joints)
        env.reset(frames=[None])
        action = np.ones((1, env.joints))
        for _ in range(150):
            result = env.step(action)
        self.assertFalse(bool(result.terminated.any()))
        self.assertLess(float(np.abs(result.state.q.numpy()[0] - result.target.numpy()[0]).mean()), 0.1)

    def test_domain_randomization_varies_the_worlds(self) -> None:
        """Worlds differ in friction, mass and nominal pose; the base model is untouched."""
        env, _ = self.env(randomize=True, num_envs=3)
        frictions = {float(model.geom_friction[0, 0]) for model in env.models}
        self.assertEqual(len(frictions), 3)
        self.assertTrue(all(0.5 <= value <= 1.25 for value in frictions))
        self.assertFalse(np.allclose(env.models[0].body_mass, env.models[1].body_mass))
        scale = env.models[0].body_mass[1:] / env.base_model.body_mass[1:]
        self.assertTrue(((scale >= 0.95) & (scale <= 1.05)).all())
        self.assertLessEqual(float(np.abs(env.world_nominal - env.nominal).max()), 0.02)
        self.assertGreater(float(np.abs(env.world_nominal[0] - env.world_nominal[1]).max()), 0)

    def test_real_and_synthetic_sources_share_one_robot(self) -> None:
        """Sources with one robot name merge; synthetic trajectories are not expert data by default."""
        from model.rl.trainer import TrainingArguments, load_embodiments

        arguments = TrainingArguments(data_config="", output_dir="")
        second = {**self.quad_source, "robot": "quad", "seed": 5}
        synthetic = {"type": "synthetic", "name": "quad_sim", "robot": "quad", "urdf": str(self.root / "quad.urdf"), "randomized_embodiments": 0,
                     "episodes_per_embodiment": 1, "steps": 10, "probe_torques": 0}
        config = {"history_size": HISTORY, "sources": [{**self.quad_source, "robot": "quad"}, second, synthetic]}
        embodiments, _ = load_embodiments(config, arguments)
        self.assertEqual([item.record.name for item in embodiments], ["quad"])
        self.assertEqual(len(embodiments[0].motions), 4)
        marked = {**synthetic, "expert": True}
        embodiments, _ = load_embodiments({"history_size": HISTORY, "sources": [{**self.quad_source, "robot": "quad"}, {**marked, "floating_base": True, "contact_links": FEET, "base_modes": ["floating"]}]}, arguments)
        self.assertGreater(len(embodiments[0].motions), 2)
        (self.root / "other.urdf").write_text(humanoid())
        other = {"type": "mujoco", "name": "b", "robot": "quad", "urdf": str(self.root / "other.urdf"), "contact_links": ["left_ankle", "right_ankle"],
                 "randomized_embodiments": 0, "episodes_per_embodiment": 1, "steps": 25, "probe_torques": 0, "gain_scale": 3.0, "dt": 0.02,
                 "nominal_pose": {"left_knee": 0.1, "right_knee": 0.1}, "contact_geometry": {"radius": 0.01, "foot_size": [0.16, 0.08]}}
        with self.assertRaisesRegex(ValueError, "same joints"):
            load_embodiments({"history_size": HISTORY, "sources": [{**self.quad_source, "robot": "quad"}, other]}, arguments)
        with self.assertRaisesRegex(ValueError, "expert"):
            load_embodiments({"history_size": HISTORY, "sources": [synthetic]}, arguments)

    def test_script_trains_two_robots_and_resumes(self) -> None:
        """The script collects, updates all networks and the encoder, evaluates, logs, checkpoints and resumes."""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        biped = {"type": "mujoco", "name": "biped", "urdf": str(self.root / "biped.urdf"), "contact_links": ["left_ankle", "right_ankle"],
                 "contact_geometry": {"radius": 0.01, "foot_size": [0.16, 0.08]}, "randomized_embodiments": 0, "episodes_per_embodiment": 2,
                 "steps": 60, "probe_torques": 0, "gain_scale": 3.0, "dt": 0.02,
                 "nominal_pose": {"left_hip_pitch": -0.05, "right_hip_pitch": -0.05, "left_knee": 0.1, "right_knee": 0.1,
                                  "left_ankle_pitch": -0.05, "right_ankle_pitch": -0.05}}
        synthetic = {"type": "synthetic", "name": "quad_sim", "urdf": str(self.root / "quad.urdf"), "randomized_embodiments": 1,
                     "episodes_per_embodiment": 1, "steps": 10, "probe_torques": 1}
        config = self.root / "rl.json"
        config.write_text(json.dumps({"history_size": HISTORY, "joint_features": "da_get_zero", "eval_fraction": 0.0, "sources": [self.quad_source, biped, synthetic]}))
        path = Path(__file__).resolve().parents[1] / "scripts" / "train_bfm.py"
        spec = importlib.util.spec_from_file_location("train_bfm", path)
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(script)
        output = self.root / "run"
        common = [
            "--simulator", "mujoco", "--data_config", str(config), "--output_dir", str(output), "--device", "cpu", "--online_parallel_envs", "4",
            "--num_seed_steps", "16", "--update_agent_every", "8", "--num_agent_updates", "1", "--batch_size", "16", "--seq_length", "4",
            "--hidden_size", "16", "--num_attention_heads", "2", "--intermediate_size", "32", "--actor_layers", "1", "--forward_layers", "1",
            "--z_dim", "8", "--num_latent_tokens", "2", "--backward_hidden_size", "16", "--discriminator_hidden_size", "16",
            "--state_hidden_size", "16", "--state_num_layers", "1", "--state_num_heads", "2", "--state_intermediate_size", "32",
            "--log_every", "8", "--eval_every", "24", "--eval_motions", "1", "--buffer_size", "200", "--max_episode_steps", "20",
            "--num_threads", "2", "--z_buffer_size", "64", "--rollout_expert_trajectories_length", "10", "--physics_batch_size", "4",
            "--physics_cache_dir", str(self.root / "cache"),
        ]
        script.main([*common, "--num_env_steps", "48"])
        checkpoint = output / "checkpoint.pt"
        state = torch.load(checkpoint, weights_only=False)
        self.assertEqual(state["frames"], 48)
        self.assertEqual(state["config"]["discount"], 0.98)
        self.assertEqual(float(state["motion_priorities"][0].min()) >= 2.0, True)
        self.assertEqual(DaGetForPhysicsPretraining.from_pretrained(output / "encoder").config.encoder["hidden_size"], 128)
        events = EventAccumulator(str(output / "runs"))
        events.Reload()
        tags = set(events.Tags()["scalars"])
        for tag in ("train/fb_loss", "train/critic_loss", "train/aux_critic_loss", "train/actor_loss", "train/disc_loss", "train/frames_per_second",
                    "train/physics/loss_fk", "train/aux_rew/penalty_undesired_contact", "rollout/quad/0-ground/fallen_fraction",
                    "eval/quad/0-ground/tracking_mpjpe", "eval/quad/0-ground/tracking_emd", "eval/quad/0-ground/getting_up_rate", "eval/biped/0-ground/goal_mpjpe"):
            self.assertIn(tag, tags)
        names = [item["name"] for item in json.loads((output / "embodiments.json").read_text())]
        self.assertEqual(names, ["quad/0-ground", "biped/0-ground"])
        script.main([*common, "--num_env_steps", "64", "--resume_from", str(checkpoint)])
        self.assertEqual(torch.load(checkpoint, weights_only=False)["frames"], 64)


class EncoderCheckpointTest(unittest.TestCase):
    """Loading the pretrained encoder."""

    def test_token_size_must_match(self) -> None:
        """A checkpoint trained with other tokens is rejected with a clear error."""
        from model.rl.trainer import build_encoder

        model = small_encoder(30)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            self.assertEqual(build_encoder(folder, 30).config.encoder["input_size"], 30)
            with self.assertRaisesRegex(ValueError, "token size"):
                build_encoder(folder, 31)


if __name__ == "__main__":
    unittest.main()
