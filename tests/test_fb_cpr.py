"""Test the cross-embodiment FB-CPR agent, its environments and its training script."""

import importlib.util
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
from model.rl import DaGetFbCprAgent, EmbodimentObserver, ExpertMotions, FbCprConfig, FrozenEncoder, ReplayBuffer, RobotState, flat_state_size
from model.rl.agent import EmaRewardNormalizer, ZBuffer, ensemble_pessimism, fb_bellman_loss, orthonormality_loss, soft_update, split_counts
from model.rl.buffers import rotation_from_gravity
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


def build_agent(**overrides) -> tuple[DaGetFbCprAgent, list[ReplayBuffer], list[ExpertMotions]]:
    """Return an agent on two fixed-base robots of different size, with filled buffers."""
    records = [fixed_record(quadruped(), "quadruped"), fixed_record(humanoid(), "humanoid")]
    max_joints = max(record.graph.num_joints for record in records)
    observers = [EmbodimentObserver(record, HISTORY, max_joints) for record in records]
    size = flat_state_size(max_joints)
    model_config = BfmConfig(embedding_size=16, backward_input_size=size, context_size=size, z_dim=Z, hidden_size=16,
                             num_attention_heads=2, intermediate_size=32, num_latent_tokens=2, actor_layers=1, forward_layers=1,
                             backward_hidden_size=32, discriminator_hidden_size=32, seq_length=4)
    values = {"batch_size": 16, "z_buffer_size": 32, **overrides}
    agent = DaGetFbCprAgent(model_config, FbCprConfig(**values), FrozenEncoder(small_encoder(observers[0].token_size), observers))
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


def snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    """Copy a module's parameters."""
    return [parameter.detach().clone() for parameter in module.parameters()]


def changed(module: torch.nn.Module, before: list[torch.Tensor]) -> bool:
    """Return whether any parameter differs from a snapshot."""
    return any(not torch.equal(parameter, old) for parameter, old in zip(module.parameters(), before))


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
        """The matrix form equals the loop over members, states and goals."""
        torch.manual_seed(0)
        members, batch = 2, 4
        measures, target = torch.randn(members, batch, batch), torch.randn(batch, batch)
        discount = torch.tensor([[0.98], [0.0], [0.98], [0.98]])
        off, diagonal = 0.0, 0.0
        for member in range(members):
            for i in range(batch):
                for j in range(batch):
                    residual = measures[member, i, j] - discount[i, 0] * target[i, j]
                    if i == j:
                        diagonal -= residual / batch
                    else:
                        off += 0.5 * residual**2 / (batch**2 - batch)
        loss, loss_diagonal, loss_off = fb_bellman_loss(measures, discount, target)
        torch.testing.assert_close(loss_off, torch.as_tensor(off))
        torch.testing.assert_close(loss_diagonal, torch.as_tensor(diagonal))
        torch.testing.assert_close(loss, loss_off + loss_diagonal)

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
        scaled = normalizer(values)
        torch.testing.assert_close(scaled, values / torch.sqrt(torch.tensor(5.0 - 4.0)), rtol=1e-4, atol=1e-4)

    def test_motions_are_resampled_to_the_control_rate(self) -> None:
        """Frames of a 30 Hz motion land on 50 Hz control steps, in time and in value."""
        from model.rl.buffers import resample_state

        record = fixed_record(quadruped(), "quadruped")
        steps, dt, control_dt = 31, 1 / 30, 0.02
        time = torch.arange(steps, dtype=torch.float64) * dt
        q = torch.sin(time)[:, None].expand(steps, record.graph.num_joints)
        qd = torch.cos(time)[:, None].expand_as(q)
        observer = EmbodimentObserver(record, HISTORY, record.graph.num_joints)
        motions = ExpertMotions(observer, [Trajectory(embodiment=0, dt=dt, positions=q, velocities=qd)], control_dt)
        frames = motions.frames[0]
        self.assertEqual(len(frames["q"]), int(1.0 / control_dt) + 1)
        expected = np.sin(np.arange(len(frames["q"])) * control_dt)
        np.testing.assert_allclose(frames["q"][:, 0], expected, atol=2e-4)
        self.assertEqual(len(motions.flat[0]), len(frames["q"]))
        zeros = torch.zeros((steps, 3), dtype=torch.float64)
        state = RobotState(q, qd, zeros, zeros, zeros, zeros[:, 0])
        rotation = torch.eye(3, dtype=torch.float64).expand(steps, 3, 3)
        same, same_rotation = resample_state(state, rotation, dt, dt)
        self.assertIs(same, state)
        self.assertIs(same_rotation, rotation)

    def test_rotation_from_gravity(self) -> None:
        """The rotation maps world down to the given base-frame direction."""
        direction = torch.nn.functional.normalize(torch.tensor([[0.3, -0.2, -1.0]], dtype=torch.float64), dim=-1)
        rotation = rotation_from_gravity(direction)
        torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), torch.eye(3, dtype=torch.float64)[None])
        down = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
        torch.testing.assert_close((rotation.transpose(-1, -2) @ down[:, None]).squeeze(-1), direction)


class ObservationTest(unittest.TestCase):
    """Flat states and buffers."""

    def test_flat_state_layout(self) -> None:
        """Each slot holds the documented feature, and unused joints stay zero."""
        record = fixed_record(quadruped(), "quadruped")
        joints = record.graph.num_joints
        max_joints = joints + 3
        observer = EmbodimentObserver(record, HISTORY, max_joints)
        state = random_state(record, 2)
        flat = observer.flat_state(state)
        self.assertEqual(flat.shape, (2, flat_state_size(max_joints)))
        torch.testing.assert_close(flat[:, :3], state.gravity_direction.float())
        positions = flat[:, 10:10 + max_joints]
        torch.testing.assert_close(positions[:, :joints], record.graph.normalize_joint_positions(state.q).float())
        mask = flat[:, 10 + 2 * max_joints : 10 + 3 * max_joints]
        torch.testing.assert_close(mask[0], torch.tensor([1.0] * joints + [0.0] * 3))
        self.assertEqual(float(flat[:, 10 + max_joints + joints : 10 + 2 * max_joints].abs().sum()), 0.0)

    def test_unknown_signals_are_hidden_from_the_flat_state(self) -> None:
        """When motions lack height or linear velocity, simulated states drop them too."""
        record = fixed_record(quadruped(), "quadruped")
        state = random_state(record, 2)
        state.linear_velocity = state.linear_velocity + 1.0
        state.height = state.height + 1.0
        observer = EmbodimentObserver(record, HISTORY, record.graph.num_joints, use_height=False, use_linear_velocity=False)
        self.assertEqual(float(observer.flat_state(state)[:, 6:10].abs().sum()), 0.0)

    def test_replay_buffer_wraps(self) -> None:
        """The ring keeps the newest transitions."""
        record = fixed_record(quadruped(), "quadruped")
        observer = EmbodimentObserver(record, HISTORY, record.graph.num_joints)
        buffer = ReplayBuffer(3, ("a",))
        for value in range(2):
            count = 2
            observation = observe(observer, random_state(record, count))
            z = torch.full((count, 1), float(value))
            buffer.add(observation, torch.zeros(count, record.graph.num_joints), observation, z, torch.zeros(count, dtype=torch.bool),
                       {"a": torch.zeros(count)})
        self.assertEqual(len(buffer), 3)
        self.assertEqual(sorted(buffer.storage["z"][:, 0].tolist()), [0.0, 1.0, 1.0])
        self.assertEqual(buffer.sample(5)["tokens"].shape[0], 5)


class AgentTest(unittest.TestCase):
    """The update rules."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.agent, self.replays, self.motions = build_agent()

    def prepared(self):
        """Return a batch with normalized states and latents."""
        agent = self.agent
        batch = agent._training_batch(self.replays)
        return batch, agent.normalize(batch.flat), agent.normalize(batch.next_flat), project_to_sphere(torch.randn(batch.flat.shape[0], Z))

    def test_batch_has_an_equal_share_of_each_robot(self) -> None:
        """Robots are padded together with zero embeddings on missing joints."""
        batch, *_ = self.prepared()
        self.assertEqual(int((batch.robots == 0).sum()), 8)
        self.assertEqual(int((batch.robots == 1).sum()), 8)
        joints = [record.record.graph.num_joints for record in self.agent.encoder.observers]
        for robot, count in enumerate(joints):
            rows = batch.robots == robot
            self.assertTrue(bool((batch.joint_mask[rows].sum(1) == count).all()))
            self.assertEqual(float(batch.embeddings[rows][:, count:].abs().sum()), 0.0)
            self.assertEqual(float(batch.action[rows][:, count:].abs().sum()), 0.0)
        self.assertEqual(batch.embeddings.shape[1], max(joints))

    def test_terminal_transitions_have_zero_discount(self) -> None:
        """The discount is zero exactly where the episode terminated, and gamma elsewhere."""
        self.replays[0].storage["terminated"][:] = True
        self.replays[1].storage["terminated"][:] = False
        batch, *_ = self.prepared()
        self.assertEqual(set(batch.discount[batch.robots == 0].flatten().tolist()), {0.0})
        self.assertEqual(set(batch.discount[batch.robots == 1].flatten().tolist()), {as_float32(0.98)})

    def test_each_update_changes_only_its_networks(self) -> None:
        """Every optimizer steps its own network, and targets wait for the soft update."""
        agent = self.agent
        model = agent.model
        batch, flat, next_flat, z = self.prepared()
        networks = {name: getattr(model, name) for name in ("forward_map", "backward_map", "actor", "critic", "aux_critic", "discriminator")}
        targets = [agent.target_forward_map, agent.target_backward_map, agent.target_critic, agent.target_aux_critic]
        steps = {
            ("forward_map", "backward_map"): lambda: agent.update_fb(batch, z, flat, next_flat),
            ("critic",): lambda: agent.update_critic(batch, z, flat, next_flat),
            ("aux_critic",): lambda: agent.update_aux_critic(batch, z, flat, next_flat),
            ("actor",): lambda: agent.update_actor(batch, z, flat),
            ("discriminator",): lambda: agent.update_discriminator(flat, z, next_flat, z),
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

    def test_encoder_stays_frozen(self) -> None:
        """A full update leaves the encoder's weights and mode alone."""
        before = snapshot(self.agent.encoder)
        self.agent.update(self.replays, self.motions)
        self.assertFalse(changed(self.agent.encoder, before))
        self.assertFalse(self.agent.encoder.pretrained.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.agent.encoder.parameters()))

    def test_full_update_moves_targets_by_tau(self) -> None:
        """All networks learn, the targets follow by their tau, and the metrics are finite."""
        agent = self.agent
        forward, target = snapshot(agent.model.critic), snapshot(agent.target_critic)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(forward, target)))
        metrics = agent.update(self.replays, self.motions)
        self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))
        self.assertIn("aux_rew/limits_dof_pos", metrics)
        tau = agent.config.critic_target_tau
        for new, old, online in zip(agent.target_critic.parameters(), target, agent.model.critic.parameters()):
            torch.testing.assert_close(new, (1 - tau) * old + tau * online, rtol=1e-5, atol=1e-6)

    def test_goal_and_expert_prompts_stay_within_a_robot(self) -> None:
        """A transition never gets a goal or a motion of another robot."""
        agent = self.agent
        count = 16
        robots = torch.tensor([0] * 8 + [1] * 8)
        goals = torch.randn(count, agent.model_config.backward_input_size)
        encoded = agent.model.project_z(agent.model.backward_map(goals)).detach()
        expert_z = project_to_sphere(torch.randn(count, Z))
        expert_robots = torch.tensor([1] * 8 + [0] * 8)
        agent.config.train_goal_ratio, agent.config.expert_asm_ratio = 1.0, 0.0
        z = agent.sample_mixed_z(robots, goals, expert_z, expert_robots)
        for row in range(count):
            source = torch.cdist(z[row:row + 1], encoded)[0].argmin()
            self.assertEqual(int(robots[source]), int(robots[row]))
        agent.config.train_goal_ratio, agent.config.expert_asm_ratio = 0.0, 1.0
        z = agent.sample_mixed_z(robots, goals, expert_z, expert_robots)
        for row in range(count):
            source = torch.cdist(z[row:row + 1], expert_z)[0].argmin()
            self.assertEqual(int(expert_robots[source]), int(robots[row]))

    def test_expert_latents_average_each_window(self) -> None:
        """An expert latent is the projected mean of B over its window."""
        agent = self.agent
        states = torch.randn(16, agent.model_config.backward_input_size)
        z = agent.encode_expert(states)
        embeddings = agent.model.backward_map(states[:4]).detach()
        torch.testing.assert_close(z[0], project_to_sphere(embeddings.mean(0)))
        torch.testing.assert_close(z[3], z[0])

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
        """Reloading a checkpoint restores the same policy, targets and statistics."""
        agent = self.agent
        agent.update(self.replays, self.motions)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "agent.pt")
            agent.save(path)
            other, _, _ = build_agent()
            other.load_state_dict(torch.load(path, weights_only=False))
        observer = agent.encoder.observers[0]
        observation = observe(observer, random_state(observer.record, 2))
        z = project_to_sphere(torch.randn(2, Z))
        other.encoder = agent.encoder
        torch.testing.assert_close(other.act(0, observation, z, mean=True), agent.act(0, observation, z, mean=True))
        torch.testing.assert_close(other.goal_latent(observation.flat), agent.goal_latent(observation.flat))
        for new, old in zip(other.target_critic.parameters(), agent.target_critic.parameters()):
            torch.testing.assert_close(new, old)

    def test_discriminator_learns_to_separate_expert_states(self) -> None:
        """With separable data, the discriminator loss falls and experts score higher."""
        agent, replays, motions = build_agent(lr_discriminator=3e-3, grad_penalty_discriminator=0.0)
        size = agent.model_config.backward_input_size
        expert, train = torch.ones(16, size), -torch.ones(16, size)
        z = project_to_sphere(torch.randn(16, Z))
        first = float(agent.update_discriminator(expert, z, train, z)["disc_loss"])
        for _ in range(40):
            last = float(agent.update_discriminator(expert, z, train, z)["disc_loss"])
        self.assertLess(last, 0.5 * first)
        discriminator = agent.model.discriminator
        self.assertGreater(float(discriminator.compute_reward(expert, z).mean()), float(discriminator.compute_reward(train, z).mean()))

    def test_critic_learns_a_constant_reward(self) -> None:
        """With terminal-only transitions, Q_R converges to the (normalized) reward."""
        agent, replays, motions = build_agent(lr_aux_critic=1e-3)
        batch = agent._training_batch(replays)
        batch.discount = torch.zeros_like(batch.discount)
        batch.aux_rewards = torch.ones_like(batch.aux_rewards)
        flat, next_flat = agent.normalize(batch.flat), agent.normalize(batch.next_flat)
        z = project_to_sphere(torch.randn(flat.shape[0], Z))
        losses = [float(agent.update_aux_critic(batch, z, flat, next_flat)["aux_critic_loss"]) for _ in range(150)]
        self.assertLess(np.mean(losses[-10:]), 0.2 * np.mean(losses[:10]))

    def test_rejects_invalid_configs(self) -> None:
        """Batch size, ensemble size and context width are validated."""
        record = fixed_record(quadruped(), "quadruped")
        observer = EmbodimentObserver(record, HISTORY, record.graph.num_joints)
        encoder = FrozenEncoder(small_encoder(observer.token_size), [observer])
        size = flat_state_size(observer.max_joints)
        common = dict(embedding_size=16, backward_input_size=size, hidden_size=16, num_attention_heads=2)
        with self.assertRaisesRegex(ValueError, "multiple"):
            DaGetFbCprAgent(BfmConfig(context_size=size, **common), FbCprConfig(batch_size=15), encoder)
        with self.assertRaisesRegex(ValueError, "context_size"):
            DaGetFbCprAgent(BfmConfig(**common), FbCprConfig(batch_size=16), encoder)
        with self.assertRaisesRegex(ValueError, "num_parallel"):
            DaGetFbCprAgent(BfmConfig(context_size=size, num_parallel=1, **common), FbCprConfig(batch_size=16), encoder)
        with self.assertRaises(ValueError):
            FbCprConfig(train_goal_ratio=0.7, expert_asm_ratio=0.6)


def as_float32(value: float) -> float:
    """Round a float the way a ``float32`` tensor stores it."""
    return torch.tensor(value, dtype=torch.float32).item()


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

    def env(self, **overrides):
        """Return the environment of a standing quadruped and its robot."""
        from model.rl.envs import MujocoEmbodimentEnv
        from model.rl.trainer import load_embodiments

        embodiment = load_embodiments({"history_size": HISTORY, "sources": [self.quad_source]}, domain_randomization=False)[0][0]
        for key, value in overrides.items():
            setattr(embodiment.env_config, key, value)
        return MujocoEmbodimentEnv(embodiment.record, embodiment.env_config), embodiment

    def test_observations_match_the_pretraining_pipeline(self) -> None:
        """Tokens, dynamics and mass matrix equal what pretraining feeds the encoder."""
        from model.training.dataset import PhysicsDataCollator, PhysicsTrajectoryDataset
        from model.training.episodes import finalize_acceleration_scales
        from model.training.sources import mujoco_source

        records, trajectories = mujoco_source(self.quad_source, None, JointFeatureConfig.da_get_zero())
        record, trajectory = records[0], trajectories[0]
        episode = process_episode(record, trajectory)
        finalize_acceleration_scales(records, [(0, episode)])
        dataset = PhysicsTrajectoryDataset(records, [(0, episode)], HISTORY, include_probes=False)
        collator = PhysicsDataCollator(records, HISTORY)
        observer = EmbodimentObserver(record, HISTORY, record.graph.num_joints, observe_base_linear_velocity=True)
        from model.rl.observation import trajectory_state

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
            observation = observer.observation(
                torch.stack(local, dim=1), torch.stack(global_features, dim=1), state.select(torch.tensor([frame])),
                state.select(torch.tensor([frame])), previous_torque,
            )
            torch.testing.assert_close(observation.tokens, batch["input_values"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(observation.dynamics, batch["dynamics_features"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(observation.mass, batch["mass_matrix"], rtol=1e-4, atol=1e-5)

    def test_reset_to_a_motion_frame_reproduces_its_flat_state(self) -> None:
        """A world reset to an expert frame has that frame's flat state (height aside)."""
        env, embodiment = self.env(num_envs=2)
        motion = embodiment.motions.frames[0]
        frames = [{key: value[time] for key, value in motion.items()} for time in (5, 20)]
        env.reset(frames=frames)
        flat = embodiment.observer.flat_state(env.state())
        expert = embodiment.motions.flat[0][[5, 20]]
        no_height = [index for index in range(flat.shape[1]) if index != 9]
        torch.testing.assert_close(flat[:, no_height], expert[:, no_height], rtol=1e-3, atol=2e-3)
        self.assertLess(float((flat[:, 9] - expert[:, 9]).abs().max()), 0.15)

    def test_quadruped_holds_its_nominal_pose(self) -> None:
        """Zero actions keep a standing robot up for the whole episode, then the time limit ends it."""
        env, embodiment = self.env(num_envs=2, max_episode_steps=100)
        env.set_motions([])
        env.reset()
        for step in range(100):
            result = env.step(np.zeros((2, env.joints)))
            self.assertFalse(bool(result.terminated.any()), f"fell at step {step}")
        self.assertTrue(bool(result.truncated.all()))
        self.assertLess(float((result.state.q.numpy() - env.nominal).__abs__().max()), 0.3)

    def test_falling_terminates_and_rewards_follow_their_formulas(self) -> None:
        """Pushing joints to extremes ends the episode; aux rewards are the BFM-Zero penalties."""
        env, _ = self.env(num_envs=2)
        env.set_motions([])
        env.reset()
        actions = np.stack([np.ones(env.joints), -np.ones(env.joints)])
        result = env.step(actions)
        torch.testing.assert_close(result.aux_rewards["penalty_action_rate"], torch.full((2,), float(env.joints), dtype=torch.float64))
        q = result.state.q.numpy()
        middle, span = 0.5 * (env.upper + env.lower), env.upper - env.lower
        soft_low, soft_high = middle - 0.475 * span, middle + 0.475 * span
        expected = (np.clip(soft_low - q, 0, None) + np.clip(q - soft_high, 0, None)).sum(axis=1)
        np.testing.assert_allclose(result.aux_rewards["limits_dof_pos"].numpy(), expected)
        ended = np.zeros(2, dtype=bool)
        for _ in range(300):
            ended |= env.step(actions).terminated.numpy()
            if ended.all():
                break
        self.assertTrue(ended.all())

    def test_fixed_base_robot_tracks_targets(self) -> None:
        """A robot without a floating base gets no free joint and follows its targets."""
        from model.rl.envs import EnvConfig, MujocoEmbodimentEnv

        record = fixed_record(quadruped(), "arm")
        env = MujocoEmbodimentEnv(record, EnvConfig(num_envs=1, action_scale=0.2, randomization=mock.Mock(enabled=False, push_interval=(1.0, 3.0))))
        self.assertEqual(env.base_model.nq, record.graph.num_joints)
        env.reset()
        action = np.ones((1, env.joints))
        for _ in range(150):
            result = env.step(action)
        self.assertFalse(bool(result.terminated.any()))
        moved = result.state.q.numpy()[0]
        target = result.target.numpy()[0]
        self.assertLess(float(np.abs(moved - target).mean()), 0.1)

    def test_domain_randomization_varies_the_worlds(self) -> None:
        """Worlds differ in friction, mass and nominal pose; the base model is untouched."""
        from model.rl.envs import MujocoEmbodimentEnv
        from model.rl.trainer import load_embodiments

        embodiment = load_embodiments({"history_size": HISTORY, "sources": [self.quad_source]})[0][0]
        embodiment.env_config.num_envs = 3
        env = MujocoEmbodimentEnv(embodiment.record, embodiment.env_config)
        frictions = {float(model.geom_friction[0, 0]) for model in env.models}
        self.assertEqual(len(frictions), 3)
        self.assertTrue(all(0.5 <= value <= 1.25 for value in frictions))
        self.assertFalse(np.allclose(env.models[0].body_mass, env.models[1].body_mass))
        scale = env.models[0].body_mass[1:] / env.base_model.body_mass[1:]
        self.assertTrue(((scale >= 0.95) & (scale <= 1.05)).all())
        self.assertLessEqual(float(np.abs(env.world_nominal - env.nominal).max()), 0.02)
        self.assertGreater(float(np.abs(env.world_nominal[0] - env.world_nominal[1]).max()), 0)

    def test_script_trains_two_robots_and_resumes(self) -> None:
        """The script collects, updates, evaluates, logs to TensorBoard, checkpoints and resumes."""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        biped = {"type": "mujoco", "name": "biped", "urdf": str(self.root / "biped.urdf"), "contact_links": ["left_ankle", "right_ankle"],
                 "contact_geometry": {"radius": 0.01, "foot_size": [0.16, 0.08]}, "randomized_embodiments": 0, "episodes_per_embodiment": 2,
                 "steps": 60, "probe_torques": 0, "gain_scale": 3.0, "dt": 0.02,
                 "nominal_pose": {"left_hip_pitch": -0.05, "right_hip_pitch": -0.05, "left_knee": 0.1, "right_knee": 0.1,
                                  "left_ankle_pitch": -0.05, "right_ankle_pitch": -0.05}}
        config = self.root / "rl.json"
        config.write_text(json.dumps({"history_size": HISTORY, "joint_features": "da_get_zero", "sources": [self.quad_source, biped]}))
        path = Path(__file__).resolve().parents[1] / "scripts" / "train_bfm.py"
        spec = importlib.util.spec_from_file_location("train_bfm", path)
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(script)
        output = self.root / "run"
        common = [
            "--data_config", str(config), "--output_dir", str(output), "--device", "cpu", "--online_parallel_envs", "4",
            "--num_seed_steps", "16", "--update_agent_every", "8", "--num_agent_updates", "1", "--batch_size", "16", "--seq_length", "4",
            "--hidden_size", "16", "--num_attention_heads", "2", "--intermediate_size", "32", "--actor_layers", "1", "--forward_layers", "1",
            "--z_dim", "8", "--num_latent_tokens", "2", "--backward_hidden_size", "16", "--discriminator_hidden_size", "16",
            "--log_every", "8", "--eval_every", "24", "--eval_motions", "1", "--buffer_size", "200", "--max_episode_steps", "20",
            "--num_threads", "2", "--z_buffer_size", "64", "--rollout_expert_trajectories_length", "10",
        ]
        script.main([*common, "--num_env_steps", "48"])
        checkpoint = output / "checkpoint.pt"
        self.assertTrue(checkpoint.exists())
        state = torch.load(checkpoint, weights_only=False)
        self.assertEqual(state["frames"], 48)
        self.assertEqual(state["config"]["discount"], 0.98)
        events = EventAccumulator(str(output / "runs"))
        events.Reload()
        tags = set(events.Tags()["scalars"])
        for tag in ("train/fb_loss", "train/critic_loss", "train/aux_critic_loss", "train/actor_loss", "train/disc_loss", "train/frames_per_second",
                    "train/aux_rew/limits_dof_pos", "eval/quad/0-ground/tracking_mpjpe", "eval/biped/0-ground/goal_survival"):
            self.assertIn(tag, tags)
        self.assertEqual(json.loads((output / "embodiments.json").read_text())[0]["name"], "quad/0-ground")
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
