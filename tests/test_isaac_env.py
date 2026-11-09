"""Test the Isaac Lab adapter's logic against a fake of the Isaac Lab API.

Isaac Lab needs an NVIDIA GPU, so these tests check the adapter's bookkeeping (joint and body
orders, frames, batching of writes, penalties), not the simulator. The fake follows the
signatures of Isaac Lab 3.0.0 (commit 4cfdfcf).
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from model import EmbodimentGraph, JointFeatureConfig, actuator_specs
from model.rl import EmbodimentObserver
from model.rl.envs import DomainRandomization, EnvConfig
from model.rl.isaac_env import IsaacEmbodimentEnv, IsaacGroup, matrix_to_xyzw, rotate_inverse_xyzw
from model.training.episodes import EmbodimentRecord, Trajectory
from model.training.ground_sim import ground_specs
from tests.test_embodiment import quadruped

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def proxy(values: torch.Tensor) -> SimpleNamespace:
    """Mimic Isaac Lab's ``ProxyArray``."""
    return SimpleNamespace(torch=values)


class FakeArticulation:
    """Record what the adapter writes, with joints and bodies in simulator order."""

    def __init__(self, count: int, record: EmbodimentRecord, extra_joint: bool = True) -> None:
        names = list(reversed(record.graph.joint_names))
        self.joint_names = (["frozen_joint"] if extra_joint else []) + names
        self.body_names = ["base", *reversed(FEET), "FL_thigh"]
        joints, bodies = len(self.joint_names), len(self.body_names)
        self.count, self.joints = count, joints
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(count, bodies, 4).clone()
        self.state = SimpleNamespace(
            joint_pos=torch.zeros(count, joints), joint_vel=torch.zeros(count, joints), default_joint_pos=torch.full((count, joints), 0.25),
            applied_torque=torch.zeros(count, joints), projected_gravity_b=torch.tensor([0.0, 0.0, -1.0]).expand(count, 3).clone(),
            root_link_ang_vel_b=torch.zeros(count, 3), root_link_lin_vel_b=torch.zeros(count, 3), root_link_pos_w=torch.zeros(count, 3),
            root_link_lin_vel_w=torch.zeros(count, 3), root_link_ang_vel_w=torch.zeros(count, 3), body_link_lin_vel_w=torch.zeros(count, bodies, 3),
            body_link_quat_w=identity, body_mass=torch.ones(count, bodies), body_inertia=torch.ones(count, bodies, 9),
            body_com_pose_b=torch.zeros(count, bodies, 7),
        )
        self.data = SimpleNamespace(**{name: proxy(value) for name, value in vars(self.state).items()})
        self.actuators = SimpleNamespace(target_command=SimpleNamespace(set_position_index=self._set_targets))
        self.calls: list[str] = []
        self.targets: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self.poses: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.velocities: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.joint_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.joint_velocity_writes: list[torch.Tensor] = []
        self.masses = self.inertias = self.coms = None
        self.materials = torch.ones(count, bodies, 3)
        self.material_ids = None
        self.root_view = SimpleNamespace(get_material_properties=lambda: self.materials, set_material_properties=self._set_materials)

    def _set_targets(self, *, value, env_ids=None):
        self.targets.append((value.clone(), env_ids))

    def _set_materials(self, materials, ids):
        self.materials, self.material_ids = materials, ids

    def set_masses_index(self, *, masses, env_ids):
        self.masses = masses

    def set_inertias_index(self, *, inertias, env_ids):
        self.inertias = inertias

    def set_coms_index(self, *, coms, env_ids):
        self.coms = coms

    def reset(self, env_ids=None):
        self.calls.append("reset")

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids):
        self.poses.append((root_pose.clone(), env_ids))

    def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids):
        self.velocities.append((root_velocity.clone(), env_ids))

    def write_joint_position_to_sim_index(self, *, position, env_ids):
        self.joint_writes.append((position.clone(), env_ids))

    def write_joint_velocity_to_sim_index(self, *, velocity, env_ids):
        self.joint_velocity_writes.append(velocity.clone())

    def write_data_to_sim(self):
        self.calls.append("write")

    def update(self, dt):
        self.calls.append(f"update:{dt}")


class FakeSensor:
    """Report contact forces, with bodies in sensor order."""

    def __init__(self, articulation: FakeArticulation) -> None:
        self.body_names = list(reversed(articulation.body_names))
        self.forces = torch.zeros(articulation.count, len(self.body_names), 3)
        self.data = SimpleNamespace(net_forces_w=proxy(self.forces))
        self.calls: list[str] = []

    def reset(self, env_ids=None):
        self.calls.append("reset")

    def update(self, dt):
        self.calls.append(f"update:{dt}")


def record() -> EmbodimentRecord:
    """Return the quadruped in position mode, with feet."""
    graph = EmbodimentGraph.from_urdf(quadruped(), JointFeatureConfig.da_get_zero(), floating_base=True)
    return EmbodimentRecord.build("quad", graph, ground_specs(actuator_specs(graph)), contact_links=FEET, surface_contacts=True)


def build(count: int = 2, randomize: bool = False, **overrides):
    """Return the adapter on fakes, with its robot."""
    robot = record()
    articulation = FakeArticulation(count, robot)
    sensor = FakeSensor(articulation)
    config = EnvConfig(num_envs=count, randomization=DomainRandomization(enabled=randomize), **overrides)
    origins = torch.tensor([[float(index) * 5, 1.0, 0.0] for index in range(count)])
    refreshes = []
    env = IsaacEmbodimentEnv(robot, config, articulation, sensor, origins, refresh=lambda: refreshes.append(1))
    return env, articulation, sensor, refreshes, robot


def frame(robot: EmbodimentRecord, rotation: np.ndarray | None = None, lying: bool = False) -> dict:
    """Return a motion frame with distinct joint values."""
    joints = robot.graph.num_joints
    result = {"q": 0.1 * np.arange(joints, dtype=np.float64), "qd": np.arange(joints, dtype=np.float64), "rotation": np.eye(3) if rotation is None else rotation,
              "linear": np.array([1.0, 0.0, 0.0]), "angular": np.array([0.0, 0.0, 2.0])}
    return {**result, "lying": True} if lying else result


class QuaternionTest(unittest.TestCase):
    """Isaac Lab 3 uses x, y, z, w quaternions."""

    def test_matrix_to_quaternion_round_trip(self) -> None:
        """Every branch of the conversion gives the rotation back."""
        rng = np.random.default_rng(0)
        from scipy.spatial.transform import Rotation

        cases = [Rotation.from_quat(rng.normal(size=4)).as_matrix() for _ in range(40)]
        cases += [Rotation.from_euler("xyz", angles).as_matrix() for angles in ([np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi], [np.pi / 2, 0, 0])]
        matrices = np.stack(cases)
        quaternion = matrix_to_xyzw(matrices)
        np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(), matrices, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(quaternion, axis=-1), 1.0)

    def test_rotate_inverse_matches_the_matrix(self) -> None:
        """Rotating into a frame equals applying the transposed matrix."""
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_euler("xyz", [0.4, -0.7, 1.1])
        quaternion = torch.tensor(rotation.as_quat(), dtype=torch.float64)
        vector = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
        np.testing.assert_allclose(rotate_inverse_xyzw(quaternion, vector).numpy(), rotation.as_matrix().T @ vector.numpy(), atol=1e-12)


class AdapterTest(unittest.TestCase):
    """Bookkeeping between the graph and the simulator."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_reset_writes_a_batch_in_simulator_order(self) -> None:
        """Poses, velocities and joints go to the right worlds, frames and joint slots."""
        env, articulation, sensor, refreshes, robot = build(count=3)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        env.reset(np.array([2, 0]), [frame(robot, rotation), frame(robot)])
        self.assertEqual(len(articulation.poses), 1)
        pose, ids = articulation.poses[0]
        self.assertEqual(ids.tolist(), [2, 0])
        np.testing.assert_allclose(pose[0, :2].numpy(), [10.0, 1.0])
        self.assertGreater(float(pose[0, 2]), 0.0)
        np.testing.assert_allclose(pose[0, 3:].numpy(), [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
        velocity = articulation.velocities[0][0]
        np.testing.assert_allclose(velocity[0].numpy(), [0.0, 1.0, 0.0, 0.0, 0.0, 2.0], atol=1e-6)
        position = articulation.joint_writes[0][0]
        names = articulation.joint_names
        for index, name in enumerate(robot.graph.joint_names):
            self.assertAlmostEqual(float(position[0, names.index(name)]), 0.1 * index, places=5)
        self.assertAlmostEqual(float(position[0, names.index("frozen_joint")]), 0.25)
        self.assertEqual(float(articulation.joint_velocity_writes[0][0, names.index("frozen_joint")]), 0.0)
        self.assertEqual(sensor.calls, ["reset"])
        self.assertEqual(len(refreshes), 1)

    def test_start_heights(self) -> None:
        """A standing start rests the feet on the floor; a lying start clears every frame."""
        env, _, _, _, robot = build()
        q = np.zeros((1, env.joints))
        height = env._start_heights(q, np.eye(3)[None], np.array([False]))[0]
        feet = robot.graph.link_poses(torch.zeros(1, env.joints, dtype=torch.float64), FEET)[..., :3, 3]
        self.assertAlmostEqual(height + float(feet[..., 2].min()), 0.03)
        turned = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        lying = env._start_heights(q, turned[None], np.array([True]))[0]
        frames = robot.graph.forward_kinematics(torch.zeros(1, env.joints, dtype=torch.float64))[..., :3, 3].numpy()
        lowest = (turned @ frames[0].T).T[:, 2].min()
        self.assertGreaterEqual(lying + lowest, 0.0)

    def test_targets_use_simulator_joints_and_hold_the_others(self) -> None:
        """Actions become PD targets on the right joints; joints outside the graph hold their default."""
        env, articulation, _, _, robot = build()
        env.prepare(np.ones((2, env.joints)) * 0.1)
        value, ids = articulation.targets[-1]
        self.assertIsNone(ids)
        names = articulation.joint_names
        self.assertAlmostEqual(float(value[0, names.index("frozen_joint")]), 0.25)
        index = robot.graph.joint_names.index("FL_thigh_joint")
        expected = float(np.clip(env.nominal[index] + 0.125, env.lower[index], env.upper[index]))
        self.assertAlmostEqual(float(value[0, names.index("FL_thigh_joint")]), expected, places=5)

    def test_state_reads_simulator_order_and_frames(self) -> None:
        """Joint signals return in graph order, and height comes from the root position."""
        env, articulation, _, _, robot = build()
        names = articulation.joint_names
        for index, name in enumerate(robot.graph.joint_names):
            articulation.state.joint_pos[:, names.index(name)] = float(index)
            articulation.state.joint_vel[:, names.index(name)] = 2.0 * index
            articulation.state.applied_torque[:, names.index(name)] = 3.0 * index
        articulation.state.root_link_pos_w[:, 2] = 0.7
        articulation.state.root_link_lin_vel_b[:, 0] = 1.5
        state = env.read_state()
        expected = torch.arange(env.joints, dtype=torch.float64)
        torch.testing.assert_close(state.q[0], expected)
        torch.testing.assert_close(state.qd[0], 2 * expected)
        torch.testing.assert_close(env._torque()[0], 3 * expected)
        self.assertAlmostEqual(float(state.height[0]), 0.7, places=5)
        self.assertEqual(float(state.linear_velocity[0, 0]), 1.5)

    def test_contact_penalties(self) -> None:
        """Slip counts feet in contact, undesired contact counts other links, and feet tilt is BFM-Zero's."""
        env, articulation, sensor, _, _ = build()
        sensor.forces[0, sensor.body_names.index("FL_foot"), 2] = 10.0
        articulation.state.body_link_lin_vel_w[0, articulation.body_names.index("FL_foot"), 0] = 3.0
        articulation.state.body_link_lin_vel_w[1, articulation.body_names.index("FL_foot"), 0] = 3.0
        np.testing.assert_allclose(env._slippage(), [3.0, 0.0])
        self.assertEqual(env._undesired_contact().tolist(), [False, False])
        sensor.forces[1, sensor.body_names.index("base"), 0] = 5.0
        self.assertEqual(env._undesired_contact().tolist(), [False, True])
        angle = 0.3
        articulation.state.body_link_quat_w[0, articulation.body_names.index("FL_foot")] = torch.tensor([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)])
        np.testing.assert_allclose(env._feet_orientation(), [np.sin(angle), 0.0], atol=1e-6)

    def test_undesired_links_follow_the_config(self) -> None:
        """Regexes choose the penalized links, as BFM-Zero's pelvis, shoulders and hips."""
        robot = record()
        articulation = FakeArticulation(1, robot)
        sensor = FakeSensor(articulation)
        env = IsaacEmbodimentEnv(robot, EnvConfig(num_envs=1, randomization=DomainRandomization(enabled=False)), articulation, sensor,
                                 torch.zeros(1, 3), undesired_contact_links=["FL_.*"])
        sensor.forces[0, sensor.body_names.index("base"), 0] = 5.0
        self.assertEqual(env._undesired_contact().tolist(), [False])
        sensor.forces[0, sensor.body_names.index("FL_thigh"), 0] = 5.0
        self.assertEqual(env._undesired_contact().tolist(), [True])

    def test_missing_names_are_reported(self) -> None:
        """Merged feet or missing joints give a clear error instead of wrong data."""
        robot = record()
        articulation = FakeArticulation(1, robot)
        articulation.body_names = ["base"]
        with self.assertRaisesRegex(ValueError, "contact links"):
            IsaacEmbodimentEnv(robot, EnvConfig(num_envs=1, randomization=DomainRandomization(enabled=False)), articulation, FakeSensor(FakeArticulation(1, robot)), torch.zeros(1, 3))
        articulation = FakeArticulation(1, robot)
        articulation.joint_names = articulation.joint_names[:-1]
        with self.assertRaisesRegex(ValueError, "joints missing"):
            IsaacEmbodimentEnv(robot, EnvConfig(num_envs=1, randomization=DomainRandomization(enabled=False)), articulation, FakeSensor(FakeArticulation(1, robot)), torch.zeros(1, 3))

    def test_randomization_writes_every_world(self) -> None:
        """Mass, inertia, base center of mass and friction are written per world, within range."""
        _, articulation, _, _, _ = build(count=3, randomize=True)
        scale = articulation.masses
        self.assertTrue(bool(((scale >= 0.95) & (scale <= 1.05)).all()))
        self.assertGreater(float((scale[0] - scale[1]).abs().max()), 0)
        torch.testing.assert_close(articulation.inertias[..., 0], scale)
        self.assertLessEqual(float(articulation.coms[:, 0, :3].abs().max()), 0.02)
        self.assertEqual(float(articulation.coms[:, 1:].abs().max()), 0.0)
        friction = articulation.materials[:, 0, 0]
        self.assertTrue(bool(((friction >= 0.5) & (friction <= 1.25)).all()))
        self.assertEqual(len(set(friction.tolist())), 3)
        self.assertEqual(articulation.material_ids.dtype, torch.int32)

    def test_pushes_are_written_as_one_batch_and_only_in_training(self) -> None:
        """Due worlds get one velocity write; evaluations get none."""
        env, articulation, _, _, _ = build(count=2, randomize=True)
        env.next_push[:] = 0.0
        env.evaluating = True
        env.prepare(np.zeros((2, env.joints)))
        self.assertEqual(articulation.velocities, [])
        env.evaluating = False
        env.prepare(np.zeros((2, env.joints)))
        self.assertEqual(len(articulation.velocities), 1)
        velocity, ids = articulation.velocities[0]
        self.assertEqual(ids.tolist(), [0, 1])
        self.assertLessEqual(float(velocity[:, :2].abs().max()), 0.5)
        self.assertEqual(float(velocity[:, 2].abs().max()), 0.0)

    def test_diverged_worlds_are_reported_and_reset(self) -> None:
        """A non-finite world ends its episode and its state is rewritten."""
        env, articulation, _, _, _ = build()
        env.prepare(np.zeros((2, env.joints)))
        articulation.state.joint_vel[1, 0] = float("nan")
        self.assertEqual(env._diverged().tolist(), [False, True])
        articulation.state.joint_vel[1, 0] = 0.0
        articulation.state.joint_pos[1, 0] = float("inf")
        result = env.collect()
        self.assertEqual(result.terminated.tolist(), [False, True])
        self.assertEqual(articulation.joint_writes[-1][1].tolist(), [1])


class GroupTest(unittest.TestCase):
    """One simulation stepping several robots."""

    def test_advance_order(self) -> None:
        """Every physics step writes all robots, steps once without rendering, and updates all."""
        first, first_articulation, first_sensor, _, _ = build()
        second, second_articulation, _, _, _ = build()
        steps = []
        simulation = SimpleNamespace(get_physics_dt=lambda: 0.005, step=lambda render=True: steps.append(render), forward=lambda: steps.append("forward"))
        group = IsaacGroup(simulation, [first, second], 4)
        first_articulation.calls.clear()
        group.advance()
        self.assertEqual(steps, [False] * 4)
        self.assertEqual(first_articulation.calls, ["write", "update:0.005"] * 4)
        self.assertEqual(second_articulation.calls[-2:], ["write", "update:0.005"])
        self.assertEqual(first_sensor.calls[-1], "update:0.005")
        group.refresh()
        self.assertEqual(steps[-1], "forward")
        self.assertEqual(first_articulation.calls[-1], "update:0.0")

    def test_rollout_runs_on_the_adapter(self) -> None:
        """The training loop's rollout builds observations from the adapter's state."""
        from model.rl.trainer import ObservationNoise, Rollout

        env, _, _, _, robot = build()
        trajectory = Trajectory(embodiment=0, dt=0.02, positions=torch.zeros(10, env.joints, dtype=torch.float64), velocities=torch.zeros(10, env.joints, dtype=torch.float64))
        observer = EmbodimentObserver(robot, 2)
        rollout = Rollout(env, observer, ObservationNoise(), torch.Generator().manual_seed(0))
        from model.rl.buffers import ExpertMotions

        env.set_motions(ExpertMotions(observer, [trajectory]).frames)
        rollout.start()
        self.assertEqual(rollout.observation.local.shape, (2, 2, env.joints, 5))
        env.prepare(np.zeros((2, env.joints)))
        next_observation, step = rollout.finish()
        self.assertEqual(next_observation.mass.shape, (2, env.joints, env.joints))
        rollout.reset_done(torch.tensor([True, False]), next_observation)
        self.assertEqual(rollout.observation.state.local.shape, (2, env.joints, 5))
        self.assertTrue(all(key in step.aux_rewards for key in ("penalty_undesired_contact", "penalty_feet_ori")))


if __name__ == "__main__":
    unittest.main()
