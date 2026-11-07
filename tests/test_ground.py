"""Test the MuJoCo ground-contact simulator (skipped without ``mujoco``)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import EmbodimentGraph, JointFeatureConfig, actuator_specs
from model.dynamics import BaseMotion, floating_mass_matrix, recursive_newton_euler
from model.training.episodes import EmbodimentRecord, process_episode
from tests.test_embodiment import humanoid, quadruped

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
# The test robots have weak 10 N m actuators, so they stand fairly straight with stiff gains.
STAND = {f"{leg}_{part}_joint": value for leg in ("FL", "FR", "RL", "RR") for part, value in (("thigh", 0.3), ("calf", -0.6))}


@unittest.skipUnless(HAS_MUJOCO, "needs mujoco (uv sync --extra sim)")
class GroundSimulationTest(unittest.TestCase):
    """The generated MJCF matches our dynamics, and ground episodes have exact labels."""

    def test_mjcf_dynamics_match_ours(self) -> None:
        """MuJoCo's mass matrix and bias forces equal the floating-base RNEA."""
        import mujoco

        from model.training.ground_sim import ContactGeometry, build_mjcf, ground_specs

        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True)
        record = EmbodimentRecord.build("quad", graph, ground_specs(actuator_specs(graph)), contact_links=FEET)
        zeros = np.zeros(graph.num_joints)
        model = mujoco.MjModel.from_xml_string(build_mjcf(record, zeros, zeros, ContactGeometry()))
        data = mujoco.MjData(model)
        names = graph.joint_names
        position = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        velocity = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        rng = np.random.default_rng(0)
        q, qd = rng.uniform(-0.5, 0.5, graph.num_joints), rng.normal(size=graph.num_joints)
        v, w = rng.normal(size=3), rng.normal(size=3)
        data.qpos[2], data.qpos[3:7], data.qpos[position] = 1.0, [1, 0, 0, 0], q
        data.qvel[0:3], data.qvel[3:6], data.qvel[velocity] = v, w, qd
        mujoco.mj_forward(model, data)
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, full, data.qM) if hasattr(data, "qM") else mujoco.mj_fullM(model, data, full)
        order = [0, 1, 2, 3, 4, 5, *velocity]
        ours = floating_mass_matrix(graph, torch.tensor(q)[None])[0].numpy()
        armature = np.zeros_like(ours)
        armature[6:, 6:] = 1e-5 * np.eye(graph.num_joints)
        np.testing.assert_allclose(full[np.ix_(order, order)], ours + armature, atol=1e-7)
        result = recursive_newton_euler(
            graph, torch.tensor(q)[None], torch.tensor(qd)[None], torch.zeros(1, graph.num_joints, dtype=torch.float64),
            torch.tensor([0, 0, -9.80665], dtype=torch.float64), BaseMotion(torch.tensor(v)[None], torch.tensor(w)[None]),
        )
        np.testing.assert_allclose(data.qfrc_bias[velocity], result.efforts[0].numpy(), atol=1e-7)
        np.testing.assert_allclose(data.qfrc_bias[:6], result.base_wrench[0].numpy(), atol=1e-7)

    def test_quadruped_stands_with_exact_labels(self) -> None:
        """A quadruped stands on its feet; labels are exact and contacts recorded."""
        from model.training.sources import mujoco_source

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "quad.urdf")
            path.write_text(quadruped())
            source = {"urdf": str(path), "contact_links": FEET, "nominal_pose": STAND, "randomized_embodiments": 1,
                      "episodes_per_embodiment": 2, "steps": 100, "probe_torques": 1, "push_probability": 0.0, "gain_scale": 3.0}
            records, trajectories = mujoco_source(source, None, JointFeatureConfig())
        self.assertEqual(len(trajectories), 2)
        trajectory = trajectories[0]
        self.assertEqual(trajectory.positions.shape[0], 100)
        self.assertGreater(float(trajectory.contacts.float().mean()), 0.8)
        self.assertTrue(trajectory.exact_dynamics)
        episode = process_episode(records[0], trajectory)
        torch.testing.assert_close(episode["required_torque"], episode["applied_torque"])
        self.assertEqual(float(episode["estimate_residual"].abs().max()), 0.0)
        self.assertTrue(bool(episode["base_acceleration_known"].all()))
        self.assertEqual(float(episode["dynamics_features"][5, 0, 6]), 1.0)

    def test_biped_is_steadied_by_the_spotter(self) -> None:
        """A biped stays up thanks to a recorded, small spotter wrench."""
        from model.training.sources import mujoco_source

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "biped.urdf")
            path.write_text(humanoid())
            pose = {"left_hip_pitch": -0.05, "right_hip_pitch": -0.05, "left_knee": 0.1, "right_knee": 0.1,
                    "left_ankle_pitch": -0.05, "right_ankle_pitch": -0.05}
            source = {"urdf": str(path), "contact_links": ["left_ankle", "right_ankle"], "nominal_pose": pose,
                      "contact_geometry": {"radius": 0.01, "foot_size": [0.16, 0.08]},
                      "randomized_embodiments": 0, "episodes_per_embodiment": 1, "steps": 150, "probe_torques": 0,
                      "gain_scale": 3.0}
            records, trajectories = mujoco_source(source, None, JointFeatureConfig())
        trajectory = trajectories[0]
        self.assertEqual(trajectory.positions.shape[0], 150)
        weight = records[0].graph.total_mass * 9.80665
        self.assertLess(float(trajectory.base_external_wrench[:, :3].norm(dim=-1).mean()), 0.3 * weight)


if __name__ == "__main__":
    unittest.main()
