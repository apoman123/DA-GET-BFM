"""Test the MuJoCo ground-contact simulator (skipped without ``mujoco``)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import EmbodimentGraph, JointFeatureConfig, actuator_specs
from model.dynamics import BaseMotion, floating_mass_matrix, recursive_newton_euler
from model.dynamics import constrained_response
from model.training.episodes import TORQUE_SOURCES, EmbodimentRecord, process_episode
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
        armature = torch.linspace(1e-4, 1e-2, graph.num_joints, dtype=torch.float64)
        graph = graph.with_armature(armature)
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
        # Our mass matrix includes each joint's rotor inertia, like MuJoCo's armature.
        ours = floating_mass_matrix(graph, torch.tensor(q)[None])[0].numpy()
        np.testing.assert_allclose(full[np.ix_(order, order)], ours, atol=1e-7)
        np.testing.assert_allclose(np.diag(ours)[6:] - np.diag(floating_mass_matrix(graph.with_armature(None), torch.tensor(q)[None])[0].numpy())[6:], armature.numpy(), atol=1e-9)
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
        self.assertGreater(float(trajectory.contacts.float().mean()), 0.7)
        self.assertTrue(trajectory.exact_dynamics)
        episode = process_episode(records[0], trajectory)
        torch.testing.assert_close(episode["required_torque"], episode["applied_torque"])
        self.assertEqual(episode["torque_source"].unique().tolist(), [TORQUE_SOURCES.index("exact")])
        # Simulator contacts are trusted: no residual check, and no contact detection.
        self.assertEqual(float(episode["contact_residual"].abs().max()), 0.0)
        self.assertFalse(bool(episode["contacts_detected"].any()))
        self.assertTrue(bool(episode["base_acceleration_known"].all()))
        self.assertEqual(episode["dynamics_features"].shape[-1], 5)
        # Response labels are dropped only around contact changes.
        changes = (trajectory.contacts[1:] != trajectory.contacts[:-1]).any(-1)
        if not bool(changes.any()):
            self.assertTrue(bool(episode["response_valid"].all()))
        self.assertTrue(bool(episode["response_valid"].any()))

    def test_constrained_response_matches_mujoco_with_stiff_contacts(self) -> None:
        """G(q) dtau equals MuJoCo's acceleration change once its contacts are made stiff.

        MuJoCo's default contacts are soft (solimp 0.9-0.95), which lets the feet
        accelerate slightly and changes the response by about 20%; with stiff contacts
        the rigid-contact model and MuJoCo agree to a few percent.
        """
        import mujoco

        from model.training.ground_sim import ContactGeometry, build_mjcf, ground_specs

        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True).with_armature(1e-3)
        record = EmbodimentRecord.build("quad", graph, ground_specs(actuator_specs(graph)), contact_links=FEET)
        zeros = np.zeros(graph.num_joints)
        model = mujoco.MjModel.from_xml_string(build_mjcf(record, zeros, zeros, ContactGeometry()))
        data = mujoco.MjData(model)
        names = graph.joint_names
        position = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        velocity = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        stand = np.array([STAND.get(name, 0.0) for name in names])
        data.qpos[position], data.qpos[3:7] = stand, [1, 0, 0, 0]
        mujoco.mj_kinematics(model, data)
        feet = [g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("foot")]
        data.qpos[2] -= min(data.geom_xpos[g][2] - model.geom_size[g, 0] for g in feet) + 5e-4
        for _ in range(2000):  # settle on all four feet under a joint-space PD
            data.qfrc_applied[velocity] = 50 * (stand - data.qpos[position]) - 2 * data.qvel[velocity]
            mujoco.mj_step(model, data)
        data.qvel[:] = 0
        model.geom_solimp[:, :2] = (0.9999, 0.99999)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 4)
        before = data.qacc.copy()
        q = torch.tensor(data.qpos[position])[None]
        response = constrained_response(graph, q, FEET, torch.ones(1, 4, dtype=torch.bool), False)[0].numpy()
        free = constrained_response(graph, q)[0].numpy()
        rng = np.random.default_rng(0)
        errors, free_errors = [], []
        for _ in range(10):
            change = 0.1 * rng.normal(size=graph.num_joints)
            data.qfrc_applied[velocity] += change
            mujoco.mj_forward(model, data)
            simulated = data.qacc - before
            data.qfrc_applied[velocity] -= change
            ours = response @ change
            errors.append(np.linalg.norm(ours[6:] - simulated[velocity]) / np.linalg.norm(simulated[velocity]))
            free_errors.append(np.linalg.norm(free[6:] @ change - simulated[velocity]) / np.linalg.norm(simulated[velocity]))
        self.assertLess(float(np.median(errors)), 0.05)
        self.assertGreater(float(np.median(free_errors)), 1.0)  # ignoring the contacts is badly wrong

    def test_estimate_error_is_internal_contact_forces(self) -> None:
        """With exact dynamics, estimated torques differ from MuJoCo's only by internal forces."""
        from model.dynamics import determinate_torque_projector, estimate_joint_efforts
        from model.training.sources import mujoco_source

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "quad.urdf")
            path.write_text(quadruped())
            source = {"urdf": str(path), "contact_links": FEET, "nominal_pose": STAND, "randomized_embodiments": 0,
                      "episodes_per_embodiment": 1, "steps": 100, "probe_torques": 0, "gain_scale": 3.0}
            records, trajectories = mujoco_source(source, None, JointFeatureConfig())
        record, trajectory = records[0], trajectories[0]
        world = torch.tensor([0, 0, -9.80665], dtype=torch.float64)
        gravity = (trajectory.base_orientation.transpose(-1, -2) @ world[:, None]).squeeze(-1)
        motion = BaseMotion(trajectory.base_linear_velocity, trajectory.base_angular_velocity,
                            trajectory.base_linear_acceleration, trajectory.base_angular_acceleration)
        estimate = estimate_joint_efforts(record.graph, trajectory.positions, trajectory.velocities, trajectory.accelerations,
                                          gravity, motion, FEET, trajectory.contacts, record.actuator,
                                          base_external_wrench=trajectory.base_external_wrench)
        error = estimate.efforts - trajectory.applied_torque
        projector, _ = determinate_torque_projector(record.graph, trajectory.positions, FEET, trajectory.contacts, False)
        determined = (projector @ error[..., None]).squeeze(-1)
        scale = trajectory.applied_torque.pow(2).mean().sqrt()
        self.assertLess(float(determined.pow(2).mean().sqrt() / scale), 0.1)
        self.assertGreater(float(error.pow(2).mean().sqrt()), 2 * float(determined.pow(2).mean().sqrt()))

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
