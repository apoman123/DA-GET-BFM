"""Test floating-base dynamics, torque estimation, multirotors, and their data pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from model import EmbodimentGraph, JointFeatureConfig, actuator_specs
from model.dynamics import (
    BaseMotion,
    contact_maps,
    determinate_torque_projector,
    distribute_contact_forces,
    estimate_joint_efforts,
    floating_forward_dynamics,
    floating_mass_matrix,
    friction_efforts,
    recursive_newton_euler,
    rotor_wrenches,
)
from model.physics import PhysicsFeatureCache, actuator_tensors
from model.training import build_physics_datasets
from model.training.episodes import EmbodimentRecord, Trajectory, process_episode
from model.training.sources import synthetic_source
from tests.test_embodiment import humanoid, joint, link, quadruped, robot

GRAVITY = 9.80665
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
ROTOR = {
    "rotor_*": {
        "mode": "thrust",
        "thrust_coefficient": 8.5e-6,
        "torque_coefficient": 1.4e-7,
        "kd": 1e-3,
        "peak_torque": 0.25,
        "continuous_torque": 0.2,
        "max_velocity": 1100,
        "max_power": 200,
    }
}


def quadrotor() -> str:
    """A 1 kg X quadrotor with four continuous rotor joints about +z."""
    elements = [link("body", 0.92, inertia=0.006)]
    for name, x, y in (("fl", 0.17, 0.17), ("bl", -0.17, 0.17), ("br", -0.17, -0.17), ("fr", 0.17, -0.17)):
        elements.append(link(f"rotor_{name}", 0.02, inertia=2e-5))
        elements.append(
            joint(f"rotor_{name}_joint", "body", f"rotor_{name}", "continuous", xyz=f"{x} {y} 0.03", axis="0 0 1",
                  extra='<limit effort="0.25" velocity="1100"/>')
        )
    return robot("quad", *elements)


def random_state(graph: EmbodimentGraph, batch: int = 3, seed: int = 0):
    """Random joint positions, velocities, accelerations, and a base twist."""
    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: torch.randn(shape, generator=generator, dtype=torch.float64)  # noqa: E731
    joints = graph.num_joints
    return draw(batch, joints) * 0.3, draw(batch, joints), draw(batch, joints), draw(batch, 3), draw(batch, 3)


class FloatingDynamicsTest(unittest.TestCase):
    """Check the floating-base equations of motion."""

    def test_mass_matrix_structure(self) -> None:
        """M_full is symmetric positive definite, holds the total mass, and contains M(q)."""
        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True)
        q = random_state(graph)[0]
        full = floating_mass_matrix(graph, q)
        torch.testing.assert_close(full, full.transpose(-1, -2))
        self.assertGreater(float(torch.linalg.eigvalsh(full).min()), 0)
        torch.testing.assert_close(full[..., 6:, 6:], graph.mass_matrix(q))
        eye = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
        torch.testing.assert_close(full[..., :3, :3], graph.total_mass * eye)

    def test_forward_dynamics_matches_rnea(self) -> None:
        """Free-floating FD accelerations give zero base wrench and the applied torques."""
        graph = EmbodimentGraph.from_urdf(humanoid(), floating_base=True)
        q, qd, tau, v, w = random_state(graph, seed=1)
        gravity = torch.tensor([0.5, -1.0, -9.7], dtype=torch.float64)
        a, alpha, qdd = floating_forward_dynamics(graph, q, qd, tau, gravity, v, w)
        result = recursive_newton_euler(graph, q, qd, qdd, gravity, BaseMotion(v, w, a, alpha))
        torch.testing.assert_close(result.base_wrench, torch.zeros_like(result.base_wrench), atol=1e-9, rtol=0)
        torch.testing.assert_close(result.efforts + friction_efforts(graph, qd), tau)

    def test_free_fall_and_still_base(self) -> None:
        """With no torque and no contact, a still robot falls at g; a fixed base gives zero base acceleration terms."""
        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True)
        q = torch.zeros(1, graph.num_joints, dtype=torch.float64)
        a, alpha, qdd = floating_forward_dynamics(graph, q, q, q, None)
        torch.testing.assert_close(a, torch.tensor([[0.0, 0.0, -GRAVITY]], dtype=torch.float64))
        torch.testing.assert_close(alpha, torch.zeros_like(alpha), atol=1e-9, rtol=0)
        torch.testing.assert_close(qdd, torch.zeros_like(qdd), atol=1e-9, rtol=0)


class TorqueEstimationTest(unittest.TestCase):
    """Check contact-force distribution and joint torque estimation."""

    def test_standing_quadruped_carries_its_weight(self) -> None:
        """Four feet share the weight with vertical forces; joint torques hold the stance."""
        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True)
        q = torch.zeros(1, graph.num_joints, dtype=torch.float64)
        estimate = estimate_joint_efforts(graph, q, q, q, contact_links=FEET)
        total = estimate.contact_forces.sum(-2)[0]
        torch.testing.assert_close(total, torch.tensor([0, 0, graph.total_mass * GRAVITY], dtype=torch.float64), atol=1e-3, rtol=1e-4)
        self.assertLess(float(estimate.contact_forces[0, :, :2].abs().max()), 1e-3)
        self.assertLess(float(estimate.residual_wrench.abs().max()), 1e-2)
        fixed = estimate_joint_efforts(EmbodimentGraph.from_urdf(quadruped()), q, q, q)
        self.assertGreater(float((estimate.efforts - fixed.efforts).abs().max()), 0.1)

    def test_biped_needs_surface_contacts(self) -> None:
        """Two point feet cannot balance every moment; flat feet (surface contacts) can."""
        graph = EmbodimentGraph.from_urdf(humanoid(), floating_base=True)
        q = 0.2 * torch.ones(1, graph.num_joints, dtype=torch.float64)
        feet = ["left_ankle", "right_ankle"]
        point = estimate_joint_efforts(graph, q, q * 0, q * 0, contact_links=feet, surface_contacts=False)
        surface = estimate_joint_efforts(graph, q, q * 0, q * 0, contact_links=feet)
        self.assertLess(float(surface.residual_wrench.abs().max()), 1e-2)
        self.assertGreater(float(point.residual_wrench.abs().max()), float(surface.residual_wrench.abs().max()))

    def test_pulling_contacts_are_dropped(self) -> None:
        """A contact that would pull on the ground is released by the active set."""
        points = torch.tensor([[[0.3, 0.0, -0.4], [-0.3, 0.0, -0.4], [2.0, 0.0, -0.4]]], dtype=torch.float64)
        wrench = torch.tensor([[0.0, 0.0, 100.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        active = torch.ones(1, 3, dtype=torch.bool)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        forces, _ = distribute_contact_forces(wrench, points, active, normal=normal)
        self.assertTrue(bool((forces[..., 2] >= -1e-6).all()))
        torch.testing.assert_close(forces.sum(-2)[0, 2], torch.tensor(100.0, dtype=torch.float64), atol=1e-2, rtol=0)

    def test_projector_removes_internal_contact_forces(self) -> None:
        """Internal forces (no net base wrench) change torques only in projected-out directions."""
        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True)
        q = 0.3 * torch.randn(4, graph.num_joints, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        for active, expected in ((torch.ones(4, 4, dtype=torch.bool), 6), (torch.tensor([[True, False, False, True]] * 4), 1), (torch.zeros(4, 4, dtype=torch.bool), 0)):
            projector, undetermined = determinate_torque_projector(graph, q, FEET, active, False)
            self.assertTrue(bool((undetermined == expected).all()))
            matrix_a, matrix_b = contact_maps(graph, q, FEET, active, False)
            internal = torch.linalg.svd(matrix_a, full_matrices=True).Vh.transpose(-1, -2)[..., :, 6:]
            torch.testing.assert_close(projector @ matrix_b @ internal, torch.zeros_like(matrix_b @ internal), atol=1e-10, rtol=0)
            torch.testing.assert_close(projector @ projector, projector, atol=1e-10, rtol=0)

    def test_hover_thrust_balances_weight(self) -> None:
        """Rotors at hover speed lift the weight; RNEA sees no base wrench left."""
        graph = EmbodimentGraph.from_urdf(quadrotor(), floating_base=True)
        actuator = actuator_tensors(actuator_specs(graph, ROTOR))
        speed = (graph.total_mass * GRAVITY / (4 * 8.5e-6)) ** 0.5
        q = torch.zeros(1, 4, dtype=torch.float64)
        qd = torch.tensor([[speed, -speed, speed, -speed]], dtype=torch.float64)
        thrust = rotor_wrenches(graph, actuator, q, qd)
        torch.testing.assert_close(thrust[..., 2].sum(), torch.tensor(graph.total_mass * GRAVITY, dtype=torch.float64))
        estimate = estimate_joint_efforts(graph, q, qd, q, actuator=actuator)
        self.assertLess(float(estimate.residual_wrench.abs().max()), 1e-9)
        torch.testing.assert_close(estimate.efforts.abs(), torch.full((1, 4), 1.4e-7 * speed**2, dtype=torch.float64))


class FloatingSimulationTest(unittest.TestCase):
    """Synthetic floating and flight data are stable and their torques are recoverable."""

    def check_recovery(self, source: dict) -> tuple[EmbodimentRecord, Trajectory]:
        records, trajectories = synthetic_source(source, None, JointFeatureConfig())
        record, trajectory = records[0], trajectories[0]
        world = torch.tensor([0, 0, -GRAVITY], dtype=torch.float64)
        gravity = (trajectory.base_orientation.transpose(-1, -2) @ world[:, None]).squeeze(-1)
        estimate = estimate_joint_efforts(
            record.graph,
            trajectory.positions,
            trajectory.velocities,
            trajectory.accelerations,
            gravity,
            BaseMotion(
                trajectory.base_linear_velocity,
                trajectory.base_angular_velocity,
                trajectory.base_linear_acceleration,
                trajectory.base_angular_acceleration,
            ),
            actuator=record.actuator,
            base_external_wrench=trajectory.base_external_wrench,
        )
        torch.testing.assert_close(estimate.efforts, trajectory.applied_torque, atol=1e-8, rtol=1e-8)
        return record, trajectory

    def test_floating_quadruped_on_harness(self) -> None:
        """A harnessed floating quadruped stays bounded; ID recovers the torques."""
        source = {"urdf": "q.urdf", "floating_base": True, "base_modes": ["floating"], "randomized_embodiments": 1,
                  "episodes_per_embodiment": 2, "steps": 30, "probe_torques": 1, "contact_links": FEET}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "q.urdf").write_text(quadruped())
            source["urdf"] = str(Path(folder, "q.urdf"))
            record, trajectory = self.check_recovery(source)
        self.assertTrue(record.graph.floating_base)
        self.assertLess(float(trajectory.base_angular_velocity.norm(dim=-1).max()), 10.0)

    def test_quadrotor_flight(self) -> None:
        """The flight controller keeps the quadrotor airborne near its waypoints."""
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "quad.urdf").write_text(quadrotor())
            source = {"urdf": str(Path(folder, "quad.urdf")), "floating_base": True, "actuator_overrides": ROTOR,
                      "randomized_embodiments": 1, "episodes_per_embodiment": 2, "steps": 60, "probe_torques": 1,
                      "randomization": {"damping": [0, 0], "friction": [0, 0]}}
            record, trajectory = self.check_recovery(source)
        self.assertTrue(record.name.endswith("flight"))
        self.assertLess(float(trajectory.base_position.abs().max()), 1.5)


class FloatingPipelineTest(unittest.TestCase):
    """Recorded floating-base data, contacts, and batches."""

    def test_recorded_standing_quadruped(self) -> None:
        """Kinematics-only data of a still quadruped yield weight-bearing torque estimates."""
        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True)
        record = EmbodimentRecord.build("quad", graph, actuator_specs(graph), contact_links=FEET)
        steps = 20
        q = torch.zeros(steps, graph.num_joints, dtype=torch.float64)
        trajectory = Trajectory(
            embodiment=0,
            dt=0.02,
            positions=q,
            base_orientation=torch.eye(3, dtype=torch.float64).expand(steps, 3, 3),
            base_position=torch.zeros(steps, 3, dtype=torch.float64),
        )
        episode = process_episode(record, trajectory)
        expected = estimate_joint_efforts(graph, q[:1], q[:1], q[:1], contact_links=FEET).efforts[0]
        torch.testing.assert_close(episode["applied_torque"][5, 0], expected.float(), atol=1e-3, rtol=1e-3)
        self.assertLess(float(episode["estimate_residual"].max()), 1e-2)
        self.assertEqual(float(episode["dynamics_features"][5, 0, 6]), 0.5)  # estimated torque flag
        self.assertEqual(float(episode["base_features"][0, 10]), 1.0)  # floating flag

    def test_feature_cache_uses_base_twist(self) -> None:
        """Base rotation changes the Coriolis features of a floating robot."""
        graph = EmbodimentGraph.from_urdf(quadruped(), floating_base=True)
        cache = PhysicsFeatureCache([graph], [actuator_specs(graph)])
        q = 0.2 * torch.ones(1, graph.num_joints)
        qd = torch.ones(1, graph.num_joints)
        still = cache.update(q, qd)
        spinning = cache.update(q, qd, base_angular_velocity=torch.tensor([[0.0, 0.0, 3.0]]))
        self.assertGreater(float((still[..., 1] - spinning[..., 1]).abs().max()), 1e-3)

    def test_hub_style_floating_source_and_batch(self) -> None:
        """A LeRobot-style floating dataset with split state columns loads and trains."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "quad.urdf").write_text(quadruped())
            graph = EmbodimentGraph.from_urdf(quadruped())
            names = list(graph.joint_names)
            rows = {"episode_index": [], "frame_index": [], "front": [], "rear": [], "pose": [], "gyro": []}
            for episode in range(2):
                for frame in range(30):
                    angle = 0.02 * np.sin(frame / 5 + episode)
                    state = [angle * (k % 3 == 1) for k in range(12)]
                    rows["episode_index"].append(episode)
                    rows["frame_index"].append(frame)
                    rows["front"].append(state[:6])
                    rows["rear"].append(state[6:])
                    rows["pose"].append([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
                    rows["gyro"].append([0.0, 0.0, 0.0])
            target = root / "data" / "chunk-000"
            target.mkdir(parents=True)
            Dataset.from_dict(rows).to_parquet(str(target / "file-000.parquet"))
            config = {
                "history_size": 2,
                "eval_fraction": 0.0,
                "sources": [
                    {
                        "type": "huggingface",
                        "path": str(root),
                        "urdf": "quad.urdf",
                        "floating_base": True,
                        "joint_names": names,
                        "state_key": ["front", "rear"],
                        "base_position_key": "pose",
                        "base_position_indices": [0, 1, 2],
                        "base_orientation_key": "pose",
                        "base_orientation_indices": [3, 4, 5, 6],
                        "base_orientation_format": "xyzw",
                        "base_angular_velocity_key": "gyro",
                        "contact_links": FEET,
                        "fps": 50,
                    },
                    {
                        "type": "synthetic",
                        "urdf": "quad.urdf",
                        "floating_base": True,
                        "base_modes": ["floating"],
                        "contact_links": FEET,
                        "randomized_embodiments": 1,
                        "episodes_per_embodiment": 1,
                        "steps": 10,
                        "probe_torques": 1,
                    },
                ],
            }
            (root / "data.json").write_text(json.dumps(config))
            datasets = build_physics_datasets(root / "data.json", cache_dir=root / "cache")
        recorded = next(episode for index, episode in datasets.train.episodes if index == 0)
        self.assertTrue(bool(recorded["base_acceleration_known"].all()))
        self.assertLess(float(recorded["estimate_residual"].median()), 1.0)
        collator = datasets.collator()
        batch = collator([datasets.train[index] for index in range(0, len(datasets.train), 7)][:8])
        self.assertTrue(bool(batch["fd_base_mask"].any()))
        self.assertFalse(bool(batch["rbd_mask"].any()))  # floating-base samples never use task F
        recorded_rows = batch["torque_label_weight"] < 1
        self.assertTrue(bool(recorded_rows.any()))
        self.assertEqual(batch["input_values"].shape[-1], datasets.token_size)
        for name, value in batch.items():
            self.assertTrue(torch.isfinite(value.float()).all(), name)
        from model import GetConfig
        from model.pretraining import GetForPhysicsPretraining, GetPhysicsPretrainingConfig

        encoder = GetConfig(input_size=datasets.token_size, hidden_size=32, num_attention_heads=4,
                            num_hidden_layers=1, intermediate_size=64, use_physics_cross_attention=True)
        model = GetForPhysicsPretraining(GetPhysicsPretrainingConfig.from_encoder_config(encoder))
        output = model(**batch)
        self.assertIn("fd_base_acceleration", output.predictions)
        self.assertTrue(torch.isfinite(output.loss))
        # The ID loss ignores the part of the target that internal contact forces could change.
        projection = batch["id_projection"]
        ambiguous = torch.eye(projection.shape[-1]) - projection
        shift = (ambiguous @ torch.randn(projection.shape[:-1])[..., None]).squeeze(-1) / batch["torque_scale"]
        shifted = dict(batch, id_target=batch["id_target"] + shift)
        torch.manual_seed(0)
        model.eval()
        with torch.no_grad():
            first = model(**batch).loss_components["id"]
            second = model(**shifted).loss_components["id"]
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
