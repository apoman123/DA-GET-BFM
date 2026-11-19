"""Test keypoint retargeting between robots and its script (proposal G3)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from scipy.spatial.transform import Rotation

from model.conversion.common import Episode, write_dataset
from model.embodiment import EmbodimentGraph, JointFeatureConfig
from model.physics import actuator_specs
from model.rl.observation import trajectory_state
from model.rl.retarget import KeypointMap, KeypointPair, reference_pose, retarget, scale_ratio
from model.training.episodes import EmbodimentRecord, exp_rotation
from model.training.sources import SOURCE_TYPES, randomize_robot
from model.urdf import parse_urdf

# The robot URDFs in assets/urdf are downloaded, not committed (scripts/download_urdfs.py).
NEEDS_URDFS = unittest.skipUnless(
    all((Path(__file__).resolve().parents[1] / "assets" / "urdf" / name).exists() for name in ("go1.urdf", "h1.urdf", "anymal_d.urdf")),
    "needs the robot URDFs: uv run python scripts/download_urdfs.py",
)


ROOT = Path(__file__).resolve().parents[1]
GO1 = ROOT / "assets" / "urdf" / "go1.urdf"
ANYMAL = ROOT / "assets" / "urdf" / "anymal_d.urdf"
GO1_FEET = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
# ANYmal D's standing pose from configs/data/ground_contact.json.
ANYMAL_POSE = {"LF_HFE": 0.6, "RF_HFE": 0.6, "LH_HFE": -0.6, "RH_HFE": -0.6, "LF_KFE": -1.0, "RF_KFE": -1.0, "LH_KFE": 1.0, "RH_KFE": 1.0}
GO1_TO_ANYMAL = {
    "keypoints": [
        {"source": "trunk", "target": "base", "orientation_weight": 1.0},
        {"source": "FL_foot", "target": "LF_FOOT"},
        {"source": "FR_foot", "target": "RF_FOOT"},
        {"source": "RL_foot", "target": "LH_FOOT"},
        {"source": "RR_foot", "target": "RH_FOOT"},
    ]
}


def go1_motion(frames: int = 30, dt: float = 0.02) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a Go1 motion around its standing pose: joints, base position, base orientation."""
    time = torch.arange(frames, dtype=torch.float64) * dt
    nominal = torch.tensor([0.0, 0.9, -1.8] * 4, dtype=torch.float64)
    joints = nominal + 0.15 * torch.sin(2 * math.pi * 1.5 * time[:, None] + torch.arange(12.0, dtype=torch.float64))
    position = torch.stack((0.3 * time, 0.05 * torch.sin(3 * time), 0.28 + 0.02 * torch.sin(5 * time)), dim=-1)
    rotation = exp_rotation(torch.stack((0.08 * torch.sin(4 * time), 0.06 * torch.cos(3 * time), 0.2 * time), dim=-1))
    return joints, position, rotation


def self_map() -> KeypointMap:
    """Return Go1's trunk (with orientation) and feet mapped onto themselves."""
    return KeypointMap((KeypointPair("trunk", "trunk", orientation_weight=1.0),) + tuple(KeypointPair(foot, foot) for foot in GO1_FEET))


@NEEDS_URDFS
class RetargetTest(unittest.TestCase):
    """Retarget Go1 motions to Go1, a scaled Go1, and ANYmal D."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the robots and one motion."""
        cls.go1 = EmbodimentGraph.from_urdf(GO1, floating_base=True)
        cls.anymal = EmbodimentGraph.from_urdf(ANYMAL, floating_base=True)
        cls.dt = 0.02
        cls.joints, cls.position, cls.rotation = go1_motion(dt=cls.dt)

    def test_retargeting_a_robot_to_itself_reproduces_the_motion(self) -> None:
        """Go1 to Go1 reproduces the keypoints, joints and base pose, floating or fixed."""
        result = retarget(self.go1, self.joints, self.go1, self_map(), dt=self.dt,
                          base_position=self.position, base_orientation=self.rotation)
        self.assertEqual(result.scale, 1.0)
        self.assertEqual(result.dt, self.dt)
        self.assertLess(float(result.keypoint_error.max()), 1e-5)
        torch.testing.assert_close(result.positions, self.joints, atol=1e-4, rtol=0)
        torch.testing.assert_close(result.base_position, self.position, atol=1e-5, rtol=0)
        torch.testing.assert_close(result.base_orientation, self.rotation, atol=1e-5, rtol=0)
        fixed = EmbodimentGraph.from_urdf(GO1, floating_base=False)
        feet = KeypointMap.from_dict({foot: foot for foot in GO1_FEET})
        result = retarget(fixed, self.joints[:10], fixed, feet, dt=self.dt)
        self.assertIsNone(result.base_position)
        self.assertLess(float(result.keypoint_error.max()), 1e-5)
        torch.testing.assert_close(result.positions, self.joints[:10], atol=1e-4, rtol=0)

    def test_uniformly_scaled_copy_gets_the_same_joint_angles(self) -> None:
        """Scaling every length by k (as randomize_robot does) keeps the angles and scales the base path."""
        factor = 1.3
        identity = {"mass": [1.0, 1.0], "length": [factor, factor], "com_offset": 0.0, "damping": [0.0, 0.0], "friction": [0.0, 0.0]}
        scaled_robot = randomize_robot(parse_urdf(GO1), identity, torch.Generator().manual_seed(0))
        scaled = EmbodimentGraph.from_urdf(scaled_robot, floating_base=True)
        self.assertAlmostEqual(scale_ratio(self.go1, scaled, self_map()), factor, places=9)
        self.assertAlmostEqual(scale_ratio(self.go1, scaled, self_map(), "keypoints"), factor, places=9)
        result = retarget(self.go1, self.joints, scaled, self_map(), dt=self.dt,
                          base_position=self.position, base_orientation=self.rotation)
        self.assertAlmostEqual(result.scale, factor, places=9)
        torch.testing.assert_close(result.positions, self.joints, atol=1e-4, rtol=0)
        torch.testing.assert_close(result.base_position, factor * self.position, atol=1e-5, rtol=0)
        self.assertLess(float(result.keypoint_error.max()), 1e-5)

    def test_go1_to_anymal_keeps_scaled_feet_inside_joint_limits(self) -> None:
        """Feet follow the scaled Go1 feet within 1 mm, joints stay in their limits, and the motion reads as BFM states."""
        keypoints = KeypointMap.from_dict(GO1_TO_ANYMAL)
        result = retarget(self.go1, self.joints, self.anymal, keypoints, dt=self.dt,
                          base_position=self.position, base_orientation=self.rotation, initial_pose=ANYMAL_POSE)
        expected_scale = self.anymal.characteristic_length / self.go1.characteristic_length
        self.assertAlmostEqual(result.scale, expected_scale, places=9)
        # Feet from the source, scaled about the world origin.
        source_feet = self.go1.link_poses(self.joints, keypoints.source_links)[..., :3, 3]
        source_feet = self.position[:, None] + source_feet @ self.rotation.transpose(-1, -2)
        target_feet = self.anymal.link_poses(result.positions, keypoints.target_links)[..., :3, 3]
        target_feet = result.base_position[:, None] + target_feet @ result.base_orientation.transpose(-1, -2)
        error = (target_feet - result.scale * source_feet).norm(dim=-1)
        self.assertLess(float(error.max()), 1e-3)
        torch.testing.assert_close(error, result.keypoint_error, atol=1e-9, rtol=0)
        lower, upper = self.anymal.lower, self.anymal.upper
        self.assertTrue(bool(((result.positions >= lower) & (result.positions <= upper)).all()))
        torch.testing.assert_close(result.base_position[:, 2], result.scale * self.position[:, 2], atol=1e-5, rtol=0)
        record = EmbodimentRecord.build("anymal_d", self.anymal, actuator_specs(self.anymal))
        state, height_known, linear_known = trajectory_state(record, result.to_trajectory())
        self.assertEqual(tuple(state.q.shape), (self.joints.shape[0], self.anymal.num_joints))
        self.assertTrue(height_known and linear_known)
        torch.testing.assert_close(state.height, result.base_position[:, 2])

    def test_keypoint_map_and_inputs_are_checked(self) -> None:
        """Maps round-trip through JSON; unknown links, empty maps and bad shapes fail loudly."""
        keypoints = KeypointMap.from_dict(GO1_TO_ANYMAL)
        self.assertEqual(KeypointMap.from_dict(keypoints.to_dict()), keypoints)
        self.assertEqual(KeypointMap.from_dict({"FR_foot": "RF_FOOT"}).pairs, (KeypointPair("FR_foot", "RF_FOOT"),))
        with self.assertRaisesRegex(ValueError, "unknown links"):
            retarget(self.go1, self.joints, self.anymal, KeypointMap.from_dict({"FR_foot": "nope"}), dt=self.dt)
        with self.assertRaises(ValueError):
            KeypointMap(())
        with self.assertRaisesRegex(ValueError, "shape"):
            retarget(self.go1, self.joints[:, :5], self.anymal, keypoints, dt=self.dt)
        with self.assertRaisesRegex(ValueError, "scale"):
            scale_ratio(self.go1, self.anymal, keypoints, "height")
        pose = reference_pose(self.go1)
        self.assertTrue(bool(((pose >= self.go1.lower) & (pose <= self.go1.upper)).all()))

    def test_script_retargets_a_table_dataset(self) -> None:
        """The script reads the table format, retargets every episode, and writes a loadable source."""
        path = ROOT / "scripts" / "retarget_motion.py"
        spec = importlib.util.spec_from_file_location("retarget_motion", path)
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(script)
        frames = 12
        quaternion = Rotation.from_matrix(self.rotation[:frames].numpy()).as_quat()
        signals = {"state": self.joints[:frames].numpy(), "base_position": self.position[:frames].numpy(), "base_orientation": quaternion}
        with tempfile.TemporaryDirectory() as folder:
            motion, output = Path(folder, "go1"), Path(folder, "anymal")
            write_dataset(motion, [Episode(signals), Episode(signals)], 1 / self.dt, list(self.go1.joint_names))
            keypoints = Path(folder, "keypoints.json")
            keypoints.write_text(json.dumps(GO1_TO_ANYMAL))
            legs = [name for name in self.anymal.joint_names if not name.startswith("inspection")]
            written = script.main([
                "--source-urdf", str(GO1), "--target-urdf", str(ANYMAL), "--motion", str(motion),
                "--keypoints", str(keypoints), "--output", str(output), "--source-floating-base",
                "--target-floating-base", "--target-joints", *legs, "--target-initial-pose", json.dumps(ANYMAL_POSE),
            ])
            self.assertEqual(written["joint_names"], legs)
            self.assertEqual(len(written["retargeting"]["max_keypoint_error"]), 2)
            # Float32 storage of the source motion limits the match to about 1e-6 m.
            self.assertLess(max(written["retargeting"]["max_keypoint_error"]), 1e-3)
            source = json.loads(Path(output, "source.json").read_text())
            records, trajectories = SOURCE_TYPES["huggingface"](source, None, JointFeatureConfig())
        self.assertEqual(records[0].graph.joint_names, tuple(legs))
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(tuple(trajectories[0].positions.shape), (frames, len(legs)))
        self.assertAlmostEqual(trajectories[0].dt, self.dt, places=6)
        scale = written["retargeting"]["scale"]
        torch.testing.assert_close(trajectories[0].base_position, scale * self.position[:frames], atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
