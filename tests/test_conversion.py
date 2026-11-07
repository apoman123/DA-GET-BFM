"""Test dataset converters, column mapping, robot analysis, and config generation."""

import argparse
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import EmbodimentGraph
from model.conversion.common import Episode, rotation_from_xyzw, write_dataset
from model.conversion.datasets import (
    ANYMAL_FEET,
    ANYMAL_JOINTS,
    LAFAN1_JOINTS,
    convert_amass_g1,
    convert_grandtour,
    convert_lafan1,
    convert_locomujoco,
    convert_mipo,
    convert_neurobem,
)
from model.conversion.mapping import inspect_dataset, match_joints, words
from model.robot_analysis import analyze_robot
from model.training import build_physics_datasets
from tests.test_embodiment import double_pendulum, humanoid, quadruped
from tests.test_floating import quadrotor

HAS_H5PY = importlib.util.find_spec("h5py") is not None


def write_zarr_array(folder: Path, name: str, values: np.ndarray) -> None:
    """Write one uncompressed zarr v2 array."""
    target = folder / name
    target.mkdir(parents=True)
    meta = {"chunks": list(values.shape), "compressor": None, "dtype": values.dtype.str, "fill_value": 0,
            "filters": None, "order": "C", "shape": list(values.shape), "zarr_format": 2}
    (target / ".zarray").write_text(json.dumps(meta))
    key = ".".join("0" for _ in values.shape)
    (target / key).write_bytes(np.ascontiguousarray(values).tobytes())


class ConverterTest(unittest.TestCase):
    """Each converter turns its source format into a loadable table."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self) -> None:
        self.folder.cleanup()

    def load(self, output: Path) -> dict:
        info = json.loads((output / "meta" / "info.json").read_text())
        source = json.loads((output / "source.json").read_text())
        self.assertTrue((output / "data" / "chunk-000" / "file-000.parquet").exists())
        return info, source

    def test_grandtour_zarr_tar(self) -> None:
        """Torques, contact states, quaternions, and twist survive conversion."""
        group = self.root / "anymal_state_state_estimator"
        group.mkdir()
        (group / ".zgroup").write_text('{"zarr_format": 2}')
        count = 1200
        time = 1000 + np.arange(count) / 400
        write_zarr_array(group, "timestamp", time)
        for name in ("joint_positions", "joint_velocities", "joint_efforts"):
            write_zarr_array(group, name, np.random.default_rng(0).normal(size=(count, 12)))
        write_zarr_array(group, "pose_pos", np.zeros((count, 3)))
        write_zarr_array(group, "pose_orien", np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)))
        write_zarr_array(group, "twist_lin", np.zeros((count, 3)))
        write_zarr_array(group, "twist_ang", np.zeros((count, 3)))
        for index, foot in enumerate(ANYMAL_FEET):
            write_zarr_array(group, f"{foot}_state", (np.arange(count) % 4 != index).astype(np.uint8))
            write_zarr_array(group, f"{foot}_wrench_force", np.zeros((count, 3)))
        archive = self.root / "state.tar"
        with tarfile.open(archive, "w") as handle:
            handle.add(group, arcname="anymal_state_state_estimator")
        info, source = self.load_conversion(convert_grandtour, archives=[archive], rate=100.0, episode_seconds=1.0)
        self.assertEqual(info["features"]["observation.state"]["names"], ANYMAL_JOINTS)
        self.assertEqual(source["torque_key"], "observation.effort")
        self.assertEqual(source["contact_links"], ANYMAL_FEET)
        self.assertEqual(info["total_episodes"], 3)

    def load_conversion(self, function, **kwargs):
        output = self.root / "out"
        function(output, **kwargs)
        return self.load(output)

    def test_neurobem_identifies_thrust(self) -> None:
        """The thrust coefficient is recovered from synthetic hover-like data."""
        rng = np.random.default_rng(1)
        count, k_f, mass = 2000, 1.5e-6, 0.772
        speeds = 1100 + 60 * rng.normal(size=(count, 4))
        table = np.zeros((count, 29))
        table[:, 0] = np.arange(count) / 400
        table[:, 10] = 1.0  # qw
        table[:, 13] = k_f * (speeds**2).sum(1) / mass  # specific force along body z
        table[:, 20:24] = speeds
        csv_folder = self.root / "processed_data"
        csv_folder.mkdir()
        header = ",".join(f"c{index}" for index in range(29))
        np.savetxt(csv_folder / "merged_flight_seg_1.csv", table, delimiter=",", header=header, comments="")
        info, source = self.load_conversion(convert_neurobem, source=csv_folder)
        model = info["identified_model"]
        self.assertAlmostEqual(model["k_f"] / k_f, 1.0, places=3)
        overrides = source["actuator_overrides"]["rotor_*"]
        self.assertEqual(overrides["mode"], "thrust")
        self.assertTrue(Path(source["urdf"]).exists())
        graph = EmbodimentGraph.from_urdf(source["urdf"], floating_base=True)
        self.assertEqual(len(graph.joint_names), 4)
        self.assertAlmostEqual(graph.total_mass, mass, places=6)

    def test_locomujoco_free_root(self) -> None:
        """MuJoCo free-root qpos/qvel become base pose and base-frame twist."""
        count = 50
        qpos = np.zeros((count, 7 + 3))
        qpos[:, 3] = 1.0
        qpos[:, 0] = np.arange(count) * 0.01
        qvel = np.zeros((count, 6 + 3))
        qvel[:, 0] = 0.4
        path = self.root / "walk.npz"
        np.savez(path, qpos=qpos, qvel=qvel, joint_names=np.array(["root", "a_joint", "b_joint", "c_joint"]),
                 frequency=np.array(40), split_points=np.array([0, 25, 50]))
        output = self.root / "loco"
        convert_locomujoco(output, files=[path])
        info, source = self.load(output)
        self.assertEqual(info["features"]["observation.state"]["names"], ["a_joint", "b_joint", "c_joint"])
        self.assertEqual(info["total_episodes"], 2)
        self.assertEqual(source["base_orientation_format"], "xyzw")

    def test_amass_and_lafan1(self) -> None:
        """Kinematic humanoid motions convert with G1 joint names."""
        joints = LAFAN1_JOINTS["g1"]
        count = 40
        path = self.root / "motion.npz"
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (count, 30, 1))
        np.savez(path, fps=np.array([30.0], dtype=np.float32), dof_names=np.array(joints),
                 dof_positions=np.zeros((count, 29), np.float32), body_positions=np.zeros((count, 30, 3), np.float32),
                 body_rotations=rotations.astype(np.float32))
        info, source = self.load_conversion(convert_amass_g1, files=[path])
        self.assertEqual(info["features"]["observation.state"]["names"], joints)
        csv = self.root / "dance.csv"
        table = np.zeros((count, 36))
        table[:, 6] = 1.0
        np.savetxt(csv, table, delimiter=",")
        output = self.root / "lafan"
        convert_lafan1(output, robot="g1", files=[csv])
        info, source = self.load(output)
        self.assertEqual(info["fps"], 30.0)
        self.assertEqual(source["contact_links"], ["left_ankle_roll_link", "right_ankle_roll_link"])

    @unittest.skipUnless(HAS_H5PY, "needs h5py (uv sync --extra validation)")
    def test_mipo_hdf5(self) -> None:
        """MIPO joints are renamed to Go1 URDF joints and contacts thresholded."""
        import h5py

        path = self.root / "trot.h5"
        count = 1200
        with h5py.File(path, "w") as handle:
            joints = handle.create_group("unitree_hardware/joint_foot")
            joints["time"] = np.arange(count) / 400
            joints["position"] = np.zeros((count, 16))
            joints["velocity"] = np.zeros((count, 16))
            effort = np.zeros((count, 16))
            effort[:, 12:] = 500.0
            joints["effort"] = effort
            imu = handle.create_group("unitree_hardware/imu")
            imu["time"] = np.arange(count) / 400
            imu["orientation"] = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1))
            imu["angular_velocity"] = np.zeros((count, 3))
            imu["linear_acceleration"] = np.tile([0.0, 0.0, 9.81], (count, 1))
        info, source = self.load_conversion(convert_mipo, files=[path])
        self.assertEqual(info["features"]["observation.state"]["names"][0], "FL_hip_joint")
        self.assertEqual(source["imu_acceleration_key"], "observation.imu_acceleration")
        self.assertEqual(source["contact_key"], "observation.contact")


class MappingTest(unittest.TestCase):
    """Column names are matched to URDF joints."""

    def test_words_and_matching(self) -> None:
        """CamelCase, Unitree leg codes, and side words normalize to shared words."""
        self.assertEqual(words("kLeftShoulderPitch"), ["left", "shoulder", "pitch"])
        self.assertEqual(words("FL1"), ["front", "left", "thigh"])
        candidates = [("observation.state", i, name, []) for i, name in enumerate(["kRightElbow", "kLeftElbow", "FL2"])]
        assigned, unmatched = match_joints(["left_elbow_joint", "right_elbow_joint", "FL_calf_joint", "waist_yaw_joint"], candidates)
        self.assertEqual(assigned["left_elbow_joint"], ("observation.state", 1))
        self.assertEqual(assigned["right_elbow_joint"], ("observation.state", 0))
        self.assertEqual(assigned["FL_calf_joint"], ("observation.state", 2))
        self.assertEqual(unmatched, ["waist_yaw_joint"])

    def test_inspect_converted_dataset(self) -> None:
        """A converted floating-base dataset maps back with base and contact keys."""
        graph = EmbodimentGraph.from_urdf(quadruped())
        names = list(graph.joint_names)
        with tempfile.TemporaryDirectory() as folder:
            count = 30
            signals = {
                "state": np.zeros((count, 12)), "velocity": np.zeros((count, 12)),
                "base_orientation": np.tile([0, 0, 0, 1.0], (count, 1)), "base_position": np.zeros((count, 3)),
                "base_angular_velocity": np.zeros((count, 3)), "contact": np.ones((count, 4)),
            }
            write_dataset(folder, [Episode(signals)], 50.0, names[::-1], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
            result = inspect_dataset(folder, names)
        source = result["source"]
        self.assertEqual(source["joint_names"], names)
        self.assertEqual(source["state_indices"], list(range(11, -1, -1)))
        self.assertEqual(source["base_orientation_format"], "xyzw")
        self.assertEqual(source["contact_key"], "observation.contact")
        self.assertEqual(source["base_angular_velocity_key"], "observation.base_angular_velocity")


class RobotAnalysisTest(unittest.TestCase):
    """The URDF analysis recognizes base types, feet, and rotors."""

    def test_robot_kinds(self) -> None:
        legged = analyze_robot(quadruped())
        self.assertEqual((legged.kind, legged.floating_base), ("quadruped", True))
        self.assertEqual(sorted(legged.contact_links), ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        self.assertFalse(legged.surface_contacts)
        biped = analyze_robot(humanoid())
        self.assertEqual(biped.kind, "biped")
        self.assertTrue(biped.surface_contacts)
        drone = analyze_robot(quadrotor(), thrust_to_weight=2.0, max_rotor_speed=1000.0)
        self.assertEqual(drone.kind, "multirotor")
        settings = drone.actuator_overrides[drone.rotor_joints[0]]
        graph = EmbodimentGraph.from_urdf(quadrotor())
        expected = 2.0 * graph.total_mass * 9.80665 / (4 * 1000.0**2)
        self.assertAlmostEqual(settings["thrust_coefficient"], expected)
        arm = analyze_robot(double_pendulum())
        self.assertEqual((arm.kind, arm.floating_base), ("fixed", False))

    def test_make_data_config(self) -> None:
        """The config script combines robot analysis, a dataset, and synthetic data."""
        import importlib

        spec = importlib.util.spec_from_file_location("make_data_config", Path(__file__).parents[1] / "scripts" / "make_data_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "quad.urdf").write_text(quadruped())
            graph = EmbodimentGraph.from_urdf(quadruped())
            signals = {"state": np.zeros((30, 12)), "base_orientation": np.tile([0, 0, 0, 1.0], (30, 1))}
            write_dataset(root / "data", [Episode(signals)], 50.0, list(graph.joint_names))
            arguments = argparse.Namespace(
                urdf=str(root / "quad.urdf"), dataset=str(root / "data"), name="quad", synthetic=True, ground=True,
                kind="auto", thrust_to_weight=2.5, max_rotor_speed=None, drag_ratio=0.016, match_threshold=0.4,
                max_episodes=None, randomized_embodiments=1,
            )
            config = module.build_config(arguments)
            types = [source["type"] for source in config["sources"]]
            self.assertEqual(types, ["huggingface", "synthetic", "mujoco"])
            recorded = config["sources"][0]
            self.assertTrue(recorded["floating_base"])
            self.assertEqual(len(recorded["contact_links"]), 4)
            config["sources"] = config["sources"][:1]
            datasets = build_physics_datasets(config, cache_dir=root / "cache")
            self.assertGreater(len(datasets.train), 0)


class QuaternionTest(unittest.TestCase):
    def test_rotation_from_xyzw(self) -> None:
        angle = 0.3
        rotation = rotation_from_xyzw(np.array([[0, 0, np.sin(angle / 2), np.cos(angle / 2)]]))[0]
        torch.testing.assert_close(torch.tensor(rotation[0, 0]), torch.tensor(np.cos(angle), dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
