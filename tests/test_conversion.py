"""Test dataset converters, column mapping, robot analysis, and config generation."""

import argparse
import importlib.util
import io
import json
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from model import EmbodimentGraph
from model.conversion.common import Episode, rotation_from_xyzw, write_dataset
from model.conversion.datasets import (
    ANYMAL_FEET,
    ANYMAL_JOINTS,
    G1_JOINTS,
    LAFAN1_JOINTS,
    convert_amass_g1,
    convert_grandtour,
    convert_lafan1,
    convert_locomujoco,
    convert_mipo,
    convert_neurobem,
    convert_xwbc,
)
from model.conversion.geometry import collision_points, ground_offset, lowest_point_heights, read_stl
from model.conversion.mapping import inspect_dataset, match_joints, words
from model.robot_analysis import analyze_robot
from model.training import build_physics_datasets
from tests.test_embodiment import double_pendulum, humanoid, quadruped
from tests.test_floating import quadrotor

HAS_H5PY = importlib.util.find_spec("h5py") is not None
G1_URDF = Path(__file__).resolve().parents[1] / "assets" / "urdf" / "g1_29dof_rev_1_0.urdf"
# The robot URDFs in assets/urdf are downloaded, not committed (scripts/download_urdfs.py).
NEEDS_G1_URDF = unittest.skipUnless(G1_URDF.exists(), "needs the robot URDFs: uv run python scripts/download_urdfs.py")
G1_MESHES = (G1_URDF.parent / "meshes" / "torso_link_rev_1_0.STL").exists()


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

    @NEEDS_G1_URDF
    def test_amass_and_lafan1(self) -> None:
        """Kinematic humanoid motions convert with G1 joint names; AMASS clips are resampled and put on the ground."""
        count = 481  # 4 s at 120 Hz
        table = np.zeros((count, 36))
        table[:, 6] = 1.0  # identity quaternion, x, y, z, w
        table[240:, 2] = -0.3  # the relative root z drifts down 0.3 m half way
        table[:, 7 + G1_JOINTS.index("left_elbow_joint")] = np.linspace(0.0, 0.6, count)
        table[:, 7 + G1_JOINTS.index("right_wrist_roll_joint")] = 0.3  # stored with the wrong sign upstream
        path = self.root / "walk_poses_120_jpos.npy"
        np.save(path, table)
        graph = EmbodimentGraph.from_urdf(G1_URDF, joint_order=G1_JOINTS, floating_base=True)

        def heights(geometry, **kwargs):
            info, source = self.load_conversion(convert_amass_g1, files=[path], urdf=str(G1_URDF), **kwargs)
            frames = pq.read_table(self.root / "out" / "data" / "chunk-000" / "file-000.parquet").to_pydict()
            q, position, orientation = (np.array(frames[key], dtype=np.float64) for key in (
                "observation.state", "observation.base_position", "observation.base_orientation"))
            return info, source, q, position, lowest_point_heights(graph, geometry, q, position, orientation)

        feet = collision_points(G1_URDF, ["left_ankle_roll_link", "right_ankle_roll_link"])
        info, source, q, position, lowest = heights(feet, ground="feet")
        self.assertEqual(info["features"]["observation.state"]["names"], G1_JOINTS)
        self.assertEqual((info["fps"], info["total_frames"]), (50.0, 201))  # 4 s at 120 Hz -> 50 Hz
        self.assertEqual(source["base_orientation_format"], "xyzw")
        np.testing.assert_allclose(q[-1, G1_JOINTS.index("left_elbow_joint")], 0.6, atol=1e-6)
        np.testing.assert_allclose(q[:, G1_JOINTS.index("right_wrist_roll_joint")], -0.3, atol=1e-6)
        self.assertGreaterEqual(lowest.min(), -1e-6)  # the envelope never puts a frame below the ground (float32 files)
        self.assertAlmostEqual(lowest[0], 0.0, places=5)  # and removes the 0.3 m drift
        self.assertAlmostEqual(lowest[-1], 0.0, places=5)
        _, _, _, _, lowest = heights(feet, ground="feet", ground_window=0.0)  # one constant per clip
        self.assertAlmostEqual(float(np.percentile(lowest, 5)), 0.0, places=5)
        self.assertGreater(lowest[0], 0.25)  # the first half floats above the drifted second half
        if G1_MESHES:
            _, _, _, _, lowest = heights(collision_points(G1_URDF))
            self.assertGreaterEqual(lowest.min(), -1e-6)
        _, _, _, position, _ = heights(feet, root_height=0.793)
        self.assertAlmostEqual(position[0, 2], 0.793, places=5)
        bad = self.root / "walk.npy"
        np.save(bad, table)
        with self.assertRaises(ValueError):
            convert_amass_g1(self.root / "bad", files=[bad], urdf=str(G1_URDF), ground="feet")
        csv = self.root / "dance.csv"
        table = np.zeros((count, 36))
        table[:, 6] = 1.0
        np.savetxt(csv, table, delimiter=",")
        output = self.root / "lafan"
        convert_lafan1(output, robot="g1", files=[csv])
        info, source = self.load(output)
        self.assertEqual(info["fps"], 30.0)
        self.assertEqual(source["contact_links"], ["left_ankle_roll_link", "right_ankle_roll_link"])

    @NEEDS_G1_URDF
    def test_xwbc_archive(self) -> None:
        """xwbc clips stream from a tar.gz, are reordered by joint name, and are put on the ground."""
        isaac_order = list(G1_JOINTS[::-1])
        count, rate = 90, 30
        archive = self.root / "unitree_g1_29dof.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, order in (("a_walk", G1_JOINTS), ("b_run", isaac_order)):
                q = np.zeros((count, 29), np.float32)
                q[:, order.index("left_elbow_joint")] = np.linspace(0, 0.5, count)
                bodies = np.zeros((count, 32, 3), np.float32)
                bodies[:, 0] = [1.0, 2.0, 0.75]  # the soles end 2-4 cm below z = 0
                quaternion = np.zeros((count, 32, 4), np.float32)
                quaternion[..., 0] = 1.0  # w, x, y, z
                buffer = io.BytesIO()
                np.savez(buffer, fps=np.array([rate]), joint_pos=q, joint_vel=np.gradient(q, 1 / rate, axis=0),
                         joint_names=np.array(order), body_pos_w=bodies, body_quat_w=quaternion,
                         body_lin_vel_w=np.zeros_like(bodies), body_ang_vel_w=np.zeros_like(bodies))
                member = tarfile.TarInfo(f"unitree_g1_29dof/ACCAD__{name}_stageii.npz")
                member.size = buffer.getbuffer().nbytes
                buffer.seek(0)
                tar.addfile(member, buffer)
        output = self.root / "xwbc"
        source = convert_xwbc(output, archive=archive, ground="feet")
        info, _ = self.load(output)
        self.assertEqual(source["joint_names"], G1_JOINTS)
        self.assertEqual((info["fps"], info["total_episodes"]), (30.0, 2))
        frames = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet").to_pydict()
        q = np.array(frames["observation.state"], dtype=np.float64)
        np.testing.assert_allclose(q[count - 1, G1_JOINTS.index("left_elbow_joint")], 0.5, atol=1e-6)
        np.testing.assert_allclose(q[-1, G1_JOINTS.index("left_elbow_joint")], 0.5, atol=1e-6)  # reordered clip
        position = np.array(frames["observation.base_position"], dtype=np.float64)
        orientation = np.array(frames["observation.base_orientation"], dtype=np.float64)
        np.testing.assert_allclose(orientation[0], [0.0, 0.0, 0.0, 1.0])
        graph = EmbodimentGraph.from_urdf(G1_URDF, joint_order=G1_JOINTS, floating_base=True)
        feet = collision_points(G1_URDF, ["left_ankle_roll_link", "right_ankle_roll_link"])
        lowest = lowest_point_heights(graph, feet, q, position, orientation)
        self.assertGreaterEqual(lowest.min(), -1e-6)
        self.assertAlmostEqual(float(np.median(lowest)), 0.0, places=4)
        np.testing.assert_allclose(np.array(frames["observation.base_linear_velocity"]), 0.0, atol=1e-6)  # the offset is constant
        convert_xwbc(self.root / "one", archive=archive, ground="none", patterns=["*b_run*"])
        info, _ = self.load(self.root / "one")
        self.assertEqual(info["total_episodes"], 1)
        frames = pq.read_table(self.root / "one" / "data" / "chunk-000" / "file-000.parquet").to_pydict()
        self.assertAlmostEqual(frames["observation.base_position"][0][2], 0.75, places=5)  # recorded height kept
        with self.assertRaises(ValueError):
            convert_xwbc(self.root / "h1", robot="unitree_h1", archive=archive)

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


    def test_unitree_g1_wrist_names_are_corrected(self) -> None:
        """Unitree G1 arm columns named yaw, roll, pitch map to the stored roll, pitch, yaw order."""
        arm = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_roll", "wrist_pitch"]
        joints = [f"left_{name}_joint" for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
        for robot_type, expected in (("unitree_g1_inspire", [0, 1, 2, 3, 4, 5, 6]), ("other_robot", [0, 1, 2, 3, 5, 6, 4])):
            with tempfile.TemporaryDirectory() as folder:
                (Path(folder) / "meta").mkdir()
                info = {"robot_type": robot_type, "fps": 30, "features": {
                    "observation.state.left_arm": {"dtype": "float32", "shape": [7], "names": arm},
                    "action.left_arm": {"dtype": "float32", "shape": [7], "names": arm},
                }}
                (Path(folder) / "meta" / "info.json").write_text(json.dumps(info))
                source = inspect_dataset(folder, joints)["source"]
            with self.subTest(robot_type=robot_type):
                self.assertEqual(source["state_indices"], expected)
                self.assertEqual(source["action_indices"], expected)


class GeometryTest(unittest.TestCase):
    """Collision geometry gives the lowest point of a posed robot, and the ground envelope keeps it above z = 0."""

    def test_lowest_point_of_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32) * 0.1
            faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
            records = b"".join(struct.pack("<3f", 0, 0, 0) + corners[list(face)].tobytes() + b"\0\0" for face in faces)
            (root / "cube.stl").write_bytes(b"solid but binary".ljust(80, b" ") + struct.pack("<I", len(faces)) + records)
            ascii_text = "solid c\n" + "".join(
                "facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in corners[list(face)]) + "endloop\nendfacet\n"
                for face in faces) + "endsolid c\n"
            (root / "cube_ascii.stl").write_text(ascii_text)
            np.testing.assert_allclose(read_stl(root / "cube.stl"), read_stl(root / "cube_ascii.stl"), atol=1e-6)
            urdf = root / "robot.urdf"
            urdf.write_text("""<robot name="r">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0 0 -0.5"/><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>
  <link name="arm"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="1 0 0"/><geometry><mesh filename="cube.stl" scale="2 2 2"/></geometry></collision>
    <collision><origin xyz="0.5 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.05" length="1"/></geometry></collision>
    <collision><origin xyz="0 0 0.3"/><geometry><sphere radius="0.05"/></geometry></collision></link>
  <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
</robot>""")
            geometry = collision_points(urdf)
            self.assertEqual(len(geometry["arm"][0]), 8 + 64 + 1)  # hull of the cube, cylinder rims, sphere
            graph = EmbodimentGraph.from_urdf(urdf, floating_base=True)
            angles = np.array([[0.0], [np.pi / 2], [-np.pi / 2]])
            upright = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
            heights = lowest_point_heights(graph, geometry, angles, np.zeros((3, 3)), upright)
            # 0: the base box bottom at -0.6; +90 deg about y turns the arm down: cube centre at z -1,
            # half size 0.2 -> -1.2; -90 deg turns it up: the sphere at x -0.3, z 0, and the box again.
            np.testing.assert_allclose(heights, [-0.6, -1.2, -0.6], atol=1e-6)
            flipped = np.tile([1.0, 0.0, 0.0, 0.0], (1, 1))  # 180 deg about x: the arm's sphere goes to z -0.3
            np.testing.assert_allclose(lowest_point_heights(graph, geometry, angles[:1], np.zeros((1, 3)), flipped), [-0.35], atol=1e-6)
            missing = root / "missing.urdf"
            missing.write_text(urdf.read_text().replace("cube.stl", "missing.stl"))
            with self.assertRaises(FileNotFoundError):
                collision_points(missing)

    def test_ground_envelope(self) -> None:
        rate = 50.0
        time = np.arange(0, 6, 1 / rate)
        flight = (time > 3.0) & (time < 3.3)
        heights = 0.01 * time + np.where(flight, 0.2 * np.sin(np.pi * (time - 3.0) / 0.3), 0.0)  # 1 cm/s drift and a 0.3 s jump
        offset = ground_offset(heights, rate, window=1.0)
        corrected = heights - offset
        self.assertGreaterEqual(corrected.min(), -1e-12)
        self.assertLess(np.abs(corrected[~flight & (time > 0.5) & (time < 5.5)]).max(), 0.006)
        self.assertGreater(corrected[flight].max(), 0.19)  # the jump is kept
        constant = ground_offset(heights, rate, window=0.0)
        self.assertTrue(np.allclose(constant, np.percentile(heights, 5)))


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
