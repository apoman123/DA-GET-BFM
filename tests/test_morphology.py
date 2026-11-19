"""Test the procedural morphology generator and the procedural training source."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from model import EmbodimentGraph, JointFeatureConfig, actuator_specs, parse_urdf
from model.morphology import (
    DEFAULT_RANGES,
    FAMILIES,
    FAMILY_TEMPLATE,
    build_urdf,
    check_physical_consistency,
    generate_robot,
    resolve_ranges,
    sample_parameters,
    shape_inertia,
    standing_torques,
    suite_seeds,
    template_summary,
    write_robot_suite,
)
from model.robot_analysis import analyze_robot
from model.training.episodes import EmbodimentRecord
from model.training.procedural import procedural_source, source_fingerprint

# The robot URDFs in assets/urdf are downloaded, not committed (scripts/download_urdfs.py).
NEEDS_URDFS = unittest.skipUnless(
    all((Path(__file__).resolve().parents[1] / "assets" / "urdf" / name).exists() for name in ("go1.urdf", "h1.urdf", "anymal_d.urdf")),
    "needs the robot URDFs: uv run python scripts/download_urdfs.py",
)


HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
SEEDS = (0, 1, 3)


class PhysicalConsistencyTest(unittest.TestCase):
    """The pseudo-inertia check accepts real solids and rejects impossible ones."""

    def test_solids_pass_and_impossible_inertias_fail(self) -> None:
        """Boxes, cylinders and spheres pass; triangle violations, m <= 0 and asymmetry fail."""
        for kind, size in (("box", (0.3, 0.1, 0.05)), ("cylinder", (0.2, 0.02)), ("sphere", (0.03,))):
            for axis in "xyz":
                self.assertTrue(check_physical_consistency(1.5, (0.1, -0.2, 0.3), shape_inertia(kind, size, 1.5, axis)))
        self.assertFalse(check_physical_consistency(1.0, (0, 0, 0), torch.diag(torch.tensor([1.0, 1.0, 3.0]))))
        self.assertFalse(check_physical_consistency(0.0, (0, 0, 0), torch.eye(3)))
        self.assertFalse(check_physical_consistency(-1.0, (0, 0, 0), torch.eye(3)))
        skew = torch.eye(3, dtype=torch.float64)
        skew[0, 1] = 0.5
        self.assertFalse(check_physical_consistency(1.0, (0, 0, 0), skew))


@NEEDS_URDFS
class GeneratorTest(unittest.TestCase):
    """Generated robots are valid URDF, physically consistent, cited, and deterministic."""

    def test_every_family_builds_a_consistent_robot(self) -> None:
        """Each family parses, builds a graph, and has positive, consistent mass properties."""
        for family in FAMILIES:
            for seed in SEEDS:
                robot = generate_robot(family, seed)
                parsed = parse_urdf(robot.urdf)
                for link in parsed.links.values():
                    if link.inertial is not None:
                        self.assertTrue(check_physical_consistency(
                            link.inertial.mass, link.inertial.origin[:3, 3], link.inertial.inertia), link.name)
                graph = EmbodimentGraph.from_urdf(parsed, floating_base=True)
                self.assertGreater(graph.total_mass, 0)
                for index in range(graph.num_joints):
                    self.assertTrue(check_physical_consistency(
                        float(graph.body_mass[index]), graph.body_com[index], graph.body_inertia[index]))
                self.assertTrue(check_physical_consistency(graph.base_mass, graph.base_com, graph.base_inertia))
                metadata = robot.metadata
                self.assertAlmostEqual(metadata["total_mass"], graph.total_mass)
                self.assertAlmostEqual(metadata["characteristic_length"], graph.characteristic_length)
                self.assertEqual(set(metadata["nominal_pose"]), set(graph.joint_names))
                self.assertTrue(all(name.endswith("_foot") for name in metadata["contact_links"]))
                # Existing consumers find the feet on their own.
                record = EmbodimentRecord.build("robot", graph, actuator_specs(graph))
                self.assertEqual(set(record.contact_links), set(metadata["contact_links"]))
                if family != "hexapod" and not metadata["params"]["knee_limit_legs"]:
                    # The URDF analysis looks at the zero pose clamped into the limits, where a
                    # leg with a restricted knee is shorter and GenBot-1K's hexapod legs are
                    # horizontal, so only check the others.
                    self.assertEqual(set(analyze_robot(parsed).contact_links), set(metadata["contact_links"]))
                # The nominal pose is inside the limits and the robot can stand on its feet.
                for name, value in metadata["nominal_pose"].items():
                    index = graph.joint_names.index(name)
                    self.assertGreaterEqual(value, float(graph.lower[index]) - 1e-12)
                    self.assertLessEqual(value, float(graph.upper[index]) + 1e-12)
                self.assertLess(metadata["standing_residual"], 1e-4 * graph.total_mass * 9.81)

    def test_humanoid_feet_stand_under_the_hips(self) -> None:
        """With knees, each ankle is straight below its hip pitch joint and the sole is flat."""
        robot = generate_robot("humanoid", 0, {"knee_joints": [1, 2, 3]})
        graph = EmbodimentGraph.from_urdf(robot.urdf, floating_base=True)
        q = torch.tensor([robot.metadata["nominal_pose"][name] for name in graph.joint_names], dtype=torch.float64)
        frames = graph.forward_kinematics(q)
        for side in ("left", "right"):
            hip = frames[graph.joint_names.index(f"{side}_hip_pitch_joint"), :3, 3]
            ankle = frames[graph.joint_names.index(f"{side}_ankle_joint")]
            self.assertAlmostEqual(float(ankle[0, 3]), float(hip[0]), places=9)
            self.assertTrue(torch.allclose(ankle[:3, :3], torch.eye(3, dtype=torch.float64), atol=1e-9))

    def test_default_ranges_are_respected(self) -> None:
        """Every sampled value comes from the cited candidate sets and intervals."""
        for family in FAMILIES:
            ranges = DEFAULT_RANGES[family]
            targets = [sorted(target) for target in ranges["knee_limit_targets"]]
            for seed in range(150):
                params = sample_parameters(family, torch.Generator().manual_seed(seed))
                self.assertIn(params["knee_joints"], ranges["knee_joints"])
                changed = [key for key in ("thigh_scale", "calf_scale", "foot_scale", "torso_scale") if params.get(key, 1.0) != 1.0]
                self.assertLessEqual(len(changed) + (params["link_scale"] != 1.0), 1)  # one factor at a time
                for key in ("link_scale", "thigh_scale", "calf_scale", "foot_scale", "torso_scale"):
                    if key in ranges:
                        self.assertIn(params[key], ranges[key])
                self.assertIn(params["knee_limit_scale"], ranges["knee_limit_scale"])
                if params["knee_limit_scale"] != 1.0:
                    self.assertIn(sorted(params["knee_limit_legs"]), targets)
                self.assertTrue(all(0.5 <= value <= 1.5 for value in params["mass_scale"].values()))
                self.assertTrue(2.3 <= params["torque_ratio"] <= 13.3)

    def test_range_overrides(self) -> None:
        """Overrides replace defaults; continuous ranges need independent sampling; typos fail."""
        ranges = {
            "knee_joints": [2], "geometry_sampling": "independent", "link_scale": {"uniform": [0.9, 1.1]},
            "thigh_calf_ratio": {"uniform": [0.75, 1.25]}, "base_density": {"uniform": [400.0, 1200.0]},
        }
        for seed in range(20):
            params = sample_parameters("quadruped", torch.Generator().manual_seed(seed), ranges)
            self.assertEqual(params["knee_joints"], 2)
            self.assertTrue(0.9 <= params["link_scale"] <= 1.1)
            self.assertTrue(0.75 - 1e-12 <= params["thigh_scale"] / params["calf_scale"] <= 1.25 + 1e-12)
            self.assertTrue(400.0 <= params["base_density"] <= 1200.0)
        robot = generate_robot("quadruped", 5, ranges)
        trunk = parse_urdf(robot.urdf).links["trunk"].inertial.mass
        volume = 0.3762 * 0.0935 * 0.114 * robot.metadata["params"]["link_scale"] ** 3
        self.assertAlmostEqual(trunk / volume, robot.metadata["params"]["base_density"], places=6)
        with self.assertRaises(ValueError):
            resolve_ranges("quadruped", {"knee_count": [1]})
        with self.assertRaises(ValueError):
            sample_parameters("quadruped", torch.Generator(), {"link_scale": {"uniform": [0.8, 1.2]}})
        with self.assertRaises(ValueError):
            resolve_ranges("biped")

    def test_parameters_shape_the_robot(self) -> None:
        """Knee count, segment lengths, knee-limit scaling, and arms appear in the URDF."""
        params = sample_parameters("quadruped", torch.Generator().manual_seed(0), {"knee_joints": [3]})
        params.update(link_scale=1.0, thigh_scale=1.6, calf_scale=0.4, foot_scale=1.0,
                      knee_limit_scale=0.2, knee_limit_legs=["FL"])
        robot = parse_urdf(build_urdf("quadruped", params))
        self.assertEqual(sum(joint.joint_type == "revolute" for joint in robot.joints.values()), 4 * 5)
        self.assertAlmostEqual(float(robot.joints["FR_knee_joint"].origin[2, 3]), -0.213 * 1.6)
        self.assertAlmostEqual(float(robot.joints["FR_knee2_joint"].origin[2, 3]), -0.12 * 0.4)
        # GenBot-1K scales the range about the nominal angle (-1.5 rad) on the targeted leg only.
        self.assertAlmostEqual(robot.joints["FL_knee_joint"].lower, -1.5 + 0.2 * (-2.72 + 1.5))
        self.assertAlmostEqual(robot.joints["FL_knee_joint"].upper, -1.5 + 0.2 * (-0.84 + 1.5))
        self.assertAlmostEqual(robot.joints["FR_knee_joint"].lower, -2.72)
        armless = generate_robot("humanoid", 0, {"arms": [False]})
        self.assertFalse(any("shoulder" in name for name in parse_urdf(armless.urdf).joints))
        legless = generate_robot("hexapod", 0, {"knee_joints": [0]})
        self.assertFalse(any("knee" in name for name in parse_urdf(legless.urdf).joints))

    def test_generation_is_deterministic(self) -> None:
        """The same seed gives the same URDF; different seeds give different robots."""
        for family in FAMILIES:
            first, second = generate_robot(family, 42), generate_robot(family, 42)
            self.assertEqual(first.urdf, second.urdf)
            self.assertEqual(first.metadata, second.metadata)
        texts = {generate_robot("quadruped", seed).metadata["params"]["torque_ratio"] for seed in range(5)}
        self.assertEqual(len(texts), 5)
        self.assertEqual(suite_seeds(3, 4), suite_seeds(3, 4))

    def test_torque_and_speed_limits(self) -> None:
        """Peak torque is at least ratio x standing torque, exactly so on the most loaded role."""
        for family in FAMILIES:
            template = template_summary(FAMILY_TEMPLATE[family])
            for seed in SEEDS:
                robot = generate_robot(family, seed, {"knee_joints": [1, 2]})
                metadata = robot.metadata
                graph = EmbodimentGraph.from_urdf(robot.urdf, floating_base=True)
                ratio = metadata["torque_ratio"]
                self.assertTrue(2.3 <= ratio <= 13.3)
                standing = standing_torques(graph, metadata["nominal_pose"], metadata["contact_links"])
                peak = graph.effort_limits
                self.assertTrue(bool((peak >= ratio * standing.abs() * (1 - 1e-12)).all()))
                loaded = int(standing.abs().argmax())
                expected = ratio * float(standing.abs().max())
                self.assertLess(abs(float(peak[loaded]) - expected), 1e-12 * expected)
                froude = math.sqrt(template["characteristic_length"] / graph.characteristic_length)
                speeds = template["velocity"].values()
                for name, value in zip(graph.joint_names, graph.velocity_limits.tolist()):
                    self.assertLess(min(abs(value / froude / speed - 1) for speed in speeds), 1e-12, name)

    def test_suite_manifest_and_script(self) -> None:
        """The suite writer and the CLI write URDFs plus a manifest with the generator metadata."""
        path = Path(__file__).resolve().parents[1] / "scripts" / "generate_morphologies.py"
        spec = importlib.util.spec_from_file_location("generate_morphologies", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_robot_suite(directory, ["quadruped", "humanoid"], 3, seed=4)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual([entry["file"] for entry in manifest["robots"]], [path.name for path in paths])
            self.assertEqual([entry["family"] for entry in manifest["robots"]], ["quadruped", "humanoid", "quadruped"])
            seeds = suite_seeds(4, 3)
            self.assertEqual(paths[1].read_text(), generate_robot("humanoid", seeds[1]).urdf)
            cli = script.main(["--family", "hexapod", "--count", "2", "--seed", "1", "--output", str(Path(directory) / "cli")])
            self.assertEqual(len(cli), 2)
            self.assertTrue((Path(directory) / "cli" / "manifest.json").exists())

    @unittest.skipUnless(HAS_MUJOCO, "needs mujoco (uv sync --extra sim)")
    def test_mujoco_loads_the_urdf(self) -> None:
        """MuJoCo's own URDF importer gives our fixed-base mass matrix."""
        import mujoco

        robot = generate_robot("humanoid", 1)
        model = mujoco.MjModel.from_xml_string(robot.urdf)
        data = mujoco.MjData(model)
        graph = EmbodimentGraph.from_urdf(robot.urdf)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in graph.joint_names]
        q = torch.linspace(-0.3, 0.3, graph.num_joints, dtype=torch.float64)
        data.qpos[[model.jnt_qposadr[joint] for joint in joints]] = q.numpy()
        mujoco.mj_forward(model, data)
        full = torch.zeros((model.nv, model.nv), dtype=torch.float64).numpy()
        mujoco.mj_fullM(model, data, full)
        dofs = [model.jnt_dofadr[joint] for joint in joints]
        self.assertTrue(torch.allclose(torch.from_numpy(full[dofs][:, dofs]), graph.mass_matrix(q), atol=1e-12))


@NEEDS_URDFS
class ProceduralSourceTest(unittest.TestCase):
    """The procedural source returns records and correctly indexed trajectories."""

    def test_synthetic_source(self) -> None:
        """Two robots with two variants each give four records; indices point at the right one."""
        with tempfile.TemporaryDirectory() as directory:
            source = {
                "type": "procedural", "families": ["quadruped", "hexapod"], "count": 2, "seed": 5,
                "simulator": "synthetic", "output_dir": directory, "randomized_embodiments": 2,
                "episodes_per_embodiment": 1, "steps": 5, "probe_torques": 1, "ranges": {"knee_joints": [1]},
            }
            records, trajectories = procedural_source(source, None, JointFeatureConfig.da_get_zero())
            self.assertEqual(len(records), 4)
            self.assertEqual(sorted(trajectory.embodiment for trajectory in trajectories), [0, 1, 2, 3])
            for trajectory in trajectories:
                record = records[trajectory.embodiment]
                self.assertEqual(trajectory.positions.shape, (5, record.graph.num_joints))
            self.assertEqual([record.family for record in records], ["quadruped"] * 2 + ["hexapod"] * 2)
            self.assertEqual(records[0].graph.num_joints % 4, 0)
            self.assertEqual(records[2].graph.num_joints % 6, 0)
            env = records[2].metadata["env"]
            self.assertEqual(set(env), {"nominal_pose", "contact_links", "contact_geometry"})
            self.assertEqual(set(env["contact_geometry"]), {"radius", "sole_offset", "foot_size"})
            self.assertEqual(list(records[2].contact_links), env["contact_links"])
            generator = records[2].metadata["generator"]
            self.assertEqual(generator["family"], "hexapod")
            self.assertEqual(generator["seed"], suite_seeds(5, 2)[1])
            for key in ("params", "total_mass", "characteristic_length", "torque_ratio"):
                self.assertIn(key, generator)
            self.assertEqual(source_fingerprint(source), source_fingerprint(dict(source)))
            self.assertNotEqual(source_fingerprint(source), source_fingerprint({**source, "seed": 6}))

    @unittest.skipUnless(HAS_MUJOCO, "needs mujoco (uv sync --extra sim)")
    def test_mujoco_source(self) -> None:
        """A generated quadruped stands on MuJoCo's ground with its nominal pose and foot size."""
        with tempfile.TemporaryDirectory() as directory:
            source = {
                "type": "procedural", "family": "quadruped", "count": 1, "seed": 2, "simulator": "mujoco",
                "output_dir": directory, "randomized_embodiments": 0, "episodes_per_embodiment": 1,
                "steps": 25, "probe_torques": 1, "ranges": {"knee_joints": [1]},
            }
            records, trajectories = procedural_source(source, None, JointFeatureConfig.da_get_zero())
            self.assertEqual(len(records), 1)
            self.assertEqual(len(trajectories), 1)
            self.assertEqual(trajectories[0].embodiment, 0)
            self.assertEqual(trajectories[0].contacts.shape[1], 4)
            self.assertTrue(bool(trajectories[0].contacts.all(dim=1).float().mean() > 0.5))


if __name__ == "__main__":
    unittest.main()
