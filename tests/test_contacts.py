"""Test contact inference from motion: the dynamics-consistent selector and its use in the labels."""

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from model import EmbodimentGraph, JointFeatureConfig, actuator_specs
from model.training.contacts import contact_probabilities, evaluate_contacts, select_contacts
from model.training.episodes import EmbodimentRecord, _BaseTrack, process_episode
from tests.test_embodiment import quadruped

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
STAND = {f"{leg}_{part}_joint": value for leg in ("FL", "FR", "RL", "RR") for part, value in (("thigh", 0.3), ("calf", -0.6))}
GRAVITY = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)


def robot() -> EmbodimentRecord:
    """Return the test quadruped with four point feet."""
    graph = EmbodimentGraph.from_urdf(quadruped(), JointFeatureConfig(), floating_base=True)
    return EmbodimentRecord.build("quad", graph, actuator_specs(graph), contact_links=FEET)


def standing(record: EmbodimentRecord, steps: int = 3) -> torch.Tensor:
    """Return ``[T, J]`` positions of the symmetric standing pose."""
    return torch.tensor([STAND.get(name, 0.0) for name in record.graph.joint_names], dtype=torch.float64).expand(steps, -1).clone()


def track(steps: int, linear_known: bool = True, linear_acceleration: torch.Tensor | None = None) -> _BaseTrack:
    """Return a base at rest (or with the given linear acceleration), in the base frame."""
    zeros = torch.zeros((steps, 3), dtype=torch.float64)
    acceleration = zeros if linear_acceleration is None else linear_acceleration.expand(steps, 3)
    return _BaseTrack(True, linear_known, zeros, zeros, acceleration, zeros, False, None, None)


class SelectorTest(unittest.TestCase):
    """The selector combines kinematic evidence with what the base wrench needs."""

    def setUp(self) -> None:
        self.record = robot()
        self.joints = self.record.graph.num_joints

    def select(self, q, qd, base):
        steps = q.shape[0]
        return select_contacts(self.record, q, qd, torch.zeros_like(q), GRAVITY.expand(steps, 3), base, self.record.actuator)

    def test_standing_on_four_feet(self) -> None:
        """Four low, still feet carrying the weight are all chosen, confidently."""
        q = standing(self.record)
        contacts, residual, confidence = self.select(q, torch.zeros_like(q), track(3))
        self.assertTrue(bool(contacts.all()))
        self.assertLess(float(residual.max()), 0.05)
        self.assertTrue(bool((confidence > 3).all()))

    def test_flight_has_no_contacts(self) -> None:
        """In free fall the base needs no wrench, and a foot that carries nothing is not a contact."""
        q = standing(self.record)
        contacts, residual, _ = self.select(q, torch.zeros_like(q), track(3, linear_acceleration=GRAVITY))
        self.assertFalse(bool(contacts.any()))
        self.assertLess(float(residual.max()), 0.05)

    def test_moving_foot_is_left_out_without_a_base_velocity(self) -> None:
        """Feet on the ground move alike relative to the base; a foot that moves differently is in the air."""
        q = standing(self.record)
        qd = torch.zeros_like(q)
        qd[:, self.record.graph.joint_names.index("RR_calf_joint")] = 5.0
        contacts, residual, _ = self.select(q, qd, track(3, linear_known=False))
        self.assertFalse(bool(contacts[:, 3].any()))
        self.assertLess(float(residual.max()), 0.05)  # the chosen feet carry the weight

    def test_probabilities_follow_height_and_speed(self) -> None:
        """A raised foot is unlikely to be down; equal feet are likely down."""
        q = standing(self.record, 1)
        flat = contact_probabilities(self.record, q, torch.zeros_like(q), GRAVITY[None], track(1))
        self.assertTrue(bool((flat > 0.9).all()))
        lifted = q.clone()
        lifted[:, self.record.graph.joint_names.index("FL_calf_joint")] = -1.5  # tuck the front-left foot
        raised = contact_probabilities(self.record, lifted, torch.zeros_like(q), GRAVITY[None], track(1))
        self.assertLess(float(raised[0, 0]), 0.1)

    def test_rejects_robots_without_feet(self) -> None:
        """A fixed base or a robot without contact links has nothing to select."""
        fixed = EmbodimentRecord.build("arm", EmbodimentGraph.from_urdf(quadruped(), floating_base=False), actuator_specs(self.record.graph))
        q = standing(self.record)
        with self.assertRaises(ValueError):
            select_contacts(fixed, q, q, q, GRAVITY.expand(3, 3), track(3))

    def test_evaluation_metrics(self) -> None:
        """Precision, recall, whole-set accuracy and the usable-frame accuracy."""
        reference = torch.tensor([[True, False], [True, True], [False, False]])
        predicted = torch.tensor([[True, True], [True, True], [False, False]])
        metrics = evaluate_contacts(predicted, reference, torch.tensor([True, True, False]))
        self.assertAlmostEqual(metrics["precision"], 3 / 4)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["set_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["usable_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["usable_set_accuracy"], 0.5)


@unittest.skipUnless(HAS_MUJOCO, "needs mujoco (uv sync --extra sim)")
class InferredContactLabelTest(unittest.TestCase):
    """Inferred contacts in the labels, against MuJoCo's contacts."""

    @classmethod
    def setUpClass(cls) -> None:
        from model.training.sources import mujoco_source

        cls.folder = tempfile.TemporaryDirectory()
        path = Path(cls.folder.name, "quad.urdf")
        path.write_text(quadruped())
        cls.source = {"type": "mujoco", "name": "quad", "urdf": str(path), "contact_links": FEET, "nominal_pose": STAND,
                      "randomized_embodiments": 0, "episodes_per_embodiment": 2, "steps": 150, "probe_torques": 0, "gain_scale": 3.0,
                      "target_amplitude": 0.05, "push_probability": 0.0}
        records, cls.trajectories = mujoco_source(cls.source, None, JointFeatureConfig())
        cls.record = records[0]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def hidden(self, trajectory):
        return replace(trajectory, contacts=None, probe_torques=None, probe_executed=None, probe_accelerations=None, probe_base_accelerations=None)

    def test_confident_frames_have_the_simulator_contacts(self) -> None:
        """Frames that keep response labels have MuJoCo's contact set almost always."""
        right, usable = 0, 0
        for trajectory in self.trajectories:
            episode = process_episode(self.record, self.hidden(trajectory), proposal_variants=0)
            self.assertTrue(bool(episode["contacts_detected"].all()))
            keep = episode["response_valid"]
            usable += int(keep.sum())
            right += int((episode["contact_flags"] == trajectory.contacts).all(-1)[keep].sum())
        self.assertGreater(usable, 20)
        self.assertGreater(right / usable, 0.9)

    def test_switches(self) -> None:
        """Inferred contacts can be kept out of the response labels; both methods run."""
        trajectory = self.hidden(self.trajectories[0])
        off = process_episode(self.record, trajectory, proposal_variants=0, detected_contact_response=False)
        self.assertFalse(bool(off["response_valid"].any()))
        strict = process_episode(self.record, trajectory, proposal_variants=0, min_contact_confidence=float("inf"))
        self.assertFalse(bool(strict["response_valid"].any()))
        height = process_episode(self.record, trajectory, proposal_variants=0, contact_detection="height")
        self.assertTrue(bool(height["response_valid"].any()))
        with self.assertRaises(ValueError):
            process_episode(self.record, trajectory, proposal_variants=0, contact_detection="guess")
        # Given contacts ignore the detection settings.
        given = process_episode(self.record, self.trajectories[0], proposal_variants=0, detected_contact_response=False)
        self.assertTrue(bool(given["response_valid"].any()))

    def test_evaluation_script(self) -> None:
        """The script reports both methods for a data config with simulator contacts."""
        import contextlib
        import io
        import sys

        path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_contact_detection.py"
        spec = importlib.util.spec_from_file_location("evaluate_contact_detection", path)
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(script)
        config = Path(self.folder.name, "data.json")
        config.write_text(json.dumps({"sources": [{**self.source, "episodes_per_embodiment": 1, "steps": 60}]}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            script.main(["--data_config", str(config)])
        lines = [line for line in output.getvalue().splitlines() if line.startswith("quad")]
        self.assertEqual([line.split()[1] for line in lines], ["height", "dynamics"])
        self.assertIn("usable set", lines[0])


if __name__ == "__main__":
    unittest.main()
