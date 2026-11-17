"""Test the deployment runner ``DaGetBfmPolicy`` against the training path."""

import collections
import traceback
import unittest

import torch
from torch.overrides import TorchFunctionMode

from model.rl import DaGetBfmPolicy
from model.rl.deployment import StaticInputs
from tests.test_fb_cpr import build_agent, observe, random_state

# Calls that wait for a GPU or copy from the host; a CUDA Graph cannot record them.
HOST_CALLS = {
    "item", "__bool__", "__int__", "__float__", "__index__", "tolist", "nonzero", "masked_select", "equal",
    "allclose", "cpu", "numpy", "unique", "tensor", "as_tensor", "from_numpy",
}


class HostCallRecorder(TorchFunctionMode):
    """Record calls that a CUDA Graph capture would reject, with the model line that made them."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: collections.Counter = collections.Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        flagged = name in HOST_CALLS
        if name in ("__getitem__", "__setitem__"):
            index = args[1] if isinstance(args[1], tuple) else (args[1],)
            flagged = any(isinstance(item, torch.Tensor) and item.dtype == torch.bool for item in index)
        if name in ("to", "cuda"):
            flagged = any(isinstance(item, (str, torch.device)) for item in (*args[1:], *kwargs.values()))
        if flagged:
            frames = [frame for frame in traceback.extract_stack()[:-1] if "/model/" in frame.filename]
            where = f"{frames[-1].filename.rsplit('/model/', 1)[1]}:{frames[-1].lineno}" if frames else "?"
            self.calls[(name, where)] += 1
        return func(*args, **kwargs)


class DeploymentTest(unittest.TestCase):
    """The runner matches ``DaGetFbCprAgent`` and records nothing a CUDA Graph rejects."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.agent, _, _ = build_agent()
        self.robot = 1
        self.observer = self.agent.observers[self.robot]
        self.state = random_state(self.observer.record, 3)
        self.observation = observe(self.observer, self.state)
        self.z = self.agent.model.sample_z(3)
        self.policy = DaGetBfmPolicy(self.agent.model, self.observer, batch_size=3)

    def act(self) -> torch.Tensor:
        item = self.observation
        return self.policy.act(item.local, item.global_features, item.dynamics, item.mass, self.z)

    def test_matches_the_training_path(self) -> None:
        """Actions equal the actor's mean and B equals the agent's, including after new weights."""
        torch.testing.assert_close(self.act(), self.agent.act(self.robot, self.observation, self.z, mean=True))
        torch.testing.assert_close(self.policy.backward(self.observation.state), self.agent._backward(self.robot, self.observation.state))
        with torch.no_grad():
            for parameter in self.agent.model.parameters():
                parameter.add_(0.01 * torch.randn_like(parameter))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.act()
        self.policy.refresh()
        torch.testing.assert_close(self.act(), self.agent.act(self.robot, self.observation, self.z, mean=True))

    def test_recorded_steps_make_no_host_calls(self) -> None:
        """What a CUDA Graph records neither waits for the device nor copies from the host."""
        self.act()
        self.policy.backward(self.observation.state)
        recorder = HostCallRecorder()
        with recorder, torch.no_grad():
            self.policy._run_actor()
            self.policy._run_backward()
        self.assertEqual(dict(recorder.calls), {})
        # The recorder does see such calls: the training path checks inputs on the device.
        with recorder, torch.no_grad():
            self.agent.act(self.robot, self.observation, self.z, mean=True)
        self.assertTrue(recorder.calls)

    def test_inputs_are_checked_before_the_copy(self) -> None:
        """A malformed mass matrix or a wrong shape fails on the CPU, as the encoder's checks would."""
        item = self.observation
        with self.assertRaisesRegex(ValueError, "mass_matrix"):
            self.policy.act(item.local, item.global_features, item.dynamics, -item.mass, self.z)
        with self.assertRaisesRegex(ValueError, "z must have shape"):
            self.policy.act(item.local, item.global_features, item.dynamics, item.mass, self.z[:1])
        with self.assertRaisesRegex(ValueError, "expected inputs"):
            self.policy.policy_inputs.fill(local=item.local)

    def test_static_inputs_share_one_buffer(self) -> None:
        """Each named input is a view of one flat buffer, in order."""
        inputs = StaticInputs({"a": (2, 3), "b": (4,)}, torch.device("cpu"))
        inputs.fill(a=torch.ones(2, 3), b=torch.full((4,), 2.0))
        torch.testing.assert_close(inputs.flat, torch.tensor([1.0] * 6 + [2.0] * 4))

    def test_cuda_graph_needs_cuda(self) -> None:
        """Asking for CUDA Graphs on a CPU is an error."""
        with self.assertRaisesRegex(ValueError, "CUDA"):
            DaGetBfmPolicy(self.agent.model, self.observer, cuda_graph=True)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
    def test_cuda_graph_replay_matches_eager(self) -> None:
        """A replayed graph gives the eager actions, and follows new inputs."""
        self.agent.model.cuda()
        graph = DaGetBfmPolicy(self.agent.model, self.observer, "cuda", batch_size=3)
        eager = DaGetBfmPolicy(self.agent.model, self.observer, "cuda", batch_size=3, cuda_graph=False)
        item = self.observation
        for scale in (1.0, 0.5):
            args = (item.local * scale, item.global_features, item.dynamics, item.mass, self.z)
            torch.testing.assert_close(graph.act(*args), eager.act(*args), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(graph.backward(item.state), eager.backward(item.state), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
