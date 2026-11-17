"""Run a trained DA-GET BFM on one robot at control rate.

``DaGetBfmPolicy`` holds every per-step input in one device buffer, so a control step
makes one host-to-device copy and one device-to-host copy. On a GPU it records the
encoder and actor (and, separately, the encoder and B) once as CUDA Graphs
(NVIDIA, "CUDA Graphs", CUDA C++ Programming Guide; ``torch.cuda.CUDAGraph``) and
replays them each step: at batch 1 the networks are bound by kernel launches, which a
replay issues as one. Input values are checked on the CPU before the copy, because a
check on the device would wait for it, which a graph cannot record.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from model.modeling_bfm import DaGetBfmModel
from model.modeling_da_get import normalize_mass_matrix
from model.physics_schema import DYNAMICS_FEATURE_SIZE
from model.rl.observation import STATE_GLOBAL_SIZE, STATE_LOCAL_SIZE, VARIABLE_LOCAL_SIZE, EmbodimentObserver, StateView
from model.training.episodes import BASE_FEATURE_SIZE

WARMUP_RUNS = 3


class StaticInputs:
    """Named views of one flat device buffer, filled through one copy from pinned host memory.

    Args:
        shapes (dict[str, tuple[int, ...]]): Shape of each input.
        device (torch.device): Device of the networks.
    """

    def __init__(self, shapes: dict[str, tuple[int, ...]], device: torch.device) -> None:
        total = sum(math.prod(shape) for shape in shapes.values())
        self.device = device
        self.flat = torch.zeros(total, device=device)
        self.host = torch.zeros(total, pin_memory=True) if device.type == "cuda" else self.flat
        self.views, self.host_views = {}, {}
        offset = 0
        for name, shape in shapes.items():
            size = math.prod(shape)
            self.views[name] = self.flat[offset:offset + size].view(shape)
            self.host_views[name] = self.host[offset:offset + size].view(shape)
            offset += size

    def fill(self, **values: torch.Tensor) -> None:
        """Copy CPU tensors into the device buffer (every input must be given)."""
        if values.keys() != self.views.keys():
            raise ValueError(f"expected inputs {sorted(self.views)}, got {sorted(values)}")
        for name, value in values.items():
            if value.shape != self.views[name].shape:
                raise ValueError(f"{name} must have shape {tuple(self.views[name].shape)}, got {tuple(value.shape)}")
            self.host_views[name].copy_(value)
        if self.host is not self.flat:
            self.flat.copy_(self.host, non_blocking=True)


class DaGetBfmPolicy:
    """Deployment runner of one robot: actions from observations, and B of reference states.

    Args:
        model (DaGetBfmModel): Trained model, used in eval mode.
        observer (EmbodimentObserver): The robot's observer, as in training.
        device (torch.device | str): Device of the model.
        batch_size (int): Environments (robots) per call.
        cuda_graph (bool | None): Replay CUDA Graphs; defaults to on for a CUDA device.

    After loading new weights, call ``refresh``: the physics memory and graphs hold the
    old ones.
    """

    def __init__(
        self,
        model: DaGetBfmModel,
        observer: EmbodimentObserver,
        device: torch.device | str = "cpu",
        batch_size: int = 1,
        cuda_graph: bool | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.observer = observer
        self.batch_size = batch_size
        self.cuda_graph = self.device.type == "cuda" if cuda_graph is None else cuda_graph
        if self.cuda_graph and self.device.type != "cuda":
            raise ValueError("cuda_graph needs a CUDA device")
        joints, history = observer.graph.num_joints, observer.history_size
        encoder = model.encoder.config
        self.uses_mass = encoder.use_mass_encoding
        self.uses_physics = model.encoder.physics_tokenizers is not None
        self.structure = observer.structure_inputs(batch_size, self.device)
        self._host_mask = torch.ones(batch_size, joints, dtype=torch.bool)
        policy = {
            "local": (batch_size, history, joints, VARIABLE_LOCAL_SIZE),
            "global_features": (batch_size, history, BASE_FEATURE_SIZE),
            "dynamics": (batch_size, joints, DYNAMICS_FEATURE_SIZE),
            "mass": (batch_size, joints, joints),
            "z": (batch_size, model.config.z_dim),
        }
        state = {
            "local": (batch_size, joints, STATE_LOCAL_SIZE),
            "global_features": (batch_size, STATE_GLOBAL_SIZE),
            "dynamics": (batch_size, joints, DYNAMICS_FEATURE_SIZE),
        }
        self.policy_inputs = StaticInputs(policy, self.device)
        self.state_inputs = StaticInputs(state, self.device)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the embodiment-only physics memory and drop recorded graphs (after a weight change)."""
        structure = self.structure
        with torch.no_grad():
            self.physics_memory = (
                self.model.encoder.build_physics_memory(
                    structure["rigid_body_features"], structure["actuator_features"], structure["joint_mask"]
                )
                if self.uses_physics
                else None
            )
        self._act: Callable[[], torch.Tensor] | None = None
        self._latent: Callable[[], torch.Tensor] | None = None

    def _encoder_inputs(self) -> dict:
        """Return the static encoder inputs: graph structure and physics memory."""
        inputs = {key: value for key, value in self.structure.items() if key not in ("rigid_body_features", "actuator_features")}
        if self.uses_physics:
            inputs["physics_memory"] = self.physics_memory
        return inputs

    def _run_actor(self) -> torch.Tensor:
        """Encoder and actor mean on the static inputs; what a graph records."""
        views = self.policy_inputs.views
        embeddings = self.model.encoder(
            input_values=self.observer.policy_tokens(views["local"], views["global_features"]),
            mass_matrix=views["mass"] if self.uses_mass else None,
            dynamics_features=views["dynamics"] if self.uses_physics else None,
            validate_inputs=False,
            **self._encoder_inputs(),
        )
        return self.model.actor(embeddings, views["z"], self.model.config.actor_std, self.structure["joint_mask"]).mean

    def _run_backward(self) -> torch.Tensor:
        """State encoder and B on the static state inputs; what a graph records."""
        views = self.state_inputs.views
        inputs = self.observer.state_inputs(StateView(views["local"], views["global_features"], views["dynamics"]))
        inputs = {**inputs, **self._encoder_inputs()}
        embeddings = self.model.encode_state(inputs, validate_inputs=False)
        return self.model.backward_map(embeddings, inputs["joint_mask"])

    def _prepare(self, function: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
        """Return ``function``, or a replay of it recorded as a CUDA Graph."""
        if not self.cuda_graph:
            return function
        current = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(current)
        # Warm up on a side stream (as PyTorch's CUDA Graphs notes ask), so that lazy
        # initializations such as cuBLAS workspaces are not recorded.
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(WARMUP_RUNS):
                function()
        current.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            output = function()

        def replay() -> torch.Tensor:
            graph.replay()
            return output

        return replay

    def _check_mass(self, mass: torch.Tensor) -> None:
        """Validate the mass matrix on the CPU (the graph runs with checks off)."""
        if self.uses_mass:
            normalize_mass_matrix(mass.float(), self._host_mask, self.model.encoder.config.mass_epsilon)

    @torch.no_grad()
    def act(self, local: torch.Tensor, global_features: torch.Tensor, dynamics: torch.Tensor, mass: torch.Tensor,
            z: torch.Tensor) -> torch.Tensor:
        """Return the mean action of each environment, on the CPU.

        Args:
            local (torch.Tensor): ``[N, H, J, 5]`` policy joint history (``Observation.local``).
            global_features (torch.Tensor): ``[N, H, 11]`` policy base history.
            dynamics (torch.Tensor): ``[N, J, 7]`` dynamics features
                (``EmbodimentObserver.policy_physics``).
            mass (torch.Tensor): ``[N, J, J]`` mass matrix (same).
            z (torch.Tensor): ``[N, z_dim]`` task latent.

        Returns:
            torch.Tensor: ``[N, J]`` actions in [-1, 1].
        """
        self._check_mass(mass)
        self.policy_inputs.fill(local=local, global_features=global_features, dynamics=dynamics, mass=mass, z=z)
        if self._act is None:
            self._act = self._prepare(self._run_actor)
        return self._act().cpu()

    @torch.no_grad()
    def backward(self, view: StateView) -> torch.Tensor:
        """Return raw ``B(s)`` ``[N, z_dim]`` of reference states, on the CPU.

        Goal reaching uses ``model.project_z(B(s_goal))``; tracking a motion streamed
        live uses the projected mean of B over the next ``seq_length`` frames.

        Args:
            view (StateView): ``N`` state frames (``EmbodimentObserver.state_view``).

        Returns:
            torch.Tensor: Backward embeddings.
        """
        self.state_inputs.fill(local=view.local, global_features=view.global_features, dynamics=view.dynamics)
        if self._latent is None:
            self._latent = self._prepare(self._run_backward)
        return self._latent().cpu()
