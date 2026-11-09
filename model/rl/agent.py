"""Cross-embodiment FB-CPR agent: the BFM-Zero update on per-DoF networks.

The losses, targets, z mixture and rollout logic follow ``FBAgent``, ``FBcprAgent`` and
``FBcprAuxAgent`` of BFM-Zero (https://github.com/LeCAR-Lab/BFM-Zero; Copyright (c) Meta
Platforms, Inc. and affiliates; CC BY-NC 4.0; Li et al., arXiv:2511.04131, 2025, and
Tirinzoni et al., ICLR 2025). See ``THIRD_PARTY_NOTICES.md``.

What changes for several robots:

* Each batch holds an equal share of every robot. Networks run per robot on the graph of
  that robot, and the results are padded and joined.
* B, D and the critics' privileged context read state tokens, so they fit any robot.
* Goal and expert prompts are only paired with transitions of the same robot, each robot
  has its own z buffer, and F and B are compared only within a robot, because a state of
  another robot is unreachable.
* The policy encoder keeps training: the F and critic losses and the physics auxiliary
  losses of pretraining update it together, with a target encoder for the Bellman targets.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import autograd, nn

from model.configuration_bfm import BfmConfig
from model.modeling_bfm import DaGetBfmModel
from model.pretraining import DaGetForPhysicsPretraining
from model.rl.buffers import ExpertMotions, ReplayBuffer
from model.rl.configuration_fb_cpr import FbCprConfig
from model.rl.observation import EmbodimentObserver, Observation, StateView, pad_joints

LATENT_CHUNK = 4096


def ensemble_pessimism(values: torch.Tensor, penalty: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the mean, the mean pairwise distance, and ``mean - penalty * distance`` over dim 0."""
    members = values.shape[0]
    difference = torch.abs(values.unsqueeze(0) - values.unsqueeze(1)).sum(dim=(0, 1)) / (members**2 - members)
    mean = values.mean(dim=0)
    return mean, difference, mean - penalty * difference


def _off_diagonal(size: int, reference: torch.Tensor, pairs: torch.Tensor | None) -> torch.Tensor:
    """Return the float mask of compared off-diagonal entries, optionally limited to ``pairs``."""
    mask = 1 - torch.eye(size, device=reference.device, dtype=reference.dtype)
    return mask if pairs is None else mask * pairs.to(reference.dtype)


def fb_bellman_loss(measures: torch.Tensor, discount: torch.Tensor, target: torch.Tensor,
                    pairs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the FB loss with its diagonal and off-diagonal terms.

    Args:
        measures (torch.Tensor): ``F(s_i, a_i, z_i)ᵀ B(s'_j)``, ``[P, B, B]``.
        discount (torch.Tensor): ``[B, 1]``, zero at terminal transitions.
        target (torch.Tensor): Target ``[B, B]``.
        pairs (torch.Tensor | None): Bool ``[B, B]`` of compared entries; all by default.
    """
    difference = measures - discount * target
    mask = _off_diagonal(measures.shape[-1], measures, pairs)
    off = 0.5 * (difference * mask).pow(2).sum() / mask.sum().clamp_min(1)
    diagonal = -torch.diagonal(difference, dim1=1, dim2=2).mean() * measures.shape[0]
    return off + diagonal, diagonal, off


def orthonormality_loss(embeddings: torch.Tensor, pairs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the loss pushing ``B Bᵀ`` toward a scaled identity, with its two terms."""
    covariance = embeddings @ embeddings.T
    mask = _off_diagonal(covariance.shape[0], covariance, pairs)
    diagonal = -covariance.diag().mean()
    off = 0.5 * (covariance * mask).pow(2).sum() / mask.sum().clamp_min(1)
    return off + diagonal, diagonal, off


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Move target parameters toward the source by ``tau``."""
    with torch.no_grad():
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.mul_(1 - tau).add_(parameter, alpha=tau)


def split_counts(total: int, groups: int) -> list[int]:
    """Split ``total`` into ``groups`` near-equal integers."""
    return [total // groups + (index < total % groups) for index in range(groups)]


class EmaRewardNormalizer(nn.Module):
    """Divide rewards by the running std of the batch mean and square (BFM-Zero's ``EMA``, ``scale=True``)."""

    def __init__(self, tau: float = 0.99, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.tau, self.epsilon = tau, epsilon
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("mean_square", torch.zeros(1))
        self.register_buffer("counter", torch.zeros(1, dtype=torch.long))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Update the statistics and return the scaled rewards."""
        self.mean.mul_(self.tau).add_((1 - self.tau) * values.mean())
        self.mean_square.mul_(self.tau).add_((1 - self.tau) * values.pow(2).mean())
        self.counter += 1
        norm = 1 - self.tau ** self.counter.float()
        variance = torch.clamp(self.mean_square / norm - (self.mean / norm).pow(2), min=self.epsilon)
        return values / variance.sqrt()


class ZBuffer:
    """Ring buffer of latents used in training, sampled for rollouts (one per robot)."""

    def __init__(self, capacity: int, dim: int, device: torch.device) -> None:
        self.storage = torch.zeros((capacity, dim), device=device)
        self.capacity, self.position, self.size = capacity, 0, 0

    def add(self, values: torch.Tensor) -> None:
        """Append latents ``[N, Z]``, keeping the newest when ``N`` exceeds the capacity."""
        values = values[-self.capacity :]
        indices = (self.position + torch.arange(values.shape[0], device=values.device)) % self.capacity
        self.storage[indices] = values
        self.position = (self.position + values.shape[0]) % self.capacity
        self.size = min(self.size + values.shape[0], self.capacity)

    def sample(self, count: int) -> torch.Tensor:
        """Return ``count`` stored latents."""
        return self.storage[torch.randint(0, self.size, (count,), device=self.storage.device)].clone()


@dataclass
class TrainingBatch:
    """One training batch, with rows in robot order.

    Attributes:
        robots (torch.Tensor): ``[B]`` robot index of each transition.
        counts (list[int]): Transitions per robot.
        joint_mask (torch.Tensor): Bool ``[B, J]``.
        observations (list[Observation]): Per robot, at s.
        next_observations (list[Observation]): Per robot, at s'.
        states (list[dict]): Per robot, state-encoder inputs at s.
        next_states (list[dict]): Per robot, state-encoder inputs at s'.
        context (torch.Tensor): ``[B, J, S]`` padded state tokens at s.
        next_context (torch.Tensor): ``[B, J, S]`` padded state tokens at s'.
        action (torch.Tensor): ``[B, J]``.
        z (torch.Tensor): ``[B, Z]`` latent used in the rollout.
        discount (torch.Tensor): ``[B, 1]``, zero at terminal transitions.
        aux_rewards (torch.Tensor): ``[B, K]`` unscaled auxiliary rewards.
    """

    robots: torch.Tensor
    counts: list[int]
    joint_mask: torch.Tensor
    observations: list[Observation]
    next_observations: list[Observation]
    states: list[dict]
    next_states: list[dict]
    context: torch.Tensor
    next_context: torch.Tensor
    action: torch.Tensor
    z: torch.Tensor
    discount: torch.Tensor
    aux_rewards: torch.Tensor


class DaGetFbCprAgent:
    """FB-CPR with auxiliary rewards, updating ``DaGetBfmModel`` and the encoder across embodiments.

    Args:
        model_config (BfmConfig): Networks.
        config (FbCprConfig): Training hyperparameters.
        pretrained (DaGetForPhysicsPretraining): Policy encoder with its physics heads.
        observers (list[EmbodimentObserver]): Robots.
        device (torch.device | str): Device of the networks.
    """

    def __init__(self, model_config: BfmConfig, config: FbCprConfig, pretrained: DaGetForPhysicsPretraining,
                 observers: list[EmbodimentObserver], device: torch.device | str = "cpu") -> None:
        if config.batch_size % model_config.seq_length:
            raise ValueError("batch_size must be a multiple of the model's seq_length")
        if model_config.num_parallel < 2:
            raise ValueError("num_parallel must be at least 2: the targets use the ensemble's uncertainty")
        if config.batch_size // model_config.seq_length < len(observers):
            raise ValueError("batch_size is too small: every robot needs an expert window per update")
        self.model_config, self.config = model_config, config
        self.device = torch.device(device)
        self.observers = observers
        self.robots = len(observers)
        self.pretrained = pretrained.to(self.device).eval().requires_grad_(config.train_encoder)
        self.target_pretrained = copy.deepcopy(self.pretrained).eval().requires_grad_(False)
        self.model = DaGetBfmModel(model_config).to(self.device)
        self.model.init_weights()
        self.target_forward_map = copy.deepcopy(self.model.forward_map).requires_grad_(False)
        self.target_backward_map = copy.deepcopy(self.model.backward_map).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.model.critic).requires_grad_(False)
        self.target_aux_critic = copy.deepcopy(self.model.aux_critic).requires_grad_(False)
        self.aux_normalizer = EmaRewardNormalizer().to(self.device)
        model = self.model
        decay = config.weight_decay
        self.optimizers = {
            "forward": torch.optim.Adam(model.forward_map.parameters(), lr=config.lr_f, weight_decay=decay),
            "backward": torch.optim.Adam(model.backward_map.parameters(), lr=config.lr_b, weight_decay=decay),
            "actor": torch.optim.Adam(model.actor.parameters(), lr=config.lr_actor, weight_decay=decay),
            "critic": torch.optim.Adam(model.critic.parameters(), lr=config.lr_critic, weight_decay=decay),
            "aux_critic": torch.optim.Adam(model.aux_critic.parameters(), lr=config.lr_aux_critic, weight_decay=decay),
            "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=config.lr_discriminator),
        }
        if config.train_encoder:
            encoder = {id(parameter) for parameter in self.pretrained.encoder.parameters()}
            heads = [parameter for parameter in self.pretrained.parameters() if id(parameter) not in encoder]
            self.optimizers["encoder"] = torch.optim.Adam(self.pretrained.encoder.parameters(), lr=config.lr_encoder)
            self.optimizers["physics_heads"] = torch.optim.Adam(heads, lr=config.lr_physics_heads)
        self.z_buffers = [ZBuffer(config.z_buffer_size, model_config.z_dim, self.device) for _ in range(self.robots)]
        self._tracking: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.reward_names = tuple(config.aux_reward_scales)
        self._reward_scales = torch.tensor([config.aux_reward_scales[name] for name in self.reward_names], device=self.device)

    def _embed(self, model: DaGetForPhysicsPretraining, robot: int, observation: Observation, grad: bool = False) -> torch.Tensor:
        """Return joint embeddings ``[N, J, E]`` of one robot's observation."""
        observer = self.observers[robot]
        local = observation.local.to(self.device)
        structure = observer.structure_inputs(local.shape[0], self.device)
        spatial, ancestors, descendants = structure.pop("graph_distances")
        with torch.set_grad_enabled(grad):
            return model.encode(
                input_values=observer.policy_tokens(local, observation.global_features.to(self.device)),
                spatial_distances=spatial,
                ancestor_distances=ancestors,
                descendant_distances=descendants,
                mass_matrix=observation.mass.to(self.device),
                dynamics_features=observation.dynamics.to(self.device),
                **structure,
            )

    def _state_inputs(self, robot: int, view: StateView) -> dict:
        """Return state-encoder inputs of one robot's frames on the agent's device."""
        moved = StateView(view.local.to(self.device), view.global_features.to(self.device), view.dynamics.to(self.device))
        return self.observers[robot].state_inputs(moved)

    @torch.no_grad()
    def act(self, robot: int, observation: Observation, z: torch.Tensor, mean: bool = False) -> torch.Tensor:
        """Return per-DoF actions ``[N, J]`` for one robot's environments."""
        embeddings = self._embed(self.pretrained, robot, observation)
        distribution = self.model.actor(embeddings, z.to(self.device), self.model_config.actor_std)
        return distribution.mean if mean else distribution.sample()

    @torch.no_grad()
    def _backward(self, robot: int, view: StateView) -> torch.Tensor:
        """Return raw ``B(s)`` of one robot's frames ``[N, ...]``, in chunks."""
        chunks = []
        for start in range(0, view.local.shape[0], LATENT_CHUNK):
            indices = torch.arange(start, min(start + LATENT_CHUNK, view.local.shape[0]))
            chunks.append(self.model.backward_map(self._state_inputs(robot, view.select(indices))))
        return torch.cat(chunks)

    @torch.no_grad()
    def tracking_latents(self, robot: int, view: StateView) -> torch.Tensor:
        """Return ``z_t`` for a motion ``[T, ...]``: the projected mean of B over its next ``seq_length`` frames."""
        embeddings = self._backward(robot, view.select(torch.arange(1, view.local.shape[0])))
        prefix = torch.cat([embeddings.new_zeros(1, embeddings.shape[-1]), embeddings.cumsum(0)])
        starts = torch.arange(embeddings.shape[0], device=embeddings.device)
        ends = torch.clamp(starts + self.model_config.seq_length, max=embeddings.shape[0])
        return self.model.project_z((prefix[ends] - prefix[starts]) / (ends - starts)[:, None].to(embeddings.dtype))

    def goal_latent(self, robot: int, view: StateView) -> torch.Tensor:
        """Return ``z`` of goal frames ``[N, ...]``."""
        return self.model.project_z(self._backward(robot, view))

    @torch.no_grad()
    def _sample_tracking_z(self, robot: int, motions: ExpertMotions, count: int, length: int) -> torch.Tensor:
        """Encode ``count`` expert windows into per-step latents ``[count, length, Z]``."""
        _, following = motions.sample_windows(count, length)
        flat = StateView(following.local.flatten(0, 1), following.global_features.flatten(0, 1), following.dynamics.flatten(0, 1))
        embeddings = self._backward(robot, flat).view(count, length, -1)
        window = self.model_config.seq_length
        averaged = torch.stack([embeddings[:, step : min(step + window, length)].mean(dim=1) for step in range(length)], dim=1)
        return self.model.project_z(averaged)

    @torch.no_grad()
    def rollout_context(self, robot: int, z: torch.Tensor | None, steps: torch.Tensor, motions: ExpertMotions) -> torch.Tensor:
        """Return the latent of each of one robot's environments for this control step.

        Args:
            robot (int): Robot index.
            z (torch.Tensor | None): ``[N, Z]`` current latents; None at the start.
            steps (torch.Tensor): ``[N]`` control steps since each episode began.
            motions (ExpertMotions): The robot's motions, for expert rollouts.

        Returns:
            torch.Tensor: ``[N, Z]`` latents.
        """
        config = self.config
        steps = steps.to(self.device)
        count = steps.shape[0]
        length = config.rollout_expert_trajectories_length
        chosen = int(config.rollout_expert_trajectories_percentage * count) if config.rollout_expert_trajectories else 0
        if z is None:
            z = self.model.sample_z(count, self.device)
            start_tracking = chosen > 0
        else:
            buffer = self.z_buffers[robot]
            mixed = config.use_mix_rollout and buffer.size > 0
            fresh = buffer.sample(count) if mixed else self.model.sample_z(count, self.device)
            z = torch.where((steps % config.update_z_every_step == 0)[:, None], fresh, z.to(self.device))
            start_tracking = chosen > 0 and bool(((steps % length) == 0).any())
        if start_tracking:
            environments = torch.randperm(count, device=self.device)[:chosen]
            self._tracking[robot] = (environments, self._sample_tracking_z(robot, motions, chosen, length))
        if chosen > 0 and robot in self._tracking:
            environments, tracking = self._tracking[robot]
            times = (steps % length)[environments]
            z[environments] = tracking[torch.arange(chosen, device=self.device), times]
        return z

    def _padded_embeddings(self, model: DaGetForPhysicsPretraining, observations: list[Observation], joints: int, grad: bool = False) -> torch.Tensor:
        """Encode each robot and join the results, padded to ``joints``."""
        return torch.cat([pad_joints(self._embed(model, robot, observation, grad), joints) for robot, observation in enumerate(observations)])

    def _training_batch(self, replays: list[ReplayBuffer], counts: list[int]) -> TrainingBatch:
        """Sample ``counts`` transitions of each robot and join them in robot order."""
        samples = [replay.sample(count, self.device) for replay, count in zip(replays, counts)]
        joints = max(sample["action"].shape[1] for sample in samples)
        robots, masks, actions = [], [], []
        for robot, sample in enumerate(samples):
            count, robot_joints = sample["action"].shape[:2]
            robots.append(torch.full((count,), robot, device=self.device))
            mask = torch.zeros((count, joints), dtype=torch.bool, device=self.device)
            mask[:, :robot_joints] = True
            masks.append(mask)
            actions.append(pad_joints(sample["action"], joints))
        states = [self._state_inputs(robot, sample["observation"].state) for robot, sample in enumerate(samples)]
        next_states = [self._state_inputs(robot, sample["next_observation"].state) for robot, sample in enumerate(samples)]

        def context(inputs: list[dict]) -> torch.Tensor:
            return torch.cat([pad_joints(item["input_values"], joints) for item in inputs])

        def join(key: str) -> torch.Tensor:
            return torch.cat([sample[key] for sample in samples])

        return TrainingBatch(
            torch.cat(robots), counts, torch.cat(masks), [sample["observation"] for sample in samples],
            [sample["next_observation"] for sample in samples], states, next_states, context(states), context(next_states),
            torch.cat(actions), join("z"), self.config.discount * (~join("terminated")[:, None]).float(), join("aux_rewards"),
        )

    def _expert_batch(self, motions: list[ExpertMotions], windows: list[int]) -> tuple[list[dict], list[dict]]:
        """Sample expert windows: each robot's state inputs at s and s', ``[W L, ...]``."""
        length = self.model_config.seq_length
        states, next_states = [], []
        for robot, (expert, count) in enumerate(zip(motions, windows)):
            current, following = expert.sample_windows(count, length)

            def flat(view: StateView) -> StateView:
                return StateView(view.local.flatten(0, 1), view.global_features.flatten(0, 1), view.dynamics.flatten(0, 1))

            states.append(self._state_inputs(robot, flat(current)))
            next_states.append(self._state_inputs(robot, flat(following)))
        return states, next_states

    @torch.no_grad()
    def encode_expert(self, next_states: list[dict]) -> torch.Tensor:
        """Return per-frame ``z`` in robot order: the projected mean of B over each expert window."""
        length = self.model_config.seq_length
        embeddings = torch.cat([self.model.backward_map(inputs) for inputs in next_states])
        windows = embeddings.view(-1, length, self.model_config.z_dim)
        return torch.repeat_interleave(self.model.project_z(windows.mean(dim=1)), length, dim=0)

    @torch.no_grad()
    def sample_mixed_z(self, robots: torch.Tensor, goals: torch.Tensor, expert_z: torch.Tensor, expert_robots: torch.Tensor) -> torch.Tensor:
        """Sample training latents: replay goals, expert windows or uniform, always from the same robot.

        Args:
            robots (torch.Tensor): ``[B]`` robot of each transition.
            goals (torch.Tensor): ``[B, Z]`` projected ``B(s')`` of the batch.
            expert_z (torch.Tensor): ``[B, Z]`` expert latents.
            expert_robots (torch.Tensor): ``[B]`` robot of each expert latent.

        Returns:
            torch.Tensor: ``[B, Z]`` latents.
        """
        config = self.config
        count = robots.shape[0]
        z = self.model.sample_z(count, self.device)
        goal_ratio, expert_ratio = config.train_goal_ratio, config.expert_asm_ratio
        probabilities = torch.tensor([goal_ratio, expert_ratio, 1 - goal_ratio - expert_ratio], device=self.device)
        mix = torch.multinomial(probabilities, count, replacement=True)
        goal_index = torch.zeros(count, dtype=torch.long, device=self.device)
        expert_index = torch.zeros(count, dtype=torch.long, device=self.device)
        for robot in range(self.robots):
            rows = torch.nonzero(robots == robot).squeeze(-1)
            experts = torch.nonzero(expert_robots == robot).squeeze(-1)
            goal_index[rows] = rows[torch.randperm(rows.shape[0], device=self.device)]
            expert_index[rows] = experts[torch.randint(0, experts.shape[0], (rows.shape[0],), device=self.device)]
        z = torch.where((mix == 0)[:, None], goals[goal_index], z)
        return torch.where((mix == 1)[:, None], expert_z[expert_index], z)

    def _discriminator_logits(self, states: list[dict], z: torch.Tensor, counts: list[int]) -> torch.Tensor:
        """Return D's logits ``[B, 1]`` of per-robot state inputs."""
        return torch.cat([self.model.discriminator.compute_logits(inputs, part) for inputs, part in zip(states, z.split(counts))])

    def _gradient_penalty(self, expert: list[dict], expert_z: torch.Tensor, train: list[dict], train_z: torch.Tensor) -> torch.Tensor:
        """Return the WGAN gradient penalty of D's head, between expert and replay samples of each robot.

        BFM-Zero penalizes gradients with respect to its flat input. Here the input of the
        head is the pooled state: second derivatives through attention would be far more
        expensive, and PyTorch's fast CPU attention does not support them.
        """
        encoder = self.model.discriminator.state_encoder
        with torch.no_grad():
            expert_features = torch.cat([encoder(inputs) for inputs in expert])
            train_features = torch.cat([encoder(inputs) for inputs in train])
        alpha = torch.rand(train_z.shape[0], 1, device=self.device)
        features = (alpha * expert_features + (1 - alpha) * train_features).requires_grad_(True)
        latents = (alpha * expert_z + (1 - alpha) * train_z).requires_grad_(True)
        logits = self.model.discriminator.logits_from_features(features, latents)
        gradients = autograd.grad(logits, (features, latents), torch.ones_like(logits), create_graph=True)
        return ((torch.cat(gradients, dim=1).norm(2, dim=1) - 1) ** 2).mean()

    def update_discriminator(self, expert: list[dict], expert_z: torch.Tensor, train: list[dict], train_z: torch.Tensor, counts: list[int]) -> dict[str, torch.Tensor]:
        """Train D to separate expert windows from replay states, each with its z."""
        expert_loss = -F.logsigmoid(self._discriminator_logits(expert, expert_z, counts))
        train_loss = F.softplus(self._discriminator_logits(train, train_z, counts))
        loss = (expert_loss + train_loss).mean()
        metrics = {"disc_expert_loss": expert_loss.mean().detach(), "disc_train_loss": train_loss.mean().detach()}
        if self.config.grad_penalty_discriminator > 0:
            penalty = self._gradient_penalty(expert, expert_z, train, train_z)
            loss = loss + self.config.grad_penalty_discriminator * penalty
            metrics["disc_wgan_gp_loss"] = penalty.detach()
        self.optimizers["discriminator"].zero_grad(set_to_none=True)
        loss.backward()
        self.optimizers["discriminator"].step()
        metrics["disc_loss"] = loss.detach()
        return metrics

    def _sample_action(self, embeddings: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sample a TD3-style clipped action."""
        return self.model.actor(embeddings, z, self.model_config.actor_std, mask).sample(clip=self.config.stddev_clip)

    def _fb_loss(self, batch: TrainingBatch, embeddings: torch.Tensor, next_embeddings: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the FB and orthonormality loss of F and B."""
        mask = batch.joint_mask
        pairs = (batch.robots[:, None] == batch.robots[None, :]) if self.config.within_robot_pairs else None
        with torch.no_grad():
            next_action = self._sample_action(next_embeddings, z, mask)
            target_forward = self.target_forward_map(next_embeddings, z, next_action, mask, context_states=batch.next_context, context_mask=mask)
            target_backward = torch.cat([self.target_backward_map(inputs) for inputs in batch.next_states])
            _, _, target = ensemble_pessimism(target_forward @ target_backward.T, self.config.fb_pessimism_penalty)
        forward = self.model.forward_map(embeddings, z, batch.action, mask, context_states=batch.context, context_mask=mask)
        backward = torch.cat([self.model.backward_map(inputs) for inputs in batch.next_states])
        measures = forward @ backward.T
        fb_loss, fb_diagonal, fb_off = fb_bellman_loss(measures, batch.discount, target, pairs)
        orth_loss, orth_diagonal, orth_off = orthonormality_loss(backward, pairs)
        loss = fb_loss + self.config.ortho_coef * orth_loss
        return loss, {
            "fb_loss": loss.detach(), "fb_diag": fb_diagonal.detach(), "fb_offdiag": fb_off.detach(),
            "orth_loss": orth_loss.detach(), "orth_loss_diag": orth_diagonal.detach(), "orth_loss_offdiag": orth_off.detach(),
            "target_M": target.mean(), "M1": measures[0].mean().detach(), "F1": forward[0].mean().detach(),
            "B_norm": backward.norm(dim=-1).mean().detach(), "z_norm": z.norm(dim=-1).mean(),
        }

    def _value_loss(self, critic: nn.Module, target_critic: nn.Module, penalty: float, reward: torch.Tensor, batch: TrainingBatch,
                    embeddings: torch.Tensor, next_embeddings: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a critic's Bellman loss, target, Q and uncertainty."""
        mask = batch.joint_mask
        with torch.no_grad():
            next_action = self._sample_action(next_embeddings, z, mask)
            next_values = target_critic(next_embeddings, z, next_action, mask, context_states=batch.next_context, context_mask=mask)
            _, uncertainty, next_value = ensemble_pessimism(next_values, penalty)
            target = reward + batch.discount * next_value
        values = critic(embeddings, z, batch.action, mask, context_states=batch.context, context_mask=mask)
        members = values.shape[0]
        loss = 0.5 * members * F.mse_loss(values, target.expand(members, -1, -1))
        return loss, target.mean(), values.mean().detach(), uncertainty.mean()

    def _critic_losses(self, batch: TrainingBatch, embeddings: torch.Tensor, next_embeddings: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return the losses of Q_D (discriminator reward) and Q_R (auxiliary rewards)."""
        config = self.config
        with torch.no_grad():
            reward = torch.cat(
                [self.model.discriminator.compute_reward(inputs, part) for inputs, part in zip(batch.states, z.split(batch.counts))]
            )
        critic_loss, target, value, uncertainty = self._value_loss(
            self.model.critic, self.target_critic, config.critic_pessimism_penalty, reward, batch, embeddings, next_embeddings, z
        )
        aux_reward = self.aux_normalizer((batch.aux_rewards * self._reward_scales).sum(-1, keepdim=True))
        aux_loss, aux_target, aux_value, aux_uncertainty = self._value_loss(
            self.model.aux_critic, self.target_aux_critic, config.aux_critic_pessimism_penalty, aux_reward, batch, embeddings, next_embeddings, z
        )
        metrics = {
            "critic_loss": critic_loss.detach(), "target_Q": target, "Q1": value, "unc_Q": uncertainty, "mean_disc_reward": reward.mean(),
            "aux_critic_loss": aux_loss.detach(), "target_auxQ": aux_target, "auxQ1": aux_value, "unc_auxQ": aux_uncertainty,
            "mean_aux_reward": aux_reward.mean(),
        }
        for index, name in enumerate(self.reward_names):
            metrics[f"aux_rew/{name}"] = batch.aux_rewards[:, index].mean()
        return critic_loss, aux_loss, metrics

    def _physics_loss(self, physics_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the pretraining losses of the encoder and its heads on a collated batch."""
        self.pretrained.train()
        output = self.pretrained(**{key: value.to(self.device) for key, value in physics_batch.items()})
        self.pretrained.eval()
        metrics = {f"physics/loss_{name}": value.detach() for name, value in (output.loss_components or {}).items()}
        metrics["physics/loss"] = output.loss.detach()
        return output.loss, metrics

    def update_value_networks(self, batch: TrainingBatch, z: torch.Tensor, physics_batch: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Train F, B, Q_D, Q_R and the encoder in one backward pass, as they share its graph."""
        config = self.config
        joints = batch.joint_mask.shape[1]
        grad = config.train_encoder and config.encoder_rl_gradients
        embeddings = self._padded_embeddings(self.pretrained, batch.observations, joints, grad)
        with torch.no_grad():
            next_embeddings = self._padded_embeddings(self.target_pretrained, batch.next_observations, joints)
        fb_loss, metrics = self._fb_loss(batch, embeddings, next_embeddings, z)
        critic_loss, aux_loss, critic_metrics = self._critic_losses(batch, embeddings, next_embeddings, z)
        metrics.update(critic_metrics)
        loss = fb_loss + critic_loss + aux_loss
        if physics_batch is not None and config.train_encoder:
            physics_loss, physics_metrics = self._physics_loss(physics_batch)
            loss = loss + config.physics_loss_weight * physics_loss
            metrics.update(physics_metrics)
        names = ["forward", "backward", "critic", "aux_critic"] + (["encoder", "physics_heads"] if config.train_encoder else [])
        for name in names:
            self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        if config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.forward_map.parameters(), config.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.backward_map.parameters(), config.clip_grad_norm)
        for name in names:
            self.optimizers[name].step()
        return metrics

    def update_actor(self, batch: TrainingBatch, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Maximize ``Fᵀz + λ_D Q_D + λ_R Q_R`` with ensemble pessimism; the encoder does not move."""
        config = self.config
        mask = batch.joint_mask
        embeddings = self._padded_embeddings(self.pretrained, batch.observations, mask.shape[1])
        action = self._sample_action(embeddings, z, mask)
        keywords = {"context_states": batch.context, "context_mask": mask}
        q_d = self.model.critic(embeddings, z, action, mask, **keywords)
        q_r = self.model.aux_critic(embeddings, z, action, mask, **keywords)
        forward = self.model.forward_map(embeddings, z, action, mask, **keywords)
        _, _, q_d = ensemble_pessimism(q_d, config.actor_pessimism_penalty)
        _, _, q_r = ensemble_pessimism(q_r, config.actor_pessimism_penalty)
        _, _, q_fb = ensemble_pessimism(self.model.fb_q_value(forward, z), config.actor_pessimism_penalty)
        weight = q_fb.abs().mean().detach() if config.scale_reg else 1.0
        loss = -q_d.mean() * config.reg_coeff * weight - q_r.mean() * config.reg_coeff_aux * weight - q_fb.mean()
        self.optimizers["actor"].zero_grad(set_to_none=True)
        loss.backward()
        if config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), config.clip_grad_norm)
        self.optimizers["actor"].step()
        return {"actor_loss": loss.detach(), "Q_discriminator": q_d.mean().detach(), "Q_aux": q_r.mean().detach(), "Q_fb": q_fb.mean().detach()}

    def update(self, replays: list[ReplayBuffer], motions: list[ExpertMotions], physics_batch: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Run one gradient step of every network, in BFM-Zero's order, then the target updates."""
        config = self.config
        length = self.model_config.seq_length
        windows = split_counts(config.batch_size // length, self.robots)
        counts = [count * length for count in windows]
        batch = self._training_batch(replays, counts)
        expert_states, expert_next = self._expert_batch(motions, windows)
        expert_z = self.encode_expert(expert_next)
        metrics = self.update_discriminator(expert_states, expert_z, batch.states, batch.z, counts)
        with torch.no_grad():
            goals = self.model.project_z(torch.cat([self.model.backward_map(inputs) for inputs in batch.next_states]))
        z = self.sample_mixed_z(batch.robots, goals, expert_z, batch.robots)
        for robot, buffer in enumerate(self.z_buffers):
            buffer.add(z[batch.robots == robot])
        relabel = torch.rand((z.shape[0], 1), device=self.device) <= config.relabel_ratio
        z = torch.where(relabel, z, batch.z)
        metrics.update(self.update_value_networks(batch, z, physics_batch))
        metrics.update(self.update_actor(batch, z))
        soft_update(self.model.forward_map, self.target_forward_map, config.fb_target_tau)
        soft_update(self.model.backward_map, self.target_backward_map, config.fb_target_tau)
        soft_update(self.model.critic, self.target_critic, config.critic_target_tau)
        soft_update(self.model.aux_critic, self.target_aux_critic, config.critic_target_tau)
        if config.train_encoder:
            soft_update(self.pretrained, self.target_pretrained, config.fb_target_tau)
        return metrics

    def state_dict(self) -> dict:
        """Return every trained state, for checkpoints."""
        return {
            "model": self.model.state_dict(),
            "pretrained": self.pretrained.state_dict(),
            "target_pretrained": self.target_pretrained.state_dict(),
            "target_forward_map": self.target_forward_map.state_dict(),
            "target_backward_map": self.target_backward_map.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "target_aux_critic": self.target_aux_critic.state_dict(),
            "aux_normalizer": self.aux_normalizer.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "model_config": asdict(self.model_config),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a state from ``state_dict``."""
        self.model.load_state_dict(state["model"])
        self.pretrained.load_state_dict(state["pretrained"])
        self.target_pretrained.load_state_dict(state["target_pretrained"])
        self.target_forward_map.load_state_dict(state["target_forward_map"])
        self.target_backward_map.load_state_dict(state["target_backward_map"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.target_aux_critic.load_state_dict(state["target_aux_critic"])
        self.aux_normalizer.load_state_dict(state["aux_normalizer"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])

    def save(self, path: str | Path) -> None:
        """Write a checkpoint file."""
        torch.save(self.state_dict(), path)
