"""Cross-embodiment FB-CPR agent: the BFM-Zero update on GET joint embeddings.

The losses, targets, z mixture and rollout logic follow ``FBAgent``, ``FBcprAgent`` and
``FBcprAuxAgent`` of BFM-Zero (https://github.com/LeCAR-Lab/BFM-Zero; Copyright (c) Meta
Platforms, Inc. and affiliates; CC BY-NC 4.0; Li et al., arXiv:2511.04131, 2025, and
Tirinzoni et al., ICLR 2025). See ``THIRD_PARTY_NOTICES.md``.

What changes for several robots:

* Each batch holds an equal share of every robot, encoded per robot by a frozen
  ``DaGetEncoder`` and padded. Actor and critics read the embeddings; critics also read
  the flat state as one privileged context token.
* Goal and expert prompts are only paired with transitions of the same robot, and each
  robot has its own z buffer: a motion of one robot says nothing about another's states.
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
from model.rl.observation import EmbodimentObserver, Observation, pad_joints


def ensemble_pessimism(values: torch.Tensor, penalty: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the mean, the mean pairwise distance, and ``mean - penalty * distance`` over dim 0."""
    members = values.shape[0]
    difference = torch.abs(values.unsqueeze(0) - values.unsqueeze(1)).sum(dim=(0, 1)) / (members**2 - members)
    mean = values.mean(dim=0)
    return mean, difference, mean - penalty * difference


def fb_bellman_loss(measures: torch.Tensor, discount: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the FB loss with its diagonal and off-diagonal terms.

    Args:
        measures (torch.Tensor): ``F(s_i, a_i, z_i)ᵀ B(s'_j)``, ``[P, B, B]``.
        discount (torch.Tensor): ``[B, 1]``, zero at terminal transitions.
        target (torch.Tensor): Target ``[B, B]``.
    """
    difference = measures - discount * target
    off_diagonal = 1 - torch.eye(measures.shape[-1], device=measures.device, dtype=measures.dtype)
    off = 0.5 * (difference * off_diagonal).pow(2).sum() / off_diagonal.sum()
    diagonal = -torch.diagonal(difference, dim1=1, dim2=2).mean() * measures.shape[0]
    return off + diagonal, diagonal, off


def orthonormality_loss(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the loss pushing ``B Bᵀ`` toward a scaled identity, with its two terms."""
    covariance = embeddings @ embeddings.T
    off_diagonal = 1 - torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    diagonal = -covariance.diag().mean()
    off = 0.5 * (covariance * off_diagonal).pow(2).sum() / off_diagonal.sum()
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
    """One padded training batch.

    Attributes:
        robots (torch.Tensor): ``[B]`` robot index of each transition.
        joint_mask (torch.Tensor): Bool ``[B, J]``.
        embeddings (torch.Tensor): ``[B, J, E]`` at s.
        next_embeddings (torch.Tensor): ``[B, J, E]`` at s'.
        flat (torch.Tensor): ``[B, S]`` raw flat state at s.
        next_flat (torch.Tensor): ``[B, S]`` raw flat state at s'.
        action (torch.Tensor): ``[B, J]``.
        z (torch.Tensor): ``[B, Z]`` latent used in the rollout.
        discount (torch.Tensor): ``[B, 1]``, zero at terminal transitions.
        aux_rewards (torch.Tensor): ``[B, K]`` unscaled auxiliary rewards.
    """

    robots: torch.Tensor
    joint_mask: torch.Tensor
    embeddings: torch.Tensor
    next_embeddings: torch.Tensor
    flat: torch.Tensor
    next_flat: torch.Tensor
    action: torch.Tensor
    z: torch.Tensor
    discount: torch.Tensor
    aux_rewards: torch.Tensor


class FrozenEncoder(nn.Module):
    """Encode observations of one robot at a time with a frozen pretrained encoder.

    Args:
        pretrained (DaGetForPhysicsPretraining): Model holding the encoder.
        observers (list[EmbodimentObserver]): Robots.
    """

    def __init__(self, pretrained: DaGetForPhysicsPretraining, observers: list[EmbodimentObserver]) -> None:
        super().__init__()
        self.pretrained = pretrained.eval().requires_grad_(False)
        self.observers = observers

    def train(self, mode: bool = True) -> FrozenEncoder:
        """Stay in evaluation mode."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, robot: int, tokens: torch.Tensor, dynamics: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """Return embeddings ``[N, J, E]`` of ``N`` observations of one robot."""
        observer = self.observers[robot]
        count, device = tokens.shape[0], tokens.device
        structure = observer.structure
        spatial, ancestors, descendants = (item.to(device).expand(count, -1, -1) for item in structure.graph_distances)

        def per_sample(values: torch.Tensor) -> torch.Tensor:
            return values.to(device).expand(count, *values.shape[1:])

        return self.pretrained.encode(
            input_values=tokens,
            parent_indices=per_sample(structure.parent_indices),
            joint_mask=per_sample(structure.joint_mask),
            adjacency_matrix=per_sample(structure.adjacency_matrix),
            spatial_distances=spatial,
            ancestor_distances=ancestors,
            descendant_distances=descendants,
            mass_matrix=mass,
            rigid_body_features=per_sample(observer.record.rigid_body_features.float()[None]),
            actuator_features=per_sample(observer.record.actuator_features.float()[None]),
            dynamics_features=dynamics,
        )


class DaGetFbCprAgent:
    """FB-CPR with auxiliary rewards, updating ``DaGetBfmModel`` across embodiments.

    Args:
        model_config (BfmConfig): Networks; ``context_size`` must equal the flat-state width.
        config (FbCprConfig): Training hyperparameters.
        encoder (FrozenEncoder): Frozen encoder of every robot.
        device (torch.device | str): Device of the networks.
    """

    def __init__(self, model_config: BfmConfig, config: FbCprConfig, encoder: FrozenEncoder, device: torch.device | str = "cpu") -> None:
        if config.batch_size % model_config.seq_length:
            raise ValueError("batch_size must be a multiple of the model's seq_length")
        if model_config.context_size != model_config.backward_input_size:
            raise ValueError("context_size must equal backward_input_size: critics read the flat state as a context token")
        if model_config.num_parallel < 2:
            raise ValueError("num_parallel must be at least 2: the targets use the ensemble's uncertainty")
        self.model_config, self.config = model_config, config
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.robots = len(encoder.observers)
        self.model = DaGetBfmModel(model_config).to(self.device)
        self.model.init_weights()
        self.target_forward_map = copy.deepcopy(self.model.forward_map).requires_grad_(False)
        self.target_backward_map = copy.deepcopy(self.model.backward_map).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.model.critic).requires_grad_(False)
        self.target_aux_critic = copy.deepcopy(self.model.aux_critic).requires_grad_(False)
        self.normalizer = nn.BatchNorm1d(model_config.backward_input_size, affine=False, momentum=config.normalizer_momentum).to(self.device)
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
        self.z_buffers = [ZBuffer(config.z_buffer_size, model_config.z_dim, self.device) for _ in range(self.robots)]
        self._tracking: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.reward_names = tuple(config.aux_reward_scales)
        self._reward_scales = torch.tensor([config.aux_reward_scales[name] for name in self.reward_names], device=self.device)

    def normalize(self, flat: torch.Tensor) -> torch.Tensor:
        """Normalize flat states with the running statistics, without updating them."""
        self.normalizer.eval()
        with torch.no_grad():
            return self.normalizer(flat)

    @torch.no_grad()
    def embed(self, robot: int, observation: Observation) -> torch.Tensor:
        """Return the joint embeddings of one robot's observation."""
        return self.encoder(
            robot, observation.tokens.to(self.device), observation.dynamics.to(self.device), observation.mass.to(self.device)
        )

    @torch.no_grad()
    def act(self, robot: int, observation: Observation, z: torch.Tensor, mean: bool = False) -> torch.Tensor:
        """Return per-DoF actions ``[N, J]`` for one robot's environments."""
        distribution = self.model.actor(self.embed(robot, observation), z.to(self.device), self.model_config.actor_std)
        return distribution.mean if mean else distribution.sample()

    def tracking_latents(self, flat: torch.Tensor) -> torch.Tensor:
        """Return ``z_t``: the projected mean of B over the next ``seq_length`` frames of ``[T, S]``."""
        return self.model.tracking_inference(self.normalize(flat.to(self.device)))

    def goal_latent(self, flat: torch.Tensor) -> torch.Tensor:
        """Return ``z`` of a goal state ``[N, S]``."""
        return self.model.goal_inference(self.normalize(flat.to(self.device)))

    @torch.no_grad()
    def _sample_tracking_z(self, motions: ExpertMotions, count: int, length: int) -> torch.Tensor:
        """Encode ``count`` expert windows into per-step latents ``[count, length, Z]``."""
        _, next_flat = motions.sample_windows(count, length)
        embeddings = self.model.backward_map(self.normalize(next_flat.reshape(count * length, -1).to(self.device)))
        embeddings = embeddings.view(count, length, -1)
        window = self.model_config.seq_length
        averaged = torch.stack(
            [embeddings[:, step : min(step + window, length)].mean(dim=1) for step in range(length)], dim=1
        )
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
            self._tracking[robot] = (environments, self._sample_tracking_z(motions, chosen, length))
        if chosen > 0 and robot in self._tracking:
            environments, tracking = self._tracking[robot]
            times = (steps % length)[environments]
            z[environments] = tracking[torch.arange(chosen, device=self.device), times]
        return z

    def _training_batch(self, replays: list[ReplayBuffer]) -> TrainingBatch:
        """Sample an equal share of every robot, encode it per robot, and pad."""
        samples = [replay.sample(count) for replay, count in zip(replays, split_counts(self.config.batch_size, self.robots))]
        joints = max(sample["action"].shape[1] for sample in samples)
        robots, masks, embeddings, next_embeddings, actions = [], [], [], [], []
        for robot, sample in enumerate(samples):
            count, robot_joints = sample["action"].shape[:2]
            robots.append(torch.full((count,), robot, device=self.device))
            mask = torch.zeros((count, joints), dtype=torch.bool, device=self.device)
            mask[:, :robot_joints] = True
            masks.append(mask)
            device = self.device
            embeddings.append(pad_joints(self.encoder(robot, sample["tokens"].to(device), sample["dynamics"].to(device), sample["mass"].to(device)), joints))
            next_embeddings.append(
                pad_joints(self.encoder(robot, sample["next_tokens"].to(device), sample["next_dynamics"].to(device), sample["next_mass"].to(device)), joints)
            )
            actions.append(pad_joints(sample["action"].to(device), joints))

        def join(key: str) -> torch.Tensor:
            return torch.cat([sample[key] for sample in samples]).to(self.device)

        terminated = join("terminated")[:, None]
        return TrainingBatch(
            torch.cat(robots), torch.cat(masks), torch.cat(embeddings), torch.cat(next_embeddings), join("flat"),
            join("next_flat"), torch.cat(actions), join("z"), self.config.discount * (~terminated).float(), join("aux_rewards"),
        )

    def _expert_batch(self, motions: list[ExpertMotions]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample expert windows: states and next states ``[B, S]`` and window robots ``[W]``."""
        length = self.model_config.seq_length
        windows = self.config.batch_size // length
        if windows < self.robots:
            raise ValueError("batch_size is too small: every robot needs an expert window per update")
        states, next_states, robots = [], [], []
        for robot, (expert, count) in enumerate(zip(motions, split_counts(windows, self.robots))):
            state, next_state = expert.sample_windows(count, length)
            states.append(state.flatten(0, 1))
            next_states.append(next_state.flatten(0, 1))
            robots.append(torch.full((count,), robot, device=self.device))
        return torch.cat(states).to(self.device), torch.cat(next_states).to(self.device), torch.cat(robots)

    @torch.no_grad()
    def encode_expert(self, next_flat: torch.Tensor) -> torch.Tensor:
        """Return per-frame ``z``: the projected mean of B over each expert window."""
        length = self.model_config.seq_length
        embeddings = self.model.backward_map(next_flat).view(-1, length, self.model_config.z_dim)
        return torch.repeat_interleave(self.model.project_z(embeddings.mean(dim=1)), length, dim=0)

    @torch.no_grad()
    def sample_mixed_z(self, robots: torch.Tensor, goals: torch.Tensor, expert_z: torch.Tensor, expert_robots: torch.Tensor) -> torch.Tensor:
        """Sample training latents: replay goals, expert windows or uniform, always from the same robot.

        Args:
            robots (torch.Tensor): ``[B]`` robot of each transition.
            goals (torch.Tensor): ``[B, S]`` normalized next states.
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
        goal_index, expert_index = torch.zeros(count, dtype=torch.long, device=self.device), torch.zeros(count, dtype=torch.long, device=self.device)
        for robot in range(self.robots):
            rows = torch.nonzero(robots == robot).squeeze(-1)
            experts = torch.nonzero(expert_robots == robot).squeeze(-1)
            goal_index[rows] = rows[torch.randperm(rows.shape[0], device=self.device)]
            expert_index[rows] = experts[torch.randint(0, experts.shape[0], (rows.shape[0],), device=self.device)]
        encoded_goals = self.model.project_z(self.model.backward_map(goals[goal_index]))
        z = torch.where((mix == 0)[:, None], encoded_goals, z)
        return torch.where((mix == 1)[:, None], expert_z[expert_index], z)

    def _gradient_penalty(self, real: torch.Tensor, real_z: torch.Tensor, fake: torch.Tensor, fake_z: torch.Tensor) -> torch.Tensor:
        """Return the WGAN gradient penalty of D between expert and replay samples."""
        alpha = torch.rand(real.shape[0], 1, device=real.device)
        states = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        latents = (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True)
        logits = self.model.discriminator.compute_logits(states, latents)
        gradients = autograd.grad(logits, (states, latents), torch.ones_like(logits), create_graph=True)
        return ((torch.cat(gradients, dim=1).norm(2, dim=1) - 1) ** 2).mean()

    def update_discriminator(self, expert: torch.Tensor, expert_z: torch.Tensor, train: torch.Tensor, train_z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Train D to separate expert windows from replay states, each with its z."""
        discriminator = self.model.discriminator
        expert_loss = -F.logsigmoid(discriminator.compute_logits(expert, expert_z))
        train_loss = F.softplus(discriminator.compute_logits(train, train_z))
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

    def _next_action(self, embeddings: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sample a TD3-style clipped action."""
        return self.model.actor(embeddings, z, self.model_config.actor_std, mask).sample(clip=self.config.stddev_clip)

    def update_fb(self, batch: TrainingBatch, z: torch.Tensor, flat: torch.Tensor, next_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Train F and B with the successor-measure Bellman loss and B's orthonormality."""
        with torch.no_grad():
            next_action = self._next_action(batch.next_embeddings, z, batch.joint_mask)
            target_forward = self.target_forward_map(batch.next_embeddings, z, next_action, batch.joint_mask, context_states=next_flat[:, None])
            target_backward = self.target_backward_map(next_flat)
            _, _, target = ensemble_pessimism(target_forward @ target_backward.T, self.config.fb_pessimism_penalty)
        forward = self.model.forward_map(batch.embeddings, z, batch.action, batch.joint_mask, context_states=flat[:, None])
        backward = self.model.backward_map(next_flat)
        measures = forward @ backward.T
        fb_loss, fb_diagonal, fb_off = fb_bellman_loss(measures, batch.discount, target)
        orth_loss, orth_diagonal, orth_off = orthonormality_loss(backward)
        loss = fb_loss + self.config.ortho_coef * orth_loss
        self.optimizers["forward"].zero_grad(set_to_none=True)
        self.optimizers["backward"].zero_grad(set_to_none=True)
        loss.backward()
        if self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.forward_map.parameters(), self.config.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.backward_map.parameters(), self.config.clip_grad_norm)
        self.optimizers["forward"].step()
        self.optimizers["backward"].step()
        return {
            "fb_loss": loss.detach(), "fb_diag": fb_diagonal.detach(), "fb_offdiag": fb_off.detach(),
            "orth_loss": orth_loss.detach(), "orth_loss_diag": orth_diagonal.detach(), "orth_loss_offdiag": orth_off.detach(),
            "target_M": target.mean(), "M1": measures[0].mean().detach(), "F1": forward[0].mean().detach(),
            "B_norm": backward.norm(dim=-1).mean().detach(), "z_norm": z.norm(dim=-1).mean(),
        }

    def _update_value_critic(
        self, critic: nn.Module, target_critic: nn.Module, optimizer: torch.optim.Optimizer, penalty: float,
        reward: torch.Tensor, batch: TrainingBatch, z: torch.Tensor, flat: torch.Tensor, next_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one Bellman update of a critic; return its loss, target, Q and uncertainty."""
        with torch.no_grad():
            next_action = self._next_action(batch.next_embeddings, z, batch.joint_mask)
            next_values = target_critic(batch.next_embeddings, z, next_action, batch.joint_mask, context_states=next_flat[:, None])
            _, uncertainty, next_value = ensemble_pessimism(next_values, penalty)
            target = reward + batch.discount * next_value
        values = critic(batch.embeddings, z, batch.action, batch.joint_mask, context_states=flat[:, None])
        members = values.shape[0]
        loss = 0.5 * members * F.mse_loss(values, target.expand(members, -1, -1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.detach(), target.mean(), values.mean().detach(), uncertainty.mean()

    def update_critic(self, batch: TrainingBatch, z: torch.Tensor, flat: torch.Tensor, next_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Train Q_D on the discriminator reward ``log D - log(1 - D)``."""
        with torch.no_grad():
            reward = self.model.discriminator.compute_reward(flat, z)
        loss, target, value, uncertainty = self._update_value_critic(
            self.model.critic, self.target_critic, self.optimizers["critic"], self.config.critic_pessimism_penalty,
            reward, batch, z, flat, next_flat,
        )
        return {"critic_loss": loss, "target_Q": target, "Q1": value, "unc_Q": uncertainty, "mean_disc_reward": reward.mean()}

    def update_aux_critic(self, batch: TrainingBatch, z: torch.Tensor, flat: torch.Tensor, next_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Train Q_R on the normalized weighted sum of the auxiliary rewards."""
        reward = self.aux_normalizer((batch.aux_rewards * self._reward_scales).sum(-1, keepdim=True))
        loss, target, value, uncertainty = self._update_value_critic(
            self.model.aux_critic, self.target_aux_critic, self.optimizers["aux_critic"], self.config.aux_critic_pessimism_penalty,
            reward, batch, z, flat, next_flat,
        )
        metrics = {"aux_critic_loss": loss, "target_auxQ": target, "auxQ1": value, "unc_auxQ": uncertainty, "mean_aux_reward": reward.mean()}
        for index, name in enumerate(self.reward_names):
            metrics[f"aux_rew/{name}"] = batch.aux_rewards[:, index].mean()
        return metrics

    def update_actor(self, batch: TrainingBatch, z: torch.Tensor, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Maximize ``Fᵀz + λ_D Q_D + λ_R Q_R`` with ensemble pessimism."""
        config = self.config
        action = self._next_action(batch.embeddings, z, batch.joint_mask)
        context = flat[:, None]
        q_d = self.model.critic(batch.embeddings, z, action, batch.joint_mask, context_states=context)
        q_r = self.model.aux_critic(batch.embeddings, z, action, batch.joint_mask, context_states=context)
        forward = self.model.forward_map(batch.embeddings, z, action, batch.joint_mask, context_states=context)
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

    def update(self, replays: list[ReplayBuffer], motions: list[ExpertMotions]) -> dict[str, torch.Tensor]:
        """Run one gradient step of every network, in BFM-Zero's order, then the target updates."""
        config = self.config
        batch = self._training_batch(replays)
        expert_flat, expert_next, expert_windows = self._expert_batch(motions)
        self.normalizer.train()
        with torch.no_grad():
            self.normalizer(batch.flat)
            self.normalizer(batch.next_flat)
        flat, next_flat = self.normalize(batch.flat), self.normalize(batch.next_flat)
        expert_flat, expert_next = self.normalize(expert_flat), self.normalize(expert_next)
        expert_z = self.encode_expert(expert_next)
        metrics = self.update_discriminator(expert_flat, expert_z, flat, batch.z)
        expert_robots = torch.repeat_interleave(expert_windows, self.model_config.seq_length)
        z = self.sample_mixed_z(batch.robots, next_flat, expert_z, expert_robots)
        for robot, buffer in enumerate(self.z_buffers):
            buffer.add(z[batch.robots == robot])
        relabel = torch.rand((z.shape[0], 1), device=self.device) <= config.relabel_ratio
        z = torch.where(relabel, z, batch.z)
        metrics.update(self.update_fb(batch, z, flat, next_flat))
        metrics.update(self.update_critic(batch, z, flat, next_flat))
        metrics.update(self.update_aux_critic(batch, z, flat, next_flat))
        metrics.update(self.update_actor(batch, z, flat))
        soft_update(self.model.forward_map, self.target_forward_map, config.fb_target_tau)
        soft_update(self.model.backward_map, self.target_backward_map, config.fb_target_tau)
        soft_update(self.model.critic, self.target_critic, config.critic_target_tau)
        soft_update(self.model.aux_critic, self.target_aux_critic, config.critic_target_tau)
        return metrics

    def state_dict(self) -> dict:
        """Return every trained state, for checkpoints."""
        return {
            "model": self.model.state_dict(),
            "target_forward_map": self.target_forward_map.state_dict(),
            "target_backward_map": self.target_backward_map.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "target_aux_critic": self.target_aux_critic.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "aux_normalizer": self.aux_normalizer.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "model_config": asdict(self.model_config),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a state from ``state_dict``."""
        self.model.load_state_dict(state["model"])
        self.target_forward_map.load_state_dict(state["target_forward_map"])
        self.target_backward_map.load_state_dict(state["target_backward_map"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.target_aux_critic.load_state_dict(state["target_aux_critic"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.aux_normalizer.load_state_dict(state["aux_normalizer"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])

    def save(self, path: str | Path) -> None:
        """Write a checkpoint file."""
        torch.save(self.state_dict(), path)
