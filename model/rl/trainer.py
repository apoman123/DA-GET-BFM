"""Online off-policy training loop of the cross-embodiment FB-CPR agent.

Mirrors BFM-Zero's ``Workspace.train_online``: parallel environments act with their
latent and fill a replay buffer, random actions seed it, and a fixed number of gradient
steps runs per collected batch. Here each robot has its own environments, replay buffer
and expert motions, and every update draws an equal share of each robot.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.configuration_bfm import BfmConfig
from model.configuration_da_get import DaGetConfig
from model.pretraining import DaGetForPhysicsPretraining, DaGetPhysicsPretrainingConfig
from model.rl.agent import DaGetFbCprAgent, FrozenEncoder
from model.rl.buffers import ExpertMotions, ReplayBuffer
from model.rl.configuration_fb_cpr import FbCprConfig
from model.rl.envs import DomainRandomization, EnvConfig, EnvStep, MujocoEmbodimentEnv
from model.rl.observation import EmbodimentObserver, Observation, RobotState, flat_state_size, trajectory_state
from model.training.dataset import joint_feature_config, load_data_config
from model.training.episodes import EmbodimentRecord
from model.training.ground_sim import ContactGeometry, ground_specs
from model.training.sources import SOURCE_TYPES


@dataclass
class ObservationNoise:
    """Additive uniform sensor noise on the policy's observations (BFM-Zero's values)."""

    enabled: bool = True
    joint_position: float = 0.01
    joint_velocity: float = 0.5
    gravity: float = 0.05
    angular_velocity: float = 0.2


@dataclass
class TrainingArguments:
    """Run settings; defaults are BFM-Zero's unless noted.

    Attributes:
        data_config (str): JSON with one ``sources`` entry per robot (see ``docs/bfm.md``).
        output_dir (str): Checkpoints and TensorBoard logs.
        encoder_checkpoint (str | None): Pretrained encoder; a random one is used if None.
        num_env_steps (int): Total environment frames (BFM-Zero: 384M).
        online_parallel_envs (int): Environments over all robots.
        buffer_size (int): Stored transitions over all robots.
        num_seed_steps (int): Frames with random actions before updates start.
        update_agent_every (int): Frames between update rounds.
        num_agent_updates (int): Gradient steps per round.
        log_every (int): Frames between logs (here: 100k).
        eval_every (int): Frames between evaluations; 0 disables them (here: 1M).
        eval_motions (int): Motions per robot in an evaluation.
        checkpoint_every (int): Frames between checkpoints.
        resume_from (str | None): Agent checkpoint to continue from.
        device (str): Device of the networks.
        num_threads (int): Physics workers.
        torch_threads (int): PyTorch CPU threads. The per-step tensors are small, so many threads only add overhead.
        seed (int): Random seed.
        domain_randomization (bool): Randomize the training environments.
        observation_noise (bool): Add sensor noise to the policy's observations.
        observe_base_linear_velocity (bool): Show it to the policy (BFM-Zero does not).
        max_episode_steps (int): Episode length in control steps.
    """

    data_config: str
    output_dir: str
    encoder_checkpoint: str | None = None
    num_env_steps: int = 384_000_000
    online_parallel_envs: int = 1024
    buffer_size: int = 5_120_000
    num_seed_steps: int = 10_240
    update_agent_every: int = 1024
    num_agent_updates: int = 16
    log_every: int = 100_000
    eval_every: int = 1_000_000
    eval_motions: int = 8
    checkpoint_every: int = 9_600_000
    resume_from: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads: int = 8
    torch_threads: int = 4
    seed: int = 0
    domain_randomization: bool = True
    observation_noise: bool = True
    observe_base_linear_velocity: bool = False
    max_episode_steps: int = 500


@dataclass
class Embodiment:
    """One robot of the training set.

    Attributes:
        record (EmbodimentRecord): Robot with position-mode actuators.
        observer (EmbodimentObserver): Builds its observations.
        motions (ExpertMotions): Its expert motions.
        env_config (EnvConfig): Its environment settings.
    """

    record: EmbodimentRecord
    observer: EmbodimentObserver
    motions: ExpertMotions
    env_config: EnvConfig = field(default_factory=EnvConfig)


def load_embodiments(data_config: str | Path | dict, observe_base_linear_velocity: bool = False,
                     domain_randomization: bool = True, max_episode_steps: int = 500) -> tuple[list[Embodiment], dict[str, Any]]:
    """Load each source of a data config as a robot with expert motions.

    Sources use the keys of the pretraining data config, plus the environment keys
    ``nominal_pose``, ``contact_geometry``, ``gain_scale`` and ``action_scale``.

    Args:
        data_config (str | Path | dict): Data config.
        observe_base_linear_velocity (bool): See ``EmbodimentObserver``.
        domain_randomization (bool): Randomize the environments.
        max_episode_steps (int): Episode length.

    Returns:
        tuple: Robots, and the loaded config.
    """
    config, base = load_data_config(data_config)
    if not config.get("sources"):
        raise ValueError("data config needs at least one source")
    feature_config = joint_feature_config(config.get("joint_features"))
    history = int(config.get("history_size", 3))
    loaded = []
    for source in config["sources"]:
        kind = source.get("type")
        if kind not in SOURCE_TYPES:
            raise ValueError(f"unknown source type {kind!r}; use one of {sorted(SOURCE_TYPES)}")
        records, trajectories = SOURCE_TYPES[kind](source, base, feature_config)
        for index, record in enumerate(records):
            motions = [trajectory for trajectory in trajectories if trajectory.embodiment == index]
            if not motions:
                raise ValueError(f"{record.name}: the source gave no motion")
            if record.graph.floating_base and not record.contact_links:
                raise ValueError(f"{record.name}: a floating robot needs contact_links for the RL environment")
            specs = ground_specs(record.specs, float(source.get("gain_scale", 1.0)))
            robot = EmbodimentRecord.build(
                record.name, record.graph, specs, contact_links=list(record.contact_links), surface_contacts=record.surface_contacts
            )
            loaded.append((robot, motions, source))
    max_joints = max(robot.graph.num_joints for robot, _, _ in loaded)
    embodiments = []
    for offset, (robot, motions, source) in enumerate(loaded):
        flags = [trajectory_state(robot, trajectory)[1:] for trajectory in motions]
        observer = EmbodimentObserver(
            robot, history, max_joints, observe_base_linear_velocity,
            use_height=all(flag[0] for flag in flags), use_linear_velocity=all(flag[1] for flag in flags),
        )
        env_config = EnvConfig(
            max_episode_steps=max_episode_steps,
            action_scale=float(source.get("action_scale", 1.25)),
            nominal_pose=source.get("nominal_pose", {}),
            geometry=ContactGeometry(**source.get("contact_geometry", {})),
            randomization=DomainRandomization(enabled=domain_randomization),
            seed=int(config.get("seed", 0)) + offset,
        )
        embodiments.append(Embodiment(robot, observer, ExpertMotions(observer, motions, env_config.control_dt), env_config))
    return embodiments, config


def build_encoder(checkpoint: str | None, token_size: int) -> DaGetForPhysicsPretraining:
    """Load the pretrained encoder, or a small untrained one when no checkpoint is given."""
    if checkpoint:
        model = DaGetForPhysicsPretraining.from_pretrained(checkpoint)
        if model.config.encoder["input_size"] != token_size:
            raise ValueError(
                f"the checkpoint's token size {model.config.encoder['input_size']} differs from {token_size}: "
                "use the same joint_features and history_size as in pretraining"
            )
        return model
    print("WARNING: no --encoder_checkpoint; using a random, frozen encoder. This is only useful for debugging.")
    encoder = DaGetConfig(input_size=token_size, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                          intermediate_size=512, use_physics_cross_attention=True)
    return DaGetForPhysicsPretraining(DaGetPhysicsPretrainingConfig.from_encoder_config(encoder))


class Rollout:
    """Run one robot's environments and keep their observation histories.

    Args:
        env (MujocoEmbodimentEnv): Environments.
        observer (EmbodimentObserver): Observer of the robot.
        noise (ObservationNoise): Sensor noise.
        generator (torch.Generator): Noise source.
    """

    def __init__(self, env: MujocoEmbodimentEnv, observer: EmbodimentObserver, noise: ObservationNoise, generator: torch.Generator) -> None:
        self.env, self.observer, self.noise, self.generator = env, observer, noise, generator
        count, joints, history = env.num_envs, env.joints, observer.history_size
        self.local = torch.zeros((count, history, joints, 5), dtype=torch.float64)
        self.global_features = torch.zeros((count, history, 11), dtype=torch.float64)
        self.perceived: RobotState | None = None
        self.torque = torch.zeros((count, joints), dtype=torch.float64)
        self.observation: Observation | None = None

    def _perceive(self, state: RobotState) -> RobotState:
        """Return the state the policy senses."""
        settings = self.noise
        if not settings.enabled:
            return state

        def noisy(values: torch.Tensor, scale: float) -> torch.Tensor:
            return values + scale * (2 * torch.rand(values.shape, generator=self.generator, dtype=values.dtype) - 1)

        direction = noisy(state.gravity_direction, settings.gravity)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return RobotState(
            noisy(state.q, settings.joint_position), noisy(state.qd, settings.joint_velocity), direction,
            noisy(state.angular_velocity, settings.angular_velocity), state.linear_velocity, state.height,
        )

    def start(self, frames: list[dict] | None = None) -> None:
        """Reset every environment and build the first observations."""
        self.env.reset(frames=frames)
        state = self.env.state()
        self.perceived = self._perceive(state)
        local, global_features = self.observer.policy_step(self.perceived, None, None, None)
        self.local[:] = local[:, None]
        self.global_features[:] = global_features[:, None]
        self.torque.zero_()
        self.observation = self.observer.observation(self.local, self.global_features, state, self.perceived, None)

    def advance(self, actions: np.ndarray) -> tuple[Observation, EnvStep]:
        """Step the environments; return the next observation before any reset."""
        step = self.env.step(actions)
        perceived = self._perceive(step.state)
        local, global_features = self.observer.policy_step(perceived, step.target, self.perceived.q, self.perceived.qd)
        self.local = torch.cat((self.local[:, 1:], local[:, None]), dim=1)
        self.global_features = torch.cat((self.global_features[:, 1:], global_features[:, None]), dim=1)
        next_observation = self.observer.observation(self.local, self.global_features, step.state, perceived, self.torque)
        self.perceived, self.torque = perceived, step.torque
        return next_observation, step

    def reset_done(self, done: torch.Tensor, next_observation: Observation) -> None:
        """Set the observations to continue from, resetting finished environments."""
        self.observation = next_observation
        indices = torch.nonzero(done).squeeze(-1)
        if indices.numel() == 0:
            return
        self.env.reset(indices.numpy())
        state = self.env.state()
        subset = state.select(indices)
        perceived = self._perceive(subset)
        local, global_features = self.observer.policy_step(perceived, None, None, None)
        self.local[indices] = local[:, None].expand(-1, self.observer.history_size, -1, -1)
        self.global_features[indices] = global_features[:, None].expand(-1, self.observer.history_size, -1)
        self.torque[indices] = 0.0
        for name in ("q", "qd", "gravity_direction", "angular_velocity", "linear_velocity", "height"):
            getattr(self.perceived, name)[indices] = getattr(perceived, name)
        fresh = self.observer.observation(self.local[indices], self.global_features[indices], subset, perceived, None)
        for name in ("tokens", "dynamics", "mass", "flat"):
            getattr(self.observation, name)[indices] = getattr(fresh, name)


@torch.no_grad()
def evaluate_robot(agent: DaGetFbCprAgent, robot: int, embodiment: Embodiment, motions: int, pool: ThreadPoolExecutor | None) -> dict[str, float]:
    """Measure tracking and goal reaching on one robot's motions, without randomization or noise.

    Uses the paper's error ``E_mpjpe = mean_t ||q_t - q_ref||``, computed while a world is
    alive, plus the fraction of worlds that stay up to the end.
    """
    selected = embodiment.motions.frames[:motions]
    flats = embodiment.motions.flat[:motions]
    config = EnvConfig(
        num_envs=len(selected), control_dt=embodiment.env_config.control_dt, physics_dt=embodiment.env_config.physics_dt,
        max_episode_steps=10**9, action_scale=embodiment.env_config.action_scale, nominal_pose=embodiment.env_config.nominal_pose,
        geometry=embodiment.env_config.geometry, randomization=DomainRandomization(enabled=False),
    )
    env = MujocoEmbodimentEnv(embodiment.record, config, pool)
    rollout = Rollout(env, embodiment.observer, ObservationNoise(enabled=False), torch.Generator())
    horizon = min(len(frame["q"]) for frame in selected)
    first = [{key: value[0] for key, value in frame.items()} for frame in selected]

    def run(latents: torch.Tensor, reference: np.ndarray) -> tuple[float, float]:
        rollout.start(first)
        alive = np.ones(len(selected), dtype=bool)
        errors, counts = np.zeros(len(selected)), np.zeros(len(selected))
        for step in range(horizon):
            action = agent.act(robot, rollout.observation, latents[:, step], mean=True).cpu().numpy()
            next_observation, result = rollout.advance(action)
            rollout.observation = next_observation
            distance = np.linalg.norm(result.state.q.numpy() - reference[:, min(step + 1, reference.shape[1] - 1)], axis=-1)
            errors += distance * alive
            counts += alive
            alive &= ~result.terminated.numpy()
        return float((errors / np.maximum(counts, 1)).mean()), float(alive.mean())

    reference = np.stack([frame["q"][:horizon] for frame in selected])
    tracking = torch.stack([agent.tracking_latents(flat[:horizon]) for flat in flats])
    tracking_error, tracking_survival = run(tracking, reference)
    goal_frames = torch.stack([flat[horizon - 1] for flat in flats])
    goal = agent.goal_latent(goal_frames)[:, None].expand(-1, horizon, -1)
    goal_reference = np.broadcast_to(reference[:, -1:], reference.shape)
    goal_error, goal_survival = run(goal, goal_reference)
    return {
        "tracking_mpjpe": tracking_error, "tracking_survival": tracking_survival,
        "goal_mpjpe": goal_error, "goal_survival": goal_survival,
    }


class FbCprTrainer:
    """Build everything from a data config and train.

    Args:
        arguments (TrainingArguments): Run settings.
        model_overrides (dict | None): ``BfmConfig`` fields to change.
        config (FbCprConfig | None): Agent hyperparameters.
    """

    def __init__(self, arguments: TrainingArguments, model_overrides: dict | None = None, config: FbCprConfig | None = None) -> None:
        self.arguments = arguments
        self.config = config or FbCprConfig()
        torch.set_num_threads(arguments.torch_threads)
        torch.manual_seed(arguments.seed)
        np.random.seed(arguments.seed)
        self.embodiments, _ = load_embodiments(
            arguments.data_config, arguments.observe_base_linear_velocity, arguments.domain_randomization, arguments.max_episode_steps
        )
        observers = [embodiment.observer for embodiment in self.embodiments]
        token_sizes = {observer.token_size for observer in observers}
        if len(token_sizes) != 1:
            raise ValueError("every robot must use the same joint features and history")
        pretrained = build_encoder(arguments.encoder_checkpoint, token_sizes.pop())
        size = flat_state_size(observers[0].max_joints)
        model_config = BfmConfig(
            embedding_size=pretrained.encoder.config.hidden_size, backward_input_size=size, context_size=size, **(model_overrides or {})
        )
        self.agent = DaGetFbCprAgent(model_config, self.config, FrozenEncoder(pretrained, observers), arguments.device)
        self.frames = 0
        if arguments.resume_from:
            state = torch.load(arguments.resume_from, map_location=arguments.device, weights_only=False)
            self.agent.load_state_dict(state)
            self.frames = int(state.get("frames", 0))
        self.pool = ThreadPoolExecutor(arguments.num_threads) if arguments.num_threads > 1 else None
        shares = np.array_split(np.arange(arguments.online_parallel_envs), len(self.embodiments))
        if min(len(share) for share in shares) == 0:
            raise ValueError("online_parallel_envs must be at least the number of robots")
        generator = torch.Generator().manual_seed(arguments.seed)
        noise = ObservationNoise(enabled=arguments.observation_noise)
        self.rollouts, self.replays = [], []
        for embodiment, share in zip(self.embodiments, shares):
            embodiment.env_config.num_envs = len(share)
            env = MujocoEmbodimentEnv(embodiment.record, embodiment.env_config, self.pool)
            env.set_motions(embodiment.motions.frames)
            self.rollouts.append(Rollout(env, embodiment.observer, noise, generator))
            capacity = max(arguments.buffer_size // len(self.embodiments), self.config.batch_size)
            self.replays.append(ReplayBuffer(capacity, self.agent.reward_names))
        self.motions = [embodiment.motions for embodiment in self.embodiments]
        self.output = Path(arguments.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

    def checkpoint(self) -> Path:
        """Write the latest checkpoint and return its path."""
        path = self.output / "checkpoint.pt"
        state = self.agent.state_dict()
        state["frames"] = self.frames
        torch.save(state, path)
        return path

    def evaluate(self, writer) -> dict[str, float]:
        """Evaluate every robot and log the results."""
        results: dict[str, float] = {}
        for robot, embodiment in enumerate(self.embodiments):
            metrics = evaluate_robot(self.agent, robot, embodiment, self.arguments.eval_motions, self.pool)
            for key, value in metrics.items():
                results[f"eval/{embodiment.record.name}/{key}"] = value
                writer.add_scalar(f"eval/{embodiment.record.name}/{key}", value, self.frames)
        writer.flush()
        return results

    def train(self) -> dict[str, float]:
        """Run the online loop until ``num_env_steps`` frames; return the last metrics."""
        from torch.utils.tensorboard import SummaryWriter

        arguments = self.arguments
        writer = SummaryWriter(str(self.output / "runs"))
        writer.add_text("arguments", json.dumps(asdict(arguments), indent=2))
        writer.add_text("agent_config", json.dumps(asdict(self.config), indent=2))
        writer.add_text("model_config", json.dumps(asdict(self.agent.model_config), indent=2))
        (self.output / "embodiments.json").write_text(
            json.dumps([{"name": e.record.name, "joints": list(e.record.graph.joint_names)} for e in self.embodiments], indent=2)
        )
        contexts: list[torch.Tensor | None] = [None] * len(self.rollouts)
        for rollout in self.rollouts:
            rollout.start()
        batch_frames = sum(rollout.env.num_envs for rollout in self.rollouts)
        last = {"update": self.frames, "log": self.frames, "eval": self.frames, "checkpoint": self.frames}
        totals: dict[str, float] = {}
        updates, last_log_time, last_log_frames = 0, time.time(), self.frames
        episode_lengths: list[list[int]] = [[] for _ in self.rollouts]
        final: dict[str, float] = {}
        while self.frames < arguments.num_env_steps:
            for robot, rollout in enumerate(self.rollouts):
                env = rollout.env
                contexts[robot] = self.agent.rollout_context(robot, contexts[robot], torch.from_numpy(env.steps.copy()), self.motions[robot])
                if self.frames < arguments.num_seed_steps:
                    action = torch.from_numpy(env.rng.uniform(-1.0, 1.0, (env.num_envs, env.joints))).float()
                else:
                    action = self.agent.act(robot, rollout.observation, contexts[robot]).cpu()
                observation = rollout.observation
                next_observation, step = rollout.advance(action.numpy())
                self.replays[robot].add(observation, action, next_observation, contexts[robot], step.terminated, step.aux_rewards)
                done = step.terminated | step.truncated
                episode_lengths[robot].extend(int(steps) for steps in env.steps[done.numpy()])
                rollout.reset_done(done, next_observation)
            self.frames += batch_frames
            ready = self.frames >= arguments.num_seed_steps and all(len(replay) > 0 for replay in self.replays)
            if ready and self.frames - last["update"] >= arguments.update_agent_every:
                last["update"] = self.frames
                for _ in range(arguments.num_agent_updates):
                    metrics = self.agent.update(self.replays, self.motions)
                    updates += 1
                    for key, value in metrics.items():
                        totals[key] = totals.get(key, 0.0) + float(value)
            if self.frames - last["log"] >= arguments.log_every:
                last["log"] = self.frames
                now = time.time()
                final = {f"train/{key}": value / max(updates, 1) for key, value in totals.items()}
                final["train/frames_per_second"] = (self.frames - last_log_frames) / max(now - last_log_time, 1e-6)
                final["train/gradient_steps"] = updates
                for name, value in final.items():
                    writer.add_scalar(name, value, self.frames)
                for robot, lengths in enumerate(episode_lengths):
                    if lengths:
                        name = f"rollout/{self.embodiments[robot].record.name}/episode_length"
                        final[name] = float(np.mean(lengths))
                        writer.add_scalar(name, final[name], self.frames)
                    lengths.clear()
                writer.flush()
                print(f"frames {self.frames}: " + ", ".join(f"{k}={v:.4g}" for k, v in sorted(final.items()) if "aux_rew" not in k), flush=True)
                totals, updates, last_log_time, last_log_frames = {}, 0, now, self.frames
            if arguments.eval_every and ready and self.frames - last["eval"] >= arguments.eval_every:
                last["eval"] = self.frames
                results = self.evaluate(writer)
                final.update(results)
                print(f"frames {self.frames}: " + ", ".join(f"{k}={v:.4g}" for k, v in sorted(results.items())), flush=True)
            if self.frames - last["checkpoint"] >= arguments.checkpoint_every:
                last["checkpoint"] = self.frames
                self.checkpoint()
        self.checkpoint()
        writer.close()
        if self.pool is not None:
            self.pool.shutdown()
        return final
