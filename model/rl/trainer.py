"""Online off-policy training loop of the cross-embodiment FB-CPR agent.

Mirrors BFM-Zero's ``Workspace.train_online``: parallel environments act with their latent
and fill a replay buffer, random actions seed it, a fixed number of gradient steps runs per
collected batch, and periodic evaluations set the motion priorities. Each robot has its own
environments, replay buffer and expert motions, and every update draws an equal share of
each robot. Physics pretraining batches keep tuning the encoder during RL.
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
from torch.utils.data import DataLoader

from model.configuration_bfm import BfmConfig
from model.configuration_da_get import DaGetConfig
from model.pretraining import DaGetForPhysicsPretraining, DaGetPhysicsPretrainingConfig
from model.rl.agent import DaGetFbCprAgent
from model.rl.buffers import ExpertMotions, ReplayBuffer
from model.rl.configuration_fb_cpr import FbCprConfig
from model.rl.envs import DomainRandomization, EmbodimentEnv, EnvConfig, EnvStep, MujocoEmbodimentEnv, MujocoGroup
from model.rl.evaluation import evaluate, motion_priorities
from model.rl.observation import EmbodimentObserver, Observation, RobotState, trajectory_state
from model.training.dataset import build_physics_datasets, joint_feature_config, load_data_config, resolve_path
from model.training.episodes import EmbodimentRecord, Trajectory
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
        data_config (str): JSON with the robots' sources (see ``docs/bfm.md``).
        output_dir (str): Checkpoints and TensorBoard logs.
        simulator (str): ``isaaclab`` or ``mujoco`` (CPU, for tests and small checks).
        encoder_checkpoint (str | None): Pretrained encoder; a new one is used if None.
        num_env_steps (int): Total environment frames (BFM-Zero: 384M).
        online_parallel_envs (int): Environments over all robots.
        buffer_size (int): Stored transitions over all robots.
        buffer_device (str): Device of the replay buffers.
        num_seed_steps (int): Frames with random actions before updates start.
        update_agent_every (int): Frames between update rounds.
        num_agent_updates (int): Gradient steps per round.
        physics_batch_size (int): Batch size of the physics auxiliary losses (here: 64).
        physics_cache_dir (str | None): Cache of the processed physics data.
        log_every (int): Frames between logs.
        eval_every (int): Frames between evaluations and priority updates; 0 disables them.
        eval_motions (int | None): Motions per robot in an evaluation; all by default.
        checkpoint_every (int): Frames between checkpoints.
        resume_from (str | None): Agent checkpoint to continue from.
        device (str): Device of the networks.
        num_threads (int): MuJoCo physics workers.
        torch_threads (int): PyTorch CPU threads.
        seed (int): Random seed.
        domain_randomization (bool): Randomize the training environments.
        observation_noise (bool): Add sensor noise to the policy's observations.
        observe_base_linear_velocity (bool): Show it to the policy (BFM-Zero does not).
        max_episode_steps (int): Episode length in control steps.
        fall_start_probability (float): Episodes that start lying down.
        terminate_on_fall (bool): End episodes when fallen (BFM-Zero: no).
        isaac_spacing (float): Distance between Isaac Lab worlds in m.
    """

    data_config: str
    output_dir: str
    simulator: str = "isaaclab"
    encoder_checkpoint: str | None = None
    num_env_steps: int = 384_000_000
    online_parallel_envs: int = 1024
    buffer_size: int = 5_120_000
    buffer_device: str = "cpu"
    num_seed_steps: int = 10_240
    update_agent_every: int = 1024
    num_agent_updates: int = 16
    physics_batch_size: int = 64
    physics_cache_dir: str | None = None
    log_every: int = 384_000
    eval_every: int = 9_600_000
    eval_motions: int | None = None
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
    fall_start_probability: float = 0.3
    terminate_on_fall: bool = False
    isaac_spacing: float = 5.0


@dataclass
class Embodiment:
    """One robot of the training set.

    Attributes:
        record (EmbodimentRecord): Robot with position-mode actuators.
        observer (EmbodimentObserver): Builds its observations.
        motions (ExpertMotions): Its expert motions.
        env_config (EnvConfig): Its environment settings.
        source (dict): The data-config source that describes its environment.
        base (Path | None): Directory of relative paths in the source.
    """

    record: EmbodimentRecord
    observer: EmbodimentObserver
    motions: ExpertMotions
    env_config: EnvConfig = field(default_factory=EnvConfig)
    source: dict = field(default_factory=dict)
    base: Path | None = None


def load_embodiments(data_config: str | Path | dict, arguments: TrainingArguments | None = None) -> tuple[list[Embodiment], dict[str, Any]]:
    """Group a data config's sources by robot; each robot's expert motions come from its sources.

    Sources of one robot share a ``robot`` name, which is what combines real and synthetic
    trajectories. ``expert`` says whether a source's trajectories are expert data for D and
    prompts; the default is true, except for ``synthetic`` sources, whose random PD motions
    only serve the physics auxiliary losses. Environment keys (``nominal_pose``,
    ``contact_geometry``, ``gain_scale``, ``action_scale``, and the Isaac Lab keys) are read
    from the robot's first expert source.

    Args:
        data_config (str | Path | dict): Data config.
        arguments (TrainingArguments | None): Environment settings; defaults if None.

    Returns:
        tuple: Robots, and the loaded config.
    """
    settings = arguments or TrainingArguments(data_config="", output_dir="")
    config, base = load_data_config(data_config)
    if not config.get("sources"):
        raise ValueError("data config needs at least one source")
    feature_config = joint_feature_config(config.get("joint_features"))
    history = int(config.get("history_size", 3))
    grouped: dict[str, tuple[EmbodimentRecord, list[Trajectory], dict]] = {}
    for source in config["sources"]:
        kind = source.get("type")
        if kind not in SOURCE_TYPES:
            raise ValueError(f"unknown source type {kind!r}; use one of {sorted(SOURCE_TYPES)}")
        if not source.get("expert", kind != "synthetic"):
            continue
        records, trajectories = SOURCE_TYPES[kind](source, base, feature_config)
        for index, record in enumerate(records):
            motions = [trajectory for trajectory in trajectories if trajectory.embodiment == index]
            name = source.get("robot", record.name)
            name = f"{name}/{index}" if "robot" in source and len(records) > 1 else name
            if record.graph.floating_base and not record.contact_links:
                raise ValueError(f"{record.name}: a floating robot needs contact_links for the RL environment")
            if name not in grouped:
                specs = ground_specs(record.specs, float(source.get("gain_scale", 1.0)))
                robot = EmbodimentRecord.build(name, record.graph, specs, contact_links=list(record.contact_links), surface_contacts=record.surface_contacts)
                grouped[name] = (robot, [], source)
            robot = grouped[name][0]
            if list(robot.graph.joint_names) != list(record.graph.joint_names):
                raise ValueError(f"{name}: sources of one robot must have the same joints in the same order")
            grouped[name][1].extend(motions)
    if not grouped:
        raise ValueError("no source provides expert motions; set \"expert\": true on at least one source")
    embodiments = []
    for offset, (robot, motions, source) in enumerate(grouped.values()):
        if not motions:
            raise ValueError(f"{robot.name}: the sources gave no motion")
        flags = [trajectory_state(robot, trajectory)[1:] for trajectory in motions]
        observer = EmbodimentObserver(
            robot, history, settings.observe_base_linear_velocity,
            use_height=all(flag[0] for flag in flags), use_linear_velocity=all(flag[1] for flag in flags),
        )
        env_config = EnvConfig(
            max_episode_steps=settings.max_episode_steps,
            action_scale=float(source.get("action_scale", 1.25)),
            armature=float(source.get("armature", 0.01)),
            fall_start_probability=settings.fall_start_probability,
            terminate_on_fall=settings.terminate_on_fall,
            nominal_pose=source.get("nominal_pose", {}),
            geometry=ContactGeometry(**source.get("contact_geometry", {})),
            randomization=DomainRandomization(enabled=settings.domain_randomization),
            seed=int(config.get("seed", 0)) + offset,
        )
        embodiments.append(Embodiment(robot, observer, ExpertMotions(observer, motions, env_config.control_dt), env_config, source, base))
    return embodiments, config


def build_encoder(checkpoint: str | None, token_size: int) -> DaGetForPhysicsPretraining:
    """Load the pretrained encoder with its heads, or create a small new one."""
    if checkpoint:
        model = DaGetForPhysicsPretraining.from_pretrained(checkpoint)
        if model.config.encoder["input_size"] != token_size:
            raise ValueError(
                f"the checkpoint's token size {model.config.encoder['input_size']} differs from {token_size}: "
                "use the same joint_features and history_size as in pretraining"
            )
        return model
    encoder = DaGetConfig(input_size=token_size, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                          intermediate_size=512, use_physics_cross_attention=True)
    return DaGetForPhysicsPretraining(DaGetPhysicsPretrainingConfig.from_encoder_config(encoder))


class Rollout:
    """Run one robot's environments and keep their observation histories.

    Args:
        env (EmbodimentEnv): Environments.
        observer (EmbodimentObserver): Observer of the robot.
        noise (ObservationNoise): Sensor noise.
        generator (torch.Generator): Noise source.
    """

    def __init__(self, env: EmbodimentEnv, observer: EmbodimentObserver, noise: ObservationNoise, generator: torch.Generator) -> None:
        self.env, self.observer, self.noise, self.generator = env, observer, noise, generator
        count, joints, history = env.num_envs, env.joints, observer.history_size
        self.local = torch.zeros((count, history, joints, 5), dtype=torch.float64)
        self.global_features = torch.zeros((count, history, 11), dtype=torch.float64)
        self.perceived: RobotState | None = None
        self.torque = torch.zeros((count, joints), dtype=torch.float64)
        self.observation: Observation | None = None
        self.evaluating = False

    def set_evaluating(self, value: bool) -> None:
        """Turn pushes and sensor noise off during evaluations."""
        self.evaluating = value
        self.env.evaluating = value

    def _perceive(self, state: RobotState) -> RobotState:
        """Return the state the policy senses."""
        settings = self.noise
        if not settings.enabled or self.evaluating:
            return state

        def noisy(values: torch.Tensor, scale: float) -> torch.Tensor:
            return values + scale * (2 * torch.rand(values.shape, generator=self.generator, dtype=values.dtype) - 1)

        direction = noisy(state.gravity_direction, settings.gravity)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return RobotState(
            noisy(state.q, settings.joint_position), noisy(state.qd, settings.joint_velocity), direction,
            noisy(state.angular_velocity, settings.angular_velocity), state.linear_velocity, state.height,
        )

    def start(self, frames: list[dict | None] | None = None) -> None:
        """Reset every environment and build the first observations."""
        self.env.reset(frames=frames)
        state = self.env.read_state()
        self.perceived = self._perceive(state)
        local, global_features = self.observer.policy_step(self.perceived, None, None, None)
        self.local[:] = local[:, None]
        self.global_features[:] = global_features[:, None]
        self.torque.zero_()
        self.observation = self.observer.observation(self.local, self.global_features, state, self.perceived, None)

    def finish(self) -> tuple[Observation, EnvStep]:
        """Collect the environment step; return the next observation before any reset."""
        step = self.env.collect()
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
        subset = self.env.read_state().select(indices)
        perceived = self._perceive(subset)
        local, global_features = self.observer.policy_step(perceived, None, None, None)
        self.local[indices] = local[:, None].expand(-1, self.observer.history_size, -1, -1)
        self.global_features[indices] = global_features[:, None].expand(-1, self.observer.history_size, -1)
        self.torque[indices] = 0.0
        for name in RobotState.__dataclass_fields__:
            getattr(self.perceived, name)[indices] = getattr(perceived, name)
        self.observation = next_observation
        self.observation.assign(indices, self.observer.observation(self.local[indices], self.global_features[indices], subset, perceived, None))


def build_group(arguments: TrainingArguments, embodiments: list[Embodiment], pool: ThreadPoolExecutor | None, simulation: Any = None):
    """Create every robot's environments in the chosen simulator."""
    if arguments.simulator == "mujoco":
        return MujocoGroup([MujocoEmbodimentEnv(item.record, item.env_config, pool) for item in embodiments])
    if arguments.simulator != "isaaclab":
        raise ValueError("simulator must be 'isaaclab' or 'mujoco'")
    from model.rl.isaac_env import IsaacRobotConfig, build_isaac_group

    robots = []
    for item in embodiments:
        urdf = resolve_path(item.source.get("simulation_urdf", item.source["urdf"]), item.base)
        settings = IsaacRobotConfig(
            str(urdf), item.source.get("ros_package_paths", []), item.source.get("undesired_contact_links"), bool(item.source.get("self_collisions", True))
        )
        robots.append((item.record, item.env_config, settings))
    return build_isaac_group(robots, simulation, arguments.isaac_spacing)


class FbCprTrainer:
    """Build everything from a data config and train.

    Args:
        arguments (TrainingArguments): Run settings.
        model_overrides (dict | None): ``BfmConfig`` fields to change.
        config (FbCprConfig | None): Agent hyperparameters.
        simulation (Any): ``isaaclab.sim.SimulationContext`` when using Isaac Lab.
    """

    def __init__(self, arguments: TrainingArguments, model_overrides: dict | None = None, config: FbCprConfig | None = None, simulation: Any = None) -> None:
        self.arguments = arguments
        self.config = config or FbCprConfig()
        torch.set_num_threads(arguments.torch_threads)
        torch.manual_seed(arguments.seed)
        np.random.seed(arguments.seed)
        self.embodiments, _ = load_embodiments(arguments.data_config, arguments)
        observers = [embodiment.observer for embodiment in self.embodiments]
        if len({observer.token_size for observer in observers}) != 1:
            raise ValueError("every robot must use the same joint features and history")
        pretrained = build_encoder(arguments.encoder_checkpoint, observers[0].token_size)
        model_config = BfmConfig(
            embedding_size=pretrained.encoder.config.hidden_size, state_token_size=observers[0].state_token_size, **(model_overrides or {})
        )
        self.agent = DaGetFbCprAgent(model_config, self.config, pretrained, observers, arguments.device)
        self.frames = 0
        if arguments.resume_from:
            state = torch.load(arguments.resume_from, map_location=arguments.device, weights_only=False)
            self.agent.load_state_dict(state)
            self.frames = int(state.get("frames", 0))
        self.pool = ThreadPoolExecutor(arguments.num_threads) if arguments.num_threads > 1 and arguments.simulator == "mujoco" else None
        shares = np.array_split(np.arange(arguments.online_parallel_envs), len(self.embodiments))
        if min(len(share) for share in shares) == 0:
            raise ValueError("online_parallel_envs must be at least the number of robots")
        for embodiment, share in zip(self.embodiments, shares):
            embodiment.env_config.num_envs = len(share)
        self.group = build_group(arguments, self.embodiments, self.pool, simulation)
        generator = torch.Generator().manual_seed(arguments.seed)
        noise = ObservationNoise(enabled=arguments.observation_noise)
        capacity = max(arguments.buffer_size // len(self.embodiments), self.config.batch_size)
        self.rollouts, self.replays = [], []
        for embodiment, env in zip(self.embodiments, self.group.envs):
            env.set_motions(embodiment.motions.frames)
            self.rollouts.append(Rollout(env, embodiment.observer, noise, generator))
            self.replays.append(ReplayBuffer(capacity, self.agent.reward_names, arguments.buffer_device))
        self.motions = [embodiment.motions for embodiment in self.embodiments]
        self.physics_batches = self._physics_batches() if self.config.train_encoder and self.config.physics_loss_weight > 0 else None
        self.output = Path(arguments.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

    def _physics_batches(self):
        """Return an endless iterator over batches of the physics auxiliary losses."""
        datasets = build_physics_datasets(self.arguments.data_config, self.arguments.physics_cache_dir)
        if datasets.token_size != self.agent.observers[0].token_size:
            raise ValueError("the physics data's token size differs from the policy's")
        loader = DataLoader(datasets.train, batch_size=self.arguments.physics_batch_size, shuffle=True, drop_last=False, collate_fn=datasets.collator())

        def cycle():
            while True:
                yield from loader

        return cycle()

    def checkpoint(self) -> Path:
        """Write the latest checkpoint, with the encoder also in ``from_pretrained`` form."""
        path = self.output / "checkpoint.pt"
        state = self.agent.state_dict()
        state["frames"] = self.frames
        state["motion_priorities"] = [motions.priorities for motions in self.motions]
        torch.save(state, path)
        self.agent.pretrained.save_pretrained(self.output / "encoder")
        return path

    def evaluate(self, writer) -> dict[str, float]:
        """Evaluate every robot, log the results, and update the motion priorities."""
        metrics, distances = evaluate(self.agent, self.rollouts, self.group, self.motions, self.arguments.eval_motions)
        for rollout, motions, emd in zip(self.rollouts, self.motions, distances):
            priorities = motion_priorities(emd)
            motions.set_priorities(torch.from_numpy(priorities))
            rollout.env.set_motion_weights(priorities)
        for key, value in metrics.items():
            writer.add_scalar(key, value, self.frames)
        writer.flush()
        return metrics

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
        fallen_fraction: list[list[float]] = [[] for _ in self.rollouts]
        final: dict[str, float] = {}
        while self.frames < arguments.num_env_steps:
            actions = []
            for robot, rollout in enumerate(self.rollouts):
                env = rollout.env
                contexts[robot] = self.agent.rollout_context(robot, contexts[robot], torch.from_numpy(env.steps.copy()), self.motions[robot])
                if self.frames < arguments.num_seed_steps:
                    action = torch.from_numpy(env.rng.uniform(-1.0, 1.0, (env.num_envs, env.joints))).float()
                else:
                    action = self.agent.act(robot, rollout.observation, contexts[robot]).cpu()
                actions.append(action)
                env.prepare(action.numpy())
            self.group.advance()
            for robot, rollout in enumerate(self.rollouts):
                observation = rollout.observation
                next_observation, step = rollout.finish()
                self.replays[robot].add(observation, actions[robot], next_observation, contexts[robot], step.terminated, step.aux_rewards)
                fallen_fraction[robot].append(float(rollout.env.fallen(step.state).mean()))
                done = step.terminated | step.truncated
                episode_lengths[robot].extend(int(steps) for steps in rollout.env.steps[done.numpy()])
                rollout.reset_done(done, next_observation)
            self.frames += batch_frames
            ready = self.frames >= arguments.num_seed_steps and all(len(replay) > 0 for replay in self.replays)
            if ready and self.frames - last["update"] >= arguments.update_agent_every:
                last["update"] = self.frames
                for _ in range(arguments.num_agent_updates):
                    physics = next(self.physics_batches) if self.physics_batches is not None else None
                    metrics = self.agent.update(self.replays, self.motions, physics)
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
                for robot, embodiment in enumerate(self.embodiments):
                    name = f"rollout/{embodiment.record.name}"
                    if episode_lengths[robot]:
                        final[f"{name}/episode_length"] = float(np.mean(episode_lengths[robot]))
                        writer.add_scalar(f"{name}/episode_length", final[f"{name}/episode_length"], self.frames)
                    final[f"{name}/fallen_fraction"] = float(np.mean(fallen_fraction[robot]))
                    writer.add_scalar(f"{name}/fallen_fraction", final[f"{name}/fallen_fraction"], self.frames)
                    episode_lengths[robot].clear()
                    fallen_fraction[robot].clear()
                writer.flush()
                print(f"frames {self.frames}: " + ", ".join(f"{k}={v:.4g}" for k, v in sorted(final.items()) if "aux_rew" not in k), flush=True)
                totals, updates, last_log_time, last_log_frames = {}, 0, now, self.frames
            if arguments.eval_every and ready and self.frames - last["eval"] >= arguments.eval_every:
                last["eval"] = self.frames
                results = self.evaluate(writer)
                final.update(results)
                print(f"frames {self.frames}: " + ", ".join(f"{k}={v:.4g}" for k, v in sorted(results.items())), flush=True)
                for rollout in self.rollouts:
                    rollout.start()
            if self.frames - last["checkpoint"] >= arguments.checkpoint_every:
                last["checkpoint"] = self.frames
                self.checkpoint()
        self.checkpoint()
        writer.close()
        if self.pool is not None:
            self.pool.shutdown()
        return final
