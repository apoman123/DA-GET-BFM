"""Physics-loss data from the policy's own rollouts, labelled without any torque.

During RL the encoder keeps training on the physics auxiliary losses. Offline data covers
the states its sources visit; ``PhysicsReplay`` adds the states the policy itself visits.
It records contiguous chunks of a few environments per robot (joint state, base twist,
gravity, the commanded targets, and the simulator's foot contacts) and turns each chunk
into a processed episode with ``model.training.episodes.process_episode``:

* contacts come from the simulator, so the contact-constrained response ``G(q)`` (tasks R
  and R^-1) is labelled with the true contact set;
* the commanded targets give exact action features, so the action-effect task E is
  conditioned on what was sent and the motor task D sees the real requests;
* FK and proposal variants of D come for free.

The simulator's applied torque is not used as an FD/ID label here. A control step spans
several physics steps whose PD torque changes with the state, while the recorded state is
the one at the end of the step, so torque and acceleration of one record do not belong to
one instant. Exact torque labels come from the offline simulated sources, which record
both at the same instant by construction.

The response labels use the robot graph of the RL record (nominal inertia plus rotor
inertia). With mass randomization on, the simulated robot differs from it by a few
percent, the same mismatch a real robot has against its URDF.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace

import torch

from model.dynamics import STANDARD_GRAVITY
from model.rl.envs import EnvStep
from model.rl.observation import RobotState
from model.training.dataset import CollatorSettings, PhysicsDataCollator, PhysicsTrajectoryDataset
from model.training.episodes import EmbodimentRecord, Trajectory, finalize_acceleration_scales, process_episode


@dataclass
class PhysicsReplaySettings:
    """How much rollout data the physics replay keeps.

    Attributes:
        envs_per_robot (int): Environments recorded per robot.
        chunk_length (int): Control steps per recorded episode.
        min_length (int): Shorter chunks (cut by a reset) are dropped (at most ``chunk_length``).
        episodes_per_robot (int): Processed episodes kept per robot (oldest dropped).
        process_per_update (int): Chunks processed into episodes before each update round.
        proposal_variants (int): Motor proposals per frame (as in the data config).
        contact_margin_steps (int): Frames around a contact change without response labels.
    """

    envs_per_robot: int = 4
    chunk_length: int = 200
    min_length: int = 20
    episodes_per_robot: int = 32
    process_per_update: int = 2
    proposal_variants: int = 2
    contact_margin_steps: int = 2


@dataclass
class _Frame:
    """One recorded state of one environment."""

    q: torch.Tensor
    qd: torch.Tensor
    gravity_direction: torch.Tensor
    angular_velocity: torch.Tensor
    linear_velocity: torch.Tensor
    contacts: torch.Tensor


def _frame(state: RobotState, contacts: torch.Tensor | None, index: int, contact_count: int) -> _Frame:
    """Copy environment ``index`` of a batched state."""
    flags = contacts[index].clone() if contacts is not None else torch.zeros(contact_count, dtype=torch.bool)
    return _Frame(
        state.q[index].clone(), state.qd[index].clone(), state.gravity_direction[index].clone(),
        state.angular_velocity[index].clone(), state.linear_velocity[index].clone(), flags,
    )


class PhysicsReplay:
    """Record rollout chunks per robot and serve physics-loss batches from them.

    Args:
        records (list[EmbodimentRecord]): RL robot of each index (with contact links and
            position-mode actuator specs).
        history_size (int): Observation history of the physics tokens.
        control_dt (float): Control period in s.
        settings (PhysicsReplaySettings | None): Sizes; defaults if None.
        collator_settings (CollatorSettings | None): Label weights and response sampling.
        seed (int): Seed of the batch sampling and the proposals.
    """

    def __init__(
        self,
        records: list[EmbodimentRecord],
        history_size: int,
        control_dt: float,
        settings: PhysicsReplaySettings | None = None,
        collator_settings: CollatorSettings | None = None,
        seed: int = 0,
    ) -> None:
        # Copies, because the data statistics are written into the records.
        self.records = [replace(record) for record in records]
        finalize_acceleration_scales(self.records, [])  # unit scales until episodes arrive
        self.history_size = history_size
        self.control_dt = control_dt
        self.settings = settings or PhysicsReplaySettings()
        self.collator = PhysicsDataCollator(self.records, history_size, collator_settings)
        self.random = random.Random(seed)
        self.seed = seed
        robots = len(records)
        self._pending: list[dict[int, _Frame | None]] = [{} for _ in range(robots)]
        self._chunks: list[dict[int, list[tuple[_Frame, torch.Tensor]]]] = [{} for _ in range(robots)]
        self._queue: deque[tuple[int, Trajectory]] = deque()
        self._episodes: list[deque] = [deque(maxlen=self.settings.episodes_per_robot) for _ in range(robots)]
        self._dataset: PhysicsTrajectoryDataset | None = None
        self.processed = 0

    def start(self, robot: int, state: RobotState, contacts: torch.Tensor | None) -> None:
        """Begin recording a robot's first environments from their current state (drops partial chunks)."""
        count = min(self.settings.envs_per_robot, state.q.shape[0])
        contact_count = len(self.records[robot].contact_links)
        self._pending[robot] = {index: _frame(state, contacts, index, contact_count) for index in range(count)}
        self._chunks[robot] = {index: [] for index in range(count)}

    def step(self, robot: int, step: EnvStep, done: torch.Tensor) -> None:
        """Append one control step of a robot (call before its resets).

        Args:
            robot (int): Robot index.
            step (EnvStep): The step, with the state before any reset.
            done (torch.Tensor): Bool ``[N]`` terminated or truncated.
        """
        contact_count = len(self.records[robot].contact_links)
        for index, previous in self._pending[robot].items():
            if previous is None:
                continue
            chunk = self._chunks[robot][index]
            chunk.append((previous, step.target[index].clone()))
            current = _frame(step.state, step.contacts, index, contact_count)
            ended = bool(done[index])
            if ended or len(chunk) >= self.settings.chunk_length:
                # A terminated world was rewritten (divergence) or fell: keep its last
                # state out, so the last recorded transition is simply unlabelled.
                final = current if not bool(step.terminated[index]) else None
                self._close(robot, chunk, final)
                self._chunks[robot][index] = []
            self._pending[robot][index] = None if ended else current

    def restart(self, robot: int, indices: torch.Tensor, state: RobotState, contacts: torch.Tensor | None) -> None:
        """Continue recording reset environments from their new state.

        Args:
            robot (int): Robot index.
            indices (torch.Tensor): Long ``[K]`` reset environments.
            state (RobotState): ``K`` new states, in the order of ``indices``.
            contacts (torch.Tensor | None): Bool ``[K, C]`` their foot contacts.
        """
        contact_count = len(self.records[robot].contact_links)
        for position, index in enumerate(indices.tolist()):
            if index in self._pending[robot]:
                self._pending[robot][index] = _frame(state, contacts, position, contact_count)
                self._chunks[robot][index] = []

    def _close(self, robot: int, chunk: list[tuple[_Frame, torch.Tensor]], final: _Frame | None) -> None:
        """Queue a finished chunk as a trajectory."""
        frames = [frame for frame, _ in chunk] + ([final] if final is not None else [])
        if len(frames) < max(min(self.settings.min_length, self.settings.chunk_length), 2):
            return
        targets = [target for _, target in chunk]
        targets += [targets[-1]] * (len(frames) - len(targets))
        record = self.records[robot]
        floating = record.graph.floating_base

        def stack(name: str) -> torch.Tensor:
            return torch.stack([getattr(frame, name) for frame in frames]).to(torch.float64)

        trajectory = Trajectory(
            embodiment=robot,
            dt=self.control_dt,
            positions=stack("q"),
            velocities=stack("qd"),
            actions=torch.stack(targets).to(torch.float64),
            action_type="position",
            gravity=STANDARD_GRAVITY * stack("gravity_direction"),
            base_angular_velocity=stack("angular_velocity") if floating else None,
            base_linear_velocity=stack("linear_velocity") if floating else None,
            contacts=torch.stack([frame.contacts for frame in frames]) if record.contact_links else None,
        )
        self._queue.append((robot, trajectory))

    def process(self, limit: int | None = None) -> int:
        """Turn queued chunks into episodes; return how many were processed."""
        limit = self.settings.process_per_update if limit is None else limit
        done = 0
        while self._queue and done < limit:
            robot, trajectory = self._queue.popleft()
            episode = process_episode(
                self.records[robot], trajectory,
                proposal_variants=self.settings.proposal_variants,
                contact_margin_steps=self.settings.contact_margin_steps,
                seed=self.seed * 1_000_003 + self.processed,
            )
            self._episodes[robot].append(episode)
            self.processed += 1
            done += 1
        if done:
            episodes = [(robot, episode) for robot, stored in enumerate(self._episodes) for episode in stored]
            finalize_acceleration_scales(self.records, episodes)
            self.collator = PhysicsDataCollator(self.records, self.history_size, self.collator.settings)
            self._dataset = PhysicsTrajectoryDataset(self.records, episodes, self.history_size)
        return done

    @property
    def queued(self) -> int:
        """Return the number of chunks waiting to be processed."""
        return len(self._queue)

    @property
    def episodes(self) -> int:
        """Return the number of stored episodes over all robots."""
        return sum(len(stored) for stored in self._episodes)

    def ready(self) -> bool:
        """Return whether a batch can be drawn."""
        return self._dataset is not None and len(self._dataset) > 0

    def batch(self, size: int) -> dict[str, torch.Tensor]:
        """Return a physics batch of ``size`` random samples of the stored episodes."""
        if not self.ready():
            raise RuntimeError("the physics replay has no processed episode yet")
        indices = [self.random.randrange(len(self._dataset)) for _ in range(size)]
        return self.collator([self._dataset[index] for index in indices])
