"""Physics-loss data from the policy's own rollouts, with exact simulator labels.

During RL the encoder keeps training on the physics auxiliary losses. Offline data covers
the states its sources visit; ``PhysicsReplay`` adds the states the policy itself visits,
such as walking, running and getting up, which the offline simulated sources (standing
and push recovery) do not reach. It records contiguous chunks of a few environments per
robot and turns each chunk into a processed episode with
``model.training.episodes.process_episode``.

Two kinds of chunks, chosen per step by what the environment reports:

* **Exact** (``EnvStep.start`` present; MuJoCo, or Isaac Lab with explicit actuators). The
  environment's label worlds (``EnvConfig.label_worlds``) report, at the start of every
  control step and after its push, the state, the torque the simulator applies over the
  first physics step and the joint and base accelerations it produces there
  (``model.rl.envs.StepStart``). Torque and acceleration belong to one instant, so the
  episode has torque source ``exact`` and gets every label: FK, forward dynamics (B),
  inverse dynamics (C, projected onto the motion-determined part when feet are down), the
  response tasks (R, R^-1), the motor task (D) and the action effect (E). Label worlds keep
  the nominal link masses and base center of mass, so the labels belong to the robot the
  tokens describe.
* **Kinematic** (no ``EnvStep.start``; Isaac Lab with implicit actuators, or worlds beyond
  ``label_worlds``). Each frame is the state at the end of the previous control step. The
  simulator's reported torque is not used: within a control step the PD torque changes at
  every physics step, so a control-rate torque and the end-of-step state are not one
  instant (and an implicit actuator's reported torque is not what PhysX applied). These
  episodes get FK, R, R^-1, D and E.

In both kinds the contacts come from the simulator (no detection), the commanded targets
give exact action features, and a transition that a push interrupts gets no effect label.

The simulators are MuJoCo (Todorov, Erez & Tassa, IROS 2012, DOI 10.1109/IROS.2012.6386109)
and Isaac Lab with PhysX (Mittal et al., arXiv:2511.04831, 2025); the replay itself is
this repo's.
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
from model.training.episodes import TORQUE_SOURCES, EmbodimentRecord, Trajectory, finalize_acceleration_scales, process_episode


@dataclass
class PhysicsReplaySettings:
    """How much rollout data the physics replay keeps.

    Attributes:
        envs_per_robot (int): Environments recorded per robot (the first ones; set the
            environment's ``label_worlds`` to at least this for exact labels).
        chunk_length (int): Control steps per recorded episode.
        min_length (int): Shorter chunks (cut by a reset) are dropped (at most ``chunk_length``).
        episodes_per_robot (int): Processed episodes kept per robot (oldest dropped).
        process_per_update (int): Chunks processed into episodes before each update round.
        proposal_variants (int): Motor proposals per frame (as in the data config).
        contact_margin_steps (int): Frames around a contact change without response labels.
        max_contact_slip (float | None): Frames with a foot in contact faster than this
            (m/s) get no response labels (as in the data config).
    """

    envs_per_robot: int = 4
    chunk_length: int = 200
    min_length: int = 20
    episodes_per_robot: int = 32
    process_per_update: int = 2
    proposal_variants: int = 2
    contact_margin_steps: int = 2
    max_contact_slip: float | None = 0.1


@dataclass
class _Frame:
    """One recorded state of one environment, with exact labels when the simulator gave them."""

    q: torch.Tensor
    qd: torch.Tensor
    gravity_direction: torch.Tensor
    angular_velocity: torch.Tensor
    linear_velocity: torch.Tensor
    contacts: torch.Tensor
    torque: torch.Tensor | None = None
    acceleration: torch.Tensor | None = None
    base_acceleration: torch.Tensor | None = None


@dataclass
class _Entry:
    """A recorded frame, the targets commanded from it, and whether its transition is clean."""

    frame: _Frame
    target: torch.Tensor
    continues: bool = True


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
        self._chunks: list[dict[int, list[_Entry]]] = [{} for _ in range(robots)]
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
        start = step.start
        labelled = 0 if start is None else start.state.q.shape[0]
        for index in list(self._chunks[robot]):
            chunk = self._chunks[robot][index]
            current = _frame(step.state, step.contacts, index, contact_count)
            if index < labelled:
                # Exact: the frame is the start of this step, with its labels. A push at this
                # step means the previous frame's transition did not end here.
                frame = _frame(start.state, start.contacts, index, contact_count)
                frame.torque = start.torque[index].clone()
                frame.acceleration = start.acceleration[index].clone()
                frame.base_acceleration = start.base_acceleration[index].clone()
                if chunk and bool(start.pushed[index]):
                    chunk[-1].continues = False
                chunk.append(_Entry(frame, step.target[index].clone()))
                final = None
            else:
                # Kinematic: the frame is the end of the previous step. A push at this step
                # changed the velocity after it was recorded.
                previous = self._pending[robot].get(index)
                if previous is None:
                    continue
                pushed = step.pushed is not None and bool(step.pushed[index])
                chunk.append(_Entry(previous, step.target[index].clone(), not pushed))
                # A terminated world was rewritten (divergence) or fell: keep its last
                # state out, so the last recorded transition is simply unlabelled.
                final = current if not bool(step.terminated[index]) else None
            ended = bool(done[index])
            if ended or len(chunk) >= self.settings.chunk_length:
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
            if index in self._chunks[robot]:
                self._pending[robot][index] = _frame(state, contacts, position, contact_count)
                self._chunks[robot][index] = []

    def _close(self, robot: int, chunk: list[_Entry], final: _Frame | None) -> None:
        """Queue a finished chunk as a trajectory."""
        frames = [entry.frame for entry in chunk] + ([final] if final is not None else [])
        if len(frames) < max(min(self.settings.min_length, self.settings.chunk_length), 2):
            return
        targets = [entry.target for entry in chunk]
        targets += [targets[-1]] * (len(frames) - len(targets))
        continues = [entry.continues for entry in chunk] + [True] * (len(frames) - len(chunk))
        record = self.records[robot]
        floating = record.graph.floating_base
        exact = all(frame.torque is not None for frame in frames)

        def stack(name: str) -> torch.Tensor:
            return torch.stack([getattr(frame, name) for frame in frames]).to(torch.float64)

        base_acceleration = stack("base_acceleration") if exact else None
        trajectory = Trajectory(
            embodiment=robot,
            dt=self.control_dt,
            positions=stack("q"),
            velocities=stack("qd"),
            accelerations=stack("acceleration") if exact else None,
            applied_torque=stack("torque") if exact else None,
            actions=torch.stack(targets).to(torch.float64),
            action_type="position",
            gravity=STANDARD_GRAVITY * stack("gravity_direction"),
            base_angular_velocity=stack("angular_velocity") if floating else None,
            base_linear_velocity=stack("linear_velocity") if floating else None,
            base_linear_acceleration=base_acceleration[:, :3] if exact and floating else None,
            base_angular_acceleration=base_acceleration[:, 3:] if exact and floating else None,
            contacts=torch.stack([frame.contacts for frame in frames]) if record.contact_links else None,
            transition_valid=torch.tensor(continues, dtype=torch.bool),
            exact_dynamics=exact,
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
                max_contact_slip=self.settings.max_contact_slip,
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

    @property
    def exact_episodes(self) -> int:
        """Return the number of stored episodes with exact torque labels."""
        source = TORQUE_SOURCES.index("exact")
        return sum(int(bool((episode["torque_source"][:, 0] == source).any())) for stored in self._episodes for episode in stored)

    def ready(self) -> bool:
        """Return whether a batch can be drawn."""
        return self._dataset is not None and len(self._dataset) > 0

    def batch(self, size: int) -> dict[str, torch.Tensor]:
        """Return a physics batch of ``size`` random samples of the stored episodes."""
        if not self.ready():
            raise RuntimeError("the physics replay has no processed episode yet")
        indices = [self.random.randrange(len(self._dataset)) for _ in range(size)]
        return self.collator([self._dataset[index] for index in indices])
