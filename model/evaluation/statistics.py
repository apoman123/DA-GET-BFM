"""Aggregate scores with the interquartile mean and bootstrap confidence intervals.

Every aggregate follows Agarwal et al. [1]:

* **Interquartile mean (IQM)**: the mean of the middle 50% of all runs pooled over
  tasks, discarding the bottom and top 25% ([1], Sec. 4; rliable's
  ``aggregate_iqm`` is ``scipy.stats.trim_mean(scores, proportiontocut=0.25,
  axis=None)``, which ``interquartile_mean`` reproduces).
* **Stratified bootstrap**: "re-sample runs with replacement independently for each
  task" ([1], Sec. 4), then take **percentile** intervals ([1]: "percentile CIs
  provide good interval estimates for as few as N = 10 runs").
* **Defaults** are rliable's ``get_interval_estimates`` defaults: ``reps=50000``,
  ``confidence_interval_size=0.95``, ``method='percentile'``
  (``rliable/library.py``, checked 2026-11-19).
* **The robot as the unit** (proposal v4, Sec. 5.7): ``robot_bootstrap_ci`` resamples
  robots with replacement, and the runs of every drawn robot with replacement. This is
  rliable's ``task_bootstrap=True`` (``StratifiedBootstrap``: "Whether to perform
  bootstrapping over tasks in addition to runs"), with robots as the tasks: the
  interval then also covers which robots happened to be in the set, which is what a
  claim about unseen robots needs.

Runs are random seeds. Henderson et al. [2] split 10 trials of one configuration into
two groups of 5 seeds and found significantly different results (``p = 0.0016``), and
conclude that "many trials must be run with different random seeds"; they give no
fixed number, so none is enforced here.

No rliable code is copied; the algorithms above are re-implemented with NumPy and SciPy,
and the random generator is a local ``numpy.random.Generator`` (rliable draws from the
global ``np.random``).

References:
    [1] R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare. Deep
        Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021.
        arXiv:2108.13264. Library: https://github.com/google-research/rliable
        (Apache-2.0).
    [2] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger. Deep
        Reinforcement Learning that Matters. AAAI 2018. DOI 10.1609/aaai.v32i1.11694.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
from scipy import stats

# rliable ``get_interval_estimates`` defaults (rliable/library.py).
RLIABLE_REPETITIONS = 50_000
RLIABLE_CONFIDENCE = 0.95
# Fraction cut from each end for the IQM (rliable ``aggregate_iqm``).
IQM_PROPORTION = 0.25
# Bootstrap replicates evaluated together by vectorized statistics (memory bound only).
_CHUNK = 2_000

Scores = Union[np.ndarray, Sequence[Sequence[float]], Mapping[str, Sequence[float]]]


def _flatten(scores: Scores) -> np.ndarray:
    """Pool every run of every task into one float array."""
    if isinstance(scores, Mapping):
        parts = [np.asarray(value, dtype=np.float64).reshape(-1) for value in scores.values()]
        return np.concatenate(parts) if parts else np.zeros(0)
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def interquartile_mean(scores: Scores) -> float:
    """Return the mean of the middle 50% of all runs pooled over tasks.

    Args:
        scores (Scores): ``[runs, tasks]`` array, or a mapping from task (or robot)
            to its runs.

    Returns:
        float: ``scipy.stats.trim_mean(scores, 0.25, axis=None)``, as rliable's
            ``aggregate_iqm``.

    Raises:
        ValueError: There are no scores or some are not finite.
    """
    flat = _flatten(scores)
    if flat.size == 0:
        raise ValueError("no scores")
    if not np.all(np.isfinite(flat)):
        raise ValueError("scores must be finite")
    return float(stats.trim_mean(flat, proportiontocut=IQM_PROPORTION, axis=None))


def mean_score(scores: Scores) -> float:
    """Return the mean over all runs pooled over tasks.

    Args:
        scores (Scores): ``[runs, tasks]`` array or a mapping from task to runs.

    Returns:
        float: Pooled mean.
    """
    flat = _flatten(scores)
    if flat.size == 0:
        raise ValueError("no scores")
    return float(flat.mean())


def _batched(statistic: Callable[[Any], float]) -> Callable[[np.ndarray], np.ndarray] | None:
    """Return a vectorized form ``[reps, pooled runs] -> [reps]`` of known pooled statistics."""
    if statistic is interquartile_mean:
        return lambda values: stats.trim_mean(values, proportiontocut=IQM_PROPORTION, axis=1)
    if statistic is mean_score:
        return lambda values: values.mean(axis=1)
    return None


@dataclass(frozen=True)
class BootstrapInterval:
    """A point estimate and its percentile bootstrap interval.

    Attributes:
        estimate (float): Statistic on the original scores.
        lower (float): Lower percentile bound.
        upper (float): Upper percentile bound.
        confidence (float): Interval size, e.g. 0.95.
        repetitions (int): Bootstrap replicates.
        unit (str): What was resampled: ``run`` (stratified over tasks) or ``robot``.
    """

    estimate: float
    lower: float
    upper: float
    confidence: float
    repetitions: int
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """Return the interval as a JSON-ready dict.

        Returns:
            dict[str, Any]: Field values.
        """
        return asdict(self)


def _as_tasks(scores: Scores) -> tuple[list[str], list[np.ndarray], bool]:
    """Split scores into per-task run arrays; remember whether they came as a matrix."""
    if isinstance(scores, Mapping):
        names = [str(name) for name in scores]
        runs = [np.asarray(value, dtype=np.float64).reshape(-1) for value in scores.values()]
        matrix = False
    else:
        array = np.asarray(scores, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError("score arrays must be [runs, tasks]")
        names = [str(index) for index in range(array.shape[1])]
        runs = [array[:, index] for index in range(array.shape[1])]
        matrix = True
    if not runs or any(item.size == 0 for item in runs):
        raise ValueError("every task needs at least one run")
    if not all(np.all(np.isfinite(item)) for item in runs):
        raise ValueError("scores must be finite")
    return names, runs, matrix


def _rebuild(names: list[str], runs: list[np.ndarray], matrix: bool) -> Scores:
    """Return resampled runs in the input's structure (matrix or mapping)."""
    if matrix and len({item.size for item in runs}) == 1:
        return np.stack(runs, axis=1)
    return {f"{name}#{index}": item for index, (name, item) in enumerate(zip(names, runs))}


def _bootstrap_distribution(
    scores: Scores,
    statistic: Callable[[Any], float],
    repetitions: int,
    generator: np.random.Generator,
    unit: str,
) -> np.ndarray:
    """Return ``repetitions`` bootstrap replicates of the statistic.

    ``unit="run"`` keeps the tasks and resamples each task's runs (stratified
    bootstrap). ``unit="robot"`` also draws the tasks (robots) with replacement first
    (rliable's ``task_bootstrap=True``).
    """
    if unit not in ("run", "robot"):
        raise ValueError("unit must be 'run' or 'robot'")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    names, runs, matrix = _as_tasks(scores)
    count = len(runs)
    batched = _batched(statistic)
    sizes = np.array([item.size for item in runs])
    results = np.empty(repetitions, dtype=np.float64)
    if batched is not None and unit == "run":
        # Pooled statistic: resample each task's runs, concatenate, evaluate per replicate.
        for start in range(0, repetitions, _CHUNK):
            stop = min(start + _CHUNK, repetitions)
            draws = [item[generator.integers(0, item.size, size=(stop - start, item.size))] for item in runs]
            results[start:stop] = batched(np.concatenate(draws, axis=1))
        return results
    if batched is not None and len(set(sizes.tolist())) == 1:
        # Robot bootstrap with equal run counts: draw tasks, then runs, all vectorized.
        stacked = np.stack(runs)  # [tasks, runs]
        width = int(sizes[0])
        for start in range(0, repetitions, _CHUNK):
            stop = min(start + _CHUNK, repetitions)
            chosen = generator.integers(0, count, size=(stop - start, count))
            inner = generator.integers(0, width, size=(stop - start, count, width))
            values = stacked[chosen[..., None], inner].reshape(stop - start, -1)
            results[start:stop] = batched(values)
        return results
    for repetition in range(repetitions):
        chosen = generator.integers(0, count, size=count) if unit == "robot" else np.arange(count)
        draws = [runs[index][generator.integers(0, runs[index].size, size=runs[index].size)] for index in chosen]
        results[repetition] = statistic(_rebuild([names[index] for index in chosen], draws, matrix))
    return results


def _percentile(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return the percentile interval of bootstrap replicates."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])
    return float(lower), float(upper)


def stratified_bootstrap_ci(
    scores: Scores,
    statistic: Callable[[Any], float] = interquartile_mean,
    repetitions: int = RLIABLE_REPETITIONS,
    confidence: float = RLIABLE_CONFIDENCE,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a stratified percentile bootstrap interval (Agarwal et al., 2021).

    Each task's runs are resampled with replacement, independently per task; the tasks
    themselves are kept.

    Args:
        scores (Scores): ``[runs, tasks]`` array, or a mapping from task to its runs
            (run counts may differ between tasks).
        statistic (Callable): Aggregate of scores in the same structure as ``scores``
            (resampled mappings get keys ``"<task>#<i>"``); IQM by default.
        repetitions (int): Bootstrap replicates (rliable default 50000).
        confidence (float): Interval size (rliable default 0.95).
        seed (int): Seed of the local random generator.

    Returns:
        BootstrapInterval: Estimate and percentile interval.

    Raises:
        ValueError: Scores are empty or nonfinite, or settings are invalid.
    """
    generator = np.random.default_rng(seed)
    samples = _bootstrap_distribution(scores, statistic, repetitions, generator, "run")
    lower, upper = _percentile(samples, confidence)
    return BootstrapInterval(float(statistic(scores)), lower, upper, confidence, repetitions, "run")


def robot_bootstrap_ci(
    scores: Scores,
    statistic: Callable[[Any], float] = interquartile_mean,
    repetitions: int = RLIABLE_REPETITIONS,
    confidence: float = RLIABLE_CONFIDENCE,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a percentile bootstrap interval with the robot as the resampling unit.

    Robots are drawn with replacement, then each drawn robot's runs (seeds) with
    replacement: rliable's ``task_bootstrap=True`` with robots as tasks. With one run
    per robot this is the plain bootstrap over robots.

    Args:
        scores (Scores): Mapping from robot to its runs, or ``[runs, robots]``.
        statistic (Callable): Aggregate (IQM by default). Resampled mappings get keys
            ``"<robot>#<i>"`` because a robot can be drawn twice.
        repetitions (int): Bootstrap replicates (rliable default 50000).
        confidence (float): Interval size (rliable default 0.95).
        seed (int): Seed of the local random generator.

    Returns:
        BootstrapInterval: Estimate and percentile interval, ``unit="robot"``.

    Raises:
        ValueError: Scores are empty or nonfinite, or settings are invalid.
    """
    generator = np.random.default_rng(seed)
    samples = _bootstrap_distribution(scores, statistic, repetitions, generator, "robot")
    lower, upper = _percentile(samples, confidence)
    return BootstrapInterval(float(statistic(scores)), lower, upper, confidence, repetitions, "robot")


def performance_gap(
    level_scores: Mapping[str, Scores],
    reference: str = "L0",
    statistic: Callable[[Any], float] = interquartile_mean,
    unit: str = "robot",
    repetitions: int = RLIABLE_REPETITIONS,
    confidence: float = RLIABLE_CONFIDENCE,
    seed: int = 0,
) -> dict[str, BootstrapInterval]:
    """Return ``statistic(Lk) - statistic(reference)`` for every other level, with intervals.

    The levels hold different robots, so each level is resampled independently and the
    interval is the percentile interval of the replicate differences. For error metrics
    (MPJPE) a positive gap means worse than the reference.

    Args:
        level_scores (Mapping[str, Scores]): Level name to its scores (robot to runs).
        reference (str): Level subtracted from the others (proposal v4: L0).
        statistic (Callable): Aggregate (IQM by default).
        unit (str): ``robot`` (robot bootstrap) or ``run`` (stratified over robots).
        repetitions (int): Bootstrap replicates (rliable default 50000).
        confidence (float): Interval size (rliable default 0.95).
        seed (int): Seed; each level gets its own generator derived from it.

    Returns:
        dict[str, BootstrapInterval]: Gap of every level except the reference.

    Raises:
        KeyError: The reference level is missing.
    """
    if reference not in level_scores:
        raise KeyError(f"reference level {reference!r} has no scores")
    seeds = np.random.SeedSequence(seed).spawn(len(level_scores))
    generators = {name: np.random.default_rng(child) for name, child in zip(level_scores, seeds)}
    base_samples = _bootstrap_distribution(level_scores[reference], statistic, repetitions, generators[reference], unit)
    base_estimate = float(statistic(level_scores[reference]))
    gaps: dict[str, BootstrapInterval] = {}
    for name, scores in level_scores.items():
        if name == reference:
            continue
        samples = _bootstrap_distribution(scores, statistic, repetitions, generators[name], unit)
        lower, upper = _percentile(samples - base_samples, confidence)
        gaps[name] = BootstrapInterval(float(statistic(scores)) - base_estimate, lower, upper, confidence, repetitions, unit)
    return gaps
