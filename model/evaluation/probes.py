"""Where is the physics: in the encoder or in the heads? (proposal v4, B0 and Sec. 5.3)

Tools, from probing work in NLP and representation learning:

* **Layer-wise embeddings.** Forward hooks on ``encoder.input_projection`` and every
  ``encoder.layers[i]`` capture the joint states after each block; ``final`` is the
  encoder output. One probe per layer, on frozen features, as Alain & Bengio [1] ("we
  take the features of each layer separately and we fit a linear classifier").
* **Linear probe.** Closed-form ridge regression, ``min ||y - Xw - b||^2 + alpha ||w||^2``
  with an unpenalized intercept. ``alpha`` is chosen on a validation split from
  scikit-learn ``RidgeCV``'s default grid ``(0.1, 1.0, 10.0)`` (documented default,
  scikit-learn 1.9); a choice at the edge of the grid is flagged.
* **MLP probe.** One hidden layer of 1000 ReLU units, as the 1000-dimensional default
  of Hewitt & Liang [2] and the "single 1000D hidden layer with a ReLU activation" of
  Zhang & Bowman [3]. Training follows Hewitt & Liang's code
  (``control-tasks/regimen.py``, Apache-2.0, adapted): Adam at learning rate 0.001,
  ``ReduceLROnPlateau(factor=0.5, patience=0)`` on the validation loss, keep the best
  parameters (improvement > 0.001), stop when 4 evaluations pass without one, at most
  40 epochs (their configs: ``epochs: 40``), batches of 20 examples (their
  ``batch_size: 20`` sentences; here 20 robot frames, a frame's joints being the
  analog of a sentence's words). Targets are standardized with training statistics for
  the MSE loss ([H] implementation choice: equal weight per output, as R²'s average).
* **Control task and selectivity** (Hewitt & Liang [2]): a control behavior is
  "sampled independently at random for each word type" from the empirical label
  distribution, and "selectivity is defined as the difference between linguistic task
  accuracy and control task accuracy". **[H] regression adaptation:** a type is a joint
  identity within a robot, ``(embodiment record, joint index)`` (the robot for
  robot-level targets); each type gets the target of one random training row, so the
  control targets follow the empirical target distribution and depend on the type
  only. Selectivity is ``R²(task) - R²(control)``. The closest regression control
  found, Liétard et al. [6], permutes the targets instead of fixing one per type.
* **Random-encoder baseline.** The same architecture with a fresh initialization,
  frozen: the "untrained LSTM baseline—frozen after random initialization" of Zhang &
  Bowman [3] and the "Untrained encoders with random weights" of Conneau et al. [4].
  Both used MLP classifiers, so linear and MLP probes both run.
* **Head-capacity test** (proposal B0's decision rule): freeze a *random* encoder,
  train only the heads, and compare its loss with the trained model's. A small gap
  means the heads alone can fit the physics.
* **Head re-initialization transfer**: freeze the trained encoder, re-initialize every
  non-encoder parameter, train the heads on training robots only, and evaluate on
  held-out robots, against the same procedure on a random encoder.

Head training uses the Hugging Face ``Trainer`` defaults that pretraining used (AdamW,
betas (0.9, 0.999), eps 1e-8, weight decay 0, gradient clipping at norm 1, linear decay
after warmup) and the learning rate, warmup and batch size of ``docs/training.md``
(5e-4, 100, 64). ``HeadTrainingSettings.from_checkpoint`` reads the run's own
``training_args.bin`` instead when it is there.

Probe targets come from the collator batch and the records:

* ``fk_position``: joint positions over the characteristic length.
* ``response_diagonal``: diagonal of the normalized response operator
  ``G[6+j, j] tau_j^peak / s_j^R`` (``physics_metrics.normalized_response_operator``),
  on valid response rows and actuated joints.
* ``mass_diagonal`` and ``gravity_torque``: columns ``mass_diagonal`` and ``gravity`` of
  the dynamics features, mapped by ``asinh(x / s)`` with the schema scale ``s``
  (``model.physics_schema``), the transform the physics tokenizer applies.
* Robot level, on the masked mean of the joint states: ``log_total_mass`` and
  ``log_characteristic_length`` ([H] log transform of positive quantities).

Note: with Physics Cross Attention the dynamics features are encoder inputs, so their
probes are expected to succeed; the random-encoder baseline and the control task tell
how much of that is the encoder's work.

References:
    [1] G. Alain, Y. Bengio. Understanding intermediate layers using linear classifier
        probes. ICLR 2017 (workshop track). arXiv:1610.01644.
    [2] J. Hewitt, P. Liang. Designing and Interpreting Probes with Control Tasks.
        EMNLP-IJCNLP 2019. DOI 10.18653/v1/D19-1275. Code:
        https://github.com/john-hewitt/control-tasks (Apache-2.0).
    [3] K. Zhang, S. R. Bowman. Language Modeling Teaches You More than Translation
        Does: Lessons Learned Through Auxiliary Syntactic Task Analysis. BlackboxNLP
        2018. DOI 10.18653/v1/W18-5448 (MLP details in arXiv:1809.10040).
    [4] A. Conneau, G. Kruszewski, G. Lample, L. Barrault, M. Baroni. What you can cram
        into a single $&!#* vector: Probing sentence embeddings for linguistic
        properties. ACL 2018. DOI 10.18653/v1/P18-1198.
    [5] B. Ai et al. Towards Embodiment Scaling Laws in Robot Locomotion. CoRL 2025.
        arXiv:2505.05753 (20% held-out fraction).
    [6] B. Liétard, M. Abdou, A. Søgaard. Do Language Models Know the Way to Rome?
        BlackboxNLP 2021. DOI 10.18653/v1/2021.blackboxnlp-1.40.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn

from model.evaluation.physics_metrics import move_batch, true_response_operators
from model.physics_schema import feature_names, feature_scales

# scikit-learn RidgeCV default alphas.
RIDGE_PENALTIES = (0.1, 1.0, 10.0)
# Hewitt & Liang: probe hidden size 1000 (also Zhang & Bowman's 1000D layer), Adam lr 0.001,
# LR halved on a plateau, early stop after 4 evaluations without improvement > 0.001, 40 epochs,
# batches of 20 examples.
MLP_HIDDEN_SIZE = 1000
MLP_LEARNING_RATE = 1e-3
MLP_MAX_EPOCHS = 40
MLP_BATCH_EXAMPLES = 20
MLP_PATIENCE = 4
MLP_MIN_IMPROVEMENT = 1e-3
# [H] validation and test fractions of probe data, mirroring the 20% held out by Ai et al.
PROBE_VAL_FRACTION = 0.2
PROBE_TEST_FRACTION = 0.2
JOINT_TARGETS = ("fk_position", "response_diagonal", "mass_diagonal", "gravity_torque")
ROBOT_TARGETS = ("log_total_mass", "log_characteristic_length")


# ----------------------------------------------------------------------------- embeddings


@torch.no_grad()
def capture_layer_embeddings(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the joint states after the input projection, every layer, and the output.

    Args:
        model (nn.Module): ``DaGetForPhysicsPretraining`` (needs ``encode`` and
            ``encoder.layers``).
        batch (Mapping[str, torch.Tensor]): Collator output.

    Returns:
        dict[str, torch.Tensor]: ``input_projection``, ``layer_0`` ... and ``final``,
            each ``[B, J, hidden]`` with padded joints zeroed.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str) -> Callable:
        def store(_module: nn.Module, _inputs: Any, output: Any) -> None:
            captured[name] = (output[0] if isinstance(output, (tuple, list)) else output).detach()

        return store

    encoder = model.encoder
    if hasattr(encoder, "input_projection"):
        handles.append(encoder.input_projection.register_forward_hook(hook("input_projection")))
    for index, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_hook(hook(f"layer_{index}")))
    was_training = model.training
    model.eval()
    try:
        final = model.encode(**batch)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    mask = batch["joint_mask"][..., None]
    order = (["input_projection"] if "input_projection" in captured else []) + [f"layer_{i}" for i in range(len(encoder.layers))]
    result = {name: captured[name].masked_fill(~mask, 0) for name in order if name in captured}
    result["final"] = final.detach().masked_fill(~mask, 0)
    return result


# ----------------------------------------------------------------------------- targets


def _schema_transform(values: torch.Tensor, group: str, column: str) -> torch.Tensor:
    """Apply the physics tokenizer's ``asinh(x / s)`` of one schema column."""
    index = feature_names(group).index(column)
    scale = feature_scales(group)[index]
    return torch.asinh(values / scale) if scale > 0 else values


def probe_targets(
    batch: Mapping[str, torch.Tensor],
    samples: Sequence[Mapping[str, Any]] | None = None,
    records: Sequence[Any] | None = None,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor]]:
    """Return per-joint and robot-level probe targets of a batch.

    Args:
        batch (Mapping[str, torch.Tensor]): Collator output.
        samples (Sequence[Mapping] | None): The dataset items of the batch, for the
            response operator and the robot targets.
        records (Sequence[EmbodimentRecord] | None): Records indexed by ``embodiment``.

    Returns:
        tuple: Joint targets ``{name: (values [B, J, D], valid [B, J])}`` and robot
            targets ``{name: [B, D]}``. Targets whose inputs are missing are left out.
    """
    mask = batch["joint_mask"].cpu()
    joint: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if "fk_position" in batch:
        joint["fk_position"] = (batch["fk_position"].cpu().double(), mask)
    if "dynamics_features" in batch:
        features = batch["dynamics_features"].cpu().double()
        names = feature_names("dynamics")
        joint["mass_diagonal"] = (_schema_transform(features[..., names.index("mass_diagonal")], "dynamics", "mass_diagonal")[..., None], mask)
        joint["gravity_torque"] = (_schema_transform(features[..., names.index("gravity")], "dynamics", "gravity")[..., None], mask)
    robot: dict[str, torch.Tensor] = {}
    if samples is not None and records is not None:
        operators, actuated = true_response_operators(batch, samples, records)
        valid = mask & actuated
        if "response_mask" in batch:
            valid = valid & batch["response_mask"].cpu()
        joint["response_diagonal"] = (operators.diagonal(dim1=-2, dim2=-1)[..., None], valid)
        graphs = [records[sample["embodiment"]].graph for sample in samples]
        robot["log_total_mass"] = torch.tensor([[math.log(max(graph.total_mass, 1e-9))] for graph in graphs], dtype=torch.float64)
        robot["log_characteristic_length"] = torch.tensor([[math.log(graph.characteristic_length)] for graph in graphs], dtype=torch.float64)
    return joint, robot


@dataclass
class ProbeData:
    """Layer features and targets of many samples, flattened to valid joints.

    Attributes:
        features (dict[str, torch.Tensor]): Layer to ``[N, hidden]`` joint rows.
        pooled (dict[str, torch.Tensor]): Layer to ``[S, hidden]`` masked means per sample.
        joint_targets (dict[str, torch.Tensor]): Target to ``[N, D]`` (NaN where invalid).
        joint_valid (dict[str, torch.Tensor]): Target to bool ``[N]``.
        robot_targets (dict[str, torch.Tensor]): Target to ``[S, D]``.
        row_sample (torch.Tensor): Long ``[N]`` sample of each row.
        row_type (torch.Tensor): Long ``[N]`` control-task type ``(embodiment, joint)``.
        sample_embodiment (torch.Tensor): Long ``[S]`` embodiment of each sample.
        sample_group (torch.Tensor): Long ``[S]`` split group (e.g. episode) of each sample.
    """

    features: dict[str, torch.Tensor]
    pooled: dict[str, torch.Tensor]
    joint_targets: dict[str, torch.Tensor]
    joint_valid: dict[str, torch.Tensor]
    robot_targets: dict[str, torch.Tensor]
    row_sample: torch.Tensor
    row_type: torch.Tensor
    sample_embodiment: torch.Tensor
    sample_group: torch.Tensor


def collect_probe_data(
    model: nn.Module,
    samples: Sequence[Mapping[str, Any]],
    collator: Callable[[Sequence[Mapping[str, Any]]], dict[str, torch.Tensor]],
    records: Sequence[Any],
    groups: Sequence[int] | None = None,
    batch_size: int = 64,
) -> ProbeData:
    """Encode samples and gather layer features with their probe targets.

    Args:
        model (nn.Module): Model whose encoder is probed.
        samples (Sequence[Mapping]): Dataset items (with ``embodiment``).
        collator (Callable): ``PhysicsDataCollator`` for ``records``.
        records (Sequence[EmbodimentRecord]): Embodiments.
        groups (Sequence[int] | None): Split group of each sample (episodes, so that
            near-identical frames of one episode stay in one split); each sample is
            its own group if None.
        batch_size (int): Samples per forward pass (``docs/training.md``: 64).

    Returns:
        ProbeData: Features and targets.
    """
    device = next(model.parameters()).device
    features: dict[str, list[torch.Tensor]] = {}
    pooled: dict[str, list[torch.Tensor]] = {}
    joint_values: dict[str, list[torch.Tensor]] = {}
    joint_valid: dict[str, list[torch.Tensor]] = {}
    robot_values: dict[str, list[torch.Tensor]] = {}
    row_sample, row_type = [], []
    max_joints = max(record.graph.num_joints for record in records)
    for start in range(0, len(samples), batch_size):
        chunk = list(samples[start : start + batch_size])
        batch = collator(chunk)
        mask = batch["joint_mask"]
        layers = capture_layer_embeddings(model, move_batch(batch, device))
        flat_mask = mask.reshape(-1)
        for name, value in layers.items():
            value = value.cpu().double()
            features.setdefault(name, []).append(value.reshape(-1, value.shape[-1])[flat_mask])
            weights = mask[..., None].double()
            pooled.setdefault(name, []).append((value * weights).sum(1) / weights.sum(1).clamp_min(1))
        joint, robot = probe_targets(batch, chunk, records)
        for name, (values, valid) in joint.items():
            values = torch.where(valid[..., None], values, torch.full_like(values, float("nan")))
            joint_values.setdefault(name, []).append(values.reshape(-1, values.shape[-1])[flat_mask])
            joint_valid.setdefault(name, []).append(valid.reshape(-1)[flat_mask])
        for name, values in robot.items():
            robot_values.setdefault(name, []).append(values)
        embodiments = torch.tensor([sample["embodiment"] for sample in chunk])
        index = torch.arange(start, start + len(chunk))
        joints = torch.arange(mask.shape[1])
        row_sample.append(index[:, None].expand_as(mask)[mask])
        row_type.append((embodiments[:, None] * max_joints + joints[None, :]).expand_as(mask)[mask])
    sample_embodiment = torch.tensor([sample["embodiment"] for sample in samples], dtype=torch.long)
    sample_group = torch.tensor(list(groups) if groups is not None else range(len(samples)), dtype=torch.long)
    return ProbeData(
        features={name: torch.cat(items) for name, items in features.items()},
        pooled={name: torch.cat(items) for name, items in pooled.items()},
        joint_targets={name: torch.cat(items) for name, items in joint_values.items()},
        joint_valid={name: torch.cat(items) for name, items in joint_valid.items()},
        robot_targets={name: torch.cat(items) for name, items in robot_values.items()},
        row_sample=torch.cat(row_sample),
        row_type=torch.cat(row_type),
        sample_embodiment=sample_embodiment,
        sample_group=sample_group,
    )


# ----------------------------------------------------------------------------- splits and scores


def split_groups(
    groups: torch.Tensor,
    val_fraction: float = PROBE_VAL_FRACTION,
    test_fraction: float = PROBE_TEST_FRACTION,
    seed: int = 0,
) -> torch.Tensor:
    """Assign every item to train (0), validation (1) or test (2) by its group.

    Args:
        groups (torch.Tensor): Long ``[N]`` group ids; a group never spans two splits.
        val_fraction (float): Fraction of groups for validation ([H] 0.2).
        test_fraction (float): Fraction of groups for test ([H] 0.2).
        seed (int): Shuffle seed.

    Returns:
        torch.Tensor: Long ``[N]`` split ids.

    Raises:
        ValueError: There are fewer than three groups or the fractions leave no training.
    """
    unique = torch.unique(groups)
    if unique.numel() < 3:
        raise ValueError("probe splits need at least three groups")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("fractions must be positive and leave groups for training")
    order = unique[torch.randperm(unique.numel(), generator=torch.Generator().manual_seed(seed))]
    count = unique.numel()
    test = max(1, round(test_fraction * count))
    val = max(1, round(val_fraction * count))
    test = min(test, count - 2)
    val = min(val, count - test - 1)
    assignment = torch.zeros(int(groups.max()) + 1, dtype=torch.long)
    assignment[order[:test]] = 2
    assignment[order[test : test + val]] = 1
    return assignment[groups]


def r2_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    """Return the coefficient of determination averaged over outputs.

    ``1 - SS_res / SS_tot`` per output column against the evaluated set's own mean,
    averaged uniformly (scikit-learn's ``uniform_average``). Constant columns are
    skipped; NaN if every column is constant or there are fewer than two rows.

    Args:
        target (torch.Tensor): ``[N, D]``.
        prediction (torch.Tensor): ``[N, D]``.

    Returns:
        float: R².
    """
    target, prediction = target.double(), prediction.double()
    if target.ndim == 1:
        target, prediction = target[:, None], prediction[:, None]
    if target.shape[0] < 2:
        return float("nan")
    residual = (target - prediction).pow(2).sum(0)
    total = (target - target.mean(0)).pow(2).sum(0)
    keep = total > 1e-12 * target.shape[0]
    if not bool(keep.any()):
        return float("nan")
    return float((1 - residual[keep] / total[keep]).mean())


def control_targets(types: torch.Tensor, targets: torch.Tensor, train: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Return control targets: one fixed random training target per type ([H] regression form).

    Args:
        types (torch.Tensor): Long ``[N]`` type of each row.
        targets (torch.Tensor): ``[N, D]`` task targets.
        train (torch.Tensor): Bool ``[N]`` training rows, the empirical distribution.
        seed (int): Seed.

    Returns:
        torch.Tensor: ``[N, D]`` control targets, constant within a type.

    Raises:
        ValueError: There are no training rows.
    """
    pool = torch.nonzero(train).squeeze(-1)
    if pool.numel() == 0:
        raise ValueError("control targets need training rows")
    unique, inverse = torch.unique(types, return_inverse=True)
    generator = torch.Generator().manual_seed(seed)
    drawn = pool[torch.randint(0, pool.numel(), (unique.numel(),), generator=generator)]
    return targets[drawn][inverse]


# ----------------------------------------------------------------------------- probes


@dataclass
class RidgeProbe:
    """A fitted ridge probe.

    Attributes:
        weight (torch.Tensor): ``[H, D]``.
        bias (torch.Tensor): ``[D]``.
        penalty (float): Chosen alpha.
        validation_r2 (float): R² of the chosen alpha on the validation split.
        penalty_at_edge (bool): The chosen alpha is the smallest or largest of the grid.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    penalty: float
    validation_r2: float
    penalty_at_edge: bool

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the probe.

        Args:
            features (torch.Tensor): ``[N, H]``.

        Returns:
            torch.Tensor: ``[N, D]``.
        """
        return features.double() @ self.weight + self.bias


def fit_ridge_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    penalties: Sequence[float] = RIDGE_PENALTIES,
) -> RidgeProbe:
    """Fit ridge regression in closed form and choose the penalty on validation R².

    Solves ``min ||y_c - X_c w||^2 + alpha ||w||^2`` on centered data with the SVD
    ``X_c = U S V^T``: ``w = V diag(s / (s^2 + alpha)) U^T y_c``, and
    ``b = mean(y) - mean(X) w``.

    Args:
        train_x (torch.Tensor): ``[N, H]``.
        train_y (torch.Tensor): ``[N, D]``.
        val_x (torch.Tensor): ``[M, H]``.
        val_y (torch.Tensor): ``[M, D]``.
        penalties (Sequence[float]): Candidate alphas (``RidgeCV`` default grid).

    Returns:
        RidgeProbe: The probe with the best validation R².
    """
    x, y = train_x.double(), train_y.double()
    if y.ndim == 1:
        y = y[:, None]
    x_mean, y_mean = x.mean(0), y.mean(0)
    u, s, vh = torch.linalg.svd(x - x_mean, full_matrices=False)
    projected = u.T @ (y - y_mean)
    best: RidgeProbe | None = None
    ordered = sorted(float(value) for value in penalties)
    for penalty in ordered:
        weight = vh.T @ ((s / (s.pow(2) + penalty))[:, None] * projected)
        bias = y_mean - x_mean @ weight
        probe = RidgeProbe(weight, bias, penalty, float("nan"), False)
        score = r2_score(val_y, probe.predict(val_x))
        probe.validation_r2 = score
        if best is None or (math.isfinite(score) and (not math.isfinite(best.validation_r2) or score > best.validation_r2)):
            best = probe
    best.penalty_at_edge = len(ordered) > 1 and best.penalty in (ordered[0], ordered[-1])
    return best


class MlpProbe(nn.Module):
    """One-hidden-layer ReLU probe ``W2 relu(W1 h + b1) + b2``."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = MLP_HIDDEN_SIZE) -> None:
        """Build the probe.

        Args:
            input_size (int): Feature width.
            output_size (int): Target width.
            hidden_size (int): Hidden units (Hewitt & Liang default: 1000).
        """
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return predictions ``[N, D]``.

        Args:
            features (torch.Tensor): ``[N, H]``.

        Returns:
            torch.Tensor: ``[N, D]``.
        """
        return self.network(features)


@dataclass
class MlpSettings:
    """Training settings of the MLP probe (Hewitt & Liang's regimen unless noted).

    Attributes:
        hidden_size (int): Hidden units (1000).
        learning_rate (float): Adam learning rate (0.001, their ``regimen.py``).
        max_epochs (int): Epoch limit (40, their configs).
        batch_examples (int): Examples (frames) per batch (20, their configs).
        patience (int): Evaluations without improvement before stopping (4).
        min_improvement (float): Required decrease of the validation loss (0.001).
        weight_decay (float): Adam weight decay (0, their configs' default).
    """

    hidden_size: int = MLP_HIDDEN_SIZE
    learning_rate: float = MLP_LEARNING_RATE
    max_epochs: int = MLP_MAX_EPOCHS
    batch_examples: int = MLP_BATCH_EXAMPLES
    patience: int = MLP_PATIENCE
    min_improvement: float = MLP_MIN_IMPROVEMENT
    weight_decay: float = 0.0


def train_mlp_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    train_examples: torch.Tensor | None = None,
    settings: MlpSettings | None = None,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Train an MLP probe with Hewitt & Liang's regimen and return its predictor.

    Args:
        train_x (torch.Tensor): ``[N, H]``.
        train_y (torch.Tensor): ``[N, D]``.
        val_x (torch.Tensor): ``[M, H]``.
        val_y (torch.Tensor): ``[M, D]``.
        train_examples (torch.Tensor | None): Long ``[N]`` example (frame) of each row;
            a batch holds the rows of ``batch_examples`` examples. Each row is its own
            example if None.
        settings (MlpSettings | None): Training settings; defaults if None.
        seed (int): Seed of initialization and shuffling.

    Returns:
        Callable: Maps ``[K, H]`` features to ``[K, D]`` predictions in target units.
    """
    settings = settings or MlpSettings()
    torch.manual_seed(seed)
    x, y = train_x.float(), train_y.double()
    if y.ndim == 1:
        y = y[:, None]
    mean, std = y.mean(0), y.std(0, unbiased=False).clamp_min(1e-12)
    y_scaled = ((y - mean) / std).float()
    val_scaled = ((val_y.double().reshape(len(val_y), -1) - mean) / std).float()
    probe = MlpProbe(x.shape[1], y.shape[1], settings.hidden_size)
    optimizer = torch.optim.Adam(probe.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=0)
    examples = train_examples if train_examples is not None else torch.arange(len(x))
    unique, inverse = torch.unique(examples, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    counts = torch.bincount(inverse, minlength=unique.numel())
    starts = torch.cumsum(counts, 0) - counts
    rows_of = [order[start : start + count] for start, count in zip(starts.tolist(), counts.tolist())]
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_index = math.inf, copy.deepcopy(probe.state_dict()), 0
    val_x = val_x.float()
    for epoch in range(settings.max_epochs):
        probe.train()
        permutation = torch.randperm(len(rows_of), generator=generator).tolist()
        for start in range(0, len(permutation), settings.batch_examples):
            rows = torch.cat([rows_of[index] for index in permutation[start : start + settings.batch_examples]])
            optimizer.zero_grad()
            loss = (probe(x[rows]) - y_scaled[rows]).pow(2).mean()
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_loss = float((probe(val_x) - val_scaled).pow(2).mean()) if len(val_x) else float(loss)
        scheduler.step(val_loss)
        if val_loss < best_loss - settings.min_improvement:
            best_loss, best_state, best_index = val_loss, copy.deepcopy(probe.state_dict()), epoch
        elif best_index < epoch - settings.patience:
            break
    probe.load_state_dict(best_state)
    probe.eval()

    def predict(features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return probe(features.float()).double() * std + mean

    return predict


def run_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    types: torch.Tensor,
    split: torch.Tensor,
    examples: torch.Tensor | None = None,
    kind: str = "linear",
    penalties: Sequence[float] = RIDGE_PENALTIES,
    mlp: MlpSettings | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a probe on the task and on its control task; report test R² and selectivity.

    Args:
        features (torch.Tensor): ``[N, H]`` rows (NaN-free).
        targets (torch.Tensor): ``[N, D]`` task targets.
        types (torch.Tensor): Long ``[N]`` control-task type of each row.
        split (torch.Tensor): Long ``[N]``: 0 train, 1 validation, 2 test.
        examples (torch.Tensor | None): Long ``[N]`` example of each row (MLP batches).
        kind (str): ``linear`` (ridge) or ``mlp``.
        penalties (Sequence[float]): Ridge grid.
        mlp (MlpSettings | None): MLP settings.
        seed (int): Seed of the control task and the MLP.

    Returns:
        dict[str, Any]: ``task_r2``, ``control_r2``, ``selectivity``, ``train_rows``,
            ``test_rows``, and for ridge the chosen ``penalty`` (task and control) and
            ``penalty_at_edge``.

    Raises:
        ValueError: ``kind`` is unknown.
    """
    if kind not in ("linear", "mlp"):
        raise ValueError("kind must be 'linear' or 'mlp'")
    train, val, test = split == 0, split == 1, split == 2
    control = control_targets(types, targets, train, seed)
    result: dict[str, Any] = {"train_rows": int(train.sum()), "test_rows": int(test.sum())}
    for name, target in (("task", targets), ("control", control)):
        if kind == "linear":
            probe = fit_ridge_probe(features[train], target[train], features[val], target[val], penalties)
            prediction = probe.predict(features[test])
            result[f"{name}_penalty"] = probe.penalty
            result[f"{name}_penalty_at_edge"] = probe.penalty_at_edge
        else:
            predictor = train_mlp_probe(
                features[train], target[train], features[val], target[val],
                None if examples is None else examples[train], mlp, seed,
            )
            prediction = predictor(features[test])
        result[f"{name}_r2"] = r2_score(target[test], prediction)
    result["selectivity"] = result["task_r2"] - result["control_r2"]
    return result


def layerwise_probes(
    data: ProbeData,
    kinds: Sequence[str] = ("linear", "mlp"),
    targets: Sequence[str] | None = None,
    penalties: Sequence[float] = RIDGE_PENALTIES,
    mlp: MlpSettings | None = None,
    val_fraction: float = PROBE_VAL_FRACTION,
    test_fraction: float = PROBE_TEST_FRACTION,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run every probe kind for every layer and target.

    Joint targets use the joint rows and the ``(embodiment, joint)`` types; robot
    targets use the per-sample pooled states and the embodiment as type. Splits are by
    ``data.sample_group``.

    Args:
        data (ProbeData): Collected features and targets.
        kinds (Sequence[str]): Probe kinds.
        targets (Sequence[str] | None): Target names; all available if None.
        penalties (Sequence[float]): Ridge grid.
        mlp (MlpSettings | None): MLP settings.
        val_fraction (float): Validation fraction of groups.
        test_fraction (float): Test fraction of groups.
        seed (int): Seed of splits, control tasks and MLPs.

    Returns:
        dict: ``{layer: {target: {kind: run_probe result}}}``.
    """
    sample_split = split_groups(data.sample_group, val_fraction, test_fraction, seed)
    row_split = sample_split[data.row_sample]
    wanted = set(targets) if targets is not None else None
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for layer, rows in data.features.items():
        layer_report: dict[str, dict[str, Any]] = {}
        for name, values in data.joint_targets.items():
            if wanted is not None and name not in wanted:
                continue
            valid = data.joint_valid[name] & torch.isfinite(values).all(-1)
            if int(valid.sum()) < 3 or len(torch.unique(row_split[valid])) < 3:
                layer_report[name] = {"skipped": "too few valid rows in some split"}
                continue
            layer_report[name] = {
                kind: run_probe(
                    rows[valid], values[valid], data.row_type[valid], row_split[valid],
                    data.row_sample[valid], kind, penalties, mlp, seed,
                )
                for kind in kinds
            }
        for name, values in data.robot_targets.items():
            if wanted is not None and name not in wanted:
                continue
            layer_report[name] = {
                kind: run_probe(
                    data.pooled[layer], values, data.sample_embodiment, sample_split,
                    torch.arange(len(values)), kind, penalties, mlp, seed,
                )
                for kind in kinds
            }
        report[layer] = layer_report
    return report


# ----------------------------------------------------------------------------- heads


def random_like(model: nn.Module, seed: int = 0) -> nn.Module:
    """Return a freshly initialized model of the same class and config.

    Args:
        model (nn.Module): A ``PreTrainedModel`` (needs ``config``).
        seed (int): Seed of the initialization.

    Returns:
        nn.Module: New model on the same device, in eval mode.
    """
    torch.manual_seed(seed)
    fresh = type(model)(copy.deepcopy(model.config))
    return fresh.to(next(model.parameters()).device).eval()


def with_reinitialized_heads(model: nn.Module, seed: int = 0) -> nn.Module:
    """Return a copy whose encoder is the model's and whose other parameters are new.

    Args:
        model (nn.Module): Trained model (``encoder`` attribute).
        seed (int): Seed of the new heads.

    Returns:
        nn.Module: New model with the trained encoder weights.
    """
    fresh = random_like(model, seed)
    fresh.encoder.load_state_dict(model.encoder.state_dict())
    return fresh


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count encoder and non-encoder (head) parameters.

    Args:
        model (nn.Module): Model with an ``encoder``.

    Returns:
        dict[str, int]: ``encoder`` and ``heads``.
    """
    encoder = {id(parameter) for parameter in model.encoder.parameters()}
    total = {id(parameter): parameter.numel() for parameter in model.parameters()}
    encoder_count = sum(count for key, count in total.items() if key in encoder)
    return {"encoder": encoder_count, "heads": sum(total.values()) - encoder_count}


@dataclass
class HeadTrainingSettings:
    """How the heads are trained in the head tests.

    Defaults: ``docs/training.md`` (2000 steps, learning rate 5e-4, warmup 100, batch
    64) and the Hugging Face ``TrainingArguments`` defaults pretraining ran with (AdamW,
    betas (0.9, 0.999), eps 1e-8, weight decay 0, max gradient norm 1, linear schedule).

    Attributes:
        steps (int): Gradient steps.
        learning_rate (float): Peak learning rate.
        warmup_steps (int): Linear warmup steps.
        batch_size (int): Samples per batch.
        weight_decay (float): AdamW weight decay.
        max_grad_norm (float): Gradient clipping; 0 disables it.
        betas (tuple[float, float]): AdamW betas.
        epsilon (float): AdamW epsilon.
        source (str): Where the values come from.
    """

    steps: int = 2000
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    batch_size: int = 64
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    source: str = "docs/training.md and transformers.TrainingArguments defaults"

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> HeadTrainingSettings:
        """Read the settings of the run that produced a checkpoint, when recorded.

        ``Trainer.save_model`` writes ``training_args.bin`` next to the weights.

        Args:
            checkpoint (str | Path): ``save_pretrained`` directory.

        Returns:
            HeadTrainingSettings: The run's values, or the defaults when absent.
        """
        path = Path(checkpoint) / "training_args.bin"
        if not path.exists():
            return cls()
        try:
            arguments = torch.load(path, weights_only=False)
        except Exception:  # noqa: BLE001 - an unreadable file falls back to the documented defaults
            return cls()
        steps = int(getattr(arguments, "max_steps", -1))
        return cls(
            steps=steps if steps > 0 else cls.steps,
            learning_rate=float(getattr(arguments, "learning_rate", cls.learning_rate)),
            warmup_steps=int(getattr(arguments, "warmup_steps", cls.warmup_steps)),
            batch_size=int(getattr(arguments, "per_device_train_batch_size", cls.batch_size)),
            weight_decay=float(getattr(arguments, "weight_decay", cls.weight_decay)),
            max_grad_norm=float(getattr(arguments, "max_grad_norm", cls.max_grad_norm)),
            betas=(float(getattr(arguments, "adam_beta1", 0.9)), float(getattr(arguments, "adam_beta2", 0.999))),
            epsilon=float(getattr(arguments, "adam_epsilon", cls.epsilon)),
            source=f"{path}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-ready dict.

        Returns:
            dict[str, Any]: Field values.
        """
        return asdict(self)


def endless_batches(
    samples: Sequence[Mapping[str, Any]],
    collator: Callable[[Sequence[Mapping[str, Any]]], dict[str, torch.Tensor]],
    batch_size: int,
    seed: int = 0,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield shuffled collated batches forever.

    Args:
        samples (Sequence[Mapping]): Dataset items, or a map-style dataset (items are
            built lazily, one batch at a time).
        collator (Callable): Collator.
        batch_size (int): Samples per batch.
        seed (int): Shuffle seed.

    Yields:
        dict[str, torch.Tensor]: Batches.
    """
    if not samples:
        raise ValueError("no samples to train on")
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(samples), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            yield collator([samples[index] for index in order[start : start + batch_size]])


def fixed_batches(
    samples: Sequence[Mapping[str, Any]],
    collator: Callable[[Sequence[Mapping[str, Any]]], dict[str, torch.Tensor]],
    batch_size: int,
    seed: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Collate evaluation batches once, with seeded response torque changes.

    The collator draws the response tasks' torque changes from the global generator,
    so it is seeded here and every model is evaluated on the same draws.

    Args:
        samples (Sequence[Mapping]): Dataset items.
        collator (Callable): Collator.
        batch_size (int): Samples per batch.
        seed (int): Seed of the torque changes.

    Returns:
        list[dict[str, torch.Tensor]]: Batches.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        return [collator(list(samples[start : start + batch_size])) for start in range(0, len(samples), batch_size)]
    finally:
        torch.random.set_rng_state(state)


@torch.no_grad()
def evaluate_losses(model: nn.Module, batches: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, float]:
    """Return the mean weighted loss and mean task losses over batches.

    Args:
        model (nn.Module): Model returning ``loss`` and ``loss_components``.
        batches (Iterable[Mapping]): Evaluation batches.

    Returns:
        dict[str, float]: ``loss`` and ``loss_<task>``.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals: dict[str, list[float]] = {}
    for batch in batches:
        output = model(**move_batch(batch, device))
        if output.loss is not None:
            totals.setdefault("loss", []).append(float(output.loss))
        for name, value in (output.loss_components or {}).items():
            totals.setdefault(f"loss_{name}", []).append(float(value))
    if was_training:
        model.train()
    return {name: sum(values) / len(values) for name, values in totals.items()}


def train_heads(model: nn.Module, batches: Iterator[Mapping[str, torch.Tensor]], settings: HeadTrainingSettings) -> list[float]:
    """Train every non-encoder parameter with the encoder frozen.

    Args:
        model (nn.Module): Model to train in place.
        batches (Iterator[Mapping]): Training batches.
        settings (HeadTrainingSettings): Optimizer and schedule.

    Returns:
        list[float]: Training loss of every step.
    """
    from transformers import get_linear_schedule_with_warmup

    device = next(model.parameters()).device
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=settings.learning_rate, betas=settings.betas, eps=settings.epsilon, weight_decay=settings.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, settings.warmup_steps, settings.steps)
    model.train()
    model.encoder.eval()
    losses = []
    for _ in range(settings.steps):
        output = model(**move_batch(next(batches), device))
        optimizer.zero_grad()
        output.loss.backward()
        if settings.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, settings.max_grad_norm)
        optimizer.step()
        scheduler.step()
        losses.append(float(output.loss.detach()))
    model.eval()
    return losses


def head_capacity_test(
    trained: nn.Module,
    train_batches: Callable[[], Iterator[Mapping[str, torch.Tensor]]],
    eval_batches: Sequence[Mapping[str, torch.Tensor]],
    settings: HeadTrainingSettings,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare the trained model with heads trained on a frozen random encoder.

    Args:
        trained (nn.Module): Trained model (not modified).
        train_batches (Callable): Returns a fresh iterator of training batches.
        eval_batches (Sequence[Mapping]): Fixed evaluation batches.
        settings (HeadTrainingSettings): Head training.
        seed (int): Seed of the random encoder and heads.

    Returns:
        dict[str, Any]: Losses of both, ``loss_ratio`` (random-encoder heads over
            trained) per loss, and the training curve's first and last loss.
    """
    reference = evaluate_losses(trained, eval_batches)
    random_model = random_like(trained, seed)
    curve = train_heads(random_model, train_batches(), settings)
    random_losses = evaluate_losses(random_model, eval_batches)
    ratio = {name: random_losses[name] / reference[name] for name in reference if name in random_losses and reference[name] > 0}
    return {
        "trained": reference,
        "random_encoder_trained_heads": random_losses,
        "loss_ratio": ratio,
        "training_loss_first_last": [curve[0], curve[-1]] if curve else [],
        "settings": settings.to_dict(),
        "reading": "loss_ratio near 1 means heads on a random encoder fit the physics about as well (physics in the heads); "
        "well above 1 means the trained encoder carries it.",
    }


def head_reinit_transfer(
    trained: nn.Module,
    train_batches: Callable[[], Iterator[Mapping[str, torch.Tensor]]],
    eval_sets: Mapping[str, Sequence[Mapping[str, torch.Tensor]]],
    settings: HeadTrainingSettings,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-train heads on a frozen encoder (trained and random) and evaluate per set.

    Args:
        trained (nn.Module): Trained model (not modified).
        train_batches (Callable): Returns a fresh iterator over training robots' batches.
        eval_sets (Mapping[str, Sequence]): Name (e.g. ``train_robots``,
            ``heldout_robots``) to fixed evaluation batches.
        settings (HeadTrainingSettings): Head training.
        seed (int): Seed of the new heads and the random encoder.

    Returns:
        dict[str, Any]: Per set, the losses of the original model, of new heads on the
            trained encoder, and of new heads on a random encoder.
    """
    reinit = with_reinitialized_heads(trained, seed)
    train_heads(reinit, train_batches(), settings)
    random_model = random_like(trained, seed + 1)
    train_heads(random_model, train_batches(), settings)
    report: dict[str, Any] = {"settings": settings.to_dict()}
    for name, batches in eval_sets.items():
        if not batches:
            report[name] = {"skipped": "no batches"}
            continue
        report[name] = {
            "original": evaluate_losses(trained, batches),
            "trained_encoder_new_heads": evaluate_losses(reinit, batches),
            "random_encoder_new_heads": evaluate_losses(random_model, batches),
        }
    return report


def probe_settings_report(penalties: Sequence[float], mlp: MlpSettings, val_fraction: float, test_fraction: float) -> dict[str, Any]:
    """Describe the probe settings and their sources for a report.

    Args:
        penalties (Sequence[float]): Ridge grid.
        mlp (MlpSettings): MLP settings.
        val_fraction (float): Validation fraction.
        test_fraction (float): Test fraction.

    Returns:
        dict[str, Any]: Values with ``source`` notes.
    """
    return {
        "ridge_penalties": {"value": list(penalties), "source": "scikit-learn RidgeCV default alphas"},
        "mlp": {"value": asdict(mlp), "source": "Hewitt & Liang 2019 (paper, regimen.py, configs); hidden 1000 also Zhang & Bowman 2018"},
        "split_fractions": {"value": {"val": val_fraction, "test": test_fraction}, "source": "[H] mirrors the 20% held out by Ai et al. 2025"},
        "control_task": {"value": "one random training target per (embodiment, joint) type", "source": "[H] regression adaptation of Hewitt & Liang 2019"},
    }


__all__ = [
    "JOINT_TARGETS",
    "ROBOT_TARGETS",
    "HeadTrainingSettings",
    "MlpProbe",
    "MlpSettings",
    "ProbeData",
    "RidgeProbe",
    "capture_layer_embeddings",
    "collect_probe_data",
    "control_targets",
    "endless_batches",
    "evaluate_losses",
    "fit_ridge_probe",
    "fixed_batches",
    "head_capacity_test",
    "head_reinit_transfer",
    "layerwise_probes",
    "parameter_counts",
    "probe_settings_report",
    "probe_targets",
    "r2_score",
    "random_like",
    "run_probe",
    "split_groups",
    "train_heads",
    "train_mlp_probe",
    "with_reinitialized_heads",
]

