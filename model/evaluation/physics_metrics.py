"""Dimensionless physics metrics of the pretraining heads (proposal v4, Sec. 5.2).

Every metric is a ratio or is computed on targets the collator already normalizes, so
robots of different size and mass can be compared and pooled:

* ``relative_frobenius_error``: ``||G_hat - G||_F / ||G||_F`` of a response operator.
* ``relative_error``: ``||y_hat - y|| / ||y||`` per sample, e.g. of the response task R.
* ``exact_label_dynamics_errors``: FD and ID errors on rows whose torque label is exact
  (``torque_label_weight == 1``, simulator torques), in the collator's normalized units
  (FD over the data RMS ``sigma^a``, ID over ``tau^cont``) and relative to the target
  energy. The ID error is projected with ``contact_projection`` exactly as the ID loss
  (task C): internal contact forces are not observable from motion.
* ``fk_error_over_length``: FK targets are already positions divided by the robot's
  characteristic length, so the Euclidean error is in units of that length.
* ``contact_f1``: precision, recall and F1 of contact predictions.
* ``operator_from_response_head``: the joint-block operator a response head implies,
  one column per forward pass with a unit torque change ``e_j`` (fraction of peak
  torque, the collator's unit; ``CollatorSettings.response_range`` draws fractions in
  [0.02, 1.0], so 1 is inside the training range). It uses only ``model(**batch)``
  and ``predictions["response_acceleration"]``, so it works for any head design.

The true normalized operator is ``G_n[i, k] = G[6 + i, k] * tau_k^peak / s_i^R`` with
``G`` from ``model.dynamics.constrained_response`` (stored per frame as
``response_operator``) and ``s^R`` the record's ``response_scale``: the same
normalization as the collator's response targets (``model.training.dataset``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

# Keys whose presence makes ``DaGetForPhysicsPretraining.forward`` run a head other than R.
OTHER_TASK_KEYS = ("fk_position", "fd_torque", "id_acceleration", "inverse_response_input", "motor_input", "effect_input")


def move_batch(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move every tensor of a batch to a device.

    Args:
        batch (Mapping[str, Any]): Collator output; non-tensors are kept.
        device (torch.device | str): Target device.

    Returns:
        dict[str, Any]: New batch.
    """
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def summarize(values: torch.Tensor) -> dict[str, float]:
    """Return mean, median and count of the finite entries.

    Args:
        values (torch.Tensor): Per-sample values; NaN marks undefined ones.

    Returns:
        dict[str, float]: ``mean``, ``median``, ``count`` (NaN statistics when empty).
    """
    finite = values[torch.isfinite(values)].double()
    if finite.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "count": 0}
    return {"mean": float(finite.mean()), "median": float(finite.median()), "count": int(finite.numel())}


def relative_frobenius_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    row_mask: torch.Tensor | None = None,
    column_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``||P - T||_F / ||T||_F`` per sample over the masked block.

    Args:
        predicted (torch.Tensor): ``[B, R, C]`` operators.
        target (torch.Tensor): ``[B, R, C]`` true operators.
        row_mask (torch.Tensor | None): Bool ``[B, R]`` valid rows (e.g. joints).
        column_mask (torch.Tensor | None): Bool ``[B, C]`` valid columns (e.g. actuated
            joints; a passive joint never receives a torque change).

    Returns:
        torch.Tensor: ``[B]`` errors; NaN where the target block is zero.
    """
    weight = torch.ones_like(target, dtype=torch.bool)
    if row_mask is not None:
        weight = weight & row_mask[:, :, None]
    if column_mask is not None:
        weight = weight & column_mask[:, None, :]
    weight = weight.to(target.dtype)
    error = ((predicted.to(target.dtype) - target) * weight).pow(2).sum((-1, -2)).sqrt()
    norm = (target * weight).pow(2).sum((-1, -2)).sqrt()
    return torch.where(norm > 0, error / norm.clamp_min(torch.finfo(target.dtype).tiny), torch.full_like(norm, float("nan")))


def relative_error(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``||p - t|| / ||t||`` per sample over the masked entries.

    Args:
        predicted (torch.Tensor): ``[B, ...]`` predictions.
        target (torch.Tensor): ``[B, ...]`` targets.
        mask (torch.Tensor | None): Bool, broadcastable to ``target``.

    Returns:
        torch.Tensor: ``[B]`` errors; NaN where the masked target is zero.
    """
    weight = torch.ones_like(target) if mask is None else mask.to(target.dtype).expand_as(target)
    dims = tuple(range(1, target.ndim))
    error = ((predicted.to(target.dtype) - target) * weight).pow(2).sum(dims).sqrt()
    norm = (target * weight).pow(2).sum(dims).sqrt()
    return torch.where(norm > 0, error / norm.clamp_min(torch.finfo(target.dtype).tiny), torch.full_like(norm, float("nan")))


def _per_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``[B, J]`` values over masked joints; return per-sample means and validity."""
    mask = mask.to(values.dtype)
    counts = mask.sum(1)
    return (values * mask).sum(1) / counts.clamp_min(1), counts > 0


def exact_label_dynamics_errors(predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Return FD and ID errors on exact-label rows only (``torque_label_weight == 1``).

    ``*_mse`` averages squared normalized errors over joints, then samples (the repo's
    loss reduction, ``masked_joint_mean``); ``*_relative_mse`` is
    ``sum(error^2) / sum(target^2)`` over the same entries.

    Args:
        predictions (Mapping[str, torch.Tensor]): ``fd_acceleration`` and ``id_torque``
            from ``DaGetForPhysicsPretraining``.
        batch (Mapping[str, torch.Tensor]): Collator output.

    Returns:
        dict[str, float]: ``fd_mse``, ``fd_relative_mse``, ``fd_samples``, and the same
            for ``id``; NaN and 0 samples when a task has no exact rows.
    """
    results: dict[str, float] = {}
    exact = batch["torque_label_weight"] == 1 if "torque_label_weight" in batch else None
    specs = (("fd", "fd_acceleration", "fd_target", "fd_mask"), ("id", "id_torque", "id_target", "id_mask"))
    for name, prediction_key, target_key, mask_key in specs:
        if prediction_key not in predictions or target_key not in batch or exact is None:
            results.update({f"{name}_mse": float("nan"), f"{name}_relative_mse": float("nan"), f"{name}_samples": 0})
            continue
        target = batch[target_key].double()
        error = predictions[prediction_key].double() - target
        if name == "id" and "contact_projection" in batch:
            scale = batch["torque_scale"].double() + 1e-6 if "torque_scale" in batch else torch.ones_like(target)
            projection = batch["contact_projection"].double()
            error = (projection @ (error * scale)[..., None]).squeeze(-1) / scale
            target = (projection @ (target * scale)[..., None]).squeeze(-1) / scale
        mask = batch[mask_key] & exact[:, None]
        per_sample, valid = _per_sample_mean(error.pow(2), mask)
        energy = (target.pow(2) * mask).sum()
        results[f"{name}_mse"] = float(per_sample[valid].mean()) if bool(valid.any()) else float("nan")
        results[f"{name}_relative_mse"] = float((error.pow(2) * mask).sum() / energy) if float(energy) > 0 else float("nan")
        results[f"{name}_samples"] = int(valid.sum())
    return results


def fk_error_over_length(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean joint-position error per sample, in characteristic lengths.

    Args:
        predicted (torch.Tensor): ``[B, J, 3]`` predicted positions over L.
        target (torch.Tensor): ``[B, J, 3]`` ``fk_position`` (positions over L).
        mask (torch.Tensor): Bool ``[B, J]`` valid joints.

    Returns:
        torch.Tensor: ``[B]`` mean ``||p_hat - p|| / L``; NaN for samples without joints.
    """
    distance = (predicted.double() - target.double()).norm(dim=-1)
    per_sample, valid = _per_sample_mean(distance, mask)
    return torch.where(valid, per_sample, torch.full_like(per_sample, float("nan")))


def contact_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None, threshold: float = 0.5) -> dict[str, float]:
    """Return precision, recall and F1 of contact predictions.

    The definitions match ``model.training.contacts.evaluate_contacts``. The threshold
    is on the probability ``sigmoid(logit)``; 0.5 is the decision rule of a calibrated
    classifier with equal error costs.

    Args:
        logits (torch.Tensor): Contact logits, any shape.
        labels (torch.Tensor): Bool or {0, 1} labels of the same shape.
        mask (torch.Tensor | None): Bool, which entries count.
        threshold (float): Probability above which a contact is predicted.

    Returns:
        dict[str, float]: ``precision``, ``recall``, ``f1``, ``count``.
    """
    predicted = torch.sigmoid(logits.double()) > threshold
    reference = labels.bool()
    if mask is not None:
        predicted, reference = predicted[mask.bool()], reference[mask.bool()]
    hits = float((predicted & reference).sum())
    precision = hits / max(float(predicted.sum()), 1.0)
    recall = hits / max(float(reference.sum()), 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "count": int(reference.numel()),
    }


@dataclass
class ResponseOperator:
    """The operator a response head implies.

    Attributes:
        joint (torch.Tensor): ``[B, J, J]``; column ``k`` is the predicted normalized
            joint-acceleration change for a unit normalized torque change of joint k.
        base (torch.Tensor | None): ``[B, 6, J]`` base rows, when the head predicts them.
    """

    joint: torch.Tensor
    base: torch.Tensor | None = None


@torch.no_grad()
def operator_from_response_head(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    amplitude: float = 1.0,
    subtract_baseline: bool = True,
    include_base: bool = False,
) -> ResponseOperator:
    """Estimate the joint-block response operator of a model's response head.

    Column ``k`` is ``(f(a e_k) - f(0)) / a`` with ``f`` the head's
    ``predictions["response_acceleration"]`` for the input ``response_torque``. The
    other tasks' inputs are dropped from the batch so only the encoder and head R run;
    new optional keys are passed through.

    Args:
        model (torch.nn.Module): Returns an output with ``.predictions`` from
            ``model(**batch)``, as ``DaGetForPhysicsPretraining``.
        batch (Mapping[str, torch.Tensor]): Collator output (encoder inputs required).
        amplitude (float): Normalized torque change ``a`` (fraction of peak torque).
        subtract_baseline (bool): Subtract ``f(0)``, the head's offset; the true
            operator maps zero change to zero.
        include_base (bool): Also return base rows (needs ``response_base_head``
            predictions; floating robots only).

    Returns:
        ResponseOperator: ``joint`` ``[B, J, J]`` and optional ``base`` ``[B, 6, J]``.

    Raises:
        KeyError: The model returned no ``response_acceleration``.
    """
    if amplitude == 0:
        raise ValueError("amplitude must be nonzero")
    was_training = model.training
    model.eval()
    inputs = {key: value for key, value in batch.items() if key not in OTHER_TASK_KEYS}
    mask = inputs["joint_mask"]
    size, joints = mask.shape
    reference = inputs["input_values"]
    inputs.setdefault("response_target", reference.new_zeros((size, joints)))
    inputs.setdefault("response_mask", mask.clone())
    if include_base:
        inputs["response_base_mask"] = torch.ones(size, dtype=torch.bool, device=mask.device)
        inputs.setdefault("response_base_target", reference.new_zeros((size, 6)))
    else:
        inputs["response_base_mask"] = torch.zeros(size, dtype=torch.bool, device=mask.device)
        inputs.setdefault("response_base_target", reference.new_zeros((size, 6)))

    def evaluate(torque: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = model(**{**inputs, "response_torque": torque})
        predictions = output.predictions
        if "response_acceleration" not in predictions:
            raise KeyError("the model returned no response_acceleration prediction")
        return predictions["response_acceleration"], predictions.get("response_base_acceleration") if include_base else None

    zero = reference.new_zeros((size, joints))
    baseline, base_baseline = evaluate(zero) if subtract_baseline else (None, None)
    columns, base_columns = [], []
    for joint in range(joints):
        torque = zero.clone()
        torque[:, joint] = amplitude
        torque = torque * mask.to(torque.dtype)
        response, base_response = evaluate(torque)
        if baseline is not None:
            response = response - baseline
        columns.append(response * mask.to(response.dtype) / amplitude)
        if include_base and base_response is not None:
            base_columns.append((base_response - base_baseline if base_baseline is not None else base_response) / amplitude)
    if was_training:
        model.train()
    joint_operator = torch.stack(columns, dim=-1)
    base_operator = torch.stack(base_columns, dim=-1) if base_columns else None
    return ResponseOperator(joint_operator, base_operator)


def normalized_response_operator(operator: torch.Tensor, record: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize one frame's true operator the way the collator normalizes R targets.

    Args:
        operator (torch.Tensor): ``[6 + J, J]`` ``constrained_response`` of the frame.
        record (EmbodimentRecord): Its embodiment (``actuator``, ``actuated``,
            ``response_scale``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Joint block ``[J, J]`` and base rows
            ``[6, J]``, both float64, with zero columns for passive joints.
    """
    actuated = record.actuated
    peak = torch.where(actuated, record.actuator["peak_torque"], torch.ones_like(record.actuator["peak_torque"])).double()
    joints = operator.shape[-1]
    scale = record.response_scale.double() if record.response_scale is not None else torch.ones(6 + joints, dtype=torch.float64)
    columns = operator.double() * (peak * actuated.double())[None, :]
    return columns[6:] / scale[6:, None], columns[:6] / scale[:6, None]


def true_response_operators(
    batch: Mapping[str, torch.Tensor],
    samples: Sequence[Mapping[str, Any]] | None = None,
    records: Sequence[Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the padded normalized true joint-block operators of a batch.

    The batch's own ``response_operator`` and ``response_scale`` are used when the
    collator provides them, so the normalization is whatever the collator applied to
    the response targets; otherwise each sample's ``response_operator`` and its record's
    ``response_scale``. Peak torques and actuated columns come from the records when
    given, else from ``response_torque_scale`` and the joint mask.

    Args:
        batch (Mapping[str, torch.Tensor]): Collator output.
        samples (Sequence[Mapping] | None): The batch's dataset items.
        records (Sequence[EmbodimentRecord] | None): Records indexed by ``embodiment``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``[B, J, J]`` float64 operators (zero on
            padding and passive columns) and bool ``[B, J]`` actuated columns.

    Raises:
        KeyError: Neither the batch nor the samples hold the operator.
    """
    mask = batch["joint_mask"].cpu()
    size, joints = mask.shape
    have_records = samples is not None and records is not None
    actuated = mask.clone()
    peak = torch.ones((size, joints), dtype=torch.float64)
    scale = torch.ones((size, joints), dtype=torch.float64)
    operator = torch.zeros((size, joints, joints), dtype=torch.float64)
    if have_records:
        actuated = torch.zeros_like(mask)
        for row, sample in enumerate(samples):
            record = records[sample["embodiment"]]
            count = record.graph.num_joints
            actuated[row, :count] = record.actuated
            record_peak = record.actuator["peak_torque"].double()
            peak[row, :count] = torch.where(record.actuated, record_peak, torch.ones_like(record_peak))
    elif "response_torque_scale" in batch:
        peak = batch["response_torque_scale"].cpu().double()
    if "response_operator" in batch and "response_scale" in batch:
        operator = batch["response_operator"].cpu().double()[:, 6 : 6 + joints, :joints].clone()
        scale = batch["response_scale"].cpu().double()
    elif have_records:
        for row, sample in enumerate(samples):
            record = records[sample["embodiment"]]
            count = record.graph.num_joints
            operator[row, :count, :count] = sample["response_operator"].double()[6:]
            if record.response_scale is not None:
                scale[row, :count] = record.response_scale.double()[6:]
    else:
        raise KeyError("no response_operator in the batch, and no samples with records")
    columns = actuated & mask
    normalized = operator * (peak * columns.double())[:, None, :] / scale.clamp_min(1e-12)[:, :, None]
    normalized = normalized * mask.double()[:, :, None]
    return normalized, columns
