"""Counterfactual physics test: change one link's mass, compare the model's operator change.

Proposal v4, Sec. 5.3 item 4: change one physics token and compare the model's change
of the response operator with the analytic change of ``G`` from
``model.dynamics.constrained_response`` (Featherstone, *Rigid Body Dynamics
Algorithms*, Springer 2008, on constrained dynamics, as cited there).

For one frame of one robot:

1. The body moved by joint ``j`` gets its mass and its inertia about the center of
   mass scaled by ``s`` (a uniform density change, which keeps the center of mass).
   Mass and inertia are scaled together as in ``model.training.sources.randomize_robot``
   and as MetaMorph's density variation (Gupta et al., ICLR 2022, arXiv:2203.11931).
   The default ``s = 1.2`` is the upper bound of both this repo's training mass
   randomization (``DEFAULT_RANDOMIZATION["mass"] = [0.8, 1.2]``) and MetaMorph's
   density range (Table 2), so the perturbed robot stays in the training distribution.
2. Every model input that depends on that mass is recomputed for both robots: the
   rigid-body features, the token hardware features (the ``child_inertial`` columns
   when present), the mass matrix ``M(q)`` and the dynamics features (``M_ii``,
   Coriolis, gravity, load). Both use the same reconstructed state, so their
   difference is the mass change only. ``reconstruction`` reports how far the
   recomputed original inputs are from the stored ones (a check of the state
   reconstruction).
3. ``Delta G = G(perturbed) - G(original)`` (joint block, contacts as recorded) and
   ``Delta G_hat`` from ``operator_from_response_head`` on the two collated robots.
   Both are expressed in the original robot's normalized units
   (``G[6+i, k] tau_k^peak / s_i^R``, the response targets' units) on actuated columns.
4. Reported: cosine similarity and relative error ``||Delta G_hat - Delta G|| /
   ||Delta G||`` (Frobenius), the magnitude ratio, and the error of ``G_hat`` itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from model.dynamics import STANDARD_GRAVITY, constrained_response
from model.evaluation.physics_metrics import move_batch, operator_from_response_head
from model.physics import dynamics_features, rigid_body_features

# Upper bound of DEFAULT_RANDOMIZATION["mass"] and of MetaMorph's density range.
DEFAULT_MASS_SCALE = 1.2
_INERTIAL_TOKEN_COLUMNS = ("mass", "ixx", "iyy", "izz", "ixy", "ixz", "iyz")


@dataclass
class FrameState:
    """The physical state of one dataset frame.

    Attributes:
        embodiment (int): Record index.
        positions (torch.Tensor): Float64 ``[J]`` q.
        velocities (torch.Tensor): Float64 ``[J]`` qd.
        gravity (torch.Tensor): Float64 ``[3]`` gravity in the base frame.
        base_linear_velocity (torch.Tensor | None): ``[3]`` (floating bases).
        base_angular_velocity (torch.Tensor | None): ``[3]`` (floating bases).
        contacts (torch.Tensor | None): Bool ``[C]`` contact state, None without contacts.
        sample (dict): The dataset item.
    """

    embodiment: int
    positions: torch.Tensor
    velocities: torch.Tensor
    gravity: torch.Tensor
    base_linear_velocity: torch.Tensor | None
    base_angular_velocity: torch.Tensor | None
    contacts: torch.Tensor | None
    sample: dict


def denormalize_positions(graph: Any, normalized: torch.Tensor) -> torch.Tensor:
    """Invert ``EmbodimentGraph.normalize_joint_positions``.

    Continuous joints return the wrapped angle (the same pose).

    Args:
        graph (EmbodimentGraph): Robot.
        normalized (torch.Tensor): ``[..., J]`` normalized positions.

    Returns:
        torch.Tensor: Float64 joint positions.
    """
    normalized = normalized.double()
    lower, upper = graph.lower.double(), graph.upper.double()
    span = upper - lower
    continuous = torch.tensor([kind == "continuous" for kind in graph.joint_types])
    limited = lower + (normalized + 1) * span / 2
    values = torch.where(span > 0, limited, normalized)
    return torch.where(continuous, normalized * math.pi, values)


def frame_state(dataset: Any, item: int) -> FrameState:
    """Reconstruct the state of one dataset sample from its stored episode.

    Raw positions and velocities are used when the episode stores them; otherwise they
    are recovered from the normalized ones (``process_episode``: positions by joint
    limits, velocities over the speed limit or 10). Gravity is
    ``STANDARD_GRAVITY`` times the stored direction; the base twist comes from the base
    features (linear velocity over the characteristic length).

    Args:
        dataset (PhysicsTrajectoryDataset): Dataset with ``index``, ``episodes`` and
            ``records``.
        item (int): Sample index.

    Returns:
        FrameState: The frame's state and the dataset item.
    """
    sample = dataset[item]
    episode_index, step = dataset.index[item][:2]
    embodiment, episode = dataset.episodes[episode_index]
    record = dataset.records[embodiment]
    graph = record.graph
    if "positions" in episode:
        positions = episode["positions"][step].double()
    else:
        positions = denormalize_positions(graph, episode["positions_normalized"][step])
    if "velocities" in episode:
        velocities = episode["velocities"][step].double()
    else:
        max_velocity = record.actuator["max_velocity"].double()
        speed = torch.where(max_velocity > 0, max_velocity, torch.full_like(max_velocity, 10.0))
        velocities = episode["velocities_normalized"][step].double() * speed
    base = episode["base_features"][step].double()
    gravity = STANDARD_GRAVITY * base[:3]
    linear = angular = None
    if graph.floating_base:
        angular = base[3:6]
        linear = base[6:9] * graph.characteristic_length
    flags = episode.get("contact_flags")
    contacts = None
    if flags is not None and flags.shape[-1] > 0 and record.contact_links:
        contacts = flags[step].bool()
    return FrameState(embodiment, positions, velocities, gravity, linear, angular, contacts, sample)


def perturb_body_mass(graph: Any, joint: int, scale: float) -> Any:
    """Return a graph whose body moved by ``joint`` has mass and inertia scaled.

    Args:
        graph (EmbodimentGraph): Robot.
        joint (int): Joint index; its moving body changes.
        scale (float): Positive factor on mass and on inertia about the center of mass.

    Returns:
        EmbodimentGraph: Copy with ``body_mass``, ``body_inertia`` and the inertial
            token columns (if any) changed.

    Raises:
        ValueError: The scale is not positive or the joint is out of range.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not 0 <= joint < graph.num_joints:
        raise ValueError(f"joint {joint} is out of range")
    mass = graph.body_mass.clone()
    inertia = graph.body_inertia.clone()
    mass[joint] = mass[joint] * scale
    inertia[joint] = inertia[joint] * scale
    features = graph.joint_features.clone()
    for name in _INERTIAL_TOKEN_COLUMNS:
        if name in graph.feature_names:
            column = graph.feature_names.index(name)
            features[joint, column] = features[joint, column] * scale
    return replace(graph, body_mass=mass, body_inertia=inertia, joint_features=features)


def perturbed_record(record: Any, graph: Any) -> Any:
    """Return a record of the perturbed graph that keeps the original's data scales.

    Args:
        record (EmbodimentRecord): Original record.
        graph (EmbodimentGraph): Perturbed graph.

    Returns:
        EmbodimentRecord: Copy with the new graph and rigid-body and token features.
    """
    return replace(record, graph=graph, rigid_body_features=rigid_body_features(graph), joint_features=graph.joint_features)


def _frame_inputs(record: Any, state: FrameState) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``M(q)`` and the dynamics features of a record at a state."""
    graph = record.graph
    mass_matrix = graph.mass_matrix(state.positions).float()
    features = dynamics_features(
        graph, record.specs, state.positions, state.velocities, state.gravity,
        state.base_linear_velocity, state.base_angular_velocity,
    )
    return mass_matrix, features


def analytic_operator(record: Any, state: FrameState) -> torch.Tensor:
    """Return ``constrained_response`` of a record at a frame's state and contacts.

    The contact model follows ``process_episode``: the record's contact links and
    surface rule (``surface_contacts``, else surface contacts for at most two feet).

    Args:
        record (EmbodimentRecord): Robot.
        state (FrameState): State and contacts.

    Returns:
        torch.Tensor: Float64 ``[6 + J, J]``.
    """
    links = list(record.contact_links)
    surface = record.surface_contacts if record.surface_contacts is not None else len(links) <= 2
    contacts = state.contacts
    return constrained_response(record.graph, state.positions, links if contacts is not None else (), contacts, surface)


def _relative(difference: torch.Tensor, reference: torch.Tensor) -> float:
    """Return ``||difference|| / ||reference||`` (NaN for a zero reference)."""
    norm = float(reference.norm())
    return float(difference.norm()) / norm if norm > 0 else float("nan")


def counterfactual_mass_response(
    model: torch.nn.Module,
    dataset: Any,
    item: int,
    collator: Any,
    joint: int | None = None,
    scale: float = DEFAULT_MASS_SCALE,
) -> dict[str, Any]:
    """Compare the model's and the analytic response-operator change for one mass change.

    Args:
        model (torch.nn.Module): ``DaGetForPhysicsPretraining``-like model.
        dataset (PhysicsTrajectoryDataset): Source of the frame.
        item (int): Sample index.
        collator (PhysicsDataCollator): Collator of the dataset (its class, history size
            and settings are reused for the two robots).
        joint (int | None): Joint whose body changes; the heaviest actuated body if None.
        scale (float): Mass and inertia factor (default 1.2, see module docstring).

    Returns:
        dict[str, Any]: ``cosine``, ``relative_error``, ``magnitude_ratio``,
            ``operator_relative_error`` (``G_hat`` vs ``G`` before the change),
            ``delta_norm`` (analytic, normalized units), ``reconstruction`` errors, and
            the robot, joint and scale.
    """
    state = frame_state(dataset, item)
    record = dataset.records[state.embodiment]
    graph = record.graph
    actuated = record.actuated
    if joint is None:
        masses = torch.where(actuated, graph.body_mass, torch.full_like(graph.body_mass, -1.0))
        joint = int(torch.argmax(masses))
    changed = perturbed_record(record, perturb_body_mass(graph, joint, scale))
    inputs = {name: _frame_inputs(entry, state) for name, entry in (("original", record), ("perturbed", changed))}
    sample = state.sample
    reconstruction = {
        "mass_matrix": _relative(inputs["original"][0].double() - sample["mass_matrix"].double(), sample["mass_matrix"].double()),
        "dynamics_features": _relative(
            inputs["original"][1].double() - sample["dynamics_features"].double(), sample["dynamics_features"].double()
        ),
    }
    original_operator = analytic_operator(record, state)
    perturbed_operator = analytic_operator(changed, state)
    if "response_operator" in sample:
        stored = sample["response_operator"].double()
        reconstruction["response_operator"] = _relative(original_operator - stored, stored)

    pair = type(collator)([record, changed], collator.history_size, collator.settings)
    rows = []
    for index, name in enumerate(("original", "perturbed")):
        mass_matrix, features = inputs[name]
        rows.append({**sample, "embodiment": index, "mass_matrix": mass_matrix, "dynamics_features": features})
    batch = pair(rows)
    device = next(model.parameters()).device
    predicted = operator_from_response_head(model, move_batch(batch, device)).joint.double().cpu()
    joints = graph.num_joints
    peak = torch.where(actuated, record.actuator["peak_torque"].double(), torch.ones(joints, dtype=torch.float64))
    if "response_scale" in batch:
        scales = batch["response_scale"].double()[:, :joints]
    else:
        scale_vector = record.response_scale.double()[6:] if record.response_scale is not None else torch.ones(joints, dtype=torch.float64)
        scales = scale_vector.expand(2, joints)
    # Model operators in physical units (per row's own scale), then both changes in the
    # original robot's normalized units.
    physical = predicted[:, :joints, :joints] * scales[:, :, None] / peak[None, None, :]
    to_normalized = peak[None, :] / scales[0][:, None]
    columns = actuated.double()[None, :]
    predicted_change = (physical[1] - physical[0]) * to_normalized * columns
    true_original = original_operator[6:] * to_normalized * columns
    true_change = (perturbed_operator[6:] - original_operator[6:]) * to_normalized * columns
    predicted_original = physical[0] * to_normalized * columns
    norm_true, norm_predicted = float(true_change.norm()), float(predicted_change.norm())
    cosine = float((predicted_change * true_change).sum()) / (norm_true * norm_predicted) if norm_true > 0 and norm_predicted > 0 else float("nan")
    return {
        "robot": record.name,
        "item": int(item),
        "joint": graph.joint_names[joint],
        "scale": float(scale),
        "cosine": cosine,
        "relative_error": _relative(predicted_change - true_change, true_change),
        "magnitude_ratio": norm_predicted / norm_true if norm_true > 0 else float("nan"),
        "operator_relative_error": _relative(predicted_original - true_original, true_original),
        "delta_norm": norm_true,
        "delta_relative_to_operator": _relative(true_change, true_original),
        "reconstruction": reconstruction,
    }


def summarize_counterfactuals(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the mean and median of the counterfactual metrics.

    Args:
        results (list[Mapping]): ``counterfactual_mass_response`` outputs.

    Returns:
        dict[str, Any]: Per metric ``mean``, ``median`` and ``count`` of finite values.
    """
    summary: dict[str, Any] = {}
    for key in ("cosine", "relative_error", "magnitude_ratio", "operator_relative_error"):
        values = torch.tensor([float(result[key]) for result in results], dtype=torch.float64)
        values = values[torch.isfinite(values)]
        summary[key] = {
            "mean": float(values.mean()) if values.numel() else float("nan"),
            "median": float(values.median()) if values.numel() else float("nan"),
            "count": int(values.numel()),
        }
    return summary
