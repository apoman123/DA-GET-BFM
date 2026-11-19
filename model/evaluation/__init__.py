"""Evaluation harness of the zero-shot proposal (Phase 0).

* ``statistics``: IQM, stratified and robot-level bootstrap intervals, gaps to L0
  (Agarwal et al., NeurIPS 2021, arXiv:2108.13264).
* ``splits``: the L0–L4 split manifest (``configs/splits/zero_shot.json``).
* ``probes``: layer-wise linear and MLP probes with control tasks, the random-encoder
  baseline, and the head-capacity and head re-initialization tests.
* ``physics_metrics``: dimensionless physics metrics and the operator implied by a
  response head.
* ``counterfactual``: one-link mass changes against the analytic change of ``G``.

``scripts/diagnose_physics_location.py`` and ``scripts/evaluate_zero_shot.py`` run them.
"""

from __future__ import annotations

from model.evaluation.counterfactual import counterfactual_mass_response, perturb_body_mass
from model.evaluation.physics_metrics import (
    ResponseOperator,
    contact_f1,
    exact_label_dynamics_errors,
    fk_error_over_length,
    operator_from_response_head,
    relative_error,
    relative_frobenius_error,
)
from model.evaluation.probes import (
    HeadTrainingSettings,
    MlpSettings,
    capture_layer_embeddings,
    fit_ridge_probe,
    head_capacity_test,
    head_reinit_transfer,
    layerwise_probes,
    random_like,
    run_probe,
)
from model.evaluation.splits import LEVELS, RobotEntry, SplitManifest, make_heldout_split
from model.evaluation.statistics import (
    BootstrapInterval,
    interquartile_mean,
    performance_gap,
    robot_bootstrap_ci,
    stratified_bootstrap_ci,
)

__all__ = [
    "LEVELS",
    "BootstrapInterval",
    "HeadTrainingSettings",
    "MlpSettings",
    "ResponseOperator",
    "RobotEntry",
    "SplitManifest",
    "capture_layer_embeddings",
    "contact_f1",
    "counterfactual_mass_response",
    "exact_label_dynamics_errors",
    "fit_ridge_probe",
    "fk_error_over_length",
    "head_capacity_test",
    "head_reinit_transfer",
    "interquartile_mean",
    "layerwise_probes",
    "make_heldout_split",
    "operator_from_response_head",
    "performance_gap",
    "perturb_body_mass",
    "random_like",
    "relative_error",
    "relative_frobenius_error",
    "robot_bootstrap_ci",
    "run_probe",
    "stratified_bootstrap_ci",
]
