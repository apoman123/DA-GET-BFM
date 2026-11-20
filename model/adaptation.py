"""Test-time adaptation of a pretrained physics model to a new robot (proposal F1).

A new robot comes with a URDF and nothing else. The repository's ``synthetic`` and
``mujoco`` sources simulate it and give exact labels for every physics task (no torque
sensor needed), so the pretraining losses themselves are a self-supervised objective
on the new robot. This module adapts ``DaGetForPhysicsPretraining`` to it with
low-rank updates and keeps it close to the pretrained model with one of two
regularizers. ``scripts/adapt_da_get.py`` runs the whole procedure.

Evidence for the procedure:

* PAD (Hansen, Jangir, Sun, Alenyà, Abbeel, Efros, Pinto and Wang, "Self-Supervised
  Policy Adaptation during Deployment", ICLR 2021, arXiv:2007.04309) shares the feature extractor between the policy and a
  self-supervised head and, at deployment, updates the extractor and that head with
  the self-supervised loss only (one gradient step per step of the environment; the
  policy head stays fixed). Their footnote 2 avoids forward dynamics in feature space
  because a constant feature solves it trivially; our forward-dynamics task predicts
  simulator labels, not features, so that collapse does not apply.
* TTT (Sun, Wang, Liu, Miller, Efros and Hardt, "Test-Time Training with Self-Supervision for Generalization under
  Distribution Shifts", ICML 2020, PMLR 119:9229-9248, arXiv:1909.13231) fine-tunes
  only the shared feature extractor on the self-supervised loss of the test data; its
  online version starts each update from the previous one. They report a negligible
  difference between updating the extractor alone and updating it with the
  self-supervised head.

Building blocks, each from the cited paper (checked against the paper's text):

* ``LoRALinear`` and ``LoRAMultiheadAttention``: LoRA (Hu, Shen, Wallis, Allen-Zhu, Li,
  Wang, Wang and Chen, "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022,
  arXiv:2106.09685, Sec. 4.1):
  ``h = W0 x + (alpha / r) B A x`` with ``W0`` frozen, ``B = 0`` at the start, and
  ``A`` random. The paper says "random Gaussian initialization for A" without a
  standard deviation; the authors' reference code (microsoft/LoRA, ``loralib``, MIT,
  Copyright (c) Microsoft Corporation)
  instead uses ``kaiming_uniform_(A, a=sqrt(5))``, the ``nn.Linear`` default, with the
  comment "this is different than what is described in the paper but should not
  affect performance". That is our default (``init="kaiming_uniform"``);
  ``init="gaussian"`` uses ``std = 1 / r``, the Gaussian option of Hugging Face
  ``peft`` (Mangrulkar, Gugger, Debut, Belkada, Paul, Bossan and Tietz, 2022,
  https://github.com/huggingface/peft, Apache-2.0, ``init_lora_weights="gaussian"``
  in ``src/peft/tuners/lora/layer.py``). With ``B = 0`` both start as an exact no-op. Hu et al. "simply set alpha
  to the first r we try and do not tune it"; ``alpha`` is therefore a required
  argument (``scripts/adapt_da_get.py`` defaults it to the rank). The rank has no
  default: Table 6 of the paper sweeps r in {1, 2, 4, 8, 64} and finds r = 1 already
  competitive for {Wq, Wv}, with the caveat (footnote 6) that "we do not expect a
  small r to work for every task or dataset".
* Which weights: Hu et al. adapt only the attention weights (Wq, Wk, Wv, Wo) and
  freeze the MLP (Sec. 4.2). In their Table 5 (fixed 18M-parameter budget), all four
  with r = 2 scores 73.7 / 91.7 on WikiSQL / MultiNLI and {Wq, Wv} with r = 4 scores
  73.7 / 91.3; their caption calls {Wq, Wv} the best overall. ``DEFAULT_LORA_TARGET``
  adapts all four projections of every attention block of the encoder: graph self
  attention and context cross attention (``nn.MultiheadAttention``) and Physics Cross
  Attention (``query``/``key``/``value``/``output`` ``nn.Linear``). The physics heads
  are not adapted by default.
* ``l2_sp_penalty``: L2-SP (Li, Grandvalet and Davoine, "Explicit Inductive Bias for
  Transfer Learning with Convolutional Networks", ICML 2018, PMLR 80:2825-2834,
  arXiv:1802.01483, Eq. 3): ``alpha/2 ||w_S - w0_S||^2 + beta/2 ||w_Sbar||^2``, where
  S are the parameters inherited from the source network and Sbar the new ones. Their
  Sec. 4.2: "alpha differs across experiments, and beta = 0.01 is consistently picked by
  cross-validation for regularizing the last layer" (their Fig. 1 grid: alpha in
  {0, 1e-3, 1e-2, 1e-1, 1}). So ``beta`` defaults to 0.01 and ``alpha`` is required.
* ``lwf_loss``: Learning without Forgetting (Li and Hoiem, ECCV 2016, LNCS Part IV,
  pp. 614-629, DOI 10.1007/978-3-319-46493-0_37; IEEE TPAMI 40(12):2935-2947, 2018,
  DOI 10.1109/TPAMI.2017.2773081, arXiv:1606.09282). The original network's
  outputs are recorded on the new task's data and the adapted network is trained to
  keep them, with the temperature-softened cross entropy of their Eq. 2-4, "T = 2
  according to a grid search on a held out set", and the loss balance weight
  "lambda_o ... set to 1 for most our experiments". For several old tasks or
  multi-label outputs they "take the sum of the loss for each old task and label".

Adaptations to this repository, and what is not from the papers:

* LoRA on ``nn.MultiheadAttention``: its forward reads ``in_proj_weight`` and
  ``out_proj.weight`` directly and never calls ``out_proj``, so wrapping
  ``out_proj`` with ``LoRALinear`` would silently do nothing (``peft`` documents the
  same pitfall and merges the weights around each forward). ``LoRAMultiheadAttention``
  instead runs the base module with the low-rank-updated weights through
  ``torch.func.functional_call``. Each of q, k, v and o has its own ``A`` and ``B``, as
  Hu et al. treat Wq, Wk, Wv, Wo as separate matrices. No code was copied from
  ``loralib`` or ``peft``.
* L2-SP with LoRA: the pretrained weight of an adapted layer is ``W0 + (alpha/r) B A``,
  so its L2-SP term is ``alpha/2 ||(alpha_lora/r) B A + (W - W0)||^2``; ``A`` and ``B``
  are a reparameterization, not new layers, and get no ``beta`` term. [H] unverified
  hypothesis: that regularizing the effective weight, rather than ``A`` and ``B``
  directly, is the right transfer of L2-SP to LoRA; Li et al. regularize full weights.
* LwF on regression heads: the physics heads are regressions (FK, FD, ID, R, R^-1,
  effect, motor margins), so their outputs are kept with a squared error. Li and
  Hoiem compare L1, L2, cross entropy and their distillation loss as the
  response-preserving loss (their Fig. 7) and find only a small advantage for
  distillation ("the similarity measure is not crucial", their Sec. 3). The two motor feasibility
  flags are independent binary labels; softening a two-class distribution
  ``(p, 1 - p)`` by their Eq. 4 gives exactly ``sigmoid(logit / T)``, so each flag uses
  their loss on that pair. We return the Kullback-Leibler form: it differs from their
  cross entropy by the teacher's entropy, a constant for the student, so the gradients
  are identical and the loss is zero for identical models. Per-joint terms are
  averaged over a robot's valid joints and then over samples, the reduction every
  physics loss uses (``docs/training.md``); the sum over prediction keys follows their
  sum over tasks and labels.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional

# Every attention projection of the encoder (Hu et al. adapt attention weights only).
# Names match both the encoder itself and DaGetForPhysicsPretraining (prefix ``encoder.``),
# but not the GraphActionMixer layers inside the physics heads.
DEFAULT_LORA_TARGET = (
    r"(encoder\.)?layers\.\d+\."
    r"(self_attention\.attention|cross_attention\.attention|physics_attention\.(query|key|value|output))"
)
MHA_PROJECTIONS = ("q", "k", "v", "o")
# beta of L2-SP: "beta = 0.01 is consistently picked by cross-validation" (Li et al., 2018, Sec. 4.2).
L2SP_BETA = 0.01
# T and lambda_o of LwF (Li and Hoiem, TPAMI 2018, Sec. 3).
LWF_TEMPERATURE = 2.0
LWF_WEIGHT = 1.0
LoRAInit = Literal["kaiming_uniform", "gaussian"]


def _init_lora_a(weight: torch.Tensor, rank: int, init: LoRAInit) -> None:
    """Initialize a LoRA ``A`` matrix in place.

    Args:
        weight (torch.Tensor): ``[r, in_features]`` parameter.
        rank (int): LoRA rank r.
        init (LoRAInit): ``"kaiming_uniform"`` (microsoft/LoRA reference code, the
            ``nn.Linear`` default) or ``"gaussian"`` (Hu et al. Sec. 4.1; standard
            deviation ``1 / r`` as in ``peft``).

    Raises:
        ValueError: ``init`` is unknown.
    """
    if init == "kaiming_uniform":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    elif init == "gaussian":
        nn.init.normal_(weight, std=1.0 / rank)
    else:
        raise ValueError(f"unknown LoRA init {init!r}; use 'kaiming_uniform' or 'gaussian'")


def _check_rank(rank: int, alpha: float) -> None:
    """Validate a LoRA rank and alpha."""
    if int(rank) != rank or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be positive and finite")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update (Hu et al., 2022).

    ``y = W0 x + b + (alpha / r) B A x``. ``B`` starts at zero, so the module starts as
    an exact copy of the wrapped layer.

    Attributes:
        base_layer (nn.Linear): Wrapped layer, ``W0`` and bias.
        lora_A (nn.Parameter): ``[r, in_features]``.
        lora_B (nn.Parameter): ``[out_features, r]``, zero at initialization.
        rank (int): r.
        alpha (float): LoRA alpha.
        scaling (float): ``alpha / r``.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, init: LoRAInit = "kaiming_uniform") -> None:
        """Wrap a linear layer.

        Args:
            base_layer (nn.Linear): Pretrained layer; its parameters are not copied.
            rank (int): Positive rank r.
            alpha (float): Positive LoRA alpha; the update is scaled by ``alpha / r``.
            init (LoRAInit): Initialization of ``A``.

        Raises:
            TypeError: ``base_layer`` is not an ``nn.Linear``.
            ValueError: Rank or alpha is invalid.
        """
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear wraps an nn.Linear")
        _check_rank(rank, alpha)
        weight = base_layer.weight
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty((self.rank, base_layer.in_features), dtype=weight.dtype, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, self.rank), dtype=weight.dtype, device=weight.device))
        _init_lora_a(self.lora_A, self.rank, init)

    def delta_weight(self) -> torch.Tensor:
        """Return the update ``(alpha / r) B A``.

        Returns:
            torch.Tensor: ``[out_features, in_features]``.
        """
        return self.scaling * (self.lora_B @ self.lora_A)

    def effective_weights(self) -> dict[str, torch.Tensor]:
        """Return the adapted weight under the base layer's parameter name.

        Returns:
            dict[str, torch.Tensor]: ``{"weight": W0 + (alpha / r) B A}``.
        """
        return {"weight": self.base_layer.weight + self.delta_weight()}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen layer and add the low-rank path.

        Args:
            inputs (torch.Tensor): ``[..., in_features]``.

        Returns:
            torch.Tensor: ``[..., out_features]``.
        """
        update = (inputs @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return self.base_layer(inputs) + update * self.scaling

    def merged(self) -> nn.Linear:
        """Fold the update into the wrapped layer and return it (``W = W0 + (alpha / r) B A``).

        Returns:
            nn.Linear: The wrapped layer with the merged weight.
        """
        with torch.no_grad():
            self.base_layer.weight.add_(self.delta_weight().to(self.base_layer.weight.dtype))
        return self.base_layer


class LoRAMultiheadAttention(nn.Module):
    """LoRA on the query, key, value and output projections of ``nn.MultiheadAttention``.

    ``nn.MultiheadAttention`` reads its projection weights directly instead of calling
    submodules, so the base module is run through ``torch.func.functional_call`` with the
    weights ``W0 + (alpha / r) B A`` substituted. Each projection has its own ``A`` and
    ``B`` (Hu et al. treat Wq, Wk, Wv, Wo as separate matrices).

    Attributes:
        base_layer (nn.MultiheadAttention): Wrapped attention.
        projections (tuple[str, ...]): Adapted projections, a subset of q, k, v, o.
        lora_A (nn.ParameterDict): ``A`` per projection, ``[r, input width]``.
        lora_B (nn.ParameterDict): ``B`` per projection, ``[embed_dim, r]``, zero at initialization.
        rank (int): r.
        alpha (float): LoRA alpha.
        scaling (float): ``alpha / r``.
    """

    def __init__(
        self,
        base_layer: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        projections: Sequence[str] = MHA_PROJECTIONS,
        init: LoRAInit = "kaiming_uniform",
    ) -> None:
        """Wrap an attention module.

        Args:
            base_layer (nn.MultiheadAttention): Pretrained attention.
            rank (int): Positive rank r.
            alpha (float): Positive LoRA alpha.
            projections (Sequence[str]): Which of ``q``, ``k``, ``v``, ``o`` to adapt.
            init (LoRAInit): Initialization of each ``A``.

        Raises:
            TypeError: ``base_layer`` is not an ``nn.MultiheadAttention``.
            ValueError: Rank, alpha or projections are invalid.
        """
        super().__init__()
        if not isinstance(base_layer, nn.MultiheadAttention):
            raise TypeError("LoRAMultiheadAttention wraps an nn.MultiheadAttention")
        _check_rank(rank, alpha)
        projections = tuple(projections)
        if not projections or set(projections) - set(MHA_PROJECTIONS) or len(set(projections)) != len(projections):
            raise ValueError(f"projections must be distinct entries of {MHA_PROJECTIONS}")
        self.base_layer = base_layer
        self.projections = projections
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        reference = base_layer.out_proj.weight
        widths = {"q": base_layer.embed_dim, "k": base_layer.kdim, "v": base_layer.vdim, "o": base_layer.embed_dim}
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in projections:
            self.lora_A[name] = nn.Parameter(torch.empty((self.rank, widths[name]), dtype=reference.dtype, device=reference.device))
            self.lora_B[name] = nn.Parameter(torch.zeros((base_layer.embed_dim, self.rank), dtype=reference.dtype, device=reference.device))
            _init_lora_a(self.lora_A[name], self.rank, init)

    def delta_weights(self) -> dict[str, torch.Tensor]:
        """Return ``(alpha / r) B A`` of every adapted projection.

        Returns:
            dict[str, torch.Tensor]: Updates keyed by ``q``, ``k``, ``v``, ``o``.
        """
        return {name: self.scaling * (self.lora_B[name] @ self.lora_A[name]) for name in self.projections}

    def effective_weights(self) -> dict[str, torch.Tensor]:
        """Return the adapted weights under the base module's parameter names.

        Returns:
            dict[str, torch.Tensor]: ``in_proj_weight`` (or ``q_proj_weight``,
                ``k_proj_weight``, ``v_proj_weight`` when key and value widths differ)
                and ``out_proj.weight``, for the adapted projections only.
        """
        base = self.base_layer
        deltas = self.delta_weights()
        weights: dict[str, torch.Tensor] = {}
        if base._qkv_same_embed_dim:
            if any(name in deltas for name in "qkv"):
                blocks = [deltas.get(name, torch.zeros_like(chunk)) for name, chunk in zip("qkv", base.in_proj_weight.chunk(3))]
                weights["in_proj_weight"] = base.in_proj_weight + torch.cat(blocks, dim=0)
        else:
            for name in "qkv":
                if name in deltas:
                    key = f"{name}_proj_weight"
                    weights[key] = getattr(base, key) + deltas[name]
        if "o" in deltas:
            weights["out_proj.weight"] = base.out_proj.weight + deltas["o"]
        return weights

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the base attention with the adapted weights.

        Args:
            *args (Any): Positional arguments of ``nn.MultiheadAttention.forward``.
            **kwargs (Any): Keyword arguments of ``nn.MultiheadAttention.forward``.

        Returns:
            Any: What ``nn.MultiheadAttention.forward`` returns.
        """
        return torch.func.functional_call(self.base_layer, self.effective_weights(), args, kwargs)

    def merged(self) -> nn.MultiheadAttention:
        """Fold the updates into the wrapped attention and return it.

        Returns:
            nn.MultiheadAttention: The wrapped module with merged weights.
        """
        with torch.no_grad():
            weights = {key: value.detach() for key, value in self.effective_weights().items()}
            for key, value in weights.items():
                self.base_layer.get_parameter(key).copy_(value)
        return self.base_layer


LORA_MODULES = (LoRALinear, LoRAMultiheadAttention)


def _patterns(target: str | Sequence[str]) -> list[re.Pattern]:
    """Compile one or more module-name regular expressions."""
    items = [target] if isinstance(target, str) else list(target)
    if not items:
        raise ValueError("target must name at least one pattern")
    return [re.compile(item) for item in items]


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at a dotted path."""
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    target: str | Sequence[str] = DEFAULT_LORA_TARGET,
    projections: Sequence[str] = MHA_PROJECTIONS,
    init: LoRAInit = "kaiming_uniform",
    freeze_base: bool = True,
) -> list[str]:
    """Wrap matching layers with LoRA, in place.

    ``nn.Linear`` modules become ``LoRALinear`` and ``nn.MultiheadAttention`` modules
    become ``LoRAMultiheadAttention``; a module matches when its dotted name fully
    matches one of the patterns.

    Args:
        module (nn.Module): Model to adapt, for example ``DaGetForPhysicsPretraining``.
        rank (int): LoRA rank r (no default: Hu et al. sweep r in {1, 2, 4, 8, 64}).
        alpha (float): LoRA alpha (Hu et al. set it to the first r they try).
        target (str | Sequence[str]): Regular expressions on module names;
            ``DEFAULT_LORA_TARGET`` is every attention projection of the encoder.
        projections (Sequence[str]): Adapted projections of matched
            ``nn.MultiheadAttention`` modules.
        init (LoRAInit): Initialization of ``A``.
        freeze_base (bool): Set ``requires_grad=False`` on every parameter that is not a
            LoRA parameter, so that ``W0`` stays frozen as in LoRA.

    Returns:
        list[str]: Names of the wrapped modules.

    Raises:
        ValueError: Nothing matched, a pattern names the ``out_proj`` of an
            ``nn.MultiheadAttention`` (which its forward never calls), or a matched
            module is neither linear nor attention.
    """
    _check_rank(rank, alpha)
    patterns = _patterns(target)
    modules = dict(module.named_modules())
    attention_outputs = {f"{name}.out_proj" if name else "out_proj" for name, child in modules.items() if isinstance(child, nn.MultiheadAttention)}
    lora_names = [name for name, child in modules.items() if isinstance(child, LORA_MODULES)]

    def already_adapted(name: str) -> bool:
        return any(not prefix or name == prefix or name.startswith(f"{prefix}.") for prefix in lora_names)

    wrapped: list[str] = []
    for name, child in modules.items():
        if not name or already_adapted(name) or not any(pattern.fullmatch(name) for pattern in patterns):
            continue
        if name in attention_outputs:
            raise ValueError(f"{name}: nn.MultiheadAttention does not call out_proj; target the attention module instead")
        if not isinstance(child, (nn.Linear, nn.MultiheadAttention)):
            raise ValueError(f"{name}: LoRA wraps nn.Linear or nn.MultiheadAttention, not {type(child).__name__}")
        wrapped.append(name)
    if not wrapped:
        raise ValueError(f"no module matched the LoRA target {target!r}")
    for name in wrapped:
        child = module.get_submodule(name)
        replacement = (
            LoRAMultiheadAttention(child, rank, alpha, projections, init)
            if isinstance(child, nn.MultiheadAttention)
            else LoRALinear(child, rank, alpha, init)
        )
        _set_submodule(module, name, replacement)
    if freeze_base:
        trainable = {id(parameter) for parameter in lora_parameters(module)}
        for parameter in module.parameters():
            parameter.requires_grad_(id(parameter) in trainable)
    return wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return the ``A`` and ``B`` parameters of every LoRA wrapper.

    Args:
        module (nn.Module): Model with injected LoRA.

    Returns:
        list[nn.Parameter]: LoRA parameters in module order.
    """
    parameters: list[nn.Parameter] = []
    for child in module.modules():
        if isinstance(child, LoRALinear):
            parameters += [child.lora_A, child.lora_B]
        elif isinstance(child, LoRAMultiheadAttention):
            for name in child.projections:
                parameters += [child.lora_A[name], child.lora_B[name]]
    return parameters


def merge_lora(module: nn.Module) -> nn.Module:
    """Fold every LoRA update into its base layer and remove the wrappers, in place.

    The merged model has the original module structure and parameter names, so it
    saves and loads like the pretrained checkpoint (Hu et al. Sec. 4.1: "explicitly
    compute and store W = W0 + BA").

    Args:
        module (nn.Module): Model with injected LoRA.

    Returns:
        nn.Module: The same object, with plain layers. A root that is itself a
            wrapper is returned as its merged base layer.
    """
    if isinstance(module, LORA_MODULES):
        return module.merged()
    names = [name for name, child in module.named_modules() if isinstance(child, LORA_MODULES)]
    # Merge innermost wrappers first so a path never goes stale.
    for name in sorted(names, key=lambda item: item.count("."), reverse=True):
        _set_submodule(module, name, module.get_submodule(name).merged())
    return module


def canonical_parameter_name(name: str) -> str:
    """Return the parameter name the model had before LoRA was injected.

    Args:
        name (str): Name in a model with LoRA wrappers.

    Returns:
        str: The name without ``.base_layer`` segments.
    """
    return name.replace(".base_layer.", ".").removeprefix("base_layer.")


def l2_sp_penalty(
    model: nn.Module,
    reference_state: Mapping[str, torch.Tensor],
    alpha: float,
    beta: float = L2SP_BETA,
) -> torch.Tensor:
    """Return the L2-SP penalty ``alpha/2 ||w_S - w0_S||^2 + beta/2 ||w_Sbar||^2`` (Li et al., 2018, Eq. 3).

    ``S`` is every trainable parameter that ``reference_state`` names with the same shape;
    ``Sbar`` is every other trainable parameter. For a LoRA layer the regularized weight
    is the adapted weight ``W0 + (alpha_lora / r) B A`` compared with the reference
    ``W0``; ``A`` and ``B`` themselves get no term ([H], see the module docstring).
    Frozen parameters outside LoRA layers add a constant and are skipped.

    Args:
        model (nn.Module): Model being adapted, with or without LoRA.
        reference_state (Mapping[str, torch.Tensor]): State dict of the pretrained model
            before any LoRA injection (its parameter names).
        alpha (float): Weight of the starting-point term; required, since Li et al.
            select it per experiment by cross-validation.
        beta (float): Weight of the L2 term on new parameters; 0.01 as picked by their
            cross-validation.

    Returns:
        torch.Tensor: Scalar penalty, zero when the model equals the reference.

    Raises:
        ValueError: ``alpha`` or ``beta`` is negative, or a LoRA layer's base weight is
            missing from ``reference_state``.
    """
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be nonnegative")
    reference_parameter = next(iter(model.parameters()))
    penalty = reference_parameter.new_zeros(())
    covered: set[str] = set()
    for prefix, child in model.named_modules():
        if not isinstance(child, LORA_MODULES):
            continue
        dotted = f"{prefix}." if prefix else ""
        for key, weight in child.effective_weights().items():
            name = canonical_parameter_name(f"{dotted}base_layer.{key}")
            if name not in reference_state:
                raise ValueError(f"reference_state has no {name!r}; pass the state dict from before LoRA injection")
            penalty = penalty + 0.5 * alpha * (weight - reference_state[name].to(weight)).pow(2).sum()
            covered.add(f"{dotted}base_layer.{key}")
        covered.update(f"{dotted}{name}" for name, _ in child.named_parameters() if name.startswith("lora_"))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name in covered:
            continue
        reference = reference_state.get(canonical_parameter_name(name))
        if reference is not None and reference.shape == parameter.shape:
            penalty = penalty + 0.5 * alpha * (parameter - reference.to(parameter)).pow(2).sum()
        else:
            penalty = penalty + 0.5 * beta * parameter.pow(2).sum()
    return penalty


def _predictions(value: Any) -> Mapping[str, torch.Tensor]:
    """Return the prediction dict of a model output or a dict."""
    predictions = getattr(value, "predictions", value)
    if not isinstance(predictions, Mapping):
        raise TypeError("expected a prediction dict or an output with .predictions")
    return predictions


def _reduce(values: torch.Tensor, joint_mask: torch.Tensor | None) -> torch.Tensor:
    """Average ``[B, J]`` values over valid joints per sample, then over samples; ``[B]`` over samples."""
    if values.ndim == 1 or joint_mask is None:
        return values.reshape(values.shape[0], -1).mean(dim=1).mean()
    mask = joint_mask.to(values.dtype)
    counts = mask.sum(dim=1)
    valid = counts > 0
    if not bool(valid.any()):
        return values.sum() * 0
    per_sample = (values * mask).sum(dim=1) / counts.clamp_min(1)
    return per_sample[valid].mean()


def lwf_loss(
    student: Any,
    teacher: Any,
    joint_mask: torch.Tensor | None = None,
    temperature: float = LWF_TEMPERATURE,
    logit_suffix: str = "_logits",
    sample_marker: str = "_base_",
) -> torch.Tensor:
    """Return the Learning-without-Forgetting loss between adapted and original predictions.

    For every prediction key both share: keys ending in ``logit_suffix`` are independent
    binary labels, compared with Li and Hoiem's temperature-softened loss in its
    Kullback-Leibler form (``sigmoid(logit / T)`` per label); every other key is a
    regression output compared with the squared error summed over its components.
    Per-joint terms (leading shape equal to ``joint_mask``) are averaged over valid
    joints and then samples; per-sample terms (keys containing ``sample_marker``, such as
    ``fd_base_acceleration``, or without a joint dimension) over samples. The keys are
    summed.

    Args:
        student (Any): Prediction dict, or an output with ``.predictions``, of the model
            being adapted.
        teacher (Any): The same for the frozen original model on the same batch;
            detached here.
        joint_mask (torch.Tensor | None): Bool ``[B, J]`` valid joints.
        temperature (float): Softening temperature T (2 in Li and Hoiem).
        logit_suffix (str): Suffix of keys that hold logits.
        sample_marker (str): Substring of keys that hold one value per sample (the
            floating-base heads), which a robot with ``J = 6`` joints could otherwise
            be mistaken for per-joint.

    Returns:
        torch.Tensor: Scalar loss, zero when the predictions are equal.

    Raises:
        ValueError: The predictions share no key, shapes differ, or the temperature is
            not positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_predictions, teacher_predictions = _predictions(student), _predictions(teacher)
    keys = sorted(set(student_predictions) & set(teacher_predictions))
    if not keys:
        raise ValueError("student and teacher predictions share no key")
    total = None
    for key in keys:
        current, recorded = student_predictions[key], teacher_predictions[key].detach()
        if current.shape != recorded.shape:
            raise ValueError(f"prediction {key!r} has shape {tuple(current.shape)} vs {tuple(recorded.shape)}")
        per_joint = (
            joint_mask is not None
            and sample_marker not in key
            and current.ndim >= 2
            and tuple(current.shape[:2]) == tuple(joint_mask.shape)
        )
        start = 2 if per_joint else 1
        if key.endswith(logit_suffix):
            # KL(teacher || student) of the softened Bernoulli pair of each label: Li and
            # Hoiem's cross entropy minus the teacher entropy (a constant for the student).
            target = torch.sigmoid(recorded / temperature)
            log_q, log_not_q = functional.logsigmoid(recorded / temperature), functional.logsigmoid(-recorded / temperature)
            log_p, log_not_p = functional.logsigmoid(current / temperature), functional.logsigmoid(-current / temperature)
            elementwise = target * (log_q - log_p) + (1 - target) * (log_not_q - log_not_p)
        else:
            elementwise = (current - recorded).pow(2)
        values = elementwise.flatten(start).sum(-1) if elementwise.ndim > start else elementwise
        term = _reduce(values, joint_mask if per_joint else None)
        total = term if total is None else total + term
    return total
