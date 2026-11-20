"""Which feet are on the ground, for data without contact sensors, and how well that is known.

Response labels (tasks R and R^-1) need the contact set, and a wrong set gives a badly
wrong label (ignoring all contacts is off by more than 100%). Simulators and robots with
foot sensors give the set; for recorded motion without them it must be inferred. Two
methods:

* ``"height"`` (``model.training.episodes.detect_contacts``): a foot is down when it is
  within ``height_tolerance`` of the lowest foot along gravity and, with a known base
  velocity, slower than ``speed_tolerance``. It assumes flat ground.
* ``"dynamics"`` (``select_contacts``): a maximum a posteriori choice among all subsets
  of the feet. Each foot has a prior probability of being down from kinematics alone
  (height above the lowest foot and, when known, its speed; ``contact_probabilities``).
  Each subset has a likelihood from dynamics: the base wrench the motion needs
  (floating-base RNEA, ``model.dynamics``) must be produced by pushing-only forces inside
  friction pyramids (the inscribed four-sided pyramid of the Coulomb cone) at the subset's
  feet (flat feet: at their four sole corners), solved as a nonnegative least-squares
  problem with FISTA, and every chosen foot must carry at least ``min_load`` of the
  weight (a foot that carries nothing is, for dynamics, a foot in the air; without this
  every foot would fit a flight phase). Without a base velocity, the feet of a subset must also move
  alike relative to the base, since feet standing still in the world all move with it
  (the cue of legged odometry). The subset with the lowest
  ``(residual / (wrench_tolerance m g))^2 / 2 + penalty sum (1 - load / min_load)_+^2
  + spread / (2 stance_speed^2) - sum log p(foot in or out)``
  wins. A foot on a
  higher step is still chosen when the weight needs it, so it does not assume flat
  ground; a foot that is kinematically down but not needed by dynamics stays in, because
  the prior keeps it.

The confidence of a frame is how much more the second-best subset costs. Response
labels are kept only above ``min_contact_confidence`` (``process_episode``): on a real
Go1 trot the low-confidence frames hold almost all wrong sets.

``evaluate_contacts`` compares an inferred set with a reference (simulator or foot
sensors); ``scripts/evaluate_contact_detection.py`` reports both methods on real and
simulated data. Related work: probabilistic contact estimation fusing kinematics and
dynamics on legged robots (Camurri et al., IEEE RA-L 2017; Bledt et al., ICRA 2018).
The subset search, the prior, the minimum load and the confidence are this repository's.

References:
    * RNEA: Luh, Walker & Paul, J. Dyn. Syst. Meas. Control 102(2):69-76, 1980, DOI
      10.1115/1.3149599; Featherstone, Rigid Body Dynamics Algorithms, Springer 2008,
      DOI 10.1007/978-1-4899-7560-7.
    * Friction pyramid with edges ``n +- mu t`` and the wrench ``(f, p x f)`` of a contact
      force: Lynch & Park, Modern Robotics, Cambridge University Press 2017, Sec. 12.2.1
      (Fig. 12.18c), DOI 10.1017/9781316661239.
    * FISTA: Beck & Teboulle, SIAM J. Imaging Sci. 2(1):183-202, 2009, DOI
      10.1137/080716542. Power iteration: von Mises & Pollaczek-Geiringer, ZAMM
      9(1):58-77, 1929, DOI 10.1002/zamm.19290090105. Tikhonov regularization:
      Tikhonov, Dokl. Akad. Nauk SSSR 151(3):501-504, 1963.
    * Stance feet at rest in the world (legged odometry): Bloesch, Hutter, Hoepflinger,
      Leutenegger, Gehring, Remy & Siegwart, RSS 2012, DOI 10.15607/RSS.2012.VIII.003.
    * Contact estimation: Camurri et al., IEEE RA-L 2(2):1023-1030, 2017, DOI
      10.1109/LRA.2017.2652491; Bledt, Wensing, Ingersoll & Kim, ICRA 2018, pp.
      4399-4406, DOI 10.1109/ICRA.2018.8460904.
    * The Go1 trot figures are from the MIPO data (``Open-IO/mipo-go1-odometry``,
      AGPL-3.0; see ``docs/data.md#references``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import torch

from model.dynamics import BaseMotion, recursive_newton_euler, rotor_wrenches

CONTACT_DETECTION_METHODS = ("height", "dynamics")


def contact_probabilities(
    record,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    gravity: torch.Tensor,
    base,
    height_tolerance: float = 0.03,
    stance_speed: float = 0.2,
    floor: float = 0.02,
) -> torch.Tensor:
    """Return each foot's prior probability of being down from kinematics alone, ``[T, C]``.

    ``p = exp(-((h / height_tolerance)^2 + (v / stance_speed)^2) / 2)``, clamped to
    ``[floor, 1 - floor]``, with ``h`` the height above the lowest foot along gravity and
    ``v`` the foot's speed (left out without a base linear velocity).

    Args:
        record (EmbodimentRecord): Robot with contact links.
        positions (torch.Tensor): ``[T, J]`` q.
        velocities (torch.Tensor): ``[T, J]`` qd.
        gravity (torch.Tensor): ``[T, 3]`` gravity in the base frame.
        base (_BaseTrack): Base signals (``model.training.episodes.reconstruct_base``).
        height_tolerance (float): Height scale in m.
        stance_speed (float): Speed scale in m/s.
        floor (float): Smallest probability of either state.

    Returns:
        torch.Tensor: ``[T, C]`` probabilities.
    """
    from model.training.episodes import contact_point_speeds

    graph = record.graph
    links = list(record.contact_links)
    points = graph.link_poses(positions, links)[..., :3, 3]
    down = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    depth = (points * down[:, None]).sum(-1)
    height = depth.max(dim=-1, keepdim=True).values - depth
    score = (height / height_tolerance).pow(2)
    if base.linear_known:
        speeds = contact_point_speeds(graph, positions, velocities, base.linear_velocity, base.angular_velocity, links)
        score = score + (speeds / stance_speed).pow(2)
    return torch.exp(-0.5 * score).clamp(floor, 1 - floor)


def _sole_points(record, positions: torch.Tensor, surface: bool, foot_size: tuple[float, float]) -> tuple[torch.Tensor, int]:
    """Return the force points of every foot ``[T, C, K, 3]`` in the base frame and ``K``.

    Point feet have one point (the link origin). Flat feet have four sole corners around
    the link origin, along the link's x and y axes.
    """
    poses = record.graph.link_poses(positions, list(record.contact_links))
    origins = poses[..., :3, 3]
    if not surface:
        return origins[..., None, :], 1
    half_length, half_width = foot_size[0] / 2, foot_size[1] / 2
    offsets = torch.tensor(
        [[sx * half_length, sy * half_width, 0.0] for sx in (-1, 1) for sy in (-1, 1)], dtype=positions.dtype
    )
    corners = origins[..., None, :] + torch.einsum("tcij,kj->tcki", poses[..., :3, :3], offsets)
    return corners, 4


def _nonnegative_least_squares(
    gram: torch.Tensor, linear: torch.Tensor, mask: torch.Tensor, iterations: int, damping: float
) -> torch.Tensor:
    """Minimize ``x^T H x / 2 - b^T x`` over ``x >= 0`` with masked entries at zero (FISTA).

    Projected FISTA (Beck & Teboulle 2009, see the module references) with the step
    ``1 / L`` and ``L`` from power iteration; the damping is a Tikhonov term.

    Args:
        gram (torch.Tensor): ``[T, n, n]`` positive semidefinite ``H``.
        linear (torch.Tensor): ``[T, n]`` ``b``.
        mask (torch.Tensor): ``[S, n]`` allowed entries per subset.
        iterations (int): Projected accelerated gradient steps.
        damping (float): Tikhonov term relative to the trace of ``H``.

    Returns:
        torch.Tensor: ``[T, S, n]`` solutions.
    """
    count = gram.shape[-1]
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    gram = gram + (damping * trace / count)[:, None, None] * torch.eye(count, dtype=gram.dtype)
    # Largest eigenvalue by power iteration (an upper bound after the 1.1 margin), for the step.
    vector = torch.ones((gram.shape[0], count), dtype=gram.dtype)
    for _ in range(30):
        vector = (gram @ vector[..., None]).squeeze(-1)
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    largest = (vector * (gram @ vector[..., None]).squeeze(-1)).sum(-1)
    step = (1.0 / (1.1 * largest.clamp_min(1e-12)))[:, None, None]
    allowed = mask.to(gram.dtype)[None]
    x = torch.zeros((gram.shape[0], mask.shape[0], count), dtype=gram.dtype)
    y, t = x, 1.0
    for _ in range(iterations):
        gradient = torch.einsum("tij,tsj->tsi", gram, y) - linear[:, None]
        following = (y - step * gradient).clamp_min(0) * allowed
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = following + ((t - 1) / t_next) * (following - x)
        x, t = following, t_next
    return x


def select_contacts(
    record,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    gravity: torch.Tensor,
    base,
    actuator: Mapping[str, torch.Tensor] | None = None,
    base_external_wrench: torch.Tensor | None = None,
    friction: float = 0.8,
    wrench_tolerance: float = 0.15,
    height_tolerance: float = 0.03,
    stance_speed: float = 0.2,
    foot_size: tuple[float, float] = (0.16, 0.08),
    iterations: int = 150,
    max_feet: int = 8,
    min_load: float = 0.05,
    unloaded_penalty: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the most probable set of feet on the ground at every frame.

    See the module docstring. The required base wrench is the floating-base RNEA's base
    wrench with the given base motion (zero base acceleration if unknown) minus known
    rotor and external wrenches.

    Args:
        record (EmbodimentRecord): Robot with a floating base and contact links.
        positions (torch.Tensor): ``[T, J]`` q.
        velocities (torch.Tensor): ``[T, J]`` qd.
        accelerations (torch.Tensor): ``[T, J]`` qdd.
        gravity (torch.Tensor): ``[T, 3]`` gravity in the base frame.
        base (_BaseTrack): Base signals.
        actuator (Mapping | None): Actuator tensors, for rotor wrenches.
        base_external_wrench (torch.Tensor | None): ``[T, 6]`` known wrench on the base.
        friction (float): Friction coefficient of the pyramids.
        wrench_tolerance (float): Residual scale as a fraction of the weight ``m g``.
        height_tolerance (float): See ``contact_probabilities``.
        stance_speed (float): See ``contact_probabilities``.
        foot_size (tuple[float, float]): Sole length and width in m (flat feet).
        iterations (int): Solver iterations per subset.
        max_feet (int): More contact links than this raise (``2^C`` subsets).
        min_load (float): Normal force, as a fraction of the weight, below which a foot of
            a subset counts as unloaded. A foot that carries nothing is not distinguishable
            from a foot in the air by dynamics, and in flight every foot would otherwise fit.
        unloaded_penalty (float): Cost ``penalty (1 - load / min_load)^2`` of each
            underloaded foot of a subset.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Bool contacts ``[T, C]``, the
            chosen subset's residual over ``m g`` ``[T]``, and the confidence ``[T]``: how
            much higher the second-best subset's cost is (a negative log-probability
            ratio). On a real Go1 trot (MIPO), frames with a confidence above 3 had the
            whole set right 98.9% of the time, against 70% over all frames.

    Raises:
        ValueError: The robot has no floating base, no contact links, or too many.
    """
    graph = record.graph
    links = list(record.contact_links)
    feet = len(links)
    if not graph.floating_base or feet == 0:
        raise ValueError(f"{record.name}: contact selection needs a floating base and contact links")
    if feet > max_feet:
        raise ValueError(f"{record.name}: {feet} contact links give too many subsets (max_feet={max_feet})")
    steps = positions.shape[0]
    dtype = positions.dtype
    wrenches = None if actuator is None else rotor_wrenches(graph, actuator, positions, velocities)
    motion = BaseMotion(base.linear_velocity, base.angular_velocity, base.linear_acceleration, base.angular_acceleration)
    required = recursive_newton_euler(
        graph, positions, velocities, accelerations, gravity, motion, wrenches, base_external_wrench
    ).base_wrench  # [T, 6]
    weight = graph.total_mass * gravity.norm(dim=-1).clamp_min(1e-9)  # [T]
    length = float(graph.characteristic_length)
    surface = record.surface_contacts if record.surface_contacts is not None else feet <= 2
    points, per_foot = _sole_points(record, positions, surface, foot_size)  # [T, C, K, 3]
    normal = -gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # [T, 3]
    helper = torch.where(
        normal[:, :1].abs() < 0.9, torch.tensor([1.0, 0.0, 0.0], dtype=dtype), torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    )
    first = torch.linalg.cross(normal, helper.expand_as(normal), dim=-1)
    first = first / first.norm(dim=-1, keepdim=True)
    second = torch.linalg.cross(normal, first, dim=-1)
    edges = torch.stack([normal + friction * sign * tangent for tangent in (first, second) for sign in (1.0, -1.0)], dim=1)  # [T, 4, 3]
    # Wrench of a unit push along every edge at every point; moments over the robot's size.
    force = edges[:, None, None].expand(steps, feet, per_foot, 4, 3)
    moment = torch.linalg.cross(points[:, :, :, None].expand_as(force), force, dim=-1) / length
    generators = torch.cat((force, moment), dim=-1).reshape(steps, -1, 6).transpose(1, 2)  # [T, 6, n]
    target = torch.cat((required[:, :3], required[:, 3:] / length), dim=-1) / weight[:, None]
    subsets = torch.tensor(list(itertools.product((False, True), repeat=feet)), dtype=torch.bool)  # [S, C]
    columns = subsets[:, :, None].expand(-1, feet, per_foot * 4).reshape(subsets.shape[0], -1)  # [S, n]
    normalized = generators  # unit pushes; the solution is then in units of the weight m g
    gram = normalized.transpose(1, 2) @ normalized
    linear = (normalized.transpose(1, 2) @ target[..., None]).squeeze(-1)
    solution = _nonnegative_least_squares(gram, linear, columns, iterations, 1e-6)
    produced = torch.einsum("tij,tsj->tsi", normalized, solution)
    residual = (produced - target[:, None]).norm(dim=-1)  # [T, S], over m g
    # Each unit push has a unit normal component, so a foot's load is the sum of its pushes.
    load = solution.reshape(steps, subsets.shape[0], feet, per_foot * 4).sum(-1)  # [T, S, C], over m g
    underload = ((min_load - load) / min_load).clamp_min(0).pow(2) * subsets[None].to(dtype)
    prior = contact_probabilities(record, positions, velocities, gravity, base, height_tolerance, stance_speed)
    chosen = subsets[None].to(dtype)
    log_prior = (chosen * prior.log()[:, None] + (1 - chosen) * (1 - prior).log()[:, None]).sum(-1)
    cost = 0.5 * (residual / wrench_tolerance).pow(2) - log_prior + unloaded_penalty * underload.sum(-1)
    if not base.linear_known:
        # Feet that stand still in the world all move with the base: their velocities relative
        # to it must agree (legged odometry). Without a base velocity this replaces the speed
        # evidence; the spread of a subset's velocities about their mean is penalized.
        from model.training.episodes import contact_point_velocities

        relative = contact_point_velocities(
            graph, positions, velocities, torch.zeros_like(base.angular_velocity), base.angular_velocity, links
        )  # [T, C, 3]
        size = chosen.sum(-1).clamp_min(1)  # [1, S]
        total = torch.einsum("sc,tcd->tsd", subsets.to(dtype), relative)
        squares = torch.einsum("sc,tc->ts", subsets.to(dtype), relative.pow(2).sum(-1))
        spread = squares - total.pow(2).sum(-1) / size
        cost = cost + 0.5 * spread / stance_speed**2
    best = cost.argmin(dim=-1)
    ordered = cost.topk(min(2, cost.shape[-1]), dim=-1, largest=False).values
    confidence = ordered[:, 1] - ordered[:, 0] if cost.shape[-1] > 1 else torch.full((steps,), math.inf, dtype=dtype)
    return subsets[best], residual.gather(1, best[:, None]).squeeze(1), confidence


def evaluate_contacts(predicted: torch.Tensor, reference: torch.Tensor, usable: torch.Tensor | None = None) -> dict[str, float]:
    """Compare inferred contacts with a reference.

    Args:
        predicted (torch.Tensor): Bool ``[T, C]``.
        reference (torch.Tensor): Bool ``[T, C]``.
        usable (torch.Tensor | None): Bool ``[T]`` frames that would get response labels.

    Returns:
        dict[str, float]: ``foot_accuracy``, ``precision``, ``recall``, ``f1``,
            ``set_accuracy`` (whole set right), and, with ``usable``, ``usable_fraction``
            and ``usable_set_accuracy`` (whole set right among usable frames).
    """
    predicted, reference = predicted.bool(), reference.bool()
    hits = (predicted & reference).sum().item()
    precision = hits / max(predicted.sum().item(), 1)
    recall = hits / max(reference.sum().item(), 1)
    exact = (predicted == reference).all(dim=-1)
    result = {
        "foot_accuracy": (predicted == reference).double().mean().item(),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "set_accuracy": exact.double().mean().item(),
    }
    if usable is not None:
        result["usable_fraction"] = usable.double().mean().item()
        result["usable_set_accuracy"] = exact[usable].double().mean().item() if bool(usable.any()) else float("nan")
    return result
