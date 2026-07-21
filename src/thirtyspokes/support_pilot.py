"""Offline economics gate for the support-ticket routing pilot.

The input is a scored outcome matrix: each de-identified ticket has the cost and
binary human/operational success label for every candidate model.  No inference
or partner data leaves the machine running this analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .cache import OutcomeCache, PoolModel


_TOKEN = re.compile(r"[a-z0-9_]+")
DEFAULT_LAMBDAS = (
    0.0, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06,
    0.07, 0.1, 0.3, 0.7, 1.5,
)


@dataclass(frozen=True)
class SupportPilotData:
    cache: OutcomeCache
    created_at: tuple[datetime, ...]
    categories: tuple[str, ...]


@dataclass(frozen=True)
class PolicyPoint:
    name: str
    accuracy: float
    mean_cost: float
    accuracy_lcb: float


@dataclass(frozen=True)
class LinearRouter:
    """The fitted prompt-visible baseline, serializable into a KOTH artifact."""

    weights: np.ndarray
    bias: np.ndarray

    def choose(self, embeddings: np.ndarray) -> np.ndarray:
        return (embeddings.astype(np.float64) @ self.weights + self.bias).argmax(axis=1)


@dataclass(frozen=True)
class PilotReport:
    tickets: int
    period_days: float
    train_tickets: int
    eval_tickets: int
    models: tuple[str, ...]
    target_accuracy: float
    target_tolerance: float
    min_advantage: float
    min_tickets: int
    min_days: float
    single_models: tuple[PolicyPoint, ...]
    category_policies: tuple[PolicyPoint, ...]
    learned_policies: tuple[PolicyPoint, ...]
    oracle: PolicyPoint
    selected: PolicyPoint | None
    baseline_name: str | None
    baseline_cost: float | None
    routing_advantage: float | None
    net_savings: float | None
    passes: bool
    reasons: tuple[str, ...]


def _parse_time(value: object, line: int) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line}: created_at must be an ISO-8601 string")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"line {line}: invalid created_at {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"line {line}: created_at must include a timezone")
    return parsed


def _hash_features(text: str, category: str, dims: int) -> np.ndarray:
    if dims < 16:
        raise ValueError("feature_dims must be at least 16")
    tokens = _TOKEN.findall(text.lower())
    terms = [f"word:{token}" for token in tokens]
    terms += [f"bigram:{a}:{b}" for a, b in zip(tokens, tokens[1:])]
    terms.append(f"category:{category.lower()}")

    out = np.zeros(dims, dtype=np.float32)
    for term in terms:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % (dims - 4)
        out[index] += 1.0 if digest[-1] & 0x80 else -1.0

    # Prompt-visible difficulty signals that do not require a model call.
    out[-4] = np.log1p(len(tokens))
    out[-3] = np.log1p(len(text))
    out[-2] = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    out[-1] = 1.0
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else out


def load_scored_tickets(path: str | Path, feature_dims: int = 128) -> SupportPilotData:
    """Load and validate the scored-ticket JSONL contract documented in docs/."""
    records: list[dict] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_no}: each record must be an object")
        record["_line"] = line_no
        records.append(record)
    if not records:
        raise ValueError("ticket file is empty")

    first_outcomes = records[0].get("outcomes")
    if not isinstance(first_outcomes, dict) or len(first_outcomes) < 2:
        raise ValueError("line 1: outcomes must contain at least two models")
    model_names = tuple(sorted(first_outcomes))
    seen_ids: set[str] = set()
    ids: list[str] = []
    times: list[datetime] = []
    categories: list[str] = []
    embeddings: list[np.ndarray] = []
    correct = np.zeros((len(records), len(model_names)), dtype=bool)
    cost = np.zeros((len(records), len(model_names)), dtype=np.float32)

    for q, record in enumerate(records):
        line = int(record["_line"])
        ticket_id = record.get("ticket_id")
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise ValueError(f"line {line}: ticket_id must be a non-empty string")
        if ticket_id in seen_ids:
            raise ValueError(f"line {line}: duplicate ticket_id {ticket_id!r}")
        seen_ids.add(ticket_id)

        text = record.get("routing_text")
        category = record.get("category")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"line {line}: routing_text must be non-empty")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"line {line}: category must be non-empty")

        outcomes = record.get("outcomes")
        if not isinstance(outcomes, dict) or tuple(sorted(outcomes)) != model_names:
            raise ValueError(f"line {line}: outcomes must contain exactly {list(model_names)}")
        for k, model in enumerate(model_names):
            outcome = outcomes[model]
            if not isinstance(outcome, dict):
                raise ValueError(f"line {line}: outcome for {model!r} must be an object")
            success = outcome.get("success")
            if success not in (True, False, 0, 1):
                raise ValueError(f"line {line}: {model!r} success must be binary")
            dollars = outcome.get("cost_usd")
            if isinstance(dollars, bool) or not isinstance(dollars, (int, float)):
                raise ValueError(f"line {line}: {model!r} cost_usd must be numeric")
            if not np.isfinite(dollars) or dollars < 0:
                raise ValueError(f"line {line}: {model!r} cost_usd must be finite and non-negative")
            correct[q, k] = bool(success)
            cost[q, k] = float(dollars)

        ids.append(ticket_id)
        times.append(_parse_time(record.get("created_at"), line))
        categories.append(category.strip())
        embeddings.append(_hash_features(text, category, feature_dims))

    mean_cost = cost.mean(axis=0)
    rank = np.argsort(mean_cost)
    roles: dict[int, str] = {}
    for position, k in enumerate(rank):
        roles[int(k)] = ("cheap" if position < len(rank) / 3 else
                         "strong" if position >= 2 * len(rank) / 3 else "mid")
    models = [PoolModel(name, float(mean_cost[k]), roles[k]) for k, name in enumerate(model_names)]
    cache = OutcomeCache(
        models=models,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        correct=correct,
        cost=cost,
        question_ids=ids,
        cost_norm=float(cost.max()) if float(cost.max()) > 0 else 1.0,
    )
    return SupportPilotData(cache, tuple(times), tuple(categories))


def chronological_split(
    data: SupportPilotData, eval_frac: float = 0.35,
) -> tuple[OutcomeCache, OutcomeCache, tuple[str, ...], tuple[str, ...]]:
    if not 0.1 <= eval_frac <= 0.5:
        raise ValueError("eval_frac must be between 0.1 and 0.5")
    if data.cache.Q < 4:
        raise ValueError("at least four tickets are required")
    order = np.array(sorted(range(data.cache.Q), key=lambda i: data.created_at[i]), dtype=int)
    n_eval = max(1, int(round(data.cache.Q * eval_frac)))
    train_idx, eval_idx = order[:-n_eval], order[-n_eval:]
    train_categories = tuple(data.categories[i] for i in train_idx)
    eval_categories = tuple(data.categories[i] for i in eval_idx)
    return (data.cache.subset(train_idx), data.cache.subset(eval_idx),
            train_categories, eval_categories)


def wilson_lcb(successes: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    radius = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float(max(0.0, (centre - radius) / denom))


def _point(name: str, cache: OutcomeCache, choice: np.ndarray) -> PolicyPoint:
    q = np.arange(cache.Q)
    successes = float(cache.correct[q, choice].sum())
    return PolicyPoint(
        name=name,
        accuracy=successes / cache.Q,
        mean_cost=float(cache.cost[q, choice].mean()),
        accuracy_lcb=wilson_lcb(successes, cache.Q),
    )


def fit_linear_router(
    train: OutcomeCache,
    lam: float,
    *,
    iters: int,
    seed: int,
) -> LinearRouter:
    """Train a small deterministic prompt router with full-batch Adam.

    The production subnet may use any miner-supplied router.  This deliberately
    cheap linear baseline answers the pilot question without turning the gate
    into a long architecture search.
    """
    if iters < 1:
        raise ValueError("train_iters must be positive")
    x = train.embeddings.astype(np.float64)
    objective = train.correct.astype(np.float64) - lam * train.normalized_cost()
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1e-3, size=(train.D, train.K))
    bias = np.zeros(train.K, dtype=np.float64)
    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = np.zeros_like(bias)
    vb = np.zeros_like(bias)
    best_weights, best_bias = weights.copy(), bias.copy()
    best_hard = -np.inf

    for step in range(1, iters + 1):
        logits = x @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        expected = (probs * objective).sum(axis=1, keepdims=True)
        grad_logits = probs * (objective - expected) / train.Q
        grad_w = x.T @ grad_logits - 1e-3 * weights
        grad_b = grad_logits.sum(axis=0)

        mw = 0.9 * mw + 0.1 * grad_w
        vw = 0.999 * vw + 0.001 * grad_w * grad_w
        mb = 0.9 * mb + 0.1 * grad_b
        vb = 0.999 * vb + 0.001 * grad_b * grad_b
        scale1, scale2 = 1.0 - 0.9 ** step, 1.0 - 0.999 ** step
        weights += 0.03 * (mw / scale1) / (np.sqrt(vw / scale2) + 1e-8)
        bias += 0.03 * (mb / scale1) / (np.sqrt(vb / scale2) + 1e-8)

        choice = (x @ weights + bias).argmax(axis=1)
        hard = float(objective[np.arange(train.Q), choice].mean())
        if hard > best_hard:
            best_hard = hard
            best_weights, best_bias = weights.copy(), bias.copy()

    return LinearRouter(best_weights, best_bias)


def _linear_router_choice(
    train: OutcomeCache,
    evaluation: OutcomeCache,
    lam: float,
    *,
    iters: int,
    seed: int,
) -> np.ndarray:
    """Compatibility wrapper used by the pilot and robustness analysis."""
    return fit_linear_router(train, lam, iters=iters, seed=seed).choose(evaluation.embeddings)


def nonadaptive_cost_at_accuracy(points: Iterable[PolicyPoint], target: float) -> float | None:
    """Cheapest fixed-model or blind two-model mixture reaching ``target``."""
    values = list(points)
    candidates = [p.mean_cost for p in values if p.accuracy >= target]
    for left in values:
        for right in values:
            if left.accuracy >= right.accuracy:
                continue
            if left.accuracy <= target <= right.accuracy:
                weight = (target - left.accuracy) / (right.accuracy - left.accuracy)
                candidates.append(left.mean_cost + weight * (right.mean_cost - left.mean_cost))
    return min(candidates) if candidates else None


def _category_policy(
    train: OutcomeCache,
    evaluation: OutcomeCache,
    train_categories: tuple[str, ...],
    eval_categories: tuple[str, ...],
    lam: float,
) -> PolicyPoint:
    objective = train.correct.astype(float) - lam * train.normalized_cost()
    fallback = int(objective.mean(axis=0).argmax())
    picks: dict[str, int] = {}
    for category in sorted(set(train_categories)):
        idx = np.array([i for i, value in enumerate(train_categories) if value == category])
        picks[category] = int(objective[idx].mean(axis=0).argmax())
    choice = np.array([picks.get(category, fallback) for category in eval_categories], dtype=int)
    return _point(f"category(lambda={lam:g})", evaluation, choice)


def run_pilot(
    data: SupportPilotData,
    *,
    target_accuracy: float,
    eval_frac: float = 0.35,
    target_tolerance: float = 0.01,
    min_advantage: float = 2.0,
    min_tickets: int = 2_000,
    min_days: float = 28.0,
    overhead_per_ticket: float = 0.0,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    train_iters: int = 250,
    seed: int = 0,
) -> PilotReport:
    if not 0 <= target_accuracy <= 1:
        raise ValueError("target_accuracy must be between 0 and 1")
    if (target_tolerance < 0 or min_advantage < 1 or min_tickets < 1 or min_days < 0 or
            overhead_per_ticket < 0):
        raise ValueError("gate thresholds and overhead must be non-negative")

    train, evaluation, train_cat, eval_cat = chronological_split(data, eval_frac)
    period_days = ((max(data.created_at) - min(data.created_at)).total_seconds() / 86_400
                   if len(data.created_at) > 1 else 0.0)
    singles = tuple(
        _point(f"single:{model.name}", evaluation, np.full(evaluation.Q, k, dtype=int))
        for k, model in enumerate(evaluation.models)
    )
    category_points = tuple(
        _category_policy(train, evaluation, train_cat, eval_cat, lam) for lam in lambdas
    )
    learned: list[PolicyPoint] = []
    for lam in lambdas:
        choice = _linear_router_choice(train, evaluation, lam, iters=train_iters, seed=seed)
        learned.append(_point(f"learned(lambda={lam:g})", evaluation, choice))
    learned_points = tuple(learned)

    q = np.arange(evaluation.Q)
    oracle_choice = np.empty(evaluation.Q, dtype=int)
    for i in q:
        correct_models = np.flatnonzero(evaluation.correct[i])
        choices = correct_models if len(correct_models) else np.arange(evaluation.K)
        oracle_choice[i] = int(choices[np.argmin(evaluation.cost[i, choices])])
    oracle = _point("oracle:cheapest-correct", evaluation, oracle_choice)

    accuracy_viable = [p for p in learned_points if p.accuracy >= target_accuracy]
    statistically_viable = [
        p for p in accuracy_viable
        if p.accuracy_lcb >= target_accuracy - target_tolerance
    ]
    selected = min(statistically_viable, key=lambda p: p.mean_cost) if statistically_viable else None
    fixed_cost = nonadaptive_cost_at_accuracy(singles, target_accuracy)
    simple = [p for p in category_points if p.accuracy >= target_accuracy]
    simple_best = min(simple, key=lambda p: p.mean_cost) if simple else None
    baselines: list[tuple[str, float]] = []
    if fixed_cost is not None:
        baselines.append(("fixed/blind-mix", fixed_cost))
    if simple_best is not None:
        baselines.append((simple_best.name, simple_best.mean_cost))
    baseline_name, baseline_cost = min(baselines, key=lambda item: item[1]) if baselines else (None, None)

    effective_cost = None if selected is None else selected.mean_cost + overhead_per_ticket
    advantage = (None if effective_cost is None or baseline_cost is None or effective_cost <= 0
                 else baseline_cost / effective_cost)
    savings = (None if effective_cost is None or baseline_cost is None or baseline_cost <= 0
               else 1.0 - effective_cost / baseline_cost)

    reasons: list[str] = []
    if data.cache.Q < min_tickets:
        reasons.append(f"need at least {min_tickets} tickets; received {data.cache.Q}")
    if period_days < min_days:
        reasons.append(f"need at least {min_days:g} days of traffic; received {period_days:.1f}")
    if not accuracy_viable:
        reasons.append("no learned policy reached the target accuracy")
    elif not statistically_viable:
        best_lcb = max(point.accuracy_lcb for point in accuracy_viable)
        reasons.append(
            f"best learned-policy accuracy LCB {best_lcb:.3f} is below "
            f"{target_accuracy - target_tolerance:.3f}"
        )
    if baseline_cost is None:
        reasons.append("no baseline reached the target accuracy")
    if advantage is None or advantage < min_advantage:
        rendered = "unavailable" if advantage is None else f"{advantage:.2f}x"
        reasons.append(f"routing advantage {rendered} is below {min_advantage:.2f}x")

    return PilotReport(
        tickets=data.cache.Q,
        period_days=period_days,
        train_tickets=train.Q,
        eval_tickets=evaluation.Q,
        models=tuple(model.name for model in data.cache.models),
        target_accuracy=target_accuracy,
        target_tolerance=target_tolerance,
        min_advantage=min_advantage,
        min_tickets=min_tickets,
        min_days=min_days,
        single_models=singles,
        category_policies=category_points,
        learned_policies=learned_points,
        oracle=oracle,
        selected=selected,
        baseline_name=baseline_name,
        baseline_cost=baseline_cost,
        routing_advantage=advantage,
        net_savings=savings,
        passes=not reasons,
        reasons=tuple(reasons),
    )


def format_report(report: PilotReport) -> str:
    def row(point: PolicyPoint) -> str:
        return (f"  {point.name:<34} acc={point.accuracy:.3f} "
                f"lcb={point.accuracy_lcb:.3f} cost/ticket=${point.mean_cost:.6f}")

    lines = [
        "Support-routing economics gate",
        (f"tickets={report.tickets} period={report.period_days:.1f}d "
         f"train={report.train_tickets} eval={report.eval_tickets}"),
        f"target={report.target_accuracy:.3f} tolerance={report.target_tolerance:.3f}",
        "",
        "Single-model baselines:",
        *(row(point) for point in report.single_models),
        "",
        "Simple category policies:",
        *(row(point) for point in report.category_policies),
        "",
        "Learned policies:",
        *(row(point) for point in report.learned_policies),
        "",
        "Ceiling:",
        row(report.oracle),
        "",
    ]
    if report.selected is not None:
        lines.append(f"selected={report.selected.name} cost/ticket=${report.selected.mean_cost:.6f}")
    if report.baseline_cost is not None:
        lines.append(f"baseline={report.baseline_name} cost/ticket=${report.baseline_cost:.6f}")
    if report.routing_advantage is not None:
        lines.append(f"routing_advantage={report.routing_advantage:.2f}x")
    if report.net_savings is not None:
        lines.append(f"net_savings={report.net_savings:.1%}")
    lines.append(f"\nVERDICT: {'PASS' if report.passes else 'FAIL'}")
    lines.extend(f"  - {reason}" for reason in report.reasons)
    return "\n".join(lines)
