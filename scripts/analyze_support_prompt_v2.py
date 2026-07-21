#!/usr/bin/env python
"""Compare the predeclared support v2 prompt with v1 and evaluate its router gate.

V2 is a development experiment on the original non-v17 training rows. It is not
a regrade of v17 and cannot promote directly to TDX without a fresh fixed holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from thirtyspokes.cache import OutcomeCache
from thirtyspokes.koth.support import SUPPORT_INTENTS, SUPPORT_MODELS
from thirtyspokes.public_support import stratified_folds
from thirtyspokes.support_pilot import (
    DEFAULT_LAMBDAS,
    fit_linear_router,
    load_scored_tickets,
    nonadaptive_cost_at_accuracy,
    wilson_lcb,
)


QUALITY_TARGET = 0.90
MIN_ONE_SIDED_LCB = 0.905
MIN_HEADROOM_PER_MILLION_USD = 5.0
CV_FOLDS = 5
CV_SEEDS = 5
V17_VM_COST_PER_TICKET_USD = 0.09624004880594202 / 2_160
ONE_SIDED_95_Z = 1.6448536269514722


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _aligned(v1_path: Path, v2_path: Path) -> tuple[list[dict], list[dict]]:
    v1 = {record["ticket_id"]: record for record in _records(v1_path)}
    v2 = {record["ticket_id"]: record for record in _records(v2_path)}
    if set(v1) != set(v2):
        missing = len(set(v1) - set(v2))
        extra = len(set(v2) - set(v1))
        raise ValueError(f"v1/v2 ticket sets differ: missing={missing} extra={extra}")
    ids = sorted(v1)
    left, right = [v1[ticket_id] for ticket_id in ids], [v2[ticket_id] for ticket_id in ids]
    for old, new in zip(left, right):
        if old["source_index"] != new["source_index"] or old["gold_intent"] != new["gold_intent"]:
            raise ValueError(f"v1/v2 source alignment differs for {old['ticket_id']}")
        if new.get("prompt_version") != "v2":
            raise ValueError(f"v2 prompt marker missing for {new['ticket_id']}")
        if tuple(sorted(new["outcomes"])) != tuple(sorted(SUPPORT_MODELS)):
            raise ValueError(f"v2 model pool differs for {new['ticket_id']}")
    return left, right


def _model_comparison(v1: list[dict], v2: list[dict]) -> list[dict]:
    result = []
    for model in SUPPORT_MODELS:
        old = np.asarray([record["outcomes"][model]["success"] for record in v1], dtype=bool)
        new = np.asarray([record["outcomes"][model]["success"] for record in v2], dtype=bool)
        old_cost = np.asarray([record["outcomes"][model]["cost_usd"] for record in v1])
        new_cost = np.asarray([record["outcomes"][model]["cost_usd"] for record in v2])
        result.append({
            "model": model,
            "v1_accuracy": float(old.mean()),
            "v2_accuracy": float(new.mean()),
            "accuracy_delta": float(new.mean() - old.mean()),
            "wrong_to_right": int((~old & new).sum()),
            "right_to_wrong": int((old & ~new).sum()),
            "v1_mean_cost_usd": float(old_cost.mean()),
            "v2_mean_cost_usd": float(new_cost.mean()),
            "mean_cost_delta_usd": float(new_cost.mean() - old_cost.mean()),
            "v2_unparsed": sum(
                record["outcomes"][model]["prediction"] is None for record in v2
            ),
            "v2_mean_tokens_in": float(np.mean([
                record["outcomes"][model]["tokens_in"] for record in v2
            ])),
        })
    return result


def _intent_comparison(v1: list[dict], v2: list[dict]) -> list[dict]:
    result = []
    for intent in SUPPORT_INTENTS:
        indices = [index for index, record in enumerate(v2) if record["gold_intent"] == intent]
        models = {}
        for model in SUPPORT_MODELS:
            old = np.mean([v1[index]["outcomes"][model]["success"] for index in indices])
            new = np.mean([v2[index]["outcomes"][model]["success"] for index in indices])
            models[model] = {"v1_accuracy": float(old), "v2_accuracy": float(new),
                             "accuracy_delta": float(new - old)}
        result.append({"gold": intent, "n": len(indices), "models": models})
    return sorted(
        result,
        key=lambda item: item["models"][SUPPORT_MODELS[0]]["accuracy_delta"],
        reverse=True,
    )


def _two_model_cache(path: Path) -> tuple[OutcomeCache, list[str]]:
    records = _records(path)
    data = load_scored_tickets(path)
    positions = {model.name: k for k, model in enumerate(data.cache.models)}
    keep = [positions[model] for model in SUPPORT_MODELS]
    cost = data.cache.cost[:, keep]
    cache = OutcomeCache(
        models=[data.cache.models[k] for k in keep],
        embeddings=data.cache.embeddings,
        correct=data.cache.correct[:, keep],
        cost=cost,
        question_ids=data.cache.question_ids,
        cost_norm=float(cost.max()),
    )
    by_id = {record["ticket_id"]: record for record in records}
    return cache, [by_id[ticket_id]["gold_intent"] for ticket_id in cache.question_ids]


def _baseline(cache: OutcomeCache) -> dict:
    accuracies = cache.model_accuracy()
    costs = cache.model_mean_cost()
    points = []
    for k, model in enumerate(cache.models):
        # nonadaptive_cost_at_accuracy needs only accuracy and mean_cost attributes.
        from thirtyspokes.support_pilot import PolicyPoint
        points.append(PolicyPoint(model.name, float(accuracies[k]), float(costs[k]), 0.0))
    cost = nonadaptive_cost_at_accuracy(points, QUALITY_TARGET)
    return {
        "target_accuracy": QUALITY_TARGET,
        "models": [
            {"model": model.name, "accuracy": float(accuracies[k]),
             "mean_cost_usd": float(costs[k])}
            for k, model in enumerate(cache.models)
        ],
        "cheapest_fixed_or_blind_mix_cost_usd": cost,
    }


def _cross_validation(cache: OutcomeCache, labels: list[str], baseline_cost: float) -> dict:
    by_lambda: dict[float, list[dict]] = defaultdict(list)
    q = np.arange(cache.Q)
    for seed in range(CV_SEEDS):
        folds = stratified_folds(labels, n_folds=CV_FOLDS, seed=seed)
        for lam in DEFAULT_LAMBDAS:
            choice = np.empty(cache.Q, dtype=int)
            for fold in range(CV_FOLDS):
                evaluation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                fitted = fit_linear_router(cache.subset(training), lam, iters=250, seed=seed)
                choice[evaluation] = fitted.choose(cache.embeddings[evaluation])
            correct = int(cache.correct[q, choice].sum())
            model_cost = float(cache.cost[q, choice].mean())
            all_in = model_cost + V17_VM_COST_PER_TICKET_USD
            headroom = baseline_cost / 2.0 - all_in
            by_lambda[lam].append({
                "seed": seed,
                "correct": correct,
                "accuracy": correct / cache.Q,
                "one_sided_95_lcb": wilson_lcb(correct, cache.Q, z=ONE_SIDED_95_Z),
                "model_cost_per_ticket_usd": model_cost,
                "all_in_per_million_usd": all_in * 1_000_000,
                "headroom_before_2x_per_million_usd": headroom * 1_000_000,
                "routing_advantage": baseline_cost / all_in,
                "gpt_fraction": float((choice == 1).mean()),
            })

    candidates = []
    for lam in DEFAULT_LAMBDAS:
        seeds = by_lambda[lam]
        qualifies = (
            min(row["one_sided_95_lcb"] for row in seeds) >= MIN_ONE_SIDED_LCB
            and min(row["headroom_before_2x_per_million_usd"] for row in seeds)
            >= MIN_HEADROOM_PER_MILLION_USD
        )
        candidates.append({
            "lambda": lam,
            "qualifies": qualifies,
            "mean_accuracy": float(np.mean([row["accuracy"] for row in seeds])),
            "min_one_sided_95_lcb": min(row["one_sided_95_lcb"] for row in seeds),
            "mean_all_in_per_million_usd": float(np.mean([
                row["all_in_per_million_usd"] for row in seeds
            ])),
            "max_all_in_per_million_usd": max(
                row["all_in_per_million_usd"] for row in seeds
            ),
            "min_headroom_before_2x_per_million_usd": min(
                row["headroom_before_2x_per_million_usd"] for row in seeds
            ),
            "mean_gpt_fraction": float(np.mean([row["gpt_fraction"] for row in seeds])),
            "seeds": seeds,
        })
    eligible = [candidate for candidate in candidates if candidate["qualifies"]]
    selected = (min(eligible, key=lambda candidate: candidate["mean_all_in_per_million_usd"])
                if eligible else None)
    return {
        "folds": CV_FOLDS,
        "seeds": CV_SEEDS,
        "quality_target": QUALITY_TARGET,
        "minimum_one_sided_95_lcb": MIN_ONE_SIDED_LCB,
        "minimum_2x_headroom_per_million_usd": MIN_HEADROOM_PER_MILLION_USD,
        "v17_vm_cost_per_million_usd": V17_VM_COST_PER_TICKET_USD * 1_000_000,
        "candidates": candidates,
        "selected": None if selected is None else {
            key: value for key, value in selected.items() if key != "seeds"
        },
        "passes_development_gate": bool(eligible),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=Path("data/bitext-support-scored-v2.jsonl"))
    parser.add_argument("--v2", type=Path, default=Path("data/bitext-support-prompt-v2.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/koth-support-prompt-v2-analysis.json"))
    args = parser.parse_args()

    v1, v2 = _aligned(args.v1, args.v2)
    cache, labels = _two_model_cache(args.v2)
    baseline = _baseline(cache)
    baseline_cost = baseline["cheapest_fixed_or_blind_mix_cost_usd"]
    if baseline_cost is None:
        raise RuntimeError("no v2 fixed/blind baseline reaches the 90% target")
    confusions = Counter(
        (record["gold_intent"], record["outcomes"][model]["prediction"], model)
        for record in v2 for model in SUPPORT_MODELS
        if not record["outcomes"][model]["success"]
    )
    analysis = {
        "status": "post-v17 development only; requires a fresh fixed holdout before TDX",
        "predeclared_gate": {
            "quality": f"all 5 CV seeds one-sided 95% LCB >= {MIN_ONE_SIDED_LCB:.3f}",
            "economics": (
                "all 5 seeds >=2x versus the v2 fixed/blind baseline after measured v17 VM cost, "
                f"with >=${MIN_HEADROOM_PER_MILLION_USD:.2f}/M headroom"
            ),
        },
        "tickets": len(v2),
        "model_comparison": _model_comparison(v1, v2),
        "intent_comparison": _intent_comparison(v1, v2),
        "v2_nonadaptive_baseline": baseline,
        "v2_router_cross_validation": _cross_validation(cache, labels, baseline_cost),
        "top_v2_model_confusions": [
            {"gold": gold, "prediction": prediction, "model": model, "errors": count}
            for (gold, prediction, model), count in confusions.most_common(20)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
