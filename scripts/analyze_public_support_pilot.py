#!/usr/bin/env python
"""Robustness analysis for a collected public support outcome matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thirtyspokes.cache import OutcomeCache
from thirtyspokes.public_support import stratified_folds
from thirtyspokes.support_pilot import (
    DEFAULT_LAMBDAS,
    SupportPilotData,
    _linear_router_choice,
    _point,
    load_scored_tickets,
    nonadaptive_cost_at_accuracy,
)


def _crossvalidated(data, labels: list[str], seed: int, target: float,
                    tolerance: float, min_advantage: float, n_folds: int) -> dict:
    cache = data.cache
    folds = stratified_folds(labels, n_folds=n_folds, seed=seed)
    choices = {lam: np.empty(cache.Q, dtype=int) for lam in DEFAULT_LAMBDAS}
    for fold in range(n_folds):
        eval_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        train, evaluation = cache.subset(train_idx), cache.subset(eval_idx)
        for lam in DEFAULT_LAMBDAS:
            choices[lam][eval_idx] = _linear_router_choice(
                train, evaluation, lam, iters=250, seed=seed
            )

    learned = tuple(_point(f"learned(lambda={lam:g})", cache, choices[lam])
                    for lam in DEFAULT_LAMBDAS)
    singles = tuple(_point(f"single:{model.name}", cache, np.full(cache.Q, k, dtype=int))
                    for k, model in enumerate(cache.models))
    baseline = nonadaptive_cost_at_accuracy(singles, target)
    eligible = [point for point in learned
                if point.accuracy >= target and point.accuracy_lcb >= target - tolerance]
    selected = min(eligible, key=lambda point: point.mean_cost) if eligible else None
    selected_lam = (None if selected is None else
                    next(lam for lam in DEFAULT_LAMBDAS
                         if selected.name == f"learned(lambda={lam:g})"))
    advantage = (baseline / selected.mean_cost
                 if baseline is not None and selected is not None and selected.mean_cost > 0 else None)
    headroom = (baseline / min_advantage - selected.mean_cost
                if baseline is not None and selected is not None else None)
    return {
        "seed": seed,
        "selected": None if selected is None else selected.name,
        "accuracy": None if selected is None else selected.accuracy,
        "accuracy_lcb": None if selected is None else selected.accuracy_lcb,
        "cost_per_ticket": None if selected is None else selected.mean_cost,
        "baseline_cost_per_ticket": baseline,
        "routing_advantage": advantage,
        "max_overhead_for_gate": headroom,
        "route_mix": (None if selected_lam is None else {
            model.name: float((choices[selected_lam] == k).mean())
            for k, model in enumerate(cache.models)
        }),
        "passes": bool(advantage is not None and advantage >= min_advantage),
    }


def _without_models(data: SupportPilotData, drop: set[int]) -> SupportPilotData:
    keep = [k for k in range(data.cache.K) if k not in drop]
    cost = data.cache.cost[:, keep]
    cache = OutcomeCache(
        models=[data.cache.models[k] for k in keep],
        embeddings=data.cache.embeddings,
        correct=data.cache.correct[:, keep],
        cost=cost,
        question_ids=data.cache.question_ids,
        cost_norm=float(cost.max()) if float(cost.max()) > 0 else 1.0,
    )
    return SupportPilotData(cache, data.created_at, data.categories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickets")
    parser.add_argument("--quality-target", type=float, default=0.90)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument("--min-advantage", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.tickets).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    data = load_scored_tickets(args.tickets)
    by_id = {record["ticket_id"]: record for record in records}
    aligned = [by_id[ticket_id] for ticket_id in data.cache.question_ids]
    labels = [record["gold_intent"] for record in aligned]

    print("=== full-matrix model statistics ===")
    for k, model in enumerate(data.cache.models):
        latencies = [record["outcomes"][model.name]["latency_seconds"] for record in aligned]
        unique = data.cache.correct[:, k] & ~np.delete(data.cache.correct, k, axis=1).any(axis=1)
        print(f"{model.name:34s} acc={data.cache.correct[:, k].mean():.4f} "
              f"cost=${data.cache.cost[:, k].sum():.4f} "
              f"latency={np.mean(latencies):.3f}s unique_correct={int(unique.sum())}")

    any_correct = data.cache.correct.any(axis=1)
    print(f"oracle_accuracy={any_correct.mean():.4f} all_models_wrong={int((~any_correct).sum())}")

    print("\n=== accuracy by language-variation tag ===")
    for tag, description in (("Z", "typo"), ("Q", "colloquial"), ("W", "offensive"),
                             ("K", "keyword"), ("I", "interrogative"), ("N", "negation")):
        idx = np.array([tag in record["source_flags"] for record in aligned])
        values = " ".join(
            f"{model.name.split('/')[-1]}={data.cache.correct[idx, k].mean():.3f}"
            for k, model in enumerate(data.cache.models)
        )
        print(f"{description:13s} n={int(idx.sum()):4d} {values}")

    print("\n=== stratified cross-validation ===")
    results = [
        _crossvalidated(data, labels, seed, args.quality_target, args.quality_tolerance,
                        args.min_advantage, args.folds)
        for seed in range(args.seeds)
    ]
    for result in results:
        print(json.dumps(result, sort_keys=True))
    passes = sum(result["passes"] for result in results)
    advantages = [result["routing_advantage"] for result in results
                  if result["routing_advantage"] is not None]
    print(f"cv_passes={passes}/{len(results)} "
          f"advantage_range={min(advantages):.2f}x..{max(advantages):.2f}x")

    accuracy = data.cache.model_accuracy()
    mean_cost = data.cache.model_mean_cost()
    dominated = {
        k for k in range(data.cache.K)
        if any((accuracy[j] >= accuracy[k] and mean_cost[j] <= mean_cost[k] and
                (accuracy[j] > accuracy[k] or mean_cost[j] < mean_cost[k]))
               for j in range(data.cache.K) if j != k)
    }
    if dominated and data.cache.K - len(dominated) >= 2:
        print("\n=== globally dominated-model ablation ===")
        print("dropped=" + ",".join(data.cache.models[k].name for k in sorted(dominated)))
        reduced = _without_models(data, dominated)
        reduced_results = [
            _crossvalidated(reduced, labels, seed, args.quality_target, args.quality_tolerance,
                            args.min_advantage, args.folds)
            for seed in range(args.seeds)
        ]
        for result in reduced_results:
            print(json.dumps(result, sort_keys=True))
        reduced_passes = sum(result["passes"] for result in reduced_results)
        reduced_advantages = [result["routing_advantage"] for result in reduced_results
                              if result["routing_advantage"] is not None]
        print(f"ablation_passes={reduced_passes}/{len(reduced_results)} "
              f"advantage_range={min(reduced_advantages):.2f}x..{max(reduced_advantages):.2f}x")


if __name__ == "__main__":
    main()
