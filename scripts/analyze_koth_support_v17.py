#!/usr/bin/env python
"""Leakage-safe postmortem for the fixed v17 support TDX shadow evidence.

The v17 answers are used only for descriptive error analysis. Candidate router
settings are evaluated on out-of-fold outcomes from the original training
matrix. The script never scores a counterfactual route on v17 because the
unchosen provider response was not collected there.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from thirtyspokes.cache import OutcomeCache
from thirtyspokes.gateway import signing
from thirtyspokes.koth.benchmarks import bench_seed
from thirtyspokes.koth.support import (
    SUPPORT_INTENTS,
    SUPPORT_MODELS,
    build_support_artifact,
    load_bitext_support_benchmark,
)
from thirtyspokes.koth.verify import _bootstrap_lcb
from thirtyspokes.public_support import extract_intent, stratified_folds
from thirtyspokes.support_pilot import (
    DEFAULT_LAMBDAS,
    _hash_features,
    fit_linear_router,
    load_scored_tickets,
    wilson_lcb,
)


TASK_ID = re.compile(r"bitext-(\d{5})$")
CURRENT_LAMBDA = 0.04
# Fixed prompt-only perturbations. These are deliberately not derived from v17 errors.
ESCALATION_FRACTIONS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05)
KNOWN_FLAGS = {
    "Z": "typo",
    "Q": "colloquial",
    "W": "offensive",
    "K": "keyword",
    "I": "interrogative",
    "N": "negation",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _two_model_training(matrix_path: Path) -> tuple[OutcomeCache, list[dict]]:
    records = [json.loads(line) for line in matrix_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    data = load_scored_tickets(matrix_path)
    positions = {model.name: k for k, model in enumerate(data.cache.models)}
    keep = [positions[model] for model in SUPPORT_MODELS]
    costs = data.cache.cost[:, keep]
    cache = OutcomeCache(
        models=[data.cache.models[k] for k in keep],
        embeddings=data.cache.embeddings,
        correct=data.cache.correct[:, keep],
        cost=costs,
        question_ids=data.cache.question_ids,
        cost_norm=float(costs.max()) if float(costs.max()) > 0 else 1.0,
    )
    by_id = {record["ticket_id"]: record for record in records}
    return cache, [by_id[ticket_id] for ticket_id in cache.question_ids]


def _group(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result = []
    for value, members in grouped.items():
        correct = sum(row["correct"] for row in members)
        result.append({
            key: value,
            "n": len(members),
            "correct": correct,
            "errors": len(members) - correct,
            "accuracy": correct / len(members),
        })
    return sorted(result, key=lambda item: (-item["errors"], str(item[key])))


def _flag_groups(rows: list[dict]) -> list[dict]:
    result = []
    for flag, name in KNOWN_FLAGS.items():
        members = [row for row in rows if flag in row["flags"]]
        correct = sum(row["correct"] for row in members)
        result.append({
            "flag": flag,
            "name": name,
            "n": len(members),
            "correct": correct,
            "errors": len(members) - correct,
            "accuracy": correct / len(members) if members else None,
        })
    return sorted(result, key=lambda item: (-item["errors"], item["flag"]))


def _intent_model_groups(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["gold"], row["model"])].append(row)
    result = []
    for (intent, model), members in grouped.items():
        correct = sum(row["correct"] for row in members)
        result.append({
            "gold": intent,
            "model": model,
            "n": len(members),
            "correct": correct,
            "errors": len(members) - correct,
            "accuracy": correct / len(members),
        })
    return sorted(result, key=lambda item: (-item["errors"], item["gold"], item["model"]))


def _margin_quartiles(rows: list[dict]) -> dict:
    margins = np.asarray([row["absolute_router_margin"] for row in rows])
    order = np.argsort(margins, kind="stable")
    buckets = np.array_split(order, 4)
    result = []
    for number, indices in enumerate(buckets, 1):
        members = [rows[int(index)] for index in indices]
        correct = sum(row["correct"] for row in members)
        values = [row["absolute_router_margin"] for row in members]
        result.append({
            "quartile": number,
            "n": len(members),
            "margin_min": min(values),
            "margin_max": max(values),
            "correct": correct,
            "errors": len(members) - correct,
            "accuracy": correct / len(members),
        })
    return {
        "warning": "The score gap is a router margin, not a calibrated probability.",
        "quartiles": result,
    }


def _overlap(rows: list[dict], epochs: list[int]) -> dict:
    epoch_tasks = {
        epoch: {row["task_id"] for row in rows if row["epoch"] == epoch}
        for epoch in epochs
    }
    pairwise = []
    for position, left in enumerate(epochs):
        for right in epochs[position + 1:]:
            pairwise.append({
                "left_epoch": left,
                "right_epoch": right,
                "shared_tasks": len(epoch_tasks[left] & epoch_tasks[right]),
            })

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    repeated = [members for members in by_task.values() if len(members) > 1]
    states = Counter()
    prediction_disagreements = 0
    response_disagreements = 0
    model_disagreements = 0
    for members in repeated:
        outcomes = {row["correct"] for row in members}
        if outcomes == {True}:
            states["always_correct"] += 1
        elif outcomes == {False}:
            states["always_wrong"] += 1
        else:
            states["correctness_flip"] += 1
        prediction_disagreements += len({row["prediction"] for row in members}) > 1
        response_disagreements += len({row["response"] for row in members}) > 1
        model_disagreements += len({row["model"] for row in members}) > 1
    frequencies = Counter(len(members) for members in by_task.values())
    return {
        "total_occurrences": len(rows),
        "unique_tasks": len(by_task),
        "duplicate_occurrences": len(rows) - len(by_task),
        "task_frequency": {str(key): value for key, value in sorted(frequencies.items())},
        "pairwise_epoch_overlap": pairwise,
        "repeated_unique_tasks": len(repeated),
        "repeated_task_outcomes": dict(states),
        "prediction_label_disagreements": prediction_disagreements,
        "raw_response_disagreements": response_disagreements,
        "route_model_disagreements": model_disagreements,
        "warning": (
            "Repeated tasks reuse the deterministic chosen route. They measure provider/output "
            "repeatability, not the unchosen model's counterfactual accuracy."
        ),
    }


def _gate_arithmetic(correct: int, n: int, target: float) -> dict:
    values = []
    needed = None
    for candidate in range(correct, n + 1):
        lcb = _bootstrap_lcb(
            [1.0] * candidate + [0.0] * (n - candidate), 0.05, 1_000, seed=0x5A17,
        )
        if candidate <= correct + 3:
            values.append({"correct": candidate, "lcb": lcb})
        if needed is None and lcb >= target:
            needed = candidate
            break
    one_sided_z = 1.6448536269514722
    return {
        "target_lcb": target,
        "predeclared_bootstrap": values,
        "minimum_correct_for_predeclared_gate": needed,
        "additional_correct_needed": needed - correct if needed is not None else None,
        "one_sided_95pct_wilson_lcb": wilson_lcb(correct, n, z=one_sided_z),
        "one_sided_wilson_minimum_correct": next(
            candidate for candidate in range(correct, n + 1)
            if wilson_lcb(candidate, n, z=one_sided_z) >= target
        ),
    }


def _summarize_seed_rows(rows: list[dict], metric: str) -> dict:
    values = [row[metric] for row in rows]
    return {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}


def _paired_training_by_intent(cache: OutcomeCache, records: list[dict]) -> list[dict]:
    labels = np.asarray([record["gold_intent"] for record in records])
    result = []
    for intent in SUPPORT_INTENTS:
        indices = np.flatnonzero(labels == intent)
        models = {}
        for position, model in enumerate(SUPPORT_MODELS):
            models[model] = {
                "accuracy": float(cache.correct[indices, position].mean()),
                "mean_cost_usd": float(cache.cost[indices, position].mean()),
            }
        result.append({
            "gold": intent,
            "n": len(indices),
            "models": models,
            "gpt_minus_gemini_accuracy": (
                models[SUPPORT_MODELS[1]]["accuracy"]
                - models[SUPPORT_MODELS[0]]["accuracy"]
            ),
        })
    return sorted(
        result,
        key=lambda item: min(value["accuracy"] for value in item["models"].values()),
    )


def _training_only_cv(
    cache: OutcomeCache,
    records: list[dict],
    *,
    folds_count: int,
    seeds_count: int,
) -> dict:
    """Evaluate the pre-existing lambda grid and fixed escalation fractions OOF only."""
    labels = [record["gold_intent"] for record in records]
    lambda_rows: dict[float, list[dict]] = defaultdict(list)
    escalation_rows: dict[float, list[dict]] = defaultdict(list)
    q = np.arange(cache.Q)

    for seed in range(seeds_count):
        folds = stratified_folds(labels, n_folds=folds_count, seed=seed)
        choices = {lam: np.empty(cache.Q, dtype=int) for lam in DEFAULT_LAMBDAS}
        current_margins = np.empty(cache.Q, dtype=float)
        for fold in range(folds_count):
            eval_idx = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            train = cache.subset(train_idx)
            evaluation_x = cache.embeddings[eval_idx].astype(np.float64)
            for lam in DEFAULT_LAMBDAS:
                fitted = fit_linear_router(train, lam, iters=250, seed=seed)
                scores = evaluation_x @ fitted.weights + fitted.bias
                choices[lam][eval_idx] = scores.argmax(axis=1)
                if lam == CURRENT_LAMBDA:
                    current_margins[eval_idx] = scores[:, 1] - scores[:, 0]

        for lam, choice in choices.items():
            lambda_rows[lam].append({
                "seed": seed,
                "accuracy": float(cache.correct[q, choice].mean()),
                "mean_model_cost_usd": float(cache.cost[q, choice].mean()),
                "gpt_fraction": float((choice == 1).mean()),
            })

        baseline = (current_margins >= 0).astype(int)
        baseline_correct = int(cache.correct[q, baseline].sum())
        for fraction in ESCALATION_FRACTIONS:
            choice = baseline.copy()
            take = round(fraction * cache.Q)
            if take:
                candidates = np.flatnonzero(choice == 0)
                closest = candidates[np.argsort(current_margins[candidates])[-take:]]
                choice[closest] = 1
            correct = int(cache.correct[q, choice].sum())
            escalation_rows[fraction].append({
                "seed": seed,
                "accuracy": correct / cache.Q,
                "mean_model_cost_usd": float(cache.cost[q, choice].mean()),
                "gpt_fraction": float((choice == 1).mean()),
                "delta_correct_vs_current": correct - baseline_correct,
            })

    lambdas = []
    for lam in DEFAULT_LAMBDAS:
        rows = lambda_rows[lam]
        lambdas.append({
            "lambda": lam,
            "accuracy": _summarize_seed_rows(rows, "accuracy"),
            "mean_model_cost_usd": _summarize_seed_rows(rows, "mean_model_cost_usd"),
            "gpt_fraction": _summarize_seed_rows(rows, "gpt_fraction"),
            "seeds": rows,
        })
    escalations = []
    for fraction in ESCALATION_FRACTIONS:
        rows = escalation_rows[fraction]
        deltas = [row["delta_correct_vs_current"] for row in rows]
        escalations.append({
            "additional_gpt_fraction": fraction,
            "accuracy": _summarize_seed_rows(rows, "accuracy"),
            "mean_model_cost_usd": _summarize_seed_rows(rows, "mean_model_cost_usd"),
            "gpt_fraction": _summarize_seed_rows(rows, "gpt_fraction"),
            "delta_correct_by_seed": deltas,
            "positive_seeds": sum(delta > 0 for delta in deltas),
            "negative_seeds": sum(delta < 0 for delta in deltas),
            "seeds": rows,
        })
    return {
        "scope": "original 2,160-row matrix only; stratified out-of-fold evaluation",
        "folds": folds_count,
        "seeds": seeds_count,
        "paired_model_accuracy_by_gold_intent": _paired_training_by_intent(cache, records),
        "candidate_lambdas": lambdas,
        "fixed_boundary_escalations": escalations,
    }


def _route_projections(
    training: OutcomeCache,
    v17_embeddings: np.ndarray,
    rows: list[dict],
    cv: dict,
    aggregate: dict,
) -> dict:
    """Project route mix/cost without using v17 gold or counterfactual answers."""
    mean_cost = {
        model: float(np.mean([row["cost_usd"] for row in rows if row["model"] == model]))
        for model in SUPPORT_MODELS
    }
    delta_per_switch = mean_cost[SUPPORT_MODELS[1]] - mean_cost[SUPPORT_MODELS[0]]
    boundary = aggregate["baseline_per_million_usd"] / 2.0
    current = fit_linear_router(training, CURRENT_LAMBDA, iters=250, seed=0)
    current_scores = v17_embeddings @ current.weights + current.bias
    current_choice = current_scores.argmax(axis=1)
    observed_choice = np.asarray([SUPPORT_MODELS.index(row["model"]) for row in rows])
    if not np.array_equal(current_choice, observed_choice):
        raise RuntimeError("re-fitted current router does not reproduce every v17 route")

    lambda_projection = []
    for result in cv["candidate_lambdas"]:
        fitted = fit_linear_router(training, result["lambda"], iters=250, seed=0)
        choice = (v17_embeddings @ fitted.weights + fitted.bias).argmax(axis=1)
        switched_to_gpt = int(((current_choice == 0) & (choice == 1)).sum())
        switched_to_gemini = int(((current_choice == 1) & (choice == 0)).sum())
        cost_delta = (switched_to_gpt - switched_to_gemini) * delta_per_switch
        projected = aggregate["total_cost_per_million_usd"] + cost_delta / len(rows) * 1e6
        lambda_projection.append({
            "lambda": result["lambda"],
            "v17_prompt_gpt_calls": int((choice == 1).sum()),
            "switched_to_gpt": switched_to_gpt,
            "switched_to_gemini": switched_to_gemini,
            "projected_all_in_per_million_usd": projected,
            "projected_passes_2x": projected <= boundary,
        })

    margins = current_scores[:, 1] - current_scores[:, 0]
    escalation_projection = []
    for result in cv["fixed_boundary_escalations"]:
        choice = current_choice.copy()
        take = round(result["additional_gpt_fraction"] * len(rows))
        if take:
            candidates = np.flatnonzero(choice == 0)
            closest = candidates[np.argsort(margins[candidates])[-take:]]
            choice[closest] = 1
        switched = int(((current_choice == 0) & (choice == 1)).sum())
        projected = (aggregate["total_cost_per_million_usd"]
                     + switched * delta_per_switch / len(rows) * 1e6)
        escalation_projection.append({
            "additional_gpt_fraction": result["additional_gpt_fraction"],
            "v17_prompt_additional_gpt_calls": switched,
            "projected_all_in_per_million_usd": projected,
            "projected_passes_2x": projected <= boundary,
        })
    return {
        "warning": (
            "Quality is not projected on v17: unchosen answers do not exist. Cost uses observed "
            "mean per-model v17 call cost and is an estimate, not invoice reconciliation."
        ),
        "mean_observed_call_cost_usd": mean_cost,
        "estimated_increment_per_gemini_to_gpt_switch_usd": delta_per_switch,
        "two_x_boundary_per_million_usd": boundary,
        "lambda_candidates": lambda_projection,
        "fixed_boundary_escalations": escalation_projection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, default=Path("data/koth-support-tdx-shadow-v17-final"),
    )
    parser.add_argument("--matrix", type=Path, default=Path("data/bitext-support-scored-v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/koth-support-v17-analysis.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 2 or args.seeds < 1:
        raise ValueError("folds must be at least two and seeds must be positive")

    summary = _read_json(args.evidence / "summary.json")
    training, training_records = _two_model_training(args.matrix)
    build = build_support_artifact(args.matrix)
    benchmark = load_bitext_support_benchmark(exclude_source_indices=build.source_indices)
    dataset = __import__("datasets").load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train",
    )
    artifact_config = json.loads(build.artifact.weights)
    weights = np.asarray(artifact_config["weights"], dtype=float)
    bias = np.asarray(artifact_config["bias"], dtype=float)

    rows = []
    for run in summary["runs"]:
        epoch = run["epoch"]
        epoch_dir = args.evidence / f"epoch-{epoch}"
        proof = _read_json(epoch_dir / "proof.json")
        trace = _read_json(epoch_dir / "trace.json")
        if signing.sha256_hex(trace) != proof["call_log_hash"]:
            raise RuntimeError(f"epoch {epoch} trace hash differs from proof")
        if proof["weights_hash"] != run["weights_hash"]:
            raise RuntimeError(f"epoch {epoch} weights hash differs between proof and summary")
        if proof["source_hash"] != build.artifact.source_hash:
            raise RuntimeError(f"epoch {epoch} source hash differs from rebuilt artifact")
        assigned = benchmark.sample(
            run["calls"], bench_seed(proof["nonce"], epoch, benchmark.name),
        )
        assigned_ids = [task.task_id for task in assigned]
        proof_ids = [result["task_id"] for result in proof["results"]]
        trace_ids = [entry["task_id"] for entry in trace]
        if assigned_ids != proof_ids or proof_ids != trace_ids:
            raise RuntimeError(f"epoch {epoch} assigned/proof/trace task order differs")
        if len(trace) != run["calls"]:
            raise RuntimeError(f"epoch {epoch} trace length differs from summary")

        for result, entry in zip(proof["results"], trace):
            match = TASK_ID.fullmatch(entry["task_id"])
            if not match:
                raise RuntimeError(f"unexpected task ID {entry['task_id']!r}")
            source_index = int(match.group(1))
            if source_index in build.source_indices:
                raise RuntimeError(f"v17 task {entry['task_id']} overlaps artifact training")
            source = dataset[source_index]
            if entry["prompt"] != source["instruction"]:
                raise RuntimeError(f"v17 prompt differs from Bitext row {source_index}")
            if result["answer"] != entry["response"]:
                raise RuntimeError(f"proof answer differs from trace response for {entry['task_id']}")
            if not np.isclose(result["cost_usd"], entry["cost_usd"], rtol=0.0, atol=1e-15):
                raise RuntimeError(f"proof cost differs from trace cost for {entry['task_id']}")
            feature = _hash_features(entry["prompt"], "public_support", training.D)
            scores = feature.astype(float) @ weights + bias
            expected_model = SUPPORT_MODELS[int(scores.argmax())]
            if entry["model"] != expected_model:
                raise RuntimeError(f"trace route differs from artifact for {entry['task_id']}")
            prediction = extract_intent(entry["response"], SUPPORT_INTENTS)
            rows.append({
                "epoch": epoch,
                "task_id": entry["task_id"],
                "source_index": source_index,
                "source_category": source["category"],
                "flags": source["flags"],
                "gold": source["intent"],
                "prediction": prediction or "<unparsed>",
                "correct": prediction == source["intent"],
                "model": entry["model"],
                "response": entry["response"],
                "cost_usd": float(entry["cost_usd"]),
                "gpt_router_margin": float(scores[1] - scores[0]),
                "absolute_router_margin": float(abs(scores[1] - scores[0])),
            })

    correct = sum(row["correct"] for row in rows)
    if len(rows) != summary["aggregate"]["calls"] or correct != summary["aggregate"]["correct"]:
        raise RuntimeError("reconstructed v17 accuracy differs from the formal summary")
    if build.artifact.weights_hash != summary["runs"][0]["weights_hash"]:
        raise RuntimeError("re-built artifact weights differ from the v17 proof")

    confusions = Counter(
        (row["gold"], row["prediction"]) for row in rows if not row["correct"]
    )
    confusion_models: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        if not row["correct"]:
            confusion_models[(row["gold"], row["prediction"])][row["model"]] += 1
    descriptive = {
        "calls": len(rows),
        "correct": correct,
        "errors": len(rows) - correct,
        "accuracy": correct / len(rows),
        "unparsed_responses": sum(row["prediction"] == "<unparsed>" for row in rows),
        "by_epoch": _group(rows, "epoch"),
        "by_selected_model": _group(rows, "model"),
        "by_gold_intent": _group(rows, "gold"),
        "by_gold_intent_and_selected_model": _intent_model_groups(rows),
        "by_source_category": _group(rows, "source_category"),
        "by_known_language_flag": _flag_groups(rows),
        "top_confusions": [
            {
                "gold": gold,
                "prediction": prediction,
                "errors": count,
                "selected_model_counts": dict(confusion_models[(gold, prediction)]),
            }
            for (gold, prediction), count in confusions.most_common(20)
        ],
        "router_margin": _margin_quartiles(rows),
        "selected_model_warning": (
            "These are observational accuracies after routing, not randomized model effects."
        ),
    }
    cv = _training_only_cv(
        training, training_records, folds_count=args.folds, seeds_count=args.seeds,
    )
    v17_embeddings = np.asarray([
        _hash_features(dataset[row["source_index"]]["instruction"], "public_support", training.D)
        for row in rows
    ], dtype=float)
    analysis = {
        "method": {
            "leakage_policy": (
                "v17 gold is descriptive only; candidate quality uses the original training "
                "matrix out of fold; v17 counterfactual quality is not estimated"
            ),
            "evidence": str(args.evidence),
            "matrix": str(args.matrix),
            "artifact_weights_hash": build.artifact.weights_hash,
            "artifact_source_hash": build.artifact.source_hash,
            "training_source_rows": len(build.source_indices),
            "v17_training_overlap": 0,
        },
        "descriptive_v17": descriptive,
        "task_overlap_and_repeatability": _overlap(rows, summary["epochs"]),
        "gate_arithmetic": _gate_arithmetic(correct, len(rows), 0.90),
        "training_only_cross_validation": cv,
        "prompt_only_route_and_cost_projection": _route_projections(
            training, v17_embeddings, rows, cv, summary["aggregate"],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
