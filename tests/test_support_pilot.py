from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from thirtyspokes.support_pilot import (
    PolicyPoint,
    chronological_split,
    fit_linear_router,
    format_report,
    load_scored_tickets,
    nonadaptive_cost_at_accuracy,
    run_pilot,
)
from thirtyspokes.public_support import extract_intent, stratified_folds, stratified_indices


def _write_matrix(path, n=80):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        hard = i % 2 == 1
        rows.append({
            "ticket_id": f"ticket-{i:04d}",
            "created_at": (start + timedelta(hours=i)).isoformat(),
            "category": "account" if hard else "faq",
            "routing_text": ("Investigate failed account migration" if hard else
                             "Where can I download an invoice"),
            "outcomes": {
                "cheap": {"success": int(not hard), "cost_usd": 0.001},
                "mid": {"success": 1, "cost_usd": 0.004},
                "strong": {"success": 1, "cost_usd": 0.010},
            },
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_load_and_chronological_split(tmp_path):
    path = tmp_path / "tickets.jsonl"
    _write_matrix(path, n=20)
    data = load_scored_tickets(path, feature_dims=32)
    train, evaluation, train_cat, eval_cat = chronological_split(data, eval_frac=0.25)

    assert data.cache.Q == 20 and data.cache.K == 3 and data.cache.D == 32
    assert train.Q == 15 and evaluation.Q == 5
    assert train.question_ids[-1] == "ticket-0014"
    assert evaluation.question_ids[0] == "ticket-0015"
    assert len(train_cat) == 15 and len(eval_cat) == 5


def test_loader_rejects_model_matrix_drift(tmp_path):
    path = tmp_path / "tickets.jsonl"
    _write_matrix(path, n=4)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["outcomes"].pop("mid")
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="outcomes must contain exactly"):
        load_scored_tickets(path)


def test_loader_accepts_two_model_matrix(tmp_path):
    path = tmp_path / "tickets.jsonl"
    _write_matrix(path, n=8)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["outcomes"].pop("strong")
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert load_scored_tickets(path).cache.K == 2


def test_nonadaptive_baseline_uses_cheapest_mixture():
    points = [
        PolicyPoint("cheap", 0.5, 1.0, 0.4),
        PolicyPoint("mid", 0.7, 4.0, 0.6),
        PolicyPoint("strong", 0.9, 10.0, 0.8),
    ]
    assert nonadaptive_cost_at_accuracy(points, 0.8) == pytest.approx(7.0)
    assert nonadaptive_cost_at_accuracy(points, 0.4) == pytest.approx(1.0)
    assert nonadaptive_cost_at_accuracy(points, 0.95) is None


def test_pilot_report_runs_and_enforces_sample_gate(tmp_path):
    path = tmp_path / "tickets.jsonl"
    _write_matrix(path)
    data = load_scored_tickets(path, feature_dims=32)
    report = run_pilot(
        data,
        target_accuracy=0.95,
        min_tickets=100,
        min_days=0,
        min_advantage=1.0,
        lambdas=(0.0, 0.5),
        train_iters=40,
    )

    assert report.oracle.accuracy == 1.0
    assert len(report.single_models) == 3
    assert len(report.learned_policies) == 2
    assert not report.passes
    assert "need at least 100 tickets" in format_report(report)


def test_public_support_stratification_and_grading():
    labels = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    selected = stratified_indices(labels, 12, seed=3)
    assert len(selected) == len(set(selected)) == 12
    assert {label: sum(labels[i] == label for i in selected) for label in set(labels)} == {
        "a": 4, "b": 4, "c": 4,
    }
    allowed = ("cancel_order", "change_order", "track_refund")
    assert extract_intent("cancel_order", allowed) == "cancel_order"
    assert extract_intent("Intent: track_refund.", allowed) == "track_refund"
    assert extract_intent("cancel_order or change_order", allowed) is None
    folds = stratified_folds(labels, n_folds=5, seed=4)
    assert set(folds) == set(range(5))
    for label in set(labels):
        assert set(folds[np.array(labels) == label]) == {0, 1, 2, 3, 4}


def test_fitted_router_object_matches_compatibility_path(tmp_path):
    from thirtyspokes.support_pilot import _linear_router_choice

    path = tmp_path / "tickets.jsonl"
    _write_matrix(path, n=40)
    cache = load_scored_tickets(path, feature_dims=32).cache
    fitted = fit_linear_router(cache, 0.1, iters=20, seed=3)
    assert np.array_equal(
        fitted.choose(cache.embeddings),
        _linear_router_choice(cache, cache, 0.1, iters=20, seed=3),
    )


def test_support_artifact_is_confined_runtime_compatible():
    import json

    from thirtyspokes.koth.runtime import Artifact, load_agent
    from thirtyspokes.koth.support import (
        SUPPORT_INTENTS,
        SUPPORT_MODELS,
        SUPPORT_ROUTER_SOURCE,
        grade_support_intent,
    )

    payload = {
        "version": 1,
        "feature_dims": 16,
        "category": "public_support",
        "models": list(SUPPORT_MODELS),
        "intents": list(SUPPORT_INTENTS),
        "lambda": 0.04,
        "weights": [[0.0, 0.0] for _ in range(16)],
        "bias": [1.0, 0.0],
    }
    artifact = Artifact(SUPPORT_ROUTER_SOURCE, json.dumps(payload).encode())
    calls = []

    def call_model(model, messages, params):
        calls.append((model, messages, params))
        return "track_refund"

    answer = load_agent(artifact.source_text, artifact.weights)("Where is my refund?", call_model)
    assert answer == "track_refund"
    assert calls[0][0] == SUPPORT_MODELS[0]
    assert calls[0][2] == {"max_tokens": 24}
    assert grade_support_intent(answer, "track_refund") == 1.0


def test_support_v2_prompt_is_model_specific_and_v1_bytes_are_frozen():
    from thirtyspokes.koth.runtime import Artifact, load_agent
    from thirtyspokes.koth.store import hash_source
    from thirtyspokes.koth.support import (
        SUPPORT_GEMINI_GUIDANCE_V2,
        SUPPORT_INTENTS,
        SUPPORT_MODELS,
        SUPPORT_ROUTER_SOURCE,
        SUPPORT_ROUTER_SOURCE_V2,
        support_system_prompt,
    )

    assert hash_source(SUPPORT_ROUTER_SOURCE) == (
        "2a8fc8fe59b836a93382ef2efc1893e6337187a141e9a09d841342a38d247ab3"
    )
    assert SUPPORT_GEMINI_GUIDANCE_V2 not in support_system_prompt(SUPPORT_MODELS[0])
    assert SUPPORT_GEMINI_GUIDANCE_V2 in support_system_prompt(
        SUPPORT_MODELS[0], version="v2"
    )
    assert SUPPORT_GEMINI_GUIDANCE_V2 not in support_system_prompt(
        SUPPORT_MODELS[1], version="v2"
    )

    def payload(bias):
        return {
            "version": 2,
            "feature_dims": 16,
            "category": "public_support",
            "models": list(SUPPORT_MODELS),
            "intents": list(SUPPORT_INTENTS),
            "lambda": 0.04,
            "weights": [[0.0, 0.0] for _ in range(16)],
            "bias": bias,
        }

    for bias, expected_model in (([1.0, 0.0], SUPPORT_MODELS[0]),
                                 ([0.0, 1.0], SUPPORT_MODELS[1])):
        calls = []
        artifact = Artifact(SUPPORT_ROUTER_SOURCE_V2, json.dumps(payload(bias)).encode())

        def call_model(model, messages, params):
            calls.append((model, messages, params))
            return "track_refund"

        load_agent(artifact.source_text, artifact.weights)("Where is my refund?", call_model)
        assert calls[0][0] == expected_model
        assert calls[0][1][0]["content"] == support_system_prompt(
            expected_model, version="v2"
        )

    with pytest.raises(ValueError, match="unknown support prompt version"):
        support_system_prompt(SUPPORT_MODELS[0], version="future")
