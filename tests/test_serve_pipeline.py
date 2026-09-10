"""The full beta pipeline: tiered serving, the learning loop, and the data flywheel.

The loop under test end-to-end: serve -> log (features + free labels) -> train -> gate -> promote
-> serve sharper. Plus the properties that keep it honest: the split is by time, the gate refuses
noise, exploration keeps the labels coming, and the shadow budget can stop the flywheel but never
serving.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from thirtyspokes.koth import harness
from thirtyspokes.koth.harness import EMBED_DIM
from thirtyspokes.serve import escalate as E
from thirtyspokes.serve.app import Stats, create_app
from thirtyspokes.serve.policy import TIER_CHEAP, TIER_STRONG, TieredPolicy

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from test_serve import FakeBackend, ask  # noqa: E402


# A deterministic "encoder": feature 0 encodes whether the ask is hard. The predictor's whole job
# is to read it, which keeps these tests about the plumbing rather than about MiniLM.
def fake_encode(prompts):
    out = np.zeros((len(prompts), EMBED_DIM))
    for i, p in enumerate(prompts):
        out[i, 0] = 1.0 if "hard" in p else -1.0
    return out


@pytest.fixture(autouse=True)
def _offline_encoder(monkeypatch):
    monkeypatch.setattr(harness, "encode", fake_encode)


def hardness_predictor() -> bytes:
    """p(escalate) high iff feature 0 is positive."""
    w = np.zeros(EMBED_DIM)
    w[0] = 8.0
    return E.save_predictor(w, 0.0)


def client(policy, backend, **kw) -> TestClient:
    return TestClient(create_app(policy, backend, stats=Stats(), **kw))


# --- the predictor file format ---------------------------------------------------------------

def test_predictor_roundtrip():
    predict = E.load_predictor(hardness_predictor())
    p = predict(fake_encode(["a hard one", "an easy one"]))
    assert p[0] > 0.99 and p[1] < 0.01


def test_predictor_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected"):
        E.load_predictor(E.save_predictor(np.zeros(7), 0.0))


def test_feature_transport_roundtrip():
    f = fake_encode(["a hard one"])[0]
    assert np.allclose(E.decode_features(E.encode_features(f)), f, atol=1e-3)


# --- the tiered policy -----------------------------------------------------------------------

def test_no_predictor_enters_cheap_with_strong_backstop():
    d = TieredPolicy(None).decide("anything")
    assert d.model == TIER_CHEAP and d.ladder == (TIER_CHEAP, TIER_STRONG)
    assert d.source == "tiered" and d.p_escalate is None


def test_predicted_hard_enters_strong():
    policy = TieredPolicy(hardness_predictor(), explore_rate=0.0)
    d = policy.decide("a hard proof")
    assert d.model == TIER_STRONG and d.ladder == (TIER_STRONG, TIER_CHEAP)
    assert d.source == "tiered-escalate" and d.p_escalate > 0.99
    e = policy.decide("an easy one")
    assert e.model == TIER_CHEAP and e.source == "tiered"


def test_explore_sends_a_predicted_hard_ask_cheap_anyway():
    """The labels the trainer needs only exist where the cheap tier was tried; exploration is what
    keeps a live predictor from starving its successor."""

    class AlwaysExplore:
        def random(self):
            return 0.0

    d = TieredPolicy(hardness_predictor(), explore_rate=0.05, rng=AlwaysExplore()).decide("hard")
    assert d.model == TIER_CHEAP and d.source == "tiered-explore"
    assert d.p_escalate > 0.99, "the prediction is still logged so the label can grade it"


# --- serving the ladder ----------------------------------------------------------------------

def test_cheap_failure_escalates_to_strong():
    backend = FakeBackend(empty_for={TIER_CHEAP})
    c = client(TieredPolicy(None), backend)
    r = ask(c)
    assert backend.calls == [TIER_CHEAP, TIER_STRONG]
    assert r.headers["x-thirtyspokes-model"] == TIER_STRONG
    assert r.json()["x_thirtyspokes"]["fell_back"] is True


def test_whole_ladder_failing_is_a_502():
    c = client(TieredPolicy(None), FakeBackend(raise_for={TIER_CHEAP, TIER_STRONG}))
    assert ask(c).status_code == 502
    assert c.get("/stats").json()["errors"] == 1


def test_wasted_cost_of_an_empty_attempt_is_counted():
    """An empty answer is billed by the provider; /stats must price the failure, not hide it."""
    backend = FakeBackend(empty_for={TIER_CHEAP}, cost=0.002)
    c = client(TieredPolicy(None), backend)
    ask(c)
    assert c.get("/stats").json()["cost_usd"] == pytest.approx(0.004)   # wasted cheap + served strong


def test_healthz_names_the_pipeline_modes():
    assert client(TieredPolicy(None), FakeBackend()).get("/healthz").json()["mode"] == "tiered"
    assert client(TieredPolicy(hardness_predictor()),
                  FakeBackend()).get("/healthz").json()["mode"] == "tiered+predictor"


# --- the learning loop, end to end -----------------------------------------------------------

def synth_log(n: int = 600, flip: float = 0.05, seed: int = 0) -> list[dict]:
    """A request log in exactly the shape serving writes: hard asks fail the cheap tier (mod some
    label noise), features attached, time-ordered."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        hard = bool(rng.random() < 0.3)
        fail = hard != (rng.random() < flip)
        f = fake_encode(["a hard one" if hard else "an easy one"])[0]
        f[1:] = rng.normal(0, 0.1, EMBED_DIM - 1)               # nuisance dimensions
        records.append({"ts": float(i), "chosen_model": TIER_CHEAP,
                        "served_model": TIER_STRONG if fail else TIER_CHEAP,
                        "fell_back": fail, "features_b64": E.encode_features(f)})
    return records


def test_train_gate_promote_serve():
    """The whole loop: a log with real signal trains a predictor that clears the gate and, once
    served, routes hard asks straight to the strong tier."""
    weights, report = E.train_from_records(synth_log(), TIER_CHEAP)
    assert weights is not None and report["auc"] >= 0.9
    ok, why = E.gate(report)
    assert ok, why

    policy = TieredPolicy(weights, explore_rate=0.0)
    assert policy.decide("a hard one").model == TIER_STRONG
    assert policy.decide("an easy one").model == TIER_CHEAP


def test_gate_refuses_noise():
    """Labels with no relation to the features must not promote — serving without a predictor is
    already on the measured frontier, so refusal costs nothing and coin-flip routing costs money."""
    rng = np.random.default_rng(1)
    records = synth_log()
    for r in records:                                        # sever the feature-label relation
        r["fell_back"] = bool(rng.random() < 0.3)
    weights, report = E.train_from_records(records, TIER_CHEAP)
    ok, why = E.gate(report)
    assert not ok and "AUC" in why


def test_trainer_needs_enough_future_asks():
    _w, report = E.train_from_records(synth_log(n=40), TIER_CHEAP)
    assert _w is None and "test rows" in report["error"]


def test_trainer_only_reads_cheap_entry_records():
    """Strong-entry records observe nothing about the cheap tier; counting them would poison the
    labels with a selection artifact."""
    records = synth_log(n=300)
    strong = [dict(r, chosen_model=TIER_STRONG) for r in synth_log(n=300, seed=9)]
    _w, report = E.train_from_records(records + strong, TIER_CHEAP)
    assert report["n_usable"] == 300


def test_serving_writes_what_the_trainer_reads():
    """Round-trip: the log serving produces is directly consumable by the trainer."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/requests.jsonl"
        backend = FakeBackend(empty_for={TIER_CHEAP})        # every ask fails cheap -> label 1
        c = client(TieredPolicy(None), backend, log_path=path, log_features=True)
        ask(c, "a hard one")
        rec = json.loads(open(path, encoding="utf-8").read().strip())

        assert rec["chosen_model"] == TIER_CHEAP and rec["fell_back"] is True
        f, y = E._rows([rec], TIER_CHEAP)
        assert f.shape == (1, EMBED_DIM) and y[0] == 1.0
        assert np.allclose(f[0], fake_encode(["a hard one"])[0], atol=1e-3)


# --- the data flywheel -----------------------------------------------------------------------

def test_shadow_samples_the_rest_of_the_pool():
    with tempfile.TemporaryDirectory() as tmp:
        shadow = f"{tmp}/shadow.jsonl"
        backend = FakeBackend()
        c = client(TieredPolicy(None), backend, shadow_rate=1.0, shadow_budget_usd=10.0,
                   shadow_path=shadow)
        ask(c)
        c.app.state.shadow_pool.shutdown(wait=True)

        rows = [json.loads(x) for x in open(shadow, encoding="utf-8")]
        from thirtyspokes.koth.harness import pool_models
        assert {r["model"] for r in rows} == set(pool_models()) - {TIER_CHEAP}
        assert all(r["well_formed"] for r in rows)
        assert c.get("/stats").json()["shadow_calls"] == len(rows)


def test_shadow_budget_stops_the_flywheel_not_the_service():
    with tempfile.TemporaryDirectory() as tmp:
        shadow = f"{tmp}/shadow.jsonl"
        backend = FakeBackend(cost=1.0)                     # each shadow call costs $1
        c = client(TieredPolicy(None), backend, shadow_rate=1.0, shadow_budget_usd=2.5,
                   shadow_path=shadow)
        for _ in range(3):
            assert ask(c).status_code == 200               # serving never blocks on the budget
        c.app.state.shadow_pool.shutdown(wait=True)

        s = c.get("/stats").json()
        assert s["requests"] == 3
        assert s["shadow_cost_usd"] <= 3.5, "the budget must bound shadow spend"
        assert s["shadow_calls"] < 3 * 6, "sampling must have stopped early"


def test_shadow_off_by_default():
    c = client(TieredPolicy(None), FakeBackend())
    ask(c)
    assert c.app.state.shadow_pool is None
    assert c.get("/stats").json()["shadow_calls"] == 0
