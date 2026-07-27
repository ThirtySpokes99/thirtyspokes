"""Phase-0 invariant tests. Small Q for speed; each asserts a load-bearing property."""

from __future__ import annotations

import numpy as np
import pytest

from thirtyspokes import oracle
from thirtyspokes.cache import OutcomeCache
from thirtyspokes.cascade import to_cascade_cache
from thirtyspokes.reign import Reign, Submission
from thirtyspokes.router import PARAM_CAP, RouterHead
from thirtyspokes.simulate import Artifact, admit_artifact
from thirtyspokes.scorer import constant_choice, per_batch_winrate, score_choice
from thirtyspokes.sepcmaes import SepCMAES
from thirtyspokes.synth import SynthConfig, generate
from thirtyspokes.train import train_router
from thirtyspokes.validate import (
    calibrate_threshold,
    fingerprint,
    is_duplicate,
    sanity_gate,
    tv_distance,
)


@pytest.fixture(scope="module")
def cache():
    return generate(SynthConfig(q=1500, seed=0))


# --- cache ------------------------------------------------------------------
def test_cache_shapes_and_norm(cache):
    assert cache.correct.shape == (cache.Q, cache.K)
    assert cache.cost.shape == (cache.Q, cache.K)
    assert cache.embeddings.shape[0] == cache.Q
    nc = cache.normalized_cost()
    assert nc.min() >= 0.0 and nc.max() <= 1.0 + 1e-6


def test_cache_save_load_roundtrip(cache, tmp_path):
    p = str(tmp_path / "c.npz")
    cache.save(p)
    c2 = OutcomeCache.load(p)
    assert np.array_equal(cache.correct, c2.correct)
    assert np.allclose(cache.cost, c2.cost)
    assert [m.name for m in c2.models] == [m.name for m in cache.models]
    assert c2.verifier_ok is not None


def test_cache_split_disjoint(cache):
    tr, ev = cache.split(eval_frac=0.3, seed=1)
    assert tr.Q + ev.Q == cache.Q
    assert set(tr.question_ids).isdisjoint(ev.question_ids)


# --- oracle -----------------------------------------------------------------
def test_per_sample_oracle_is_upper_bound(cache):
    rep = oracle.compute(cache, lam=0.3)
    # Guaranteed properties only (review M6): the per-sample oracle dominates any
    # choice vector, including the routable one; and for lam < 1 it always picks
    # a correct model when one exists, so its accuracy >= any single model's.
    # NOTE: routable >= best_single is NOT guaranteed (cross-fit clusters can
    # mislead on redundant pools) and is deliberately not asserted here.
    assert rep.ps_oracle_obj.score >= rep.routable_obj.score - 1e-9
    assert rep.ps_oracle_acc >= rep.best_single_acc - 1e-9


def test_oracle_gate_distinguishes_pools():
    # complementary pool has real accuracy headroom; a redundant strength-ladder
    # pool has near-zero *accuracy* headroom (routable), so the gate separates them.
    complementary = oracle.compute(generate(SynthConfig(q=4000, specialization=0.95, seed=2)), lam=0.3)
    redundant = oracle.compute(generate(SynthConfig(q=4000, specialization=0.0, seed=2)), lam=0.3)
    assert complementary.routable_accuracy_gap_points > redundant.routable_accuracy_gap_points + 2.0
    assert complementary.passes
    assert not redundant.passes


# --- scorer -----------------------------------------------------------------
def test_score_choice_matches_manual(cache):
    ch = constant_choice(cache, 0)
    sb = score_choice(cache, ch, lam=0.0)
    assert sb.accuracy == pytest.approx(cache.model_accuracy()[0], abs=1e-6)


def test_per_batch_winrate_sums_to_one(cache):
    choices = {f"m{k}": constant_choice(cache, k) for k in range(cache.K)}
    wr = per_batch_winrate(cache, choices, lam=0.3, n_batches=50, seed=0)
    assert sum(wr.values()) == pytest.approx(1.0, abs=1e-6)


# --- router head ------------------------------------------------------------
def test_router_distribution_normalized(cache):
    head = RouterHead(cache.D, cache.K, hidden=8)
    dist = head.distribution(head.init_theta(0), cache.embeddings[:64])
    assert np.allclose(dist.sum(axis=1), 1.0, atol=1e-6)
    assert head.n_params < PARAM_CAP


def test_router_param_cap_enforced(cache):
    with pytest.raises(ValueError):
        RouterHead(cache.D, cache.K, hidden=1_000_000)


# --- sep-CMA-ES -------------------------------------------------------------
def test_sepcmaes_minimizes_sphere():
    es = SepCMAES(np.full(6, 3.0), sigma0=1.0, seed=0)
    for _ in range(80):
        pop = es.ask()
        es.tell(pop, (pop ** 2).sum(axis=1))
    assert float((es.mean ** 2).sum()) < 1e-3


# --- training ---------------------------------------------------------------
def test_training_beats_best_single(cache):
    tr_c, ev_c = cache.split(eval_frac=0.4, seed=0)
    rep = oracle.compute(ev_c, lam=0.3)
    tr = train_router(tr_c, lam=0.3, iters=120, seed=0)
    assert tr.evaluate(ev_c).score > rep.best_single_obj.score
    assert tr.evaluate(ev_c).score <= rep.ps_oracle_obj.score + 1e-6


# --- cascade ----------------------------------------------------------------
def test_cascade_cache_wellformed(cache):
    cc = to_cascade_cache(cache)
    assert cc.Q == cache.Q and cc.K == cache.K
    assert cc.correct.dtype == bool
    assert cc.verifier_ok is None  # verifier consumed


def test_cascade_shares_lambda_normalizer(cache):
    """Review M2: the derived cache must inherit the base normalizer, otherwise
    every cross-action-space comparison flatters the cascade."""
    cc = to_cascade_cache(cache)
    assert cc.cost_max == pytest.approx(cache.cost_max)
    # a cascade trajectory may legitimately normalize above 1.0
    assert cc.normalized_cost().max() >= 1.0


def test_cascade_top_rung_equals_most_expensive_single_shot(cache):
    """Review M7: the one property that IS guaranteed — starting at the top rung
    means one call to the most expensive model, banked unconditionally, so its
    correctness and cost equal that model's single-shot column exactly.
    (The old test asserted 'escalation only adds options', which is false: low
    rungs can bank verifier-approved wrong answers.)"""
    cc = to_cascade_cache(cache)
    prices = [m.price_per_token for m in cache.models]
    top = int(np.argmax(prices))
    assert np.array_equal(cc.correct[:, cache.K - 1], cache.correct[:, top])
    assert np.allclose(cc.cost[:, cache.K - 1], cache.cost[:, top])


# --- reign ------------------------------------------------------------------
def test_reign_split_and_burn():
    r = Reign()
    # only two candidates -> slots 3,4,5 burn
    res = r.update([Submission("a", "hka", 1, 0.9), Submission("b", "hkb", 2, 0.8)])
    assert res.slots[0] == "a" and res.slots[1] == "b"
    assert res.burn == pytest.approx(0.15 + 0.12 + 0.08)
    assert res.weights["uid0"] == pytest.approx(res.burn)
    assert res.weights["a"] == pytest.approx(0.40)


def test_reign_earliest_commit_breaks_ties():
    r = Reign()
    # identical scores; earlier commit_block should take slot 1
    res = r.update([Submission("late", "h1", 50, 0.5), Submission("early", "h2", 10, 0.5)])
    assert res.slots[0] == "early"


def test_reign_eps_protects_fresh_incumbent():
    r = Reign(eps0=0.05, eps_floor=0.001, tau=8.0)
    r.update([Submission("champ", "h", 1, 0.70)])          # champ crowned, age 0
    # a challenger only 0.02 better should NOT dethrone a fresh incumbent (eps~0.05)
    res = r.update([Submission("champ", "h", 1, 0.70), Submission("chal", "h2", 2, 0.72)])
    assert res.slots[0] == "champ"


def test_reign_deregistered_hotkey_burns():
    r = Reign()
    res = r.update([Submission("a", "hka", 1, 0.9)], deregistered={"hka"})
    assert "a" not in res.weights
    assert res.weights["uid0"] == pytest.approx(1.0)


# --- validate ---------------------------------------------------------------
def test_fingerprint_detects_copy(cache):
    head = RouterHead(cache.D, cache.K, hidden=8)
    probe = cache.embeddings[:128]
    t = head.init_theta(0)
    t_copy = t + np.random.default_rng(0).normal(0, 1e-4, size=t.shape)
    t_other = head.init_theta(999)
    fp, fp_copy, fp_other = (fingerprint(head, x, probe) for x in (t, t_copy, t_other))
    assert tv_distance(fp, fp_copy) < tv_distance(fp, fp_other)
    cal = calibrate_threshold([fp, fp_other], margin=0.5)
    assert is_duplicate(fp, fp_copy, cal["recommended_threshold"])
    assert not is_duplicate(fp, fp_other, cal["recommended_threshold"])


def test_sanity_gate_rejects_constant_policy(cache):
    head = RouterHead(cache.D, cache.K, hidden=8)
    w1, b1, w2, b2 = head.unpack(head.init_theta(0, scale=0.0))
    b2[0] = 50.0
    theta = np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel()])
    assert not sanity_gate(head, theta, cache.embeddings[:128]).ok


def test_sanity_gate_accepts_real_router(cache):
    tr_c, _ = cache.split(0.5, seed=0)
    tr = train_router(tr_c, lam=0.3, iters=60, seed=0)
    assert sanity_gate(tr.head, tr.theta, cache.embeddings[:128]).ok


# --- review regression tests --------------------------------------------------
def test_cost_norm_shared_across_split_and_saveload(cache, tmp_path):
    """Review L2: children/batches share the parent's lambda normalizer."""
    tr, ev = cache.split(0.4, seed=3)
    assert tr.cost_max == pytest.approx(cache.cost_max)
    assert ev.cost_max == pytest.approx(cache.cost_max)
    sub = cache.subset(np.arange(10))
    assert sub.cost_max == pytest.approx(cache.cost_max)
    p = str(tmp_path / "c.npz")
    ev.save(p)
    assert OutcomeCache.load(p).cost_max == pytest.approx(cache.cost_max)


def test_reign_no_inter_incumbent_inversion():
    """Review M3: a fresher member with a strictly lower raw score must not
    dethrone an older champion — members rank by raw score alone."""
    r = Reign(eps0=0.02, eps_floor=0.002, tau=8.0)
    r.update([Submission("A", "hA", 1, 0.700)])                       # A crowned
    r.update([Submission("A", "hA", 1, 0.700), Submission("B", "hB", 2, 0.695)])
    res = r.update([Submission("A", "hA", 1, 0.700), Submission("B", "hB", 2, 0.695)])
    assert res.slots[0] == "A" and not res.coronation


def test_reign_age_persists_across_reign_exit():
    """Review M4: artifact age keeps counting while the artifact stays a
    candidate, so protection decayed over a long tenure is not refreshed by
    falling out of / re-entering the reign."""
    r = Reign(eps0=0.02, eps_floor=0.002, tau=8.0)
    for _ in range(15):
        r.update([Submission("A", "hA", 1, 0.700)])
    aged_eps = r.eps(r._artifact_age(Submission("A", "hA", 1, 0.700)))
    assert aged_eps < 0.006  # decayed well below eps0
    # a challenger beating A by just over the decayed eps now succeeds
    res = r.update([Submission("A", "hA", 1, 0.700), Submission("C", "hC", 9, 0.707)])
    assert res.slots[0] == "C"


def test_reign_recommit_enters_as_challenger():
    """Architecture 6f: a re-commit is a new artifact — it forfeits the slot's
    protection (and seniority) instead of inheriting it."""
    r = Reign(eps0=0.02, eps_floor=0.002, tau=8.0)
    r.update([Submission("A", "hA", 100, 0.700)])
    # B at +0.011 cannot dethrone a protected fresh incumbent...
    res = r.update([Submission("A", "hA", 100, 0.700), Submission("B", "hB", 200, 0.711)])
    assert res.slots[0] == "A"
    # ...but if A re-commits (new block), A's new artifact is a challenger too,
    # and raw scores decide: B wins.
    res = r.update([Submission("A", "hA", 999, 0.700), Submission("B", "hB", 200, 0.711)])
    assert res.slots[0] == "B"


def test_reign_rejects_duplicate_miner_ids():
    """Review M5: one miner must not stack multiple slots."""
    r = Reign()
    with pytest.raises(ValueError):
        r.update([Submission("A", "h1", 1, 0.9), Submission("A", "h2", 2, 0.8)])


def _mk_live_artifact(cache, mid, block, theta_seed, noise=0.0):
    head = RouterHead(cache.D, cache.K, hidden=8)
    theta = head.init_theta(theta_seed)
    if noise:
        theta = theta + np.random.default_rng(1).normal(0, noise, theta.shape)
    return Artifact(mid, f"hk_{mid}", block, head, theta)


def test_admit_self_resubmission_not_flagged_as_copy(cache):
    """Review H1: refining and resubmitting your own router is never a 'copy'."""
    probe = cache.embeddings[:128]
    live, dq, notes = {}, {}, []
    a1 = _mk_live_artifact(cache, "M", 100, theta_seed=5)
    admit_artifact(a1, live, probe, 0.05, dq, notes)
    a2 = _mk_live_artifact(cache, "M", 200, theta_seed=5, noise=1e-4)  # tiny refinement
    admit_artifact(a2, live, probe, 0.05, dq, notes)
    assert not live["M"].disqualified
    assert live["M"].art.commit_block == 200      # refinement replaced the old artifact
    assert "M" not in dq


def test_admit_rejected_resubmission_keeps_prior_artifact(cache):
    """Review H1: a resubmission that duplicates SOMEONE ELSE is rejected, but
    the miner's previously-admitted artifact keeps earning."""
    probe = cache.embeddings[:128]
    live, dq, notes = {}, {}, []
    other = _mk_live_artifact(cache, "O", 50, theta_seed=11)
    admit_artifact(other, live, probe, 0.05, dq, notes)
    mine = _mk_live_artifact(cache, "M", 100, theta_seed=22)          # independent
    admit_artifact(mine, live, probe, 0.05, dq, notes)
    steal = _mk_live_artifact(cache, "M", 200, theta_seed=11, noise=1e-4)  # copies O
    admit_artifact(steal, live, probe, 0.05, dq, notes)
    assert not live["M"].disqualified
    assert live["M"].art.commit_block == 100      # prior artifact retained
    assert "M" not in dq
    assert any("resubmission by M rejected" in n for n in notes)


def test_thinker_boosts_worker_probability():
    """The composition effect must be real: accumulated Thinker context raises a
    Worker's success probability (this is why orchestration can beat best-single)."""
    from thirtyspokes.synth_tinyrouter import SyntheticBackend, TinyConfig
    be = SyntheticBackend(TinyConfig(q=500, seed=0))
    m = np.full(be.Q, be.K - 1)  # strongest model as worker
    p0 = be.worker_prob(m, np.zeros(be.Q)).mean()
    p_boost = be.worker_prob(m, np.full(be.Q, 1.5)).mean()
    assert p_boost > p0 + 0.02


def test_orchestrator_episode_invariants():
    """Vectorized multi-turn loop produces sane outputs; Verifier can't fire with
    no answer (so a pure-verifier policy never 'stops early' correct)."""
    from thirtyspokes.orchestrator import make_policy_head, run_episodes
    from thirtyspokes.synth_tinyrouter import SyntheticBackend, TinyConfig
    be = SyntheticBackend(TinyConfig(q=400, seed=1))
    head = make_policy_head(be.embeddings.shape[1], be.K)
    r = run_episodes(head, head.init_theta(0), be, max_turns=5)
    assert r.correct.dtype == bool and r.correct.shape == (be.Q,)
    assert np.all(r.cost >= 0)
    assert np.all(r.turns_used >= 1) and np.all(r.turns_used <= 5)


def test_admit_dedup_is_order_symmetric(cache):
    """Review M1: the LATER commit loses regardless of admission order — a
    copier claiming an earlier block cannot dodge comparison by arriving first."""
    probe = cache.embeddings[:128]
    live, dq, notes = {}, {}, []
    victim = _mk_live_artifact(cache, "victim", 500, theta_seed=7)
    admit_artifact(victim, live, probe, 0.05, dq, notes)
    front = _mk_live_artifact(cache, "front", 400, theta_seed=7, noise=1e-4)  # earlier block
    admit_artifact(front, live, probe, 0.05, dq, notes)
    # earlier block wins the artifact; the later one is flagged as the copy
    assert live["victim"].disqualified and dq["victim"] == "copy_of:front"
    assert not live["front"].disqualified


# --- KingChain: single king + equal-share chain of ex-kings --------------------------------------
def _sub(mid, score, block=1):
    from thirtyspokes.reign import Submission
    return Submission(mid, mid, block, score)


def test_kingchain_pays_king_and_ex_kings_equally():
    from thirtyspokes.reign import KingChain
    kc = KingChain(chain_size=3)
    r = kc.update([_sub("a", 0.90), _sub("b", 0.50)])
    assert r.weights == {"a": 1.0} and r.coronation          # vacant crown -> best takes it, sole payee

    r = kc.update([_sub("a", 0.90), _sub("b", 0.99)])        # b clears eps -> crowned; a -> chain
    assert r.slots[:2] == ["b", "a"]
    assert r.weights == {"b": 0.5, "a": 0.5}, "king + ex-king must split equally"

    r = kc.update([_sub("a", 0.90), _sub("b", 0.99), _sub("c", 1.20)])   # c crowned; b -> chain
    assert r.slots[:3] == ["c", "b", "a"]
    assert all(abs(w - 1 / 3) < 1e-9 for w in r.weights.values()), r.weights


def test_kingchain_eps_protects_the_king_from_noise():
    """A challenger must CLEAR the king's eps; a marginal lead must not flip the crown."""
    from thirtyspokes.reign import KingChain
    kc = KingChain(eps0=0.02)
    kc.update([_sub("a", 0.90)])
    r = kc.update([_sub("a", 0.90), _sub("b", 0.905)])       # +0.005 < eps
    assert r.slots[0] == "a" and not r.coronation
    r = kc.update([_sub("a", 0.90), _sub("b", 0.95)])        # +0.05 > eps
    assert r.slots[0] == "b" and r.coronation


def test_kingchain_ex_kings_decay_out_of_the_chain():
    from thirtyspokes.reign import KingChain
    kc = KingChain(chain_size=2)                              # king + ONE ex-king
    kc.update([_sub("a", 0.50)])
    kc.update([_sub("a", 0.50), _sub("b", 0.60)])             # b king, a in chain
    r = kc.update([_sub("a", 0.50), _sub("b", 0.60), _sub("c", 0.70)])   # c king, b in chain
    assert "a" not in r.weights, "a must decay out once the chain is full"
    assert set(r.weights) == {"c", "b"}


def test_kingchain_burns_when_nobody_is_registered():
    from thirtyspokes.reign import KingChain
    kc = KingChain()
    kc.update([_sub("a", 0.9)])
    r = kc.update([_sub("a", 0.9)], deregistered={"a"})
    assert r.burn == 1.0 and r.weights == {"uid0": 1.0}


def test_kingchain_copy_cannot_steal_the_crown_by_seniority():
    """The economics that stop copy-mining must survive the switch to equal-share.

    A copier that reproduces the king's score EXACTLY still loses: ties resolve to the earlier
    commit block. (A behaviourally identical copy is DQ'd upstream as `copy_of:` and never even
    reaches here — this is the second line of defence, for a copy that merely matches the score.)
    """
    from thirtyspokes.reign import KingChain, Submission
    kc = KingChain()
    orig = Submission("orig", "orig", 100, 0.90)             # committed at block 100
    copy = Submission("copy", "copy", 900, 0.90)             # same score, committed LATER
    kc.update([orig])
    r = kc.update([orig, copy])
    assert r.slots[0] == "orig", "later commit must not take the crown on an exact tie"
    assert "copy" not in r.weights, "a copy that never held the crown earns nothing"


def test_kingchain_recommit_keeps_the_crown_but_not_a_fresh_eps_clock():
    """Re-committing (i.e. IMPROVING your agent) must not cost you the crown, and must not buy you
    a refreshed eps shield either.

    The eps clock is keyed to the REIGN (`_king_since`), not the artifact. Artifact-keyed age made
    the incumbent lookup miss on a re-commit, so a 15-epoch champion that shipped an improvement
    silently forfeited its earned protection and lost the crown to a challenger it still out-scored
    — punishing exactly the behaviour the subnet exists to reward. Reign-keyed age fixes that while
    still closing the arch-6f gaming vector, because re-committing no longer resets the clock.
    """
    from thirtyspokes.reign import KingChain, Submission
    kc = KingChain(eps0=0.02)
    old = Submission("a", "a", 10, 0.90)
    for _ in range(12):
        kc.update([old])                                     # a is a long-lived incumbent (eps decays)
    aged_eps = kc.eps(kc._king_age())
    fresh = Submission("a", "a", 500, 0.90)                  # same miner, NEW (improved) artifact
    # (a) a challenger INSIDE the margin still cannot take the crown from a re-committing king
    r = kc.update([fresh, Submission("b", "b", 20, 0.90 + aged_eps / 2)])
    assert r.slots[0] == "a", "shipping an improvement must not forfeit earned incumbency"
    # (b) but the re-commit bought no fresh shield: the margin is still the AGED one, not eps0
    assert kc.eps(kc._king_age()) < kc.eps0 / 2, "a re-commit must not refresh eps protection"
    # (c) and a challenger that genuinely clears the aged margin still wins
    r = kc.update([fresh, Submission("b", "b", 20, 0.99)])
    assert r.slots[0] == "b", "a genuinely better challenger must still be able to dethrone"


def test_kingchain_snapshot_restore_roundtrip():
    from thirtyspokes.reign import KingChain
    kc = KingChain()
    kc.update([_sub("a", 0.5)])
    kc.update([_sub("a", 0.5), _sub("b", 0.7)])
    kc2 = KingChain()
    kc2.restore(kc.snapshot())
    assert kc2.king.sub.miner_id == "b"
    assert [s.miner_id for s in kc2.chain] == ["a"]
    assert kc2.update([_sub("a", 0.5), _sub("b", 0.7)]).weights == {"b": 0.5, "a": 0.5}


# --- LIVENESS: emissions pay for WORK, never for a vested seat -----------------------------------
# Regression tests for the emission-capture exploits found by auditing the mining side. Before these,
# `KingChain` consulted only seat MEMBERSHIP + registration at payout, never whether the holder had
# actually submitted, so a miner could be crowned once and then draw its share forever.

def test_kingchain_idle_ex_king_earns_nothing():
    """THE headline exploit: take the crown once, go dark, collect forever.

    Measured before the fix: an ex-king that never submitted again drew 50% of emissions across 50
    idle epochs — the same rate as the miner that actually worked and paid for inference every one
    of them."""
    from thirtyspokes.reign import KingChain
    kc = KingChain()
    kc.update([_sub("idle", 0.50)])                          # idle takes the vacant crown
    kc.update([_sub("idle", 0.50), _sub("worker", 0.99)])    # worker dethrones it -> idle -> chain
    assert "idle" in {s.miner_id for s in kc.chain}
    earned = sum(kc.update([_sub("worker", 0.99)]).weights.get("idle", 0.0) for _ in range(50))
    assert earned == 0.0, "an ex-king that stops submitting must earn nothing"


def test_kingchain_evicts_a_seat_after_the_absence_grace():
    """A missed epoch is forgiven (real CVM boots are flaky); sustained absence is not."""
    from thirtyspokes.reign import KingChain
    kc = KingChain(absent_grace=3)
    kc.update([_sub("a", 0.50)])
    kc.update([_sub("a", 0.50), _sub("b", 0.99)])            # b king, a in chain
    for i in range(1, 3):                                    # a misses 1..2 -> still seated
        kc.update([_sub("b", 0.99)])
        assert "a" in {s.miner_id for s in kc.chain}, f"evicted too early at miss {i}"
    kc.update([_sub("b", 0.99)])                             # third consecutive miss -> evicted
    assert "a" not in {s.miner_id for s in kc.chain}
    # and coming back does not restore the seat — you must retake the crown
    r = kc.update([_sub("a", 0.50), _sub("b", 0.99)])
    assert "a" not in r.weights


def test_kingchain_absent_king_keeps_the_crown_and_the_eps_bar():
    """Withholding one epoch used to hand the crown to ANY submitter, unguarded: a 21-epoch
    champion lost to a 0.01-scoring agent by missing a single upload. An absent king now retains
    the title (unpaid) and goes on setting the eps bar until its grace runs out."""
    from thirtyspokes.reign import KingChain
    kc = KingChain(absent_grace=3)
    for _ in range(21):
        kc.update([_sub("champ", 0.90)])
    r = kc.update([_sub("usurper", 0.01)])                   # champ withholds ONE epoch
    assert r.slots[0] == "champ" and not r.coronation, "absence must not be a cheap handoff"
    assert r.weights.get("champ", 0.0) == 0.0, "...but an absent king is not paid either"
    assert r.weights == {"usurper": 1.0} or r.burn == 1.0
    # a genuinely better challenger can still take an absent king's crown immediately
    r = kc.update([_sub("better", 0.99)])
    assert r.slots[0] == "better" and r.coronation


def test_kingchain_delinquent_king_vacates_and_gets_no_pension():
    """Once the grace is exhausted the crown truly vacates — and a king that lost it by going dark
    does NOT get a pension seat, or 'abdicate then collect' would just be the exploit again."""
    from thirtyspokes.reign import KingChain
    kc = KingChain(absent_grace=2)
    kc.update([_sub("dead", 0.90)])
    kc.update([_sub("live", 0.10)])                          # miss 1: dead retains (live is worse)
    assert kc.king.sub.miner_id == "dead"
    r = kc.update([_sub("live", 0.10)])                      # miss 2: grace exhausted -> vacates
    assert r.slots[0] == "live" and kc.king.sub.miner_id == "live"
    assert "dead" not in {s.miner_id for s in kc.chain}, "a delinquent king must not be pensioned"
    assert "dead" not in r.weights


def test_kingchain_sybil_crown_rotation_captures_nothing():
    """A cartel used to rotate the crown through 5 self-owned hotkeys and then stop entirely,
    holding 100% of emissions and capping any honest miner at 1/5 of the pot."""
    from thirtyspokes.reign import KingChain
    kc = KingChain(absent_grace=3)
    sybils = [f"S{i}" for i in range(5)]
    for i, s in enumerate(sybils):                            # each takes a turn submitting alone
        kc.update([_sub(s, 0.30, block=i + 1)])
    r = kc.update([_sub("S4", 0.30, block=5)])                # cartel now mostly idle
    assert sum(w for m, w in r.weights.items() if m in sybils) <= 1.0
    idle = [m for m in sybils if m != "S4"]
    assert all(r.weights.get(m, 0.0) == 0.0 for m in idle), "idle sybils must earn nothing"
    r = kc.update([_sub("honest", 0.99, block=99)])           # an honest miner arrives
    assert r.weights.get("honest") == 1.0, "a better honest miner must not be capped by idle seats"


def test_kingchain_burns_when_no_miner_is_live():
    """An epoch nobody submits must BURN, not silently keep paying the last slate."""
    from thirtyspokes.reign import KingChain
    kc = KingChain()
    kc.update([_sub("a", 0.9)])
    kc.update([_sub("a", 0.9), _sub("b", 0.95)])
    r = kc.update([])                                         # nobody submits
    assert r.burn == 1.0 and r.weights == {"uid0": 1.0}


def test_kingchain_live_set_overrides_candidacy():
    """Under `scoring_mode="accumulate"` a miner that MISSED the epoch is still scored (miss=0) and
    so is still a candidate. Paying on candidacy would reopen the pension, so liveness is passed
    explicitly and must win over mere presence in `candidates`."""
    from thirtyspokes.reign import KingChain
    kc = KingChain()
    kc.update([_sub("a", 0.90)])
    kc.update([_sub("a", 0.90), _sub("b", 0.99)])             # b king, a in chain
    r = kc.update([_sub("a", 0.90), _sub("b", 0.99)], live={"b"})   # a is a candidate but ABSENT
    assert r.weights == {"b": 1.0}, "a scored-but-absent candidate must not be paid"


def test_kingchain_liveness_state_survives_a_restart():
    """Miss counters must persist, else every validator restart forgives every absence and the
    pension reopens for `absent_grace` epochs at a time."""
    from thirtyspokes.reign import KingChain
    kc = KingChain(absent_grace=2)
    kc.update([_sub("a", 0.50)])
    kc.update([_sub("a", 0.50), _sub("b", 0.99)])
    kc.update([_sub("b", 0.99)])                              # a: miss 1
    kc2 = KingChain(absent_grace=2)
    kc2.restore(kc.snapshot())
    assert kc2._misses.get("a") == 1
    kc2.update([_sub("b", 0.99)])                             # miss 2 -> evicted, not forgiven
    assert "a" not in {s.miner_id for s in kc2.chain}
