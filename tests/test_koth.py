"""KOTH-TEE tests: the artifact binding is enforced (runtime hashes what it runs;
validator binds to the downloaded commit), the proof verifies + grades, the Pareto
guard verdicts are sound, and the anti-cheat backstops catch a memorizer (statistical
collapse), a hardcoder (public source), and a copier (behavioral dedup)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from thirtyspokes.gateway.gateway import MockBackend
from thirtyspokes.gateway.signing import Signer
from thirtyspokes.koth import benchmarks, commit
from thirtyspokes.koth.proof import Proof
from thirtyspokes.koth.runtime import Artifact, KOTHRuntime, load_agent, runtime_measurement
from thirtyspokes.koth.store import hash_source
from thirtyspokes.koth.subnet import _ROUTER_SRC, _art, _keys, N_PER_BENCH, run_simulation
from thirtyspokes.koth.verify import (
    BenchStat,
    Challenge,
    ProofVerdict,
    adjudicate_challenge,
    behavioral_duplicates,
    dethrone_guard,
    memorization_collapsed,
    scan_source,
    verify_proof,
)
from thirtyspokes.tee.attestation import Platform


@pytest.fixture
def env():
    suite = benchmarks.default_suite()
    pool, allk, weak = _keys(suite)
    return SimpleNamespace(platform=Platform(), backend=MockBackend(), suite=suite,
                           approved={runtime_measurement()}, pool=pool, allk=allk, weak=weak)


def _proof(env, artifact, hotkey, *, epoch=1, nonce="n1"):
    proof, _trace = KOTHRuntime(env.backend, env.platform).run(
        artifact, hotkey=hotkey, epoch=epoch, nonce=nonce, suite=env.suite, n_per_bench=N_PER_BENCH)
    return proof


def _verify(env, proof, artifact, hotkey, **over):
    kw = dict(approved_measurements=env.approved, platform_public_hex=env.platform.public_hex,
              expect_epoch=1, expect_nonce="n1", expect_hotkey=hotkey,
              expect_source_hash=artifact.source_hash, expect_weights_hash=artifact.weights_hash,
              suite=env.suite, n_per_bench=N_PER_BENCH)
    kw.update(over)
    return verify_proof(proof, **kw)


# --- binding + attestation ---------------------------------------------------
def test_valid_proof_verifies_and_scores(env):
    a = _art(_ROUTER_SRC, env.allk)
    v = _verify(env, _proof(env, a, "hk"), a, "hk")
    # A perfect slice must NOT certify Q_lcb = 1.0: eight questions cannot rule out a worse true
    # accuracy, and claiming they can is what made a lucky slice able to dethrone anyone.
    assert v.valid and 0.7 < v.score < 1.0
    assert all(bs.acc == 1.0 for bs in v.per_bench.values())


def test_runtime_computes_hashes_from_what_it_runs(env):
    """A2/A1: run artifact B, verify against artifact A's committed hashes -> mismatch.
    The runtime stamps hashes of what it actually loaded, so B can't masquerade as A."""
    a = _art(_ROUTER_SRC, env.allk)          # committed/public artifact
    b = _art(_ROUTER_SRC, env.pool)          # a different artifact actually run
    v = _verify(env, _proof(env, b, "hk"), a, "hk")   # expect A's hashes
    assert v.reason == "artifact_binding_mismatch"


def test_binding_tamper_breaks_report_data(env):
    a = _art(_ROUTER_SRC, env.allk)
    tampered = replace(_proof(env, a, "hk"), total_cost_usd=0.0)
    assert _verify(env, tampered, a, "hk").reason == "report_data_mismatch"


def test_replay_and_hotkey_and_forged_quote(env):
    a = _art(_ROUTER_SRC, env.allk)
    assert _verify(env, _proof(env, a, "hk", nonce="STALE"), a, "hk").reason == "epoch_nonce_mismatch"
    assert _verify(env, _proof(env, a, "hk"), a, "hk", expect_hotkey="other").reason == "hotkey_mismatch"
    p = Proof(1, "n1", "hk", a.source_hash, a.weights_hash, "r", (), 0.0, 0, "x",
              measurement=runtime_measurement()).attested_by(Platform(Signer()))
    assert _verify(env, p, a, "hk").reason == "bad_platform_quote"
    p2 = Proof(1, "n1", "hk", a.source_hash, a.weights_hash, "r", (), 0.0, 0, "x",
               measurement="MODIFIED").attested_by(env.platform)
    assert _verify(env, p2, a, "hk").reason == "unapproved_runtime"


def test_load_agent_from_bytes(env):
    a = _art(_ROUTER_SRC, {"Q": "A"})
    agent = load_agent(a.source_text, a.weights)
    assert agent("Q", lambda *x, **k: "") == "A"


# --- Pareto dethrone guard (verdicts; validator clamps on ANY not-ok) --------
def _vd(accs, *, cost=0.1):
    per = {n: BenchStat(8, a, max(0.0, a - 0.05), 0.0) for n, a in accs.items()}
    return ProofVerdict(True, "ok", per, cost, sum(accs.values()) / len(accs))


# --- P0: the live suite must SAMPLE, not hand out the whole pool every epoch ------------------
def test_real_suite_pool_far_exceeds_the_per_epoch_sample():
    """P0 regression. `real_suite` loaded n_load=16 -> half() -> an 8-item pool, and the validator
    draws n_per_bench=8 => it handed out the ENTIRE pool every epoch. The scored set was 16 fixed
    public questions forever: the epoch nonce selected nothing, the anti-grind commit window had no
    sample variance to protect, and the set was trivially memorizable."""
    import inspect

    from thirtyspokes.koth.benchmarks import BenchTask, RealBenchmark, bench_seed, real_suite
    n_load = inspect.signature(real_suite).parameters["n_load"].default
    pool_per_bench = n_load // 2                      # real_suite half()s into pool / held-out
    assert pool_per_bench >= 50 * 8, (                # >= 50x the daemon's n_per_bench=8
        f"pool of {pool_per_bench} is not >> n_per_bench=8 — the slice cannot vary")

    # and the draw genuinely varies epoch to epoch
    pool = [BenchTask(f"q{i}", f"p{i}", "g") for i in range(pool_per_bench)]
    b = RealBenchmark("mmlu", 0.5, pool, pool, lambda a, g: a == g)
    s1 = {t.task_id for t in b.sample(8, bench_seed("beacon", 1, "mmlu"))}
    s2 = {t.task_id for t in b.sample(8, bench_seed("beacon", 2, "mmlu"))}
    assert s1 != s2 and len(s1) == 8


# --- P0: --probe-bank must actually switch on the probe audit ---------------------------------
def test_probe_bank_implies_probe_audit_mode():
    """P0 regression. The daemon never passed `audit_mode`, so it stayed "grounding" while the
    validator gated the whole probe path on `audit_mode == "probe"` — making --probe-bank a silent
    no-op (the bank was loaded, handed over, and ignored)."""
    from thirtyspokes.koth.neuron import resolve_audit_mode
    assert resolve_audit_mode("grounding", "/path/bank.json") == "probe"   # the bug
    assert resolve_audit_mode("probe", "/path/bank.json") == "probe"
    assert resolve_audit_mode("grounding", None) == "grounding"            # default unchanged
    assert resolve_audit_mode("probe", None) == "probe"                    # public-slice probe


# --- P0: a saturated suite must not freeze emissions on the earliest committer -----------------
def test_cost_dethrones_at_quality_parity_but_never_beats_quality(env):
    """P0 regression. Q_lcb saturates: at the accuracy ceiling no challenger can be "confidently
    dominant", so `dethrone_guard` protected the incumbent forever and the reign fell back to
    commit-block seniority => emissions frozen on whoever committed first. Cost is the axis that
    never saturates, so parity + materially cheaper is now a valid dethrone."""
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign

    def vd(acc, cost):
        per = {n: BenchStat(8, acc, acc, cost / 2) for n in ("mmlu", "math")}
        return ProofVerdict(True, "ok", per, cost, acc)

    king = vd(1.0, 0.100)                                    # saturated + expensive
    assert dethrone_guard(vd(1.0, 0.050), king) == (True, "ok")          # 50% cheaper at parity -> wins
    assert dethrone_guard(vd(1.0, 0.095), king)[1] == "no_confident_gain"  # 5% cheaper -> not enough
    assert dethrone_guard(vd(0.90, 0.010), king)[1].startswith("regression")  # cheap but WORSE -> blocked

    # the ranking scalar must order equal-quality miners by cost, yet never outrank real quality
    v = KOTHValidator(env.approved, "pub", None, Reign(), env.suite, None, None,
                      budget=0.5, cost_tiebreak=0.02)
    assert v._reign_scalar(vd(1.0, 0.05)) > v._reign_scalar(vd(1.0, 0.10))   # cheaper ranks higher
    # a genuinely better agent that is also maximally expensive still beats a perfect-cost worse one
    better_pricey, worse_free = vd(1.0, 0.5), vd(0.875, 0.0)   # 8/8 vs 7/8 on both benchmarks
    assert v._reign_scalar(better_pricey) > v._reign_scalar(worse_free)


def test_guard_blocks_regression_and_marginal_gain(env):
    king = _vd({"math": 0.80, "mmlu": 0.80, "gpqa": 0.80, "swe": 0.80})
    assert dethrone_guard(_vd({"math": 1.0, "mmlu": 0.8, "gpqa": 0.8, "swe": 0.5}), king)[1] == "regression:swe"
    # not-worse everywhere but confidently dominant nowhere -> the validator clamps this too
    assert dethrone_guard(_vd({"math": 0.81, "mmlu": 0.81, "gpqa": 0.81, "swe": 0.81}), king)[1] == "no_confident_gain"
    assert dethrone_guard(_vd({"math": 0.95, "mmlu": 0.85, "gpqa": 0.82, "swe": 0.81}), king) == (True, "ok")


# --- anti-cheat backstops ----------------------------------------------------
def test_memorization_two_proportion_test():
    assert not memorization_collapsed(0.90, 32, 0.88, 32)   # honest: no significant drop
    assert memorization_collapsed(1.00, 32, 0.10, 32)       # memorizer: collapse
    assert memorization_collapsed(1.00, 32, 0.0, 0)         # empty fresh -> fail closed


def test_behavioral_dedup_keeps_earliest_commit():
    fps = {"A": ("x", "y", "z"), "B": ("x", "y", "z"), "C": ("p", "q", "r")}
    losers = behavioral_duplicates(fps, {"A": 0, "B": 5, "C": 0}, agree=0.95)
    assert losers == {"B": "A"}                             # B duplicates the earlier-committed A


def test_source_scan_and_bound_fraud_proof():
    hard = "ANSWER_TABLE = {'q': 'A'}\n"
    assert scan_source(hard)[0] and not scan_source("def route(p): return llm(p)")[0]
    # fraud-proof binds: fabricated source against an honest miner is rejected
    assert adjudicate_challenge(Challenge("v", "t", hard), hash_source(hard)) == (True, "hardcoded_answers")
    assert adjudicate_challenge(Challenge("v", "t", hard), "0" * 64) == (False, "unbound_source")


def test_commit_is_hotkey_salted():
    data = commit.commit_string("hk-A", "user/repo", "rev1", "src", "wts")
    assert commit.verify_commit(data, "hk-A", "src", "wts")
    assert not commit.verify_commit(data, "hk-B", "src", "wts")


# --- decoupled daemon flow (real subnet architecture) ------------------------
def test_proof_json_roundtrip_preserves_binding(env):
    a = _art(_ROUTER_SRC, env.allk)
    proof = _proof(env, a, "hk")
    back = Proof.from_json(proof.to_json())
    assert back.report_data() == proof.report_data()        # binding survives serialization
    assert _verify(env, back, a, "hk").valid                 # downloaded proof still verifies


def test_freeloader_that_stops_mining_earns_nothing_end_to_end(env):
    """END-TO-END emission-capture regression, through the real validator (verify_proof -> grounding
    -> dedup -> eligibility -> KingChain -> set_weights).

    Measured before the fix: a miner running the CHEAPEST pool model took the vacant crown in one
    epoch, went permanently dark, and still captured 54% of emissions over 12 epochs — out-earning
    the honest miner that worked every single epoch. The validator recorded `no_proof` for it every
    epoch and paid it anyway, because payout read seat membership rather than work.
    """
    import json
    import tempfile

    from thirtyspokes.gateway.signing import Signer
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.neuron import store_get_proof
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import KingChain
    from thirtyspokes.subnet.chain import LocalFileChain

    root = tempfile.mkdtemp()
    chain = LocalFileChain(f"{root}/chain"); store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    relay = ("import json\ndef build_agent(w):\n m=json.loads(w.decode())['model']\n"
             " def agent(p,cm):\n  return cm(m,[{'role':'user','content':p}],{'max_tokens':16})\n return agent\n")
    art = lambda m: Artifact(relay, json.dumps({"model": m}).encode(), m)  # noqa: E731
    val = KOTHValidator({runtime_measurement()}, env.platform.public_hex, chain, KingChain(),
                        env.suite, store, MockPool(), n_per_bench=8, budget_per_task=1.0, f_min=0.0)
    mk = lambda m, r: KOTHMinerNeuron(Signer().public_hex, MockPool(), env.platform, env.suite,  # noqa: E731
                                      chain, store, art(m), r)
    freeloader, honest = mk("cheap", "f/repo"), mk("strong", "h/repo")
    for m in (freeloader, honest):
        m.publish()
    gp = store_get_proof(store)

    earned = {freeloader.hotkey: 0.0, honest.hotkey: 0.0}
    for i in range(12):
        chain.advance(EPOCH_BLOCKS)
        e = current_epoch(chain)
        for m in ([freeloader] if i == 0 else [honest]):     # freeloader mines ONCE, then goes dark
            m.run_once(e)
        rep = val.run_epoch(gp)
        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            hk = uid_hk.get(uid, "")
            if hk in earned:
                earned[hk] += w

    assert earned[freeloader.hotkey] <= 1.0, "one mined epoch must not pay out for eleven idle ones"
    assert earned[honest.hotkey] > earned[freeloader.hotkey] * 5, \
        "the miner doing the work must dominate the one that quit"
    assert val.reign.king.sub.hotkey == honest.hotkey


def test_epoch_with_no_submissions_burns_instead_of_freezing_weights(env):
    """If nobody submits, the validator must still settle the epoch and BURN.

    It used to skip `set_weights` entirely when there was nothing to score, leaving the previous
    slate standing on-chain — so a network where every miner stopped kept paying its last five
    miners in full, forever, for producing nothing."""
    import tempfile

    from thirtyspokes.koth.epoch import EPOCH_BLOCKS
    from thirtyspokes.koth.neuron import store_get_proof
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import KingChain
    from thirtyspokes.subnet.chain import LocalFileChain

    root = tempfile.mkdtemp()
    chain = LocalFileChain(f"{root}/chain"); store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    val = KOTHValidator({runtime_measurement()}, env.platform.public_hex, chain, KingChain(),
                        env.suite, store, MockPool(), n_per_bench=8)
    chain.advance(EPOCH_BLOCKS)
    rep = val.run_epoch(store_get_proof(store))              # no miners committed at all
    assert rep.weights_by_uid == {0: 1.0}, "an empty epoch must burn to uid 0"
    # it must actually reach the chain — the bug was that set_weights was skipped entirely
    assert val._last_submitted_weights == {0: 1.0}
    assert chain._load()["weights"] == {"0": 1.0}


def test_decoupled_neuron_flow_scores_via_store():
    """Miner uploads proof to its repo; validator downloads from the chain+store and
    scores it — the two never call each other."""
    from thirtyspokes.koth.neuron import run_local
    out = run_local(epochs=3, verbose=False)
    em = out["emissions"]
    # KingChain economics: only the king and registered EX-kings are paid, equally. The weak miner
    # never takes the crown, so it earns nothing — and nothing burns while a king is seated (the
    # 5-slot reign used to burn its unfilled slots).
    assert em.get("strong", 0) > 0
    assert em.get("weak", 0) == 0, "a miner that never held the crown must not be paid"
    assert em.get("burn", 0) == 0, "no burn while a king is seated"


# --- WS1: pinned pool + gradeable router ----------------------------------------
_V2_ROUTER = (
    "import json\n"
    "def build_agent(weights):\n"
    "    model = json.loads(weights.decode()).get('model', 'strong')\n"
    "    def agent(prompt, call_model):\n"
    "        return call_model(model, [{'role': 'user', 'content': prompt}], {'max_tokens': 16})\n"
    "    return agent\n")


def test_pinned_pool_router_scores_and_rejects_unpinned(env):
    import json
    from thirtyspokes.koth.pool import MockPool, PinnedBackend, UnpinnedModelError
    pool = MockPool()
    backend = PinnedBackend(pool, pool.allowed)
    rt = KOTHRuntime(backend, env.platform)
    strong = Artifact(_V2_ROUTER, json.dumps({"model": "strong"}).encode(), "strong")
    proof, _ = rt.run(strong, hotkey="hk", epoch=1, nonce="n1", suite=env.suite, n_per_bench=8)
    v = _verify(env, proof, strong, "hk")
    assert v.valid and v.per_bench["math"].acc >= 0.9    # strong pool model solves synthetic tasks
    # a router that names a model outside the allow-list cannot produce a proof
    bad = Artifact(_V2_ROUTER, json.dumps({"model": "secret-gpt-5"}).encode(), "x")
    with pytest.raises(UnpinnedModelError):
        rt.run(bad, hotkey="hk", epoch=1, nonce="n1", suite=env.suite, n_per_bench=8)


# --- grounding memorization defense (docs/DESIGN.md §6: default "grounding" mode, no miner code runs) ---
def test_grounding_check_predicate():
    from thirtyspokes.eval.math_tasks import grade as gmath
    from thirtyspokes.koth.benchmarks import BenchTask, RealBenchmark, grade_choice
    from thirtyspokes.koth.proof import BenchmarkResult, Proof
    from thirtyspokes.koth.verify import grounding_check
    suite = [RealBenchmark("mmlu", 0.5, [BenchTask("m1", "q", "B")], [], grade_choice),
             RealBenchmark("math", 0.5, [BenchTask("g1", "q", 42.0)], [], gmath)]
    res = [BenchmarkResult("mmlu", "m1", "B", 0.0), BenchmarkResult("math", "g1", "42", 0.0)]
    mk = lambda: Proof(1, "n", "hk", "s", "w", "mid", tuple(res), 0.0, 2, "clh", "meas")  # noqa: E731
    # honest: the answer is present in a logged response -> grounded
    assert grounding_check(mk(), [{"task_id": "m1", "response": "The answer is B"},
                                  {"task_id": "g1", "response": "= 42"}], suite) == (True, "ok")
    # memorizer: throwaway calls return C/99, agent returns memorized B/42 -> ungrounded
    assert grounding_check(mk(), [{"task_id": "m1", "response": "C"},
                                  {"task_id": "g1", "response": "99"}], suite) == (False, "ungrounded")
    # KNOWN residual: laundering (the model echoes the injected answer) passes
    assert grounding_check(mk(), [{"task_id": "m1", "response": "B"},
                                  {"task_id": "g1", "response": "42"}], suite)[0] is True
    # tolerance: 1/2 ungrounded (0.5) DQs by default but is admitted under a looser threshold
    part = [{"task_id": "m1", "response": "B"}, {"task_id": "g1", "response": "99"}]
    assert grounding_check(mk(), part, suite)[0] is False
    assert grounding_check(mk(), part, suite, max_ungrounded=0.6)[0] is True


def test_grounding_dqs_memorizer_via_runtime(env):
    """Real runtime trace: a relay agent (answers from the pool) is grounded; a memorizer
    (answers from its weights key, ignoring the pool) is DQ'd `ungrounded`."""
    import json

    from thirtyspokes.koth.verify import grounding_check
    relay = Artifact(_V2_ROUTER, json.dumps({"model": "strong"}).encode(), "relay")
    p_r, t_r = KOTHRuntime(env.backend, env.platform).run(
        relay, hotkey="r", epoch=1, nonce="n1", suite=env.suite, n_per_bench=N_PER_BENCH)
    assert grounding_check(p_r, t_r, env.suite) == (True, "ok")
    mem = _art(_ROUTER_SRC, env.allk)                     # answer-from-weights = the memorizer pattern
    p_m, t_m = KOTHRuntime(env.backend, env.platform).run(
        mem, hotkey="m", epoch=1, nonce="n1", suite=env.suite, n_per_bench=N_PER_BENCH)
    ok, why = grounding_check(p_m, t_m, env.suite)
    assert not ok and why == "ungrounded"


def test_copy_dedup_on_scored_answers():
    """Grounding mode fingerprints copy-dedup on the attested scored answers (no probe)."""
    from thirtyspokes.koth.verify import behavioral_duplicates
    fp = {"orig": ("A", "B", "C"), "copy": ("A", "B", "C"), "diff": ("A", "B", "D")}
    losers = behavioral_duplicates(fp, {"orig": 1, "copy": 2, "diff": 1}, agree=0.95)
    assert losers == {"copy": "orig"}                    # earliest commit kept; identical vector DQ'd


# --- small-sample honesty: the LCB must actually price 8-question uncertainty --------------------
def test_lcb_never_certifies_a_perfect_small_slice():
    """A perfect slice must not report lcb = 1.0.

    The resampling bootstrap was degenerate on all-same slices — resampling 8 identical values gives
    8 identical means, so 8/8 returned lcb = 1.000, i.e. *zero* claimed uncertainty from eight
    questions. That cleared `dethrone_guard`'s `lcb_c > acc_king + margin` against any king below
    0.97, making "resubmit until a slice comes up perfect" a free lottery ticket for the crown.
    """
    from thirtyspokes.koth.verify import _bootstrap_lcb
    perfect8 = _bootstrap_lcb([1.0] * 8, 0.05, 1000, 7)
    assert 0.7 < perfect8 < 0.8, "8/8 must be bounded well below 1.0"
    # more evidence at the same accuracy must bind tighter — that is the whole incentive to keep mining
    assert _bootstrap_lcb([1.0] * 32, 0.05, 1000, 7) > perfect8
    # monotone in accuracy, and an empty/zero slice floors at 0
    seq = [_bootstrap_lcb([1.0] * k + [0.0] * (8 - k), 0.05, 1000, 7) for k in range(9)]
    assert seq == sorted(seq) and seq[0] == 0.0
    # deterministic: two validators must agree without drawing the same resamples
    assert _bootstrap_lcb([1.0] * 5 + [0.0] * 3, 0.05, 1000, 1) == \
           _bootstrap_lcb([1.0] * 5 + [0.0] * 3, 0.05, 1000, 999)


def test_lucky_perfect_slice_cannot_dethrone_a_better_king():
    """End of the variance-grinding lottery: a genuinely worse agent that draws a perfect 8-question
    slice must NOT clear the guard against a king with higher true accuracy."""
    from thirtyspokes.koth.verify import BenchStat, ProofVerdict, _bootstrap_lcb, dethrone_guard
    lucky_lcb = _bootstrap_lcb([1.0] * 8, 0.05, 1000, 7)            # the REAL bound, not a stub
    lucky = ProofVerdict(True, "ok", {"math": BenchStat(8, 1.0, lucky_lcb, 0.01)}, 0.01, lucky_lcb, 1.0)
    king = ProofVerdict(True, "ok", {"math": BenchStat(8, 0.95, 0.80, 0.01)}, 0.01, 0.95, 0.95)
    ok, why = dethrone_guard(lucky, king)
    assert not ok and why == "no_confident_gain"


def test_scoring_mode_defaults_to_accumulate(env):
    """Per-epoch scoring makes the crown a per-epoch lottery and charges nothing for a hidden epoch.
    Accumulation is the default so ranking is stable and decoupled validators agree."""
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import KingChain
    val = KOTHValidator(env.approved, "pub", None, KingChain(), env.suite, None, None)
    assert val.scoring_mode == "accumulate"


# --- docs/DESIGN.md §5b: evidence accumulation (scoring_mode="accumulate", the DEFAULT) -----------
def test_accumulate_mode_crowns_and_pools(env):
    """Accumulate mode pools per-artifact evidence over epochs, crowns the stronger miner, and the
    accumulator holds MORE than one epoch's tasks (the whole point — 32 q/epoch ranks stably)."""
    import json
    import tempfile

    from thirtyspokes.gateway.signing import Signer
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.neuron import store_get_proof
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import SUITE_VERSION, runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore, hash_source, hash_weights
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import LocalFileChain
    root = tempfile.mkdtemp()
    chain = LocalFileChain(f"{root}/chain"); store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    relay = ("import json\ndef build_agent(w):\n m=json.loads(w.decode())['model']\n"
             " def agent(p,cm):\n  return cm(m,[{'role':'user','content':p}],{'max_tokens':16})\n return agent\n")
    art = lambda m: Artifact(relay, json.dumps({"model": m}).encode(), m)  # noqa: E731
    val = KOTHValidator({runtime_measurement()}, env.platform.public_hex, chain,
                        Reign(eps0=0.0, eps_floor=0.0), env.suite, store, MockPool(), n_per_bench=8,
                        budget_per_task=1.0, f_min=0.0, scoring_mode="accumulate", half_life_epochs=50)
    strong = KOTHMinerNeuron(Signer().public_hex, MockPool(), env.platform, env.suite, chain, store,
                             art("strong"), "s/repo")
    weak = KOTHMinerNeuron(Signer().public_hex, MockPool(), env.platform, env.suite, chain, store,
                           art("cheap"), "w/repo")
    for m in (strong, weak):
        m.publish()
    gp = store_get_proof(store)
    rep = None
    for _ in range(6):
        chain.advance(EPOCH_BLOCKS)
        for m in (strong, weak):
            m.run_once(current_epoch(chain))
        rep = val.run_epoch(gp)
    assert rep.scored[strong.hotkey] > rep.scored[weak.hotkey] > 0        # accumulated ranking
    a = art("strong")
    ev = val._evidence.for_artifact(strong.hotkey, hash_source(a.source_text), hash_weights(a.weights),
                                    SUITE_VERSION)
    assert ev.total_n() > env.suite[0].weight * 0 + 8 * len(env.suite)    # > one epoch's tasks (pooled)


def test_accumulate_miss_is_zero(env):
    """A no-proof epoch accumulates (n_expected, 0) — withholding drags the pooled score down, and a
    miss is a reign candidate (decaying), not a hard DQ."""
    from thirtyspokes.koth.validator import KOTHValidator, _MinerEval
    from thirtyspokes.koth.verify import BenchStat, ProofVerdict
    from thirtyspokes.reign import Reign
    val = KOTHValidator(env.approved, "pub", None, Reign(), env.suite, None, None, n_per_bench=8,
                        scoring_mode="accumulate", budget_per_task=1.0, f_min=0.0)
    vd = ProofVerdict(True, "ok", {b.name: BenchStat(8, 1.0, 1.0, 0.0) for b in env.suite}, 0.0, 1.0, 1.0)
    s1 = val._accumulate({"m": _MinerEval(verdict=vd, sh="s", wh="w")}, {})
    dq = {"m": "no_proof"}
    s2 = val._accumulate({"m": _MinerEval(sh="s", wh="w")}, dq)           # a withheld epoch
    assert s2["m"] < s1["m"] and "m" not in dq                           # miss lowers score, still a candidate


# --- F7: intra-epoch commit-reveal (opt-in commit_window) bounds best-of-N grinding -------------
def test_commit_window_binds_proof_to_one_run(env):
    """A commit_window validator scores only a proof COMMITTED on-chain in-window and revealed exactly:
    honest (commits its report_data in-window) is scored; no-commit → no_proof_commit; a revealed proof
    that differs from the commit (post-commit best-of-N swap) → commit_mismatch; a commit past the window
    (grinding past the wall-clock bound) → commit_out_of_window."""
    import json
    import tempfile

    from thirtyspokes.gateway.signing import Signer
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.neuron import store_get_proof
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import LocalFileChain
    root = tempfile.mkdtemp()
    chain = LocalFileChain(f"{root}/chain"); store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    relay = ("import json\ndef build_agent(w):\n m=json.loads(w.decode())['model']\n"
             " def agent(p,cm):\n  return cm(m,[{'role':'user','content':p}],{'max_tokens':16})\n return agent\n")
    art = lambda: Artifact(relay, json.dumps({"model": "strong"}).encode(), "strong")  # noqa: E731
    W = 5
    val = KOTHValidator({runtime_measurement()}, env.platform.public_hex, chain,
                        Reign(eps0=0.0, eps_floor=0.0), env.suite, store, MockPool(), n_per_bench=8,
                        budget=999.0, f_min=0.0, commit_window=W, scoring_mode="per_epoch")
    mk = lambda repo, cp: KOTHMinerNeuron(Signer().public_hex, MockPool(), env.platform, env.suite,  # noqa: E731
                                          chain, store, art(), repo, commit_proofs=cp)
    honest, nocommit, swap, late = mk("h/r", True), mk("n/r", False), mk("s/r", False), mk("l/r", True)
    for m in (honest, nocommit, swap, late):
        m.publish()
    chain.advance(EPOCH_BLOCKS); e = current_epoch(chain)                 # e=1, window [100, 105]
    for m in (honest, nocommit, swap):
        m.run_once(e)                                                    # commit (honest) at block 100
    chain.commit_proof(swap.hotkey, e, "0" * 64)                         # swap: revealed proof != commit
    chain.advance(W + 3)                                                 # block 108, still epoch 1
    late.run_once(e)                                                     # late: commits at 108 > 105
    rep = val.run_epoch(store_get_proof(store), epoch=e)
    assert honest.hotkey in rep.scored and rep.scored[honest.hotkey] > 0
    assert rep.dq.get(nocommit.hotkey) == "no_proof_commit"
    assert rep.dq.get(swap.hotkey) == "commit_mismatch"
    assert rep.dq.get(late.hotkey) == "commit_out_of_window"


# --- F2: grace window — score a SETTLED epoch, never the live one, so validators agree -----------
def test_grace_window_defers_scoring(env):
    """F2: inside epoch E's grace the validator still scores E-1 (E's submissions haven't settled);
    only once the grace passes does it advance to E — it never scores the live epoch."""
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import MockChain
    chain = MockChain()
    val = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, None, None,
                        epoch_blocks=100, grace_blocks=20)
    chain.advance(100)                                     # epoch 1 just opened (block 100)
    assert val._settle_epoch() == 0                        # inside epoch 1's grace -> still score epoch 0
    chain.advance(19)                                      # block 119, still < 120
    assert val._settle_epoch() == 0
    chain.advance(1)                                       # block 120 = 100 + grace
    assert val._settle_epoch() == 1                        # grace passed -> now safe to score epoch 1


def test_grace_window_presence_deterministic(env):
    """F2: scoring a settled epoch is identical regardless of WHEN the validator polls (submissions
    have settled), and an absent miner — no proof committed/uploaded for that epoch — is a
    deterministic miss=0 that ranks below a present miner."""
    import json
    import tempfile

    from thirtyspokes.gateway.signing import Signer
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.neuron import store_get_proof
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import LocalFileChain
    root = tempfile.mkdtemp()
    chain = LocalFileChain(f"{root}/chain"); store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    relay = ("import json\ndef build_agent(w):\n m=json.loads(w.decode())['model']\n"
             " def agent(p,cm):\n  return cm(m,[{'role':'user','content':p}],{'max_tokens':16})\n return agent\n")
    art = lambda: Artifact(relay, json.dumps({"model": "strong"}).encode(), "strong")  # noqa: E731
    W, G = 5, 20
    mk = lambda repo: KOTHMinerNeuron(Signer().public_hex, MockPool(), env.platform, env.suite,  # noqa: E731
                                      chain, store, art(), repo, commit_proofs=True)
    present, absent = mk("p/r"), mk("a/r")
    for m in (present, absent):
        m.publish()
    chain.advance(EPOCH_BLOCKS)                            # epoch 1
    present.run_once(1)                                    # commit + upload for epoch 1; absent never submits
    chain.advance(G + 3)                                   # block 123: past grace, still epoch 1

    def score():
        v = KOTHValidator({runtime_measurement()}, env.platform.public_hex, chain,
                          Reign(eps0=0.0, eps_floor=0.0), env.suite, store, MockPool(), n_per_bench=8,
                          budget=999.0, f_min=0.0, budget_per_task=999.0, commit_window=W,
                          grace_blocks=G, scoring_mode="accumulate")
        return v.run_epoch(store_get_proof(store))         # epoch derived from the grace window
    repA = score()
    chain.advance(11)                                      # a later poll-time, same settled epoch
    repB = score()
    assert repA.epoch == 1 and repB.epoch == 1             # grace picked the settled epoch, not the live one
    assert repA.scored[present.hotkey] > 0
    assert repA.scored[absent.hotkey] < repA.scored[present.hotkey]   # miss=0 dragged the absent miner down
    assert repA.scored == repB.scored and repA.dq == repB.dq         # identical across poll-times (F2)


def test_fresh_suite_disjoint_and_solvable(env):
    """WS4: fresh benchmarks generate disjoint sample/probe slices (nothing to memorize)
    yet stay gradeable, so a genuine router still scores."""
    from thirtyspokes.koth.benchmarks import fresh_suite
    from thirtyspokes.koth.devkit import evaluate
    from thirtyspokes.koth.miner import reference_artifact
    from thirtyspokes.koth.pool import MockPool
    suite = fresh_suite()
    for b in suite:
        s = {t.task_id for t in b.sample(8, 123)}
        p = {t.task_id for t in b.probe(8, 123)}
        assert s and p and not (s & p)                              # disjoint
    r = evaluate(reference_artifact("strong"), pool_backend=MockPool(), suite=suite,
                 n_per_bench=8, nonce="f1")
    assert r["valid"] and r["Q_lcb"] > 0.7          # solver still grades fresh tasks (8/8 -> LCB .747)


def test_devkit_matches_validator(env):
    """WS5: the dev kit reuses verify_proof, so a miner's local Q_lcb == the validator's."""
    from thirtyspokes.koth.devkit import evaluate
    from thirtyspokes.koth.miner import reference_artifact
    from thirtyspokes.koth.pool import MockPool
    art = reference_artifact("strong")
    r = evaluate(art, pool_backend=MockPool(), suite=env.suite, n_per_bench=8, nonce="p1")
    assert r["valid"] and r["eligible"] and r["Q_lcb"] > 0.5 and r["n_pool_calls"] > 0
    proof, _ = KOTHRuntime(MockPool(), env.platform).run(
        art, hotkey="dev", epoch=0, nonce="p1", suite=env.suite, n_per_bench=8)
    v = _verify(env, proof, art, "dev", expect_epoch=0, expect_nonce="p1")
    assert abs(r["Q_lcb"] - round(v.score, 4)) < 1e-6                # local == validator


_NOCALL_SRC = ("def build_agent(weights):\n"
               "    def agent(prompt, call_model):\n"
               "        return 'X'\n"          # answers free, never calls the pool
               "    return agent\n")


def test_v2_pinned_pool_gates(tmp_path):
    """End-to-end v2: pinned pool + trace + budgeted-quality. An honest router that
    orchestrates the pool wins; a 'free answer, no pool call' cheater is DQ'd; a
    tampered trace is rejected."""
    import json
    from thirtyspokes.koth.benchmarks import default_suite
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.neuron import store_get_proof
    from thirtyspokes.koth.pool import MockPool, PinnedBackend
    from thirtyspokes.koth.runtime import runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import LocalFileChain

    platform = Platform()
    pool = MockPool()
    backend = PinnedBackend(pool, pool.allowed)
    suite = default_suite()
    chain = LocalFileChain(str(tmp_path / "chain")); chain.register("burn")
    store = LocalBundleStore(str(tmp_path / "store"))
    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, Reign(),
                        suite, store, backend, n_per_bench=8, budget=1.0, f_min=0.1,
                        pool_spec={"kind": "mockpool"})
    miners = {
        "honest": KOTHMinerNeuron(Signer().public_hex, backend, platform, suite, chain, store,
                                  Artifact(_V2_ROUTER, json.dumps({"model": "strong"}).encode(), "s"), "honest/repo"),
        "nocall": KOTHMinerNeuron(Signer().public_hex, backend, platform, suite, chain, store,
                                  Artifact(_NOCALL_SRC, b"{}", "n"), "nocall/repo"),
    }
    name = {m.hotkey: n for n, m in miners.items()}
    for m in miners.values():
        m.publish()
    chain.advance(EPOCH_BLOCKS)
    epoch = current_epoch(chain)
    for m in miners.values():
        m.run_once(epoch)
    rep = val.run_epoch(store_get_proof(store))

    dq = {name.get(hk, hk): r for hk, r in rep.dq.items()}
    assert dq.get("nocall") == "no_pool_call"                        # must orchestrate the pool
    uid_hk = chain.hotkeys()
    won = {name.get(uid_hk.get(u, ""), "burn"): w for u, w in rep.weights_by_uid.items()}
    assert won.get("honest", 0) > 0                                  # honest router is paid

    # a tampered trace fails its binding
    from thirtyspokes.gateway import signing
    pj = json.loads(store.download_proof("honest/repo", epoch))
    trace = json.loads(store.download_trace("honest/repo", epoch))
    trace.append({"task_id": "forged", "model": "strong", "prompt": "x", "response": "y", "cost_usd": 0.0})
    assert signing.sha256_hex(trace) != pj["call_log_hash"]


# --- WS0: sandbox isolates untrusted miner code from the validator --------------
def test_sandbox_scrubs_validator_secrets(monkeypatch):
    from thirtyspokes.koth.sandbox import run_agent_probe
    monkeypatch.setenv("WALLET_SECRET", "s3cr3t-hotkey")
    evil = ("import os\n"
            "def build_agent(weights):\n"
            "    def agent(prompt, call_model):\n"
            "        return 'LEAK:' + str(os.environ.get('WALLET_SECRET'))\n"
            "    return agent\n")
    ans = run_agent_probe(evil, b"{}", ["q1"])
    assert "s3cr3t-hotkey" not in ans[0] and ans[0] == "LEAK:None"   # scrubbed env


def test_sandbox_fails_closed_on_hostile_source():
    from thirtyspokes.koth.sandbox import SandboxError, run_agent_probe
    with pytest.raises(SandboxError):
        run_agent_probe("def build_agent(w):\n    raise RuntimeError('boom')\n", b"{}", ["q1"])


# --- H3: no-egress confinement (agent's only channel is call_model, metered parent-side) -----
_TASK = [{"task_id": "t1", "benchmark": "b", "prompt": "Compute 2+2."}]
_CALLS_POOL = ("def build_agent(w):\n"
               "    def agent(prompt, call_model):\n"
               "        return call_model('mid', [{'role':'user','content':prompt}], {'max_tokens':8})\n"
               "    return agent\n")

needs_confinement = pytest.mark.skipif(
    not __import__("thirtyspokes.koth.confine", fromlist=["confine"]).confinement_available(),
    reason="no unprivileged netns confinement on this host")


def test_confine_root_skips_userns():
    """F3: as root the sandbox creates net/mount/pid namespaces DIRECTLY (no unprivileged userns),
    which is what makes it bootstrap on the locked measured image where Ubuntu 24.04 restricts /
    disables unprivileged user namespaces. Non-root still wraps in a userns."""
    import os

    from thirtyspokes.koth.confine import _argv
    argv = _argv(True)
    assert argv[0] == "unshare" and "--net" in argv          # no-egress netns
    assert ("--user" not in argv) if os.geteuid() == 0 else ("--user" in argv)


def test_confinement_probe_matches_execution_path():
    """Regression: `confinement_available()` must PROBE the namespaces `_argv` actually executes,
    never guess from a kernel knob. The old guess read `/proc/sys/kernel/unprivileged_userns_clone`
    — absent on Ubuntu 24.04, where AppArmor blocks unprivileged userns — so it claimed True and
    every sandbox spawn then died (`uid_map: Operation not permitted`), turning each audit into
    `audit_error:SandboxError` (and reddening CI)."""
    import subprocess as sp

    from thirtyspokes.koth.confine import _argv, _ns_flags, confinement_available
    flags = _ns_flags()
    argv = _argv(True)
    assert argv[0] == "unshare" and argv[1:1 + len(flags)] == flags      # detection == execution
    if confinement_available():        # if it CLAIMS available, a real unshare must actually work
        assert sp.run(["unshare", *flags, "--", "true"], capture_output=True).returncode == 0


def test_confine_meters_parent_side():
    """The agent reaches the pool only via the metered call_model RPC; the parent records the
    authoritative trace + per-task cost. Works with or without netns (portable path)."""
    from thirtyspokes.koth.confine import run_agent_confined
    from thirtyspokes.koth.pool import MockPool
    results, trace, _hardened = run_agent_confined(_CALLS_POOL, b"{}", _TASK, backend=MockPool())
    assert results[0]["answer"] == "4"
    assert len(trace) == 1 and trace[0]["cost_usd"] > 0
    assert results[0]["cost_usd"] == trace[0]["cost_usd"]     # cost attributed from the trace


@needs_confinement
def test_confine_blocks_network_egress():
    """Inside the confinement the agent has no route off-box — it cannot exfiltrate the slice."""
    from thirtyspokes.koth.confine import run_agent_confined
    from thirtyspokes.koth.pool import MockPool
    src = ("import socket\n"
           "def build_agent(w):\n"
           "    def agent(prompt, call_model):\n"
           "        try:\n"
           "            s=socket.create_connection(('1.1.1.1',53),timeout=3); s.close(); return 'EGRESS'\n"
           "        except Exception as e:\n"
           "            return 'BLOCKED:'+type(e).__name__\n"
           "    return agent\n")
    results, _tr, _hardened = run_agent_confined(src, b"{}", _TASK, backend=MockPool())
    assert results[0]["answer"].startswith("BLOCKED")


def test_confine_fails_closed_on_timeout():
    from thirtyspokes.koth.confine import SandboxError, run_agent_confined
    from thirtyspokes.koth.pool import MockPool
    hang = ("import time\n"
            "def build_agent(w):\n"
            "    def agent(prompt, call_model):\n"
            "        time.sleep(30); return 'x'\n"
            "    return agent\n")
    with pytest.raises(SandboxError):
        run_agent_confined(hang, b"{}", _TASK, backend=MockPool(), timeout=2.0)


def test_confined_run_retries_a_transient_spawn_crash(monkeypatch):
    """Real testnet finding (2026-07-14): the confined child's namespace/process spawn intermittently
    races with cold-boot resource contention and dies before the first handshake write
    (SandboxError: protocol error: Broken pipe), ~30% of real boots. It happens before any metered
    call, so a fresh retry is safe (no double-counted cost) and cheap (respawns a subprocess, not the
    VM). `_run_confined` must absorb a couple of these before giving up."""
    from thirtyspokes.koth.confine import SandboxError
    from thirtyspokes.koth.miner import reference_artifact
    from thirtyspokes.koth.pool import MockPool, PinnedBackend

    calls = {"n": 0}
    real_rows = ([{"benchmark": "b", "task_id": "t1", "answer": "ok", "cost_usd": 0.0}], [], True)

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise SandboxError("protocol error: [Errno 32] Broken pipe")
        return real_rows

    monkeypatch.setattr("thirtyspokes.koth.confine.run_agent_confined", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)     # no real backoff delay in tests

    rt = KOTHRuntime(PinnedBackend(MockPool(), MockPool().allowed), Platform(), confine=True)
    results, trace, hardened = rt._run_confined(reference_artifact("strong"), _TASK)
    assert calls["n"] == 3 and results[0].answer == "ok" and hardened is True


def test_confined_run_gives_up_after_repeated_crashes(monkeypatch):
    """The retry is bounded — a genuinely broken agent/environment still fails closed rather than
    retrying forever."""
    from thirtyspokes.koth.confine import SandboxError
    from thirtyspokes.koth.miner import reference_artifact
    from thirtyspokes.koth.pool import MockPool, PinnedBackend

    def always_fails(*a, **kw):
        raise SandboxError("protocol error: [Errno 32] Broken pipe")

    monkeypatch.setattr("thirtyspokes.koth.confine.run_agent_confined", always_fails)
    monkeypatch.setattr("time.sleep", lambda s: None)

    rt = KOTHRuntime(PinnedBackend(MockPool(), MockPool().allowed), Platform(), confine=True)
    with pytest.raises(SandboxError):
        rt._run_confined(reference_artifact("strong"), _TASK)


def test_runtime_confined_matches_inprocess():
    """confine=True produces the same answers + a valid proof as confine=False (parity)."""
    from thirtyspokes.gateway import signing
    from thirtyspokes.koth.miner import reference_artifact
    from thirtyspokes.koth.pool import MockPool, PinnedBackend
    from thirtyspokes.koth.runtime import KOTHRuntime
    from thirtyspokes.tee.attestation import Platform
    suite = benchmarks.default_suite()
    art = reference_artifact("strong")
    backend = PinnedBackend(MockPool(), MockPool().allowed)
    kw = dict(hotkey="m", epoch=1, nonce="n", suite=suite, n_per_bench=2)
    p_in, _ = KOTHRuntime(backend, Platform(), confine=False).run(art, **kw)
    p_cf, tr = KOTHRuntime(backend, Platform(), confine=True).run(art, **kw)
    a_in = [(r.benchmark, r.task_id, r.answer) for r in p_in.results]
    a_cf = [(r.benchmark, r.task_id, r.answer) for r in p_cf.results]
    assert a_in == a_cf and p_cf.n_calls == p_in.n_calls
    assert signing.sha256_hex(tr) == p_cf.call_log_hash          # trace binding intact


# --- end-to-end --------------------------------------------------------------
def test_simulation_every_defense_fires():
    out = run_simulation(epochs=4, verbose=False)
    dq = out["disqualifications"]
    assert dq.get("lying-rt") == "unapproved_runtime"
    assert dq.get("fake-quote") == "bad_platform_quote"
    assert dq.get("replayer") == "epoch_nonce_mismatch"
    assert dq.get("hardcoder") == "hardcoded_answers"
    # the memorizer keeps its answer table in weights.bin. `scan_weights` now catches it BEFORE the
    # probe audit runs — a cheaper, earlier catch — so accept either memorization defense firing.
    assert dq.get("memorizer") in {"answers_in_weights", "memorization", "ungrounded", "laundered"}
    assert dq.get("substituter") == "artifact_binding_mismatch"     # binding enforced
    assert dq.get("copier", "").startswith("copy_of:")              # behavioral dedup
    em = out["emissions"]
    # KingChain: the crown (and the pay) goes to the strongest honest miner; the weaker honest miner
    # never reigns, so it earns nothing. Every adversary above was DQ'd, so none of them is paid.
    assert em.get("honest-strong", 0) > 0
    assert em.get("honest-weak", 0) == 0


# --- WS7: real Intel TDX attestation (verified offline against a captured quote) -----
import base64
import hashlib
import pathlib

_FIXTURE = pathlib.Path(__file__).parent / "data" / "tdx_quote_v4.b64"
# the report_data the fixture was generated over (scripts/koth_tdx_smoke fixture capture)
_FIXTURE_RD_HASH = hashlib.sha256(b"koth-tdx-fixture").hexdigest()


def _load_fixture_quote() -> bytes:
    return base64.b64decode(_FIXTURE.read_text())


def test_tdx_quote_verifies_to_intel_root():
    """A genuine TDX v4 quote from real hardware verifies through the full DCAP chain to
    the pinned Intel SGX Root CA, and surfaces the attested MRTD/RTMRs."""
    from thirtyspokes.koth import tdx
    q = _load_fixture_quote()
    rd = tdx.report_data_from_hash(_FIXTURE_RD_HASH)
    vd = tdx.verify_quote(q, expect_report_data=rd, approved_mrtd=None)
    assert vd.ok, vd.reason
    assert len(vd.mr_td) == 96 and len(vd.rtmrs) == 4       # 48-byte measurements, hex
    assert vd.report_data == rd


def test_tdx_report_data_binding_is_load_bearing():
    """Changing the expected payload (as a proof-tamper would) breaks the hardware binding."""
    from thirtyspokes.koth import tdx
    q = _load_fixture_quote()
    wrong = tdx.report_data_from_hash(hashlib.sha256(b"tampered").hexdigest())
    vd = tdx.verify_quote(q, expect_report_data=wrong, approved_mrtd=None)
    assert not vd.ok and vd.reason == "report_data_mismatch"


def test_tdx_mrtd_gate_and_malformed_quote():
    from thirtyspokes.koth import tdx
    q = _load_fixture_quote()
    rd = tdx.report_data_from_hash(_FIXTURE_RD_HASH)
    assert tdx.verify_quote(q, expect_report_data=rd, approved_mrtd={"00" * 48}).reason \
        == "mrtd_not_approved"
    # random bytes are a verdict, not a crash
    assert not tdx.verify_quote(b"\x00" * 200, expect_report_data=rd).ok


def test_tdx_forged_signature_rejected():
    """Flip a byte in the TD-quote signature region -> the attestation-key check fails
    (the cert chain still verifies, proving we check the quote body signature too)."""
    from thirtyspokes.koth import tdx
    q = bytearray(_load_fixture_quote())
    q[640] ^= 0xFF                                          # inside the TD ECDSA signature
    rd = tdx.report_data_from_hash(_FIXTURE_RD_HASH)
    vd = tdx.verify_quote(bytes(q), expect_report_data=rd, approved_mrtd=None)
    assert not vd.ok and vd.reason == "td_quote_sig"


def test_tdx_unavailable_off_hardware():
    """Off a TDX guest the generation path fails cleanly and the field-verifier rejects
    non-TDX quotes (mock ed25519 platform_sig)."""
    from thirtyspokes.koth import tdx
    from thirtyspokes.tee.attestation import Quote
    if not tdx.tdx_available():
        with pytest.raises(RuntimeError):
            tdx.get_quote(b"\x00" * 64)
    assert tdx.verify_tdx_quote_field(Quote("m", "r", "ed25519sig")).reason == "not_a_tdx_quote"


def test_collateral_cache_keyed_by_pck_leaf_not_fmspc():
    """Regression (found on GCP TDX): two different physical CPUs share an FMSPC but each needs its
    OWN DCAP collateral (cross-CPU collateral fails 'qe_report signature invalid'), so a validator
    scoring many miners must key the collateral cache by the PCK leaf cert, not the FMSPC."""
    import hashlib
    from thirtyspokes.koth import collateral, tdx
    q = _load_fixture_quote()
    key = collateral._cache_key(q)
    assert key == hashlib.sha256(tdx.parse_quote(q).cert_pems[0]).hexdigest()  # per-CPU, not per-FMSPC
    assert len(key) == 64


# --- H1: full DCAP verification (TCB status / CRL / QE-identity) offline via pinned collateral ---
_COLLATERAL = pathlib.Path(__file__).parent / "data" / "tdx_collateral_v4.json"
_FIXTURE_NOW = 1783520381        # the wall-clock the fixture quote+collateral were captured at


def _dcap_qvl_available() -> bool:
    try:
        import dcap_qvl  # noqa: F401
        return True
    except Exception:
        return False


needs_dcap = pytest.mark.skipif(not _dcap_qvl_available(), reason="dcap-qvl not installed (tee/eval extra)")


@needs_dcap
def test_tdx_full_dcap_verifies_tcb_uptodate():
    """dcap-qvl full verification (chain+CRL+QE-identity+TCB) of the captured quote against
    the captured collateral at the captured instant → UpToDate, offline + deterministic."""
    from thirtyspokes.koth import tdx
    q, col = _load_fixture_quote(), _COLLATERAL.read_text()
    rd = tdx.report_data_from_hash(_FIXTURE_RD_HASH)
    vd = tdx.verify_quote_full(q, expect_report_data=rd, collateral=col, now=_FIXTURE_NOW)
    assert vd.ok, vd.reason
    assert vd.tcb_status == "UpToDate"
    assert len(vd.mr_td) == 96


@needs_dcap
def test_tdx_full_dcap_gates_and_binding():
    from thirtyspokes.koth import tdx
    q, col = _load_fixture_quote(), _COLLATERAL.read_text()
    rd = tdx.report_data_from_hash(_FIXTURE_RD_HASH)
    common = dict(collateral=col, now=_FIXTURE_NOW)
    # tamper -> rejected locally before any network/collateral use
    assert tdx.verify_quote_full(q, expect_report_data=tdx.report_data_from_hash(
        hashlib.sha256(b"x").hexdigest()), **common).reason == "report_data_mismatch"
    # a strict owner TCB policy rejects even UpToDate, with a distinct reason
    assert tdx.verify_quote_full(q, expect_report_data=rd, tcb_accept=frozenset(),
                                 **common).reason == "tcb_up_to_date"
    # MRTD / RTMR image gates
    assert tdx.verify_quote_full(q, expect_report_data=rd, approved_mrtd={"00" * 48},
                                 **common).reason == "mrtd_not_approved"
    assert tdx.verify_quote_full(q, expect_report_data=rd, approved_rtmr={0: "11" * 48},
                                 **common).reason == "rtmr0_not_approved"


@needs_dcap
def test_tdx_full_dcap_detects_expiry_via_now():
    """Verifying far in the future (collateral long past its nextUpdate) fails — proves the
    `now`/expiry check is live, not stubbed."""
    from thirtyspokes.koth import tdx
    q, col = _load_fixture_quote(), _COLLATERAL.read_text()
    rd = tdx.report_data_from_hash(_FIXTURE_RD_HASH)
    vd = tdx.verify_quote_full(q, expect_report_data=rd, collateral=col,
                               now=_FIXTURE_NOW + 10 * 365 * 24 * 3600)
    assert not vd.ok and vd.reason.startswith(("dcap_verify_failed", "tcb_"))


# --- H2: RTMR runtime self-measurement + owner governance -----------------------------------
def test_rtmr_extend_formula_and_expected():
    """RTMR3 extend is SHA384(base || digest); `expected_rtmr3` matches, and the same digest
    from the runtime-id inputs is deterministic (owner ⇄ miner agree)."""
    import hashlib
    from thirtyspokes.koth import rtmr
    d = rtmr.runtime_digest(runtime_measurement="rt-1", suite_version="s-1",
                            pool_allow_list=["b", "a"])
    assert d == rtmr.runtime_digest(runtime_measurement="rt-1", suite_version="s-1",
                                    pool_allow_list=["a", "b"])       # order-independent
    want = hashlib.sha384(bytes.fromhex(rtmr.RTMR3_ZERO) + d).hexdigest()
    assert rtmr.expected_rtmr3(d) == want
    assert rtmr.owner_expected_rtmr3(runtime_measurement="rt-1", suite_version="s-1",
                                     pool_allow_list=["a", "b"]) == want


def test_owner_governance_round_trips_on_chain():
    from thirtyspokes.koth.owner import build_record
    from thirtyspokes.subnet.chain import MockChain
    rec = build_record(mrtd=["aa" * 48], rtmr1="11" * 48, rtmr2="22" * 48,
                       pool_allow_list=["openai/gpt-4o-mini"])
    chain = MockChain()
    assert chain.owner_measurements() is None
    chain.publish_owner_measurements(rec)
    got = chain.owner_measurements()
    assert got["mrtd"] == ["aa" * 48] and got["rtmr"]["1"] == "11" * 48
    assert "UpToDate" in got["tcb_accept"]


def test_validator_governance_overrides_gate():
    """`_effective_governance` prefers the owner's on-chain record over constructor defaults."""
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import MockChain
    from thirtyspokes.koth.validator import KOTHValidator
    chain = MockChain()
    val = KOTHValidator({runtime_measurement()}, "pub", chain, Reign(), benchmarks.default_suite(),
                        store=None, backend=MockBackend(), approved_mrtd={"ff" * 48})
    _, mrtd0, rtmr0, tcb0, _pc0 = val._effective_governance()
    assert mrtd0 == {"ff" * 48} and rtmr0 is None            # constructor default
    chain.publish_owner_measurements({"version": 1, "runtime_measurements": [runtime_measurement()],
                                      "mrtd": ["aa" * 48], "rtmr": {"3": "cc" * 48},
                                      "tcb_accept": ["UpToDate"]})
    _, mrtd1, rtmr1, tcb1, _pc1 = val._effective_governance()
    assert mrtd1 == {"aa" * 48} and rtmr1 == {3: "cc" * 48} and tcb1 == frozenset({"UpToDate"})


# --- production fail-closed enforcement (the miner-CVM trust boundary, docs/DESIGN.md §8) -----------
def _as_tdx(proof):
    """Relabel a proof's quote as a hardware (tdx:) quote so we can exercise the enforce
    pre-crypto gates (mrtd/rtmr/tcb unset) without needing a real quote."""
    return replace(proof, quote=replace(proof.quote, platform_sig="tdx:AAAA"))


_FULL_RTMR = {1: "11" * 48, 2: "22" * 48, 3: "33" * 48}


def test_enforce_rejects_mock_quote(env):
    """A mock-TEE proof (public vendor key, no hardware root) is DQ'd under enforce even when
    the measurement gate is fully configured — mock == zero security."""
    a = _art(_ROUTER_SRC, env.allk)
    v = _verify(env, _proof(env, a, "hk"), a, "hk", enforce=True, approved_mrtd={"aa" * 48},
                approved_rtmr=_FULL_RTMR, tcb_accept=frozenset({"UpToDate"}))
    assert not v.valid and v.reason == "mock_quote_rejected"


def test_enforce_fails_closed_on_unset_gate(env):
    """With a real (tdx:) quote but the owner's measured-image gate unset, verification fails
    closed — it never silently skips the MRTD/RTMR/TCB checks and accepts a tampered runtime."""
    a = _art(_ROUTER_SRC, env.allk)
    p = _as_tdx(_proof(env, a, "hk"))
    assert _verify(env, p, a, "hk", enforce=True).reason == "mrtd_gate_unset"
    assert _verify(env, p, a, "hk", enforce=True,
                   approved_mrtd={"aa" * 48}).reason == "rtmr_gate_unset"
    assert _verify(env, p, a, "hk", enforce=True, approved_mrtd={"aa" * 48},
                   approved_rtmr={1: "11" * 48}).reason == "rtmr_gate_unset"   # need 1,2,3
    assert _verify(env, p, a, "hk", enforce=True, approved_mrtd={"aa" * 48},
                   approved_rtmr=_FULL_RTMR).reason == "tcb_policy_unset"


def test_nonenforcing_path_unchanged(env):
    """The offline sim / mock-TEE testnet path (enforce=False) still verifies a mock proof."""
    a = _art(_ROUTER_SRC, env.allk)
    assert _verify(env, _proof(env, a, "hk"), a, "hk").valid


def test_validator_governance_ready_preflight(env):
    """The daemon preflight refuses to start an enforcing validator until the owner has pinned
    a full measured-image set on-chain; a non-enforcing validator is always advisory-ready."""
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import MockChain
    from thirtyspokes.koth.validator import KOTHValidator
    chain = MockChain()
    val = KOTHValidator({runtime_measurement()}, "pub", chain, Reign(), env.suite,
                        store=None, backend=MockBackend(), enforce=True)
    assert val.governance_ready() == (False, "no_approved_mrtd")
    chain.publish_owner_measurements({"version": 1, "runtime_measurements": [runtime_measurement()],
                                      "mrtd": ["aa" * 48], "rtmr": {"1": "11" * 48, "2": "22" * 48,
                                      "3": "33" * 48}, "tcb_accept": ["UpToDate"]})
    assert val.governance_ready() == (True, "ok")
    lax = KOTHValidator({runtime_measurement()}, "pub", chain, Reign(), env.suite,
                        store=None, backend=MockBackend())        # enforce=False (sim default)
    assert lax.governance_ready()[0]


# --- S2: secret, owner-held, rotated memorization probe (koth/heldout.py, docs/DESIGN.md §6/§9) -----
def _bank(tasks_by_bench):
    from thirtyspokes.koth.heldout import SecretProbeBank
    from thirtyspokes.koth.benchmarks import BenchTask
    return SecretProbeBank({n: [BenchTask(*t) for t in ts] for n, ts in tasks_by_bench.items()})


def test_secret_probe_commit_is_stable_order_independent_and_rotates():
    t = [("m-1", "p1", 1.0), ("m-2", "p2", 2.0)]
    b1, b2 = _bank({"math": t}), _bank({"math": list(reversed(t))})
    assert b1.commit() == b2.commit()                              # order-independent
    b3 = _bank({"math": t + [("m-3", "p3", 3.0)]})
    assert b3.commit() != b1.commit()                             # rotation changes the commit
    from thirtyspokes.koth.heldout import SecretProbeBank
    assert SecretProbeBank.from_json(b1.to_json()).commit() == b1.commit()   # round-trip


def test_secret_probe_is_used_and_not_publicly_derivable(env):
    """When a secret bank is supplied, the validator's probe slice is drawn from it — not from
    the public dataset a miner could reproduce."""
    from thirtyspokes.subnet.chain import MockChain
    from thirtyspokes.reign import Reign
    from thirtyspokes.koth.validator import KOTHValidator
    bank = _bank({"math": [(f"secret-math-{i}", f"Compute {i}+{i}.", float(2 * i)) for i in range(12)]})
    val = KOTHValidator(env.approved, "pub", MockChain(), Reign(), env.suite, store=None,
                        backend=MockBackend(), probe_bank=bank, n_per_bench=8)
    probe = val._shared_probe("nonceXYZ", bank)
    math_ids = [t.task_id for b, t in probe if b.name == "math"]
    assert math_ids and all(i.startswith("secret-math") for i in math_ids)    # from the bank
    public = {t.task_id for t in next(b for b in env.suite if b.name == "math").probe(8, 123)}
    assert not (set(math_ids) & public)                          # a miner deriving the public probe misses


def test_governance_ready_requires_matching_probe_bank(env):
    from thirtyspokes.subnet.chain import MockChain
    from thirtyspokes.reign import Reign
    from thirtyspokes.koth.validator import KOTHValidator
    bank = _bank({"math": [("s-1", "p", 1.0)]})
    chain = MockChain()
    chain.publish_owner_measurements({"version": 1, "runtime_measurements": [runtime_measurement()],
        "mrtd": ["aa" * 48], "rtmr": {"1": "11" * 48, "2": "22" * 48, "3": "33" * 48},
        "tcb_accept": ["UpToDate"], "probe_commit": bank.commit()})
    mk = lambda pb: KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                                  backend=MockBackend(), enforce=True, probe_bank=pb, audit_mode="probe")
    assert mk(None).governance_ready() == (False, "probe_bank_missing")
    assert mk(_bank({"math": [("other", "p", 9.0)]})).governance_ready() == (False, "probe_bank_mismatch")
    assert mk(bank).governance_ready() == (True, "ok")


def test_governance_last_known_good(env):
    """F9: a vanished/failed on-chain governance read reuses the last-known-good record instead of
    silently dropping the gate to constructor defaults (which under enforce DQs the whole subnet)."""
    from thirtyspokes.reign import Reign
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.subnet.chain import MockChain
    chain = MockChain()
    chain.publish_owner_measurements({"version": 1, "runtime_measurements": [runtime_measurement()],
        "mrtd": ["aa" * 48], "rtmr": {"1": "11" * 48, "2": "22" * 48, "3": "33" * 48},
        "tcb_accept": ["UpToDate"]})
    val = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                        backend=MockBackend(), enforce=True)
    _a, m1, r1, t1, _ = val._effective_governance()
    assert m1 == {"aa" * 48} and set(r1) == {1, 2, 3}
    chain.owner_measurements = lambda: None                  # transient failure / record removed
    _a, m2, r2, t2, _ = val._effective_governance()
    assert (m2, r2, t2) == (m1, r1, t1)                      # reused, not dropped to None


def test_run_epoch_refuses_when_secret_probe_required_but_unverified(env):
    from thirtyspokes.subnet.chain import MockChain
    from thirtyspokes.reign import Reign
    from thirtyspokes.koth.validator import KOTHValidator
    bank = _bank({"math": [("s-1", "p", 1.0)]})
    chain = MockChain()
    chain.publish_owner_measurements({"version": 1, "runtime_measurements": [runtime_measurement()],
        "mrtd": ["aa" * 48], "rtmr": {"1": "11" * 48, "2": "22" * 48, "3": "33" * 48},
        "tcb_accept": ["UpToDate"], "probe_commit": bank.commit()})
    val = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                        backend=MockBackend(), enforce=True, probe_bank=None, audit_mode="probe")
    rep = val.run_epoch(lambda *a: None)
    assert rep.dq.get("*") == "probe_bank_unverified" and not rep.weights_by_uid


# --- S2b: cohort-relative memorization test (probe difficulty must not false-DQ honest miners) --
def _detector(env, *, min_cohort=3, max_probe_drop=0.25):
    from thirtyspokes.subnet.chain import MockChain
    from thirtyspokes.reign import Reign
    from thirtyspokes.koth.validator import KOTHValidator
    return KOTHValidator(env.approved, "pub", MockChain(), Reign(), env.suite, store=None,
                         backend=MockBackend(), min_cohort=min_cohort, max_probe_drop=max_probe_drop)


N = 64   # 4 benchmarks x 16 tasks — enough power that a 15-pt difficulty gap is "significant"


def test_absolute_test_false_dqs_honest_miner_on_a_harder_probe():
    """REGRESSION GUARD for the bug that motivated the cohort test: with allowance=0 a perfectly
    honest router is called a memorizer purely because the secret probe is harder."""
    from thirtyspokes.koth.verify import memorization_collapsed
    assert memorization_collapsed(0.95, N, 0.80, N)                # 15-pt difficulty gap -> "DQ"


def test_cohort_allowance_absolves_honest_cohort_on_a_hard_probe(env):
    """Everyone drops the same 23 pts (the probe is just hard) -> nobody is a memorizer."""
    audits = {f"hk{i}": (0.95, 0.72, N, N) for i in range(5)}
    assert _detector(env)._detect_memorizers(audits) == set()


def test_memorizer_still_caught_among_an_honest_cohort(env):
    """Honest cohort drops 23 pts; the memorizer collapses to chance -> only it is flagged."""
    audits = {f"honest{i}": (0.95, 0.72, N, N) for i in range(4)}
    audits["memorizer"] = (0.95, 0.25, N, N)
    assert _detector(env)._detect_memorizers(audits) == {"memorizer"}


def test_colluding_all_memorizer_cohort_cannot_inflate_the_allowance(env):
    """If EVERY miner memorized, the median drop is huge — the owner's max_probe_drop cap stops
    them hiding behind it, so all are still flagged."""
    audits = {f"m{i}": (0.95, 0.25, N, N) for i in range(5)}
    assert _detector(env, max_probe_drop=0.25)._detect_memorizers(audits) == set(audits)


def test_small_cohort_falls_back_to_owner_declared_allowance(env):
    """< min_cohort miners: no cohort to calibrate. Fall back to max_probe_drop, so an honest
    miner on a hard probe survives while a memorizer is still caught."""
    d = _detector(env, min_cohort=3, max_probe_drop=0.25)
    assert d._detect_memorizers({"honest": (0.95, 0.72, N, N)}) == set()          # 23-pt drop ok
    assert d._detect_memorizers({"memo": (0.95, 0.25, N, N)}) == {"memo"}         # 70-pt drop caught


def test_cohort_probe_allowance_median_cap_and_fallback():
    from thirtyspokes.koth.verify import cohort_probe_allowance as cpa
    assert cpa([0.1, 0.2, 0.9], min_cohort=3, max_drop=0.5) == 0.2                # median, not mean
    assert cpa([0.6, 0.7, 0.8], min_cohort=3, max_drop=0.25) == 0.25              # capped
    assert cpa([0.1], min_cohort=3, max_drop=0.25) == 0.25                        # small-cohort fallback
    assert cpa([-0.3, -0.2, -0.1], min_cohort=3, max_drop=0.25) == 0.0            # negative -> 0


# --- R: validator restart-safety (persist reign standings + king dethrone-guard baseline) ----
def test_reign_snapshot_restore_preserves_standings():
    """A restored reign continues identically to one that never restarted (same eps/age history)."""
    from thirtyspokes.reign import Reign, Submission
    def run(r, epoch_subs):
        for subs in epoch_subs:
            r.update(subs)
        return r
    seq = [[Submission("a", "a", 1, 0.9), Submission("b", "b", 2, 0.5)],
           [Submission("a", "a", 1, 0.9), Submission("b", "b", 2, 0.88)]]   # b challenges a
    cont = run(Reign(), seq)                                      # no restart
    r1 = Reign(); r1.update(seq[0])                              # restart between epochs
    r2 = Reign(); r2.restore(r1.snapshot()); r2.update(seq[1])
    assert [m.sub.miner_id for m in r2.members] == [m.sub.miner_id for m in cont.members]
    assert [m.age for m in r2.members] == [m.age for m in cont.members]


def test_validator_state_round_trip_keeps_king_baseline(env, tmp_path):
    from thirtyspokes.subnet.chain import MockChain
    from thirtyspokes.reign import Reign
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.koth.verify import BenchStat
    val = KOTHValidator(env.approved, "pub", MockChain(), Reign(), env.suite, store=None,
                        backend=MockBackend())
    val._king_id = "king-hk"
    val._king_vd = ProofVerdict(True, "ok", {"math": BenchStat(8, 0.9, 0.8, 0.01)},
                                total_cost_usd=0.05, score=0.42, total_score=0.44)
    path = str(tmp_path / "state.json")
    val.save_state(path)
    fresh = KOTHValidator(env.approved, "pub", MockChain(), Reign(), env.suite, store=None,
                          backend=MockBackend())
    assert fresh.load_state(path)
    assert fresh._king_id == "king-hk"
    assert fresh._king_vd.per_bench["math"].acc == 0.9 and fresh._king_vd.total_cost_usd == 0.05
    # a restored king still guards: a regressing challenger cannot dethrone it
    from thirtyspokes.koth.verify import dethrone_guard
    weak = ProofVerdict(True, "ok", {"math": BenchStat(8, 0.5, 0.4, 0.01)}, 0.05, 0.20, 0.20)
    ok, _ = dethrone_guard(weak, fresh._king_vd)
    assert not ok


def test_validator_state_round_trip_keeps_pending_chain_transaction(env, tmp_path):
    from thirtyspokes.koth.validator import EpochReport, KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import MockChain

    path = str(tmp_path / "state.json")
    val = KOTHValidator(env.approved, "pub", MockChain(), Reign(), env.suite, store=None,
                        backend=MockBackend())
    val._last_scored_epoch = 42
    val._pending_weights = {4: 1.0}
    val._pending_report = EpochReport(42, {"miner": 0.5}, {}, ["miner"], {4: 1.0}, {})
    val.save_state(path)

    fresh = KOTHValidator(env.approved, "pub", MockChain(), Reign(), env.suite, store=None,
                          backend=MockBackend())
    assert fresh.load_state(path)
    assert fresh.last_scored_epoch == 42
    assert fresh.pending_weights == {4: 1.0}
    assert fresh.pending_report.epoch == 42
    assert fresh.pending_report.weights_by_uid == {4: 1.0}


def test_validator_neuron_replays_pending_weights_without_rescoring(env, tmp_path):
    from thirtyspokes.koth.neuron import KOTHValidatorNeuron
    from thirtyspokes.koth.validator import EpochReport, KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import MockChain

    path = str(tmp_path / "state.json")
    chain = MockChain()
    val = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                        backend=MockBackend())
    val._last_scored_epoch = 17
    val._pending_weights = {3: 0.25, 4: 0.75}
    val._pending_report = EpochReport(17, {}, {}, [], {3: 0.25, 4: 0.75}, {})
    val.save_state(path)

    fresh = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                          backend=MockBackend())
    neuron = KOTHValidatorNeuron(fresh, store=object(), state_path=path)
    assert neuron._flush_pending()
    assert chain._weights == {3: 0.25, 4: 0.75}
    assert fresh.last_scored_epoch == 17
    assert fresh.pending_weights is None
    assert fresh.last_submitted_weights == {3: 0.25, 4: 0.75}

    again = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                          backend=MockBackend())
    assert again.load_state(path)
    assert again.last_scored_epoch == 17
    assert again.pending_weights is None


def test_validator_neuron_skips_a_persisted_identical_distribution(env, tmp_path):
    from thirtyspokes.koth.neuron import KOTHValidatorNeuron
    from thirtyspokes.koth.validator import EpochReport, KOTHValidator
    from thirtyspokes.reign import Reign
    from thirtyspokes.subnet.chain import MockChain

    path = str(tmp_path / "state.json")
    chain = MockChain()
    val = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                        backend=MockBackend())
    val._last_scored_epoch = 18
    val._last_submitted_weights = {3: 0.5, 4: 0.5}
    val._pending_weights = {3: 0.5, 4: 0.5}
    val._pending_report = EpochReport(18, {}, {}, [], {3: 0.5, 4: 0.5}, {})
    val.save_state(path)

    def must_not_submit(_weights):
        raise AssertionError("an unchanged distribution must not reset the chain rate-limit clock")

    chain.set_weights = must_not_submit
    fresh = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                          backend=MockBackend())
    neuron = KOTHValidatorNeuron(fresh, store=object(), state_path=path)
    assert neuron._flush_pending()
    assert fresh.pending_weights is None
    assert fresh.pending_report is None
    assert fresh.last_submitted_weights == {3: 0.5, 4: 0.5}


def test_validator_neuron_stop_wakes_a_long_poll():
    import threading

    from thirtyspokes.koth.neuron import KOTHValidatorNeuron

    class IdleValidator:
        pending_weights = None
        pending_report = None
        last_scored_epoch = 7

        @staticmethod
        def _settle_epoch():
            return 7

    neuron = KOTHValidatorNeuron(IdleValidator(), store=object())
    worker = threading.Thread(target=neuron.run_forever, kwargs={"poll_s": 60})
    worker.start()
    neuron.request_stop()
    worker.join(timeout=1)
    assert not worker.is_alive(), "SIGTERM stop requests must wake the daemon's poll immediately"


def test_standings_rpc_is_bounded_and_restores_chain_settings(env):
    from types import SimpleNamespace

    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign

    substrate = SimpleNamespace(retry_timeout=60.0, max_retries=5)

    class Chain:
        subtensor = SimpleNamespace(substrate=substrate)

        def hotkeys(self):
            assert substrate.retry_timeout == 10.0
            assert substrate.max_retries == 1
            raise TimeoutError("public RPC stalled")

    val = KOTHValidator(env.approved, "pub", Chain(), Reign(), env.suite, store=object(),
                        backend=MockBackend())
    assert val.flush_standings("repo") is False
    assert substrate.retry_timeout == 60.0
    assert substrate.max_retries == 5


def test_validator_neuron_rolls_back_a_failed_epoch_before_retry(env):
    from thirtyspokes.koth.neuron import KOTHValidatorNeuron
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import Reign, Submission
    from thirtyspokes.subnet.chain import MockChain

    chain = MockChain()
    chain.reconnect = lambda: setattr(chain, "reconnected", True)
    val = KOTHValidator(env.approved, "pub", chain, Reign(), env.suite, store=None,
                        backend=MockBackend())
    before = val.snapshot()

    def fail_after_mutation(*_args, **_kwargs):
        val.reign.update([Submission("miner", "miner", 1, 0.5)])
        val._history.append({"epoch": 99})
        val._king_id = "miner"
        raise TimeoutError("public chain RPC stalled")

    val.run_epoch = fail_after_mutation
    neuron = KOTHValidatorNeuron(val, store=object())
    assert neuron._stage_epoch(99) is False
    assert val.snapshot() == before
    assert chain.reconnected is True


# --- H4: chain hardening (SDK-shape handling + governance on the file-backed chain) ----------
def test_bittensor_latest_commitment_shape():
    """bittensor 10.x `get_all_revealed_commitments` returns {hotkey: ((block,data),…)}; we take
    the latest — this handles that shape without a live chain (regression for the SDK drift)."""
    from thirtyspokes.subnet.chain import BittensorChain
    assert BittensorChain._latest(((5, "a"), (9, "b"), (3, "c"))) == (9, "b")
    assert BittensorChain._latest(()) is None
    assert BittensorChain._latest(None) is None


def test_bittensor_hotkeys_uses_the_minimal_keys_storage_map():
    from types import SimpleNamespace

    from thirtyspokes.subnet.chain import BittensorChain

    calls = []

    class Substrate:
        def query_map(self, **kwargs):
            calls.append(kwargs)
            return [(0, SimpleNamespace(value="burn")), (3, "miner")]

    chain = object.__new__(BittensorChain)
    chain.netuid = 526
    chain.subtensor = SimpleNamespace(substrate=Substrate())
    assert chain.hotkeys() == {0: "burn", 3: "miner"}
    assert calls == [{"module": "SubtensorModule", "storage_function": "Keys", "params": [526]}]


def test_bittensor_close_releases_the_sdk_connection():
    from types import SimpleNamespace

    from thirtyspokes.subnet.chain import BittensorChain

    closed = []
    chain = object.__new__(BittensorChain)
    chain.subtensor = SimpleNamespace(close=lambda: closed.append(True))
    chain.close()
    assert closed == [True]


def test_bittensor_weight_write_is_inclusion_only_and_bounded():
    from types import SimpleNamespace

    from thirtyspokes.subnet.chain import BittensorChain

    class Subtensor:
        def __init__(self):
            self.substrate = SimpleNamespace(retry_timeout=60.0, max_retries=5)
            self.kwargs = None

        def get_uid_for_hotkey_on_subnet(self, _hotkey, _netuid):
            return 7

        def weights(self, _netuid):
            return [(7, [(3, 65535)])]

        def blocks_since_last_update(self, _netuid, _uid):
            return 101

        def weights_rate_limit(self, _netuid):
            return 100

        def set_weights(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(success=True, message="included")

    chain = object.__new__(BittensorChain)
    chain.netuid = 526
    chain.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
    chain.subtensor = Subtensor()
    chain.set_weights({4: 1.0})

    assert chain.subtensor.kwargs["wait_for_inclusion"] is True
    assert chain.subtensor.kwargs["wait_for_finalization"] is False
    assert chain.subtensor.kwargs["wait_for_revealed_execution"] is False
    assert chain.subtensor.kwargs["max_attempts"] == 2
    assert chain.subtensor.substrate.retry_timeout == 60.0
    assert chain.subtensor.substrate.max_retries == 5


def test_bittensor_weight_write_reports_the_rate_limit_before_submission():
    from types import SimpleNamespace

    from thirtyspokes.subnet.chain import BittensorChain, WeightRateLimited

    class Subtensor:
        substrate = SimpleNamespace(retry_timeout=60.0, max_retries=5)

        def get_uid_for_hotkey_on_subnet(self, _hotkey, _netuid):
            return 7

        def weights(self, _netuid):
            return [(7, [(3, 65535)])]

        def blocks_since_last_update(self, _netuid, _uid):
            return 62

        def weights_rate_limit(self, _netuid):
            return 100

        def set_weights(self, **_kwargs):
            raise AssertionError("rate-limited weights must not be submitted")

    chain = object.__new__(BittensorChain)
    chain.netuid = 526
    chain.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
    chain.subtensor = Subtensor()
    with pytest.raises(WeightRateLimited, match="39 more block"):
        chain.set_weights({4: 0.5, 3: 0.5})


def test_bittensor_weight_replay_skips_an_identical_onchain_distribution():
    from types import SimpleNamespace

    from thirtyspokes.subnet.chain import BittensorChain

    class Subtensor:
        substrate = SimpleNamespace(retry_timeout=60.0, max_retries=5)

        def get_uid_for_hotkey_on_subnet(self, _hotkey, _netuid):
            return 7

        def weights(self, _netuid):
            return [(7, [(3, 16384), (4, 49151)])]

        def set_weights(self, **_kwargs):
            raise AssertionError("an idempotent replay must not submit another extrinsic")

    chain = object.__new__(BittensorChain)
    chain.netuid = 526
    chain.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
    chain.subtensor = Subtensor()
    chain.set_weights({3: 0.25, 4: 0.75})


def test_localfilechain_governance_and_register(tmp_path):
    from thirtyspokes.subnet.chain import LocalFileChain
    ch = LocalFileChain(str(tmp_path))
    assert ch.owner_measurements() is None
    ch.register("hk1")
    ch.publish_owner_measurements({"version": 2, "mrtd": ["bb" * 48]})
    assert ch.owner_measurements()["mrtd"] == ["bb" * 48]
    assert "hk1" in ch.hotkeys().values()


# --- F7 must never destroy the artifact commit (regression: testnet 526) ----------------------
class SingleSlotChain:
    """Models the REAL chain, which MockChain does not: one commitment slot per hotkey.

    `commit()` (set_reveal_commitment) parks a TIMELOCKED payload in that slot; it only lands in the
    separate append-only revealed map after `reveal_delay` blocks. `commit_proof()` (set_commitment)
    writes the SAME slot, so issuing one while the artifact commit is still timelocked wipes it and
    the artifact NEVER reveals. Observed live on testnet 526."""

    def __init__(self, reveal_delay: int = 6):
        from thirtyspokes.subnet.chain import Commitment
        self._C = Commitment
        self.reveal_delay = reveal_delay
        self.block = 0
        self.slot = None                 # ("timelocked"|"plain", data, block)
        self.revealed: list = []
        self.proof_commits: dict = {}

    def advance(self, n=1):
        self.block += n
        if self.slot and self.slot[0] == "timelocked" and self.block - self.slot[2] >= self.reveal_delay:
            self.revealed.append(self._C("hk", 1, self.slot[1], self.block))
            self.slot = None

    def register(self, hotkey): pass
    def current_block(self): return self.block
    def commit(self, hotkey, data): self.slot = ("timelocked", data, self.block)
    def commit_proof(self, hotkey, epoch, digest):
        self.slot = ("plain", f"{epoch}|{digest}", self.block)      # CLOBBERS a pending artifact commit
        self.proof_commits[(hotkey, epoch)] = (digest, self.block)
    def proof_commit(self, hotkey, epoch): return self.proof_commits.get((hotkey, epoch))
    def revealed_commitments(self): return list(self.revealed)
    def hotkeys(self): return {1: "hk"}
    def set_weights(self, w): pass
    def beacon(self, epoch): return f"beacon-{epoch}"


def _miner_on(chain, *, commit_proofs):
    import tempfile
    from thirtyspokes.koth.benchmarks import default_suite
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.tee.attestation import Platform
    import json
    relay = ("import json\n"
             "def build_agent(w):\n"
             "    m = json.loads(w.decode())['model']\n"
             "    def agent(prompt, call_model):\n"
             "        return call_model(m, [{'role': 'user', 'content': prompt}], {'max_tokens': 16})\n"
             "    return agent\n")
    art = Artifact(relay, json.dumps({"model": "strong"}).encode(), "strong")
    return KOTHMinerNeuron("hk", MockPool(), Platform(), default_suite(), chain,
                           LocalBundleStore(tempfile.mkdtemp()), art,
                           "hk/repo", n_per_bench=2, commit_proofs=commit_proofs)


def test_proof_commit_never_clobbers_a_pending_artifact_commit():
    """The bug: committing a proof while the artifact commit is timelocked wipes it, so it never
    reveals, so validators bind no artifact and score nobody. The miner must hold proof commits
    back until the artifact has revealed."""
    chain = SingleSlotChain(reveal_delay=6)
    m = _miner_on(chain, commit_proofs=True)
    m.publish()                                   # artifact commit -> timelocked slot
    assert chain.slot[0] == "timelocked"

    m.run_once(epoch=1)                           # would previously clobber it
    assert chain.slot is not None and chain.slot[0] == "timelocked", "artifact commit was destroyed"
    assert chain.proof_commit("hk", 1) is None    # proof commit correctly held back

    for _ in range(6):                            # let the timelock elapse
        chain.advance()
    assert len(chain.revealed) == 1, "artifact commit never revealed"

    m.run_once(epoch=2)                           # now safe: revealed record is in the other map
    assert chain.proof_commit("hk", 2) is not None
    assert len(chain.revealed) == 1               # and the revealed record survives


def test_commit_proofs_is_off_by_default():
    """The F7 write is useless unless validators run --commit-window, and harmful otherwise."""
    chain = SingleSlotChain()
    m = _miner_on(chain, commit_proofs=False)
    m.publish()
    m.run_once(epoch=1)
    assert chain.proof_commit("hk", 1) is None
    assert chain.slot[0] == "timelocked"          # artifact commit intact


# --- the public leaderboard feed (standings.json) ------------------------------------------------
def test_standings_payload_is_real_and_flush_never_raises():
    """The dashboard feed must carry REAL scored data, and publishing it must never be able to
    break scoring (a store outage is presentational, not a scoring failure)."""
    import json
    import tempfile

    from thirtyspokes.gateway.signing import Signer
    from thirtyspokes.koth.benchmarks import default_suite
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.neuron import store_get_proof
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import Artifact, runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import KingChain
    from thirtyspokes.subnet.chain import LocalFileChain
    from thirtyspokes.tee.attestation import Platform

    root = tempfile.mkdtemp(prefix="koth_standings_")
    backend, platform = MockPool(), Platform()
    chain, store = LocalFileChain(f"{root}/chain"), LocalBundleStore(f"{root}/store")
    chain.register("burn")
    suite = default_suite()
    relay = ("import json\n"
             "def build_agent(w):\n"
             "    m = json.loads(w.decode())['model']\n"
             "    def agent(p, call_model):\n"
             "        return call_model(m, [{'role': 'user', 'content': p}], {'max_tokens': 16})\n"
             "    return agent\n")
    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, KingChain(),
                        suite, store, backend, n_per_bench=8, budget=0.5, f_min=0.1)
    miners = [KOTHMinerNeuron(Signer().public_hex, backend, platform, suite, chain, store,
                              Artifact(relay, json.dumps({"model": m}).encode(), m), f"{m}/repo")
              for m in ("strong", "cheap")]
    for m in miners:
        m.publish()
    chain.advance(EPOCH_BLOCKS)
    for m in miners:
        m.run_once(current_epoch(chain))
    rep = val.run_epoch(store_get_proof(store))

    st = val.standings(rep)
    assert st["schema"] == 1
    assert st["network"] is None and st["netuid"] is None  # local chain has no false provenance
    assert st["mechanism"]["kind"] == "king+equal_share_chain"
    assert st["king"] is not None and st["king"]["king"] is True
    assert st["king"]["q_lcb"] > 0
    assert st["king"]["per_bench"], "the king's per-benchmark accuracy must be published"
    assert st["scored"], "scored miners must be published"
    # equal-share: every paid miner gets the same weight, and it is the mechanism's own number
    assert st["mechanism"]["weight_each"] > 0

    # publish + read back through the store
    assert val.flush_standings("val/repo", rep) is True
    assert json.loads(store.download_standings("val/repo"))["king"]["hotkey"] == st["king"]["hotkey"]

    # a broken store must NOT raise into the scoring loop — it must just report False
    class Broken:
        def upload_standings(self, *_a, **_k):
            raise OSError("store is down")
    val.store = Broken()
    assert val.flush_standings("val/repo", rep) is False


def test_validator_prefers_owner_hf_token_for_official_feed():
    from thirtyspokes.koth.neuron import validator_hf_token

    assert validator_hf_token({"OWNER_HF_TOKEN": "owner", "HF_TOKEN": "generic"}) == "owner"
    assert validator_hf_token({"HF_TOKEN": "generic"}) == "generic"
    assert validator_hf_token({}) is None


def test_hf_standings_publish_repairs_existing_private_repo():
    """Anonymous dashboard reads require public visibility, including for a pre-existing repo."""
    from thirtyspokes.koth.store import HFBundleStore

    calls = []

    class Api:
        def create_repo(self, *args, **kwargs):
            calls.append(("create", args, kwargs))

        def update_repo_settings(self, *args, **kwargs):
            calls.append(("public", args, kwargs))

    store = object.__new__(HFBundleStore)
    store.api = Api()
    store.token = "test-token"
    store._upload_json = lambda repo, path, payload: calls.append(
        ("upload", (repo, path, payload), {}))

    store.upload_standings("thirtyspokes/standings", '{"schema": 1}')

    assert [call[0] for call in calls] == ["create", "public", "upload"]
    assert calls[1][2]["private"] is False


def test_dethrone_guard_lets_quality_beat_a_cheap_weak_king():
    """A far better agent must be able to take a cheap king's crown even though it costs more.

    The guard used to veto any challenger costing >10% more than the king, regardless of quality —
    a one-way ratchet that made a cheap-but-weak first mover permanently unbeatable (seen live: a
    challenger at Q_lcb 0.999 clamped under a king at 0.282, forever). Cost is bounded by the
    owner's absolute budget (`eligible`), not by a ceiling that only ever moves down.
    """
    from thirtyspokes.koth.verify import BenchStat, ProofVerdict, dethrone_guard

    def vd(acc, cost):
        bs = BenchStat(n=8, acc=acc, lcb=acc - 0.02, cost_usd=cost)
        return ProofVerdict(True, "ok", {"math": bs, "mmlu": bs}, total_cost_usd=cost, score=acc)

    cheap_king = vd(0.30, 0.001)          # weak but nearly free — took a vacant crown first
    strong_chal = vd(0.99, 0.40)          # far better, 400x pricier, still inside a 0.5 budget
    ok, why = dethrone_guard(strong_chal, cheap_king)
    assert ok, f"quality-dominant challenger must be able to dethrone a cheap king, got {why}"

    # a challenger that is NOT better, only pricier, still cannot buy the crown
    ok, why = dethrone_guard(vd(0.30, 0.40), cheap_king)
    assert not ok and why in ("no_confident_gain", "cost_regression")

    # and a regression on any benchmark still blocks, however cheap
    worse = ProofVerdict(True, "ok",
                         {"math": BenchStat(8, 0.99, 0.97, 0.0), "mmlu": BenchStat(8, 0.10, 0.08, 0.0)},
                         total_cost_usd=0.0001, score=0.5)
    ok, why = dethrone_guard(worse, cheap_king)
    assert not ok and why.startswith("regression:")


def test_grounding_validator_needs_no_llm_and_never_infers():
    """A grounding-mode validator does NO inference, so it needs no LLM key and no GPU — it just
    verifies proofs. The daemon used to demand OPENROUTER_API_KEY unconditionally, forcing every
    validator to hold a billing credential for a path that is off by default. Scoring a full epoch
    with a backend that RAISES on any call proves the guarantee end-to-end."""
    import json
    import tempfile

    import pytest

    from thirtyspokes.gateway.signing import Signer
    from thirtyspokes.koth.benchmarks import default_suite
    from thirtyspokes.koth.epoch import EPOCH_BLOCKS, current_epoch
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.neuron import NoInferenceBackend, store_get_proof
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import Artifact, runtime_measurement
    from thirtyspokes.koth.store import LocalBundleStore
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.reign import KingChain
    from thirtyspokes.subnet.chain import LocalFileChain
    from thirtyspokes.tee.attestation import Platform

    root = tempfile.mkdtemp(prefix="koth_noinf_")
    platform = Platform()
    chain, store = LocalFileChain(f"{root}/chain"), LocalBundleStore(f"{root}/store")
    chain.register("burn")
    suite = default_suite()
    relay = ("import json\n"
             "def build_agent(w):\n"
             "    m = json.loads(w.decode())['model']\n"
             "    def agent(p, call_model):\n"
             "        return call_model(m, [{'role': 'user', 'content': p}], {'max_tokens': 16})\n"
             "    return agent\n")

    # the validator's backend raises on ANY call; the miners have a real (mock) pool
    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, KingChain(),
                        suite, store, NoInferenceBackend(), n_per_bench=8, budget=0.5, f_min=0.1,
                        audit_mode="grounding")
    miners = [KOTHMinerNeuron(Signer().public_hex, MockPool(), platform, suite, chain, store,
                              Artifact(relay, json.dumps({"model": m}).encode(), m), f"{m}/repo")
              for m in ("strong", "cheap")]
    for m in miners:
        m.publish()
    chain.advance(EPOCH_BLOCKS)
    for m in miners:
        m.run_once(current_epoch(chain))

    rep = val.run_epoch(store_get_proof(store))     # must NOT raise -> no inference happened
    assert rep.scored, "miners must still be scored without the validator ever calling a model"
    assert not rep.dq, f"no miner should be DQ'd here: {rep.dq}"

    # and the guard really does bite if something tries to infer
    with pytest.raises(RuntimeError, match="NO inference"):
        NoInferenceBackend().complete("any", [], {})


def test_commit_pins_the_exact_published_bytes_and_revision():
    """The essential guarantee: what the enclave RAN == what the miner PUBLISHED == what is committed.

    * the commit carries the store's IMMUTABLE revision (an HF commit SHA in production), not the
      literal "rev1" it used to hardcode and the validator then ignored;
    * the validator downloads AT that revision and recomputes the hashes itself;
    * swapping the artifact after committing does NOT let you keep the crown — the recomputed hashes
      no longer satisfy the on-chain commit.
    """
    import json
    import tempfile

    from thirtyspokes.gateway.signing import Signer
    from thirtyspokes.koth import commit as commitmod
    from thirtyspokes.koth.benchmarks import default_suite
    from thirtyspokes.koth.miner import KOTHMinerNeuron
    from thirtyspokes.koth.pool import MockPool
    from thirtyspokes.koth.runtime import Artifact
    from thirtyspokes.koth.store import LocalBundleStore, hash_source, hash_weights
    from thirtyspokes.subnet.chain import LocalFileChain
    from thirtyspokes.tee.attestation import Platform

    root = tempfile.mkdtemp(prefix="koth_bind_")
    chain, store = LocalFileChain(f"{root}/chain"), LocalBundleStore(f"{root}/store")
    src = "def build_agent(w):\n    return lambda p, c: 'x'\n"
    art = Artifact(src, b'{"model": "cheap"}', "cheap")
    m = KOTHMinerNeuron(Signer().public_hex, MockPool(), Platform(), default_suite(),
                        chain, store, art, "me/repo")
    m.publish()

    c = [x for x in chain.revealed_commitments() if x.hotkey == m.hotkey][0]
    repo, revision, _salt = commitmod.parse_commit(c.data)
    assert repo == "me/repo"
    assert revision not in ("rev1", "", None), "the commit must pin a real revision, not a literal"

    # the validator's path: download AT the committed revision, recompute, check the commit
    got = store.download(repo, revision)
    sh, wh = hash_source(got.source_text), hash_weights(got.weights)
    assert (sh, wh) == (art.source_hash, art.weights_hash)
    assert commitmod.verify_commit(c.data, m.hotkey, sh, wh), "published bytes must satisfy the commit"

    # now the miner swaps the artifact in its repo but keeps the old on-chain commit
    evil = Artifact("def build_agent(w):\n    return lambda p, c: 'CHEAT'\n", b'{"model": "cheap"}', "cheap")
    store.upload(repo, evil)
    swapped = store.download(repo, revision)
    sh2, wh2 = hash_source(swapped.source_text), hash_weights(swapped.weights)
    assert not commitmod.verify_commit(c.data, m.hotkey, sh2, wh2), \
        "a swapped artifact must NOT satisfy the original commit -> bad_commit"


def test_branch_names_are_never_accepted_as_a_revision():
    """A branch is mutable, so it must never appear as an on-chain revision — not as a fallback, not
    as a default. Rejected at BOTH ends: the store refuses to download one, and the validator DQs a
    commit that pins one (`unpinned_revision`) rather than quietly fetching a moving head."""
    import tempfile

    import pytest

    from thirtyspokes.koth.store import LocalBundleStore, is_pinned_revision

    for bad in ("main", "master", "HEAD", "refs/heads/main", "v1", "rev1", "", None, "  "):
        assert not is_pinned_revision(bad), f"{bad!r} must not count as pinned"
    for good in ("a" * 40, "0123456789abcdef0123456789abcdef01234567", "abc1234"):
        assert is_pinned_revision(good)

    store = LocalBundleStore(tempfile.mkdtemp())
    with pytest.raises(ValueError, match="unpinned revision"):
        store.download("me/repo", "main")


# --- HuggingFace: where the owner publishes the measured runtime image -------------------------
def test_runtime_image_naming_and_public_urls():
    """Our own bucket + paths, and a URL a miner can curl with no token (verified live)."""
    from thirtyspokes.koth import imagestore

    assert imagestore.DEFAULT_BUCKET == "thirtyspokes/cvm-runtime-image"
    assert imagestore.image_path("v14") == "runtime/v14/koth-runtime.tar.gz"
    assert imagestore.manifest_path("v14") == "runtime/v14/manifest.json"
    assert imagestore.LATEST_PATH == "runtime/latest.json"
    assert imagestore.bucket_uri() == "hf://buckets/thirtyspokes/cvm-runtime-image"
    assert imagestore.public_url(imagestore.image_path("v14")) == (
        "https://huggingface.co/buckets/thirtyspokes/cvm-runtime-image/resolve/"
        "runtime/v14/koth-runtime.tar.gz")


def test_runtime_manifest_carries_what_a_miner_needs_to_verify():
    """The manifest is a convenience, not a trust anchor — but it must carry everything needed to
    rebuild the image and to check it against the validator's on-chain gate."""
    import sys
    import tempfile

    sys.path.insert(0, "scripts")
    from publish_runtime_image import build_manifest

    with tempfile.NamedTemporaryFile("wb", suffix=".tar.gz", delete=False) as f:
        f.write(b"not-a-real-image")
        img = f.name
    man = build_manifest(version="v14", image_path=img,
                         uki_sha256="aa" * 32, roothash="bb" * 32,
                         mrtd="cc" * 48, rtmr1="dd" * 48, rtmr2="00" * 48,
                         pool=["openai/gpt-4o", "meta-llama/llama-3.1-8b-instruct"])
    assert man["image"]["sha256"]                     # the bytes are hashed
    assert man["bucket"] == "thirtyspokes/cvm-runtime-image"
    assert man["image"]["url"].startswith("https://huggingface.co/buckets/thirtyspokes/")
    # reproducibility: the miner can check out THIS recipe commit and rebuild
    assert man["reproducible"]["recipe"] == "scripts/build_koth_image_prod.sh"
    assert man["reproducible"]["uki_sha256"] == "aa" * 32
    # the gate the validator actually applies
    assert man["measurements"]["rtmr1"] == "dd" * 48
    assert man["pool_allow_list"] == ["meta-llama/llama-3.1-8b-instruct", "openai/gpt-4o"]


# --- governance: hash on-chain, record in the bucket --------------------------------------------
def test_governance_commit_fits_a_plain_immediate_commitment():
    """The whole point of hash-committing: the record is ~657 bytes and a plain commitment's Raw
    field caps at 128 ("Value 'Raw657' not present in type_mapping" — the chain said so). The hash
    fits, so governance becomes IMMEDIATELY visible instead of timelocked for ~72 minutes."""
    from thirtyspokes.koth import governance

    rec = {"version": 1, "mrtd": ["9b" * 48], "rtmr": {"1": "6d" * 48, "2": "00" * 48, "3": "d2" * 48},
           "tcb_accept": ["UpToDate"], "pool_allow_list": ["openai/gpt-4o"],
           "runtime_measurements": ["5c" * 32]}
    assert len(governance.canonical(rec)) > 128, "the record itself does NOT fit on-chain"

    d = governance.digest(rec)
    commit = governance.commit_string(d)
    assert len(commit) == 73 <= 128, f"the commit must fit a plain Raw field, got {len(commit)}"
    assert governance.parse_commit(commit) == d
    assert governance.parse_commit("not-a-governance-commit") is None


def test_governance_record_is_content_addressed_and_tamper_evident():
    """The bucket needs no trust: the record's URL is derived from its own hash, and a modified
    record fails verification. The chain is the trust root; the bucket is transport."""
    import pytest

    from thirtyspokes.koth import governance

    rec = {"version": 1, "rtmr": {"1": "6d" * 48}, "pool_allow_list": ["openai/gpt-4o"]}
    d = governance.digest(rec)
    raw = governance.canonical(rec)

    assert governance.record_path(d) == f"governance/{d}.json"     # addressed BY its hash
    assert governance.verify(raw, d) == rec                        # honest record round-trips

    # an attacker swaps the approved RTMR1 to bless their own tampered image
    evil = dict(rec, rtmr={"1": "ff" * 48})
    with pytest.raises(ValueError, match="hash mismatch"):
        governance.verify(governance.canonical(evil), d)

    # and the digest is stable regardless of key order / whitespace
    assert governance.digest({"pool_allow_list": ["openai/gpt-4o"], "rtmr": {"1": "6d" * 48},
                              "version": 1}) == d


# --- CRITICAL-1: the enclave's pool connection must not be steerable by the miner ----------------
# The measured runtime takes its pool credential from a blob the MINER supplies (GCE metadata
# `koth-secrets`). It used to be splatted wholesale into os.environ, and the HTTP client reads its
# TLS trust store and proxy routing FROM the environment. Two extra metadata lines therefore let a
# miner MITM its own pool: fabricate answers (inject memorised gold), fabricate `usage.cost` and
# token counts (cost is read out of the response body), spend nothing — and still emit a fully
# valid attested proof, because the enclave faithfully reports what it was told.

def test_secrets_blob_is_allowlisted():
    """Only the pool credential may cross from the miner-supplied blob into the environment."""
    import tempfile

    from thirtyspokes.eval import config
    blob = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    blob.write(
        "OPENROUTER_API_KEY=sk-or-real\n"
        "SSL_CERT_FILE=/run/koth/agent/weights.bin\n"      # CA smuggled in as 'weights'
        "HTTPS_PROXY=http://127.0.0.1:8080\n"
        "export REQUESTS_CA_BUNDLE=/tmp/evil.pem\n"
        "OPENAI_BASE_URL=http://127.0.0.1:9/v1\n"
        "# a comment\n")
    blob.close()
    env: dict = {}
    accepted, rejected = config.load_secrets_env(blob.name, env=env)
    assert accepted == ["OPENROUTER_API_KEY"]
    assert env == {"OPENROUTER_API_KEY": "sk-or-real"}, "nothing else may reach the environment"
    assert set(rejected) == {"SSL_CERT_FILE", "HTTPS_PROXY", "REQUESTS_CA_BUNDLE", "OPENAI_BASE_URL"}


def test_scrub_network_env_clears_every_tls_and_proxy_override():
    from thirtyspokes.eval import config
    env = {k: "x" for k in config.UNSAFE_NET_ENV} | {"OPENROUTER_API_KEY": "keep", "PATH": "keep"}
    dropped = config.scrub_network_env(env)
    assert set(dropped) == set(config.UNSAFE_NET_ENV)
    assert env == {"OPENROUTER_API_KEY": "keep", "PATH": "keep"}


def test_pool_client_ignores_env_tls_and_proxy_overrides(monkeypatch, tmp_path):
    """The backstop: even with the attack env set, the client must keep the real trust anchor.

    This is the PoC that proved the hole, inverted into a regression test — it fails loudly if
    anyone reconstructs the client without `trust_env=False`.
    """
    import certifi
    from thirtyspokes.gateway.gateway import OpenRouterBackend

    evil_ca = tmp_path / "evil_ca.pem"          # a CA the miner controls
    evil_ca.write_text(open(certifi.where()).read().split("-----END CERTIFICATE-----")[0]
                       + "-----END CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(evil_ca))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:8080")

    be = OpenRouterBackend("sk-test")
    client = be._client._client
    assert client.trust_env is False, "env-derived TLS/proxy must be off"
    assert not [m for m in client._mounts.values() if m is not None], "no proxy may be mounted"
    n_roots = len(client._transport._pool._ssl_context.get_ca_certs())
    assert n_roots > 20, f"trust store was replaced by the miner's CA (only {n_roots} roots)"


# --- CRITICAL-3: the no-egress guarantee must fail CLOSED and be ATTESTED -----------------------
# `run_agent_confined` silently degraded to a plain subprocess when the namespace probe failed
# (which it did on ANY exception, including its own probe timeout, then lru_cached that for the
# boot). An unconfined agent has network egress: it can call an off-allow-list model with a key
# embedded in its own weights.bin, voiding PinnedBackend + the MeteringProxy + the budget ceiling,
# while one token call through call_model satisfies `no_pool_call`. Nothing recorded the downgrade,
# so a fully-enforcing validator could not tell the two runs apart.

def test_confinement_probe_does_not_cache_a_timeout(monkeypatch):
    """A slow probe is inconclusive, not a 'no'. Caching it disabled confinement for the whole
    boot — a load-sensitive trigger on a machine whose size the miner chooses."""
    import subprocess as sp

    from thirtyspokes.koth import confine
    monkeypatch.setattr(confine, "_probe_cache", None)
    monkeypatch.setattr(confine.shutil, "which", lambda _: "/usr/bin/unshare")
    monkeypatch.setattr(confine.sys, "platform", "linux")
    calls = {"n": 0}

    def always_timeout(*a, **kw):
        calls["n"] += 1
        raise sp.TimeoutExpired(cmd="unshare", timeout=30)

    monkeypatch.setattr(confine.subprocess, "run", always_timeout)
    assert confine.confinement_available() is False
    assert calls["n"] == 3, "a timeout must be retried, not taken as a verdict"
    assert confine._probe_cache is None, "an inconclusive probe must NOT be cached"


def test_require_confinement_refuses_to_run_unconfined():
    """Production fails closed instead of silently handing the agent a network."""
    from thirtyspokes.koth.confine import SandboxError, run_agent_confined
    from thirtyspokes.koth.pool import MockPool
    with pytest.raises(SandboxError, match="refusing to run the agent unconfined"):
        run_agent_confined(_CALLS_POOL, b"{}", _TASK, backend=MockPool(),
                           hardened=False, require=True)


def test_proof_attests_the_confinement_mode(env):
    """The proof records what ACTUALLY happened, and the quote covers it."""
    import dataclasses

    from thirtyspokes.koth.proof import Proof
    a = _art(_ROUTER_SRC, env.allk)
    p = _proof(env, a, "hk")                     # in-process runtime -> not confined
    assert p.confined is False
    assert "confined" in p._payload(), "must be inside report_data, i.e. covered by the quote"
    # flipping the flag breaks the binding — a miner cannot claim confinement it did not have
    forged = dataclasses.replace(p, confined=True)
    assert forged.report_data() != p.quote.report_data
    assert Proof.from_json(p.to_json()).confined is False


def test_absent_confined_field_reads_as_false():
    """An older proof that never asserted confinement must not be read as having had it."""
    import json

    from thirtyspokes.koth.proof import Proof
    p = Proof(epoch=1, nonce="n", hotkey="hk", source_hash="sh", weights_hash="wh",
              model_id="m", results=(), total_cost_usd=0.0, n_calls=0,
              call_log_hash="x", measurement="m", confined=True)
    d = json.loads(p.to_json()); d.pop("confined")
    assert Proof.from_json(json.dumps(d)).confined is False


def test_enforcing_validator_rejects_an_unconfined_proof(env):
    """The gate that makes this enforceable regardless of how the miner configured its runtime:
    the measured-image gates prove WHICH image booted, not what it did inside."""
    a = _art(_ROUTER_SRC, env.allk)
    p = _proof(env, a, "hk")
    assert p.confined is False
    v = _verify(env, p, a, "hk", enforce=True, approved_mrtd={"m"},
                approved_rtmr={1: "a", 2: "b", 3: "c"}, tcb_accept=frozenset({"UpToDate"}))
    assert not v.valid and v.reason in {"unconfined_agent", "mock_quote_rejected"}


# --- CRITICAL-2: memorization of the PUBLIC benchmark ------------------------------------------
# The scored pool is 500 public items/benchmark with public gold. The beacon makes the SLICE
# unpredictable but not the POOL, so a miner memorises all 500 -- a few hundred KB in weights.bin,
# which `scan_source` never looked at. Plain grounding ("answer appears in some response") was
# defeated by laundering: put the memorised answer in your own prompt and the model echoes it.

def _tr(*pairs):
    return [{"task_id": "t", "prompt": p, "response": r} for p, r in pairs]


def test_first_appearance_grounding_catches_laundering():
    """The provenance question: did the answer surface on the RESPONSE side before it ever
    surfaced on the PROMPT side? Information the agent already had shows up in a prompt first."""
    from thirtyspokes.koth.verify import _grounded_one
    # free-form: the answer (11) is NOT in the prompt by construction
    assert _grounded_one("11", _tr(("Compute 3+4*2.", "11")), "number") == (True, "ok")
    assert _grounded_one("11", _tr(("hi", "hello")), "number") == (False, "ungrounded")
    assert _grounded_one("11", _tr(("Repeat exactly: 11", "11")), "number") == (False, "laundered")


def test_first_appearance_does_not_flag_honest_orchestration():
    """Verify-loops and cheap->strong escalation must keep passing: their answer still ORIGINATES
    in a response, it is merely quoted back afterwards."""
    from thirtyspokes.koth.verify import _grounded_one
    verify_loop = _tr(("Compute 3+4*2.", "11"), ("Is 11 correct?", "11"))
    escalation = _tr(("Compute 3+4*2.", "10"), ("Compute 3+4*2.", "11"))
    assert _grounded_one("11", verify_loop, "number") == (True, "ok")
    assert _grounded_one("11", escalation, "number") == (True, "ok")


def test_multiple_choice_is_exempt_from_the_provenance_rule():
    """MCQ is undefendable here BY CONSTRUCTION -- every option is in the prompt, so the rule would
    flag every honest agent. It must fall back to plain grounding rather than DQ everyone. This is
    why the ranking weights are 100% free-form (`real_suite`)."""
    from thirtyspokes.koth.verify import _grounded_one, answer_token
    mcq = _tr(("Q: 2+2? A) 3 B) 4", "B"))
    assert answer_token(mcq[0]["prompt"], "choice") == "B", "the gold is in the prompt already"
    assert _grounded_one("B", mcq, "choice") == (True, "ok"), "must not flag an honest MCQ agent"


def test_ranking_weights_are_entirely_free_form():
    """No ranked benchmark may use an answer format whose provenance cannot be checked."""
    from thirtyspokes.koth.verify import _PROVENANCE_KINDS, _bench_kind
    from thirtyspokes.koth.benchmarks import default_suite
    for b in default_suite():
        if b.weight > 0:
            assert _bench_kind(b) in _PROVENANCE_KINDS, f"{b.name} is ranked but undefendable"


def test_scan_weights_finds_an_answer_table_but_not_random_weights():
    """weights.bin was the obvious hiding place and was never examined. Decoys of the same
    digit-length are the control, so short numeric golds matching by chance don't false-positive."""
    import json
    import numpy as np
    from thirtyspokes.koth.verify import scan_weights
    golds = [18, 7, 4213, 96, 51, 8, 1240, 33]
    table = json.dumps({f"q{i}": g for i, g in enumerate(golds)}).encode()
    assert scan_weights(table, golds, salt="n1") == (True, "answers_in_weights")
    # a real float32 weight blob must NOT trip it, however numerically dense
    rng = np.random.default_rng(0)
    real = rng.standard_normal(200_000).astype("float32").tobytes()
    assert scan_weights(real, golds, salt="n1")[0] is False
    assert scan_weights(b"", golds, salt="n1")[0] is False


# --- the router scalar: "best answer at the lowest price, for a given ask" -----------------------
# The old scalar (Q_lcb - lambda*cost) scores the OUTCOME, is ABSOLUTE rather than baseline-relative
# (on a 95%-accurate pool ~98% of it measures the POOL), and fixes ONE quality/price exchange rate.
# `router_headroom` scores the DECISION against what was achievable AT THAT PRICE.

def _ref():
    """3 asks x 2 models. cheap solves only ask0; pricey solves all."""
    import numpy as np
    return np.array([[1., 1.], [0., 1.], [0., 1.]]), np.array([[0.001, 0.10]] * 3)


def test_router_headroom_is_zero_on_the_featureless_frontier():
    """Both 'always cheapest' and 'always priciest' are ON the Zero frontier -- neither demonstrated
    any routing skill, so both must score 0 regardless of their very different accuracy."""
    from thirtyspokes.koth.verify import router_headroom
    S, C = _ref()
    assert abs(router_headroom(1 / 3, 0.001, S, C)) < 1e-9      # always cheap
    assert abs(router_headroom(1.0, 0.10, S, C)) < 1e-9         # always pricey


def test_router_headroom_rewards_matching_quality_at_lower_price():
    """The property the OLD quality-only measure got backwards: an oracle router that reaches full
    quality by paying the cheap model where it suffices scores 1.0, not negative."""
    from thirtyspokes.koth.verify import router_headroom
    S, C = _ref()
    oracle_cost = (0.001 + 0.10 + 0.10) / 3        # cheap on ask0, pricey on the other two
    assert router_headroom(1.0, oracle_cost, S, C) == pytest.approx(1.0, abs=1e-6)
    assert router_headroom(0.0, 0.05, S, C) < 0     # worse than the baseline -> negative


def test_router_headroom_ignores_asks_nobody_can_solve():
    """A query no pool model answers must not drag every miner down -- routing cannot fix it."""
    from thirtyspokes.koth.verify import oracle_frontier, zero_frontier
    import numpy as np
    S = np.array([[1., 1.], [0., 0.]]); C = np.array([[0.001, 0.10]] * 2)
    # both frontiers cap at 0.5: the unsolvable ask is simply unreachable for anyone
    assert max(q for _, q in zero_frontier(S, C)) == pytest.approx(0.5)
    assert max(q for _, q in oracle_frontier(S, C)) == pytest.approx(0.5)


def test_validator_uses_the_router_scalar_when_a_pool_reference_exists(env):
    """With a reference the reign scalar becomes frontier-relative headroom; without one it falls
    back to the legacy Q_lcb - lambda*cost, so existing deployments are unchanged."""
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.koth.verify import BenchStat, ProofVerdict
    from thirtyspokes.reign import KingChain
    S, C = _ref()
    vd = ProofVerdict(True, "ok", {"math": BenchStat(3, 1.0, 0.7, 0.201)}, 0.201, 0.7, 1.0)

    plain = KOTHValidator(env.approved, "pub", None, KingChain(), env.suite, None, None)
    legacy = plain._reign_scalar(vd, epoch=1, nonce="n")
    assert legacy == pytest.approx(vd.score - 0.02 * min(1.0, 0.201 / 0.5))

    routed = KOTHValidator(env.approved, "pub", None, KingChain(), env.suite, None, None,
                           pool_reference=lambda e, n: (S, C))
    assert routed._reign_scalar(vd, epoch=1, nonce="n") == pytest.approx(1.0, abs=1e-6)


def test_pool_reference_outage_falls_back_instead_of_breaking_scoring():
    """A reference fetch that raises must never take the subnet down."""
    from thirtyspokes.koth.validator import KOTHValidator
    from thirtyspokes.koth.benchmarks import default_suite
    from thirtyspokes.reign import KingChain

    def boom(epoch, nonce):
        raise RuntimeError("reference unavailable")

    v = KOTHValidator({"m"}, "pub", None, KingChain(), default_suite(), None, None,
                      pool_reference=boom)
    assert v._router_scalar(0.9, 0.01, 1, "n") is None
