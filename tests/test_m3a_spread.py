"""The two numbers M3a's report adds to `score.derive_exchange`'s verdict.

`derive_exchange` decides DOMINATOR on `q_strong <= q_cheap` — two arm means, compared bare, with
no noise band of the kind `RELSPEND_SPREAD_FLOOR` gives the other guard. That flag is the whole of
the spend plan §7 step 3a's stop condition, and this repository has already RETRACTED a launch
verdict that turned out to be one noisy slice. So the report says how wide the margin is and how
often the two arms actually disagreed, and those two statements have to be right.

The arms are paired on the same tasks, which is what makes a sign test the honest read: the pairing
is the design of the sweep, and throwing it away for a two-sample comparison on the means would
discard the variance the pairing already removed.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from thirtyspokes.v3 import score
from thirtyspokes.v3.score import Exchange
from thirtyspokes.v3.types import EpisodeResult


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m3a_spread.py"
    spec = importlib.util.spec_from_file_location("m3a_spread", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m3a = _load_script()


def result(task_id: str, score: float, *, benchmark: str = "livecodebench") -> EpisodeResult:
    return EpisodeResult(task_id=task_id, benchmark=benchmark, steps=(), graded_score=score,
                         spend_usd=0.01, stopped_reason="")


def arms(cheap: dict[str, float], strong: dict[str, float], *, benchmark: str = "livecodebench"):
    return [m3a.Arm(name="always_cheapest",
                    results=tuple(result(t, s, benchmark=benchmark) for t, s in cheap.items())),
            m3a.Arm(name="always_strongest",
                    results=tuple(result(t, s, benchmark=benchmark) for t, s in strong.items()))]


# --- the pairing --------------------------------------------------------------------------------


def test_wins_are_counted_per_task_rather_than_off_the_means():
    """Two arms with the SAME mean can still disagree on every task, and that is the case the mean
    cannot see — it is also the case where the dominator flag is decided by which way the last tie
    happened to fall."""
    pairs = arms(cheap={"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0},
                 strong={"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0})

    assert m3a.paired_wins(pairs, "livecodebench") == (2, 2, 0)


def test_a_task_only_one_arm_reached_is_not_counted_as_a_win_for_the_other():
    """§5.4's futility abort and §6.3c's exclusions both leave an arm short of the slice. Counting a
    task the strongest arm never bought as a loss would turn "we stopped paying" into "money buys
    nothing", which is the exact claim the flag is making."""
    pairs = arms(cheap={"a": 1.0, "b": 1.0, "c": 1.0},
                 strong={"a": 1.0})

    assert m3a.paired_wins(pairs, "livecodebench") == (0, 0, 1)


def test_only_the_named_benchmark_is_paired():
    pairs = [m3a.Arm(name="always_cheapest",
                     results=(result("a", 0.0), result("x", 1.0, benchmark="hle"))),
             m3a.Arm(name="always_strongest",
                     results=(result("a", 1.0), result("x", 0.0, benchmark="hle")))]

    assert m3a.paired_wins(pairs, "livecodebench") == (1, 0, 0)
    assert m3a.paired_wins(pairs, "hle") == (0, 1, 0)


# --- the p value --------------------------------------------------------------------------------


def _exact_two_sided(up: int, down: int) -> float:
    """The closed form, written out separately: both tails of Binomial(n, 1/2) at or past min(up, down)."""
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    lower = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    upper = sum(math.comb(n, i) for i in range(n - k, n + 1)) / 2 ** n
    return min(1.0, lower + upper)


def test_the_sign_test_is_the_exact_binomial_and_not_an_approximation():
    """Checked against the closed form at the counts a real slice produces, including the small ones
    where a normal approximation is worst and the decision is most fragile."""
    for up in range(0, 25):
        for down in range(0, 25):
            assert abs(m3a.sign_test(up, down) - _exact_two_sided(up, down)) < 1e-12, (up, down)


def test_arms_that_never_disagree_are_not_evidence_of_anything():
    """All ties is the reading a small slice most often produces, and it must come back as p=1
    rather than as a small number that would read as a confirmed dominator."""
    assert m3a.sign_test(0, 0) == 1.0


def test_a_one_task_margin_does_not_read_as_a_finding():
    """The failure the margin line exists to prevent: one task separating the arms is a coin flip,
    and DOMINATOR would fire on it just as hard as on a rout."""
    assert m3a.sign_test(0, 1) == 1.0
    assert m3a.sign_test(8, 16) > 0.05
    assert m3a.sign_test(2, 20) < 0.001                 # and a rout still reads as one


def test_the_test_is_symmetric_in_which_arm_won():
    for up, down in ((3, 11), (0, 7), (9, 9), (20, 4)):
        assert m3a.sign_test(up, down) == m3a.sign_test(down, up)


# --- the provider-error fit ----------------------------------------------------------------------


def step(model_id: str | None, *, cost: float, success: bool):
    return SimpleNamespace(model_id=model_id, cost_usd=cost, success=success, parse_failed=False)


def episode(task_id: str, score_: float, spend: float, *steps, benchmark: str = "livecodebench"):
    return SimpleNamespace(task_id=task_id, benchmark=benchmark, graded_score=score_,
                           spend_usd=spend, stopped_reason="stop", steps=list(steps))


def test_a_wrong_answer_is_not_a_provider_error():
    """§8b.3's distinction, and the whole validity of the correction rests on it: `scaffold` records
    a worker exception as `success=False, cost_usd=0.0` and a WRONG answer as `success=False,
    cost_usd>0`. Excluding wrong answers would not correct a measurement, it would delete it."""
    assert m3a.errored(episode("a", 0.0, 0.0, step("m", cost=0.0, success=False)))
    assert not m3a.errored(episode("b", 0.0, 0.01, step("m", cost=0.01, success=False)))
    assert not m3a.errored(episode("c", 1.0, 0.01, step("m", cost=0.01, success=True)))


def test_a_step_with_no_model_is_never_a_provider_error():
    """`model_id` is None on STOP and on a PARSE FAILURE, and a parse failure is recorded exactly
    as a provider error is — no worker was called, so it cost nothing and did not succeed. It is a
    fact about the Conductor's output, not about a provider, and counting it would delete episodes
    where the pool answered perfectly well."""
    assert not m3a.errored(episode("a", 1.0, 0.01, step("m", cost=0.01, success=True),
                                   step(None, cost=0.0, success=False)))


def test_an_errored_task_leaves_BOTH_arms():
    """§6.3c: an infrastructure failure drops the task from both arms and the denominator. Dropping
    it from the erroring arm alone hands the other an easier slice — which here would mean
    correcting a measurement by biasing it the other way."""
    cheap = [episode("t1", 0.0, 0.0, step("cheap", cost=0.0, success=False)),      # cheap errored
             episode("t2", 1.0, 0.01, step("cheap", cost=0.01, success=True)),
             episode("t3", 1.0, 0.01, step("cheap", cost=0.01, success=True))]
    strong = [episode("t1", 1.0, 0.05, step("strong", cost=0.05, success=True)),
              episode("t2", 0.0, 0.0, step("strong", cost=0.0, success=False)),    # strong errored
              episode("t3", 1.0, 0.05, step("strong", cost=0.05, success=True))]
    arms = [m3a.Arm(name="always_cheapest", results=tuple(cheap)),
            m3a.Arm(name="always_strongest", results=tuple(strong))]

    table, kept = m3a.exchange_without_errors(arms, {"livecodebench": Exchange(
        "livecodebench", 0.0, 0.04, ())})

    # BOTH directions: t1 is the erroring arm's own task and t2 is the other arm's. A filter that
    # asked only about the cheap arm would keep t2 and score the strongest arm a zero it never
    # earned — correcting the measurement by biasing it the other way.
    assert kept == {"livecodebench": 1}, "only t3 survives; an error in either arm drops the pair"
    # On the surviving pair both arms score 1.0, so the strongest bought nothing: DOMINATOR.
    assert table["livecodebench"].flags and "dominator" in table["livecodebench"].flags[0]


def test_the_corrected_fit_reuses_the_published_c_b_rather_than_re_pinning_it():
    """The two lambdas are printed side by side and read as comparable. Re-pinning C_b on the
    surviving episodes would make them differ in two things at once, and the reader could not tell
    which one moved the number."""
    cheap = [episode(f"t{i}", 0.2, 0.01, step("cheap", cost=0.01, success=True)) for i in range(4)]
    strong = [episode(f"t{i}", 0.8, 0.05, step("strong", cost=0.05, success=True)) for i in range(4)]
    arms = [m3a.Arm(name="always_cheapest", results=tuple(cheap)),
            m3a.Arm(name="always_strongest", results=tuple(strong))]

    table, _ = m3a.exchange_without_errors(arms, {"livecodebench": Exchange(
        "livecodebench", 0.0, 0.123, ())})

    assert table["livecodebench"].c_per_task == 0.123


def test_a_run_without_both_fixed_arms_yields_no_corrected_fit():
    """`--report-only` over a directory holding one arm is a real state — an interrupted run writes
    episodes incrementally — and a slope needs both ends."""
    arms = [m3a.Arm(name="always_cheapest",
                    results=(episode("t1", 1.0, 0.01, step("cheap", cost=0.01, success=True)),))]

    assert m3a.exchange_without_errors(arms, {}) == ({}, {})


# --- selecting which arms to buy ------------------------------------------------------------------


def test_the_three_arms_are_the_ones_the_table_is_fitted_from():
    """`--arms` chooses from this registry and the lambda fit reads all three by name, so a rename
    here silently breaks the fit rather than the flag."""
    from thirtyspokes.v3.pool import Catalog
    from thirtyspokes.v3.types import CatalogEntry
    catalog = Catalog(entries=(CatalogEntry("a/cheap", 0.05, 0.2, 128_000),
                               CatalogEntry("b/mid", 0.30, 1.2, 128_000),
                               CatalogEntry("c/strong", 1.00, 4.0, 128_000)))

    policies = m3a.arm_policies(catalog)

    assert tuple(policies) == m3a.ARM_NAMES
    # The cascade must end somewhere the cheapest arm does not, or the pair measures zero (§5.6).
    assert policies["always_cheapest"].rungs[-1] != policies["cascade"].rungs[-1]


def _dry_run(monkeypatch, tmp_path, *argv) -> str:
    """`--dry-run` through `main`, with only the network seams replaced."""
    from thirtyspokes.v3.pool import Catalog
    from thirtyspokes.v3.types import CatalogEntry
    catalog = Catalog(entries=(CatalogEntry("a/cheap", 0.05, 0.2, 128_000),
                               CatalogEntry("b/mid", 0.30, 1.2, 128_000),
                               CatalogEntry("c/strong", 1.00, 4.0, 128_000)))
    monkeypatch.setattr(m3a.pool_mod, "fetch", lambda *a, **k: None)
    monkeypatch.setattr(m3a.pool_mod, "snapshot", lambda payload: catalog)
    monkeypatch.setattr(m3a, "probe_pool", lambda cat: cat)
    monkeypatch.setattr(m3a, "slice_for",
                        lambda names, per, *, seed: ((), [object()] * 100))
    captured: list[str] = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: captured.append(" ".join(str(x) for x in a)))
    flags = [a for a in argv if a != "--no-dry-run-marker"]
    dry = ["--dry-run"] if "--no-dry-run-marker" not in argv else []
    m3a.main(["--out", str(tmp_path), *dry, "--episode-usd", "0.10", *flags])
    return "\n".join(captured)


def test_re_buying_one_rung_is_priced_as_one_rung(monkeypatch, tmp_path):
    """The point of the flag: the pre-flight ceiling prices what is actually BOUGHT. Estimating
    three arms for a one-arm re-run would refuse the cheap fix the reliability guard asks for."""
    one = _dry_run(monkeypatch, tmp_path, "--arms", "always_cheapest", "--budget-usd", "100")
    all_three = _dry_run(monkeypatch, tmp_path, "--budget-usd", "100")

    assert "100 tasks x 1 arm(s) = 100 episodes" in one
    assert "estimate $10.00" in one
    assert "100 tasks x 3 arm(s) = 300 episodes" in all_three
    assert "estimate $30.00" in all_three
    # And the ceiling each arm runs under moves with the split. An arm capped for a three-arm run
    # would stop short of its slice and publish as futility rather than as a budget bug.
    assert "per-arm ceiling $100.00 across 1 arm(s): always_cheapest" in one
    assert "per-arm ceiling $33.33 across 3 arm(s)" in all_three


def test_the_measurement_is_more_patient_than_the_mechanism(monkeypatch, tmp_path):
    """MEASURED 2026-09-05: the cheap rung lost 19-35% of its calls to HTTP 429 across two runs,
    every one after the client's three attempts. A validator must not stall a duel behind a slow
    provider, so the mechanism keeps the tight default; a measurement fitting a constant that gets
    pinned per corpus can afford to wait, and `reliability` refuses the reading either way."""
    import inspect
    from thirtyspokes.v3.openrouter import OpenRouterClient
    client_defaults = inspect.signature(OpenRouterClient.__init__).parameters

    out = _dry_run(monkeypatch, tmp_path, "--budget-usd", "100")

    assert "worker retries: 8 attempts, 2.0s backoff" in out
    assert 8 > client_defaults["attempts"].default
    assert 2.0 > client_defaults["backoff"].default


class _Aborted(Exception):
    """Stops `main` at the client, which is the first thing it builds after the dry-run gate."""


def test_the_retry_settings_actually_reach_the_worker(monkeypatch, tmp_path):
    """The failure a flag test cannot see: parsed, printed, and never passed. Measured by building
    the client and stopping there — it is the first thing `main` constructs once it commits to
    spending, so nothing else has to be faked."""
    seen = {}

    def fake_client(key, **kw):
        seen.update(kw)
        raise _Aborted

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setattr(m3a, "OpenRouterClient", fake_client)
    with pytest.raises(_Aborted):
        _dry_run(monkeypatch, tmp_path, "--budget-usd", "100", "--attempts", "11",
                 "--backoff", "3.5", "--no-dry-run-marker")

    assert seen == {"attempts": 11, "backoff": 3.5}


def test_a_one_arm_run_is_still_refused_against_a_ceiling_it_exceeds(monkeypatch, tmp_path):
    """The ceiling is not weakened by narrowing the run — it is still checked before any call."""
    with pytest.raises(SystemExit, match="REFUSED before spending"):
        _dry_run(monkeypatch, tmp_path, "--arms", "always_cheapest", "--budget-usd", "5")


def test_a_partial_run_that_cannot_fit_a_slope_says_which_arm_is_missing(monkeypatch, tmp_path):
    """`--arms always_cheapest` in a directory with nothing else cannot fit lambda: the slope needs
    both fixed arms and C_b is pinned from the cascade. Better a named refusal than a KeyError."""
    (tmp_path / "episodes-always_cheapest.jsonl").write_text("", encoding="utf-8")
    arms = m3a.load_arms(tmp_path)

    assert [a.name for a in arms] == ["always_cheapest"]
    assert [n for n in m3a.ARM_NAMES if n not in {a.name for a in arms}] == [
        "always_strongest", "cascade"]


# --- lambda's interval, and what it cannot establish -----------------------------------------------


def _sweep(cheap_scores, strong_scores, *, benchmark="livecodebench", spend=(0.001, 0.005)):
    cheap = [episode(f"t{i}", q, spend[0], step("c", cost=spend[0], success=q > 0),
                     benchmark=benchmark) for i, q in enumerate(cheap_scores)]
    strong = [episode(f"t{i}", q, spend[1], step("s", cost=spend[1], success=q > 0),
                      benchmark=benchmark) for i, q in enumerate(strong_scores)]
    return [m3a.Arm(name="always_cheapest", results=tuple(cheap)),
            m3a.Arm(name="always_strongest", results=tuple(strong))]


def test_a_flag_decided_by_a_hair_reports_as_a_coin_flip():
    """The failure mode the firing rate exists to expose: `derive_exchange` publishes DOMINATOR from
    one comparison of two means, so a slice where the arms trade single tasks flags exactly as hard
    as a rout does. Resampling the tasks turns that into a rate, and a rate near a half cannot be
    read as a fact about the pool.

    An exact tie is NOT this case — `q_strong <= q_cheap` makes a tie flag every time, and rightly:
    money that bought nothing is the finding. The unreadable case is arms a hair apart."""
    cheap  = [1.0, 0.0] + [0.5] * 18                                  # each arm wins one task
    strong = [0.0, 1.0] + [0.5] * 18
    arms = _sweep(cheap, strong)
    table = {"livecodebench": Exchange("livecodebench", 0.0, 0.004, ())}

    lo, hi, rate = m3a.lambda_stability(arms, table, n_boot=400)["livecodebench"]

    assert 0.05 < rate < 0.95, f"arms a hair apart must not report as settled (got {rate})"
    assert lo <= 0.0 <= hi


def test_a_rout_reports_as_a_rout():
    """The control. Without it the test above passes for a function that always returns 0.5."""
    arms = _sweep([0.0] * 20, [1.0] * 20)                            # strongest wins every task
    table = {"livecodebench": Exchange("livecodebench", 0.0, 0.004, ())}

    _lo, _hi, rate = m3a.lambda_stability(arms, table, n_boot=400)["livecodebench"]

    assert rate == 0.0, "money buying quality on every task is not a coin flip"


def test_the_resample_is_paired_so_task_difficulty_cannot_masquerade_as_a_slope():
    """Both arms ran the identical slice, so the task is the unit of evidence. An unpaired draw
    would price difficulty as disagreement between the arms — the variance the pairing removes."""
    hard_easy = [0.0] * 10 + [1.0] * 10
    arms = _sweep(hard_easy, hard_easy)                              # identical arms, wide spread

    _lo, _hi, rate = m3a.lambda_stability(
        arms, {"livecodebench": Exchange("livecodebench", 0.0, 0.004, ())}, n_boot=400)["livecodebench"]

    # Identical arms: every resample must tie, so the guard fires on every one of them.
    assert rate == 1.0


def test_a_run_missing_a_fixed_arm_has_no_interval_rather_than_a_wrong_one():
    arms = [m3a.Arm(name="always_cheapest", results=(episode("t1", 1.0, 0.001),))]

    assert m3a.lambda_stability(arms, {}) == {}


def test_the_interval_is_a_tail_bound_and_not_the_point_estimate_again():
    """A 2.5th percentile that quietly returned the middle of the distribution would report an
    interval far narrower than the evidence supports — the exact direction of error that makes a
    number look settled. So the lower bound has to sit BELOW the fit it brackets."""
    cheap = [0.0] * 20
    strong = [1.0] * 15 + [0.0] * 5                 # money wins, but not on every task
    arms = _sweep(cheap, strong)
    table = {"livecodebench": Exchange("livecodebench", 0.0, 0.004, ())}
    point = score.derive_exchange(list(arms[0].results), list(arms[1].results),
                                  {"livecodebench": 0.004})["livecodebench"].lam

    lo, hi, _rate = m3a.lambda_stability(arms, table, n_boot=600)["livecodebench"]

    assert lo < point < hi, f"interval [{lo}, {hi}] does not bracket the fit {point}"
    assert (point - lo) > 0.15 * point, "a lower bound this tight is the median wearing a hat"


def test_the_interval_and_the_fit_are_computed_over_the_same_episodes():
    """A 429 is scored as the model answering nothing, so a bootstrap over the RAW arms resamples
    the provider's throttle as if it were the pool's quality. Measured 2026-09-06 on run 3: the raw
    fit reported DOMINATOR firing in 0% of resamples while the same run with errored tasks dropped
    flagged DOMINATOR outright. The two numbers must describe one set of episodes."""
    clean = [episode(f"c{i}", 0.0, 0.001, step("c", cost=0.001, success=False)) for i in range(10)]
    # Ten tasks where the cheap arm was throttled: zero score at zero cost, model never answered.
    throttled = [episode(f"x{i}", 0.0, 0.0, step("c", cost=0.0, success=False)) for i in range(10)]
    strong = [episode(f"c{i}", 1.0, 0.005, step("s", cost=0.005, success=True)) for i in range(10)]
    strong += [episode(f"x{i}", 0.0, 0.005, step("s", cost=0.005, success=False)) for i in range(10)]
    arms = [m3a.Arm(name="always_cheapest", results=tuple(clean + throttled)),
            m3a.Arm(name="always_strongest", results=tuple(strong))]
    table = {"livecodebench": Exchange("livecodebench", 0.0, 0.004, ())}

    _table, kept = m3a.exchange_without_errors(arms, table)
    _lo, _hi, _rate = m3a.lambda_stability(arms, table, n_boot=200)["livecodebench"]

    # The throttled ten are gone from the fit; the interval must be drawn from the same ten that
    # remain, never from the twenty the raw arm reports.
    assert kept == {"livecodebench": 10}
    cheap_seen, strong_seen = m3a._clean_pairs(arms)
    assert set(cheap_seen) == set(strong_seen) == {f"c{i}" for i in range(10)}


# --- the pool is pinned in two places, and they must not drift -------------------------------------


def test_the_probe_pool_is_the_one_the_pinned_table_carries():
    """`probe_pool` refuses a substitute, so the code treats the pin as authoritative — and the pin
    that RUNS is `runs/m3a-exchange.json`'s `_pool`, which `world.pins` narrows the catalog to. Two
    copies of a decision, and the one the validator trusts is not the one the script carries.

    Order matters as much as membership: `reference.cascade` takes entries 0, 2 and 4 of the
    PRICE-SORTED list, so the table and the tuple have to agree about which model is the cheapest
    rung, not merely about the set of five.
    """
    import json
    table = json.loads((Path(__file__).resolve().parents[1] / "runs" / "m3a-exchange.json").read_text())
    assert tuple(table["_pool"]) == m3a.PROBE_POOL, (
        "runs/m3a-exchange.json's `_pool` and scripts/m3a_spread.py disagree about the probe pool. "
        "Re-pin BOTH — the ladder lambda prices is derived from this order.")


# --- the fit, through this script's own arm names ---------------------------------------------------


def test_the_fit_runs_over_the_scripts_arm_names_and_comes_back_in_them():
    """This script spells arms by file stem (`always_cheapest`); `reference` spells policies with a
    hyphen. The first re-render under the King0 fit refused on exactly that mismatch, because no
    test had driven `fit_pool` through the script's names. Both directions are pinned: the fit
    accepts the stems, and King0 and the pair come back as stems the report can look up."""
    cheap = [episode(f"t{i}", 0.3, 0.001, step("c", cost=0.001, success=False)) for i in range(4)]
    strong = [episode(f"t{i}", 0.6, 0.010, step("s", cost=0.010, success=True)) for i in range(4)]
    casc = [episode(f"t{i}", 0.9, 0.004, step("k", cost=0.004, success=True)) for i in range(4)]
    arms = [m3a.Arm(name="always_cheapest", results=tuple(cheap)),
            m3a.Arm(name="always_strongest", results=tuple(strong)),
            m3a.Arm(name="cascade", results=tuple(casc))]

    fit = m3a.fit_arms(arms)

    assert (fit.king_zero, fit.floor, fit.top) == ("cascade", "always_cheapest", "cascade")
    assert fit.exchange["livecodebench"].flags == ()
    assert fit.exchange["livecodebench"].lam > 0.0


def test_the_table_written_is_the_fit_the_report_recommends():
    """The report said "the fit to pin if the rung is not re-run" beside the error-excluded λ while
    the JSON — the artifact `world.pins` loads — kept the raw fit. One function decides which table
    is written, and with provider errors present it is the error-excluded one."""
    cheap = [episode(f"t{i}", 0.3, 0.001, step("c", cost=0.001, success=False)) for i in range(6)]
    cheap += [episode("x1", 0.0, 0.0, step("c", cost=0.0, success=False))]        # a 429, scored 0
    casc = [episode(f"t{i}", 0.9, 0.004, step("k", cost=0.004, success=True)) for i in range(6)]
    casc += [episode("x1", 0.9, 0.004, step("k", cost=0.004, success=True))]
    strong = [episode(f"t{i}", 0.6, 0.010, step("s", cost=0.010, success=True)) for i in range(6)]
    strong += [episode("x1", 0.6, 0.010, step("s", cost=0.010, success=True))]
    arms = [m3a.Arm(name="always_cheapest", results=tuple(cheap)),
            m3a.Arm(name="always_strongest", results=tuple(strong)),
            m3a.Arm(name="cascade", results=tuple(casc))]
    fit = m3a.fit_arms(arms)

    table, which = m3a.table_to_pin(arms, fit)
    clean, _ = m3a.exchange_without_errors(arms, fit.exchange, fit.top)

    assert "PROVIDER ERRORS EXCLUDED" in which
    assert table["livecodebench"].lam == clean["livecodebench"].lam
    assert table["livecodebench"].lam != fit.exchange["livecodebench"].lam

    no_errors = [m3a.Arm(name=a.name, results=tuple(r for r in a.results if r.task_id != "x1"))
                 for a in arms]
    table2, which2 = m3a.table_to_pin(no_errors, m3a.fit_arms(no_errors))
    assert "raw fit" in which2 and table2 == m3a.fit_arms(no_errors).exchange



# --- drawing SWE-bench Pro from what the grading daemon can actually run ------------------------------


def test_swebench_pro_is_drawn_only_from_instances_whose_image_is_present(monkeypatch):
    """The corpus is disk-bound ~50x, so a seeded draw over all 731 would name tasks no box can
    grade. The probe draws from `usable_ids()` — recorded by provisioning, not chosen here — and the
    draw is seeded so two boxes holding the same images draw the same slice."""
    from thirtyspokes.v3.benchmarks import swebench_pro as mod
    from thirtyspokes.v3.types import TaskSpec
    every = tuple(TaskSpec(task_id=f"swep-{i}", benchmark="swebench_pro", prompt="p", tools=())
                  for i in range(40))
    monkeypatch.setattr(mod.SweBenchPro, "__init__", lambda self: None)
    monkeypatch.setattr(mod.SweBenchPro, "load", lambda self: every)
    monkeypatch.setattr(mod.SweBenchPro, "usable_ids", lambda self: tuple(f"swep-{i}" for i in range(5, 15)))

    _, drawn_a = m3a.slice_for(["swebench_pro"], 6, seed="x")
    _, drawn_b = m3a.slice_for(["swebench_pro"], 6, seed="x")
    _, drawn_c = m3a.slice_for(["swebench_pro"], 6, seed="y")

    assert len(drawn_a) == 6 and {t.task_id for t in drawn_a} <= {f"swep-{i}" for i in range(5, 15)}
    assert [t.task_id for t in drawn_a] == [t.task_id for t in drawn_b], "not seeded"
    assert [t.task_id for t in drawn_a] != [t.task_id for t in drawn_c]


def test_swebench_pro_with_no_image_present_refuses_before_any_call(monkeypatch):
    from thirtyspokes.v3.benchmarks import swebench_pro as mod
    monkeypatch.setattr(mod.SweBenchPro, "__init__", lambda self: None)
    monkeypatch.setattr(mod.SweBenchPro, "usable_ids", lambda self: ())
    with pytest.raises(SystemExit, match="no instance image is present"):
        m3a.slice_for(["swebench_pro"], 6, seed="x")
