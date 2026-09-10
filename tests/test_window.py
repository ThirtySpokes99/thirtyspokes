"""The window's properties (docs/WHITEPAPER.md §6.2/§6.3/§6.3b/§6.3c, the build plan M7, M7a).

The threat model here is an OWNER with discretion, not a miner with a GPU. Nothing in a window is
secret — these are public benchmarks a miner can download in full, and §2.1 already removed the value
of memorising their answers. What must hold is that the window a validator scores is the one the
schedule pinned before anybody committed, that its slice is derived rather than chosen, that both
arms face the identical tasks AND the identical catalog, and that anything unverifiable stops the
window rather than degrading it.

The three slice invariants (§6.3b) get their own tests because each is a silent dependency of a
scoring rule somewhere else: leave-one-out needs every benchmark present every window, equal-weight
aggregation needs roughly equal counts, and `quality_b` needs a minimum count under which the
benchmark is dropped VISIBLY.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from thirtyspokes.v3 import window as win
from thirtyspokes.v3.config import MAX_STEPS, TEMPERATURE
from thirtyspokes.v3.pool import thaw
from thirtyspokes.v3.scaffold import task_order
from thirtyspokes.v3.types import Catalog, CatalogEntry, EpisodeResult, TaskSpec
from thirtyspokes.koth import holdout_feed
from thirtyspokes.koth.reference import envelope

BENCHMARKS = ("terminal-bench", "deepswe", "cybergym", "hle-tools")

CATALOG = Catalog(entries=(
    CatalogEntry("cheap/model", 0.05, 0.20, 128_000),
    CatalogEntry("strong/model", 2.50, 10.00, 400_000),
))

NONCE = "0x9f2c41ab"


def pool(per_benchmark: int = 40, benchmarks=BENCHMARKS, sizes=None) -> tuple[TaskSpec, ...]:
    """A task pool: `benchmarks` strata, `sizes[b]` tasks each (default `per_benchmark`)."""
    sizes = sizes or {}
    return tuple(TaskSpec(task_id=f"{b}-{i:04d}", benchmark=b, prompt=f"{b} task {i}",
                          tools=("bash",))
                 for b in benchmarks
                 for i in range(sizes.get(b, per_benchmark)))


def scheduled(tasks, *, window: int = 7, per_benchmark: int = 10, minimum: int = 5):
    """The owner's whole pre-commit step: entries for the run, chained into ONE manifest root."""
    entries = win.schedule(range(1, 9), tasks=tasks, per_benchmark=per_benchmark, minimum=minimum)
    return entries[window - 1], holdout_feed.manifest(entries)


def published(record: dict, sign=None) -> bytes:
    """The bytes a validator fetches: the record, signed by the owner."""
    return json.dumps(envelope(record, sign or signer()[0])).encode()


def signer(key: str = "owner"):
    def sign(data: bytes) -> str:
        return hashlib.sha256(key.encode() + data).hexdigest()

    def verify_sig(data: bytes, sig: str, ss58: str) -> bool:
        return sig == hashlib.sha256(ss58.encode() + data).hexdigest()
    return sign, verify_sig


def episodes(task_ids, benchmark: str = "terminal-bench", score: float = 1.0):
    return tuple(EpisodeResult(task_id=t, benchmark=benchmark, steps=(), graded_score=score,
                               spend_usd=0.01, stopped_reason="stop") for t in task_ids)


# --- M7 exit 1: the window is self-contained ---------------------------------------------------

def test_the_window_record_carries_everything_both_arms_need():
    """§6.2. Slice, catalog snapshot, task order, MAX_STEPS, decode params, scaffold version —
    anything missing is a difference between the two arms that the duel cannot see."""
    tasks = pool()
    entry, man = scheduled(tasks)
    record = win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)

    assert record["task_ids"] and record["nonce"] == NONCE
    assert thaw(record["catalog"]).entries == CATALOG.entries
    assert record["schedule"]["max_steps"] == MAX_STEPS
    assert record["schedule"]["temperature"] == TEMPERATURE
    assert record["schedule"]["scaffold_version"] == win.SCAFFOLD_VERSION
    assert win.verify(json.loads(json.dumps(record)), window=7, man=man, tasks=tasks)


def test_the_catalog_snapshot_is_what_makes_the_duel_paired():
    """Model availability and prices move through the day. King and challenger are evaluated minutes
    apart, so the action space has to come from the record and not from a live fetch — otherwise a
    challenger can win on a model the king never saw."""
    tasks = pool()
    entry, man = scheduled(tasks)
    record = json.loads(json.dumps(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)))

    later = Catalog(entries=(CatalogEntry("cheap/model", 0.90, 3.00, 128_000),))
    assert later.entries != CATALOG.entries              # the live list really did move

    king_arm = win.verify(record, window=7, man=man, tasks=tasks)
    challenger_arm = win.verify(record, window=7, man=man, tasks=tasks)
    assert king_arm.catalog.entries == challenger_arm.catalog.entries == CATALOG.entries


def test_the_recorded_task_order_is_the_one_the_scaffold_will_re_derive():
    """§4: both arms must be zeroed on the identical tail when an allowance runs out. The record
    pins an order and `Scaffold.run_window` derives one from the nonce; if they could disagree, the
    pinned order would be decoration."""
    tasks = pool()
    entry, man = scheduled(tasks)
    verified = win.verify(json.loads(json.dumps(
        win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE))),
        window=7, man=man, tasks=tasks)
    assert task_order(verified.tasks, verified.nonce) == verified.tasks


# --- M7 exit 2: one manifest root, committed before challengers commit -------------------------

def test_the_schedule_entry_holds_nothing_that_is_unknown_before_the_window_opens():
    """This is what makes pre-commitment possible at all. The nonce and the catalog come into
    existence at window open, so they live in the published file; everything that must not move
    after challengers commit lives in the chained entry."""
    tasks = pool()
    entry, _ = scheduled(tasks)
    assert "nonce" not in entry and "catalog" not in entry and "task_ids" not in entry
    assert set(entry) == {"v", "epoch", "scaffold_version", "max_steps", "temperature",
                          "max_conductor_tokens", "per_benchmark", "minimum", "pool_digest"}


def test_one_root_covers_the_whole_schedule_so_no_per_window_extrinsic_is_needed():
    """The chain gives the owner ONE commitment slot and `set_commitment` overwrites it; the slot
    already holds governance. A per-window write would erase it and every validator would refuse to
    score (`koth/reference.py` hit this wall), so the whole run is chained into a single root."""
    tasks = pool()
    entries = win.schedule(range(1, 31), tasks=tasks, per_benchmark=10, minimum=5)
    man = holdout_feed.manifest(entries)
    assert isinstance(man["root"], str) and len(man["digests"]) == 30
    assert all(holdout_feed.in_manifest(e, man) for e in entries)


def test_a_swapped_slice_does_not_re_derive_and_is_refused():
    """The owner cannot choose a slice after seeing who committed: the validator re-derives it from
    the nonce and its own copy of the public pool rather than trusting the published IDs."""
    tasks = pool()
    entry, man = scheduled(tasks)
    record = json.loads(json.dumps(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)))
    unpicked = next(t.task_id for t in tasks if t.task_id not in record["task_ids"])
    record["task_ids"][3] = unpicked

    with pytest.raises(win.WindowError, match="does not re-derive"):
        win.verify(record, window=7, man=man, tasks=tasks)


def test_a_task_pool_edited_after_the_schedule_was_committed_is_refused():
    """Public benchmarks are patched at unchanged IDs, so the pool digest binds every field of every
    task — not just the IDs the draw reads. Otherwise a window could be scored on different text
    than the schedule committed to and nothing downstream would notice."""
    tasks = pool()
    entry, man = scheduled(tasks)
    record = json.loads(json.dumps(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)))

    patched = (TaskSpec(tasks[0].task_id, tasks[0].benchmark, "an easier version of the same task",
                        tasks[0].tools),) + tasks[1:]
    with pytest.raises(win.WindowError, match="different task pool"):
        win.verify(record, window=7, man=man, tasks=patched)


def test_the_draw_parameters_cannot_move_after_the_schedule_is_committed():
    """Under D12 the owner does not pick the slice — but picking how many tasks each benchmark
    contributes, or where the minimum sits, would pick it indirectly. Both are chained."""
    tasks = pool()
    entry, man = scheduled(tasks)
    record = json.loads(json.dumps(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)))
    record["schedule"]["per_benchmark"] = 12

    with pytest.raises(win.WindowError, match="not the one pinned"):
        win.verify(record, window=7, man=man, tasks=tasks)


def test_the_slice_is_unknowable_until_the_nonce_exists():
    """D12's whole protection: tasks are reusable and nothing burns, so what keeps a duel honest is
    that no one knows which tasks will be scored at the moment they commit."""
    tasks = pool()
    entry, _ = scheduled(tasks)
    first = win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)["task_ids"]
    other = win.build(entry, tasks=tasks, catalog=CATALOG, nonce="0xdeadbeef")["task_ids"]
    assert set(first) != set(other)
    assert win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)["task_ids"] == first


# --- M7 exit 3: missing or unverifiable -> fail closed ------------------------------------------

def test_a_missing_window_file_fails_closed():
    """No file, no duels, no shot spent — challengers roll over with their one submission intact.
    Scoring a window nobody committed to would spend a miner's single shot on it."""
    _, verify_sig = signer()
    _, man = scheduled(pool())

    def absent(path: str) -> bytes:
        raise FileNotFoundError(path)

    with pytest.raises(win.WindowError, match="missing or unauthentic"):
        win.load(7, fetch_bytes=absent, owner_ss58="owner", verify_sig=verify_sig, man=man,
                 tasks=pool())


def test_a_window_not_signed_by_the_owner_fails_closed():
    """Authenticity comes from the owner's signature, never from where the bytes were found — the
    path is derived and therefore guessable by anyone."""
    tasks = pool()
    entry, man = scheduled(tasks)
    record = win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)
    impostor_sign, _ = signer("impostor")
    _, verify_sig = signer("owner")
    raw = published(record, impostor_sign)

    with pytest.raises(win.WindowError, match="missing or unauthentic"):
        win.load(7, fetch_bytes=lambda _: raw, owner_ss58="owner", verify_sig=verify_sig, man=man,
                 tasks=tasks)


def test_a_correctly_signed_window_for_a_different_window_fails_closed():
    """A signature proves the owner wrote the bytes, not that they describe THIS window. An older
    window's file would score both arms on a slice the challengers already saw published."""
    tasks = pool()
    entry, man = scheduled(tasks, window=3)
    sign, verify_sig = signer()
    raw = published(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE), sign)

    with pytest.raises(win.WindowError, match="not window 4"):
        win.load(4, fetch_bytes=lambda _: raw, owner_ss58="owner", verify_sig=verify_sig, man=man,
                 tasks=tasks)


def test_another_windows_schedule_entry_inside_this_windows_file_fails_closed():
    """§6.3. Manifest membership is not enough on its own: the manifest chains N entries and all N
    are equally authentic, while the draw is a pure function of the entry (D12) — so N entries are
    N slices. An owner who may file window 4's entry under window 1 picks among the whole schedule's
    slices AFTER challengers have committed, which is exactly the discretion the chained manifest
    exists to remove, and nothing about it looks like a failure: every signature and every hash still
    verifies. The entry's own index must be bound to the window being scored.
    """
    tasks = pool()
    entries = win.schedule(range(1, 9), tasks=tasks, per_benchmark=10, minimum=5)
    man = holdout_feed.manifest(entries)
    smuggled = json.loads(json.dumps(
        win.build(entries[3], tasks=tasks, catalog=CATALOG, nonce=NONCE)))
    smuggled["epoch"] = 1                       # the only index today's caller-facing check reads

    assert holdout_feed.in_manifest(smuggled["schedule"], man)   # still validly chained
    honest = win.build(entries[0], tasks=tasks, catalog=CATALOG, nonce=NONCE)
    assert smuggled["task_ids"] != honest["task_ids"]            # and it IS a different slice

    with pytest.raises(win.WindowError, match="schedule entry for 4"):
        win.verify(smuggled, window=1, man=man, tasks=tasks)


def test_a_window_with_no_action_space_fails_closed():
    """An empty catalog is a window in which no `DELEGATE` can ever be valid: every episode would
    die on parse failures and both arms would score zero, which reads as a tie rather than as the
    broken window it is."""
    tasks = pool()
    entry, man = scheduled(tasks)
    record = json.loads(json.dumps(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)))
    record["catalog"] = "[]"

    with pytest.raises(win.WindowError, match="no action space"):
        win.verify(record, window=7, man=man, tasks=tasks)


def test_the_whole_hand_off_round_trips_when_nothing_is_wrong():
    """The other half of fail-closed: the checks must admit a genuine window, or the arena stalls
    forever on its own caution."""
    tasks = pool()
    entry, man = scheduled(tasks)
    sign, verify_sig = signer()
    raw = published(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE), sign)

    loaded = win.load(7, fetch_bytes=lambda p: raw, owner_ss58="owner", verify_sig=verify_sig,
                      man=man, tasks=tasks)
    assert loaded.epoch == 7 and loaded.nonce == NONCE
    assert loaded.benchmarks == tuple(sorted(BENCHMARKS))
    assert len(loaded.tasks) == 10 * len(BENCHMARKS)
    assert win.window_path(7) == "v3/window/7.json" != win.window_path(8)


# --- M7 exit 4: freshness and staleness are published per window --------------------------------

def test_freshness_and_staleness_are_published_per_window():
    """Both are what make ossification visible (M7a exit 4), and both are published numbers rather
    than gates. `staleness` is task-level recurrence, not the deprecated encoder's median-max-cosine
    (§9): under D12 tasks recur by design, so the exact question is answerable from labelled IDs."""
    tasks = pool()
    entry, _ = scheduled(tasks)
    first = win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)
    assert first["freshness"]["staleness"] is None        # window 1 has nothing to be stale against
    assert first["freshness"]["category_freshness"]["recycled_fraction"] == 0.0

    second = win.build(entry, tasks=tasks, catalog=CATALOG, nonce="0x00ff",
                       revealed=first["task_ids"])
    repeats = len(set(second["task_ids"]) & set(first["task_ids"]))
    assert second["freshness"]["staleness"] == repeats / len(second["task_ids"])
    # Every benchmark recurs every window by construction (§6.3b invariant 1), so category
    # freshness reads 1.0 — the instrument that bites here is the task-level one.
    assert second["freshness"]["category_freshness"]["recycled_fraction"] == 1.0


# --- §6.3b: the three slice invariants ----------------------------------------------------------

def test_every_admitted_benchmark_appears_in_every_slice():
    """Leave-one-out (§5.2) drops one benchmark at a time. A benchmark absent from a window makes
    LOO run over a different set that window, and verdicts stop being comparable across windows."""
    tasks = pool(benchmarks=tuple(f"bench-{i:02d}" for i in range(12)))
    for nonce in ("0x1", "0x2", "0x3", "0xfeed", "0xbeef"):
        sliced = win.draw(tasks, nonce=nonce, window=1, per_benchmark=10, minimum=5)
        assert len(sliced.benchmarks) == 12
        assert {t.benchmark for t in sliced.tasks} == {f"bench-{i:02d}" for i in range(12)}


def test_counts_are_equal_per_benchmark_however_lopsided_the_pool_is():
    """§5.1 weights benchmarks equally, and slices carry equal counts so that the equal-weight
    reading and the natural per-task reading coincide. A pool-proportional draw would hand the
    benchmark with the most tasks a bigger vote than the one nobody chose."""
    tasks = pool(sizes={"terminal-bench": 500, "deepswe": 31, "cybergym": 30, "hle-tools": 30})
    sliced = win.draw(tasks, nonce=NONCE, window=1, per_benchmark=10, minimum=5)
    assert {n for _, n in sliced.counts} == {10}


def test_a_benchmark_below_the_minimum_is_dropped_and_the_narrowing_is_recorded():
    """§6.3b invariant 3. Below the minimum `quality_b` is too noisy to carry a twelfth of the
    score, so the window is built WITHOUT the benchmark — and says so, because leave-one-out is
    about to run over a smaller set than last window's."""
    tasks = pool(sizes={"terminal-bench": 40, "deepswe": 40, "cybergym": 3, "hle-tools": 40})
    entry, man = scheduled(tasks)
    record = json.loads(json.dumps(win.build(entry, tasks=tasks, catalog=CATALOG, nonce=NONCE)))

    assert "cybergym" not in record["benchmarks"]
    assert record["narrowed"] == [["cybergym", 3]]        # recorded, never silent
    assert not any(t.startswith("cybergym") for t in record["task_ids"])
    verified = win.verify(record, window=7, man=man, tasks=tasks)
    assert verified.benchmarks == ("deepswe", "hle-tools", "terminal-bench")


def test_the_draw_is_stratified_within_benchmarks_never_across():
    """Sampling across benchmarks would couple them: a benchmark dropped or admitted (§6.1) would
    reshuffle every other benchmark's slice, and two windows differing only in the admitted set
    would be incomparable for a reason nobody chose."""
    four = pool()
    five = four + pool(benchmarks=("automationbench",))
    a = win.draw(four, nonce=NONCE, window=1, per_benchmark=10, minimum=5)
    b = win.draw(five, nonce=NONCE, window=1, per_benchmark=10, minimum=5)
    for benchmark in BENCHMARKS:
        picks = {t.task_id for t in a.tasks if t.benchmark == benchmark}
        assert picks == {t.task_id for t in b.tasks if t.benchmark == benchmark}


def test_both_arms_draw_the_identical_slice_whatever_order_the_pool_arrives_in():
    """The `scoreable_ids` failure mode: a draw that depends on the caller's list order agrees
    across the two arms only as long as everything upstream also agrees. Ranking by
    `sha256(nonce|window|benchmark|task_id)` depends on which tasks are eligible and nothing else."""
    tasks = pool()
    king = win.draw(tasks, nonce=NONCE, window=1, per_benchmark=10, minimum=5)
    challenger = win.draw(tuple(reversed(tasks)), nonce=NONCE, window=1, per_benchmark=10,
                          minimum=5)
    assert king == challenger


def test_a_draw_below_the_floor_it_enforces_is_refused():
    """With `per_benchmark < minimum`, "drawn below the minimum" and "available below the minimum"
    stop being the same condition and the window enforces a noise floor it violates everywhere."""
    with pytest.raises(win.WindowError, match="below the floor"):
        win.draw(pool(), nonce=NONCE, window=1, per_benchmark=4, minimum=8)


# --- §6.3c: task exclusion is symmetric ---------------------------------------------------------

def test_a_grader_failure_drops_the_task_from_both_arms_and_the_denominator():
    """§8b.3's other half: a worker model erroring is an outcome and belongs in the score, but
    Docker dying is a fact about the validator. Scoring it zero injects the owner's infrastructure
    noise into a miner's result."""
    ids = [f"t{i}" for i in range(6)]
    arms = win.exclude(episodes(ids, score=1.0), episodes(ids, score=0.5), failed=["t2"])

    assert [r.task_id for r in arms.king] == [i for i in ids if i != "t2"]
    assert [r.task_id for r in arms.challenger] == [i for i in ids if i != "t2"]
    assert len(arms.king) == len(arms.challenger) == 5
    assert arms.excluded == (("t2", "grader"),)


def test_a_task_missing_from_one_arm_leaves_the_denominator_for_both():
    """The same asymmetry arriving by a different route — a crashed arm rather than a named grader
    failure. An exclusion applied to one arm only would compare different task sets and hand the
    other arm an easier slice."""
    king = episodes([f"t{i}" for i in range(5)])
    challenger = episodes([f"t{i}" for i in range(5) if i != 1] + ["t9"])
    arms = win.exclude(king, challenger)

    assert [r.task_id for r in arms.king] == ["t0", "t2", "t3", "t4"]
    assert [r.task_id for r in arms.challenger] == ["t0", "t2", "t3", "t4"]
    assert arms.excluded == (("t1", "missing from challenger arm"),
                             ("t9", "missing from king arm"))


def test_the_surviving_arms_stay_row_aligned_for_the_paired_duel():
    """`duel.BenchmarkPair` pairs positionally — row `i` must be the same task for both arms. The
    challenger arm may arrive in any order (parallel execution finishes out of order), so alignment
    has to be produced here rather than assumed."""
    ids = [f"t{i}" for i in range(6)]
    shuffled = [ids[i] for i in (4, 0, 5, 3, 1, 2)]
    arms = win.exclude(episodes(ids), episodes(shuffled), failed=["t3"])

    assert [r.task_id for r in arms.king] == [r.task_id for r in arms.challenger]
    assert "t3" not in [r.task_id for r in arms.king]


def test_exclusions_are_counted_and_reasoned_so_the_reveal_can_publish_them():
    """§8b.3: "exclusions are counted and published". A window that dropped many tasks has degraded
    power, and a reader comparing it to the next window has to be able to see that."""
    ids = [f"t{i}" for i in range(8)]
    arms = win.exclude(episodes(ids), episodes(ids[:-1]), failed=["t0", "t1"])
    assert len(arms.excluded) == 3
    assert dict(arms.excluded)["t7"] == "missing from challenger arm"
    assert len(arms.king) == len(ids) - 3


def test_a_duplicated_task_is_refused_rather_than_mispaired():
    """A repeated task ID makes positional pairing ambiguous, and the wrong pairing is invisible in
    every downstream number — so it stops the duel here instead."""
    with pytest.raises(ValueError, match="same task twice"):
        win.exclude(episodes(["t0", "t1", "t1"]), episodes(["t0", "t1"]))
