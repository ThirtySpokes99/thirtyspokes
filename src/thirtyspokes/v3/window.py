"""The window: what the owner pins before commits, and what a validator scores from
(docs/WHITEPAPER.md §6.2, §6.3, §6.3b, §6.3c, D12; the build plan M7, M7a).

A window is **self-contained** (§6.2) — slice, catalog snapshot, task order, `MAX_STEPS`, decode
params and scaffold version, in one record. The catalog is the part that is easy to leave out and
fatal to omit: OpenRouter's list moves through the day, so king and challenger evaluated minutes
apart would face different action spaces and the duel would not be paired. A challenger could then
win because a model the king never saw came online between the two arms.

THE COMMITMENT, AND WHY IT REUSES `holdout_feed` RATHER THAN INVENTING ANYTHING
------------------------------------------------------------------------------
The chain gives each hotkey exactly ONE commitment slot (`CommitmentOf[(netuid, hotkey)]`) and
`set_commitment` OVERWRITES it. The owner's slot already holds the governance record, so a
per-window write would erase governance and every validator would refuse to score — `koth/
reference.py` hit this exact wall and `koth/holdout_feed.py` is the answer that came out of it:
**derive the path, commit one chained-manifest root for the whole schedule.** So the owner's recipe
is two lines and no per-window extrinsic:

    entries = window.schedule(range(1, 31), tasks=POOL, per_benchmark=21, minimum=8)
    root    = holdout_feed.manifest(entries)["root"]        # ONE commitment, whole schedule

HOW "COMMITTED BEFORE CHALLENGERS COMMIT" (§6.3) AND "DRAWN AFTER THEY COMMIT" (D12) ARE BOTH TRUE
--------------------------------------------------------------------------------------------------
They look contradictory and are not, because under D12 **the owner does not choose the slice at
all** — it is a pure function of `(chain nonce, window, pool)`. The manifest therefore pins the
half that is knowable in advance and must not move afterwards: the draw parameters, the episode
pins, and a digest of the eligible task pool. The published window file then carries the drawn
slice and the catalog snapshot, and a validator **re-derives the slice and compares** rather than
trusting it (`verify`). That is strictly stronger than a per-window hash: an owner who wanted a
flattering slice would have to change a parameter or the pool, and both are chained.

**The residual is the nonce, and it is a caller's responsibility, not this module's.** Whoever picks
the nonce picks the slice. `verify` re-derives from the nonce *in the record*; the caller must check
that nonce against the chain-derived value for the window, exactly as `koth/reference.py` does.

WHAT D12 REMOVES. No burn ledger, no disjointness bookkeeping, no "this task is spent" state
(M7a exit 2). Tasks are reusable; freshness comes from unpredictability, not scarcity. These are
public benchmarks a miner can download in full anyway, so withholding which twenty were scored never
denied them anything — what protects the duel is that the slice is unknown at commit time. And §2.1
already removed the thing worth memorising: the Conductor cannot emit answer text to a worker, so a
model that has memorised every solution in a benchmark gains nothing from it.

THE THREE SLICE INVARIANTS (§6.3b), each of which a scoring rule silently depends on
------------------------------------------------------------------------------------
1. **Every admitted benchmark appears in every slice.** Leave-one-out (§5.2) drops one benchmark at
   a time; a benchmark missing from a window makes LOO run over a different set that window and
   verdicts stop being comparable across windows. Hence a stratified draw — *within* each benchmark,
   never across. A global draw would also let the biggest task pool dominate by size alone.
2. **Roughly equal counts per benchmark**, so §5.1's equal-weight-per-benchmark aggregation and the
   natural per-task reading coincide instead of pulling against each other.
3. **A minimum per benchmark.** Below it `quality_b` is too noisy to carry a twelfth of the score,
   so the window is built WITHOUT that benchmark and the narrowing is RECORDED — a narrowed window
   must be visible in the reveal rather than silently scored over a different set.

FAIL CLOSED (M7 exit 3). Missing file, bad signature, wrong window, unpinned schedule, a slice that
does not re-derive: every one of them raises `WindowError`, and the caller's only correct response
is to skip the window — no duels, no shot spent, challengers roll over with their one submission
intact. Scoring against a stale or substituted window would spend a miner's single shot on a
measurement nobody committed to, which is the "Kind" property broken at its most expensive point.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..gateway import signing
from ..koth import holdout_feed
from ..koth.corpus import category_freshness
from ..koth.reference import open_envelope
from .config import MAX_CONDUCTOR_TOKENS, MAX_STEPS, TEMPERATURE
from .pool import freeze, thaw
from .scaffold import task_order
from .types import Catalog, EpisodeResult, TaskSpec

# The pinned harness identity (§6.2). A window records it so that two windows run under different
# scaffolds are never compared as though the difference were the miners'. Lives here rather than in
# `config.py` only because that module was not this agent's to edit; it belongs with the other pins.
SCAFFOLD_VERSION = "v3-scaffold-1"


class WindowError(Exception):
    """A window that cannot be scored. Every raise means "skip this window" (M7 exit 3)."""


@dataclass(frozen=True)
class Slice:
    """One window's drawn tasks, in the nonce-derived order both arms will run them in.

    `narrowed` is `(benchmark, tasks available)` for every admitted benchmark that could not supply
    its minimum. It is a field rather than a log line because §6.3b requires the narrowing to reach
    the reveal: leave-one-out ran over the benchmarks in `benchmarks`, and a reader comparing this
    window to the next has to be able to see that the sets differ.
    """

    tasks: tuple[TaskSpec, ...]
    counts: tuple[tuple[str, int], ...]      # (benchmark, tasks drawn), sorted by benchmark
    narrowed: tuple[tuple[str, int], ...]    # (benchmark, tasks available) for the dropped ones

    @property
    def benchmarks(self) -> tuple[str, ...]:
        """The benchmarks PRESENT — the set leave-one-out runs over (§6.3b). Derived from `counts`
        so it can never disagree with the tasks that were actually drawn."""
        return tuple(name for name, _ in self.counts)


@dataclass(frozen=True)
class Window:
    """A verified window, in the form the two arms consume it.

    `tasks` are already in the pinned order, so `Scaffold.run_window(w.tasks, nonce=w.nonce, ...)`
    re-derives that same order and both arms are zeroed on the identical tail when an allowance runs
    out (§4). `record` is kept whole because it is what the reveal publishes.
    """

    epoch: int
    nonce: str
    tasks: tuple[TaskSpec, ...]
    catalog: Catalog
    benchmarks: tuple[str, ...]
    record: dict
    # The slice's task ids that appeared in an earlier published reveal's traces — marked when the
    # slice is drawn (`build`) so the reveal can report every arm on seen and unseen rows apart
    # (§5.2c's memorisation meter). A diagnostic; the verdict does not read it.
    seen: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Paired:
    """Both arms after symmetric exclusion (§6.3c), row-aligned and same-length.

    `excluded` is `(task_id, reason)` and is published: §8b.3 requires the exclusion count in the
    reveal, because a window that dropped many tasks has degraded power and the reader must know.
    """

    king: tuple[EpisodeResult, ...]
    challenger: tuple[EpisodeResult, ...]
    excluded: tuple[tuple[str, str], ...]


def window_path(window: int) -> str:
    """Where window N's file lives. A pure function of the window, so a validator addresses it
    without being told anything and authenticity comes from the owner's signature rather than from
    where the bytes were found (`koth/holdout_feed.epoch_path`, same reason)."""
    return f"v3/window/{int(window)}.json"


def pool_digest(tasks: Iterable[TaskSpec]) -> str:
    """One digest over the eligible pool — what the schedule pins so the pool cannot move later.

    Binds EVERY field of every task, not just the IDs. The IDs are what the draw reads, but the
    prompt is what the scaffold forwards byte-for-byte (§2.1), and public benchmarks are patched at
    unchanged IDs. A pool digest over IDs alone would let a window be scored on different text than
    the one the schedule committed to, and nothing downstream would notice. `group` is bound for the
    same reason and is the one field a worker never sees: it is the duel's resample unit (`duel.py`),
    so re-labelling the pool moves every published lower bound.

    Sorted, so it depends on which tasks are eligible and never on the order a caller listed them.
    """
    return signing.sha256_hex(
        sorted([t.benchmark, t.task_id, t.prompt, list(t.tools), t.group] for t in tasks))


def draw(tasks: Iterable[TaskSpec], *, nonce: str, window: int,
         per_benchmark: int, minimum: int) -> Slice:
    """The window's slice: stratified WITHIN each benchmark, seeded by the nonce (§6.3b, D12).

    Ranking by `sha256(nonce|window|benchmark|task_id)` rather than by a seeded shuffle is the
    `scaffold.task_order` choice for the same reason: the result then depends only on which tasks
    are eligible, never on the order the caller listed them in, and it is reproducible from the
    published record by anyone with the public pool — no library's RNG stream in the loop. Including
    the benchmark in the key is what makes stratum 1's draw independent of stratum 2's, so adding a
    thirteenth benchmark does not reshuffle the first twelve (`koth/benchmarks.bench_seed` pins the
    same three inputs).

    `per_benchmark < minimum` is refused rather than run: with it, "drawn below the minimum" and
    "available below the minimum" stop being the same condition, and the window would enforce a
    noise floor it violates on every stratum.
    """
    if per_benchmark < minimum:
        raise WindowError(f"per_benchmark={per_benchmark} is below minimum={minimum}: every "
                          f"stratum would be drawn below the floor the minimum defines")
    by_benchmark: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        by_benchmark.setdefault(task.benchmark, []).append(task)

    picked: list[TaskSpec] = []
    counts: list[tuple[str, int]] = []
    narrowed: list[tuple[str, int]] = []
    for benchmark in sorted(by_benchmark):
        stratum = by_benchmark[benchmark]
        if len(stratum) < minimum:
            # Built WITHOUT it, and recorded (§6.3b invariant 3). Never silent: leave-one-out is
            # about to run over a different set of benchmarks than last window's.
            narrowed.append((benchmark, len(stratum)))
            continue
        ranked = sorted(stratum, key=lambda t: _draw_key(nonce, window, t.benchmark, t.task_id))
        take = ranked[:per_benchmark]
        picked.extend(take)
        counts.append((benchmark, len(take)))

    if not picked:
        raise WindowError("no benchmark could supply its minimum task count")
    return Slice(tasks=task_order(picked, nonce), counts=tuple(counts), narrowed=tuple(narrowed))


def schedule_entry(window: int, *, tasks: Iterable[TaskSpec],
                   per_benchmark: int, minimum: int, digest: str | None = None) -> dict:
    """The pre-committable half of one window: everything that must not move after challengers
    commit, and nothing that is unknowable before the window opens.

    The catalog and the nonce are deliberately absent — both come into existence at window open,
    which is why the record and the schedule are two objects rather than one.

    The index key is `epoch` because that is the key `holdout_feed.manifest` chains on; a v3 window
    is that machinery's epoch unit, and forking the chain to rename a field would be exactly the
    reinvention §6.3 forbids.
    """
    return {"v": 1,
            "epoch": int(window),
            "scaffold_version": SCAFFOLD_VERSION,
            "max_steps": MAX_STEPS,
            "temperature": TEMPERATURE,
            "max_conductor_tokens": MAX_CONDUCTOR_TOKENS,
            "per_benchmark": int(per_benchmark),
            "minimum": int(minimum),
            "pool_digest": pool_digest(tasks) if digest is None else digest}


def schedule(windows: Iterable[int], *, tasks: Iterable[TaskSpec],
             per_benchmark: int, minimum: int) -> list[dict]:
    """Entries for a whole run, to be chained into ONE commitment by `holdout_feed.manifest`.

    Under D12 the entries differ only in their index — the slice is not chosen here, it is derived
    at window open from a nonce nobody controls. That is the point: what the chain pins is the
    parameters and the pool, so the owner keeps no discretion at scoring time.
    """
    pool = tuple(tasks)
    # ONE digest for the whole schedule. `schedule` hands every entry the SAME pool and
    # `pool_digest` is pure, so the per-entry call recomputed a bit-identical value once per
    # window: measured 50.3 ms at a 6812-task pool, i.e. 18.4 s over a 365-window schedule, paid
    # blocking in the validator's `__post_init__` on every start — including the restart path
    # 8b.5 exists to make cheap. `schedule_entry` keeps computing its own when called standalone.
    digest = pool_digest(pool)
    return [schedule_entry(w, tasks=pool, per_benchmark=per_benchmark, minimum=minimum,
                           digest=digest)
            for w in windows]


def build(entry: dict, *, tasks: Iterable[TaskSpec], catalog: Catalog, nonce: str,
          revealed: Iterable[str] = ()) -> dict:
    """The self-contained window file (§6.2), for the owner to sign and publish at window open.

    `revealed` is the task IDs scored in earlier windows — it feeds the published freshness numbers
    only, never the draw. Under D12 a repeat is legal; what matters is that it is *visible*.
    """
    pool = tuple(tasks)
    if pool_digest(pool) != entry.get("pool_digest"):
        raise WindowError("the task pool is not the one this window's schedule entry pinned")
    sliced = draw(pool, nonce=nonce, window=entry["epoch"],
                  per_benchmark=entry["per_benchmark"], minimum=entry["minimum"])
    already = set(revealed)
    return {"v": 1,
            "epoch": entry["epoch"],
            "schedule": entry,
            "nonce": nonce,
            # The action space, frozen (§6.2). Without it the two arms face different worlds and the
            # duel is not paired.
            "catalog": freeze(catalog),
            "freshness": _freshness(sliced.tasks, pool, revealed),
            # Which of the drawn tasks an earlier reveal already showed (D15 traces are public, so
            # these are the rows a miner could have trained on). Carried, like `freshness`, not
            # verified: it is computed from reveal history the verifier may not hold.
            "seen": sorted(t.task_id for t in sliced.tasks if t.task_id in already),
            **_slice_body(sliced)}


def verify(record: dict, *, window: int, man: dict, tasks: Iterable[TaskSpec]) -> Window:
    """Bind a fetched window to the committed schedule, or refuse it (M7 exits 1-3).

    The check is THREE things and not two (§6.3): the schedule entry is in the committed manifest,
    the entry's own window index equals the window being scored, and the published slice re-derives
    from that entry and the record's nonce. The slice is RE-DERIVED from the record's nonce and the
    validator's own copy of the public pool, then compared. Trusting the published IDs would leave
    the owner exactly the discretion §6.3 exists to remove; re-deriving means a swapped slice is
    caught by arithmetic rather than by a hash of the owner's own choosing.

    What is deliberately NOT checked here: that the nonce is the chain's. Whoever picks the nonce
    picks the slice, so the caller must compare it against the chain-derived value for this window
    (`koth/reference.py` does the same). And the freshness block is a published diagnostic computed
    from reveal history the validator may not hold, so it is carried, not verified.
    """
    if not isinstance(record, dict) or record.get("v") != 1:
        raise WindowError("not a window record")
    if record.get("epoch") != int(window):
        raise WindowError(f"window file describes {record.get('epoch')!r}, not window {window}")
    entry = record.get("schedule")
    if not isinstance(entry, dict) or not holdout_feed.in_manifest(entry, man):
        raise WindowError(f"window {window} is not the one pinned in the committed schedule")
    # Membership is NOT enough (§6.3). The manifest chains N entries and all N are equally
    # authentic, so an owner free to put window 23's entry inside window 7's file gets N candidate
    # slices to pick from AFTER challengers have committed — the draw is a pure function of the
    # entry (D12), so N entries are N slices — and every signature and every hash still verifies.
    # That is the chained manifest's whole purpose restored to the owner, silently.
    if entry.get("epoch") != int(window):
        raise WindowError(f"window {window}'s file carries the schedule entry for "
                          f"{entry.get('epoch')!r}: N entries are N slices")

    pool = tuple(tasks)
    if pool_digest(pool) != entry["pool_digest"]:
        raise WindowError(f"window {window} was scheduled against a different task pool")
    nonce = record.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        raise WindowError(f"window {window} carries no nonce, so its slice cannot be re-derived")

    sliced = draw(pool, nonce=nonce, window=entry["epoch"],
                  per_benchmark=entry["per_benchmark"], minimum=entry["minimum"])
    if _slice_body(sliced) != {key: record.get(key) for key in _SLICE_KEYS}:
        raise WindowError(f"window {window}'s slice does not re-derive from its nonce")

    try:
        catalog = thaw(record["catalog"])
    except Exception as exc:                     # noqa: BLE001 — any unreadable catalog is fatal
        raise WindowError(f"window {window} has no readable catalog snapshot: {exc}") from exc
    if not catalog.entries:
        raise WindowError(f"window {window}'s catalog snapshot is empty: there is no action space")

    drawn = {t.task_id for t in sliced.tasks}
    return Window(epoch=entry["epoch"], nonce=nonce, tasks=sliced.tasks, catalog=catalog,
                  benchmarks=sliced.benchmarks, record=record,
                  seen=frozenset(str(t) for t in (record.get("seen") or ()) if str(t) in drawn))


def load(window: int, *, fetch_bytes, owner_ss58: str, verify_sig, man: dict,
         tasks: Iterable[TaskSpec]) -> Window:
    """Fetch, authenticate, verify — the validator's whole window hand-off, fail-closed.

    One function and one exception type on purpose. Composed at each call site instead, some caller
    eventually forgets one of the three checks, and the failure mode of a forgotten check here is
    scoring a window nobody committed to. `fetch_bytes(path) -> bytes` is the store seam (R2 in
    production, `holdout_feed.publish`'s counterpart); everything it can raise becomes a
    `WindowError`, because "missing" and "unverifiable" have the same correct response: skip.
    """
    try:
        raw = fetch_bytes(window_path(window))
        record = open_envelope(raw, owner_ss58=owner_ss58, verify_sig=verify_sig)
    except Exception as exc:                     # noqa: BLE001 — fail closed on ANY failure
        raise WindowError(f"window {window} is missing or unauthentic: {exc}") from exc
    return verify(record, window=window, man=man, tasks=tasks)


def exclude(king: Sequence[EpisodeResult], challenger: Sequence[EpisodeResult],
            *, failed: Iterable[str] = ()) -> Paired:
    """Symmetric task exclusion (§6.3c): a task drops from BOTH arms or from neither.

    `failed` is the tasks whose GRADER or sandbox failed — §8b.3's other half. A worker model
    erroring is an outcome and belongs in the score; Docker dying is a fact about the validator's
    infrastructure, and scoring it zero would inject the owner's ops noise into a miner's result.
    Both arms ran the identical task list, so an exclusion that applied to one arm only would
    compare different task sets and hand the other an easier slice.

    A task present in only one arm is excluded for the same reason — that IS the asymmetry, arriving
    by a different route (a crashed arm rather than a named grader failure), and it is the one this
    rule would miss if `failed` were the only input.

    Output is row-aligned in the king's order, which is the window's pinned order: `duel
    .BenchmarkPair` pairs positionally, so alignment here is what its length check has to find.
    """
    _refuse_duplicates(king, "king")
    _refuse_duplicates(challenger, "challenger")
    by_challenger = {result.task_id: result for result in challenger}
    by_king = {result.task_id: result for result in king}
    dropped = {task_id: "grader" for task_id in failed}
    for task_id in by_king:
        if task_id not in by_challenger:
            dropped.setdefault(task_id, "missing from challenger arm")
    for task_id in by_challenger:
        if task_id not in by_king:
            dropped.setdefault(task_id, "missing from king arm")

    kept = [task_id for task_id in by_king if task_id not in dropped]
    return Paired(king=tuple(by_king[task_id] for task_id in kept),
                  challenger=tuple(by_challenger[task_id] for task_id in kept),
                  excluded=tuple(sorted(dropped.items())))


_SLICE_KEYS = ("task_ids", "benchmarks", "counts", "narrowed")


def _slice_body(sliced: Slice) -> dict:
    """The slice as the record stores it — lists, so a JSON round trip compares equal to itself and
    `verify` can diff a re-derived draw against a fetched file without a shape conversion."""
    return {"task_ids": [t.task_id for t in sliced.tasks],
            "benchmarks": list(sliced.benchmarks),
            "counts": [[name, n] for name, n in sliced.counts],
            "narrowed": [[name, n] for name, n in sliced.narrowed]}


def _freshness(picked: Sequence[TaskSpec], pool: Sequence[TaskSpec],
               revealed: Iterable[str]) -> dict:
    """What this window re-asks (M7 exit 4) — published, never a gate.

    `category_freshness` is `koth/corpus.py`'s, with the benchmark as the category (§5.3: "the
    category becomes the benchmark"). `staleness` is NOT that module's median-max-cosine: §9
    deprecates the pinned MiniLM encoder that instrument needed, and under D12 the exact question is
    available anyway — tasks recur by design, so the fraction of this slice already scored in an
    earlier window is measurable from labelled IDs rather than estimated in embedding space. It is
    the number M7a's residual risk makes worth watching: a miner who profiles the pool
    (~$60k) is buying exactly the tasks this fraction counts.

    None when nothing has been revealed yet, matching `corpus.staleness` — window 1 has nothing to
    be stale against, and 0.0 would read as a measured freshness rather than an absent one.
    """
    seen = set(revealed)
    benchmark_of = {t.task_id: t.benchmark for t in pool}
    tags = [t.benchmark for t in picked]
    revealed_tags = sorted({benchmark_of[task_id] for task_id in seen if task_id in benchmark_of})
    repeats = sum(1 for t in picked if t.task_id in seen)
    return {"category_freshness": category_freshness(tags, revealed_tags),
            "staleness": (repeats / len(picked)) if seen else None}


def _draw_key(nonce: str, window: int, benchmark: str, task_id: str) -> tuple[str, str]:
    return signing.sha256_hex(f"{nonce}|{window}|{benchmark}|{task_id}"), task_id


def _refuse_duplicates(results: Sequence[EpisodeResult], arm: str) -> None:
    """A repeated task ID would make positional pairing ambiguous, and the wrong pairing is not
    visible in any downstream number — so it is refused loudly here."""
    ids = [r.task_id for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{arm} arm reports the same task twice; the arms cannot be paired")
