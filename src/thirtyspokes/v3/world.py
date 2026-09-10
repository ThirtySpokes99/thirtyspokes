"""The production world — what `orchestra-validator --world` resolves to (§5.1b, §5.5, §6.2).

`validator.main` requires `--world module:attr`: a zero-argument callable returning the published
`Pins` — the lambda/C table, King0 and its contrast, over the real corpus. Nothing in the tree
returned one, because `Pins` lived in `simulate.py` beside the mock world it was built for, and its
contents are measurements rather than constants somebody can write down. So the validator was
unrunnable in production for a reason no error message named: the flag was required, and there was
nothing to pass it.

THE TABLE IS LOADED, NEVER FITTED HERE. `lambda_b` is pinned per corpus and published (§5.1b),
precisely so scores stay comparable across windows and the target does not move under miners. A
module that derived it at startup would re-derive it on every restart, from whatever catalog and
corpus happened to be current — which is the same number drifting silently, dressed as a constant.
`scripts/m3a_spread.py` measures it once and writes `m3a-exchange.json`; this reads that file.

KING0 IS CHOSEN BY THE TABLE, NOT DECLARED. §5.5 says King0 is the best FIXED policy, because the
real alternative to running this subnet is shipping a fixed cascade — if a trained Conductor cannot
beat one, the subnet pays for something the owner could deploy for free. Which policy that is, is an
M3a finding, so `pins` picks it by `final` under the loaded table rather than taking anyone's word.

AND THE CONTRAST IS `reference.contrast_for`, NOT A RULE OF THIS MODULE'S OWN. §5.6 requires the
reference pair to differ in the ACTION IT TAKES rather than in its label — two policies whose
episodes end on the same model measure a spread of exactly zero, so every window refuses itself while
the corpus is perfectly healthy and the arena never opens. That rule, its measured counterexample
(cascade vs always-strongest, +0.0000) and its refusal of ladders it did not build already live in
`reference.contrast_for`. Picking the pair again here would be a second copy of the one decision
whose first copy was already wrong once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import reference
from .benchmarks.base import Benchmark
from .devkit import LocalWorld, suite_grade, suite_inspect
from .openrouter import OpenRouterClient
from .pool import Catalog
from .reference import FixedPolicy
from .score import Exchange
from .types import TaskSpec


class WorldError(RuntimeError):
    """A production world that cannot be built. Never a repair: every fallback available here would
    substitute a number the owner did not measure into the price of a crown."""


def load_exchange(path: Path | str) -> dict[str, Exchange]:
    """The published lambda/C table, as `scripts/m3a_spread.py` writes it.

    A benchmark carrying flags is kept rather than dropped: `score.score_arm` excludes a flagged
    benchmark itself, and it must be able to say WHICH one and why (§5.1b requires the exclusion to
    be announced). Dropping it here would make the exclusion silent, which is the failure the flag
    exists to prevent.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise WorldError(f"{path} is not a published lambda/C table")
    table: dict[str, Exchange] = {}
    for benchmark, row in raw.items():
        if benchmark.startswith("_"):            # metadata, not a benchmark: `_pool` (`load_pool`)
            continue
        try:
            table[benchmark] = Exchange(benchmark=benchmark, lam=float(row["lam"]),
                                        c_per_task=float(row["c_per_task"]),
                                        flags=tuple(row.get("flags", ())))
        except (KeyError, TypeError, ValueError) as exc:
            raise WorldError(f"{path}: benchmark {benchmark!r} is not a usable row: {exc}") from exc
    return table


def load_pool(path: Path | str) -> tuple[str, ...] | None:
    """The pool the table was fitted on, as `scripts/m3a_spread.py` records it under `_pool`.

    None for a table that predates the record. Measured 2026-09-07: without it the daemon priced
    spend at the probe pool's slope while King₀ climbed the whole catalog to `openai/gpt-4`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = raw.get("_pool") if isinstance(raw, dict) else None
    if pool is None:
        return None
    if not isinstance(pool, list) or not all(isinstance(m, str) for m in pool) or len(pool) < 2:
        raise WorldError(f"{path}: `_pool` must list at least two model ids, got {pool!r}")
    return tuple(pool)


def _policy(name: str, catalog: Catalog) -> FixedPolicy:
    """One fixed policy by the name `reference` already uses for it."""
    builders = {reference.ALWAYS_CHEAPEST: reference.always_cheapest,
                reference.ALWAYS_STRONGEST: reference.always_strongest,
                reference.CASCADE: reference.cascade}
    if name not in builders:
        raise WorldError(f"{name!r} is not a fixed policy this module can build; expected one of "
                         f"{sorted(builders)}")
    return builders[name](catalog)


def pins(*, exchange_path: Path | str, catalog: Catalog, benchmarks: Sequence[Benchmark],
         tasks: Sequence[TaskSpec], worker, quality: Mapping[str, float] | None = None,
         pool: Sequence[str] | None = None):
    """Assemble the published `Pins`. Imported lazily so `simulate` stays the only owner of the type.

    `quality` is M3a's measured `final` per fixed policy, used to pick King0 (§5.5). Omitted, the
    cascade is used and SAID SO — the plan's own expectation, not a silent default, because the
    policy that holds the throne decides what every challenger must beat.
    """
    from .simulate import Pins
    from .pool import CatalogError, narrow

    table = load_exchange(exchange_path)
    # THE ARMS RUN ON THE POOL λ WAS FITTED ON. `pool` is the table's own `_pool` record; without
    # narrowing, King₀'s cascade climbs today's whole catalog to its dearest rung and the price
    # says nothing about the ladder — measured 2026-09-07 as `openai/gpt-4` on the first launch.
    if pool is not None:
        try:
            catalog = narrow(catalog, pool)
        except CatalogError as exc:
            raise WorldError(f"the lambda table was fitted on a pool this catalog cannot supply: "
                             f"{exc}") from exc
    missing = {task.benchmark for task in tasks} - set(table)
    if missing:
        raise WorldError(
            f"the lambda/C table prices no {sorted(missing)}, but the corpus draws from it. A "
            f"benchmark scored under a table that never measured it is priced by a constant nobody "
            f"derived; re-run M3a over the whole corpus before launching.")

    # §5.1b EXCLUDES a flagged benchmark from scoring, so a table where every benchmark carries a
    # flag builds a world whose `final` is a mean over nothing. That is not a quiet edge case: the
    # section says outright that if every benchmark fires "there is no corpus to launch on ... the
    # response is to refuse to launch, not to score the whole arena on unpriced quality". Measured
    # 2026-09-03, M3a flagged BOTH admitted benchmarks as dominators, so this is the live case and
    # not a hypothetical.
    priced = [b for b, row in table.items() if not row.flags
              and b in {task.benchmark for task in tasks}]
    if not priced:
        flagged = "\n".join(f"    {b}: {'; '.join(table[b].flags)}" for b in sorted(table)
                             if table[b].flags)
        raise WorldError(
            "every benchmark in this corpus is excluded by its own lambda guard, so `final` would "
            "be a mean over nothing and every duel would compare two empty scores:\n" + flagged +
            "\n\n§5.1b: refuse to launch rather than score the arena on unpriced quality. This is a "
            "finding about the model market, not a configuration error — re-run M3a on a pool where "
            "paying more buys quality, or admit a benchmark where it does.")

    # ONE priced benchmark is not a launchable corpus either, and it used to pass this gate.
    # `duel` refuses a slice it cannot judge breadth on — leave-one-out over a single benchmark
    # scores the challenger on an empty slice — so with one benchmark admitted, `preflight` reports
    # green and then EVERY duel raises, caught per challenger by `Validator._admit`'s broad handler
    # and published as a refusal of the miner. The corpus would read as a field of broken
    # challengers. Measured 2026-09-05: the full M3a run prices livecodebench and flags hle, which
    # is exactly one, so this is the live case and not a hypothetical.
    if len(priced) < 2:
        drawn = {task.benchmark for task in tasks}
        flagged = [f"    {b}: {'; '.join(table[b].flags)}" for b in sorted(table) if table[b].flags]
        # Measured 2026-09-07 on the first live launch: the table priced two benchmarks and the
        # slice drew ONE, and this refusal said "the benchmarks this corpus lost:" over an empty
        # list. A priced benchmark that was not drawn is a different mistake from one the guard
        # excluded, and it has a different fix, so it is named as what it is.
        not_drawn = [f"    {b}: priced (lambda={table[b].lam:+.4f}) but not in this slice"
                     for b in sorted(table) if not table[b].flags and b not in drawn]
        raise WorldError(
            f"only {priced[0]!r} is priced AND drawn, and a duel cannot be judged on one "
            f"benchmark: `duel` refuses a slice whose leave-one-out would score a challenger on an "
            f"empty slice, so every window would raise while this gate said the launch was fine.\n"
            + ("Excluded by their own guard:\n" + "\n".join(flagged) + "\n" if flagged else "")
            + ("Priced by the table but absent from the corpus — draw them:\n"
               + "\n".join(not_drawn) + "\n" if not_drawn else "")
            + "\n§5.2's breadth rule is what makes a crown a statement about routing rather than "
              "about one benchmark: draw at least two priced benchmarks, admit a second that prices, "
              "or re-run M3a on a pool where a flagged one does.")

    chosen = max(quality, key=lambda name: quality[name]) if quality else reference.CASCADE
    king_zero = _policy(chosen, catalog)
    contrast = _policy(reference.contrast_for(chosen), catalog)
    # Belt and braces on the one failure that is invisible in production: a pair sharing a top rung
    # scores +0.0000 forever and the arena never opens, which reads as a quiet validator rather than
    # as an error. `contrast_for` prevents it by construction; this asserts it on the real catalog,
    # where two names can still collapse to one model if the pool has a single usable entry.
    if king_zero.rungs[-1:] == contrast.rungs[-1:]:
        raise WorldError(
            f"the reference pair does not separate: King0 ({chosen}) and its contrast "
            f"({contrast.name}) both end on {king_zero.rungs[-1]}, so their spread is zero by "
            f"construction and the power gate would refuse every window while the corpus is "
            f"healthy (§5.6). The pool is too narrow to measure a band.")

    world = LocalWorld(catalog=catalog, tasks=tuple(tasks), grade=suite_grade(benchmarks),
                       worker=worker, exchange=table, inspect=suite_inspect(benchmarks))
    return Pins(world=world, king_zero=king_zero, contrast=contrast)


def from_env():
    """`--world thirtyspokes.v3.world:from_env` — the whole production world from the environment.

    Every input is named by an environment variable rather than a default, because each one is a
    thing an owner must have measured or chosen: `V3_EXCHANGE` is M3a's published table,
    `V3_BENCHMARKS` the admitted corpus, `V3_TASKS_PER_BENCHMARK` the slice width the pool is
    drawn from. A default here would be this module choosing the price of a crown.
    """
    import os

    from .benchmarks.hle import HumanitysLastExam
    from .benchmarks.real import LCB_RELEASES, LiveCodeBench
    from .pool import fetch, snapshot

    # §5.5: King0 is the best FIXED policy as MEASURED, because the real alternative to running this
    # subnet is shipping that policy for free. `pins` falls back to the cascade when handed no
    # measurement, which is the plan's expectation but not a measurement — and King0 is what every
    # challenger must beat, so defaulting it silently prices every crown against a policy nobody
    # checked. The production seam therefore requires the name M3a actually reported.
    # Every required variable is read BEFORE anything is opened, so an absent one stops the daemon
    # naming itself rather than surfacing as a file error on a path that was never set. Measured
    # 2026-09-07: reading the table's pool first turned a missing OPENROUTER_API_KEY into
    # `FileNotFoundError` on whatever V3_EXCHANGE happened to hold.
    exchange_path = os.environ["V3_EXCHANGE"]
    api_key = os.environ["OPENROUTER_API_KEY"]
    king_zero = os.environ.get("V3_KING0")
    if not king_zero:
        raise WorldError(
            "V3_KING0 is unset. §5.5 makes King0 the best FIXED policy as measured, and this is "
            "the seam that scores real windows: read the winner off M3a's report and name it "
            f"explicitly (one of {sorted((reference.ALWAYS_CHEAPEST, reference.ALWAYS_STRONGEST, reference.CASCADE))}). "
            "Defaulting it would price every crown against a policy nobody measured.")

    names = os.environ.get("V3_BENCHMARKS", "livecodebench,hle").split(",")
    per = int(os.environ.get("V3_TASKS_PER_BENCHMARK", "125"))
    built: list[Benchmark] = []
    tasks: list[TaskSpec] = []
    for name in (n.strip() for n in names if n.strip()):
        if name == "livecodebench":
            bench = LiveCodeBench(releases=LCB_RELEASES)
            tasks.extend(bench.draw_across_releases(per, seed=os.environ.get("V3_SEED", "v3")))
        elif name == "hle":
            bench = HumanitysLastExam()
            tasks.extend(bench.load()[:per])
        else:
            raise WorldError(f"no adapter for benchmark {name!r}")
        built.append(bench)
    pool = load_pool(exchange_path)
    if pool is None:
        raise WorldError(
            f"{exchange_path} records no `_pool`, so the ladder it prices is unknown and the daemon "
            f"would run King0 over the whole catalog at a price fitted on five models. Re-render "
            f"it: `scripts/m3a_spread.py --report-only` writes the pool.")
    return pins(exchange_path=exchange_path, catalog=snapshot(fetch()), benchmarks=built,
                tasks=tasks, quality={king_zero: 1.0}, pool=pool,
                worker=OpenRouterClient(api_key))
