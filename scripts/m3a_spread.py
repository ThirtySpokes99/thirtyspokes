#!/usr/bin/env python
"""M3a — the spread smoke, and the only thing that produces the constants the scorer needs.

    Is there GROSS spread between fixed policies on this traffic?

It is a coarse question on purpose (the build plan M3a): at ~15 tasks per benchmark the detectable
difference is roughly 12 points, so this sees large effects and is blind to small ones. That is
well matched to what it decides — with `EPS = 0.05` and lambda-pricing, a spread under ~12 points
leaves almost nothing for a miner to compete for.

WHAT IT PRODUCES, WHICH IS MORE THAN A VERDICT. `validator.main` requires `--world module:attr`
returning the published `Pins`: the lambda/C table, King0 and its contrast. There is no such module in
the tree, because its contents are measurements and this is the measurement. So this script writes
`m3a-exchange.json` alongside its report, and that file is what a production world is built from.
M3a does not merely gate the launch; nothing can be scored until it has run.

THE POOL IS PINNED, NOT CHOSEN HERE. the spend plan §4.2 froze five models against a live
catalog, checked each honours `temperature` (§4.3), and recorded why each is in the list. Re-deriving
"the cheap models" from today's catalog would silently re-decide that, and the four-figure arm §4.1
documents is exactly what a re-derivation produced last time. Every id is verified present in today's
catalog and a missing one REFUSES rather than substitutes: a quietly swapped rung changes what the
lambda table prices.

WHAT THIS COSTS AND WHAT STOPS IT. Three arms over `--tasks-per-benchmark` tasks on a cheap+mid pool,
budgeted per arm and refused before any call if the estimate exceeds `--budget-usd`. `--dry-run`
prints the estimate and calls nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thirtyspokes.v3 import pool as pool_mod                                    # noqa: E402
from thirtyspokes.v3 import reference, score                                    # noqa: E402
from thirtyspokes.v3.config import EPISODE_CONCURRENCY, N_BOOT                          # noqa: E402
from thirtyspokes.v3.benchmarks.hle import HumanitysLastExam                    # noqa: E402
from thirtyspokes.v3.benchmarks.real import LCB_RELEASES, LiveCodeBench         # noqa: E402
from thirtyspokes.v3.devkit import suite_grade, suite_inspect                   # noqa: E402
from thirtyspokes.v3.openrouter import OpenRouterClient                         # noqa: E402
from thirtyspokes.v3.pool import Catalog                                        # noqa: E402
from thirtyspokes.v3.scaffold import Scaffold                                   # noqa: E402
from thirtyspokes.v3.types import EpisodeResult, TaskSpec                       # noqa: E402
from thirtyspokes.v3.validator import episode_json                              # noqa: E402

PROBE_POOL = (
    "qwen/qwen3-30b-a3b-instruct-2507",          # always-cheapest, cascade rung 1
    "mistralai/mistral-small-3.2-24b-instruct",  # filler
    "qwen/qwen3-coder",                          # cascade rung 2
    "z-ai/glm-4.7",                              # filler
    "anthropic/claude-haiku-4.5",                # always-strongest, cascade rung 3
)
"""the spend plan §4.2, frozen there against a live catalog with each model's role recorded.

Price-sorted, so `reference.cascade(pool, rungs=3)` takes entries 0, 2 and 4 and yields exactly the
ladder that document specifies with no special-casing. Cheapest to dearest spans 25x, entirely inside
cheap+mid: the frontier band is one to two orders above the top rung, and a frontier episode is
~$7.50 against ~$0.04 here, which is the whole difference between this test and the four-figure one
an earlier draft of the plan proposed.
"""

# A frontier-free pool still has a worst case, so the estimate is per episode rather than per token:
# measured cheap episodes run ~$0.04 and the cascade's top rung is ~25x the floor, so $0.35 is a
# deliberately pessimistic per-episode ceiling for the refusal below rather than a prediction.
#
# IT IS A CEILING AND NOT A FORECAST, AND THE DIFFERENCE IS LARGE ENOUGH TO MATTER. Measured
# 2026-09-02 on a 6-episode HLE smoke: **$0.00183 per episode**, 190x below this bound. A knowledge
# benchmark answers in a few hundred tokens; a code benchmark writes a program and grades it, so the
# same number will not hold there. Overriding this with `--episode-usd` is therefore how a run is
# sized from measurement rather than from a bound, and leaving it alone is how an UNMEASURED
# benchmark stays sized from the bound.
PESSIMISTIC_EPISODE_USD = 0.35


@dataclass(frozen=True)
class Arm:
    """One fixed policy's run over the whole slice."""

    name: str
    results: tuple[EpisodeResult, ...]

    def quality(self, benchmark: str | None = None) -> float:
        rows = self._rows(benchmark)
        return statistics.fmean(r.graded_score for r in rows) if rows else 0.0

    def spend(self, benchmark: str | None = None) -> float:
        return sum(r.spend_usd for r in self._rows(benchmark))

    def _rows(self, benchmark: str | None) -> tuple[EpisodeResult, ...]:
        return tuple(r for r in self.results
                     if benchmark is None or r.benchmark == benchmark)


def probe_pool(catalog: Catalog) -> Catalog:
    """Today's catalog narrowed to the pinned five — through `pool.narrow`, the same narrowing the
    production world applies from the table's own `_pool` record, so the ladder λ is fitted on and
    the ladder the daemon prices are one definition."""
    try:
        return pool_mod.narrow(catalog, PROBE_POOL)
    except pool_mod.CatalogError as exc:
        raise SystemExit(f"M3a refuses to run: {exc}") from exc


def slice_for(bench_names: Sequence[str], per_benchmark: int, *, seed: str):
    """The task slice and the graders for it, per benchmark.

    LiveCodeBench draws STRATIFIED ACROSS ITS SIX RELEASES rather than from the newest file alone.
    The census (the LCB census) asks for exactly this: the cost of this test is driven by the
    task count and not by which tasks, so a balanced draw answers the enable question for money that
    was going to be spent anyway. Each task carries its release, so the report can span by release.
    """
    benchmarks, tasks = [], []
    for name in bench_names:
        if name == "livecodebench":
            bench = LiveCodeBench(releases=LCB_RELEASES)
            drawn = bench.draw_across_releases(per_benchmark, seed=seed)
        elif name == "r2egym":
            # DeepSWE on the owner's list of twelve. Drawn by seed from the whole 6,812 rather than
            # from `usable_ids()`, so the draw is a property of the seed and not of which images this
            # box happens to hold — `scripts/provision_r2egym.py` pulls exactly this slice, and a
            # missing image is then a loud refusal at grade time rather than a silently different
            # corpus on every machine.
            from thirtyspokes.v3.benchmarks.r2egym import R2EGym
            bench = R2EGym()
            loaded = list(bench.load())
            random.Random(f"{seed}:r2egym").shuffle(loaded)
            drawn = tuple(loaded[:per_benchmark])
        elif name == "swebench_pro":
            # Drawn from `usable_ids()` — the instances whose images the grading daemon holds — and
            # NOT from the 731 by seed as r2egym is, because the corpus is disk-bound by ~50x
            # (the corpus record, the adapter's own arithmetic) and a seeded draw over the
            # whole set would name tasks no box can grade. The question this probe answers is
            # whether the gradeable stratum prices under the King0 pair; which stratum is gradeable
            # is recorded by the provisioning step, not chosen here.
            from thirtyspokes.v3.benchmarks.swebench_pro import SweBenchPro
            bench = SweBenchPro()
            usable = set(bench.usable_ids())
            if not usable:
                raise SystemExit("swebench_pro: no instance image is present on the grading "
                                 "daemon; pull the provisioned tags first (V3_DOCKER_HOST)")
            loaded = [t for t in bench.load() if t.task_id in usable]
            random.Random(f"{seed}:swebench_pro").shuffle(loaded)
            drawn = tuple(loaded[:per_benchmark])
        elif name == "hle":
            bench = HumanitysLastExam()
            loaded = list(bench.load())
            # Seeded shuffle, not `loaded[:n]`: the first N rows of a published dataset are an
            # ordering somebody chose, and a spread measured on them could be a property of that
            # ordering. LiveCodeBench above already draws by seed; this matches it.
            random.Random(f"{seed}:hle").shuffle(loaded)
            drawn = tuple(loaded[:per_benchmark])
        else:
            raise SystemExit(f"unknown benchmark {name!r}; M3a runs r2egym, hle, livecodebench, swebench_pro")
        benchmarks.append(bench)
        tasks.extend(drawn)
    return benchmarks, tuple(tasks)


def arm_policies(catalog: Catalog) -> dict:
    """The three fixed policies by name, in the order an arm is bought.

    A dict rather than the inline tuple it replaces, because `--arms` selects from it: the report's
    own reliability guard tells the operator to "re-run the flagged rung before pinning lambda" and
    there was no way to do that — `run_arm` truncates its file, so the only path was re-buying all
    three arms to replace one.
    """
    return {"always_cheapest": reference.always_cheapest(catalog),
            "always_strongest": reference.always_strongest(catalog),
            "cascade": reference.cascade(catalog, rungs=3)}


ARM_NAMES = ("always_cheapest", "always_strongest", "cascade")

# This script names arms by FILE STEM (`episodes-always_cheapest.jsonl`) and `reference` names
# policies with hyphens (`always-cheapest`). `fit_pool` keys on the latter, so the two spellings meet
# exactly here, in both directions — measured 2026-09-07: the first re-render under the King0 fit
# refused with "missing ['always-cheapest', 'always-strongest'] from ['always_cheapest', ...]".
_TO_REF = {"always_cheapest": reference.ALWAYS_CHEAPEST,
           "always_strongest": reference.ALWAYS_STRONGEST,
           "cascade": reference.CASCADE}
_FROM_REF = {ref: name for name, ref in _TO_REF.items()}


def table_to_pin(arms: Sequence[Arm], fit: reference.PoolFit) -> tuple[dict, str]:
    """The exchange table `m3a-exchange.json` carries, and one line saying which fit it is.

    The report has said "<- the fit to pin if the rung is not re-run" beside the error-excluded λ
    since 2026-09-05 while the JSON kept the RAW fit — advice in the report, the other number in the
    artifact `world.pins` actually loads. A 429 is scored as the model answering nothing, so the raw
    fit prices the provider's throttle as the pool's slope; measured on run 3 that is λ=0.186 raw
    against 0.090 with the errored tasks dropped from both arms. When provider errors are present
    the artifact is the error-excluded table; when there are none the two are identical.
    """
    clean, kept = exchange_without_errors(arms, fit.exchange, fit.top)
    if clean and any(errored(r) for arm in arms for r in arm.results):
        return clean, (f"table written: PROVIDER ERRORS EXCLUDED, paired n={kept} — the fit the "
                       f"report recommends, not the raw one")
    return fit.exchange, "table written: raw fit (no provider errors in either fitted arm)"


def write_table(out: Path, table: Mapping) -> None:
    """The table, WITH the pool it was fitted on. `world.pins` narrows the daemon's catalog to
    `_pool` so King₀'s ladder is the one λ prices; a table without it cannot launch (`from_env`)."""
    rows = {b: {"lam": e.lam, "c_per_task": e.c_per_task, "flags": list(e.flags)}
            for b, e in table.items()}
    rows["_pool"] = list(PROBE_POOL)
    (out / "m3a-exchange.json").write_text(json.dumps(rows, indent=2, sort_keys=True),
                                           encoding="utf-8")


def fit_arms(arms: Sequence[Arm]) -> reference.PoolFit:
    """`reference.fit_pool` over this script's arms, with King0 and the pair spelled as arm names."""
    fit = reference.fit_pool({_TO_REF[arm.name]: arm.results for arm in arms})
    return reference.PoolFit(king_zero=_FROM_REF[fit.king_zero], floor=_FROM_REF[fit.floor],
                             top=_FROM_REF[fit.top], exchange=fit.exchange)


def run_arm(name: str, policy, tasks: Sequence[TaskSpec], catalog: Catalog, *,
            worker, grade, inspect, budget_usd: float, out: Path,
            concurrency: int = EPISODE_CONCURRENCY) -> Arm:
    """One policy over the slice, with every episode written and FLUSHED as it lands (exit 0).

    Incremental because an arm is hours of paid model calls: a crash at task 190 of 200 that lost
    the first 189 would mean buying them all again. `flush` on every line rather than at close,
    since the failure this guards against is the process not reaching close. The encoding is
    `validator.episode_json`'s — the one the reveal publishes — so a saved episode and a published
    one are the same shape rather than two that have to be kept in step.
    """
    scaffold = Scaffold(catalog, policy, worker, grade, inspect=inspect)
    started = time.time()
    path = out / f"episodes-{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        def keep(episode) -> None:
            handle.write(json.dumps(episode_json(episode.result), sort_keys=True) + "\n")
            handle.flush()

        # Concurrent, as `Validator._arm` runs an arm: serial, 248 tasks at tens of seconds each is
        # hours per arm, and this measurement is already the thing everything waits on.
        episodes = scaffold.run_window(tasks, nonce=f"m3a-{name}", budget_usd=budget_usd,
                                       on_episode=keep, concurrency=concurrency)
    results = tuple(e.result for e in episodes)
    print(f"  {name:18s} {len(results):3d} episodes  ${sum(r.spend_usd for r in results):7.4f}  "
          f"quality {statistics.fmean([r.graded_score for r in results] or [0]):.4f}  "
          f"{time.time() - started:6.1f}s")
    return Arm(name=name, results=results)


def disagreement(a: Arm, b: Arm) -> float:
    """The fraction of tasks the two policies score differently — §5.4's unmeasured 20%.

    A paired comparison extracts information only from tasks where the arms disagree, so this is the
    number the task-count table is built on, and it has never been measured for sequential policies
    on agentic work. Reported here as a by-product.
    """
    left = {r.task_id: r.graded_score for r in a.results}
    right = {r.task_id: r.graded_score for r in b.results}
    shared = set(left) & set(right)
    if not shared:
        return float("nan")
    return sum(left[t] != right[t] for t in shared) / len(shared)



def load_arms(out: Path) -> list[Arm]:
    """Rebuild the arms from the saved episode files, so a report can be re-rendered for free.

    The episodes are the expensive artefact and the report is a rendering of them, so a change to
    the rendering — the reliability guard was added after the first run was already in flight —
    must never mean paying for the calls again. `SimpleNamespace` rather than the real dataclasses
    because every consumer here reads attributes and constructs nothing.
    """
    arms = []
    for path in sorted(out.glob("episodes-*.jsonl")):
        results = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            steps = [SimpleNamespace(model_id=st["model_id"], cost_usd=st["cost_usd"],
                                     success=st["success"], parse_failed=st["parse_failed"])
                     for st in raw["steps"]]
            results.append(SimpleNamespace(
                task_id=raw["task_id"], benchmark=raw["benchmark"],
                graded_score=raw["graded_score"], spend_usd=raw["spend_usd"],
                stopped_reason=raw["stopped_reason"], steps=steps))
        arms.append(Arm(name=path.stem.removeprefix("episodes-"), results=tuple(results)))
    if not arms:
        raise SystemExit(f"no episode files in {out}/ to report on")
    return arms


def reliability(arms: Sequence[Arm]) -> dict[str, tuple[int, int]]:
    """Per model: (calls that errored, calls made). THE VALIDITY GUARD, and it is not optional.

    `scaffold` records a worker exception as `success=False, cost_usd=0.0` and a wrong answer as
    `success=False, cost_usd>0`. §8b.3 makes both a failed rung on purpose — a Conductor that keeps
    delegating to a flaky model should be scored for it — but a MEASUREMENT comparing model quality
    cannot treat them alike: a model erroring half the time reports as a bad model, and what was
    actually measured is a provider.

    ROUTING_MEASUREMENTS §16 is the precedent and it cost a whole pass: four of seven pinned models
    returned EMPTY on 39-74% of code problems, empty graded as wrong, and the reading was reported as
    quality. The guard that exists to refuse such a reading was not applied. It is applied here.
    """
    counts: dict[str, list[int]] = {}
    for arm in arms:
        for result in arm.results:
            for step in result.steps:
                if step.model_id is None:
                    continue
                row = counts.setdefault(step.model_id, [0, 0])
                row[1] += 1
                if step.cost_usd == 0.0 and not step.success:
                    row[0] += 1
    return {model: (bad, total) for model, (bad, total) in counts.items()}


def errored(result) -> bool:
    """A step that cost nothing and did not succeed is a provider failure, not a wrong answer."""
    return any(st.cost_usd == 0.0 and not st.success for st in result.steps
               if st.model_id is not None)


def _clean_pairs(arms: Sequence[Arm], top: str = "always_strongest") -> tuple[dict, dict]:
    """The tasks BOTH fitted arms answered without a provider failure, keyed by task id.

    `top` is the arm λ is fitted against — King₀ as `reference.fit_pool` chose it. The default is the
    two-arm case `fit_pool` itself falls back to when no cascade was swept.

    One definition, used by every statistic the report derives from the pair, so the fit and its
    interval can never be computed over different episodes (§6.3c: an infrastructure failure drops
    the task from both arms — dropping it from the erroring arm alone hands the other an easier
    slice).
    """
    by_name = {arm.name: arm for arm in arms}
    if not {"always_cheapest", top} <= by_name.keys():
        return {}, {}
    cheap = {r.task_id: r for r in by_name["always_cheapest"].results}
    strong = {r.task_id: r for r in by_name[top].results}
    keep = [t for t in cheap.keys() & strong.keys()
            if not errored(cheap[t]) and not errored(strong[t])]
    return {t: cheap[t] for t in keep}, {t: strong[t] for t in keep}


def exchange_without_errors(arms: Sequence[Arm], exchange,
                            top: str = "always_strongest") -> tuple[dict, dict]:
    """The lambda table a fit would give with every provider-errored task dropped from BOTH arms,
    and the paired task count each benchmark kept.

    NOT a replacement for the published table, and deliberately not written to disk: §8b.3 makes an
    error a failed rung on purpose and that is right for SCORING a Conductor. It is wrong for
    MEASURING a pool, which is what this fit is — `reliability` already refuses a reading past 25%
    and warns past 10%, but a warning that cannot say HOW FAR the errors moved lambda leaves the
    owner unable to tell a 5% correction from a doubling. Measured 2026-09-05: the cheap rung
    errored on 18.5% of its calls and the raw livecodebench slope came out 2.2x the corrected one.

    Dropped from BOTH arms, pairwise, because that is §6.3c's rule for an infrastructure failure —
    dropping the errored task from one arm alone hands the other an easier slice.

    `c_per_task` is carried over from the published table rather than re-pinned, so the two lambdas
    differ only in the episodes that fed them and stay directly comparable.
    """
    cheap, strong = _clean_pairs(arms, top)
    keep = sorted(cheap)
    if not keep:
        return {}, {}
    kept: dict[str, int] = {}
    for t in keep:
        kept[cheap[t].benchmark] = kept.get(cheap[t].benchmark, 0) + 1
    return score.derive_exchange([cheap[t] for t in keep], [strong[t] for t in keep],
                                 {b: ex.c_per_task for b, ex in exchange.items()}), kept


def lambda_stability(arms: Sequence[Arm], exchange, *, top: str = "always_strongest",
                     n_boot: int = N_BOOT, seed: int = 20260906) -> dict:
    """Per benchmark: lambda's bootstrap interval, and HOW OFTEN EACH GUARD FIRES.

    *** THIS BOUNDS TASK SAMPLING AND NOTHING ELSE. IT DOES NOT ESTABLISH THAT THE FLAG IS
    STABLE, AND IT MUST NOT BE READ THAT WAY. ***

    Measured 2026-09-06, and the measurement is the reason this warning is longer than the code.
    The cheap arm was replicated three times over the identical 248 tasks. livecodebench flagged
    DOMINATOR in TWO draws and priced in the third; pooled over all three the paired difference is
    +0.0102 with a 95% interval of [-0.0438, +0.0642], straddling zero. Yet run 1's own bootstrap —
    this function, on that draw — reported DOMINATOR firing in 3% of resamples, which reads as
    settled. It is not wrong; it is answering a different question. Resampling TASKS cannot see
    variance BETWEEN RUNS, and between-run variance is what moved this flag: the worker is
    nondeterministic at `temperature: 0.0` (six identical calls to one HLE task returned six
    distinct answers from a single pinned endpoint), so a rerun is a fresh draw of every score.

    So: a wide interval or a firing rate near a half is evidence the slice is too small. A narrow
    one is NOT evidence the flag will survive a rerun. Only replication is, and the report says so
    beneath every one of these lines.

    The resample is PAIRED and WITHIN benchmark, because the two arms ran the identical slice and
    the task is the unit of evidence — an unpaired draw would price task difficulty as if it were
    disagreement between the arms, which is the variance the pairing exists to remove.

    `derive_exchange` is CALLED, never re-implemented: this module must not carry a second copy of
    §5.1b's formula, for the reason `simulate.priced` gives — the day the rule changes, the report
    and the scorer would disagree with nothing able to see it.
    """
    # THE SAME EPISODES `exchange_without_errors` FITS, and that is not a detail. A 429 is scored
    # as the model answering nothing, so a bootstrap over the raw arms resamples the provider's
    # throttle as if it were the pool's quality. Measured 2026-09-06 on run 3: the raw fit reports
    # DOMINATOR firing in 0% of resamples — maximum confidence — while the same run with the
    # errored tasks dropped flags DOMINATOR outright. Reporting the first beside the second is how
    # a reader takes false comfort from a number that is describing a rate limit.
    cheap, strong = _clean_pairs(arms, top)
    shared = sorted(cheap)
    if not shared:
        return {}
    per_bench: dict[str, list[str]] = {}
    for t in shared:
        per_bench.setdefault(cheap[t].benchmark, []).append(t)
    costs = {b: ex.c_per_task for b, ex in exchange.items()}

    rng = random.Random(seed)
    lams: dict[str, list[float]] = {b: [] for b in per_bench}
    fired: dict[str, int] = {b: 0 for b in per_bench}
    for _ in range(n_boot):
        drawn = [rng.choice(ts) for ts in per_bench.values() for _ in ts]
        table = score.derive_exchange([cheap[t] for t in drawn], [strong[t] for t in drawn], costs)
        for b, ex in table.items():
            lams[b].append(ex.lam)
            fired[b] += bool(ex.flags)
    out = {}
    for b, values in lams.items():
        values.sort()
        lo = values[int(0.025 * len(values))]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        out[b] = (lo, hi, fired[b] / n_boot)
    return out


def paired_wins(arms: Sequence[Arm], benchmark: str,
                top: str = "always_strongest") -> tuple[int, int, int]:
    """(top won, cheapest won, tied) over the tasks BOTH fitted arms actually reached."""
    by_name = {arm.name: arm for arm in arms}
    cheap = {r.task_id: r.graded_score for r in by_name["always_cheapest"].results
             if r.benchmark == benchmark}
    strong = {r.task_id: r.graded_score for r in by_name[top].results
              if r.benchmark == benchmark}
    shared = cheap.keys() & strong.keys()
    up = sum(1 for t in shared if strong[t] > cheap[t])
    down = sum(1 for t in shared if strong[t] < cheap[t])
    return up, down, len(shared) - up - down


def sign_test(up: int, down: int) -> float:
    """Exact two-sided binomial p over the discordant pairs, under a null of no difference."""
    n = up + down
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(up, down) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def report(arms: Sequence[Arm], benchmarks: Sequence[str], tasks: Sequence[TaskSpec],
           exchange, *, king: str = "always_strongest", top: str = "always_strongest") -> str:
    lines = ["", "=" * 78, "M3a — SPREAD SMOKE", "=" * 78, ""]
    # §5.5 names King0 from this line, and `world.from_env` requires it be named EXPLICITLY (V3_KING0):
    # a default there would price every crown against a policy nobody measured.
    lines += [f"KING0 = {king}   (the best fixed policy by quality; pass it as V3_KING0)",
              f"lambda is fitted between always_cheapest and {top} (reference.fit_pool)", ""]
    lines.append(f"{'arm':20s} {'quality':>9s} {'spend $':>9s}")
    for arm in arms:
        lines.append(f"{arm.name:20s} {arm.quality():9.4f} {arm.spend():9.4f}")

    lines += ["", "-- per benchmark (exit 1): spread between best and worst fixed policy --", ""]
    for bench in benchmarks:
        qualities = {arm.name: arm.quality(bench) for arm in arms}
        best, worst = max(qualities.values()), min(qualities.values())
        n = sum(1 for t in tasks if t.benchmark == bench)
        lines.append(f"{bench:20s} n={n:<4d} spread {best - worst:+.4f}   "
                     + "  ".join(f"{k}={v:.3f}" for k, v in qualities.items()))
        cheap = qualities.get("always_cheapest", 0.0)
        lines.append(f"{'':20s} cheapest-dominance: cheapest {'IS' if cheap >= best else 'is NOT'} "
                     f"the best fixed policy here")

    lines += ["", "-- disagreement rate (exit 2): §5.4 assumes 20%, never measured until now --", ""]
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            lines.append(f"{a.name} vs {b.name}: {disagreement(a, b):.3f}")

    lines += ["", "-- provisional lambda / C_b (exit 3) — cheap+mid pool, NOT launch grade --", ""]
    clean_table, clean_n = exchange_without_errors(arms, exchange, top)
    stability = lambda_stability(arms, exchange, top=top)
    for bench, ex in sorted(exchange.items()):
        flags = ", ".join(ex.flags) if ex.flags else "none"
        lines.append(f"{bench:20s} lambda={ex.lam:+.5f}  C={ex.c_per_task:.5f}  flags: {flags}")
        # `score.derive_exchange` decides DOMINATOR on `q_strong <= q_cheap` — two means, compared
        # bare, with no noise band of the kind `RELSPEND_SPREAD_FLOOR` gives the other guard. The
        # flag is the whole of §7 step 3a's stop condition, and this repository has already
        # RETRACTED one launch verdict that was a single noisy slice, so the report says how wide
        # the margin is and how often the two arms actually disagreed. The arms are paired on the
        # same tasks, which is what makes the sign test the right read rather than the arm means.
        up, down, tied = paired_wins(arms, bench, top)
        by_name = {arm.name: arm for arm in arms}
        margin = by_name[top].quality(bench) - by_name["always_cheapest"].quality(bench)
        lines.append(f"{'':20s} {top} - cheapest = {margin:+.4f}; {top} won {up}, "
                     f"cheapest won {down}, tied {tied}; sign test p={sign_test(up, down):.4f}")
        # And what the same fit says once the provider's failures are out of it. Printed only when
        # it differs, so a clean run says nothing extra.
        clean = clean_table.get(bench)
        if clean is not None and (abs(clean.lam - ex.lam) > 1e-9 or clean.flags != ex.flags):
            lines.append(f"{'':20s} errors excluded (n={clean_n.get(bench, 0)}): "
                         f"lambda={clean.lam:+.5f}  "
                         f"flags: {', '.join(clean.flags) if clean.flags else 'none'}  "
                         f"<- the fit to pin if the rung is not re-run")
        band = stability.get(bench)
        if band is not None:
            lo, hi, rate = band
            note = ("  <- the slice cannot resolve this flag" if 0.05 < rate < 0.95 else "")
            lines.append(f"{'':20s} bootstrap over TASKS (provider errors excluded, n="
                         f"{clean_n.get(bench, 0)}): lambda in [{lo:+.5f}, {hi:+.5f}], "
                         f"DOMINATOR fires in {rate:.0%} of resamples{note}")

    if stability:
        lines += ["",
                  "  A NARROW BOOTSTRAP IS NOT A PINNABLE LAMBDA. It resamples TASKS, so it cannot",
                  "  see variance BETWEEN RUNS, and the worker is nondeterministic at temperature 0",
                  "  (measured 2026-09-06: six identical calls to one task, six distinct answers,",
                  "  one pinned endpoint). Replicating the cheap arm three times FLIPPED",
                  "  livecodebench's flag, while each single draw's own bootstrap called itself",
                  "  settled at 3%. Re-run an arm with --arms before pinning lambda: a flag seen",
                  "  once has not been measured."]

    # `release_of` is an INSTANCE method — it reads the loaded release index — and calling it on the
    # class raised only in the report, after every paid episode had already been bought. The arms
    # were saved incrementally so nothing was lost, which is the property `run_arm` exists for.
    releases: dict[str, list[float]] = {}
    _lcb = LiveCodeBench(releases=LCB_RELEASES)
    for arm in arms:
        for r in arm.results:
            if r.benchmark == "livecodebench":
                rel = _lcb.release_of(r.task_id) or "unknown"
                releases.setdefault(rel, []).append(r.graded_score)
    if releases:
        lines += ["", "-- LiveCodeBench by release: the enable question, answered for free --", ""]
        for rel, scores in sorted(releases.items()):
            lines.append(f"{rel:16s} n={len(scores):<4d} mean {statistics.fmean(scores):.4f}")
        spans = [statistics.fmean(v) for v in releases.values()]
        lines.append(f"{'span':16s} {max(spans) - min(spans):+.4f} across releases")

    rel = reliability(arms)
    lines += ["", "-- reliability (the validity guard): a call that cost $0 ERRORED --", ""]
    worst = 0.0
    for model, (bad, total) in sorted(rel.items()):
        rate = bad / total if total else 0.0
        worst = max(worst, rate)
        mark = "  <-- CONTAMINATED" if rate > 0.25 else ("  <- watch" if rate > 0.10 else "")
        lines.append(f"{model:45s} {bad:4d}/{total:<4d} errored  {rate:6.1%}{mark}")
    if worst > 0.25:
        lines += ["",
                  "*** THIS READING IS REFUSED AS A QUALITY MEASUREMENT. ***",
                  "A model erroring on more than a quarter of its calls is being scored for its",
                  "provider's availability, not its answers. §8b.3 makes an error a failed rung on",
                  "purpose, which is right for SCORING a Conductor and wrong for MEASURING a pool.",
                  "ROUTING_MEASUREMENTS §16 reported such a pass as quality and had to retract it.",
                  "Re-run the affected rung before believing the spread or the lambda table above."]
    elif worst > 0.10:
        lines += ["", "NOTE: an error rate above 10% is enough to move a 15-task spread. Treat the",
                  "spread as provisional and re-run the flagged rung before pinning lambda."]

    lines += ["", "-- what this can conclude --",
              "  clear spread  -> proceed.",
              "  no spread     -> look again before building further. NOT a kill: a cheap+mid pool",
              "                   showing nothing is weak evidence, and on hard agentic benchmarks",
              "                   cheap models may all score near zero for a floor-effect reason",
              "                   that says nothing about routing.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks-per-benchmark", type=int, default=15)
    parser.add_argument("--benchmarks", nargs="+", default=["livecodebench", "hle"])
    parser.add_argument("--budget-usd", type=float, default=40.0,
                        help="hard ceiling for the WHOLE run; refused before any call if the "
                             "pessimistic estimate exceeds it")
    parser.add_argument("--episode-usd", type=float, default=PESSIMISTIC_EPISODE_USD,
                        help="per-episode cost the pre-flight estimate is built from. The default "
                             "is a deliberately pessimistic ceiling; pass a MEASURED figure to size "
                             "a run from evidence (HLE measured $0.00183/episode on 2026-09-02)")
    parser.add_argument("--out", type=Path, default=Path("m3a-results"))
    parser.add_argument("--concurrency", type=int, default=EPISODE_CONCURRENCY, metavar="N",
                        help=f"episodes in flight per arm, as the validator runs them (default "
                             f"{EPISODE_CONCURRENCY})")
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES),
                        metavar="ARM",
                        help="which arms to buy; the rest are read from --out as they were left. "
                             "Re-running one rung is what the reliability guard asks for when a "
                             "provider errored, and it costs a third of the run rather than all of "
                             f"it. Choices: {', '.join(ARM_NAMES)}")
    # MEASURED 2026-09-05, and the reason this is a flag rather than the client's default: the
    # cheap rung failed 19-35% of its calls across two runs, every one of them an HTTP 429 after
    # the client's three attempts. A rung that loses a fifth of its calls to a throttle is being
    # scored for the provider's rate limit rather than its answers, which is the reading
    # `reliability` refuses. The mechanism keeps the tighter default — a validator must not stall a
    # duel behind a slow provider — but a MEASUREMENT can afford to wait, and this one is fitting a
    # constant that gets pinned per corpus.
    parser.add_argument("--attempts", type=int, default=8, metavar="N",
                        help="worker retry attempts per call (client default 3)")
    parser.add_argument("--backoff", type=float, default=2.0, metavar="SECONDS",
                        help="base backoff between attempts (client default 1.5)")
    parser.add_argument("--seed", default="m3a")
    parser.add_argument("--report-only", action="store_true",
                        help="re-render the report from an existing --out directory; calls nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the estimate and the resolved pool, call nothing")
    args = parser.parse_args(argv)

    if args.report_only:
        arms = load_arms(args.out)
        by_name = {arm.name: arm for arm in arms}
        fit = fit_arms(arms)
        tasks = [SimpleNamespace(benchmark=r.benchmark)
                 for r in by_name["always_cheapest"].results]
        text = report(arms, args.benchmarks, tasks, fit.exchange, king=fit.king_zero, top=fit.top)
        # The table is re-written too: a re-render is exactly when the FIT changed under the same
        # episodes (2026-09-07: the pair moved from always-strongest to King0), and the table is the
        # artifact `world.pins` consumes — a report that said one thing beside a JSON that said
        # another would launch on the stale one.
        table, which = table_to_pin(arms, fit)
        write_table(args.out, table)
        text += "\n" + which + "\n"
        (args.out / "m3a-report.txt").write_text(text, encoding="utf-8")
        print(text)
        return 0

    catalog = probe_pool(pool_mod.snapshot(pool_mod.fetch()))
    print("probe pool (price-sorted):")
    for entry in catalog.entries:
        print(f"  {entry.model_id:45s} ${entry.price_in_per_mtok:7.3f} / "
              f"${entry.price_out_per_mtok:7.3f} per Mtok")

    benchmarks, tasks = slice_for(args.benchmarks, args.tasks_per_benchmark, seed=args.seed)
    episodes = len(tasks) * len(args.arms)
    estimate = episodes * args.episode_usd
    print(f"\n{len(tasks)} tasks x {len(args.arms)} arm(s) = {episodes} episodes; "
          f"pessimistic estimate ${estimate:.2f} against a ${args.budget_usd:.2f} ceiling")
    if estimate > args.budget_usd:
        raise SystemExit(f"REFUSED before spending: the pessimistic estimate ${estimate:.2f} "
                         f"exceeds --budget-usd ${args.budget_usd:.2f}. Lower "
                         f"--tasks-per-benchmark or raise the ceiling deliberately.")
    # The ceiling each arm is actually run under, printed BEFORE --dry-run returns because it is
    # what decides whether an arm finishes its slice: an arm that hits it stops early, and a short
    # arm reads as futility rather than as a budget that was split for a run of a different size.
    per_arm = args.budget_usd / len(args.arms)
    print(f"per-arm ceiling ${per_arm:.2f} across {len(args.arms)} arm(s): {', '.join(args.arms)}")
    print(f"worker retries: {args.attempts} attempts, {args.backoff}s backoff "
          f"(a 429 that outlives them is scored as the model answering nothing)")
    if args.dry_run:
        print("\n--dry-run: nothing was called.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    worker = OpenRouterClient(os.environ["OPENROUTER_API_KEY"],
                              attempts=args.attempts, backoff=args.backoff)
    grade, inspect = suite_grade(benchmarks), suite_inspect(benchmarks)

    print()
    policies = arm_policies(catalog)
    for name in args.arms:
        run_arm(name, policies[name], tasks, catalog, worker=worker, grade=grade, inspect=inspect,
                budget_usd=per_arm, out=args.out, concurrency=args.concurrency)

    # Read every arm back off disk, including the ones just bought, so a partial run and a whole one
    # take the identical path to the table. `load_arms` is what `--report-only` already trusts.
    arms = load_arms(args.out)
    by_name = {arm.name: arm for arm in arms}
    missing = [name for name in ARM_NAMES if name not in by_name]
    if missing:
        raise SystemExit(
            f"cannot fit lambda: {args.out}/ holds no episodes for {missing}. A slope needs both "
            f"fixed arms and C_b is pinned from the cascade, so --arms can only re-buy a rung in a "
            f"directory that already has the others.")
    # King0 first, then lambda fitted to King0 — `reference.fit_pool` is the one definition of that
    # pair, shared with `simulate.pin_corpus`, the dev kit and the archetype world. Fitting it here
    # against always-strongest measured whether a stronger SINGLE MODEL buys quality, which on this
    # pool it does not (2026-09-06: a coin flip on both benchmarks); the mechanism's question is
    # whether the best FIXED POLICY beats the floor, and the cascade does, with room.
    fit = fit_arms(arms)
    exchange = fit.exchange
    table, which = table_to_pin(arms, fit)
    write_table(args.out, table)

    text = report(arms, args.benchmarks, tasks, exchange, king=fit.king_zero, top=fit.top)
    text += "\n" + which + "\n"
    (args.out / "m3a-report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"episodes, exchange table and report written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
