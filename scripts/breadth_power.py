#!/usr/bin/env python
"""Can v3 breadth (§5.2 conditions 2-3) be a hard gate at the corpus size that exists? NO — and the
reason is not the one the corpus record §3 gave.

RUN: `python scripts/breadth_power.py all` (~10 min on 12 cores) or a single section by name.
Every number in the corpus record §3 and `WHITEPAPER.md` §5.3 that is dated 2026-09-01
comes out of this file.

THE QUESTION. `duel.challenger_wins` gates on the SIGN of each benchmark's delta (over a
`BREADTH_FLOOR`-zeroed vector). At the viable corpus — N = 3 benchmarks, n_b ~= 83 tasks each — the
per-benchmark bootstrap SE is ~0.055, eleven times the floor, so those signs are coin flips on a
benchmark the challenger never touched. The hypothesis tested here was that the failure is a
property of gating SIGNS AT ZERO rather than of breadth: a rule calibrated against the RIGHT null
("the true advantage is UNIFORM, equal to the observed aggregate D") instead of against zero should
have real power, because a specialist that clears `eps` must put ~N*eps on one benchmark, which is
2.7 SE and not a coin flip.

THE ANSWER, IN FOUR MEASUREMENTS, EACH A SECTION BELOW.

  1. `baseline` — **the premise for demoting breadth is false.** §3 option 2 says keeping breadth
     as a gate refuses "roughly half of genuinely broad challengers at N = 3". Measured against
     condition 1 alone it refuses 0.3-1.4 points of them. What refuses broad challengers is `eps`
     (a true +0.05 is a coin flip against a 0.05 bar by construction), not breadth. Breadth
     meanwhile does real work on the shapes it can see: a one-benchmark specialist 50.2% vs 82.5%,
     the predator 24.8% vs 57.2%. Demoting it is a ~1-point saving for a 30-point loss.

  2. `slice` — **no larger slice repairs it, and the lever runs backwards.** A TWO-of-three
     specialist is crowned 24.2% at n_b=20, 55.5% at 83, 73.8% at 1,000 and 99.2% at 16,000.
     Arithmetic, not noise: at N = 3 the median of `[x, x, d3]` is `x > 0` for any `d3` and every
     leave-one-out subset keeps a winner, so the crown rate converges UP to condition 1's, which
     converges to 100%. Buying tasks makes the shipped gate strictly less safe.

  3. `candidate` — **the calibrated rule works, on the archetypes it was designed against.**
     `max(d_b) - median(d_b)` against a bootstrap null re-centred on H0 ("uniform, equal to D"),
     refused in the upper tail at alpha = 0.05, takes the loud one-benchmark specialist from 50.2%
     to 6.0% and the predator from 24.8% to 2.8%, at a 4-9 point cost to broad challengers, and its
     false-refusal rate on a truly uniform challenger IS alpha. That is a real result and it is why
     this file exists.

  4. `attacks` — **and it does not survive an adaptive miner or the corpus's own task structure.**
     (a) R2E-Gym draws many tasks per repository; tasks arrive in correlated blocks and the paired
     bootstrap resamples them iid, so the null is too narrow. A genuinely uniform +0.08 challenger
     is refused at 24% where the dial reads 5% — and the design effect is invisible to the bootstrap
     an operator would tune against, so the dial is not merely miscalibrated, it is uncalibratable.
     (b) D14 publishes the king's weights, so the profile is a design variable. Two knobs that cost
     the miner nothing it knows — route the off-subject benchmarks to a different equal-quality
     model, and spend ~half as much there — take a PURE one-benchmark lookup table from 16.8% to
     50.2%, seven points off having no breadth rule at all, and defeat the SHIPPED rule at the same
     time (56.5% against condition 1's 57.2%). (c) On `[+0.183, +0.183,
     -0.200]` the replacement is a strict REGRESSION: it crowns 50.7% where the shipped rule crowns
     18.8%, and the gap widens with slice size, because `max - median` at N = 3 is `d(1) - d(2)` and
     therefore provably blind to how bad the worst benchmark is.

  5. `pooled` — **and the direction §3 recommended exploring is refused by an invariance.** "Judge
     breadth on pooled evidence rather than per benchmark — require the aggregate to survive
     dropping the challenger's best quartile of TASKS, which has the whole slice's n behind it."
     With roughly equal tasks per benchmark (§6.3b requires it) the pooled trimmed mean is a
     function of the multiset of per-task advantages alone, so it is EXACTLY invariant under
     permuting which benchmark each task came from — five relabellings give it to nine decimals
     while `loo_min` swings from -0.007 to +0.072. Breadth is a question about those labels. And
     the free parameter has no window: below the challenger's per-task win rate the statistic is
     identically `(D - f)/(1 - f)`, so `trimmed > 0` is `delta > f` — condition 1 with `eps`
     renamed — and above it the statistic is negative for everyone, uniform +0.15 included.

SO: **(c) — breadth cannot be a complete hard gate at N = 3, and no rule family measured repairs it
at a feasible slice.** But the corrected conclusion is NOT to demote it. Measurement 1 says the
shipped rule is nearly free and does real work on one-benchmark shapes; measurement 4 says every
replacement offered is worse under an adaptive miner. `duel.py` is therefore left exactly as it is,
and what changes is the DOC: the residual hole is named as the two-of-three shape at N = 3
specifically, not as "breadth is unjudgeable".

WHAT THE NUMBERS REST ON, STATED SO A LATER READER CAN PRICE THEM
-----------------------------------------------------------------
Quality is PASS/FAIL and PAIRED, because R2E-Gym, LiveCodeBench and HLE all are. The paired per-task
difference is in {-1, 0, +1} and cannot be modelled by adding a normal deviate:

    king K_i ~ Bernoulli(q);  C_i = 1 w.p. `a` where K_i = 0;  C_i = 0 w.p. `r` where K_i = 1
    E[C-K] = (1-q)a - q*r = mu      P(C != K) = (1-q)a + q*r = d      SD[C-K] = sqrt(d - mu^2)

At d = 0.20, mu = 0 that is 0.447, which is §5.2b's "SD ~= 0.45 at ~20% discordance" and reproduces
§3's per-benchmark SE of ~0.5/sqrt(n). It also carries the [0,1] bound a Gaussian silently violates:
`mu` cannot exceed `1 - q`, so some concentrated archetypes are `Infeasible` at N = 12 rather than
merely large. That is a finding about the attack surface, not a limitation here — it is raised, never
clipped, because a clipped archetype is a different archetype wearing the requested one's name.

Traceable to the repo: `SWEEP_CHEAP_USD` / `SWEEP_STRONG_USD` from the spend plan §5 test 2 and
its top rung; `EPS` / `BREADTH_FLOOR` / `N_BOOT` imported from `v3.config`, never re-expressed;
`DISCORDANCE = 0.20` from §5.2b. ASSUMED, with no measurement behind them and flagged rather than
buried: `BASE_QUALITY = 0.30` (it matters only through the headroom `1 - q`), `SWEEP_QUALITY_GAP`,
`SPEND_TASK_SIGMA`, and `SPEND_JITTER_SIGMA` — the last is the ONLY source of per-benchmark movement
for a copy, and `attacks` sweeps it.

Every archetype here is a POINT MASS and every rule-to-rule comparison is PAIRED (same data seeds),
so differences are far tighter than the 2.5pp binomial SE of any single 400-trial cell. A real
challenger population is a mixture, and nothing here measures one.
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from thirtyspokes.v3.config import BREADTH_FLOOR, EPS, N_BOOT
from thirtyspokes.v3.duel import BenchmarkPair, _delta, duel

# --- the priced-score world (score.py §5.1b, as a function the bootstrap can re-evaluate) ---------

SWEEP_CHEAP_USD = 0.002       # the spend plan §5 test 2 MID: $0.212 / 100 worker calls
SWEEP_STRONG_USD = 0.0305     # claude-haiku-4.5 at $1/$5 per Mtok on the measured 500-in/6,000-out
SWEEP_QUALITY_GAP = 0.20      # ASSUMED — the spread smoke that would measure it has not been run
C_PINNED = (SWEEP_CHEAP_USD + SWEEP_STRONG_USD) / 2.0
LAMBDA = SWEEP_QUALITY_GAP / ((SWEEP_STRONG_USD - SWEEP_CHEAP_USD) / C_PINNED)
PRICE = LAMBDA / C_PINNED     # quality points per dollar; ~7.0 USD^-1

BASE_QUALITY = 0.30           # ASSUMED: HLE ~0.15, LiveCodeBench ~0.35, R2E-Gym ~0.30
DISCORDANCE = 0.20            # §5.2b, via the spend plan §5 test 3
SPEND_MEAN_USD = 0.010        # a cascade arm's mean per-task spend, between the two rungs
SPEND_TASK_SIGMA = 0.6        # ASSUMED: lognormal sd of per-task spend (task length spread)
SPEND_JITTER_SIGMA = 0.10     # ASSUMED: the two arms' spend divergence where routing AGREES

SEED = 20260901
TRIALS = 400
BOOT = 200                    # n_boot only moves `lcb`; drift vs N_BOOT is <=1.0pp, measured below
ALPHA = 0.05                  # the candidate rule's dial, from its own ROC and not from convention


def final_b(benchmark: str, quality: float, spend: float) -> float:
    """`duel.FinalB`: `quality_b - lambda_b * spend_b / C_b`, one exchange rate for every benchmark.

    §5.1b says the real rates differ per benchmark. They are not varied here because the rate only
    scales how spend noise enters `final`, spend noise is ~2% of quality noise per task, and a
    per-benchmark rate would be a second invented number for no change in any verdict.
    """
    return quality - PRICE * spend


# --- the generative model -------------------------------------------------------------------------


class Infeasible(ValueError):
    """This archetype cannot exist on a [0,1] graded benchmark at this base quality."""


@dataclass(frozen=True)
class Bench:
    """One benchmark's generative parameters. The defaults are the honest, non-adaptive case.

    `spend_factor`, `discordance` and `cluster*` are the three axes the `attacks` section moves; the
    first two are knobs a miner controls at no cost to what it knows, the third is a property of the
    corpus (R2E-Gym draws many tasks per repository).
    """

    mu: float                                  # TRUE advantage, delivered through quality
    n: int = 83
    base_quality: float = BASE_QUALITY
    discordance: float = DISCORDANCE
    spend_factor: float = 1.0                  # challenger mean spend / king mean spend
    spend_jitter: float = SPEND_JITTER_SIGMA
    spend_task_sigma: float = SPEND_TASK_SIGMA
    cluster: int = 1                           # tasks per correlated block (1 = iid)
    cluster_tau: float = 0.0                   # sd of the per-block advantage offset


def _rates(quality: float, mu: np.ndarray, discordance: float) -> tuple[np.ndarray, np.ndarray]:
    """`(gain, regress)` — the per-task flip rates delivering true advantage `mu` at `discordance`.

    Discordance is RAISED to `|mu|` where the requested value is smaller: two arms cannot differ in
    the mean by more than they differ per task, so a benchmark carrying a large concentrated win is
    necessarily more discordant than 0.20. Per-task cluster offsets are clipped into the feasible
    band; a benchmark whose MEAN is infeasible raises instead, which is what keeps `Infeasible` a
    finding rather than a silent substitution.
    """
    mu = np.clip(mu, -quality + 1e-9, (1.0 - quality) - 1e-9)
    lo, hi = np.abs(mu), np.minimum(2.0 * (1.0 - quality) - mu, 2.0 * quality + mu)
    d = np.clip(discordance, lo, hi)
    return (d + mu) / (2.0 * (1.0 - quality)), (d - mu) / (2.0 * quality)


def sample(specs: list[Bench], rng: np.random.Generator) -> tuple[BenchmarkPair, ...]:
    """One duel's worth of paired data. Benchmarks are named `b00...` so `duel`'s sort order is the
    spec list's order — a caller reading `by_benchmark` positionally reads what this put there."""
    pairs = []
    for i, s in enumerate(specs):
        if abs(s.mu) > (1.0 - s.base_quality if s.mu > 0 else s.base_quality):
            raise Infeasible(
                f"a true advantage of {s.mu:+.4f} needs more headroom than a pass/fail benchmark at "
                f"base quality {s.base_quality:.2f} has: the most a challenger can gain is "
                f"{1.0 - s.base_quality:+.4f} and the most it can lose is {-s.base_quality:+.4f}")
        mu = np.full(s.n, s.mu)
        if s.cluster_tau > 0.0:   # cluster == 1 is the IID CONTROL: same marginal law, no blocks
            blocks = int(np.ceil(s.n / s.cluster))
            mu = mu + np.repeat(rng.normal(0.0, s.cluster_tau, blocks), s.cluster)[:s.n]
        gain, regress = _rates(s.base_quality, mu, s.discordance)
        king = rng.random(s.n) < s.base_quality
        gained = rng.random(s.n) < gain
        regressed = rng.random(s.n) < regress
        challenger = np.where(king, ~regressed, gained)

        king_spend = SPEND_MEAN_USD * _unit_lognormal(rng, s.spend_task_sigma, s.n)
        chal_spend = king_spend * s.spend_factor * _unit_lognormal(rng, s.spend_jitter, s.n)
        pairs.append(BenchmarkPair(f"b{i:02d}", king.astype(float), king_spend,
                                   challenger.astype(float), chal_spend))
    return tuple(pairs)


def _unit_lognormal(rng: np.random.Generator, sigma: float, n: int) -> np.ndarray:
    """Lognormal with mean exactly 1, so a jitter is noise and never a systematic discount."""
    return np.ones(n) if sigma <= 0.0 else np.exp(rng.normal(-0.5 * sigma * sigma, sigma, n))


# --- the archetypes -------------------------------------------------------------------------------

TARGET = 1.1 * EPS      # a win sized to land JUST over the aggregate bar — the hardest specialist
LOUD = 1.5 * EPS        # comfortably clear of it, so condition 1 never decides — the §5.3 shape
BROAD = 0.08            # a broad challenger big enough to be crownable against eps at all


def _spike(n: int, k: int, aggregate: float, floor: float = 0.0) -> list[float]:
    """`k` benchmarks carry the whole win; the other `n - k` sit at `floor`."""
    return [(aggregate * n - floor * (n - k)) / k] * k + [floor] * (n - k)


ARCHETYPES = {
    # --- must be refused ---
    "copy": lambda n: [0.0] * n,
    "specialist1": lambda n: _spike(n, 1, TARGET),
    "specialist1_loud": lambda n: _spike(n, 1, LOUD),
    "specialist2": lambda n: _spike(n, 2, TARGET),
    "specialist_predator": lambda n: _spike(n, 1, TARGET, floor=-0.02),
    # --- must be crowned ---
    "uniform_003": lambda n: [0.03] * n,
    "uniform_005": lambda n: [0.05] * n,
    "uniform_008": lambda n: [0.08] * n,
    "uniform_015": lambda n: [0.15] * n,
    "broad_uneven": lambda n: [BROAD * (0.5 + i / (n - 1)) for i in range(n)],
    # --- its own axis: whether this SHOULD be crowned is a judgement call, not a measurement ---
    "broad_one_loss": lambda n: [-0.02] + [(BROAD * n + 0.02) / (n - 1)] * (n - 1),
}

REFUSE = ("copy", "specialist1", "specialist1_loud", "specialist2", "specialist_predator")


def archetype(name: str, n_benchmarks: int, n_tasks: int, **kwargs) -> list[Bench]:
    """A spec list. `copy` routes IDENTICALLY to the king (discordance 0) — the only place a
    per-benchmark delta of exactly zero is physically honest."""
    mu = ARCHETYPES[name](n_benchmarks)
    disc = 0.0 if name == "copy" else DISCORDANCE
    return [Bench(mu=m, n=n_tasks, discordance=max(disc, abs(m)), **kwargs) for m in mu]


# --- the rules --------------------------------------------------------------------------------------
#
# CONDITION 1 IS INSIDE EVERY RULE, NEVER BESIDE IT. `challenger_wins` is `delta > eps AND lcb > 0
# AND breadth`, so a breadth rule is only ever asked about challengers that already cleared the
# aggregate. Measuring breadth alone would overstate every specialist threat, because a specialist
# that cannot clear `eps` is not a threat at all.


def condition_one(verdict) -> bool:
    """§5.2 condition 1, verbatim."""
    return verdict.delta > verdict.eps and verdict.lcb > 0.0


def current_breadth(verdict) -> bool:
    """§5.2 conditions 2 and 3 — the SHIPPED rule, read off `DuelVerdict` rather than re-expressed."""
    return verdict.loo_min > 0.0 and verdict.median > 0.0


def dispersion_p(verdict, pairs, *, seed: int, n_boot: int) -> float:
    """THE CANDIDATE. `p` for `max(d_b) - median(d_b)` against a bootstrap null re-centred on H0.

    H0 is "the challenger's true advantage is THE SAME on every benchmark, equal to the observed
    aggregate D" — NOT "is zero", which is the null the shipped rule tests and the reason its signs
    are coin flips. The null is built from `duel()`'s own paired resamples, re-centred:

        e*_b = d*_b - d_b + D        p = (1 + #{T(e*) >= T(d)}) / (B + 1)      refuse iff p <= alpha

    Two properties that fall out of the construction rather than out of tuning:

      * The false-refusal rate on a genuinely uniform challenger IS alpha, independent of n_b, of N
        and of how noisy the benchmarks are — the cost is set by a dial rather than by the noise.
        Measured in `candidate`, and BROKEN by correlated tasks in `attacks`.
      * D drops out. `max - median` is invariant to adding a constant to all N deltas, so `- d_b + D`
        equals `- d_b + anything` and the observed aggregate's own noise cannot bias the test.
        Asserted in `verify`.

    The bootstrap loop is `duel()`'s, REPLAYED rather than reimplemented — same sorted pairs, same
    seeded rng, same `_delta` — so `boot` here is bit-for-bit the matrix `duel` took `noise_floor`
    from (asserted in `verify`). A shipped version would return that matrix instead of replaying it,
    which is why this rule was priced at zero extra compute.
    """
    pairs = tuple(sorted(pairs, key=lambda p: p.benchmark))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(pairs)))
    for b in range(n_boot):
        rows = [rng.integers(0, p.n, size=p.n) for p in pairs]
        _, boot[b] = _delta(pairs, final_b, rows)

    def concentration(d: np.ndarray) -> np.ndarray:
        """Vectorised over leading axes, so the whole null costs one call."""
        return d.max(-1) - np.median(d, -1)

    observed = np.array([d for _, d in verdict.by_benchmark])
    null = boot - observed + verdict.delta
    # +1 in both places: a p-value that can be exactly 0 would refuse on a bootstrap that never drew
    # anything as extreme, which at B = 200 is a 1-in-201 event under the null, not proof.
    return float((1 + np.sum(concentration(null) >= concentration(observed))) / (n_boot + 1))


# --- the measurement ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """Four crown rates on ONE set of trials, so every comparison in this file is paired."""

    cond1: float          # condition 1 alone — the ceiling any breadth rule works under
    current: float        # the shipped rule
    dispersion: float     # condition 1 + the candidate at ALPHA
    both: float           # condition 1 + shipped AND candidate — the composite
    roc: dict             # crown rate for the candidate at each alpha on the ROC
    deltas: np.ndarray    # mean per-benchmark delta, so a table can show WHAT was measured
    noise: np.ndarray     # mean per-benchmark bootstrap SE

    def refusal(self, rate: float) -> float:
        """A rule's own false-refusal rate: what it turned away of what condition 1 let through."""
        return float("nan") if self.cond1 <= 0.0 else 1.0 - rate / self.cond1


ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def measure(specs: list[Bench], *, trials: int = TRIALS, seed: int = SEED, n_boot: int = BOOT,
            candidate: bool = True) -> Cell:
    """Crown rates over `trials` sampled slices. `candidate=False` skips the second bootstrap, which
    halves the cost of the sections that only need the shipped rule."""
    seeds = np.random.default_rng(seed).integers(1, 2 ** 31 - 1, size=(trials, 2))
    gate = cur = 0
    roc = dict.fromkeys(ALPHAS, 0)
    both = 0
    deltas, noise = [], []
    for data_seed, duel_seed in seeds:
        pairs = sample(specs, np.random.default_rng(int(data_seed)))
        verdict = duel(pairs, final_b, seed=int(duel_seed), n_boot=n_boot)
        deltas.append([d for _, d in verdict.by_benchmark])
        noise.append(verdict.noise_floor)
        if not condition_one(verdict):
            continue
        gate += 1
        breadth = current_breadth(verdict)
        cur += breadth
        if candidate:
            p = dispersion_p(verdict, pairs, seed=int(duel_seed), n_boot=n_boot)
            for a in ALPHAS:
                roc[a] += p > a
            both += breadth and p > ALPHA
    return Cell(gate / trials, cur / trials, roc[ALPHA] / trials, both / trials,
                {a: c / trials for a, c in roc.items()},
                np.asarray(deltas).mean(axis=0), np.asarray(noise).mean(axis=0))


def _job(kwargs):
    try:
        return measure(**kwargs)
    except Infeasible:
        return None


def run(jobs, workers=None) -> list:
    """`measure` over a list of kwargs dicts, one process per core. `None` marks a cell whose
    archetype is `Infeasible` at that shape — which is a result, not an error."""
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_job, jobs))


# --- section 1: the baseline, and the correction it forces ----------------------------------------


def baseline(trials: int, n_boot: int, n_benchmarks: int = 3) -> None:
    """The shipped rule against condition 1 alone, over the archetypes and four slice sizes.

    The gap between the two columns is the ONLY part breadth is responsible for — a refusal
    condition 1 would have made anyway is not breadth's doing, and conflating the two is what
    produced §3's "refuses roughly half of genuinely broad challengers".
    """
    sizes = (20, 83, 250, 1000)
    jobs = [dict(specs=archetype(a, n_benchmarks, n), trials=trials, n_boot=n_boot, candidate=False)
            for a in ARCHETYPES for n in sizes]
    out = run(jobs)

    print("=" * 100)
    print(f"1. BASELINE — the SHIPPED rule (condition 1 alone in brackets), N = {n_benchmarks}, "
          f"{trials} trials")
    print("=" * 100)
    print("   " + "archetype".ljust(22) + "".join(f"n_b={n:<14}" for n in sizes))
    for i, a in enumerate(ARCHETYPES):
        cells = []
        for j in range(len(sizes)):
            c = out[i * len(sizes) + j]
            cells.append("infeasible".ljust(18) if c is None
                         else f"{c.current:5.1%} ({c.cond1:5.1%})".ljust(18))
        print("   " + a.ljust(22) + "".join(cells))

    print("\n   what breadth ITSELF costs and buys at the real slice (n_b = 83), as a refusal rate")
    for i, a in enumerate(ARCHETYPES):
        c = out[i * len(sizes) + 1]
        if c is None or c.cond1 <= 0.0:
            continue
        verb = "REFUSED (wanted)" if a in REFUSE else "refused (cost)"
        print(f"      {a:<22} {c.refusal(c.current):6.1%} {verb}")
    print("\n   -> breadth's cost on broad challengers is single-digit POINTS, not half. The half is")
    print("      `eps`: a true +0.05 against a 0.05 bar is a coin flip by construction (uniform_005")
    print("      never rises above ~50% at ANY slice size, in the condition-1 column).")


# --- section 2: does a bigger slice fix it? -------------------------------------------------------


def slice_size(trials: int, n_boot: int, n_benchmarks: int = 3) -> None:
    """The outcome-(b) question: is there a slice size at which the SHIPPED rule works?

    Answered on the shape that decides it — a two-of-three specialist — and on a one-of-three for
    contrast, out to slice sizes far past anything the corpus or the budget could supply.
    """
    sizes = (20, 83, 250, 1000, 4000, 16000)
    names = ("specialist1_loud", "specialist2", "specialist_predator", "uniform_008", "uniform_003")
    jobs = [dict(specs=archetype(a, n_benchmarks, n), trials=trials, n_boot=n_boot, candidate=False)
            for a in names for n in sizes]
    out = run(jobs)

    print("\n" + "=" * 100)
    print(f"2. SLICE SIZE — the shipped rule (condition 1 in brackets) as n_b grows, N = "
          f"{n_benchmarks}")
    print("=" * 100)
    print("   " + "archetype".ljust(22) + "".join(f"n_b={n:<14}" for n in sizes))
    for i, a in enumerate(names):
        cells = []
        for j in range(len(sizes)):
            c = out[i * len(sizes) + j]
            cells.append("infeasible".ljust(18) if c is None
                         else f"{c.current:5.1%} ({c.cond1:5.1%})".ljust(18))
        print("   " + a.ljust(22) + "".join(cells))
    print("\n   -> `specialist2` rises MONOTONICALLY with the slice. At N = 3 the median of")
    print("      [x, x, d3] is x > 0 for any d3 and every leave-one-out subset keeps a winner, so")
    print("      breadth is arithmetically blind to a 2-of-3 win and the rate converges up to")
    print("      condition 1's, which converges to 100%. MORE TASKS MAKE THE GATE LESS SAFE.")
    print("      The corpus supplies n_b = 83 (the corpus record 2); 16,000 x 3 benchmarks x 9 arms")
    print("      is ~$3,500/window against a $100 authorised budget, and LiveCodeBench has 112 rows.")

    # THE LEVER THAT DOES HAVE THE RIGHT SIGN. A specialist's spike must be N*D to clear the
    # aggregate, so a wider corpus makes concentration harder to hide — and it is the median's
    # blindness that N repairs, since the median of [x, x, d3] is x but the median of
    # [x, x, d3, d4, d5] is not. Reported at fixed n_b AND at a fixed total task budget, because
    # only the second is a decision an owner can actually take.
    shapes = [("N=3 n_b=83  (T=250)", 3, 83), ("N=5 n_b=50  (T=250)", 5, 50),
              ("N=3 n_b=250 (T=750)", 3, 250), ("N=5 n_b=150 (T=750)", 5, 150),
              ("N=5 n_b=83  (T=415)", 5, 83)]
    names = ("specialist1_loud", "specialist2", "uniform_008")
    jobs = [dict(specs=archetype(a, nb, n), trials=trials, n_boot=n_boot, candidate=False)
            for a in names for _, nb, n in shapes]
    out = run(jobs)
    print("\n   BENCHMARK COUNT at a fixed TOTAL task budget — the shipped rule (condition 1)")
    print("   " + "archetype".ljust(22) + "".join(s.ljust(22) for s, _, _ in shapes))
    for i, a in enumerate(names):
        cells = []
        for j in range(len(shapes)):
            c = out[i * len(shapes) + j]
            cells.append("infeasible".ljust(22) if c is None
                         else f"{c.current:5.1%} ({c.cond1:5.1%})".ljust(22))
        print("   " + a.ljust(22) + "".join(cells))
    print("   -> N is the only lever measured with the right sign, and it is a corpus decision")
    print("      rather than a budget one: accepting HLE (decision 1) buys breadth, buying tasks")
    print("      does not. It does not reach a safe point either — N = 12 would, and there are 3.")


# --- section 3: the candidate ---------------------------------------------------------------------


def candidate(trials: int, n_boot: int, n_benchmarks: int = 3, n_tasks: int = 83) -> None:
    """The calibrated dispersion rule at the operative slice, with its alpha ROC.

    Reported at every alpha because the dial is the rule's whole claim: if the false-refusal rate on
    a genuinely uniform challenger really is alpha, an owner can price breadth instead of inheriting
    whatever the noise happens to give.
    """
    jobs = [dict(specs=archetype(a, n_benchmarks, n_tasks), trials=trials, n_boot=n_boot)
            for a in ARCHETYPES]
    out = run(jobs)

    print("\n" + "=" * 100)
    print(f"3. THE CANDIDATE — max(d)-median(d) vs a uniform-D bootstrap null. N = {n_benchmarks}, "
          f"n_b = {n_tasks}, {trials} trials")
    print("=" * 100)
    print("   " + "archetype".ljust(22) + "".join(f"a={a:<7}" for a in ALPHAS)
          + "|  current    both      cond1")
    for a, c in zip(ARCHETYPES, out):
        if c is None:
            print("   " + a.ljust(22) + "infeasible")
            continue
        print("   " + a.ljust(22) + "".join(f"{c.roc[x]:6.1%} " for x in ALPHAS)
              + f"| {c.current:7.1%} {c.both:8.1%} {c.cond1:8.1%}")

    print(f"\n   CALIBRATION — refusal rate on a challenger that IS uniform, at alpha = {ALPHA}")
    for a, c in zip(ARCHETYPES, out):
        if c is None or not a.startswith("uniform") or c.cond1 <= 0.0:
            continue
        print(f"      {a:<22} {c.refusal(c.dispersion):6.3f}   (nominal {ALPHA})")
    print("   -> this is the structural difference from the shipped rule: the false-refusal cost is")
    print("      set by a DIAL rather than by the noise. Section 4a is what breaks that guarantee.")


# --- section 4: the attacks that decide it --------------------------------------------------------


def attacks(trials: int, n_boot: int) -> None:
    """Three probes. The first two break the candidate; the third shows it is a regression."""
    print("\n" + "=" * 100)
    print(f"4. THE ATTACKS. N = 3, n_b = 83, {trials} trials, alpha = {ALPHA}")
    print("=" * 100)

    # (a) CORRELATED TASKS. R2E-Gym draws many tasks per repository and a router that suits a repo is
    # better on all of its tasks, so advantages arrive in blocks. `duel()` resamples tasks IID, so
    # the null is too narrow — and the design effect is invisible to that same bootstrap, which is
    # what makes this uncalibratable rather than merely miscalibrated. `block = 1` is the IID
    # CONTROL: identical per-task offset law, correlation removed.
    print("\n  (a) CORRELATED TASKS — a genuinely UNIFORM +0.08 challenger, refusal rate vs nominal")
    shapes = ((1, 0.0), (1, 0.20), (10, 0.20), (10, 0.30), (17, 0.30))
    jobs = [dict(specs=[Bench(mu=0.08, cluster=m, cluster_tau=t)] * 3, trials=trials,
                 n_boot=n_boot) for m, t in shapes]
    print("      " + "structure".ljust(28) + "".join(f"a={a:<7}" for a in ALPHAS) + "| current")
    for (m, t), c in zip(shapes, run(jobs)):
        print("      " + f"block={m:<3} tau={t:.2f}".ljust(28)
              + "".join(f"{c.refusal(c.roc[a]):6.3f} " for a in ALPHAS)
              + f"| {c.refusal(c.current):6.3f}"
              + ("   <- IID CONTROL: same marginal law, no blocks" if m == 1 and t > 0 else ""))
    print("      -> the dial reads 0.05 and the rule refuses ~a quarter of honest broad challengers")
    print("         on the block structure R2E-Gym actually has. The IID control (block=1, same")
    print("         marginal law) stays nominal, so it is the CORRELATION and not the variance.")

    # (b) THE ADAPTIVE MINER. D14 publishes the king's weights, so the per-benchmark profile is a
    # design variable. Both knobs below cost the miner nothing it knows: routing the off-subject
    # benchmarks to a DIFFERENT model of equal expected quality only adds variance, and spending
    # less there is a real (small) saving. The quality spike is resized so the aggregate stays at
    # 1.1*eps whatever the knobs buy, i.e. the miner gets no free aggregate out of this.
    print("\n  (b) THE ADAPTIVE MINER — a PURE one-benchmark lookup table, aggregate pinned at "
          "1.1*eps")
    aggregate = TARGET
    knobs = [(1.0, 0.20), (1.0, 0.40), (0.7, 0.40), (0.5, 0.20), (0.5, 0.40)]
    jobs = []
    for factor, disc in knobs:
        lift = PRICE * SPEND_MEAN_USD * (1.0 - factor)      # what the spend cut alone buys in final
        spike = 3.0 * aggregate - 2.0 * lift
        off = Bench(mu=0.0, discordance=disc, spend_factor=factor)
        jobs.append(dict(specs=[Bench(mu=spike, discordance=max(DISCORDANCE, spike)), off, off],
                         trials=trials, n_boot=n_boot))
    print("      " + "miner tuning".ljust(28) + "".join(f"a={a:<7}" for a in ALPHAS)
          + "| current    cond1   mean per-benchmark delta")
    for (factor, disc), c in zip(knobs, run(jobs)):
        tag = f"spend x{factor:.1f}, off-disc {disc:.2f}"
        print("      " + tag.ljust(28) + "".join(f"{c.roc[a]:6.1%} " for a in ALPHAS)
              + f"| {c.current:6.1%} {c.cond1:8.1%}   "
              + np.array2string(c.deltas, precision=3, floatmode="fixed"))
    print("      -> two knobs take a pure lookup table from 16.8% to 50.2%, within 7 points of")
    print("         having no breadth rule at all — and they defeat the SHIPPED rule at the same")
    print("         time (56.5% against condition 1's 57.2%). The spend cut is the FLATTENER, not")
    print("         the win: it lifts the untouched benchmarks' delta without changing any routing,")
    print("         which is exactly the hole BREADTH_FLOOR closed at 1e-5 dollars and this reopens")
    print("         at ~$0.83 a window. NEITHER RULE FAMILY SURVIVES THIS.")

    # (c) THE REGRESSION. `max - median` at N = 3 is `d(1) - d(2)`, so it is blind to how bad the
    # WORST benchmark is — algebraically, not as a small-sample artefact. The shipped rule reads
    # that benchmark and refuses; the replacement does not.
    print("\n  (c) TWO OF THREE CARRY IT AND THE THIRD IS A DISASTER — [+0.183, +0.183, -0.200]")
    sizes = (83, 250, 1000)
    jobs = [dict(specs=[Bench(mu=0.183, n=n), Bench(mu=0.183, n=n),
                        Bench(mu=-0.200, n=n, discordance=0.20)], trials=trials, n_boot=n_boot)
            for n in sizes]
    print("      " + "slice".ljust(28) + "".join(f"a={a:<7}" for a in ALPHAS)
          + "| current    both     cond1")
    for n, c in zip(sizes, run(jobs)):
        print("      " + f"n_b={n}".ljust(28) + "".join(f"{c.roc[a]:6.1%} " for a in ALPHAS)
              + f"| {c.current:6.1%} {c.both:8.1%} {c.cond1:8.1%}")
    print("      -> a STRICT REGRESSION against the rule it would replace, and the gap widens with")
    print("         the slice. Keeping the shipped conditions alongside it (`both`) recovers this,")
    print("         but `both` still inherits (a) and (b), which is what closes the family.")


# --- section 5: the pooled family, refused by an invariance rather than by a rate -----------------


def pooled(*_dispatch, n_tasks: int = 83) -> None:
    """the corpus record §3 option 3 — "judge breadth on pooled evidence rather than per
    benchmark", e.g. require the aggregate to survive dropping the challenger's best quartile of
    TASKS, "which has the whole slice's n behind it instead of one stratum's".

    It cannot work, and the reason is a provable invariance rather than a sample size. `final_b` is
    affine in (quality, spend), so a task's contribution to its benchmark's delta is well defined;
    with roughly equal tasks per benchmark (which §6.3b requires) the POOLED trimmed mean is a
    function of the multiset of per-task advantages alone. It is therefore exactly invariant under
    permuting which benchmark each task came from — and breadth is a question about precisely those
    labels. Pooling puts the whole slice's `n` behind a question that has no answer in the pooled
    data.

    The second half is the free parameter, which has no working window either. Below the challenger's
    per-task WIN RATE the pooled trimmed mean is `(D - f)/(1 - f)`, so "trimmed > 0" is just
    "delta > f" — condition 1 with a renamed `eps`. Above the win rate every survivor is a
    regression and the statistic is negative for everyone, uniform +0.15 included. On pass/fail
    grading the win rate is ~0.12-0.18, and the quartile §3 names sits on the wrong side of it.

    Takes the dispatcher's `(trials, n_boot)` and needs neither, which is the section in one line:
    there is no crown rate here to be uncertain about, only an identity and an invariance.
    """
    rng = np.random.default_rng(SEED)

    def advantages(specs):
        """Per-task paired advantage in `final`, per benchmark — the decomposition the family needs."""
        pairs = sample(specs, rng)
        return [(p.challenger_quality - p.king_quality)
                - PRICE * (p.challenger_spend - p.king_spend) for p in pairs]

    def trimmed(per_benchmark, f: float) -> float:
        pool = np.sort(np.concatenate(per_benchmark))
        keep = pool[:len(pool) - int(round(f * len(pool)))]
        return float(keep.mean())

    def loo_min(per_benchmark) -> float:
        d = [float(x.mean()) for x in per_benchmark]
        return min(float(np.mean(d[:i] + d[i + 1:])) for i in range(len(d)))

    print("\n" + "=" * 100)
    print(f"5. THE POOLED FAMILY (the corpus record 3 option 3) — N = 3, n_b = {n_tasks}")
    print("=" * 100)
    print("\n  (a) the pooled statistic CANNOT SEE the benchmark labels breadth asks about")
    per_benchmark = advantages(archetype("specialist1_loud", 3, n_tasks))
    pool = np.concatenate(per_benchmark)
    print(f"      {'relabelling':<22}{'pooled trim f=0.25':>22}{'loo_min':>14}")
    for k in range(5):
        shuffled = rng.permutation(pool) if k else pool
        split = [shuffled[i * n_tasks:(i + 1) * n_tasks] for i in range(3)]
        tag = "as measured" if k == 0 else f"random relabel {k}"
        print(f"      {tag:<22}{trimmed(split, 0.25):>22.9f}{loo_min(split):>14.4f}")
    print("      -> identical to nine decimals while leave-one-out moves by tens of points. Every")
    print("         member of this family is a statistic of the pooled multiset, so it answers a")
    print("         question with no benchmark in it. The between-benchmark contrast is")
    print("         irreducibly PER-STRATUM evidence; pooling does not buy `n` for it.")

    print("\n  (b) and the trim fraction has no working window — the cliff is the per-task win rate")
    print(f"      {'archetype':<20}{'D':>8}{'win rate':>10}" +
          "".join(f"{'f=%.2f' % f:>20}" for f in (0.05, 0.10, 0.25)))
    for name in ("specialist1_loud", "uniform_015", "uniform_005"):
        per_benchmark = advantages(archetype(name, 3, 4000))
        pool = np.concatenate(per_benchmark)
        aggregate = trimmed(per_benchmark, 0.0)
        cells = "".join(f"{trimmed(per_benchmark, f):+9.4f} ({(aggregate - f) / (1 - f):+.4f})"
                        for f in (0.05, 0.10, 0.25))
        # win rate = tasks the challenger actually SOLVED and the king did not; the spend jitter
        # makes every other task a tiny nonzero, which is why the threshold is 0.5 and not 0.
        print(f"      {name:<20}{aggregate:>+8.4f}{float((pool > 0.5).mean()):>10.3f}{cells}")
    print("      -> measured, with `(D - f)/(1 - f)` in brackets. Below the win rate the pooled")
    print("         trimmed mean IS that identity, i.e. `trimmed > 0` is `delta > f` — condition 1")
    print("         with `eps` renamed, and f = eps = 0.05 reproduces condition 1 exactly. Above it")
    print("         every survivor is a regression and the statistic is negative for EVERYONE, the")
    print("         genuine uniform +0.15 included. §3's quartile sits on the far side of the cliff.")


# --- verification -----------------------------------------------------------------------------------


def verify(trials: int, n_boot: int) -> None:
    """The checks that have to hold before any table above is worth reading."""
    print("=" * 100)
    print("VERIFICATION")
    print("=" * 100)
    print(f"   EPS={EPS}  BREADTH_FLOOR={BREADTH_FLOOR}  N_BOOT={N_BOOT}  lambda_b={LAMBDA:.4f}  "
          f"C_b={C_PINNED:.4f}  price={PRICE:.2f}/USD  per-task paired SD={np.sqrt(DISCORDANCE):.3f}")

    pairs = sample(archetype("specialist1", 3, 83), np.random.default_rng(11))
    v = duel(pairs, final_b, seed=7, n_boot=64)

    # [1] the candidate replays `duel`'s own bootstrap: same seed, same rng consumption order, same
    # `_delta`. If this drifts, the rule is not free at the seam and its whole cost claim is wrong.
    rng = np.random.default_rng(7)
    boot = np.array([_delta(tuple(sorted(pairs, key=lambda p: p.benchmark)), final_b,
                            [rng.integers(0, p.n, size=p.n) for p in pairs])[1] for _ in range(64)])
    assert np.array_equal(boot.std(axis=0, ddof=1), np.array(v.noise_floor)), "replay drifted"
    print("\n   [1] the candidate's null is duel()'s own resample matrix, bit-for-bit  OK")

    # [2] the statistic is location-invariant, so the observed aggregate's own noise cannot enter
    # the null it is tested against (the self-reference trap the null re-centring could have had).
    p_real = dispersion_p(v, pairs, seed=7, n_boot=64)
    shifted = replace(v, delta=v.delta + 0.37)
    assert dispersion_p(shifted, pairs, seed=7, n_boot=64) == p_real, "D did not drop out"
    print("   [2] `max - median` is location-invariant, so D drops out of the null       OK")

    # [3] the per-benchmark SE this whole question turns on: ~0.5/sqrt(n) at 20% discordance, which
    # is the corpus record 3's table and eleven times BREADTH_FLOOR at the slice the corpus can fill.
    print("\n   [3] the per-benchmark bootstrap SE against the floor it is compared to")
    for n in (20, 83, 250, 1000):
        c = measure(archetype("uniform_005", 3, n), trials=40, n_boot=n_boot, candidate=False)
        print(f"       n_b={n:<6} measured SE={c.noise.mean():.4f}   predicted {np.sqrt(DISCORDANCE / n):.4f}"
              f"   BREADTH_FLOOR={BREADTH_FLOOR}   ratio {c.noise.mean() / BREADTH_FLOOR:5.1f}x")

    # [4] n_boot: the sweeps run at 200 because 1,000 makes this a long job, and the only thing
    # n_boot moves is `lcb`, which binds only where `delta` sits near `eps`.
    print(f"\n   [4] n_boot sensitivity — {trials} trials, same seed and therefore the same data")
    cells = [("specialist1", 83), ("specialist2", 83), ("uniform_005", 83), ("uniform_008", 83)]
    jobs = [dict(specs=archetype(a, 3, n), trials=trials, n_boot=b)
            for a, n in cells for b in (n_boot, N_BOOT)]
    out = run(jobs)
    for i, (a, n) in enumerate(cells):
        lo, hi = out[2 * i], out[2 * i + 1]
        print(f"       {a:<22} n_boot={n_boot}: current {lo.current:6.1%} dispersion "
              f"{lo.dispersion:6.1%}   N_BOOT: {hi.current:6.1%} / {hi.dispersion:6.1%}")


SECTIONS = {"verify": verify, "baseline": baseline, "slice": slice_size,
            "candidate": candidate, "attacks": attacks, "pooled": pooled}


def main() -> None:
    p = argparse.ArgumentParser(description="is v3 breadth judgeable at a feasible slice size?")
    p.add_argument("section", nargs="?", default="all", choices=[*SECTIONS, "all"])
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--n-boot", type=int, default=BOOT)
    args = p.parse_args()

    for name, fn in SECTIONS.items():
        if args.section in (name, "all"):
            fn(args.trials, args.n_boot)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
