"""The adversarial archetype suite — seven Conductors the mechanism has to rank correctly.

A scoring rule is worth exactly what it REFUSES to reward, and this project has twice paid to learn
that the refusing does not happen by itself. A 6.4K matmul head memorised a random rung table for
1,000 tasks and looked excellent in-sample (§1.1 cites it as the reason no parameter cap is claimed
here). A five-agent red team found copy-mining reaching ~86% of emissions in a design that had no
answer to it. Neither was caught by a mechanism that noticed something odd; both were caught by an
adversary somebody wrote down and ran. So the adversaries are written down here, run through the
SHIPPED scaffold against the SHIPPED scorer, and their order is asserted in
`tests/test_archetypes.py` rather than printed for a human to eyeball.

WHAT AN ARCHETYPE MAY KNOW. It is *built* with knowledge of this world — that is what "trained"
means — and at inference it reads nothing but the rendered state, which is all
`conductor.Conductor` offers. So the specialist's one memorised benchmark and the saboteur's choice
of the pool's dominated model are legitimate constructions rather than cheats: they are what a
miner's weights would encode. Nothing here touches a `TaskSpec` or a grader while an episode runs,
and every worker call goes through `Scaffold._call_worker`, so §2.1 holds for the adversaries too —
which is the point. The mechanism must beat them without a detector.

WHY THE WORLD LOOKS LIKE THIS. Three properties, each of which some archetype needs in order to be
a real adversary rather than a straw man:

  * **A strictly dominated model.** `mock/boutique` is weaker than `mock/mid` AND dearer than it —
    beaten on both axes by something already in the catalog. Without one, "always the same model"
    lands on the pool's own cost/quality line (§5.1b fits λ so the extremes tie), the degenerate and
    the miser score the same, and the ranking's bottom half tests nothing. Real catalogs of ~300
    models are full of dominated rows; a world without one is the easy case.
  * **One benchmark that a lookup table is worth more than `eps` on.** `automationbench` sits
    exactly one strength point above the cheapest model: the miser solves none of it and a $0.05
    model solves all of it. A specialist that memorises that single row moves the twelve-benchmark
    aggregate by +0.063 — past `eps` — which is what makes exit 1 a real test. If the subject
    benchmark were ordinary, the specialist would fail condition 1 and "the specialist loses" would
    pass for the wrong reason.
  * **A hard tier of four benchmarks**, whose easiest task already needs the middle rung — and one
    of which, `agents-last-exam`, runs past the strongest model's reach entirely, so some tasks are
    unsolvable at any price and every rung spent on them is pure loss (`benchmarks.mock` builds the
    world for exactly this case). There a cascade's discarded rungs cost more than its routing
    saves, so the honest archetype LOSES those four benchmarks in the duel it wins 8-4 — which is
    what gives the breadth rule something to be right about (M8 exit 2).

WHAT THIS WORLD SAYS ABOUT THE HONEST ARCHETYPE, MEASURED RATHER THAN ASSUMED. `HONEST` is §5.5's
free fixed cascade — the policy the owner could deploy without a subnet, and therefore the bar. It
tops the seven archetypes, but it is NOT the best fixed ladder over this catalog: three of the
thirty-one ladders beat it (`(nano, mid)` by +0.0787, the constant `mid` by +0.0621,
`(nano, small, mid)` by +0.0025), and every one of them wins the same way — by declining to buy the
hardest tail at top-rung prices. That is this repository's own finding reproduced inside the suite
(a good-value model is a formidable baseline), and it is pinned by a test rather than left as a
remark, with the corollary that matters for anyone tempted to fix it by repricing: **the top rung
cannot be made worth including by making it cheaper.** λ_b is fitted between the cheapest and the
strongest policy, so the priced cost of one full top-rung call is invariant to that rung's price;
only its measured quality can change the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .benchmarks.mock import MockBenchmark, worker_answers
from .config import MAX_STEPS
from .conductor import Conductor
from .devkit import suite_grade
from .reference import FixedPolicy, always_cheapest, always_strongest, cascade, fit_pool
from .scaffold import Scaffold
from .score import Exchange
from .types import Catalog, CatalogEntry, EpisodeResult, TaskSpec
from .worker import MockWorker

# --- the pool -------------------------------------------------------------------------------------
# Prices are dollars per million tokens, as OpenRouter reports them and as the render shows them
# (§3). Deliberately fake IDs: nothing here is a claim about a real provider.
CATALOG = Catalog(entries=(
    CatalogEntry("mock/apex", 4.00, 16.00, 400_000),
    CatalogEntry("mock/boutique", 2.50, 10.00, 128_000),
    CatalogEntry("mock/mid", 2.00, 8.00, 256_000),
    CatalogEntry("mock/nano", 0.05, 0.20, 64_000),
    CatalogEntry("mock/small", 1.00, 4.00, 128_000),
))

# What each model can solve (`benchmarks.mock`: a model at or above a task's difficulty completes
# every subgoal, and loses one per point it falls short).
STRENGTH: Mapping[str, int] = {
    "mock/nano": 1, "mock/small": 2, "mock/boutique": 2, "mock/mid": 4, "mock/apex": 6,
}

# THE POOL'S DOMINATED MODEL: weaker than `mock/mid` and dearer than it. "The worst available model"
# has to mean worst *value*, because that is the axis `final_b` scores on — a saboteur that picked
# the weakest CHEAP model would be the miser, and would rank above the degenerate rather than below
# it.
DOMINATED = "mock/boutique"

# One worker call is priced as 10k input + 10k output tokens at the catalog's own prices, so the
# dollars this world charges and the prices the Conductor is shown in the prompt cannot drift apart.
# A world whose catalog lied about relative cost would make every budget-aware archetype a fiction.
CALL_MTOK = 0.01
COST: Mapping[str, float] = {e.model_id: (e.price_in_per_mtok + e.price_out_per_mtok) * CALL_MTOK
                             for e in CATALOG.entries}

# --- the corpus -----------------------------------------------------------------------------------
# §6.1's twelve names, so a narrowed slice or a per-benchmark delta reads as the real thing. The
# difficulty mixes span the whole range the mechanism has to price: benchmarks the cheapest model
# clears, benchmarks only the top rung clears, and one that runs past every model in the pool.
_CORPUS = (
    #  name                 difficulty   subgoals
    ("terminal-bench",      (1, 3),      4),
    ("deepswe",             (4, 6),      4),
    ("agents-last-exam",    (4, 9),      1),   # past the strongest model: partly unsolvable
    ("automationbench",     (2, 2),      1),   # the specialist's subject — see the module docstring
    ("hle-tools",           (4, 6),      4),
    ("gdpval-aa",           (1, 4),      4),
    ("frontier-swe",        (4, 6),      1),
    ("programbench",        (1, 3),      1),
    ("cybergym",            (1, 4),      4),
    ("exploitbench",        (2, 5),      1),
    ("nl2repo",             (1, 2),      4),
    ("swe-marathon",        (1, 6),      1),
)

# ~20 tasks per benchmark is §6.3b invariant 2's "roughly equal counts" at §5.4's ~250-task target.
BENCHMARKS = tuple(MockBenchmark(name, STRENGTH, n_tasks=20, difficulty=difficulty,
                                 subgoals=subgoals)
                   for name, difficulty, subgoals in _CORPUS)
TASKS: tuple[TaskSpec, ...] = tuple(task for b in BENCHMARKS for task in b.load())
GRADE = suite_grade(BENCHMARKS)

# The window nonce: task order and every bootstrap seed derive from it (§4, §5.2), so the whole
# suite is reproducible from this one string.
NONCE = "v3-archetypes"

# The one benchmark the specialist has memorised, and the model that solves it. Exported because the
# breadth test asserts that this is the benchmark leave-one-out drops — naming it there rather than
# repeating the string is what keeps the fixture and the assertion from drifting apart.
SUBJECT = "automationbench"
SUBJECT_MODEL = "mock/small"


def arm(conductor: Conductor, tasks: tuple[TaskSpec, ...] = TASKS) -> tuple[EpisodeResult, ...]:
    """One policy over the corpus, through the shipped `Scaffold` — an arm, ready to score.

    Unbudgeted, because these arms measure ROUTING. §4's exhaustion zeroes a tail identically for
    both arms and is tested where it belongs (`tests/test_scaffold.py`); letting it fire here
    would silently convert the saboteur's ranking — it spends 6.5x the honest cascade — into a
    statement about the allowance rather than about the policy.

    A fresh `MockWorker` per call: it accumulates `seen` and `charged`, and a shared instance would
    mix one archetype's calls into the next one's record.
    """
    worker = MockWorker(answers=worker_answers(STRENGTH), costs=COST)
    scaffold = Scaffold(CATALOG, conductor, worker, GRADE)
    return tuple(e.result for e in scaffold.run_window(tasks, nonce=NONCE,
                                                       budget_usd=float("inf")))


def exchange() -> dict[str, Exchange]:
    """The world's measured λ_b / C_b table, from §6.1's own fixed-policy sweep.

    Derived through `reference.fit_pool`, never written down: the property the whole ranking rests
    on — the floor and King₀ score exactly the same (§5.1b, M8 exit 3) — is a consequence of fitting
    λ to this pool, and a hardcoded table would be a guess that merely resembled one. On this world
    King₀ is the cascade, so `MISER` ties the cascade rather than `SPENDTHRIFT`; `SPENDTHRIFT` sits
    below the line, which is what buying the strongest model's quality at its own price earns.
    """
    return fit_pool({policy.name: arm(policy) for policy in
                     (always_cheapest(CATALOG), always_strongest(CATALOG), cascade(CATALOG))}).exchange


# --- reading the rendered state ---------------------------------------------------------------

# `render._task`'s own line. Mirrored here rather than imported because it is the render's format,
# not a constant it exports, and a test renders a real state and asserts the read-back, so a
# template change fails there instead of quietly turning the specialist into a generalist.
#
# TAKEN FROM THE LEFT (`partition`), which is the opposite of `reference._progress`'s `rpartition`
# and for the same reason: the task section comes BEFORE the untrusted benchmark text, so the first
# occurrence is the real one and a forged `# TASK` header inside a task statement can only ever
# appear later. (`reference` reads history, which comes after that text, so it takes the last.)
_TASK_HEADING = "\n# TASK\nbenchmark: "


def _benchmark_of(prompt: str) -> str:
    """Which benchmark this episode belongs to, read out of the rendered state and nothing else."""
    _, heading, rest = prompt.partition(_TASK_HEADING)
    if not heading:
        raise ValueError("the rendered state carries no task section: a benchmark-conditional "
                         "policy cannot know which benchmark it is on, and guessing would score a "
                         "policy nobody wrote")
    return rest.split("\n", 1)[0]


# --- the seven archetypes -----------------------------------------------------------------------


@dataclass(frozen=True)
class Specialist:
    """§5.3's lookup-table miner: one memorised benchmark, the incumbent's policy everywhere else.

    THE FAILURE MODE THE BREADTH RULE EXISTS FOR. A miner cannot smuggle answers (§2.1), but can
    still learn "benchmark X → model Y wins" — and with twelve benchmarks that policy is twelve
    facts. This archetype is the first fact. It is deliberately built as *the king's own policy plus
    one row*, so every per-benchmark delta except one is exactly zero and leave-one-out lands on
    exactly 0.0: the refusal is arithmetic, not a threshold that happened to catch it.

    Note what CANNOT distinguish it from an honest router: both read the benchmark tag out of the
    same rendered state and both dispatch on it. Measurement 11 found copier agreement and honest
    convergence to be the same signal, i.e. undetectable, so the mechanism does not try to tell the
    two apart by inspection — it prices breadth (§5.2 condition 2) and lets the specialist keep its
    subject.
    """

    subject: str
    on_subject: Conductor
    elsewhere: Conductor
    name: str = "specialist"

    def act(self, prompt: str) -> str:
        policy = self.on_subject if _benchmark_of(prompt) == self.subject else self.elsewhere
        return policy.act(prompt)


@dataclass(frozen=True)
class Copier:
    """A different artifact that behaves identically to the king (D14, M8 exit 9).

    A separate object rather than the king itself, because the claim being tested is about
    BEHAVIOUR, not identity: two hotkeys, two uploads, byte-identical decisions. It ties on every
    task, so it ties on every resample, so `delta == 0.0` exactly and the strict `eps` beat refuses
    it. That is what converts anti-copy from a detection problem this repo measured to be
    unsolvable into an economic one — copying can tie, and a tie keeps the crown where it is.
    """

    king: Conductor
    name: str = "copier"

    def act(self, prompt: str) -> str:
        return self.king.act(prompt)


# §5.5's free cascade: cheapest first, escalate on an observed failure, stop on success. The only
# signal available to it is §3's sequential one — in this world difficulty is a hash of the task ID
# (`benchmarks.mock`), so no amount of reading the prompt reveals it — which is exactly what makes
# this the honest policy rather than a lucky one. It is the bar a Conductor must clear (§5.5) and it
# is built from `reference.cascade` rather than re-listed, so the archetype and the reference arm
# cannot drift apart.
HONEST = FixedPolicy("honest", cascade(CATALOG).rungs)

# Frugal incompetence (M8 exit 5) and the pool's own exchange rate paid in full (exit 4). These two
# ARE `reference`'s extremes under archetype names, and that is the point rather than a shortcut:
# λ_b is fitted between them, so their `final` scores tie by construction and the ranking's middle
# tier is the λ invariant restated end-to-end, through real episodes instead of synthetic arrays.
MISER = FixedPolicy("miser", always_cheapest(CATALOG).rungs)
SPENDTHRIFT = FixedPolicy("spendthrift", always_strongest(CATALOG).rungs)

SPECIALIST = Specialist(SUBJECT, FixedPolicy("subject", (SUBJECT_MODEL,)), MISER)

# One model, always, whatever the state says — a policy that has collapsed to a constant. Which
# constant is the whole question: on the cheapest or the strongest model this is the miser or the
# spendthrift, and both of those sit ON the pool line. It fixes on the dominated model, which is the
# only version of "degenerate" the mechanism has anything to say about.
DEGENERATE = FixedPolicy("degenerate", (DOMINATED,))

# The griefer: the dominated model, called until the step cap stops it (§8b.2). Same quality as the
# degenerate — it never calls anything else — at MAX_STEPS times the spend, so the pair is "same
# quality, more spend" at the bottom of the ranking exactly as honest/spendthrift is at the top.
# Conservative on purpose: built on `FixedPolicy`, it still stops on a solve, so it understates what
# a hostile model would burn rather than overstating it.
SABOTEUR = FixedPolicy("saboteur", (DOMINATED,) * MAX_STEPS)

COPIER = Copier(HONEST)

# In the order §5's exit criteria predict, best first. `SPENDTHRIFT` and `MISER` are one tier — the
# λ invariant makes them tie — and the test asserts that as an equality rather than an ordering.
ARCHETYPES: tuple[tuple[str, Conductor], ...] = (
    ("honest", HONEST),
    ("specialist", SPECIALIST),
    ("spendthrift", SPENDTHRIFT),
    ("miser", MISER),
    ("degenerate", DEGENERATE),
    ("saboteur", SABOTEUR),
    ("copier", COPIER),
)
