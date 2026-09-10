"""The corpus the duel is scored on — the contract, the offline worlds, and the real adapters.

`base.py` is the contract every adapter implements; `mock.py` is the deterministic offline world
every other test builds against. `real.py`, `hle.py`, `r2egym.py` and `swebench_pro.py` hold the
adapters that fetch real task sets and grade in Docker, and `ADAPTERS` at the foot of this file is
the registry of them. Importing one costs nothing — every fetch and every container is behind a
call — so this package stays importable and testable with neither network nor Docker (the build plan M2a
is explicitly `$0`, harness only).

THE TWELVE §6.1 NAMED, AND WHAT IS ACTUALLY THERE. Terminal-Bench 3.0 · DeepSWE v1.1 · Agents' Last
Exam · AutomationBench · HLE w/ Tools · GDPval-AA v2 · FrontierSWE · ProgramBench · CyberGym ·
ExploitBench · NL2Repo · SWE-Marathon. Checked against HuggingFace on 2026-09-01, **four of those
twelve resolve** (DeepSWE, HLE, GDPval, CyberGym); Terminal-Bench is real but GitHub-distributed,
and seven did not resolve at any obvious path. `real.py` therefore ships three adapters, two of
which (LiveCodeBench, SWE-bench Verified) are not on the list at all — they were chosen for being
tractable and real. Its module docstring carries the per-benchmark record.

*"Twelve" is the target, not a constant* (§6.3b): admission drops benchmarks and rotation adds them,
so nothing anywhere may hard-code the number — and on today's measurement the reachable pool is
nearer seven, so a check written against twelve would refuse every window that can actually be run.

BEFORE AN ADAPTER IS WRITTEN — three checks, in this order, each of which can veto the work outright
(§6.1, risk register #4: several of the twelve could not be confirmed at that exact name/version, so
this is a live veto rather than a formality):

  1. **AVAILABILITY AT A PINNED VERSION.** A dataset revision, not a branch name. The corpus is what
     `λ_b` and `C_b` are fitted against and they are pinned per corpus (§5.1b), so a benchmark that
     moves under the arena silently reprices every score computed against it — and the miners who
     trained on the published traces trained against the old one.
  2. **A SHIPPED GRADER**, used as shipped. Never a reimplementation: §3's argument that scaffold
     quality *cancels* between the two arms holds only because both arms are graded identically by
     something neither of them chose, and a home-grown grader would move the level per benchmark
     while looking like a score.
  3. **A LICENCE TO REDISTRIBUTE** via the owner's R2 (§6.1.2, M2a exit 4 — non-trivial for HLE and
     SWE-bench derivatives). A window must be self-contained or the two arms face different worlds
     (§6.2), so a benchmark that cannot be redistributed cannot be committed into one. Record the
     licence per benchmark; "probably fine" is how a subnet acquires a takedown mid-window.

WHAT AN ADAPTER MUST THEN SUPPLY:

  * **`name`**, equal to the `TaskSpec.benchmark` tag on every task it loads. `score.py` groups by
    that tag and leave-one-out drops by it, so a mismatch splits one benchmark into two and doubles
    its weight in an aggregate that is defined to weight benchmarks equally (§5.1).
  * **`load()` with stable task IDs that survive a re-fetch.** Slices are drawn by ID and the reveal
    publishes them (§6.2); an ID that renumbers makes two windows incomparable and makes the
    published traces (D15) unmatchable to the tasks they came from.
  * **`tools()`** — the exact tool surface, stamped onto every task it loads. It is rendered into
    the prompt (`render._task`), so a declaration that disagrees with what the sandbox provides is
    a prompt that lies to the Conductor about its own action space.
  * **`grade()` at the benchmark's finest granularity** — fraction of tests passing, subgoals
    completed. See `base.Grade`: binary is the expensive default, not the safe one.
  * **Grading inside a `--network none` container** (M2a exit 5). The grader executes untrusted
    worker output on the single machine that holds the subnet's scoring authority, which is the same
    threat model that makes `.bin` uploads refused (§1.1.2).
  * **A task-count census.** It feeds M7a's pool-profiling cost estimate and M3a exit 4.
  * **An admission probe** (§6.1.3): a real spread between fixed policies over ~50 tasks. A
    benchmark where the scaffold does ~all the work contributes cost and no signal.

WHAT AN ADAPTER MUST NOT SUPPLY: anything derived from the Conductor. Tool definitions, the sandbox
and the task statement all come from the benchmark, and the task text reaches a worker byte for byte
(§2.1). That invariant is what permits public benchmarks to be used at all, and an adapter is one
place where a "helpful" prompt wrapper would look like an ordinary improvement.

The cost probe (M2a exit 6) belongs with the first adapters and is the **last** window in which
`MAX_STEPS` and the token caps may move (`config.py`): it measures real turns and dollars per
episode, and everything downstream is sized from it rather than from the plan's estimate table.
"""

from .hle import HumanitysLastExam
from .r2egym import R2EGym
from .real import LiveCodeBench, SweBenchVerified
from .swebench_pro import SweBenchPro

# THE REGISTRY — every adapter this package ships that a window can load. It lives here rather than
# in `real.py` because `hle.py`, `r2egym.py` and `swebench_pro.py` each import `real.py` for the
# `Provenance` record and the fetch helpers, so a registry inside that module could only reach them
# through a circular import. The one that used to sit there named the two adapters that happened to
# live in the same file; the other four were imported nowhere in `src/` and no window could load
# them, including the one written to move `N`.
#
# REGISTERED IS NOT ADMITTED, and running the two together would put a corpus decision in an import
# list. This tuple is what is reachable and gradeable; which benchmarks carry weight in a window is
# §6.1.3's admission decision, taken on measured spread and recorded in the corpus record.
# R2E-Gym is registered and measured dead under the current one-shot scaffold; both SWE-bench
# variants are registered and disk-bound to task counts no stratum can fill. Nothing here says
# otherwise, because that is not what registration is for.
#
# NAMES MUST BE UNIQUE ACROSS IT, which is a scoring rule and not tidiness. `score.py` groups by
# `TaskSpec.benchmark` and leave-one-out drops by it (§5.2), so two entries answering to the same
# `name` split one benchmark into two and DOUBLE its weight inside an aggregate defined to weight
# benchmarks equally (§5.1). Nothing would raise; the verdict would just be wrong.
#
# WHAT IS DELIBERATELY ABSENT, since a registry is also a record of the refusals:
#   * `real.HumanitysLastExam` — superseded by `hle.py`, which carries the gate handling and the
#     same `name = "hle"`. Exactly one of the two may ever be registered, and it is the one whose
#     fetch passes `OWNER_HF_TOKEN`; the copy here would 401 even once the terms are accepted.
#   * `mock.MockBenchmark` — the deterministic offline world the rest of the suite is built on. It
#     fetches no task set and grades nothing a model wrote.
#   * `gdpval.py` — REFUSED rather than missing (`gdpval.ADMITTED is False`): its shipped grader is
#     a human expert panel whose automated stand-in is a model, which check 2 above vetoes before
#     any code is written. It ships no `Benchmark` class at all, which is that refusal in code.
ADAPTERS = (LiveCodeBench, SweBenchVerified, HumanitysLastExam, R2EGym, SweBenchPro)
