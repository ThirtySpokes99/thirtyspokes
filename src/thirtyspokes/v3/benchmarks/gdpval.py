"""GDPval is REFUSED, and this module is the evidence rather than an adapter (§6.1 check 2).

`openai/gdpval` is on §6.1's list of twelve, it resolves, it is ungated, and it is 220 tasks across
44 occupations — every reason to want it. It is nonetheless **not mechanically gradeable**, so
admitting it would mean putting a judge inside the scorer, and `benchmarks/__init__.py`'s second
pre-adapter check ("a shipped grader, used as shipped") vetoes it before any code is written. What
follows is the measurement that veto rests on, taken at the pinned revision on 2026-09-01.

THREE INDEPENDENT DISQUALIFICATIONS. Any one of them is fatal; the point of listing all three is that
no amount of adapter work removes them, so this is not a "later" item.

1. **THE SHIPPED GRADER IS A HUMAN EXPERT PANEL, AND ITS AUTOMATED STAND-IN IS A MODEL.** GDPval's
   own metric is a *win rate* from blinded expert pairwise comparison — occupation experts ranking
   unlabelled deliverables, at "over an hour" per comparison for this gold subset. OpenAI also
   published an experimental automated grader (hosted at evals.openai.com), which agrees with the
   human experts **66% of the time** against a 71% human inter-rater agreement. Neither can be used
   here, for the reasons `real.py` already records against HLE's `exactMatch` half — and both are
   worse in the same direction. A model judge costs a completion call per graded task, which §5.1b
   prices into `final_b` and would therefore charge a miner for the grader; and it is
   non-deterministic, so the same submission can score differently in the king's arm and the
   challenger's, which is the one thing a PAIRED comparison cannot survive (§5.2). On HLE the judge
   decides a letter comparison, which is what makes the substitution there defensible. Here it would
   decide a median of 47 open-ended semantic questions per task about the contents of a binary
   office file: the judge is not a tie-break at the margin, it *is* the grader, and a grader that is
   itself a scored policy is exactly what the arena exists to avoid. The 34% disagreement figure is
   worth reading against §5.4's assumed 20% — no task count buys back a grader that wrong.

2. **THE ANSWER IS A FILE, AND THE SCAFFOLD RETURNS A STRING.** `scaffold._call_worker` makes one
   completion call and receives text. 185 of the 220 tasks name a gold deliverable and **182 of them
   are binary office or media formats** (.pdf ×85, .xlsx ×65, .docx ×64, .pptx ×17, .zip, .mp4).
   The rubrics say so in terms — 28.0% of all 10,453 criteria assert something about the artifact's
   structure ("the workbook contains a worksheet named exactly 'Sample Size Calculation'", "the deck
   contains between 8 and 10 slides", "the delivered file's total length is 3 minutes 24 seconds").
   A text answer fails those for a reason that has nothing to do with routing, which is the same
   defect `real.hle_rows` filters image-bearing questions out for.

3. **125 OF THE 220 PROMPTS ARE INCOMPLETE WITHOUT AN ATTACHMENT.** They carry reference files
   (.xlsx ×86, .pdf ×74, .docx ×67, plus audio and CAD) and refer to them directly — "the attached
   spreadsheet titled 'Population' contains…". `TaskSpec.prompt` is a plain string forwarded byte
   for byte (§2.1), so those tasks would be graded on a question that was never fully asked.

**AND THE SURVIVORS DO NOT RESCUE IT.** Filtering to tasks that are self-contained AND whose gold
deliverable is a text format leaves **3 of 220** (`TEXT_SUFFIXES` is deliberately generous — it
counts .ipynb, .yaml and .overpassql). Their rubrics are prose too: "the query targets Interstate 40
using an interstate-safe selector", "describes checksum/ETag validation at file or part completion".
A handful of criteria per task *are* machine-checkable ("the OpenAPI YAML parses as valid YAML"), and
grading a task on the four criteria of fifty that a parser can reach would move that benchmark's
level while still looking like a score — `real.py`'s own warning about a reimplemented grader, at its
worst. Three tasks also cannot fill a stratum: §6.3b wants ~125 tasks per benchmark, and the reason
GDPval was worth a day is that it might have moved `N`.

**THE RUBRIC IS NOT EVEN A `Grade`.** Nothing in `rubric_json` is a machine-readable check: across
all 10,453 criteria the `required`, `read_only` and `form_content` fields are null **every time**, so
the criterion text is the whole instrument. The weights are a scored sum, not a count — 1 to 20
points each, and **94 criteria across 18 tasks carry NEGATIVE scores, down to −85**. Even handed a
perfect judge, a submission tripping one produces a total `Grade` refuses to hold, correctly:
`quality_b` is a mean of these and `final_b` subtracts a priced spend from it.

THE FIVE FACTS THE ROUND ASKS FOR, since a refusal has to be as documented as an admission:

| | GDPval |
|---|---|
| revision | `11e7900cdcac61bc4daf59e65feb238acda98fbf` (a commit, checked by `Provenance`) |
| licence | **none declared** — no `license` tag and no `license:` in the card's front matter |
| rows vs gradeable | 220 rows, **0 gradeable** — the widest candidate subset is 3, and it fails too |
| grader provenance | neither as-shipped nor reimplemented — the shipped grader is a **judge** |
| granularity | N/A. Were there a judge: a weighted sum with penalties, not `Grade(k, n)` |

Two smaller notes for whoever revisits this. The card carries a contamination canary string
(`Canary gdpval:fdea:…`), which is worth preserving if any of this text is ever republished. And the
licence gap is worse than SWE-bench Verified's: there the card declares nothing but the harness is
MIT, whereas here the artifacts are real business documents from named occupations with no grant of
any kind attached — so §6.1.2's redistribution question does not even have a document to read.

WHAT WOULD CHANGE THIS ANSWER, so the refusal is falsifiable: a tool-using scaffold that lets a
worker read the reference files and emit a real .xlsx (finding 1 in `real.py`), *plus* a mechanical
rubric — criteria as executable predicates rather than prose. The first is plausible future work.
The second is a different dataset.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

# The fetch helpers, not a copy of them: an adapter and a refusal should be reading the same bytes
# through the same pinned-revision path, or the census here is not about the data `real.py` would
# have loaded. Nothing else is imported, and nothing here is added to `benchmarks.ADAPTERS`.
from .real import Provenance, _hf_file, _parquet_rows

NAME = "gdpval"
REPO = "openai/gdpval"
REVISION = "11e7900cdcac61bc4daf59e65feb238acda98fbf"
FILE = "data/train-00000-of-00001.parquet"

# A third value beside `real.GRADER_AS_SHIPPED` and `real.GRADER_REIMPLEMENTED`, defined here rather
# than added there because it is not an option an adapter may pick: it is the reason there is no
# adapter. The shipped grader exists, works, and is a person or a model.
GRADER_BY_JUDGE = "by-judge"

# There is no admitted benchmark in this module. Stated as data because §6.3b's rule is that `N` is
# the admitted set, and a reader counting files in this package would otherwise count this one.
ADMITTED = False

# Deliberately generous, because the filter it feeds is the one that decides whether a gradeable
# SUBSET exists: every format here is one a completion could plausibly emit as text. Being generous
# and still landing on 3 of 220 is the finding.
TEXT_SUFFIXES = frozenset({".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".py", ".ipynb",
                           ".sql", ".overpassql", ".xml", ".html", ".js", ".sh"})

PROVENANCE = Provenance(
    repo=REPO, revision=REVISION,
    licence="NONE DECLARED — the dataset repo carries no `license` tag and the card's front matter "
            "has no `license:` key; the arXiv paper page's CC BY 4.0 is the paper's licence, not a "
            "grant over 549 real business documents",
    redistribution="refused-by-absence: §6.1.2 needs a licence to read and there is not one",
    grader=GRADER_BY_JUDGE,
    grader_source="the dataset repo ships NO grading code (552 files = 549 reference/deliverable "
                  "artifacts + README + .gitattributes + the parquet). GDPval's own metric is a win "
                  "rate from blinded human expert pairwise comparison, over an hour per comparison; "
                  "the published automated stand-in is a hosted model grader at 66% agreement with "
                  "those experts (human inter-rater agreement is 71%).",
    deviation="N/A — nothing was adapted. Any v3 score here would be a home-grown grader over a "
              "prose rubric, which is the deviation, not a description of one.",
    granularity="N/A. The rubric is a WEIGHTED SUM with penalties (scores 1..20, and 94 criteria at "
                "-1..-85), so it is not `Grade(completed, total)` even with a perfect judge",
    usable_tasks=0)


@dataclass(frozen=True)
class Census:
    """The counts each disqualification rests on, so the verdict is checkable rather than recalled.

    Written down as a value and re-derived by `census` for the same reason `real.Provenance` is a
    field and not a docstring: the corpus rotates and `openai/gdpval` can gain rows. A refusal whose
    evidence lives only in prose is one that gets re-argued from memory by the next person, and a
    refusal that cannot notice the data moving under it is worse than no refusal at all.
    """

    tasks: int
    needing_reference_files: int        # the prompt is incomplete without an attachment
    with_a_file_deliverable: int        # the gold answer is a file, of any format
    with_a_binary_deliverable: int      # …and one the scaffold's string answer cannot be
    self_contained_and_textual: int     # the best case: no attachment, text-format deliverable
    rubric_criteria: int
    criteria_with_a_machine_readable_check: int
    criteria_with_a_negative_score: int


# What a run against the pinned revision saw on 2026-09-01. `test_bench_gdpval.py` re-derives it.
MEASURED = Census(tasks=220,
                  needing_reference_files=125,
                  with_a_file_deliverable=185,
                  with_a_binary_deliverable=182,
                  self_contained_and_textual=3,
                  rubric_criteria=10453,
                  criteria_with_a_machine_readable_check=0,
                  criteria_with_a_negative_score=94)

# The number that would have moved `N` (§1.2 of the spend plan). It is zero, and it is not the same
# number as `MEASURED.self_contained_and_textual`: those 3 tasks are *loadable*, and still need a
# judge to score. "Loads 220 rows" is not "moved N".
GRADEABLE_TASKS = 0


def _is_textual(deliverables: list[str]) -> bool:
    """Whether every gold deliverable is a format a completion could have emitted as its answer."""
    return bool(deliverables) and all(
        pathlib.PurePath(name).suffix.lower() in TEXT_SUFFIXES for name in deliverables)


def census(rows: list[dict]) -> Census:
    """Re-derive `MEASURED` from the dataset's own rows.

    Pure over rows so the whole verdict can be exercised on a fixture offline, and run against the
    real parquet in the one network-marked test — the same split `real.py`'s pure row functions and
    their tests already use.
    """
    criteria = [item for row in rows for item in json.loads(row["rubric_json"])]
    return Census(
        tasks=len(rows),
        needing_reference_files=sum(1 for row in rows if row["reference_files"]),
        with_a_file_deliverable=sum(1 for row in rows if row["deliverable_files"]),
        with_a_binary_deliverable=sum(1 for row in rows if row["deliverable_files"]
                                      and not _is_textual(row["deliverable_files"])),
        self_contained_and_textual=sum(1 for row in rows if not row["reference_files"]
                                       and _is_textual(row["deliverable_files"])),
        rubric_criteria=len(criteria),
        # `required`, `read_only` and `form_content` are the only fields that could have carried an
        # executable check. All three are null on every criterion, which is what makes "the criterion
        # text is the whole instrument" a measurement rather than a reading of the samples.
        criteria_with_a_machine_readable_check=sum(
            1 for item in criteria
            if any(item.get(field) is not None
                   for field in ("required", "read_only", "form_content"))),
        criteria_with_a_negative_score=sum(1 for item in criteria if item["score"] < 0))


def load_rows() -> list[dict]:
    """The 220 rows at the pinned revision. Raises `BenchmarkUnavailable` offline (from `_hf_file`).

    Present so the refusal is re-runnable and not merely asserted; it is NOT a `Benchmark.load` —
    there is no `TaskSpec` here, because a task nothing can grade is not a task this corpus has.
    """
    return _parquet_rows(_hf_file(REPO, FILE, REVISION))
