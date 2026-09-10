"""Humanity's Last Exam — the corpus's one non-code benchmark, and the test of whether the protocol
is general (M2a, §6.1).

WHY THIS ONE IS WORTH A MODULE OF ITS OWN. Every other benchmark in the corpus is code-shaped:
LiveCodeBench runs a program, SWE-bench applies a patch and runs a test suite, and both of them need
Docker, a per-task image or a driver script. A corpus made only of those lets a code-shaped
assumption hide inside `base.Benchmark` — a sandbox that is always required, a `Grade` whose `total`
is always a test count, a `tools()` that is always a shell. This adapter needs no container, no
image and no subprocess, and its `Grade` is out of 1. **The protocol survived that unchanged**, which
is the finding: `load` / `tools` / `grade` express a knowledge question with nothing bent. What is
*not* general is the package checklist in `benchmarks/__init__.py`, which requires "grading inside a
`--network none` container" of every adapter without qualification. Here there is nothing to
execute — grading is a string comparison against a gold letter — so the honest reading of that rule
is "any grader that EXECUTES worker output must do so in a container", and this module is the
counter-example that shows the difference matters.

**THIS REPLACES `real.HumanitysLastExam`, AND IT IS THIS ONE `benchmarks.ADAPTERS` REGISTERS.** That
one carries the same pin and the same three filters, but it cannot fetch the dataset (mechanical
note 1) and it answers to the same `name = "hle"`. `score.py` groups by `TaskSpec.benchmark` and
leave-one-out drops by it, so two adapters answering to one name in the same admitted set would split
a benchmark in two and double its weight in an aggregate that is defined to weight benchmarks equally
(§5.1). Until that block is deleted, exactly one of the two may be admitted.

**THE TASK SET COULD NOT BE FETCHED, THE CENSUS WAS NOT TAKEN, AND N HAS NOT MOVED YET.** `cais/hle`
is gated (`gated: auto`), and the account holding `OWNER_HF_TOKEN` has not accepted its terms:
`HfApi.auth_check("cais/hle", repo_type="dataset")` refuses with *"Access to dataset cais/hle is
restricted and you are not in the authorized list"*, measured 2026-09-01. The trap worth recording is
that `dataset_info` **succeeds** on a gated repo without acceptance — it returns the sha, the card
and the file list — so a reachability check written against metadata reports a dataset that no
`hf_hub_download` can pull. That is why the revision and the row schema below are pinned and correct
while `usable_tasks` is `None`.

Unblocking it is one action by the owner and it is theirs to take rather than this module's: the gate
asks the accepting account to *share its contact information* with the benchmark's authors, which is
a representation made in the owner's name. Accept at https://huggingface.co/datasets/cais/hle with
the account behind `OWNER_HF_TOKEN`, then take the census with

    python -c "from thirtyspokes.v3.benchmarks.hle import HumanitysLastExam as H; \
               print(len(H().load()))"

and write that number into `HLE_PROVENANCE.usable_tasks`. `test_bench_hle.py` holds a test that
is skipped while the gate is closed and fails until the recorded census matches what loads, so the
`None` is a tracked debt rather than a note.

**WHAT TO EXPECT FROM THE CLICK, AND WHY IT IS AN ESTIMATE AND NOT A CENSUS.** Counted on a
third-party re-upload of the same 2500-row public set — a copy, not the pin, and deliberately not
named here because the card refuses exactly that redistribution — the split is 591 multiple-choice
against 1909 `exactMatch`, 342 rows carrying an image, and **513 rows that are both multiple-choice
and text-only**. All 513 of those carry a gold that is exactly one capital letter, so the third
filter below costs nothing and the expected census is 513 rather than some fraction of it. The
marginals are 23.6% and 13.7% against the publisher's own published 24% and 14%, which is what makes
the copy credible enough to plan against; what it cannot settle is whether the copy is byte-identical
to the pinned revision. So the number to expect is **513, four and a half times LiveCodeBench's
112** — enough to make a 125-task stratum drawable (§6.3b, spend plan §1.2), which at N = 2 it is
not. Nothing may be sized from it until it is counted at the pin.

THE GRADER IS A REIMPLEMENTATION AND HAD TO BE. HLE ships an LLM judge (`hle_eval/
run_judge_results.py` prompts a frontier model per question) and §6.1.2 asks for the shipped grader,
but that one is disqualified twice over here: it costs a model call per graded task on top of the
episode's own spend, which `final_b` prices and would therefore charge a miner for the *grader*
(§5.1b), and it is non-deterministic, so the same submission can score differently in the king's arm
and the challenger's — the one thing a PAIRED comparison cannot survive (§5.2). On the
multiple-choice subset the judge's whole job is comparing an option letter, which is what makes the
substitution defensible; on the `exactMatch` three-quarters of the benchmark it is not, and those
rows are dropped rather than graded by a strict string match that would score near zero for everyone
and hand §6.1.3's admission probe a benchmark with no spread for an artefact of the grader.

THE PROMPT ASKS FOR THE LETTER, AND THAT IS WHAT MAKES A JUDGE-FREE GRADER HONEST. HLE's own system
prompt asks for `Answer: {your chosen answer}`, under which a model may legitimately answer with the
option's *text*; a letter comparison would then score the answer's formatting rather than its
correctness — and since the Conductor chooses which worker answers, that noise lands directly on the
routing decision being measured. Narrowing the slot to the option letter is owner text, identical for
both arms and for every miner, exactly as LiveCodeBench's "output ONLY the program source" is; the
question itself is still forwarded byte for byte (§2.1). `hle_option` is lenient in the other
direction as well, reading the letter out of `C) Paris`, so a model that answers in the shipped
format is not punished for it.

REDISTRIBUTION IS REFUSED IN TERMS, and this is the benchmark that makes §6.1.2's requirement bite.
The card says, verbatim: *"Please help us protect the integrity of this benchmark by not publicly
sharing, re-uploading, or distributing the dataset."* The licence is MIT and the licence to USE is
plainly granted; what is refused is putting a copy on the owner's R2 for a window to be
self-contained from. With one validator (D1) nothing else ever fetches the slice, and the published
traces carry `StepRecord.rendered_state_digest` rather than rendered state, so a committed list of
task IDs plus the owner's own copy is as reproducible as a single-validator design gets. **If a
future reveal ever publishes rendered states verbatim, that decision reverses and this benchmark
leaves the corpus.**

TWO MECHANICAL NOTES.

1. **The fetch passes a token, and `real._hf_file` does not.** `hf_hub_download` picks up `HF_TOKEN`
   or a stored login and knows nothing about `OWNER_HF_TOKEN`, which is the name this repository
   keeps the owner's credential under (`koth/neuron.py`). So a gated benchmark loaded through
   `real._hf_file` would 401 *even after the terms are accepted*, for a reason that looks exactly
   like the gate itself.
2. **Only five of the twelve columns are read.** The pinned parquet is 274 MB for 2500 rows, and
   nearly all of it is the two image columns (`image_preview`, `rationale_image`); `to_pylist()`
   over the whole table would materialise those as Python bytes for every row to reach five string
   fields. `HLE_COLUMNS` is the projection, and it is also a statement of exactly which fields this
   adapter depends on.
"""

from __future__ import annotations

import os
import re

from ..types import TaskSpec
from .base import Grade
from .real import GRADER_REIMPLEMENTED, REDISTRIBUTION_REFUSED, BenchmarkUnavailable, Provenance

HLE_NAME = "hle"
HLE_REPO = "cais/hle"

# The commit `cais/hle` resolved to on 2026-09-01, read from the Hub's own metadata (which is public
# even while the files are gated). A branch name would not be a pin: `λ_b` and `C_b` are fitted per
# corpus and pinned (§5.1b), so a benchmark that moves under the arena reprices every score already
# computed against it.
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
HLE_FILE = "data/test-00000-of-00001.parquet"

# The fields this adapter depends on, and the projection that keeps the two image columns out of
# memory — see mechanical note 2. Verified against the dataset card's declared features at the pinned
# revision, so the field NAMES are the repo's own; what the closed gate leaves unverified is the
# VALUES, which have been counted only on a copy.
HLE_COLUMNS = ("id", "question", "answer", "answer_type", "image")

# In this repository the owner's HuggingFace credential is `OWNER_HF_TOKEN` (`koth/neuron.py`), which
# `hf_hub_download` does not look for. `HF_TOKEN` second, because that is what the library and the
# CLI both use and an operator who exported it means it.
HLE_TOKEN_ENV = ("OWNER_HF_TOKEN", "HF_TOKEN")

# Appended to every load failure. The overwhelmingly likely cause is the gate, and a message that
# names the one action that fixes it is worth more than one that reports a 403.
HLE_GATE_NOTE = (f"{HLE_REPO} is gated (`gated: auto`): the account behind "
                 f"{'/'.join(HLE_TOKEN_ENV)} must accept its terms at "
                 f"https://huggingface.co/datasets/{HLE_REPO} — note that `dataset_info` succeeds "
                 "without acceptance, so metadata reachability is not access")

# Multiple-choice rows only. The other three quarters of HLE are `exactMatch` short answers whose
# shipped grader is an LLM judge — see the module docstring for why that cannot be this seam's
# grader, and why a strict string match is not an acceptable substitute for it.
HLE_ANSWER_TYPE = "multipleChoice"

# The answer line HLE's own format asks for, kept verbatim so a response formatted for the shipped
# judge is formatted for this grader too.
HLE_ANSWER_MARKER = "answer:"

# What an admitted row's gold may look like: ONE option token and nothing else. Deliberately stricter
# than the reader below, because the two rules protect different things — a lenient gold would admit
# a multi-answer row like `C, D`, whose first letter then grades as the whole answer and scores a
# submission that named half of it correct. On the copy counted in the module docstring every one of
# the 513 golds was a bare capital letter, so the tolerance for `(A)` and `A.` costs nothing there
# and is kept only because that copy is not the pin.
HLE_GOLD = re.compile(r"\(?[A-Za-z]\)?\.?")

# What a submission's answer line may look like: the option letter, optionally parenthesised, ending
# the line or followed by a separator — `B`, `(B)`, `B.`, `B) Paris`. Lenient on purpose, because the
# shipped judge accepts an answer that repeats the option text and a letter-only comparison would
# score formatting instead of knowledge — and which model formats how is a routing decision, so that
# noise would land on the very thing being measured.
HLE_OPTION = re.compile(r"\(?([A-Za-z])\)?\s*(?:$|[.):,;-])")

# Emphasis and quoting stripped off both ends of the answer line before it is read. Models routinely
# bold the answer (`**B**`), and a grader that scored that zero would be measuring markdown. Nothing
# INSIDE the line is touched, so this cannot turn prose into an option token.
HLE_WRAPPERS = "*_`\"' \t"

HLE_PROVENANCE = Provenance(
    repo=HLE_REPO, revision=HLE_REVISION,
    licence="MIT on the card, and the card also asks in terms that the dataset not be publicly "
            "shared, re-uploaded or distributed; the repo is gated behind accepted terms that share "
            "the accepting account's contact details with the authors",
    redistribution=REDISTRIBUTION_REFUSED,
    grader=GRADER_REIMPLEMENTED,
    grader_source="`hle_grade`: the option letter after HLE's own `Answer:` line, compared with the "
                  "gold. The shipped grader (`hle_eval/run_judge_results.py`) is an LLM judge and is "
                  "refused here for two reasons — it charges a model call per graded task to "
                  "`final_b`, and it is non-deterministic, which a PAIRED duel cannot survive.",
    deviation="published accuracy is judge-scored over all 2500 questions; this is a letter "
              "comparison over the text-only multiple-choice subset, under a prompt that asks for "
              "the letter rather than for `{your chosen answer}`. Comparable across arms, not "
              "against the leaderboard.",
    granularity="1/1 — binary, and unavoidably so: a multiple-choice question has no partial credit "
                "to report. This is the benchmark that proves `Grade` carries a binary grader "
                "honestly (`Grade(0, 1)` publishes its own coarseness) rather than the one that "
                "helps §5.4's task count, and every duel that includes it pays binary variance.",
    # COUNTED 2026-09-02, the day the owner accepted the gate. 2,500 rows in, 513 out: the filters
    # are multiple-choice AND text-only, and both are `load`'s, not a publisher's composition figure.
    # This field was `None` until it could be counted, and `test_bench_hle.py` failed on purpose
    # from the moment access was granted until it was — that tripwire did its job and the number
    # below is the one the adapter actually yields, verified twice.
    usable_tasks=513)


def hle_token() -> str | None:
    """The owner's HuggingFace credential, under either name, or None.

    Returned rather than exported: `hf_hub_download` is given it explicitly so that nothing else in
    the process acquires the owner's token as a side effect of loading a benchmark.
    """
    for name in HLE_TOKEN_ENV:
        token = os.environ.get(name)
        if token:
            return token
    return None


def hle_parquet() -> list[dict]:
    """The pinned task file, projected to `HLE_COLUMNS`, from the cache when it is already there.

    `hf_hub_download` and not `datasets.load_dataset`: the loader resolves a *config*, one more
    upstream-controlled indirection between a pinned revision and the rows that come back. A file
    name at a commit has none.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:                                   # pragma: no cover — env-dependent
        raise BenchmarkUnavailable("huggingface_hub is required to load a real benchmark "
                                   "(`uv pip install -e '.[benchmarks]'`)") from exc
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:                                   # pragma: no cover — env-dependent
        raise BenchmarkUnavailable("pyarrow is required to read a parquet task set "
                                   "(`uv pip install -e '.[benchmarks]'`)") from exc
    try:
        path = hf_hub_download(HLE_REPO, HLE_FILE, repo_type="dataset", revision=HLE_REVISION,
                               token=hle_token())
    except Exception as exc:                                     # noqa: BLE001 — gate, 404, offline
        raise BenchmarkUnavailable(f"could not fetch {HLE_FILE} from {HLE_REPO}@{HLE_REVISION}: "
                                   f"{exc}\n{HLE_GATE_NOTE}") from exc
    return pq.read_table(path, columns=list(HLE_COLUMNS)).to_pylist()


def hle_prompt(row: dict) -> str:
    """The question as it will be forwarded to a worker, plus the answer-format instruction.

    The question is HLE's own text byte for byte (§2.1); the three lines under it are owner text,
    identical for every miner and both arms, exactly as LiveCodeBench's "output ONLY the program
    source" is. The format is HLE's own two-line skeleton with one slot narrowed — see the module
    docstring for why asking for the letter is what makes a judge-free grader honest.
    """
    return (f"{row['question']}\n\n"
            "Respond in the following format:\n"
            "Explanation: {your reasoning}\n"
            "Answer: {the letter of your chosen option}")


def hle_option(text: str) -> str:
    """The option letter a response or a gold carries, case-folded, or "" if it carries none.

    TWO RULES POINTING AT OPPOSITE ENDS OF THE TEXT, ON PURPOSE. With HLE's own `Answer:` marker
    present, the answer is the FIRST non-empty line after the LAST such marker: last, because a model
    that echoes the format instruction before answering would otherwise be graded on the instruction;
    first after it, because anything below is the reasoning or the confidence line the shipped format
    also asks for. With no marker at all it is the LAST non-empty line — an unformatted reply reasons
    first and answers at the end, so reading its first line would grade the reasoning.

    A line that is not an option token returns "" rather than itself. There is no free-text
    comparison to fall back to: an admitted gold is a single letter, so prose could only ever match
    it by accident, and "the answer is A common misconception" must not score a question whose answer
    is A.
    """
    body = str(text or "")
    lowered = body.lower()
    marked = HLE_ANSWER_MARKER in lowered
    if marked:
        body = body[lowered.rfind(HLE_ANSWER_MARKER) + len(HLE_ANSWER_MARKER):]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    found = HLE_OPTION.match((lines[0] if marked else lines[-1]).strip(HLE_WRAPPERS))
    return found.group(1).casefold() if found else ""


def hle_grade(submission: str, gold: str) -> Grade:
    """Binary, and `Grade(_, 1)` says so — §5.4's cost of a benchmark that cannot do better.

    The gold is read through the same function as the submission so that the two are normalised
    identically, and an unreadable gold scores zero rather than matching an unreadable answer: rows
    are admitted only when `HLE_GOLD` accepts them, so this can only happen to a caller grading
    against something `hle_rows` would have dropped.
    """
    chosen, correct = hle_option(submission), hle_option(gold)
    return Grade(1 if correct and chosen == correct else 0, 1)


def hle_rows(rows: list[dict]) -> dict[str, dict]:
    """The mechanically gradeable subset, keyed by stable task ID and in file order.

    Three filters, each removing rows this seam cannot honestly score rather than rows it finds
    inconvenient:

      * `exactMatch` rows, whose shipped grader is an LLM judge this seam refuses to run and whose
        strict-string-match substitute would score near zero for every policy;
      * image-bearing rows, which the worker never sees — the scaffold forwards `TaskSpec.prompt`, a
        string, so a multimodal question would be graded on a question that was never fully asked,
        which every model fails for a reason that has nothing to do with routing;
      * rows whose gold is not one option token, which `hle_grade` could only decide by judging free
        text.

    File order and the dataset's own `id`, not an index: the window's task order comes from the nonce
    alone (`scaffold.task_order`), and an ID that renumbers between two loads makes two windows
    incomparable and the published traces (D15) unmatchable to the tasks they came from.
    """
    usable = {}
    for row in rows:
        if row.get("answer_type") != HLE_ANSWER_TYPE:
            continue
        if str(row.get("image") or "").strip():
            continue
        if not HLE_GOLD.fullmatch(str(row.get("answer") or "").strip()):
            continue
        usable[f"{HLE_NAME}-{row['id']}"] = row
    return usable


class HumanitysLastExam:
    """Expert-written knowledge questions, graded by comparing one option letter.

    The cheapest benchmark in the corpus to grade — no Docker, no per-task image, no repository
    checkout, no subprocess — which is what makes it the one most likely to survive a tight validator
    budget (§8b.2), and the one whose grading cost does not grow with the slice.
    """

    name = HLE_NAME
    provenance = HLE_PROVENANCE

    def __init__(self) -> None:
        self._rows: dict[str, dict] | None = None

    def load(self) -> tuple[TaskSpec, ...]:
        return tuple(TaskSpec(task_id=task_id, benchmark=self.name, prompt=hle_prompt(row),
                              tools=self.tools())
                     for task_id, row in self._loaded().items())

    def tools(self) -> tuple[str, ...]:
        """Empty, and here for a reason that is about the benchmark rather than about the harness: a
        knowledge question has no tree to read and no tool surface to declare in the first place,
        which is the part that shows the protocol is not code-shaped."""
        return ()

    def environment(self, task: TaskSpec) -> None:
        """None, exactly because `tools()` is empty (`base.Benchmark.environment`)."""
        return None

    def grade(self, submission: str, task: TaskSpec) -> Grade:
        """Whether the submitted option letter is the gold one. Nothing is executed, so — unlike
        every other adapter in this package — there is no sandbox here and no failure this can raise
        that is the validator's rather than the submission's."""
        return hle_grade(submission, self._loaded()[task.task_id]["answer"])

    def _loaded(self) -> dict[str, dict]:
        if self._rows is None:
            self._rows = hle_rows(hle_parquet())
        return self._rows
