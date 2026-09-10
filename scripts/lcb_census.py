"""Census LiveCodeBench's six release files — the free corpus lever of the spend plan §1.3, priced.

WHY THIS EXISTS. `real.LiveCodeBench` reads `test6.jsonl` alone (175 rows, 112 usable), and §1.2
measured what that costs: a ~250-task slice over N = 2 benchmarks wants 125 LiveCodeBench tasks, the
benchmark holds 112, so a window would draw EVERY task the benchmark has and §6.3b/M7a's protection
— "the slice is drawn after commits, so a miner cannot know which tasks will be scored" — protects
nothing. The other five files are the same problem set at the SAME commit, so enlarging the pool is
a download and a census rather than an integration. The price is contamination: the earlier releases
are older competition problems, far likelier to sit in a worker model's pretraining data, and §5.3's
whole anti-memorisation argument is about exactly that. This script produces the numbers that
decision needs; it changes no adapter and enables nothing.

NOTHING HERE RE-EXPRESSES `real.py`'S CRITERIA. `usable` is `real.lcb_rows` and the graded
denominator is `len(real.lcb_cases(row))`, both called on the row itself, so a census that disagreed
with the adapter would be one bug rather than two definitions of "usable". That is also why the
script reports GRADEABLE separately: `lcb_rows` admits a row on platform and starter code, while
`LiveCodeBench.grade` REFUSES a row with no stdin case (it raises rather than scoring zero, §8b.3),
so "usable" and "a task a duel can actually score" are two different counts and only the second one
moves N.

NO RELEASE FILE IS KEPT ON DISK. 4.49 GB of files against the ~79 GB free that §1.3 measured and
that several jobs share, so rows are streamed and parsed as they arrive and the bytes are dropped; a
file already in the HuggingFace cache is read from there instead of re-fetched. Standing disk cost
is zero. It also means a rerun of the five uncached files re-downloads them — the census is a
one-off input to an owner decision, not something a window runs.

    python scripts/lcb_census.py                       # all six release files (~4.5 GB streamed)
    python scripts/lcb_census.py --files test6.jsonl   # one file, from the cache
    python scripts/lcb_census.py --json census.json    # the numbers, machine-readable
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from thirtyspokes.v3.benchmarks import real  # noqa: E402

# The six files in the order `code_generation_lite.py`'s ALLOWED_FILES concatenates them. A release
# tag names a PREFIX of this list (`release_v4` == files 1-4), so every file is an INCREMENT and the
# cumulative column below is what a given release tag would actually load.
RELEASE_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
                 "test6.jsonl")

# How many bytes of the HTTP body to pull at a time. The default 512 would make 4.49 GB nine million
# round trips through the iterator; a single row also runs to ~12 MB in `test4.jsonl` (the private
# test cases dominate the file), so the buffer has to be comfortable with a line that size anyway.
STREAM_CHUNK_BYTES = 1 << 22


@dataclasses.dataclass
class FileCensus:
    """One release file's numbers. Counts are over rows; `usable_ids` is kept for the overlap check.

    `usable_rows` and `gradeable_rows` are separate because the adapter treats them separately (see
    the module docstring), and `full_granularity_rows` is here because §5.4's task count depends on
    it: a task graded out of 12 carries far less variance than one graded out of 2, so a release
    whose problems ship few cases buys fewer *effective* tasks than its usable count suggests.
    """

    filename: str
    total_rows: int = 0
    usable_rows: int = 0
    gradeable_rows: int = 0
    public_only_rows: int = 0
    full_granularity_rows: int = 0
    # Graded denominators below the cap, as `{n: rows}`. Kept because `LCB_PROVENANCE.granularity`
    # asserts "every usable problem has at least 2 cases" and that was measured on `test6.jsonl`
    # alone: an older release holding a one-case problem would make the adapter BINARY on that task
    # while its provenance string still says otherwise, and §5.4 sizes a duel off exactly this.
    cases_below_max: dict[str, int] = dataclasses.field(default_factory=dict)
    date_min: str = ""
    date_max: str = ""
    usable_date_min: str = ""
    usable_date_max: str = ""
    difficulty: dict[str, int] = dataclasses.field(default_factory=dict)
    usable_by_year: dict[str, int] = dataclasses.field(default_factory=dict)
    usable_ids: list[str] = dataclasses.field(default_factory=list)


def _stream_rows(filename: str, revision: str = real.LCB_REVISION):
    """Yield the parsed rows of one release file without leaving it on disk.

    Reads the HuggingFace cache when the file is already there — `test6.jsonl` is, because the
    adapter loads it — and streams over HTTPS otherwise. `hf_hub_download` is deliberately not used
    for the uncached files: it would leave 4.35 GB in the cache to answer a question that is asked
    once, on a volume §1.3 measured at 91% full.
    """
    from huggingface_hub import hf_hub_url, try_to_load_from_cache

    cached = try_to_load_from_cache(real.LCB_REPO, filename, repo_type="dataset", revision=revision)
    if isinstance(cached, str):
        with open(cached) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return

    import requests

    url = hf_hub_url(real.LCB_REPO, filename, repo_type="dataset", revision=revision)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=STREAM_CHUNK_BYTES, decode_unicode=True):
            if line:
                yield json.loads(line)


def census_rows(filename: str, rows) -> FileCensus:
    """The census of one file, over an iterable of its rows.

    Separated from the fetch so the counting is a pure function of the rows: the same code produces
    the cached file's numbers and a streamed one's, and a disagreement between two runs is a change
    in the data rather than in how it was read.
    """
    out = FileCensus(filename=filename)
    difficulty: collections.Counter = collections.Counter()
    small: collections.Counter = collections.Counter()
    by_year: collections.Counter = collections.Counter()
    dates: list[str] = []
    usable_dates: list[str] = []
    for row in rows:
        out.total_rows += 1
        date = str(row.get("contest_date") or "")[:10]
        if date:
            dates.append(date)
        if not real.lcb_rows([row]):        # the adapter's own filter, called rather than restated
            continue
        out.usable_rows += 1
        out.usable_ids.append(f"{real.LCB_NAME}-{row['question_id']}")
        difficulty[row["difficulty"]] += 1
        if date:
            usable_dates.append(date)
            by_year[date[:4]] += 1
        public = len([c for c in json.loads(row["public_test_cases"])
                      if c.get("testtype") == "stdin"])
        cases = len(real.lcb_cases(row))
        if cases:
            out.gradeable_rows += 1
        if cases and cases <= public:
            # `lcb_cases` falls back to the public samples when the private block will not decode.
            # Such a task is graded on 1-5 samples the problem statement already showed the worker,
            # which is a weaker check rather than a broken one — but it is a different check, and a
            # release made mostly of them would inflate the usable count without adding evidence.
            out.public_only_rows += 1
        if cases == real.LCB_MAX_TESTS:
            out.full_granularity_rows += 1
        elif cases:
            small[cases] += 1
    out.cases_below_max = {str(n): small[n] for n in sorted(small)}
    out.difficulty = dict(sorted(difficulty.items()))
    out.usable_by_year = dict(sorted(by_year.items()))
    out.date_min, out.date_max = (min(dates), max(dates)) if dates else ("", "")
    out.usable_date_min, out.usable_date_max = ((min(usable_dates), max(usable_dates))
                                                if usable_dates else ("", ""))
    return out


def report(censuses: list[FileCensus]) -> str:
    """The three tables the §1.3 decision needs, as markdown, so the write-up quotes rather than
    retypes them: the per-file census with its cumulative column, the contamination profile by
    contest year, and the slice arithmetic §1.2 point 4 breaks on."""
    lines = ["| file | rows | usable | gradeable | cum. gradeable | contest dates (usable) |",
             "|---|---|---|---|---|---|"]
    cumulative = 0
    for c in censuses:
        cumulative += c.gradeable_rows
        window = f"{c.usable_date_min} .. {c.usable_date_max}" if c.usable_date_min else "-"
        lines.append(f"| `{c.filename}` | {c.total_rows} | {c.usable_rows} | {c.gradeable_rows} | "
                     f"{cumulative} | {window} |")

    years = sorted({year for c in censuses for year in c.usable_by_year})
    lines += ["", "| file | " + " | ".join(years) + " | easy/medium/hard | 12-case | public-only |",
              "|---|" + "---|" * (len(years) + 3)]
    for c in censuses:
        cells = " | ".join(str(c.usable_by_year.get(year, 0)) for year in years)
        d = c.difficulty
        lines.append(f"| `{c.filename}` | {cells} | "
                     f"{d.get('easy', 0)}/{d.get('medium', 0)}/{d.get('hard', 0)} | "
                     f"{c.full_granularity_rows} | {c.public_only_rows} |")

    total = sum(c.gradeable_rows for c in censuses)
    below: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()
    for c in censuses:
        below.update({int(n): k for n, k in c.cases_below_max.items()})
        seen.update(c.usable_ids)
    duplicates = [task_id for task_id, n in seen.items() if n > 1]
    lines += ["",
              f"gradeable total: {total}",
              f"graded out of {real.LCB_MAX_TESTS}: {sum(c.full_granularity_rows for c in censuses)}"
              f"; below the cap: {dict(sorted(below.items()))}",
              f"duplicate task IDs across files: {len(duplicates)}"
              + (f" (e.g. {duplicates[:3]})" if duplicates else ""),
              # §1.2 point 4: at N = 2 a ~250-task slice wants 125 tasks from this benchmark, and a
              # stratum that is the whole benchmark is not a draw. The fraction is the number that
              # decides whether the corpus lever fixes it.
              f"125-task stratum as a fraction of the pool: {125 / total:.1%} "
              f"(at test6.jsonl alone: {125 / censuses[-1].gradeable_rows:.1%})"
              if total and censuses[-1].gradeable_rows else ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="+", default=list(RELEASE_FILES),
                        help="release files to census, in ALLOWED_FILES order")
    parser.add_argument("--json", type=pathlib.Path, help="write the raw census here")
    args = parser.parse_args(argv)

    censuses = []
    for filename in args.files:
        print(f"[census] {filename} …", file=sys.stderr, flush=True)
        censuses.append(census_rows(filename, _stream_rows(filename)))
        print(f"[census] {filename}: {censuses[-1].gradeable_rows} gradeable of "
              f"{censuses[-1].total_rows}", file=sys.stderr, flush=True)

    print(f"livecodebench/code_generation_lite @ {real.LCB_REVISION}")
    print(report(censuses))
    if args.json:
        args.json.write_text(json.dumps([dataclasses.asdict(c) for c in censuses], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
