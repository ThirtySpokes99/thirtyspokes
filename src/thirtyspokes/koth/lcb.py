"""LiveCodeBench — the only traffic this project measured that is worth routing over.

WHY THIS BENCHMARK EXISTS IN THE SUITE. The scored suite ranks 100% on GSM8K, and eight measurements
say that is the wrong traffic to compete over: every pool model lands at 79-97%, the oracle gap is
**+0.019**, and the best router beat the featureless baseline by **+0.002**. Since the router scalar
divides by that gap, a saturated benchmark does not merely fail to reward routing — it turns
8-question sampling noise into a leaderboard. SWE-bench Pro one-shot is degenerate from the other
end (0/120 solved; the published 62-80% come from agentic scaffolds, so the *format* is wrong).

LiveCodeBench is the one regime found where models land in the MIDDLE and disagree: scores span
0.47-0.78, 13.9% of problems are unsolvable by anyone, it produced the first non-zero sole-correct
rate (kimi-k3, 8.3%), and the oracle gap is **+0.083** — above the validator's `min_headroom_gap`
floor, where math sits far below it. It is tractable where SWE-bench Pro was not: problems are
self-contained (no repo, no per-task multi-GB image) and grading is running a program against test
cases in one throwaway container.

HONEST CAVEAT, AND IT IS THE WHOLE PRODUCT QUESTION. On that same measurement the learned router
still LOST to the featureless baseline (-0.019 vs Zero), and on the clean easy+medium tiers the gap
falls to +0.042 with the router tying Zero exactly. So this benchmark makes the competition
*measurable*; it does not promise anyone can win it. That is the point of the degeneracy gate — the
subnet now reports the achievable gap every epoch instead of assuming it.

WHAT ADOPTING IT COST (done, pre-launch — `benchmarks.real_suite`, `SUITE_VERSION` = koth-suite-4).
`SUITE_VERSION` is baked into `runtime_measurement()` and into RTMR3, so ranking on this means a
rebuilt measured image and a fresh owner approval on-chain. Two operational consequences follow:

  * VALIDATORS AND THE OWNER NEED DOCKER. Grading is *executing* untrusted model output, not parsing
    it, so the grader runs each submission in a throwaway `--network none` container. Miners do not
    need it — they only generate answers.
  * THE PUBLIC POOL IS SMALLER: 112 problems (56 scored / 56 held-out) against math's 1000. The
    epoch nonce still selects an unpredictable 8, but a memorizer has fewer solutions to store. The
    load-bearing defence is unchanged — `grounding_check` requires every scored answer to derive
    from a logged pool response, so a stored program that never touches the pool is DQ'd — but note
    `scan_weights` is INERT here: it needs digit-shaped golds to build same-shape decoys, and a
    gold here is a list of test cases. See docs/DESIGN.md §6.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np

from .benchmarks import BenchTask

SANDBOX_IMAGE = "python:3.11-slim"
MAX_TESTS = 12               # public + private, capped — one problem ships 2305 cases
PER_CASE_TIMEOUT = 10        # seconds inside the container
ANSWER_KIND = "code"
TIERS = ("easy", "medium", "hard")


class GradingUnavailable(RuntimeError):
    """The grading HARNESS failed (no Docker, image pull failed, container died) — as opposed to the
    submitted program failing its tests. The two must never be conflated: scoring a miner 0 because
    the validator's Docker was down would penalise the miner for the validator's infrastructure."""


def load_lcb(seed: int = 0) -> dict[str, list[BenchTask]]:  # pragma: no cover — HF + dataset seam
    """Real LiveCodeBench problems, keyed by the dataset's own difficulty label.

    stdin-style (atcoder) problems only: they grade by running the program and comparing stdout,
    one uniform convention. LeetCode-style problems use a functional calling convention that would
    need a second harness, and mixing the two silently changes what "solved" means. That leaves 112
    problems — 26 easy, 26 medium, 60 hard.

    The difficulty label is why this dataset is here rather than a code benchmark without one: the
    mixed-difficulty stream that makes routing meaningful is REAL, not approximated. It is returned
    per tier rather than flattened because the per-epoch draw must STRATIFY over it — see
    `LCBBenchmark._draw`.
    """
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    with open(p) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["platform"] == "atcoder" and not r["starter_code"].strip()]
    rng = np.random.default_rng(seed)
    out: dict[str, list[BenchTask]] = {}
    for diff in TIERS:
        tier = [r for r in rows if r["difficulty"] == diff]
        order = rng.permutation(len(tier))          # seeded: miner and validator build it identically
        out[diff] = [BenchTask(f"lcb-{tier[int(i)]['question_id']}",
                               prompt_for(tier[int(i)]), tests_for(tier[int(i)])) for i in order]
    return out


def tests_for(row) -> list[dict]:  # pragma: no cover — dataset seam
    import base64
    import pickle
    import zlib
    cases = json.loads(row["public_test_cases"])
    try:
        # the pickle yields the JSON *string*, not the list (a 2305-char string, not 2305 cases)
        raw = pickle.loads(zlib.decompress(base64.b64decode(row["private_test_cases"])))
        cases += json.loads(raw) if isinstance(raw, str) else raw
    except Exception:                     # noqa: BLE001 — public-only is still a valid check
        pass
    return [c for c in cases if c.get("testtype") == "stdin"][:MAX_TESTS]


def prompt_for(row) -> str:  # pragma: no cover — dataset seam
    return (f"{row['question_content']}\n\n"
            "Write a complete Python 3 program that reads from standard input and writes the answer "
            "to standard output. Output ONLY the program source, no prose and no code fences.")


def extract_code(text: str) -> str:
    """Pull the program out of a model response — the same parser the pool-matrix experiments used,
    so a score here means what it meant there."""
    t = str(text or "")
    if "```" in t:
        for b in (b for b in t.split("```") if b.strip()):
            b = b[len("python"):] if b.lstrip().lower().startswith("python") else b
            if "input" in b or "print" in b:
                return b.strip() + "\n"
    return t.strip() + "\n"


_DRIVER = (
    "import json,subprocess,sys\n"
    "cs=json.load(open('/w/tests.json'))\n"
    "ok=0\n"
    "for c in cs:\n"
    "    try:\n"
    "        r=subprocess.run([sys.executable,'/w/sol.py'],input=c['input'],\n"
    f"                         capture_output=True,text=True,timeout={PER_CASE_TIMEOUT})\n"
    "        if r.stdout.split()==c['output'].split(): ok+=1\n"
    "    except Exception: pass\n"
    "print('KOTH_PASS',ok,len(cs))\n")


def run_tests(code: str, cases: list[dict], *, image: str = SANDBOX_IMAGE) -> float:
    """1.0 iff EVERY case matches, else 0.0. One container per submission, all cases inside it.

    `--network none` because the graded artifact is untrusted model output: it must not phone home,
    and a program that could reach the network could fetch the expected outputs. Deterministic across
    validators — same program, same cases, same verdict — which is what lets two decoupled validators
    agree without talking to each other.

    Raises `GradingUnavailable` if the harness itself could not run, so an infrastructure failure is
    never charged to the miner as a wrong answer.
    """
    if not cases:
        raise GradingUnavailable("no test cases for this task")
    if not code.strip():
        return 0.0                            # an empty submission is a genuine failure, not an error
    # THE SANDBOX DIR MUST EXIST AT THE SAME PATH IN BOTH NAMESPACES. `docker run -v` is resolved by
    # the DAEMON, so when the validator is itself containerised and talks to the host daemon through a
    # mounted socket, a path inside this container does not exist on the host: the bind silently mounts
    # nothing, /w/drv.py is absent, and grading fails. MEASURED on testnet 526 — the container had a
    # working docker client (`docker version` reported the host server) and still could not grade a
    # single code task. Point KOTH_GRADE_DIR at a directory bind-mounted at the IDENTICAL path on both
    # sides (docker-compose.yml does this) and the mount resolves for the daemon and for us alike.
    with tempfile.TemporaryDirectory(dir=os.environ.get("KOTH_GRADE_DIR") or None) as d:
        pathlib.Path(d, "sol.py").write_text(code)
        pathlib.Path(d, "tests.json").write_text(json.dumps(cases))
        pathlib.Path(d, "drv.py").write_text(_DRIVER)
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--memory", "1g",
                 "-v", f"{d}:/w:ro", image, "python", "/w/drv.py"],
                capture_output=True, text=True,
                timeout=PER_CASE_TIMEOUT * len(cases) + 60)
        except FileNotFoundError as e:
            raise GradingUnavailable("docker not available") from e
        except subprocess.TimeoutExpired:
            return 0.0                        # the SUBMISSION hung, not the harness
    for line in r.stdout.splitlines():
        if line.startswith("KOTH_PASS"):
            _, ok, tot = line.split()
            return 1.0 if int(ok) == int(tot) else 0.0
    raise GradingUnavailable(f"grader produced no verdict (rc={r.returncode}): {r.stderr[-200:]}")


def grade_code(answer: str, gold) -> float:
    return run_tests(extract_code(answer), list(gold or []))


class LCBBenchmark:
    """Code benchmark with a difficulty-STRATIFIED per-epoch draw.

    Why stratify rather than reuse the uniform `RealBenchmark._draw`: the usable pool is 26 easy /
    26 medium / 60 hard, so a uniform 8-task draw lands ~4-5 hard problems every epoch. The hard tier
    is exactly where this project's own measurement found 35/72 reference cells empty from token-cap
    truncation, and where the achievable gap is least trustworthy (on the clean easy+medium tiers it
    falls from +0.083 to +0.042). A hard-heavy slice would therefore feed the router scalar the
    noisiest part of the distribution and call it signal.

    Stratifying also fixes the *composition* of what miners are scored on epoch to epoch, which
    matters for the accumulator: pooling binomials across epochs assumes the draws are i.i.d., and a
    slice whose difficulty mix wanders violates that quietly.
    """

    answer_kind = ANSWER_KIND

    def __init__(self, name: str, weight: float, pool: dict, held: dict):
        self.name, self.weight = name, weight
        self._pool, self._held = pool, held

    @staticmethod
    def _shares(n: int) -> list[int]:
        """Split n across the tiers as evenly as possible, deterministically (3,3,2 for n=8)."""
        return [n // len(TIERS) + (1 if i < n % len(TIERS) else 0) for i in range(len(TIERS))]

    def _draw(self, by_tier: dict, n: int, seed: int) -> list[BenchTask]:
        rng = np.random.default_rng(seed)
        out: list[BenchTask] = []
        for tier, want in zip(TIERS, self._shares(n)):
            tasks = by_tier.get(tier, [])
            if not tasks or want <= 0:
                continue
            idx = rng.choice(len(tasks), size=min(want, len(tasks)), replace=False)
            out.extend(tasks[int(i)] for i in idx)
        return out

    def sample(self, n, seed): return self._draw(self._pool, n, seed)
    def probe(self, n, seed): return self._draw(self._held, n, seed)
    def grade(self, answer, gold): return grade_code(answer, gold)


def make_lcb(seed: int = 0, weight: float = 1.0) -> LCBBenchmark:
    """The benchmark object, with a per-tier disjoint pool/held-out split so the memorization probe
    draws problems the scored slice never contains — at every difficulty, not just on average."""
    by_tier = load_lcb(seed=seed)
    pool = {t: v[: len(v) // 2] for t, v in by_tier.items()}
    held = {t: v[len(v) // 2:] for t, v in by_tier.items()}
    return LCBBenchmark("code", weight, pool, held)
