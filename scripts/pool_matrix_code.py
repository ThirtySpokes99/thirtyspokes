"""Counterfactual pool matrix over LiveCodeBench (the mid-range code benchmark).

Math and SWE-bench Pro both produce DEGENERATE traffic, from opposite ends:
  * math (GSM8K + MATH-500 L5): every model 79-97%, saturated at the CEILING, oracle gap +0.019,
    learned router beat the Zero baseline by +0.002.
  * SWE-bench Pro one-shot: every model 0/20, saturated at the FLOOR. Nothing to compare.

Routing needs traffic where models land in the MIDDLE and disagree. LiveCodeBench is the cheapest
benchmark with that property, and it ships the one thing the others lack: a per-problem DIFFICULTY
label (easy/medium/hard), so the mixed-difficulty stream is real rather than approximated.

Why it is tractable where SWE-bench Pro was not: problems are self-contained (no repo, no per-task
multi-GB image), and grading is running a program against test cases in one throwaway container.

Claude Opus 5 is dropped from the pool -- on the SWE matrix it was $5.37 of $8.42 (64%) and never solved
anything. The pool keeps 6 labs and ~60x price spread without it.

  set -a && . ./.env && set +a
  python scripts/pool_matrix_code.py --generate
  python scripts/pool_matrix_code.py --grade
  python scripts/routing_headroom.py --matrix data/pool_matrix_code.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import pickle
import subprocess
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend

MODELS = [
    "openai/gpt-5-nano",              # $0.05 / $0.40
    "deepseek/deepseek-v4-flash",     # $0.09 / $0.19
    "google/gemini-3.1-flash-lite",   # $0.25 / $1.50
    "minimax/minimax-m2.7",           # $0.25 / $1.00
    "z-ai/glm-5.2",                   # $0.71 / $2.23
    "moonshotai/kimi-k3",             # $3.00 / $15.00
]
N_PER_DIFF = 12                # 12 easy + 12 medium + 12 hard = 36 problems
MAX_TOKENS = 16_384          # hard LCB problems hit the 8k cap even with reasoning bounded
REASONING = {"effort": "low"}  # The n=20 run found: unbounded thinking burned the whole budget and emitted nothing
MAX_TESTS = 12                 # public + private, capped (one problem ships 2305 cases)
SANDBOX = "python:3.11-slim"
GEN = pathlib.Path(__file__).resolve().parent.parent / "data" / "pool_matrix_code_generations.jsonl"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "pool_matrix_code.jsonl"


def load_problems():
    """stdin-style problems only (atcoder): uniform run-and-compare grading, no LeetCode functional
    calling convention. Stratified by the dataset's own difficulty label."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p)]
    rows = [r for r in rows if r["platform"] == "atcoder" and not r["starter_code"].strip()]
    out, rng = [], np.random.default_rng(11)
    for diff in ("easy", "medium", "hard"):
        pool = [r for r in rows if r["difficulty"] == diff]
        for i in rng.permutation(len(pool))[:N_PER_DIFF]:
            out.append(pool[int(i)])
    return out


def tests_for(row) -> list[dict]:
    cases = json.loads(row["public_test_cases"])
    try:
        # pickle yields the JSON *string*, not the list (a 2305-char string, not 2305 cases)
        raw = pickle.loads(zlib.decompress(base64.b64decode(row["private_test_cases"])))
        cases += json.loads(raw) if isinstance(raw, str) else raw
    except Exception:                                    # noqa: BLE001 — public-only is still a valid check
        pass
    return [c for c in cases if c.get("testtype") == "stdin"][:MAX_TESTS]


def prompt_for(row) -> str:
    return (f"{row['question_content']}\n\n"
            "Write a complete Python 3 program that reads from standard input and writes the answer "
            "to standard output. Output ONLY the program source, no prose and no code fences.")


def extract_code(text: str) -> str:
    t = str(text or "")
    if "```" in t:
        blocks = [b for b in t.split("```") if b.strip()]
        for b in blocks:
            b = b[len("python"):] if b.lstrip().lower().startswith("python") else b
            if "input" in b or "print" in b:
                return b.strip() + "\n"
    return t.strip() + "\n"


def run_tests(code: str, cases: list[dict], timeout: int = 20) -> float | None:
    """1.0 iff every case matches. One container per (solution), all cases inside it."""
    if not code.strip() or not cases:
        return 0.0 if cases else None
    driver = (
        "import json,subprocess,sys\n"
        "cs=json.load(open('/w/tests.json'))\n"
        "ok=0\n"
        "for c in cs:\n"
        "    try:\n"
        "        r=subprocess.run([sys.executable,'/w/sol.py'],input=c['input'],\n"
        "                         capture_output=True,text=True,timeout=10)\n"
        "        if r.stdout.split()==c['output'].split(): ok+=1\n"
        "    except Exception: pass\n"
        "print('KOTH_PASS',ok,len(cs))\n")
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "sol.py").write_text(code)
        pathlib.Path(d, "tests.json").write_text(json.dumps(cases))
        pathlib.Path(d, "drv.py").write_text(driver)
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--memory", "1g",
                 "-v", f"{d}:/w:ro", SANDBOX, "python", "/w/drv.py"],
                capture_output=True, text=True, timeout=timeout * len(cases) + 60)
        except subprocess.TimeoutExpired:
            return 0.0
    for line in r.stdout.splitlines():
        if line.startswith("KOTH_PASS"):
            _, ok, tot = line.split()
            return 1.0 if int(ok) == int(tot) else 0.0
    return None


def generate():
    probs = load_problems()
    print(f"{len(probs)} problems ({N_PER_DIFF} each of easy/medium/hard)", flush=True)
    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=600.0, max_retries=3,
                           price_fn=config.price_for)
    done = set()
    if GEN.exists():
        done = {(json.loads(l)["task_id"], json.loads(l)["model"])
                for l in GEN.read_text().splitlines() if l.strip()}
    jobs = [(r, m) for r in probs for m in MODELS if (r["question_id"], m) not in done]
    print(f"{len(jobs)} cells to generate", flush=True)
    fh = GEN.open("a"); n = {"i": 0}

    def run(job):
        r, model = job
        t0 = time.time()
        try:
            text, tin, tout, cost = be.complete(
                model, [{"role": "user", "content": prompt_for(r)}],
                {"max_tokens": MAX_TOKENS, "reasoning": REASONING})
            return dict(task_id=r["question_id"], model=model, tier=r["difficulty"], kind="lcb",
                        code=extract_code(text), tokens_in=tin, tokens_out=tout,
                        cost_usd=float(cost), latency_s=round(time.time() - t0, 1))
        except Exception as e:                            # noqa: BLE001
            return dict(task_id=r["question_id"], model=model, tier=r["difficulty"], kind="lcb",
                        code="", tokens_in=0, tokens_out=0, cost_usd=0.0, latency_s=0.0,
                        error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(run, j) for j in jobs]):
            c = f.result(); fh.write(json.dumps(c) + "\n"); fh.flush()
            n["i"] += 1
            if n["i"] % 20 == 0:
                print(f"  {n['i']}/{len(jobs)}", flush=True)
    fh.close()
    rows = [json.loads(l) for l in GEN.read_text().splitlines() if l.strip()]
    print(f"\ngenerated {len(rows)} cells, ${sum(r['cost_usd'] for r in rows):.2f}, "
          f"{sum(1 for r in rows if not r['code'].strip())} empty", flush=True)


def grade():
    probs = {r["question_id"]: r for r in load_problems()}
    cases = {qid: tests_for(r) for qid, r in probs.items()}
    gens = [json.loads(l) for l in GEN.read_text().splitlines() if l.strip()]
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["task_id"], json.loads(l)["model"])
                for l in OUT.read_text().splitlines() if l.strip()}
    todo = [g for g in gens if (g["task_id"], g["model"]) not in done]
    print(f"{len(todo)} cells to grade", flush=True)
    subprocess.run(["docker", "pull", SANDBOX], capture_output=True, timeout=600)
    fh = OUT.open("a"); n = {"i": 0}

    def run(g):
        s = run_tests(g["code"], cases.get(g["task_id"], []))
        return dict(task_id=g["task_id"], model=g["model"], tier=g["tier"], kind="lcb",
                    response=g["code"][:300], tokens_in=g.get("tokens_in", 0),
                    tokens_out=g.get("tokens_out", 0), cost_usd=g["cost_usd"],
                    latency_s=g.get("latency_s", 0.0), score=s)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(run, g) for g in todo]):
            c = f.result(); fh.write(json.dumps(c) + "\n"); fh.flush()
            n["i"] += 1
            if n["i"] % 20 == 0:
                print(f"  {n['i']}/{len(todo)}", flush=True)
    fh.close()
    print(f"\ngraded -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args()
    generate() if a.generate else grade() if a.grade else print(__doc__)
