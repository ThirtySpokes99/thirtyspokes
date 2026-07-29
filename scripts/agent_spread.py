#!/usr/bin/env python
"""Do independently-built AGENTS differ enough for a subnet to have a competition? Live, real cost.

THE SUBNET-VIABILITY QUESTION, in the right frame. ThirtySpokes pays miners emissions for out-scoring
each other. If every competent miner lands on the same number, nobody ever clears the reign's eps
margin, emissions freeze on whoever committed first, and the subnet is dead however good the
mechanism is. Spread between miners IS the product.

WHY THIS IS NOT ALREADY ANSWERED. Fourteen measurements closed one FAMILY of competition axes:
pick-one routing, cascade routing, verification, the fixed-harness router, specialist pools. Every
one asks "how should the owner's model pool be chosen among?", and all of them answered ~nothing.
But KOTH does not require that axis — its original design has miners shipping an arbitrary AGENT
(source + weights, any scaffold) scored on quality-per-cost. "Does composing models beat the best
single model" and "do two different agents score differently" are different quantities, and only the
first was measured. Two scaffolds around the SAME model differ enormously in the wild; that is what a
SWE-bench leaderboard is.

THE CONTROL THAT MAKES THIS HONEST. Any set of agents will show *some* spread; sampling noise
guarantees it. So one arm is a deliberate near-duplicate -- the same strategy under a paraphrased
prompt. That is a miner with no new idea, and the spread it produces is the NUISANCE FLOOR. Genuine
strategies have to beat that floor, not merely beat zero. Without this arm the experiment would
report a positive result on almost any pool, which is the exact failure that nearly landed in
measurement 14.

Pre-committed verdicts, so the reading is not fitted to the outcome:
    real spread <= nuisance floor   -> the differences are noise; the subnet ossifies. DEAD AXIS.
    spread < eps floor (0.002)      -> no miner can ever dethrone another. DEAD AXIS.
    eps floor < spread < eps0 (.02) -> MARGINAL: only early-reign challenges succeed.
    spread > eps0 (0.02)            -> COMPETITIVE: the mechanism you already have is launchable.

Scored EXACTLY as `KOTHValidator._reign_scalar` does: `Q_lcb - 0.02 * min(1, cost_per_task/budget)`,
where Q_lcb is the Wilson LOWER bound per benchmark. Quality first, cost as a small tiebreak that
never runs out. Getting this wrong is not cosmetic -- a first draft used a cost weight of 0.5, which
let cost swing the score by half a point and scored a strategy 0.37 away from a PARAPHRASE of itself.
That would have reported cost jitter as competition.

Run: set -a && . ./.env && set +a && python scripts/agent_spread.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend

EPS0, EPS_FLOOR = 0.02, 0.002        # reign.KingChain incumbency margin
CHEAP, STRONG = "qwen/qwen3.7-flash", "deepseek/deepseek-v4-pro"
RESULTS = os.environ.get("RESULTS", "data/agent_spread.jsonl")   # --suite code uses _lcb


# --- tasks -----------------------------------------------------------------------------------
def load_code_tasks(n: int, seed: int) -> list[dict]:
    """LiveCodeBench, graded by actually running the program in Docker.

    THE POINT OF THIS SUITE. On GSM8K/MMLU-Pro the top agents sit at 0.90-0.93 — close enough to the
    ceiling that spread compresses as miners improve, which is precisely how the routing axis died:
    not "no difference today" but "the difference evaporates once everyone is competent". Code is
    where nobody is near ceiling, so it tests whether spread is DURABLE rather than merely present.
    """
    from thirtyspokes.koth.lcb import load_lcb
    by_tier = load_lcb(seed=seed)
    rows = [t for tier in sorted(by_tier) for t in by_tier[tier]]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n]
    return [{"id": f"lcb-{rows[i].task_id}", "bench": "lcb", "kind": "code",
             "prompt": rows[i].prompt, "gold": rows[i].gold} for i in idx]


def load_tasks(n_per: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    out = []
    gs = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    for i, r in enumerate(gs.select(range(n_per))):
        out.append({"id": f"gsm8k-{i}", "bench": "gsm8k", "kind": "number",
                    "prompt": r["question"], "gold": r["answer"].split("####")[-1].strip()})
    mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=seed)
    for i, r in enumerate(mp.select(range(n_per))):
        opts = "\n".join(f"{chr(65+j)}) {c}" for j, c in enumerate(r["options"]))
        out.append({"id": f"mmlupro-{i}", "bench": "mmlu-pro", "kind": "choice",
                    "prompt": f"{r['question']}\n{opts}", "gold": str(r["answer"]).strip().upper()})
    return out


def grade(text: str, t: dict) -> float:
    if t["kind"] == "code":
        from thirtyspokes.koth.lcb import grade_code
        return grade_code(text, t["gold"])
    if t["kind"] == "number":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", str(text).replace("$", ""))
        if not nums:
            return 0.0
        try:
            return float(abs(float(nums[-1].replace(",", "")) - float(t["gold"])) < 1e-4)
        except ValueError:
            return 0.0
    up = str(text).upper()
    m = re.search(r"ANSWER[:\s]*([A-J])", up) or None
    hits = [m.group(1)] if m else re.findall(r"\b([A-J])\b", up)
    return float(bool(hits) and hits[-1] == t["gold"])


# --- the agents: distinct honest strategies a miner might actually ship ---------------------------
FINAL = "End with 'Answer: X' where X is the final answer only."
CODE_FINAL = ("Return ONLY a complete Python program that reads from stdin and writes to stdout, "
              "in a single ```python code block. No explanation.")


def final_for(t: dict) -> str:
    return CODE_FINAL if t["kind"] == "code" else FINAL


def _one(be, model, prompt, max_tokens=1200, temp=0.0):
    text, _i, _o, c = be.complete(model, [{"role": "user", "content": prompt}],
                                  {"max_tokens": max_tokens, "temperature": temp,
                                   "reasoning": {"effort": "low"}})
    return str(text), float(c)


def a_direct(be, t):
    return _one(be, CHEAP, f"{t['prompt']}\n\nAnswer immediately, no working. {final_for(t)}",
                200 if t["kind"] != "code" else 2000)


def a_cot(be, t):
    return _one(be, CHEAP, f"{t['prompt']}\n\nThink step by step, then answer. {final_for(t)}")


def a_cot_alt(be, t):
    """NUISANCE CONTROL — same strategy, paraphrased. A miner with no new idea."""
    return _one(be, CHEAP, f"{t['prompt']}\n\nWork through this carefully before responding. {final_for(t)}")


def extract_answer(text: str, kind: str) -> str | None:
    """The agent's own answer token — NO gold. Used for majority voting, which must be blind: an
    earlier version voted on `grade(o, t)`, i.e. grouped samples by whether they were CORRECT. That
    hands the agent the answer key and would have inflated this arm into a fake winner."""
    if kind == "code":
        from thirtyspokes.koth.lcb import extract_code
        c = extract_code(str(text))
        return " ".join(c.split()) or None      # whitespace-normalised program text
    if kind == "number":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", str(text).replace("$", ""))
        return nums[-1].replace(",", "") if nums else None
    up = str(text).upper()
    m = re.search(r"ANSWER[:\s]*([A-J])", up)
    if m:
        return m.group(1)
    hits = re.findall(r"\b([A-J])\b", up)
    return hits[-1] if hits else None


def a_self_consistency(be, t):
    """Three samples, majority vote on the ANSWER. Real technique, 3x the cost."""
    outs, cost = [], 0.0
    for _ in range(3):
        txt, c = _one(be, CHEAP, f"{t['prompt']}\n\nThink step by step, then answer. {final_for(t)}",
                      temp=0.8)
        outs.append(txt); cost += c
    votes = defaultdict(int)
    for o in outs:
        tok = extract_answer(o, t["kind"])
        if tok is not None:
            votes[tok] += 1
    if not votes:
        return outs[0], cost
    win = max(votes, key=votes.get)
    return next(o for o in outs if extract_answer(o, t["kind"]) == win), cost


def a_draft_critique(be, t):
    draft, c1 = _one(be, CHEAP, f"{t['prompt']}\n\nThink step by step, then answer. {final_for(t)}")
    rev, c2 = _one(be, CHEAP, f"{t['prompt']}\n\nA proposed solution:\n{draft[:1500]}\n\n"
                              f"Find any error and give the corrected final answer. {final_for(t)}")
    return rev, c1 + c2


def a_cascade(be, t):
    """Cheap first; escalate to the strong model only when the cheap answer is unparseable."""
    txt, c = _one(be, CHEAP, f"{t['prompt']}\n\nThink step by step, then answer. {final_for(t)}")
    if extract_answer(txt, t["kind"]) is not None:
        return txt, c
    txt2, c2 = _one(be, STRONG, f"{t['prompt']}\n\nThink step by step, then answer. {final_for(t)}")
    return txt2, c + c2


def a_strong(be, t):
    return _one(be, STRONG, f"{t['prompt']}\n\nThink step by step, then answer. {final_for(t)}")


AGENTS = {"direct": a_direct, "cot": a_cot, "cot-alt": a_cot_alt,
          "self-consistency": a_self_consistency, "draft-critique": a_draft_critique,
          "cascade": a_cascade, "strong-only": a_strong}
NUISANCE_PAIR = ("cot", "cot-alt")


def main() -> None:
    ap = argparse.ArgumentParser(description="inter-miner spread on the AGENT axis")
    ap.add_argument("--suite", choices=["easy", "code"], default="easy",
                    help="easy = GSM8K + MMLU-Pro (top agents ~0.93, near ceiling); "
                         "code = LiveCodeBench (far from ceiling — tests whether spread is DURABLE)")
    ap.add_argument("--n-per", type=int, default=30, help="tasks per benchmark")
    # The validator's ACTUAL scalar: quality first, cost as a small tiebreak that never runs out.
    # Using a big cost weight (an earlier draft used 0.5) lets cost swing the score by half a point,
    # so the measurement reports cost jitter as competition — a paraphrase of one strategy scored
    # 0.37 away from it. Quality has to dominate, exactly as it does on-chain.
    ap.add_argument("--cost-tiebreak", type=float, default=0.02,
                    help="KOTHValidator.cost_tiebreak")
    ap.add_argument("--budget-per-task", type=float, default=0.02,
                    help="KOTHValidator.budget_per_task")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    global RESULTS
    if args.suite == "code":
        RESULTS = os.environ.get("RESULTS", "data/agent_spread_lcb.jsonl")
    tasks = (load_code_tasks(args.n_per, args.seed) if args.suite == "code"
             else load_tasks(args.n_per, args.seed))
    print(f"{len(tasks)} tasks | {len(AGENTS)} agents | pool: {CHEAP} + {STRONG}", flush=True)

    done: dict[tuple[str, str], dict] = {}
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line); done[(r["agent"], r["id"])] = r
            except Exception:      # noqa: BLE001 — torn last line after a kill
                pass
        print(f"resuming: {len(done)} runs on disk", flush=True)

    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=180.0, max_retries=2,
                           price_fn=config.price_for)
    work = [(a, t) for t in tasks for a in AGENTS if (a, t["id"]) not in done]
    out = open(RESULTS, "a")
    n = [0]

    def run(job):
        a, t = job
        try:
            text, cost = AGENTS[a](be, t)
            rec = {"agent": a, "id": t["id"], "bench": t["bench"],
                   "correct": grade(text, t), "cost": cost}
        except Exception as e:    # noqa: BLE001 — a provider failure is data
            rec = {"agent": a, "id": t["id"], "bench": t["bench"], "correct": 0.0, "cost": 0.0,
                   "error": type(e).__name__}
        n[0] += 1
        if n[0] % 100 == 0:
            print(f"  {n[0]}/{len(work)} runs", flush=True)
        return rec

    if work:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec in ex.map(run, work):
                out.write(json.dumps(rec) + "\n"); out.flush()
                done[(rec["agent"], rec["id"])] = rec
    out.close()

    # --- score exactly as the validator would: quality with a cost gradient ----------------------
    from thirtyspokes.koth.evidence import wilson_lcb
    ids = [t["id"] for t in tasks]
    benches = sorted({t["bench"] for t in tasks})
    by_b = {b: [t["id"] for t in tasks if t["bench"] == b] for b in benches}
    acc = {a: np.array([done[(a, i)]["correct"] for i in ids if (a, i) in done]) for a in AGENTS}
    cost = {a: np.array([done[(a, i)]["cost"] for i in ids if (a, i) in done]) for a in AGENTS}

    def scalar(agent, subset=None):
        """Q_lcb - cost_tiebreak * min(1, cost_per_task/budget), i.e. `_reign_scalar` verbatim.
        Q_lcb is the Wilson LOWER bound per benchmark, so a lucky small sample cannot rank."""
        keep = set(subset) if subset is not None else None
        q = 0.0
        for b in benches:
            rows = [i for i in by_b[b] if (agent, i) in done and (keep is None or i in keep)]
            if not rows:
                continue
            c = sum(done[(agent, i)]["correct"] for i in rows)
            q += wilson_lcb(c, len(rows)) / len(benches)
        rows = [i for i in ids if (agent, i) in done and (keep is None or i in keep)]
        cpt = sum(done[(agent, i)]["cost"] for i in rows) / max(len(rows), 1)
        return q - args.cost_tiebreak * min(1.0, cpt / args.budget_per_task)

    score = {a: scalar(a) for a in AGENTS}
    errs = {a: sum(1 for i in ids if done.get((a, i), {}).get("error")) for a in AGENTS}

    print(f"\n{'agent':18s} {'acc':>7s} {'$/task':>9s} {'scalar':>9s}"
          f"   (cost_tiebreak={args.cost_tiebreak})")
    for a in sorted(AGENTS, key=lambda x: -score[x]):
        print(f"{a:18s} {acc[a].mean():7.3f} {cost[a].mean():9.5f} {score[a]:9.4f}"
              + (f"   [{errs[a]} errors]" if errs[a] else ""))

    real = [a for a in AGENTS if a != NUISANCE_PAIR[1]]          # exclude the deliberate twin
    spread = max(score[a] for a in real) - min(score[a] for a in real)
    nuisance = abs(score[NUISANCE_PAIR[0]] - score[NUISANCE_PAIR[1]])

    # paired bootstrap over TASKS: does the spread survive resampling the task set?
    rng = np.random.default_rng(0)
    boots = []
    arr = np.array(ids)
    for _ in range(args.boot):
        pick = arr[rng.integers(0, len(arr), len(arr))]
        s = {a: scalar(a, subset=pick) for a in real}
        boots.append(max(s.values()) - min(s.values()))
    lo, hi = float(np.percentile(boots, 5)), float(np.percentile(boots, 95))

    print(f"\nSPREAD across genuine strategies = {spread:.4f}   90% CI [{lo:.4f}, {hi:.4f}]")
    print(f"NUISANCE floor ({NUISANCE_PAIR[0]} vs its paraphrase) = {nuisance:.4f}")
    print(f"reign eps: floor {EPS_FLOOR}, eps0 {EPS0}")

    print("\nVERDICT")
    if spread <= nuisance:
        print("  DEAD AXIS — the spread between genuine strategies is no larger than between a\n"
              "  strategy and a paraphrase of itself. What looks like competition is sampling noise.")
    elif lo < EPS_FLOOR:
        print(f"  DEAD/UNSAFE — the spread's lower bound ({lo:.4f}) is under the eps floor "
              f"({EPS_FLOOR}).\n  A miner could not reliably dethrone another; emissions ossify on "
              "earliest commit.")
    elif lo < EPS0:
        print(f"  MARGINAL — spread clears the floor but its lower bound ({lo:.4f}) is under eps0 "
              f"({EPS0}).\n  Only early-reign challenges would succeed; expect ossification as the "
              "margin tightens.")
    else:
        print(f"  COMPETITIVE — spread exceeds eps0 even at its lower bound ({lo:.4f}) and beats the\n"
              f"  nuisance floor ({nuisance:.4f}). Miners CAN differentiate on this axis, so the\n"
              "  mechanism already built is launchable against it.")
    print("\nScope: one model pool, two benchmarks, agents written by one author in one sitting.\n"
          "Real miners iterate adversarially for months, so treat this as a LOWER bound on spread —\n"
          "but a lower bound that sits under eps is still decisive against launching.")


if __name__ == "__main__":
    main()
