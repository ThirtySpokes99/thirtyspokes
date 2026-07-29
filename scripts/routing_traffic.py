#!/usr/bin/env python
"""Traffic a router could actually be paid to route: a REAL difficulty distribution, exactly graded.

Every measurement in this project ran on exam sets, and `difficulty_predictability.py` showed what
that costs: inside one exam set, "can the cheap model handle this?" is predictable at AUC 0.500 --
chance. Exam sets are curated to be uniformly hard, which deletes the exact variance a router exists
to detect. Measuring routing there is like testing a thermostat in a room held at constant
temperature.

Production traffic looks nothing like that. It is dominated by asks a 7B model handles perfectly, with
a thin tail that genuinely needs a frontier model, and the difference is usually obvious from the text:

    "what is 47 + 128"                    trivial      (procedural)
    "how many letters in 'strawberry'"    trivial      (procedural)
    <a GSM8K word problem>                easy
    <an MMLU question>                    medium
    <a 10-option MMLU-Pro question>       medium-hard
    <an AIME competition problem>         hard

That spread is where routing earns its keep: send 60% of traffic to a model costing 1/20th and the
saving is real even if no accuracy is gained anywhere. An exam set has no such tail, so it can only
ever measure the residual -- per-item luck -- which is genuinely unpredictable.

THE TRIVIAL TIER IS PROCEDURAL, AND THAT IS DELIBERATE. Generated fresh from a seed, it is:
  * exactly gradeable, so validators stay cheap and deterministic (no judge, no Docker);
  * unlimited, which is the only real defence against the routing-table memoriser -- measurement 10
    showed a capped head fits a RANDOM table for 1,000 tasks, and the edge only falls below 15% past
    ~20,000 asks. A generated tier has no fixed bank to memorise at all;
  * uncontaminated, since the items did not exist before the seed was drawn.

So this is not only a better measuring instrument, it is a candidate traffic spec for the subnet.

Emits JSONL: {id, tier, kind, prompt, gold}. `--stats` prints the mix without writing.
"""
from __future__ import annotations

import argparse
import json
import random

# Production-like: most asks are easy. Deliberately NOT uniform — a uniform mix would be another
# exam set with extra steps, and would quietly reproduce the flat-difficulty problem this exists to
# fix. Tune here if the target traffic differs; this is the single most consequential number in the
# file, because it sets how much of the routable value is cost saving versus quality gain.
MIX = {"trivial": 0.35, "easy": 0.20, "medium": 0.15, "medium-hard": 0.10,
       "hard": 0.05, "hard-proc": 0.15}


# --- the procedural trivial tier ------------------------------------------------------------------
def _arith(rng) -> tuple[str, str]:
    a, b = rng.randint(12, 999), rng.randint(12, 999)
    op = rng.choice(["+", "-", "*"])
    val = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"What is {a} {op} {b}? Reply with the number only.", str(val)


def _count_letters(rng) -> tuple[str, str]:
    word = rng.choice(["strawberry", "bookkeeper", "mississippi", "parallel", "committee",
                       "possession", "millennium", "necessary", "occurrence", "embarrass"])
    ch = rng.choice(sorted(set(word)))
    return (f"How many times does the letter '{ch}' appear in the word '{word}'? "
            f"Reply with the number only."), str(word.count(ch))


def _unit(rng) -> tuple[str, str]:
    n = rng.randint(2, 99)
    kind = rng.choice(["m_cm", "kg_g", "h_min", "km_m"])
    q, val = {
        "m_cm": (f"How many centimetres are in {n} metres?", n * 100),
        "kg_g": (f"How many grams are in {n} kilograms?", n * 1000),
        "h_min": (f"How many minutes are in {n} hours?", n * 60),
        "km_m": (f"How many metres are in {n} kilometres?", n * 1000),
    }[kind]
    return f"{q} Reply with the number only.", str(val)


def _percent(rng) -> tuple[str, str]:
    p = rng.choice([10, 20, 25, 50])
    base = rng.randrange(40, 2000, 20)
    return f"What is {p}% of {base}? Reply with the number only.", str(base * p // 100)


def _sequence(rng) -> tuple[str, str]:
    start, step = rng.randint(1, 20), rng.randint(2, 12)
    seq = [start + step * i for i in range(4)]
    return (f"What comes next in this sequence: {', '.join(map(str, seq))}? "
            f"Reply with the number only."), str(start + step * 4)


TRIVIAL_GENS = [_arith, _count_letters, _unit, _percent, _sequence]


def trivial_tasks(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        prompt, gold = TRIVIAL_GENS[i % len(TRIVIAL_GENS)](rng)
        out.append({"id": f"trivial-{i}", "tier": "trivial", "kind": "number",
                    "prompt": prompt, "gold": gold})
    return out


# --- the graded tiers, from real datasets ---------------------------------------------------------
def easy_tasks(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    return [{"id": f"easy-{i}", "tier": "easy", "kind": "number",
             "prompt": r["question"] + "\n\nReply with the final number only.",
             "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(ds.select(range(min(n, len(ds)))))]


def _mcq(r, i, tier, letters):
    opts = "\n".join(f"{letters[j]}) {c}" for j, c in enumerate(r["options" if "options" in r else "choices"]))
    return {"id": f"{tier}-{i}", "tier": tier, "kind": "choice",
            "prompt": f"{r['question']}\n{opts}\n\nReply with the single letter only.",
            "gold": (str(r["answer"]).strip().upper() if isinstance(r["answer"], str)
                     else letters[int(r["answer"])])}


def medium_tasks(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
    L = "ABCD"
    return [_mcq(r, i, "medium", L) for i, r in enumerate(ds.select(range(min(n, len(ds)))))]


def medium_hard_tasks(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=seed)
    L = "ABCDEFGHIJ"
    return [_mcq(r, i, "medium-hard", L) for i, r in enumerate(ds.select(range(min(n, len(ds)))))]


def hard_tasks(n: int, seed: int) -> list[dict]:
    """AIME, not MMLU-Pro. MEASURED: the cheapest model in the pool solves MMLU-Pro outright at
    1/15th the price of the frontier one, so a tier built from it has no capability gradient and no
    routing decision — the dominator-pool problem, rebuilt by hand. A tier is only "hard" if the
    cheap model actually FAILS it; otherwise the traffic guarantees the negative result it is
    supposed to test. AIME answers are integers 0-999, so grading stays exact."""
    from datasets import load_dataset
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train").shuffle(seed=seed)
    return [{"id": f"hard-{i}", "tier": "hard", "kind": "number",
             "prompt": r["Question"] + "\n\nReply with the final integer answer only.",
             "gold": str(r["Answer"]).strip()}
            for i, r in enumerate(ds.select(range(min(n, len(ds)))))]


# --- procedurally HARD: uncontaminated, and difficulty is a dial ----------------------------------
def _long_arith(rng) -> tuple[str, str]:
    """Multi-step arithmetic deep enough that a small model loses the thread. Exact by construction,
    and impossible to have memorised because the numbers are drawn at run time."""
    v = rng.randint(200, 999)
    steps, expr = [], f"{v}"
    for _ in range(rng.randint(6, 9)):
        op = rng.choice(["+", "-", "*"])
        k = rng.randint(11, 99) if op != "*" else rng.randint(3, 17)
        v = {"+": v + k, "-": v - k, "*": v * k}[op]
        expr += f" {op} {k}"
    return (f"Evaluate step by step, respecting standard operator precedence:\n{expr}\n\n"
            f"Reply with the final integer only."), str(eval(expr))


def _modular(rng) -> tuple[str, str]:
    b, e, m = rng.randint(7, 97), rng.randint(50, 400), rng.randint(101, 997)
    return (f"Compute {b}^{e} mod {m}. Reply with the integer only."), str(pow(b, e, m))


def _combinatorics(rng) -> tuple[str, str]:
    from math import comb
    n, k = rng.randint(18, 40), rng.randint(5, 12)
    return (f"How many ways are there to choose {k} items from {n} distinct items? "
            f"Reply with the integer only."), str(comb(n, k))


def _digit_puzzle(rng) -> tuple[str, str]:
    n = rng.randint(10**7, 10**9)
    return (f"What is the sum of the decimal digits of {n} * {rng.randint(7, 29)}? "
            f"Reply with the integer only."), None      # filled below


HARD_GENS = [_long_arith, _modular, _combinatorics]


def hard_proc_tasks(n: int, seed: int) -> list[dict]:
    """The hard tier, generated. AIME turned out to be contaminated — a cheap flash model scored
    0.938 on it, which is recall, not capability, so it could not separate the pool. These are drawn
    at run time and cannot have been memorised, while staying exactly gradeable."""
    rng = random.Random(seed + 7717)
    out = []
    for i in range(n):
        prompt, gold = HARD_GENS[i % len(HARD_GENS)](rng)
        out.append({"id": f"hardproc-{i}", "tier": "hard-proc", "kind": "number",
                    "prompt": prompt, "gold": gold})
    return out


BUILDERS = {"trivial": trivial_tasks, "easy": easy_tasks, "medium": medium_tasks,
            "medium-hard": medium_hard_tasks, "hard": hard_tasks,
            "hard-proc": hard_proc_tasks}


def build(n: int, seed: int = 0) -> list[dict]:
    tasks: list[dict] = []
    for tier, share in MIX.items():
        tasks += BUILDERS[tier](max(1, round(n * share)), seed)
    random.Random(seed).shuffle(tasks)          # interleave, as a real stream would arrive
    return tasks


def grade(text: str, t: dict) -> float:
    import re
    if t["kind"] == "number":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", str(text).replace("$", "").replace(",", ""))
        if not nums:
            return 0.0
        try:
            return float(abs(float(nums[-1]) - float(t["gold"])) < 1e-4)
        except ValueError:
            return 0.0
    up = str(text).upper()
    m = re.search(r"ANSWER[:\s]*([A-J])", up)
    hits = [m.group(1)] if m else re.findall(r"\b([A-J])\b", up)
    return float(bool(hits) and hits[-1] == t["gold"])


def main() -> None:
    ap = argparse.ArgumentParser(description="build routing-shaped traffic")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/routing_traffic.jsonl")
    ap.add_argument("--stats", action="store_true", help="print the mix without writing")
    args = ap.parse_args()

    tasks = build(args.n, args.seed)
    from collections import Counter
    c = Counter(t["tier"] for t in tasks)
    print(f"{len(tasks)} tasks: " + "  ".join(f"{k}={c[k]} ({c[k]/len(tasks):.0%})" for k in MIX))
    for tier in MIX:
        ex = next(t for t in tasks if t["tier"] == tier)
        print(f"\n[{tier}] {ex['prompt'][:150].replace(chr(10), ' / ')}\n   gold={ex['gold']}")
    if args.stats:
        return
    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
