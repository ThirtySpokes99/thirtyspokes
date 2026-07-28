#!/usr/bin/env python
"""The last open question: does a pool of GENUINE language specialists give a router something to win?

Measurement 13 killed the 17.5% specialist result (n=118 -> n=785, CI straddles zero) but could not
test the hypothesis properly, because all 11 RouterBench models are Western-trained and gpt-4 won
Chinese AND English traffic alike. A dominator pool cannot demonstrate specialist routing; it can only
re-demonstrate the dominator-pool problem. That was recorded as UNTESTABLE, not closed.

This closes it. The pinned pool contains six models from Chinese labs (Qwen, DeepSeek x2, MiniMax,
GLM, Kimi) against four Western ones (OpenAI x2, Google, xAI) -- labs whose pretraining mixes differ
in the one way that plausibly produces competence differing in KIND rather than TIER. Traffic is
CMMLU (Chinese, 11,582 questions over 67 subjects) against MMLU (English), matched MCQ format so the
only difference is language and subject matter, not answer parsing.

Three strata, because "language specialist" could mean two different things and they should not be
conflated:
    cn-specific   subjects about Chinese language/history/culture -- maximum specialist signal
    cn-general    ordinary subjects asked IN CHINESE -- isolates language from subject matter
    en-control    the same kind of ordinary subjects in English -- the baseline

THE DECISIVE STATISTIC IS NOT ACCURACY, IT IS WHO WINS. If one model tops every stratum, the pool is
a capability ladder again and routing has nothing to decide, exactly as measurements 1-13 found. If
different models top different strata, specialist diversity is real and the remaining question is
whether a router can capture the band it opens -- which is measured here too, as the ACHIEVABLE
language-router: pick the best model per stratum on a train split, route test asks by their stratum.
That is the strongest router that could possibly exist for this traffic, since the stratum is given
rather than predicted. If even THAT does not beat best-single, no learned router can.

Pre-committed verdicts, so the reading is not fitted to the outcome:
    same winner everywhere            -> dominator pool again; the specialist thesis is closed
    different winners, router <= best -> diversity is real but unexploitable; closed
    different winners, router > best  -> the first positive result in this project; measure at power

Cost: ~3,600 calls of a few hundred tokens. Resumable -- every call is appended to RESULTS as it
lands, and a rerun skips what is already there.

Run: set -a && . ./.env && set +a && python scripts/language_specialist_experiment.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend

# The owner-pinned pool (scripts/pool_candidate.py): latest model per family, Anthropic excluded by
# owner decision. Six Chinese labs vs four Western ones -- the split this experiment turns on.
POOL = [
    "qwen/qwen3.7-flash",             # Qwen      CN
    "deepseek/deepseek-v4-flash",     # DeepSeek  CN
    "minimax/minimax-m3",             # MiniMax   CN
    "deepseek/deepseek-v4-pro",       # DeepSeek  CN
    "openai/gpt-5.6-luna",            # OpenAI    US
    "z-ai/glm-5.2",                   # GLM       CN
    "openai/gpt-5.6-terra",           # OpenAI    US
    "google/gemini-3.6-flash",        # Google    US
    "x-ai/grok-4.5",                  # xAI       US
    "moonshotai/kimi-k3",             # Moonshot  CN
]
LAB = {"qwen/qwen3.7-flash": "CN", "deepseek/deepseek-v4-flash": "CN", "minimax/minimax-m3": "CN",
       "deepseek/deepseek-v4-pro": "CN", "z-ai/glm-5.2": "CN", "moonshotai/kimi-k3": "CN",
       "openai/gpt-5.6-luna": "US", "openai/gpt-5.6-terra": "US",
       "google/gemini-3.6-flash": "US", "x-ai/grok-4.5": "US"}

# CMMLU subjects that are ABOUT Chinese language/history/culture -- where a Chinese-trained model
# should hold an edge that is not merely "reads Chinese".
CN_SPECIFIC = ["ancient_chinese", "chinese_literature", "chinese_history", "chinese_foreign_policy",
               "chinese_civil_service_exam", "chinese_driving_rule", "chinese_food_culture",
               "chinese_teacher_qualification", "modern_chinese", "traditional_chinese_medicine"]
# ordinary subjects, present in BOTH CMMLU and MMLU, so language is the only thing that changes
SHARED = ["high_school_physics", "high_school_mathematics", "college_medicine", "virology",
          "professional_law", "marketing", "nutrition", "world_religions", "astronomy", "anatomy"]

STRATA = ["cn-specific", "cn-general", "en-control"]
RESULTS = os.environ.get("RESULTS", "data/language_specialist_v2.jsonl")
CMMLU_ZIP = "data/cmmlu/cmmlu_v1_0_1.zip"


def _prompt(q: str, opts: list[str], chinese: bool) -> str:
    body = "\n".join(f"{chr(65 + j)}) {c}" for j, c in enumerate(opts))
    tail = ("\n\n只回答一个字母（A/B/C/D），不要解释。" if chinese
            else "\n\nAnswer with the single letter (A/B/C/D) only.")
    return f"{q}\n{body}{tail}"


def load_cmmlu(subjects: list[str], n: int, stratum: str, seed: int) -> list[dict]:
    z = zipfile.ZipFile(CMMLU_ZIP)
    have = {p.split("/")[-1][:-4] for p in z.namelist() if p.startswith("test/")}
    rows = []
    for s in subjects:
        if s not in have:
            continue
        for i, r in enumerate(csv.DictReader(io.TextIOWrapper(z.open(f"test/{s}.csv"), encoding="utf-8"))):
            rows.append({"id": f"{s}-{i}", "stratum": stratum, "subject": s,
                         "prompt": _prompt(r["Question"], [r["A"], r["B"], r["C"], r["D"]], True),
                         "gold": r["Answer"].strip().upper()})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_mmlu(subjects: list[str], n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    rows = []
    for s in subjects:
        try:
            ds = load_dataset("cais/mmlu", s, split="test")
        except Exception:      # noqa: BLE001 — a missing subject is data, not a crash
            continue
        for i, r in enumerate(ds):
            rows.append({"id": f"en-{s}-{i}", "stratum": "en-control", "subject": s,
                         "prompt": _prompt(r["question"], list(r["choices"]), False),
                         "gold": chr(65 + int(r["answer"]))})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def extract(resp: str) -> str | None:
    up = str(resp).strip().upper()
    m = re.search(r"ANSWER[:\s]*([A-D])", up)
    if m:
        return m.group(1)
    hits = re.findall(r"\b([A-D])\b", up)
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description="language-specialist routing ceiling, live")
    ap.add_argument("--n-per", type=int, default=100, help="questions per stratum")
    ap.add_argument("--models", default=",".join(POOL))
    # 1500, not 300: at 300 the reasoning models in this pool spend the whole budget thinking and
    # return an EMPTY string, which scores as wrong. That depressed every Chinese lab's ENGLISH score
    # (deepseek-v4-pro 14/20 empty) and manufactured a fake "language specialisation" result. The
    # parse-rate guard below exists so this can never be mistaken for competence again.
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    tasks = (load_cmmlu(CN_SPECIFIC, args.n_per, "cn-specific", args.seed)
             + load_cmmlu(SHARED, args.n_per, "cn-general", args.seed)
             + load_mmlu(SHARED, args.n_per, args.seed))
    by_stratum = defaultdict(list)
    for t in tasks:
        by_stratum[t["stratum"]].append(t["id"])
    print(f"{len(tasks)} tasks " + " ".join(f"{s}={len(by_stratum[s])}" for s in STRATA)
          + f" | {len(models)} models -> {len(tasks) * len(models)} calls", flush=True)

    # resume: every landed call is on disk, so a rerun costs only what is missing
    done: dict[tuple[str, str], dict] = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[(r["model"], r["id"])] = r
                except Exception:    # noqa: BLE001 — a torn last line is expected after a kill
                    pass
        print(f"resuming: {len(done)} calls already on disk", flush=True)

    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=180.0, max_retries=2,
                           price_fn=config.price_for)
    work = [(m, t) for t in tasks for m in models if (m, t["id"]) not in done]
    out = open(RESULTS, "a")
    n_done = [0]

    def run(job):
        m, t = job
        try:
            text, _i, _o, c = be.complete(m, [{"role": "user", "content": t["prompt"]}],
                                          {"max_tokens": args.max_tokens,
                                           "reasoning": {"effort": "low"}})
            got = extract(text)
            rec = {"model": m, "id": t["id"], "stratum": t["stratum"],
                   "correct": float(got == t["gold"]), "cost": float(c),
                   # recorded because "wrong" and "never answered" are different failures, and
                   # conflating them is what produced the retracted v1 result
                   "parsed": got is not None, "empty": not str(text).strip()}
        except Exception as e:   # noqa: BLE001 — a model erroring is data, not a crash
            rec = {"model": m, "id": t["id"], "stratum": t["stratum"], "correct": 0.0,
                   "cost": 0.0, "parsed": False, "empty": True, "error": type(e).__name__}
        n_done[0] += 1
        if n_done[0] % 200 == 0:
            print(f"  {n_done[0]}/{len(work)} calls", flush=True)
        return rec

    if work:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec in ex.map(run, work):
                out.write(json.dumps(rec) + "\n"); out.flush()
                done[(rec["model"], rec["id"])] = rec
    out.close()

    # ---------------- analysis ----------------
    err = defaultdict(int)
    for (m, _t), r in done.items():
        err[m] += 1 if r.get("error") else 0
    models = [m for m in models if err[m] < len(tasks) * 0.5]     # drop models that mostly failed

    def acc(m, ids):
        v = [done[(m, i)]["correct"] for i in ids if (m, i) in done]
        return sum(v) / max(len(v), 1)

    def cost(m, ids):
        return sum(done[(m, i)]["cost"] for i in ids if (m, i) in done)

    def parse_rate(m, ids):
        v = [done[(m, i)].get("parsed", True) for i in ids if (m, i) in done]
        return sum(v) / max(len(v), 1)

    print("\n=== accuracy by stratum (all asks) ===")
    print("model".ljust(30) + "lab" + "".join(s.rjust(13) for s in STRATA) + "     $total")
    for m in models:
        row = m.ljust(30) + LAB.get(m, "??").ljust(4)
        for s in STRATA:
            row += f"{acc(m, by_stratum[s]):13.3f}"
        row += f"{sum(cost(m, by_stratum[s]) for s in STRATA):11.3f}"
        print(row + (f"   [{err[m]} errors]" if err[m] else ""))

    # CONFOUND GUARD. A model that cannot emit a parseable answer in one language scores 0 there and
    # looks like a specialist in the other. That is a harness failure wearing the costume of the
    # result this experiment is looking for, so it is checked BEFORE the finding is read.
    print("\n=== 0. PARSE-RATE GUARD — did every model actually answer in every language? ===")
    worst = 1.0
    for m in models:
        rates = {s: parse_rate(m, by_stratum[s]) for s in STRATA}
        spread = max(rates.values()) - min(rates.values())
        worst = min(worst, min(rates.values()))
        flag = "  <-- SKEWED, results unusable" if spread > 0.10 else ""
        print(f"  {m:30s} " + " ".join(f"{s}={rates[s]:.2f}" for s in STRATA) + flag)
    if worst < 0.90:
        print(f"\n  WARNING: lowest parse rate {worst:.2f}. Raise --max-tokens and re-run; an\n"
              "  unanswered ask is not a wrong answer, and scoring it as one fabricates specialisation.")

    print("\n=== 1. SPECIALIST DIVERSITY — who wins each stratum? ===")
    winners = {}
    for s in STRATA:
        w = max(models, key=lambda m: (acc(m, by_stratum[s]), -cost(m, by_stratum[s])))
        winners[s] = w
        print(f"  {s:12s} {w:30s} ({LAB.get(w)})  acc={acc(w, by_stratum[s]):.3f}")
    distinct = len(set(winners.values()))
    print(f"  -> {distinct} distinct winner(s) across {len(STRATA)} strata")
    # SIGNIFICANCE GUARD. "a different model tops this stratum" is not evidence of specialisation if
    # the margin is inside sampling noise -- at n=100 and acc~0.9 one standard error is ~0.03, so a
    # 3pp lead is a coin flip. Requiring 2 SE is what separates a real specialist from the model that
    # happened to win. Without this the experiment would report diversity on nearly any pool.
    import math
    cn_w, en_w = winners["cn-specific"], winners["en-control"]
    a_cn, a_en = acc(cn_w, by_stratum["cn-specific"]), acc(en_w, by_stratum["cn-specific"])
    n_cn = len(by_stratum["cn-specific"])
    se = math.sqrt(max(a_cn * (1 - a_cn), 1e-9) / max(n_cn, 1)) * math.sqrt(2)
    margin = a_cn - a_en
    print(f"  cn-specific: {cn_w.split('/')[-1]} {a_cn:.3f} vs en-winner {en_w.split('/')[-1]} "
          f"{a_en:.3f}  margin={margin:+.3f}  2SE={2*se:.3f}")
    cn_win = (cn_w != en_w) and margin > 2 * se
    print("  -> " + ("DIFFERENT models win Chinese vs English by more than noise: genuine "
                     "language specialisation" if cn_win else
                     "NO significant language specialisation: the Chinese and English winners are "
                     "the same model, or the margin is inside sampling noise"))

    print("\n=== 2. CAN IT BE EXPLOITED? achievable stratum-router vs best-single ===")
    rng = random.Random(args.seed)
    train, test = set(), set()
    for s in STRATA:                                  # 50/50 split within each stratum
        ids = by_stratum[s][:]; rng.shuffle(ids); k = len(ids) // 2
        train.update(ids[:k]); test.update(ids[k:])
    test_by = {s: [i for i in by_stratum[s] if i in test] for s in STRATA}
    test_ids = [i for s in STRATA for i in test_by[s]]

    best_single = max(models, key=lambda m: acc(m, [i for i in test_ids if i in train] or list(train)))
    bs_acc, bs_cost = acc(best_single, test_ids), cost(best_single, test_ids)
    pick = {s: max(models, key=lambda m: (acc(m, [i for i in by_stratum[s] if i in train]),
                                          -cost(m, [i for i in by_stratum[s] if i in train])))
            for s in STRATA}
    r_correct = sum(done[(pick[s], i)]["correct"] for s in STRATA for i in test_by[s]
                    if (pick[s], i) in done)
    r_cost = sum(done[(pick[s], i)]["cost"] for s in STRATA for i in test_by[s]
                 if (pick[s], i) in done)
    r_acc = r_correct / max(len(test_ids), 1)
    hits = ocost = 0.0
    for i in test_ids:
        win = [m for m in models if done.get((m, i), {}).get("correct", 0.0) >= 1.0]
        if win:
            hits += 1
            ocost += min(done[(m, i)]["cost"] for m in win)
    o_acc = hits / max(len(test_ids), 1)

    print(f"  best-single   {best_single:30s} acc={bs_acc:.4f}  ${bs_cost:.3f}")
    print(f"  stratum-router{'':30s} acc={r_acc:.4f}  ${r_cost:.3f}   "
          f"(picks: {', '.join(f'{s}->{pick[s].split('/')[-1]}' for s in STRATA)})")
    print(f"  per-ask oracle{'':30s} acc={o_acc:.4f}  ${ocost:.3f}")
    print(f"\n  router - best-single = {r_acc - bs_acc:+.4f} acc, {r_cost - bs_cost:+.4f} $")

    print("\nVERDICT")
    if not cn_win:
        print("  DOMINATOR POOL AGAIN. One model tops both languages, so there is nothing to route\n"
              "  even with six Chinese labs in the pool. The specialist thesis is CLOSED: it was the\n"
              "  last mechanism by which routing could have had a moat.")
    elif r_acc - bs_acc <= 0:
        print("  Specialist diversity is REAL but UNEXPLOITABLE: even the achievable router — handed\n"
              "  the stratum instead of having to predict it — does not beat always calling the best\n"
              "  single model. A learned router can only do worse. CLOSED.")
    else:
        print(f"  POSITIVE: the achievable router beats best-single by {r_acc - bs_acc:+.4f} acc.\n"
              "  This is the first positive result in the project. Re-measure at power before\n"
              "  acting on it, and check the gain survives paying for the routing decision itself.")


if __name__ == "__main__":
    main()
