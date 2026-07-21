"""Frontier-pool ceiling on HARD traffic — the last regime where routing could still have a moat.

Every prior no-go (RouterBench, MATH, agentic SWE, specialist-diverse) used *cheap generalists on
ordinary traffic* and hit the same wall: the models had converged, the cheapest was also the best, so
there was nothing to route. That leaves one untested regime, and it is the only one where the thesis
can still be alive: FRONTIER models on GENUINELY HARD questions, where different models plausibly
fail on different problems.

This measures, on hard gradeable traffic (GPQA-Diamond + MATH level 5), the three numbers that decide
it — and, unlike the domain-router demonstrator, the achievable router is LEARNED FROM THE PROMPT
(complementarity on hard traffic is not domain-shaped, so a domain label is the wrong signal):

  * BEST-SINGLE  — best model overall (picked on train, measured on test).
  * ORACLE       — per-query best. An UPPER BOUND no router can exceed. If this ≈ best-single, the
                   models fail on the SAME questions, there is no complementarity, and the thesis is
                   dead for ANY traffic — not just this sample.
  * ROUTER       — the achievable router: per-model P(correct | prompt embedding), cross-validated,
                   route to argmax (cost-aware tie-break). This is what you could actually ship.

The gate: ORACLE must meaningfully beat BEST-SINGLE (else there is nothing to route), AND the ROUTER
must capture enough of that gap to matter (else the headroom is real but unreachable — which is what
killed the RouterBench case: the gap existed but was per-query noise, not a learnable signal).

Run: set -a && . ./.env && set +a && python scripts/frontier_ceiling_experiment.py
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from thirtyspokes.eval import config
from thirtyspokes.gateway.gateway import OpenRouterBackend

# Frontier pool, spread across cost. The point is CAPABILITY DIVERSITY, not price tiers: if these
# models fail on the same hard questions, no pool of them can be routed profitably.
DEFAULT_MODELS = [
    "anthropic/claude-opus-4.7",
    "openai/gpt-4o",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-v3.2-exp",
    "meta-llama/llama-3.3-70b-instruct",   # cheap anchor: is the cheapest STILL best-single?
]

# Token budgets must be big enough for REASONING models, or the benchmark silently scores them ~0.
# Measured: gemini-2.5-pro spends its whole completion budget on reasoning tokens and never reaches an
# answer at 2k (2044/2048 used, no answer) or even 8k; it needs ~18k to finish one MATH-5 problem. A
# small budget therefore doesn't measure capability, it measures verbosity — and would manufacture a
# "no complementarity" result. Same budget for every model, so the comparison stays fair.
MC_TOKENS = 16384
MATH_TOKENS = 32768


def _extract_letter(resp: str, n_opts: int = 10) -> str | None:
    last = chr(64 + n_opts)                                  # 'D' for 4 options, 'J' for 10
    m = re.search(rf"ANSWER[:\s]*\(?([A-{last}])\)?", str(resp).upper())
    if m:
        return m.group(1)
    hits = re.findall(rf"\b([A-{last}])\b", str(resp).strip().upper())
    return hits[-1] if hits else None


def load_hard_tasks(n_mc: int, n_math: int, seed: int) -> list[dict]:
    """Hard + exactly gradeable.

    MMLU-Pro: 10-option, reasoning-heavy MMLU (frontier models ~70-80% vs ~88% on plain MMLU) —
    hard enough that models genuinely differ. (GPQA-Diamond would be the sharper probe but is a GATED
    HF dataset; request access and pass --use-gpqa to swap it in.)
    MATH level 5: the hardest competition tier.
    """
    from datasets import load_dataset

    from thirtyspokes.eval import hard_tasks

    tasks: list[dict] = []
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n_mc, len(ds)), replace=False)
    for i in idx:
        r = ds[int(i)]
        opts = r["options"]
        body = "\n".join(f"{chr(65 + j)}) {o}" for j, o in enumerate(opts))
        gold = str(r["answer"]).strip().upper()
        tasks.append({
            "id": f"mmlupro-{int(i)}", "src": "mmlu_pro",
            "prompt": (f"{r['question']}\n{body}\n\nThink briefly, then end with "
                       f"'Answer: X' where X is the option letter."),
            "grade": (lambda resp, g=gold, k=len(opts): float(_extract_letter(resp, k) == g)),
            "max_tokens": MC_TOKENS,
        })
    for t in hard_tasks.load_math(n_math, min_level=5, seed=seed):
        tasks.append({"id": t.task_id, "src": "math5", "prompt": t.prompt,
                      "grade": (lambda r, g=t.gold: hard_tasks.grade_hard(r, g)),
                      "max_tokens": MATH_TOKENS})
    return tasks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-mc", type=int, default=120, help="MMLU-Pro (hard 10-option MC) questions")
    p.add_argument("--n-math", type=int, default=120)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = config.LiveConfig(); cfg.require_key()
    be = OpenRouterBackend(cfg.api_key, cfg.base_url, timeout=180.0, max_retries=2,
                           price_fn=config.price_for)
    tasks = load_hard_tasks(args.n_mc, args.n_math, args.seed)
    print(f"{len(tasks)} hard tasks x {len(models)} models = {len(tasks)*len(models)} real calls",
          flush=True)

    corr: dict = defaultdict(dict)
    cost: dict = defaultdict(dict)
    errors: dict = defaultdict(int)

    def ask(job):
        t, m = job
        try:
            text, _i, _o, c = be.complete(m, [{"role": "user", "content": t["prompt"]}],
                                          {"max_tokens": t["max_tokens"]})
            return t, m, t["grade"](text), c, None
        except Exception as e:  # noqa: BLE001 — a model erroring is data, not a crash
            return t, m, 0.0, 0.0, e

    jobs = [(t, m) for t in tasks for m in models]
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for t, m, ok, c, err in ex.map(ask, jobs):
            corr[m][t["id"]] = ok
            cost[m][t["id"]] = c
            if err is not None:
                errors[m] += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} calls", flush=True)

    models = [m for m in models if errors[m] < len(tasks)]
    ids = [t["id"] for t in tasks]
    src = {t["id"]: t["src"] for t in tasks}
    y = np.array([[corr[m][i] for m in models] for i in ids])       # (n_tasks, n_models)
    c_mat = np.array([[cost[m][i] for m in models] for i in ids])

    # ---- per-model accuracy + the complementarity diagnostic --------------------------------
    print("\n=== per-model accuracy ===")
    print("model".ljust(38) + " mmluPro  math5  overall     $cost")
    for j, m in enumerate(models):
        g = np.mean([y[k, j] for k, i in enumerate(ids) if src[i] == "mmlu_pro"])
        mm = np.mean([y[k, j] for k, i in enumerate(ids) if src[i] == "math5"])
        print(f"{m.ljust(38)}{g:6.2f}{mm:8.2f}{y[:, j].mean():9.2f}{c_mat[:, j].sum():10.3f}")

    best_j = int(np.argmax(y.mean(axis=0)))
    best_m = models[best_j]
    oracle = (y.max(axis=1) >= 1.0).astype(float)
    # THE number: how often does best-single fail but someone else succeed? That is all a router
    # could ever recover. If it is ~0, no pool of these models is routable, on any traffic.
    routable = float(np.mean((y[:, best_j] < 1.0) & (oracle >= 1.0)))
    print(f"\noracle acc = {oracle.mean():.3f} | best-single ({best_m}) = {y[:, best_j].mean():.3f}")
    print(f"routable headroom (best-single WRONG but some model RIGHT) = {routable:.3f}")
    print(f"all-models-wrong (unroutable, nobody solves it)            = {1 - oracle.mean():.3f}")

    # ---- ACHIEVABLE router: learned from the prompt, cross-validated -------------------------
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    print("\nencoding prompts (MiniLM) ...", flush=True)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = enc.encode([t["prompt"] for t in tasks], show_progress_bar=False,
                   normalize_embeddings=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    router_hit = np.zeros(len(ids))
    router_cost = np.zeros(len(ids))
    bs_hit = np.zeros(len(ids))
    bs_cost = np.zeros(len(ids))
    for tr, te in kf.split(X):
        # per-model P(correct | prompt), trained on the training fold only
        prob = np.zeros((len(te), len(models)))
        for j, _m in enumerate(models):
            yj = y[tr, j]
            if len(set(yj)) < 2:                       # degenerate: model always right/wrong on fold
                prob[:, j] = yj.mean()
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], yj)
            prob[:, j] = clf.predict_proba(X[te])[:, 1]
        # best-single is also re-picked per fold, on the training fold (no leakage)
        bj = int(np.argmax(y[tr].mean(axis=0)))
        pick = prob.argmax(axis=1)                      # route to most-likely-correct
        for k, ti in enumerate(te):
            router_hit[ti] = y[ti, pick[k]]
            router_cost[ti] = c_mat[ti, pick[k]]
            bs_hit[ti] = y[ti, bj]
            bs_cost[ti] = c_mat[ti, bj]

    oc_cost = float(sum(
        (c_mat[k][y[k] >= 1.0].min() if oracle[k] >= 1.0 else c_mat[k].min())
        for k in range(len(ids))))

    n = len(ids)
    print(f"\n=== ceiling (n={n} hard queries, {args.folds}-fold CV) ===")
    print(f"  BEST-SINGLE  {best_m:32s} acc={bs_hit.mean():.3f}  cost=${bs_cost.sum():.3f}")
    print(f"  ROUTER       {'learned P(correct|prompt)':32s} acc={router_hit.mean():.3f}  "
          f"cost=${router_cost.sum():.3f}")
    print(f"  ORACLE       {'per-query best (upper bound)':32s} acc={oracle.mean():.3f}  "
          f"cost=${oc_cost:.3f}")

    gap_o = oracle.mean() - bs_hit.mean()
    gap_r = router_hit.mean() - bs_hit.mean()
    print("\n=== verdict ===")
    print(f"  oracle headroom over best-single : {gap_o:+.3f}  (upper bound — nothing to route if ~0)")
    print(f"  ACHIEVABLE gap (router − best-single): {gap_r:+.3f}  (what you could ship)")
    captured = (gap_r / gap_o * 100) if gap_o > 1e-9 else 0.0
    print(f"  fraction of the headroom the router captures: {captured:.0f}%")
    if gap_o < 0.02:
        print("\n  => NO-GO: the frontier models fail on the SAME hard questions. No complementarity,")
        print("     so no pool of them is routable — this is traffic-independent. Thesis dead.")
    elif gap_r < 0.01:
        print("\n  => NO-GO: real headroom exists but is NOT learnable from the prompt (per-query")
        print("     noise, not signal) — the RouterBench failure mode, now at the frontier.")
    else:
        print("\n  => SIGNAL: routing beats best-single on hard traffic. Quantify, then find the traffic.")
    if errors:
        print(f"  note: model errors {dict(errors)}")


if __name__ == "__main__":
    main()
