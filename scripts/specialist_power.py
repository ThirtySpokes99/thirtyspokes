#!/usr/bin/env python
"""The last open branch: is LANGUAGE-specialist traffic routable, or was 17.5% small-n noise?

Twelve measurements have closed general routing and the fixed-harness architecture built on it. One
thread was left live: `where_routing_works.py` ranked `chinese_zodiac` FIRST at 17.5% captured — the
predicted direction, since language competence is the one axis where models plausibly differ in KIND
rather than in TIER. But n=118, which is far too small to distinguish signal from a lucky split.

RouterBench holds ~785 Chinese-language rows across 16 slices. Pooling them gives 6.6x the sample, and
that is enough to settle three questions in order. The order matters: each one makes the next
meaningful, and failing an earlier one makes the later ones unanswerable rather than negative.

  1. ARE THERE SPECIALISTS AT ALL? The signature of a specialist pool is that the best model on
     Chinese traffic is a DIFFERENT model from the best on English traffic. If one model wins both,
     this pool contains no language specialists and the branch cannot be tested with this data --
     that is "untestable", NOT "closed", and the two must not be confused.

  2. IS THERE A BAND? achievable gap + nestedness, the same instruments used throughout. Specialist
     traffic should show LOW nestedness: competence that is not a total order.

  3. DOES A ROUTER CAPTURE IT? held-out capture over repeated splits, with a bootstrap CI. The
     single-split 17.5% has no error bar; a CI that straddles zero settles it. The bar is not "is it
     positive" but "does it clear the reign's eps floor", since capture worth less than the
     incumbency margin cannot be competed on even if real.

An English control bank of matched size is measured alongside every number, so an effect has to be
specific to the Chinese traffic rather than a property of small banks in general.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from thirtyspokes import realdata as rd
from thirtyspokes.oracle import per_sample_oracle_choice
from thirtyspokes.scorer import constant_choice, score_choice
from thirtyspokes.train import train_router

EPS0, EPS_FLOOR = 0.02, 0.002


def nestedness(S: np.ndarray) -> float:
    B = S >= 0.5
    pairs = nested = 0
    for i in range(B.shape[1]):
        for j in range(B.shape[1]):
            if i != j:
                pairs += 1
                nested += 0 if (B[:, i] & ~B[:, j]).any() else 1
    return nested / max(pairs, 1)


def describe(name: str, cache, lam: float) -> dict:
    """Per-traffic summary. `best_k` is ranked by SCORE, not raw accuracy, because the pool is priced
    and the subnet pays the cost-aware objective — the cheapest adequate model has repeatedly been
    the one to beat."""
    scores = [score_choice(cache, constant_choice(cache, k), lam).score for k in range(cache.K)]
    best_k = int(np.argmax(scores))
    B = np.asarray(cache.correct, bool)
    orc = score_choice(cache, per_sample_oracle_choice(cache, lam), lam).score
    sole = int((B & (B.sum(axis=1, keepdims=True) == 1)).sum())
    name_k = cache.models[best_k].name
    print(f"{name:22s} n={cache.Q:5d}  best={name_k:34s} acc={B[:, best_k].mean():.3f}  "
          f"band={orc-scores[best_k]:.4f}  nested={nestedness(B.astype(float)):.3f}  "
          f"sole={sole:4d} ({100*sole/cache.Q:4.1f}%)", flush=True)
    return {"name": name, "best_k": best_k, "best_model": name_k, "band": orc - scores[best_k],
            "n": cache.Q}


def capture_ci(cache, lam: float, hidden: int, iters: int, splits: int) -> tuple[float, float, float]:
    """Held-out capture over repeated random splits — mean and a percentile CI.

    Repeated splits, not one: the 17.5% this script exists to check came from a single split at
    n=118, where the split itself is the dominant source of variance. Each split re-derives its own
    best-single and oracle, because a band measured on other asks does not describe these ones.
    """
    caps = []
    for s in range(splits):
        train, test = cache.split(0.4, seed=s)
        best = max(score_choice(test, constant_choice(test, k), lam).score for k in range(test.K))
        orc = score_choice(test, per_sample_oracle_choice(test, lam), lam).score
        if orc - best < 1e-9:
            continue
        tr = train_router(train, lam=lam, hidden=hidden, iters=iters, seed=s)
        caps.append((tr.evaluate(test).score - best) / (orc - best))
    a = np.array(caps)
    return float(a.mean()), float(np.percentile(a, 5)), float(np.percentile(a, 95))


def main() -> None:
    p = argparse.ArgumentParser(description="power test for the specialist-routing branch")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--splits", type=int, default=8)
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")
    if len(emb) != len(df):
        raise SystemExit(f"embedding rows {len(emb)} != df rows {len(df)}")
    models = rd.ALL_MODELS                       # widest pool = best chance of a specialist

    is_cn = df["eval_name"].str.lower().str.startswith("chinese")
    cn_idx = np.flatnonzero(is_cn.to_numpy())
    # English control of matched size, drawn from the general traffic
    en_idx = np.flatnonzero(~is_cn.to_numpy())
    en_idx = np.random.default_rng(0).choice(en_idx, size=len(cn_idx), replace=False)
    print(f"pool K={len(models)}  chinese n={len(cn_idx)} across "
          f"{df.loc[is_cn, 'eval_name'].nunique()} slices  |  english control n={len(en_idx)}\n")

    banks = {}
    for name, idx in (("chinese (specialist)", cn_idx), ("english (control)", np.sort(en_idx))):
        banks[name] = rd.to_cache(df.iloc[idx].reset_index(drop=True), models, emb[idx])

    print("1. SPECIALIST DIVERSITY — is the best model different on the two traffics?")
    info = {n: describe(n, c, args.lam) for n, c in banks.items()}
    cn, en = info["chinese (specialist)"], info["english (control)"]
    specialists = cn["best_k"] != en["best_k"]
    print(f"\n   best on chinese: {cn['best_model']}\n   best on english: {en['best_model']}")
    if not specialists:
        print("   -> SAME MODEL WINS BOTH. This pool holds no language specialists, so it cannot\n"
              "      test the branch. UNTESTABLE, not closed — settling it needs a pool built to\n"
              "      contain genuine specialists (e.g. a Chinese-trained model against a Western one).")
    else:
        print("   -> DIFFERENT WINNERS: genuine specialist diversity is present in this pool.")

    print("\n2/3. DOES A ROUTER CAPTURE THE BAND? held-out, repeated splits, 90% CI")
    for name, cache in banks.items():
        m, lo, hi = capture_ci(cache, args.lam, args.hidden, args.iters, args.splits)
        band = info[name]["band"]
        need = EPS_FLOOR / band if band > 1e-9 else float("inf")
        verdict = ("CI straddles zero — no measurable routing value" if lo <= 0 <= hi else
                   "positive but below the eps floor — real yet not competable-on" if hi < need else
                   "CLEARS THE EPS FLOOR — a live specialist measurement is warranted")
        print(f"   {name:22s} capture={m:+7.1%}  90% CI [{lo:+.1%}, {hi:+.1%}]  "
              f"needs >={need:.1%} to clear eps  -> {verdict}", flush=True)

    print("\nVERDICT")
    if not specialists:
        print("  The specialist branch remains UNTESTABLE on RouterBench: its pool has one dominant\n"
              "  model on both traffics. Any live measurement must first assemble a pool where\n"
              "  different models genuinely win different languages — otherwise it re-measures the\n"
              "  dominator-pool problem that closed every earlier attempt.")
    else:
        print("  Specialist diversity is present; read the capture CIs above to decide whether a\n"
              "  paid live measurement on purpose-built specialist traffic is justified.")


if __name__ == "__main__":
    main()
