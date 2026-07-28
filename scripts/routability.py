"""What SHAPE of traffic is actually routable? Measured on RouterBench, free.

The live suite ranks on LiveCodeBench and measures an achievable gap of ~0.04 -- with only ONE of six
models ever sole-correct. The hypothesis this tests: LCB fails not because it is a bad benchmark but
because its competence is NESTED. Problems sit on a single difficulty axis, so strong models solve a
superset of what weak ones solve. When competence is a total order, routing collapses to "pick the
cheapest model above the difficulty threshold" -- and there is nothing to learn per-query.

Routing needs the opposite: models that are each best at DIFFERENT things, at different prices.

Three diagnostics per candidate traffic, using the subnet's own scoring functions so the numbers are
directly comparable to what the validator computes:

  achievable_gap  what a perfect per-ask router adds over randomising over fixed models
  sole-correct    asks exactly one model solves -- the raw routing signal
  nestedness      fraction of ordered model pairs where solved(a) is a SUBSET of solved(b).
                  1.0 = a pure capability ladder = nothing to route.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thirtyspokes.koth.verify import achievable_gap

MODELS = [
    "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]
RNG = np.random.default_rng(0)


def matrices(df: pd.DataFrame):
    S = df[MODELS].to_numpy(float)
    C = df[[f"{m}|total_cost" for m in MODELS]].to_numpy(float)
    return S, C


def nestedness(S: np.ndarray) -> float:
    """Fraction of ordered model pairs whose solved-sets are nested (a subset of b).

    A pure capability ladder scores 1.0: every weaker model's wins are contained in every stronger
    model's. That is the shape where routing has nothing to decide -- you only ever need to know how
    hard the question is, never which model suits it."""
    B = S >= 0.5
    pairs = nested = 0
    for i in range(B.shape[1]):
        for j in range(B.shape[1]):
            if i == j:
                continue
            pairs += 1
            if not (B[:, i] & ~B[:, j]).any():      # solved(i) subset-of solved(j)
                nested += 1
    return nested / max(pairs, 1)


def report(name: str, df: pd.DataFrame) -> dict:
    if len(df) < 40:
        return {}
    S, C = matrices(df)
    keep = ~np.isnan(S).any(axis=1) & ~np.isnan(C).any(axis=1)
    S, C = S[keep], C[keep]
    if len(S) < 40:
        return {}
    gap = achievable_gap(S, C)
    B = S >= 0.5
    sole = ((B) & (B.sum(axis=1, keepdims=True) == 1)).sum(axis=0)
    print(f"{name:34s} n={len(S):6d}  gap={gap:.4f}  nested={nestedness(S):.3f}  "
          f"sole={int(sole.sum()):4d} ({100*sole.sum()/len(S):5.1f}%)  "
          f"acc[min..max]={B.mean(axis=0).min():.2f}..{B.mean(axis=0).max():.2f}", flush=True)
    return {"name": name, "n": len(S), "gap": gap, "sole_frac": sole.sum() / len(S)}


def main() -> None:
    df = pd.read_pickle("data/routerbench_0shot.pkl")
    print("=" * 108)
    print("SINGLE-SOURCE traffic  (what the live suite looks like: one benchmark)")
    print("=" * 108)
    singles = []
    for name, sub in sorted(df.groupby("eval_name"), key=lambda kv: -len(kv[1]))[:12]:
        r = report(name, sub)
        if r:
            singles.append(r)

    print()
    print("=" * 108)
    print("MIXED traffic  (many sources blended -- what production traffic actually looks like)")
    print("=" * 108)
    counts = df["eval_name"].value_counts()
    big = [n for n in counts.index if counts[n] >= 200]
    for k in (2, 4, 8, 16, len(big)):
        k = min(k, len(big))
        picks = list(RNG.choice(big, size=k, replace=False)) if k < len(big) else big
        per = max(40, 4000 // k)
        sub = pd.concat([df[df.eval_name == n].sample(min(per, counts[n]), random_state=1)
                         for n in picks])
        report(f"blend of {k:2d} sources", sub)

    print()
    print("=" * 108)
    print("EXPLICIT easy+hard pairing  (a difficulty SPREAD, not a difficulty LEVEL)")
    print("=" * 108)
    acc_by = {n: (df[df.eval_name == n][MODELS].to_numpy(float) >= 0.5).mean()
              for n in big}
    easy = max(acc_by, key=acc_by.get)
    hard = min(acc_by, key=acc_by.get)
    print(f"easiest source: {easy} (mean acc {acc_by[easy]:.2f})   "
          f"hardest: {hard} (mean acc {acc_by[hard]:.2f})")
    for frac in (0.25, 0.5, 0.75):
        n_e = int(2000 * frac)
        sub = pd.concat([df[df.eval_name == easy].sample(min(n_e, counts[easy]), random_state=2),
                         df[df.eval_name == hard].sample(min(2000 - n_e, counts[hard]), random_state=2)])
        report(f"{int(frac*100)}% easy / {100-int(frac*100)}% hard", sub)


if __name__ == "__main__":
    main()
