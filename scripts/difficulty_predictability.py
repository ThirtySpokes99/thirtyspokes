#!/usr/bin/env python
"""Is "can the cheap model handle this?" READABLE FROM THE PROMPT? The question routing rests on.

A routing model's entire product is one judgement: this query is cheap-solvable, that one is not.
Everything else -- the cascade, the frontier scalar, the harness -- is machinery around that call. So
if the judgement is not learnable from the prompt, routing is not learnable, full stop.

Fourteen measurements in this project concluded exactly that, and all fourteen used EXAM benchmarks:
LiveCodeBench, GSM8K, MMLU, MMLU-Pro, RouterBench, CMMLU. That is a confound, not a control. Within
one exam set every item is a similar task at a similar difficulty, so the only thing that varies
across models is which items they happen to get right -- luck, which is genuinely unpredictable. The
finding "the routable signal is luck and luck is not in the prompt" may therefore be a property of
the INSTRUMENT rather than of routing.

THE DECOMPOSITION THAT SETTLES IT. Predict "the cheap model solves this" from the prompt embedding
and measure held-out AUC two ways:

    ACROSS sources   train and test on the whole blend, where difficulty genuinely varies
                     (commonsense vs competition code are not the same ask)
    WITHIN a source  train and test inside ONE benchmark, where difficulty is roughly fixed

If WITHIN is ~0.5 while ACROSS is high, difficulty IS legible in the prompt whenever it actually
varies, and every prior measurement was run on the one distribution engineered to remove that
variance. If BOTH are ~0.5, the negative result survives its own confound and routing is dead on
evidence rather than on instrument choice.

A third arm, ESCALATION, is the decision a cascade actually makes: "strong solves this AND cheap does
not". That is rarer and harder than plain difficulty, and it is the signal the ladder is paid to find.

Pre-committed readings, so the interpretation is not fitted to the outcome:
    within ~= across ~= 0.5   routing is unlearnable; the fourteen measurements stand
    within ~0.5 << across     the benchmarks were the confound; rerun the thesis on real traffic
    within high              difficulty is legible even inside one exam set -- and then the
                             near-zero captured value measured earlier needs a different explanation

Offline and free. Needs data/routerbench_0shot.pkl + data/emb_minilm.npy.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MODELS = [
    "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]


def auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float | None:
    """Held-out AUC for predicting y from the prompt embedding.

    AUC rather than accuracy because the classes are unbalanced and shifting: on an easy source
    almost everything is solved, and a constant predictor would score well on accuracy while being
    exactly the useless router this is trying to detect. AUC asks the routing question directly --
    can you RANK queries by how likely the cheap model is to cope?
    """
    y = np.asarray(y, int)
    if len(y) < 120 or y.mean() in (0.0, 1.0) or min(y.sum(), len(y) - y.sum()) < 25:
        return None                      # too few of one class to rank meaningfully
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    if min(y[te].sum(), len(te) - y[te].sum()) < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))


def main() -> None:
    p = argparse.ArgumentParser(description="is difficulty legible in the prompt?")
    p.add_argument("--cheap", default="mistralai/mistral-7b-chat")
    p.add_argument("--strong", default="gpt-4-1106-preview")
    p.add_argument("--min-n", type=int, default=200, help="min rows for a source to be reported")
    args = p.parse_args()

    df = pd.read_pickle("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")
    if len(emb) != len(df):
        raise SystemExit(f"embedding rows {len(emb)} != df rows {len(df)}")
    cheap_ok = (df[args.cheap].to_numpy(float) >= 0.5).astype(int)
    strong_ok = (df[args.strong].to_numpy(float) >= 0.5).astype(int)
    escalate = ((strong_ok == 1) & (cheap_ok == 0)).astype(int)
    src = df["eval_name"].to_numpy()

    print(f"cheap = {args.cheap}   strong = {args.strong}")
    print(f"n = {len(df)}   cheap solves {cheap_ok.mean():.1%}   strong solves {strong_ok.mean():.1%}"
          f"   escalation cases {escalate.mean():.1%}\n")

    print("=" * 74)
    print("ACROSS SOURCES — the whole blend, where difficulty genuinely varies")
    print("=" * 74)
    a_cheap = auc(emb, cheap_ok)
    a_esc = auc(emb, escalate)
    print(f"  predict 'cheap solves it'        AUC = {a_cheap:.3f}")
    print(f"  predict 'must escalate'          AUC = {a_esc:.3f}")

    print()
    print("=" * 74)
    print("WITHIN A SOURCE — one benchmark at a time, difficulty roughly fixed")
    print("=" * 74)
    rows = []
    for name, sub in sorted(df.groupby("eval_name"), key=lambda kv: -len(kv[1])):
        if len(sub) < args.min_n:
            continue
        m = np.flatnonzero(src == name)
        a = auc(emb[m], cheap_ok[m])
        e = auc(emb[m], escalate[m])
        if a is None:
            continue
        rows.append((name, len(m), a, e))
        print(f"  {name:34s} n={len(m):5d}  cheap AUC = {a:.3f}"
              + (f"   escalate AUC = {e:.3f}" if e is not None else "   escalate: too few cases"))

    within = float(np.mean([r[2] for r in rows])) if rows else float("nan")
    print(f"\n  mean WITHIN-source AUC = {within:.3f}   over {len(rows)} sources")

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    gap = a_cheap - within
    print(f"  across = {a_cheap:.3f}   within = {within:.3f}   gap = {gap:+.3f}")
    if a_cheap < 0.6 and within < 0.6:
        print("  ROUTING IS UNLEARNABLE — difficulty is not legible in the prompt even across\n"
              "  sources. The fourteen prior measurements survive the confound; the negative\n"
              "  result is about routing, not about the benchmarks.")
    elif gap > 0.08:
        print("  THE BENCHMARKS WERE THE CONFOUND. Difficulty is legible whenever it actually\n"
              "  varies, and near-invisible inside any single exam set — which is the only\n"
              "  distribution the prior measurements ever used. Routing must be re-measured on\n"
              "  traffic with a real difficulty spread before the thesis can be called closed.")
    else:
        print("  MIXED — difficulty is somewhat legible both within and across sources. The\n"
              "  bottleneck is then NOT prompt-legibility, and the near-zero captured value\n"
              "  measured earlier needs a different explanation (pool nestedness, or cost\n"
              "  structure leaving no room between best-single and the oracle).")
    print("\nNote: even the 'across' arm here is a blend of EXAM sets. Real traffic has a trivial\n"
          "tail ('hi', 'translate this') that no academic benchmark contains, so this is a LOWER\n"
          "bound on how legible difficulty is in production.")


if __name__ == "__main__":
    main()
