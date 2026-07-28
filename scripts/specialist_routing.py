#!/usr/bin/env python
"""The last open cell: does routing work on MIXED-DOMAIN traffic with a SPECIALIST pool?

Every prior measurement in this investigation missed this combination, and it is exactly where the
surviving hypothesis lives:

  head_spread.py        mixed traffic, but the CURATED pool -- a pure capability ladder
                        (mistral -> mixtral -> Yi -> gpt-3.5 -> gpt-4). No specialists.  -> 1.1%
  where_routing_works.py  the full 11-model pool, but ONE BENCHMARK AT A TIME -- within a single
                        domain there is no domain signal to route on.        -> median -1.6%

Routing on domain needs both at once: traffic that spans domains, and a pool where different models
genuinely win different ones. RouterBench's 11-model pool contains `code-llama-instruct-34b` (a code
specialist) and `Yi-34B-Chat` (Chinese), so the combination is testable here for free.

Why it could break the negative result: difficulty and luck are close to unpredictable from a prompt,
which is what every measurement so far actually established. DOMAIN is trivially predictable from a
prompt. If competence is non-nested across domains, a router only has to recognise the domain -- an
easy problem -- rather than forecast which model gets lucky.

Reports:
  1. the specialisation structure itself (per-model accuracy by domain, and who wins each)
  2. captured fraction in three configurations, isolating the cause:
       A  mixed-domain + specialist pool   <- the untested cell
       B  mixed-domain + ladder pool       <- controls for the POOL
       C  single-domain + specialist pool  <- controls for the TRAFFIC
     A >> B and A >> C  =>  specialisation is the answer and the suite has a specification.
     A ~ B ~ C          =>  the thesis is closed for good.

Offline, free.
"""
from __future__ import annotations

import argparse

import numpy as np

from thirtyspokes import realdata as rd
from thirtyspokes.oracle import per_sample_oracle_choice
from thirtyspokes.scorer import constant_choice, score_choice
from thirtyspokes.train import train_router

# domains chosen so that DIFFERENT model families plausibly win different ones
DOMAINS = {
    "code": ["mbpp"],
    "law": ["mmlu-professional-law"],
    "commonsense": ["hellaswag", "winogrande"],
    "math": ["grade-school-math"],
    "knowledge": ["mmlu-miscellaneous", "mmlu-professional-psychology"],
}
LADDER = rd.CURATED_POOL          # capability tiers, no specialists
SPECIALIST = rd.ALL_MODELS        # includes code-llama (code) and Yi-34B (Chinese)


def band_and_capture(cache, seeds, hidden, lam, iters, tag):
    tr, te = cache.split(0.3, seed=0)
    best1 = max(score_choice(te, constant_choice(te, k), lam).score for k in range(te.K))
    orc = score_choice(te, per_sample_oracle_choice(te, lam), lam).score
    band = orc - best1
    got = np.array([train_router(tr, lam=lam, hidden=hidden, iters=iters,
                                 seed=s).evaluate(te).score for s in range(seeds)])
    cap = (got.max() - best1) / band if band > 1e-9 else 0.0
    print(f"{tag:44s} n={cache.Q:5d} K={cache.K:2d} band={band:.4f} "
          f"router={got.max():+.4f} captured={100*cap:6.1f}% spread={got.max()-got.min():.5f}",
          flush=True)
    return cap


def main() -> None:
    p = argparse.ArgumentParser(description="mixed-domain x specialist-pool routing")
    p.add_argument("--per-domain", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb = np.load("data/emb_minilm.npy")

    # ---------- 1. is there any specialisation to route on? --------------------------------
    print("=" * 108)
    print("SPECIALISATION STRUCTURE — per-model accuracy by domain (winner starred)")
    print("=" * 108)
    sel_by_domain = {}
    for dom, names in DOMAINS.items():
        m = df["eval_name"].isin(names).to_numpy()
        if m.sum() < 60:
            continue
        sel_by_domain[dom] = m
    accs = {}
    for dom, m in sel_by_domain.items():
        a = np.array([(df.loc[m, mod].astype(float).to_numpy() >= 0.5).mean()
                      for mod in SPECIALIST])
        accs[dom] = a
    print(f"{'model':36s} " + "".join(f"{d:>13s}" for d in accs))
    winners = {d: int(np.argmax(a)) for d, a in accs.items()}
    for i, mod in enumerate(SPECIALIST):
        cells = "".join(f"{a[i]:12.3f}" + ("*" if winners[d] == i else " ")
                        for d, a in accs.items())
        print(f"{mod[:36]:36s} {cells}")
    distinct = len(set(winners.values()))
    print(f"\ndistinct per-domain winners: {distinct} of {len(winners)} domains "
          f"-> {'NON-NESTED (routable in principle)' if distinct > 1 else 'ONE MODEL DOMINATES ALL'}")

    # ---------- 2. the three configurations -------------------------------------------------
    rng = np.random.default_rng(0)
    idx = []
    for dom, m in sel_by_domain.items():
        cand = np.flatnonzero(m)
        idx.append(rng.choice(cand, min(args.per_domain, len(cand)), replace=False))
    mixed = np.concatenate(idx)
    print("\n" + "=" * 108)
    print("CAPTURED FRACTION — isolating pool vs traffic")
    print("=" * 108)
    A = band_and_capture(rd.to_cache(df.iloc[mixed].reset_index(drop=True), SPECIALIST, emb[mixed]),
                         args.seeds, args.hidden, args.lam, args.iters,
                         "A  mixed-domain + SPECIALIST pool  (untested cell)")
    B = band_and_capture(rd.to_cache(df.iloc[mixed].reset_index(drop=True), LADDER, emb[mixed]),
                         args.seeds, args.hidden, args.lam, args.iters,
                         "B  mixed-domain + ladder pool      (pool control)")
    Cs = []
    for dom, m in sel_by_domain.items():
        cand = np.flatnonzero(m)
        take = rng.choice(cand, min(args.per_domain * len(sel_by_domain), len(cand)), replace=False)
        if len(take) < 200:
            continue
        Cs.append(band_and_capture(
            rd.to_cache(df.iloc[take].reset_index(drop=True), SPECIALIST, emb[take]),
            args.seeds, args.hidden, args.lam, args.iters,
            f"C  single-domain ({dom}) + SPECIALIST pool"))

    print("\n" + "=" * 108)
    c_med = float(np.median(Cs)) if Cs else 0.0
    print(f"A (mixed+specialist) = {100*A:.1f}%   B (mixed+ladder) = {100*B:.1f}%   "
          f"C (single+specialist, median) = {100*c_med:.1f}%")
    if A > 0.25 and A > 2 * max(B, c_med):
        print("VERDICT: SPECIALISATION IS THE ANSWER — the suite needs mixed-domain traffic over a "
              "specialist pool, and the thesis reopens with a concrete specification.")
    else:
        print("VERDICT: NO — the untested cell behaves like the rest. Routing does not become "
              "learnable with domain signal and specialists, and the thesis is closed for good.")


if __name__ == "__main__":
    main()
