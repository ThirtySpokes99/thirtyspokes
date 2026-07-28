#!/usr/bin/env python
"""Phase-0 gate, part 3: is VERIFICATION the competitive surface routing wasn't?

Parts 1-2 measured the routing surface and it is empty. A capped head captures 1.1% of the pick-one
band with 0.005 spread across seeds; under every cascade verifier quality it captures <=9% with
0.001-0.008 spread -- at or below the reign's eps in all six configurations. The subnet would
ossify on earliest-commit regardless.

The same sweep pointed somewhere else entirely. Reading down the best-fixed-entry column as verifier
quality went from useless to perfect: +0.8126 -> +0.9305. A 0.118 swing driven ENTIRELY by
verification, with no routing model involved. The harness delivers the value; the router does not.

So invert the artifact. The harness always enters at the cheapest rung and escalates; the MINER'S
MODEL decides whether to accept the answer in hand. That is a genuinely different learning problem --
post-hoc judgement of (question, answer) rather than prediction of model success from a bare prompt --
and it is the one this data says the score actually responds to.

Same gate as before, so the numbers are directly comparable:
  band     perfect-verifier score minus the best trivial policy (always-accept / always-escalate)
  captured share of that band a trained capped head reaches
  spread   held-out score range across independently-seeded honest miners  <- the ossification gate

Offline. Needs data/routerbench_0shot.pkl + data/emb_minilm.npy + sentence-transformers.
"""
from __future__ import annotations

import argparse
import ast

import numpy as np

from thirtyspokes import realdata as rd
from thirtyspokes.router import RouterHead
from thirtyspokes.sepcmaes import SepCMAES

EPS0, EPS_FLOOR = 0.02, 0.002


def clean(x) -> str:
    """RouterBench stores responses as the repr of a one-element list."""
    s = str(x)
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)) and v:
                return str(v[0])
        except Exception:  # noqa: BLE001 — malformed rows fall through as raw text
            pass
    return s


def ladder(vok, correct, cost, order):
    """Always enter at the cheapest rung; escalate while the verifier rejects.

    Identical semantics to `cascade.to_cascade_cache` at r=0, so the offline trainer and the live
    harness would agree. Returns (banked_correct, total_cost)."""
    Q = len(correct)
    done = np.zeros(Q, bool)
    banked = np.zeros(Q, bool)
    total = np.zeros(Q, np.float32)
    for pos, m in enumerate(order):
        active = ~done
        total[active] += cost[active, m]
        newly = active & (vok[:, m] | (pos == len(order) - 1))
        banked[newly] = correct[newly, m]
        done[newly] = True
    return banked, total


def score_of(vok, correct, cost, order, cost_norm, lam):
    banked, total = ladder(vok, correct, cost, order)
    return float(banked.mean()) - lam * float((total / cost_norm).mean())


def main() -> None:
    p = argparse.ArgumentParser(description="verifier action-space competitive spread")
    p.add_argument("--n", type=int, default=6000, help="questions to sample")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=120)
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb_all = np.load("data/emb_minilm.npy")
    idx = np.random.default_rng(0).choice(len(df), min(args.n, len(df)), replace=False)
    sub, pemb = df.iloc[idx].reset_index(drop=True), emb_all[idx]
    cache = rd.to_cache(sub, rd.CURATED_POOL, pemb)
    order = np.argsort([m.price_per_token for m in cache.models])   # cheap -> expensive
    print(f"Q={cache.Q} K={cache.K} lam={args.lam} hidden={args.hidden}", flush=True)
    print("ladder order:", [cache.models[m].name for m in order], flush=True)

    # --- embed every model's ANSWER (the new information the verifier gets) -------------------
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    A = np.zeros((cache.Q, cache.K, pemb.shape[1]), dtype=np.float32)
    for k, m in enumerate(rd.CURATED_POOL):
        texts = [clean(t) for t in sub[f"{m}|model_response"].to_numpy()]
        A[:, k, :] = enc.encode(texts, batch_size=256, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        print(f"  embedded answers: {m}", flush=True)

    # features per (q,k): the question AND the answer in hand
    X = np.concatenate([np.repeat(pemb[:, None, :], cache.K, axis=1), A], axis=2)  # (Q,K,2D)
    D2 = X.shape[2]
    correct, cost, cnorm = cache.correct, cache.cost, cache.cost_max

    tr = np.random.default_rng(1).permutation(cache.Q)
    cut = int(0.7 * cache.Q)
    itr, ite = tr[:cut], tr[cut:]

    def sc(vok, sel):
        return score_of(vok[sel], correct[sel], cost[sel], order, cnorm, args.lam)

    # --- trivial policies + the perfect-verifier ceiling --------------------------------------
    always = np.ones_like(correct, dtype=bool)      # accept immediately -> always cheapest
    never = np.zeros_like(correct, dtype=bool)      # never accept -> always escalate to the top
    perfect = correct.copy()                        # accept iff the answer is actually right
    base = {"always-accept (cheapest)": sc(always, ite),
            "never-accept (priciest)": sc(never, ite),
            "PERFECT verifier": sc(perfect, ite)}
    for name, v in base.items():
        print(f"  {name:28s} {v:+.4f}", flush=True)
    trivial = max(base["always-accept (cheapest)"], base["never-accept (priciest)"])
    band = base["PERFECT verifier"] - trivial
    print(f"  band (perfect - best trivial) = {band:.4f}", flush=True)

    # --- N honest miners, identical data, different seeds -------------------------------------
    head = RouterHead(D2, 2, args.hidden)
    print(f"verifier head params = {head.n_params}", flush=True)
    Xtr, Xte = X[itr].reshape(-1, D2), X[ite].reshape(-1, D2)

    def vok_of(theta, Xf, n):
        acc = head.distribution(theta, Xf)[:, 1] > 0.5
        return acc.reshape(n, cache.K)

    scores = []
    for seed in range(args.seeds):
        def neg(theta):
            return -score_of(vok_of(theta, Xtr, len(itr)), correct[itr], cost[itr],
                             order, cnorm, args.lam)
        es = SepCMAES(head.init_theta(seed), sigma0=0.3, seed=seed + 1)
        best_t, best_n = es.mean.copy(), neg(es.mean)
        for _ in range(args.iters):
            pop = es.ask()
            fits = np.array([neg(x) for x in pop])
            es.tell(pop, fits)
            b = int(fits.argmin())
            if fits[b] < best_n:
                best_n, best_t = float(fits[b]), pop[b].copy()
        held = score_of(vok_of(best_t, Xte, len(ite)), correct[ite], cost[ite],
                        order, cnorm, args.lam)
        scores.append(held)
        print(f"  seed {seed}: held-out {held:+.4f}", flush=True)

    scores = np.array(scores)
    spread = scores.max() - scores.min()
    captured = (scores.max() - trivial) / band if band > 1e-9 else 0.0
    print()
    print("=" * 78)
    print(f"SPREAD   {spread:.5f}   (min {scores.min():+.4f} / max {scores.max():+.4f})")
    print(f"CAPTURED {100*captured:.1f}% of the {band:.4f} verification band")
    print(f"VERDICT  " + ("OSSIFIES — below the eps floor, no miner can dethrone another"
                          if spread < EPS_FLOOR else
                          "MARGINAL — inside the eps band" if spread < EPS0 else
                          "COMPETITIVE — spread exceeds eps; better verifiers can take the crown"))


if __name__ == "__main__":
    main()
