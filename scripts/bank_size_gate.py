#!/usr/bin/env python
"""Is there ANY bank size where the fixed-harness architecture is defensible? Real data, free.

Phase 4 left the design pinned between two bounds that pull on the same knob:

  small bank -> a routing-table memoriser fits the whole thing and scores like an oracle
                (`memoriser_capacity.py`: 100% of the oracle at 1,000 asks, 12% at 20,000)
  large bank -> the memoriser's edge decays, but honest routers converge on the one best head and
                copy-dedup starts disqualifying them (`router_subnet.py`, measured overlap at 8k)

Both of those were measured on SYNTHETIC worlds, where the routable signal was placed by hand. This
runs the same question against RouterBench: 11 real models, real per-ask outcomes, real prices, real
MiniLM embeddings.

THE REDUCTION. The subnet scores asks drawn from its public bank, so every scored ask is one a miner
could have trained on. "Memoriser" and "honest router" are therefore not two agents — they are one
fit, read two ways:

    IN-SAMPLE capture   score on the bank it trained on  = what a memoriser actually collects
    HELD-OUT capture    score on asks it has never seen  = genuine, transferable routing value
    EDGE                the difference                   = what enumeration buys over generalising

Capture is normalised against the band a perfect per-ask oracle opens over the best single model, so
1.0 is oracle routing and 0.0 is "you may as well always call the best model".

WHAT WOULD VINDICATE THE ARCHITECTURE: some bank size where held-out capture is clearly positive AND
the edge has collapsed. That is a bank big enough to defeat memorisation while still leaving a real
router something to win. If held-out capture is ~0 at every size, then at large banks there is no
attack because there is nothing left to attack — and the architecture has no viable operating point
on this traffic regardless of how the suite is assembled.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from thirtyspokes import realdata as rd
from thirtyspokes.oracle import per_sample_oracle_choice
from thirtyspokes.scorer import constant_choice, score_choice
from thirtyspokes.train import train_router

EPS_FLOOR = 0.002        # reign incumbency margin: capture worth less than this cannot be competed on


def band(cache, lam: float) -> tuple[float, float]:
    """(best-single, oracle) on THIS slice — the two ends of the routable band. Both must be
    recomputed per slice: a band measured on the training asks does not describe the held-out ones."""
    best = max(score_choice(cache, constant_choice(cache, k), lam).score for k in range(cache.K))
    orc = score_choice(cache, per_sample_oracle_choice(cache, lam), lam).score
    return best, orc


def capture(score: float, best: float, orc: float) -> float:
    return (score - best) / (orc - best) if orc - best > 1e-9 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="bank-size gate for the fixed-harness architecture")
    p.add_argument("--sizes", default="200,1000,5000,20000")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--pool", choices=["curated", "all"], default="curated")
    p.add_argument("--heldout", type=int, default=4000, help="disjoint asks for the generalisation read")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--out", default="")
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    models = rd.CURATED_POOL if args.pool == "curated" else rd.ALL_MODELS
    emb = np.load("data/emb_minilm.npy")
    if len(emb) != len(df):
        raise SystemExit(f"embedding rows {len(emb)} != df rows {len(df)}")
    full = rd.to_cache(df, models, emb)
    print(f"pool={args.pool} K={full.K} Q={full.Q} D={full.D} hidden={args.hidden} lam={args.lam}",
          flush=True)

    # one fixed held-out slice for every bank size, so the generalisation reads are comparable
    rest, held = full.split(args.heldout / full.Q, seed=0)
    hb, ho = band(held, args.lam)
    print(f"held-out slice: n={held.Q}  best-single={hb:+.4f}  oracle={ho:+.4f}  band={ho-hb:.4f}\n",
          flush=True)

    print(f"{'bank':>7s} {'band':>7s} {'in-sample':>10s} {'held-out':>9s} {'EDGE':>7s}  verdict")
    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        if n > rest.Q:
            continue
        bank, _ = rest.split(1.0 - n / rest.Q, seed=1)     # the PUBLIC bank of this size
        bb, bo = band(bank, args.lam)
        ins, outs = [], []
        for seed in range(args.seeds):
            tr = train_router(bank, lam=args.lam, hidden=args.hidden, iters=args.iters, seed=seed)
            ins.append(capture(tr.evaluate(bank).score, bb, bo))
            outs.append(capture(tr.evaluate(held).score, hb, ho))
        i_cap, o_cap = float(np.mean(ins)), float(np.mean(outs))
        edge = i_cap - o_cap
        # a bank is only usable if enumeration buys little AND generalisation is worth something
        verdict = ("MEMORISABLE — enumeration buys most of the oracle" if edge > 0.15 else
                   "NOTHING TO WIN — generalisation captures ~nothing" if o_cap < 0.05 else
                   "USABLE — edge collapsed, real routing value remains")
        print(f"{bank.Q:7d} {bo-bb:7.4f} {i_cap:10.1%} {o_cap:9.1%} {edge:7.1%}  {verdict}",
              flush=True)
        rows.append({"bank": bank.Q, "band": bo - bb, "in_sample": i_cap, "held_out": o_cap,
                     "edge": edge, "verdict": verdict})

    print()
    usable = [r for r in rows if r["verdict"].startswith("USABLE")]
    if usable:
        print(f"VIABLE OPERATING POINT at bank >= {min(r['bank'] for r in usable)}: "
              f"held-out capture {max(r['held_out'] for r in usable):.1%}, edge collapsed.")
    else:
        print("NO VIABLE OPERATING POINT on this traffic. Small banks are memorisable; at large\n"
              "banks the memoriser's edge is gone but so is the routing value, so the architecture\n"
              "is not defending anything worth defending. Fixing this needs traffic where a router\n"
              "generalises — not a bigger bank of the same traffic.")
    if rows:
        best = max(rows, key=lambda r: r["held_out"])
        print(f"\nbest held-out capture anywhere: {best['held_out']:.1%} at bank={best['bank']} "
              f"(band {best['band']:.4f}); reign eps floor {EPS_FLOOR} needs capture "
              f">= {EPS_FLOOR / max(best['band'], 1e-9):.1%} just to be competable-on.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
