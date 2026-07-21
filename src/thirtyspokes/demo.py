"""End-to-end Phase-0 demonstration — runs every gate in sequence.

    python -m thirtyspokes.demo        (or: thirtyspokes demo)

Walks the architecture's Phase-0 checklist:
  1. Oracle-gap admission gate  — the ceiling principle (complementary vs redundant pool)
  2. Single-shot training       — a router closes part of the gap
  3. Cascade (v1.5) action space — same machinery, lower cost at similar accuracy
  4. Adversarial competition    — reign + dedup + sanity gate reward only genuine work
"""

from __future__ import annotations

import numpy as np

from . import oracle
from .cascade import to_cascade_cache
from .scorer import per_batch_winrate, constant_choice
from .simulate import run_competition
from .synth import SynthConfig, generate
from .train import train_router


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run(seed: int = 7, lam: float = 0.3) -> None:
    _rule("1. ORACLE-GAP ADMISSION GATE  (the ceiling principle)")
    print("A pool is worth building on only if the *routable* accuracy gap clears the")
    print("bar. Note how the per-sample oracle (noise-inflated) roughly doubles the")
    print("honest cross-fit routable ceiling -- using it would greenlight bad pools.")
    print("\n-- complementary pool (specialization=0.85) --")
    comp = generate(SynthConfig(q=8000, specialization=0.85, seed=seed))
    print(oracle.compute(comp, lam=lam).summary())
    print("\n-- redundant pool (specialization=0.0) --")
    redun = generate(SynthConfig(q=8000, specialization=0.0, seed=seed))
    print(oracle.compute(redun, lam=lam).summary())

    _rule("2. SINGLE-SHOT ROUTER TRAINING  (gap closure)")
    train, ev = comp.split(eval_frac=0.35, seed=seed)
    rep = oracle.compute(ev, lam=lam)
    tr = train_router(train, lam=lam, iters=220, seed=seed)
    ev_s = tr.evaluate(ev)
    bs, orc = rep.best_single_obj.score, rep.routable_obj.score
    closed = (ev_s.score - bs) / (orc - bs) * 100 if orc > bs else float("nan")
    print(f"  best single (obj)  : {bs:+.4f}  [{rep.best_single_obj_model}]")
    print(f"  routable ceiling   : {orc:+.4f}  (achievable, cross-fit)")
    print(f"  per-sample oracle  : {rep.ps_oracle_obj.score:+.4f}  (noise-inflated upper bound)")
    note = " (>100%: router beats the conservative cross-fit estimate)" if closed > 100 else ""
    print(f"  trained router     : {ev_s.score:+.4f}   -> closed {closed:.1f}% of the routable gap{note}")
    print(f"  accuracy           : {ev_s.accuracy:.4f} vs best-single {rep.best_single_acc:.4f}"
          f"  ({(ev_s.accuracy-rep.best_single_acc)*100:+.2f} pts)")
    print(f"  normalized cost    : {ev_s.norm_cost:.4f}  (frontier-only would be ~1.0)")

    _rule("3. CASCADE (v1.5) ACTION SPACE  (accuracy lever, still precomputable)")
    casc = to_cascade_cache(comp)               # inherits comp's lambda normalizer
    ctrain, cev = casc.split(eval_frac=0.35, seed=seed)
    ctr = train_router(ctrain, lam=lam, iters=220, seed=seed)
    cev_s = ctr.evaluate(cev)
    # honest comparison: RAW cost per question (same currency units), same normalizer
    raw_single = float(ev.cost[np.arange(ev.Q), tr.choose(ev.embeddings)].mean())
    raw_casc = float(cev.cost[np.arange(cev.Q), ctr.choose(cev.embeddings)].mean())
    raw_frontier = float(ev.cost[:, -1].mean())
    print("  cascade actions = starting rungs on the cost ladder (verify->escalate)")
    print(f"  single-shot router : score={ev_s.score:+.4f}  acc={ev_s.accuracy:.4f}  raw cost/q={raw_single:8.0f}")
    print(f"  cascade router     : score={cev_s.score:+.4f}  acc={cev_s.accuracy:.4f}  raw cost/q={raw_casc:8.0f}")
    print(f"  always-frontier    :                  acc={float(ev.correct[:, -1].mean()):.4f}  raw cost/q={raw_frontier:8.0f}")
    da = (cev_s.accuracy - ev_s.accuracy) * 100
    dc = (raw_casc - raw_single) / raw_single * 100
    print(f"  cascade vs single  : {da:+.2f} accuracy pts for {dc:+.1f}% raw cost "
          f"(higher objective; both ~{raw_frontier/max(raw_casc, 1e-9):.1f}x+ cheaper than always-frontier)")

    _rule("4. ADVERSARIAL COMPETITION  (reign + dedup + sanity gate)")
    master = generate(SynthConfig(q=12000, specialization=0.85, seed=seed))
    comp_rep = run_competition(master, lam=lam, epochs=24, seed=seed)
    print(comp_rep.summary())

    _rule("PHASE-0 VERDICT")
    print("  - oracle-gap gate separates worth-building pools from redundant ones")
    print("  - a tiny CMA-ES head closes a real fraction of the gap (encoder-bounded)")
    print("  - the cascade buys accuracy with modest extra spend (same precompute cache;")
    print("    both routers stay far cheaper than always-frontier)")
    print("  - the reign rewards early genuine improvement; hoarding/copying/degeneracy lose")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
