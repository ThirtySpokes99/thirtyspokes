"""thirtyspokes CLI — drive every Phase-0 gate from the terminal.

    thirtyspokes gen-cache   --out cache.npz [--specialization S]
    thirtyspokes oracle-gap  [--cache cache.npz | --synth] [--lam L] [--cascade]
    thirtyspokes train       [--cache cache.npz | --synth] [--lam L] [--iters N]
    thirtyspokes simulate    [--lam L] [--epochs E]
    thirtyspokes demo        run the whole pipeline (scripts/run_phase0.py)
"""

from __future__ import annotations

import argparse

from . import oracle
from .cache import OutcomeCache
from .cascade import to_cascade_cache
from .synth import SynthConfig, generate


def _load_or_synth(args) -> OutcomeCache:
    if getattr(args, "cache", None):
        return OutcomeCache.load(args.cache)
    cfg = SynthConfig(q=args.q, specialization=args.specialization, seed=args.seed)
    return generate(cfg)


def cmd_gen_cache(args) -> None:
    cache = generate(SynthConfig(q=args.q, specialization=args.specialization, seed=args.seed))
    cache.save(args.out)
    print(f"wrote synthetic cache: Q={cache.Q} K={cache.K} D={cache.D} -> {args.out}")
    print("  models: " + ", ".join(f"{m.name}({m.role})" for m in cache.models))


def cmd_oracle_gap(args) -> None:
    cache = _load_or_synth(args)
    if args.cascade:
        cache = to_cascade_cache(cache)
        print("[cascade action space: actions = starting rungs on the cost ladder]")
    print(oracle.compute(cache, lam=args.lam, threshold_points=args.threshold).summary())


def cmd_train(args) -> None:
    from .train import train_router  # lazy: heavier import path
    cache = _load_or_synth(args)
    if args.cascade:
        cache = to_cascade_cache(cache)
    train, ev = cache.split(eval_frac=0.35, seed=args.seed)
    rep = oracle.compute(ev, lam=args.lam)
    tr = train_router(train, lam=args.lam, iters=args.iters, seed=args.seed, verbose=True)
    ev_s = tr.evaluate(ev)
    bs, orc = rep.best_single_obj.score, rep.routable_obj.score
    closed = (ev_s.score - bs) / (orc - bs) * 100 if orc > bs else float("nan")
    print(f"\neval: {ev_s}")
    print(f"best_single={bs:+.4f}  routable_ceiling={orc:+.4f}  realized={ev_s.score:+.4f}")
    print(f"gap closed: {closed:.1f}% of the achievable (routable) gap")
    print(f"accuracy vs best single: {(ev_s.accuracy - rep.best_single_acc)*100:+.2f} points")


def cmd_simulate(args) -> None:
    from .simulate import run_competition
    master = generate(SynthConfig(q=args.q, specialization=args.specialization, seed=args.seed))
    rep = run_competition(master, lam=args.lam, epochs=args.epochs, seed=args.seed)
    print(rep.summary())


def cmd_demo(args) -> None:
    from .demo import run
    run()


def cmd_measure(args) -> None:
    """Run the oracle-gap gate on a REAL pool from RouterBench (offline)."""
    import os

    import numpy as np

    from . import realdata as rd
    if not os.path.exists(args.routerbench):
        print(f"missing {args.routerbench}. Fetch once with:\n"
              "  python -c \"from huggingface_hub import hf_hub_download as d; "
              "d('withmartian/routerbench','routerbench_0shot.pkl',repo_type='dataset',local_dir='data')\"")
        return
    df = rd.load_routerbench(args.routerbench)
    pool = rd.CURATED_POOL if args.pool == "curated" else rd.ALL_MODELS
    if args.embeddings and os.path.exists(args.embeddings):
        emb = np.load(args.embeddings)
        enc = f"semantic({emb.shape[1]}d)"
    else:
        domains, names = rd._factorize(df["eval_name"].to_numpy())
        emb = rd.domain_onehot_embeddings(df, domains, len(names))
        enc = "domain-only"
    cache = rd.to_cache(df, pool, emb)
    print(f"[real pool: {len(pool)} models, {cache.Q} questions, encoder={enc}]")
    print(oracle.compute(cache, lam=args.lam, n_groups=args.n_groups).summary())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thirtyspokes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--cache", help="path to a saved .npz cache (else synthesize)")
        sp.add_argument("--synth", action="store_true", help="force synthetic cache")
        sp.add_argument("--q", type=int, default=8000, help="synthetic question count")
        sp.add_argument("--specialization", type=float, default=0.85,
                        help="pool complementarity knob (0=redundant,1=complementary)")
        sp.add_argument("--lam", type=float, default=0.3, help="cost weight lambda")
        sp.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("gen-cache", help="generate + save a synthetic outcome cache")
    g.add_argument("--out", required=True)
    g.add_argument("--q", type=int, default=8000)
    g.add_argument("--specialization", type=float, default=0.85)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gen_cache)

    o = sub.add_parser("oracle-gap", help="measure the oracle gap (admission gate)")
    common(o)
    o.add_argument("--cascade", action="store_true", help="score the v1.5 cascade action space")
    o.add_argument("--threshold", type=float, default=5.0, help="admission threshold (accuracy points)")
    o.set_defaults(func=cmd_oracle_gap)

    t = sub.add_parser("train", help="train a router and report gap closure")
    common(t)
    t.add_argument("--cascade", action="store_true")
    t.add_argument("--iters", type=int, default=220)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("simulate", help="run the adversarial competition")
    common(s)
    s.add_argument("--epochs", type=int, default=24)
    s.set_defaults(func=cmd_simulate)

    d = sub.add_parser("demo", help="run the full Phase-0 pipeline")
    d.set_defaults(func=cmd_demo)

    m = sub.add_parser("measure", help="oracle-gap gate on a REAL RouterBench pool")
    m.add_argument("--routerbench", default="data/routerbench_0shot.pkl")
    m.add_argument("--embeddings", default="data/emb_minilm.npy",
                   help="optional .npy semantic embeddings; falls back to domain-only")
    m.add_argument("--pool", choices=["curated", "all"], default="curated")
    m.add_argument("--lam", type=float, default=0.3)
    m.add_argument("--n-groups", type=int, default=96, dest="n_groups")
    m.set_defaults(func=cmd_measure)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
