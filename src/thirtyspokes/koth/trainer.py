"""Train a routing model — the miner's whole job (`orchestra-koth-train`).

The subnet's competition is one loop: a miner improves a routing head, a validator scores the
decisions it makes. This is the miner's half. It produces the exact artifact `orchestra-koth-miner
--routing-model` consumes, and scores it the way the validator will, so "what I see locally" is
"what I am paid for".

WHAT A MINER IS OPTIMISING. Given a task, the head picks WHERE TO ENTER a cheap->expensive ladder.
The harness then invokes that rung, runs the pinned verifier, escalates while the verifier rejects,
and banks the first accepted answer. So the head is learning two things at once — which asks are
cheap-solvable, and how far to trust a cheap answer — and it is scored on quality minus price, not
on quality.

TRAINING DATA. You need, per task, what every pool model would have produced: `outcomes.json` with
`scores`/`costs`/`verifier_ok` matrices plus the prompts. Two ways to get one:

  * `build` — run the pinned pool over suite tasks yourself. You pay for this; it is the real cost of
    entering, and it is yours to keep and reuse across many training runs.
  * the owner's per-epoch reference records, which carry the same matrices for the scored slices.

Training itself is free and takes seconds: the head is ~6K parameters. Miners compete on insight into
the traffic, not on compute — which is the point of the fixed harness, and also the reason the
competition can ossify if the traffic has nothing to learn (see docs/ROUTING_MEASUREMENTS.md).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import harness as H


def build_outcomes(suite, n_per_bench: int, backend, *, seed: int = 0,
                   params: dict | None = None, progress=print) -> dict:
    """Run every pinned pool model over a slice and record what each produced.

    This is the miner's own reference build. It is deliberately the same shape as the owner's
    per-epoch record (`koth/reference.py`), so a head trained on one can be trained on the other
    without translation.
    """
    from .benchmarks import bench_seed
    pool = H.pool_models()
    p = dict(params or {"max_tokens": 16384, "reasoning": {"effort": "low"}})
    tasks = [(b, t) for b in suite for t in b.sample(n_per_bench, bench_seed(str(seed), 0, b.name))]
    S = np.zeros((len(tasks), len(pool)))
    C = np.zeros((len(tasks), len(pool)))
    V = np.zeros((len(tasks), len(pool)), dtype=bool)
    for i, (bench, t) in enumerate(tasks):
        for k, model in enumerate(pool):
            try:
                text, _in, _out, cost = backend.complete(
                    model, [{"role": "user", "content": t.prompt}], dict(p))
            except Exception as e:      # noqa: BLE001 — a provider failure is DATA about that rung
                progress(f"  {model} failed on {t.task_id}: {type(e).__name__}")
                text, cost = "", 0.0
            S[i, k] = bench.grade(str(text), t.gold)
            C[i, k] = float(cost)
            V[i, k] = H.verifier_ok(str(text), bench)
        if (i + 1) % 10 == 0:
            progress(f"  {i + 1}/{len(tasks)} tasks")
    return {"models": list(pool), "task_ids": [t.task_id for _b, t in tasks],
            "prompts": [t.prompt for _b, t in tasks], "scores": S.tolist(),
            "costs": C.tolist(), "verifier_ok": V.tolist()}


def train_head(outcomes: dict, *, hidden: int = H.DEFAULT_HIDDEN, lam: float = 0.5,
               seed: int = 0, iters: int = 600, holdout: float = 0.3) -> tuple[bytes, dict]:
    """Fit a head to the best ladder entry point per ask, and report HELD-OUT decision quality.

    Held out, always. A head scored on the asks it trained on tells you nothing — measurement 12
    found in-sample capture of 60% next to held-out capture of ~0 on the same data, which is the
    exact difference between a router and a lookup table. The number printed here is the one that
    predicts your emissions.
    """
    from sklearn.neural_network import MLPClassifier

    from .verify import cascade_outcomes
    pool = outcomes["models"]
    order = list(range(len(pool)))                       # outcomes are already cheap -> expensive
    correct, cost = cascade_outcomes(outcomes, order)
    obj = correct - lam * (cost / (float(np.max(cost)) or 1.0))
    y = obj.argmax(axis=1)                               # the label: best entry point per ask

    emb = H.encode(list(outcomes["prompts"]))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1.0 - holdout))
    tr, te = idx[:cut], idx[cut:]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="adam",
                            max_iter=iters, random_state=seed).fit(emb[tr], y[tr])
    # transplant into the harness's own parameterisation: RouterHead is [W1,b1,W2,b2] with tanh,
    # the exact shape of a one-hidden-layer MLP, so the fit loads through `load_head` unchanged.
    w1, w2 = clf.coefs_
    b1, b2 = clf.intercepts_
    W2 = np.zeros((hidden, len(pool)))
    B2 = np.full(len(pool), -10.0)                       # absent classes must not win the argmax
    for j, c in enumerate(clf.classes_):
        W2[:, c], B2[c] = w2[:, j], b2[j]
    theta = np.concatenate([w1.ravel(), b1.ravel(), W2.ravel(), B2.ravel()])

    head, th = H.load_head(H.save_head(theta, hidden), k=len(pool))   # validate the artifact NOW
    picks = head.distribution(th, emb[te]).argmax(axis=1)
    got, best, worst = obj[te, picks], obj[te].max(axis=1), obj[te].min(axis=1)
    span = best - worst
    stats = {
        "n_train": len(tr), "n_holdout": len(te), "params": head.n_params,
        "decision_quality": float(np.mean(np.where(span > 1e-9, (got - worst) / np.maximum(span, 1e-9), 1.0))),
        "regret_vs_oracle": float(np.mean(best - got)),
        # the baseline that matters: always entering at the cheapest rung. A head that cannot beat
        # this is costing its miner money for nothing.
        "always_cheapest": float(obj[te, 0].mean()),
        "router": float(got.mean()),
        "oracle": float(best.mean()),
    }
    return H.save_head(theta, hidden), stats


def main() -> None:  # pragma: no cover — CLI over live/offline seams
    ap = argparse.ArgumentParser(description="train a ThirtySpokes routing model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="run the pinned pool over suite tasks -> outcomes.json (COSTS MONEY)")
    b.add_argument("--n-per-bench", type=int, default=40)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="outcomes.json")

    t = sub.add_parser("train", help="fit a routing head from outcomes.json -> weights.npz")
    t.add_argument("--outcomes", default="outcomes.json")
    t.add_argument("--out", default="weights.npz")
    t.add_argument("--hidden", type=int, default=H.DEFAULT_HIDDEN)
    t.add_argument("--lam", type=float, default=0.5, help="cost weight in the training objective")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--iters", type=int, default=600)
    args = ap.parse_args()

    if args.cmd == "build":
        from ..eval import config
        from ..gateway.gateway import OpenRouterBackend
        from .benchmarks import real_suite
        from .pool import PinnedBackend
        cfg = config.LiveConfig(); cfg.require_key()
        backend = PinnedBackend(
            OpenRouterBackend(cfg.api_key, cfg.base_url, price_fn=config.price_for),
            set(H.pool_models()))
        oc = build_outcomes(real_suite(), args.n_per_bench, backend, seed=args.seed)
        with open(args.out, "w") as f:
            json.dump(oc, f)
        spent = float(np.asarray(oc["costs"]).sum())
        print(f"wrote {args.out}: {len(oc['task_ids'])} tasks x {len(oc['models'])} models, "
              f"${spent:.4f} spent")
        return

    with open(args.outcomes) as f:
        oc = json.load(f)
    weights, stats = train_head(oc, hidden=args.hidden, lam=args.lam, seed=args.seed,
                                iters=args.iters)
    with open(args.out, "wb") as f:
        f.write(weights)
    print(f"wrote {args.out}  ({stats['params']} params, cap {H.PARAM_CAP})")
    print(f"  held out on {stats['n_holdout']} asks (trained on {stats['n_train']})")
    print(f"  decision quality  {stats['decision_quality']:.4f}   (1.0 = oracle routing)")
    print(f"  regret vs oracle  {stats['regret_vs_oracle']:.4f}")
    print(f"  router {stats['router']:+.4f}  vs always-cheapest {stats['always_cheapest']:+.4f}"
          f"  vs oracle {stats['oracle']:+.4f}")
    if stats["router"] <= stats["always_cheapest"]:
        print("  WARNING: your head does not beat always entering at the cheapest rung. Shipping it "
              "would spend more for no gain — retrain, or check that your outcomes have a real "
              "difficulty spread.")


if __name__ == "__main__":
    main()
