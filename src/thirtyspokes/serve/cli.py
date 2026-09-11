"""`thirtyspokes-serve` — run the beta routing API.

    export OPENROUTER_API_KEY=sk-or-...

    # the recommended beta pipeline: two tiers + learning loop + data flywheel
    thirtyspokes-serve --tiered --log requests.jsonl --log-features \\
                    --shadow-rate 0.02 --shadow-budget-usd 5

    # once enough traffic is logged, train + promote the escalate predictor:
    thirtyspokes-serve-train --log requests.jsonl --out escalate.npz
    thirtyspokes-serve --tiered --escalate-weights escalate.npz --log requests.jsonl --log-features

    # the two A/B arms it is judged against:
    thirtyspokes-serve --baseline-only               # always the cheap reliable model
    thirtyspokes-serve --weights head.npz            # a subnet miner's 7-way head

The product claim is a comparison — "the pipeline beats calling the cheap reliable model directly" —
so run the arms over the same traffic and compare /stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .app import DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_S, create_app
from .policy import DEFAULT_BASELINE, RoutingPolicy, TieredPolicy


def build(args) -> object:
    from ..eval.config import LiveConfig  # noqa: PLC0415 — optional deps
    from ..gateway.gateway import OpenRouterBackend  # noqa: PLC0415

    cfg = LiveConfig()
    cfg.require_key()

    if args.tiered:
        esc = Path(args.escalate_weights).read_bytes() if args.escalate_weights else None
        policy = TieredPolicy(esc, threshold=args.threshold, explore_rate=args.explore_rate)
    elif args.baseline_only:
        policy = RoutingPolicy(None, baseline=args.baseline)
    elif args.weights:
        policy = RoutingPolicy(Path(args.weights).read_bytes(), baseline=args.baseline,
                               servable=set(args.servable) if args.servable else None)
    else:
        raise SystemExit("pick a mode: --tiered (the pipeline), --weights <head.npz> (a miner "
                         "head), or --baseline-only (the control)")

    backend = OpenRouterBackend(cfg.api_key, base_url=cfg.base_url, timeout=args.timeout)
    shadow_path = args.shadow_log or (str(Path(args.log).with_suffix(".shadow.jsonl"))
                                      if args.log and args.shadow_rate > 0 else None)
    return create_app(policy, backend, log_path=args.log, log_prompts=args.log_prompts,
                      log_features=args.log_features,
                      shadow_rate=args.shadow_rate, shadow_budget_usd=args.shadow_budget_usd,
                      shadow_path=shadow_path,
                      request_timeout=args.timeout, max_tokens=args.max_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="ThirtySpokes beta routing API")
    mode = ap.add_argument_group("mode (pick one)")
    mode.add_argument("--tiered", action="store_true",
                      help="the pipeline: enter cheap, escalate on predicted or observed failure")
    mode.add_argument("--weights", help="serve a subnet miner's 7-way head (the miner A/B arm)")
    mode.add_argument("--baseline-only", action="store_true",
                      help="always call the baseline (the A/B control)")

    tiered = ap.add_argument_group("tiered options")
    tiered.add_argument("--escalate-weights",
                        help="escalate predictor from thirtyspokes-serve-train; omit to run "
                             "cheap-first + escalate-on-failure only")
    tiered.add_argument("--threshold", type=float, default=0.5,
                        help="p(escalate) above which an ask enters at the strong tier")
    tiered.add_argument("--explore-rate", type=float, default=0.05,
                        help="fraction of predicted-escalate asks that enter cheap anyway, so the "
                             "training loop keeps receiving the labels that could correct the "
                             "predictor. 0 stops learning; it does not stop serving")

    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="fallback + control model (default: cheapest realised, 0%% empty)")
    ap.add_argument("--servable", nargs="*",
                    help="restrict which pool models a miner head may serve; default whole pool")

    logs = ap.add_argument_group("logging + flywheel")
    logs.add_argument("--log", help="append one JSON record per request to this file")
    logs.add_argument("--log-prompts", action="store_true",
                      help="log prompt TEXT, not just its hash. Requires consent from whoever "
                           "sent the traffic — logged prompts can end up in a published corpus")
    logs.add_argument("--log-features", action="store_true",
                      help="log the prompt EMBEDDING (needed by thirtyspokes-serve-train). No text, "
                           "but semantic — keep the log private")
    logs.add_argument("--shadow-rate", type=float, default=0.0,
                      help="fraction of requests replayed across the rest of the pool in the "
                           "background — the data flywheel. Costs real money (~$0.25/ask over the "
                           "full pool on code traffic); requires --shadow-budget-usd")
    logs.add_argument("--shadow-budget-usd", type=float, default=0.0,
                      help="hard cap on total shadow spend; when spent, sampling stops and "
                           "serving continues")
    logs.add_argument("--shadow-log", help="shadow output path (default: <log>.shadow.jsonl)")

    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help="per-call wall clock before falling through the ladder")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import uvicorn  # noqa: PLC0415

    uvicorn.run(build(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
