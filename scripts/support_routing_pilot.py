#!/usr/bin/env python
"""Run the support-ticket routing economics gate on a scored JSONL export."""

from __future__ import annotations

import argparse

from thirtyspokes.support_pilot import format_report, load_scored_tickets, run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickets", help="path to the scored-ticket JSONL file")
    parser.add_argument("--quality-target", type=float, required=True,
                        help="partner's minimum acceptable held-out success rate")
    parser.add_argument("--quality-tolerance", type=float, default=0.01,
                        help="allowed accuracy non-inferiority margin (default: 0.01)")
    parser.add_argument("--eval-frac", type=float, default=0.35,
                        help="newest chronological fraction held out (default: 0.35)")
    parser.add_argument("--min-advantage", type=float, default=2.0,
                        help="minimum cost advantage over best simple baseline")
    parser.add_argument("--min-tickets", type=int, default=2_000)
    parser.add_argument("--min-days", type=float, default=28.0)
    parser.add_argument("--overhead-per-ticket", type=float, default=0.0,
                        help="router/TEE/subnet overhead added to routed cost")
    parser.add_argument("--feature-dims", type=int, default=128)
    parser.add_argument("--train-iters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_scored_tickets(args.tickets, feature_dims=args.feature_dims)
    report = run_pilot(
        data,
        target_accuracy=args.quality_target,
        eval_frac=args.eval_frac,
        target_tolerance=args.quality_tolerance,
        min_advantage=args.min_advantage,
        min_tickets=args.min_tickets,
        min_days=args.min_days,
        overhead_per_ticket=args.overhead_per_ticket,
        train_iters=args.train_iters,
        seed=args.seed,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
