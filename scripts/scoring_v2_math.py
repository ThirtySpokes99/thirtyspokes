"""Reproduces every number in docs/DESIGN.md (§1 discrimination, §5 time-to-crown).

No API, no network — pure arithmetic on the Wilson lower bound + the EWMA accumulator.
    python scripts/scoring_v2_math.py
"""
import math

Z = 1.645          # 95% one-sided
P = 0.75           # representative miner accuracy
H = 200            # half-life, epochs
N_EP = 8           # scored tasks per benchmark per epoch
EPOCH_MIN = 20     # 100 blocks x 12s
USD_PER_Q = 0.0017 # measured gpt-4o rate
TASKS_PER_EPOCH = 32   # 8/bench x 4 benches


def wilson_lcb(p: float, n: float, z: float = Z) -> float:
    if n <= 0:
        return 0.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d


def main() -> None:
    lam = 0.5 ** (1 / H)
    n_sat = N_EP / (1 - lam)
    n_at = lambda t: n_sat * (1 - lam ** t)  # noqa: E731
    gap = lambda n: P - wilson_lcb(P, n)     # noqa: E731

    print(f"lambda={lam:.6f}  n_sat={n_sat:.0f}/bench\n")
    print("§1  discrimination: n to resolve delta=0.10")
    print(f"  one-sided z={Z}: n = {2*Z*Z*P*(1-P)/0.01:.0f}/bench   "
          f"two-sided z=1.96: n = {2*1.96**2*P*(1-P)/0.01:.0f}/bench\n")

    print("§5  challenger age -> pooled n -> LCB penalty")
    for t in (1, 12, 72, 200):
        n = n_at(t)
        print(f"  t={t:3d} ep ({t*EPOCH_MIN/60:5.1f} h)  n={n:7.0f}  penalty={gap(n):.3f}")
    print(f"  saturated            n={n_sat:7.0f}  penalty={gap(n_sat):.3f}\n")

    print("§5  time-to-crown vs true edge (mature king at lcb = acc - %.3f)" % gap(n_sat))
    for delta in (0.05, 0.02, 0.005):
        target = delta + gap(n_sat)
        t = next(t for t in range(1, 10_000) if gap(n_at(t)) < target)
        hrs = t * EPOCH_MIN / 60
        print(f"  delta={delta:.3f} -> {t:4d} ep ({hrs:5.1f} h = {hrs/24:.2f} d)  "
              f"challenge cost ~${t * TASKS_PER_EPOCH * USD_PER_Q:.2f}")

    # --- §5.1: choosing n_per_bench.  n_sat ~ 1.443*n_ep*H ; n(t) ~ n_ep*t for t<<H ---
    B, CHEAP_Q = 4, 0.0000075
    print(f"\n§5.1  n_ep buys SPEED (t_crown ~ 1/n_ep); H buys the CEILING.  suite={B} benchmarks")
    print(f"  {'n/bench':>7} {'q/ep':>5} {'n_sat':>6} {'king':>6} {'D=.05':>11} {'D=.02':>11} "
          f"{'$/day strong':>13} {'$/day cheap':>12}")
    for n_ep in (4, 8, 16, 32):
        ns = n_ep / (1 - lam)
        nt = lambda t: ns * (1 - lam ** t)  # noqa: E731
        kp = gap(ns)
        ts = {d: next(t for t in range(1, 20_000) if gap(nt(t)) < d + kp) for d in (0.05, 0.02)}
        q = n_ep * B
        print(f"  {n_ep:>7} {q:>5} {ns:>6.0f} {kp:>6.3f} "
              f"{ts[0.05]:>4}ep/{ts[0.05]*EPOCH_MIN/60:>4.1f}h {ts[0.02]:>4}ep/{ts[0.02]*EPOCH_MIN/60:>4.1f}h "
              f"{q*72*USD_PER_Q:>13.2f} {q*72*CHEAP_Q:>12.3f}")

    # --- §5.1 constraint 1: the probe is on the VALIDATOR's bill and is sized by n_per_bench ---
    print("\n§5.1  validator probe bill (100 miners) — why n_per_bench must decouple from probe_n")
    for n_ep in (8, 32):
        for k in (1, 6):
            pq = n_ep * B
            print(f"  n/bench={n_ep:<3} probe_every_k={k}: {pq:>3} q/miner -> "
                  f"${pq*100*(72/k)*USD_PER_Q:>8.2f}/day/validator")


if __name__ == "__main__":
    main()
