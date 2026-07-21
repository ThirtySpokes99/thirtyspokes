"""docs/DESIGN.md §5b simulation — does evidence accumulation (v2) beat per-epoch scoring (v1)?

Offline, no API. A synthetic field of miners with known true accuracies; each epoch each validator
draws a noisy n=32 observation. v1 scores that epoch's Wilson-LCB; v2 accumulates (EWMA, half-life H)
and scores the pooled Wilson-LCB. Both feed the real `reign.Reign` (v1 with its eps hysteresis, v2 with
eps=0 since accumulation is the noise defense). Measures: ranking stability / false-dethrone, validator
agreement (two independent validators), time-to-crown vs true edge Δ, and grinder/withholder resistance.

    python scripts/scoring_v2_sim.py
"""
import math

import numpy as np

from thirtyspokes.reign import Reign, Submission

N_PER_EPOCH = 32                 # 8/bench x 4 benches
H = 200                          # EWMA half-life (epochs)
LAM = 0.5 ** (1 / H)
Z = 1.645


def wilson_lcb(k, n, z=Z):
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


class Track:
    """One (version, validator) scoring track over the field."""

    def __init__(self, accumulate, eps0):
        self.acc = accumulate
        self.reign = Reign(eps0=eps0, eps_floor=0.0 if eps0 == 0 else 0.002)
        self.n = {}   # miner -> accumulated n
        self.k = {}   # miner -> accumulated k

    def score(self, miner, k_obs, present):
        if self.acc:
            self.n[miner] = LAM * self.n.get(miner, 0.0) + N_PER_EPOCH   # miss counts as (n, 0) too
            self.k[miner] = LAM * self.k.get(miner, 0.0) + (k_obs if present else 0)
            return wilson_lcb(self.k[miner], self.n[miner])
        return wilson_lcb(k_obs, N_PER_EPOCH) if present else 0.0

    def step(self, subs):
        return self.reign.update(subs)


def draw(rng, acc, best_of=1):
    ks = rng.binomial(N_PER_EPOCH, acc, size=best_of)
    return int(ks.max())


# --- Experiment 1: ranking stability + validator agreement --------------------------------------
def exp1():
    field = {"king": 0.75, "noise": 0.75, "h+.005": 0.755, "h+.02": 0.77, "h+.05": 0.80}
    best = "h+.05"
    epochs = 300
    print("=== Exp 1: ranking stability + validator agreement "
          "(king .75, noise .75, +.005, +.02, +.05; 300 epochs, 2 validators) ===")
    print(f"{'':14}{'v1 (per-epoch)':>18}{'v2 (accumulated)':>18}")
    out = {}
    for ver, accumulate, eps0 in (("v1", False, 0.02), ("v2", True, 0.0)):
        rngs = [np.random.default_rng(11), np.random.default_rng(22)]     # two independent validators
        tracks = [Track(accumulate, eps0) for _ in range(2)]
        crown_changes = 0; miscrown = 0; div = 0.0; prev_crown = None
        for _e in range(epochs):
            weights = []
            for vi in range(2):
                subs = [Submission(m, m, 0, tracks[vi].score(m, draw(rngs[vi], a), True))
                        for m, a in field.items()]
                res = tracks[vi].step(subs)
                weights.append({m: res.weights.get(m, 0.0) for m in field})
            div += sum(abs(weights[0][m] - weights[1][m]) for m in field)        # validator L1 divergence
            cons = {m: 0.5 * (weights[0][m] + weights[1][m]) for m in field}      # Yuma-style mean
            crown = max(cons, key=cons.get)
            if prev_crown is not None and crown != prev_crown:
                crown_changes += 1
            prev_crown = crown
            if crown != best:
                miscrown += 1
        out[ver] = (crown_changes, 100 * miscrown / epochs, div / epochs)
    for label, i in (("crown churn", 0), ("mis-crown %", 1), ("val L1 diverge", 2)):
        fmt = (lambda x: f"{x:.0f}") if i == 0 else ((lambda x: f"{x:.1f}%") if i == 1 else (lambda x: f"{x:.4f}"))
        print(f"{label:14}{fmt(out['v1'][i]):>18}{fmt(out['v2'][i]):>18}")
    return out


# --- Experiment 2: time-to-crown vs true edge ---------------------------------------------------
def exp2():
    print("\n=== Exp 2: time-to-crown vs true edge Δ (MATURE king .75 vs a fresh honest+Δ challenger) ===")
    print("  key point: v1 crowns on a lucky slice (≈0 epochs); v2 requires ACCUMULATED confidence,")
    print("  so time-to-crown is monotone in Δ. (Sim accumulates 32/epoch; §5's 8/bench numbers are ~4× these.)")
    print(f"{'Δ':>6}{'v1 epochs':>14}{'v2 epochs':>14}")
    n_sat = N_PER_EPOCH / (1 - LAM)
    for delta in (0.05, 0.02, 0.005):
        row = [f"{delta:.3f}"]
        for ver, accumulate, eps0 in (("v1", False, 0.02), ("v2", True, 0.0)):
            rng = np.random.default_rng(7)
            tr = Track(accumulate, eps0)
            if accumulate:                       # a long-established incumbent (saturated accumulator)
                tr.n["king"], tr.k["king"] = n_sat, 0.75 * n_sat
            ttc = None
            for e in range(3000):
                ks = tr.score("king", draw(rng, 0.75), True)
                cs = tr.score("chal", draw(rng, 0.75 + delta), True)
                res = tr.step([Submission("king", "king", 0, ks), Submission("chal", "chal", 1, cs)])
                if res.slots and res.slots[0] == "chal" and ttc is None:
                    ttc = e
                    break
            row.append(str(ttc) if ttc is not None else ">3000")
        print(f"{row[0]:>6}{row[1]:>14}{row[2]:>14}")


# --- Experiment 3: grinder + withholder ---------------------------------------------------------
def exp3():
    print("\n=== Exp 3: grinder (best-of-5, true .75) + withholder (publishes only good epochs) "
          "vs an honest .78 king; share of total emissions over 400 epochs ===")
    print(f"{'':14}{'v1':>10}{'v2':>10}")
    epochs = 400
    out = {}
    for ver, accumulate, eps0 in (("v1", False, 0.02), ("v2", True, 0.0)):
        rng = np.random.default_rng(5)
        tr = Track(accumulate, eps0)
        emit = {"king": 0.0, "grinder": 0.0, "withholder": 0.0}
        for _e in range(epochs):
            subs = [Submission("king", "king", 0, tr.score("king", draw(rng, 0.78), True))]
            subs.append(Submission("grinder", "grinder", 0,                    # best-of-5 inflation
                                   tr.score("grinder", draw(rng, 0.75, best_of=5), True)))
            k_w = draw(rng, 0.75)
            publish = k_w / N_PER_EPOCH >= 0.81                                # cherry-pick good epochs
            s_w = tr.score("withholder", k_w, publish)
            if publish:
                subs.append(Submission("withholder", "withholder", 0, s_w))
            res = tr.step(subs)
            for m in emit:
                emit[m] += res.weights.get(m, 0.0)
        tot = sum(emit.values()) or 1.0
        out[ver] = {m: 100 * v / tot for m, v in emit.items()}
    for m in ("king", "grinder", "withholder"):
        print(f"{m:14}{out['v1'][m]:>9.1f}%{out['v2'][m]:>9.1f}%")


def main():
    print(f"lambda={LAM:.6f}  half-life H={H}  n/epoch={N_PER_EPOCH}\n")
    o1 = exp1()
    exp2()
    exp3()
    print("\n=== verdict (docs/DESIGN.md §5b success criteria) ===")
    print(f"  validator divergence  v1 {o1['v1'][2]:.4f} -> v2 {o1['v2'][2]:.4f}  "
          f"({'PASS' if o1['v2'][2] < o1['v1'][2] else 'FAIL'}: v2 validators agree more)")
    print(f"  crown churn (stability) v1 {o1['v1'][0]:.0f} -> v2 {o1['v2'][0]:.0f}  "
          f"({'PASS' if o1['v2'][0] < o1['v1'][0] else 'FAIL'})")
    print(f"  mis-crown rate         v1 {o1['v1'][1]:.1f}% -> v2 {o1['v2'][1]:.1f}%  "
          f"({'PASS' if o1['v2'][1] <= o1['v1'][1] else 'FAIL'})")
    print("  (grinder: accumulation does NOT dilute best-of-N — needs the wall-clock/intra-epoch"
          " commit, per §4.5; withholder: miss=0 penalizes it under v2.)")


if __name__ == "__main__":
    main()
