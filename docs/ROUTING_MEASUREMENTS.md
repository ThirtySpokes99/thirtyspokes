# Is routing worth paying for? — eight measurements

*The decisive negative result for the ThirtySpokes subnet thesis. Every number here is reproducible
from the scripts named at the bottom; verdicts were pre-committed in each script's docstring before
the run, so the interpretation is not fitted to the outcome.*

## Verdict

**No.** Across three action spaces, two learner classes and five feature regimes, a trained router
captures **1–12%** of a routable band that is consistently real and large (0.12–0.40). The **median
captured fraction across benchmarks is −1.6%** — on most traffic a trained router generalises *worse*
than simply always calling the best single model. Differences between independently-trained miners sit
at or below the reign's incumbency margin almost everywhere, so a subnet built on this ossifies on
whoever commits first.

The mechanism is not at fault. It is hardware-proven end to end: a reproducible measured image, real
Intel TDX attestation, on-chain governance, and a live miner scored by an enforcing validator. The
problem is that **the thing it pays miners to learn does not appear to be learnable at a price a
router can afford.**

## The measurements

| # | what | surface / learner | band | captured | seed spread |
|---|---|---|---|---|---|
| 1 | GSM8K (former live suite) | oracle gap only | 0.019 | — | — |
| 2 | LiveCodeBench (live suite), 6-model pool | oracle gap only | 0.0401 | — | — |
| 3 | LiveCodeBench, pinned 2-model pool | oracle gap only | **0.0000** | — | — |
| 4 | RouterBench pick-one | capped head, MiniLM | 0.1214 | **1.1%** | 0.0052 |
| 5 | Cascade, 6 verifier qualities | capped head, MiniLM | 0.0014–0.1214 | **≤9%** | 0.0010–0.0080 |
| 6 | Verification | capped head, MiniLM | 0.1255 | **2.5%** | 0.0036 |
| 7 | Verification | GBM, 778 features | 0.1325 | **5.0%** | — |
| 8 | Verification + cross-model agreement | GBM, 781 features | 0.1325 | **7.1%** | — |
| 9 | Per-benchmark learnability, 11-model pool | capped head, MiniLM | 0.117–0.396 | **median −1.6%, max 17.5%** | 0.000–0.063 |

Reign incumbency margin for comparison: **eps 0.002 → 0.02**.

## Two mechanisms, not one bad dataset

**1. The routable signal is luck, and luck is not in the prompt.** Where models differ is largely
*which questions they happen to get right*, which is close to unpredictable from prompt text. The
per-ask oracle exploits exactly that hidden variation — on RouterBench it reaches **higher accuracy
than gpt-4 at one-tenth the cost** (0.9371 @ 0.0062 vs 0.8450 @ 0.0648). A capped head reaches
**1.1%** of that, and a gradient-boosted model on rich features was previously measured barely beating
it (`encoder_ceiling_experiment.py`). The gap is not encoder capacity.

**2. Cascade errors are asymmetric, which strangles verification.** A false *reject* only wastes
money; a false *accept* banks a wrong answer irreversibly. So a verifier must run at low false-positive
rate — and there, recall collapses and the cascade degenerates into "always pay for the strong model":

| threshold | tpr | fpr | cascade | captured |
|---|---|---|---|---|
| 0.50 | 0.889 | 0.307 | +0.7337 | **−51.6%** (worse than doing nothing) |
| 0.80 | 0.679 | 0.121 | +0.7987 | −2.6% |
| 0.90 | 0.502 | 0.055 | +0.8116 | **+7.1%** (best) |

Capturing the band needs roughly **tpr 0.9 @ fpr 0.05 ⇒ AUC ≈ 0.97**. The best cheap verifier measured
**0.88**, and that number is *optimistic*: cross-model agreement — the strongest signal found, worth
+0.09 AUC on its own — requires paying for a second opinion that this accounting did not charge.

The underlying reason is structural: **verifying is about as hard as answering.** A verifier reliable
enough to catch a strong model's errors must be about as capable as that model — at which point you
have paid for it and saved nothing.

**Corollary — a perfect verifier makes routing redundant.** With an oracle verifier, simply always
entering the ladder at the cheapest rung scores **+0.9305** against a per-ask oracle of **+0.9319**.
The band left for a routing model is **0.0014**. The value lives in the harness, not the miner.

## The final measurement: specialisation, tested and refuted

The one surviving hypothesis was that routing fails on *difficulty* (unpredictable from a prompt) but
would work on *domain* (trivially predictable). Testing it needs a combination no earlier measurement
had: mixed-domain traffic **and** a pool containing genuine specialists. Three-way control:

| configuration | captured |
|---|---|
| **A** mixed-domain + specialist pool (11 models incl. code-llama) | **13.3%** |
| **B** mixed-domain + ladder pool (5 curated, no specialists) | **15.6%** |
| **C** single-domain + specialist pool (median of 5 domains) | **2.6%** |

**Half the hypothesis was right.** Domain routing *is* learnable — mixed-domain traffic captures
13–16% against 2.6% single-domain, because a router can read the domain off the prompt where it
cannot read luck.

**It changes nothing, because the pool has a dominator.** Per-domain accuracy over the 11-model pool:

| model | code | law | commonsense | math | knowledge |
|---|---|---|---|---|---|
| **gpt-4-1106-preview** | **0.686** | **0.675** | **0.837** | **0.956** | **0.910** |
| code-llama-instruct-34b | 0.518 | 0.004 | 0.228 | 0.476 | 0.007 |
| Yi-34B-Chat | 0.386 | 0.484 | 0.730 | 0.680 | 0.796 |

**Distinct per-domain winners: 1 of 5.** The nominal specialist is *worse at its own specialty* than
the generalist (0.518 vs 0.686) and useless elsewhere — narrow and strictly worse, not orthogonal.
Its wins are a subset of gpt-4's. Adding specialists to the pool made routing slightly **worse**
(13.3% vs 15.6%): extra actions that are never correct are pure liability.

**So the requirement, stated exactly:**

> Routing needs **a model that beats the frontier model at something, cheaply** — genuinely
> orthogonal competence, not a cheaper or narrower model.

No pool measured in this project has one. Not RouterBench's 2024 models here, and not the 2026
frontier models in the specialist-ceiling run. Two model generations, same structure.

## What this does *not* claim

- **Specialisation is untested at power.** The most specialisation-bearing benchmark (`chinese_zodiac`)
  ranked **first** at 17.5% captured — the predicted direction — but on n=118 test rows. All
  `chinese_*` benchmarks together are ~500 rows. RouterBench cannot settle whether routing over
  genuine language/domain *specialists* (as opposed to capability tiers) works. That would need a
  purpose-built pool and dataset.
- **One benchmark did clear the bar.** `hellaswag` (n=3013) captured 11.9% with seed spread **0.027**,
  above eps₀ — the only genuinely competitive spread observed. Its captured value (+0.027) is however
  about the same size as its spread, so skill and training noise are not separable there.
- These are 2024-era pools (RouterBench) and a 2-model live pool. A pool with genuinely non-nested
  competence could behave differently.

## Nestedness — why code and math are the worst possible traffic

`nestedness` = fraction of ordered model pairs where `solved(a) ⊆ solved(b)`. At 1.0 the pool is a pure
capability ladder and routing has nothing to decide — only difficulty matters, and difficulty is the
unpredictable part.

| traffic | gap | nestedness | sole-correct |
|---|---|---|---|
| LiveCodeBench | 0.0401 | high — 5 of 6 models *never* sole-correct | 8.3%, one model |
| RouterBench `mmlu-professional-law` | 0.487 | 0.100 | 12.4% |
| RouterBench `hellaswag` | 0.403 | 0.000 | 4.6% |
| RouterBench blends (4–35 sources) | 0.35–0.41 | 0.000 | 6–8% |

Blending sources does **not** manufacture headroom (blends ≈ singles); task type and pool diversity do.

## Operational findings worth keeping

- **Copy-dedup is miscalibrated.** Independently-trained *honest* routers reach **0.954** action
  agreement against a 0.95 copy threshold — the current rule would disqualify honest miners. Any
  future dedup must use soft distributions with a threshold above the measured honest baseline.
- **The scalar needs a wide band to work.** At LCB's 0.0647 band the Wilson penalty at n=8 is 0.25 —
  4× the entire measurable range, producing scores near −5 and requiring ~660 pooled asks to mean
  anything. The degeneracy gate (`min_headroom_gap`) exists for exactly this and correctly refused.

## Reproducing

```bash
python scripts/routability.py          # gap / nestedness / sole-correct for any traffic
python scripts/head_spread.py          # (4) pick-one spread + honest-router dedup baseline
python scripts/cascade_spread.py       # (5) cascade across verifier qualities
python scripts/verifier_spread.py      # (6) learned verifier, capped head
python scripts/verifier_ceiling.py     # (7,8) strongest-learner verifier ceiling
python scripts/where_routing_works.py  # (9) per-benchmark learnability
```

Needs `data/routerbench_0shot.pkl` + `data/emb_minilm.npy` (see `realdata.py` for the fetch).

## What survives

1. **The attestation stack** — reproducible measured image, real TDX quotes, artifact binding, on-chain
   governance. It makes *"this exact artifact produced this score"* checkable without re-running it.
   That capability is independent of every economic finding above.
2. **The measurement instruments** — `achievable_gap`, nestedness, the spread/ossification gate. They
   caught three would-be mistakes during this investigation and apply to any future thesis.
3. **The negative result itself**, which is what this document is for.
