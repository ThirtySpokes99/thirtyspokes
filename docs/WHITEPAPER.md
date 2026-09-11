# Thirty Spokes v3 — subnet architecture

*Specification. Replaces the matmul routing head and the precomputed outcome matrix with a
generative routing model — the **Conductor** — scored on real agentic work. Owner decisions are
recorded in §10; open items and known risks in §11. Implemented and tested; the build plan and the
measurement records this document cites (the build plan, the M3a record, the corpus record,
the spend plan, the LCB census, the sandbox record, `CONFIDENTIAL_SERVING.md`) are kept with the
owner's operating records outside this tree.*

Three properties the mechanism has to hold at once:

* **Kind** — a miner's one shot is never spent by our ambiguity. Invalid submissions fail loudly and
  early, the dev kit shows exactly what the validator will compute, and no silent divergence between
  local and scored behaviour is tolerated (this is why `sv::` is refused rather than ignored).
* **Knight** — the crown is defended, not held. The king is re-evaluated on every window's fresh
  slice; its recorded score never protects it, only its artifact does.
* **Spirit** — what is rewarded is the part that generalises. A win must be broad across benchmarks
  (§5.2 conditions 2 and 3), because a win concentrated in a few is a lookup table wearing a
  policy's coat.

---

## 0. What this is, in one paragraph

A miner ships **their own full model weights**, on a pinned architecture. That model is a
*Conductor*: given an agentic task, it reasons internally and then delegates the task to a real
model on OpenRouter, observes the outcome, and either retries with a different model or stops. It
never writes the answer and never touches the worker's input. A single owner-run validator
evaluates it against the sitting king on the same fresh slice of twelve agentic benchmarks, paid
for out of each miner's own OpenRouter credit. It is scored on **quality minus what that quality
cost**, at an exchange rate measured from the pool rather than chosen, and it takes the crown only
if it wins **broadly** — a win that survives dropping its best benchmark *and* that moved more than
half the corpus in its favour. The king earns 0.85 of emissions; the five most recent former kings
share a decaying 0.15.

---

## 1. The artifact

| | |
|---|---|
| **reference model** | Qwen3.6-35B-A3B, pinned revision. Owner publishes repo, revision, `config.json` hash, tokenizer hash, and the full tensor-name/shape inventory. |
| **artifact** | **the miner's own full model weights** — every tensor, not a delta. |
| **architecture** | **FIXED.** Identical to the reference: layer count, hidden size, expert count and routing width, vocabulary, RoPE settings, tensor names and shapes. Weights are free; the shape of the thing is not. |
| **format** | `safetensors` only. `.bin` / `.pt` / `.pth` are **refused**. |
| **dtype** | pinned (bf16). |
| **size** | ~70 GB per hotkey at bf16. |
| **decoding** | greedy, `temperature=0`, pinned `max_tokens`, pinned system prompt. |
| **upload** | owner-provided R2, **one time per hotkey** (see §7). |

**Weights are free, architecture is not.** A miner may reach their weights however they like — full
fine-tune, continued pretraining, distillation, RL, model merging, or LoRA training followed by a
merge back into full weights. That last route matters because it lets a miner without a cluster
compete at all. **What it costs them is not asserted here.** Training compute is bought by the miner,
wherever they choose, and this project has never measured it; a figure we have not run would be a
promise made on a stranger's behalf. What is uploaded is always the merged full model.

What a miner's entry *does* cost, in the parts this mechanism controls: a Bittensor registration
burn at whatever the network charges when they register; the bandwidth and storage of a ~70 GB
upload; and the spend allowance they fund for their own duels (§4), which is the only one this
document can size. Measured over 30 episodes per policy in the M3a run, per-episode spend ran from
$0.001 under always-cheapest to $0.067 under always-strongest, so a ~250-task arm implies roughly
**$0.25 to $17 per window** depending on how expensive the routing is. Small sample, two benchmarks,
and measured on M3a's deliberately frontier-free cheap+mid pool (top rung ~$6/Mtok). Under D4 a miner
routes over the whole admitted catalogue, which carries rungs two orders above that, so the range is
a FLOOR — §5.6's "$250+ of real inference per duel" is the same quantity with frontier rungs in it,
and both are order-of-magnitude rather than quotes.

### 1.1 What full weights cost, and what has to replace it

An adapter over a pinned base enforced three properties *by construction*. Uploading full weights
gives all three back as things that must be checked, and each one is load-bearing:

1. **Architecture identity.** With an adapter, a miner literally could not change the model. Now they
   can upload anything — a distilled 3B, a 70B, a different family. So the validator must verify
   `config.json` field-by-field against the pinned reference, verify the tokenizer is byte-identical,
   and verify every tensor name and shape matches the inventory. Anything unmatched is **refused,
   not coerced**. Without this check "the architecture is fixed" is a sentence in a document rather
   than a property of the system.

2. **No code execution.** `.bin`/`.pt`/`.pth` are pickle archives, and loading one runs arbitrary
   code **on the owner's validator**. `safetensors` is a pure tensor container with no code path, so
   requiring it is what preserves the "no miner code ever executes" property (§2.1) through the new
   upload format. This is not a preference; a pickle upload is remote code execution on the single
   machine that holds the subnet's scoring authority.

   **The pickle refusal is necessary and NOT sufficient, and an earlier draft of this section stopped
   at it.** A model tree carries a second execution path that has nothing to do with the weight
   format: the *custom-code* path. So the admission gate must also refuse

   * **any bundled `*.py`** in the tree — a `modeling_*.py` or `configuration_*.py` riding beside
     perfectly clean `safetensors`;
   * **`auto_map` in `config.json`**, which is the field that names those files;
   * **`trust_remote_code` in `config.json`** (or any tokenizer config), which is the flag that asks
     for them.

   The reason this is not paranoia is that the custom-code path is the *ordinary* way to serve a
   non-upstream architecture, so a serving stack that reaches for it is behaving normally — the
   miner's file executes the moment the validator loads the model, and no pickle was ever involved.
   Refused, never stripped: silently deleting a `modeling_*.py` and loading the rest would serve an
   architecture the miner did not upload, which is a different failure and a worse one. The
   architecture is already pinned field-by-field (point 1), so a tree that *needs* custom code is by
   construction not the reference architecture and has no legitimate reason to ship it.

3. **Comparability.** Two models uploaded at different precisions are not the same experiment —
   quantisation changes quality. The dtype is pinned so that duels compare routing, not numerics.

**No parameter cap, and none is claimed.** The architecture pin fixes the parameter count exactly,
which is a stronger constraint than a cap ever was. Memorisation is not defended here — it is
defended by the scoring rule (§5.3), as it must be: this repo measured a 6.4K matmul head memorising
a random rung table for 1,000 tasks outright, so size has never been a memorisation bound.

---

## 2. The action space

The Conductor emits one action per step, parsed against a fixed grammar. **It is never executed and
never forwarded.**

```
REASON    <free text, token-capped>     internal only — never leaves the Conductor
DELEGATE  <model_id>                    scaffold sends the ORIGINAL task to this model
RETRY     <model_id>                    same, after observing an outcome
STOP                                    submit the best result so far
```

`<model_id>` must appear in the window's committed catalog snapshot (§6.2). Anything else is a
**parse failure**: the step is refused, counted, and the episode continues (three consecutive parse
failures force `STOP`).

### 2.1 The invariant that makes this cheat-proof

> **The only things that reach a worker model are the pinned task text and a validated model ID.**

The scaffold — owner code, not miner code — constructs every worker request from the benchmark's
own task statement. The Conductor cannot rewrite it, prepend to it, or append to it. There is
therefore no channel through which a miner can smuggle a memorised solution into the answer path,
and no need for a detector to look for one. The owner logs every worker request and can assert
byte-equality with the task statement.

This is the property that permits public benchmarks to be used at all. It replaces
`koth/verify.py::grounding_check` (removed with v2, 2026-09-07), which existed to catch answers that did not derive from a pool
response — a check this design makes unnecessary rather than merely stronger.

A model ID is a short catalog slug with no room to carry a solution, and an unrecognised one is
refused rather than forwarded, so widening the pool to all of OpenRouter (§6.2) does not reopen the
channel.

---

## 3. The episode

Pinned scaffold, identical for every miner, identical for king and challenger:

```
for each task in the window slice:
    state ← { task text, tools this benchmark exposes, catalog snapshot,
              history: (model, outcome, cost) per step so far,
              spend so far, step index }
    loop, up to MAX_STEPS:
        action ← Conductor(state)              # greedy decode, miner's full weights
        DELEGATE/RETRY → scaffold calls model_id with the ORIGINAL task, records cost + output
        STOP           → break
    grade the final result with the benchmark's own grader
```

**Sequential, not one-shot.** `RETRY` is what makes this more than pick-one selection: the
Conductor observes that a model failed and escalates or switches. The design rationale is structural
— reacting to an observed failure is strictly more information than choosing blind, and it costs
nothing against the §2.1 invariant because the task text is forwarded verbatim every time.

*(A prior measurement on this repo's single-turn LiveCodeBench matrix found a fixed
verify-and-escalate cascade beating the best single model by +0.0982 with no learning. That is
suggestive of where the headroom sits, but it is a **different regime** — 7 models, one decision per
task, single-turn code generation — and the magnitude does not carry over to twelve agentic
benchmarks over ~300 models. M3a measures it fresh.)*

**The catalog goes in context**, compacted to `id / $in / $out / ctx`, cached once per window since
it is constant across steps. A budget-aware policy has to see prices to be budget-aware.

**Scaffold quality sets the level, not the ranking.** These benchmarks are scaffold-dominated —
this repo measured SWE-bench Pro one-shot at 0/120 while published agentic results run 62–80%. Since
the scaffold is pinned and the duel is paired, it cancels: absolute scores will trail public
leaderboards, and the *difference* between king and challenger remains attributable to routing.

**Every arm in a window faces the same worker draws (§5.2c, D17).** A `DELEGATE`/`RETRY` is bought
live through the gateway the first time any arm in the window asks for that `(task, model,
attempt)` — `attempt` counting retries of the same model within the episode — and replayed from the
window's outcome table for every later arm that asks for the same key. The Conductor never changes
what a worker answers, only which keys get drawn and in what order, and the task text it triggers is
the verbatim statement either way, so the table is a sufficient statistic for scoring any Conductor
on that slice. The episode above is unchanged; what changed is that a draw is one draw.

---

## 4. Money

Each miner registers their own OpenRouter API key beside their submission and sets their own
per-window cap; the validator runs that miner's arm on that key (D18, 2026-09-08). The key travels
**sealed** to the owner's mailbox key (`thirtyspokes-owner key`) and signed by the hotkey — the
mirror of §7's credential envelope — and only the validator opens it (`funding.py`). The window's
allowance is the miner's cap bounded by what the key itself reports it can still spend, read at
arm time; the request body is still the owner's pinned one, so the two arms of a duel send the
same bytes to different accounts.

- **The budget is the miner's, and mutable.** Miners may raise or lower it at any time.
- **It is snapshotted at window open** and frozen for that window's duels. The validator must read
  it at a definite instant or the score is undefined, and a mid-duel raise once spend climbs would
  otherwise be free.
- **Exhaustion zeroes the remainder.** When the allowance runs out, every remaining task in the
  slice scores 0.
- **Task order is derived from the window nonce** and is identical for both arms, so both are
  zeroed on the same tail.
- **The king must stay funded.** It is re-evaluated every window (D13); an unfunded king scores zeroes
  and loses.
- **A replayed delegate costs the miner what a bought one costs.** Under §5.2c an arm is debited the
  row's priced cost for every delegate it makes, hit or miss — the same receipt, the same allowance,
  the same exhaustion — so nothing in this section knows the table exists and "you pay for the calls
  your Conductor makes" stays exactly true. What reaches a provider is the fills alone, on
  whichever key filled them — the owner's for the reference arms and King₀, the miner's for their
  own arm's fills; the reveal publishes both, per arm, as `charged_usd` and `provider_usd`.
- **A key that cannot pay defers the entry, with its shot intact.** A key the provider refuses, an
  account with nothing left, or a cap of zero is found before the arm is opened (one probe), and
  the entry rolls over as unfunded rather than being judged at zero. Mid-arm, a 401/402 on the
  key is the meter's refusal — counted as `unfunded_calls`, never stored as the model's outcome.
- **The owner may still credit a hotkey by hand** (`thirtyspokes-owner credit`); it widens that
  window's allowance and is the owner's grant, not the rail.

Consequence, recorded rather than argued: because allowances are miner-set and unequal, the
leaderboard measures routing skill *and* capital together. That is the owner's decision (§10-D5).

---

## 5. Scoring

### 5.1 The per-arm score

Graded per-task scores from each benchmark's own grader. **Graded, not pass/fail, wherever the
benchmark supports it** — fraction of tests passing, subgoals completed. Binary outcomes carry
maximum variance; partial credit materially cuts the task count needed for the duel to resolve
anything (§5.4).

**Aggregated with equal weight per BENCHMARK, not per task:**

```
quality_b = mean graded score over benchmark b's tasks in the slice
quality   = mean over the 12 benchmarks of quality_b
```

Averaging raw tasks instead would let whichever benchmark contributed the most tasks — or has the
highest variance — silently dominate the total. That is an implicit weighting nobody chose, and it
is directly at odds with rewarding generalisation. Slices should also carry roughly equal task
counts per benchmark so the two notions coincide.

### 5.1b Cost is in the score — the quality/cost exchange rate

A policy that matches the king's quality at a third of the spend is a better router, and the score
has to say so. But the two obvious ways of saying it are both broken:

* **`quality / cost`** collapses to *always call the cheapest model*. The ratio is maximised at the
  bottom of the quality range, so it rewards frugal incompetence.
* **`quality − λ·cost` with a hand-picked λ** buries the most important number in the mechanism —
  the exchange rate between accuracy and dollars — in a constant somebody guessed.

**v3 scores, PER BENCHMARK, `final_b = quality_b − λ_b · (spend_b / C_b)`, with λ MEASURED, not
chosen; the reported score is `final = mean over admitted benchmarks of final_b`.**

`λ_b` is the slope of that benchmark's own cost/quality tradeoff, from the fixed-policy sweep the
admission gate already runs (§6.1), **between the floor and King₀** — always-cheapest and the best
fixed policy by quality (§5.5):

```
λ_b = (quality_King₀,b − quality_cheapest,b) / (relspend_King₀,b − relspend_cheapest,b)
```

**Why King₀ and not always-strongest — measured 2026-09-06, and it decided whether the corpus could
launch.** An earlier draft fitted λ between the two price extremes. That measures whether a stronger
*single model* buys quality, which on the live probe pool it does not: over three replicated draws,
always-strongest minus always-cheapest was +0.010 ± 0.054 on LiveCodeBench and −0.051 ± 0.066 on HLE,
both straddling zero, so the dominator guard was a coin flip and one benchmark or the other was
excluded on every run. The cascade — King₀ — beat the floor by +0.148 ± 0.051 and +0.139 ± 0.071 on
the same tasks. The mechanism's question is whether the best fixed policy beats the floor, not
whether the dearest model does; fitted to that pair λ is +0.09 and +0.08 with both benchmarks
priced. It is also the pair §5.6's power gate already measures the band on (`contrast_for`), so the
price and the gate describe one line. When King₀ *is* always-cheapest there is nothing above the
floor to fit to; the fit then reaches for always-strongest so the dominator guard fires and names
the finding. `reference.fit_pool` is the one definition of this pair; King₀ is chosen on quality
before λ exists, because a choice made under the fitted table would be choosing between two arms the
table was built to make tie.

**λ is per-benchmark, not global.** The exchange rate between accuracy and dollars genuinely differs
across a terminal-ops benchmark and a cyber one — different task lengths, different model spreads.
A single global λ would price them all at one benchmark's rate, so `final_b` values would not be
commensurable and the equal-weight mean in §5.1 would be adding unlike things. Since `quality_b` and
`spend_b` are computed per benchmark anyway, `λ_b` costs nothing extra.

**`C_b` is a PINNED per-task dollar constant, not King₀'s live spend.** An earlier draft normalised
by `spend_King₀`, which would have forced a full King₀ arm every window purely to obtain a
denominator. It is unnecessary: the duel is **paired**, both arms face the same slice, and the delta
is

```
final_ch − final_king  =  (q_ch − q_king) − Σ_b (λ_b / C_b)(spend_ch,b − spend_king,b)
```

so `C_b` only rescales `λ_b`. Any fixed choice is sound, and a pinned constant is stable across
windows where a live baseline would drift. Pin `C_b` from M3a's measured mean cost per task on that
benchmark.

That choice has a precise consequence: **always-cheapest and King₀ score exactly the same.** The
score's iso-lines run parallel to the line joining them, so buying quality at King₀'s own exchange
rate earns nothing — and always-strongest, which pays top-rung prices for quality the cascade gets
cheaper, now sits *below* that line. A policy scores above King₀ if and only if it sits *above* the
line — more quality per dollar than the best fixed policy gives away for free. That is the whole
definition of routing skill, and it is now the objective rather than a commentary on it.

Consequences worth stating:

* Neither degenerate policy is rewarded. Frugal-and-bad and rich-and-good land on the same score.
* **It defuses the allowance asymmetry.** Under §4 a miner sets their own budget, so a richer
  challenger could previously outspend the king into a win. Now spending is priced, and the extra
  quality bought at the pool's own rate scores zero.
* `spend` is per-benchmark too, so `final_b` exists and the leave-one-out rule (§5.2) runs on the
  cost-aware score rather than on quality alone. Breadth is required in the *tradeoff*, not just in
  accuracy.

**Guards. A BENCHMARK WHOSE GUARD FIRES IS EXCLUDED FROM SCORING — it is not scored on pure
quality.** Two conditions fire a guard: `relspend_strongest − relspend_cheapest` below a floor (λ is
undefined — the slope's denominator is not separable from the sweep's own noise), and
`quality_King₀ ≤ quality_cheapest` (the pool buys nothing with money, so λ ≤ 0 — the dominator
condition this project has hit five times). Either way the benchmark leaves the admitted set until λ
is re-derived, and the exclusion is *announced*: a silent change to what miners are optimising would
break the "Kind" property outright.

**An earlier draft fell back to pure quality instead, and that fallback re-opened the exact hole
§5.1b was written to close.** With λ_b clamped to 0 the benchmark is scored on quality with spend
**unpriced**, so this section's own claim — "the extra quality bought at the pool's own rate scores
zero" — is false precisely there. Two such benchmarks are enough to satisfy the breadth rule
(§5.2/§5.3), so a challenger byte-identical to the king on ten benchmarks that simply burns its
allowance on the two flagged ones takes the crown, at thousands of times the king's spend. And the
dominator guard is not an exotic corner: it fires exactly on the condition this repository has
measured five times, so this is the ordinary case rather than the tail.

Exclusion is the right repair rather than a stricter fallback, because of what a fired guard
*means*: it says this benchmark could not be priced from the fixed-policy sweep, which is a
benchmark §6.1's admission gate should have dropped. **A benchmark the admission gate should have
dropped must not price a crown.** λ is pinned per corpus rather than per window (below), so a guard
fires once at derivation time and the admitted set is stable across windows — leave-one-out keeps
running over one set and verdicts stay comparable, which a per-window exclusion would destroy
(§6.3b). If every benchmark fires, there is no corpus to launch on; that is a finding about the
model market (ROUTING_MEASUREMENTS), and the response is to refuse to launch, not to score the whole
arena on unpriced quality.

**λ is pinned per corpus, not recomputed per window.** It is the exchange rate miners train
against; drifting it silently every window would make scores incomparable across windows and move
the target under everyone. Re-derive it when the corpus or catalog changes materially, and announce
the change.

**λ must be RE-DERIVED after M3b, before launch.** M3a fits `λ_b` on the three M2a benchmarks only.
The other nine arrive later with their own cost/quality slopes, and launching on three benchmarks'
worth of exchange rates would misprice three quarters of the corpus. Every admitted benchmark
carries its own `λ_b` and `C_b`, both fixed at M3b and published with the pinned constants.

### 5.2 The duel

Paired: king and challenger on the **same** slice, same scaffold, same catalog. The king is
re-evaluated on every window's fresh slice, so no stale score ever defends the crown — measured once
per slice and shared by that window's duels (§5.2a).

The challenger wins iff **all three** hold:

1. **Aggregate** — paired delta in `final` > `eps` *and* the bootstrap lower bound on the paired
   difference is > 0. (Existing machinery, `duel.duel`; `n_boot` carries over, `N_ASKS` is replaced
   by the task count of §5.4. **`eps` does NOT carry over — see below.**)
2. **Breadth, by leave-one-out** — removing the challenger's single best benchmark, it must still
   win:

   ```
   min over b in benchmarks of [ aggregate delta in final, computed WITHOUT benchmark b ] > 0
   ```

3. **Breadth, by median** — the *typical* benchmark must have moved the challenger's way:

   ```
   median over b in benchmarks of [ per-benchmark delta in final ] > 0
   ```

All three run on `final` (§5.1b), so a challenger must be broadly better *at its price*, not merely
broadly more accurate. Conditions 2 and 3 are gated at zero rather than at `eps`, for the reason
§5.2b gives: `eps` is the anti-noise margin on the headline comparison, and requiring it on every
subset as well would silently multiply the bar and refuse challengers for being merely broad.
Remaining ties break on **earliest commit block**.

**Condition 3 was added after conditions 1–2 were measured and found not to bind** (§5.3 below:
leave-one-out alone crowns a challenger that wins two benchmarks of twelve and is byte-identical on
the other ten, and even one that *loses* ten). It is an **addition, not a replacement** — the two
ask different questions and neither implies the other:

* leave-one-out asks **"is this win carried by one benchmark?"** — it can pass on a two-benchmark
  specialist, because dropping either winner leaves the other;
* the median asks **"did more than half the corpus move the right way?"** — it can pass on a win
  that *is* carried by one benchmark, if the remaining eleven happen to split six-to-five positive.

Both directions are measured, not argued: a challenger at `[+0.315, +0.315, 0×10]` passes LOO at
+0.0286 and fails the median at exactly 0, and one at `[+0.90, +0.01×6, −0.10×5]` passes the median
at +0.0100 and fails LOO at −0.0400.

**Condition 3's counterexample is N-dependent, and at N = 3 it evaporates — read §5.3's 2026-09-01
subsection before relying on it.** Ten identical benchmarks put the median at exactly 0 *because
there are ten of them*. At N = 3 a two-benchmark specialist `[+0.0825, +0.0825, 0]` has a median of
+0.0825, so condition 3 does not refuse the shape it was added for; two of three is a majority. The
corpus is N = 3 (the corpus record §2), so this is the live case and not a hypothetical.

### 5.2a The king's arm is computed ONCE PER WINDOW — a correctness rule, not a saving

Every duel in a window runs on that window's single slice. So the king's performance *on that slice*
is one quantity, and re-running it per challenger does not measure it again — it re-measures the same
thing with fresh noise.

That is a bug, not merely waste. Suppose challengers A and B both duel the king on window 7. Re-run
per duel, A is compared against king-run-1 and B against king-run-2. Run-to-run variance then makes
the two comparisons *inconsistent*: B can be genuinely better than A yet lose while A wins, purely
because king-run-2 came out higher. Batch coronation (§5.2) already picks a champion by comparing
challengers to each other — and that comparison is only meaningful if they faced the identical
baseline.

**So: one king arm per window, reused by every duel in it.** Three consequences, all good:

* **Window-internal consistency.** All challengers are ranked against one number, so the champion
  selection is well defined.
* **The griefing vector closes.** With per-duel re-runs, an attacker spends one allowance to burn one
  of the king's — near 1:1, cheap for a rival who wants the crown. Batched, the king pays once per
  window no matter how many challenge, so the ratio becomes N:1 against the attacker, who is also
  paying a registration burn on top.
* **D6's intent is preserved exactly.** The king is still re-evaluated on fresh data every window;
  its recorded score still never defends it, only its artifact does. What changes is that it is
  measured once per *slice* rather than once per *opponent* — which is what "re-evaluated on this
  data" means in the first place.

### 5.2c Every arm faces the same worker draws — the window's outcome table

§5.2a is the rule one level up: the king's *arm* is one measurement per slice. This is the same rule
one level down, at the unit the money is spent in. A worker's answer to a delegate is a draw from
that model, and a Conductor does not change it — it chooses which `(task, model, attempt)` pairs get
drawn and in what order, and the scaffold sends the verbatim task text every time (§2.1). So a table
of real draws, one per distinct key per window, is a *sufficient statistic* for scoring any Conductor
on that slice: bought live the first time any arm asks, stored with its grade, and replayed for every
later arm that asks for the same key.

**It is a correctness rule before it is a saving, and the measurement that says so is M3a's.** The
cheap arm was replicated three times over the identical 248 tasks; the dominator flag moved between
draws, and always-cheapest duelled against a second draw of itself cleared three of the four verdict
conditions. That is the noise §5.2a removes between king runs, still present between the king and
each challenger. Against one table, identical policies tie exactly — `delta = 0` to the last bit —
which is what "a near-copy of the king never wins" (§5.2b) was always supposed to mean.

What the rule fixes, and what it leaves alone:

* **The unit is one delegate, read loop included**, keyed `(task_id, model_id, attempt)`, where
  `attempt` is the count of prior delegates to that same model on that same task within the episode,
  plus one. A `RETRY` of the same model is attempt 2: a new draw, never the first one handed back —
  otherwise retrying would be free and useless at once.
* **A provider failure is a row** (§8b.3: an outcome about that rung), so every arm sees the same
  dead rung rather than a fresh coin flip on the provider's availability; the reveal publishes the
  window's stored-failure rate so a 429 storm is visible. **A grader or sandbox failure is never a
  row** (§6.3c): the row is written only after the benchmark's own grader returned, so the table can
  hold no zero that was really the owner's dead sandbox. A refusal by the meter — an exhausted
  allowance — is neither: a dead step for that arm, and not a draw of anything.
* **The money is unchanged for the miner.** A hit is debited exactly as a miss (§4). The real
  outlay — the owner's on the reference arms and King₀, each miner's on their own arm's fills (D18)
  — is the fills only, and the reveal's `metering` splits the two per arm: `charged_usd`, which
  must equal `recorded_usd`, and `provider_usd`, which is at most `charged_usd`; with `calls`,
  `hits` and `fills`. The arms run reference → king → challengers so the owner's fills cover the
  cascade's rungs, and a challenger that agrees with the king pays for its agreement without its
  own key being charged for it.
* **One window.** The table lives in the validator's state beside the §8b.5 checkpoints and a
  restart re-buys nothing; nothing is carried to the next window. Cross-window reuse would need a
  row's tokens re-priced against each window's catalog snapshot and a drift gate, and is a
  follow-up, not this. **Never a model**: a row is a real draw bought through the gateway. Predicting
  a worker's outcome instead is what this project has measured to be either wrong or the router
  itself, and it stays refused.
* **Two diagnostics, gating nothing.** Each window re-buys a few of the keys it replayed, live, on
  the owner's allowance (bounded by `RETEST_MAX_KEYS` and `RETEST_MAX_USD`) and publishes how often
  the fresh draw disagreed with the stored one in grade and in cost — the baseline any cross-window
  reuse would have to clear. And the slice is marked at draw time by which of its tasks an earlier
  reveal's traces already showed, and every arm's `quality` and `final` are published on the seen
  and the unseen rows separately, as a memorisation meter (§6.4's honest cost, measured).
* **Publication is D15's, unchanged.** The reveal carries the arms' traces exactly as before; the
  table itself is not published. The §2.1 audit rows are still logged on a hit, one per turn of the
  delegate the row records, so `text == compose(task.prompt, observations)` holds over a replayed
  arm by the same recomputation.

### 5.2b `eps = 0.05` — what that buys and what it costs

**Pinned initial value: `eps = 0.05` on the `final` scale** — five points of benchmark score. It does
*not* inherit the arena's 0.02, which was calibrated against *capture*, a ratio with a different
scale and a different null distribution.

Starting conservative is the right direction of error: a too-large `eps` refuses real challengers,
a too-small one crowns noise, and only the second is irreversible in its effects on the ledger.
Lowering `eps` later loosens the bar and is a safe governance change; raising it mid-flight is not.

**Measured 2026-09-01 (`scripts/breadth_power.py`), and it makes D10's one-way-downward pin
stronger than the argument above: raising `eps` does not buy safety at all, it costs it.** Against an
adversary that sizes its win to whatever bar is set — the only safe assumption, since the bar is
public — `eps` 0.05 → 0.15 moves the worst specialist's crown rate the *wrong* way, 55.5% → 69.8%,
while `uniform_008` collapses 86.5% → 0.2% and `uniform_015` falls to 50%. The mechanism: a
challenger sized at 1.1·`eps` carries an absolute cushion of 0.1·`eps` over a fixed aggregate SE, so
a larger `eps` means a *larger* cushion and condition 1 lets more specialists through, while breadth
reads signs and is scale-free so it gains nothing. **`eps` is a pure true-accept knob in both
directions**; it is not, and cannot be made into, an anti-specialist one.

Two arithmetic consequences, both of which need watching:

**1. At ~250 tasks, `eps = 0.05` behaves like a ~7–8 point bar.** With ~20% of tasks discordant, the
per-task paired difference has SD ≈ 0.45, so at n = 250 the standard error of the mean delta is
≈ 0.028. A challenger genuinely at +0.05 then clears the point estimate about half the time, and its
bootstrap lower bound sits at ≈ 0.003 — barely positive. Both conditions together give it well under
even odds. For +0.05 to be *reliably* winnable the slice needs ≈ 650–700 tasks per arm, not 250.

So either raise the task count toward that, or accept the honest reading: **at 250 tasks, `eps=0.05`
means a challenger must be roughly 7–8 points better to win dependably.** Graded scores (§5.1) shrink
the SD and improve this; the figures above assume binary and are therefore conservative.

**2. `band / eps` is the number of coronations this subnet can ever have.** Once a challenger sits
within `eps` of the achievable ceiling, every later duel lands inside the margin and the crown
ossifies on whoever holds it. The count of distinct coronations the arena can ever host is therefore
bounded by `band / eps`.

**The band is UNKNOWN and cannot be borrowed.** Every headroom figure in this repository was measured
on single-turn LiveCodeBench with a 7-model pool and a 2-rung entry decision. v3 is twelve
multi-step agentic benchmarks over ~300 models with a sequential policy — a different action space,
a different pool, a different task distribution. Nothing about the old magnitude transfers, in either
direction: the band could be several times larger (more models, more decisions per episode, more
diverse tasks) or smaller (agentic scaffolds may wash out model differences). It has to be measured.

That is why M3a reports `band / eps` as a headline number. **If `band / eps < 3`, either `eps` comes
down or the arena has a two-or-three-king lifetime** — a finding about the corpus, not a tuning
problem.

Re-derive `eps` empirically once M4 produces real traces, using the `arena_calibration.py` method:
two equal-quality policies trained on disjoint samples, measure the null spread of the paired delta,
and set `eps` above the noise floor rather than by taste.

### 5.3 Why breadth is the anti-overfitting mechanism — and why leave-one-out *and* a median, not a sign test

With the matmul artifact, held-out re-scoring was free, which is what let held-out scoring *be* the
defence and retire every detector. Live agentic evaluation costs real money, so that lever is gone
and conditions 2 and 3 replace it.

It is aimed at a specific, likely strategy. A miner cannot smuggle answers (§2.1), but can still
learn *"benchmark X → model Y wins"*. Without a breadth requirement that policy is only twelve
facts, every competent miner finds all twelve, they tie, and the crown falls to commit-block
seniority — measurement 16's category recognition with 12 rows instead of 27.

**An earlier draft of this spec gated on "≥7 of 12 benchmarks, sign test p < 0.10". Those two
clauses contradict each other and the rule gated nothing:**

| threshold | one-sided p under H₀ |
|---|---|
| ≥7 of 12 | **0.387** |
| ≥8 of 12 | 0.194 |
| ≥9 of 12 | 0.073 |

Seven of twelve happens by coin flip more than a third of the time. Raising the bar to nine is the
wrong repair: with ~20 tasks per benchmark, per-benchmark win/loss is very noisy, so a genuinely
broad +3-point challenger can land 8–4 by luck and be refused. A sign test over twelve noisy bits is
a weak instrument in *both* directions, because it throws away the graded scores it was computed
from.

Leave-one-out keeps them. It asks the question the failure mode actually poses — *is this win
carried by one benchmark?* — uses the full magnitudes rather than their signs, and demands nothing
about winning everywhere. It is also legible to a miner: **your win must survive losing your best
subject.**

**LEAVE-ONE-OUT ALONE DOES NOT DO THE JOB, AND WHAT IT MISSES IS THE FAILURE MODE NAMED THREE
PARAGRAPHS ABOVE.** Sweep the same aggregate win over k = 1..12 benchmarks and breadth binds **only
at k = 1**. A challenger that beats the king on two of twelve and is byte-identical on the other ten
is crowned: delta +0.0525, `loo_min` +0.0286, published sign test 2 of 12 at p = 0.9968. Dropping
either winner leaves the other, so every subset clears zero. The same shape survives being broadly
*worse*: a challenger that loses 0.02 on ten benchmarks and wins 0.50 on two is also crowned, at
delta +0.0667 and `loo_min` +0.0273. Under D14 the king's weights are public, so **"the king plus two
benchmark-specific overrides" is a buildable submission** — the twelve-facts strategy above with ten
of the facts left out, and the rule this section sells as the defence against it says yes.

Hence §5.2's condition 3: **the median per-benchmark delta must be > 0.** It refuses both cases by
construction — ten identical benchmarks put the median at exactly 0, ten small losses put it at
−0.02 — while the direction that must *not* be refused passes untouched. Legible in the same one
line: **more than half your subjects must have improved.**

**What that bar is, stated plainly, because it sits very close to the one this section rejects.** A
median over twelve benchmarks above zero is approximately *"wins a majority"* — the ≥7-of-12
threshold the table above dismisses. The distinction is real, and it is not a hedge:

* **What §5.3 rejects is demanding STATISTICAL SIGNIFICANCE from twelve noisy bits.** `p < 0.10`
  needs ≥9 of 12. That refuses genuinely broad challengers: at ~21 tasks per benchmark the
  per-benchmark direction is a noisy bit, so a real +3-point challenger lands 8–4 by luck and is
  turned away. The measured archetype duel is exactly that case — the honest cascade beats the
  spendthrift on 8 of 12 at one-sided p = 0.1938, and the draft rule would have refused it.
* **A bare majority is a far weaker bar than significance**, and that same 8–4 duel clears it
  comfortably: its median delta is +0.1716. So the rule that would have refused a real challenger
  still refuses it; the median does not.

Two further reasons the median is not the sign test wearing a new hat:

* **It keeps the magnitudes where they matter.** The sign test collapses each benchmark to a bit;
  the median is an order statistic *of the deltas themselves*, so it reports the middle benchmark's
  actual number. At an even benchmark count it is the mean of the two middle deltas, which means
  `median > 0` is **not exactly ≥7 of 12**: a 6–6 split passes when the smallest win outweighs the
  smallest loss and fails when it does not. At an odd count — a narrowed window (§6.3b) — it *is*
  exactly a strict majority.
* **It gates, and no p-value does.** The threshold is zero on a score, not a significance level on a
  binomial, so nothing here re-introduces the instrument §5.3 threw out.

**Neither breadth condition implies the other**, which is why both are required rather than the
median replacing leave-one-out — §5.2 carries the two measured counterexamples.

The sign test is still computed and **published as a diagnostic**; it just does not gate.

#### What breadth actually costs and buys at N = 3 — measured 2026-09-01, and it changed the recommendation

Everything above this line was measured on a fixture in which a non-winning benchmark's delta is
bit-exactly 0.0 and the corpus is twelve benchmarks. The real corpus is **three** benchmarks at ~83
tasks each (the corpus record §2), where the per-benchmark bootstrap SE is 0.049 — ten times
`BREADTH_FLOOR`. `scripts/breadth_power.py` re-measures the rule there against a generative model
that carries what the fixture did not: pass/fail paired quality (SD 0.447 per task at 20%
discordance), the [0,1] headroom bound, and spend noise on every task including where the two arms
route identically. **No code changed as a result. Three things about the rule did.**

**1. Breadth is nearly free, and the corpus record §3 was wrong to say otherwise.** That
document recommended demoting breadth to a diagnostic on the strength of a false-refusal cost of
"roughly half". Measured against condition 1 alone, breadth refuses **0.0–1.7%** of genuinely broad
challengers at N = 3, n_b = 83, and **26–57%** of one-benchmark specialists (`specialist1_loud` 50.2%
vs condition 1's 82.5%; the predator that is genuinely −0.02 elsewhere, 24.8% vs 57.2%). What refuses
broad challengers is `eps`, exactly as §5.2b's arithmetic says. **Breadth stays a gate.**

**2. The residual hole is one shape, and it is not a sample-size problem.** A challenger that wins
**two of the three** benchmarks and copies the king on the third is crowned at 55.5% against
condition 1's 56.5% — breadth does essentially nothing — and the rate **rises with the slice**: 62.5%
at n_b = 250, 73.8% at 1,000, 99.2% at 16,000. At N = 3 the median of `[x, x, d₃]` is `x` and every
leave-one-out subset keeps a winner, so more tasks only make condition 1 more reliable at letting the
shape through. **Do not buy tasks to make breadth safer.** The lever with the right sign is N: at a
fixed total task budget, N = 5 moves `specialist1_loud` 50.2% → 42.5% and `specialist2` 55.5% → 52.8%
for nothing, which is a second reason to accept the HLE gate.

**3. Four replacement rule families were measured and all four fail. They are recorded in
the corpus record §3 so that nobody re-proposes them**, and the one that came closest is worth
naming here because it looks right and is not: a **calibrated dispersion test** — refuse when the
per-benchmark deltas are more spread out than a bootstrap null re-centred on H₀ *"the true advantage
is uniform, equal to the observed aggregate D"*. It takes the loud one-benchmark specialist from
50.2% to **6.0%** at a 4–9 point cost, and its false-refusal rate on a truly uniform challenger **is
α** rather than being set by the noise. It is still refused, on three counts:

* **tasks are not iid.** R2E-Gym draws many tasks per repository, so advantages arrive in correlated
  blocks and the paired bootstrap resamples them as if they were independent. On that structure a
  genuinely uniform +0.08 challenger is refused **24%** of the time while the dial reads 5% — and the
  design effect is invisible to the very bootstrap an operator would tune against, so the dial is
  uncalibratable rather than merely miscalibrated. Any future rule that reads the bootstrap's own
  spread inherits this and needs a block bootstrap first.
* **D14 makes the profile a design variable.** Two knobs that cost a miner nothing it knows — route
  the untouched benchmarks to a *different* model of equal expected quality, and spend half as much
  there — take a **pure** one-benchmark lookup table from 16.8% back to **50.2%**. They defeat the
  shipped rule at the same time (56.5% against condition 1's 57.2%): this is the spend hole
  `BREADTH_FLOOR` closed at a hundredth of a cent, reopened at ~$0.83 a window because `final_b` is
  still continuous in spend. **`BREADTH_FLOOR` prices the manufactured attack, not the bought one.**
* **it is blind where the shipped rule is not.** `max − median` at N = 3 is `d₍₁₎ − d₍₂₎`, so it
  cannot see the worst benchmark at all: on `[+0.183, +0.183, −0.200]` it crowns 51.5% where the
  shipped rule crowns 19.0%, widening to 70.2% vs 7.5% at n_b = 1,000.

The general lesson, since it will outlive the specific rule: **a breadth rule calibrated against the
observed data's own noise is only as good as the resampling scheme's model of that noise**, and this
corpus violates the iid assumption that scheme is built on. A rule that gates on signs (what ships)
degrades gracefully into "the aggregate decides"; a rule that gates on a p-value degrades into
refusing honest challengers at a rate nobody published.

Supporting measures, all cheap:

- **Slice rotation per window**, drawn after challengers commit (§6.1).
- **Per-benchmark deltas published every window** — `arena.by_category` already does this; the
  "category" becomes the benchmark. Reported as *quality*, never as a capture ratio: averaging
  per-category ratios reported −2.7761 where the pooled truth was −0.2638 (ROUTING_MEASUREMENTS §17).
- **A procedural-task fraction in every slice.** Public benchmarks cannot be fresh; generated tasks
  can. They are the only rows where memorisation is structurally impossible, and they calibrate how
  much of a score is recall.
- **Held-out benchmarks.** Once enough windows exist, withhold 2–3 benchmarks entirely from some
  windows and publish transfer. This is the direct test of the thing the subnet claims to sell.

### 5.4 How many tasks a duel needs

A paired comparison only extracts information from tasks where king and challenger *disagree*.
Under McNemar, assuming a **20% disagreement rate** — an assumption, not a measurement:

| to detect | tasks per arm |
|---|---|
| +10 points | **~225** |
| +5 points | **~670** |

At 25 tasks a challenger must be ~40 points better to be distinguishable from noise. **Target ≥ 250
tasks per arm.** At 2 hours wall clock, ~40-way parallelism and ~15 min/task, that is ~320 — so
wall clock is not the constraint. Cost is.

**The 20% is a placeholder and the table moves with it.** Disagreement rate is a property of the
corpus and the policies, and it has never been measured for sequential policies over agentic
benchmarks. If real duels disagree on 40% of tasks the counts roughly halve; at 10% they roughly
double. M3a should report the observed disagreement rate between fixed policies as a by-product,
which is enough to re-derive this table before any real money is committed to a slice size.

Two multipliers: graded scoring (§5.1), and challenger-futility early stopping — **implemented as an
EXACT ceiling rather than the one-sided quantile test below**. `score.best_possible_final` bounds the
highest `final` an arm can still reach, from two properties of the scoring rule: a graded score is
refused outside [0, 1], so an unrun task adds at most 1.0 to `quality_b`, and spend only grows, so
the most favourable remaining spend is zero. The arm is abandoned when even that ceiling cannot clear
`king + eps` — a bar known before any challenger runs, because §5.2a already measured the king over
the whole slice.

The exactness is the point and not polish. §7 gives a hotkey ONE submission ever, so an abort that
fires on a challenger who would have won spends it forever; an observed-quantile heuristic can do
that and the reference implementation says of itself that it is "not a mathematical bound". A bound
cannot — **provided it agrees with the score about which tasks are in the comparison.** The first
version did not: `_GraderGuard` returns a placeholder 0.0 for a task whose grader died and then
§6.3c drops that task from the denominator, so a ceiling blind to the exclusion set read those
placeholders as scores and aborted arms that went on to win. Both the ceiling and the bar it is
compared against are therefore recomputed against the guard's live set at every batch boundary; the
set only grows, so a ceiling computed with it can only rise, which is the safe direction. Abandoned tasks carry `futile`, distinct from both clocks and from `budget_exhausted`,
because §5.8 makes "could no longer win" a different published finding from "ran out of money".

The original sketch, kept for the record: (a one-sided
sequential test that aborts hopeless challengers before they spend the full allowance —
already implemented in `teutonic/`, `test_early_stopping_is_one_sided_challenger_futility`).

### 5.5 King₀ is the best FIXED policy, not the untrained base model

The tempting baseline is "the reference model, untrained" — what zero miner contribution buys. It is
the wrong bar. **The real alternative to running this subnet is shipping a fixed cascade**, which
needs no miner, no training and no arena. If a trained Conductor cannot beat one, the subnet is
paying for something the owner could deploy for free.

So King₀ is **the best fixed policy found by the admission gate** (§6.1), whichever that turns out to
be. Which policy wins, and by how much, is a v3 measurement — not an inheritance from the
single-turn experiments in `ROUTING_MEASUREMENTS.md (removed with v2, 2026-09-07)`, whose pool, action space and task distribution
all differ. Setting the bar anywhere lower pays for work that is already available at zero cost.

**THE CROWN REVERTS TO KING₀ WHEN THE KING DEREGISTERS.** §4 already zeroes an unfunded king and §5.7
already describes what happens "if King₀ reigns … or the crown reverted", but nothing said *when* a
reversion happens, and the omission is not cosmetic. The reigning hotkey is what §5.7's pension
excludes from the lineage — one hotkey, one share — so a king that has left the metagraph but is
still recorded as reigning stays excluded from its own lineage, and **every surviving pensioner is
paid one rank too high** while the sixth-most-recent drops off the end for nothing.

So: each window, resolve the reigning hotkey against the live metagraph, exactly as §5.7 already
resolves pensioners. If it no longer holds its UID, the crown reverts to King₀ and the deposed king
**stays in the lineage, taking the rank-1 pension slot it was dethroned into** — a slot whose share
then burns, because the hotkey does not resolve. The pension is paid by position in the lineage, not
by position among those who resolved (§5.7), so the ranks behind it do not move up. King₀'s 0.85
burns for as long as the throne is vacant, and the first challenger to clear the verdict against it
takes the crown, exactly as at cold start (§8b.8).

A reversion is **not a coronation** and is never appended to the lineage: nobody won anything, so a
record of a coronation that did not happen would corrupt every later pension rank.

### 5.6 The power gate lost its data source and needs a third arm

Before any artifact is scored, a window must be shown capable of ranking anyone at all. The existing
`arena.power_of` did this by constructing a synthetic challenger that plays the per-task best action
some fraction of the time — **which required the precomputed outcome matrix.** v3 deletes that
matrix (§9), so the gate as written cannot run.

Replacement: **run two fixed reference policies — King₀ and a contrast — once per window, on a
SUBSAMPLE of the slice.** (Which contrast is not free to choose; see the end of this section.)
Their spread is the window's measured signal; if they are
indistinguishable on this data, the window cannot separate policies and no duel should be scored on
it.

Two design points that keep this affordable:

* **A subsample, not a full arm.** Detecting *gross signal failure* — "these two policies are
  indistinguishable here" — needs far less data than ranking two close competitors. ~60 tasks per
  reference arm is enough to catch a dead window, against ~250 for a duel.
* **Once per window, shared by every duel in it**, exactly as the king's arm is (§5.2a). Two
  reference arms at ~60 tasks against `N` duels at ~250 amortises to a small fraction of window
  cost, and it falls further the more challengers a window carries.

This is the check that this repository has already paid to learn the value of: measurement 15 ran an
epoch whose oracle beat best-single by +0.0078 and the mechanism paid out anyway, because nothing
asked whether the epoch was worth running. At $250+ of real inference per duel — the frontier-rung
figure, against §1's $0.25–$17 floor measured on a cheap+mid pool — that mistake is no
longer cheap.

**THE REFERENCE PAIR MUST BE ABLE TO SEPARATE A HEALTHY WINDOW, AND "ALWAYS-STRONGEST" IS NOT A
CONTRAST FOR EVERY KING₀.** The rule is: *the contrast must differ from King₀ in the ACTION IT TAKES,
not merely in its name.* Two policies whose episodes end on the same model, on the same tasks, have
identical quality by construction, a measured spread of exactly zero, and every window then refuses
itself — **the arena never opens**, while the corpus is perfectly healthy and every miner's shot sits
in the queue.

This is not hypothetical; it is the shipped default. §5.5 makes King₀ the best fixed policy, which is
the cascade whenever the cascade wins the sweep, and the substitution rule as first written returns
always-strongest for everything that is not itself always-strongest. But **the cascade's top rung is
the always-strongest model**, and a task's score is the best any step reached (§3), so on a pool
where price tracks capability — the normal case — the cascade solves whatever the strongest model
solves and escalates to it on everything else. Measured on this repo's archetype world: King₀ =
cascade, contrast = always-strongest, spread **+0.0000**, gate refuses. The escape from the
degenerate "compare a policy with itself" case landed straight back in it by a different route.

So the contrast is chosen against King₀'s *ladder*, not against its label: pick a fixed policy whose
rungs are not a superset of King₀'s reachable set — always-cheapest is the natural choice against any
cascade, since it stops at the bottom rung — and, because "not a superset" is an argument about the
pool rather than a proof, **assert the pair separates on the admission sweep before it is pinned**.
A reference pair that measures zero spread on M3a's own data is a mis-chosen pair, and the failure
must surface there, at $40, rather than as an arena that silently never opens.

### 5.7 Emissions — the king and a five-deep pension

Winner-take-all was the v1 rule and it has a real cost: a miner who expects to be dethroned within a
window has almost no reason to enter, and a dethroned king has every reason to grief the new one.
v3 pays a decaying tail to the five most recent former kings.

| position | share |
|---|---|
| **reigning king** | **0.850** |
| ex-king, 1 back | 0.050 |
| ex-king, 2 back | 0.040 |
| ex-king, 3 back | 0.030 |
| ex-king, 4 back | 0.020 |
| ex-king, 5 back | 0.010 |
| | **1.000** |

Rules, each of which closes a way the schedule could be gamed or silently distorted:

* **Ordering is by recency of coronation**, most recent first. Dethroning shifts everyone down one
  and drops the sixth off the end.
* **The reigning king does not also occupy a pension slot.** One hotkey, one share.
* **A hotkey crowned more than once holds one slot only**, at its most recent position. Otherwise
  alternating coronations would let one operator occupy several slots.
* **Unfilled slots BURN. They are never redistributed to the king.** Early on there are no ex-kings,
  and folding their 0.15 into the crown would hand the first winner 100% of emissions — exactly the
  outcome the pension exists to soften. Same for a deregistered ex-king: its share burns.
* **Pay by HOTKEY, resolved to a UID at weight-setting time — never by a remembered UID.** Bittensor
  recycles UIDs: if ex-king `hk_A` held UID 5 and is deregistered, UID 5 is reassigned to some
  unrelated `hk_B`. A schedule that stored `{5: 0.04}` would then pay a stranger a pension they
  never earned, silently and every window. The lineage stores hotkeys; each window it resolves
  hotkey → UID against the live metagraph and **verifies the UID still belongs to that hotkey**.
  Unresolvable → that slot burns.
* **If King₀ reigns** — nobody has ever won, or the crown reverted — its 0.85 burns, while any
  existing pensioners are still paid. They earned it against a live opponent.

The pension is deliberately shallow (0.15 total, 0.05 at the top). It is a soft landing, not an
income: at 0.05 a former king earns about one seventeenth of the crown, so defending is worth
vastly more than having once won.

### 5.8 Publish the budget-exhaustion point

When one arm is zeroed on a 150-task tail (§4) and the other is not, the verdict was decided by
funding rather than routing. That is a legitimate outcome under D5, but it must be visible in the
reveal rather than buried inside an aggregate — otherwise the leaderboard reports skill for a result
that measured capital.

---

## 6. The corpus

### 6.1 Twelve benchmarks

Terminal-Bench 3.0 · DeepSWE v1.1 · Agents' Last Exam · AutomationBench · HLE w/ Tools ·
GDPval-AA v2 · FrontierSWE · ProgramBench · CyberGym · ExploitBench · NL2Repo · SWE-Marathon.

Each must clear, before it can carry weight:

1. **Availability** — the benchmark exists at a pinned version and ships a grader.
2. **Licence to redistribute** via the owner's R2. Non-trivial for HLE and SWE-bench derivatives.
3. **Admission** — a real spread between fixed policies (always-cheap / always-strong / random) on
   a ~50-task probe. A benchmark where the scaffold does ~all the work contributes cost and no
   signal.

Slices are drawn per window, **after** challengers commit, stratified across all twelve.

### 6.2 What the owner commits per window

Not just the slice — the whole window must be self-contained, or the two arms face different worlds:

- the task slice (IDs + graders)
- the **OpenRouter catalog snapshot**: model IDs, prices, context lengths, from one
  `GET /api/v1/models`. Model availability and prices move; without a snapshot the king and
  challenger evaluated minutes apart have different action spaces and the duel is not paired.
- the task order (nonce-derived)
- `MAX_STEPS`, decoding parameters, scaffold version

### 6.3 How it is committed — reuse `holdout_feed`, do not invent

**The chain gives each hotkey exactly one commitment slot and `set_commitment` overwrites it.** The
owner's slot already holds the governance record (`kothgov1|…`); a per-window write would erase it
and every validator would refuse to score. `reference.py` hit this exact wall.

So: **derive the path, commit one chained-manifest root** for the whole schedule
(`holdout_feed.manifest`), publish each window's file to R2 at scoring time, authenticate by the
owner's signature. This also preserves the property a bare per-window hash loses — the owner cannot
choose a slice *after* seeing who committed.

**A window record must BIND its schedule entry to the window being verified.** Membership in the
manifest is not enough on its own: the manifest chains *N* entries, all of them equally authentic, so
a validator that checks only "this entry is in the manifest" accepts window 23's entry inside window
7's file. The draw is a pure function of the entry (§6.3b, D12), so **N entries are N slices**, and
the owner picks among them *after* challengers have committed — which restores precisely the
discretion the chained manifest exists to remove, while every signature and every hash still
verifies.

The check is therefore three things and not two: the entry is in the committed manifest, **the
entry's own window index equals the window being scored**, and the published slice re-derives from
that entry and the record's nonce. Same argument, same shape, as the one already made about the
nonce below — whoever gets to choose an input gets to choose the slice.

---

### 6.3b Slice composition — what every window's draw must guarantee

The random draw (D12) is not unconstrained. Three invariants, each of which the scoring rules
silently depend on:

1. **Every admitted benchmark appears in every slice.** The leave-one-out breadth rule (§5.2)
   removes one benchmark at a time; if a benchmark is absent from a window's slice, LOO runs over a
   different set that window and verdicts stop being comparable. The draw is stratified: it samples
   *within* each benchmark, never *across* them.
2. **Roughly equal task counts per benchmark**, so §5.1's equal-weight aggregation and the natural
   per-task reading coincide. At ~250 tasks over 12 benchmarks that is ~21 each.
3. **A minimum task count per benchmark**, below which `quality_b` is too noisy to carry a
   twelfth of the score. If a benchmark cannot supply its minimum, the window is **built without
   it** and the breadth rule runs over the benchmarks actually present — recorded in the reveal, so
   a narrowed window is visible rather than silent.

**"Twelve" is the target, not a constant.** Admission (§6.1) may drop benchmarks, and rotation may
add them. Everywhere this document says twelve, the rule is *the admitted set* — `N` benchmarks, LOO
over `N`, equal weight `1/N`. A mechanism hard-coded to 12 would break the first time a benchmark
failed its gate.

### 6.3c Task exclusion is symmetric

When a grader or sandbox fails (§8b.3) the task is dropped **from both arms**, never from one. Both
arms run the identical task list, so an asymmetric exclusion would compare different task sets and
hand one arm an easier slice. If the grader fails for either arm, the task leaves the denominator for
both.

### 6.4 The owner publishes decision traces — the entry ramp

To train a Conductor a miner needs `(state, action, outcome, cost)` traces, and generating them means
running agentic benchmarks against many models. That is thousands of dollars *before* a miner can
train anything, let alone compete. Left alone, this subnet would be open only to entrants who can
afford to buy their own training set — and a competition among the well-capitalised is not a
competition about routing.

**So every window publishes the full decision traces of the arms it ran** — the king's and the
reference arm's: rendered state, action taken, worker outcome, cost, per step. It is a by-product of
running the subnet and costs the owner nothing extra. (The window's outcome table, §5.2c, is not
published: what is public is the arms' traces, exactly as here, not the draw behind every key.)

Three things this buys:

* **Entry cost collapses.** A new miner starts from real traces on real benchmarks instead of a
  five-figure data-collection bill.
* **The competition becomes about learning, not about capital.** Everyone sees the same data; the
  differentiation is what you do with it.
* **It compounds.** Each window adds traces, so the public training corpus grows even though task
  *slices* repeat (§M7a).

The honest cost: shared data pushes miners toward similar policies, and convergence means ties, and
ties mean seniority. That pressure is real and it is the same ossification risk the corpus rotation
and breadth rule already exist to counter. It is the right trade — a subnet nobody can afford to
enter has no competition to ossify.

---

## 7. Miner workflow

1. Produce full model weights on the pinned architecture, by any route — full fine-tune, continued
   pretraining, distillation, RL, merging, or LoRA-then-merge. Starting from another miner's
   published weights is permitted (see §11 for the free-riding risk this carries).
2. Register a fresh hotkey. **One shot — a hotkey with any prior entry is refused forever.**
3. Obtain a prefix-scoped, one-time R2 credential: poll the owner's public mailbox
   (`store.thirtyspokes.ai`) for an envelope
   sealed to the hotkey's public key and signed by the owner (`teutonic/access/`, proven pattern;
   one-shot is enforced server-side — *"hotkey submission eligibility is permanently consumed"*).
4. Upload the model tree (~70 GB, safetensors only, multipart); publish `manifest.json` **last** as
   the commit marker.
5. Commit readiness on chain, packing the digests into one commitment
   (`r2ready:v1:<86 base64url chars>` — teutonic's encoding, two SHA-256 in one slot).
6. Register your own OpenRouter key: `thirtyspokes-miner register-key` seals it to the owner's
   mailbox key, signs the record with your hotkey and puts it beside your tree, with the cap one
   window may spend (§4, D18). It uses the same credential as step 3, so it runs before or after
   the commit for as long as that credential lives, and re-running it rotates the key or moves the
   cap; after it has expired and the shot is spent, a key-only credential (`thirtyspokes-owner issue
   --key-only`, scoped to the `openrouter/` sub-prefix the tree's verification ignores) rotates
   the key without ever re-opening the tree. The validator reads it fresh every window and runs your arm on it through its own gateway,
   which still builds the pinned request (§11-1 records what that does and does not close).
7. If crowned: keep the key funded. The king's arm runs once per window, for as long as it reigns.

## 8. Validator workflow (owner-run, single)

1. At window open: fetch the committed window file, verify signature + manifest chain, snapshot
   each miner's allowance.
2. Run the **power gate**. Fail → no duels this window, no shot spent, challengers roll over.
3. For each queued challenger in `(commit_block, hotkey)` order:
   - refuse if the hotkey is spent; download the model tree, verify the commitment binding, and run
     the §1.1 admission checks (architecture, tokenizer, tensor inventory, safetensors, dtype)
   - run king and challenger over the identical slice, in parallel, each on its own key
   - apply the three-condition verdict (§5.2)
4. Highest-capture winner takes the crown; ties break on earliest commit block.
5. Persist history → set weights → publish the reveal (slice, per-benchmark deltas, catalog,
   spend). Persist before setting weights: re-judging after a crash is not idempotent under the
   one-shot rule.

## 8b. Operational rules — the queue, timeouts, and what counts as a failure

A duel spends real money on someone else's machine over two hours, so the ways it can go wrong are
mechanism surface, not ops trivia. Each rule below closes a way the arena could be stalled, drained,
or made unfair.

### 8b.1 The queue, and the deregistration trap

Challengers are evaluated in `(commit_block, hotkey)` order. Two limits:

* **`MAX_DUELS_PER_WINDOW = 8`.** Beyond it, the remainder roll over to the next window with their
  shot **unspent**. A queue is never dropped, only deferred. The eight is derived, not chosen:
  challenger arms are serial on the single owner-run validator (the king and one challenger are
  already resident at ~70 GB each, §11-7) and each is bounded by the two-hour per-duel wall clock, so
  a 24-hour window holds twelve arms — less the king's arm (§5.2a) and the two reference arms (§5.6),
  less the time to pull and load a ~70 GB tree before each challenger.
* **`MAX_QUEUE_DEPTH = 16`, and the trap it exists for.** A challenger who registers, funds a key,
  uploads ~70 GB and then waits in a long queue can be **deregistered before ever being evaluated** —
  they paid a registration burn and got nothing, which is exactly the outcome the "Kind" property
  forbids. Bittensor's immunity period protects a UID earning nothing for a fixed number of blocks
  and no longer, so the cap is expressed as a **drain time** rather than a headcount: at
  `2 × MAX_DUELS_PER_WINDOW` the back of the queue is reached within two windows.

  That makes the owner's chain-side obligation a single sentence: **set `immunity_period` to cover
  three windows** — one to wait for the next window to open, two to drain — and re-derive both
  numbers together if the window length moves. Publish the current wait, and **refuse commits beyond
  the cap** rather than accepting them; taking a registration burn for a slot that will expire before
  it is judged is the outcome the cap exists to prevent, and taking the money first makes it worse
  rather than kinder.

### 8b.2 Timeouts and pathological models

Nothing stops a miner uploading a model that is architecturally legal and behaviourally useless —
one that emits `MAX_TOKENS` of `REASON` at every step, never reaching an action. Without limits that
is a denial-of-service on the validator and a drain on the king's allowance.

| limit | on breach |
|---|---|
| per-Conductor-call token cap | truncate; unparseable output counts as a parse failure |
| per-episode wall clock | abandon the episode, score the task 0 for that arm |
| per-duel wall clock | abandon remaining tasks, score them 0 for that arm |
| model load timeout / OOM | submission is **invalid**, shot spent, king not charged |

The load check runs **before** the king's arm is touched, so a broken challenger never costs the
king anything.

**What the per-duel wall clock bounds, since a limit with no stated scope cannot be enforced.** It
bounds **one arm**, not the pair: §5.4's sizing is per arm (~250 tasks at ~15 min with ~40-way
parallelism ≈ 2 h), and under §5.2a the king's arm is a separate measurement shared by every duel in
the window rather than half of any one duel. The deadline is checked **between tasks**, so an episode
already running may overrun it by up to the per-episode clock; the honest bound on an arm is
therefore duel + episode, and the per-duel clock must exceed the per-episode one or the arm is
abandoned before its first episode could ever time out.

Abandoned tasks score 0 **and say why**: they carry a `stopped_reason` of their own, distinct from
the per-episode clock's. §5.8 requires a verdict decided by something other than routing to be
visible in the reveal rather than buried in an aggregate, and one reason string for both clocks would
render a pathological model and an overrunning validator as the same row. One case deserves naming:
if the **king's** arm hits the clock, §5.2a means the zeroed tail is shared by every duel in that
window, so the owner's own slowness would decide all of them at once — a window whose king arm was
truncated should be treated as the power gate treats a dead window (no duels, no shot spent) rather
than scored.

### 8b.3 Worker failures are data; grader failures are not

A distinction that matters, and getting it backwards would corrupt scores:

* **A worker model erroring, refusing, or timing out is an OUTCOME.** It is information about that
  rung — a Conductor that keeps delegating to a flaky model should be scored for it. Recorded as a
  failed step; the episode continues.
* **A grader or sandbox failing is NOT.** Docker dying is a fact about the validator, not the miner.
  Such a task is **excluded from both arms** and from the denominator. Scoring it zero would inject
  the owner's infrastructure noise into a miner's result.

Exclusions are counted and published: if many tasks drop, the window's power is degraded and the
reader should know.

### 8b.4 Inference determinism on an MoE

Expert routing in an MoE can vary with batch composition, so the "same" model can produce different
outputs across runs. Batch size, sampling parameters and kernel selection are pinned, and greedy
decoding is required. This does not have to be bit-exact — §5.2a removes the case where it would
matter most, by measuring the king once per slice rather than once per opponent.

### 8b.5 Validator restart mid-duel

Resume from the last completed task rather than restarting the duel: a restart would re-spend money
already spent, and under one-shot rules a partially-recorded duel must never be re-judged from
scratch. Per-task results are checkpointed as they land.

### 8b.6 Weight cadence and owner downtime

The window must be no shorter than the chain's `weights_rate_limit`, or the validator will attempt to
set weights faster than the chain accepts and silently keep an older schedule.

**If the validator is down for a window, weights are left untouched** — the previous schedule
persists and the king keeps earning. That is the documented consequence of the owner-liveness
dependency (D1), and it is the right failure: the alternative, decaying weights toward nothing, would
punish a king for the owner's outage.

### 8b.7 Storage retention

Every submission is ~70 GB held in owner R2. At a hundred entrants that is 7 TB (~$105/month at R2
rates); at a thousand it is 70 TB. Unbounded retention is not a plan.

Retention: **the reigning king and the five pensioners are kept indefinitely** — they are the public
artifacts D14 makes derivable — and losing submissions are kept for a published grace window, then
deleted. The manifest hash and the history entry are retained forever regardless, so the *record* of
every submission is permanent even when its weights are not.

### 8b.8 Cold start

Window 1 has no king and no pensioners. King₀ — the best fixed policy from M3a — is a policy in the
owner's scaffold, not a model, so **the owner pays for its arm**, which it does anyway as the
reference arm (§5.6). The first challenger to clear `eps` against it takes the crown; until then
0.85 burns and there are no pensioners to pay.

---

### 8b.9 Deployment — what a validator host actually needs

**Serving is the validator's job, not an operator's (2026-09-08).** The daemon launches vLLM on the
serving host for each tree it must serve — `--serve-host`, `--serve-cards`, `--serve-trees`; the
launch line is `serve.launch_command`, the host receives that script and nothing else — on a free
card or the least recently used one, waits for the endpoint to list the committed name, and re-serves
a Conductor whose card was given away in between. With two cards a window launches about seven
times. `--serve-url` remains for an operator who serves a fixed cast by hand.

Every item below was found by standing the Conductor up on a real 2× RTX PRO 6000 Blackwell box on
2026-09-01, not by reasoning about it. **Six of the eight failures arrive AFTER a successful 65 GiB
model load**, which is what makes them worth writing down: they read as serving bugs, so an operator
looks at vLLM flags and model files rather than at build dependencies.

#### Split the roles — the box that holds the keys must not execute untrusted code

**REVERSED ON 2026-09-01, BY THE OWNER.** Until that date this subsection was headed *"the GPU host
must not grade"* and gave the graders a fleet of their own. The owner's instruction is the opposite —
*"we should run all possible logic on gpu machines. only necessary ones should be on controller
server … to maximize the safety"* — and it is written down as a reversal rather than quietly edited,
because the sentence it replaces was a rule this project relied on.

The reversal is right because the old wording named the wrong invariant. The argument underneath
§8b.9 was never *"the GPU box is special"*. It was: **one machine holds the chain signing key, the R2
credentials and sole authority over emissions, and that machine must not be the one running a program
a miner's model wrote.** The old arrangement satisfied that on paper only. Measured on the
controller on 2026-09-01: with the sandbox seam unset — the default — grading ran on the **root**
daemon as real host root (`/proc/self/uid_map` = `0 0 4294967295`, no user namespace), that daemon
also runs several unrelated projects' production containers, and the preflight gate returned **exit 0
with a success line and no warning** on exactly that configuration. Co-locating the graders with the
Conductor serves the invariant better than the old table did, because the old table's sandbox-fleet
row was never deployed — the graders ran on the Orchestrator.

| host | runs | needs Docker | holds |
|---|---|---|---|
| **Orchestrator** — the controller box | window build, verdicts, weights. **Executes no untrusted code.** | **no** | chain hotkey, R2, gateway creds |
| **Conductor + sandbox fleet** — the GPU machines | serving the miner's model, the agentic tool calls **and the graders** | **yes** | ***no credentials at all*** |

Serving still needs no Docker of its own: weights are parsed by `safetensors` with no code path, and
§2.1 guarantees the model's output never reaches an executor — so the Docker requirement on that row
comes entirely from grading. **The "no credentials at all" cell is now load-bearing for the whole
split**, and nothing in the code can check it: a grading container reads whatever the mounting uid
can read, the bind mount is chosen by the operator rather than by the sandbox, and containment is not
confinement. The moment a hotkey, an R2 secret or a gateway key appears on a grading host, the split
is gone whatever the hostnames say.

**This is not general hygiene, it is specific to this corpus.** CyberGym grades by executing crash
inputs against vulnerable binaries — deliberately running exploit code. On a single-validator subnet
(D1) there is exactly one machine with emissions authority, so it gets one job.

**Mechanism for the sandbox row: `V3_DOCKER_HOST`.** The split is expressed by one environment
variable (`src/thirtyspokes/v3/benchmarks/sandbox.py` — `DOCKER_HOST_ENV`, `docker_host()`,
`_docker()`), and all four docker call sites — `run`, `image_present`, `available`, `_kill` — carry
it as an explicit `--host`, which outranks an ambient `DOCKER_HOST` or `docker context`. **Unset is
refused**: `run` (the only function here that executes anything), `available` (the "may this process
grade" predicate) and `preflight` all raise before a command line is built, because the measured
default above was the catastrophic one. A single box that genuinely means to grade on its own daemon
declares `V3_DOCKER_HOST=local`, which produces byte for byte the command line an unset seam used to
produce. The declaration is a statement of intent and not a reachability check — nothing in the
process can tell the keys-holding daemon from the fleet's, so the seam makes the split *answerable*
from inside the process and never *enforced* by it.

**The constraint the code cannot remove: `docker run -v` is resolved by the DAEMON.** The grading
daemon must see the bind path at the **identical path with identical contents**, so pointing
`V3_DOCKER_HOST` at another machine moves the *daemon* and leaves the *payload* behind. A genuinely
remote fleet therefore needs either `V3_GRADE_DIR` on storage mounted at the same absolute path on
both sides, or a payload-transfer step that does not exist today; a rootless daemon on the same box
additionally needs that payload readable by the daemon's uid, which `workspace()`'s `0700` tempdir is
not unless the grading process runs as that uid. **`preflight` is the gate** —
`python -m thirtyspokes.v3.benchmarks.sandbox`, run **where the grading PROCESS runs and with its
environment**, which is the box with the daemon only when those are the same box — and it checks
the mount **by content** with a freshly minted token rather than by comparing path strings, which is
what the testnet-526 failure would have defeated. It also refuses a daemon whose cgroup is not
delegated, i.e. one that accepts `--memory`, `--pids-limit` and `--cpus` and silently ignores them.
Operator detail and the full red-team result: the sandbox record.

**WHICH PROCESS RUNS WHERE — settled 2026-09-01, and it is not the same question as which daemon.**
`V3_DOCKER_HOST` moves the daemon; nothing moves the process but a deployment. Three arrangements
exist and the obvious one is refused:

* **The validator process on the GPU box, grading on its own daemon.** Simplest, and **NO**. That one
  process signs `set_weights` with the chain hotkey and holds the R2 credentials, the owner wallet
  and the gateway's provider key. Putting it on the sandbox host does not move untrusted execution
  away from the credentials — **it moves the credentials to the untrusted execution**, and the row
  above says *no credentials at all*. The GPU box is rented hardware inside a user namespace whose
  outer root belongs to a third party, with `/root` a `fuse.gocryptfs` mount that root can read while
  it is mounted (the sandbox record §2.5's P3 already concedes the provider sees everything on
  that box). Reversing the credentials
  cell the way the grading cell was reversed is an owner decision and has not been made.
* **The validator process on the Orchestrator, `V3_DOCKER_HOST=ssh://…` at the GPU daemon, and
  `V3_GRADE_DIR` on storage both machines see at the IDENTICAL absolute path.** **This is the
  deployed shape.** The controller keeps the keys and executes nothing a miner's model wrote; every
  container — the graders *and* the read channel — lands on the GPU host.
* **A split process** — an episode runner on the sandbox host, verdicts and weights on the
  Orchestrator — is the destination, and it needs a task/result transport and a gateway endpoint that
  do not exist. Named, not built.

The chosen shape's price is stated rather than smoothed over: **an identical path is not a shared
path.** Measured between these two boxes on 2026-09-01, the transport works (`docker --host
ssh://root@<gpu> version` from the controller answers with the GPU daemon's version) and the payload
does not: with the same absolute path created on both and no shared filesystem underneath, the
daemon created an empty source directory and `preflight` refused by content. The controller has
neither an `sshfs` nor an NFS client today, so **the shape is wired, gated and unrun.** The
declaration is now a required argument (`--sandbox-host`, with `--grade-dir`) rather than an
inherited variable, and `sandbox.check_grading_host()` refuses at launch both an undeclared process
and a remote endpoint with no shared grade directory — the arrangement that provably cannot work —
so that failure costs one line of output instead of a window's spend turned into §8b.3 exclusions.

**The cost this buys, recorded rather than hidden: a grading escape now lands beside the Conductor
(D13).** `validator.run_window` runs the reference arms, then the king's arm once (§5.2a/D13), then
each challenger's arm in a loop, all against one serving process, with grading inside `_arm` episode
by episode. An escape from a grading container on a co-located host lands in the same namespace as
that serving process, and three things follow: **self-inflation mid-arm** — the escape happens while
grading the challenger's *own* answers at task *k*, and tasks *k+1…n* of its own arm plus its §8b.5
checkpoints are still ahead of it; **cross-miner griefing** of every challenger served afterwards;
and **exposure of other challengers' artifacts**, which D14 makes public only for the king. D13
bounds the damage in the one direction that matters — the king's arm is measured first and cannot be
corrupted retroactively — and `--network none` means this requires an actual runc/kernel escape
rather than a socket. It does not make co-location free: **the graders and the served model share a
fate.**

Two mitigations, cheapest first. **Separate uids on the same box** — graders under a rootless daemon
owned by an unprivileged user, `V3_DOCKER_HOST` pointed at its socket — collapses all three, because
an escape landing as that uid cannot ptrace, signal or read the serving process; this is what the
seam was built for. **Separate hosts** — a cheap CPU-only box, since graders need no GPU — restores
the three-way split completely. **Grading only between arms does not work and must not be relied
on**: an escape persists, so a process planted during arm *i* is still there for arm *i+1* unless the
server is restarted between arms, which is a 65.53 GiB reload each time.

**A mechanism benefit that came with the move.** Grading needs disk for images, and the two boxes are
not comparable: measured 2026-09-01, the GPU host has **1.2 T free**, the controller **434 G of 879 G
free** (and stood at **24 G free** earlier the same day, mid-corpus-work). At the measured R2E-Gym
image footprint — median 1.88 GB, mean 2.15 GB, the sandbox record §7.2, which corrects
`r2egym.py`'s docstring by ~3.6× — a 250-task slice is **~540 GB**: comfortable there, arithmetically
impossible here. SWE-bench Verified's recorded *"gradeable: 7 (disk)"* is therefore a fact about the
controller and not about the benchmark. Measured both ways on the images themselves, its recorded
3.59–7.26 GB **is** `docker images` — the correct convention; only the R2E-Gym column was in the
`docker image inspect` one §7.2 shows undercounts by 3.6×. So the count is answerable: at a measured
mean of 5.24 GB, **1.2 T holds ~229 Verified images** against the 83–125 a stratum needs. Disk
therefore excludes neither SWE variant. What still excludes them is the action space, not the box —
see the corpus record §2.

#### The runtime build toolchain, in the order it bites

vLLM **JIT-compiles** kernels for this model's hybrid attention/SSM layers and its sm_120 fused-MoE
path, so the host needs a CUDA *build* toolchain, not merely a CUDA runtime.

1. **PEP 668.** System `pip` is blocked on modern images (`externally-managed-environment`). Use a
   venv — not `--break-system-packages`, which damages an OS Python you do not own.
2. **`nvcc` is not at `/usr/local/cuda`.** It ships *inside the torch cu13 wheels*
   (`…/site-packages/nvidia/cu13/bin`). Set `CUDA_HOME` there.
3. **`ninja` and `cmake`** are pip-installed into the venv, so the venv `bin` must be on `PATH`.
   `gcc`/`g++`/`ld` were present system-wide here; verify on any new host.
4. **The pip CUDA packages install MUTUALLY INCONSISTENT VERSIONS of themselves.** Measured:
   `nvidia-cuda-nvcc` 13.3.73 alongside `nvidia-cuda-runtime` 13.0.96 and `nvidia-cuda-nvrtc`
   13.0.88. CCCL asserts exact compiler/header equality, so the build dies with *"CUDA compiler and
   CUDA toolkit headers are incompatible"*. **Pin nvcc, runtime and nvrtc to the same minor
   version.** Do not reach for `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK`: bypassing a compiler/header
   assertion on the machine holding sole scoring authority invites silent numerical divergence.
5. **The wheels ship only versioned libraries** (`libcudart.so.13`). `ld` wants the unversioned
   `libcudart.so` / `libnvrtc.so` symlinks, and the lib directory must be on `LIBRARY_PATH`.
6. **Do not install `flashinfer-cubin` to dodge the JIT.** It is version-locked to
   `flashinfer-python`, and no matching build existed (cubin topped out at 0.6.13 against vLLM
   0.28.0's pinned 0.6.16.post3) — installing it converts an optional accelerator into a hard
   startup assertion.

#### Three vLLM flags that are not optional

* **`--kernel-config` with the triton MoE and linear backends (and `VLLM_USE_FLASHINFER_SAMPLER=0`).**
  Measured 2026-09-07 on the serving box: vLLM's default flashinfer/CUTLASS fused-MoE path JIT-compiles
  sm_120 kernels with nvcc at startup and dies in ptxas (the split toolchain emits PTX newer than its
  ptxas accepts), and the flashinfer sampler JIT refuses without `CUDA_HOME`. Triton compiles its own
  kernels and needs neither; `serve.launch_command` emits both. This selects kernels — it bypasses no
  correctness assertion.

* **`--max-num-seqs` must sit below the Mamba block ceiling.** For a hybrid attention/SSM model the
  **Mamba state cache, not the KV cache, sets the concurrency ceiling**. Measured: 693,301 KV tokens
  but only **722 Mamba cache blocks**, so vLLM's default `--max-num-seqs 1024` fails startup
  outright with `exceeds available Mamba cache blocks`. Nothing warns in advance.
* **`--enable-prefix-caching`.** M0 exit 7 pins the prompt order `[catalog | task | history]`
  precisely so the constant catalog prefix is prefilled once per window rather than on all ~6,000
  Conductor calls. Without this flag that design pays nothing.

(`--disable-log-requests` was removed in current vLLM and will refuse to start.)

#### First start looks exactly like a hang

The first launch on a new host compiles CUTLASS fused-MoE kernels — **minutes with no log output**,
`ps` showing ~0% CPU because the work is in `cicc` subprocesses. An operator will conclude it has
hung and kill it. Confirm progress by watching object files accumulate in
`~/.cache/flashinfer/*/cached_ops/fused_moe_*/` rather than by watching the log. Subsequent starts
are fast; the cache is worth preserving across container rebuilds.

#### Measured, for sizing

| | |
|---|---|
| weights resident | **65.53 GiB** (36.0B params, bf16) |
| model load | 40 s |
| KV cache | **693,301 tokens** at `--max-model-len 32768` |
| Conductor latency | **median 0.38 s/call** |
| prefix-cache reuse | **2.0×** on a 9,344-token shared prefix |

At ~6,000 Conductor calls per duel that is 0.63 h even at 1-way concurrency, and ~0.04 h at 16-way —
so **the Conductor is not the duel's bottleneck**, confirming the assumption §5.4's sizing rests on.

#### Verify before scoring

`scripts/serve_probe.py` checks the four properties that fail *silently* — the endpoint answers,
the duel completes, and the number is wrong: served identity matches the pinned revision, greedy
decode repeats byte-for-byte, the prefix is genuinely reused, and per-call latency leaves the window
intact. Run it against any endpoint before it scores a duel.

---

## 9. What this deprecates

`fugu.FuguArtifact` · `harness.load_head` · `duel.head_entries` / `fugu_entries` · `matrix.py`'s
precomputed matrix and free held-out scoring · `ARENA_LADDER` and `entry_outcomes` · the pinned
MiniLM encoder and `EMBED_DIM` · the 50K/2M param caps · `sv::` · `verify.grounding_check`.

Kept: one-shot hotkeys · hotkey-salted commit binding · paired duels with eps + LCB · the power
gate · batch coronation with earliest-commit tiebreak · the chained manifest · `arena.by_category`
and `corpus.category_freshness` · King₀ burn.

**ROUTING_MEASUREMENTS §17 no longer constrains the product.** It measured singular-value
adaptation of a 22M frozen encoder feeding a matmul head — the architecture being replaced. The
finding stands as a result about frozen-embedding routers, which is what the literature uses; it is
not a limit on a generative Conductor, and its own limits section says so.

## 10. Owner decisions taken

| | |
|---|---|
| D1 | Single validator, owner-run. Trades away third-party re-derivation of scores; consistent with the accepted owner-liveness dependency. |
| D2 | Agentic tasks; all twelve benchmarks; generalisation over specialisation. |
| D3 | No matmul. Generative Conductor over a pinned base. |
| D4 | Any OpenRouter model — no pinned pool. |
| D5 | Miner-set, mutable allowances. Accepts that ranking reflects capital as well as skill. |
| D6 | 2-hour duels; the king is re-evaluated on fresh data every window and must stay funded. *(Refined by D13: measured once per window's slice rather than once per opponent — same freshness, one consistent baseline.)* |
| D7 | **Base model: Qwen3.6-35B-A3B** (MoE, 3B active, Apache 2.0, 262K ctx). Chosen on active-parameter serving cost — the Conductor runs ~5,000–15,000 times per duel — and on being small enough that a miner without a cluster has a LoRA route at all — the cost of that route is the miner's and is not estimated here. |
| D8 | Objective is `final_b = quality_b − λ_b·(spend_b/C_b)`, averaged equal-weight over admitted benchmarks. **`λ_b` measured per benchmark** from the fixed-policy sweep so the two degenerate policies tie; `C_b` a pinned constant (free under a paired comparison). Breadth enforced by **leave-one-out** on `final`, not a sign test. *(Supersedes drafts in which cost was a tiebreak, and in which λ was global.)* |
| D9 | Emissions: king 0.85; the five most recent ex-kings share **0.15** as 0.05 / 0.04 / 0.03 / 0.02 / 0.01. Unfilled or deregistered slots **burn**, never reflow to the king. |
| D10 | **`eps = 0.05`** initial, on the `final` scale. Conservative on purpose — lowering it later is a safe loosening, raising it is not. Watch `band / eps` (§5.2b): it bounds how many coronations the arena can ever have. |
| D11 | Miners upload **full model weights**, not adapters. Architecture stays pinned and is verified explicitly (§1.1); `safetensors` only, dtype pinned. Costs ~70 GB/hotkey and doubles validator VRAM; buys miners an unconstrained training route. |
| D12 | Tasks are **reusable**; each window draws a random slice after commits; nothing burns. Freshness comes from unpredictability, not scarcity — and §2.1 already removed the value of memorising answers. |
| D13 | **The king's arm is computed once per window**, reused by every duel in it. A correctness rule (all challengers ranked against one baseline) that also collapses the king-griefing ratio from 1:1 to N:1. Preserves D6's intent: the king is still re-measured on fresh data every window. |
| D14 | **Derivatives allowed; the king's weights are public.** `eps = 0.05` means a derivative must add 5–8 real points, which is improvement rather than free-riding; the ex-king pension pays the predecessor for being built upon; and detection was never available (measurement 11). |
| D15 | **The owner publishes decision traces every window.** Without them only well-capitalised miners could afford a training set. Accepts convergence pressure as the price of an open field. |
| D16 | **Breadth takes a third condition: the MEDIAN per-benchmark delta must be > 0** (§5.2, §5.3). Leave-one-out alone was measured and binds only at k = 1 — two winning benchmarks out of twelve, with the other ten byte-identical, is crowned at `loo_min` +0.0286, and so is a challenger that *loses* ten and wins two. Under D14 "the king plus two benchmark-specific overrides" is buildable, so the rule §5.3 sells as the anti-overfitting defence did not defend. **An addition, not a replacement**: LOO says no single benchmark carried the win, the median says more than half moved the right way, and neither implies the other. Stated honestly, the median is close to "wins a majority" — the bar §5.3 rejects a *sign test* for. What §5.3 rejects is demanding **significance** (`p < 0.10` needs ≥9 of 12, which refuses real challengers); a bare majority is far weaker, and the measured 8–4 archetype duel — refused by significance at p = 0.1938 — passes it at a median of +0.1716. |
| D17 | **Every arm in a window is scored against the same sampled worker outcomes** (§5.2c). One live draw per distinct `(task, model, attempt)` per window through the existing gateway, replayed for every later arm; a grader failure is never a row, a provider failure is; the miner is debited hit or miss and the owner pays the fills; the traces D15 publishes are unchanged and the table is not published; one window only, with a test-retest sample and a seen/unseen split as diagnostics. Adopted 2026-09-08 for correctness (identical policies tie exactly) and for cost. A learned simulator of outcomes stays rejected. |
| D18 | **Miners pay on their own OpenRouter key** (§4, §7 step 6). The key is sealed to the owner's mailbox key and signed by the hotkey, kept beside the submission, read fresh every window, probed before the arm opens, and bounded by the miner's own per-window cap and what the key reports it can still spend. Adopted 2026-09-08 over an on-chain TAO rail. Re-opens §11-1's substitution channel as stated there; the reveal publishes the endpoint evidence per arm. Hand credits remain as the owner's grant. |

## 11. Open items and known risks

1. **Model substitution via the miner's key.** OpenRouter supports BYOK and provider preferences,
   so `openai/gpt-5.4` need not mean the same thing across miners — and with an open catalog this
   covers hundreds of models. **Largest remaining hole.** The built `gateway/` closes it: owner
   holds provider credentials, miner pre-funds, the owner makes the call. Its known weakness
   (miner-forgeable metering) does not apply when the owner runs it.

   **Re-opened by D18 (2026-09-08), deliberately.** The owner chose the miner-key rail — each
   arm runs on the key its miner registered — over an on-chain payment rail that would have kept
   this closed, because miners paying the owner in TAO was judged the worse trade. What the gateway
   still does on the miner's key: builds the one pinned body (`PROVIDER_ROUTING`: no fallbacks,
   parameters required, price-sorted), meters every call, and records which endpoint answered
   each fill. What the reveal publishes per arm: `own_key`, the endpoints that served each model,
   and `drift` — the `model@endpoint` pairs a miner-key arm was served by that no arm on the
   owner's key was served by that window. What none of that reaches: BYOK and account-level
   provider settings on the miner's account, which no request or response field exposes. The
   drift list is evidence for the owner to act on, not a gate.
2. ~~**Griefing the king.**~~ **RESOLVED by D13 (§5.2a).** The king's arm is computed once per window
   and reused by every duel in it, so the king pays once per window however many challenge and the
   attacker's ratio falls from ~1:1 to N:1 — while they also pay a registration burn on top. This was
   adopted for **correctness** rather than for cost: every challenger in a window must be ranked
   against the *same* baseline, or champion selection compares against different measurements.
3. ~~**Derivative submissions.**~~ **RESOLVED — derivatives are allowed, and the mechanism already
   prices them.** The original worry was "a copy ties and loses, but a copy + ε wins," which was
   written when `eps` was 0.02 on a *ratio* scale. Under D10 `eps = 0.05` on the accuracy scale, and
   at ~250 tasks the effective bar is nearer 7–8 points (§5.2b). **A derivative must therefore add
   five to eight real points of quality-per-dollar** — that is not free-riding, that is an
   improvement, and it is exactly what the subnet exists to buy.

   **`eps` alone did not in fact price the cheapest derivative, and D16 is the repair.** "The king
   plus two benchmark-specific overrides" clears `eps` at +0.0525 while being byte-identical to the
   king on ten of twelve benchmarks — an improvement bought on two subjects, not five to eight
   points of routing. The median condition refuses it (§5.3), so what remains admitted is a
   derivative that is broadly better, which is what this entry always claimed to be describing.

   Two further reasons this is the right call rather than a tolerated risk:

   * **The pension compensates the predecessor.** A dethroned king draws 0.05 → 0.01 across five
     windows (§5.7). So a miner whose work gets improved upon is paid for having been the platform,
     which is precisely the incentive an open derivative culture needs.
   * **Detection was never available anyway.** Measurement 11 found copy-dedup and honest convergence
     to be the same signal, so a "declared parent hash" rule would be unenforceable — a miner who
     omits the declaration cannot be caught. A rule nobody can enforce is worse than no rule, because
     it punishes only the honest.

   **This requires the king's weights to be public**, which is also what makes "fine-tune another
   miner's model" possible at all. Published weights raise the floor for every entrant and make the
   competition open-source-shaped rather than secret-shaped.
4. **Benchmark verification.** Several of the twelve could not be confirmed at that exact
   name/version. Each needs §6.1's three checks before integration.
5. **Ops load.** ~500 sandboxed agentic executions per duel (both arms), parallel, with Docker and
   per-benchmark graders, on one owner-run machine.
6. ~~**Reasoning token cap.**~~ **RESOLVED — pinned in M0 and enforced in §8b.2**, alongside the
   per-episode and per-duel wall clocks that bound a pathological model.
7. ~~**Serving VRAM.**~~ **RESOLVED — 2× H200 (282 GB), accepted by the owner.** Two 35B models at
   bf16 is ~140 GB, leaving ~140 GB for KV cache; reference arms run sequentially rather than as a
   third resident model. Original note retained for the sizing rationale: A duel serves the king AND the challenger, and
   both are now full 35B models rather than two adapters on one shared base: ~70 GB each at bf16,
   ~140 GB plus KV cache. That is 2× H200 or 4× H100 on the owner's single validator, where the
   adapter design needed one base resident and hot-swapped deltas. Reference-arm windows (§5.6) add
   a third model. **Sizing this is a launch prerequisite, not a detail.**

8. **Storage and transfer** (retention policy now in §8b.7). ~70 GB per hotkey. R2 egress is free, which is
   why it is the right store, but ingest time is hours for a miner on a normal connection and each
   duel pulls a full tree to the validator. Multipart upload is mandatory (teutonic already
   configures 64 MB parts; 4 streams per file, 2 files at a time — the 16 x 16 it shipped with collapsed a 68 GB upload on a ~35 MB/s link, measured 2026-09-08).

---

## 12. Build order

1. **Pin the base model; write the scaffold, grammar and render template.** Everything depends on
   the action space and prompt format being fixed. *Blocking.*
2. **Admission-gate all twelve** on a ~50-task probe with fixed policies. Cheap, and it says which
   benchmarks can carry weight before any machinery is built.
3. **Measure the band** with fixed policies (always-cheap / always-strong / random / fixed cascade).
   If the spread is small, no Conductor can earn anything and the power gate would refuse windows
   anyway. This is the step the measurement record says would have saved most of the previous
   programme.
4. Owner-run gateway (§11-1).
5. R2 / mailbox / one-shot flow, lifted from `teutonic/`.
6. Duel loop, verdict, reveal.
