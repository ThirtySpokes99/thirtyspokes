# ThirtySpokes — mining guide

You compete to build the **routing model** that gets the highest benchmark quality *per dollar* over
an owner-pinned pool of models. You ship weights; the subnet owns the engine that runs them. **King = the eligible miner with the highest
cost-budgeted quality.** For *how the mechanism works* see [`DESIGN.md`](DESIGN.md); to stand up a
subnet see [`DEPLOYING.md`](DEPLOYING.md).

> **Live on Bittensor MAINNET — netuid 99 (`finney`, the code default).** Governance record **v7**
> (`96855d0f…` — the same digest pinned in the image table below) was published on-chain on
> **2026-08-01** and confirmed live: the deployed runtime measurement matches this repo, the subnet is
> full (256/256 registered), and non-owner miners are earning. Testnet **526** (`--network test`)
> remains available for offline/dev bring-up, and several command examples below still show it —
> **substitute `--netuid 99 --network finney` to mine mainnet.**
>
> *(This caveat has flip-flopped in earlier revisions; the on-chain fact as of 2026-08-01 is netuid 99.
> DEPLOYING.md and VALIDATOR.md still carry the pre-launch "testnet 526" framing and lag this.)*
>
> Validators run **fail-closed**: to earn anything you must run the owner's measured runtime image on a
> confidential VM. A mock TEE or stock CVM earns **nothing**.

## What you build

**One thing: a routing model.** A small set of weights that, given a task, decides *which pool model
should answer it*. You do not write code — the subnet owns the engine, and no miner code runs
anywhere in the system.

```
your task → [frozen encoder, owned by the harness] → embedding
          → [YOUR HEAD: ~6K weights]               → distribution over pool models
          → [harness] calls that model, escalates if the verifier rejects, returns its answer
```

Your artifact is a single `weights.npz` holding `theta` (a flat float vector) and `hidden` (one
integer). It loads through `np.load(allow_pickle=False)` into a strict shape check, capped at
50,000 parameters. The encoder, the head architecture, the ladder, the verifier and the task
sampling are all fixed and bound into the measured image.

**Why this shape.** A routing head *cannot emit an answer* — it emits a choice, and the harness
returns the chosen model's response verbatim. So answer-memorisation is impossible by construction,
and the whole apparatus that used to police it (grounding checks, source scans, confinement of your
code) simply does not apply to you.

**What you are competing on** is one judgement, made per task: *is this cheap-solvable, and how far
should I trust a cheap answer?* The harness enters the ladder where your head says, runs the pinned
verifier, and escalates while the verifier rejects. So a good head learns both which asks are easy
and when a cheap answer should not be believed.

### Train it

Training is free and takes seconds — the head is ~6K parameters. What costs money is the **training
data**: what every pool model would have produced on each task.

**Size the cache before you trust anything it tells you.** The live slice is drawn fresh from ~1000
asks per benchmark each epoch; a cache is a sample of that, and a small one is both noisy AND
systematically easier. Measured on a real 48-ask cache: the head's HELD-OUT accuracy read
1.00/1.00/1.00 while the same head delivered 0.63/0.75/0.75 live — that is not overfitting (held-out
beat in-sample), the cache was just not representative. On that same cache the head's edge over
always-cheapest was 0.0005 against a sampling error of ±0.204: the data could not tell whether the
head helped at all. `train_head` now prints that interval and says so; if it warns, collect more asks
rather than spending a TDX epoch to find out.

```bash
# 1. Run the pinned pool over suite tasks. YOU pay for this; the data is yours and reusable.
#    --n-per-bench 40 gives 120 asks; treat that as a floor, not a target.
uv run orchestra-koth-train build --n-per-bench 40 --out outcomes.json

# 2. Fit a head. Scored HELD OUT, and against the baseline that matters.
uv run orchestra-koth-train train --outcomes outcomes.json --out weights.npz
```

The trainer prints:

```
decision quality  0.71        1.0 = you matched the per-ask oracle
regret vs oracle  0.04
router +0.83  vs always-cheapest +0.79  vs oracle +0.91
```

**Read the middle number first.** `always-cheapest` is what you get by ignoring the task entirely and
always entering at the bottom rung. If your head does not beat it, you are spending more for nothing
and the trainer says so. Beating the oracle is impossible; beating always-cheapest is the whole job.

Everything is scored **held out**, on asks the head never trained on. That is deliberate: a head
fitted to the tasks it will be scored on can reach 60% capture in-sample and ~0% on unseen asks —
a lookup table, not a router — and only the held-out number predicts your emissions.

You are free to ignore this trainer entirely and fit `theta` however you like: a different optimiser,
different features from the same embeddings, a different objective. The format is the contract, not
the method.

## What secures this — the trust boundary (read this)

Your routing head runs inside a confidential VM that samples and runs the benchmark — so the
subnet does **not** trust your machine, it trusts the owner-published **measured runtime image**:

- You boot the owner's published, locked-down image ([`DEPLOYING.md`](DEPLOYING.md) §Measured image)
  on a TDX/SEV-SNP CVM. It loads *your* public `weights.npz` via `np.load(allow_pickle=False)` — never
  pickle, so **no miner code executes anywhere** — hashes exactly those bytes into the attestation,
  runs the fixed harness, and meters your cost. The harness's only outbound channel is the metered
  `call_model`.
- **What you compete on is the head** (fully public); **what is fixed and measured is the runtime** —
  encoder, head architecture, ladder, verifier and task sampling, all bound into the image.
- Validators run **fail-closed**: a modified runtime, a stock CVM, or the mock TEE is **rejected, not
  scored** (`mock_quote_rejected` / `unapproved_runtime`). There is no "trust me" path — the measured
  image is the only way in.

This is *why* a static public benchmark is safe here: a routing head **cannot emit an answer** — it
picks which pinned model answers, and the harness returns that model's response verbatim — so answer
memorisation is impossible by construction (`koth/harness.py`). See [`DESIGN.md`](DESIGN.md) §1 (trust
model) + §6 (why benchmarks are safe).

> The mock TEE exists only for **offline development** (the local simulator). Mainnet validators reject
> its proofs, and the validator daemon refuses to disable its gate on mainnet — so don't tune against it.

## The rules (enforced by the runtime)

- **Pinned pool only.** Your head's action space *is* the owner's allow-list — it cannot choose a
  model outside it, so `UnpinnedModelError` is structurally unreachable for a routing head.
- **You pay your own inference** (your OpenRouter key). Cost is metered from the real bill.

> **Retired for routing heads.** The old free-agent rules — `no_pool_call`, the `ungrounded` grounding
> check, source/weight scans — **do not apply to a routing head** (`koth/harness.py`): a head cannot
> emit an answer, so answering "from your own weights" is impossible by construction, and the harness
> always calls a pinned model and returns its response verbatim. Those checks survive only on the
> legacy `--source`/`--weights` free-agent path, which this subnet no longer competes on.

## What you're scored on

**The ranked benchmark is LiveCodeBench** (`code`). MMLU and GSM8K are also run, at **weight 0** —
they are eligibility floors, not ranking signal: you must clear `f_min` on them, but being brilliant
at them earns you nothing. Each epoch you get 8 code problems drawn unpredictably from a 56-problem
public pool, **stratified by difficulty** (3 easy / 3 medium / 2 hard), plus 8 from each floor.

Math used to carry the ranking weight and no longer does, for a reason worth understanding before you
optimise: on GSM8K every pool model scores 79-97%, so a *perfect* router beats a coin-flip over the
pool by about **+0.019** — less than the noise in an 8-question slice. There was nothing there to win.
On LiveCodeBench that number is **+0.083**. See [`DESIGN.md`](DESIGN.md) §5e.

## How you're scored

**You are not scored on how accurate you are. You are scored on how well you ROUTED.**

Each epoch the owner publishes a *pool reference*: what every pinned model scored, and cost, on the
exact asks you were given. Your score is where you landed between two baselines **evaluated at the
price you actually paid**:

```
headroom = (Q_lcb − zero(your_cost)) / (oracle(your_cost) − zero(your_cost))
```

- **`zero(c)`** — what a *featureless* policy gets at price `c`, by randomly picking pool models.
  **This is the bar.** Score 0.0 and you have demonstrated nothing, however high your accuracy.
- **`oracle(c)`** — the best a perfect per-ask router could do at that price. Score 1.0 and you
  matched it.
- **Below 0.0** means you did worse than picking at random.

The consequence to internalise: **matching quality at a fraction of the price scores well, not
badly.** Calling the strongest model on everything puts you *on* the zero frontier — maximum accuracy,
zero headroom, score ≈ 0. The gain comes from knowing *which* asks need the expensive model.

`Q_lcb` is your accuracy at its **Wilson lower confidence bound**, pooled across every epoch you have
run the same artifact — so a lucky slice cannot buy a crown, and a good agent's score climbs as
evidence accumulates. Re-publishing a changed artifact **resets that evidence**: dethroning costs real
attested epochs. No separate cost penalty is subtracted — cost is already inside both baselines.

If the owner publishes no reference, or the traffic that epoch turns out too saturated to measure
routing on, scoring falls back to absolute quality-minus-a-small-cost-term and the feed says so
(`diagnostics.pool_reference.routable = false`).

**What is measured but never paid for.** Your escalation behaviour, latency, token counts and per-ask
regret are all extracted from your attested trace and published in the standings feed. None of them
is a reward term — every one becomes farmable the moment it pays. Read them to debug your agent, not
to game the scalar.

To take the **crown** you must beat the king's pooled score by an **epsilon incumbency margin** that
decays as its artifact ages — so noise cannot flip the crown, but a genuinely better agent always
gets there. Because the score is pooled Wilson-bounded evidence, that means out-performing it *over
attested epochs*, not winning one lucky slice; and because re-publishing resets your own evidence,
every artifact change is a real bet. Emissions split
**equally** across the king + a chain of up to 4 recent ex-kings (5 slots, ≈20% each when all five are
paid — a dethroned king that keeps competing goes on earning while it decays out of the chain, so
there's no cliff to camp against), with an epsilon incumbency margin protecting the king +
earliest-commit tiebreak.

**A seat pays only while you keep mining.** You are paid for an epoch only if you submitted a valid
proof *in that epoch* — holding the crown once does not vest anything. Miss **3 consecutive epochs**
and you lose the seat entirely and must retake the crown to get back in. A missed epoch or two is
forgiven (real CVM boots are flaky), and while you are the king, being absent costs you the epoch's
pay but not the crown — you keep the title, and challengers still have to clear your epsilon margin,
until your grace runs out.

You are **eligible** only if ALL hold, else you earn nothing that epoch: accumulated **cost per task**
≤ the owner's ceiling; accuracy ≥ `f_min` on **every** benchmark, floors included; ≥1 pool call on
every scored task.

**So: clear the floors, then beat the frontier.** Being accurate is table stakes — the score is what
you added *over what your money could have bought anyway*. (Full detail: [`DESIGN.md`](DESIGN.md)
§5.0 for the scalar, §5e for the suite.)

## Test locally before you spend a cent on-chain

The dev kit runs the validator's own scoring code on your artifact. It has two modes, and the
difference matters:

```bash
# 1. FREE smoke test — synthetic benchmarks + a mock pool, no key, no Docker, no cost.
#    Proves your head loads, routes, and produces graded answers. It CANNOT tell you whether you
#    can route well: the mock pool's difficulty is invented, so its headroom is meaningless.
uv run orchestra-koth-dev --routing-model weights.npz

# 2. REAL — the live suite over the owner-pinned pool. This is what the validator scores.
#    Needs OPENROUTER_API_KEY (you pay). The pool is NOT yours to choose here: it is your head's
#    action space, pinned in the harness, so --pool is ignored on this path.
uv run orchestra-koth-dev --routing-model weights.npz --real
```

The report includes a `decisions` block — which rung your head entered for each task and how far it
escalated. That is your actual output; the accuracy figures beside it are largely the pool's work.
Both print per-benchmark acc/lcb, total_cost, `Q_lcb`, `eligible`, `n_pool_calls` from the same code
path the validator runs. Iterate on (1) until nothing is broken, then on (2) until `eligible: true`
under the budget with a `Q_lcb` you would bet a CVM boot on.

Note what neither mode gives you: your **headroom**. That needs the owner's pool reference for a live
epoch — so a strong `Q_lcb` here says you are accurate, not that you out-routed the frontier. Compare
your cost against calling each pool model on everything: if you are not both cheaper *and* comparably
accurate, your headroom will be near zero on-chain.

## Submit + run

```bash
uv pip install -e ".[chain,eval,tee]"     # tee extra = full DCAP verification on a CVM
export OPENROUTER_API_KEY=...             # you pay your own inference
huggingface-cli login                     # to publish your public bundle
btcli wallet new_coldkey && btcli wallet new_hotkey
# MAINNET (live): netuid 99 is FULL (256/256) — registration recycles TAO and evicts the
# lowest-immunity neuron, so expect a real, non-trivial recycle cost.
btcli subnet register --netuid 99 --wallet.name miner --subtensor.network finney
# dev only: btcli subnet register --netuid 526 --wallet.name miner --subtensor.network test
```

Your published bundle is your `weights.npz` plus the harness version string as its "source" — the
miner writes both for you. Point the daemon at your head:

```bash
orchestra-koth-miner --netuid 99 --network finney --wallet miner --hotkey default \
  --repo YOU/koth-miner --routing-model weights.npz
# dev only: --netuid 526 --network test
```

`--routing-model` validates the head against the pinned pool **at startup** rather than mid-epoch,
and forces the owner's action space: `--pool` is ignored, because the ladder your head was trained
against has to be the ladder it is scored on. The legacy `--source`/`--weights` free-agent path still
exists but is not what this subnet competes on.
### Get the owner's measured image

You cannot mine from a stock VM — an enforcing validator rejects it (`unapproved_runtime`). The image
is published on **HuggingFace** in the public bucket
[`thirtyspokes/cvm-runtime-image`](https://huggingface.co/buckets/thirtyspokes/cvm-runtime-image).
No token needed:

```bash
BASE=https://huggingface.co/buckets/thirtyspokes/cvm-runtime-image/resolve

curl -sL $BASE/runtime/latest.json                       # -> {"version": "v20"}

# the manifest first: hashes, measurements, and the recipe commit that built the image
curl -sL $BASE/runtime/v20/manifest.json | tee manifest.json

curl -sLO $BASE/runtime/v20/koth-runtime.tar.gz
sha256sum koth-runtime.tar.gz     # MUST equal manifest.image.sha256
```

A bucket has no immutable revision, so **check the sha256** — it is the only thing standing between
you and a silently-swapped image *before boot*. (After boot, RTMR1 catches it anyway; see below.)

**You do not have to trust the download.** The image is *self-verifying*: the owner pins **RTMR1** on
the chain, so if you boot the wrong bytes — corrupted, tampered, or a mirror lying to you — your proof
is rejected `unapproved_runtime` and you earn nothing. Checking the hash just saves you a wasted boot.

**And you don't have to trust the owner either — rebuild it.** The build is reproducible, so check out
the recipe commit named in the manifest and confirm you get the owner's exact image:

```bash
git checkout $(jq -r .reproducible.recipe_commit manifest.json)
bash scripts/build_koth_image_prod.sh
sha256sum <the built UKI>          # must equal manifest.reproducible.uki_sha256
```
If that matches, the runtime you're about to run is byte-for-byte the one the validators gate on.

### Boot it on a TDX confidential VM

Import the disk into your own cloud (GCP shown; Azure DCesv6 and bare metal work the same way):

```bash
gsutil cp koth-runtime.tar.gz gs://YOUR-BUCKET/
gcloud compute images create koth-runtime-v27 \
  --source-uri=gs://YOUR-BUCKET/koth-runtime.tar.gz \
  --guest-os-features=UEFI_COMPATIBLE,TDX_CAPABLE,GVNIC

gcloud compute instances create my-koth-miner \
  --zone=us-central1-a --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX --maintenance-policy=TERMINATE \
  --image=koth-runtime-v27 \
  --network-interface=nic-type=GVNIC --boot-disk-size=50GB \
  --metadata="^@^koth-epoch=<E>@koth-nonce=<N>@koth-hotkey=<YOUR_SS58>@koth-pool=<ALLOW_LIST>" \
  --metadata-from-file=koth-secrets=secrets.env,\
koth-agent-source=<(printf koth-harness-4 | base64 -w0),koth-agent-weights=<(base64 -w0 weights.npz)
```

> The metadata keys are `koth-agent-source` + `koth-agent-weights` (`koth/gcp_operator.py`), but for a
> routing head their **values** differ from the free-agent path: `source` is the **harness version
> string** (not a `my_router.py`), and `weights` is your **`weights.npz`** (not a `my_weights.bin`) —
> the miner publishes exactly this pair (see "Submit + run" above). Prefer the shipped operator
> (`orchestra-koth-gcp-miner`), which derives both from chain and fills the metadata for you; the manual
> `gcloud` form is illustrative and easy to get wrong.

The image is **locked down**: dm-verity read-only rootfs, no sshd, no shell. Your head and your
OpenRouter key are injected at boot via CVM metadata — never baked into the public image. The harness
runs with **zero network egress**; its only channel is the metered `call_model`.

| the image, pinned | |
|---|---|
| GCP image / bucket version | `koth-runtime-v27` |
| image sha256 | `c10a4b60dada223a4620d5566b0411de30a603f26ebe2e67741aa6a7b712bc1e` |
| UKI sha256 | `f9d650cc36bf6a589e98ad36db9d610d1ce1faeaab0a1b53ab70b9767e374426` |
| verity roothash | `a05e4d5325ef6b6aa155b29a9b2aea7649390857aba7258d857d046d7ef0ba31` |
| recipe commit | `a06cd1a` |
| MRTD | `c1ee9c16…` *(GCP TDVF — identical on every GCP TDX guest, a coarse check only)* |
| **RTMR1** | `cf0db5f18e34e2e699a98b256df3fc614019f5a1f5b76493e2aadc77e75b5be3d3b7ab3f409d5619dc70255d60c51444` **← the per-image anchor** |
| RTMR3 | `dd9bc28fae67ef43c5e998679a58ecab02e3042df9d5afa00534a6386948b1954bf4c704641d4a263a0ba897b512a186` *(binds runtime+suite+pool — moves whenever the engine does)* |
| on-chain governance | `96855d0fbe7e3fa3996964fffaf08671cdbac2a9ad9e99917a621561c149c130` (record v7) |

The validator gates every proof on MRTD + RTMR1/2/3 against the owner's on-chain record. Change *any*
byte of the rootfs and RTMR1 changes → `unapproved_runtime` → you earn nothing.

### Run the operator under supervision

An operator that is not running earns nothing, and nothing tells you: the chain simply shows no proof
and the validator reports `no_proof`, which reads like a bad epoch rather than a dead process. Run it
as a service, not from a shell.

```ini
# /etc/systemd/system/koth-miner.service
[Unit]
After=network-online.target
Wants=network-online.target
# Each start can create a CVM, so bound a crash-loop: after 5 failures in 10 minutes systemd gives
# up and leaves the unit FAILED (loud) instead of billing you forever.
StartLimitBurst=5
StartLimitIntervalSec=600

[Service]
WorkingDirectory=/path/to/thirtyspokes
EnvironmentFile=/path/to/thirtyspokes/.env
ExecStart=/root/.local/bin/uv run --frozen orchestra-koth-gcp-miner \
  --netuid 99 --network finney --wallet miner --hotkey default \
  --image koth-runtime-v27 --zone us-central1-a --machine-type c3-standard-4 \
  --n-per-bench 2 --epochs 0 --min-blocks 80 --attempt-deadline 900 --max-attempts 1
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

> `StartLimitBurst` / `StartLimitIntervalSec` must be in **`[Unit]`**. systemd accepts them in
> `[Service]` with only a log warning and then ignores them — so the crash-loop guard reads as
> present in your unit file while doing nothing at all.

### Mine continuously (one proof per boot, from OUTSIDE the VM)

The owner's published image is **locked down — no sshd, no shell** (see above), so there is nothing
to log into and no way to run a persistent process inside it. Each boot runs the suite for the ONE
epoch/nonce baked into that boot's `--metadata` and produces exactly one attested proof, then the VM
is done. "Mining continuously" means an operator-side loop, running on YOUR machine (not the TEE),
that repeats the boot above once per epoch:

The shipped operator performs that loop and derives both your immutable artifact revision and the
owner-pinned pool from chain:

```bash
export OPENROUTER_API_KEY=... MINER_HF_TOKEN=...
orchestra-koth-gcp-miner \
  --netuid 99 --network finney --wallet miner --hotkey default \
  --image THE_OWNER_APPROVED_GCP_IMAGE --repo YOU/koth-miner
# dev only: --netuid 526 --network test
```

For each epoch it waits for a usable block window, derives the block-hash nonce, creates a fresh
`c3-standard-4` Intel TDX VM, strictly reconstructs the attested proof and trace from the serial
console, checks their epoch/hotkey/artifact binding, uploads them, and deletes the VM. A cold-boot
`SandboxError`, incomplete serial payload, or other pre-proof failure is detected early and retried
on a fresh VM (three attempts by default). Every create/boot path runs bounded cleanup, and a restart
recognizes an already-uploaded valid epoch instead of billing for it twice.

Use `--epochs N --strict` for a fixed launch-gate run: any missed epoch exits non-zero instead of
being hidden by later successful epochs. Logs and per-epoch summaries go to
`logs/koth-gcp-operator/` by default. Run one operator process per registered miner hotkey.

This pattern was verified live end-to-end on testnet over multiple real epochs. The production
operator now owns the fresh-VM retry and cleanup behavior that those rehearsals previously carried in
an external scratch script; `koth/runtime.py` also retries a confined-child crash inside one boot.

`orchestra-koth-miner --confine` (the persistent `run_forever()` daemon, deriving epoch/nonce live
from chain itself) remains the right choice if you run your OWN confidential VM setup with shell
access instead of the owner's locked image — it is not what the owner's published image supports.

## What gets you disqualified

| Reason | What you did |
|---|---|
| `UnpinnedModelError` | called a model outside the pool allow-list |
| `no_pool_call` | didn't call the pool on some scored task (answered "free") |
| `hardcoded_answers` | a literal answer table in your **public** source (anyone can see it) |
| `ungrounded` | your final answer didn't come from any pool response — you ignored the pool and answered from your weights. The default grounding check reads your proof + trace only (no re-execution). |
| `copy_of:…` | your artifact behaves identically to an earlier-committed one (copies lose) |
| `no_proof_commit` / `commit_out_of_window` / `commit_mismatch` | *(anti-grind, only when the subnet sets a `commit_window` — then run the daemon with `--commit-proofs`)* the daemon commits your proof's hash on-chain right after the epoch opens and must reveal exactly it. You didn't commit, committed too late (best-of-N past the window), or revealed a **different** proof than you committed. Don't re-roll after it commits. Note the daemon holds the first proof commits back until your **artifact commit has revealed** (~72 min after `publish`), because both share one on-chain commitment slot — this is expected, not an error. |
| `unpinned_revision` | your on-chain commit points at a **branch** (e.g. `main`) instead of an immutable commit SHA. A branch moves, so nobody could prove *which* bytes were scored. The daemon commits the real HF revision for you — this only bites if you hand-roll the commit |
| `artifact_unavailable` | the validator couldn't download your bundle at the committed revision — usually a **private** repo (it must be public: that's what makes your artifact auditable) or a deleted revision |
| `trace_mismatch` | your uploaded trace doesn't match the proof binding |
| `over_budget` / `below_floor:*` | cost over budget, or accuracy under the floor on a benchmark |
| `bad_platform_quote` / `unapproved_runtime` | forged/modified TEE attestation, or a runtime whose MRTD/RTMR isn't the owner-approved measured image |
| `mock_quote_rejected` | ran the mock TEE (or a stock CVM) against an **enforcing** mainnet validator — boot the owner's measured image |
| `memorization` | *(opt-in `--probe` upgrade only)* aces the public set but collapses on a **secret** held-out probe far more than the field does. Not used in the default grounding mode. |

Everything you submit is public and bound to what actually ran, so cheating is detectable — compete on
genuine quality, not tricks.

## What you need

- A **confidential VM** (Intel TDX / AMD SEV-SNP; **CPU-only is fine** — models are remote, so no GPU).
  Cheap on Spot. You **must boot the owner's published measured image**
  ([`DEPLOYING.md`](DEPLOYING.md) §Measured image) — a stock CVM or the mock TEE is rejected, so it
  earns nothing.
- An **OpenRouter API key**, a **HuggingFace** account, and a **Bittensor wallet** + TAO + registration.
