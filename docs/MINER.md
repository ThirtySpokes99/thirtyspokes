# ThirtySpokes — mining guide

You compete to build the **routing / orchestration agent** that gets the highest benchmark quality
*per dollar* over an owner-pinned pool of models. **King = the eligible miner with the highest
cost-budgeted quality.** For *how the mechanism works* see [`DESIGN.md`](DESIGN.md); to stand up a
subnet see [`DEPLOYING.md`](DEPLOYING.md).

> **Live on Bittensor mainnet — netuid 99.** Validators run **fail-closed**: to earn anything you must
> run the owner's measured runtime image on a confidential VM. A mock TEE or stock CVM earns **nothing**.

## What you build

One thing: a Python source file defining `build_agent(weights) -> agent`, where
`agent(prompt, call_model) -> answer`. `call_model(model, messages, params)` is your **only** channel
to the pool — it returns the model's response text and is metered for cost.

```python
# my_router.py  — a cost-aware cascade: cheap first, escalate hard prompts
import json
def build_agent(weights):
    cfg = json.loads(weights.decode())            # your routing config / trained params
    cheap, strong = cfg["cheap"], cfg["strong"]
    def agent(prompt, call_model):
        ans = call_model(cheap, [{"role": "user", "content": prompt}], {"max_tokens": 256})
        if len(prompt) > 400 or "prove" in prompt.lower():      # your routing logic
            ans = call_model(strong, [{"role": "user", "content": prompt}], {"max_tokens": 512})
        return ans
    return agent
```

Your bundle = `source.py` + `weights.bin` (opaque bytes — a trained model, a config, whatever) and it
is **public** on your HuggingFace repo. The competitive surface is open: a trained routing model,
orchestration (ensemble / verify / decompose / cascade), efficient prompting — anything that lifts
quality or cuts cost. **You write your own training — the subnet provides no trainer (that's your
edge).** Serialize your policy into `weights.bin` and publish it beside your `source.py`.

## What secures this — the trust boundary (read this)

Your agent runs in **your own** confidential VM, and that VM samples and runs the benchmark — so the
subnet does **not** trust your machine, it trusts the owner-published **measured runtime image**:

- You boot the owner's published, locked-down image ([`DEPLOYING.md`](DEPLOYING.md) §Measured image)
  on a TDX/SEV-SNP CVM. It loads *your* public `source.py` + `weights.bin` as the untrusted agent,
  hashes exactly those bytes into the attestation, meters your cost, and runs your agent with **zero
  network egress** (its only channel is the metered `call_model`).
- **What you compete on is the agent** (fully public); **what is fixed and measured is the runtime.**
- Validators run **fail-closed**: a modified runtime, a stock CVM, or the mock TEE is **rejected, not
  scored** (`mock_quote_rejected` / `unapproved_runtime`). There is no "trust me" path — the measured
  image is the only way in.

This is *why* static public benchmarks are safe: the score is bound to your public artifact running
under an audited runtime, so any cheat is visible in your source or caught by the validator. See
[`DESIGN.md`](DESIGN.md) §1 (trust model) + §6 (why benchmarks are safe).

> The mock TEE exists only for **offline development** (the local simulator). Mainnet validators reject
> its proofs, and the validator daemon refuses to disable its gate on mainnet — so don't tune against it.

## The rules (enforced by the runtime)

- **Pinned pool only.** Call *only* the owner's allow-listed models — anything else → `UnpinnedModelError`.
- **≥1 pool call per scored task.** Answering "for free" from your own weights → `no_pool_call`.
- **Your answer must come from the pool.** Each scored answer must match a response your agent got from
  a pool model — the **grounding check** (proof-inspection only, no re-execution). Ignoring the pool and
  answering from your weights (a lookup table / self-contained model) → `ungrounded`. Relaying,
  cascading, ensembling, and verifying are all fine.
- **You pay your own inference** (your OpenRouter key). Cost is metered from the real bill.

## How you're scored

**Quality first, then cost.** You are ranked by

```
S = Q_lcb − λ · (your_cost / B)      Q_lcb = Σ_benchmark w · lcb(accuracy),   λ = 0.02
```

`Q_lcb` is the weighted per-benchmark accuracy at its bootstrap **lower confidence bound** (so a lucky
run can't win). The cost term is deliberately **small — it can never beat a real accuracy gain**; it
only separates miners who are otherwise **equal**. That matters because accuracy *saturates*: once
you're at the ceiling, **being cheaper is how you keep climbing.**

To take the **crown** you must be not-worse than the king on every benchmark, and then either be
**confidently better on ≥1 benchmark**, *or* **match its quality at ≥10% lower cost**. Emissions split
**equally** across the king + a chain of up to 4 recent ex-kings (5 slots, ≈20% each when full — a
dethroned king keeps earning while it decays out of the chain, so there's no cliff to camp against),
with an epsilon incumbency margin protecting the king + earliest-commit tiebreak.

You are **eligible** only if ALL hold, else you earn nothing that epoch: `total_cost ≤ B` (the owner's
per-slice budget — a hard ceiling); accuracy ≥ `f_min` on **every** benchmark; ≥1 pool call on every
scored task.

**So: hit the quality bar on every benchmark, then get relentlessly cheaper.** (Full detail:
[`DESIGN.md`](DESIGN.md) §5.)

## Test locally before you spend a cent on-chain

The dev kit runs the exact validator scoring on your artifact:
```bash
uv run orchestra-koth-dev --source my_router.py --weights my_weights.bin
# -> per-benchmark acc/lcb, total_cost, Q_lcb, eligible, n_pool_calls  (byte-identical to the validator)
```
Iterate here until your `Q_lcb` is high and `eligible: true` under the budget.

## Submit + run

```bash
uv pip install -e ".[chain,eval,tee]"     # tee extra = full DCAP verification on a CVM
export OPENROUTER_API_KEY=...             # you pay your own inference
huggingface-cli login                     # to publish your public bundle
btcli wallet new_coldkey && btcli wallet new_hotkey
btcli subnet register --netuid 99 --wallet.name miner --subtensor.network finney
```
### Get the owner's measured image

You cannot mine from a stock VM — an enforcing validator rejects it (`unapproved_runtime`). The image
is published on **HuggingFace** in the public bucket
[`thirtyspokes/cvm-runtime-image`](https://huggingface.co/buckets/thirtyspokes/cvm-runtime-image).
No token needed:

```bash
BASE=https://huggingface.co/buckets/thirtyspokes/cvm-runtime-image/resolve

curl -sL $BASE/runtime/latest.json                       # -> {"version": "v14"}

# the manifest first: hashes, measurements, and the recipe commit that built the image
curl -sL $BASE/runtime/v14/manifest.json | tee manifest.json

curl -sLO $BASE/runtime/v14/koth-runtime.tar.gz
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
gcloud compute images create koth-runtime-v14 \
  --source-uri=gs://YOUR-BUCKET/koth-runtime.tar.gz \
  --guest-os-features=UEFI_COMPATIBLE,TDX_CAPABLE,GVNIC

gcloud compute instances create my-koth-miner \
  --zone=us-central1-a --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX --maintenance-policy=TERMINATE \
  --image=koth-runtime-v14 \
  --network-interface=nic-type=GVNIC --boot-disk-size=50GB \
  --metadata="^@^koth-epoch=<E>@koth-nonce=<N>@koth-hotkey=<YOUR_SS58>@koth-pool=<ALLOW_LIST>" \
  --metadata-from-file=koth-secrets=secrets.env,\
koth-agent-source=<(base64 -w0 my_router.py),koth-agent-weights=<(base64 -w0 my_weights.bin)
```

The image is **locked down**: dm-verity read-only rootfs, no sshd, no shell. Your agent and your
OpenRouter key are injected at boot via CVM metadata — never baked into the public image. The agent
runs with **zero network egress**; its only channel is the metered `call_model`.

| the image, pinned | |
|---|---|
| GCP image | `koth-runtime-v14` |
| UKI sha256 | `4757aaabe0cfef16732da7db6c92043cde55552c379d6048b8c7149fa5daff63` |
| MRTD | `9bf86e6280ec4282…` *(GCP TDVF — identical on every GCP TDX guest, a coarse check only)* |
| **RTMR1** | `6d8893eb9255aad4b52045250f6aaecefbf93163ebcc2b9e3149930314df6c04…` **← the per-image anchor** |
| RTMR2 | `0000…` *(zero under direct-UKI boot)* |
| RTMR3 | derived from runtime+suite+pool, extended by the runtime at startup |

The validator gates every proof on MRTD + RTMR1/2/3 against the owner's on-chain record. Change *any*
byte of the rootfs and RTMR1 changes → `unapproved_runtime` → you earn nothing.

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
  --netuid 526 --network test --wallet miner --hotkey default \
  --image THE_OWNER_APPROVED_GCP_IMAGE --repo YOU/koth-miner
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
