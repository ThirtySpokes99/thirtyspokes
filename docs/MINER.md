# ThirtySpokes v3 — mining guide

**This is the v3 mechanism (v3).** The first-generation subnet, where miners ran
benchmarks in their own TEE and uploaded a proof, was retired on 2026-09-07 and its code and guides
removed; v3 is in testnet rehearsal before it opens on mainnet 99.

Under v3 you submit **a model, not a proof**. The validator runs it. Your model is the *Conductor*:
per task it reasons, then delegates the task to a real OpenRouter model that answers. You are scored
on the answers your routing produced and the dollars it spent.

---

## 1. What you submit

Full weights against the pinned reference architecture — **not an adapter, not a LoRA** (D11).

| | |
|---|---|
| architecture | `Qwen/Qwen3.6-35B-A3B`, pinned |
| dtype | `bfloat16`, pinned — two precisions are not the same experiment |
| format | `.safetensors` only; `.bin` / `.pt` / `.pth` are **refused** |
| size | ~70 GB per submission |

The format pin is what makes "no miner code ever executes" survive a full-weights upload: a pickle
archive is code, and `safetensors` is not.

**A hotkey gets exactly one submission, ever** (§7). Not one per window — one. A malformed upload,
a wrong dtype, a stray `.bin` spends it. Use the dev kit (`thirtyspokes-dev`) against your own
artifact before you spend the shot; it runs the same admission checks the validator will.

**Copy `config.json` and the tokenizer files from the reference tree verbatim.** This is the trap
most likely to cost you the shot, and it does not look like a mistake. `save_pretrained` rewrites
both, and a different `transformers` version writes them differently — it renames fields, moves them
between blocks, reorders keys. Your weights can be perfect and the tree still refused, because the
tokenizer is compared by digest and the architectural fields are compared value by value. Train
however you like; when you save, put the reference's own `config.json` and tokenizer back. Then run
`check` until it says admitted.

### 1a. Step by step — from an empty wallet to a spent shot

The whole path is `btcli` for the chain-side account work and `thirtyspokes-miner` for everything
the mechanism owns. Nothing here is optional and nothing past step 7 is reversible.

#### 0. Prerequisites

* Python 3.12 and [`uv`](https://docs.astral.sh/uv/); a machine with the ~70 GB model tree on local
  disk (the upload reads it twice — once to hash, once to send).
* `btcli` — the command line that ships **inside the `bittensor` package** the `miner` extra
  installs (`uv run btcli --version` prints `11.x`). Used only to create the hotkey and register
  it. The separately published `bittensor-cli` 9.x is a different program with different flags
  (`--wallet-name`, `--no-use-password`); every command below is the 11.x spelling.
* A coldkey with **free** TAO for the registration burn plus fees: `btcli subnets burn-cost 99 -n finney`
  prints the burn; the transaction fee and the MEV-shield carrier fee add roughly τ0.01, and
  `btcli subnets register` (step 2) shows the total and asks before spending it.
* The reference tree (step 4) and your trained weights on the pinned architecture (§1).

```bash
git clone https://github.com/thirtyspokes99/thirtyspokes && cd thirtyspokes
uv venv --python 3.12 && uv pip install -e ".[dev,miner]"
uv run thirtyspokes-miner --help
```

`miner` is what the miner commands import beyond the core — the S3 client for the upload, the chain SDK and wallet, and the envelope cryptography; the offline suite never touches them.

#### 1. Create an **ed25519** hotkey — not btcli's default

The owner publishes your upload credential as an envelope **sealed to your hotkey's own public
key**, which needs the ed25519→X25519 conversion. `btcli` creates **sr25519** hotkeys unless told
otherwise, and an sr25519 hotkey cannot open the envelope. `submit` refuses such a wallet by name,
but it refuses *after* you have paid to register it, so create the key correctly first:

```bash
WALLET=your-coldkey-wallet      # an existing coldkey wallet (or: btcli wallet new-coldkey -w "$WALLET")
HOTKEY=v3-1                    # a NEW hotkey name; one hotkey = one submission, ever

btcli wallet new-hotkey -w "$WALLET" -H "$HOTKEY" --crypto-type ed25519
btcli wallet show -w "$WALLET" -H "$HOTKEY"          # hotkey_crypto_type must read ed25519
```

`-w` names the coldkey wallet and `-H` the hotkey inside it; both are global options and sit
anywhere on the line. The mnemonic is printed once — keep it. btcli 11 stores hotkeys
**unencrypted** (its `wallet unlock` is for the coldkey), which is what `submit` needs: it reads
the seed out of the keyfile to open the envelope. A hotkey keyfile that *is* encrypted came from
another tool; decrypt it in place first, which prompts for its password:

```bash
uv run python -c "from bittensor_wallet import Wallet; Wallet(name='$WALLET', hotkey='$HOTKEY').hotkey_file.decrypt()"
```

The coldkey stays encrypted; it is never read here.

Without `btcli`, the same key from Python:

```bash
uv run python - <<'EOF'
from bittensor_wallet import Wallet, Keypair
w = Wallet(name="your-coldkey-wallet", hotkey="v3-1")
w.set_hotkey(Keypair.create_from_mnemonic(Keypair.generate_mnemonic(12), crypto_type=0),  # 0 = ed25519
             encrypt=False, overwrite=False)
print(w.hotkey.ss58_address)
EOF
```

**Verify before you spend anything.** This reads only the keyfile:

```bash
uv run thirtyspokes-miner --netuid 99 --wallet "$WALLET" --hotkey "$HOTKEY" hotkey
```

```
hotkey             5Dm…N9Mt
scheme             ed25519 — the mailbox envelope can be sealed to it
keyfile            readable
safe to register:  btcli subnets register --netuid 99 -n finney -w … -H …
```

A refusal here names the fix (`--crypto-type ed25519`, or unlock the keyfile). A refusal at
`submit` would have cost a registration.

#### 2. Register the hotkey on the subnet

The burn, the transaction fee and the MEV-shield carrier fee all come out of the coldkey's **free**
balance, and `burned_register` cannot be submitted unshielded — so an unfunded coldkey stops with
`MEV-shielded submission needs free TAO for the outer carrier fee` before anything is sent. Check,
and fund if needed (`--dest` takes a wallet name or an ss58 address):

```bash
btcli wallet balance -w "$WALLET" -n finney                                          # free TAO must cover burn + ~τ0.01
btcli wallet transfer -w <funded-wallet> -n finney --dest "$WALLET" --amount-tao 0.05   # if it does not
```

On testnet (`-n test`, netuid 526) the same transfer from any funded testnet coldkey does; testnet
TAO has no value. Then:

```bash
btcli subnets register --netuid 99 -n finney -w "$WALLET" -H "$HOTKEY" --dry-run   # fee and effect, nothing sent
btcli subnets register --netuid 99 -n finney -w "$WALLET" -H "$HOTKEY"             # asks, then spends
```

This spends the burn from the coldkey (it prompts for the coldkey password; `-y` skips only the
confirmation) and gives the hotkey a UID and a registration block — the two chain facts your upload prefix and mailbox key are derived
from. **Register one fresh hotkey per submission.** A hotkey that has ever committed is spent
forever, and re-registering it does not reset that. Check with:

```bash
btcli wallet overview -w "$WALLET" -n finney --netuid 99      # the hotkey's UID appears once registered
```

The subnet's `immunity_period` covers three windows; your UID is safe from deregistration for at
least that long, which is the whole time the queue can take to reach you (§7).

#### 3. Send the owner your identity

```bash
uv run thirtyspokes-miner --netuid 99 --wallet "$WALLET" --hotkey "$HOTKEY" identity
```

```
hotkey             5Dm…N9Mt
uid                17
registration block 7953991
registration id    3f9c…a12e
upload prefix      submissions/3f9c…a12e/
  mailbox keys to poll:
    generation 1: mailbox/v1/3f9c…a12e/00000000000000000001.bin
    generation 2: …
```

Send the owner the hotkey (the rest they re-derive from the chain — nothing you type can change
where your credential is scoped to). They run `thirtyspokes-owner issue`, which publishes your
envelope on the subnet's public store, `https://store.thirtyspokes.ai` — the tools poll it by
default. **You don't need anything back from the owner.** Their ed25519 public key is pinned in the
tools (`config.OWNER_MAILBOX_KEY`, `ae6cea37…3d21`), and `submit` refuses any envelope not signed by
it. Anyone who hands you a *different* owner key is not the owner: passing it would make
`register-key` seal your OpenRouter key to them. Register your OpenRouter key once the envelope is up (step 6): an entry
with no key and no credit is deferred, not judged.

#### 4. Get the reference tree and prepare yours

The validator compares your tree to the pinned reference; `check` needs that reference locally.

```bash
uv run hf download Qwen/Qwen3.6-35B-A3B \
    --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir ./reference
```

Then, whatever you trained with, **save your weights and put the reference's own `config.json`
and tokenizer files back** (see §1 — `save_pretrained` rewrites both, and the tokenizer is compared
by digest). Your tree must contain only `.safetensors` shards, their index, `config.json`,
`generation_config.json` if present, and the tokenizer files — no `*.py`, no `.bin`/`.pt`/`.pth`,
no `auto_map`, no `trust_remote_code`. Run the dev kit until the endings look like routing:

```bash
uv run thirtyspokes-dev --model-tree ./weights --reference ./reference \
    --conductor my_serving:conductor --budget-usd 1
```

#### 5. `check` — the validator's gate, locally, until it says admitted

```bash
uv run thirtyspokes-miner --netuid 99 --wallet "$WALLET" --hotkey "$HOTKEY" check \
    --model ./weights --reference ./reference
```

`admitted: this tree passes the checks the validator runs (§1.1)` — or the validator's own refusal
string, verbatim. It opens no wallet and touches no network.

#### 6. Register your OpenRouter key

Your arm runs on **your own** OpenRouter key (whitepaper §4, D18). Create one at openrouter.ai —
give it a credit limit there if you want a hard ceiling on your exposure — put it in a file, and
register it with the cap one window may spend:

```bash
printf '%s' 'sk-or-v1-…' > ~/.openrouter-key && chmod 600 ~/.openrouter-key
uv run thirtyspokes-miner --netuid 99 --wallet "$WALLET" --hotkey "$HOTKEY" register-key \
    --key-file ~/.openrouter-key --cap-usd 20
```

The key is sealed to the owner's key **before** it leaves this machine and the record is signed by
your hotkey; it lands beside your tree as `openrouter/key.json`, and only the owner's validator opens
it. It uses the same envelope `submit` opens, so run it any time that credential is valid — before
or after step 7 — and run it again to rotate the key or move the cap. Once that credential has expired and the
shot is spent, ask the owner for a **key-only** credential (`thirtyspokes-owner issue --key-only`)
and run `register-key --key-credential --generation N`: it can write your key and nothing of
your tree. §6 has the order of
magnitude for the cap; the validator reads the record fresh every window and bounds the cap by
what the key reports it can still spend.

#### 7. `submit` — this spends the shot

```bash
uv run thirtyspokes-miner --netuid 99 --wallet "$WALLET" --hotkey "$HOTKEY" submit \
    --model ./weights --reference ./reference
```

**Your upload is private.** It goes to the subnet's private models bucket, readable only through
your own scoped credential and by the validator — never by other miners, and not before or after
your duel. Only if you **win the crown** does the validator publish the copy it verified to the
public models bucket, at `models/sha256/<your manifest digest>/` under
`https://models.thirtyspokes.ai` (see [Downloading the king](#downloading-the-king)); that copy is
what D14 makes public, and it is never deleted. A losing submission's weights are deleted 14 days
after it is judged, and its manifest is kept.

In order, and it stops at the first refusal: re-runs the admission gate; refuses if this hotkey has
already committed; fetches and opens your envelope (checks the owner's signature, your hotkey, your
registration, the prefix and the expiry); builds and signs the manifest; uploads every file to your
prefix with 64 MiB multipart parts — **hours** for 70 GB on an ordinary uplink, resumable, and files
already present with the right digest are skipped; publishes `manifest.json` last; then commits the
ready signal on chain. That last step is the shot.

```
submitted 31 files (70,318,145,536 bytes uploaded, 0 already present)
  registration  3f9c…a12e
  manifest      e43a72c5…df91
  committed by  5Dm…N9Mt
THE SHOT IS NOW SPENT. Keep the OpenRouter account behind your registered key funded (§4, docs/MINER.md §6).
```

If no key is registered yet the last line says so instead; the entry is deferred as unfunded until
one is (step 6), with the shot intact.

Keep the `manifest` digest: the validator serves your model under the name
`v3-<your hotkey>@<manifest>` and every reveal names you by hotkey.

**If the credential expires mid-upload** (one day by default): ask the owner for a rotation and
re-run the same command with `--generation 2`. Already-uploaded files are skipped, the manifest is
rebuilt identically, and the commit happens once.

#### 8. What happens next

You are in the queue in `(commit block, hotkey)` order; three challengers are judged per window
and the wait is published. `https://store.thirtyspokes.ai/v3/queue.json` lists everyone
committed and not yet judged, so your position is visible before the window that judges you
runs — and so is a commit that landed past the queue's depth cap. When your window runs, the validator downloads your tree, re-hashes every
file against your manifest, admits it, serves it, runs the king's arm and yours on the same slice,
and publishes the verdict — `final`, the four conditions, per-benchmark deltas, and where your
allowance ran out if it did — in the window's reveal and on the [dashboard](https://thirtyspokes.ai/dashboard).
A refusal is final for that hotkey; a new attempt is a new hotkey, a new registration and a new
70 GB.

#### Downloading the king

Only the reigning king's weights are public (D14), and building on them is allowed. Every window's
reveal says where they are: `record.crown_model` gives the king's `url` and `manifest_url` under
`https://models.thirtyspokes.ai`, and the `manifest_sha256` that manifest must hash to. It is `null`
while King₀ reigns, because King₀ is a fixed policy with no weights to download. The bucket cannot be
listed, so the manifest is how the files are found: each is at `url` + its `path`, with its `size`
and `sha256`.

```bash
STORE=https://store.thirtyspokes.ai
WINDOW=$(curl -fsS "$STORE/v3/latest.json" | jq .window)
curl -fsS "$STORE/v3/reveal/$WINDOW.json" | jq .record.crown_model > king.json   # null: King₀ reigns
URL=$(jq -r .url king.json)
curl -fsS "${URL}manifest.json" -o manifest.json
echo "$(jq -r .manifest_sha256 king.json)  manifest.json" | sha256sum -c -
jq -r '.files[].path' manifest.json | while read -r path; do
    mkdir -p "king/$(dirname "$path")"
    curl -fsS "$URL$path" -o "king/$path"
done
jq -r '.files[] | "\(.sha256)  \(.path)"' manifest.json | (cd king && sha256sum -c -)
```

The two `sha256sum -c` lines are the check: the first ties the manifest to the digest the reveal
names, and the second ties every file to the manifest. A king is ~70 GB, so the loop takes a while;
any HTTP client that fetches the same paths will do.

#### Troubleshooting

| you see | it means | do |
|---|---|---|
| `… is not an ed25519 hotkey …` | the hotkey is sr25519 (btcli's default) | create a new one with `--crypto-type ed25519`; do not register the old one |
| `the hotkey keyfile at … is encrypted` | the seed cannot be read | decrypt it in place (the one-liner in step 1); btcli 11 never encrypts hotkeys, so this file came from another tool |
| `MEV-shielded submission needs free TAO for the outer carrier fee` | the coldkey's free balance is empty | fund the coldkey (step 2); the burn, the fee and the shield's carrier fee all come from it, and `burned_register` cannot run unshielded |
| `No such option: --wallet-name` | you are running `bittensor-cli` 9.x, a different program | use the `btcli` inside the `bittensor` 11 package this repo installs (`uv run btcli --version`); its flags are `-w`, `-H`, `-n` |
| `… is the legacy (v9) btcli config and is ignored` | an old `~/.bittensor/config.yml` from 9.x | harmless; `btcli config set` carries values over, then delete the old file |
| `no hotkey keyfile at …` | the hotkey is not on this machine | create it (step 1), or copy the registered hotkey's keyfile from the machine that holds it |
| `identity` fails with no registration | the hotkey is not registered on this netuid | step 2 |
| `this tree would be REFUSED at admission` | config/tokenizer/tensor/format mismatch | copy the reference's `config.json` + tokenizer back; remove any `*.py`, pickle shards; re-run `check` |
| `mailbox envelope is not signed by the owner` | an `--owner-key` override that is not the owner's, or a `--mailbox-url` that is not the subnet's store | on netuid 99 drop both flags — the owner key is pinned in the tools; pass them only for a rehearsal you were told to run |
| `--owner-key is required off finney netuid 99` | you are on another network or netuid, where the pinned key belongs to the wrong owner | pass that subnet owner's `--owner-key` |
| `mailbox credential has already expired` | the upload outran the credential | ask for a rotation; `--generation 2` |
| `register-key` says the credential expired and your shot is spent | the submission credential is gone for good | ask the owner for a key-only credential (`issue --key-only`); re-run with `--key-credential --generation N` |
| `… has already committed manifest …` | this hotkey's shot is spent | a new hotkey |
| the upload died mid-way (`SSL … EOF`, `Connection reset`) | the link dropped for longer than the retries cover | run `submit` again: it re-hashes the tree, then skips every file the bucket already holds with the right digest and sends only the rest; the shot is spent only when `manifest.json` lands and the ready signal is committed |
| your entry shows `deferred` in the reveal | no key and no credit, or the queue was full | `register-key` (step 6); a deferred shot is intact |
| `deferred` with `OpenRouter refused the registered key (HTTP 401)` | the key is dead or mistyped | register a live one (`register-key` again) |
| `deferred` with `has nothing to spend this window` | the account behind the key is empty, its limit is used up, or the cap is zero | add credits at openrouter.ai, raise the key's limit, or re-register with a larger `--cap-usd` |
| `deferred` with `the registered OpenRouter key cannot be used` | the record was sealed to another owner key, or signed by another hotkey | re-run `register-key` from this wallet without `--owner-key` — on netuid 99 the pinned key is the right one |
| `no OpenRouter key: pass --key-file, or set OPENROUTER_API_KEY` | `register-key` found no key | the file is empty, or the variable is unset in this shell |

---

**The dev kit runs your slice the way the validator will**, including the same number of episodes
in flight (`EPISODE_CONCURRENCY`, currently 16). That matters for more than speed: a serial run over
a production-sized slice takes hours and ends in `duel_wall_clock`, an ending the scored arm would
never have given you. Pass `--concurrency 1` when you want to read one episode's trace at a time.

## 2. What your model has to emit

Your Conductor sees a rendered state and replies with **one action as the last line**. Anything else
is a parse failure, and three consecutive failures force `STOP` (§2, §8b.2).

```
REASON <free text>          optional, may precede an action; capped at 512 tokens
DELEGATE <model-id>         send the task to this OpenRouter model
RETRY <model-id>            send it again to a different (or the same) model
STOP                        submit the best answer so far
```

Bounds you train against, all pinned: `MAX_STEPS = 12` per task, `REASON_TOKEN_CAP = 512`,
`MAX_CONDUCTOR_TOKENS = 576`. The score keeps the **best** answer across your delegations, so a
retry can only help your quality — and always costs dollars.

**Any OpenRouter model is allowed.** There is no owner-curated shortlist; the catalog is filtered
only for structural viability (a model that cannot emit text, or has no price, cannot be routed to).

---

## 3. What you cannot do, and why it is not worth trying

**Nothing your model writes ever reaches the worker.** §2.1: the only things that reach an answering
model are the benchmark's own task text and a validated model ID. Your reasoning, your trailing text,
your choice of argument — none of it travels. This is a property of the seam's *shape*, not a filter:
`_call_worker` has no parameter such content could pass through, and the audit republishes every
worker request as `compose(task.prompt, observations)` so anyone can recompute it.

Your `REASON` text is kept verbatim in the published trace. It is refused at the seam, not scrubbed
from the record.

So the answer path is not where the competition is. The competition is **which model you pick, and
what you spend**.

---

## 4. The read channel

On benchmarks that declare it, the *worker* you delegate to may inspect the repository before
answering. You do not drive this and cannot influence it — the worker issues its own reads, and the
budget is per delegate:

```
READ <path> [start_line]   up to 200 numbered lines of one file
LIST <path>                the entries of one directory, sorted (up to 200)
FIND <text>                up to 50 `path:line:text` matches for a FIXED string
```

At most **3** inspections per delegate, ~8 KB per observation. Why it exists, measured: without it,
**0 of 75 R2E-Gym episodes produced a patch `git apply` would accept**, because a model that has
never seen the file invents the context lines — and `git apply` matches on context alone. The gold
patch scored 1.0 on 24 of 25 of those same tasks through the identical grader.

What this means for you: **the read budget resets per delegate.** A retry to a stronger model does
not inherit the cheap model's reads. Escalating is a fresh start, not a continuation — price that in.

---

## 5. How you are scored

Per benchmark: `final_b = quality_b − λ_b · spend_b / C_b`, with **equal weight per benchmark**, so
no benchmark dominates by contributing more tasks. `λ_b` is set so that always-routing-cheapest and
always-routing-strongest score the same — spending more is neither rewarded nor punished in itself,
only relative to what it buys.

To take the crown from the king, **all** of these must hold (§5.2):

1. the paired delta in `final` exceeds `EPS = 0.05`, **and** its bootstrap lower bound clears zero;
2. **leave-one-out**: drop your single best benchmark and you still win;
3. **median**: the typical benchmark moved your way.

Conditions 2 and 3 are why a routing table that memorises one benchmark does not win. Both run over
the benchmarks *present* in that window, not a fixed count.

Two consequences worth internalising:

* **A near-copy of the king never wins.** It ties, and a tie fails a strict `EPS` beat. There is no
  copy detector; the economics do the work.
* **`EPS` is a real bar.** At ~250 tasks per arm it behaves like a 7–8 point margin. A challenger
  genuinely at +0.05 is crowned about half the time.

---

## 6. Budget

You pay for your arm **on your own OpenRouter key** (step 6 of §1a; whitepaper §4, D18). The
owner's validator opens the key you sealed to it, runs every one of your arm's worker calls on it
through the owner's gateway — the same pinned request body every other arm sends — and meters each
call's real cost against your **cap**, the per-window ceiling you set when you registered the key,
bounded by what the key reports it can still spend. Every debit is against a signed receipt you can
check; the reveal's `metering` row for your arm says `own_key: true`, lists the endpoints that
served each model you bought, and lists as `drift` any endpoint the owner's own arms were not served
by that window. The owner may also credit your hotkey by hand; that widens the window's allowance
and is a grant, not the rail.

Register the key before your entry is judged: an arm with no key and no credit is **deferred**,
not judged, and so is one whose key the provider refuses or whose account is empty — the shot stays
intact and the entry is judged in the first window after it is funded. If an arm exhausts its cap
mid-slice, the remaining tasks are published as `budget_exhausted` and score zero — distinct in
the reveal from a model that stalled and from a validator that ran out of clock, because those are
three different findings. A 401 or 402 from the provider mid-arm is counted as `unfunded_calls`,
never as your model's failure.

If you become king, keep the account behind the key funded: your arm is re-run every window as the
baseline every challenger is measured against, on the same key.

**What the key does not buy you.** The request body is the owner's — no fallbacks, parameters
required, price-sorted — and your account's own routing settings are yours to keep honest: an
endpoint the owner never saw shows up in the reveal as `drift`, and a `served_model` that differs
from the model you asked for is listed with it. Whitepaper §11-1 states what that evidence does
and does not reach.

**You pay for the calls your Conductor makes, whether or not the owner had to buy them.** Every arm
in a window is scored against the same worker draws (whitepaper §5.2c): a delegate some earlier arm
in the window already made is replayed from the window's outcome table rather than bought again,
and your allowance is debited its priced cost exactly as if it had been bought — same receipt, same
exhaustion rule. The reveal's `metering` shows, for your arm, `charged_usd` (what you paid, equal to
your recorded spend) beside `provider_usd` (what the provider was actually paid for your arm) with
`hits` and `fills`. Your Conductor is scored on the same draws the king's was, so a model that
makes the king's choices scores the king's score to the last bit — and a tie does not clear `EPS`.

**How much cap is enough?** You set it, to whatever you like — there is no minimum, no maximum
and no figure the subnet expects of you. The honest answer to how much is *enough* is that it depends
on how expensive your routing is, because that is the thing being scored: a policy that always calls
the cheapest model needs a fraction of what one that always calls the strongest does.

For an order of magnitude — a measurement of what POLICIES cost, not a budget anyone is asking you
for — the owner's M3a run recorded per-episode spend over 30 episodes per fixed policy: **$0.001**
always calling the cheapest model, **$0.067** always calling the strongest, **$0.062** for a
cheap-then-escalate cascade. Against the design target of ~250 tasks per arm that is roughly **$0.25
to $17 per window**. Small sample, two benchmarks — and measured on a deliberately frontier-free cheap+mid pool whose
top rung is about $6 per million tokens. You may route to any admitted model and the catalogue
carries rungs a hundred times dearer, so this is a floor rather than a bracket. Fund above your own estimate: an arm that exhausts mid-slice scores zero on every task it
never reached, and that is a worse outcome than over-funding.

**What else entering costs you.** Three things, none of which the owner sets:

* the **Bittensor registration burn** for your hotkey, at whatever the network charges when you
  register;
* your **training compute**, bought wherever you like — this project has never measured it and does
  not estimate it for you;
* the **upload bandwidth and time** for a ~70 GB tree — hours on a normal connection. The storage
  itself is the owner's: submissions live in owner-provided R2 and §8b.7 bounds that bill, not
  yours.

The subnet charges you nothing beyond the allowance you fund and spend yourself.

---

## 7. The queue

Commits beyond `MAX_QUEUE_DEPTH` are **refused, not accepted and deferred** — taking a registration
burn for a slot that would expire before it is judged is the outcome the cap exists to prevent.
Queued challengers roll over with their shot unspent; a queue is never dropped, only delayed.

Six challenger arms are judged per 24-hour window.
