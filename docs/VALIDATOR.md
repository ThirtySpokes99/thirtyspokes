# ThirtySpokes v3 — validating guide

**This is the v3 mechanism (v3).** The first-generation subnet, where a validator
verified attestation proofs and ran no inference, was retired on 2026-09-07 and its code and guides
removed; v3 is in testnet rehearsal before it opens on mainnet 99.

Under v3 a validator is the opposite of that: it **runs the miner's model itself** (the Conductor),
lets it route each task to a real OpenRouter model (the worker), grades the answers with the
benchmark's own grader, and compares two policies in a paired duel. There is one validator and the
owner runs it (D1).

---

## 1. What one window does

1. Draw a slice of tasks, stratified across the admitted benchmarks (§6.3b).
2. Run the **king's arm** once (§5.2a) — the current champion's policy over the whole slice.
3. Run up to `MAX_DUELS_PER_WINDOW` **challenger arms** against that same slice.
4. Score each arm: `final_b = quality_b − λ_b · spend_b / C_b`, equal weight per benchmark (§5.1).
5. Decide each duel: the paired delta must clear `EPS`, its bootstrap lower bound must clear zero,
   and both breadth conditions must hold (§5.2).
6. Crown the largest winning delta, set weights, publish the reveal.

**Six challengers per 24-hour window**, not eight — see §5 below, it is the read channel's price.

---

## 2. Before the first window: the grading host

**A validator will not start without `--sandbox-host`, and that is deliberate.** Grading executes
code a model wrote. §8b.9 requires that to happen somewhere other than the box holding the chain
signing key, and the code enforces it rather than trusting the operator: with `V3_DOCKER_HOST`
undeclared, `sandbox.run`, `available` and `preflight` all refuse **before issuing any subprocess**.

Set the grading host up first. Three things, none of which the code can do for you:

* **shared storage** at the *identical absolute path* on both machines (an identical path is not a
  shared path — `preflight` checks the bind by content and refuses until bytes actually arrive);
* **one forwarded Docker socket**, because every container crosses the transport. A per-call
  `ssh://` host multiplexes through `ControlMaster` (0.84 s per container against 2.73 s without),
  but every container is then a *session* on that connection and sshd caps those at `MaxSessions`
  (10 by default): at 16 episodes in flight the overflow opens fresh connections, and on a provider
  box whose host key has changed since it was recorded those fail — measured 2026-09-07, the arm
  died on its first grade. Forwarded channels are not sessions, so a single `ssh -L` carrying the
  daemon's socket has no such cap and no per-container handshake at all;
* an explicit **`IdentityFile`** if your key is not a default name, or ssh never offers it.

Concretely, on the controller (measured 2026-09-01; the controller shipped with no sshfs, NFS, CIFS or rclone):

```bash
apt-get install -y sshfs                       # fuse3 is already present
sshfs -o reconnect,ServerAliveInterval=15,allow_other root@<gpu>:/var/v3/grade /var/v3/grade
# and after EVERY sshfs mount that carries a model tree, raise the kernel read-ahead on it:
echo 16384 > /sys/class/bdi/$(grep " /var/v3/grade " /proc/self/mountinfo | awk '{print $3}')/read_ahead_kb
```

The read-ahead line is not optional for the tree mounts. Measured 2026-09-08 on the owner's link: one
sequential reader gets 3–7 MB/s at the default 128 KB and ~20 MB/s at 16 MB, and the value is fixed per
open file, so set it before a fetch or a submit starts, not while one runs.

and in `~/.ssh/config`, scoped to the one host so no other destination changes behaviour:

```
Host <gpu-ip>
    Port <port>
    IdentityFile /root/.ssh/<key>     # not a default name, so ssh never offers it unasked
    IdentitiesOnly yes
    ControlMaster auto
    ControlPath /root/.ssh/cm/%r@%h:%p
    ControlPersist 10m
```

Then forward the daemon's socket over ONE connection and keep it up. As a systemd unit on the
controller (the one this repository's owner runs; `Restart=always` reconnects after a provider
restart, and `StrictHostKeyChecking=yes` makes a changed host key a loud stop rather than a
silent new peer):

```ini
# /etc/systemd/system/v3-docker-tunnel.service
[Service]
ExecStartPre=/bin/mkdir -p /run/v3
ExecStart=/usr/bin/ssh -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes \
    -o StreamLocalBindUnlink=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=yes -L /run/v3/docker.sock:/var/run/docker.sock <gpu-ip>
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
```

Then prove it, on the box that will grade:

```bash
V3_DOCKER_HOST=unix:///run/v3/docker.sock V3_GRADE_DIR=/var/v3/grade \
  python -m thirtyspokes.v3.benchmarks.sandbox --image <a provisioned image>
```

A success line means the daemon answers, the resource caps are enforced, the image is present **on
that daemon**, and the bind resolves. Anything else raises and names the fix. Run it again after any
reboot: neither an sshfs mount nor an ssh config survives one by default, and the failure mode is
`preflight` refusing rather than a window grading wrongly.

---

## 2a. The owner's keys, with btcli 11

The daemon signs weights and the schedule root with the **owner hotkey** that `--wallet` /
`--hotkey` name (wallet and hotkey *names* under `~/.bittensor/wallets`, not addresses). It never
reads the coldkey. `btcli` here is the one inside the `bittensor` 11 package the `chain` extra
installs (`uv run btcli --version` prints `11.x`); the separately published `bittensor-cli` 9.x is a
different program with different flags (`--wallet-name`, `--hotkey`) and is not what this guide
uses. Every command below was run against 11.1.0.

```bash
btcli wallet new-coldkey -w <owner-wallet>                       # once; prompts for a password — keep it encrypted
btcli wallet new-hotkey  -w <owner-wallet> -H <owner-hotkey>     # sr25519 is fine here: nothing is sealed to this key
btcli wallet show        -w <owner-wallet> -H <owner-hotkey>     # both addresses; the hotkey is what --hotkey names

btcli subnets register --netuid <n> -n finney -w <owner-wallet> -H <owner-hotkey> --dry-run   # fee and effect, nothing sent
btcli subnets register --netuid <n> -n finney -w <owner-wallet> -H <owner-hotkey>             # asks, prompts for the coldkey password, spends
btcli stake add --netuid <n> -n finney -w <owner-wallet> --hotkey <owner-hotkey> --amount-tao <τ>   # the stake behind the validator permit
btcli wallet overview -w <owner-wallet> -n finney --netuid <n>   # the hotkey's UID and stake on the subnet
btcli subnets metagraph <n> -n finney                            # the subnet's view of it
```

`-n`, `-w` and `-H` are global options and sit anywhere on the line; `-y` skips a confirmation,
`--dry-run` previews any mutation, and `--wallet-password-file <file>` supplies the coldkey password
without a prompt. btcli 11 stores the hotkey file unencrypted, which is what a daemon that signs
every window needs. Weights are set by the daemon itself under this hotkey (§8); `btcli weights` is not part of
the workflow.

---

## 3. Launching

Every argument below is required except `--poll-seconds`, `--check` and the serving flags'
defaults.

```bash
orchestra-validator \
  --state /var/lib/v3 --netuid <n> --network finney \
  --wallet <owner-wallet> --hotkey <owner-hotkey> \
  --genesis-block <b> --window-blocks <n> --windows <n> --immunity-blocks <n> \
  --world <module:attr> --reference-tree /srv/reference \
  --serve-host root@<serving-host> --serve-cards 0:8002,1:8003 --serve-trees /var/v3/trees \
  --sandbox-host unix:///run/v3/docker.sock --grade-dir /var/v3/grade \
  --r2-endpoint <url> --r2-bucket <bucket> \
  --per-benchmark <n> --minimum <n>
```

**`--check` runs the launch-time gates against the chain and exits.** Use it before every real
start: it is the cheap version of discovering at task 173 that a flag was wrong.

Three of these are worth understanding rather than copying:

* **`--serve-host` is validator-managed serving, and it is how a subnet runs unattended.** Asked
  for a tree's committed name, the daemon launches vLLM for it on the serving host over ssh — with
  `serve.launch_command`, the only line that carries the pinned flags — on a free card or the one
  used longest ago, waits for `/v1/models` to list the name, and serves the arm. With two cards a
  window launches about seven times (every challenger at admission, the king, every challenger
  again for its arm), a few minutes each; a Conductor whose card was given away is re-served on
  its next turn. `--serve-cards` names the cards and their ports, reached here on the forwarded
  ports (§2); `--serve-trees` is the host-side directory behind `<state>/trees`; `--serve-preamble`
  is the toolchain export the launch script runs first. **The host gets a launch script and
  nothing else** — no Docker, no credential; weights are parsed by `safetensors` with no code
  path, and §2.1 guarantees the model's output never reaches an executor. One server per distinct
  tree: vLLM answers every alias with its first served name, so two names on one server are
  refused by the identity check (measured 2026-09-08).
* **`--serve-url`** is the operator-managed alternative: the endpoint(s) on which you have ALREADY
  launched the artifacts under test, comma-separated with one server per card. The daemon refuses
  a name no endpoint lists and launches nothing. Use it for a rehearsal or a fixed cast.
* **`--grade-dir`** must resolve **at the daemon**, not merely on this box. Required whenever
  `--sandbox-host` names another machine. The launch gate refuses a path that is not a directory
  here, because a typo would otherwise drop every task from both arms while the window spent its
  whole allowance.
* **`--minimum`** is the count below which a benchmark is dropped and the window narrows (§6.3b). A
  narrowed window is published as narrowed; breadth is judged over the benchmarks actually present.

---

## 3a2. The immunity period

`--immunity-blocks` declares what you believe the subnet's `immunity_period` is. The daemon reads the
real one off the chain and **refuses to start if the two disagree**, because the queue cap is derived
from that number: a challenger at the back of the queue waits two windows, and immunity has to cover
three. Set it too short and entrants who paid a registration burn and uploaded ~70 GB are
deregistered before they are ever judged.

---

## 3b. Funding — the owner account, and the miners' own keys

Every model call is metered by the gateway against an allowance. The **owner account** pays the
two reference arms every window, the king's arm while King zero reigns, and the retest sample;
**each miner pays on their own OpenRouter key** (whitepaper §4, D18): `orchestra-miner
register-key` seals it to your mailbox key — the one `orchestra-owner key` prints and miners pass
as `--owner-key` — and the daemon opens it with the seed in `<state>/mailbox-key.hex`, so the
daemon's `--state` must be the directory `orchestra-owner` uses, and that key must never be
rotated behind miners' backs (every key sealed to it would stop opening). At each window the
daemon reads the record fresh, probes the key (`GET /auth/key`, `/credits`), binds the hotkey to a
client on that key with the miner's cap bounded by what the key reports, and journals the cap as
`op: cap` in `allowances.jsonl`. A key that is refused, empty or capped at zero defers the entry.

The owner account is the one balance nothing else fills, so this is a step, not a detail:

```bash
orchestra-owner --state /srv/v3-state credit --hotkey owner --usd 40 --ref "extrinsic 0x91af"
orchestra-owner --state /srv/v3-state credit --hotkey <MINER_SS58> --usd 12.50 --ref "invoice 7"
orchestra-owner --state /srv/v3-state balances
```

`--ref` is free text and is never parsed. A credit to a miner's hotkey is your grant on top of
the rail — it widens that window's allowance whether or not a key is registered — and how it was
settled is yours to choose; the gateway only records what you say beside the credit.

Credits are appended to `<state>/allowances.jsonl` and survive a restart. A credit made while the
daemon is running is picked up at the start of the next window — no restart needed. Pass the daemon
the **same `--state`**, or it will read a different file and start against a wallet you believe you
filled.

**Why the daemon refuses to start on an empty owner account.** It is the one misconfiguration whose
symptom points somewhere else. Unfunded, both reference arms trip the exhaustion check at their
first step, score zero, and the power gate reports that the slice *cannot separate two policies
known to differ* — so the window is published blaming the corpus, every window, burning emissions,
while `--check` stays green. `preflight` refuses instead, and names this command.

---

## 3a. Issuing submission credentials — a second program, on purpose

**The validator daemon cannot issue a credential, and that is deliberate**: it wires its mailbox with
a minter that always raises, because a daemon able to mint could hand out write access to a
submission prefix from inside the loop that scores it. Issuing is `orchestra-owner`.

```bash
# A miner sends you their `identity` output. You need only the hotkey.
orchestra-owner --state /var/lib/v3 issue \
  --hotkey <miner ss58> --netuid <n> --wallet <owner-wallet> \
  --r2-endpoint <url> --r2-bucket <bucket>

# Who holds credentials, and which must now be revoked.
orchestra-owner --state /var/lib/v3 status

# A KEY-ONLY credential (D18): write access to `openrouter/` under the miner's prefix and nothing
# else, issued to a spent hotkey too — how a miner rotates a dead or leaked OpenRouter key after
# their submission credential expired. Its own generation series; nothing to revoke on a spent shot.
orchestra-owner --state /var/lib/v3 issue --key-only \
  --hotkey <miner ss58> --netuid <n> --wallet <owner-wallet> \
  --r2-endpoint <url> --r2-bucket <bucket>
```

Miners poll `https://store.thirtyspokes.ai` — the custom domain connected to the production bucket
(2026-09-08) — and the dashboard reads reveals from it, so `--r2-bucket` here and on the daemon must
name the bucket behind that domain, with public access on. Not the bucket's r2.dev address: Cloudflare
rate-limits it and calls it unfit for production. The domain answers 404 for a missing key; it
answers 403 only to Python's default User-Agent, which is the zone's bot protection and which the
miner tool sidesteps by naming itself.

`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` come from the environment. The uid and
registration block are read from the chain and **cannot be passed by hand** — the miner derives the
same prefix independently from the same four facts, so an operator-supplied number would scope the
token to a prefix the miner does not compute.

Two operational rules, both of which cost real money to get wrong:

* **`--state` must be the daemon's `--state`.** The one-shot ledger is `<state>/mailbox.json`. Two
  ledgers are two one-shots: each file would enforce "one submission per hotkey" correctly while the
  rule was false overall. (Sharing it is safe — both processes take an exclusive lock and re-read
  before writing. That was added when this tool made the ledger's second writer real; before it, the
  daemon's next write erased an issue made after it started.)
* **`status` is your revocation list, and it is not advisory.** When a shot is spent the miner still
  holds a write credential for a prefix the chain now names. `status` prints `REVOKE NOW` with the
  generations to revoke; revoke all of them, not just the newest, because rotations are cumulative.
  The cryptographic backstop is independent — every file is re-hashed against the commitment at duel
  time — so a missed revocation is a swap that fails rather than a swap that scores, but it is still
  a miner able to destroy their own submission after committing it.

---

## 4. What is published, and what to watch

Every window publishes the verdict and its diagnostics, because a crown decided by something other
than routing must be visible rather than inferred (§5.8):

| field | read it for |
|---|---|
| `delta`, `lcb` | the headline comparison and its bootstrap lower bound |
| `by_benchmark` | where the win came from |
| `loo_min`, `median` | the two breadth conditions |
| `sign_wins`, `sign_p` | a diagnostic that **does not gate** — published so a disagreement with breadth is visible |
| `noise_floor` | per-benchmark bootstrap SE. When it exceeds the effects a duel is trying to resolve, breadth is unjudgeable at that slice size and no threshold repairs it |
| `DUEL_WALL_CLOCK_REASON` | tasks the arm never reached — a fact about the validator, never scored as the miner's |

`stopped_reason` distinguishes three things that must not share a row: a miner who ran out of money
(`budget_exhausted`), a model that stalled (the episode clock's `wall_clock`), and a validator that
ran out of clock (`DUEL_WALL_CLOCK_REASON`).

**The window's outcome table (§5.2c, D17).** Every arm in a window is scored against the same
sampled worker outcomes: the first arm to make a `(task, model, attempt)` delegate buys it live
through the gateway, and every later arm that makes the same delegate replays the stored row — no
provider call, no grader run — while still being debited the row's price. The table lives at
`<state>/checkpoints/window-<n>/outcomes.jsonl`, one JSON line per fill, so a restart re-buys
nothing; it is never carried to the next window and never published. Three things to read:

| field | read it for |
|---|---|
| `metering[].charged_usd` vs `provider_usd` | what the arm was debited (equal to `recorded_usd`) against what the provider was paid; with `hits` and `fills` |
| `outcome_table` | rows, fills, hits, the stored-failure rate (a 429 storm shows here) and the provider's outlay for the window |
| `retest` | the drift audit: a few replayed keys re-bought live on your allowance (`RETEST_MAX_KEYS`, `RETEST_MAX_USD` in `config.py`) and how often grade or cost disagreed — diagnostic, gates nothing |
| `seen_unseen` | each arm's `quality` and `final` on the slice's tasks an earlier reveal already showed, and on the rest — the memorisation meter, diagnostic |
| `metering[].own_key` | whether the arm ran on the miner's registered key (D18) rather than a credited allowance |
| `metering[].endpoints` | per model the arm bought live, which endpoint answered (`provider`, or `provider:served_model` when the response named a different model) |
| `metering[].drift` | the `model@endpoint` pairs a miner-key arm was served by that no arm on YOUR key was served by that window — §11-1's residual made visible; evidence, not a gate |

Arms run reference → king → challengers, so your allowance pays the fills for the cascade's rungs
and a challenger that agrees with the king pays for its agreement without the provider being asked
twice. Budget the owner account for the reference arms, the king's arm while King₀ reigns, and the
retest sample.

---

## 5. The read channel, and what it costs you

Workers may inspect the repository before answering — three read-only verbs, at most
`MAX_READS_PER_DELEGATE` times per delegate, executed in a throwaway container on the sandbox host.
It exists because it was measured to be necessary: under a one-shot scaffold, **0 of 75 R2E-Gym
episodes produced a patch `git apply` would accept**, while the gold patch scored 1.0 on 24 of 25 of
those same tasks through the identical grader. The failure was the action space, not the models.

Its price lands on you, not on the miner:

* an arm needs **~8731 s** (read budget plus multiplexed transport) against a **9000 s** clock;
* which is why a 24-hour window holds **six** challenger arms rather than eight.

> **Do not read that as fitting.** Measured 2026-09-02, the 8731 s was taken with a *local* grade
> dir. If `--grade-dir` is on a shared mount to the sandbox host, each read pays ~30 filesystem
> round trips — **1652 ms per read over sshfs against 0.24 ms local**, ~1650 s per arm — and the
> honest total is **~10,383 s against a 9000 s clock**. `preflight` passing means the payload
> arrives, not that it arrives in time. The sandbox record (kept with the owner's operating records)
> has the numbers and the two fixes; until one of them lands, run the validator where its grade dir is local to the daemon.

> **How urgent that is — 2026-09-02.** **No window you can run today pays this price.** Every benchmark
> in the N = 2 corpus — LiveCodeBench, and HLE when its gate is accepted — returns `tools() == ()`: no
> read is ever issued, the read budget is zero, six-arms-not-eight is not yet the binding arithmetic, and (B) fits with room. What the section
> above prices is **admitting a tool-enabled benchmark**, and that stopped being hypothetical on
> 2026-09-02 — the channel was measured on R2E-Gym and moved the apply rate from **0/162 to 7/162**
> with 4 solves (the corpus record, §1.2a). R2E-Gym is still **not** admitted; its routable
> band is `+0.0000`. So read this as a **build item that must land before the first tool-enabled
> window, not as a defect in the window you are running** — and note that it cannot be deferred into
> that window, because the shortfall is ~1,400 s and no clock move covers it.

If you shorten the clock, cut `MAX_READS_PER_DELEGATE` first and read the census beside it — two
reads serve 71.1% of the corpus against 92.2% at three. But that constant is **one-way**: miners
train against the episode shape it produces, so it must be settled before the first window a
tool-enabled benchmark appears in. The clock is not one-way. Spend the reversible knob.

---

## 6. Running it wrong, and how each shows up

| symptom | cause |
|---|---|
| `--sandbox-host` argparse error | you are running a pre-2026-09 checkout or an old command line; the flag is required |
| `V3_DOCKER_HOST is unset…` | the grading host was never declared; nothing executed, which is the safe direction |
| `…did not arrive at /w` | an identical path is not a shared path (§2) |
| `Permission denied (publickey…)` | the key is not a default name and no `IdentityFile` names it |
| every task excluded, allowance spent | a `--grade-dir` the daemon cannot read — run `--check` and `preflight` |
| arms overrunning the clock | the socket tunnel is down and each container is opening its own SSH connection (2.73 s instead of 0.84 s); `systemctl status v3-docker-tunnel` |
| `…produced no V3_CASES line` with an ssh warning in the text | grades overflowed `MaxSessions` on a per-call `ssh://` host and the fallback connections failed their host-key check; use the forwarded socket (§2) |
