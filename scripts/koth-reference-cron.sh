#!/usr/bin/env bash
# Owner cron: publish the per-epoch pool reference for ThirtySpokes (netuid 526, testnet).
#
# WHY THIS RUNS ON A TIMER. The validators' scalar is frontier-relative: a miner is scored on how
# much quality it added over what its money could have bought from the pool anyway. That frontier is
# this record. Miss an epoch and that epoch silently scores on absolute accuracy instead — no error,
# no alert, just a different (worse) number. So the failure mode of NOT running this is invisible,
# which is exactly why it belongs in cron rather than in someone's memory.
#
# Cadence: an epoch is 100 blocks (~20 min at 12 s/block). Running every 15 minutes guarantees at
# least one publish per epoch. Re-running inside the same epoch is harmless — the record path is
# derived from (epoch, nonce), so it overwrites itself rather than accumulating.
#
# Cost: n_per_bench x |pool| = 8 x 2 = 16 calls per epoch. Only RANKED benchmarks are measured.
set -euo pipefail

REPO=/workspace/venus/minirouter_workspace/thirtyspokes
LOG=/var/log/koth-reference.log

# NO --pool HERE, DELIBERATELY. This used to pin "meta-llama/llama-3.1-8b-instruct,openai/gpt-4o" —
# two models, while a routing head emits rungs into the harness's 7-rung ROUTING_POOL. That does not
# fail: `decision_regret` drops every rung past the reference's width, so most of each miner's
# actual contribution would vanish from the scalar with no error anywhere. The reference now defaults
# to `harness.pool_models()`, so the published columns and the action space come from ONE source and
# cannot drift. Re-adding a --pool flag here re-opens that hole.

# ABSOLUTE PATH, because cron's PATH is /usr/bin:/bin and nothing else. The first three ticks of
# this job died on `uv: command not found` and the only trace was two lines in this log — the exact
# silent-failure mode the reference is most vulnerable to, since a missing reference does not error
# anywhere: validators simply fall back to absolute scoring and nobody is told.
UV=/root/.local/bin/uv
[ -x "$UV" ] || { echo "FATAL: uv not found at $UV" >&2; exit 1; }

cd "$REPO"
set -a; . ./.env; set +a

# --deadline-s must stay inside the epoch: a reference that outruns its epoch describes a slice
# nobody is scored on any more. Cells outstanding at the deadline count as failed and their rows
# drop, so a hung provider call costs coverage rather than the whole epoch.
exec >>"$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) publishing pool reference ==="
"$UV" run --frozen orchestra-koth-reference \
  --netuid "$NETUID" --network "$NETWORK" \
  --wallet "$OWNER_WALLET" --hotkey "$OWNER_HOTKEY1" \
  --n-per-bench 2 \
  --deadline-s 600 --call-timeout 120
