"""Every v3 pin, in one module (docs/WHITEPAPER.md §1/§5/§8b, the build plan M0).

They live together because they are read from opposite ends of the system — the parser, the
scaffold, admission, the duel and the emission schedule — and a constant that sits next to its one
consumer is a constant nobody audits as a whole. This module is also what a miner reads to know
exactly what they are training against, which is the "Kind" property in its cheapest form.

WHICH OF THESE ARE ONE-WAY DOORS, AND WHY EACH ONE IS ONE
---------------------------------------------------------
*One-way* here means: changing it after launch does not deprecate work, it destroys work.

* `REFERENCE_MODEL`, `REFERENCE_REVISION`, `REFERENCE_CONFIG_HASH`, `REFERENCE_TOKENIZER_HASH` —
  this IS the pinned architecture (§1). A miner ships ~70 GB of full weights against it and a hotkey
  gets exactly one submission ever, so rotating the reference invalidates every miner's model at
  once and spends every shot already taken. Risk register #8: "rotation = full reset".
* `DTYPE`, `ALLOWED_WEIGHT_SUFFIXES`, `REFUSED_WEIGHT_SUFFIXES` — §1.1. The dtype is pinned so that
  duels compare routing rather than numerics (two precisions are not the same experiment), and the
  format pin is what preserves "no miner code ever executes" through a full-weights upload.
* `MAX_STEPS`, `REASON_TOKEN_CAP`, `MAX_CONDUCTOR_TOKENS` — the policy's horizon. Every miner trains
  a model to act inside these bounds, so widening or narrowing them later re-scores everyone against
  an episode shape they did not train for.
* `MAX_READS_PER_DELEGATE` — for the same reason, one level down: it is the worker's horizon, it is
  stated in the task text a window commits to (`tools.PROTOCOL`), and miners train against the
  episode shape it produces. It must be settled before the first window a tool-enabled benchmark
  appears in, and it was — by census, 2026-09-01, at the constant. The five other read-channel caps
  are shape bounds and are ordinary knobs.
* `EPS` — one-way **downward only** (D10, §5.2b). Too large refuses real challengers; too small
  crowns noise, and only the second is irreversible in its effects on the ledger. Lowering it later
  is a safe governance loosening; raising it mid-flight is not.

Not one-way, and deliberately so: `N_BOOT`, the wall clocks, the queue caps and the emission shares.
All are governance knobs that can move at a window boundary without invalidating anything already
trained.

WHAT IS STILL PROVISIONAL. `MAX_STEPS`, `REASON_TOKEN_CAP` and `MAX_CONDUCTOR_TOKENS` are pinned
here because M1 onward is shaped by them, but M2a's cost probe (the build plan M2a exit 6) is the *last*
window in which they may move — it measures real turns and dollars per episode, and the plan is
explicit that downstream sizing comes from that probe rather than from its own estimate table. Once
a miner has trained against them they are frozen for good.
"""

from __future__ import annotations

# --- the reference artifact (§1, D7) --------------------------------------------------------------
# Qwen3.6-35B-A3B chosen on ACTIVE parameter count: the Conductor runs ~5,000-15,000 times per duel
# (~500 episodes x 10-30 steps), so serving cost tracks active params, and 3B active gives dense-3B
# economics at MoE capability. Apache 2.0 (miners must be able to redistribute derived WEIGHTS, not
# just use them), and small enough that a LoRA route exists for a miner without a cluster. What that
# route COSTS is deliberately not asserted here: it is the miner's own compute, bought wherever they
# like, and this project has never measured it. An entry-cost figure nobody has run is a promise made
# on a stranger's behalf.
REFERENCE_MODEL = "Qwen/Qwen3.6-35B-A3B"

# A SENTINEL, NOT AN EMPTY STRING. These three are pinned at M0 against the real artifact — the
# owner publishes the revision, the `config.json` hash and the tokenizer hash — and until then every
# admission check must refuse. The sentinel is truthy on purpose: an empty string would be skipped
# by the natural `if REFERENCE_CONFIG_HASH:` guard, turning "not yet pinned" into "not checked",
# which is the failure mode §1.1 exists to prevent. No real digest can collide with it.
UNPINNED = "UNPINNED"

# PINNED 2026-09-01 against the real artifact on HuggingFace. Reproduce with
# `scripts/pin_reference.py`, which is the only sanctioned way to change these three.
REFERENCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
REFERENCE_CONFIG_HASH = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
# sha256 over the SORTED tokenizer artefact set, each contributing (filename, sha256(bytes)) — not
# `tokenizer.json` alone. A tokenizer is `merges.txt` + `vocab.json` + `tokenizer.json` +
# `tokenizer_config.json` + `preprocessor_config.json` together, and swapping any one of them
# changes what a prompt tokenises to while leaving a single-file digest intact.
REFERENCE_TOKENIZER_HASH = "b852bad3930e0505d85e82d121c4d50856813ff4fb1d689d6b78a13708aaaf07"
REFERENCE_TOKENIZER_FILES = ("merges.txt", "preprocessor_config.json", "tokenizer.json",
                             "tokenizer_config.json", "vocab.json")

# WHAT THE REAL ARTIFACT TURNED OUT TO BE, because two properties of it are load-bearing and neither
# was assumed when this file was first written.
#
# 1. IT IS MULTIMODAL. `Qwen3_5MoeForConditionalGeneration` carries a 27-layer vision tower and
#    image/video token ids alongside the text model — 1,045 tensors over 26 shards, 71.9 GB, 36.0B
#    parameters at bf16. Admission (§1.1) must therefore verify the VISION tensors too: they are
#    part of the pinned architecture whether or not routing ever uses them, and an inventory that
#    covered only `text_config` would leave a quarter of the tree unchecked.
#
# 2. IT IS A HYBRID ATTENTION/SSM MODEL, not dense attention. 40 layers with
#    `full_attention_interval = 4` means only 10 carry a growing KV cache; the other 30 are linear
#    layers with a CONSTANT-size recurrent state. With `num_key_value_heads = 2` and
#    `head_dim = 256` the KV cost is 2*10*2*256*2B = 20 KB/token — about ONE TENTH of the ~190
#    KB/token this project assumed when sizing validator hardware. `mamba_ssm_dtype = float32` is
#    also a determinism input (§8b.4): the SSM state is the other place a "same" model can diverge
#    run to run, alongside MoE expert routing.
REFERENCE_TENSOR_COUNT = 1045
REFERENCE_WEIGHTS_BYTES = 71_900_000_000

# Pinned dtype (§1.1 point 3): quantisation changes quality, so two models uploaded at different
# precisions are not the same experiment and a duel between them would compare numerics, not routing.
DTYPE = "bfloat16"

# §1.1 point 2, and risk register #10. `.bin` / `.pt` / `.pth` are pickle archives, and loading one
# runs arbitrary code ON THE OWNER'S VALIDATOR — the single machine that holds the subnet's scoring
# authority. `safetensors` is a pure tensor container with no code path. This is not a preference
# about file formats; it is what carries the "no miner code ever executes" property (§2.1) through
# the switch from adapters to full-weight uploads. Refused, never skipped: a refusal list rather
# than a bare allow-list so the error a miner sees names the actual problem.
ALLOWED_WEIGHT_SUFFIXES = (".safetensors",)
REFUSED_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth")

# --- the episode (§3, §8b.2) ----------------------------------------------------------------------
# Greedy, pinned (§1). Sampling would make the king's arm a different measurement every window and
# would put run-to-run noise inside a paired comparison that exists to remove it.
TEMPERATURE = 0.0

# NO CHAT-TEMPLATE KNOB IS PINNED HERE, AND THAT IS DELIBERATE — measured 2026-09-01 against the
# live model. Asked via /v1/chat/completions to "Reply with exactly: DELEGATE openai/gpt-5.4":
#     default (thinking ON) -> "Here's a thinking process:\n\n1. **Analyze User Input:** ..."
#     enable_thinking=False -> "DELEGATE openai/gpt-5.4"
# So on the CHAT endpoint the model's native thinking mode consumes the whole token budget before
# reaching an action line, making every turn a parse failure (§2) — three of those force STOP, so
# every episode would end having decided nothing while the duel still ran and still crowned someone.
#
# `serve.py` avoids this at the root rather than with a flag: it calls /v1/completions, which
# applies no chat template at all. That is the stronger fix for an independent reason — the bytes
# `render()` produced are the bytes the model reads and the bytes `StepRecord.rendered_state_digest`
# commits to, whereas a chat template would silently rewrite them, and its kwargs' defaults differ
# across serving stacks and versions. A pinned `enable_thinking` here would be a constant nothing
# reads, which is how `sv::` became dead weight charged against a param cap.

# Conductor turns per episode. A fixed cheap->strong cascade needs 3-4; 12 leaves room for a
# genuinely exploratory policy while bounding the worst case at 11 worker calls on one task, which
# is what a pathological model would otherwise spend of the king's allowance every episode.
MAX_STEPS = 12

# Internal reasoning per turn (§2). Never forwarded to a worker, never seen by anything but the
# Conductor itself — capped because §8b.2's pathological model is one that emits MAX_TOKENS of
# REASON at every step and never reaches an action.
REASON_TOKEN_CAP = 512

# The per-Conductor-call generation cap (§8b.2: on breach, truncate; unparseable output counts as a
# parse failure). MUST exceed `REASON_TOKEN_CAP` with room for the action line: a Conductor that can
# spend its entire generation budget reasoning can never emit an action, so every turn truncates
# into a parse failure and the episode dies at MAX_CONSECUTIVE_PARSE_FAILURES having decided
# nothing. That is a silently fatal interaction between two independent-looking pins, so the
# headroom is asserted in tests/test_action.py rather than left to whoever edits one of them.
MAX_CONDUCTOR_TOKENS = 576

# §2: anything that is not exactly one valid action is a parse failure — the step is refused,
# counted, and the episode continues. Three CONSECUTIVE failures force STOP, because a model that
# cannot emit a parseable action is not routing, and letting it burn the full step budget is a drain
# on the validator (§8b.2) that buys no information. Consecutive rather than cumulative: a policy
# that fumbles one turn and recovers is still routing.
MAX_CONSECUTIVE_PARSE_FAILURES = 3

# §8b.2's outer bounds, from §5.4's ~15 min/task and D6's 2-hour duel. On breach the episode (resp.
# the remaining tasks) is abandoned and scored 0 FOR THAT ARM — a pathological model must not be
# able to stall the validator or hold the window open.
EPISODE_WALL_CLOCK_SECONDS = 900.0
# COUPLED TO `MAX_READS_PER_DELEGATE` BELOW, and the coupling is the design's largest cost. A read
# turn is one extra worker round trip plus one container — MEASURED at 0.26-0.32 s, not the 0.177 s
# this arithmetic was written against, which moves nothing because the container was never the
# dominant term — so ≈5.3 s, and a 3-delegate episode with three reads each runs ~63 s against
# today's ~16 s. At N = 3 with ~83 tool-enabled tasks in a 250-task slice that is 83x63 + 167x16 =
# 7901 s, about 10% over this bound; restoring margin means 9000.0, which is a governance move at a
# window boundary (the wall clocks are explicitly not one-way doors) and which then makes
# `MAX_DUELS_PER_WINDOW` ~6 and `MAX_QUEUE_DEPTH` 12 by their own derivations. §8b.1's obligation is
# a DRAIN TIME, so that rule is unchanged; only the headcount moves.
#
# WHAT DOES MOVE THE ARITHMETIC is FIND's tree walk, priced for the first time by the same
# measurement (the sandbox record §7.4): the median call costs 1.3 s but a no-match walk of the
# largest repositories in the corpus costs 8-18 s, so a three-read delegate there is ~31 s rather
# than ~16 s and an all-FIND episode on them ~110 s. `TOOL_TIMEOUT_SECONDS` bounds that per call and
# the two wall clocks bound it per episode and per arm; a task an arm never reached is published
# with `DUEL_WALL_CLOCK_REASON` rather than scored (§5.8), so the tail is visible rather than silent.
#
# THE ALTERNATIVE THAT NEEDS NO GOVERNANCE CHANGE — two reads per delegate — IS NOW MEASURED AND
# REFUSED: the census beside `MAX_READS_PER_DELEGATE` puts two reads at 71.1% of the corpus served
# against 92.2% at three. It would save a REVERSIBLE clock move by spending 21.1 points of the
# corpus on an IRREVERSIBLE pin, which is the wrong way round.
#
# MOVED 7200 -> 9000, 2026-09-01, and the trigger was a measurement rather than the read budget
# alone. Arrangement (B) (the sandbox record §1.3a) sends every container across an ssh transport:
# **0.84 s each multiplexed, 2.73 s without**, measured 5 reps. At ~1000 containers an arm that is
# ~830 s ON TOP of the read budget, so an arm needs 7901 + 830 = **8731 s** and 7200 could not hold
# it. Unmultiplexed the same arm is ~10,600 s and NO plausible clock covers it, which is why
# `ControlMaster` is a deployment requirement and not a tuning note.
#
# *** THAT 8731 s IS AN UNDER-COUNT, AND 9000 DOES NOT HOLD ARRANGEMENT (B) TODAY. *** Corrected
# 2026-09-02. It was measured with a LOCAL grade dir, so the payload's own filesystem cost was
# never in it. Under (B) `V3_GRADE_DIR` is a shared mount, and one read issues ~30 filesystem
# round trips (mkdtemp, payload mkdir, two writes, the log's create/write/stat/tail/re-read,
# rmtree). Measured on the real mount: **0.24 ms per read local, 1652 ms over sshfs** at 94.66 ms
# per round trip — a 6949x penalty, or **~1650 s per arm at ~1000 reads**. The honest total is
# ~10,383 s against this 9000 s clock.
#
# DO NOT RAISE THE CLOCK TO PAPER OVER THIS. The cost is ~95 ms of geography per round trip, and
# raising the clock cuts `MAX_DUELS_PER_WINDOW` further for a constant nobody should be paying. The
# fixes are, in order: move the container log off shared storage (it is a host-side `stdout=`
# redirect the daemon never opens, so this is also a containment improvement) which removes about
# half the round trips; and then the split the deployment note already names as the destination —
# an episode runner ON the sandbox host with verdicts and weights here, where the grade dir is
# local again and the figure returns to 0.24 ms. See the sandbox record 1.3a.
#
# WHAT IT COSTS, because this is not free: a 24-hour window holds 86400/9000 = 9.6 arms instead of
# 12. Net of the king's arm and the two reference arms (~a sixth) that is 8.0 available where it was
# 10.0, so `MAX_DUELS_PER_WINDOW` drops 8 -> 6 to keep the same ~75-80% loading and leave the ~70 GB
# model pull its headroom. **The subnet judges 25% fewer challengers per day.** That is the price of
# the read channel, paid on a knob this file calls reversible, and it is the cheaper of the two
# available prices — the other was 21.1 points of corpus on a one-way pin.
DUEL_WALL_CLOCK_SECONDS = 9000.0

# The `stopped_reason` a task carries when the per-duel clock abandoned it (§8b.2). DISTINCT FROM THE
# EPISODE CLOCK'S `"wall_clock"` ON PURPOSE, and that is the whole reason it is pinned: an episode
# that stalled is a fact about the challenger's model, while a task the arm never reached is a fact
# about the clock, and §5.8's rule — a verdict decided by something other than routing must be
# visible in the reveal rather than buried in an aggregate — covers both. One reason string for both
# would make a pathological model and an overrunning validator the same row in the published trace.
# It lives here rather than inline beside the other reasons because the scaffold writes it and the
# reveal reads it, and a literal two modules must agree on is a literal that eventually drifts.
DUEL_WALL_CLOCK_REASON = "duel_wall_clock"
# The `stopped_reason` a task carries when §5.4's futility check abandoned it — DISTINCT from both
# clocks and from `budget_exhausted` for the reason §5.8 gives: a verdict decided by something other
# than routing must be legible in the reveal rather than buried in an aggregate. These four now
# describe four different findings — the model stalled, the validator ran out of clock, the miner
# ran out of money, and the arm could no longer win — and one string for any two of them would
# publish them as the same row.
#
# Not a knob. Whether to stop is decided by an EXACT bound (`score.best_possible_final`), so there
# is no threshold to tune and no quantile to choose: the arm is abandoned only when no continuation
# whatsoever could clear `EPS`. A tunable here would be a way to abandon arms that could still win,
# and §7 gives a hotkey one submission ever.
FUTILE_REASON = "futile"

# --- the worker read channel (§2.1, `tools.py`) ---------------------------------------------------
# Reads one WORKER may make inside ONE delegate, before its reply is taken as the submission. It
# exists because of a measurement: 0 of 75 R2E-Gym episodes produced a patch `git apply` would
# accept, while the gold patch scored 1.0 on 24 of 25 of the same tasks through the identical grader
# (the sandbox record §7). The patches were well-formed diffs with invented context lines, and
# `git apply` matches context, so the missing input is the bytes of the file.
#
# PER DELEGATE, NEVER PER EPISODE, and that is §2.1 one level down: the transcript is a local
# variable inside one `_call_worker` call, so no Conductor decision — not RETRY, not its timing —
# can change what any worker sees. The worst case is `MAX_STEPS - 1 = 11` delegates x 3 reads = 33
# read turns in one episode, which `EPISODE_WALL_CLOCK_SECONDS` already bounds at ~220 s.
#
# ONE-WAY in the sense `MAX_STEPS` is: miners train against it, so it must be settled before the
# first window a tool-enabled benchmark appears in. The five caps below are shape bounds and are not.
#
# PINNED AT 3 BY THE FILES-PER-GOLD-COMMIT CENSUS the paragraph above used to call for — all 6,812
# gradeable R2E-Gym tasks, offline from the parquet the adapter already downloads, no containers and
# no model calls (the sandbox record §7.4). It answered a different question than the one it was
# set, and that is the useful half:
#
#   * FILE COUNT IS NOT WHAT BINDS. The gold patch touches a median of ONE non-test source file
#     (mean 1.10, p95 2, max 4; 89.7% touch exactly one), so three would be generous on that axis.
#   * LOCALISATION IS. The `[ISSUE]` text the worker is actually shown names a path it could derive
#     — the repo-relative path (0.15% of tasks), the basename (0.43%) or an importable dotted module
#     (16.65%) — on only 16.9% of them. On the other ~84% the first read is spent FINDING the file,
#     so the budget arithmetic is 1 FIND + reads and never reads alone.
#   * ONE FILE IS NOT ONE READ, but nearly. Covering the gold hunks' line ranges with the window a
#     READ really returns needs a median of 1 read (mean 1.43, p90 2): 67.3% need one, 91.4% two,
#     98.2% three.
#
# SERVED FRACTION AT THIS PIN, which is the number that decides it and is recorded whichever way it
# fell: 92.2% of the corpus under the realistic model (one FIND when the issue names no file, then
# the covering READs), 87.3% under the pessimistic one (a FIND per unnamed file), 98.2% if the
# worker were simply told where to look. Cutting to 2 serves 71.1% — it buys back a reversible
# governance move (`DUEL_WALL_CLOCK_SECONDS`) by spending 21.1 POINTS OF THE CORPUS on a pin that
# cannot be spent back. Going to 4 buys +6.1 points (to 98.3%) for 33% more read wall clock, and
# `OBSERVATION_BYTES` below buys about half of that for none.
#
# WHAT 3 DOES NOT SERVE, plainly: 7.8% of tasks — the >1-source-file and ≥3-dispersed-hunk ones.
# They are not lost, they are unaided: the delegate still answers and is still graded, with partial
# credit live on 27.9% of the corpus. And the census's own open question is whether ONE FIND
# localises at all — `FIND_MAX` stops the walk at 50 hits in sorted order, so a common string can
# fill the budget before reaching the file. If it often does, the honest column is the pessimistic
# 87.3% and the case for a 4th read is stronger than this comment makes it.
MAX_READS_PER_DELEGATE = 3

# What one observation may add to the NEXT turn's prompt — the read loop's dollar cost, since a read
# turn's other half is ~20 output tokens rather than a patch. 4 KB ≈ 1,024 tokens, so three reads add
# ~6,000 cumulative input tokens: +$0.0063 on a strong-tier delegate against a $0.0169 one-shot
# baseline, i.e. +37% on a 3-delegate all-strong episode ($0.051 -> $0.070). `final_b` already prices
# that out of the miner's own allowance, so no λ_b/C_b refit is required. Truncation is MARKED, never
# silent (`tools.TRUNCATED`).
#
# MEASURED AT ALL 14,102 GOLD HUNK STARTS: this is the cap that actually binds a READ — one read
# returns a median of 96 lines and never once reached `READ_LINES` — and raising it to 8192 takes
# the served fraction at three reads from 92.2% to 95.2% and at two from 71.1% to 76.3%, for ~+$0.001
# per delegate and no wall clock at all. Beyond 8192 there is nothing left to buy (mean reads needed
# 1.32 at 8192 and 1.32 uncapped).
#
# MOVED 4096 -> 8192, 2026-09-01. It was described here as "the cheap half of a 4th read", and that
# is exactly what it buys: **+3.0 points of served corpus (92.2% -> 95.2%) at three reads**, for
# ~+$0.001 per delegate and NO wall clock — where a 4th read buys +6.1 points and costs 33% more
# read time on an arm that is already 270 s from its ceiling. Spent before the read budget, as the
# note said it should be. Beyond 8192 the measurement says there is nothing left, so this is the end
# of that road rather than a step along it.
OBSERVATION_BYTES = 8192

# Bounds READ's container-side loop, and makes the reply shape predictable: the worker needs the line
# NUMBERS to write a diff that applies, which is the whole point of the channel.
#
# WAS DEAD, IS NOW LIVE — and the prediction that it would become live is what makes it worth
# recording. At `OBSERVATION_BYTES = 4096` the byte budget stopped the read first on 100% of the
# corpus's gold hunk windows (max 193 lines returned against this 200), so this constant bounded
# nothing; the note here said it was kept "because it becomes live the moment the byte budget is
# raised". The byte budget was raised to 8192 on 2026-09-01 and it did: **36.1% of gold hunk windows
# now reach 200 lines** and are cut by this cap rather than by bytes. So it is doing work for the
# first time, and it is the constant to revisit — not `OBSERVATION_BYTES` — if reads come back short.
READ_LINES = 200

# Bounds LIST's directory loop. LIST is the one verb whose necessity is assumed rather than measured
# (`tools.PROTOCOL`): 6849 of 8101 R2E-Gym rows carry a generated `[ISSUE]` block, and a worker whose
# issue text contains no literal FIND can hit has no other way to orient. The check that would settle
# it is free and offline — the fraction of prompts containing a token that appears verbatim in the
# tree — and if FIND hits on substantially every task the tool set is two verbs.
LIST_MAX = 200

# Bounds FIND's tree walk: it stops at 50 hits rather than filling a buffer it will then truncate.
FIND_MAX = 50

# The sandbox clock for ONE read. A breach is a `SandboxError` and therefore an exclusion from both
# arms (§8b.3), never a zero.
#
# MEASURED 2026-09-01 THROUGH THE SHIPPED PATH — `tools.observe` -> `tools.in_sandbox` -> the real
# `READER`, the real `sandbox.run` flags and these caps — over seven R2E-Gym images spanning the
# corpus (1.14-6.82 GB, sympy and pandas among them; the sandbox record §7.4). 30.0 did NOT breach:
# the worst single call anywhere was 18.39 s, a no-match FIND on the largest tree (orange3, 36,538
# files / 2,034 MB) with a cold page cache, i.e. every byte decoded. What the measurement changes is
# the SHAPE of the risk, not a broken bound:
#
#   * LIST and READ are 0.26-0.37 s whatever the tree, and are 100% container start — the reader's
#     own work is below the CLI's ~50 ms quantum even for a 1,045-entry listing or a 13,010-line
#     file. Nothing in this clock is about them.
#   * FIND is a linear CPU-bound walk at 141-207 MB/s: `0.3 s + tree_bytes / 160 MB/s` predicts every
#     row measured, across a 22x range of tree sizes. So the number that decides this pin is the
#     biggest `/testbed` a benchmark may ship and the single-core throughput of a host §8b.9
#     deliberately leaves unspecified.
#
# 90.0 = 18.39 s (measured worst) x 1.2 (the corpus tags not sampled) x 3 (a sandbox host at a third
# of this box's decode throughput) = 66 s, rounded up: 4.9x on the corpus maximum, admitting a ~14 GB
# repository. The one input that does not transfer is the box — a 30-core EPYC with flash-backed
# storage where eight concurrent walks cost nothing — which is exactly what the 3x is for.
#
# WHAT EACH DIRECTION COSTS, because they are wrong in different currencies and that asymmetry is
# the whole argument. TOO LOOSE spends wall clock on a container already useless, and it is bounded
# twice: a breach ends that delegate's read loop (`_GraderGuard.inspecting` returns None), so the
# cost is one timeout per delegate and not per read, and one 90 s hang overshoots
# `EPISODE_WALL_CLOCK_SECONDS` by ≤10%. It does make a DELIBERATE stall dearer — 11 delegates x 90 s
# reaches the 900 s episode clock, where 30.0 capped the same episode at ~330 s — but a model stalls
# the arm it is running in, `DUEL_WALL_CLOCK_SECONDS` ends that arm either way, every unreached task
# is published with `DUEL_WALL_CLOCK_REASON` rather than scored, and a KING arm that hits the clock
# settles the window DEAD rather than deciding it (§8b.2).
# TOO TIGHT drops the task from BOTH arms and from the denominator, and does it NON-RANDOMLY: cost
# is linear in tree size, so a tight clock preferentially deletes pandas (21.2% of gradeable rows)
# and orange3 (7.1%) — it shrinks the slice exactly where the repositories are largest, which is a
# bias in what the arena measures rather than noise in it. 60.0 is the defensible conservative
# alternative (3.3x); at or below 45 the margin is being spent on a host nobody has measured.
TOOL_TIMEOUT_SECONDS = 90.0


# --- the queue (§8b.1) ----------------------------------------------------------------------------
# Duels one window runs. Beyond it the remainder ROLL OVER with their shot unspent: a queue is never
# dropped, only deferred, because dropping a challenger spends the one thing §7 says a hotkey has
# exactly one of.
#
# DERIVED FROM THE VALIDATOR'S WALL CLOCK RATHER THAN CHOSEN. Challenger arms are serial on the
# single owner-run validator — the king and one challenger are already resident at ~70 GB each (risk
# register #7), so a second challenger does not fit — and each arm is bounded by
# `DUEL_WALL_CLOCK_SECONDS`, now **2.5 hours**. A 24-hour window is therefore 9.6 arms; the king's
# arm (§5.2a) and the two reference arms (§5.6, ~60 tasks against a duel's ~250) take about a sixth
# of them, leaving 8.0, and each challenger arm is preceded by pulling and loading a ~70 GB model
# tree. Six leaves that headroom instead of pretending the download is free.
#
# RE-DERIVED 8 -> 6, 2026-09-01, because it was never a chosen number: it follows from the clock,
# and the clock moved to 9000 s to fit the read channel plus (B)'s ssh transport. At the old two-hour
# arm the window held 12 arms, 10.0 net, and eight of those was 80% loading; eight of 8.0 would be
# 100% and the model pull would have nowhere to fit. Six of 8.0 is 75%, which restores it.
#
# `MAX_QUEUE_DEPTH` follows automatically below, and the owner's chain-side obligation is stated in
# WINDOWS rather than blocks precisely so it stays correct through a move like this one — see the
# note there. The visible cost is throughput: **six challengers judged per window instead of eight.**
EPISODE_CONCURRENCY = 16
"""Episodes in flight per arm. NOT a one-way door, unlike this file's header constants.

An episode is independent of every other episode — same catalog, same history, same grader — so this
changes how many run at once and nothing about the shape a miner trains against; `1` reproduces a
serial arm exactly (pinned by test). §5.4 sizes ~250 tasks per arm to about two hours ASSUMING
parallelism, and measured 2026-09-03 an episode takes 45 s to 1.4 min: serially that is 3-6 h against
`DUEL_WALL_CLOCK_SECONDS = 9000`, so a serial arm cannot finish a full slice at all.

What widens with it, both bounds the serial arm already had: the allowance can overrun by up to this
many DELEGATEs rather than one, and a crash can lose this many in-flight episodes rather than none.
16 is chosen well below §5.4's "~40-way" so the overrun stays a small multiple of one episode's
spend; raise it only against a measured provider rate limit, since beyond that concurrency buys
429s rather than throughput.
"""

# LOWERED 6 -> 3, 2026-09-11, because 6 was a number the clock could not honour. Windows 1 and 2 on
# netuid 99 ran two 60-task reference arms in 26 and 36 minutes, which puts a 250-task served arm
# near 1.5 h; against a 5.78 h window (the most `immunity_period = 5000` permits) the king's arm and
# the two reference arms take ~2.1 h and leave ~2.5 duels. Claiming 6 made `check_launch` certify a
# drain of two windows while the real one was nearly five — immunity covering 17 h against a 28 h
# wait, which is §8b.1's trap with the gate reporting green.
#
# It is the CLAIM that was wrong, not the window: `MAX_QUEUE_DEPTH` is derived from this number, so
# lowering it shortens the promised drain to something the clock delivers. Raise it again when the
# window does — at `immunity_period >= 21600` a 7200-block window fits ~8.5 and 6 becomes true.
UNCLAIMED_GRACE_SECONDS = 7 * 86_400
"""How long bytes nobody ever committed to are kept (§8b.7's gap, reachable once credentials issue
themselves).

§8b.7 deletes a LOSER's weights a fortnight after judgment, which presumes every upload is judged.
An upload that never gets a ready signal never is: it is not in any queue, no window ever sees it,
and nothing deletes it. That was tolerable while a credential took an owner's attention per miner;
with issuing automatic, every registered hotkey can park ~70 GB in a bucket the owner pays for, and
256 uids makes that ~18 TB nobody is accountable for.

Seven days rather than the fortnight a judged loser gets, because the two are not the same promise:
a judged submission earned its grace by being scored, and this one is a prefix somebody started
filling and walked away from. It is measured from the last object WRITTEN, so a 70 GB upload that
takes two days keeps resetting its own clock and is never swept mid-flight.
"""

KING_ZERO_NAME = "thirtyspokes-genesis"
"""What King₀ is CALLED, as against `V3_KING0` which is the policy that implements it.

Two names because they answer different questions and drift apart the moment either moves: the
policy id (`cascade`) selects an arm and belongs to the world the validator was pinned against, while
this is the identity a reader sees on the throne. A dashboard that printed the policy id was telling
miners the genesis king is a routing strategy, which is true and useless — they need to know which
king they have to beat.
"""

MAX_DUELS_PER_WINDOW = 3

# Challengers that may be queued at once — and what it is DERIVED FROM is the point of it.
#
# §8b.1's deregistration trap: a challenger registers a hotkey, pays the registration burn, uploads
# ~70 GB and then waits. Bittensor's immunity period protects a UID earning nothing for a fixed
# number of blocks and no longer, so a queue that takes longer than immunity to drain deregisters
# challengers BEFORE they are ever evaluated. They paid and got nothing, which is the "Kind" property
# broken at its most expensive point.
#
# So the cap is a DRAIN TIME wearing a headcount's clothes: two windows from the back of the queue to
# the front. That makes the owner's chain-side obligation one legible sentence — set
# `immunity_period` to cover THREE windows, one to wait for the next window to open and two to drain
# — and it stays correct when the window length moves, where a bare `16` would silently stop being
# safe. Commits beyond the cap are REFUSED rather than accepted (§8b.1, the build plan M9 exit 10): taking
# a burn for a slot that will expire before it is judged is the outcome the cap exists to prevent,
# and accepting the money first makes it worse rather than kinder.
MAX_QUEUE_DEPTH = 2 * MAX_DUELS_PER_WINDOW

# --- the verdict (§5.2, D10) ----------------------------------------------------------------------
# On the `final` scale (quality minus priced spend) — five points of benchmark score.
#
# IT DOES NOT INHERIT THE ARENA'S 0.02. That value was calibrated by `scripts/arena_calibration.py`
# against *capture*, a ratio with a different scale and a different null distribution; carrying it
# across would be the exact class of mistake the build plan §0 forbids. 0.05 is conservative on purpose
# (§5.2b): at ~250 tasks per arm it behaves like a ~7-8 point bar, which is also what makes a
# derivative of the king have to add real points rather than epsilon (D14).
#
# The number to watch is `band / eps`: it bounds how many coronations this arena can EVER have,
# because once a challenger sits within eps of the achievable ceiling every later duel lands inside
# the margin and the crown ossifies. The band is unknown and cannot be borrowed from this repo's
# single-turn measurements; production reference arms (§5.6) measure it.
EPS = 0.05

# The floor a per-benchmark delta must clear to count toward BREADTH (§5.2 conditions 2 and 3;
# `duel.DuelVerdict.counted`). NOT applied to condition 1 — the aggregate reports what a challenger
# actually did; breadth asks the different question of where it moved something real.
#
# WHY IT EXISTS. Both breadth conditions gate at a strict `> 0` and `final_b` is CONTINUOUS IN
# SPEND, so on a benchmark it routes identically on, a challenger manufactures a positive delta by
# spending a hair less. Measured by the red team: the king plus two benchmark overrides plus a
# **1e-5 dollar** spend reduction on four others is crowned, median +3.03e-06. At EPS/10 a
# manufactured 1e-6 movement is zeroed while a real 0.03 one survives — three orders of margin
# either side, so the constant is not delicately tuned.
#
# *** WHAT THIS DOES NOT FIX, AND THE ATTEMPT THAT PROVED IT. ***
# The same defect also appears as a RATE: with ordinary measurement noise instead of a fixture's
# exact zeros, a pure one-benchmark specialist is crowned 38-49% of the time, because both breadth
# conditions become coin flips on the sign of noise. The obvious fix — floor at k x the per-benchmark
# bootstrap SE — was implemented and MEASURED TO BE WRONG: at ~20 tasks per benchmark that SE is
# order 0.11, so a 1-SE floor zeroes a genuine uniform +0.03 challenger along with the noise. It was
# refused by `test_a_uniform_three_point_challenger_is_refused_by_eps_not_by_breadth`, which is
# exactly the "refuses real challengers" failure §5.3 forbids.
#
# So the noise-driven coin flip is a SAMPLE-SIZE problem, not a threshold problem: no per-benchmark
# threshold can separate a 0.03 signal from a 0.11 standard error. It is fixed by tasks per
# benchmark, which makes the breadth rule and the slice size ONE decision. The per-benchmark SE is
# published every window (`DuelVerdict.noise_floor`) so the gap stays visible rather than implicit.
BREADTH_FLOOR = EPS / 10.0

# Bootstrap resamples for the paired lower bound. Carries over from `koth/duel.py`, where the
# calibration's finding was that THE BOOTSTRAP DOES THE WORK, NOT EPS — the lower bound refuses a
# delta it cannot separate from zero, which is what kept the measured false-positive rate at 0.0000.
N_BOOT = 1000

# --- emissions (§5.7, D9) -------------------------------------------------------------------------
# The king plus a five-deep decaying pension, most recent ex-king first. Deliberately shallow: at
# 0.05 a former king earns about a seventeenth of the crown, so defending is worth vastly more than
# having once won. Unfilled or unresolvable slots BURN and are never reflowed to the king — folding
# them in would hand the first winner 100% of emissions, which is the outcome the pension exists to
# soften.
EMISSION_KING = 0.85
EMISSION_PENSION = (0.05, 0.04, 0.03, 0.02, 0.01)


# How long a caller waits for the FIRST call of a gateway's life to settle and publish a price
# (`OwnerGateway.call`). Before any call settles there is no per-call ceiling to reserve against, so
# the first reserves the whole balance; a concurrent caller waits for it rather than being refused as
# unfunded, which is a different and much worse thing to tell a miner who has paid. Bounded so a hung
# provider degrades to the ordinary refusal instead of parking an arm: the episode and duel clocks
# (§8b.2) are the outer bounds, and this sits well inside both.
PRICE_PROBE_WAIT_SECONDS = 30.0


# How much of its slice an arm must reach before it may take the CROWN (§4, §5.8, §5.2).
#
# The zero-spend guard this replaces closed exactly one point — an arm that spent $0 — and an
# allowance of one nanodollar walked straight past it: the gateway checks `balance <= 0` BEFORE a
# call and debits after, so any positive balance buys one metered call. That arm answers almost none
# of the slice, scores ~0 quality at ~0 spend, and `final = quality - lam*spend/C` puts it at ~0.0 —
# which beats any king whose priced spend exceeds its quality. Measured: a $1e-9 allowance crowned a
# do-nothing challenger with 0.85 of emissions.
#
# THIS BOUNDS THE CROWN, NOT THE SCORE, and the distinction is §5.8's. §4 decides what happens to a
# starved arm's tail — the remainder is zeroed — and §5.8 requires that cut to be PUBLISHED rather
# than hidden, so that a verdict decided by funding is legible instead of reading as skill. Refusing
# to score such an arm would defeat both. What must not follow from an unfunded arm is a CORONATION:
# a challenger may win by buying the same quality for less, and may not win by declining to buy
# anything. So the arm is run, scored and published exactly as before, and is simply not eligible to
# take the throne.
#
# A half is chosen rather than tuned, and the fixture shows why no sharper line exists: a legitimate
# starved arm reached 1 of 9 tasks and the nanodollar attack reached 0 of 9. Those are indistinguish-
# able by participation, which is why participation gates eligibility and not scoring.
MIN_SLICE_REACHED = 0.5


# The `stopped_reason` a task carries when §6.3c had already dropped it before this arm reached it.
# Distinct from the other endings for §5.8's reason: it is not a fact about the model, the clock or
# the miner's wallet — it is a task the WINDOW removed, and it will leave the denominator entirely.
EXCLUDED_REASON = "excluded"


# §5.2c's drift audit. Per window the owner re-buys a few of the outcome table's HIT keys live and
# the reveal publishes how often the fresh draw disagreed with the stored one — the gate that any
# cross-window reuse of rows would need, run from day one so its baseline exists. Diagnostic in v1:
# it gates nothing. Bounded twice, by count and by dollars, because it is owner money spent on a
# question the window has already answered.
RETEST_MAX_KEYS = 8
RETEST_MAX_USD = 1.0
# A re-bought cost is "the same" within this relative tolerance. Token counts move run to run even
# at temperature 0 (measured 2026-09-06: six identical calls to one task, six distinct answers), so
# exact equality would report the provider's own nondeterminism as drift.
RETEST_COST_TOLERANCE = 0.25

# The subnet's public store: the custom domain on the owner's production R2 bucket (connected
# 2026-09-08). Miners poll their credential envelope there and read window files and reveals from
# it; the dashboard reads reveals from it. The tools default to it; a rehearsal passes its own
# `--mailbox-url`. Not the bucket's r2.dev address — Cloudflare rate-limits that one and calls it
# unfit for production, and the zone's bot protection refuses Python's default User-Agent on both.
PUBLIC_STORE = "https://store.thirtyspokes.ai"

# The owner's mailbox public key on netuid 99 — ed25519, hex. What `thirtyspokes-owner key` prints
# from the seed in `<state>/mailbox-key.hex` (generated 2026-09-10), and the miner tools' default
# `--owner-key` on the subnet named below.
#
# PINNED BECAUSE THE KEY A MINER SUPPLIES IS THEIR ENTIRE TRUST ANCHOR. `access.open_credential`
# verifies an envelope against the key its caller passes and ignores the `owner_identity` the
# envelope carries — correctly, since anyone can sign an envelope and name themselves in it. So
# whoever hands a miner their owner key decides what that miner trusts: an impersonator's key makes
# `submit` open the impersonator's envelope, and makes `register-key` seal the miner's funded
# OpenRouter key to the impersonator and upload it to a store anyone can read. Handing the key out
# one miner at a time is exactly the channel an impersonator can fake; shipped inside the code the
# miner is already running, faking it means compromising this repository.
#
# PUBLIC, and safe to publish by construction. The secret is the SEED behind it, which never leaves
# the owner's state directory: it opens every OpenRouter key ever sealed to this value.
#
# ONE-WAY IN PRACTICE. Rotating the seed orphans every key sealed to the old value (VALIDATOR.md
# §3b), so this changes only together with every miner re-registering their key, in the same release.
OWNER_MAILBOX_KEY = "ae6cea3711968cc229e79ce2d6e76ac789ad974e908974ac1fca55eb10ca3d21"
# The subnet that key belongs to, as `(network, netuid)`. The default applies HERE ONLY: a testnet
# rehearsal that forgot `--owner-key` would otherwise seal a miner's funded OpenRouter key to
# netuid 99's owner without a word, so anywhere else the flag is required.
OWNER_MAILBOX_SUBNET = ("finney", 99)
