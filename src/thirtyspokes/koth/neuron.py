"""Deployable neuron wiring + an offline decoupled demo (`orchestra-koth-local`).

Proves the REAL architecture (distinct from the sim): the miner produces + uploads
its attested proof to its repo; the validator independently polls the chain,
downloads each bundle + proof, and verifies/grades/scores — the two never call each
other. The offline demo uses a file-backed chain + local store + mock TEE so it runs with no
infra; production swaps in BittensorChain + real DCAP attestation (unchanged control flow).
"""

from __future__ import annotations

import tempfile
import threading
import time
from collections import defaultdict

from ..gateway.signing import Signer
from ..reign import KingChain
from ..subnet.chain import LocalFileChain
from ..tee.attestation import Platform
from .benchmarks import default_suite
from .epoch import EPOCH_BLOCKS, current_epoch
from .miner import KOTHMinerNeuron
from .proof import Proof
from .runtime import runtime_measurement
from .store import LocalBundleStore
from .validator import KOTHValidator

# Bittensor mainnet aliases. `--insecure` (mock TEE accepted, measured-image gate skipped) turns the
# whole security model off, so the validator daemon REFUSES it on these networks.
MAINNET_NETWORKS = frozenset({"finney", "mainnet"})

# RE-SUBMIT AN UNCHANGED SLATE THIS OFTEN. Weights used to be written only when the DISTRIBUTION
# changed, and the KingChain is built to hold one — eps hysteresis and incumbency exist precisely so
# the crown does not turn over on noise. So a healthy subnet is the case that breaks it: measured on
# netuid 99, the validator scored 20+ consecutive epochs correctly while `last_update` aged 2234
# blocks without a single submission. Past the subnet's `activity_cutoff` (5000 there) Yuma stops
# counting the validator entirely, so a perfectly-behaved validator disappears from consensus for
# being too stable. 180 blocks sits above the 100-block `weights_rate_limit` (a refresh is never
# rejected for arriving too soon) and far below any cutoff worth supporting.
WEIGHT_REFRESH_BLOCKS = 180


def validator_hf_token(environ=None) -> str | None:
    """Credential used only by the validator's publishing side.

    Miner uploads and owner publications are deliberately separate identities.  The official
    validator is owner-operated, so it may reuse the owner's org-scoped token to publish the public
    dashboard feed.  ``HF_TOKEN`` remains a compatibility fallback for third-party deployments.
    """
    import os

    env = os.environ if environ is None else environ
    return env.get("OWNER_HF_TOKEN") or env.get("HF_TOKEN") or None


def resolve_audit_mode(audit_mode: str, probe_bank_path: str | None) -> str:
    """A secret probe bank is read ONLY by the probe audit, so supplying one implies probe mode.

    The daemon used to never pass `audit_mode` at all — it stayed at the "grounding" default while
    the validator gated the entire probe path on `audit_mode == "probe"`. So `--probe-bank` loaded
    the bank, handed it to the validator, and was then silently ignored: a no-op flag."""
    return "probe" if probe_bank_path else audit_mode


class NoInferenceBackend:
    """The backend a grounding-mode validator gets: it has none, and must never need one.

    Grounding is a pure proof-inspection — the validator reads the attested proof + trace and runs no
    miner code. If anything ever calls a model, that is a bug (and a silent bill), so this raises
    instead of quietly working. The opt-in `--audit-mode probe` swaps in a real backend."""

    def complete(self, model, messages, params):
        raise RuntimeError(
            "the validator tried to call a model in grounding mode, which must do NO inference. "
            "Grounding inspects the proof + trace only. If you meant to re-execute miner agents, "
            "run with --audit-mode probe (which needs OPENROUTER_API_KEY and costs real money).")


def store_get_proof(store):
    """Validator-side fetch: download the proof + trace the miner uploaded to its repo."""
    import json

    def get_proof(hk, epoch, nonce, repo):
        pj = store.download_proof(repo, epoch)
        tj = store.download_trace(repo, epoch)
        if pj is None or tj is None:
            return None
        return Proof.from_json(pj), json.loads(tj)
    return get_proof


class KOTHValidatorNeuron:  # pragma: no cover — daemon loop
    def __init__(self, validator: KOTHValidator, store, state_path: str | None = None,
                 standings_repo: str | None = None):
        self.val = validator
        # Docker runs this Python process as PID 1, whose default SIGTERM disposition is ignored.
        # A real release rehearsal therefore reached Docker's timeout and was SIGKILLed.  The daemon
        # installs an explicit handler below; the event wakes poll waits without interrupting an
        # in-flight score/save/weight transaction.
        self._stop = threading.Event()
        self.get_proof = store_get_proof(store)
        self.state_path = state_path
        # The public leaderboard feed: this validator's OWN repo, published once per settled epoch.
        # It is that validator's opinion, not a consensus view — the dashboard reads one publisher.
        self.standings_repo = standings_repo
        if state_path and self.val.load_state(state_path):
            print(f"[koth-validator] restored reign + king baseline from {state_path}")

    def _save(self) -> None:
        if self.state_path:
            self.val.save_state(self.state_path)

    def request_stop(self) -> None:
        """Finish the active atomic operation, then leave the daemon loop promptly."""
        self._stop.set()

    def _weights_are_stale(self) -> bool:
        """Have this validator's on-chain weights aged past `WEIGHT_REFRESH_BLOCKS`?

        `blocks_since_weights` exists only on the live chain seam. The offline chains (MockChain,
        LocalFileChain) have no activity cutoff to age out of, so they report nothing and never go
        stale — which keeps every sim's submission pattern exactly as it was.
        """
        since = getattr(self.val.chain, "blocks_since_weights", None)
        if since is None:
            return False
        try:
            n = since()
        except Exception:  # noqa: BLE001 — an unreadable chain must not manufacture a submission
            return False
        return n is not None and int(n) >= WEIGHT_REFRESH_BLOCKS

    def _flush_pending(self) -> bool:
        """Finish one durably staged epoch; safe to repeat after a crash or RPC failure."""
        weights = self.val.pending_weights
        report = self.val.pending_report
        if weights is None and report is None:
            return True
        if weights is not None:
            # SUBMIT ON CHANGE **OR** ON AGE. Change alone was the old rule, and it silently stopped
            # writing weights for as long as the reign held (see WEIGHT_REFRESH_BLOCKS).
            stale = self._weights_are_stale()
            submit = bool(weights) and (weights != self.val.last_submitted_weights or stale)
            ready = time.monotonic() >= getattr(self, "_weight_retry_not_before", 0.0)
            try:
                if submit and ready:
                    # `force` only when the slate is unchanged: it bypasses the chain seam's
                    # already-current short-circuit, which would otherwise drop the refresh.
                    self.val.chain.set_weights(weights, force=stale)
            except Exception as exc:  # noqa: BLE001 — keep the durable pending update for retry
                retry_blocks = getattr(exc, "retry_blocks", None)
                if retry_blocks is not None:
                    self._weight_retry_not_before = time.monotonic() + min(60.0, retry_blocks * 12.0)
                reconnect = getattr(self.val.chain, "reconnect", None)
                if retry_blocks is None and reconnect is not None:
                    try:
                        reconnect()
                    except Exception as reconnect_exc:  # noqa: BLE001 — retry remains fail-closed
                        print(f"[koth-validator] chain reconnect failed "
                              f"({type(reconnect_exc).__name__}: {reconnect_exc})", flush=True)
                print(f"[koth-validator] weight submission pending; will retry ({type(exc).__name__}: "
                      f"{exc})", flush=True)
                return False
            if submit and not ready:
                return False              # inside the backoff window: retry, do not settle the epoch
            if weights:
                self.val.record_submitted_weights(weights)
            self._weight_retry_not_before = 0.0
            # Chain inclusion is the scoring transaction boundary. Persist it BEFORE optional
            # dashboard work, whose public RPC/HF reads must never cause a weight replay.
            self.val.complete_pending_weights()
            self._save()
        if report is not None:
            if self.standings_repo:
                self.val.flush_standings(self.standings_repo, report)
            self.val.complete_pending_report()
            self._save()
        return True

    def _stage_epoch(self, epoch: int) -> bool:
        """Score and durably stage one epoch, rolling back every in-memory mutation on failure."""
        before = self.val.snapshot()
        try:
            rep = self.val.run_epoch(self.get_proof, epoch=epoch, submit_weights=False)
            # SUCCESS MUST BE AUDIBLE. This previously logged only on failure, so a validator scoring
            # every epoch correctly printed nothing after the startup banner — indistinguishable from
            # one that had silently wedged, which is precisely how the reference cron failed unnoticed
            # for three ticks. One line per epoch: who scored, who was disqualified and why, whether
            # the traffic was routable at all, and where the weight went.
            king = self.val.reign.members[0].sub.hotkey[:10] if self.val.reign.members else "-"
            ref = (rep.audit or {}).get("pool_reference") or {}
            routable = ref.get("routable")
            print(f"[koth-validator] epoch {epoch}: scored={len(rep.scored)} dq={len(rep.dq)} "
                  f"king={king} uids={len(rep.weights_by_uid)}"
                  # SAY SO WHEN ROUTING IS NOT BEING SCORED. Without a pool reference the scalar
                  # silently degrades to absolute accuracy — the subnet keeps paying, the log keeps
                  # looking healthy, and the thing the subnet exists to measure is simply not
                  # measured. It used to show up only as the ABSENCE of these two fields, which is
                  # invisible unless you already know to look for it. Live on 76745: the owner's
                  # reference lost every row to one hung cell and no line anywhere said so.
                  + (f" routable={routable} gap={ref.get('achievable_gap')}" if ref
                     else " reference=MISSING(scoring absolute accuracy, NOT routing)")
                  + (f" dq_reasons={sorted(set(rep.dq.values()))}" if rep.dq else ""), flush=True)
        except Exception as exc:  # noqa: BLE001 — public RPC failures are retried from a clean state
            self.val.restore(before)
            reconnect = getattr(self.val.chain, "reconnect", None)
            if reconnect is not None:
                try:
                    reconnect()
                except Exception as reconnect_exc:  # noqa: BLE001 — retry loop remains fail-closed
                    print(f"[koth-validator] chain reconnect failed ({type(reconnect_exc).__name__}: "
                          f"{reconnect_exc})", flush=True)
            print(f"[koth-validator] epoch {epoch} failed before durable staging; will retry "
                  f"({type(exc).__name__}: {exc})", flush=True)
            return False
        self._save()                               # durable BEFORE the chain write
        return self._flush_pending()

    def run_forever(self, poll_s: float = 5.0):
        restored = self.val.last_scored_epoch
        # The STARTUP settle is as exposed as the in-loop one: on a fresh boot with no persisted
        # epoch this is the first chain read, and an RPC hiccup here killed the daemon before the
        # retry loop below could ever protect it. Retry until the chain answers.
        last = restored
        while last is None and not self._stop.is_set():
            try:
                last = self.val._settle_epoch() - 1        # first boot: latest settled epoch
            except Exception as exc:                       # noqa: BLE001 — chain/network flakiness
                print(f"[koth-validator] chain unreadable at startup "
                      f"({type(exc).__name__}: {str(exc)[:100]}) — retrying", flush=True)
                self._stop.wait(poll_s)
        while not self._stop.is_set():
            # A TRANSIENT CHAIN RPC FAILURE MUST NOT KILL THE DAEMON. Observed in production: the
            # substrate node returned a body with no "result", `_get_block_number` did
            # `response["result"]["number"]` on None, and the TypeError propagated out of
            # `run_forever` — the validator process EXITED and only came back because Docker
            # restarted it. It had been crash-looping, losing in-memory reign state and re-reading
            # from disk each time. A validator is a long-lived daemon on a network that hiccups; the
            # correct response to an unreadable block is to wait and ask again.
            try:
                if not self._flush_pending():
                    self._stop.wait(poll_s)
                    continue
                if self.val.last_scored_epoch is not None:
                    last = max(last, self.val.last_scored_epoch)
                settled = self.val._settle_epoch()         # latest past-grace epoch; skip the live one
                for e in range(last + 1, settled + 1):     # score each settled epoch once, in order
                    if self._stop.is_set():
                        break
                    if not self._stage_epoch(e):
                        break
                    last = e
            except Exception as exc:                       # noqa: BLE001 — chain/network flakiness
                print(f"[koth-validator] transient error ({type(exc).__name__}: {str(exc)[:120]}) "
                      f"— retrying in {poll_s:.0f}s", flush=True)
            self._stop.wait(poll_s)


class _RoutingPool:
    """Mock backend over the PINNED routing pool, with a real quality/price gradient.

    MockPool serves cheap/mid/strong, but a routing head emits rungs into `harness.ROUTING_POOL`, so
    a local run needs a backend that answers to those exact ids — otherwise the miner cannot even
    invoke the ladder it was trained against.
    """

    def __init__(self):
        from . import harness as H
        self.models = H.pool_models()
        self.price = {m: H.price_of(m) for m in self.models}

    def complete(self, model, messages, params):
        if model not in self.price:
            raise ValueError(f"unpinned model {model}")
        rung = self.models.index(model)
        prompt = str(messages[-1]["content"])
        # higher rungs answer correctly more often — a real ladder, so routing has something to learn
        ok = (hash((prompt, rung)) % 10) < (4 + rung)
        text = "Answer: 4" if ok else "..."
        tin, tout = 32, 8
        return text, tin, tout, (tin + tout) / 1000.0 * self.price[model] / 1000.0


def run_local_routing(epochs: int = 3, verbose: bool = True) -> dict:
    """THE ROUTING LOOP THROUGH THE REAL DAEMONS — the seam nothing else covers.

    `run_local` below exercises the free-agent path, and `orchestra-koth-router-sim` exercises the
    validator against hand-built proofs. Neither runs a ROUTING artifact through
    `KOTHMinerNeuron.run_once` -> store -> `KOTHValidator.run_epoch`, which is the path a live miner
    actually takes. Both bugs found this session lived in exactly that kind of uncovered second path:
    the image's runner, and the reference pool. So this closes it.
    """
    import numpy as np

    from . import harness as H
    from .miner import routing_artifact
    from ..router import RouterHead
    root = tempfile.mkdtemp(prefix="koth_routing_")
    backend, platform = _RoutingPool(), Platform()
    chain = LocalFileChain(f"{root}/chain")
    store = LocalBundleStore(f"{root}/store")
    chain.register("burn")
    suite = default_suite()
    k, hid, N_PER = len(H.pool_models()), 8, 4
    n = RouterHead(H.EMBED_DIM, k, hid).n_params

    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, KingChain(),
                        suite, store, backend, n_per_bench=N_PER, budget=0.5, f_min=0.0,
                        routing_pool=H.pool_models())
    miners = {}
    for label, seed in (("router-a", 1), ("router-b", 5)):
        theta = np.random.default_rng(seed).normal(0, 0.4, n)
        # n_per_bench passed to BOTH sides: a mismatch is `unexpected_task` for every miner
        miners[label] = KOTHMinerNeuron(Signer().public_hex, backend, platform, suite, chain, store,
                                        routing_artifact(H.save_head(theta, hid)), f"{label}/repo",
                                        n_per_bench=N_PER)
    name = {m.hotkey: nm for nm, m in miners.items()}
    for m in miners.values():
        m.publish()
    get_proof = store_get_proof(store)

    emissions: dict = defaultdict(float)
    dq: dict = {}
    decisions = 0
    for _e in range(epochs):
        chain.advance(EPOCH_BLOCKS)
        epoch = current_epoch(chain)
        for m in miners.values():
            m.run_once(epoch)
        rep = val.run_epoch(get_proof)
        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            emissions[name.get(uid_hk.get(uid, ""), "burn")] += w
        dq.update({name.get(h, h): r for h, r in rep.dq.items()})
        decisions += sum(len(v) for v in (rep.audit.get("decision") or {}).values())

    if verbose:
        print(f"routing loop through the REAL daemons (file-chain + store + mock TEE) @ {root}")
        total = sum(v for kk, v in emissions.items() if kk != "burn") or 1.0
        for who in list(miners) + ["burn"]:
            v = emissions.get(who, 0.0)
            share = "  --" if who == "burn" else f"{v/total*100:5.1f}%"
            tag = f"  [DQ: {dq[who]}]" if who in dq else ""
            print(f"  {who:10s} {v:6.3f}  {share}{tag}")
        if dq:
            print(f"  disqualifications: {dq}")
    return {"emissions": dict(emissions), "dq": dq, "root": root}


def run_local(epochs: int = 3, verbose: bool = True) -> dict:
    """Offline decoupled demo: miners upload proofs, validator downloads + scores — and in the
    default "grounding" mode the validator runs NO miner code. Honest miners RELAY a pool model's
    answer, so their answers are grounded in the trace; a memorizer that ignored the pool would be
    DQ'd `ungrounded`. Strong routes to the accurate model, weak to the cheap one."""
    import json

    from .pool import MockPool
    from .runtime import Artifact
    root = tempfile.mkdtemp(prefix="koth_local_")
    backend, platform = MockPool(), Platform()               # differentiated pool: cheap/mid/strong
    chain = LocalFileChain(f"{root}/chain")
    store = LocalBundleStore(f"{root}/store")
    chain.register("burn")                                   # uid 0 = burn
    suite = default_suite()

    _RELAY = ("import json\n"
              "def build_agent(w):\n"
              "    m = json.loads(w.decode())['model']\n"
              "    def agent(prompt, call_model):\n"
              "        return call_model(m, [{'role': 'user', 'content': prompt}], {'max_tokens': 16})\n"
              "    return agent\n")
    art = lambda model: Artifact(_RELAY, json.dumps({"model": model}).encode(), model)  # noqa: E731

    # ONE slice size, passed to BOTH sides. The validator re-derives the slice and rejects a proof
    # whose task set differs (`unexpected_task`), so miner and validator n_per_bench are not two
    # settings — they are one setting stored twice, and letting them default independently is what
    # disqualified every miner when the validator default moved 8 -> 16.
    N_PER = 8
    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, KingChain(),
                        suite, store, backend, n_per_bench=N_PER, budget=0.5, f_min=0.1)
    miners = {
        "strong": KOTHMinerNeuron(Signer().public_hex, backend, platform, suite, chain, store,
                                  art("strong"), "strong/repo", n_per_bench=N_PER),
        "weak": KOTHMinerNeuron(Signer().public_hex, backend, platform, suite, chain, store,
                                art("cheap"), "weak/repo", n_per_bench=N_PER),
    }
    name = {m.hotkey: n for n, m in miners.items()}
    for m in miners.values():
        m.publish()                                          # upload bundle + commit (once)
    get_proof = store_get_proof(store)

    emissions: dict = defaultdict(float)
    for _e in range(epochs):
        chain.advance(EPOCH_BLOCKS)
        epoch = current_epoch(chain)
        for m in miners.values():
            m.run_once(epoch)                                # produce + upload proof to repo
        rep = val.run_epoch(get_proof)                       # download + verify + score
        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            emissions[name.get(uid_hk.get(uid, ""), "burn")] += w

    if verbose:
        print(f"decoupled run OK (file-chain + local store + mock TEE) @ {root}")
        total = sum(v for k, v in emissions.items() if k != "burn") or 1.0
        for who, v in sorted(emissions.items(), key=lambda kv: -kv[1]):
            share = "  --" if who == "burn" else f"{v/total*100:5.1f}%"
            print(f"  {who:8s} {v:6.3f}  {share}")
    return {"emissions": dict(emissions), "root": root}


def main() -> None:
    run_local()


def validator_main() -> None:  # pragma: no cover — live daemon (needs bittensor + HF + wallet)
    import argparse
    import signal

    from ..eval import config
    from ..gateway.gateway import OpenRouterBackend
    from ..subnet.chain import BittensorChain
    from .benchmarks import real_suite
    from .runtime import mock_vendor_platform, runtime_measurement
    from .store import HFBundleStore

    p = argparse.ArgumentParser(description="KOTH validator daemon (Bittensor mainnet; fail-closed)")
    p.add_argument("--netuid", type=int, required=True)
    p.add_argument("--wallet", required=True, help="Bittensor wallet (coldkey) name")
    p.add_argument("--hotkey", default="default",
                   help="hotkey NAME inside the wallet (the one registered on the subnet — it signs "
                        "set_weights). A wallet can hold several; only 'default' was reachable before.")
    p.add_argument("--network", default="finney")
    # 2, and the ceiling is WALL CLOCK, not statistics. A miner's run must finish inside one epoch
    # (100 blocks ~= 20 min) or its proof carries a stale nonce and can never validate. Measured on
    # hardware at ~90s per pool call over 3 benchmarks: 16 -> 48 tasks ~= 72 min, IMPOSSIBLE; 12
    # tasks straddled the boundary and landed only sometimes. 4 (12 tasks, ~1080s) was the
    # first proposal here and the epoch-fit test REJECTED it: it needs ~90% of the epoch, which
    # is exactly the coin-flip that was observed. 2 -> 6 tasks ~= 540s, inside a 60% budget.
    # An earlier 8 -> 16 raise here argued statistical power and was wrong twice: unachievable, and
    # unnecessary, because `accumulate` pools evidence ACROSS epochs — the Wilson bound applies to
    # the pooled count, so per-epoch size never governed confidence.
    p.add_argument("--n-per-bench", type=int, default=2)
    p.add_argument("--poll", type=float, default=12.0)
    p.add_argument("--insecure", action="store_true",
                   help="DEV/OFFLINE ONLY: accept the mock TEE + skip the measured-image gate "
                        "(NO SECURITY — miners can forge every score). REFUSED on mainnet.")
    p.add_argument("--state", default="koth_validator_state.json",
                   help="path for persisted reign+king state (restart-safety)")
    p.add_argument("--owner-hotkey",
                   help="ss58 of the subnet owner whose on-chain measured-image governance you trust. "
                        "Defaults to the chain's own SubnetOwnerHotkey — override only for a fork.")
    p.add_argument("--standings-repo",
                   help="publish standings.json (the public leaderboard feed the dashboard fetches) "
                        "to this HF repo each settled epoch, e.g. you/thirtyspokes-standings. "
                        "Presentational only — a publish failure never affects scoring or weights.")
    p.add_argument("--probe-bank", help="path to the secret memorization-probe bank JSON "
                                        "(koth/heldout.py) matching the owner's on-chain probe_commit. "
                                        "Implies --audit-mode probe.")
    p.add_argument("--audit-mode", choices=["grounding", "probe"], default="grounding",
                   help="memorization backstop. 'grounding' (default): pure proof-inspection, the "
                        "validator runs NO miner code, costs $0. 'probe': re-executes each miner's "
                        "agent on a held-out slice — costs inference AND runs untrusted miner code in "
                        "the sandbox.")
    p.add_argument("--commit-window", type=int, default=None,
                   help="F7 anti-grind: require each proof committed on-chain within N blocks of the "
                        "epoch open (miners commit automatically). Off by default; calibrate N to one "
                        "benchmark run's wall-clock before enabling.")
    p.add_argument("--no-pool-reference", action="store_true",
                   help="ignore the owner's per-epoch pool reference and score on absolute accuracy "
                        "(Q_lcb - lambda*cost) instead of frontier-relative routing headroom")
    p.add_argument("--min-headroom-gap", type=float, default=0.05,
                   help="refuse to score frontier-relatively when the pool reference shows less "
                        "achievable headroom than this (saturated traffic -> the ratio is noise)")
    p.add_argument("--skip-preflight", action="store_true",
                   help="start even if the startup preflight finds a blocking problem "
                        "(slice disagreement, ungradeable code benchmark, wrong governance). "
                        "The problem does not go away; you just find it hours later.")
    p.add_argument("--grace-blocks", type=int, default=None,
                   help="F2: score an epoch only N blocks after it opens, so submissions settle and "
                        "decoupled validators agree on presence. Keep N < 100 (epoch length).")
    args = p.parse_args()

    # --insecure turns OFF the whole security model (mock TEE accepted, measured-image gate skipped ->
    # a miner can forge every score). It exists for offline/dev bring-up only and is REFUSED on mainnet.
    if args.insecure and args.network in MAINNET_NETWORKS:
        raise SystemExit(
            f"[koth-validator] FATAL: --insecure is refused on mainnet ({args.network}). It accepts the "
            "mock TEE and skips the owner's measured-image gate, so miners could forge every score. Use "
            "it only for offline/dev bring-up on a non-mainnet network (e.g. --network test).")

    # ONE SOURCE FOR THE GRACE POINT. The owner publishes it, miners size their runs against it, and
    # a validator that keeps its own copy WILL drift from it: compose here defaulted to 50 while
    # governance said 85, so every `compose up --force-recreate` silently reverted a hand-set 85 and
    # the validator scored seven minutes early, disqualifying proofs that met the published deadline.
    # Unset now means "use what the owner published", so there is nothing to keep in sync. An
    # explicit value still wins, and the preflight checks it against the published one.
    if args.grace_blocks is None:
        try:
            _gov = BittensorChain(args.netuid, args.wallet, args.network,
                                  hotkey=args.hotkey).owner_measurements() or {}
            args.grace_blocks = int(_gov.get("grace_blocks") or 0)
            print(f"[koth-validator] grace: using the published {args.grace_blocks} blocks", flush=True)
        except Exception as exc:      # noqa: BLE001 — an unreadable chain must not block startup
            args.grace_blocks = 0
            print(f"[koth-validator] grace: chain unreadable ({type(exc).__name__}), scoring at the "
                  f"epoch boundary", flush=True)

    # PREFLIGHT (koth/doctor.py). A validator that cannot grade the ranked benchmark, or disagrees
    # with the miners about the slice, or measures a runtime the owner never approved, comes up
    # looking perfectly healthy and scores zeros for hours — and every one of those symptoms points
    # at the MINERS. Check it in seconds here instead. `--insecure` is offline bring-up: report but
    # do not block, since there is no published governance and often no docker daemon there.
    if not args.skip_preflight:
        from . import doctor
        checks = [doctor.check_slice_agreement(args.n_per_bench), doctor.check_slice_fits_epoch(args.n_per_bench),
                  doctor.check_code_grading(), doctor.check_build_freshness(),
                  doctor.check_governance(args.netuid, args.network, args.wallet, args.hotkey,
                                          grace_blocks=args.grace_blocks)]
        doctor.preflight_or_exit("koth-validator", checks, block=not args.insecure)

    chain = BittensorChain(args.netuid, args.wallet, args.network, hotkey=args.hotkey,
                           owner_ss58=args.owner_hotkey)     # None -> resolved from the chain
    platform = mock_vendor_platform()          # public_hex only used on the (rejected) mock path
    hf_token = validator_hf_token()
    if args.standings_repo and not hf_token:
        raise SystemExit(
            "[koth-validator] FATAL: --standings-repo requires OWNER_HF_TOKEN (preferred for the "
            "official validator) or HF_TOKEN with write access to that Hugging Face repo.")
    store = HFBundleStore(token=hf_token)
    # Secure by default: enforce the hardware measured-image gate. The owner's approved set
    # (MRTD + RTMR1/2/3 + TCB policy) is read per-epoch from the chain via governance.
    probe_bank = None
    audit_mode = resolve_audit_mode(args.audit_mode, args.probe_bank)

    # A grounding-mode validator does NO inference: it inspects the proof + trace and runs no miner
    # code (measured: 0 pool calls per epoch). It therefore needs no LLM key — the daemon used to
    # demand one unconditionally, forcing every validator to hold a billing credential for a path
    # that is off by default and never executed. Only the opt-in `probe` audit re-executes agents,
    # and only that needs a key. In grounding mode we install a backend that RAISES if anything ever
    # tries to infer, so a regression that silently starts spending money fails loudly instead.
    if audit_mode == "probe":
        cfg = config.LiveConfig(); cfg.require_key()
        backend = OpenRouterBackend(cfg.api_key, cfg.base_url, price_fn=config.price_for)
    else:
        backend = NoInferenceBackend()
    if args.probe_bank:
        from .heldout import SecretProbeBank
        probe_bank = SecretProbeBank.from_file(args.probe_bank)
        if args.audit_mode != "probe":
            print("[koth-validator] --probe-bank given -> audit_mode=probe (re-executes miner agents)")
    # THE ROUTER SCALAR'S INPUT. Without a pool reference the validator can only score how accurate
    # a miner was; with one it can score how well it ROUTED — against what was achievable at the
    # price it paid. The owner publishes it per epoch (`orchestra-koth-reference`); a missing or
    # stale one degrades to the absolute scalar rather than stalling the subnet.
    pool_reference = None
    if not args.no_pool_reference:
        from .reference import chain_reader
        pool_reference = chain_reader(chain, n_per_bench=args.n_per_bench)
    val = KOTHValidator({runtime_measurement()}, platform.public_hex, chain, KingChain(),
                        real_suite(), store, backend, n_per_bench=args.n_per_bench,
                        enforce=not args.insecure, probe_bank=probe_bank, audit_mode=audit_mode,
                        commit_window=args.commit_window, grace_blocks=args.grace_blocks,
                        pool_reference=pool_reference,
                        min_headroom_gap=args.min_headroom_gap)
    # RETRY THE PREFLIGHT — it reads the chain AND fetches the governance record over HTTP, so a
    # single transient failure must not permanently kill the validator. Observed on the first real
    # `docker compose up`: the record was fetchable from the same container seconds later, but the
    # daemon had already exited FATAL and, under `restart: unless-stopped`, crash-looped forever.
    # Cold start is the worst moment for this — the suite's dataset downloads run first and share the
    # (unauthenticated, rate-limited) Hub budget with this fetch. `_effective_governance` caches
    # last-known-good for mid-run blips; at startup there is nothing cached, so the tolerance has to
    # be here. Bounded, then still fail closed: an unset gate would admit a forged runtime.
    ok, why = False, "not_checked"
    for attempt in range(1, 7):
        ok, why = val.governance_ready()
        if ok:
            break
        print(f"[koth-validator] governance preflight attempt {attempt}/6 failed ({why}); "
              f"retrying in {10 * attempt}s", flush=True)
        time.sleep(10 * attempt)
    if not ok:
        raise SystemExit(
            f"[koth-validator] FATAL ({why}) after 6 attempts: enforcing mode requires the owner's "
            "approved measured-image set (MRTD + RTMR1/2/3 + TCB policy) published on-chain — without "
            "it a trustless miner could run a tampered runtime and forge its score. The subnet owner "
            "must publish it with `orchestra-koth-owner`.")
    neuron = KOTHValidatorNeuron(val, store, state_path=args.state,
                                 standings_repo=args.standings_repo)

    # PID 1 ignores SIGTERM's default action. Install an explicit handler that requests a safe loop
    # exit rather than raising through a score/save/chain transaction. Docker Compose grants enough
    # time for a bounded in-flight RPC to finish; an idle poll wakes immediately via the Event.
    def stop(_signum, _frame):
        neuron.request_stop()

    signal.signal(signal.SIGTERM, stop)
    print(f"[koth-validator] netuid={args.netuid} enforce={val.enforce} polling every {args.poll}s",
          flush=True)
    try:
        neuron.run_forever(poll_s=args.poll)
    except KeyboardInterrupt:
        neuron.request_stop()
    finally:
        # Substrate keeps a non-daemon WebSocket worker alive. The loop can return cleanly while the
        # container still hangs until Docker sends SIGKILL unless we close that connection explicitly.
        close = getattr(chain, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # noqa: BLE001 — shutdown must continue after a broken socket
                print(f"[koth-validator] chain close failed ({type(exc).__name__}: {exc})", flush=True)
    print("[koth-validator] stopped", flush=True)


if __name__ == "__main__":
    main()
