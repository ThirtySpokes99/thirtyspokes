"""Preflight a deployment before it runs — `orchestra-koth-doctor`.

WHY THIS EXISTS. Bringing the subnet up live surfaced nine bugs, and the expensive ones were not
logic errors: they were DEPLOYMENT SHAPE. The validator container had no docker client, so the ranked
benchmark could not be graded. The sandbox bind-mount resolved in the host namespace, so it still
could not be graded once the client was there. The slice size was set to a value at which no miner
could finish inside an epoch. Miner and validator disagreed on that slice, disqualifying everyone.
The published image did not match the measurements pinned on chain.

Every one of those is checkable in seconds and each cost hours to find, because the symptom always
appeared somewhere else: a grading failure looked like a bad miner, a slice mismatch looked like
cheating, an unreadable image looked like an unapproved runtime. Each check below exists because
something upstream lied about the cause.

Read-only and cheap: no inference, no chain writes, no VMs. Run it before the daemons.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import threading

OK, WARN, FAIL = "ok", "warn", "FAIL"

# A preflight must always terminate. Every chain read here is bounded by this and degrades to a
# warning, because "the chain is slow" is not a deployment fault and must not hold up the daemon.
CHAIN_READ_TIMEOUT_S = 45.0


def _r(status: str, name: str, detail: str) -> tuple[str, str, str]:
    return (status, name, detail)


def _bounded(fn, timeout: float = CHAIN_READ_TIMEOUT_S):
    """Run `fn` on a daemon thread and give up on it. Returns `(value, error_or_None)`.

    Used for every chain read here. `join(timeout=...)` is the only bound that actually holds: the
    SDK can sit in its own retry ladder inside `Subtensor(...)` before any timeout we configure is
    reachable, so wrapping the call is the only way a preflight is guaranteed to finish. The thread
    is left running rather than killed — it is a daemon, it holds no locks we need, and the process
    is about to either start the daemon or exit.
    """
    box: dict = {}

    def _run():
        try:
            box["v"] = fn()
        except BaseException as e:      # noqa: BLE001 — reported, never raised on a dead thread
            box["e"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if "v" in box:
        return box["v"], None
    if "e" in box:
        return None, box["e"]
    return None, TimeoutError(f"no answer in {timeout:.0f}s")


def check_slice_fits_epoch(n_per_bench: int) -> tuple[str, str, str]:
    """The constraint that made mining IMPOSSIBLE and took two wasted TDX runs to find.

    A proof completed after its epoch expires carries a stale nonce and can never validate — it is
    unrecoverable, not merely late. So slice x benchmarks x per-call latency must fit one epoch.
    """
    from .benchmarks import real_suite
    from .epoch import EPOCH_BLOCKS
    from .harness import MIN_TASK_S, RUN_BUDGET_S
    # The harness refuses further escalation past TASK_BUDGET_S, so this is an actual CEILING, not
    # the average-latency estimate it used to be. That estimate was the weaker claim: it passed at
    # 45% of the window on the same day a single task ran ~950s and lost the epoch.
    # Measured against the whole epoch. A validator's --grace-blocks makes the REAL deadline tighter
    # still (85 blocks ~= 1020s), which is what RUN_BUDGET_S is sized for — see harness.RUN_BUDGET_S.
    block_s = 12.0
    n_bench = len(real_suite())
    # boot + attest/emit + upload, plus the watchdog's per-task slack (koth/runtime.py): the
    # watchdog waits budget+5s before abandoning, so the worst case is the budget plus that slack.
    overhead = 90.0 + n_per_bench * n_bench * 5.0
    need = RUN_BUDGET_S + overhead
    budget = EPOCH_BLOCKS * block_s
    frac = need / budget
    d = (f"{n_per_bench} x {n_bench} benchmarks under a shared {RUN_BUDGET_S:.0f}s run budget "
         f"= {need:.0f}s worst case vs ~{budget:.0f}s/epoch ({frac:.0%} of the window)")
    # THE FAILURE MODE MOVED WITH THE BUDGET. A shared run budget means an oversized slice no longer
    # overruns the epoch — it starves: every task collapses to the MIN_TASK_S floor and answers get
    # truncated wholesale, which grades as incapability rather than as a misconfiguration. So the
    # question is no longer "does it fit" but "can every task still afford a real answer".
    n_tasks = n_per_bench * n_bench
    floor_need = n_tasks * MIN_TASK_S
    if floor_need > RUN_BUDGET_S:
        return _r(FAIL, "slice fits epoch",
                  f"{n_tasks} tasks x {MIN_TASK_S:.0f}s floor = {floor_need:.0f}s exceeds the "
                  f"{RUN_BUDGET_S:.0f}s run budget — every task would be truncated at the floor")
    # The WARN threshold moved 0.6 -> 0.85 WITH the semantics. It used to gate an average-latency
    # estimate, where 60% of the window was genuinely risky; it now gates a hard ceiling the harness
    # enforces, and a typical run lands far below it. Leaving it at 0.6 would have made the correct
    # production slice warn on every start, which is how operators learn to ignore warnings.
    if frac > 1.0:
        return _r(FAIL, "slice fits epoch", d + " — NO valid proof can ever be produced")
    if frac > 0.85:
        return _r(WARN, "slice fits epoch", d + " — no headroom for boot and attestation")
    return _r(OK, "slice fits epoch", d)


def check_slice_agreement(runtime_n: int | None = None) -> tuple[str, str, str]:
    """Miner, validator, operator and reference cron store ONE setting in four places. A mismatch is
    `unexpected_task` — a DQ — for every miner, and each side looks individually correct.

    `runtime_n` is THIS process's effective value. It matters because the live mismatch was not a
    source default at all: a validator was raised to 16 on the command line while the miners stayed
    where they were, so every source default still agreed and every miner was disqualified anyway.
    A daemon passes its own resolved setting; the standalone CLI checks defaults only.
    """
    import inspect
    import pathlib
    import re

    from .miner import KOTHMinerNeuron
    vals = {"miner class": inspect.signature(KOTHMinerNeuron.__init__).parameters["n_per_bench"].default}
    # The repo is found relative to this file, which only holds when running from a checkout. A
    # CONTAINERISED validator has the package under site-packages, so the CLI defaults and the cron
    # are simply not there — and the check used to report a confident "ok" after comparing the
    # process against itself, which is worse than admitting it cannot see them. KOTH_REPO_ROOT lets
    # a container mount the checkout and get the real check.
    here = pathlib.Path(os.environ.get("KOTH_REPO_ROOT")
                        or pathlib.Path(__file__).resolve().parents[3])
    checked_sources = 0                      # sources OTHER than this process's own defaults
    for label, rel, pat in [
        ("validator cli", "src/thirtyspokes/koth/neuron.py", r'"--n-per-bench", type=int, default=(\d+)'),
        ("miner cli", "src/thirtyspokes/koth/miner.py", r'"--n-per-bench", type=int, default=(\d+)'),
        ("operator cli", "src/thirtyspokes/koth/gcp_operator.py", r'"--n-per-bench", type=int, default=(\d+)'),
        ("reference cron", "scripts/koth-reference-cron.sh", r"--n-per-bench (\d+)"),
    ]:
        f = here / rel
        if not f.exists():
            continue
        m = re.search(pat, f.read_text())
        if m:
            vals[label] = int(m.group(1))
            checked_sources += 1

    # ALSO READ WHAT IS ACTUALLY DEPLOYED. This check listed `scripts/koth-reference-cron.sh` and
    # kept passing after the reference moved to a systemd unit — validating a file nothing runs,
    # while the live setting sat somewhere it never looked. A check that quietly stops covering the
    # thing it names is worse than no check, so it now reads the units too when they are present.
    unit_dir = pathlib.Path(os.environ.get("KOTH_UNIT_DIR", "/etc/systemd/system"))
    for unit in sorted(unit_dir.glob("koth-*.service")):
        try:
            m = re.search(r"--n-per-bench[= ](\d+)", unit.read_text())
        except OSError:
            continue
        if m:
            vals[f"unit {unit.stem}"] = int(m.group(1))
            checked_sources += 1
    if runtime_n is not None:
        vals["this process"] = runtime_n
    uniq = set(vals.values())
    d = ", ".join(f"{k}={v}" for k, v in vals.items())
    if len(uniq) != 1:
        return _r(FAIL, "slice agreement", d)
    if checked_sources == 0:
        return _r(WARN, "slice agreement",
                  d + f" — could not read the CLI/cron defaults under {here}, so this compared the "
                      f"process against itself only (set KOTH_REPO_ROOT to check them)")
    return _r(OK, "slice agreement", d)


def check_code_grading() -> tuple[str, str, str]:
    """The ranked benchmark is code, graded by running it in a throwaway container. Two independent
    ways this silently fails, both hit live:
      * no docker CLI — `docker.io` ships the DAEMON, the client is `docker-cli` (a *Recommends*
        that --no-install-recommends drops), so the package installs and leaves no binary;
      * the scratch dir is bind-mounted by PATH, resolved by the DAEMON. Containerised validators
        talking to the host daemon mount a path that does not exist there, binding nothing.
    """
    if shutil.which("docker") is None:
        return _r(FAIL, "code grading", "no `docker` client on PATH — install docker-cli, not docker.io")
    try:
        v = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, text=True, timeout=20)
        if v.returncode != 0:
            return _r(FAIL, "code grading", f"docker client cannot reach a daemon: {v.stderr.strip()[:80]}")
        server = v.stdout.strip()
    except Exception as e:      # noqa: BLE001
        return _r(FAIL, "code grading", f"docker unusable: {type(e).__name__}")

    # the mount must resolve in BOTH namespaces — prove it rather than assume it
    import tempfile
    base = os.environ.get("KOTH_GRADE_DIR") or None
    try:
        with tempfile.TemporaryDirectory(dir=base) as d:
            open(os.path.join(d, "probe.txt"), "w").write("koth")
            r = subprocess.run(["docker", "run", "--rm", "--network", "none",
                                "-v", f"{d}:/w:ro", "python:3.11-slim", "cat", "/w/probe.txt"],
                               capture_output=True, text=True, timeout=180)
        if r.stdout.strip() != "koth":
            return _r(FAIL, "code grading",
                      f"daemon {server} reachable but the sandbox mount resolves elsewhere "
                      f"(set KOTH_GRADE_DIR to a path bind-mounted identically on both sides)")
    except Exception as e:      # noqa: BLE001
        return _r(FAIL, "code grading", f"sandbox probe failed: {type(e).__name__}: {str(e)[:60]}")
    return _r(OK, "code grading", f"daemon {server}, sandbox mount resolves"
                                  + (f" via KOTH_GRADE_DIR={base}" if base else ""))


def check_governance(netuid: int, network: str, wallet: str, hotkey: str,
                     grace_blocks: int | None = None) -> tuple[str, str, str]:
    """Does the measurement THIS code computes match what the owner pinned on chain? A mismatch means
    every honest miner is rejected as `unapproved_runtime`, and the symptom points at the miner."""
    from .runtime import runtime_measurement

    def _read():
        from ..subnet.chain import BittensorChain
        c = BittensorChain(netuid, wallet, network, hotkey=hotkey)
        return c.owner_measurements() or {}

    # Bounded for the same reason as the reference check: this reads the chain too, and an
    # unbounded preflight blocks the daemon it exists to protect.
    rec, err = _bounded(_read)
    if err is not None:
        return _r(WARN, "governance", f"chain unreadable ({type(err).__name__}) — cannot verify")
    on_chain = list(rec.get("runtime_measurements") or [])
    mine = runtime_measurement()
    if not on_chain:
        return _r(FAIL, "governance", "no approved-measurement record published")
    if mine not in on_chain:
        return _r(FAIL, "governance",
                  f"this build measures {mine[:16]}… but chain pins {on_chain[0][:16]}… — "
                  f"every proof from this code would be rejected as unapproved_runtime")
    # The published grace point is the deadline a proof must beat; RUN_BUDGET_S is sized for it, so a
    # mismatch means every honest miner's run is either wasted (too slow) or needlessly cramped.
    grace = rec.get("grace_blocks")
    # THE VALIDATOR'S OWN GRACE MUST NOT BE TIGHTER THAN THE PUBLISHED ONE. Miners size their runs
    # against the number the owner publishes; a validator scoring EARLIER than that misses proofs
    # that arrived exactly on the contract and disqualifies honest miners as `no_proof`. Scoring
    # later is harmless — it only waits longer — so that is a warning, not a failure.
    if grace and grace_blocks is not None and int(grace_blocks) != int(grace):
        if int(grace_blocks) < int(grace):
            return _r(FAIL, "governance",
                      f"this validator scores at {grace_blocks} blocks but the owner publishes "
                      f"{grace} — it would miss proofs that met the published deadline and DQ "
                      f"honest miners as no_proof")
        return _r(WARN, "governance",
                  f"this validator scores at {grace_blocks} blocks, later than the published "
                  f"{grace}; harmless, but it disagrees with what miners are told")
    if grace:
        from .harness import RUN_BUDGET_S
        need = RUN_BUDGET_S + 90.0              # boot + attest/emit + upload
        if need > int(grace) * 12:
            return _r(FAIL, "governance",
                      f"run budget {RUN_BUDGET_S:.0f}s + overhead = {need:.0f}s cannot beat the "
                      f"published grace of {grace} blocks ({int(grace) * 12}s) — every proof would "
                      f"land after scoring, complete and valid and unscored")
    pool = list(rec.get("pool_allow_list") or [])
    from .harness import pool_models
    if pool and sorted(pool) != sorted(pool_models()):
        return _r(FAIL, "governance",
                  f"on-chain pool ({len(pool)}) != harness ROUTING_POOL ({len(pool_models())}) — "
                  f"decisions would be scored against a ladder miners never saw")
    return _r(OK, "governance", f"measurement {mine[:16]}… pinned, pool {len(pool)} rungs")


def check_reference_freshness(netuid: int, network: str, wallet: str, hotkey: str,
                              bucket: str | None = None) -> tuple[str, str, str]:
    """Is the owner still publishing a pool reference for the CURRENT epoch?

    `_load_reference` is deliberately forgiving ("a reference outage is an owner problem, it must
    degrade rather than stall the subnet"), so when the record stops arriving `headroom_lcb` returns
    None and every miner falls through to the legacy `Q_lcb - cost` scalar. The validator keeps
    running, keeps scoring, keeps setting weights — on a different quantity than anyone intends.

    The epoch line DOES say so (`neuron.py` prints `reference=MISSING(scoring absolute accuracy, NOT
    routing)`), so this is not a missing signal; it is a signal that only exists inside one process's
    log stream, where it reads as one field among many and only while someone is watching. Mainnet
    ran 48 epochs that way. A preflight turns it into a startup gate instead of a thing you notice.

    Checks the BUCKET, not the process, for two reasons: a wedged builder is still `active` (a
    provider call hanging past its own timeout has held a run for 35+ minutes here), and the bucket
    is readable from anywhere — so this works from a machine that runs neither daemon.

    BOUNDED, because a preflight that hangs is worse than the fault it looks for: it blocks the
    daemon it is supposed to protect. Building a `Subtensor` can take the SDK's long retry path
    (`chain._configure_rpc`: five 60-second waits) before any of our own timeouts apply — measured
    here at 3s on a good endpoint and >9 minutes on a bad one, for the same call. A slow chain is a
    reason to say "cannot check", never a reason to wait.
    """
    from . import imagestore

    def _epoch_now():
        from ..subnet.chain import BittensorChain
        from .epoch import EPOCH_BLOCKS
        c = BittensorChain(netuid, wallet, network, hotkey=hotkey)
        return int(c.subtensor.get_current_block()) // EPOCH_BLOCKS

    now, err = _bounded(_epoch_now)
    if err is not None:
        return _r(WARN, "reference", f"chain unreadable ({type(err).__name__}) — cannot check")

    try:
        import re

        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("OWNER_HF_TOKEN") or os.environ.get("HF_TOKEN"))
        entries = api.list_bucket_tree(bucket or imagestore.default_bucket(), "reference",
                                       recursive=True)
        epochs = {int(m.group(1)) for e in entries
                  for m in [re.search(r"reference/(\d+)-", getattr(e, "path", str(e)))] if m}
    except Exception as e:      # noqa: BLE001 — an unreadable bucket is not proof of an outage
        return _r(WARN, "reference", f"bucket unreadable ({type(e).__name__}) — cannot check")

    if not epochs:
        return _r(FAIL, "reference",
                  "no pool reference has ever been published — every miner is scored on the legacy "
                  "quality/cost scalar, not on routing")
    latest = max(epochs)
    # One epoch of slack: the current epoch's record is legitimately still being built.
    behind = now - latest
    if behind > 1:
        return _r(FAIL, "reference",
                  f"newest reference is epoch {latest}, chain is at {now} ({behind} epochs behind) — "
                  f"the router scalar is dead and miners are silently being ranked on the legacy "
                  f"quality/cost scalar instead")
    return _r(OK, "reference", f"epoch {latest} published, chain at {now}")


#: Bittensor's own names for the chains. "mainnet" is NOT one of them — the SDK treats an unknown
#: value as a hostname and dies resolving it.
KNOWN_NETWORKS = ("finney", "test", "local", "archive", "subvortex")


def check_network_name(network: str) -> tuple[str, str, str]:
    """Is `--network` a name the SDK knows? Mainnet is `finney`; `mainnet` is not a network.

    This check exists because getting it wrong is WORSE than a hard failure here. An unknown name
    fails DNS inside `Subtensor(...)`, both chain checks below catch the exception and downgrade to
    "chain unreadable — cannot check", and warnings never block. The preflight then prints "no
    blocking problems" having verified neither governance nor the reference — the two things it is
    most needed for. Found in a live `.env` as `NETWORK=mainnet`, which `main()` uses as the default.
    """
    if network in KNOWN_NETWORKS:
        return _r(OK, "network", network)
    hint = " — mainnet is called `finney`" if network.lower() in ("mainnet", "main") else ""
    return _r(FAIL, "network",
              f"{network!r} is not a bittensor network{hint}. Every chain check would report "
              f"'unreadable' and pass as a warning, so this preflight would verify nothing.")


def check_build_freshness() -> tuple[str, str, str]:
    """Is the code in this container the code in the checkout?

    `docker compose restart` reuses the existing image, so a fix can look deployed and not be. It bit
    twice in one day: the v25 cutover (caught only because the runtime measurement had changed) and
    the reference-fallback marker (caught by noticing the log line was missing). Whenever the change
    does NOT move the measurement, nothing notices at all.
    """
    built = os.environ.get("KOTH_BUILD_COMMIT")
    if not built:
        return _r(OK, "build freshness", "not containerised (running from the checkout)")
    root = os.environ.get("KOTH_REPO_ROOT")
    if not root:
        return _r(WARN, "build freshness", f"image built from {built[:12]} — mount the checkout at "
                                           f"KOTH_REPO_ROOT to compare it with HEAD")
    git = pathlib.Path(root) / ".git"
    try:
        ref = (git / "HEAD").read_text().strip()
        if ref.startswith("ref: "):
            name = ref[5:]
            loose = git / name
            if loose.exists():
                ref = loose.read_text().strip()
            else:
                # PACKED REFS. A repack — or any `git gc`, or a history rewrite — moves refs out of
                # refs/heads/ into packed-refs, and reading only the loose path then fails. The check
                # went blind exactly when history changed, which is when it matters most.
                packed = (git / "packed-refs").read_text() if (git / "packed-refs").exists() else ""
                ref = next((ln.split()[0] for ln in packed.splitlines()
                            if not ln.startswith(("#", "^")) and ln.rstrip().endswith(" " + name)), "")
                if not ref:
                    return _r(WARN, "build freshness",
                              f"{name} is neither loose nor in packed-refs under {root}")
    except OSError as e:                # noqa: BLE001
        return _r(WARN, "build freshness", f"cannot read HEAD under {root} ({type(e).__name__})")
    if built == "unknown":
        return _r(WARN, "build freshness",
                  f"image carries no commit stamp; checkout is at {ref[:12]} — rebuild with "
                  f"KOTH_BUILD_COMMIT=$(git rev-parse HEAD) to make staleness visible")
    if built != ref:
        return _r(WARN, "build freshness",
                  f"image built from {built[:12]} but the checkout is at {ref[:12]} — `docker "
                  f"compose restart` does NOT rebuild; use `up -d --build` if you meant to deploy")
    return _r(OK, "build freshness", f"image matches the checkout at {ref[:12]}")


def render(results: list[tuple[str, str, str]]) -> str:
    width = max(len(r[1]) for r in results)
    mark = {OK: "  ok ", WARN: " warn", FAIL: " FAIL"}
    return "\n".join(f"[{mark[s]}] {n:<{width}}  {d}" for s, n, d in results)


def preflight_or_exit(role: str, results: list[tuple[str, str, str]], *, block: bool = True) -> None:
    """Run at daemon startup so a misconfigured deployment REFUSES TO START.

    Every check here has already broken a live run, and each one failed the same expensive way: the
    daemon came up, looked healthy, and scored nothing for hours while the symptom pointed at some
    other component. Failing at startup turns a multi-hour misdiagnosis into one line of output.

    `block=False` (dev/offline bring-up) prints the same report but proceeds — an offline run has no
    published governance and often no docker, and refusing there would break the sim path.
    """
    print(f"[{role}] preflight\n{render(results)}", flush=True)
    bad = [r for r in results if r[0] == FAIL]
    if not bad:
        return
    if not block:
        print(f"[{role}] {len(bad)} preflight problem(s) — continuing anyway (dev mode)", flush=True)
        return
    raise SystemExit(
        f"[{role}] FATAL: {len(bad)} blocking preflight problem(s): "
        + "; ".join(f"{n}: {d}" for _, n, d in bad)
        + ". Each of these has silently broken a live run before. Fix, or pass --skip-preflight to "
          "start anyway (it will not make the problem go away).")


def main() -> None:
    p = argparse.ArgumentParser(description="preflight a ThirtySpokes deployment (read-only)")
    p.add_argument("--netuid", type=int)
    p.add_argument("--network", default=os.environ.get("NETWORK", "finney"))
    p.add_argument("--wallet", default=os.environ.get("VALIDATOR_WALLET", ""))
    p.add_argument("--hotkey", default=os.environ.get("VALIDATOR_HOTKEY", "default"))
    p.add_argument("--n-per-bench", type=int, default=None,
                   help="slice to check (default: the validator CLI's own default)")
    args = p.parse_args()

    import inspect
    import re
    import pathlib
    n = args.n_per_bench
    if n is None:
        src = (pathlib.Path(__file__).resolve().parents[0] / "neuron.py").read_text()
        m = re.search(r'"--n-per-bench", type=int, default=(\d+)', src)
        n = int(m.group(1)) if m else 2

    results = [check_slice_agreement(args.n_per_bench), check_slice_fits_epoch(n),
               check_code_grading(), check_network_name(args.network)]
    if args.netuid and args.wallet:
        results.append(check_governance(args.netuid, args.network, args.wallet, args.hotkey))
        results.append(check_reference_freshness(args.netuid, args.network, args.wallet,
                                                 args.hotkey))
    else:
        results.append(_r(WARN, "governance", "skipped — pass --netuid and --wallet to check"))
        results.append(_r(WARN, "reference", "skipped — pass --netuid and --wallet to check"))

    print("ThirtySpokes preflight\n")
    print(render(results))
    bad = [r for r in results if r[0] == FAIL]
    print()
    if bad:
        print(f"{len(bad)} BLOCKING problem(s). Each of these has silently broken a live run before, "
              f"and the symptom appeared somewhere else.")
        raise SystemExit(1)
    print("No blocking problems. Warnings above are worth reading before a long run.")


if __name__ == "__main__":
    main()
