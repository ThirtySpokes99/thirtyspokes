#!/usr/bin/env python3
"""M9 exit 11 — one window of the real daemon, on one real benchmark, against one real challenger.

`test_validator.py` holds every property of the daemon's order and durability, but it holds them
with a scripted worker, a mock benchmark and a grader that returns a number. That is the right place
for those properties and the wrong place to learn what this codebase has never been told: whether the
whole thing runs when the model is 35B of weights on a card, the tasks are LiveCodeBench, the grader
is Docker, and every episode is a receipt against a real balance. This script is that one run.

WHAT IS REAL HERE, because it is what the exit criterion is for:

* the worker — `OpenRouterClient` behind the production `OwnerGateway`, spending real dollars and
  reconciling every arm against real receipts;
* the corpus — a real benchmark's real tasks, graded by the real Docker sandbox, on the machine
  `--sandbox-host` names (§8b.9: this process applies a model's patch and runs its tests);
* the challenger — a served conductor, reached through `local_serving`, refusing to answer unless
  the endpoint reports the artifact the chain committed to;
* the store path — the production `S3Bucket`, `build_manifest`, `upload_tree` and
  `fetch_submission`, so the tree is hashed on the way out and every shard re-hashed on the way in;
* the mechanism — `validator.Validator.run_window`, unmodified.

WHAT IS NOT, and why each is not a hole in the test:

* THE CHAIN is `MockChain`. A real ready signal costs a registration burn on a live subnet and takes
  the one commitment slot a hotkey ever gets (§7). `MockChain` recycles UIDs and honours the weight
  rate limit, which is the whole of what this window reads from a chain, and `--check` against the
  real chain is the mode that exercises the rest.
* THE BUCKET'S FAR SIDE is a directory. R2 credentials are an owner secret and `r2_bucket` is
  configuration with no logic in it; `S3Bucket` — the code that pages a listing and hashes a
  download — is the real one, over a client that links rather than copies so a 72 GB tree does not
  cost 72 GB twice to smoke.

Run it in two terminals. This one prints the `--served-model-name` the artifact must be launched
under (it is a function of the manifest digest, so it cannot be known before the manifest is built)
and then waits for that endpoint to report it::

    python scripts/live_smoke.py --model-dir /srv/v3/challenger \\
        --reference-tree /srv/v3/reference --exchange m3a-exchange.json \\
        --king0 always_cheapest --serve-url http://127.0.0.1:8002/v1 \\
        --sandbox-host unix:///var/run/docker.sock --tasks 4 --budget-usd 2

Exit 0 means the window settled and every arm reconciled against the meter. Exit 1 means it did not,
and the reveal says which of the two it was.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thirtyspokes.gateway import signing                                         # noqa: E402
from thirtyspokes.v3 import serve as serve_mod                                  # noqa: E402
from thirtyspokes.v3 import world as world_mod                                  # noqa: E402
from thirtyspokes.v3.access import Mailbox, Registration, encode_ss58           # noqa: E402
from thirtyspokes.v3.admission import describe                                  # noqa: E402
from thirtyspokes.v3.benchmarks import sandbox                                  # noqa: E402
from thirtyspokes.v3.benchmarks.hle import HumanitysLastExam                    # noqa: E402
from thirtyspokes.v3.benchmarks.real import LCB_RELEASES, LiveCodeBench         # noqa: E402
from thirtyspokes.v3.chain import MockChain, ReadySignal                        # noqa: E402
from thirtyspokes.v3.gateway import OwnerGateway                                # noqa: E402
from thirtyspokes.v3.openrouter import OpenRouterClient                         # noqa: E402
from thirtyspokes.v3.pool import fetch, snapshot                                # noqa: E402
from thirtyspokes.v3.simulate import DUELLED                                     # noqa: E402
from thirtyspokes.v3.store import (S3Bucket, build_manifest, fetch_manifest,  # noqa: E402
                                    upload_tree)
from thirtyspokes.v3.validator import (Cadence, Owner, Validator, allowance_journal,  # noqa: E402
                                        format_reveal, local_serving, reveal_path,
                                        served_name)
from thirtyspokes.v3.window import build as build_window                        # noqa: E402
from thirtyspokes.v3.window import window_path                                  # noqa: E402

NETUID = 99
BUCKET = "v3-smoke"
# One window, and a cadence `preflight` accepts: `window_blocks` must clear MockChain's 100-block
# weight rate limit and `immunity_blocks` must cover three windows (§8b.1).
CADENCE = Cadence(genesis_block=1_000, window_blocks=200, windows=(1,), immunity_blocks=600)
NONCE = "v3-live-smoke"


class DirectoryClient:
    """A boto3-shaped S3 client over one directory. `S3Bucket` is the code under test above it.

    `upload_file`/`download_file` hard-link when the filesystem allows it. A conductor tree is tens
    of gigabytes and this smoke moves it twice — out to the bucket and back into the duel's scratch
    directory — so copying would spend an hour of disk on bytes that never leave the machine. The
    digest checks in `fetch_submission` still read every byte, which is the part that matters: a
    link that pointed at the wrong file would fail them exactly as a bad copy would.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.meta: dict[str, dict[str, str]] = {}

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict | None = None,
                   ContentType: str | None = None) -> None:
        path = self._path(Key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(Body))
        self.meta[Key] = dict(Metadata or {})

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        import io
        return {"Body": io.BytesIO(self._path(Key).read_bytes()),
                "Metadata": dict(self.meta.get(Key, {}))}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"ContentLength": self._path(Key).stat().st_size,
                "Metadata": dict(self.meta.get(Key, {}))}

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "",
                        ContinuationToken: str | None = None) -> dict[str, Any]:
        keys = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())
        rows = [{"Key": k, "Size": self._path(k).stat().st_size}
                for k in keys if k.startswith(Prefix)]
        return {"Contents": rows, "IsTruncated": False}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self._path(Key).unlink(missing_ok=True)
        self.meta.pop(Key, None)

    def upload_file(self, filename: str, Bucket: str, Key: str, ExtraArgs=None, Config=None) -> None:
        self._link(Path(filename), self._path(Key))
        self.meta[Key] = dict((ExtraArgs or {}).get("Metadata", {}))

    def download_file(self, Bucket: str, Key: str, filename: str, Config=None) -> None:
        self._link(self._path(Key), Path(filename))

    @staticmethod
    def _link(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        try:
            os.link(src, dest)
        except OSError:
            shutil.copyfile(src, dest)


def miner_key(seed: str) -> Ed25519PrivateKey:
    """The smoke's miner identity. A real miner's is `miner.hotkey_seed` over a wallet keyfile; the
    hotkey here is derived from the same public key so `fetch_submission`'s signature check is the
    real one rather than a check of the smoke against itself."""
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed.encode()).digest())


def build_benchmarks(names: Sequence[str], per_benchmark: int, seed: str):
    """Every named benchmark, `per_benchmark` tasks each — AT LEAST TWO, or the window cannot run.

    Measured 2026-09-07, on the first live launch: with one benchmark drawn, `world.pins` refused —
    correctly — because `duel` cannot judge breadth on a single benchmark, however many the table
    prices. The corpus is what is DRAWN, not what is priced."""
    benchmarks, tasks = [], []
    for name in names:
        if name == "livecodebench":
            bench = LiveCodeBench(releases=LCB_RELEASES)
            drawn = list(bench.draw_across_releases(per_benchmark, seed=seed))
        elif name == "hle":
            bench = HumanitysLastExam()
            drawn = list(bench.load()[:per_benchmark])
        else:
            raise SystemExit(f"no adapter for benchmark {name!r} (livecodebench, hle)")
        benchmarks.append(bench); tasks.extend(drawn)
    return benchmarks, tasks


def already_committed(bucket: S3Bucket, prefix: str, manifest) -> str | None:
    """The digest already committed under `prefix` if it is THIS tree's, else None — or a refusal.

    Run 2b of this smoke (2026-09-07) restarted the wedged window and died in `upload_tree`, which
    refuses a prefix that already carries a manifest: correct, since a second upload into a
    committed prefix would replace the bytes the chain names. A restart is not a second submission.
    The same tree under the same hotkey is resumed; a DIFFERENT tree is the second submission §7
    forbids and is refused here by name, before a byte moves.
    """
    if prefix + "manifest.json" not in bucket.list(prefix):
        return None
    existing = fetch_manifest(bucket, prefix)
    if existing.sha256 != manifest.sha256:
        raise SystemExit(f"{prefix} already holds a DIFFERENT submission ({existing.sha256[:12]}… vs "
                         f"this tree's {manifest.sha256[:12]}…). One hotkey, one submission (§7): "
                         f"this is not a resume. Use a fresh --out to start over.")
    return existing.sha256


def wait_for(url: str, served: str, seconds: float) -> None:
    """Block until the endpoint reports the artifact, or say what it reported instead.

    `local_serving` raises on a mismatch and that raise is a REFUSAL the daemon records against the
    challenger, so a smoke that let the operator race the server would publish a lost duel and call
    it a finding. Waiting here keeps the two failures apart: this one is an empty port, the daemon's
    is a model that will not serve.
    """
    import httpx

    deadline = time.monotonic() + seconds
    seen: list[str] = []
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=10.0) as client:
                body = client.get(url.rstrip("/") + "/models").json()
            seen = [row.get("id", "") for row in body.get("data", ())]
            if served in seen:
                return
        except Exception:                                    # a port that is not up yet, repeatedly
            seen = []
        time.sleep(5.0)
    raise SystemExit(f"{url} never reported {served!r} (it reported {seen or 'nothing'}). Launch it "
                     f"with the line printed above — `--served-model-name` is the only channel the "
                     f"artifact's identity has onto the wire (§8b.2).")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="live_smoke",
        description="One window of the real validator daemon over live seams (the build plan M9 exit 11).")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="the challenger's tree, as a miner would upload it")
    parser.add_argument("--reference-tree", type=Path, required=True,
                        help="the pinned reference the challenger is admitted against (§1.1)")
    parser.add_argument("--exchange", type=Path, required=True,
                        help="M3a's published lambda/C table (scripts/m3a_spread.py)")
    parser.add_argument("--king0", required=True,
                        help="the best FIXED policy as M3a measured it (§5.5)")
    parser.add_argument("--serve-url", required=True,
                        help="the OpenAI-compatible endpoint holding the challenger")
    parser.add_argument("--sandbox-host", required=True,
                        help="the docker daemon a graded patch executes on (§8b.9)")
    parser.add_argument("--grade-dir", type=Path,
                        help="where the sandbox stages a task, if not the default")
    parser.add_argument("--benchmarks", nargs="+", default=["livecodebench", "hle"],
                        choices=("livecodebench", "hle"),
                        help="at least two: a duel cannot judge breadth on one (§5.2)")
    parser.add_argument("--tasks", type=int, default=40,
                        help="tasks PER BENCHMARK; the duel buys one arm of these per side. The "
                             "power gate needs its paired bootstrap lower bound above zero: at the "
                             "measured spread (cascade +0.14 over the floor, per-task SD ~0.3) "
                             "that takes ~40, and 8 measured lcb +0.0000")
    parser.add_argument("--minimum", type=int, default=2,
                        help="tasks a window needs before it settles anything")
    parser.add_argument("--budget-usd", type=float, default=2.0,
                        help="the challenger's pre-funded allowance (§4)")
    parser.add_argument("--owner-usd", type=float, default=5.0,
                        help="the owner's allowance, which pays the king and both reference arms")
    parser.add_argument("--wait-seconds", type=float, default=1_800.0,
                        help="how long to wait for the endpoint to report the artifact")
    parser.add_argument("--out", type=Path, default=Path("v3-live-smoke"),
                        help="scratch root: bucket, state, reveal")
    args = parser.parse_args(argv)

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is unset. This smoke spends real money on purpose — "
                         "that is the half of the daemon the offline tests cannot reach.")

    # §8b.9 before anything else, exactly as `validator.main` does it: where a model's code executes
    # is an argument, and `sandbox.docker_host()` reads it back at every call site.
    os.environ[sandbox.DOCKER_HOST_ENV] = args.sandbox_host
    if args.grade_dir is not None:
        os.environ[sandbox.GRADE_DIR_ENV] = str(args.grade_dir)
    print(f"[smoke] {sandbox.check_grading_host()}")

    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    bucket = S3Bucket(DirectoryClient(root / "bucket"), BUCKET)

    benchmarks, tasks = build_benchmarks(args.benchmarks, args.tasks, NONCE)
    print(f"[smoke] {len(tasks)} tasks over {args.benchmarks}, {args.king0} on the throne")
    # The same rule as `world.from_env`: the arms run on the pool λ was fitted on. Measured on the
    # first run of this smoke (2026-09-07): without it the cascade climbed the whole catalog to
    # `openai/gpt-4` and the reference arm cost $0.038 a task against a table fitted at $0.005.
    pool = world_mod.load_pool(args.exchange)
    if pool is None:
        raise SystemExit(f"{args.exchange} records no `_pool`; re-render it with "
                         f"`scripts/m3a_spread.py --report-only` so the smoke prices the ladder "
                         f"it runs")
    pins = world_mod.pins(exchange_path=args.exchange, catalog=snapshot(fetch()),
                          benchmarks=benchmarks, tasks=tasks, quality={args.king0: 1.0},
                          pool=pool, worker=OpenRouterClient(key))

    chain = MockChain(immunity=CADENCE.immunity_blocks)
    chain.register("burn")                                   # uid 0 — where King0's share burns (§5.7)
    gateway = OwnerGateway(OpenRouterClient(key), journal=allowance_journal(root / "state"))
    gateway.fund("owner", args.owner_usd)
    owner_signer = signing.Signer()

    validator = Validator(
        pins=pins, reference=describe(args.reference_tree), chain=chain, store=bucket,
        mailbox=Mailbox(root / "state" / "mailbox.json", signing.Signer(), _never_mint),
        gateway=gateway,
        owner=Owner(ss58=owner_signer.public_hex, sign=owner_signer.sign,
                    verify_sig=lambda data, sig, who: signing.verify(who, data, sig)),
        cadence=CADENCE, serve=local_serving(args.serve_url), beacon=lambda _window: NONCE,
        netuid=NETUID, root=root / "state",
        per_benchmark=args.tasks, minimum=args.minimum)
    chain.commit_schedule(validator.manifest["root"])         # the owner's side of §6.3

    # --- the miner's side, every step the production path -------------------------------------
    signing_key = miner_key("v3-live-smoke-challenger")
    hotkey = encode_ss58(signing_key.public_key().public_bytes_raw())
    chain.block = 500
    registration = Registration(netuid=NETUID, uid=chain.register(hotkey), hotkey=hotkey,
                                registration_block=500)
    manifest = build_manifest(args.model_dir, registration,
                              lambda body: signing_key.sign(body).hex())
    committed = already_committed(bucket, registration.prefix, manifest)
    if committed is None:
        print(f"[smoke] uploading {len(manifest.files)} files under {registration.prefix}")
        upload_tree(args.model_dir, bucket, registration.prefix, manifest)
    else:
        print(f"[smoke] resuming: {registration.prefix} already holds this submission "
              f"({committed[:12]}…), not uploading twice")
    chain.block = 1_010
    chain.commit_ready(hotkey, ReadySignal(registration_id=registration.registration_id,
                                           manifest_sha256=manifest.sha256))
    gateway.fund(hotkey, args.budget_usd)

    commitment = next(c for c in chain.commitments() if c.hotkey == hotkey)
    served = served_name(commitment)
    print("\n[smoke] launch the challenger under EXACTLY this name, then leave it up:\n")
    print("    " + serve_mod.launch_command(str(args.model_dir), served, gpu=0, port=8002) + "\n")
    wait_for(args.serve_url, served, args.wait_seconds)
    print(f"[smoke] {args.serve_url} is serving {served}")

    # --- the owner opens the window, and the daemon runs it ------------------------------------
    chain.block = CADENCE.opens_at(1)
    record = build_window(validator._entries[1], tasks=pins.world.tasks, catalog=pins.world.catalog,
                          nonce=NONCE, revealed=sorted(validator.history.revealed))
    bucket.put(window_path(1), signing.canonical(
        {"record": record, "sig": owner_signer.sign(signing.canonical(record))}))

    reveal = validator.run_window(1)
    print()
    print(format_reveal(reveal))
    # Read the reveal back out of the bucket rather than re-serialising it here: what a miner
    # audits is the published object, and a smoke that saved its own copy would not have
    # checked that the publish happened at all.
    (root / "reveal.json").write_bytes(bucket.get(reveal_path(1)))

    spent = sum(arm.audit.metered_usd for arm in reveal.arms)
    print(f"\n[smoke] ${spent:.4f} through the meter across {len(reveal.arms)} arms; "
          f"crowned {reveal.report.crowned or 'nobody'}")
    problem = verdict_of(reveal)
    if problem:
        print(f"[smoke] FAILED: {problem}")
        return 1
    print("[smoke] M9 exit 11: a real challenger was duelled on real tasks for real money")
    return 0


def verdict_of(reveal) -> str | None:
    """Why this window is NOT exit 11, or None if it is.

    The first run of this smoke (2026-09-07) exited 0 and printed the exit-11 line over a window in
    which the power gate had refused itself at 8 tasks an arm (lcb +0.0000), the challenger was
    DEFERRED, the king's arm never ran and the served conductor received no request at all. The
    check was "some outcome exists", and a deferral is an outcome. Exit 11 is a challenger DUELLED:
    its arm bought, metered and reconciled. Anything short of that is the mechanism refusing —
    correctly — and this smoke saying so rather than certifying a window that touched no seam
    past the reference arms.
    """
    unreconciled = [arm.arm for arm in reveal.arms if not arm.scoreable]
    if unreconciled:
        return f"{unreconciled} did not reconcile against their receipts"
    duelled = [o for o in reveal.report.outcomes if o.status == DUELLED]
    if not duelled:
        statuses = [(o.hotkey[:12], o.status, o.detail[:90]) for o in reveal.report.outcomes]
        return (f"no challenger was duelled — the served conductor was never scored: {statuses}. "
                f"A DEFERRED challenger usually means the power gate refused the window (too few "
                f"tasks for its bootstrap to clear zero); raise --tasks")
    if not any(o.arm is not None for o in duelled):
        return "a challenger was marked duelled but carries no scored arm"
    return None


def _never_mint(prefix: str):
    raise RuntimeError("the validator daemon does not issue credentials (asked for %s)" % prefix)


if __name__ == "__main__":
    raise SystemExit(main())
