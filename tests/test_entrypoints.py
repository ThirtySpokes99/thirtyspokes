"""The three CLIs, driven as an operator drives them — the code `# pragma: no cover` excused.

WHY THIS FILE EXISTS. A 110-agent audit on 2026-09-03 found two CRITICAL defects that had shipped
inside `main()` bodies carrying that pragma, past 1484 passing tests: `thirtyspokes-owner
commit-schedule` called `_load(args.world)()` when `devkit._load` already invokes the attribute, so
the ONLY writer of the §6.3 schedule root raised `TypeError` before it reached the chain and the
validator — which refuses to start without a root — could not be started at all; and
`miner.hotkey_seed` read `wallet.hotkey.private_key`, an attribute the pinned `bittensor_wallet`
Keypair does not have, so every real submission would have died at the one moment a miner cannot
afford a surprise. Both were one-line wiring mistakes in the layer nothing exercised.

So what is tested here is the WIRING, not the parts: every module these commands call already has
its own tests, and every one of them was green while the subnet was unrunnable. The seams are
replaced at their module attributes — `chain.BittensorChain`, `store.r2_bucket`,
`bittensor_wallet.Wallet`, `pool.fetch` — because those are the four places that would reach a
network, a wallet or an account, and replacing them there leaves the whole of `main()` in the test.

THE ASSERTIONS ARE CROSS-CHECKS AGAINST THE OTHER SIDE, never against a literal this file also
computes. The root `commit-schedule` writes is compared to the one a real `Validator` computes from
the same world (that equality is the whole of §6.3 — a root that only matches itself is a number
nobody can check a window against); the mailbox key `thirtyspokes-miner identity` tells a miner to
poll is compared to the key `thirtyspokes-owner issue` actually published to; and the envelope the
owner published is opened with the miner's own seed against the public key `owner ... key` printed.
Each of those spans two programs that derive their answer independently, which is what makes them
able to fail.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_admission import _reference_tree
from test_gateway import FakeProvider
from test_store import BUCKET, FakeS3

# OPTIONAL AT THE SEAM, GUARDED AT THE IMPORT. `bittensor` lives in the `chain` extra, not the base
# install, so a bare module-level import turns a checkout without that extra into a collection error
# for this whole file — nineteen tests silently not running rather than failing, which is how a
# missing dependency reads as a passing suite. The convention here is `importorskip`
# (`test_client.py`, `test_enclave_cli.py`).
bittensor_wallet = pytest.importorskip("bittensor_wallet")

from thirtyspokes.v3.benchmarks import sandbox
from thirtyspokes.v3.gateway import OwnerGateway, allowance_journal
from thirtyspokes.v3 import chain as chain_module
from thirtyspokes.v3 import miner as miner_tool
from thirtyspokes.v3 import owner as owner_tool
from thirtyspokes.v3 import validator as validator_tool
from thirtyspokes.v3.validator import ValidatorError
from thirtyspokes.v3 import pool as pool_module
from thirtyspokes.v3 import reference, world
from thirtyspokes.v3 import store as store_module
from thirtyspokes.v3.access import (
    Registration,
    credential_from_envelope,
    encode_ss58,
    mailbox_key,
    open_credential,
)
from thirtyspokes.v3.benchmarks import hle as hle_module
from thirtyspokes.v3.benchmarks import real as real_module
from thirtyspokes.v3.chain import ChainError, MockChain
from thirtyspokes.v3.pool import Catalog
from thirtyspokes.v3.store import S3Bucket
from thirtyspokes.v3.types import CatalogEntry, TaskSpec
from thirtyspokes.v3.validator import Cadence, Validator

ENDPOINT = "https://acct.r2.cloudflarestorage.com"
MINER_SEED = bytes.fromhex("33" * 32)


def _address(seed: bytes) -> str:
    return encode_ss58(Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw())


MINER = _address(MINER_SEED)
STRANGER = _address(bytes.fromhex("44" * 32))

REGISTRATION_BLOCK = 5_000_000
NETUID = 7


def _entry(model_id: str, price: float) -> CatalogEntry:
    return CatalogEntry(model_id=model_id, price_in_per_mtok=price, price_out_per_mtok=price * 4,
                        context_length=128_000)


CATALOG = Catalog(entries=(_entry("cheap/one", 0.05), _entry("mid/two", 0.30),
                           _entry("strong/three", 1.00)))
# TWO priced benchmarks, because `duel` refuses a slice it cannot judge breadth on and `world.pins`
# now refuses that corpus at launch rather than letting every window raise.
TASKS = tuple(TaskSpec(task_id=f"{b}-{i}", benchmark=b, prompt="q", tools=())
              for b in ("livecodebench", "hle") for i in range(6))
TABLE = {"livecodebench": {"lam": 0.25, "c_per_task": 0.04, "flags": []},
         "hle": {"lam": 0.18, "c_per_task": 0.002, "flags": []}}

_TABLE_DIR = Path(tempfile.mkdtemp(prefix="v3-entrypoints-"))


def production_world():
    """The `--world module:attr` factory both `commit-schedule` and the daemon resolve.

    MODULE-LEVEL AND ZERO-ARGUMENT because that is the contract `devkit._load` imposes — it imports
    the module and CALLS the attribute — and because the defect this file was written for was a
    second call on the value it returns. A fixture could not stand in: nothing in `_load`'s path can
    be handed a `tmp_path`, so the table it reads is written beside the module instead.
    """
    table = _TABLE_DIR / "m3a-exchange.json"
    table.write_text(json.dumps(TABLE), encoding="utf-8")
    return world.pins(exchange_path=table, catalog=CATALOG, benchmarks=(), tasks=TASKS,
                      worker=object())


def validator_root(root: Path, *, windows: int, per_benchmark: int, minimum: int) -> str:
    """The root the DAEMON computes, from the real `Validator.__post_init__`.

    Recomputing `schedule(...)` and `holdout_feed.manifest(...)` here instead would compare the
    owner's arithmetic against a copy of the owner's arithmetic; §6.3 needs the two programs to
    agree, so the comparison has to come from the other program. The seams left `None` are the ones
    `__post_init__` never touches — it reads `pins.world.tasks`, `cadence.windows` and the two
    counts, and nothing else, which is exactly the state a daemon has before its first poll.
    """
    daemon = Validator(
        pins=production_world(), reference=None, chain=None, store=None, mailbox=None,
        private_models=None, public_models=None,
        gateway=None, owner=None,
        cadence=Cadence(genesis_block=1, window_blocks=360, windows=tuple(range(1, windows + 1)),
                        immunity_blocks=1_080),
        serve=None, beacon=None, netuid=NETUID, root=root, per_benchmark=per_benchmark,
        minimum=minimum)
    return daemon.manifest["root"]


@pytest.fixture
def chain() -> MockChain:
    """A registered miner on a chain both CLIs are pointed at, as the real one would be."""
    mock = MockChain()
    mock.advance(REGISTRATION_BLOCK)
    mock.register(MINER)
    return mock


@pytest.fixture
def bucket() -> S3Bucket:
    return S3Bucket(FakeS3(), BUCKET)


@pytest.fixture
def wired(monkeypatch, chain, bucket) -> dict:
    """The four seams a `main()` reaches a service through, replaced at their module attributes.

    Every one of these is imported INSIDE the function under test (`from .chain import
    BittensorChain`, `from .store import r2_bucket`, `from bittensor_wallet import Wallet`), so the
    name is resolved on the module at call time and patching it there leaves the entire body of
    `main()` — argument parsing, ordering, derivation, printing — running as an operator runs it.
    """
    built: list[dict] = []
    opened: list = []

    def fake_chain(**kwargs):
        built.append(kwargs)
        return chain

    def fake_bucket(credential):
        opened.append(credential)
        return bucket

    class NoWallet:
        def __init__(self, *args, **kwargs):
            raise AssertionError("this command must not open a wallet")

    monkeypatch.setattr(chain_module, "BittensorChain", fake_chain)
    monkeypatch.setattr(store_module, "r2_bucket", fake_bucket)
    monkeypatch.setattr(bittensor_wallet, "Wallet", NoWallet)
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "parent-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "parent-secret")
    return {"chain": built, "bucket": opened}


# Where `issue` scopes a miner's credential. The envelope still goes to BUCKET, the public store.
PRIVATE_BUCKET = "v3-private-models"


def owner_argv(state: Path, *rest: str) -> list[str]:
    return ["--state", str(state), *rest]


def issue_argv(state: Path, hotkey: str = MINER) -> list[str]:
    return owner_argv(state, "issue", "--hotkey", hotkey, "--netuid", str(NETUID),
                      "--network", "test", "--wallet", "owner-wallet",
                      "--owner-hotkey", "owner-hot", "--r2-endpoint", ENDPOINT,
                      "--r2-bucket", BUCKET, "--r2-private-model-bucket", PRIVATE_BUCKET)


# --- thirtyspokes-owner commit-schedule: the command that was dead on arrival -------------------


def test_commit_schedule_puts_the_root_the_validator_computes_on_chain(tmp_path, wired, chain):
    """THE DEFECT THIS FILE EXISTS FOR. `_load` already calls the attribute, so the extra `()` this
    command used to make called `Pins(...)` and raised `TypeError` — and this is the only writer of
    the §6.3 root, without which `Validator.preflight` refuses to start. The subnet was unrunnable
    and nothing said so.

    The root must be the DAEMON's, not merely some root: `preflight` compares what is on chain
    against what it computed and stops the daemon on a mismatch, so a command that committed a
    well-formed root of its own would fail exactly as loudly as committing nothing at all.
    """
    owner_tool.main(owner_argv(tmp_path / "state", "commit-schedule",
                               "--netuid", str(NETUID), "--network", "test",
                               "--wallet", "owner-wallet", "--owner-hotkey", "owner-hot",
                               "--world", f"{__name__}:production_world",
                               "--windows", "4", "--per-benchmark", "7", "--minimum", "3"))

    assert chain.schedule_root() == validator_root(tmp_path / "daemon", windows=4,
                                                   per_benchmark=7, minimum=3)


def test_commit_schedule_reads_the_chain_under_the_hotkey_the_validator_reads_back_with(
        tmp_path, wired):
    """§6.3's own refusal text: the root is read back from the SIGNING KEY's commitment slot, so a
    root committed under a different hotkey is indistinguishable from no root at all. Every one of
    these four values decides which slot is written."""
    owner_tool.main(owner_argv(tmp_path / "state", "commit-schedule",
                               "--netuid", str(NETUID), "--network", "test",
                               "--wallet", "owner-wallet", "--owner-hotkey", "owner-hot",
                               "--world", f"{__name__}:production_world",
                               "--windows", "2", "--per-benchmark", "5", "--minimum", "2"))

    assert wired["chain"] == [{"netuid": NETUID, "wallet_name": "owner-wallet",
                               "network": "test", "hotkey": "owner-hot"}]


def test_commit_schedule_replaces_a_retired_governance_record_only_when_told(tmp_path, wired,
                                                                             chain):
    """Testnet 526 and mainnet 99 both hold KOTH's `kothgov1|…` in the owner's slot. The CLI must
    refuse by default and overwrite only under `--replace-governance-record`."""
    chain._schedule = "kothgov1|" + "de" * 16
    argv = owner_argv(tmp_path / "state", "commit-schedule",
                      "--netuid", str(NETUID), "--network", "test",
                      "--wallet", "owner-wallet", "--owner-hotkey", "owner-hot",
                      "--world", f"{__name__}:production_world",
                      "--windows", "2", "--per-benchmark", "5", "--minimum", "2")
    with pytest.raises(ChainError, match="not a schedule root"):
        owner_tool.main(argv)
    assert chain._schedule.startswith("kothgov1|")
    owner_tool.main(argv + ["--replace-governance-record"])
    assert chain.schedule_root() == validator_root(tmp_path / "daemon", windows=2,
                                                   per_benchmark=5, minimum=2)


def test_the_committed_root_moves_when_the_schedule_parameters_do(tmp_path, wired, chain):
    """A root that did not depend on the window count or the stratification would pin nothing, and
    the owner could re-derive any slate under it after seeing who committed."""
    roots = []
    for windows, per_benchmark, minimum in ((4, 7, 3), (5, 7, 3), (4, 6, 3), (4, 7, 2)):
        owner_tool.main(owner_argv(tmp_path / "state", "commit-schedule",
                                   "--netuid", str(NETUID), "--wallet", "owner-wallet",
                                   "--world", f"{__name__}:production_world",
                                   "--windows", str(windows),
                                   "--per-benchmark", str(per_benchmark),
                                   "--minimum", str(minimum)))
        roots.append(chain.schedule_root())
    assert len(set(roots)) == len(roots), "the root does not distinguish the schedules it pins"


# --- thirtyspokes-owner issue / status / key ----------------------------------------------------


def test_issue_publishes_an_envelope_the_miner_can_actually_open(tmp_path, wired, chain, bucket,
                                                                 capsys):
    """§7 step 3 end to end through the CLI: mint, seal, publish — and the seal is verified from the
    MINER's side, with the miner's own seed and the public key `owner ... key` prints.

    Opening it is the only assertion that can fail for the right reason. `open_credential` checks
    the signature, the registration and the generation, so an envelope published under a per-run
    signing key, sealed to the wrong hotkey, or scoped to a registration derived from anything but
    the four chain facts is refused here rather than three hours into a miner's upload.
    """
    state = tmp_path / "state"
    owner_tool.main(issue_argv(state))
    printed = capsys.readouterr().out
    owner_tool.main(owner_argv(state, "key"))
    owner_public_hex = capsys.readouterr().out.strip()

    registration = Registration(netuid=NETUID, uid=0, hotkey=MINER,
                                registration_block=REGISTRATION_BLOCK)
    key = mailbox_key(registration.registration_id, 1)
    envelope = open_credential(bucket.get(key), MINER_SEED, owner_public_hex=owner_public_hex,
                               registration=registration, generation=1)
    credential = credential_from_envelope(envelope)
    # Published on the public store, but scoped to the PRIVATE models bucket: a miner's weights
    # land where nobody can download them unless they win.
    assert credential.bucket == PRIVATE_BUCKET and credential.endpoint == ENDPOINT
    # The operator is told where it went, and the parent credential really was the one from the env.
    assert key in printed and registration.prefix in printed
    assert wired["bucket"][0].access_key_id == "parent-key"


def test_issue_refuses_to_scope_a_credential_to_the_public_store(tmp_path, wired, bucket):
    """The one misconfiguration nothing downstream would notice: a credential scoped to the bucket
    behind the public domain opens and uploads exactly like a private one, and every byte the miner
    sends is then downloadable by anyone before it has won anything. Refused, and nothing published.
    """
    argv = issue_argv(tmp_path / "state")
    argv[argv.index("--r2-private-model-bucket") + 1] = BUCKET

    with pytest.raises(SystemExit) as caught:
        owner_tool.main(argv)

    assert "must not be the public store" in str(caught.value)
    assert bucket.client.objects == {}


def test_issue_refuses_an_unregistered_hotkey_with_a_message_and_a_nonzero_exit(tmp_path, wired,
                                                                                bucket):
    """An `AccessError` reaching the operator as a traceback would be read as a bug in the tool; the
    exit code is what a deployment script reads. Nothing may be published either — a credential
    scoped to a guessed registration is the one failure the derivation exists to prevent."""
    with pytest.raises(SystemExit) as caught:
        owner_tool.main(issue_argv(tmp_path / "state", STRANGER))
    assert "thirtyspokes-owner:" in str(caught.value) and "holds no uid" in str(caught.value)
    assert bucket.client.objects == {}


def test_status_reads_the_very_ledger_issue_wrote(tmp_path, wired, capsys):
    """`--state` is the DAEMON's state directory and the ledger is one file inside it (§7's one shot
    holds only if there is one ledger). A `status` that looked anywhere else would report "no
    credentials issued" forever, which is also what it reports when nothing was issued — the two
    states are indistinguishable to an operator, and the second is the one that hides a live
    credential from the revocation list `format_status` exists to produce."""
    state = tmp_path / "state"
    owner_tool.main(owner_argv(state, "status"))
    assert capsys.readouterr().out.strip() == "no credentials issued"

    owner_tool.main(issue_argv(state))
    capsys.readouterr()
    owner_tool.main(owner_argv(state, "status"))
    report = capsys.readouterr().out

    registration = Registration(netuid=NETUID, uid=0, hotkey=MINER,
                                registration_block=REGISTRATION_BLOCK)
    assert MINER in report and registration.prefix in report
    assert "REVOKE NOW" not in report          # the shot is unspent: nothing to revoke yet


def test_key_is_the_same_key_on_every_run_and_is_the_one_that_signed_the_envelope(tmp_path, wired,
                                                                                   capsys):
    """`Signer()` GENERATES a keypair, so a tool that built one per invocation would print a
    different `--owner-key` every time and sign each envelope with a key no miner is holding — and
    `open_credential`'s "not signed by the owner" refusal, which exists to stop a miner being
    pointed at somebody else's bucket, would fire on the owner's own envelopes."""
    state = tmp_path / "state"
    owner_tool.main(owner_argv(state, "key"))
    first = capsys.readouterr().out.strip()
    owner_tool.main(owner_argv(state, "key"))
    assert capsys.readouterr().out.strip() == first
    assert len(bytes.fromhex(first)) == 32

    owner_tool.main(issue_argv(state))
    capsys.readouterr()
    owner_tool.main(owner_argv(state, "key"))
    assert capsys.readouterr().out.strip() == first, "issuing rotated the published owner key"


# --- thirtyspokes-miner check / identity --------------------------------------------------------


def test_check_runs_the_validators_gate_with_no_wallet_and_no_chain(tmp_path, wired, capsys):
    """§7 step 1 is local and comes BEFORE anything on the network, and this is what that buys: a
    miner can check a tree before they have a funded, registered, ed25519 hotkey. `wired` opens the
    wallet only over an `AssertionError`, and a chain read would leave a record here — either would
    make the gate cost a miner something before it has told them anything."""
    tree = _reference_tree(tmp_path / "model")
    miner_tool.main(["--netuid", str(NETUID), "--wallet", "miner-wallet", "check",
                     "--model", str(tree), "--reference", str(_reference_tree(tmp_path / "ref"))])
    assert "admitted" in capsys.readouterr().out
    assert wired["chain"] == []


def test_check_refuses_in_the_validators_own_words_and_exits_nonzero(tmp_path, wired):
    """`check_admission` returns the refusal VERBATIM so a miner greps for the string the validator
    would print; a CLI that summarised it would put a divergence between local and scored behaviour
    behind a nicety. The exit code is what tells a submit script to stop."""
    tree = _reference_tree(tmp_path / "model")
    (tree / "pytorch_model.bin").write_bytes(b"\x80\x04")     # a pickle: §1.1's first refusal
    with pytest.raises(SystemExit) as caught:
        miner_tool.main(["--netuid", str(NETUID), "--wallet", "miner-wallet", "check",
                         "--model", str(tree),
                         "--reference", str(_reference_tree(tmp_path / "ref"))])
    message = str(caught.value)
    assert "thirtyspokes-miner:" in message and "pickle archive refused" in message
    assert "REFUSED at admission" in message


def test_identity_names_the_exact_mailbox_key_the_owner_publishes_to(tmp_path, monkeypatch, wired,
                                                                     chain, bucket, capsys):
    """§7's whole point: both sides derive the identity from the same four chain facts and neither
    chooses it. So the key a miner is told to poll and the key the owner uploaded to are the same
    string, computed by two programs that never speak — and if they are not, the miner polls an
    empty key forever while a live credential sits in the bucket under another name."""
    class FakeWallet:
        def __init__(self, name, hotkey):
            self.name, self.hotkey_name = name, hotkey
            self.hotkey = type("Keypair", (), {"ss58_address": MINER})()

    monkeypatch.setattr(bittensor_wallet, "Wallet", FakeWallet)
    state = tmp_path / "state"
    owner_tool.main(issue_argv(state))
    published = next(iter(bucket.client.objects))
    capsys.readouterr()

    miner_tool.main(["--netuid", str(NETUID), "--wallet", "miner-wallet", "--hotkey", "default",
                     "identity"])
    printed = capsys.readouterr().out

    assert published in printed
    registration = Registration(netuid=NETUID, uid=0, hotkey=MINER,
                                registration_block=REGISTRATION_BLOCK)
    assert registration.registration_id in printed and registration.prefix in printed
    assert str(REGISTRATION_BLOCK) in printed and MINER in printed


def test_identity_derives_from_the_wallets_own_address_not_from_the_wallet_name(tmp_path,
                                                                                monkeypatch, wired,
                                                                                capsys):
    """The identity is the HOTKEY's, read off the wallet. A command that derived it from anything an
    operator typed would hand a miner a prefix that is somebody else's — the failure `Registration`
    is derived rather than typed to make impossible."""
    class WrongKeyWallet:
        def __init__(self, name, hotkey):
            self.hotkey = type("Keypair", (), {"ss58_address": STRANGER})()

    monkeypatch.setattr(bittensor_wallet, "Wallet", WrongKeyWallet)
    with pytest.raises(SystemExit) as caught:
        miner_tool.main(["--netuid", str(NETUID), "--wallet", "miner-wallet", "identity"])
    assert "holds no uid" in str(caught.value)


# --- thirtyspokes.v3.world:from_env — the production `--world` seam ------------------------------


class FakeLCB:
    """`benchmarks.real.LiveCodeBench` without the download, and the second priced benchmark the
    corpus needs: one is not a corpus a duel can judge breadth on."""

    name = "livecodebench"

    def __init__(self, releases=()) -> None:
        self.releases = releases

    def draw_across_releases(self, n, *, seed):
        return tuple(TaskSpec(task_id=f"lcb-{i}", benchmark="livecodebench", prompt="q", tools=())
                     for i in range(n))

    def tools(self):
        return ()

    def environment(self, task):
        return None


class FakeHLE:
    """`benchmarks.hle.HumanitysLastExam` without the download. Only `name` and `load` are reached:
    `suite_grade`/`suite_inspect` key their dispatch on the name and defer the rest to a lambda."""

    name = "hle"

    def load(self):
        return [TaskSpec(task_id=f"hle-{i}", benchmark="hle", prompt="q", tools=())
                for i in range(4)]

    def tools(self):
        return ()

    def environment(self, task):
        return None


@pytest.fixture
def env(monkeypatch, tmp_path) -> Path:
    """A complete production environment with no service in it, and NOTHING inherited: every
    variable `from_env` reads is cleared first, so a test asserting a refusal cannot be quietly
    satisfied by the operator's own shell."""
    for name in ("V3_KING0", "V3_BENCHMARKS", "V3_TASKS_PER_BENCHMARK", "V3_EXCHANGE",
                 "V3_SEED", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    table = tmp_path / "m3a-exchange.json"
    table.write_text(json.dumps({"hle": {"lam": 0.25, "c_per_task": 0.002, "flags": []},
                                 "livecodebench": {"lam": 0.30, "c_per_task": 0.04, "flags": []},
                                 # The pool the table was fitted on — every model of CATALOG, so
                                 # the narrowing is a no-op here and the tests below stay about
                                 # what they were about. `test_world` tests the narrowing.
                                 "_pool": [e.model_id for e in CATALOG.entries]}),
                     encoding="utf-8")
    monkeypatch.setattr(hle_module, "HumanitysLastExam", FakeHLE)
    monkeypatch.setattr(real_module, "LiveCodeBench", FakeLCB)
    monkeypatch.setattr(pool_module, "fetch", lambda *a, **k: "catalog-payload")
    monkeypatch.setattr(pool_module, "snapshot", lambda payload: CATALOG)
    monkeypatch.setenv("V3_KING0", reference.ALWAYS_CHEAPEST)
    monkeypatch.setenv("V3_BENCHMARKS", "hle,livecodebench")
    monkeypatch.setenv("V3_TASKS_PER_BENCHMARK", "3")
    monkeypatch.setenv("V3_EXCHANGE", str(table))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    return table


def test_from_env_builds_the_published_world_the_daemon_scores_with(env, monkeypatch):
    """The happy path, which nothing exercised: `--world thirtyspokes.v3.world:from_env` is the
    ONLY production spelling of the flag, and `_load` calls it at daemon start — before the chain,
    the gateway or a single dollar. Everything it assembles is checked, because a world assembled
    wrong is scored with rather than refused."""
    keys: list[str] = []
    monkeypatch.setattr(world, "OpenRouterClient", lambda key, **kw: keys.append(key) or "worker")

    pins = world.from_env()

    # Both benchmarks named in V3_BENCHMARKS, each drawn to V3_TASKS_PER_BENCHMARK=3, and in the
    # order the variable lists them — the slice width is per benchmark, not per corpus.
    assert [task.task_id for task in pins.world.tasks] == [
        "hle-0", "hle-1", "hle-2", "lcb-0", "lcb-1", "lcb-2"]
    assert pins.world.exchange["hle"].lam == 0.25          # the published table, not a fitted one
    # ALWAYS_CHEAPEST rather than the cascade on purpose: `pins` FALLS BACK to the cascade when
    # handed no measurement, so a happy path that named the cascade would pass just as well with
    # V3_KING0 thrown away — which is the §5.5 defect the variable exists to prevent.
    assert pins.king_zero.name == reference.ALWAYS_CHEAPEST
    assert pins.contrast.name == reference.contrast_for(reference.ALWAYS_CHEAPEST)
    assert pins.king_zero.rungs[-1] != pins.contrast.rungs[-1]
    assert keys == ["or-test-key"], "the worker was not built with OPENROUTER_API_KEY"


def test_from_env_refuses_an_unmeasured_king_zero_rather_than_defaulting_it(env, monkeypatch):
    """§5.5: King0 is the best fixed policy AS MEASURED, and it is the bar every challenger must
    beat. Defaulting it silently would price every crown against a policy nobody checked, so the
    refusal has to name the variable and the choices."""
    monkeypatch.delenv("V3_KING0")
    with pytest.raises(world.WorldError) as caught:
        world.from_env()
    assert "V3_KING0" in str(caught.value) and reference.ALWAYS_CHEAPEST in str(caught.value)

    monkeypatch.setenv("V3_KING0", "always-fastest")
    with pytest.raises(world.WorldError, match="not a fixed policy"):
        world.from_env()


def test_from_env_refuses_a_benchmark_it_has_no_adapter_for(env, monkeypatch):
    """A typo in `V3_BENCHMARKS` that silently dropped the benchmark would launch a corpus narrower
    than the one M3a priced, and §5.2's breadth rule would then be applied to it."""
    monkeypatch.setenv("V3_BENCHMARKS", "hle,swebench-pro")
    with pytest.raises(world.WorldError, match="no adapter for benchmark"):
        world.from_env()


def test_from_env_names_the_environment_variable_it_is_missing(env, monkeypatch):
    """Each input is a thing the owner MEASURED or chose, so an absent one must stop the daemon and
    say which — never be defaulted, and never surface as something an operator reads as a crash in
    somebody else's code.

    Measured 2026-09-03: `V3_EXCHANGE` and `OPENROUTER_API_KEY` are read with `os.environ[...]`, so
    what an operator actually gets today is a bare `KeyError` naming the variable, not the paragraph
    `V3_KING0` gets. This accepts either — what must hold is that the run STOPS and the message
    names the variable, which is what a deployment can act on."""
    for name in ("V3_EXCHANGE", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name)
        with pytest.raises((world.WorldError, KeyError)) as caught:
            world.from_env()
        assert name in str(caught.value)
        monkeypatch.setenv(name, "restored")


# --- miner.hotkey_seed: the other defect the audit found -----------------------------------------


class PinnedKeypair:
    """The pinned Keypair's surface where it matters: `sign`, `verify`, `public_key`,
    `ss58_address` — and NO private material. Re-verified against `bittensor_wallet` 4.1.0 here:
    `Keypair.create_from_seed(...)` exposes no `private_key`, no `seed_hex` and no `secret_key`.

    A plain class rather than a Mock, and that is the point of it: an earlier `hotkey_seed` read
    `wallet.hotkey.private_key`, which every permissive double in a test suite would have answered.
    Here the attribute simply does not exist, so the code under test meets the same `AttributeError`
    a miner's one submission would have met.
    """

    def __init__(self, seed: bytes) -> None:
        self._key = Ed25519PrivateKey.from_private_bytes(seed)
        self.public_key = self._key.public_key().public_bytes_raw()
        self.ss58_address = encode_ss58(self.public_key)

    def sign(self, data: bytes) -> bytes:
        return self._key.sign(data)

    def verify(self, data: bytes, signature: bytes) -> bool:
        return True


class FakeKeyfile:
    def __init__(self, path: Path, payload, *, encrypted: bool = False) -> None:
        self.path = str(path)
        self._payload = payload
        self._encrypted = encrypted

    def is_encrypted(self) -> bool:
        return self._encrypted

    def exists_on_device(self) -> bool:
        return self._payload is not None

    @property
    def data(self) -> bytes:
        return json.dumps(self._payload).encode() if isinstance(self._payload, dict) \
            else self._payload


class FakeHotkeyWallet:
    def __init__(self, path: Path, payload, *, seed: bytes = MINER_SEED,
                 encrypted: bool = False) -> None:
        self.hotkey = PinnedKeypair(seed)
        self.hotkey_file = FakeKeyfile(path, payload, encrypted=encrypted)


def test_the_seed_comes_from_the_keyfile_and_never_from_the_keypair(tmp_path):
    """THE SECOND CRITICAL DEFECT. `hotkey_seed` used to read `wallet.hotkey.private_key`, which the
    pinned Keypair does not define — so every real submission would have died with `AttributeError`
    at the step that spends the shot. The seed lives in the keyfile, which is where
    `Keypair.create_from_seed` got it, and `PinnedKeypair` here has no private attribute to fall
    back to."""
    wallet = FakeHotkeyWallet(tmp_path / "hotkey", {"secretSeed": "0x" + MINER_SEED.hex()})
    assert miner_tool.hotkey_seed(wallet) == MINER_SEED


def test_a_64_byte_private_key_field_yields_the_half_that_re_derives_the_address(tmp_path):
    """Keyfiles in the wild carry `privateKey` as seed||public as well as a bare seed, so the field
    is a candidate rather than an answer — each half is tried and the one that re-derives the
    wallet's own address is returned."""
    keypair = PinnedKeypair(MINER_SEED)
    wallet = FakeHotkeyWallet(tmp_path / "hotkey",
                              {"privateKey": (MINER_SEED + keypair.public_key).hex()})
    assert miner_tool.hotkey_seed(wallet) == MINER_SEED


def test_a_seed_that_does_not_re_derive_the_address_is_refused_rather_than_used(tmp_path):
    """THE FAILURE IS SILENT IF IT IS NOT CHECKED HERE. A wrong seed derives a different X25519 key,
    and the miner simply cannot open an envelope that was sealed to them correctly — with nothing to
    point at. The refusal has to name the ed25519 requirement, because an sr25519 hotkey (btcli's
    default) is the ordinary way to arrive at it."""
    wallet = FakeHotkeyWallet(tmp_path / "hotkey", {"secretSeed": "0x" + ("55" * 32)})
    with pytest.raises(miner_tool.MinerError) as caught:
        miner_tool.hotkey_seed(wallet)
    assert "ed25519" in str(caught.value) and "sr25519" in str(caught.value)


def test_an_encrypted_keyfile_is_named_as_such_rather_than_reported_as_a_missing_seed(tmp_path):
    """An ordinary state with an obvious remedy. A miner reading "no usable seed" would go looking
    for a bug that is not there, and the remedy — decrypt the keyfile — would never be tried."""
    wallet = FakeHotkeyWallet(tmp_path / "hotkey", b"\x00encrypted", encrypted=True)
    with pytest.raises(miner_tool.MinerError) as caught:
        miner_tool.hotkey_seed(wallet)
    # The remedy, not the word: `tmp_path` carries this test's own name, so matching on "encrypted"
    # alone passed with the check deleted — the unreadable ciphertext fell through to "cannot read
    # the keyfile at <path containing 'encrypted'>". Measured while mutation-testing this file.
    assert "hotkey_file.decrypt()" in str(caught.value) and "is encrypted" in str(caught.value)


def test_check_certifies_the_subnet_the_daemon_will_refuse_to_run_on(tmp_path, monkeypatch):
    """`--check` is documented as the pre-launch gate and is the one mode safe to point at a live
    chain, so an owner runs it, reads "launch gates pass", and starts the daemon.

    While the chain-side cross-check lived only in `Validator.preflight`, that pair disagreed:
    `--check` certified a subnet whose `immunity_period` was a third of what `--immunity-blocks`
    claimed, and the daemon then refused to start on the very same configuration. Both callers now
    share one function, so a certified launch is a launch that starts.
    """
    chain = chain_module.MockChain(immunity=200)      # the subnet says 200
    chain.register("owner")
    monkeypatch.setattr(chain_module, "BittensorChain", lambda **kw: chain)
    # `--check` reports the grading host before it reaches the gate; that is a different seam and
    # not what this test is about.
    monkeypatch.setattr(sandbox, "check_grading_host", lambda: "grading host ok")
    # `main` writes `--sandbox-host` and `--grade-dir` into the process environment — the single
    # source of truth `sandbox.docker_host()` reads — so the fake host must not outlive this test.
    # Registering the names with monkeypatch restores whatever they held. Measured 2026-09-07: a
    # suite run with a real V3_DOCKER_HOST had every later grading test dial `ssh://x`.
    monkeypatch.delenv(sandbox.DOCKER_HOST_ENV, raising=False)
    monkeypatch.delenv(sandbox.GRADE_DIR_ENV, raising=False)

    argv = ["--netuid", "99", "--wallet", "w", "--hotkey", "h", "--network", "finney",
            "--genesis-block", "1000", "--window-blocks", "200", "--windows", "3",
            "--immunity-blocks", "600",                # the owner claims 600
            "--world", "x:y", "--reference-tree", str(tmp_path), "--serve-url", "http://x",
            "--state", str(tmp_path), "--r2-endpoint", "http://x", "--r2-bucket", "b",
            "--r2-private-model-bucket", "p", "--r2-public-model-bucket", "m",
            "--public-model-base-url", "https://models.example.org",
            "--grade-dir", str(tmp_path), "--sandbox-host", "ssh://x",
            "--per-benchmark", "20", "--minimum", "5", "--check"]

    with pytest.raises(ValidatorError) as caught:
        validator_tool.main(argv)

    assert "immunity_period" in str(caught.value)


def test_the_daemon_refuses_to_start_unless_its_three_buckets_are_distinct(tmp_path, monkeypatch):
    """Private submissions are a deployment property, so the deployment is where it is refused: a
    private models bucket that is the public store would publish every challenger, and a public
    models bucket that is the private one would publish no king. Refused before dialling the chain.
    """
    monkeypatch.setattr(chain_module, "BittensorChain",
                        lambda **kw: pytest.fail("the daemon reached the chain"))
    monkeypatch.delenv(sandbox.DOCKER_HOST_ENV, raising=False)
    monkeypatch.delenv(sandbox.GRADE_DIR_ENV, raising=False)
    argv = ["--netuid", "99", "--wallet", "w", "--hotkey", "h", "--network", "finney",
            "--genesis-block", "1000", "--window-blocks", "200", "--windows", "3",
            "--immunity-blocks", "600", "--world", "x:y", "--reference-tree", str(tmp_path),
            "--serve-url", "http://x", "--state", str(tmp_path), "--r2-endpoint", "http://x",
            "--r2-bucket", "store", "--r2-private-model-bucket", "store",
            "--r2-public-model-bucket", "kings", "--sandbox-host", "ssh://x",
            "--public-model-base-url", "https://models.example.org",
            "--per-benchmark", "20", "--minimum", "5", "--check"]

    with pytest.raises(SystemExit) as caught:
        validator_tool.main(argv)

    assert "three distinct buckets" in str(caught.value)


@pytest.mark.parametrize("url", ["http://models.example.org", "models.example.org",
                                 "https://models.example.org/?token=x"])
def test_the_daemon_refuses_a_public_model_base_url_that_is_not_plain_https(tmp_path, monkeypatch,
                                                                            url):
    """The address is published in every reveal as where anyone downloads the king: plain http can be
    rewritten by any network in between, a bare host is not something a downloader can join a path
    onto, and a query string would hand whatever it carries to everyone who reads the reveal."""
    monkeypatch.setattr(chain_module, "BittensorChain",
                        lambda **kw: pytest.fail("the daemon reached the chain"))
    monkeypatch.delenv(sandbox.DOCKER_HOST_ENV, raising=False)
    monkeypatch.delenv(sandbox.GRADE_DIR_ENV, raising=False)
    argv = ["--netuid", "99", "--wallet", "w", "--hotkey", "h", "--network", "finney",
            "--genesis-block", "1000", "--window-blocks", "200", "--windows", "3",
            "--immunity-blocks", "600", "--world", "x:y", "--reference-tree", str(tmp_path),
            "--serve-url", "http://x", "--state", str(tmp_path), "--r2-endpoint", "http://x",
            "--r2-bucket", "store", "--r2-private-model-bucket", "private",
            "--r2-public-model-bucket", "kings", "--public-model-base-url", url,
            "--sandbox-host", "ssh://x", "--per-benchmark", "20", "--minimum", "5", "--check"]

    with pytest.raises(SystemExit) as caught:
        validator_tool.main(argv)

    assert "--public-model-base-url must be a plain https:// address" in str(caught.value)


def test_credit_puts_money_where_the_daemon_will_look_for_it(tmp_path, capsys):
    """The money path end to end, across the two programs that have to agree about it.

    Neither half was exercised by anything: `credit` could be changed to record nothing and fund
    nobody, and the daemon's `journal=allowance_journal(args.state)` could be dropped entirely, and
    the suite stayed green both times. That is the gap the commit adding the money path said it was
    closing.
    """
    state = tmp_path / "state"
    state.mkdir()

    owner_tool.main(["--state", str(state), "credit", "--hotkey", "owner",
                     "--usd", "40", "--ref", "extrinsic 0x91af"])
    owner_tool.main(["--state", str(state), "credit", "--hotkey", "5MINER", "--usd", "12.5"])

    # Built exactly as `validator.main` builds it — same derivation, same file.
    daemon = OwnerGateway(FakeProvider(), journal=allowance_journal(state))

    assert daemon.balance("owner") == 40.0
    assert daemon.balance("5MINER") == 12.5
    # WHAT THIS DOES NOT COVER, stated rather than implied: `validator.main` is wiring behind a live
    # chain, R2 credentials and a wallet, and stays `# pragma: no cover`. This pins the two halves
    # that CAN be driven — `credit` writes, and a gateway built from the same `--state` reads what it
    # wrote. Deleting `journal=allowance_journal(args.state)` from `main` would still pass; the
    # protection there is that both sides derive the path from one function, which the test below
    # pins.


def test_balances_reports_what_credit_wrote(tmp_path, capsys):
    state = tmp_path / "state"
    state.mkdir()
    owner_tool.main(["--state", str(state), "balances"])
    assert "no allowances credited" in capsys.readouterr().out

    owner_tool.main(["--state", str(state), "credit", "--hotkey", "5MINER", "--usd", "7.25"])
    capsys.readouterr()
    owner_tool.main(["--state", str(state), "balances"])

    out = capsys.readouterr().out
    assert "5MINER" in out and "7.2500" in out


def test_the_journal_path_is_derived_from_state_so_the_two_cannot_disagree(tmp_path):
    """The failure this forecloses is an owner crediting one file while the daemon reads another,
    and starting against a wallet they believe they filled."""
    assert allowance_journal(tmp_path) == tmp_path / "allowances.jsonl"


def test_from_env_refuses_a_table_that_does_not_say_which_pool_it_priced(env, monkeypatch):
    """MEASURED 2026-09-07, first live launch: the table was fitted over five models, `from_env`
    handed the daemon the whole catalog, and King0's cascade climbed to `openai/gpt-4` — a rung the
    price had never seen, at twenty times the per-call cost. A table without its `_pool` cannot say
    what ladder it prices, so the production seam refuses it and names the re-render that fixes it."""
    env.write_text(json.dumps({"hle": {"lam": 0.25, "c_per_task": 0.002, "flags": []},
                               "livecodebench": {"lam": 0.30, "c_per_task": 0.04, "flags": []}}),
                   encoding="utf-8")
    monkeypatch.setattr(world, "OpenRouterClient", lambda key, **kw: "worker")

    with pytest.raises(world.WorldError, match="records no `_pool`.*--report-only"):
        world.from_env()


# --- thirtyspokes-miner hotkey: the ed25519 check, BEFORE the registration burn -----------------


def test_hotkey_command_approves_an_ed25519_keyfile_without_touching_the_chain(tmp_path, monkeypatch,
                                                                             wired, capsys):
    """`submit` refuses a sr25519 wallet, but only after the miner has paid to register it. This
    command is the same `hotkey_seed` check run first, and it must not need a chain: a miner runs it
    before they have a uid."""
    monkeypatch.setattr(bittensor_wallet, "Wallet", lambda name, hotkey: FakeHotkeyWallet(
        tmp_path / "hotkey", {"secretSeed": "0x" + MINER_SEED.hex()}))
    miner_tool.main(["--netuid", str(NETUID), "--network", "test", "--wallet", "w", "--hotkey", "h",
                     "hotkey"])
    printed = capsys.readouterr().out
    assert MINER in printed and "ed25519" in printed
    assert "btcli subnets register --netuid" in printed and "--crypto-type" not in printed
    assert wired["chain"] == [], "the check must not open a chain connection"


def test_hotkey_command_refuses_a_non_ed25519_keyfile_and_names_the_btcli_flag(tmp_path, monkeypatch,
                                                                             wired):
    """The remedy has to name btcli's real flag: it is `--crypto-type`, and a miner who follows a
    wrong flag name into a second sr25519 hotkey has paid two registration burns for nothing."""
    monkeypatch.setattr(bittensor_wallet, "Wallet", lambda name, hotkey: FakeHotkeyWallet(
        tmp_path / "hotkey", {"secretSeed": "0x" + ("55" * 32)}))
    with pytest.raises(SystemExit) as caught:
        miner_tool.main(["--netuid", str(NETUID), "--wallet", "w", "--hotkey", "h", "hotkey"])
    message = str(caught.value)
    assert "not an ed25519 hotkey" in message and "--crypto-type ed25519" in message
    assert wired["chain"] == []


def test_a_missing_keyfile_is_named_as_such_rather_than_reported_as_a_decode_error(tmp_path,
                                                                                monkeypatch, wired):
    """Measured 2026-09-07 on the GPU box before the keyfile had been copied there: the wallet API
    returns None for a file that is not on the device, and the tool said "cannot convert 'NoneType'
    object to bytes" — a bug report where the fix is `scp`."""
    monkeypatch.setattr(bittensor_wallet, "Wallet",
                        lambda name, hotkey: FakeHotkeyWallet(tmp_path / "hotkey", None))
    with pytest.raises(SystemExit) as caught:
        miner_tool.main(["--netuid", str(NETUID), "--wallet", "w", "--hotkey", "h", "hotkey"])
    message = str(caught.value)
    assert "no hotkey keyfile at" in message and "NoneType" not in message


def test_the_mailbox_poll_names_itself_because_the_public_domain_refuses_the_default_agent(monkeypatch):
    """Cloudflare's r2.dev answers 403 to `Python-urllib/…` and 200 to any named agent on the SAME
    object (measured 2026-09-08 on the owner's bucket): both rehearsal submits failed at their first
    step with a "403 Forbidden" that no bucket setting could explain, since the object was public and
    curl fetched it. The header is the whole fix, so it is the whole assertion."""
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"sealed"

    def fake_urlopen(request, timeout):
        seen["agent"] = request.get_header("User-agent")
        seen["url"] = request.full_url
        return Response()

    monkeypatch.setattr(miner_tool.urllib.request, "urlopen", fake_urlopen)
    assert miner_tool.fetch_envelope("https://pub.example/", "mailbox/v1/x/1.bin") == b"sealed"
    assert seen["agent"] == miner_tool.USER_AGENT and "urllib" not in seen["agent"]
    assert seen["url"] == "https://pub.example/mailbox/v1/x/1.bin"


def test_the_miner_tools_poll_the_subnets_public_store_unless_told_otherwise():
    """`store.thirtyspokes.ai` is the store (config.PUBLIC_STORE); a miner types it nowhere, and a
    rehearsal against another bucket still can."""
    from thirtyspokes.v3 import miner as miner_tool
    from thirtyspokes.v3.config import PUBLIC_STORE

    parser = miner_tool._parser()
    common = ["--netuid", "99", "--wallet", "w"]
    assert parser.parse_args([*common, "submit", "--model", "m", "--reference", "r",
                              "--owner-key", "ab"]).mailbox_url == PUBLIC_STORE
    assert parser.parse_args([*common, "register-key", "--cap-usd", "5",
                              "--owner-key", "ab"]).mailbox_url == PUBLIC_STORE
    assert parser.parse_args([*common, "register-key", "--cap-usd", "5", "--owner-key", "ab",
                              "--mailbox-url", "https://x.example"]).mailbox_url == "https://x.example"
    assert PUBLIC_STORE == "https://store.thirtyspokes.ai"


def test_the_owner_key_is_pinned_for_netuid_99_and_required_everywhere_else():
    """A miner is never told the owner key, so there is nothing for an impersonator to substitute —
    the key a miner supplies is the only thing `open_credential` trusts and what `register-key`
    seals a funded OpenRouter key to. But a default that followed a miner onto a rehearsal would seal
    a real key to netuid 99's owner without a word, so off that subnet the flag stays required."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from thirtyspokes.v3 import miner as miner_tool
    from thirtyspokes.v3.config import OWNER_MAILBOX_KEY, OWNER_MAILBOX_SUBNET

    # A malformed pin would make every submit on the subnet refuse every genuine envelope.
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(OWNER_MAILBOX_KEY))
    assert OWNER_MAILBOX_SUBNET == ("finney", 99)

    parser = miner_tool._parser()
    on_99 = parser.parse_args(["--netuid", "99", "--wallet", "w",
                               "submit", "--model", "m", "--reference", "r"])
    assert miner_tool._owner_key(on_99) == OWNER_MAILBOX_KEY

    override = parser.parse_args(["--netuid", "99", "--wallet", "w",
                                  "register-key", "--cap-usd", "5", "--owner-key", "ab"])
    assert miner_tool._owner_key(override) == "ab"

    for where in (["--netuid", "526", "--network", "test"],   # the rehearsal subnet
                  ["--netuid", "98"],                          # another finney subnet
                  ["--netuid", "99", "--network", "test"]):    # netuid 99 is not unique across networks
        rehearsal = parser.parse_args([*where, "--wallet", "w", "register-key", "--cap-usd", "5"])
        with pytest.raises(SystemExit, match="--owner-key is required"):
            miner_tool._owner_key(rehearsal)
