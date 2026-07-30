"""Chain integration — commit-reveal, metagraph reads, weight setting.

Two implementations behind one interface:
  * MockChain      — in-memory; the offline simulator + tests use this.
  * BittensorChain — real substrate via the `bittensor` SDK (lazy-imported).

The interface is deliberately tiny: what a miner needs to commit and what a
validator needs to read commitments + set weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Commitment:
    hotkey: str
    uid: int
    data: str          # the orch1|<repo>|<salted-hash> reveal string
    block: int


def _mock_beacon(epoch: int) -> str:
    """Deterministic mock beacon (PREDICTABLE — real unpredictability comes from the
    live block hash in BittensorChain.beacon)."""
    import hashlib
    return hashlib.sha256(f"koth-beacon|{epoch}".encode()).hexdigest()[:16]


GOV_PREFIX = "kothgov1|"       # owner-published approved-measurement record (H2/H4 governance)


class WeightRateLimited(RuntimeError):
    def __init__(self, retry_blocks: int):
        self.retry_blocks = retry_blocks
        super().__init__(f"weight update is rate-limited for {retry_blocks} more block(s)")


def _decode_raw_commitment(fields) -> str | None:
    """Pull the digest string out of a substrate CommitmentInfo `fields` list (one Raw(bytes) entry).

    The SCALE enum variant is `Raw<N>` where N is the payload's BYTE LENGTH (`Raw70`, `Raw32`, …) —
    never a bare `Raw`. Matching the literal key "Raw" therefore always missed, so every F7 proof
    commit read back as None: with `--commit-window` on, that DQs every miner as
    `commit_out_of_window`. Observed live on testnet 526."""
    for entry in fields or []:
        if hasattr(entry, "get"):
            raw = next((v for k, v in entry.items() if str(k).startswith("Raw")), None)
        else:
            raw = entry
        if isinstance(raw, str):
            return bytes.fromhex(raw[2:]).decode() if raw.startswith("0x") else raw
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode()
    return None


class Chain(Protocol):
    def current_block(self) -> int: ...
    def commit(self, hotkey: str, data: str) -> None: ...
    def revealed_commitments(self) -> list[Commitment]: ...
    def hotkeys(self) -> dict[int, str]: ...           # uid -> hotkey
    def set_weights(self, weights: dict[int, float]) -> None: ...
    def beacon(self, epoch: int) -> str: ...           # per-epoch seed for the task nonce
    # owner-governed approved measurements (MRTD/RTMR/runtime/TCB); None if unpublished
    def owner_measurements(self) -> dict | None: ...
    # subnet economics for the public feed (alpha price + emission); None offline
    def market(self) -> dict | None: ...
    # F7 anti-grind + F2 presence: a per-(hotkey, epoch), immediately-visible proof binding (the
    # proof's report_data + its inclusion block). Keyed by epoch so a validator scoring a SETTLED
    # (past-grace) epoch can still read that epoch's commit. Only used when the validator sets
    # commit_window.
    def commit_proof(self, hotkey: str, epoch: int, digest: str) -> None: ...
    def proof_commit(self, hotkey: str, epoch: int) -> tuple[str, int] | None: ...   # (digest, block)|None


def _require_accepted(res, what: str) -> None:
    """Raise unless the chain accepted the extrinsic.

    bittensor's write helpers return `(ok, message)` (and have changed shape before), so a discarded
    result makes a REJECTED write indistinguishable from a successful one. That is not hypothetical:
    an artifact commit reported success, never appeared after its full 360-block reveal window, and
    the failure was only visible by re-submitting and printing the return value. Silent-success on a
    chain write means an operator debugs the wrong layer for hours.
    """
    ok, msg = res if isinstance(res, tuple) else (bool(res), "")
    if not ok:
        raise RuntimeError(f"{what} REJECTED by the chain: {msg or res!r}")


@dataclass
class MockChain:
    """In-memory chain for offline runs. Registration = assigning a uid+hotkey."""

    _block: int = 0
    _uid_of: dict[str, int] = field(default_factory=dict)
    _commit: dict[str, Commitment] = field(default_factory=dict)   # hotkey -> latest
    _proof_commit: dict[tuple, tuple] = field(default_factory=dict)  # (hotkey, epoch) -> (digest, block)
    _weights: dict[int, float] = field(default_factory=dict)
    _next_uid: int = 0
    _owner_meas: dict | None = None

    def register(self, hotkey: str) -> int:
        if hotkey not in self._uid_of:
            self._uid_of[hotkey] = self._next_uid
            self._next_uid += 1
        return self._uid_of[hotkey]

    def deregister(self, hotkey: str) -> None:
        self._uid_of.pop(hotkey, None)
        self._commit.pop(hotkey, None)
        for k in [k for k in self._proof_commit if k[0] == hotkey]:
            del self._proof_commit[k]

    def advance(self, n: int = 1) -> None:
        self._block += n

    # --- Chain interface ---
    def current_block(self) -> int:
        return self._block

    def commit(self, hotkey: str, data: str) -> None:
        uid = self._uid_of.get(hotkey)
        if uid is None:
            raise ValueError(f"hotkey {hotkey} not registered")
        self._commit[hotkey] = Commitment(hotkey, uid, data, self._block)

    def revealed_commitments(self) -> list[Commitment]:
        return [c for c in self._commit.values() if c.hotkey in self._uid_of]

    def commit_proof(self, hotkey: str, epoch: int, digest: str) -> None:
        if hotkey not in self._uid_of:
            raise ValueError(f"hotkey {hotkey} not registered")
        self._proof_commit[(hotkey, epoch)] = (digest, self._block)

    def proof_commit(self, hotkey: str, epoch: int) -> tuple | None:
        return self._proof_commit.get((hotkey, epoch))

    def hotkeys(self) -> dict[int, str]:
        return {uid: hk for hk, uid in self._uid_of.items()}

    def set_weights(self, weights: dict[int, float]) -> None:
        self._weights = dict(weights)

    def beacon(self, epoch: int) -> str:
        return _mock_beacon(epoch)

    def publish_owner_measurements(self, record: dict) -> None:
        self._owner_meas = dict(record)

    def owner_measurements(self) -> dict | None:
        return self._owner_meas

    def market(self) -> dict | None:
        return None                     # no economics offline


class LocalFileChain:
    """File-backed chain so independent miner + validator PROCESSES can interact
    (Stage-0 stand-in for BittensorChain). State is a JSON file re-read per call."""

    def __init__(self, root: str):
        import json
        import os
        self._json, self._os = json, os
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "chain.json")
        if not os.path.exists(self.path):
            self._save({"block": 0, "uids": {}, "next_uid": 0, "commits": {}, "weights": {}})

    def _load(self) -> dict:
        with open(self.path) as f:
            return self._json.load(f)

    def _save(self, s: dict) -> None:
        with open(self.path, "w") as f:
            self._json.dump(s, f)

    def register(self, hotkey: str) -> int:
        s = self._load()
        if hotkey not in s["uids"]:
            s["uids"][hotkey] = s["next_uid"]; s["next_uid"] += 1
            self._save(s)
        return s["uids"][hotkey]

    def advance(self, n: int = 1) -> None:
        s = self._load(); s["block"] += n; self._save(s)

    def current_block(self) -> int:
        return self._load()["block"]

    def commit(self, hotkey: str, data: str) -> None:
        s = self._load()
        if hotkey not in s["uids"]:
            raise ValueError(f"hotkey {hotkey} not registered")
        s["commits"][hotkey] = {"uid": s["uids"][hotkey], "data": data, "block": s["block"]}
        self._save(s)

    def revealed_commitments(self) -> list[Commitment]:
        s = self._load()
        return [Commitment(hk, c["uid"], c["data"], c["block"])
                for hk, c in s["commits"].items() if hk in s["uids"]]

    def commit_proof(self, hotkey: str, epoch: int, digest: str) -> None:
        s = self._load()
        if hotkey not in s["uids"]:
            raise ValueError(f"hotkey {hotkey} not registered")
        s.setdefault("proof_commits", {}).setdefault(hotkey, {})[str(epoch)] = \
            {"digest": digest, "block": s["block"]}
        self._save(s)

    def proof_commit(self, hotkey: str, epoch: int) -> tuple | None:
        c = self._load().get("proof_commits", {}).get(hotkey, {}).get(str(epoch))
        return (c["digest"], c["block"]) if c else None

    def hotkeys(self) -> dict[int, str]:
        return {uid: hk for hk, uid in self._load()["uids"].items()}

    def set_weights(self, weights: dict[int, float]) -> None:
        s = self._load(); s["weights"] = {str(u): w for u, w in weights.items()}; self._save(s)

    def beacon(self, epoch: int) -> str:
        return _mock_beacon(epoch)

    def publish_owner_measurements(self, record: dict) -> None:
        s = self._load(); s["owner_meas"] = dict(record); self._save(s)

    def owner_measurements(self) -> dict | None:
        return self._load().get("owner_meas")

    def market(self) -> dict | None:
        return None                     # no economics offline


class BittensorChain:  # pragma: no cover — needs a live chain + wallet
    """Real substrate integration (verified against the bittensor 10.x SDK — method names +
    return shapes introspected, not guessed). Commit-reveal via `set_reveal_commitment`;
    metagraph for uid->hotkey; `set_weights` for emissions; `burned_register` to register.

    NOTE the reveal window: `set_reveal_commitment(blocks_until_reveal=…)` is the anti-front-run
    delay (default ~360 blocks). Registration is normally done once via `btcli`; `register()`
    here is a convenience that burns to register the hotkey."""

    def __init__(self, netuid: int, wallet_name: str, network: str = "finney",
                 *, hotkey: str = "default", owner_ss58: str | None = None,
                 epoch_blocks: int = 100, reveal_blocks: int = 360):
        try:
            import bittensor as bt  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("pip install bittensor to use the live chain") from e
        self.bt = bt
        self.netuid = netuid
        self.network = network
        self.epoch_blocks = epoch_blocks       # beacon block = epoch * epoch_blocks (matches epoch.py)
        self.reveal_blocks = reveal_blocks
        subtensor_cls = getattr(bt, "Subtensor", None) or bt.subtensor   # 10.x / 8.x
        wallet_cls = getattr(bt, "Wallet", None) or bt.wallet
        self.subtensor = subtensor_cls(network=network)
        self.wallet = wallet_cls(name=wallet_name, hotkey=hotkey)
        self.owner_ss58 = owner_ss58           # subnet owner hotkey publishing the approved set
        self._configure_rpc()

    def _configure_rpc(self) -> None:
        # The SDK defaults to five 60-second waits for every public RPC. A live testnet endpoint
        # stopped responding mid-epoch, leaving both validators apparently alive for five minutes
        # on one read. Keep every chain operation bounded; the production epoch loop rolls back its
        # in-memory snapshot and retries a failed pre-commit epoch safely.
        substrate = self.subtensor.substrate
        if hasattr(substrate, "retry_timeout"):
            substrate.retry_timeout = min(float(substrate.retry_timeout), 20.0)
        if hasattr(substrate, "max_retries"):
            substrate.max_retries = min(int(substrate.max_retries), 2)

    def reconnect(self) -> None:
        """Replace a timed-out SDK connection instead of reusing its poisoned WebSocket."""
        self.close(raise_error=False)
        subtensor_cls = getattr(self.bt, "Subtensor", None) or self.bt.subtensor
        self.subtensor = subtensor_cls(network=self.network)
        self._configure_rpc()

    def close(self, *, raise_error: bool = True) -> None:
        """Release the SDK WebSocket worker so a validator process can actually terminate."""
        try:
            self.subtensor.close()
        except Exception:  # noqa: BLE001 — reconnect must still replace a broken connection
            if raise_error:
                raise

    def hotkey_ss58(self) -> str:
        return self.wallet.hotkey.ss58_address

    def register(self, hotkey: str) -> None:
        """Burn-register this wallet's hotkey on the subnet (idempotent — skips if already on
        the metagraph). Registration is usually done once out-of-band via `btcli`."""
        if self.wallet.hotkey.ss58_address in set(self.hotkeys().values()):
            return
        _require_accepted(
            self.subtensor.burned_register(wallet=self.wallet, netuid=self.netuid,
                                           wait_for_inclusion=True, wait_for_finalization=True),
            "burned registration")

    def current_block(self) -> int:
        return int(self.subtensor.get_current_block())

    def commit(self, hotkey: str, data: str) -> None:
        """Publish the artifact commitment, and RAISE if the chain did not accept it.

        This used to discard the return value. `set_reveal_commitment` returns `(ok, message)`, so a
        rejected extrinsic looked exactly like a successful one — measured: a commit reported success,
        never appeared after its full 360-block reveal window, and the failure was only visible by
        re-submitting and printing the result. A miner that believes it committed will mine for hours
        and be scored `no_proof` every epoch with nothing anywhere saying why.
        """
        res = self.subtensor.set_reveal_commitment(
            wallet=self.wallet, netuid=self.netuid, data=data,
            blocks_until_reveal=self.reveal_blocks)
        # bittensor returns (bool, str) here, but has changed this shape before; treat a bare False
        # or a (False, msg) as failure and anything else as success rather than guessing.
        ok, msg = res if isinstance(res, tuple) else (bool(res), "")
        if not ok:
            raise RuntimeError(f"artifact commit REJECTED by the chain: {msg or res!r}. "
                               f"Nothing is bound on-chain, so every proof this miner uploads will "
                               f"be unscorable.")

    @staticmethod
    def _latest(entries) -> tuple[int, str] | None:
        """`get_all_revealed_commitments` returns {hotkey: ((block, data), …)} — pick latest."""
        if not entries:
            return None
        block, data = max(entries, key=lambda bd: bd[0])
        return int(block), data

    @staticmethod
    def decode_revealed(pair) -> tuple[int, str] | None:
        """Decode ONE `RevealedCommitments` entry `(message, block)`; None if undecodable.

        The node serializes the SCALE-encoded message as a plain `str` when those bytes happen to
        be valid UTF-8 and as a `0x…` hex string when they don't — which turns on the compact
        length-prefix byte, i.e. on the payload LENGTH. The SDK's `get_revealed_commitment_by_hotkey`
        assumes hex always (`bytes.fromhex`) AND decodes a hotkey's entries as one batch, so a single
        `str`-form entry raises and hides EVERY commitment that hotkey ever made — observed live on
        testnet 526, where it silently masked a valid artifact commit. That is a permanent DQ, so we
        decode both forms, one entry at a time."""
        try:
            msg, block = pair
            if isinstance(msg, str):
                raw = bytes.fromhex(msg[2:]) if msg.startswith("0x") else msg.encode("utf-8")
            else:
                raw = bytes(msg)
            offset = {0: 1, 1: 2}.get(raw[0] & 0b11, 4)      # SCALE compact length prefix
            return int(block), raw[offset:].decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001 — one malformed entry must not blind us to the rest
            return None

    def _revealed_for(self, hotkey_ss58: str) -> tuple[int, str] | None:
        """Latest revealed commitment for ONE hotkey, decoding each entry independently (see
        `decode_revealed`). Nothing readable -> treated as no commitment."""
        try:
            q = self.subtensor.query_module(module="Commitments", name="RevealedCommitments",
                                            params=[self.netuid, hotkey_ss58])
            entries = q.value_serialized
        except Exception:  # noqa: BLE001 — no commitment / RPC hiccup is "nothing to read"
            return None
        decoded = [d for d in (self.decode_revealed(p) for p in entries or []) if d]
        return self._latest(decoded)

    def revealed_commitments(self) -> list[Commitment]:
        out = []
        for uid, hk in self.hotkeys().items():
            latest = self._revealed_for(hk)
            if latest and not latest[1].startswith(GOV_PREFIX):
                out.append(Commitment(hk, uid, latest[1], latest[0]))
        return out

    def hotkeys(self) -> dict[int, str]:
        # We need exactly one field: uid -> hotkey. `subtensor.metagraph()` also fetches stakes,
        # neuron metadata, and MetagraphInfo; the sole public testnet endpoint repeatedly timed out
        # on that unrelated runtime API and blocked every validator. `Keys` is the canonical direct
        # storage map and returned the same five live uid/hotkey pairs immediately on netuid 526.
        rows = self.subtensor.substrate.query_map(
            module="SubtensorModule", storage_function="Keys", params=[self.netuid])
        return {int(uid): str(getattr(hotkey, "value", hotkey)) for uid, hotkey in rows}

    def set_weights(self, weights: dict[int, float]) -> None:
        """Set weights with a bounded inclusion wait and idempotent recovery.

        Finalization is not needed here: once the extrinsic is included, a rare reorg can be
        repaired by the next epoch's identical update.  Waiting for finalization on a public RPC
        stalled a live validator indefinitely.  The SDK's default RPC budget is also five one-minute
        waits, so temporarily bound this write to two 20-second waits and fail loudly.  The validator
        persists pending weights before calling this method and retries after restart.
        """
        uids = list(weights.keys())
        total = sum(weights.values()) or 1.0
        vals = [weights[u] / total for u in uids]        # normalize; SDK converts to u16 on-chain
        if self._weights_are_current(weights):
            return

        uid = self.subtensor.get_uid_for_hotkey_on_subnet(
            self.wallet.hotkey.ss58_address, self.netuid)
        since = self.subtensor.blocks_since_last_update(self.netuid, uid) if uid is not None else None
        limit = self.subtensor.weights_rate_limit(self.netuid)
        if since is not None and limit is not None and since <= limit:
            raise WeightRateLimited(int(limit - since + 1))

        substrate = self.subtensor.substrate
        old_timeout = getattr(substrate, "retry_timeout", None)
        old_retries = getattr(substrate, "max_retries", None)
        try:
            if old_timeout is not None:
                substrate.retry_timeout = min(float(old_timeout), 20.0)
            if old_retries is not None:
                substrate.max_retries = min(int(old_retries), 2)
            response = self.subtensor.set_weights(
                wallet=self.wallet, netuid=self.netuid, uids=uids, weights=vals,
                max_attempts=2, raise_error=False,
                wait_for_inclusion=True, wait_for_finalization=False,
                wait_for_revealed_execution=False,
            )
        finally:
            if old_timeout is not None:
                substrate.retry_timeout = old_timeout
            if old_retries is not None:
                substrate.max_retries = old_retries

        if not getattr(response, "success", False) and not self._weights_are_current(weights):
            message = getattr(response, "message", None) or "unknown chain error"
            raise RuntimeError(f"set_weights was not included: {message}")

    @staticmethod
    def _same_weight_distribution(requested: dict[int, float], current) -> bool:
        """Compare normalized distributions, ignoring zero entries and u16 quantization."""
        want = {int(uid): float(weight) for uid, weight in requested.items() if weight > 0}
        got = {int(uid): float(weight) for uid, weight in (current or []) if weight > 0}
        want_total, got_total = sum(want.values()), sum(got.values())
        if not want_total or not got_total or set(want) != set(got):
            return False
        return all(abs(want[uid] / want_total - got[uid] / got_total) <= 2 / 65535
                   for uid in want)

    def _weights_are_current(self, requested: dict[int, float]) -> bool:
        """True when this validator already has the requested distribution on-chain."""
        try:
            uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                self.wallet.hotkey.ss58_address, self.netuid)
            if uid is None:
                return False
            rows = self.subtensor.weights(self.netuid)
            current = next((row for source_uid, row in rows if int(source_uid) == int(uid)), None)
            return self._same_weight_distribution(requested, current)
        except Exception:  # noqa: BLE001 — preflight failure must not suppress a real submission
            return False

    def beacon(self, epoch: int) -> str:
        # the real, unpredictable-before-the-epoch seed: the block hash at epoch start
        return str(self.subtensor.get_block_hash(epoch * self.epoch_blocks))[:16]

    def publish_owner_measurements(self, record: dict) -> None:
        """Publish the approved-measurement record: bytes to the bucket, HASH to the chain.

        This used to shove the whole record into a TIMELOCKED reveal-commitment. That was never a
        design choice — it was the only thing that fit: the record is ~657 bytes and a plain
        commitment's Raw field caps at 128 ("Value 'Raw657' not present in type_mapping"). The
        timelock bought nothing (the payload is encrypted until reveal, so miners get NO advance
        warning of a change) and cost a lot: an emergency TCB tightening could not take effect for
        ~72 minutes.

        Now the chain carries the sha256 (73 bytes -> a plain, IMMEDIATELY-VISIBLE commitment) and the
        record itself is served from the owner's public bucket, named by its own hash. Validators fetch
        it and verify it against the chain, so the bucket needs no trust — a tampered record fails the
        hash. Same asymmetry as the runtime image: the chain is the trust root, the bucket is transport.
        """
        from ..koth import governance
        d = governance.publish(record)
        self.commit_governance(d)

    def commit_governance(self, record_digest: str) -> None:
        from ..koth import governance
        # CHECKED: a silently-rejected governance write leaves validators on the PREVIOUS approved
        # measurement, so every miner running the new image is rejected as `unapproved_runtime` and
        # the fault looks like it belongs to the miners.
        _require_accepted(
            self.subtensor.set_commitment(wallet=self.wallet, netuid=self.netuid,
                                          data=governance.commit_string(record_digest)),
            "governance commitment")

    def governance_digest(self) -> str | None:
        """The record hash the OWNER committed (plain commitment -> readable immediately)."""
        from ..koth import governance
        owner = self.owner_hotkey()
        if not owner:
            return None
        try:
            rec = self.subtensor.substrate.query("Commitments", "CommitmentOf",
                                                 [self.netuid, owner])
            v = getattr(rec, "value", None)
            raw = _decode_raw_commitment(v["info"]["fields"]) if v else None
        except Exception:  # noqa: BLE001 — unreadable = no governance = fail closed upstream
            return None
        return governance.parse_commit(raw or "")

    def owner_hotkey(self) -> str | None:
        """Who the subnet OWNER is, read from the chain (`SubnetOwnerHotkey`).

        The validator daemon never passed an `owner_ss58`, so `owner_measurements()` returned None
        unconditionally -> `governance_ready()` always failed -> the daemon could NEVER start in
        enforce mode, whatever the owner published. Resolving the owner from the chain also removes a
        trust hole: a validator that has to be *told* who the owner is can be pointed at an attacker's
        hotkey and made to accept the attacker's measured image. The chain is the authority."""
        if self.owner_ss58:
            return self.owner_ss58                     # explicit override (tests / a forked subnet)
        if getattr(self, "_owner_cache", None):
            return self._owner_cache
        try:
            r = self.subtensor.substrate.query("SubtensorModule", "SubnetOwnerHotkey", [self.netuid])
            self._owner_cache = str(getattr(r, "value", r)) or None
        except Exception:  # noqa: BLE001 — unreadable owner = no governance = fail closed upstream
            self._owner_cache = None
        return self._owner_cache

    def owner_measurements(self) -> dict | None:
        """The approved-measurement record: read its HASH from the chain, fetch the bytes, verify.

        The fetch is content-addressed — the URL is derived from the on-chain hash — so a tampered
        record cannot even be addressed, let alone accepted. Any failure returns None, which the
        validator treats as "no governance" and fails closed (it refuses to score).
        """
        from ..koth import governance
        d = self.governance_digest()
        if not d:
            return None
        if getattr(self, "_gov_cache", (None, None))[0] == d:
            return self._gov_cache[1]           # same record as last epoch — don't refetch
        try:
            rec = governance.fetch(d)
        except Exception:  # noqa: BLE001 — unreachable/tampered record = no governance
            return None
        self._gov_cache = (d, rec)
        return rec

    # --- subnet economics for the public feed ----------------------------------------------------
    # Read straight off the chain. The sibling subnet pulls these from a paid third-party API
    # (TaoMarketCap, needs an API key); everything except the TAO/USD rate is on-chain, and a
    # validator should not need an API key to render a leaderboard. TAO/USD genuinely is not
    # on-chain, so it stays optional: set TAO_USD to price the columns in dollars, else the feed
    # publishes alpha only and the dashboard omits the $ columns rather than inventing a rate.
    def market(self) -> dict | None:
        import os
        try:
            d = self.subtensor.subnet(self.netuid)
            alpha_price_tao = float(getattr(d.price, "tao", d.price))
            gross_alpha_per_block = float(getattr(d.alpha_out_emission, "tao", d.alpha_out_emission))
            # SubnetOwnerCut is a network-wide u16 fraction (18% today). The remainder splits 50/50
            # between miners and validators, so miners get (1 - owner_cut) / 2. Derived from chain,
            # not hardcoded, so it stays right if governance changes the cut.
            cut_raw = self.subtensor.substrate.query("SubtensorModule", "SubnetOwnerCut")
            owner_cut = float(getattr(cut_raw, "value", cut_raw)) / 65535.0
            miner_share = (1.0 - owner_cut) / 2.0
            tao_usd = os.environ.get("TAO_USD")
            return {
                "alpha_price_tao": alpha_price_tao,
                "tao_usd": float(tao_usd) if tao_usd else None,
                "alpha_price_usd": (alpha_price_tao * float(tao_usd)) if tao_usd else None,
                "owner_cut": round(owner_cut, 6),
                "miner_share": round(miner_share, 6),
                "alpha_per_block_gross": gross_alpha_per_block,
                "alpha_per_block_miners": gross_alpha_per_block * miner_share,
                "blocks_per_hour": 300,          # 12s blocks
            }
        except Exception:  # noqa: BLE001 — presentational; never break scoring over a price read
            return None

    # --- F7 anti-grind + F2 presence: per-epoch proof binding -------------------------------------
    # The proof commit must be IMMEDIATELY visible (the validator reads it while scoring) and
    # block-timestamped — so it uses the PLAIN commitment map (`set_commitment`), NOT the delayed
    # reveal-commitment the artifact + governance use. That map is single-slot per hotkey, so we tag
    # the data with the epoch (`<epoch>|<digest>`) and the reader checks it. CONSTRAINT: this only
    # retains the LATEST epoch's commit, so the validator must score INTRA-epoch (grace_blocks <
    # epoch_blocks) — while the scored epoch is still the miner's latest commit. SEAM: verify on
    # testnet that the plain-commitment map is independent of the reveal map (so a proof commit never
    # clobbers the artifact commit) and that the accessor returns the inclusion block.
    def commit_proof(self, hotkey: str, epoch: int, digest: str) -> None:
        # CHECKED: a silently-rejected proof commit DQs the miner as `no_proof_commit` every epoch
        # under --commit-window, with nothing anywhere explaining why its work is not counting.
        _require_accepted(
            self.subtensor.set_commitment(wallet=self.wallet, netuid=self.netuid,
                                          data=f"{epoch}|{digest}"),
            "proof commitment")

    def proof_commit(self, hotkey: str, epoch: int) -> tuple[str, int] | None:
        # CommitmentOf[(netuid, hotkey)] -> {info: {fields: [Raw(bytes)]}, block}. Decode `<epoch>|<digest>`;
        # return (digest, block) only if the tag matches the epoch being scored. Field decode is
        # SDK/runtime-specific — a seam to pin on testnet; any read error means "nothing committed".
        try:
            rec = self.subtensor.substrate.query("Commitments", "CommitmentOf", [self.netuid, hotkey])
            v = getattr(rec, "value", None)
            raw = _decode_raw_commitment(v["info"]["fields"]) if v else None
            tag, _, digest = (raw or "").partition("|")
            return (digest, int(v["block"])) if raw and tag == str(epoch) else None
        except Exception:  # noqa: BLE001
            return None
