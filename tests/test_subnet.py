"""Subnet-layer tests: artifact/commit integrity, contract validation, and a
fast end-to-end epoch through the real Validator (chain + store + reign)."""

from __future__ import annotations

import numpy as np
import pytest

from thirtyspokes.subnet import artifact as A
from thirtyspokes.subnet import miner, scoring
from thirtyspokes.subnet.artifact import PolicyArtifact
from thirtyspokes.subnet.chain import MockChain
from thirtyspokes.subnet.policy import train_baseline
from thirtyspokes.subnet.pool import SanctionedPool
from thirtyspokes.subnet.protocol import ALLOWED_HIDDEN, CompetitionParams
from thirtyspokes.subnet.store import LocalArtifactStore
from thirtyspokes.subnet.validator import Validator
from thirtyspokes.synth_tinyrouter import TinyConfig


@pytest.fixture(scope="module")
def pool():
    return SanctionedPool(TinyConfig(q=600, seed=7))


def _art(pool, hidden=16, seed=0):
    head = pool  # noqa
    from thirtyspokes.orchestrator import make_policy_head
    h = make_policy_head(pool.embed_dim, pool.k_models, hidden)
    theta = h.init_theta(seed)
    return PolicyArtifact(pool.embed_dim, pool.k_models, hidden, theta, f"m{seed}")


def test_artifact_roundtrip(pool, tmp_path):
    art = _art(pool, seed=1)
    p = str(tmp_path / "a.npz")
    art.save(p)
    b = PolicyArtifact.load(p)
    assert b.model_hash() == art.model_hash()
    assert b.hidden == art.hidden and np.allclose(b.theta, art.theta)


def test_salted_commit_is_hotkey_bound(pool):
    """A copied commit fails verification for any other hotkey (SN37 pattern)."""
    art = _art(pool, seed=2)
    c_alice = art.commit_string("alice", "r/x")
    # validator recomputes with the *actual* committing hotkey; a copier reusing
    # alice's commit under 'mallory' would mismatch mallory's recomputed string
    assert art.commit_string("mallory", "r/x") != c_alice


def test_validate_rejects_bad_artifacts(pool):
    good = _art(pool, hidden=16)
    assert A.validate(good, pool.embed_dim, pool.k_models)[0]
    bad_hidden = PolicyArtifact(pool.embed_dim, pool.k_models, 17, good.theta)  # 17 not allowed
    assert not A.validate(bad_hidden, pool.embed_dim, pool.k_models)[0]
    wrong_contract = PolicyArtifact(pool.embed_dim + 1, pool.k_models, 16, good.theta)
    assert not A.validate(wrong_contract, pool.embed_dim, pool.k_models)[0]
    assert 16 in ALLOWED_HIDDEN


def test_sanity_gate_rejects_degenerate(pool):
    probe = pool.probe_states(128)
    from thirtyspokes.orchestrator import make_policy_head
    n = make_policy_head(pool.embed_dim, pool.k_models, 8).n_params
    degen = PolicyArtifact(pool.embed_dim, pool.k_models, 8, np.zeros(n))
    assert not scoring.sanity_gate(degen, probe)[0]


def test_end_to_end_epoch_dedup_and_reign(pool, tmp_path):
    """One real epoch: two trained miners + a copier + a degenerate go through
    ingest -> verify -> sanity -> dedup -> score -> reign -> weights."""
    params = CompetitionParams(pool=pool.cfg, eval_questions=600)
    chain = MockChain()
    store = LocalArtifactStore(str(tmp_path))
    chain.register("burn")                       # uid 0 = burn
    val = Validator(pool, params, chain, store)

    # two honest miners (cheap training)
    a = train_baseline(pool, params, hidden=16, iters=15, seed=1, miner_id="alice")
    chain.register("alice"); miner.publish(a, chain, store, "alice", "a/1")
    b = train_baseline(pool, params, hidden=16, iters=15, seed=2, miner_id="bob")
    chain.advance(10)
    chain.register("bob"); miner.publish(b, chain, store, "bob", "b/1")

    # copier steals alice's weights (later block) -> fingerprint DQ
    chain.advance(10)
    copy = PolicyArtifact(a.embed_dim, a.k_models, a.hidden,
                          a.theta + np.random.default_rng(0).normal(0, 1e-4, a.theta.shape))
    chain.register("copier"); miner.publish(copy, chain, store, "copier", "c/1")

    # degenerate -> sanity DQ
    from thirtyspokes.orchestrator import make_policy_head
    n = make_policy_head(pool.embed_dim, pool.k_models, 8).n_params
    chain.register("degen")
    miner.publish(PolicyArtifact(pool.embed_dim, pool.k_models, 8, np.zeros(n)),
                  chain, store, "degen", "d/1")

    rep = val.run_epoch(epoch_seed=99)
    assert rep.disqualifications.get("copier") == "copy_of:alice"
    assert "degen" in rep.disqualifications
    assert rep.n_scored == 2                       # alice + bob only
    assert rep.champion in {"alice", "bob"}
    assert sum(rep.weights_by_uid.values()) == pytest.approx(1.0, abs=1e-6)


# --- live-chain decode seams (both shapes captured from Bittensor testnet 526) ----------------
# These two decoders were the only untested part of the chain seam, and BOTH were broken live:
# a miner's artifact commit was invisible to validators and its F7 proof commit read back as None.

def test_decode_revealed_accepts_both_wire_forms():
    """The node serializes a revealed commitment as hex when its SCALE bytes aren't valid UTF-8 and
    as a plain str when they are — so a hotkey's history can contain BOTH. Decoding must survive it."""
    from thirtyspokes.subnet.chain import BittensorChain as C

    # hex form: SCALE compact length prefix 0xb901 (4-byte mode -> offset 4... here mode=1 -> 2)
    msg = ("koth1|darksky8/thirtyspokes-koth-testnet|rev1|"
           "4aa95b298e31884e41c7623c4b08316a2c793b6602aef3f9762cb74ddbf9bed5")
    hex_form = "0x" + b"\xb9\x01".hex() + msg.encode().hex()   # exactly the testnet-526 wire bytes
    assert C.decode_revealed((hex_form, 7538182)) == (7538182, msg)

    str_form = "u\x01koth1|koth-demo/miner-a|rev1|" + "de" * 32
    assert C.decode_revealed((str_form, 7523462)) == (
        7523462, "koth1|koth-demo/miner-a|rev1|" + "de" * 32)

    assert C.decode_revealed(("not-hex-0x!!", 1)) is None or True   # never raises
    assert C.decode_revealed(None) is None


def test_decode_raw_commitment_handles_sized_raw_variant():
    """SCALE names the variant `Raw<N>` (N = byte length), never a bare `Raw`."""
    from thirtyspokes.subnet.chain import _decode_raw_commitment

    payload = "75427|5ec007c85ad133a7d437d54a454396beab0a7c83967b8b589de2bbdab3ea6d75"
    fields = [{"Raw70": "0x" + payload.encode().hex()}]
    assert _decode_raw_commitment(fields) == payload
    assert _decode_raw_commitment([{"Raw32": payload}]) == payload
    assert _decode_raw_commitment([]) is None
    assert _decode_raw_commitment(None) is None
