"""Policy artifact — what a miner actually submits (ARCHITECTURE.md §3 "miner-owns").

A miner's whole submission is: their trained policy weights + a declared
architecture (hidden size, from the allowed set) that loads into the fixed
orchestration engine. NOT the engine, the pool, or the eval.

Provides: save/load (npz), content hash, salted on-chain commit string
(hash(model_hash ‖ hotkey) — SN37's verified anti-copy pattern), and validation
against the protocol's I/O contract + size cap.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..orchestrator import make_policy_head
from ..router import RouterHead
from . import protocol


@dataclass
class PolicyArtifact:
    embed_dim: int          # frozen-encoder dim (fixed by the pool/contract)
    k_models: int           # pool size (fixed by the contract)
    hidden: int             # the miner's chosen capacity (customizable)
    theta: np.ndarray       # trained weights
    miner_id: str = ""

    # --- build the runnable policy head (the fixed I/O family) ---------------
    def head(self) -> RouterHead:
        return make_policy_head(self.embed_dim, self.k_models, self.hidden)

    # --- content hash + salted commit ---------------------------------------
    def model_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.embed_dim}|{self.k_models}|{self.hidden}|".encode())
        h.update(np.ascontiguousarray(self.theta, dtype=np.float64).tobytes())
        return h.hexdigest()

    def commit_string(self, hotkey: str, repo: str) -> str:
        """orch1|<repo>|<sha256(model_hash ‖ hotkey)> — copied commits fail for
        any other hotkey (salted)."""
        salted = hashlib.sha256(f"{self.model_hash()}{hotkey}".encode()).hexdigest()
        return f"{protocol.COMMIT_PREFIX}|{repo}|{salted}"

    # --- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, embed_dim=self.embed_dim, k_models=self.k_models,
                 hidden=self.hidden, theta=self.theta, miner_id=self.miner_id)

    @classmethod
    def load(cls, path: str) -> "PolicyArtifact":
        d = np.load(path, allow_pickle=True)
        return cls(int(d["embed_dim"]), int(d["k_models"]), int(d["hidden"]),
                   d["theta"], str(d["miner_id"]))


def validate(art: PolicyArtifact, embed_dim: int, k_models: int) -> tuple[bool, str]:
    """Check an artifact conforms to the protocol contract before eval."""
    if art.embed_dim != embed_dim or art.k_models != k_models:
        return False, "contract_mismatch"
    if art.hidden not in protocol.ALLOWED_HIDDEN:
        return False, f"hidden_not_allowed:{art.hidden}"
    head = art.head()
    if head.n_params > protocol.MAX_POLICY_PARAMS:
        return False, f"param_cap_exceeded:{head.n_params}"
    if art.theta.shape != (head.n_params,):
        return False, "theta_shape_mismatch"
    if not np.all(np.isfinite(art.theta)):
        return False, "non_finite_theta"
    return True, "ok"
