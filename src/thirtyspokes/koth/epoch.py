"""Shared epoch + nonce derivation (docs/DESIGN.md §4).

In the live subnet the miner and validator are decoupled processes that never talk
directly, so both must derive the SAME epoch and task-sampling nonce from on-chain
state alone. Epoch = block height / EPOCH_BLOCKS; the nonce seeds the benchmark
slice (`benchmarks.bench_seed`).

Offline runs derive the nonce from the epoch number (deterministic, both sides agree).
Production seeds from the block *hash* at epoch start so the slice
is unpredictable before the epoch — `Chain.beacon(epoch)` is the seam for that.
"""

from __future__ import annotations

from ..gateway import signing

EPOCH_BLOCKS = 100


def current_epoch(chain, epoch_blocks: int = EPOCH_BLOCKS) -> int:
    return chain.current_block() // epoch_blocks


def epoch_nonce(epoch: int, beacon: str = "") -> str:
    """Shared per-epoch nonce. `beacon` (a block hash) makes it unpredictable when
    available; empty falls back to the epoch number (offline)."""
    return signing.sha256_hex(f"koth-epoch|{epoch}|{beacon}")[:16]
