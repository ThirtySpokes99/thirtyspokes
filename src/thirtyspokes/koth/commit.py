"""On-chain commit string for the KOTH-TEE design (docs/DESIGN.md §3).

A miner commits, to its OWN chain hotkey, a pointer to its OWN public HF repo
revision plus a hotkey-salted binding hash. Two properties:
  * BINDING: the hash covers `source_hash`+`weights_hash`, so the commitment names
    exactly the public artifact the proof is attested against.
  * ANTI-COPY: it is salted with the hotkey (SN37's verified pattern), so a copied
    commitment fails verification under any other hotkey. Front-running is handled
    by the chain's commit-reveal delay (`set_reveal_commitment`, blocks_until_reveal).
"""

from __future__ import annotations

from ..gateway import signing

COMMIT_PREFIX = "koth1"


def commit_string(hotkey: str, repo: str, revision: str,
                  source_hash: str, weights_hash: str) -> str:
    """koth1|<repo>|<revision>|<sha256(source_hash‖weights_hash‖hotkey)>."""
    salted = signing.sha256_hex(f"{source_hash}|{weights_hash}|{hotkey}")
    return f"{COMMIT_PREFIX}|{repo}|{revision}|{salted}"


def parse_commit(data: str) -> tuple[str, str, str] | None:
    """-> (repo, revision, salted_hash) or None if not a KOTH commit."""
    parts = data.split("|")
    if len(parts) != 4 or parts[0] != COMMIT_PREFIX:
        return None
    return parts[1], parts[2], parts[3]


def verify_commit(data: str, hotkey: str, source_hash: str, weights_hash: str) -> bool:
    """Recompute the salted hash from the revealed artifact hashes + hotkey."""
    parsed = parse_commit(data)
    if parsed is None:
        return False
    _repo, _rev, salted = parsed
    return salted == signing.sha256_hex(f"{source_hash}|{weights_hash}|{hotkey}")
