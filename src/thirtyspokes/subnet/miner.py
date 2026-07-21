"""Miner neuron — train a policy, upload it, commit the salted hash on-chain.

The full miner loop (ARCHITECTURE.md §3 "miner-owns"): train offline → upload
weights (private) → commit hash(model_hash ‖ hotkey) on-chain → (flip public).
Miners may swap `train_baseline` for any custom policy/trainer; only the artifact
that loads into the fixed engine is submitted.
"""

from __future__ import annotations

from .artifact import PolicyArtifact
from .policy import train_baseline
from .protocol import CompetitionParams


def publish(art: PolicyArtifact, chain, store, hotkey: str, repo: str) -> str:
    """Upload the artifact and commit its salted hash. Returns the commit string."""
    store.upload(repo, art)                       # (private in production)
    commit = art.commit_string(hotkey, repo)
    chain.commit(hotkey, commit)                  # salted -> copy-proof for other hotkeys
    return commit


def train_and_publish(pool, params: CompetitionParams, chain, store, hotkey: str,
                      repo: str, hidden: int = 32, iters: int = 140, seed: int = 0,
                      verbose: bool = False) -> PolicyArtifact:
    art = train_baseline(pool, params, hidden=hidden, iters=iters, seed=seed,
                         miner_id=hotkey, verbose=verbose)
    publish(art, chain, store, hotkey, repo)
    return art


def main() -> None:  # pragma: no cover — real-chain entrypoint
    import argparse

    from .chain import BittensorChain
    from .pool import SanctionedPool
    from .protocol import NETUID, CompetitionParams
    from .store import HFArtifactStore

    ap = argparse.ArgumentParser(prog="orchestra-miner")
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--hotkey", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    params = CompetitionParams()
    pool = SanctionedPool(params.pool)
    chain = BittensorChain(NETUID, args.wallet)
    train_and_publish(pool, params, chain, HFArtifactStore(), args.hotkey, args.repo,
                      hidden=args.hidden, iters=args.iters, verbose=True)


if __name__ == "__main__":
    main()
