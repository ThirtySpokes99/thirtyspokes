"""Validator neuron — the epoch loop (the heart of the subnet).

Each epoch: sample fresh tasks over the fixed pool -> read on-chain commitments ->
download + verify (salted hash) + validate + sanity-gate + fingerprint-dedup each
policy -> execute it through the fixed engine and score (quality - lam*cost) ->
update the rank-weighted reign -> set weights on chain. Every validator does this
independently; no shared backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..reign import Reign, Submission
from . import artifact as artifact_mod
from . import scoring
from .protocol import DUP_TV_THRESHOLD, PROBE_SIZE, CompetitionParams


@dataclass
class EpochReport:
    epoch: int
    n_commits: int
    n_scored: int
    disqualifications: dict[str, str]
    slots: list[str]
    champion: str
    coronation: bool
    weights_by_uid: dict[int, float]
    scores: dict[str, float] = field(default_factory=dict)


def _parse_repo(commit_data: str) -> str | None:
    parts = commit_data.split("|")
    return parts[1] if len(parts) >= 3 else None


class Validator:
    def __init__(self, pool, params: CompetitionParams, chain, store):
        self.pool = pool
        self.params = params
        self.chain = chain
        self.store = store
        self.reign = Reign(split=params.reign_split, eps0=params.eps0,
                           eps_floor=params.eps_floor, tau=params.eps_tau)
        self.probe = pool.probe_states(PROBE_SIZE)
        self._epoch = 0

    def run_epoch(self, epoch_seed: int | None = None) -> EpochReport:
        self._epoch += 1
        seed = epoch_seed if epoch_seed is not None else (10_000 + self._epoch)
        backend = self.pool.sample_tasks(seed)
        commits = self.chain.revealed_commitments()
        dq: dict[str, str] = {}

        # --- ingest: download, verify salted hash, validate, sanity, fingerprint ---
        cands = []   # (hotkey, uid, block, artifact, fingerprint)
        for c in commits:
            repo = _parse_repo(c.data)
            if repo is None or not self.store.exists(repo):
                dq[c.hotkey] = "missing_artifact"; continue
            art = self.store.download(repo)
            if art.commit_string(c.hotkey, repo) != c.data:
                dq[c.hotkey] = "salted_hash_mismatch"; continue   # copy for another hotkey / tampered
            ok, reason = artifact_mod.validate(art, self.pool.embed_dim, self.pool.k_models)
            if not ok:
                dq[c.hotkey] = reason; continue
            ok, reason = scoring.sanity_gate(art, self.probe)
            if not ok:
                dq[c.hotkey] = reason; continue
            cands.append((c.hotkey, c.uid, c.block, art, scoring.fingerprint(art, self.probe)))

        # --- order-symmetric fingerprint dedup: earliest commit block wins ---
        cands.sort(key=lambda x: (x[2], x[0]))     # by (block, hotkey)
        accepted = []
        for hk, uid, block, art, fp in cands:
            dup_of = next((a[0] for a in accepted
                           if scoring.is_duplicate(fp, a[4], DUP_TV_THRESHOLD)), None)
            if dup_of is not None:
                dq[hk] = f"copy_of:{dup_of}"
            else:
                accepted.append((hk, uid, block, art, fp))

        # --- score + reign ---
        live = set(self.chain.hotkeys().values())
        subs, scores = [], {}
        for hk, uid, block, art, fp in accepted:
            s = scoring.score(art, backend, self.params)
            scores[hk] = s.objective
            subs.append(Submission(miner_id=hk, hotkey=hk, commit_block=block, score=s.objective))
        dereg = {hk for hk, *_ in accepted if hk not in live}
        res = self.reign.update(subs, deregistered=dereg) if subs else None

        # --- map reign weights (keyed by hotkey) -> on-chain weights (by uid) ---
        weights_by_uid: dict[int, float] = {}
        if res is not None:
            hk_to_uid = {hk: uid for uid, hk in self.chain.hotkeys().items()}
            for mid, w in res.weights.items():
                uid = 0 if mid == self.reign.burn_uid else hk_to_uid.get(mid)
                if uid is not None:
                    weights_by_uid[uid] = weights_by_uid.get(uid, 0.0) + w
            self.chain.set_weights(weights_by_uid)

        return EpochReport(
            epoch=self._epoch, n_commits=len(commits), n_scored=len(accepted),
            disqualifications=dq,
            slots=res.slots if res else [], champion=(res.slots[0] if res and res.slots else ""),
            coronation=res.coronation if res else False,
            weights_by_uid=weights_by_uid, scores=scores,
        )


def main() -> None:  # pragma: no cover — real-chain entrypoint
    import argparse
    import time

    from .chain import BittensorChain
    from .pool import SanctionedPool
    from .protocol import NETUID, CompetitionParams
    from .store import HFArtifactStore

    ap = argparse.ArgumentParser(prog="orchestra-validator")
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--epoch-blocks", type=int, default=360)
    args = ap.parse_args()
    params = CompetitionParams()
    v = Validator(SanctionedPool(params.pool), params,
                  BittensorChain(NETUID, args.wallet), HFArtifactStore())
    while True:
        rep = v.run_epoch(epoch_seed=v.chain.current_block())
        print(f"epoch {rep.epoch}: scored {rep.n_scored}/{rep.n_commits} "
              f"champion={rep.champion} coronation={rep.coronation}")
        time.sleep(args.epoch_blocks * 12)


if __name__ == "__main__":
    main()
