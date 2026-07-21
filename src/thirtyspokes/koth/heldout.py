"""Secret, owner-held, rotated memorization probe (docs/DESIGN.md §6 + the §9 review seam).

The memorization audit (`verify.memorization_collapsed`) is the ONLY backstop against a
miner weight-memorizing the static public benchmark: the validator runs the downloaded
artifact on a held-out slice and flags a significant accuracy drop vs. the claimed score.
It bites *only if the miner could not have seen the probe questions*.

In `benchmarks.py` the probe is `bench.probe(n, seed)` — drawn from the SAME public dataset
with a seed derived from the PUBLIC epoch nonce. So a miner can derive the exact probe slice
and memorize it too, and the audit is defeated. This module closes that: the probe comes from
an owner-curated **secret bank** the miner has never seen, **committed on-chain** (a hash, for
public auditability), distributed only to the (trusted, stake-gated) validators, and **rotated**.

Flow:
  * The owner assembles a `SecretProbeBank` {benchmark_name -> [BenchTask,...]} of private tasks
    (freshly written / licensed / generated — NOT in the public pool), computes `commit()`, and
    publishes that hash on-chain via governance (`owner.build_record(probe_commit=...)`), while
    distributing the bank file to validators out of band.
  * Each validator loads the bank and checks `bank.commit() == chain probe_commit` before scoring
    (`KOTHValidator.governance_ready`, fail-closed under enforce). It then draws the epoch probe
    slice from the bank, seeded by the unpredictable beacon nonce. Miners hold no bank, so they
    cannot know — let alone memorize — the probed questions.
  * Rotation: publish a new bank + commit (bump the governance `version`); the old slice is retired.

The bank is DATA the owner sources; this module is the mechanism (commit, verify, draw, rotate).
"""

from __future__ import annotations

import json

import numpy as np

from ..gateway import signing
from .benchmarks import BenchTask


def _task_dict(t: BenchTask) -> dict:
    return {"task_id": t.task_id, "prompt": t.prompt, "gold": t.gold}


class SecretProbeBank:
    """A private held-out task bank, keyed by benchmark name. Its `commit()` is the only
    thing that goes on-chain; the tasks themselves stay with validators."""

    def __init__(self, tasks_by_bench: dict[str, list[BenchTask]]):
        # store sorted by task_id so the commit is order-independent
        self._by_bench = {name: sorted(tasks, key=lambda t: t.task_id)
                          for name, tasks in tasks_by_bench.items() if tasks}

    def has(self, name: str) -> bool:
        return name in self._by_bench

    def benchmarks(self) -> list[str]:
        return sorted(self._by_bench)

    def size(self, name: str) -> int:
        return len(self._by_bench.get(name, ()))

    def draw(self, name: str, n: int, seed: int) -> list[BenchTask]:
        """Deterministic held-out slice for a benchmark, seeded by the beacon nonce."""
        tasks = self._by_bench.get(name, [])
        if not tasks:
            return []
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tasks), size=min(n, len(tasks)), replace=False)
        return [tasks[int(i)] for i in idx]

    def commit(self) -> str:
        """Canonical, order-independent hash of the whole bank — the on-chain commitment.
        Any change to any task/gold changes it (so rotation is visible + auditable)."""
        payload = {name: [_task_dict(t) for t in tasks]      # tasks already sorted by id
                   for name, tasks in sorted(self._by_bench.items())}
        return signing.sha256_hex({"koth-probe-bank-v1": payload})

    # --- serialization for owner->validator distribution ---------------------
    def to_json(self) -> str:
        return json.dumps({name: [_task_dict(t) for t in tasks]
                           for name, tasks in sorted(self._by_bench.items())})

    @classmethod
    def from_json(cls, s: str) -> "SecretProbeBank":
        d = json.loads(s)
        return cls({name: [BenchTask(t["task_id"], t["prompt"], t["gold"]) for t in tasks]
                    for name, tasks in d.items()})

    @classmethod
    def from_file(cls, path: str) -> "SecretProbeBank":
        with open(path) as f:
            return cls.from_json(f.read())
