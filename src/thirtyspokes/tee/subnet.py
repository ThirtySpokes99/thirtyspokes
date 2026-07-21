"""Option-A subnet — TEE-attested competition, end-to-end and runnable offline.

Validator issues a fresh per-task nonce, the miner runs its agent in a TEE runtime
(here: the mock runtime + mock platform), and the validator verifies the
attestation + grades + reads metered cost -> reign -> weights. Demonstrates the
adversaries that BEAT the gateway now fail: there is no off-runtime path, a
lying/modified runtime is rejected by measurement, a self-signed quote by the
hardware root, and a replayed attestation by the issued nonce.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..gateway.gateway import MockBackend
from ..gateway.grader import grade_math
from ..gateway.signing import Signer
from ..gateway.grader import generate_math_tasks
from ..reign import Reign, Submission
from ..subnet.chain import MockChain
from .attestation import AttestationReport, Platform
from .runtime import TEERuntime, runtime_measurement
from .verify import verify_attestation


def calling_agent(model: str, n_calls: int, answer: str):
    """Agent that makes `n_calls` metered calls, then returns `answer`.
    Its ONLY channel to models is the injected call_model — it cannot escape it."""
    def agent(prompt, call_model):
        for _ in range(n_calls):
            call_model(model, [{"role": "user", "content": prompt}], {"max_tokens": 128})
        return answer
    return agent


# --- miner variants (honest + adversaries) -----------------------------------
class Miner:
    hotkey: str
    def produce(self, task, epoch, nonce) -> AttestationReport: ...


class HonestMiner(Miner):
    def __init__(self, signer, backend, platform, model, n_calls, skill=1.0):
        self.signer = signer; self.hotkey = signer.public_hex
        self.rt = TEERuntime(backend, platform)
        self.model, self.n_calls, self.skill = model, n_calls, skill

    def produce(self, task, epoch, nonce):
        from ..gateway.signing import sha256_hex
        h = int(sha256_hex(f"{self.hotkey}{task.task_id}")[:8], 16) / 0xFFFFFFFF
        answer = str(int(task.gold)) if h < self.skill else str(int(task.gold) + 1)
        return self.rt.run(calling_agent(self.model, self.n_calls, answer),
                           task_id=task.task_id, epoch=epoch, nonce=nonce, prompt=task.prompt)


class LyingRuntimeMiner(Miner):
    """Runs a MODIFIED runtime that reports cost 0 — but its measurement differs."""
    def __init__(self, signer, backend, platform):
        self.signer = signer; self.hotkey = signer.public_hex
        self.backend, self.platform = backend, platform

    def produce(self, task, epoch, nonce):
        rep = AttestationReport(task.task_id, epoch, nonce, str(int(task.gold)),
                                total_cost_usd=0.0, n_calls=0, call_log_hash="x",
                                measurement="MODIFIED-runtime-reports-zero-cost")
        return rep.attested_by(self.platform)     # genuine hardware, but unapproved image


class FakeQuoteMiner(Miner):
    """Self-signs the quote (no real TEE) — fails the hardware-root check."""
    def __init__(self, signer, backend, platform):
        self.signer = signer; self.hotkey = signer.public_hex

    def produce(self, task, epoch, nonce):
        fake_platform = Platform(self.signer)     # miner's own key, not the vendor root
        rep = AttestationReport(task.task_id, epoch, nonce, str(int(task.gold)),
                                0.0, 0, "x", runtime_measurement())
        return rep.attested_by(fake_platform)


class ReplayMiner(Miner):
    """Reuses a stale nonce (an old attestation) — fails the issued-nonce check."""
    def __init__(self, signer, backend, platform, model):
        self.signer = signer; self.hotkey = signer.public_hex
        self.rt = TEERuntime(backend, platform); self.model = model

    def produce(self, task, epoch, nonce):
        return self.rt.run(calling_agent(self.model, 1, str(int(task.gold))),
                           task_id=task.task_id, epoch=epoch, nonce="STALE-nonce", prompt=task.prompt)


# --- validator ---------------------------------------------------------------
@dataclass
class EpochReport:
    epoch: int
    scored: dict
    dq: dict
    slots: list
    weights_by_uid: dict


class TEEValidator:
    def __init__(self, approved_measurements, platform_public_hex, chain, reign,
                 lam=0.5, cost_ref=0.02, n_tasks=12):
        self.approved = set(approved_measurements)
        self.platform_pub = platform_public_hex
        self.chain, self.reign = chain, reign
        self.lam, self.cost_ref, self.n_tasks = lam, cost_ref, n_tasks
        self._nc = 0
        self._reg_block = {}
        self._epoch = 0

    def register(self, hotkey):
        self.chain.register(hotkey)
        self._reg_block.setdefault(hotkey, self.chain.current_block())

    def _issue_nonce(self):
        self._nc += 1
        return f"vnonce-{self._nc}"

    def run_epoch(self, miners, epoch_seed):
        self._epoch += 1
        tasks = generate_math_tasks(self.n_tasks, epoch_seed)
        scored, dq = {}, {}
        for m in miners:
            per, reason = [], None
            for t in tasks:
                nonce = self._issue_nonce()             # fresh, validator-issued
                try:
                    rep = m.produce(t, self._epoch, nonce)
                except Exception as e:
                    reason = f"produce_error:{type(e).__name__}"; break
                v = verify_attestation(
                    rep, approved_measurements=self.approved, platform_public_hex=self.platform_pub,
                    expect_task_id=t.task_id, expect_epoch=self._epoch, expect_nonce=nonce,
                    gold=t.gold, grade_fn=grade_math, lam=self.lam, cost_ref=self.cost_ref)
                if not v.valid:
                    reason = v.reason; break
                per.append(v.score)
            if reason:
                dq[m.hotkey] = reason
            else:
                scored[m.hotkey] = float(np.mean(per))

        subs = [Submission(hk, hk, self._reg_block.get(hk, 0), s) for hk, s in scored.items()]
        res = self.reign.update(subs) if subs else None
        weights = {}
        if res is not None:
            hk_to_uid = {hk: uid for uid, hk in self.chain.hotkeys().items()}
            for mid, w in res.weights.items():
                uid = 0 if mid == self.reign.burn_uid else hk_to_uid.get(mid)
                if uid is not None:
                    weights[uid] = weights.get(uid, 0.0) + w
            self.chain.set_weights(weights)
        return EpochReport(self._epoch, scored, dq, res.slots if res else [], weights)


def run_simulation(epochs: int = 5, verbose: bool = True) -> dict:
    backend = MockBackend()
    platform = Platform()                                  # the mock hardware root
    approved = {runtime_measurement()}                    # only the audited runtime is approved
    chain = MockChain(); chain.register("burn")
    val = TEEValidator(approved, platform.public_hex, chain, Reign(), n_tasks=12)

    k = {n: Signer() for n in ["honest-cheap", "honest-exp", "lying-rt", "fake-quote", "replayer"]}
    miners = [
        HonestMiner(k["honest-cheap"], backend, platform, "tiny", 1, skill=1.0),
        HonestMiner(k["honest-exp"], backend, platform, "frontier", 4, skill=1.0),
        LyingRuntimeMiner(k["lying-rt"], backend, platform),
        FakeQuoteMiner(k["fake-quote"], backend, platform),
        ReplayMiner(k["replayer"], backend, platform, "tiny"),
    ]
    name = {m.hotkey: n for n, m in zip(k, miners)}
    for m in miners:
        val.register(m.hotkey)

    emissions = defaultdict(float); dq_final = {}
    for e in range(epochs):
        rep = val.run_epoch(miners, epoch_seed=6000 + e)
        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            emissions[name.get(uid_hk.get(uid, ""), "burn")] += w
        dq_final.update({name.get(hk, hk): r for hk, r in rep.dq.items()})
    if verbose:
        total = sum(v for kk, v in emissions.items() if kk != "burn") or 1.0
        print("cumulative emissions:")
        for who, v in sorted(emissions.items(), key=lambda kv: -kv[1]):
            tag = f"  [DQ: {dq_final[who]}]" if who in dq_final else ("  [burn]" if who == "burn" else "")
            share = "  -- " if who == "burn" else f"{v/total*100:5.1f}%"
            print(f"  {who:14s} {v:6.3f}  {share}{tag}")
        print("\ndisqualifications:", dq_final)
    return {"emissions": dict(emissions), "disqualifications": dq_final}


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()
