"""Option-2 subnet — the full free-agent competition, integrated and runnable offline.

Ties the linchpin (metered gateway + verifiable grading + proof verification) to
the reign and chain into a complete subnet loop:

  each epoch: sample FRESH tasks -> miners run THEIR OWN agents through the gateway
  (paying real metered cost) and submit answer+proof -> validator VERIFIES each
  proof (grade answer + check receipts + binding) and scores quality - lam*cost ->
  rank-weighted reign -> set weights.

Validators never re-run agents; miners bear inference cost. Miners are arbitrary
here — a HonestAgent (any cost/quality point) or an adversary — exactly the point
of Option 2: the subnet doesn't constrain how a miner works, only proves what it
did and what it cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..reign import Reign, Submission
from ..subnet.chain import MockChain
from .gateway import MeteringGateway, MockBackend, gateway_call
from .grader import generate_math_tasks, grade_math
from .receipts import Receipt, TaskSubmission
from .signing import Signer, sha256_hex
from .verify import verify_submission

_NONCE = [0]


def _nonce() -> str:
    _NONCE[0] += 1
    return f"n{_NONCE[0]}"


# --- miner agents (arbitrary; the subnet doesn't constrain them) -------------
class AgentMiner:
    hotkey: str

    def run_task(self, gw: MeteringGateway, task) -> tuple[TaskSubmission, dict]:
        raise NotImplementedError


class HonestAgent(AgentMiner):
    """Thinks with `model` for `n_calls`, then emits an answer via the gateway.
    `skill` = P(correct); cheaper/weaker models cost less but may answer wrong."""

    def __init__(self, signer: Signer, model: str, n_calls: int, skill: float = 1.0):
        self.signer = signer
        self.hotkey = signer.public_hex
        self.model = model
        self.n_calls = n_calls
        self.skill = skill

    def run_task(self, gw, task):
        receipts = {}
        for _ in range(self.n_calls):
            _, r = gateway_call(gw, self.signer, self.model,
                                [{"role": "user", "content": task.prompt}],
                                {"max_tokens": 64}, _nonce())
            receipts[r.call_id] = r
        # deterministic correctness from a hash (no RNG): correct if skill high
        h = int(sha256_hex(f"{self.hotkey}{task.task_id}")[:8], 16) / 0xFFFFFFFF
        answer = str(int(task.gold)) if h < self.skill else str(int(task.gold) + 1)
        _, rf = gateway_call(gw, self.signer, self.model,
                             [{"role": "user", "content": answer}],
                             {"finalize": True}, _nonce())
        receipts[rf.call_id] = rf
        return TaskSubmission(self.hotkey, task.task_id, answer, tuple(receipts)), receipts


class BypassAgent(AgentMiner):
    """Solves off-gateway, submits the answer with no backing work -> rejected."""

    def __init__(self, signer):
        self.signer = signer
        self.hotkey = signer.public_hex

    def run_task(self, gw, task):
        return TaskSubmission(self.hotkey, task.task_id, str(int(task.gold)), ()), {}


class ForgerAgent(AgentMiner):
    """Self-signs a fake cheap receipt -> gateway-signature check fails."""

    def __init__(self, signer):
        self.signer = signer
        self.hotkey = signer.public_hex

    def run_task(self, gw, task):
        ans = str(int(task.gold))
        fake = Receipt("fake", self.hotkey, "frontier", "ph", sha256_hex(ans),
                       1, 1, 1e-5, 1).signed_by(self.signer)
        return TaskSubmission(self.hotkey, task.task_id, ans, ("fake",)), {"fake": fake}


# --- the validator epoch loop ------------------------------------------------
@dataclass
class EpochReport:
    epoch: int
    scored: dict[str, float]
    disqualifications: dict[str, str]
    slots: list[str]
    weights_by_uid: dict[int, float]


class Option2Validator:
    def __init__(self, gateway: MeteringGateway, chain: MockChain, reign: Reign,
                 lam: float = 0.5, cost_ref: float = 0.02, tasks_per_epoch: int = 12):
        self.gw = gateway
        self.chain = chain
        self.reign = reign
        self.lam = lam
        self.cost_ref = cost_ref
        self.n_tasks = tasks_per_epoch
        self._reg_block: dict[str, int] = {}
        self._epoch = 0

    def register(self, hotkey: str) -> None:
        self.chain.register(hotkey)
        self.gw.register(hotkey)
        self._reg_block.setdefault(hotkey, self.chain.current_block())

    def run_epoch(self, miners: list[AgentMiner], epoch_seed: int) -> EpochReport:
        self._epoch += 1
        tasks = generate_math_tasks(self.n_tasks, epoch_seed)
        scored: dict[str, float] = {}
        dq: dict[str, str] = {}

        for m in miners:
            per_task = []
            reason = None
            for task in tasks:
                sub, receipts = m.run_task(self.gw, task)         # miner works (pays cost)
                v = verify_submission(sub, receipts, self.gw.public_hex, task.gold,
                                      grade_math, self.lam, self.cost_ref)
                if not v.valid:
                    reason = v.reason
                    break
                per_task.append(v.score)
            if reason:
                dq[m.hotkey] = reason
            else:
                scored[m.hotkey] = float(np.mean(per_task))

        subs = [Submission(hk, hk, self._reg_block.get(hk, 0), s) for hk, s in scored.items()]
        res = self.reign.update(subs) if subs else None

        weights_by_uid: dict[int, float] = {}
        if res is not None:
            hk_to_uid = {hk: uid for uid, hk in self.chain.hotkeys().items()}
            for mid, w in res.weights.items():
                uid = 0 if mid == self.reign.burn_uid else hk_to_uid.get(mid)
                if uid is not None:
                    weights_by_uid[uid] = weights_by_uid.get(uid, 0.0) + w
            self.chain.set_weights(weights_by_uid)

        return EpochReport(self._epoch, scored, dq,
                           res.slots if res else [], weights_by_uid)


# --- end-to-end simulation ---------------------------------------------------
def run_simulation(epochs: int = 6, verbose: bool = True) -> dict:
    gw = MeteringGateway(MockBackend(), set())
    chain = MockChain()
    chain.register("burn")                       # uid 0 = burn
    reign = Reign()
    val = Option2Validator(gw, chain, reign, lam=0.5, tasks_per_epoch=12)

    keys = {n: Signer() for n in ["cheap", "expensive", "wrong", "forger", "bypasser"]}
    miners: list[AgentMiner] = [
        HonestAgent(keys["cheap"], "small", n_calls=1, skill=1.0),      # correct, tiny cost
        HonestAgent(keys["expensive"], "frontier", n_calls=4, skill=1.0),  # correct, big cost
        HonestAgent(keys["wrong"], "small", n_calls=1, skill=0.3),      # cheap but often wrong
        ForgerAgent(keys["forger"]),
        BypassAgent(keys["bypasser"]),
    ]
    name_of = {m.hotkey: n for n, m in zip(["cheap", "expensive", "wrong", "forger", "bypasser"], miners)}
    for m in miners:
        val.register(m.hotkey)

    from collections import defaultdict
    emissions: dict[str, float] = defaultdict(float)
    dq_final: dict[str, str] = {}
    for e in range(epochs):
        rep = val.run_epoch(miners, epoch_seed=5000 + e)
        uid_hk = chain.hotkeys()
        for uid, w in rep.weights_by_uid.items():
            emissions[name_of.get(uid_hk.get(uid, ""), "burn")] += w
        dq_final.update({name_of.get(hk, hk): r for hk, r in rep.disqualifications.items()})
        if verbose:
            slots = [name_of.get(hk, hk) for hk in rep.slots]
            sc = {name_of.get(hk, hk): round(s, 3) for hk, s in rep.scored.items()}
            print(f"[epoch {e}] champion={slots[0] if slots and slots[0] else '-':10s} scores={sc}")

    if verbose:
        total = sum(v for k, v in emissions.items() if k != "burn") or 1.0
        print("\ncumulative emissions:")
        for who, v in sorted(emissions.items(), key=lambda kv: -kv[1]):
            tag = f"  [DQ: {dq_final[who]}]" if who in dq_final else ("  [burn]" if who == "burn" else "")
            share = "  -- " if who == "burn" else f"{v/total*100:5.1f}%"
            print(f"  {who:12s} {v:6.3f}  {share}{tag}")
        print("\ndisqualifications:", dq_final)
    return {"emissions": dict(emissions), "disqualifications": dq_final}


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()
