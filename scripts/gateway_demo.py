#!/usr/bin/env python
"""Linchpin demo: metered gateway + verifiable grading, and every attack failing.

Runs fully offline (mock provider, procedural math tasks). Shows honest cost-
efficient work winning, and cost-lying / gateway-bypass / receipt-forgery /
receipt-theft all rejected — the metered-gateway + verifiable-grading trust flow, end to end.
"""
from __future__ import annotations

from thirtyspokes.gateway.gateway import MeteringGateway, MockBackend, gateway_call
from thirtyspokes.gateway.grader import generate_math_tasks, grade_math
from thirtyspokes.gateway.receipts import Receipt, TaskSubmission
from thirtyspokes.gateway.signing import Signer, sha256_hex
from thirtyspokes.gateway.verify import verify_submission

LAM, COST_REF = 0.5, 0.02
_N = [0]


def nonce():
    _N[0] += 1
    return f"n{_N[0]}"


def agent(gw, miner, task, model, n_calls):
    receipts = {}
    for _ in range(n_calls):
        _, r = gateway_call(gw, miner, model, [{"role": "user", "content": task.prompt}],
                            {"max_tokens": 64}, nonce())
        receipts[r.call_id] = r
    ans = str(int(task.gold))
    _, rf = gateway_call(gw, miner, model, [{"role": "user", "content": ans}],
                         {"finalize": True}, nonce())
    receipts[rf.call_id] = rf
    return TaskSubmission(miner.public_hex, task.task_id, ans, tuple(receipts)), receipts


def main() -> None:
    task = generate_math_tasks(1, seed=42)[0]
    gw = MeteringGateway(MockBackend(), set())
    who = {n: Signer() for n in ["cheap", "expensive", "bypasser", "forger", "thief"]}
    for s in who.values():
        gw.register(s.public_hex)

    def show(name, sub, receipts):
        v = verify_submission(sub, receipts, gw.public_hex, task.gold, grade_math, LAM, COST_REF)
        verdict = f"score={v.score:+.3f} (q={v.quality:.0f}, cost=${v.cost_usd:.5f})" if v.valid \
            else f"REJECTED: {v.reason}"
        print(f"  {name:22s} {verdict}")

    print(f"task: {task.prompt}  (gold={int(task.gold)})   lambda={LAM}\n")

    sc, rc = agent(gw, who["cheap"], task, "small", 1)
    show("honest, cheap", sc, rc)
    se, re = agent(gw, who["expensive"], task, "frontier", 4)
    show("honest, expensive", se, re)

    # bypasser: correct answer, no gateway work
    bp = who["bypasser"]
    show("gateway-bypasser", TaskSubmission(bp.public_hex, task.task_id, str(int(task.gold)), ()), {})

    # forger: self-signed fake cheap receipt
    fg = who["forger"]
    fake = Receipt("cx", fg.public_hex, "frontier", "ph", sha256_hex(str(int(task.gold))),
                   1, 1, 1e-5, 1).signed_by(fg)
    show("receipt-forger", TaskSubmission(fg.public_hex, task.task_id, str(int(task.gold)), ("cx",)),
         {"cx": fake})

    # thief: reuse cheap miner's real receipts
    th = who["thief"]
    show("receipt-thief", TaskSubmission(th.public_hex, task.task_id, str(int(task.gold)), tuple(rc)), rc)

    print("\n=> only honest, gateway-backed work scores; cheaper wins; every forgery fails.")


if __name__ == "__main__":
    main()
