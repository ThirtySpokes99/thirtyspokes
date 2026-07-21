"""The TinyRouter-faithful multi-turn orchestration engine + trainable policy.

Per turn the policy (the miner artifact) sees the task embedding + a few state
scalars and picks a (model, role); the episode runs up to `max_turns` turns,
accumulating a shared transcript, and a Verifier turn can accept and stop early.
Fully vectorized over questions (every question takes its turn simultaneously
under the shared policy), so training with CMA-ES is fast.

Roles: THINKER (add reasoning that boosts a later Worker), WORKER (produce an
answer), VERIFIER (accept the current answer and stop, or reject and continue).
The backend (synthetic now, live models later) supplies the per-turn effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .router import RouterHead
from .sepcmaes import SepCMAES

THINKER, WORKER, VERIFIER = 0, 1, 2
N_ROLES = 3
N_STATE = 5  # [turn, thinker_context, has_answer, n_think, last_rejected]


@dataclass
class EpisodeResult:
    correct: np.ndarray   # (Q,) bool — final answer correct
    cost: np.ndarray      # (Q,) float — summed model-call cost
    turns_used: np.ndarray
    role_counts: dict     # role -> total count across all turns


def make_policy_head(embed_dim: int, k_models: int, hidden: int = 24) -> RouterHead:
    """A policy head mapping [embedding + state scalars] -> logits over K*3 actions."""
    return RouterHead(d=embed_dim + N_STATE, k=k_models * N_ROLES, hidden=hidden)


def _state_features(backend, turn, tc, wc, n_think, last_rej, max_turns) -> np.ndarray:
    Q = backend.Q
    scal = np.stack([
        np.full(Q, turn / max_turns),
        np.tanh(tc / 3.0),                 # bounded view of accumulated thinker context
        (wc >= 0).astype(np.float64),      # has a working answer
        n_think / max_turns,
        last_rej,
    ], axis=1)
    return np.hstack([backend.embeddings, scal]).astype(np.float32)


def run_episodes(head: RouterHead, theta: np.ndarray, backend, max_turns: int) -> EpisodeResult:
    Q, K = backend.Q, backend.K
    tc = np.zeros(Q)              # thinker context
    wc = np.full(Q, -1)          # working correct: -1 none / 0 wrong / 1 right
    n_think = np.zeros(Q, dtype=np.int64)
    done = np.zeros(Q, dtype=bool)
    cost = np.zeros(Q)
    last_rej = np.zeros(Q)
    turns_used = np.zeros(Q, dtype=np.int64)
    role_counts = {THINKER: 0, WORKER: 0, VERIFIER: 0}

    for turn in range(max_turns):
        active = ~done
        if not active.any():
            break
        feats = _state_features(backend, turn, tc, wc, n_think, last_rej, max_turns)
        L = head.logits(theta, feats).reshape(Q, K, N_ROLES)
        # mask Verifier when there is no answer yet (pointless / invalid)
        L[wc == -1, :, VERIFIER] = -1e9
        action = L.reshape(Q, K * N_ROLES).argmax(axis=1)
        model = action // N_ROLES
        role = action % N_ROLES
        turns_used[active] += 1

        is_think = active & (role == THINKER)
        is_work = active & (role == WORKER)
        is_ver = active & (role == VERIFIER)

        # THINKER: accumulate diminishing context
        inc = backend.thinker_increment(model, n_think)
        tc[is_think] += inc[is_think]
        n_think[is_think] += 1

        # WORKER: draw an answer using current (prior-turn) thinker context
        corr = backend.worker_correct(model, tc, turn)
        wc[is_work] = corr[is_work].astype(np.int64)
        last_rej[is_work] = 0

        # VERIFIER: accept (stop) or reject (continue)
        acc = backend.verifier_accept(model, wc, turn)
        accept = is_ver & acc
        done[accept] = True
        last_rej[is_ver & (~acc)] = 1

        # cost: one model call per acting question
        acted = is_think | is_work | is_ver
        cost[acted] += backend.price[model[acted]]
        role_counts[THINKER] += int(is_think.sum())
        role_counts[WORKER] += int(is_work.sum())
        role_counts[VERIFIER] += int(is_ver.sum())

    return EpisodeResult(correct=(wc == 1), cost=cost, turns_used=turns_used,
                         role_counts=role_counts)


@dataclass
class TrainedOrchestrator:
    head: RouterHead
    theta: np.ndarray
    lam: float
    max_turns: int
    cost_ref: float

    def run(self, backend) -> EpisodeResult:
        return run_episodes(self.head, self.theta, backend, self.max_turns)


def train_orchestrator(backend, lam: float, hidden: int = 24, iters: int = 140,
                       max_turns: int = 5, train_mask: np.ndarray | None = None,
                       seed: int = 0, verbose: bool = False) -> TrainedOrchestrator:
    head = make_policy_head(backend.embeddings.shape[1], backend.K, hidden)
    cost_ref = float(max_turns * backend.price.max())
    if train_mask is None:
        train_mask = np.ones(backend.Q, dtype=bool)

    def neg_fitness(theta: np.ndarray) -> float:
        r = run_episodes(head, theta, backend, max_turns)
        obj = r.correct.astype(float) - lam * (r.cost / cost_ref)
        return -float(obj[train_mask].mean())

    es = SepCMAES(head.init_theta(seed), sigma0=0.3, seed=seed + 1)
    best_theta, best = es.mean.copy(), neg_fitness(es.mean)
    for it in range(iters):
        pop = es.ask()
        fits = np.array([neg_fitness(x) for x in pop])
        es.tell(pop, fits)
        i = int(fits.argmin())
        if fits[i] < best:
            best, best_theta = float(fits[i]), pop[i].copy()
        m = neg_fitness(es.mean)
        if m < best:
            best, best_theta = m, es.mean.copy()
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f"  iter {it:4d}  train_obj={-best:+.4f}  sigma={es.sigma:.3f}")

    return TrainedOrchestrator(head, best_theta, lam, max_turns, cost_ref)
