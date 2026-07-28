#!/usr/bin/env python
"""Is verification unlearnable, or were the FEATURES the bottleneck?

Every negative result in this investigation shares one confound: a tiny CMA-ES head on frozen 384-d
MiniLM embeddings. Pick-one routing captured 1.1% of its band; cascade routing <=9%; a learned
verifier ~1.6%. Before concluding that the whole thesis is dead, that confound has to be isolated --
otherwise we may have measured "sentence embeddings do not carry correctness", not "correctness is
unpredictable".

So this builds the STRONGEST verifier the data allows and reads off what it is actually worth:

  features   MiniLM(answer) + MiniLM(question) + model identity + cheap surface cues
             (length, digits, refusal/hedge markers, parse-ability)
  learner    HistGradientBoostingClassifier -- same family `encoder_ceiling_experiment.py` used to
             establish the ROUTING ceiling, so the two answers are directly comparable
  readout    held-out AUC, the realized (tpr, fpr) operating point, and -- the number that matters --
             the CASCADE SCORE that verifier delivers when plugged into the real ladder

Reference points it sits between:
  always-escalate (pay for the best model every time)   -- the trivial policy to beat
  perfect verifier (accept iff actually correct)        -- the unreachable ceiling

near trivial   -> verification is FUNDAMENTALLY hard here; features were not the bottleneck
near perfect   -> the tiny head / MiniLM was the bottleneck and the subnet has a real surface

Caches answer embeddings to data/ so reruns are cheap. Offline, free.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from thirtyspokes import realdata as rd

REFUSAL = re.compile(r"i (cannot|can't|am unable|do not know|don't know)|as an ai|sorry", re.I)


def clean(x) -> str:
    s = str(x)
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)) and v:
                return str(v[0])
        except Exception:  # noqa: BLE001
            pass
    return s


def ladder(vok, correct, cost, order):
    Q = len(correct)
    done = np.zeros(Q, bool); banked = np.zeros(Q, bool); total = np.zeros(Q, np.float32)
    for pos, m in enumerate(order):
        active = ~done
        total[active] += cost[active, m]
        newly = active & (vok[:, m] | (pos == len(order) - 1))
        banked[newly] = correct[newly, m]
        done[newly] = True
    return banked, total


def score_of(vok, correct, cost, order, cnorm, lam):
    b, t = ladder(vok, correct, cost, order)
    return float(b.mean()) - lam * float((t / cnorm).mean())


def surface_feats(texts: list[str]) -> np.ndarray:
    """Cheap cues a verifier could plausibly exploit without knowing the answer."""
    out = np.zeros((len(texts), 5), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = (len(t), t.count("\n"), sum(c.isdigit() for c in t),
                  1.0 if REFUSAL.search(t) else 0.0, 1.0 if t.strip() else 0.0)
    out[:, 0] = np.log1p(out[:, 0])
    out[:, 2] = out[:, 2] / np.maximum(np.expm1(out[:, 0]), 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="strongest-learner verifier ceiling")
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--lam", type=float, default=0.5)
    args = p.parse_args()

    df = rd.load_routerbench("data/routerbench_0shot.pkl")
    emb_all = np.load("data/emb_minilm.npy")
    idx = np.random.default_rng(0).choice(len(df), min(args.n, len(df)), replace=False)
    sub, pemb = df.iloc[idx].reset_index(drop=True), emb_all[idx]
    cache = rd.to_cache(sub, rd.CURATED_POOL, pemb)
    order = np.argsort([m.price_per_token for m in cache.models])
    Q, K = cache.Q, cache.K
    print(f"Q={Q} K={K} lam={args.lam}", flush=True)

    cache_path = pathlib.Path(f"data/ans_emb_{args.n}.npy")
    texts_by_k = [[clean(t) for t in sub[f"{m}|model_response"].to_numpy()]
                  for m in rd.CURATED_POOL]
    if cache_path.exists():
        A = np.load(cache_path)
        print(f"loaded cached answer embeddings {A.shape}", flush=True)
    else:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        A = np.zeros((Q, K, pemb.shape[1]), dtype=np.float32)
        for k in range(K):
            A[:, k, :] = enc.encode(texts_by_k[k], batch_size=256, show_progress_bar=False,
                                    convert_to_numpy=True, normalize_embeddings=True)
            print(f"  embedded {rd.CURATED_POOL[k]}", flush=True)
        np.save(cache_path, A)

    # --- assemble the richest per-(q,k) feature matrix the data allows ------------------------
    # CROSS-MODEL AGREEMENT — the one signal class the embedding/surface features cannot express,
    # and the strongest cheap verification signal known in practice (self-consistency / majority
    # voting). If two independent models produce the same answer, that is evidence of correctness
    # that requires no knowledge of the answer itself. RouterBench gives us five responses per
    # question, so it is measurable here even though a real cascade would have to pay for the
    # second opinion. If agreement does not rescue verification, nothing cheap will.
    agree = np.zeros((Q, K, 3), dtype=np.float32)
    for k in range(K):
        sims = np.einsum("qd,qkd->qk", A[:, k, :], A)        # cosine (embeddings are normalized)
        others = np.delete(sims, k, axis=1)
        agree[:, k, 0] = others.mean(axis=1)                  # mean agreement with the rest
        agree[:, k, 1] = others.max(axis=1)                   # closest peer
        agree[:, k, 2] = (others > 0.95).sum(axis=1)          # near-exact matches

    rows, labels, qid, kid = [], [], [], []
    for k in range(K):
        sf = surface_feats(texts_by_k[k])
        onehot = np.zeros((Q, K), dtype=np.float32); onehot[:, k] = 1.0
        rows.append(np.hstack([A[:, k, :], pemb, sf, onehot, agree[:, k, :]]))
        labels.append(cache.correct[:, k])
        qid.append(np.arange(Q)); kid.append(np.full(Q, k))
    X = np.vstack(rows); y = np.concatenate(labels)
    qid = np.concatenate(qid); kid = np.concatenate(kid)
    print(f"feature matrix {X.shape}  positives={y.mean():.3f}", flush=True)

    # split by QUESTION so no answer to a train question appears in test
    rng = np.random.default_rng(1)
    perm = rng.permutation(Q); cut = int(0.7 * Q)
    trq, teq = set(perm[:cut].tolist()), set(perm[cut:].tolist())
    m_tr = np.array([q in trq for q in qid]); m_te = ~m_tr

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         early_stopping=True, random_state=0)
    clf.fit(X[m_tr], y[m_tr])
    p_te = clf.predict_proba(X[m_te])[:, 1]
    auc = roc_auc_score(y[m_te], p_te)
    print(f"\nheld-out verifier AUC = {auc:.4f}   (0.5 = no signal)", flush=True)

    # rebuild a (Q_test, K) acceptance matrix
    te_qs = np.array(sorted(teq)); pos = {q: i for i, q in enumerate(te_qs)}
    P = np.zeros((len(te_qs), K), dtype=np.float32)
    for prob, q, k in zip(p_te, qid[m_te], kid[m_te]):
        P[pos[q], k] = prob
    C, Cost = cache.correct[te_qs], cache.cost[te_qs]
    cnorm = cache.cost_max

    triv_lo = score_of(np.ones_like(C, bool), C, Cost, order, cnorm, args.lam)
    triv_hi = score_of(np.zeros_like(C, bool), C, Cost, order, cnorm, args.lam)
    perfect = score_of(C.copy(), C, Cost, order, cnorm, args.lam)
    trivial = max(triv_lo, triv_hi)
    band = perfect - trivial
    print(f"always-cheapest {triv_lo:+.4f}   always-escalate {triv_hi:+.4f}   "
          f"PERFECT {perfect:+.4f}   band {band:.4f}", flush=True)

    print(f"\n{'threshold':>10s} {'tpr':>7s} {'fpr':>7s} {'cascade':>9s} {'captured':>9s}")
    best = (-1e9, None)
    for th in np.arange(0.05, 1.0, 0.05):
        vok = P >= th
        tpr = float(vok[C].mean()) if C.any() else 0.0
        fpr = float(vok[~C].mean()) if (~C).any() else 0.0
        s = score_of(vok, C, Cost, order, cnorm, args.lam)
        cap = (s - trivial) / band if band > 1e-9 else 0.0
        print(f"{th:10.2f} {tpr:7.3f} {fpr:7.3f} {s:+9.4f} {100*cap:8.1f}%", flush=True)
        if s > best[0]:
            best = (s, th, tpr, fpr, cap)
    s, th, tpr, fpr, cap = best
    print()
    print("=" * 72)
    print(f"BEST learned verifier: threshold {th:.2f}  tpr {tpr:.3f}  fpr {fpr:.3f}")
    print(f"  cascade score {s:+.4f}  vs trivial {trivial:+.4f}  vs perfect {perfect:+.4f}")
    print(f"  CAPTURED {100*cap:.1f}% of the verification band")
    print("  VERDICT: " + ("FUNDAMENTAL — the strongest learner barely beats the trivial policy; "
                           "features were not the bottleneck"
                           if cap < 0.15 else
                           "FEATURE-BOUND — a strong learner reaches a useful operating point; "
                           "the tiny head / MiniLM was the limit"))


if __name__ == "__main__":
    main()
