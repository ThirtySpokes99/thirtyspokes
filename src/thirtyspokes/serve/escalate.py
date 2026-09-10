"""The escalate predictor — the one learnable decision the measurements left standing.

WHY THIS SHAPE. Seven-way pick-one routing is measured dead: the two routing heads earning on
mainnet today, replayed against the real 112-ask matrix, sit ON the featureless frontier at 12-18x
its cost (docs/ROUTER_V2.md (removed with v2, 2026-09-07) §8). What survives, measured twice on production-shaped traffic, is the
BINARY decision — "will the cheap tier fail this ask?" — at held-out AUC 0.8355 (support) and 0.812
(production-shaped difficulty). That is also where the wider field landed: RouteLLM routes between
exactly two tiers, and FrugalGPT's cascades win on when-to-escalate, not on which-of-k. So the
learned object here is a logistic probe on the pinned encoder's embeddings: p(cheap tier fails).

WHERE THE LABELS COME FROM. Serving generates them for free. Every request that enters at the cheap
tier observes whether it produced a usable answer — that IS the escalate label, no grader, no gold,
no extra spend. The request log is therefore a labelled dataset that grows at traffic rate, and this
module turns it into a predictor. Two lessons from the measurement record are baked in rather than
rediscovered:

  * the split is BY TIME, never random. Measurement 16: random splits of self-similar traffic leak
    (median held-out cosine 0.9034 to nearest train ask) and flatter capture by 0.5. Training on the
    past and testing on the future is the deployment question asked honestly.
  * classes are balanced in training. At a ~6% escalate base rate an unweighted fit collapses to
    the majority class and reads as "nothing to learn" (the support-matrix trap).

THE PROMOTION GATE. A predictor serves only if its held-out-by-time AUC clears MIN_AUC = 0.70 — the
same bar the suite admission gate (C, legibility) sets for traffic, reused deliberately: a predictor
below it is indistinguishable from routing on noise, and serving it would spend latency and money on
coin flips. Below the bar the pipeline keeps serving cheap-first + escalate-on-failure, which is
already on the measured frontier. Nothing is lost by refusing.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np

from ..koth.harness import EMBED_DIM

MIN_AUC = 0.70          # suite admission gate C's legibility bar, reused for the same reason
MIN_TEST = 50           # below this the AUC's own error bar spans the gate

# The trainer's own end-to-end check that a written predictor round-trips: not a security bound,
# just the file format for a (w, b) logistic probe over the pinned encoder's features.


def save_predictor(w: np.ndarray, b: float) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, w=np.asarray(w, dtype=np.float32).reshape(-1), b=np.float32(b))
    return buf.getvalue()


def load_predictor(blob: bytes):
    """-> callable (Q, EMBED_DIM) features -> (Q,) p(escalate). Strict shape, no pickle."""
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    w = np.asarray(z["w"], dtype=np.float64).reshape(-1)
    b = float(np.asarray(z["b"]).reshape(()))
    if w.shape != (EMBED_DIM,):
        raise ValueError(f"predictor has {w.shape[0]} weights, expected {EMBED_DIM}")

    def predict(features: np.ndarray) -> np.ndarray:
        z = np.asarray(features, dtype=np.float64) @ w + b
        return 1.0 / (1.0 + np.exp(-z))
    return predict


# --- feature transport: embeddings in the request log ---------------------------------------
# float16 base64 — 768 bytes per request against ~4KB as JSON floats. Embeddings are SEMANTIC:
# they do not contain the prompt text but they are partially invertible, so a log with features
# must be treated as sensitive and never published. That is still a strictly better default than
# logging prompt text, which is why features are a separate opt-in from --log-prompts.

def encode_features(features: np.ndarray) -> str:
    return base64.b64encode(np.asarray(features, dtype=np.float16).tobytes()).decode()


def decode_features(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float64)


# --- training from the request log ----------------------------------------------------------

def _rows(records: list[dict], tier_cheap: str) -> tuple[np.ndarray, np.ndarray]:
    """(features, label) from log records, time-ordered.

    A record is usable iff the CHEAP tier was actually attempted (entered there) and features were
    logged. The label is "the cheap tier did not produce the served answer": a fallback or an
    outright error. Records that entered at the strong tier observe nothing about the cheap tier
    and are excluded — which is why serving keeps an explore rate: a predictor that routed
    everything to the strong tier would otherwise starve its own successor of labels.
    """
    feats, ys, ts = [], [], []
    for r in records:
        if r.get("chosen_model") != tier_cheap or "features_b64" not in r:
            continue
        f = decode_features(r["features_b64"])
        if f.shape != (EMBED_DIM,):
            continue
        feats.append(f)
        ys.append(1.0 if (r.get("fell_back") or "error" in r) else 0.0)
        ts.append(float(r.get("ts", 0.0)))
    if not feats:
        return np.empty((0, EMBED_DIM)), np.empty(0)
    order = np.argsort(np.asarray(ts), kind="stable")
    return np.asarray(feats)[order], np.asarray(ys)[order]


def train_from_records(records: list[dict], tier_cheap: str,
                       test_frac: float = 0.2) -> tuple[bytes | None, dict]:
    """Fit the probe on the oldest (1-test_frac) of the log, evaluate on the newest test_frac.

    Returns (weights_or_None, report). Weights are None when there is nothing to fit — the report
    says why. The report is the promotion evidence and should be kept next to the predictor.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = _rows(records, tier_cheap)
    n = len(y)
    report: dict = {"n_usable": n, "base_rate": round(float(y.mean()), 4) if n else None}
    n_test = int(n * test_frac)
    if n_test < MIN_TEST:
        report["error"] = f"only {n_test} test rows (need {MIN_TEST}); keep serving and collecting"
        return None, report
    Xtr, ytr, Xte, yte = X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        report["error"] = "a split is single-class — the cheap tier is (n)ever failing; nothing to route"
        return None, report

    # class_weight balanced: at a ~6% base rate an unweighted fit predicts "never escalate"
    # for every ask and reads as AUC ~0.5 — a training artifact, not a verdict on the traffic.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    picked = p >= 0.5
    report.update({
        "n_train": len(ytr), "n_test": len(yte),
        "auc": round(float(roc_auc_score(yte, p)), 4),
        "recall_at_0.5": round(float((picked & (yte == 1)).sum() / max(1, (yte == 1).sum())), 4),
        "fpr_at_0.5": round(float((picked & (yte == 0)).sum() / max(1, (yte == 0).sum())), 4),
        "escalate_rate_at_0.5": round(float(picked.mean()), 4),
    })
    return save_predictor(clf.coef_.reshape(-1), float(clf.intercept_[0])), report


def gate(report: dict, min_auc: float = MIN_AUC) -> tuple[bool, str]:
    """Promote only what would beat serving without a predictor. Refusal is the safe state."""
    if "error" in report:
        return False, report["error"]
    if report["auc"] < min_auc:
        return False, (f"held-out-by-time AUC {report['auc']} < {min_auc} — below the legibility "
                       f"bar this is routing on noise; cheap-first without a predictor is better")
    return True, f"AUC {report['auc']} clears {min_auc} on {report['n_test']} future asks"


def main() -> None:
    """`orchestra-serve-train` — turn a request log into a servable escalate predictor."""
    from .policy import TIER_CHEAP
    ap = argparse.ArgumentParser(description="train the escalate predictor from a request log")
    ap.add_argument("--log", required=True, help="requests.jsonl written with --log-features")
    ap.add_argument("--out", default="escalate.npz")
    ap.add_argument("--min-auc", type=float, default=MIN_AUC)
    ap.add_argument("--tier-cheap", default=TIER_CHEAP)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.log).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    weights, report = train_from_records(records, args.tier_cheap)
    print(json.dumps(report, indent=2))
    ok, why = gate(report, args.min_auc) if weights is not None else (False, report.get("error", ""))
    if not ok:
        raise SystemExit(f"NOT PROMOTED: {why}")
    Path(args.out).write_bytes(weights)
    Path(args.out + ".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PROMOTED: {why}\nwrote {args.out} — serve with: orchestra-serve --escalate-weights {args.out}")


if __name__ == "__main__":
    main()
