"""The rotating corpus — building each window's dataset, and the alarm that says it went stale
(docs/ARENA.md (removed with v2, 2026-09-07) §3, §3.5).

The owner rotates the dataset per arena epoch. This module is the owner-side toolkit for doing
that without discretion leaking in:

  * `build_window_record` — the self-contained window file. Carries `epoch` alongside `window`
    (same value) so `holdout_feed.manifest` chains window records unchanged: the whole rotation
    schedule is committed in ONE root before any challenger commits, which is the double-blind
    ordering invariant (§3.5-1) in code.
  * `stratified_pick` — slices balanced across sources, so no single category can own a window
    and a category-lookup head cannot win on composition luck.
  * `staleness` — median max-cosine of the new window's features against everything already
    revealed. This is measurement 16's leak diagnostic turned into the arena's ossification
    alarm: drifting toward 1.0 means duels are re-asking old questions, all competent heads will
    tie, and the sitting king is being defended by the corpus instead of its own signal. The
    number is published in every reveal — the owner watches it, and so can everyone else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The arena ladder is imported where the arena's own builders need it, not here: v3 uses only
# `category_freshness`, which is pure, and a module-level import of `duel` dragged the retired
# matmul path back into v3's closure (M10, 2026-09-07).


# The owner publishes each window's dataset and each reveal here. Derived from the netuid rather
# than defaulted to one name, because the damaging mistake is silent: a testnet validator writing
# into the mainnet arena repo corrupts the public record that the whole one-shot rule is audited
# against, and nothing about it looks like a failure at the time. Same reasoning as `in_window()`
# in the daemon tests — the errors worth engineering out are the ones that do not raise.
MAINNET_NETUID = 99


def category_freshness(tags: list[str], revealed_tags) -> dict:
    """How much of this window re-asks a CATEGORY that has already been revealed.

    `staleness` answers the same question in embedding space and is the better instrument when
    categories are unlabelled; this one is exact when they are labelled, and it is the quantity
    §3-requirement-4 actually names. The two disagree usefully: a window can be lexically fresh
    (low staleness) while re-asking every category it ever asked, which is precisely the regime
    where a category-lookup artifact keeps winning and the arena reads it as skill.

    Recycled categories are not forbidden — a corpus with 27 intents cannot rotate forever — but
    the fraction has to be VISIBLE, because it bounds how much of a duel's outcome can be
    generalisation rather than recall.
    """
    seen = set(revealed_tags or ())
    recycled = [t for t in tags if t in seen]
    fresh_tags = sorted(set(tags) - seen)
    return {"n": len(tags), "n_categories": len(set(tags)),
            "recycled_fraction": (len(recycled) / len(tags)) if tags else 0.0,
            "fresh_categories": fresh_tags,
            "recycled_categories": sorted(set(tags) & seen)}


def staleness(new_features, revealed_features) -> float | None:
    """Median over the new window's asks of max-cosine to ANY previously revealed ask.

    None when nothing has been revealed yet (window 1 has nothing to leak from). Calibration
    points from measurement 16: 0.9034 was a template corpus whose random split flattered capture
    by +0.54; production-shaped fresh traffic measured ~0.61. Read ≥0.9 as "this window re-asks
    old questions", not as a hard gate — the number is published so the trend is public.
    """
    new = np.asarray(new_features, dtype=float)
    old = np.asarray(revealed_features, dtype=float) if revealed_features is not None else None
    if old is None or len(old) == 0:
        return None
    new = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
    old = old / np.maximum(np.linalg.norm(old, axis=1, keepdims=True), 1e-12)
    return float(np.median((new @ old.T).max(axis=1)))


# --- sources: what may enter the corpus, and the gate each must clear ---------------------------


def build_window(window: int, sources: list[Source], n: int, seed: int, *, netuid: int,
                 include_prompts: bool = True, encode=None,
                 revealed_features=None, revealed_categories=None) -> tuple[dict, dict]:
    """Assemble one window's dataset from admitted sources. -> (record, report).

    Only admitted sources contribute; the report names every refusal so a rotation that quietly
    lost a source is visible rather than silent. `encode` (the pinned encoder) is needed for the
    staleness diagnostic and for embedding-only mode.
    """
    verdicts = [admit_source(s) for s in sources]
    ok = [s for s, v in zip(sources, verdicts) if v.admitted]
    if not ok:
        raise ValueError("no admitted source: refusing to build a window nothing can be won on")

    picked = stratified_pick({s.name: list(range(len(s))) for s in ok}, n, seed)
    by_name = {s.name: s for s in ok}
    task_ids, prompts, rows_s, rows_c, rows_w, srcs = [], [], [], [], [], []
    for name in sorted(picked):
        s = by_name[name]
        for i in sorted(picked[name]):
            task_ids.append(s.task_ids[i])
            prompts.append(s.prompts[i])
            rows_s.append(np.asarray(s.success)[i])
            rows_c.append(np.asarray(s.cost)[i])
            rows_w.append(np.asarray(s.well_formed)[i])
            # Tagged at CATEGORY granularity when the source labels one. `sources` is the axis the
            # reveal reports per-category capture over, and a tag of `public:bitext` for all 2,160
            # asks would make that report a single row — i.e. no report at all.
            srcs.append(f"{s.kind}:{s.name}:{s.categories[i]}" if s.categories
                        else f"{s.kind}:{s.name}")

    features = np.asarray(encode(prompts), dtype=float) if encode is not None else None
    record = build_window_record(
        window, task_ids, np.asarray(rows_s), np.asarray(rows_c), np.asarray(rows_w),
        netuid=netuid, prompts=prompts if include_prompts else None,
        features=None if include_prompts else features, sources=srcs)

    report = {
        "window": window, "n": len(task_ids),
        "sources": {v.name: {"admitted": v.admitted, "reasons": v.reasons} for v in verdicts},
        "composition": {k: len(v) for k, v in picked.items()},
        "staleness": (staleness(features, revealed_features)
                      if features is not None else None),
        "category_freshness": category_freshness(srcs, revealed_categories),
    }
    return record, report
