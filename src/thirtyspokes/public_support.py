"""Public Bitext support-intent experiment helpers.

Bitext is a hybrid-synthetic, exactly labelled customer-support dataset.  These
helpers keep its sampling and grading deterministic; live provider calls remain
in ``scripts/collect_public_support_pilot.py``.
"""

from __future__ import annotations

from collections import defaultdict
import re

import numpy as np


def stratified_indices(labels: list[str], n: int, seed: int = 0) -> list[int]:
    """Select ``n`` examples as evenly as possible across every label."""
    if n < 1 or n > len(labels):
        raise ValueError(f"n must be between 1 and {len(labels)}")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    names = sorted(grouped)
    rng = np.random.default_rng(seed)
    base, remainder = divmod(n, len(names))
    selected: list[int] = []
    for position, name in enumerate(names):
        take = base + int(position < remainder)
        candidates = np.asarray(grouped[name], dtype=int)
        if take > len(candidates):
            raise ValueError(f"label {name!r} has only {len(candidates)} examples; need {take}")
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def stratified_folds(labels: list[str], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign every row to a balanced, deterministic fold within each label."""
    if not 2 <= n_folds <= len(labels):
        raise ValueError("n_folds must be between 2 and the number of rows")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for indices in grouped.values():
        shuffled = np.asarray(indices, dtype=int)
        rng.shuffle(shuffled)
        folds[shuffled] = np.arange(len(shuffled)) % n_folds
    return folds


def extract_intent(text: str, allowed: list[str] | tuple[str, ...]) -> str | None:
    """Parse one allowed intent from a terse model response."""
    normalized = text.strip().lower().strip("`\"' .")
    allowed_set = set(allowed)
    if normalized in allowed_set:
        return normalized
    matches = [
        label for label in allowed
        if re.search(rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])", normalized)
    ]
    return matches[0] if len(matches) == 1 else None
