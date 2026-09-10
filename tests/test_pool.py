"""`pool.narrow` — today's catalog restricted to the pool λ was fitted on (§5.1b, §6.2).

One definition, shared by the M3a script that fits the table and the production world that prices
arms with it. Measured 2026-09-07: with the narrowing only in the script, the daemon's King0 climbed
the whole catalog to `openai/gpt-4` while λ priced a five-model ladder.
"""

from __future__ import annotations

import pytest

from thirtyspokes.v3.pool import CatalogError, narrow
from thirtyspokes.v3.types import Catalog, CatalogEntry


def _entry(model_id: str, price: float) -> CatalogEntry:
    return CatalogEntry(model_id=model_id, price_in_per_mtok=price, price_out_per_mtok=price * 4,
                        context_length=128_000)


CATALOG = Catalog(entries=(_entry("z/dear", 5.0), _entry("a/cheap", 0.05), _entry("m/mid", 0.5),
                           _entry("x/other", 1.0)))


def test_narrowing_keeps_the_pinned_models_in_price_order_whatever_order_they_were_pinned_in():
    """`reference.cascade` takes rungs by position in the PRICE-sorted catalog, so the order the
    pool was written down in must not leak into which model is the cheapest rung."""
    narrowed = narrow(CATALOG, ["z/dear", "a/cheap", "m/mid"])

    assert [e.model_id for e in narrowed.entries] == ["a/cheap", "m/mid", "z/dear"]
    assert "x/other" not in {e.model_id for e in narrowed.entries}


def test_a_pinned_model_missing_from_the_catalog_is_refused_by_name_not_substituted():
    with pytest.raises(CatalogError, match=r"absent from today's usable catalog: \['gone/model'\]"):
        narrow(CATALOG, ["a/cheap", "gone/model"])


def test_a_pool_of_one_is_not_a_ladder():
    with pytest.raises(CatalogError, match="at least two"):
        narrow(CATALOG, ["a/cheap"])


def test_a_duplicated_pin_is_one_rung():
    assert [e.model_id for e in narrow(CATALOG, ["m/mid", "a/cheap", "m/mid"]).entries] == [
        "a/cheap", "m/mid"]
