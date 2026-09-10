"""The production world — the object `--world` had to return and nothing returned (§5.1b, §5.5).

`validator.main` requires `--world module:attr` yielding the published `Pins`, and until `world.py`
there was no such callable in the tree: `Pins` lived beside the mock world in `simulate.py`, so the
flag was required and had nothing to satisfy it. These tests pin the three things that module must
not get wrong — the table is loaded rather than fitted, King0 is chosen rather than declared, and
the reference pair is one that can actually separate a window.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirtyspokes.v3 import reference, world
from thirtyspokes.v3.types import CatalogEntry, TaskSpec
from thirtyspokes.v3.pool import Catalog

# Two benchmarks that price and one that does not. TWO, because `duel` refuses a slice it cannot
# judge breadth on, so a single priced benchmark is not a launchable corpus and `pins` now says so —
# a fixture built on one would have been testing a world no window could run.
TABLE = {"livecodebench": {"lam": 0.25, "c_per_task": 0.04, "flags": []},
         "r2egym": {"lam": 0.18, "c_per_task": 0.06, "flags": []},
         "hle": {"lam": 0.0, "c_per_task": 0.002, "flags": ["dominator: no quality bought"]}}


def entry(model_id: str, price: float) -> CatalogEntry:
    return CatalogEntry(model_id=model_id, price_in_per_mtok=price,
                        price_out_per_mtok=price * 4, context_length=128_000)


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(entries=(entry("cheap/one", 0.05), entry("mid/two", 0.30),
                            entry("strong/three", 1.00)))


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "m3a-exchange.json"
    path.write_text(json.dumps(TABLE), encoding="utf-8")
    return path


def tasks(*benchmarks: str) -> tuple[TaskSpec, ...]:
    return tuple(TaskSpec(task_id=f"{b}-{i}", benchmark=b, prompt="q", tools=())
                 for i, b in enumerate(benchmarks))


def build(table: Path, catalog: Catalog, *, benchmarks=("livecodebench", "r2egym"), quality=None):
    return world.pins(exchange_path=table, catalog=catalog, benchmarks=(), worker=object(),
                      tasks=tasks(*benchmarks), quality=quality)


def test_the_published_table_is_loaded_verbatim_including_its_flags(table, catalog):
    """A flagged benchmark is KEPT. `score_arm` excludes it and must be able to say which and why;
    dropping it here would make an exclusion §5.1b requires to be announced completely silent."""
    pins = build(table, catalog)
    assert pins.world.exchange["hle"].lam == 0.0
    assert pins.world.exchange["hle"].flags == ("dominator: no quality bought",)
    assert pins.world.exchange["livecodebench"].lam == 0.25


def test_a_corpus_the_table_does_not_price_is_refused(table, catalog):
    """A benchmark scored under a table that never measured it is priced by a constant nobody
    derived — and it would price a crown."""
    with pytest.raises(world.WorldError, match="prices no"):
        build(table, catalog, benchmarks=("livecodebench", "r2egym", "swebench_pro"))


def test_king_zero_defaults_to_the_cascade_and_its_contrast_is_the_cheapest(table, catalog):
    """§5.5's expectation, and §5.6's pair. Always-strongest would share the cascade's TOP rung."""
    pins = build(table, catalog)
    assert pins.king_zero.name == reference.CASCADE
    assert pins.contrast.name == reference.ALWAYS_CHEAPEST
    assert pins.king_zero.rungs[-1] != pins.contrast.rungs[-1]


def test_king_zero_is_chosen_by_measurement_when_one_is_supplied(table, catalog):
    """§5.5: King0 is whichever fixed policy M3a found best, not whichever this module prefers."""
    pins = build(table, catalog, quality={reference.ALWAYS_CHEAPEST: 0.9, reference.CASCADE: 0.1})
    assert pins.king_zero.name == reference.ALWAYS_CHEAPEST
    # And the contrast flips with it, or the pair would compare a policy against itself.
    assert pins.contrast.name == reference.ALWAYS_STRONGEST
    assert pins.king_zero.rungs[-1] != pins.contrast.rungs[-1]


def test_a_pool_too_narrow_to_separate_is_refused_rather_than_scored_forever(table):
    """THE FAILURE THAT IS INVISIBLE IN PRODUCTION. A pair sharing a top rung measures +0.0000 every
    window, so the power gate refuses each one and the arena never opens while the corpus is
    perfectly healthy — a quiet validator, not an error. Measured on this repo's archetype world as
    cascade vs always-strongest (§5.6); here the same collapse arrives from a one-model pool."""
    with pytest.raises(world.WorldError, match="does not separate"):
        build(table, Catalog(entries=(entry("only/one", 0.05),)))


def test_a_corpus_with_one_priced_benchmark_is_refused_at_launch_not_at_every_duel(table, catalog):
    """The gap this gate closes, and it is the LIVE case: the full M3a run of 2026-09-05 prices
    livecodebench and flags hle, which is exactly one.

    `duel` already refuses a slice it cannot judge breadth on — leave-one-out over a single
    benchmark scores the challenger on an empty slice, and a NaN must never decide a crown
    (`test_invariant2.py` pins that raise). What was missing is that the refusal arrived per
    DUEL: `preflight` reported green, and then `Validator._admit`'s broad handler caught the
    ValueError and published it as a refusal of the challenger, every window, for every miner. A
    corpus that cannot be duelled has to be refused where it costs a message rather than a field of
    miners who each read it as their own failure."""
    with pytest.raises(world.WorldError, match="cannot be judged on one benchmark"):
        build(table, catalog, benchmarks=("livecodebench", "hle"))


def test_a_priced_benchmark_the_slice_did_not_draw_is_named_as_such_not_as_lost(table, catalog):
    """MEASURED 2026-09-07, first live launch: the table priced two benchmarks, the smoke drew one,
    and the refusal said "the benchmarks this corpus lost:" above an empty list. A benchmark the
    guard excluded and a benchmark nobody drew have different fixes; the message says which."""
    with pytest.raises(world.WorldError) as caught:
        build(table, catalog, benchmarks=("livecodebench",))          # r2egym priced, not drawn
    text = str(caught.value)
    assert "r2egym: priced" in text and "not in this slice" in text
    assert "draw them" in text
    assert "hle: dominator" in text                                     # the guard's own exclusion


def test_the_one_priced_refusal_names_the_benchmark_that_survived_and_the_ones_that_did_not(
        table, catalog):
    """Same requirement §5.1b puts on the all-flagged refusal: an exclusion must be ANNOUNCED, or
    the owner goes looking for a configuration bug that is not there."""
    with pytest.raises(world.WorldError) as caught:
        build(table, catalog, benchmarks=("livecodebench", "hle"))
    assert "livecodebench" in str(caught.value) and "hle" in str(caught.value)
    assert "dominator" in str(caught.value)


def test_a_malformed_table_is_refused_rather_than_defaulted(tmp_path, catalog):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hle": {"c_per_task": 0.002}}), encoding="utf-8")
    with pytest.raises(world.WorldError, match="not a usable row"):
        build(bad, catalog)
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(world.WorldError, match="not a published"):
        build(empty, catalog)


def test_a_corpus_where_every_benchmark_is_flagged_refuses_to_build(table, catalog):
    """§5.1b: a flagged benchmark is EXCLUDED from scoring, so a corpus of only flagged benchmarks
    makes `final` a mean over nothing and every duel compares two empty scores. The section says to
    refuse to launch rather than score the arena on unpriced quality, and this is the live case —
    M3a flagged both admitted benchmarks as dominators on 2026-09-03."""
    with pytest.raises(world.WorldError, match="every benchmark in this corpus is excluded"):
        build(table, catalog, benchmarks=("hle",))


def test_the_refusal_names_which_guard_fired_on_which_benchmark(table, catalog):
    """§5.1b requires an exclusion to be ANNOUNCED. A refusal that did not say which benchmark or
    why would send the owner to look for a configuration bug that is not there."""
    with pytest.raises(world.WorldError) as caught:
        build(table, catalog, benchmarks=("hle",))
    assert "hle" in str(caught.value) and "dominator" in str(caught.value)



# --- the table carries the pool it priced, and the daemon runs on that pool --------------------------


def test_the_daemon_runs_king_zero_on_the_pool_lambda_was_fitted_on(table, catalog):
    """MEASURED 2026-09-07, first live launch: the table was fitted over the five-model probe pool,
    `pins` was handed today's whole catalog, and King0's cascade climbed to `openai/gpt-4` — a top
    rung the price had never seen, at twenty times the per-call cost of the one it had. λ prices
    spend at one pool's slope; the arms it prices have to climb that pool. So the catalog is
    narrowed to the table's `_pool` before any policy is built from it."""
    wide = Catalog(entries=catalog.entries + (entry("frontier/dearest", 40.0),))

    unpinned = build(table, wide)
    pinned = world.pins(exchange_path=table, catalog=wide, benchmarks=(), worker=object(),
                        tasks=tasks("livecodebench", "r2egym"),
                        pool=[e.model_id for e in catalog.entries])

    assert unpinned.king_zero.rungs[-1] == "frontier/dearest"          # what the launch did
    assert pinned.king_zero.rungs[-1] == "strong/three"                 # what the price is for
    assert "frontier/dearest" not in {e.model_id for e in pinned.world.catalog.entries}


def test_a_pinned_model_the_catalog_no_longer_carries_is_refused_by_name(table, catalog):
    """A substitute would change the ladder without changing the price. Refused, and the refusal
    names the model and the re-pin, rather than quietly serving a different pool at the old λ."""
    with pytest.raises(world.WorldError, match="fitted on a pool this catalog cannot supply.*gone/model"):
        world.pins(exchange_path=table, catalog=catalog, benchmarks=(), worker=object(),
                   tasks=tasks("livecodebench", "r2egym"), pool=["cheap/one", "gone/model"])


def test_load_pool_reads_the_record_and_load_exchange_does_not_mistake_it_for_a_benchmark(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(TABLE), encoding="utf-8")
    assert world.load_pool(bare) is None

    with_pool = tmp_path / "pooled.json"
    with_pool.write_text(json.dumps({**TABLE, "_pool": ["a/one", "b/two"]}), encoding="utf-8")
    assert world.load_pool(with_pool) == ("a/one", "b/two")
    assert set(world.load_exchange(with_pool)) == set(TABLE), "`_pool` is metadata, not a row"

    short = tmp_path / "short.json"
    short.write_text(json.dumps({**TABLE, "_pool": ["only/one"]}), encoding="utf-8")
    with pytest.raises(world.WorldError, match="at least two model ids"):
        world.load_pool(short)
