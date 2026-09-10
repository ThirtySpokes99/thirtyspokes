"""The window's catalog: fetch it once, freeze it, and validate every ID against it (§6.2).

WHY A SNAPSHOT AND NOT A LIVE FETCH. OpenRouter's model list moves — models appear, disappear and
reprice through the day. King and challenger are evaluated minutes apart, so a live fetch would hand
them different action spaces and the duel would stop being paired: a challenger could win because a
model the king never saw came online between the two arms. §6.2 therefore makes the catalog part of
what the owner commits per window, alongside the slice, the task order and the decode params. This
module is the seam between the live list and that frozen record.

WHAT `validate` IS FOR, GIVEN `action.parse` ALREADY CHECKS MEMBERSHIP. It is the same check made a
second time, immediately before the worker call, so the §2.1 invariant is enforced *at the call
site* rather than inherited from a parser three modules away. A model ID is the only Conductor-made
value that reaches a provider at all, and the cost of re-checking it is a set lookup.
"""

from __future__ import annotations

import json
from collections.abc import Sequence, Mapping

from .types import Catalog, CatalogEntry

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class CatalogError(Exception):
    """A snapshot that cannot be an action space, or an ID that is not in one."""


def fetch(url: str = OPENROUTER_MODELS_URL,
          timeout: float = 30.0) -> dict:  # pragma: no cover — live seam
    """`GET /api/v1/models` -> the raw payload, for `snapshot` to freeze (§6.2).

    Kept as a bare fetch that returns the payload unchanged: the raw list is what the owner archives
    alongside the window record, so which rows `snapshot` dropped stays derivable afterwards rather
    than being a decision this process made and forgot.
    """
    import httpx  # noqa: PLC0415

    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def snapshot(payload: Mapping) -> Catalog:
    """The raw `/models` payload -> the window's frozen action space.

    ROWS THAT CANNOT BE PRICED ARE DROPPED, NOT REFUSED. OpenRouter reports `-1` for variable-priced
    endpoints (its own auto-router among them), and a row with no usable price is unusable twice
    over: `final_b` prices spend (§5.1b), so a call to it could not be scored, and the catalog is in
    the prompt precisely so a budget-aware policy can see prices (§3), which that row would not
    carry. Dropping one row costs the window nothing; refusing the whole payload over one would
    stall a window on a vendor's listing quirk.

    A DUPLICATE ID IS REFUSED. It makes the action space ambiguous — a validated ID must name
    exactly one model, or "the two arms faced the same catalog" stops meaning anything.

    Entries are sorted by ID so that two fetches returning the same models in a different order
    produce the same snapshot, and therefore the same cacheable prompt prefix (M0 exit 7). That
    makes the committed record verifiable by re-derivation rather than only by trust.
    """
    entries: dict[str, CatalogEntry] = {}
    for row in payload.get("data", ()):
        entry = _entry(row)
        if entry is None:
            continue
        if unusable(row, entry) is not None:
            continue
        if entry.model_id in entries:
            raise CatalogError(f"duplicate model id in catalog payload: {entry.model_id!r}")
        entries[entry.model_id] = entry
    return Catalog(entries=tuple(sorted(entries.values(), key=lambda e: e.model_id)))


# Endpoint variants that cannot serve a duel, whatever model sits behind them.
UNUSABLE_SUFFIXES = (":free", ":batch")


def unusable(row: Mapping, entry: CatalogEntry) -> str | None:
    """Why this row cannot be a rung, or None if it can. STRUCTURAL VIABILITY ONLY.

    D4 ("any OpenRouter model — no pinned pool") stands: this is not a curated allow-list, it is a
    refusal of endpoints that cannot complete an episode by construction. Every exclusion names a
    mechanism rather than a preference, and any model with a working endpoint keeps its place in the
    action space.

    MEASURED ON THE LIVE 415-ENTRY CATALOG, which is why this exists at all. Run against the
    unfiltered catalog, all three fixed reference policies select something broken:

        always_cheapest  -> 'cohere/north-mini-code:free'
        cascade rung 2   -> 'google/gemini-3.6-flash:batch'
        always_strongest -> 'openai/o1-pro'   ($150 / $600 per Mtok)

    * `:free` endpoints are rate-limited (order 20 req/min). A 250-task arm through one is a wall of
      429s, and because `openrouter.py` retries inside ONE wall-clock budget the arm dies on the
      clock — so the failure reads as a slow policy rather than an unusable rung. It also puts a $0
      floor under lambda's denominator (§5.1b), fitting the exchange rate between an arm that spent
      nothing and one that spent everything.
    * `:batch` variants are asynchronous with turnaround in hours, against
      `EPISODE_WALL_CLOCK_SECONDS = 900`. The rung can never return.
    * A model whose `supported_parameters` omits `temperature` has no qualifying provider under
      `openrouter.PROVIDER_ROUTING`'s `require_parameters: True` combined with
      `worker.WORKER_PARAMS = {"temperature": 0.0}`, so every call 503s. The whole OpenAI
      gpt-5/o-series family declares this, `openai/o1-pro` among them — which is how a $918 arm
      became the default "strongest".

    This is a correctness fix, not a cost one. The owner's two reference arms measure the band every
    window (§5.6), and `the build plan`'s cheap path rests entirely on them being ordinary operating cost.
    At o1-pro prices one reference arm is $48-220 per window, and that argument dies the first time
    a window opens.
    """
    if any(entry.model_id.endswith(s) for s in UNUSABLE_SUFFIXES):
        return "rate-limited or asynchronous endpoint variant"
    if entry.price_in_per_mtok <= 0.0 and entry.price_out_per_mtok <= 0.0:
        # The suffix check above misses `openrouter/free`, which carries the name rather than the
        # suffix. Price is the discriminator that catches both: a $0 endpoint is free-tier and
        # therefore rate-limited, and it puts a ZERO under lambda's denominator (§5.1b) — the
        # exchange rate would be fitted between an arm that spent nothing and one that spent
        # everything, which is a division by the thing the score is trying to measure.
        return "zero-priced (free tier): rate-limited, and a $0 floor under lambda's denominator"
    supported = row.get("supported_parameters")
    if isinstance(supported, (list, tuple)) and "temperature" not in supported:
        return "declares no temperature support; refused by require_parameters routing"
    out = ((row.get("architecture") or {}).get("output_modalities"))
    if isinstance(out, (list, tuple)) and list(out) != ["text"]:
        return f"emits {'+'.join(out)}, not text alone; a generation model cannot answer a task"
    return None


def narrow(catalog: Catalog, model_ids: Sequence[str]) -> Catalog:
    """Today's catalog restricted to a pinned list of models, refusing rather than substituting.

    `snapshot` has already dropped what is structurally unusable (`:free`, `:batch`, no
    `temperature`, unpriceable). This is the deliberate narrowing on top of it — and it is the
    narrowing λ was FITTED ON. Measured 2026-09-07 on the first live launch: the table was fitted
    over the five-model probe pool, `world.from_env` handed the daemon the whole catalog, and King₀'s
    cascade climbed to `openai/gpt-4` as its top rung — a ladder the table had never priced, at
    twenty times the per-call cost of the one it had. λ prices spend at one pool's slope; the arms
    it prices must run on that pool. A pinned model missing from today's catalog is refused by name,
    because a substitute would change the ladder without changing the price.
    """
    present = {entry.model_id: entry for entry in catalog.entries}
    missing = [model_id for model_id in model_ids if model_id not in present]
    if missing:
        raise CatalogError(
            f"pinned models absent from today's usable catalog: {missing}. A substitute would "
            f"change the ladder the lambda table prices; re-pin the pool (the spend plan "
            f"§4.2) and re-run M3a rather than serving a different pool at the old price.")
    if len(set(model_ids)) < 2:
        raise CatalogError(f"a pool needs at least two models to be a ladder, not {list(model_ids)}")
    return Catalog(entries=tuple(sorted((present[m] for m in dict.fromkeys(model_ids)),
                                        key=lambda e: e.price_in_per_mtok + e.price_out_per_mtok)))


def freeze(catalog: Catalog) -> str:
    """The catalog as the window record stores it: canonical JSON, so it hashes stably (§6.3).

    Explicit field names rather than positional rows: the record outlives this module, and a reader
    a year from now must not have to know the field order of a dataclass to read what the arms were
    allowed to call.
    """
    return json.dumps([{"model_id": e.model_id,
                        "price_in_per_mtok": e.price_in_per_mtok,
                        "price_out_per_mtok": e.price_out_per_mtok,
                        "context_length": e.context_length}
                       for e in catalog.entries],
                      sort_keys=True, separators=(",", ":"))


def thaw(text: str) -> Catalog:
    """The frozen record -> the catalog both arms run against. Round-trips `freeze` exactly."""
    return Catalog(entries=tuple(
        CatalogEntry(model_id=row["model_id"],
                     price_in_per_mtok=row["price_in_per_mtok"],
                     price_out_per_mtok=row["price_out_per_mtok"],
                     context_length=row["context_length"])
        for row in json.loads(text)))


def validate(catalog: Catalog, model_id: str) -> str:
    """The ID, if this window's snapshot contains it exactly. Otherwise `CatalogError`.

    Exact membership, never a prefix or fuzzy match (`action.py`'s reasoning applies verbatim): a
    repaired near-miss would send the task somewhere the Conductor did not choose, and the arm would
    be scored for a policy nobody submitted.
    """
    if model_id not in catalog.ids:
        raise CatalogError(f"{model_id!r} is not in this window's catalog snapshot")
    return model_id


def _entry(row: object) -> CatalogEntry | None:
    """One `/models` row -> a catalog entry, or None if it cannot be an action.

    Prices are converted from OpenRouter's per-token strings to the dollars-per-million-tokens the
    catalog and the prompt speak in, because per-token prices render as `1e-06` — a number no policy
    can reason about and a format that is all rounding noise.
    """
    if not isinstance(row, Mapping):
        return None
    model_id = row.get("id")
    pricing = row.get("pricing")
    if not isinstance(model_id, str) or not isinstance(pricing, Mapping):
        return None
    price_in = _price(pricing.get("prompt"))
    price_out = _price(pricing.get("completion"))
    context_length = row.get("context_length")
    if price_in is None or price_out is None:
        return None
    if not isinstance(context_length, int) or context_length <= 0:
        return None
    return CatalogEntry(model_id=model_id, price_in_per_mtok=price_in,
                        price_out_per_mtok=price_out, context_length=context_length)


def _price(raw: object) -> float | None:
    """Dollars per million tokens, or None for a price that is absent, unparseable or negative."""
    try:
        price = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return price * 1e6 if price >= 0.0 else None
