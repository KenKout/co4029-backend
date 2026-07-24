"""Per-call cost computation, backed by the ``ai_model_pricing`` table.

Pricing used to be a hand-maintained Python dict requiring a release to
change. It now lives in the ``ai_model_pricing`` table (see migration
``0020_ai_model_pricing``) so admins can add/edit/remove model rates via
the admin UI without a deploy. Models absent from the table still record a
``NULL`` cost rather than guessing — honest is better than wrong.

Rates are cached in-process for ``_CACHE_TTL_SECONDS`` to avoid a DB
round-trip on every LLM/embedding call; the cache is refreshed lazily on
the first call after it expires. Admin pricing writes do not need to bust
this cache explicitly — a slightly stale rate for a few seconds is an
acceptable trade-off for not adding a pub/sub invalidation path.
"""

from __future__ import annotations

import time
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.models import AIModelPricing

_CACHE_TTL_SECONDS = 30.0

# module-level cache: model_name -> (input_per_1m, output_per_1m)
_cache: dict[str, tuple[Decimal, Decimal]] = {}
_cache_loaded_at: float = 0.0


async def _load_price_table(db: AsyncSession) -> dict[str, tuple[Decimal, Decimal]]:
    global _cache, _cache_loaded_at
    now = time.monotonic()
    if _cache and (now - _cache_loaded_at) < _CACHE_TTL_SECONDS:
        return _cache

    rows = (await db.execute(select(AIModelPricing))).scalars().all()
    _cache = {row.model_name: (row.input_usd_per_1m, row.output_usd_per_1m) for row in rows}
    _cache_loaded_at = now
    return _cache


def invalidate_pricing_cache() -> None:
    """Force the next :func:`compute_cost` call to re-read the DB.

    Called by the admin pricing CRUD service after create/update/delete so
    changes take effect immediately instead of waiting out the TTL.
    """
    global _cache_loaded_at
    _cache_loaded_at = 0.0


async def compute_cost(
    db: AsyncSession,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> Decimal | None:
    """Return the estimated USD cost for a single call, or ``None``.

    Returns ``None`` when:
      * the model has no row in ``ai_model_pricing`` (we refuse to guess), or
      * ``input_tokens`` is ``None`` (the upstream did not report usage).

    ``output_tokens=None`` is treated as ``0`` so embedding-shaped responses
    that omit the output side still produce a cost.
    """
    if input_tokens is None:
        return None
    price_table = await _load_price_table(db)
    entry = price_table.get(model)
    if entry is None:
        return None
    input_rate, output_rate = entry
    input_cost = input_rate * input_tokens / Decimal(1_000_000)
    output_cost = output_rate * (output_tokens or 0) / Decimal(1_000_000)
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Offline fallback for the eval harness (``eval/judges/judge.py``), which
# deliberately never opens a production ``AsyncSession`` (see that module's
# docstring — eval runs must not touch the audit DB). This static table is
# a frozen snapshot maintained separately from the admin-configurable
# ``ai_model_pricing`` DB table; keep it in sync manually if judge-model
# pricing drifts. Update the dated comment when you touch this table.
#
# Rates are USD per 1,000,000 tokens (per-1M convention, matching the DB
# table). Updated 2026-07-24 to per-1M — values are the prior per-1K rates
# ×1000, so per-call cost is unchanged.
# ---------------------------------------------------------------------------

# Updated: 2026-07-24 (per-1M convention; OpenAI list pricing).
_STATIC_PRICE_TABLE: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("5.00"), Decimal("15.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "text-embedding-3-small": (Decimal("0.02"), Decimal("0")),
    "text-embedding-3-large": (Decimal("0.13"), Decimal("0")),
}


def compute_cost_static(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> Decimal | None:
    """Sync, DB-free cost estimate for the eval harness only.

    Production call sites (``LLMGateway``, ``EmbeddingClient``, Whisper
    extraction) must use the async, DB-backed :func:`compute_cost` instead.
    """
    entry = _STATIC_PRICE_TABLE.get(model)
    if entry is None or input_tokens is None:
        return None
    input_rate, output_rate = entry
    input_cost = input_rate * input_tokens / Decimal(1000)
    output_cost = output_rate * (output_tokens or 0) / Decimal(1000)
    return input_cost + output_cost
