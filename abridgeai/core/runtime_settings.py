"""Resolve runtime settings for one organization.

Precedence, highest first:

1. a ``system_settings`` row for this organization
2. a ``system_settings`` row with ``organization_id IS NULL`` (the deployment default)
3. the environment variable named by the spec
4. the spec's ``default``

Levels 3 and 4 are what make adding a setting safe: a deployment that has never
opened the admin page keeps behaving exactly as it did, because the resolver
falls through to the same value the code used to hard-code.

Raw SQL rather than the ORM. This module sits in ``core`` so that ``ai`` and
every feature can call it without tripping the import-linter's cross-feature
independence contract; importing a feature-owned model here would invert the
layering. It is the same reasoning that makes
``ai/knowledge_graph/tenancy.py`` hand-write its query.

Caching
-------
One row set per organization is cached in-process for a short TTL, because the
ingestion pipeline resolves settings once per document and the admin surface
changes them rarely. Writes call :func:`invalidate_settings_cache` so an edit
is visible on the next ingest instead of after the TTL, mirroring
``ai/llm/pricing.py``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.settings_registry import SETTINGS_REGISTRY, SettingSpec

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 30.0

# (org_id or None) -> {setting_key: json value}
_cache: dict[str | None, dict[str, Any]] = {}
_cache_loaded_at: dict[str | None, float] = {}

_LOAD_SQL = text(
    """
    SELECT setting_key, setting_value_json, organization_id
    FROM system_settings
    WHERE organization_id IS NULL
       OR organization_id = CAST(:organization_id AS uuid)
    """
)


def invalidate_settings_cache() -> None:
    """Force the next resolve to re-read the table."""
    _cache.clear()
    _cache_loaded_at.clear()


def _env_override(spec: SettingSpec) -> bool | int | float | None:
    """Read the spec's environment variable, or ``None`` if unset/unparseable.

    A malformed value is ignored rather than fatal: an operator's typo in
    ``.env`` should not take ingestion down, and the log line plus the fall
    through to the default makes the mistake visible without an outage.
    """
    if spec.env_var is None:
        return None
    raw = os.environ.get(spec.env_var)
    if raw is None or not raw.strip():
        return None
    text_value = raw.strip()
    try:
        if spec.type == "bool":
            lowered = text_value.lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(text_value)
        if spec.type == "int":
            return int(text_value)
        return float(text_value)
    except ValueError:
        logger.warning(
            "runtime settings: %s=%r is not a valid %s; falling back to the default",
            spec.env_var,
            raw,
            spec.type,
        )
        return None


async def _load_rows(db: AsyncSession, organization_id: UUID | None) -> dict[str, Any]:
    """Return ``{key: value}`` for one org, org rows shadowing global rows."""
    cache_key = str(organization_id) if organization_id else None
    now = time.monotonic()
    if cache_key in _cache and (now - _cache_loaded_at.get(cache_key, 0.0)) < _CACHE_TTL_SECONDS:
        return _cache[cache_key]

    result = await db.execute(
        _LOAD_SQL, {"organization_id": str(organization_id) if organization_id else None}
    )
    globals_: dict[str, Any] = {}
    org_values: dict[str, Any] = {}
    for row in result.mappings():
        target = globals_ if row["organization_id"] is None else org_values
        target[row["setting_key"]] = row["setting_value_json"]

    merged = {**globals_, **org_values}
    _cache[cache_key] = merged
    _cache_loaded_at[cache_key] = now
    return merged


async def resolve_settings(
    db: AsyncSession,
    organization_id: UUID | None = None,
) -> dict[str, bool | int | float]:
    """Every registered setting, resolved for ``organization_id``.

    Returns the full registry — callers get a complete mapping and never have
    to handle a missing key. A stored value that no longer satisfies its spec
    (bounds tightened in a later release, say) is discarded in favour of the
    next level down rather than propagated.

    Never raises on a database error: an unreachable settings table degrades to
    environment and defaults, which is the behaviour that existed before this
    module. Failing an ingest because a tuning knob could not be read would
    trade a real capability for a cosmetic one.
    """
    try:
        stored = await _load_rows(db, organization_id)
    except Exception:  # noqa: BLE001 -- settings are tuning; never fail the caller
        logger.warning("runtime settings: load failed; using env + defaults", exc_info=True)
        stored = {}

    from abridgeai.core.settings_registry import (  # noqa: PLC0415 -- avoid cycle at import time
        SettingValidationError,
        coerce_and_validate,
    )

    resolved: dict[str, bool | int | float] = {}
    for key, spec in SETTINGS_REGISTRY.items():
        if key in stored:
            try:
                resolved[key] = coerce_and_validate(key, stored[key])
                continue
            except SettingValidationError:
                logger.warning(
                    "runtime settings: stored value for %s is invalid for the current "
                    "spec; falling through to env/default",
                    key,
                )
        env_value = _env_override(spec)
        resolved[key] = env_value if env_value is not None else spec.default
    return resolved


async def resolve_setting(
    db: AsyncSession,
    key: str,
    organization_id: UUID | None = None,
) -> bool | int | float:
    """Single-key convenience wrapper over :func:`resolve_settings`."""
    return (await resolve_settings(db, organization_id))[key]


__all__ = [
    "invalidate_settings_cache",
    "resolve_setting",
    "resolve_settings",
]
