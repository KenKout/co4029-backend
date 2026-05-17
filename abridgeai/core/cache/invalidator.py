"""SQLAlchemy `after_flush` listener that drops cache keys touched by writes.

`INVALIDATION_RULES` maps a model class to the cache keys whose
contents may have been invalidated by a write. The listener walks every
flushed instance (new/dirty/deleted), looks up its model class in the
rules table, then issues `DEL` (or `SCAN MATCH ... DEL` for patterns
with unresolved placeholders) for the rendered key — interpolating
placeholders from the instance's mapped columns.

Phase 0.7 ships with `INVALIDATION_RULES = {}`. T1.x populates User /
AuthSession / UserRoleAssignment entries; T7.5.13 adds CardReview /
StudentCardState rules. Wiring is done here so downstream phases only
edit the rules dict.

Pattern-delete (T7.5.13)
------------------------
Rules whose key pattern contains placeholders that the model can't
resolve (e.g. ``LESSON_UNLOCK`` needs ``lesson_id`` but
``StudentCardState`` only knows ``question_id``) substitute the
unresolved slot with ``*`` and route through ``SCAN MATCH``. This lets
SR writes fan out across every cached lesson-unlock entry for the
affected student in one flush — the cache stays correct without
forcing the SR service to know about cache topology.

User-cascade strategy
---------------------
A `User` write also kills every active session for that user. The
listener consults the reverse-index SET `user_sessions:{user_id}` via
`SMEMBERS`, then `DEL`s every `session:{sid}` plus the index itself.
This avoids `SCAN MATCH session:*` (O(N) over the keyspace).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from redis.exceptions import RedisError
from sqlalchemy import event
from sqlalchemy.orm import Session, UOWTransaction

from .client import RedisFallbackError, get_cache
from .keys import (
    CARDS_DUE,
    KR_ESTIMATE,
    LESSON_UNLOCK,
    SESSION,
    USER_SESSIONS_INDEX,
    CacheKey,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

INVALIDATION_RULES: dict[type, list[CacheKey]] = {}

INVALIDATION_RULES_BY_TABLE: dict[str, list[CacheKey]] = {
    "card_reviews": [LESSON_UNLOCK, CARDS_DUE, KR_ESTIMATE],
    "student_card_state": [LESSON_UNLOCK, CARDS_DUE, KR_ESTIMATE],
    "lessons": [LESSON_UNLOCK],
}

USER_SESSION_CASCADE_MODELS: set[type] = set()

_registered = False

_pending_tasks: set[asyncio.Task[None]] = set()

# Cap pattern-delete fan-out per key to keep flush latency bounded. SCAN
# returns the cursor in batches, so we read at most ``_GLOB_SCAN_LIMIT``
# matches; anything beyond falls back to the next write or TTL expiry.
_GLOB_SCAN_LIMIT = 256
_GLOB_WILDCARD = "*"

# Cross-feature placeholder aliases. The SR feature stores ``student_id``
# while the cache key namespace uses ``user_id`` -- they refer to the
# same entity. Listing the alias here keeps the rules dict declarative
# and avoids per-feature interpolation hooks.
_PLACEHOLDER_ALIASES: dict[str, tuple[str, ...]] = {
    "user_id": ("student_id",),
    "student_id": ("user_id",),
}


async def drain_invalidations() -> None:
    """Await every in-flight invalidation task (test helper)."""
    if not _pending_tasks:
        return
    await asyncio.gather(*list(_pending_tasks), return_exceptions=True)


def _resolve_placeholder(name: str, instance: object) -> str | None:
    """Find a value for ``name`` on ``instance``, trying aliases.

    Order: literal name → alias names → ``<name without _id>`` →
    ``id``. Returns ``None`` when nothing resolves so the caller can
    decide whether to skip the rule or substitute a wildcard.
    """
    candidates: list[str] = [name]
    candidates.extend(_PLACEHOLDER_ALIASES.get(name, ()))
    if name.endswith("_id"):
        candidates.append(name.removesuffix("_id"))
        candidates.append("id")
    for cand in candidates:
        if hasattr(instance, cand):
            value = getattr(instance, cand)
            if value is not None:
                return str(value)
    return None


def _interpolate(key: CacheKey, instance: object) -> str | None:
    ctx: dict[str, str] = {}
    for name in key.placeholders:
        value = _resolve_placeholder(name, instance)
        if value is None:
            return None
        ctx[name] = value
    return key.format(**ctx)


def _interpolate_with_glob(key: CacheKey, instance: object) -> str | None:
    """Interpolate ``key`` substituting ``*`` for unresolved placeholders.

    Returns ``None`` only when no placeholder resolves at all -- a
    rule that hits nothing on the instance shouldn't fan out across
    the entire namespace.
    """
    ctx: dict[str, str] = {}
    resolved_any = False
    for name in key.placeholders:
        value = _resolve_placeholder(name, instance)
        if value is None:
            ctx[name] = _GLOB_WILDCARD
        else:
            resolved_any = True
            ctx[name] = value
    if not resolved_any:
        return None
    return key.format(**ctx)


def _gather_keys(instances: Iterable[object]) -> tuple[set[str], set[str]]:
    """Return ``(exact_keys, glob_patterns)`` to invalidate."""
    exact: set[str] = set()
    globs: set[str] = set()
    for obj in instances:
        rules = INVALIDATION_RULES.get(type(obj))
        if not rules:
            for cls, cls_rules in INVALIDATION_RULES.items():
                if isinstance(obj, cls):
                    rules = cls_rules
                    break
        if not rules:
            tablename = getattr(obj, "__tablename__", None)
            if isinstance(tablename, str):
                rules = INVALIDATION_RULES_BY_TABLE.get(tablename)
        if not rules:
            continue
        for rule in rules:
            rendered = _interpolate(rule, obj)
            if rendered is not None:
                exact.add(rendered)
                continue
            glob = _interpolate_with_glob(rule, obj)
            if glob is not None and _GLOB_WILDCARD in glob:
                globs.add(glob)
    return exact, globs


def _gather_user_cascades(instances: Iterable[object]) -> set[str]:
    user_ids: set[str] = set()
    for obj in instances:
        cls = type(obj)
        if cls in USER_SESSION_CASCADE_MODELS or any(
            isinstance(obj, c) for c in USER_SESSION_CASCADE_MODELS
        ):
            uid = getattr(obj, "id", None) or getattr(obj, "user_id", None)
            if uid is not None:
                user_ids.add(str(uid))
    return user_ids


async def _expand_glob(client: Redis, pattern: str) -> list[str]:
    """Resolve a glob pattern via SCAN, capped at ``_GLOB_SCAN_LIMIT``."""
    matched: list[str] = []
    try:
        async for key in client.scan_iter(match=pattern, count=64):
            matched.append(key)
            if len(matched) >= _GLOB_SCAN_LIMIT:
                break
    except (RedisError, RedisFallbackError, OSError) as exc:
        logger.warning(
            "cache.scan_failed",
            extra={
                "event": "cache_scan_failed",
                "pattern": pattern,
                "err": repr(exc),
            },
        )
    return matched


async def _invalidate_async(
    keys: set[str],
    user_ids: set[str],
    glob_patterns: set[str],
) -> None:
    client: Redis = get_cache()
    try:
        for pattern in glob_patterns:
            for matched in await _expand_glob(client, pattern):
                keys.add(matched)
        if user_ids:
            for uid in user_ids:
                index_key = USER_SESSIONS_INDEX.format(user_id=uid)
                members: set[str] = await client.smembers(index_key)  # type: ignore[misc]
                for sid in members:
                    keys.add(SESSION.format(session_id=sid))
                keys.add(index_key)
        if keys:
            await client.delete(*keys)
    except (RedisError, RedisFallbackError, OSError) as exc:
        logger.warning(
            "cache.invalidate_failed",
            extra={
                "event": "cache_invalidate_failed",
                "keys": sorted(keys),
                "err": repr(exc),
            },
        )


def _schedule_invalidation(keys: set[str], user_ids: set[str], glob_patterns: set[str]) -> None:
    if not keys and not user_ids and not glob_patterns:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(_invalidate_async(keys, user_ids, glob_patterns))
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "cache.invalidate_failed",
                extra={"event": "cache_invalidate_failed", "err": repr(exc)},
            )
        return
    task = loop.create_task(_invalidate_async(keys, user_ids, glob_patterns))
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


def register_cache_invalidator() -> None:
    """Wire the global SQLAlchemy `after_flush` cache hook (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True

    @event.listens_for(Session, "after_flush")
    def _cache_after_flush(  # noqa: ARG001 — SQLAlchemy event signature
        session: Session, flush_context: UOWTransaction
    ) -> None:
        instances = list(session.new) + list(session.dirty) + list(session.deleted)
        keys, globs = _gather_keys(instances)
        user_ids = _gather_user_cascades(instances)
        _schedule_invalidation(keys, user_ids, globs)


__all__ = [
    "INVALIDATION_RULES",
    "USER_SESSION_CASCADE_MODELS",
    "drain_invalidations",
    "register_cache_invalidator",
]
