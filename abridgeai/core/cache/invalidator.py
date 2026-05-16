"""SQLAlchemy `after_flush` listener that drops cache keys touched by writes.

`INVALIDATION_RULES` maps a model class to the cache keys whose
contents may have been invalidated by a write. The listener walks every
flushed instance (new/dirty/deleted), looks up its model class in the
rules table, then issues `DEL` for the rendered key — interpolating
placeholders from the instance's mapped columns.

Phase 0.7 ships with `INVALIDATION_RULES = {}`. T1.x populates User /
AuthSession / UserRoleAssignment entries; T7.5.13 adds CardReview /
StudentCardState rules. Wiring is done here so downstream phases only
edit the rules dict.

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
from .keys import SESSION, USER_SESSIONS_INDEX, CacheKey

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

INVALIDATION_RULES: dict[type, list[CacheKey]] = {}

USER_SESSION_CASCADE_MODELS: set[type] = set()

_registered = False

_pending_tasks: set[asyncio.Task[None]] = set()


async def drain_invalidations() -> None:
    """Await every in-flight invalidation task (test helper)."""
    if not _pending_tasks:
        return
    await asyncio.gather(*list(_pending_tasks), return_exceptions=True)


def _interpolate(key: CacheKey, instance: object) -> str | None:
    ctx: dict[str, str] = {}
    for name in key.placeholders:
        candidates: list[str] = [name]
        if name.endswith("_id"):
            candidates.append(name.removesuffix("_id"))
            candidates.append("id")
        value: object | None = None
        for cand in candidates:
            if hasattr(instance, cand):
                value = getattr(instance, cand)
                if value is not None:
                    break
        if value is None:
            return None
        ctx[name] = str(value)
    return key.format(**ctx)


def _gather_keys(instances: Iterable[object]) -> set[str]:
    keys: set[str] = set()
    for obj in instances:
        rules = INVALIDATION_RULES.get(type(obj))
        if not rules:
            for cls, cls_rules in INVALIDATION_RULES.items():
                if isinstance(obj, cls):
                    rules = cls_rules
                    break
        if not rules:
            continue
        for rule in rules:
            rendered = _interpolate(rule, obj)
            if rendered is not None:
                keys.add(rendered)
    return keys


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


async def _invalidate_async(keys: set[str], user_ids: set[str]) -> None:
    client: Redis = get_cache()
    try:
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


def _schedule_invalidation(keys: set[str], user_ids: set[str]) -> None:
    if not keys and not user_ids:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(_invalidate_async(keys, user_ids))
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "cache.invalidate_failed",
                extra={"event": "cache_invalidate_failed", "err": repr(exc)},
            )
        return
    task = loop.create_task(_invalidate_async(keys, user_ids))
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
        keys = _gather_keys(instances)
        user_ids = _gather_user_cascades(instances)
        _schedule_invalidation(keys, user_ids)


__all__ = [
    "INVALIDATION_RULES",
    "USER_SESSION_CASCADE_MODELS",
    "drain_invalidations",
    "register_cache_invalidator",
]
