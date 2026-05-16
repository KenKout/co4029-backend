"""Async Redis client wrapper with singleton + graceful-fallback contract.

`get_cache()` returns a process-wide singleton matching the same shape as
`core.db.get_engine`. Construction is lazy so test code can monkeypatch
`core.config.get_settings` before the first call.

Operations raise `RedisFallbackError` when the underlying connection
fails. The `@cached` decorator catches this and falls through to the
source — callers should never catch it themselves; cache is meant to be
invisible failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from redis.asyncio import Redis
from redis.asyncio import from_url as redis_from_url
from redis.exceptions import RedisError

from abridgeai.core.config import get_settings

logger = logging.getLogger(__name__)


class RedisFallbackError(RuntimeError):
    """Raised when Redis is unreachable; signals fallback-to-source."""


_client: Redis | None = None

_SOCKET_CONNECT_TIMEOUT: Final = 0.5
_SOCKET_TIMEOUT: Final = 1.0


def get_cache() -> Redis:
    """Return the process-wide async Redis client (lazy singleton)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis_from_url(  # type: ignore[no-untyped-call]
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=_SOCKET_CONNECT_TIMEOUT,
            socket_timeout=_SOCKET_TIMEOUT,
            retry_on_timeout=False,
        )
    return _client


async def close_cache() -> None:
    """Tear down the cache connection pool (test/shutdown hook)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def reset_cache_client_for_tests() -> None:
    """Force re-creation of the singleton on next `get_cache()` call.

    Test-only; production code never resets the live client.
    """
    global _client
    _client = None


def wrap_redis_errors(exc: BaseException) -> RedisFallbackError:
    """Coerce a Redis-side exception into the fallback signal."""
    return RedisFallbackError(f"redis unavailable: {exc!r}")


async def read_session_cache(session_id: str) -> dict[str, Any] | None:
    """Read a cached session payload, logging WARN on a revoked HIT.

    A `revoked_at` value present in the cached payload means the
    invalidator failed to clear the entry — a security signal that
    callers should treat as cache-stale and re-check the database.
    """
    from .keys import SESSION

    client = get_cache()
    key = SESSION.format(session_id=session_id)
    try:
        raw = await client.get(key)
    except (RedisError, RedisFallbackError, OSError) as exc:
        logger.warning(
            "cache.get_failed",
            extra={"event": "cache_get_failed", "key": key, "err": repr(exc)},
        )
        return None
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict) and payload.get("revoked_at") is not None:
        logger.warning(
            "cache.revoked_session_hit",
            extra={
                "event": "cache_revoked_session_hit",
                "session_id": session_id,
                "revoked_at": payload.get("revoked_at"),
            },
        )
    return payload if isinstance(payload, dict) else None


__all__ = [
    "Redis",
    "RedisError",
    "RedisFallbackError",
    "close_cache",
    "get_cache",
    "read_session_cache",
    "reset_cache_client_for_tests",
    "wrap_redis_errors",
]
