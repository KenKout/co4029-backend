"""Explicit read-through primitives for callers the `@cached` decorator can't serve.

`@cached` keys off the wrapped function's arguments and hands back whatever
`json.loads` produced. That is the right tool for a dict-returning query. It is
the wrong tool when the caller needs to

* re-validate the payload into a typed DTO (a Pydantic model, not a bare dict);
* decide *not* to cache a particular result (a ``None`` lookup must not pin a
  404 for the whole TTL — publishing a course would then take effect late);
* compute the TTL at call time (e.g. clamp it below a presigned URL's expiry).

These helpers keep the failure contract identical to the decorator: every Redis
error is swallowed and logged, and the caller proceeds against the source of
truth. A cache outage costs latency, never correctness.

Each call carries a ``namespace`` (the `CacheKey.pattern`, not the rendered
key) so hit/miss/failure counters can be aggregated per cache rather than per
entity id.
"""

from __future__ import annotations

import json
import logging
from typing import Final

from redis.exceptions import RedisError

from .client import RedisFallbackError, get_cache

logger = logging.getLogger(__name__)

_REDIS_FAILURES: Final = (RedisError, RedisFallbackError, OSError)


async def get_json(key: str, *, namespace: str) -> object | None:
    """Return the decoded payload at ``key``, or ``None`` on miss/failure.

    A decode failure is treated as a miss and logged: a payload written by an
    older DTO shape must not break the read path.
    """
    client = get_cache()
    try:
        raw = await client.get(key)
    except _REDIS_FAILURES as exc:
        logger.warning(
            "cache.get_failed",
            extra={
                "event": "cache_get_failed",
                "namespace": namespace,
                "key": key,
                "err": repr(exc),
            },
        )
        return None

    if raw is None:
        logger.debug(
            "cache.miss",
            extra={"event": "cache_miss", "namespace": namespace, "key": key},
        )
        return None

    try:
        payload: object = json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "cache.decode_failed",
            extra={
                "event": "cache_decode_failed",
                "namespace": namespace,
                "key": key,
                "err": repr(exc),
            },
        )
        return None

    logger.debug(
        "cache.hit",
        extra={
            "event": "cache_hit",
            "namespace": namespace,
            "key": key,
            "bytes": len(raw),
        },
    )
    return payload


async def set_json(key: str, value: object, *, ttl: int, namespace: str) -> bool:
    """Write ``value`` as JSON under ``key``; ``False`` when the write failed.

    The return value exists for tests and metrics — callers MUST NOT branch on
    it, because a failed cache write is not a failed request.
    """
    try:
        encoded = json.dumps(value, default=str)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "cache.encode_failed",
            extra={
                "event": "cache_encode_failed",
                "namespace": namespace,
                "key": key,
                "err": repr(exc),
            },
        )
        return False

    client = get_cache()
    try:
        await client.set(key, encoded, ex=ttl)
    except _REDIS_FAILURES as exc:
        logger.warning(
            "cache.set_failed",
            extra={
                "event": "cache_set_failed",
                "namespace": namespace,
                "key": key,
                "err": repr(exc),
            },
        )
        return False

    logger.debug(
        "cache.stored",
        extra={
            "event": "cache_stored",
            "namespace": namespace,
            "key": key,
            "ttl": ttl,
            "bytes": len(encoded),
        },
    )
    return True


__all__ = ["get_json", "set_json"]
