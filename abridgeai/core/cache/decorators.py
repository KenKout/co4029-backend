"""`@cached` decorator factory: read-through async Redis with safe fallback."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast

from redis.exceptions import RedisError

from .client import RedisFallbackError, get_cache
from .keys import CacheKey

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _build_key(key: CacheKey, bound: dict[str, Any]) -> str:
    ctx = {name: bound[name] for name in key.placeholders if name in bound}
    missing = [n for n in key.placeholders if n not in ctx]
    if missing:
        raise KeyError(
            f"@cached({key.pattern!r}): missing placeholder(s) "
            f"{missing!r} in call args; available={list(bound)}"
        )
    return key.format(**{k: str(v) for k, v in ctx.items()})


def _bind_args(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def cached(
    key: CacheKey,
    *,
    ttl: int | None = None,
    bypass: bool = False,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Wrap an async function with read-through Redis caching.

    Args:
        key: A `CacheKey` constant; placeholders resolved from call kwargs.
        ttl: Override `key.ttl_seconds` (rarely needed).
        bypass: Skip cache entirely (testing/debug).

    Cache miss or any Redis failure falls through to the source function;
    the decorator never blocks the caller on cache problems.
    """

    effective_ttl = ttl if ttl is not None else key.ttl_seconds

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if bypass:
                return await func(*args, **kwargs)

            try:
                bound = _bind_args(func, args, kwargs)
                cache_key = _build_key(key, bound)
            except KeyError:
                raise

            client = get_cache()

            try:
                raw = await client.get(cache_key)
            except (RedisError, RedisFallbackError, OSError) as exc:
                logger.warning(
                    "cache.get_failed",
                    extra={"event": "cache_get_failed", "key": cache_key, "err": repr(exc)},
                )
                return await func(*args, **kwargs)

            if raw is not None:
                try:
                    cached_value = json.loads(raw)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "cache.decode_failed",
                        extra={
                            "event": "cache_decode_failed",
                            "key": cache_key,
                            "err": repr(exc),
                        },
                    )
                else:
                    logger.debug(
                        "cache.hit",
                        extra={"event": "cache_hit", "key": cache_key},
                    )
                    return cast(R, cached_value)

            logger.debug(
                "cache.miss",
                extra={"event": "cache_miss", "key": cache_key},
            )
            value = await func(*args, **kwargs)

            try:
                await client.set(cache_key, json.dumps(value, default=str), ex=effective_ttl)
            except (RedisError, RedisFallbackError, OSError, TypeError) as exc:
                logger.warning(
                    "cache.set_failed",
                    extra={"event": "cache_set_failed", "key": cache_key, "err": repr(exc)},
                )

            return value

        return wrapper

    return decorator


__all__ = ["cached"]
