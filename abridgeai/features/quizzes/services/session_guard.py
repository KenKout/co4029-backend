"""Redis-backed ownership for a live quiz attempt.

The key is deliberately ephemeral: authentication session ownership protects
ordinary concurrent browsers without introducing a persistent lease model. All
read/compare/write behavior happens inside Redis Lua scripts so simultaneous
resume requests cannot both win.
"""

from __future__ import annotations

from uuid import UUID

from redis.exceptions import RedisError

from abridgeai.core.cache.client import RedisFallbackError, get_cache
from abridgeai.core.cache.keys import QUIZ_ATTEMPT_SESSION

SESSION_GUARD_TTL_SECONDS = QUIZ_ATTEMPT_SESSION.ttl_seconds

_CLAIM_SCRIPT = """
-- CLAIM_OR_RENEW: compare and set atomically.
local current = redis.call('GET', KEYS[1])
if not current then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2], 'NX')
  return 1
end
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
  return 1
end
return 0
"""

_TAKEOVER_SCRIPT = """
-- TAKEOVER: explicit user-confirmed replacement.
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
return 1
"""

_RELEASE_SCRIPT = """
-- RELEASE: compare and delete; stale owners cannot delete replacements.
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class QuizSessionGuardUnavailable(RuntimeError):  # noqa: N818 -- stable domain name
    """Redis could not evaluate a security-sensitive ownership operation."""


def session_key(attempt_id: UUID) -> str:
    """Return the central-namespace key for an attempt."""
    return QUIZ_ATTEMPT_SESSION.format(attempt_id=attempt_id)


async def _eval(script: str, attempt_id: UUID, session_id: UUID) -> bool:
    key = session_key(attempt_id)
    try:
        result = await get_cache().eval(
            script,
            1,
            key,
            str(session_id),
            str(SESSION_GUARD_TTL_SECONDS),
        )
    except (RedisError, RedisFallbackError, OSError, TimeoutError) as exc:
        raise QuizSessionGuardUnavailable("quiz session guard unavailable") from exc
    return bool(int(result or 0))


async def claim(attempt_id: UUID, session_id: UUID) -> bool:
    """Claim an unowned attempt or renew the caller's existing claim."""
    return await _eval(_CLAIM_SCRIPT, attempt_id, session_id)


async def validate_and_renew(attempt_id: UUID, session_id: UUID) -> bool:
    """Validate ownership and renew it, claiming an expired/missing key."""
    return await _eval(_CLAIM_SCRIPT, attempt_id, session_id)


async def takeover(attempt_id: UUID, session_id: UUID) -> bool:
    """Replace the current owner after an explicit takeover confirmation."""
    return await _eval(_TAKEOVER_SCRIPT, attempt_id, session_id)


async def release(attempt_id: UUID, session_id: UUID) -> bool:
    """Delete the key only when it still belongs to the caller."""
    return await _eval(_RELEASE_SCRIPT, attempt_id, session_id)


__all__ = [
    "QUIZ_ATTEMPT_SESSION",
    "SESSION_GUARD_TTL_SECONDS",
    "QuizSessionGuardUnavailable",
    "claim",
    "release",
    "session_key",
    "takeover",
    "validate_and_renew",
]
