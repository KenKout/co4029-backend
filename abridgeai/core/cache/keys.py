"""Centralised cache-key namespace.

Every cacheable concept declares a `CacheKey` constant here so the key
pattern, TTL, and human-readable purpose live in one place. Code paths
SHOULD NOT hard-code key strings — always reference a constant.

Conventions
-----------
* Pattern uses `str.format` placeholders (``"perm:user:{user_id}"``).
* TTL is the safety-net upper bound. Auto-invalidation (T0.29 listener)
  removes the entry on relevant DB writes; TTL only kicks in if Redis
  pub/sub or the listener fails.
* Security-critical keys (``SESSION``, ``SESSION_BY_REFRESH_HASH``,
  ``USER_STATUS``) keep TTL ≤ 60 s — a stolen-token blast-radius cap.

User → sessions reverse index
-----------------------------
``user_sessions:{user_id}`` is a Redis SET storing every active session
id for that user. Maintained by:

* ``cache.add_user_session(user_id, session_id)`` → ``SADD`` on session create.
* ``cache.remove_user_session(user_id, session_id)`` → ``SREM`` on revoke.

User-level invalidation cascade (e.g. admin disables a user) does:

* ``SMEMBERS user_sessions:{user_id}`` → list of session_ids.
* ``DEL session:{sid}`` for each.
* ``DEL user_sessions:{user_id}`` (the index itself).

This avoids ``SCAN MATCH session:*`` (O(N) over all keys, unsafe at scale).
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Typed cache-key descriptor; call :meth:`format` to materialise."""

    pattern: str
    ttl_seconds: int
    description: str

    def format(self, **ctx: object) -> str:
        """Render the key. Missing placeholder raises KeyError (fail loud)."""
        return self.pattern.format(**ctx)

    @property
    def placeholders(self) -> tuple[str, ...]:
        names: list[str] = []
        for _, field, _, _ in string.Formatter().parse(self.pattern):
            if field is not None and field != "":
                names.append(field)
        return tuple(names)


PERM_USER: Final = CacheKey(
    pattern="perm:user:{user_id}",
    ttl_seconds=300,
    description="Effective permissions for a user (any course/scope).",
)

PERM_COURSE: Final = CacheKey(
    pattern="perm:course:{user_id}:{course_id}",
    ttl_seconds=300,
    description="Effective permissions for a (user, course) pair.",
)

# ---------------------------------------------------------------------------
# Authoring & content read-through cache.
# ---------------------------------------------------------------------------

LESSON_UNLOCK: Final = CacheKey(
    pattern="lesson_unlock:{user_id}:{lesson_id}",
    ttl_seconds=60,
    description="Whether a lesson is unlocked for a learner (SR engine).",
)

COURSE_CONTENT_PUBLISHED: Final = CacheKey(
    pattern="course_content:published:{course_id}",
    ttl_seconds=300,
    description="Published course tree (modules/lessons/items) for learner reads.",
)

CARDS_DUE: Final = CacheKey(
    pattern="cards_due:{user_id}",
    ttl_seconds=60,
    description="Spaced-repetition cards due for a user (SR queue).",
)

KG_LESSON_CONCEPTS: Final = CacheKey(
    pattern="kg:lesson:{lesson_id}",
    ttl_seconds=1800,
    description="Knowledge-graph concept set for a lesson (Neo4j projection).",
)

PRESIGNED_URL: Final = CacheKey(
    pattern="presigned:{material_version_id}",
    ttl_seconds=3000,
    description="S3 presigned download URL (cached < URL TTL of 3600s).",
)

KR_ESTIMATE: Final = CacheKey(
    pattern="kr:{user_id}:{lesson_id}",
    ttl_seconds=300,
    description="Knowledge-readiness estimate (lesson, user).",
)

COMPLIANCE: Final = CacheKey(
    pattern="compliance:{user_id}:{lesson_id}",
    ttl_seconds=3600,
    description="Compliance-mode lesson completion state.",
)

# ---------------------------------------------------------------------------
# Identity & session — security-critical: short TTL + immediate invalidation.
# ---------------------------------------------------------------------------

SESSION: Final = CacheKey(
    pattern="session:{session_id}",
    ttl_seconds=60,
    description="Cached AuthSession state for JWT validation hot path.",
)

SESSION_BY_REFRESH_HASH: Final = CacheKey(
    pattern="session:refresh:{refresh_hash}",
    ttl_seconds=60,
    description="Refresh-token hash → session_id lookup.",
)

USER_STATUS: Final = CacheKey(
    pattern="user:status:{user_id}",
    ttl_seconds=60,
    description="Per-user account status (active/disabled).",
)

USER_SESSIONS_INDEX: Final = CacheKey(
    pattern="user_sessions:{user_id}",
    ttl_seconds=24 * 60 * 60,  # bound; refreshed on each SADD
    description="Reverse-index SET of session_ids owned by a user (cascade).",
)


__all__ = [
    "CARDS_DUE",
    "COMPLIANCE",
    "COURSE_CONTENT_PUBLISHED",
    "CacheKey",
    "KG_LESSON_CONCEPTS",
    "KR_ESTIMATE",
    "LESSON_UNLOCK",
    "PERM_COURSE",
    "PERM_USER",
    "PRESIGNED_URL",
    "SESSION",
    "SESSION_BY_REFRESH_HASH",
    "USER_SESSIONS_INDEX",
    "USER_STATUS",
]
