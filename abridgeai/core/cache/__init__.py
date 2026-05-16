"""Public cache surface — import from here, not submodules."""

from .client import RedisFallbackError, close_cache, get_cache, read_session_cache
from .decorators import cached
from .invalidator import (
    INVALIDATION_RULES,
    USER_SESSION_CASCADE_MODELS,
    drain_invalidations,
    register_cache_invalidator,
)
from .keys import (
    CARDS_DUE,
    COMPLIANCE,
    COURSE_CONTENT_PUBLISHED,
    KG_LESSON_CONCEPTS,
    KR_ESTIMATE,
    LESSON_UNLOCK,
    PERM_COURSE,
    PERM_USER,
    PRESIGNED_URL,
    SESSION,
    SESSION_BY_REFRESH_HASH,
    USER_SESSIONS_INDEX,
    USER_STATUS,
    CacheKey,
)

__all__ = [
    "CARDS_DUE",
    "COMPLIANCE",
    "COURSE_CONTENT_PUBLISHED",
    "CacheKey",
    "INVALIDATION_RULES",
    "KG_LESSON_CONCEPTS",
    "KR_ESTIMATE",
    "LESSON_UNLOCK",
    "PERM_COURSE",
    "PERM_USER",
    "PRESIGNED_URL",
    "RedisFallbackError",
    "SESSION",
    "SESSION_BY_REFRESH_HASH",
    "USER_SESSIONS_INDEX",
    "USER_SESSION_CASCADE_MODELS",
    "USER_STATUS",
    "cached",
    "close_cache",
    "drain_invalidations",
    "get_cache",
    "read_session_cache",
    "register_cache_invalidator",
]
