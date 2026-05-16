"""Application-level exceptions used across features.

These are domain exceptions thrown by the service layer. The HTTP error
mapping (404, 401, 403) lives in the API layer (T1.x routers / FastAPI
exception handler), keeping services free of HTTP awareness.
"""

from __future__ import annotations


class AppError(Exception):
    """Base for application errors. Maps to HTTP 500 unless caught by a
    more specific subclass handler."""


class NotFoundError(AppError):
    """The requested resource does not exist (or is hidden by soft-delete /
    scope filtering)."""


class UnauthorizedError(AppError):
    """The caller is not authenticated, or authentication is invalid (bad
    refresh token, bad MFA code). Distinct from :class:`ForbiddenError`,
    which is "authenticated but lacks permission"."""


class ForbiddenError(AppError):
    """The caller is authenticated but the current credential does not grant
    the requested action."""


__all__ = ["AppError", "ForbiddenError", "NotFoundError", "UnauthorizedError"]
