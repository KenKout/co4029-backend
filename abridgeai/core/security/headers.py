"""OWASP-recommended security headers middleware (T0.28).

Adds the standard browser-side defenses (HSTS, X-Content-Type-Options,
X-Frame-Options, Referrer-Policy, Permissions-Policy, Content-Security-Policy)
to every HTTP response. The CSP is environment-aware: a strict policy is
applied in production; a relaxed variant is applied outside production so
the FastAPI Swagger UI (which inlines scripts/styles and pulls assets from
``cdn.jsdelivr.net``) still functions during local development.

Mounted in :mod:`abridgeai.api` BEFORE the audit-log middleware so the
audit trail records the final, headers-decorated response.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from starlette.types import ASGIApp


_SECURITY_HEADERS: dict[str, str] = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

# Strict CSP for production: no inline scripts, no eval, no third-party CDNs.
_CSP_PROD = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "frame-ancestors 'none'"
)

# Relaxed CSP for non-production: allows Swagger UI (inline + jsdelivr CDN).
_CSP_DEV = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://cdn.jsdelivr.net; "
    "frame-ancestors 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds OWASP-recommended security headers to every response."""

    def __init__(self, app: ASGIApp, *, environment: str = "production") -> None:
        super().__init__(app)
        self._csp = _CSP_DEV if environment != "production" else _CSP_PROD

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        response.headers.setdefault("Content-Security-Policy", self._csp)
        return response


__all__ = ["SecurityHeadersMiddleware"]
