"""Starlette middleware that binds the current request's actor to the audit context.

Reads `request.state.user` (set by the auth dependency `get_current_user`
elsewhere in the app). For unauthenticated requests, or requests handled
before the auth dep runs (e.g. early middleware errors), the actor stays
None and the audit listener falls through to the system-NULL path.

Uses `current_actor_var.set(...)` + `reset(token)` to guarantee clean
teardown per request, so concurrent requests cannot leak actor identity
across each other.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .context import current_actor_var


class AuditContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        actor_id = None
        user = getattr(request.state, "user", None)
        if user is not None:
            actor_id = getattr(user, "id", None)
        token = current_actor_var.set(actor_id)
        try:
            return await call_next(request)
        finally:
            current_actor_var.reset(token)


__all__ = ["AuditContextMiddleware"]
