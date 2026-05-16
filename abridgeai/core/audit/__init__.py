"""Audit auto-population infrastructure.

Public API:
- `current_actor_var` — ContextVar carrying the active actor's UUID.
- `get_current_actor()` — read accessor.
- `AuditContextMiddleware` — HTTP path actor binder.
- `register_audit_listener()` — wires SQLAlchemy `before_flush` (idempotent).
- `set_worker_actor()` — ARQ worker path actor setter (re-export from workers).
"""

from .context import current_actor_var, get_current_actor
from .listener import register_audit_listener
from .middleware import AuditContextMiddleware

__all__ = [
    "AuditContextMiddleware",
    "current_actor_var",
    "get_current_actor",
    "register_audit_listener",
]
