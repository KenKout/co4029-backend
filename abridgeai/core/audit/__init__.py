"""Audit auto-population infrastructure.

Public API:
- ``current_actor_var`` -- ContextVar carrying the active actor's UUID.
- ``get_current_actor()`` -- read accessor.
- ``register_audit_listener()`` -- wires SQLAlchemy ``before_flush`` (idempotent).

The HTTP-path actor bind lives in ``abridgeai.core.security.get_current_user``
(immediately after the principal is resolved). ARQ workers bind via
``set_worker_actor`` in ``abridgeai.workers.actor``.
"""

from .context import current_actor_var, get_current_actor
from .listener import register_audit_listener

__all__ = [
    "current_actor_var",
    "get_current_actor",
    "register_audit_listener",
]
