"""Audit auto-population infrastructure.

Public API:
- ``current_actor_var`` -- ContextVar carrying the active actor's UUID.
- ``get_current_actor()`` -- read accessor.
- ``register_audit_listener()`` -- wires SQLAlchemy ``before_flush`` (idempotent).
- ``audit_maintenance()`` -- opt into deleting append-only audit rows
  (retention only; migration 0105 blocks the DELETE without it).

The HTTP-path actor bind lives in ``abridgeai.core.security.get_current_user``
(immediately after the principal is resolved). ARQ workers bind via
``set_worker_actor`` in ``abridgeai.workers.actor``.
"""

from .context import current_actor_var, get_current_actor
from .listener import register_audit_listener
from .maintenance import audit_maintenance

__all__ = [
    "audit_maintenance",
    "current_actor_var",
    "get_current_actor",
    "register_audit_listener",
]
