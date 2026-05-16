"""SQLAlchemy `before_flush` listener that populates audit columns.

For every flush against any `Session`:
- New instances of `AuditedByMixin` get `created_by` AND `updated_by`
  filled from `current_actor_var` (only if not already set explicitly).
- Dirty instances of `AuditedByMixin` get `updated_by` overwritten with
  the current actor.

If `current_actor_var` is unset (None), the listener is a no-op — audit
columns stay NULL. This is the system-operation path (background jobs,
migrations, fixture seeding without an attributed caller).

Registration is idempotent: calling `register_audit_listener()` more than
once (e.g. test harness re-imports) does not double-fire the hook.
"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.orm import Session

from .context import current_actor_var

_registered = False


def register_audit_listener() -> None:
    """Wire the global `before_flush` audit hook (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True

    from ..db.mixins import AuditedByMixin

    @event.listens_for(Session, "before_flush")
    def _audit_before_flush(session, flush_context, instances) -> None:  # noqa: ARG001
        actor_id = current_actor_var.get()
        if actor_id is None:
            return
        for obj in session.new:
            if isinstance(obj, AuditedByMixin):
                if obj.created_by is None:
                    obj.created_by = actor_id
                if obj.updated_by is None:
                    obj.updated_by = actor_id
        for obj in session.dirty:
            if isinstance(obj, AuditedByMixin) and session.is_modified(obj):
                obj.updated_by = actor_id


__all__ = ["register_audit_listener"]
