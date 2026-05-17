"""Guard against accidental hard-deletes of SoftDeleteMixin rows.

The orchestration contract for soft-deletable entities is:

* WRITE path: callers must use ``abridgeai.core.db.recursive_delete.soft_delete_cascade``
  to set ``deleted_at``/``deleted_by``. The recursive walker handles ONETOMANY
  descent and audit-trail stamping in one transactional unit.
* The READ-side filter (``core/db/soft_delete.py``) hides tombstoned rows from
  ORM queries automatically.

This module installs a ``before_flush`` listener that intercepts any
``Session.delete()`` call targeting a ``SoftDeleteMixin`` instance and raises
``RuntimeError`` rather than letting SQLAlchemy issue a physical SQL DELETE.

Why raise rather than silently rewrite to soft-delete:

1. ``soft_delete_cascade`` requires an actor (``actor_id`` for ``deleted_by``);
   the contextvar fallback works but is invisible at the call site. Failing
   loud forces callers to be explicit.
2. The cascade walk is async (``await session.refresh(...)``); rewriting
   inside a sync ``before_flush`` hook would require running an event loop
   or queueing — fragile.
3. Raising surfaces bugs at the point of mistake. Silent rewrite hides them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import event

from abridgeai.core.db.mixins import SoftDeleteMixin

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, UOWTransaction


def register_hard_delete_guard(session_class: type[Session]) -> None:
    """Install before_flush listener that rejects hard-delete of SoftDeleteMixin rows."""

    @event.listens_for(session_class, "before_flush")
    def _reject_hard_delete_of_softdelete(  # noqa: ARG001  -- SQLAlchemy event signature
        session: Session,
        flush_context: UOWTransaction,
        instances: object,
    ) -> None:
        for obj in session.deleted:
            if isinstance(obj, SoftDeleteMixin):
                obj_id = getattr(obj, "id", "<unknown>")
                raise RuntimeError(
                    f"Hard-delete of {type(obj).__name__}(id={obj_id}) is forbidden. "
                    f"Use abridgeai.core.db.recursive_delete.soft_delete_cascade() "
                    f"to soft-delete with audit trail and ONETOMANY cascade."
                )


__all__ = ["register_hard_delete_guard"]
