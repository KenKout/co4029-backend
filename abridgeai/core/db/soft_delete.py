"""Global soft-delete filter via SQLAlchemy do_orm_execute event.

Registers a single Session-level event listener that auto-applies
``WHERE deleted_at IS NULL`` to every SELECT touching a SoftDeleteMixin
subclass. Idempotent — safe to call multiple times.

Opt-out per-query: pass ``execution_options(include_deleted=True)`` to
the statement (admin/audit views).

Writes (INSERT/UPDATE/DELETE) bypass the filter — the contract is
read-side only. Soft-deletion is performed by setting ``deleted_at``
(an UPDATE), not by issuing a DELETE.
"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.orm import ORMExecuteState, Session, with_loader_criteria

from abridgeai.core.db.mixins import SoftDeleteMixin

_REGISTERED = False


def register_soft_delete_filter() -> None:
    """Wire the do_orm_execute listener once per process."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    @event.listens_for(Session, "do_orm_execute")
    def _apply_soft_delete_filter(execute_state: ORMExecuteState) -> None:
        if execute_state.execution_options.get("include_deleted", False):
            return
        if not execute_state.is_select:
            return
        execute_state.statement = execute_state.statement.options(
            with_loader_criteria(
                SoftDeleteMixin,
                lambda cls: cls.deleted_at.is_(None),
                include_aliases=True,
            )
        )


__all__ = ["register_soft_delete_filter"]
