"""Recursive soft-delete service.

App-level cascade replacement for the DB-level CASCADE rules that T0.14
flipped to NO ACTION on every FK targeting a SoftDeleteMixin parent.

Walks SQLAlchemy ``inspect(type(instance)).relationships`` at runtime,
recurses into ONETOMANY children that themselves carry SoftDeleteMixin,
and stamps ``deleted_at`` / ``deleted_by`` on every reachable instance.
The whole traversal runs inside the caller's transaction; if any part
fails the caller is responsible for rolling back (we ``flush`` only —
no commit, no nested savepoint).

Direction handling:
- ONETOMANY  → recurse (these are the children we own and must clean up)
- MANYTOONE  → skip (this is the parent direction; would walk upward)
- MANYTOMANY → skip (link-table semantics depend on the association
  table; if the assoc carries SoftDeleteMixin it is reachable through
  some ONETOMANY relationship from one of its sides anyway)

Cycle safety:
- Tracks visited ``(table_name, id)`` pairs in a set; revisits are
  silent no-ops. Required because real-world schemas have self-FKs
  (e.g. ``module_prerequisites`` — Module → Module via secondary).

dry_run:
- ``dry_run=True`` returns the planned ``affected`` list without
  writing ``deleted_at`` / ``deleted_by`` and without flushing.
  Useful for admin "preview before delete" UX.

Actor identity:
- Explicit ``actor_id`` argument wins. Otherwise falls back to
  ``current_actor_var.get()`` (the same contextvar the audit listener
  reads). NULL actor is allowed (system path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.relationships import ONETOMANY

from abridgeai.core.audit.context import current_actor_var
from abridgeai.core.db.mixins import SoftDeleteMixin


@dataclass
class SoftDeleteResult:
    affected: list[tuple[str, UUID]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.affected)


async def soft_delete_cascade(
    session: AsyncSession,
    instance: Any,
    actor_id: UUID | None = None,
    *,
    dry_run: bool = False,
) -> SoftDeleteResult:
    """Soft-delete ``instance`` and every SoftDelete descendant reachable
    via ONETOMANY relationships.

    Parameters
    ----------
    session : AsyncSession
        Open async session. Caller owns the transaction lifecycle.
    instance : Any
        ORM instance to soft-delete. Must inherit ``SoftDeleteMixin``;
        non-mixin instances cause an empty result (silent no-op).
    actor_id : UUID | None
        Explicit actor for ``deleted_by``. If None, falls back to
        ``current_actor_var.get()``.
    dry_run : bool, keyword-only
        If True, returns the planned list of affected rows without
        writing or flushing.

    Returns
    -------
    SoftDeleteResult
        ``affected`` lists (table_name, id) pairs in traversal order.
    """
    result = SoftDeleteResult()
    if not isinstance(instance, SoftDeleteMixin):
        return result

    visited: set[tuple[str, UUID]] = set()
    effective_actor = actor_id if actor_id is not None else current_actor_var.get()
    now = datetime.now(timezone.utc)

    async def _walk(obj: Any) -> None:
        if not isinstance(obj, SoftDeleteMixin):
            return
        mapper = inspect(type(obj))
        table_name = mapper.persist_selectable.name
        obj_id = getattr(obj, "id", None)
        if obj_id is None:
            return
        key: tuple[str, UUID] = (table_name, obj_id)
        if key in visited:
            return
        visited.add(key)
        result.affected.append(key)

        for rel in mapper.relationships:
            if rel.direction is not ONETOMANY:
                continue
            # Async-safe lazy-load: refresh attaches the collection to
            # the loaded object even when expire_on_commit=False has
            # already detached it.
            await session.refresh(obj, attribute_names=[rel.key])
            children = getattr(obj, rel.key) or []
            for child in children:
                await _walk(child)

        if not dry_run:
            obj.deleted_at = now
            obj.deleted_by = effective_actor

    await _walk(instance)

    if not dry_run:
        await session.flush()

    return result


__all__ = ["SoftDeleteResult", "soft_delete_cascade"]
