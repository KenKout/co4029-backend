"""Invalidation for the published course-content cache: delete, then delete again.

Why this lives apart from :mod:`invalidator`
--------------------------------------------
The generic rules delete their keys in ``after_flush``, before the transaction
commits. For short-TTL, per-user keys that is an acceptable trade: the worst
case is one learner re-reading their own state a moment early.

``course_content:published:{course_id}`` is different. It is shared by every
learner on the course and rebuilt by a read that costs half a dozen batch
queries, so a flush-time delete on its own opens a window — between the delete
and the commit — where a concurrent reader repopulates the key from the
*pre-write* snapshot, and that stale tree then survives the full TTL, for
everyone.

So the keys go out twice. The flush delete keeps the window short for a long
transaction and covers sessions that flush without ever committing; the commit
delete is the one that actually guarantees correctness, because nothing can
repopulate from the old snapshot after it. A rollback simply drops the
stash — the flush delete it already sent cost a rebuild, nothing more.

Resolving a write to a course id
--------------------------------
The published tree is assembled from several tables, and a write to any of them
can change it. Most carry ``course_id`` (or, for ``courses`` itself, ``id``)
directly on the row. ``lessons`` and ``module_items`` only know their
``module_id``, so those are resolved with one batched lookup against
``modules`` — issued inside the flush, where the parent row is guaranteed to
already be visible to this transaction.

Deliberately NOT wired
----------------------
``user_profiles`` is embedded in the cached tree (instructor display name and
avatar) but is not invalidated here: resolving a profile to its courses is a
fan-out join, and the write is rare while the TTL is short. A renamed
instructor appears within ``COURSE_CONTENT_PUBLISHED.ttl_seconds``. Teaching-
team *membership* (``user_role_assignments``) does carry ``course_id``, so
adding or removing a teacher is invalidated immediately.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import Any, Final

from sqlalchemy import String, column, event, select, table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, UOWTransaction

from .keys import COURSE_CONTENT_PUBLISHED

logger = logging.getLogger(__name__)

#: Session-scoped stash of keys awaiting a successful commit.
_STASH_KEY: Final = "abridgeai_course_content_invalidations"

#: Tables whose rows name their course directly: ``table -> attribute``.
#: Every one of these feeds the published tree that
#: :func:`~abridgeai.features.courses.services.catalog.get_published_course_content_for_learner`
#: serialises — course metadata, modules, the polymorphic item targets, the
#: teaching team, and the career-path placements shown on the course block.
_DIRECT_COURSE_ATTR: Final[dict[str, str]] = {
    "courses": "id",
    "modules": "course_id",
    "quizzes": "course_id",
    "interview_configs": "course_id",
    "career_course_items": "course_id",
    "user_role_assignments": "course_id",
}

#: Tables that reach their course through ``modules.course_id``.
_VIA_MODULE: Final[frozenset[str]] = frozenset({"lessons", "module_items"})

#: Lightweight Core table — avoids importing the courses feature's ORM models
#: into ``core`` (which ``core.db`` imports at module load: a cycle).
_modules_table: Final = table(
    "modules",
    column("id", UUID(as_uuid=True)),
    column("course_id", UUID(as_uuid=True)),
)


def _tablename(instance: object) -> str | None:
    name = getattr(instance, "__tablename__", None)
    return name if isinstance(name, str) else None


def _direct_course_ids(instances: Iterable[object]) -> tuple[set[str], set[str]]:
    """Split flushed instances into ``(course_ids, module_ids)``."""
    course_ids: set[str] = set()
    module_ids: set[str] = set()
    for obj in instances:
        tablename = _tablename(obj)
        if tablename is None:
            continue
        attr = _DIRECT_COURSE_ATTR.get(tablename)
        if attr is not None:
            # `user_role_assignments.course_id` is NULL for org-scoped rows —
            # those grant nothing on a course, so there is nothing to drop.
            value = getattr(obj, attr, None)
            if value is not None:
                course_ids.add(str(value))
            continue
        if tablename in _VIA_MODULE:
            module_id = getattr(obj, "module_id", None)
            if module_id is not None:
                module_ids.add(str(module_id))
    return course_ids, module_ids


def _module_lookup(session: Session, module_ids: set[str]) -> tuple[Any, list[Any]]:
    """Return dialect-compatible columns and bind values for module ids."""
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "postgresql":
        return _modules_table, [uuid.UUID(module_id) for module_id in sorted(module_ids)]
    return (
        table("modules", column("id", String()), column("course_id", String())),
        sorted(module_ids),
    )


def _courses_for_modules(session: Session, module_ids: set[str]) -> set[str]:
    """Resolve ``module_ids`` to their course ids with one batched SELECT.

    Runs with autoflush disabled: we are inside ``after_flush`` and a nested
    flush would re-enter the unit of work. Any failure degrades to TTL expiry
    rather than breaking the write.
    """
    if not module_ids:
        return set()
    modules_table, lookup_ids = _module_lookup(session, module_ids)
    stmt = select(modules_table.c.course_id).where(modules_table.c.id.in_(lookup_ids))
    try:
        with session.no_autoflush:
            rows = session.execute(stmt).scalars().all()
    except SQLAlchemyError as exc:
        logger.warning(
            "cache.course_content_resolve_failed",
            extra={
                "event": "cache_course_content_resolve_failed",
                "module_ids": sorted(module_ids),
                "err": repr(exc),
            },
        )
        return set()
    return {str(course_id) for course_id in rows if course_id is not None}


def collect_course_content_keys(session: Session, instances: Iterable[object]) -> set[str]:
    """Cache keys invalidated by ``instances``; empty when none apply."""
    course_ids, module_ids = _direct_course_ids(instances)
    course_ids |= _courses_for_modules(session, module_ids)
    return {COURSE_CONTENT_PUBLISHED.format(course_id=cid) for cid in course_ids}


def _stash(session: Session, keys: set[str]) -> None:
    if not keys:
        return
    session.info.setdefault(_STASH_KEY, set()).update(keys)


def pop_stashed_keys(session: Session) -> set[str]:
    """Remove and return the keys this session has queued for deletion."""
    stashed = session.info.pop(_STASH_KEY, None)
    return set(stashed) if stashed else set()


def register_course_content_invalidation() -> None:
    """Wire the flush → commit invalidation pair (idempotent per process)."""
    # Imported here rather than at module scope: `invalidator` imports this
    # module to call the present function, so a top-level import would cycle.
    from .invalidator import schedule_key_deletion  # noqa: PLC0415

    @event.listens_for(Session, "after_flush")
    def _collect(  # noqa: ARG001 — SQLAlchemy event signature
        session: Session, flush_context: UOWTransaction
    ) -> None:
        instances = list(session.new) + list(session.dirty) + list(session.deleted)
        keys = collect_course_content_keys(session, instances)
        _stash(session, keys)
        schedule_key_deletion(keys)

    @event.listens_for(Session, "after_commit")
    def _publish(session: Session) -> None:
        schedule_key_deletion(pop_stashed_keys(session))

    @event.listens_for(Session, "after_rollback")
    def _discard(session: Session) -> None:
        pop_stashed_keys(session)


__all__ = [
    "collect_course_content_keys",
    "pop_stashed_keys",
    "register_course_content_invalidation",
]
