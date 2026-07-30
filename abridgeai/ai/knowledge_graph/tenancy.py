"""Resolve the owning organization for a set of lessons.

Every Neo4j read is tenant-scoped on ``Concept.org_id``, but the callers that
need that scope (quiz / interview retrieval, remediation) hold lesson ids, not
an org id. This module bridges the two with one indexed SQL hop.

Raw SQL on purpose: ``ai/`` must not import ``features.*`` ORM models, the
same constraint that makes ``ai/retrieval/pgvector.py`` hand-write its
queries. The import-linter contract depends on it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def organization_id_for_lessons(
    db: AsyncSession,
    lesson_ids: Iterable[UUID | str],
) -> UUID | None:
    """Return the organization owning ``lesson_ids``, or ``None``.

    Returns ``None`` when the lessons resolve to more than one organization.
    That should be impossible — a quiz or interview is scoped to one module,
    which belongs to one course, which belongs to one org — so it means the
    caller was handed a mixed set. Refusing to pick one is the safe answer:
    the KG lookup degrades to empty rather than guessing a tenant and reading
    another customer's graph.
    """
    ids = [str(lid) for lid in lesson_ids if lid is not None]
    if not ids:
        return None

    rows = (
        await db.execute(
            text(
                """
                SELECT DISTINCT c.organization_id AS org_id
                FROM lessons l
                JOIN modules m ON m.id = l.module_id
                JOIN courses c ON c.id = m.course_id
                WHERE l.id = ANY(CAST(:lesson_ids AS uuid[]))
                """
            ),
            {"lesson_ids": ids},
        )
    ).all()

    if len(rows) != 1:
        if len(rows) > 1:
            logger.error(
                "lessons %s span %d organizations; refusing to scope a KG read",
                ids,
                len(rows),
            )
        return None
    return UUID(str(rows[0].org_id))


async def organization_id_for_course(
    db: AsyncSession,
    course_id: UUID | str,
) -> UUID | None:
    """Return the organization owning ``course_id``, or ``None`` if unknown."""
    row = (
        await db.execute(
            text("SELECT organization_id FROM courses WHERE id = CAST(:cid AS uuid)"),
            {"cid": str(course_id)},
        )
    ).first()
    return UUID(str(row.organization_id)) if row else None


__all__ = ["organization_id_for_course", "organization_id_for_lessons"]
