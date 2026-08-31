"""Interview retrieval anchors (T6.4).

Ports anchor logic from
``backend/app/ai/haystack/pipelines/interview_generation.py:123-146``.

Anchor precedence (interview flavour):

    1. Teacher-supplied ``focus_topics`` from ``run.config_json``.
    2. KG concepts surfaced for the module's lessons (top
       :data:`MAX_KG_CONCEPTS_AS_ANCHORS`).
    3. Lesson titles for ``InterviewConfig.module_id``.
    4. ``InterviewConfig.title`` as last resort.

Unlike the quiz retriever, the interview path always loads the KG
context — interview question relevance comes from concept relations
(plan §6.4 MUST NOT: "Skip KG context"). The ``kg_context_enabled``
flag gates the network call, not the anchor list shape.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from abridgeai.ai.knowledge_graph.retrieval import retrieve_kg_context_for_lesson_ids
from abridgeai.ai.knowledge_graph.schemas import Concept
from abridgeai.ai.knowledge_graph.tenancy import organization_id_for_lessons

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import InterviewConfig

logger = logging.getLogger(__name__)

MAX_ANCHORS = 10
MAX_KG_CONCEPTS_AS_ANCHORS = 8


async def build_interview_anchors(
    db: AsyncSession,
    config: InterviewConfig,
    run_config: dict[str, Any],
    *,
    kg_context_enabled: bool = True,
) -> tuple[list[str], list[Concept]]:
    """Build the anchor list + KG concept list for an interview run.

    Returns ``(anchors, kg_concepts)``. The KG concepts are surfaced
    separately so :func:`retrieval_metadata` can audit "what KG context
    was loaded" independent of which strings ended up driving the
    embedding query.
    """

    focus_topics = [
        t.strip()
        for t in (run_config.get("focus_topics") or [])
        if isinstance(t, str) and t.strip()
    ]

    lesson_ids = await _module_lesson_ids(db, config.module_id)
    kg_concepts: list[Concept] = []
    if kg_context_enabled and lesson_ids:
        try:
            # See the matching note in the quiz anchor builder: the previous
            # call passed lesson UUIDs where concept NAMES were expected, so
            # this lookup never matched and the interview ran vector-only.
            org_id = await organization_id_for_lessons(db, lesson_ids)
            if org_id is not None:
                kg = await retrieve_kg_context_for_lesson_ids(
                    lesson_ids,
                    org_id=org_id,
                    depth=2,
                )
                kg_concepts = list(kg.concepts)
        except Exception as exc:  # pragma: no cover - graceful degrade
            logger.warning("KG anchor lookup failed for interview: %s", exc)
            kg_concepts = []

    if focus_topics:
        return focus_topics[:MAX_ANCHORS], kg_concepts

    anchors: list[str] = [c.name for c in kg_concepts[:MAX_KG_CONCEPTS_AS_ANCHORS]]

    if not anchors:
        anchors = await _module_lesson_titles(db, lesson_ids)

    if not anchors and config.title:
        anchors = [config.title]

    return _dedupe_preserving_order(anchors)[:MAX_ANCHORS], kg_concepts


async def _module_lesson_ids(db: AsyncSession, module_id: UUID | None) -> list[UUID]:
    """Resolve all non-deleted lesson ids for a module."""

    if module_id is None:
        return []

    # Lesson.position is intentionally absent (§A2) — lesson ordering lives
    # on the parent module_items.position link, not on lessons itself.
    stmt = text(
        "SELECT l.id FROM lessons l "
        "JOIN module_items mi ON mi.lesson_id = l.id AND mi.deleted_at IS NULL "
        "WHERE l.module_id = :module_id "
        "  AND l.deleted_at IS NULL "
        "ORDER BY mi.position"
    )
    result = await db.execute(stmt, {"module_id": module_id})
    return [row[0] for row in result.all() if row[0] is not None]


async def _module_lesson_titles(db: AsyncSession, lesson_ids: list[UUID]) -> list[str]:
    if not lesson_ids:
        return []

    stmt = text(
        "SELECT title FROM lessons "
        "WHERE id = ANY(CAST(:lesson_ids AS uuid[])) "
        "  AND deleted_at IS NULL"
    ).bindparams(bindparam("lesson_ids", type_=ARRAY(PG_UUID(as_uuid=True))))

    result = await db.execute(stmt, {"lesson_ids": lesson_ids})
    return [str(row[0]) for row in result.all() if row[0]]


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = [
    "MAX_ANCHORS",
    "MAX_KG_CONCEPTS_AS_ANCHORS",
    "build_interview_anchors",
]
