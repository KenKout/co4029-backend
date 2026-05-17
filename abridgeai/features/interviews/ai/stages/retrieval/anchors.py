"""Interview retrieval anchors + student-weakness chunk lookup (T6.4).

Ports anchor logic from
``backend/app/ai/haystack/pipelines/interview_generation.py:123-146``
and extends it with the **interview-specific** student-weakness anchor
the plan calls out in §6.4 ("recent quiz misses — signal of student
gap").

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

Student-weakness chunks are returned **separately** from the anchor
list. They feed an additional vector-search-free chunk pool that the
generation stage merges into its grounding context. This keeps the
"weak topics" signal from polluting the anchor pre-filter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from abridgeai.ai.knowledge_graph.schemas import Concept
from abridgeai.ai.retrieval import (
    ChunkWithDistance,
    retrieve_kg_context_for_anchors,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import InterviewConfig

logger = logging.getLogger(__name__)

MAX_ANCHORS = 10
MAX_KG_CONCEPTS_AS_ANCHORS = 8
MAX_WEAK_TOPIC_CHUNKS = 12


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
            kg = await retrieve_kg_context_for_anchors(
                [str(lid) for lid in lesson_ids],
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


async def fetch_weak_topic_chunks(
    db: AsyncSession,
    *,
    student_id: UUID,
    module_id: UUID,
    limit: int = MAX_WEAK_TOPIC_CHUNKS,
) -> list[ChunkWithDistance]:
    """Pull chunks linked to quiz questions the student got wrong.

    Anchored to ``module_id`` so we don't drag in unrelated misses from
    other modules in the same course. Joins:

        quiz_attempts → quiz_attempt_answers (is_correct=false) →
        quiz_questions.source_refs[*].chunk_id → document_chunks.

    The lookup is best-effort — any DB error degrades to an empty list
    so the generation stage can still proceed with the primary
    grounding context.
    """

    if limit <= 0:
        return []

    stmt = text(
        "WITH wrong_chunks AS ( "
        "  SELECT DISTINCT CAST(sr->>'chunk_id' AS uuid) AS chunk_id "
        "  FROM quiz_attempts qa "
        "  JOIN quiz_attempt_answers qaa ON qaa.attempt_id = qa.id "
        "  JOIN quiz_questions qq ON qq.id = qaa.question_id "
        "  JOIN quizzes q ON q.id = qa.quiz_id "
        "  CROSS JOIN LATERAL jsonb_array_elements( "
        "    CASE jsonb_typeof(qq.source_refs) "
        "      WHEN 'array' THEN qq.source_refs ELSE '[]'::jsonb "
        "    END "
        "  ) AS sr "
        "  WHERE qa.student_id = :student_id "
        "    AND qaa.is_correct = FALSE "
        "    AND q.module_id = :module_id "
        "    AND qq.deleted_at IS NULL "
        "    AND q.deleted_at IS NULL "
        "    AND sr ? 'chunk_id' "
        ") "
        "SELECT dc.id AS chunk_id, dc.material_version_id, dc.course_id, "
        "       dc.lesson_id, dc.content "
        "FROM wrong_chunks wc "
        "JOIN document_chunks dc ON dc.id = wc.chunk_id "
        "LIMIT :limit"
    )

    try:
        result = await db.execute(
            stmt,
            {
                "student_id": student_id,
                "module_id": module_id,
                "limit": int(limit),
            },
        )
    except Exception as exc:  # pragma: no cover - graceful degrade
        logger.warning("Weak-topic chunk lookup failed: %s", exc)
        return []

    rows = result.mappings().all()
    return [
        ChunkWithDistance(
            chunk_id=row["chunk_id"],
            material_version_id=row["material_version_id"],
            course_id=row["course_id"],
            lesson_id=row["lesson_id"],
            content=row["content"],
            distance=1.0,  # weak-topic chunks have no vector similarity score
            embedding=None,
        )
        for row in rows
    ]


async def _module_lesson_ids(db: AsyncSession, module_id: UUID | None) -> list[UUID]:
    """Resolve all non-deleted lesson ids for a module."""

    if module_id is None:
        return []

    stmt = text(
        "SELECT id FROM lessons "
        "WHERE module_id = :module_id "
        "  AND deleted_at IS NULL "
        "ORDER BY position"
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
    "MAX_WEAK_TOPIC_CHUNKS",
    "build_interview_anchors",
    "fetch_weak_topic_chunks",
]
