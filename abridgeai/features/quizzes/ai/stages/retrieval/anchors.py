"""Anchor builder for quiz retrieval (T5.4).

Ports ``_build_query_anchors`` + ``_kg_lesson_ids`` + ``_source_lesson_titles``
from ``backend/app/ai/haystack/pipelines/quiz_generation.py:1019-1141``.

Anchor precedence (FR-11):
    1. Teacher-supplied ``focus_topics`` (verbatim).
    2. KG concept labels (top 8) merged with body-section headings from
       the lesson outline (top 8 per outline).
    3. Lesson titles only.
    4. ``quiz.title`` as last resort.

The builder is intentionally decoupled from the retrieval orchestrator:
it returns a plain ``list[str]`` so the caller can audit, log, or
override the chosen anchors before embedding.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from abridgeai.ai.retrieval import retrieve_kg_context_for_anchors

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.quizzes.models import Quiz

logger = logging.getLogger(__name__)

MAX_ANCHORS = 10
MAX_KG_CONCEPTS_AS_ANCHORS = 8
MAX_OUTLINE_SECTIONS_AS_ANCHORS = 8


async def build_query_anchors(
    db: AsyncSession,
    quiz: Quiz,
    config: dict[str, Any],
    *,
    question_hint: str | None = None,
    kg_context_enabled: bool = True,
) -> list[str]:
    """Build the ordered anchor list driving multi-anchor retrieval.

    Parameters
    ----------
    db
        Async session for lesson-title fallback.
    quiz
        Quiz draft (used for ``module_id`` + ``title`` fallback).
    config
        Generation-run config dict; reads ``focus_topics`` and
        ``source_lesson_ids``.
    question_hint
        Optional precomputed anchor (e.g. when regenerating one
        question). When supplied it is returned as a single-element list
        and short-circuits all other precedence rules.
    kg_context_enabled
        When ``False`` the KG anchor lookup is skipped and the builder
        falls through to lesson titles / quiz title.

    Returns
    -------
    list[str]
        Up to :data:`MAX_ANCHORS` anchor strings. May be empty when the
        quiz has no module, no lessons, and no title.
    """

    if question_hint is not None:
        cleaned = question_hint.strip()
        return [cleaned] if cleaned else []

    focus_topics = [
        t.strip()
        for t in (config.get("focus_topics") or [])
        if isinstance(t, str) and t.strip()
    ]
    if focus_topics:
        return focus_topics[:MAX_ANCHORS]

    anchors: list[str] = []

    lesson_ids = _kg_lesson_ids(config, quiz)
    if kg_context_enabled and lesson_ids:
        try:
            kg = await retrieve_kg_context_for_anchors(
                [str(lid) for lid in lesson_ids],
                depth=2,
            )
            anchors.extend(c.name for c in kg.concepts[:MAX_KG_CONCEPTS_AS_ANCHORS])
        except Exception as exc:
            logger.warning("KG anchor lookup failed: %s", exc)

    if anchors:
        deduped = _dedupe_preserving_order(anchors)
        return deduped[:MAX_ANCHORS]

    titles = await _source_lesson_titles(db, config)
    if titles:
        return titles[:MAX_ANCHORS]

    return [quiz.title] if quiz.title else []


def _kg_lesson_ids(config: dict[str, Any], quiz: Quiz) -> list[UUID]:
    """Parse ``source_lesson_ids`` from the run config.

    Mirrors legacy precedence: when no explicit lessons are configured,
    return ``[]`` rather than dragging in the entire module's KG.
    """

    raw_ids = config.get("source_lesson_ids") or []
    parsed: list[UUID] = []
    for raw_id in raw_ids:
        try:
            parsed.append(UUID(str(raw_id)))
        except ValueError:
            continue
    if not parsed and quiz.module_id is not None:
        return []
    return parsed


async def _source_lesson_titles(
    db: AsyncSession, config: dict[str, Any]
) -> list[str]:
    lesson_ids = _kg_lesson_ids_loose(config)
    if not lesson_ids:
        return []

    stmt = text(
        "SELECT title FROM lessons "
        "WHERE id = ANY(CAST(:lesson_ids AS uuid[])) "
        "  AND deleted_at IS NULL"
    ).bindparams(bindparam("lesson_ids", type_=ARRAY(PG_UUID(as_uuid=True))))

    result = await db.execute(stmt, {"lesson_ids": lesson_ids})
    return [str(row[0]) for row in result.all() if row[0]]


def _kg_lesson_ids_loose(config: dict[str, Any]) -> list[UUID]:
    parsed: list[UUID] = []
    for raw_id in config.get("source_lesson_ids") or []:
        try:
            parsed.append(UUID(str(raw_id)))
        except ValueError:
            continue
    return parsed


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
    "build_query_anchors",
]
