"""Quiz persistence stage (T5.9).

Ports ``_persist_questions`` + ``_replace_question_in_place`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:871-958``. Both
helpers ``db.add()`` + ``db.flush()`` only; the caller owns the
transaction. ``QuizQuestion`` audit columns auto-populate via the T0.8
``before_flush`` listener; ``QuizQuestionRevision.created_by`` is set
explicitly because that table uses ``CreatedAtMixin`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from sqlalchemy import delete, func, select

from abridgeai.features.quizzes.models import (
    Quiz,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.retrieval import ChunkWithDistance


class _RunLike(Protocol):
    """Stub of ``GenerationRun`` — only ``requested_by`` is read."""

    requested_by: UUID | None


def _structure_source_refs(
    raw_refs: object,
    chunks: list[ChunkWithDistance],
) -> list[dict[str, Any]]:
    """Hydrate generator chunk-id refs into JSONB ref dicts.

    Output schema: ``[{"chunk_id", "material_version_id", "course_id",
    "lesson_id"}]``. Order preserved; duplicates dropped; unknown ids
    skipped (mirrors legacy ``structure_source_refs``).
    """

    if not isinstance(raw_refs, list):
        return []

    chunks_by_id = {str(chunk.chunk_id): chunk for chunk in chunks}
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_ref in raw_refs:
        chunk_id = _ref_chunk_id(raw_ref)
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        refs.append(
            {
                "chunk_id": str(chunk.chunk_id),
                "material_version_id": str(chunk.material_version_id),
                "course_id": str(chunk.course_id) if chunk.course_id else None,
                "lesson_id": str(chunk.lesson_id) if chunk.lesson_id else None,
            }
        )
    return refs


def _ref_chunk_id(raw_ref: object) -> str | None:
    if isinstance(raw_ref, str):
        return raw_ref
    if isinstance(raw_ref, dict):
        value = raw_ref.get("chunk_id") or raw_ref.get("id")
        return str(value) if value else None
    return None


async def persist_questions(
    db: AsyncSession,
    run: _RunLike,
    quiz: Quiz,
    chunks: list[ChunkWithDistance],
    questions: list[dict[str, Any]],
) -> list[QuizQuestion]:
    """Create QuizQuestion + options + initial revision rows.

    Each ``questions`` entry: ``question_type``, ``prompt_text``,
    ``hint_text``, ``explanation``, ``difficulty``, ``bloom_level``,
    ``expected_response_time_ms``, ``source_refs`` (raw chunk-id list),
    ``original_generated_payload``, ``options`` (kwargs for
    ``QuizQuestionOption``). Positions append after the current max.
    """

    if not questions:
        return []

    position_result = await db.execute(
        select(func.coalesce(func.max(QuizQuestion.position), 0)).where(
            QuizQuestion.quiz_id == quiz.id
        )
    )
    start_position = int(position_result.scalar_one())

    persisted: list[QuizQuestion] = []
    for offset, payload in enumerate(questions, start=1):
        structured_refs = _structure_source_refs(payload.get("source_refs"), chunks)
        question = QuizQuestion(
            quiz_id=quiz.id,
            position=start_position + offset,
            question_type=payload["question_type"],
            prompt_text=payload["prompt_text"],
            hint_text=payload.get("hint_text"),
            explanation=payload.get("explanation"),
            difficulty=payload.get("difficulty"),
            bloom_level=payload.get("bloom_level"),
            review_status="pending",
            expected_response_time_ms=payload.get("expected_response_time_ms"),
            source_refs=structured_refs,
            original_generated_payload=payload.get("original_generated_payload"),
        )
        db.add(question)
        await db.flush()

        for option_payload in payload.get("options", []):
            db.add(QuizQuestionOption(question_id=question.id, **option_payload))

        db.add(
            QuizQuestionRevision(
                question_id=question.id,
                revision_no=1,
                source_kind="ai",
                payload_json=payload.get("original_generated_payload") or {},
                created_by=run.requested_by,
            )
        )
        await db.flush()
        persisted.append(question)

    return persisted


async def replace_question_in_place(
    db: AsyncSession,
    run: _RunLike,
    question: QuizQuestion,
    payload: dict[str, Any],
    chunks: list[ChunkWithDistance] | None = None,
) -> QuizQuestion:
    """Regenerate ``question`` in-place: same id, bumped ``revision_no``,
    review state reset to ``pending``, options atomically replaced.
    When ``chunks`` is provided, ``payload['source_refs']`` is re-hydrated.
    """

    structured_refs: list[dict[str, Any]] | Any = payload.get("source_refs")
    if chunks is not None:
        structured_refs = _structure_source_refs(structured_refs, chunks)

    question.question_type = payload["question_type"]
    question.prompt_text = payload["prompt_text"]
    question.hint_text = payload.get("hint_text")
    question.explanation = payload.get("explanation")
    question.difficulty = payload.get("difficulty")
    question.bloom_level = payload.get("bloom_level")
    question.review_status = "pending"
    question.expected_response_time_ms = payload.get("expected_response_time_ms")
    question.source_refs = structured_refs if structured_refs is not None else []
    question.original_generated_payload = payload.get("original_generated_payload")
    question.reviewed_by = None
    question.reviewed_at = None

    await db.execute(
        delete(QuizQuestionOption).where(QuizQuestionOption.question_id == question.id)
    )
    for option_payload in payload.get("options", []):
        db.add(QuizQuestionOption(question_id=question.id, **option_payload))

    revision_no_result = await db.execute(
        select(func.coalesce(func.max(QuizQuestionRevision.revision_no), 0)).where(
            QuizQuestionRevision.question_id == question.id
        )
    )
    db.add(
        QuizQuestionRevision(
            question_id=question.id,
            revision_no=int(revision_no_result.scalar_one()) + 1,
            source_kind="ai",
            payload_json=payload.get("original_generated_payload") or {},
            created_by=run.requested_by,
        )
    )
    await db.flush()
    return question


__all__ = ["persist_questions", "replace_question_in_place"]
