"""Authoring quiz queries (teacher / draft surface).

Plan §5499-5503. Unlike :mod:`abridgeai.features.quizzes.queries.published`
these queries return quizzes / questions in *all* states (draft,
published, archived) so the teacher dashboard can show in-progress
work. Soft-deleted rows are still excluded (T0.7 loader-criteria).

The dedup helper :func:`list_existing_module_question_keys` is
co-located here because it powers the AI generation pipeline's
collision-detection stage (T5.4) — the original implementation lives
at ``backend/app/ai/haystack/pipelines/quiz_generation.py:1235-1257``
and used a Python-side hash. Here we delegate to a SQL file so the
hash is computed in PostgreSQL (sha256 of prompt + sorted chunk_ids)
and the network payload is just the hash set.
"""

from __future__ import annotations

from importlib import resources
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.quizzes.models import Quiz, QuizQuestion

_MODULE_QUESTION_KEYS_SQL = text(
    resources.files("abridgeai.features.quizzes.queries.sql")
    .joinpath("module_question_keys.sql")
    .read_text(encoding="utf-8")
)


async def get_quiz_for_authoring(db: AsyncSession, quiz_id: UUID) -> Quiz | None:
    """Quiz by id including draft / archived states.

    Soft-deleted rows are still filtered out (T0.7). Teachers see all
    statuses so the authoring dashboard can render in-progress quizzes.
    """
    stmt = select(Quiz).where(Quiz.id == quiz_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_quizzes_for_course(db: AsyncSession, course_id: UUID) -> list[Quiz]:
    """All quizzes (every status) for a course's authoring view.

    Ordered by ``created_at`` ascending so teachers see authoring
    history. Status filtering / sorting is the caller's job.
    """
    stmt = select(Quiz).where(Quiz.course_id == course_id).order_by(Quiz.created_at)
    return list((await db.execute(stmt)).scalars().all())


async def list_existing_module_question_keys(db: AsyncSession, quiz_id: UUID) -> set[str]:
    """Return the dedup-key set for every question in ``quiz_id``'s module.

    Each key is a sha256 hash of ``prompt_text`` plus an md5 of the
    sorted ``source_refs`` JSON, computed inside PostgreSQL via
    ``sql/module_question_keys.sql``. The pipeline uses this set as
    the seed for collision detection so a regenerate cannot re-emit a
    near-duplicate question already accepted elsewhere in the module.

    Ports the legacy
    ``_existing_module_question_keys`` helper (backend/app/ai/haystack/
    pipelines/quiz_generation.py:1235-1257) — output shape is now a
    flat ``set[str]`` since the chunk-clash layer 3 is being moved to
    the AI pipeline (T5.4) where it can use richer fixture data.
    """
    rows = (await db.execute(_MODULE_QUESTION_KEYS_SQL, {"quiz_id": quiz_id})).all()
    return {row.question_key for row in rows}


async def list_questions_with_source_refs(db: AsyncSession, quiz_id: UUID) -> list[QuizQuestion]:
    """Questions for a quiz including their ``source_refs`` payloads.

    Used by analytics + KG remediation lookups (§C5). The raw ORM
    rows are returned so callers can read either ``source_refs`` (KG
    chunk references) or ``original_generated_payload`` without
    re-fetching.
    """
    stmt = (
        select(QuizQuestion).where(QuizQuestion.quiz_id == quiz_id).order_by(QuizQuestion.position)
    )
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "get_quiz_for_authoring",
    "list_existing_module_question_keys",
    "list_questions_with_source_refs",
    "list_quizzes_for_course",
]
