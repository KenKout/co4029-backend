"""Authoring interview queries (teacher / draft surface).

Plan §6.3. Unlike :mod:`abridgeai.features.interviews.queries.published`
these queries return interview configs / outcomes / questions in *all*
states (draft, published, archived) so the teacher dashboard can show
in-progress work. Soft-deleted rows are excluded by the T0.7
loader-criteria listener; the authoring path has no restore surface
(per Metis E3a — confirmed no callers reach the deleted set).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewQuestion,
)


async def get_interview_for_authoring(
    db: AsyncSession,
    config_id: UUID,
) -> InterviewConfig | None:
    """Interview config by id including draft / archived states.

    Soft-deleted rows are filtered by the T0.7 loader-criteria listener;
    Metis E3a confirmed no authoring caller needs the deleted set.
    """
    stmt = select(InterviewConfig).where(InterviewConfig.id == config_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_interviews_for_course(
    db: AsyncSession,
    course_id: UUID,
    *,
    include_archived: bool = False,
) -> list[InterviewConfig]:
    """All interview configs for a course's authoring view.

    Ordered by ``created_at`` ascending. Archived configs are excluded
    by default; pass ``include_archived=True`` to include them
    (mirrors the T3.4 / T5.3 ``include_archived`` flag).
    """
    stmt = select(InterviewConfig).where(InterviewConfig.course_id == course_id)
    if not include_archived:
        stmt = stmt.where(InterviewConfig.status != "archived")
    stmt = stmt.order_by(InterviewConfig.created_at)
    return list((await db.execute(stmt)).scalars().all())


async def get_interview_with_full_config(
    db: AsyncSession, config_id: UUID
) -> tuple[InterviewConfig, list[InterviewOutcome], list[InterviewQuestion]] | None:
    """Authoring view including outcomes + questions in any review state.

    Returns the ``(config, outcomes, questions)`` triple. Unlike the
    learner-side ``get_interview_for_taking``, this surfaces:

    * configs in any status (draft / published / archived),
    * questions in any ``review_status`` (pending / approved / edited
      / rejected),
    * authoring-side fields (``difficulty``, ``source_refs_json``,
      ``ai_generated``, ``reviewed_by``, ``reviewed_at``) — the
      schemas/authoring.py layer is responsible for shaping the
      outbound payload.
    """
    config = await get_interview_for_authoring(db, config_id)
    if config is None:
        return None
    outcomes = await list_outcomes_for_config(db, config_id)
    questions = await list_questions_for_config(db, config_id)
    return config, outcomes, questions


async def list_outcomes_for_config(db: AsyncSession, config_id: UUID) -> list[InterviewOutcome]:
    """Outcomes (rubric criteria) for any interview, ordered by position."""
    stmt = (
        select(InterviewOutcome)
        .where(InterviewOutcome.interview_config_id == config_id)
        .order_by(InterviewOutcome.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_questions_for_config(
    db: AsyncSession,
    config_id: UUID,
    *,
    review_status: str | None = None,
) -> list[InterviewQuestion]:
    """Questions for any interview config, optionally filtered by review.

    ``review_status`` is one of ``pending`` / ``approved`` / ``edited``
    / ``rejected``; ``None`` returns every state (default — the
    teacher review queue needs to see pending + edited side-by-side).
    """
    stmt = select(InterviewQuestion).where(InterviewQuestion.interview_config_id == config_id)
    if review_status is not None:
        stmt = stmt.where(InterviewQuestion.review_status == review_status)
    stmt = stmt.order_by(InterviewQuestion.position)
    return list((await db.execute(stmt)).scalars().all())


async def next_outcome_position(db: AsyncSession, config_id: UUID) -> int:
    """``MAX(position) + 1`` for ``INSERT`` on a new outcome.

    Returns 1 when the config has no outcomes yet. The
    ``uq_interview_outcomes_position`` UNIQUE constraint enforces
    no-collision; service layer uses this helper before
    ``db.add(InterviewOutcome(position=...))``.
    """
    stmt = select(func.coalesce(func.max(InterviewOutcome.position), 0)).where(
        InterviewOutcome.interview_config_id == config_id
    )
    return int((await db.execute(stmt)).scalar_one()) + 1


async def lock_question_append(db: AsyncSession, config_id: UUID) -> None:
    """Serialize automatic question position allocation for one config.

    The transaction-scoped advisory lock survives through the caller's commit,
    including an empty question list where row-level locks cannot help.
    """
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"interview_questions_append:{config_id}"},
    )


async def lock_bank_group(db: AsyncSession, course_id: UUID, group_id: UUID) -> None:
    """Serialize writes that reshape one logical bank group.

    Sibling adds and grouped edits reach a group through DIFFERENT rows, so
    ``SELECT ... FOR UPDATE`` on the row the request names cannot serialize
    them against each other. The transaction-scoped advisory lock is keyed on
    the durable group id instead and is held until the caller commits.
    """
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"interview_question_bank_group:{course_id}:{group_id}"},
    )


async def next_question_position(db: AsyncSession, config_id: UUID) -> int:
    """``MAX(position) + 1`` for ``INSERT`` on a new question.

    ``InterviewQuestion.position`` is nullable in baseline DDL — a
    NULL position skips the UNIQUE constraint. Callers that want
    explicit ordering use this helper. Returns 1 for empty configs.
    """
    stmt = select(func.coalesce(func.max(InterviewQuestion.position), 0)).where(
        InterviewQuestion.interview_config_id == config_id
    )
    current = (await db.execute(stmt)).scalar_one()
    return int(current or 0) + 1


__all__ = [
    "get_interview_for_authoring",
    "get_interview_with_full_config",
    "list_interviews_for_course",
    "list_outcomes_for_config",
    "list_questions_for_config",
    "lock_bank_group",
    "lock_question_append",
    "next_outcome_position",
    "next_question_position",
]
