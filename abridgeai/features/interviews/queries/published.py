"""Visibility-filtered interview queries (learner / take surface).

Plan §6.3. Returns ORM models; the schema layer enforces redaction
(``hidden_reasoning``, ``internal_summary_json`` are private — never
exposed in public schemas).

Soft-delete is filtered automatically by the T0.7
``with_loader_criteria`` listener — every ``select(InterviewConfig)`` /
``select(InterviewOutcome)`` / ``select(InterviewQuestion)`` emits
``WHERE deleted_at IS NULL`` for free.

Status gating: every query in this module also filters
``status='published'`` for ``InterviewConfig`` and (for questions)
``review_status='approved'``. Drafts and pending-review questions
never reach the learner.

Attempt / prerequisite policy
-----------------------------
:func:`get_interview_for_taking` validates a learner's eligibility to
take an interview. The strict gates (max-attempts, prereq-pass) are
enforced at the service layer in T6.10/T6.11; this query simply
returns the (config, outcomes, questions) triple if the interview is
published and the learner has not exhausted ``max_attempts``. Per the
task brief: "if you can't validate prereqs in this query, return raw
rows and let service do that check" — we ship raw rows.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewQuestion,
    InterviewSession,
)


async def get_published_interview(db: AsyncSession, config_id: UUID) -> InterviewConfig | None:
    """Single published interview config by id, or ``None``.

    Excludes drafts, archived, and soft-deleted rows. Existence is not
    leaked — service maps ``None`` to HTTP 404 uniformly.
    """
    stmt = select(InterviewConfig).where(
        InterviewConfig.id == config_id,
        InterviewConfig.status == "published",
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_published_interviews_for_module(
    db: AsyncSession, module_id: UUID
) -> list[InterviewConfig]:
    """Published interview configs attached to ``module_id``.

    An interview is "attached" to a module when EITHER:

    * ``InterviewConfig.module_id`` matches (the canonical owning
      column on the ``interview_configs`` table), OR
    * a ``module_items`` row with ``item_type='interview'`` points at
      the config from that module (the catalog-level link table).

    The cross-feature ``module_items`` arm is queried via raw SQL so
    the import-linter contract (features are independent) stays clean
    — interviews does not import the courses ORM.
    """
    rows = (
        await db.execute(
            text(
                "SELECT ic.id FROM interview_configs ic "
                "WHERE ic.deleted_at IS NULL "
                "  AND ic.status = 'published' "
                "  AND ( "
                "    ic.module_id = :module_id "
                "    OR EXISTS ( "
                "      SELECT 1 FROM module_items mi "
                "      WHERE mi.module_id = :module_id "
                "        AND mi.interview_config_id = ic.id "
                "        AND mi.deleted_at IS NULL "
                "    ) "
                "  ) "
                "ORDER BY ic.created_at"
            ),
            {"module_id": module_id},
        )
    ).all()
    if not rows:
        return []
    config_ids = [row.id for row in rows]
    stmt = (
        select(InterviewConfig)
        .where(
            InterviewConfig.id.in_(config_ids),
            InterviewConfig.status == "published",
        )
        .order_by(InterviewConfig.created_at)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_published_interviews_for_course(
    db: AsyncSession, course_id: UUID
) -> list[InterviewConfig]:
    """Published interview configs anchored to ``course_id``.

    Direct ``InterviewConfig.course_id`` filter — no module_items
    detour because course-scoped surfacing is feature-internal.
    """
    stmt = (
        select(InterviewConfig)
        .where(
            InterviewConfig.course_id == course_id,
            InterviewConfig.status == "published",
        )
        .order_by(InterviewConfig.created_at)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_published_outcomes_for_config(
    db: AsyncSession, config_id: UUID
) -> list[InterviewOutcome]:
    """Outcomes (rubric criteria) for a published interview, ordered."""
    stmt = (
        select(InterviewOutcome)
        .where(InterviewOutcome.interview_config_id == config_id)
        .order_by(InterviewOutcome.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_published_questions_for_config(
    db: AsyncSession, config_id: UUID
) -> list[InterviewQuestion]:
    """Approved questions only — pending / rejected / edited drop out.

    Returns ordered by ``position``. The service layer is responsible
    for any per-session sampling (the bank may have more than the
    interview asks).
    """
    stmt = (
        select(InterviewQuestion)
        .where(
            InterviewQuestion.interview_config_id == config_id,
            InterviewQuestion.review_status == "approved",
        )
        .order_by(InterviewQuestion.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_interview_for_taking(
    db: AsyncSession,
    config_id: UUID,
    user_id: UUID,
) -> tuple[InterviewConfig, list[InterviewOutcome], list[InterviewQuestion]] | None:
    """Validate and return everything needed to start an interview.

    Returns ``None`` (→ 404) when the interview is missing /
    unpublished / soft-deleted. Returns the
    ``(config, outcomes, questions)`` triple otherwise.

    Attempt-cap enforcement: when ``InterviewConfig.max_attempts`` is
    set, the count of existing ``InterviewSession`` rows for
    ``(user_id, config_id)`` is compared against the ceiling. NULL
    ``max_attempts`` disables the gate (unbounded retakes).

    Prerequisite enforcement (per UC-LEARN-02 thesis: "must have passed
    prerequisite quizzes") happens in the service layer — this query
    returns raw rows and lets ``services/lifecycle.py`` resolve
    cross-feature prereq state.
    """
    config = await get_published_interview(db, config_id)
    if config is None:
        return None

    if config.max_attempts is not None and config.max_attempts > 0:
        attempts_stmt = select(func.count(InterviewSession.id)).where(
            InterviewSession.interview_config_id == config_id,
            InterviewSession.student_id == user_id,
        )
        used = (await db.execute(attempts_stmt)).scalar_one()
        if used >= config.max_attempts:
            return None

    outcomes = await list_published_outcomes_for_config(db, config_id)
    questions = await list_published_questions_for_config(db, config_id)
    return config, outcomes, questions


__all__ = [
    "get_interview_for_taking",
    "get_published_interview",
    "list_published_interviews_for_course",
    "list_published_interviews_for_module",
    "list_published_outcomes_for_config",
    "list_published_questions_for_config",
]
