"""Shared authoring guards for the interview services layer (T6.11).

A tiny leaf module so :mod:`.authoring`, :mod:`.outcomes`, :mod:`.generation_runs`
and :mod:`.bank` can all use the same config-lookup and outcome-threshold
guards without a cycle: nothing here imports back into its siblings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.features.courses.api import public as courses_public
from abridgeai.features.interviews.models import InterviewOutcome
from abridgeai.features.interviews.queries import authoring as authoring_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import InterviewConfig


async def _require_config(db: AsyncSession, config_id: UUID) -> InterviewConfig:
    """Return the interview config or raise NotFound; permission is enforced
    at the router edge."""
    config = await authoring_queries.get_interview_for_authoring(db, config_id)
    if config is None:
        raise NotFoundError(f"Interview config {config_id} not found")
    return config


def _assert_threshold_within_outcome_count(
    min_outcomes_to_pass: int | None,
    outcome_count: int,
) -> None:
    """Reject a pass threshold above the number of active outcomes."""
    if min_outcomes_to_pass is not None and min_outcomes_to_pass > outcome_count:
        raise AppError(
            "min_outcomes_to_pass_exceeds_active_outcomes: minimum outcomes to pass "
            f"({min_outcomes_to_pass}) cannot exceed active learning outcomes ({outcome_count})"
        )


async def _assert_config_threshold_valid(db: AsyncSession, config: InterviewConfig) -> None:
    """Validate the config's stored threshold against its live outcomes."""
    outcomes = await authoring_queries.list_outcomes_for_config(db, config.id)
    _assert_threshold_within_outcome_count(config.min_outcomes_to_pass, len(outcomes))


async def _require_outcome(
    db: AsyncSession,
    config_id: UUID,
    outcome_id: UUID,
) -> InterviewOutcome:
    """Return the outcome scoped to its config or raise NotFound."""
    outcome = await db.get(InterviewOutcome, outcome_id)
    if outcome is None or outcome.interview_config_id != config_id:
        raise NotFoundError(f"Interview outcome {outcome_id} not found")
    return outcome


def _normalise_title(value: str) -> str:
    """Collapse whitespace + casefold: the comparison a teacher reads as equal."""
    return " ".join(value.split()).casefold()


async def _require_module_in_course(db: AsyncSession, module_id: UUID, course_id: UUID) -> None:
    """Raise when ``module_id`` does not belong to ``course_id``."""
    module = await courses_public.get_module_by_id(db, module_id)
    if module is None or module.course_id != course_id:
        raise AppError(f"Module {module_id} is not part of course {course_id}")


async def _require_lessons_in_course(
    db: AsyncSession, lesson_ids: list[UUID], course_id: UUID
) -> None:
    """Raise when any lesson id is unknown or lives in another course."""
    for lesson_id in lesson_ids:
        lesson = await courses_public.get_lesson_by_id(db, lesson_id)
        if lesson is None:
            raise AppError(f"Lesson {lesson_id} not found")
        module = await courses_public.get_module_by_id(db, lesson.module_id)
        if module is None or module.course_id != course_id:
            raise AppError(f"Lesson {lesson_id} is not part of course {course_id}")


__all__ = [
    "_assert_config_threshold_valid",
    "_assert_threshold_within_outcome_count",
    "_normalise_title",
    "_require_config",
    "_require_lessons_in_course",
    "_require_module_in_course",
    "_require_outcome",
]
