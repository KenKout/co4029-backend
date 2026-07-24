"""Authorization + access-resolution helpers for the discussions feature.

Two perimeters, mirroring the house pattern:

* **Teacher (authoring)** — post / edit / close / delete a topic, and
  moderate (delete) any comment. Walks the sub-resource id UP to the
  owning ``course_id`` and runs the standard owner-or-grant check
  (``can_manage_course``), same shape as ``quizzes/routers/_deps.py``.
* **Student (learner)** — read topics + comments and post/edit/delete
  their OWN comments. Gated by an *active* enrollment in the owning
  course via ``enrollments.api.public.is_user_enrolled``. A viewer who
  can manage the course also passes (teachers can read their own
  lesson's discussion without being "enrolled").

All cross-feature reads route through the ``courses`` / ``enrollments``
public APIs so the "features are independent" import-linter contract
holds. A caller who fails the gate always gets 404 on the sub-resource
(no existence leak), matching the quizzes/materials perimeter.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import can_manage_course
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.discussions.models import (
    LessonDiscussionComment,
    LessonDiscussionTopic,
)
from abridgeai.features.enrollments.api import public as enrollments_api

_MANAGE_PERM = "course.update"


def not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


async def resolve_lesson_course(db: AsyncSession, lesson_id: UUID) -> UUID | None:
    """Walk ``lesson_id -> module -> course`` and return the course id.

    Reads flow through ``courses.api.public`` (never the courses ORM) so
    the import-linter contract holds. Returns ``None`` when any hop is
    missing / soft-deleted (router maps to 404).
    """
    lesson = await courses_api.get_lesson_by_id(db, lesson_id)
    if lesson is None:
        return None
    module = await courses_api.get_module_by_id(db, lesson.module_id)
    if module is None:
        return None
    return module.course_id


async def can_manage(db: AsyncSession, user: CurrentUser, course_id: UUID) -> bool:
    """True iff the user owns or has ``course.update`` on the course."""
    return await can_manage_course(db, user.user_id, course_id, manage_perm=_MANAGE_PERM)


async def can_view_discussion(db: AsyncSession, user: CurrentUser, course_id: UUID) -> bool:
    """Read gate: an active enrollment OR course-manage rights.

    Teachers reading their own lesson's discussion are not necessarily
    "enrolled", so managers pass without an enrollment row.
    """
    if await enrollments_api.is_user_enrolled(db, student_id=user.user_id, course_id=course_id):
        return True
    return await can_manage(db, user, course_id)


async def get_topic_or_404(db: AsyncSession, topic_id: UUID) -> LessonDiscussionTopic:
    topic = (
        await db.execute(
            select(LessonDiscussionTopic).where(LessonDiscussionTopic.id == topic_id)
        )
    ).scalar_one_or_none()
    if topic is None:
        raise not_found("discussion_topic", topic_id)
    return topic


async def get_comment_or_404(db: AsyncSession, comment_id: UUID) -> LessonDiscussionComment:
    comment = (
        await db.execute(
            select(LessonDiscussionComment).where(LessonDiscussionComment.id == comment_id)
        )
    ).scalar_one_or_none()
    if comment is None:
        raise not_found("discussion_comment", comment_id)
    return comment


__all__ = [
    "can_manage",
    "can_view_discussion",
    "get_comment_or_404",
    "get_topic_or_404",
    "not_found",
    "resolve_lesson_course",
]
