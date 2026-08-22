"""Course lifecycle → learner/teacher notifications (cross-feature).

Emits in-app (and, when an ARQ pool is supplied, email) notifications when a
course relationship changes in a way the recipient should know about:

* a teacher is assigned to a **published** course,
* a student is enrolled in a **published** course,
* a course that already has teachers/students assigned is **published**.

Only *published* courses trigger the assignment/enrolment notifications: being
attached to a draft is not yet actionable for the recipient (they can't open a
course they can't see). Publishing is the moment a draft becomes visible, so it
back-fills notifications to everyone already attached.

Each notification carries an ``action_url`` deep-link so the inbox row opens the
relevant course directly:

* students  → ``/courses/{slug}/learn`` (the learner "continue learning" route),
* teachers  → ``/teacher/courses/{id}`` (the authoring workspace).

Dispatch goes through the blessed cross-feature surfaces
(:mod:`notifications.api.public`, :mod:`identity.api.public`) so the
import-linter feature-independence contract holds. Failures are swallowed and
logged: a notification must never roll back or 500 the course operation that
triggered it. ``send_notification`` does not commit — the writes join the
caller's transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.observability import get_logger
from abridgeai.features.identity.api.public import get_user_locale
from abridgeai.features.notifications.api import public as notifications_api

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

_CATEGORY = "course_announcement"


def _student_action_url(course_slug: str) -> str:
    return f"/courses/{course_slug}/learn"


def _teacher_action_url(course_id: UUID) -> str:
    return f"/teacher/courses/{course_id}"


async def notify_teacher_assigned(
    db: AsyncSession,
    *,
    teacher_user_id: UUID,
    course_id: UUID,
    course_title: str,
    arq_pool: object | None = None,
) -> None:
    """Tell a teacher they were assigned to a published course."""
    try:
        locale = await get_user_locale(db, teacher_user_id)
        await notifications_api.send_notification(
            db,
            recipient_user_id=teacher_user_id,
            notification_type=_CATEGORY,
            title=notifications_api.course_teacher_assigned_title(
                course_title=course_title, locale=locale
            ),
            body=notifications_api.course_teacher_assigned_body(
                course_title=course_title, locale=locale
            ),
            entity_type="course",
            entity_id=course_id,
            action_url=_teacher_action_url(course_id),
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001 — never break the assignment on a notify failure
        _logger.exception(
            "course_teacher_assigned_notify_failed",
            teacher_user_id=str(teacher_user_id),
            course_id=str(course_id),
        )


async def notify_student_enrolled(
    db: AsyncSession,
    *,
    student_user_id: UUID,
    course_id: UUID,
    course_title: str,
    course_slug: str,
    arq_pool: object | None = None,
) -> None:
    """Tell a student they were enrolled in a published course."""
    try:
        locale = await get_user_locale(db, student_user_id)
        await notifications_api.send_notification(
            db,
            recipient_user_id=student_user_id,
            notification_type=_CATEGORY,
            title=notifications_api.course_enrolled_title(course_title=course_title, locale=locale),
            body=notifications_api.course_enrolled_body(course_title=course_title, locale=locale),
            entity_type="course",
            entity_id=course_id,
            action_url=_student_action_url(course_slug),
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001
        _logger.exception(
            "course_student_enrolled_notify_failed",
            student_user_id=str(student_user_id),
            course_id=str(course_id),
        )


async def notify_course_published(
    db: AsyncSession,
    *,
    course_id: UUID,
    course_title: str,
    course_slug: str,
    teacher_user_ids: list[UUID],
    student_user_ids: list[UUID],
    arq_pool: object | None = None,
) -> None:
    """Notify all attached teachers + enrolled students that a course published.

    Each recipient is handled independently; one bad row does not stop the rest.
    """
    for teacher_id in teacher_user_ids:
        try:
            locale = await get_user_locale(db, teacher_id)
            await notifications_api.send_notification(
                db,
                recipient_user_id=teacher_id,
                notification_type=_CATEGORY,
                title=notifications_api.course_published_teacher_title(
                    course_title=course_title, locale=locale
                ),
                body=notifications_api.course_published_teacher_body(
                    course_title=course_title, locale=locale
                ),
                entity_type="course",
                entity_id=course_id,
                action_url=_teacher_action_url(course_id),
                arq_pool=arq_pool,
            )
        except Exception:  # noqa: BLE001
            _logger.exception(
                "course_published_teacher_notify_failed",
                teacher_user_id=str(teacher_id),
                course_id=str(course_id),
            )

    for student_id in student_user_ids:
        try:
            locale = await get_user_locale(db, student_id)
            await notifications_api.send_notification(
                db,
                recipient_user_id=student_id,
                notification_type=_CATEGORY,
                title=notifications_api.course_published_student_title(
                    course_title=course_title, locale=locale
                ),
                body=notifications_api.course_published_student_body(
                    course_title=course_title, locale=locale
                ),
                entity_type="course",
                entity_id=course_id,
                action_url=_student_action_url(course_slug),
                arq_pool=arq_pool,
            )
        except Exception:  # noqa: BLE001
            _logger.exception(
                "course_published_student_notify_failed",
                student_user_id=str(student_id),
                course_id=str(course_id),
            )


async def notify_syllabus_import_succeeded(
    db: AsyncSession,
    *,
    manager_user_id: UUID,
    course_id: UUID,
    course_title: str,
    outcome_count: int,
    warnings: list[str],
    arq_pool: object | None = None,
) -> None:
    """Tell the importing manager their syllabus became a draft course.

    Deep-links to the authoring workspace rather than the learner route:
    the course is a draft, so there is nothing for a learner to open yet
    and the manager's next step is reviewing what was imported.
    """
    try:
        locale = await get_user_locale(db, manager_user_id)
        await notifications_api.send_notification(
            db,
            recipient_user_id=manager_user_id,
            notification_type=_CATEGORY,
            title=notifications_api.syllabus_import_succeeded_title(
                course_title=course_title, locale=locale
            ),
            body=notifications_api.syllabus_import_succeeded_body(
                course_title=course_title,
                outcome_count=outcome_count,
                warnings=warnings,
                locale=locale,
            ),
            entity_type="course",
            entity_id=course_id,
            action_url=_teacher_action_url(course_id),
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001 — never fail an import over its notification
        _logger.exception(
            "syllabus_import_succeeded_notify_failed",
            manager_user_id=str(manager_user_id),
            course_id=str(course_id),
        )


async def notify_syllabus_import_failed(
    db: AsyncSession,
    *,
    manager_user_id: UUID,
    import_id: UUID,
    filename: str | None,
    reason: str,
    arq_pool: object | None = None,
) -> None:
    """Tell the importing manager why no course came out of their upload.

    ``entity_id`` is the import attempt, not a course — a failed import has
    no course to point at, and the attempt row is what holds the file and
    the reason.
    """
    try:
        locale = await get_user_locale(db, manager_user_id)
        await notifications_api.send_notification(
            db,
            recipient_user_id=manager_user_id,
            notification_type=_CATEGORY,
            title=notifications_api.syllabus_import_failed_title(
                filename=filename, locale=locale
            ),
            body=notifications_api.syllabus_import_failed_body(reason=reason, locale=locale),
            entity_type="course",
            entity_id=import_id,
            action_url="/dept/courses",
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001
        _logger.exception(
            "syllabus_import_failed_notify_failed",
            manager_user_id=str(manager_user_id),
            import_id=str(import_id),
        )


__all__ = [
    "notify_course_published",
    "notify_student_enrolled",
    "notify_syllabus_import_failed",
    "notify_syllabus_import_succeeded",
    "notify_teacher_assigned",
]
