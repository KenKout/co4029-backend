"""Public, typed cross-feature dispatch API for the notifications feature.

This module re-exports :func:`send_notification` and the related
:data:`EMAIL_NOTIFICATION_TASK_NAME` constant from
:mod:`abridgeai.features.notifications.services.dispatch` so cross-
feature producers (SR remediation + scan-due-cards today; quizzes,
interviews, enrollments tomorrow) can import them via the blessed
``api.public`` path. The current ``ignore_imports`` exceptions for
``spaced_repetition.services.remediation -> notifications.services.dispatch``
and ``spaced_repetition.workers.scan_due_cards -> notifications.services.dispatch``
collapse once those call sites are migrated to this surface in Wave 5.

The plan body referenced ``dispatch_remediation_for_card_failure`` --
the actual callable is the more general :func:`send_notification`,
which the SR card-failure path already invokes with
``notification_type='spaced_repetition'``.

No new logic lives here; this is a typed pass-through. Tests cover the
re-export identity + that the public name resolves to the same callable
as the underlying implementation.
"""

from __future__ import annotations

from abridgeai.features.notifications import messages
from abridgeai.features.notifications.services.dispatch import (
    EMAIL_NOTIFICATION_TASK_NAME,
    send_notification,
)

# Re-export the localized message builders so cross-feature producers render
# notification copy through the blessed public surface instead of importing
# ``notifications.messages`` directly (which the independence contract forbids —
# only ``api.public`` is a sanctioned cross-feature import target).
course_teacher_assigned_title = messages.course_teacher_assigned_title
course_teacher_assigned_body = messages.course_teacher_assigned_body
course_enrolled_title = messages.course_enrolled_title
course_enrolled_body = messages.course_enrolled_body
course_published_teacher_title = messages.course_published_teacher_title
course_published_teacher_body = messages.course_published_teacher_body
course_published_student_title = messages.course_published_student_title
course_published_student_body = messages.course_published_student_body
syllabus_import_succeeded_title = messages.syllabus_import_succeeded_title
syllabus_import_succeeded_body = messages.syllabus_import_succeeded_body
syllabus_import_failed_title = messages.syllabus_import_failed_title
syllabus_import_failed_body = messages.syllabus_import_failed_body

__all__ = [
    "EMAIL_NOTIFICATION_TASK_NAME",
    "course_enrolled_body",
    "course_enrolled_title",
    "course_published_student_body",
    "course_published_student_title",
    "course_published_teacher_body",
    "course_published_teacher_title",
    "course_teacher_assigned_body",
    "course_teacher_assigned_title",
    "send_notification",
    "syllabus_import_failed_body",
    "syllabus_import_failed_title",
    "syllabus_import_succeeded_body",
    "syllabus_import_succeeded_title",
]
