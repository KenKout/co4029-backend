"""Path-change review → student notifications (cross-feature).

A Faculty Dean's moves on a path-change request are invisible to the student who
filed it unless we say so. Three moments are worth a notification, and they map
one-to-one onto the request's non-terminal → terminal transitions:

* **in progress** — the dean opened the request and is checking the student's
  record. Nothing has changed; this exists purely to answer "has anyone looked
  at this?", which is the question a silent queue provokes.
* **rejected** — with the structured reason (and the dean's own words when they
  picked ``other``), because "rejected" with no reason is what makes a student
  re-file the same request.
* **approved** — the switch already happened when this fires.

All three go out under the ``path_change_review`` category so a student gets one
preference toggle for "decisions about my path change" rather than per-event
switches.

Dispatch goes through the blessed cross-feature surfaces
(:mod:`notifications.api.public`, :mod:`identity.api.public`) so the
import-linter feature-independence contract holds. Failures are swallowed and
logged: a notification must never roll back or 500 the review decision that
triggered it — a dean's approval is an academic record change, and losing it
because an inbox row could not be written would be far worse than a missing
notification. ``send_notification`` does not commit; the writes join the
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

_CATEGORY = "path_change_review"

_ACTION_URL = "/learning-programs"
"""Where the student reviews their programs, requests, and rejection history.

Deliberately not a career-path deep link: after a rejection the interesting
screen is their own request record, not the path they were refused.
"""


async def notify_path_change_in_progress(
    db: AsyncSession,
    *,
    student_user_id: UUID,
    request_id: UUID,
    program_name: str,
    target_path_name: str,
    arq_pool: object | None = None,
) -> None:
    """Tell the student their request was picked up for review."""
    try:
        locale = await get_user_locale(db, student_user_id)
        await notifications_api.send_notification(
            db,
            recipient_user_id=student_user_id,
            notification_type=_CATEGORY,
            title=notifications_api.path_change_in_progress_title(
                program_name=program_name, locale=locale
            ),
            body=notifications_api.path_change_in_progress_body(
                target_path_name=target_path_name, locale=locale
            ),
            entity_type="path_change_request",
            entity_id=request_id,
            action_url=_ACTION_URL,
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001 — never break the review on a notify failure
        _logger.exception(
            "path_change_in_progress_notify_failed",
            student_user_id=str(student_user_id),
            request_id=str(request_id),
        )


async def notify_path_change_rejected(
    db: AsyncSession,
    *,
    student_user_id: UUID,
    request_id: UUID,
    program_name: str,
    target_path_name: str,
    reason_code: str,
    reason_detail: str | None,
    arq_pool: object | None = None,
) -> None:
    """Tell the student their request was rejected, and why."""
    try:
        locale = await get_user_locale(db, student_user_id)
        await notifications_api.send_notification(
            db,
            recipient_user_id=student_user_id,
            notification_type=_CATEGORY,
            title=notifications_api.path_change_rejected_title(
                program_name=program_name, locale=locale
            ),
            body=notifications_api.path_change_rejected_body(
                target_path_name=target_path_name,
                reason_code=reason_code,
                reason_detail=reason_detail,
                locale=locale,
            ),
            entity_type="path_change_request",
            entity_id=request_id,
            action_url=_ACTION_URL,
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001
        _logger.exception(
            "path_change_rejected_notify_failed",
            student_user_id=str(student_user_id),
            request_id=str(request_id),
        )


async def notify_path_change_approved(
    db: AsyncSession,
    *,
    student_user_id: UUID,
    request_id: UUID,
    program_name: str,
    target_path_name: str,
    arq_pool: object | None = None,
) -> None:
    """Tell the student the switch was approved and has taken effect."""
    try:
        locale = await get_user_locale(db, student_user_id)
        await notifications_api.send_notification(
            db,
            recipient_user_id=student_user_id,
            notification_type=_CATEGORY,
            title=notifications_api.path_change_approved_title(
                program_name=program_name, locale=locale
            ),
            body=notifications_api.path_change_approved_body(
                target_path_name=target_path_name, locale=locale
            ),
            entity_type="path_change_request",
            entity_id=request_id,
            action_url=_ACTION_URL,
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001
        _logger.exception(
            "path_change_approved_notify_failed",
            student_user_id=str(student_user_id),
            request_id=str(request_id),
        )


__all__ = [
    "notify_path_change_approved",
    "notify_path_change_in_progress",
    "notify_path_change_rejected",
]
