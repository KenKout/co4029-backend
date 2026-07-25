"""Teacher completion/failure notifications for interview AI generation.

When a teacher kicks off interview question generation, the pipeline runs
asynchronously in an ARQ worker (``run_interview_generation_task``) and can take
seconds to minutes across its stages (retrieval → generation → validation →
persistence). Without a signal the teacher has to sit and poll the run status.
This module dispatches an ``interview_generation`` notification — success OR
failure — to the teacher who initiated the run, via the existing
``send_notification`` surface (in-app always-on; email preference-gated + async).

Mirrors :mod:`abridgeai.features.quizzes.services.completion_notify`.

Design notes
------------
* Best-effort: the entrypoint swallows its own errors so a notification write
  can never break (or roll back) the generation transaction it rides on — same
  contract as the quiz completion notify.
* Recipient is ``GenerationRun.requested_by`` (the teacher who started the run).
* ``action_url`` is the TEACHER interview-config page so "Take action" jumps
  straight to the freshly generated questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.observability import get_logger
from abridgeai.features.notifications.services.dispatch import send_notification

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

_CATEGORY = "interview_generation"
_ENTITY_TYPE = "interview"


async def notify_interview_generation_outcome(
    db: AsyncSession,
    *,
    recipient_user_id: UUID | None,
    course_id: UUID | None,
    config_id: UUID | None,
    interview_title: str | None,
    succeeded: bool,
    error_message: str | None = None,
    arq_pool: object | None = None,
) -> None:
    """Dispatch an ``interview_generation`` notification to the initiating teacher.

    Best-effort — any failure is logged and swallowed so it can never break the
    generation transaction this rides on. Skips silently when there is no
    recipient (system-initiated run) or no interview/course to deep-link to.
    """
    try:
        if recipient_user_id is None or config_id is None or course_id is None:
            _logger.warning(
                "interview_gen_notify_skipped_missing_target",
                recipient=str(recipient_user_id),
                config_id=str(config_id),
                course_id=str(course_id),
            )
            return

        title_text = interview_title or "Interview"
        action_url = f"/teacher/courses/{course_id}/interview-configs/{config_id}"

        if succeeded:
            title = "Interview questions ready"
            body = (
                f'AI generation finished for "{title_text}". '
                "Open the interview to review the new questions."
            )
        else:
            title = "Interview generation failed"
            detail = f" ({error_message})" if error_message else ""
            body = (
                f'AI generation failed for "{title_text}"{detail}. '
                "Open the interview to try again."
            )

        await send_notification(
            db,
            recipient_user_id=recipient_user_id,
            notification_type=_CATEGORY,
            title=title,
            body=body,
            entity_type=_ENTITY_TYPE,
            entity_id=config_id,
            action_url=action_url,
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001 — best-effort; never break the caller.
        _logger.exception(
            "interview_gen_notify_failed",
            config_id=str(config_id),
            succeeded=succeeded,
        )


__all__ = ["notify_interview_generation_outcome"]
