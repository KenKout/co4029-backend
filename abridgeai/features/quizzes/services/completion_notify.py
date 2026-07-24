"""Teacher completion/failure notifications for quiz AI generation (T7.6).

When a teacher kicks off quiz question generation, the pipeline runs
asynchronously in an ARQ worker (``run_quiz_generation_task``) and can take
seconds to minutes across its stages (retrieval → ideation → generation →
validation → dedup → persistence). Without a signal the teacher has to sit and
poll the run status. This module dispatches a ``quiz_generation`` notification
— success OR failure — to the teacher who initiated the run, via the existing
``send_notification`` surface (in-app always-on; email preference-gated + async).

Design notes
------------
* Best-effort: the entrypoint swallows its own errors so a notification write
  can never break (or roll back) the generation transaction it rides on — same
  contract as the SR remediation dispatcher and the material completion notify.
* Recipient is ``GenerationRun.requested_by`` (the teacher who started the run).
* ``action_url`` is the TEACHER quiz-manage page so "Take action" jumps straight
  to the freshly generated questions.
* Cross-feature Postgres reads go through raw ``text(...)`` — no foreign feature
  ORM imports, keeping the import-linter contracts satisfied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.observability import get_logger
from abridgeai.features.notifications.services.dispatch import send_notification

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

_CATEGORY = "quiz_generation"
_ENTITY_TYPE = "quiz"


async def notify_quiz_generation_outcome(
    db: AsyncSession,
    *,
    recipient_user_id: UUID | None,
    course_id: UUID | None,
    quiz_id: UUID | None,
    quiz_title: str | None,
    succeeded: bool,
    error_message: str | None = None,
    arq_pool: object | None = None,
) -> None:
    """Dispatch a ``quiz_generation`` notification to the initiating teacher.

    Best-effort — any failure is logged and swallowed so it can never break the
    generation transaction this rides on. Skips silently when there is no
    recipient (system-initiated run) or no quiz/course to deep-link to.
    """
    try:
        if recipient_user_id is None or quiz_id is None or course_id is None:
            _logger.warning(
                "quiz_gen_notify_skipped_missing_target",
                recipient=str(recipient_user_id),
                quiz_id=str(quiz_id),
                course_id=str(course_id),
            )
            return

        title_text = quiz_title or "Quiz"
        action_url = f"/teacher/courses/{course_id}/quizzes/{quiz_id}"

        if succeeded:
            title = "Quiz questions ready"
            body = (
                f'AI generation finished for "{title_text}". '
                "Open the quiz to review the new questions."
            )
        else:
            title = "Quiz generation failed"
            detail = f" ({error_message})" if error_message else ""
            body = (
                f'AI generation failed for "{title_text}"{detail}. '
                "Open the quiz to try again."
            )

        await send_notification(
            db,
            recipient_user_id=recipient_user_id,
            notification_type=_CATEGORY,
            title=title,
            body=body,
            entity_type=_ENTITY_TYPE,
            entity_id=quiz_id,
            action_url=action_url,
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001 — best-effort; never break the caller.
        _logger.exception(
            "quiz_gen_notify_failed",
            quiz_id=str(quiz_id),
            succeeded=succeeded,
        )


__all__ = ["notify_quiz_generation_outcome"]
