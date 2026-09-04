"""Lesson-discussion comment notifications (cross-feature).

A new comment on a discussion topic is a push moment for two audiences:

* **thread participants** — everyone who commented before (any role) is
  following the exchange and should see the new reply;
* **course teachers** — when the comment comes from a STUDENT, the course
  managers (owner + assigned teachers) must know a student is asking
  something, without having to poll the lesson page.

Sent under the ``course_discussion`` category so users get ONE preference
toggle for lesson-discussion activity, distinct from course announcements.

Dispatch goes through the blessed cross-feature surfaces
(:mod:`notifications.api.public`, :mod:`courses.api.public`,
:mod:`identity.api.public`) so the import-linter feature-independence
contract holds. Failures are swallowed and logged: a notification must
never roll back or 500 the comment that triggered it. ``send_notification``
does not commit; the writes join the caller's transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.observability import get_logger
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.discussions import queries
from abridgeai.features.discussions.models import (
    LessonDiscussionComment,
    LessonDiscussionTopic,
)
from abridgeai.features.identity.api import public as identity_api
from abridgeai.features.notifications.api import public as notifications_api

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

_CATEGORY = "course_discussion"

# Where the comment lives: the lesson's discussion panel on the course learn
# page. Both students and course teachers can open it (teachers pass the
# discussion gate via course-manage rights).
#
# ``&tab=discussion`` is load-bearing, not decoration: the learn page defaults
# to the Lesson Notes tab, so without it the recipient landed on the right
# LESSON but had to find and click "Discussion" to reach the comment the
# notification was about. The learn route reads the param and opens that tab
# (see the SPA's LearnSearch.tab); an unknown value falls back to the default
# tab, so this link degrades safely if the tab set is ever renamed.
_ACTION_URL = "/courses/{course_slug}/learn?item={lesson_slug}&tab=discussion"

_SNIPPET_LEN = 120


async def notify_comment_participants(
    db: AsyncSession,
    *,
    comment: LessonDiscussionComment,
    topic: LessonDiscussionTopic,
    actor_id: UUID,
    actor_can_manage: bool,
    arq_pool: object | None = None,
) -> None:
    """Fan out the "new comment" notification to thread + teachers."""
    try:
        lesson = await courses_api.get_lesson_by_id(db, topic.lesson_id)
        if lesson is None:
            return
        module = await courses_api.get_module_by_id(db, lesson.module_id)
        if module is None:
            return
        course = await courses_api.get_course_by_id(db, module.course_id)
        if course is None:
            return

        commenter = await identity_api.get_user_by_id(db, actor_id)
        if commenter is not None:
            commenter_label = (commenter.display_name or "").strip() or commenter.primary_email
        else:
            commenter_label = str(actor_id)

        snippet = (comment.body or "").strip().replace("\n", " ")
        if len(snippet) > _SNIPPET_LEN:
            snippet = f"{snippet[:_SNIPPET_LEN]}…"

        # Thread participants: everyone who already commented (the fresh
        # comment is included by the list query; drop the author).
        recipients = {
            row.author_id for row in await queries.list_comments_for_topic(db, topic.id)
        }
        recipients.discard(actor_id)

        # Student comment -> course teachers must see it.
        if not actor_can_manage:
            recipients.update(
                uid
                for uid in await courses_api.list_course_manager_ids(db, module.course_id)
                if uid != actor_id
            )

        action_url = _ACTION_URL.format(course_slug=course.slug, lesson_slug=lesson.slug)
        for recipient_id in recipients:
            try:
                locale = await identity_api.get_user_locale(db, recipient_id)
                await notifications_api.send_notification(
                    db,
                    recipient_user_id=recipient_id,
                    notification_type=_CATEGORY,
                    title=notifications_api.discussion_comment_title(
                        topic_title=topic.title, locale=locale
                    ),
                    body=notifications_api.discussion_comment_body(
                        commenter_label=commenter_label,
                        topic_title=topic.title,
                        course_title=course.title,
                        comment_snippet=snippet,
                        locale=locale,
                    ),
                    entity_type="lesson_discussion_comment",
                    entity_id=comment.id,
                    action_url=action_url,
                    arq_pool=arq_pool,
                )
            except Exception:  # noqa: BLE001 — never break the comment on a notify failure
                _logger.exception(
                    "discussion_comment_notify_failed",
                    recipient_user_id=str(recipient_id),
                    comment_id=str(comment.id),
                )
    except Exception:  # noqa: BLE001 — context resolution must never 500 the comment
        _logger.exception(
            "discussion_comment_notify_context_failed",
            comment_id=str(comment.id),
        )


__all__ = ["notify_comment_participants"]
