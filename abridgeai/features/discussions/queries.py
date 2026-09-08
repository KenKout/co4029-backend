"""Feature-local ORM reads/writes for the discussions feature.

Every SELECT inherits the global soft-delete loader-criteria filter
(``core.db.soft_delete``), so deleted topics/comments never surface here.
Author display names are resolved through ``identity.api.public`` (the
blessed cross-feature surface) rather than joining the users ORM, keeping
the ``features-are-independent`` import-linter contract clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import func, select

from abridgeai.features.discussions.models import (
    LessonDiscussionComment,
    LessonDiscussionTopic,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_topics_for_lesson(
    db: AsyncSession, lesson_id: UUID
) -> list[LessonDiscussionTopic]:
    """Topics on a lesson, newest first."""
    stmt = (
        select(LessonDiscussionTopic)
        .where(LessonDiscussionTopic.lesson_id == lesson_id)
        .order_by(LessonDiscussionTopic.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_topics_for_course(
    db: AsyncSession, course_id: UUID
) -> list[LessonDiscussionTopic]:
    """Course-wide topics, newest first.

    Only topics attached to the COURSE — a course-level board that also swept
    in every lesson's topics would bury the course-wide announcements it exists
    to hold, and each lesson already shows its own.
    """
    stmt = (
        select(LessonDiscussionTopic)
        .where(LessonDiscussionTopic.course_id == course_id)
        .order_by(LessonDiscussionTopic.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def comment_counts_for_topics(
    db: AsyncSession, topic_ids: list[UUID]
) -> dict[UUID, int]:
    """Batched ``{topic_id: comment_count}`` (no N+1); missing ids → absent."""
    if not topic_ids:
        return {}
    stmt = (
        select(
            LessonDiscussionComment.topic_id,
            func.count(LessonDiscussionComment.id),
        )
        .where(LessonDiscussionComment.topic_id.in_(topic_ids))
        .group_by(LessonDiscussionComment.topic_id)
    )
    rows = (await db.execute(stmt)).all()
    return {row[0]: int(row[1]) for row in rows}


async def mention_counts_for_topics(
    db: AsyncSession, topic_ids: list[UUID], *, viewer_id: UUID
) -> dict[UUID, int]:
    """Batched ``{topic_id: replies-to-me}`` for one viewer.

    A mention is a reply, so "how many times was I named in this topic" is
    "how many comments answer one of mine". Replies the viewer wrote to their
    own comment do not count — self-replying is a normal way to add a thought
    and badging yourself for it would make the marker meaningless.

    Batched for the same reason the comment counts are: a topic list otherwise
    costs one query per row.
    """
    if not topic_ids:
        return {}
    mine = select(LessonDiscussionComment.id).where(
        LessonDiscussionComment.topic_id.in_(topic_ids),
        LessonDiscussionComment.author_id == viewer_id,
    )
    stmt = (
        select(
            LessonDiscussionComment.topic_id,
            func.count(LessonDiscussionComment.id),
        )
        .where(
            LessonDiscussionComment.topic_id.in_(topic_ids),
            LessonDiscussionComment.parent_comment_id.in_(mine),
            LessonDiscussionComment.author_id != viewer_id,
        )
        .group_by(LessonDiscussionComment.topic_id)
    )
    rows = (await db.execute(stmt)).all()
    return {row[0]: int(row[1]) for row in rows}


async def get_topic(db: AsyncSession, topic_id: UUID) -> LessonDiscussionTopic | None:
    stmt = select(LessonDiscussionTopic).where(LessonDiscussionTopic.id == topic_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_comments_for_topic(
    db: AsyncSession, topic_id: UUID
) -> list[LessonDiscussionComment]:
    """Comments on a topic, oldest first (chronological discussion order)."""
    stmt = (
        select(LessonDiscussionComment)
        .where(LessonDiscussionComment.topic_id == topic_id)
        .order_by(LessonDiscussionComment.created_at.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_comment(
    db: AsyncSession, comment_id: UUID
) -> LessonDiscussionComment | None:
    stmt = select(LessonDiscussionComment).where(
        LessonDiscussionComment.id == comment_id
    )
    return (await db.execute(stmt)).scalar_one_or_none()


__all__ = [
    "comment_counts_for_topics",
    "get_comment",
    "get_topic",
    "list_comments_for_topic",
    "mention_counts_for_topics",
    "list_topics_for_course",
    "list_topics_for_lesson",
]
