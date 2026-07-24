"""Discussions feature router — lesson-level topics + comments.

Two perimeters share one router (see ``deps.py``):

* Teachers who can manage the course open / edit / close / delete topics
  and may delete (moderate) any comment.
* Enrolled students read topics + comments and post / edit / delete their
  OWN comments.

Read + comment endpoints gate on ``can_view_discussion`` (active
enrollment OR course-manage). Topic-authoring + moderation endpoints gate
on ``can_manage``. Every failed gate returns 404 on the sub-resource (no
existence leak), matching the quizzes/materials perimeter.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.discussions import deps, queries
from abridgeai.features.discussions.models import (
    LessonDiscussionComment,
    LessonDiscussionTopic,
)
from abridgeai.features.discussions.schemas import (
    DiscussionCommentAuthor,
    DiscussionCommentCreate,
    DiscussionCommentRead,
    DiscussionCommentUpdate,
    DiscussionTopicCreate,
    DiscussionTopicList,
    DiscussionTopicRead,
    DiscussionTopicUpdate,
)
from abridgeai.features.identity.api import public as identity_api

router = APIRouter(tags=["discussions"])


# ── helpers ────────────────────────────────────────────────────────────────


async def _require_lesson_view(
    db: AsyncSession, lesson_id: UUID, user: CurrentUser
) -> UUID:
    """Resolve the lesson's course and assert the viewer may see it."""
    course_id = await deps.resolve_lesson_course(db, lesson_id)
    if course_id is None:
        raise deps.not_found("lesson", lesson_id)
    if not await deps.can_view_discussion(db, user, course_id):
        raise deps.not_found("lesson", lesson_id)
    return course_id


async def _require_lesson_manage(
    db: AsyncSession, lesson_id: UUID, user: CurrentUser
) -> UUID:
    course_id = await deps.resolve_lesson_course(db, lesson_id)
    if course_id is None:
        raise deps.not_found("lesson", lesson_id)
    if not await deps.can_manage(db, user, course_id):
        raise deps.not_found("lesson", lesson_id)
    return course_id


async def _topic_course_or_404(
    db: AsyncSession, topic: LessonDiscussionTopic
) -> UUID:
    course_id = await deps.resolve_lesson_course(db, topic.lesson_id)
    if course_id is None:
        raise deps.not_found("discussion_topic", topic.id)
    return course_id


def _topic_read(
    topic: LessonDiscussionTopic, *, comment_count: int, can_manage: bool
) -> DiscussionTopicRead:
    return DiscussionTopicRead(
        id=topic.id,
        lesson_id=topic.lesson_id,
        title=topic.title,
        body_markdown=topic.body_markdown,
        status=topic.status,
        created_by=topic.created_by,
        created_at=topic.created_at,
        updated_at=topic.updated_at,
        comment_count=comment_count,
        can_manage=can_manage,
    )


def _comment_read(
    comment: LessonDiscussionComment,
    *,
    author: DiscussionCommentAuthor | None,
    is_own: bool,
    can_delete: bool,
) -> DiscussionCommentRead:
    return DiscussionCommentRead(
        id=comment.id,
        topic_id=comment.topic_id,
        author_id=comment.author_id,
        body=comment.body,
        parent_comment_id=comment.parent_comment_id,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
        author=author,
        is_own=is_own,
        can_delete=can_delete,
    )


# ── topics ─────────────────────────────────────────────────────────────────


@router.get(
    "/lessons/{lesson_id}/discussion/topics",
    response_model=DiscussionTopicList,
)
async def list_lesson_topics(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscussionTopicList:
    """Topics on a lesson (enrolled students + course managers).

    Returns an envelope so ``can_manage`` is available even when there are
    zero topics — the client needs it to decide whether to show the
    teacher's "post a topic" affordance.
    """
    course_id = await _require_lesson_view(db, lesson_id, current_user)
    viewer_can_manage = await deps.can_manage(db, current_user, course_id)
    topics = await queries.list_topics_for_lesson(db, lesson_id)
    counts = await queries.comment_counts_for_topics(db, [t.id for t in topics])
    return DiscussionTopicList(
        can_manage=viewer_can_manage,
        topics=[
            _topic_read(
                t,
                comment_count=counts.get(t.id, 0),
                can_manage=viewer_can_manage,
            )
            for t in topics
        ],
    )


@router.post(
    "/lessons/{lesson_id}/discussion/topics",
    response_model=DiscussionTopicRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_lesson_topic(
    lesson_id: UUID,
    payload: DiscussionTopicCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscussionTopicRead:
    """Open a new topic on a lesson (course managers only)."""
    await _require_lesson_manage(db, lesson_id, current_user)
    topic = LessonDiscussionTopic(
        lesson_id=lesson_id,
        title=payload.title,
        body_markdown=payload.body_markdown,
        status="open",
        created_by=current_user.user_id,
    )
    db.add(topic)
    await db.flush()
    await db.commit()
    await db.refresh(topic)
    return _topic_read(topic, comment_count=0, can_manage=True)


@router.patch(
    "/discussion/topics/{topic_id}",
    response_model=DiscussionTopicRead,
)
async def update_topic(
    topic_id: UUID,
    payload: DiscussionTopicUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscussionTopicRead:
    """Edit a topic or toggle open/closed (course managers only)."""
    topic = await deps.get_topic_or_404(db, topic_id)
    course_id = await _topic_course_or_404(db, topic)
    if not await deps.can_manage(db, current_user, course_id):
        raise deps.not_found("discussion_topic", topic_id)
    if payload.title is not None:
        topic.title = payload.title
    if payload.body_markdown is not None:
        topic.body_markdown = payload.body_markdown
    if payload.status is not None:
        topic.status = payload.status
    await db.commit()
    await db.refresh(topic)
    counts = await queries.comment_counts_for_topics(db, [topic.id])
    return _topic_read(topic, comment_count=counts.get(topic.id, 0), can_manage=True)


@router.delete(
    "/discussion/topics/{topic_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_topic(
    topic_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a topic and its comments (course managers only)."""
    topic = await deps.get_topic_or_404(db, topic_id)
    course_id = await _topic_course_or_404(db, topic)
    if not await deps.can_manage(db, current_user, course_id):
        raise deps.not_found("discussion_topic", topic_id)
    # Soft-delete the topic; the recursive soft-delete walker stamps the
    # ONETOMANY comments (relationship cascade) with deleted_at.
    from abridgeai.core.db.recursive_delete import soft_delete_cascade  # noqa: PLC0415

    await soft_delete_cascade(db, topic, actor_id=current_user.user_id)
    await db.commit()


# ── comments ─────────────────────────────────────────────────────────────


@router.get(
    "/discussion/topics/{topic_id}/comments",
    response_model=list[DiscussionCommentRead],
)
async def list_topic_comments(
    topic_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DiscussionCommentRead]:
    """Comments on a topic (enrolled students + course managers)."""
    topic = await deps.get_topic_or_404(db, topic_id)
    course_id = await _topic_course_or_404(db, topic)
    if not await deps.can_view_discussion(db, current_user, course_id):
        raise deps.not_found("discussion_topic", topic_id)
    viewer_can_manage = await deps.can_manage(db, current_user, course_id)
    comments = await queries.list_comments_for_topic(db, topic_id)
    authors = await identity_api.get_users_by_ids(
        db, [c.author_id for c in comments]
    )
    result: list[DiscussionCommentRead] = []
    for c in comments:
        dto = authors.get(c.author_id)
        author = (
            DiscussionCommentAuthor(id=c.author_id, display_name=dto.display_name)
            if dto is not None
            else DiscussionCommentAuthor(id=c.author_id)
        )
        is_own = c.author_id == current_user.user_id
        result.append(
            _comment_read(
                c,
                author=author,
                is_own=is_own,
                can_delete=is_own or viewer_can_manage,
            )
        )
    return result


@router.post(
    "/discussion/topics/{topic_id}/comments",
    response_model=DiscussionCommentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_comment(
    topic_id: UUID,
    payload: DiscussionCommentCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscussionCommentRead:
    """Post a comment (enrolled students + course managers).

    Rejects with 404 when the topic is closed for viewers who cannot
    manage the course — a closed topic accepts no new student comments.
    """
    topic = await deps.get_topic_or_404(db, topic_id)
    course_id = await _topic_course_or_404(db, topic)
    if not await deps.can_view_discussion(db, current_user, course_id):
        raise deps.not_found("discussion_topic", topic_id)
    viewer_can_manage = await deps.can_manage(db, current_user, course_id)
    if topic.status == "closed" and not viewer_can_manage:
        raise deps.not_found("discussion_topic", topic_id)
    comment = LessonDiscussionComment(
        topic_id=topic_id,
        author_id=current_user.user_id,
        body=payload.body,
    )
    db.add(comment)
    await db.flush()
    await db.commit()
    await db.refresh(comment)
    dto = (await identity_api.get_users_by_ids(db, [current_user.user_id])).get(
        current_user.user_id
    )
    author = (
        DiscussionCommentAuthor(id=current_user.user_id, display_name=dto.display_name)
        if dto is not None
        else DiscussionCommentAuthor(id=current_user.user_id)
    )
    return _comment_read(comment, author=author, is_own=True, can_delete=True)


@router.patch(
    "/discussion/comments/{comment_id}",
    response_model=DiscussionCommentRead,
)
async def update_comment(
    comment_id: UUID,
    payload: DiscussionCommentUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscussionCommentRead:
    """Edit a comment body (author only)."""
    comment = await deps.get_comment_or_404(db, comment_id)
    if comment.author_id != current_user.user_id:
        raise deps.not_found("discussion_comment", comment_id)
    comment.body = payload.body
    await db.commit()
    await db.refresh(comment)
    dto = (await identity_api.get_users_by_ids(db, [comment.author_id])).get(
        comment.author_id
    )
    author = (
        DiscussionCommentAuthor(id=comment.author_id, display_name=dto.display_name)
        if dto is not None
        else DiscussionCommentAuthor(id=comment.author_id)
    )
    return _comment_read(comment, author=author, is_own=True, can_delete=True)


@router.delete(
    "/discussion/comments/{comment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_comment(
    comment_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete a comment (author, or a course manager moderating)."""
    comment = await deps.get_comment_or_404(db, comment_id)
    is_own = comment.author_id == current_user.user_id
    if not is_own:
        topic = await deps.get_topic_or_404(db, comment.topic_id)
        course_id = await _topic_course_or_404(db, topic)
        if not await deps.can_manage(db, current_user, course_id):
            raise deps.not_found("discussion_comment", comment_id)
    from abridgeai.core.db.recursive_delete import soft_delete_cascade  # noqa: PLC0415

    await soft_delete_cascade(db, comment, actor_id=current_user.user_id)
    await db.commit()


__all__ = ["router"]
