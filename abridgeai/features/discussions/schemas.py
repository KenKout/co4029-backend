"""Pydantic v2 DTOs for the discussions feature.

Learner + authoring share these read shapes; the write shapes are split
by actor (students create comments; teachers create topics).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── Author identity (shared by topics + comments) ─────────────────────────


class DiscussionCommentAuthor(BaseModel):
    """Minimal author identity shown beside a topic or a comment.

    Declared above the topic shapes because ``DiscussionTopicRead`` embeds it;
    the name keeps its original ``Comment`` spelling so existing importers and
    the generated client types do not churn.

    ``avatar_url`` is a short-lived presigned GET URL minted per request by
    :mod:`abridgeai.features.discussions.authors` — never a raw bucket/key.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    display_name: str | None = None
    avatar_url: str | None = None


# ── Topics ────────────────────────────────────────────────────────────────


class DiscussionTopicCreate(BaseModel):
    """Teacher payload to open a new topic.

    Scope is not in the body — it comes from the route the payload is posted
    to (``/lessons/{id}/...`` or ``/courses/{id}/...``), so a client cannot
    claim one scope in the URL and another in the JSON.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=255)
    body_markdown: str | None = Field(default=None, max_length=20_000)


class DiscussionTopicUpdate(BaseModel):
    """Teacher payload to edit a topic or toggle its open/closed status."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=255)
    body_markdown: str | None = Field(default=None, max_length=20_000)
    status: str | None = Field(default=None, pattern="^(open|closed)$")


class DiscussionTopicRead(_ORMModel):
    """A discussion topic + its comment count.

    ``comment_count`` is filled by the query layer (batched), not stored.
    ``can_manage`` is set per-request by the router so the client knows
    whether to show edit/close/delete controls — never persisted.

    ``author`` is the resolved identity behind ``created_by`` (display name +
    presigned avatar), so the client can attribute a topic without a second
    round-trip per row. ``None`` when ``created_by`` is NULL.
    """

    id: UUID
    #: Exactly one of these is set — the topic's scope. A course-scoped topic
    #: has ``lesson_id`` NULL and vice versa (CHECK-enforced).
    lesson_id: UUID | None = None
    course_id: UUID | None = None
    title: str
    body_markdown: str | None = None
    status: str
    created_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    comment_count: int = 0
    #: Replies addressed to the VIEWER in this topic — what the red badge
    #: counts. Per-request, never persisted, and 0 for a viewer who has not
    #: commented (nobody can have replied to them).
    mention_count: int = 0
    can_manage: bool = False
    author: DiscussionCommentAuthor | None = None


class DiscussionTopicList(BaseModel):
    """Envelope for a topic list (lesson-scoped or course-scoped).

    ``can_manage`` rides at the top level so the client knows whether to
    show the teacher's "post a topic" affordance even when ``topics`` is
    empty (a bare list can't carry that signal for zero rows).
    """

    can_manage: bool = False
    topics: list[DiscussionTopicRead] = Field(default_factory=list)


# ── Comments ──────────────────────────────────────────────────────────────


class DiscussionCommentCreate(BaseModel):
    """Student/teacher payload to post a comment on a topic.

    ``parent_comment_id`` makes the comment a REPLY, and a reply is the only
    way to name someone: the client prefixes the body with the parent author's
    handle, and the notification goes to that author. There is deliberately no
    free-form mention field — see migration 0111's docstring.
    """

    model_config = ConfigDict(extra="forbid")

    body: str = Field(min_length=1, max_length=5_000)
    #: The comment being replied to. Must belong to the same topic; a reply to
    #: a reply is flattened onto the top-level parent by the router, so threads
    #: stay one level deep.
    parent_comment_id: UUID | None = None


class DiscussionCommentUpdate(BaseModel):
    """Author payload to edit their own comment body."""

    model_config = ConfigDict(extra="forbid")

    body: str = Field(min_length=1, max_length=5_000)


class DiscussionCommentRead(_ORMModel):
    """One comment, with a resolved author summary and viewer affordances.

    ``is_own`` / ``can_delete`` are set per-request by the router (a
    student can edit/delete their own comment; a course manager can delete
    any). They are never persisted.
    """

    id: UUID
    topic_id: UUID
    author_id: UUID
    body: str
    parent_comment_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    author: DiscussionCommentAuthor | None = None
    is_own: bool = False
    can_delete: bool = False


__all__ = [
    "DiscussionCommentAuthor",
    "DiscussionCommentCreate",
    "DiscussionCommentRead",
    "DiscussionCommentUpdate",
    "DiscussionTopicCreate",
    "DiscussionTopicList",
    "DiscussionTopicRead",
    "DiscussionTopicUpdate",
]
