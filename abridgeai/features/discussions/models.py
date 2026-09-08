"""Discussions feature ORM models.

Course discussion: a teacher who can manage the course posts a **topic**
(open questions / prompt) either on one lesson or on the course as a whole;
enrolled students post **comments** to discuss it.

Two tables (migrations 0039 + 0111):

* ``lesson_discussion_topics`` — one row per teacher-authored topic. Exactly
  one of ``lesson_id`` / ``course_id`` is set (CHECK
  ``ck_lesson_discussion_topics_scope``), which is what makes a topic
  lesson-scoped or course-scoped. ``status`` gates whether new comments are
  accepted (``open`` / ``closed``).
* ``lesson_discussion_comments`` — one row per student/teacher comment on a
  topic. ``parent_comment_id`` is the one-level reply thread: a reply carries
  its parent, and the client renders the pair. It doubles as the mention
  target — a reply is the only way to name someone, so who was mentioned is
  the parent's author rather than anything parsed out of the body.
The ``lesson_`` prefix on the first two tables predates course-scoped topics
and is now a misnomer. It is kept deliberately: renaming would churn every
model, query, test and generated client type to buy nothing at runtime.

Mixin policy mirrors the content aggregate (quizzes / lessons): both
tables are content the author can edit and soft-delete, so both carry
``UUIDPrimaryKeyMixin`` + ``TimestampMixin`` + ``AuditedByMixin`` +
``SoftDeleteMixin``. Soft-delete is enforced globally by the loader
criteria listener (``core.db.soft_delete``) — no per-table registration.

FK on-delete policy (see ``scripts/audit_fks.py``): parents that carry
``SoftDeleteMixin`` are soft-deleted via UPDATE, never hard-deleted on
the happy path, so FKs into them use ``NO ACTION`` (not CASCADE) to match
the house rule — the recursive soft-delete walker stamps children.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class LessonDiscussionTopic(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """A teacher-authored topic, scoped to one lesson OR to the whole course."""

    __tablename__ = "lesson_discussion_topics"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'closed')",
            name="ck_lesson_discussion_topics_status",
        ),
        # Exactly one scope. Both set would put the topic in two places with
        # two permission perimeters; neither set would orphan it.
        CheckConstraint(
            "(lesson_id IS NULL) <> (course_id IS NULL)",
            name="ck_lesson_discussion_topics_scope",
        ),
    )

    lesson_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        # lessons carries SoftDeleteMixin → NO ACTION (soft-deleted via UPDATE).
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        nullable=True,
        index=True,
    )
    #: Set instead of ``lesson_id`` for a course-wide topic.
    course_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body_markdown: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'open'")
    )

    comments: Mapped[list[LessonDiscussionComment]] = relationship(
        back_populates="topic",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class LessonDiscussionComment(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """A single comment on a discussion topic (student or teacher)."""

    __tablename__ = "lesson_discussion_comments"

    topic_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        # topics carries SoftDeleteMixin → NO ACTION (soft-deleted via UPDATE).
        ForeignKey("lesson_discussion_topics.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    # Author of the comment. NO ACTION mirrors the house policy for FKs into
    # users (never hard-deleted on the happy path).
    author_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # One-level reply thread: a reply names the comment it answers. Replies
    # are never nested further — the client renders parent + children, and a
    # reply to a reply is stored against the same top-level parent, so a thread
    # cannot grow into a tree nobody can follow on a phone.
    parent_comment_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lesson_discussion_comments.id", ondelete="NO ACTION"),
        nullable=True,
        index=True,
    )

    topic: Mapped[LessonDiscussionTopic] = relationship(back_populates="comments")


__all__ = [
    "LessonDiscussionComment",
    "LessonDiscussionTopic",
]
