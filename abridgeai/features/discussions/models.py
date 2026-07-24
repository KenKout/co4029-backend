"""Discussions feature ORM models.

Lesson-level discussion: a teacher who can manage the course posts a
**topic** (open questions / prompt) on a lesson; enrolled students post
**comments** to discuss it.

Two tables (migration 0039):

* ``lesson_discussion_topics`` — one row per teacher-authored topic,
  attached to a lesson. ``status`` gates whether new comments are
  accepted (``open`` / ``closed``).
* ``lesson_discussion_comments`` — one row per student/teacher comment
  on a topic. ``parent_comment_id`` is a nullable self-FK reserved for a
  future one-level reply thread; v1 renders flat (the column exists so
  threading is a later add with no migration).

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
    """A teacher-authored discussion topic attached to a lesson."""

    __tablename__ = "lesson_discussion_topics"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'closed')",
            name="ck_lesson_discussion_topics_status",
        ),
    )

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        # lessons carries SoftDeleteMixin → NO ACTION (soft-deleted via UPDATE).
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        nullable=False,
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
    # Reserved for a future one-level reply thread. Nullable self-FK; v1
    # renders flat and never sets this. Kept so threading is a later add
    # without a schema migration.
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
