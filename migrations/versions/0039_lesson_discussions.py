"""Lesson discussion topics + comments (teacher-posted topics, student comments).

Revision ID: 0039_lesson_discuss
Revises: 0038_iq_src_modules
Create Date: 2026-07-24 00:00:00.000000

Adds two tables backing lesson-level discussion:

* ``lesson_discussion_topics`` — a teacher who can manage the course posts a
  topic (title + optional markdown body, e.g. open questions) on a lesson.
  ``status`` gates whether new comments are accepted ('open' / 'closed').
* ``lesson_discussion_comments`` — enrolled students (and teachers) comment
  on a topic. ``parent_comment_id`` is a nullable self-FK reserved for a
  future one-level reply thread (v1 renders flat).

Both tables carry the content mixins (id / created_at / updated_at /
created_by / updated_by / deleted_at / deleted_by) matching the courses +
quizzes aggregate. FKs into soft-deleted parents (lessons, topics, users)
use NO ACTION per the house rule (parents are soft-deleted via UPDATE; the
recursive soft-delete walker stamps children).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0039_lesson_discuss"
down_revision = "0038_iq_src_modules"
branch_labels = None
depends_on = None


def _audit_and_timestamp_columns() -> list[sa.Column]:
    """The shared TimestampMixin + AuditedByMixin + SoftDeleteMixin columns."""
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by", postgresql.UUID(as_uuid=True), nullable=True),
    ]


def upgrade() -> None:
    op.create_table(
        "lesson_discussion_topics",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("lesson_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("body_markdown", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'open'"),
            nullable=False,
        ),
        *_audit_and_timestamp_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_lesson_discussion_topics"),
        sa.ForeignKeyConstraint(
            ["lesson_id"],
            ["lessons.id"],
            name="fk_lesson_discussion_topics_lesson_id",
            ondelete="NO ACTION",
        ),
        sa.CheckConstraint(
            "status IN ('open', 'closed')",
            name="ck_lesson_discussion_topics_status",
        ),
    )
    op.create_index(
        "ix_lesson_discussion_topics_lesson_id",
        "lesson_discussion_topics",
        ["lesson_id"],
    )

    op.create_table(
        "lesson_discussion_comments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("author_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("parent_comment_id", postgresql.UUID(as_uuid=True), nullable=True),
        *_audit_and_timestamp_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_lesson_discussion_comments"),
        sa.ForeignKeyConstraint(
            ["topic_id"],
            ["lesson_discussion_topics.id"],
            name="fk_lesson_discussion_comments_topic_id",
            ondelete="NO ACTION",
        ),
        sa.ForeignKeyConstraint(
            ["author_id"],
            ["users.id"],
            name="fk_lesson_discussion_comments_author_id",
            ondelete="NO ACTION",
        ),
        sa.ForeignKeyConstraint(
            ["parent_comment_id"],
            ["lesson_discussion_comments.id"],
            name="fk_lesson_discussion_comments_parent_comment_id",
            ondelete="NO ACTION",
        ),
    )
    op.create_index(
        "ix_lesson_discussion_comments_topic_id",
        "lesson_discussion_comments",
        ["topic_id"],
    )
    op.create_index(
        "ix_lesson_discussion_comments_author_id",
        "lesson_discussion_comments",
        ["author_id"],
    )
    op.create_index(
        "ix_lesson_discussion_comments_parent_comment_id",
        "lesson_discussion_comments",
        ["parent_comment_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_lesson_discussion_comments_parent_comment_id",
        table_name="lesson_discussion_comments",
    )
    op.drop_index(
        "ix_lesson_discussion_comments_author_id",
        table_name="lesson_discussion_comments",
    )
    op.drop_index(
        "ix_lesson_discussion_comments_topic_id",
        table_name="lesson_discussion_comments",
    )
    op.drop_table("lesson_discussion_comments")
    op.drop_index(
        "ix_lesson_discussion_topics_lesson_id",
        table_name="lesson_discussion_topics",
    )
    op.drop_table("lesson_discussion_topics")
