"""Course-level discussion topics.

Revision ID: 0111_discussion_scope
Revises: 0110_auth_events
Create Date: 2026-09-08 00:00:00.000000

``lesson_discussion_topics.lesson_id`` becomes NULLABLE and gains a sibling
``course_id``, with a CHECK that exactly one is set. A topic is therefore
scoped to a lesson OR to a course, and every comment / moderation /
notification path already written keeps working unchanged — which is why this
widens the existing table instead of adding a parallel
``course_discussion_topics`` that would fork the whole feature (comments, soft
delete, author resolution, notify fan-out) into two copies.

The table keeps its ``lesson_`` name. Renaming it would churn every model,
query, test and generated client type for no behavioural gain; the docstring in
``discussions/models.py`` carries the caveat instead.

NOT included, deliberately: free-form ``@`` mentions. An earlier draft of this
migration also added a ``discussion_comment_mentions`` ledger and a
``course_mention`` notification category, both of which exist to answer "who
did this comment name, and have they been told?" — a question that only arises
when an author can type any name. With mentions limited to direct replies the
target IS ``parent_comment_id``: there is nothing to parse, nothing to gate
against the roster, and nothing to diff on edit, because an edit cannot change
who a reply replies to. Those tables are future work if free-form mentions land.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0111_discussion_scope"
down_revision = "0110_auth_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "lesson_discussion_topics",
        "lesson_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )
    op.add_column(
        "lesson_discussion_topics",
        sa.Column(
            "course_id",
            postgresql.UUID(as_uuid=True),
            # courses carries SoftDeleteMixin -> NO ACTION, house rule.
            sa.ForeignKey("courses.id", ondelete="NO ACTION"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_lesson_discussion_topics_course_id",
        "lesson_discussion_topics",
        ["course_id"],
    )
    # Exactly one scope. Without this a topic could be attached to both (two
    # places to find it, two permission perimeters) or to neither (orphaned,
    # unreachable, and invisible to every list query).
    op.create_check_constraint(
        "ck_lesson_discussion_topics_scope",
        "lesson_discussion_topics",
        "(lesson_id IS NULL) <> (course_id IS NULL)",
    )


def downgrade() -> None:
    # Course-scoped topics have no home once the column is gone, and their
    # comments would be unreachable rather than merely hidden, so they are
    # removed with the column rather than left dangling.
    op.execute(
        "DELETE FROM lesson_discussion_comments WHERE topic_id IN "
        "(SELECT id FROM lesson_discussion_topics WHERE course_id IS NOT NULL)"
    )
    op.execute("DELETE FROM lesson_discussion_topics WHERE course_id IS NOT NULL")
    op.drop_constraint(
        "ck_lesson_discussion_topics_scope",
        "lesson_discussion_topics",
        type_="check",
    )
    op.drop_index(
        "ix_lesson_discussion_topics_course_id",
        table_name="lesson_discussion_topics",
    )
    op.drop_column("lesson_discussion_topics", "course_id")
    op.alter_column(
        "lesson_discussion_topics",
        "lesson_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
