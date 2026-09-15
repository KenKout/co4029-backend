"""Preserve the exact message targeted by a discussion reply.

Revision ID: 0120_discussion_direct_reply
Revises: 0119_program_default_path
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0120_discussion_direct_reply"
down_revision = "0119_program_default_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "lesson_discussion_comments",
        sa.Column(
            "reply_to_comment_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_discussion_comments_reply_to",
        "lesson_discussion_comments",
        "lesson_discussion_comments",
        ["reply_to_comment_id"],
        ["id"],
        ondelete="NO ACTION",
    )
    # Historical rows only retain their flattened parent. Treat that root as
    # the best available direct target; new rows preserve the exact message.
    op.execute(
        "UPDATE lesson_discussion_comments "
        "SET reply_to_comment_id = parent_comment_id "
        "WHERE parent_comment_id IS NOT NULL"
    )
    op.create_index(
        "ix_lesson_discussion_comments_reply_to_comment_id",
        "lesson_discussion_comments",
        ["reply_to_comment_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_lesson_discussion_comments_reply_to_comment_id",
        table_name="lesson_discussion_comments",
    )
    op.drop_constraint(
        "fk_discussion_comments_reply_to",
        "lesson_discussion_comments",
        type_="foreignkey",
    )
    op.drop_column("lesson_discussion_comments", "reply_to_comment_id")
