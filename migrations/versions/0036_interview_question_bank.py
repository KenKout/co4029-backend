"""Create interview_question_bank_items (course-scoped reusable questions).

Revision ID: 0036_iq_bank
Revises: 0035_gen_run_progress
Create Date: 2026-07-23 00:00:00.000000

A course-scoped pool of reusable interview questions (§QBank-1). Teachers
add generated/authored interview questions into the bank, then copy from
it into any interview config in the same course. Copy semantics: importing
creates a fresh interview_questions row, so the bank item and the imported
question diverge independently (deleting either does not affect the other).

Scope is course-level only. Mirrors the copyable subset of
interview_questions columns; no per-config state (position, review_status,
linked_outcome_id) lives here.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0036_iq_bank"
down_revision = "0035_gen_run_progress"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interview_question_bank_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("course_id", sa.UUID(), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(length=30), nullable=False),
        sa.Column("difficulty", sa.String(length=20), nullable=True),
        sa.Column("model_answer", sa.Text(), nullable=True),
        sa.Column(
            "tags_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("source_config_id", sa.UUID(), nullable=True),
        # TimestampMixin
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        # AuditedByMixin
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column("updated_by", sa.UUID(), nullable=True),
        # SoftDeleteMixin
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by", sa.UUID(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["course_id"], ["courses.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_config_id"], ["interview_configs.id"], ondelete="SET NULL"
        ),
        sa.CheckConstraint(
            "question_type IN ('conceptual', 'behavioral', 'technical', "
            "'situational', 'system_design')",
            name="ck_iq_bank_question_type",
        ),
        sa.CheckConstraint(
            "difficulty IS NULL OR difficulty IN ('junior', 'mid_level', 'senior')",
            name="ck_iq_bank_difficulty",
        ),
    )
    op.create_index(
        "ix_interview_question_bank_items_course_id",
        "interview_question_bank_items",
        ["course_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_interview_question_bank_items_course_id",
        table_name="interview_question_bank_items",
    )
    op.drop_table("interview_question_bank_items")
