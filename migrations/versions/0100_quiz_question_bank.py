"""Create the course-scoped curated Quiz Question Bank.

Revision ID: 0100_quiz_question_bank
Revises: 0099_ic_cooldown_minutes
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0100_quiz_question_bank"
down_revision = "0099_ic_cooldown_minutes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_question_bank_items",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("course_id", sa.UUID(), nullable=False),
        sa.Column("learning_outcome_id", sa.UUID(), nullable=True),
        sa.Column("source_question_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.String(20), server_default=sa.text("'draft'"), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("question_type", sa.String(30), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("hint_text", sa.Text(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("difficulty", sa.String(20), nullable=True),
        sa.Column("bloom_level", sa.String(20), nullable=True),
        sa.Column("expected_response_time_ms", sa.Integer(), nullable=True),
        sa.Column("expected_ef_ceiling", sa.Numeric(4, 2), nullable=True),
        sa.Column(
            "source_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "original_generated_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("prompt_format", sa.String(10), server_default=sa.text("'plain'"), nullable=False),
        sa.Column("hint_format", sa.String(10), server_default=sa.text("'plain'"), nullable=False),
        sa.Column(
            "explanation_format", sa.String(10), server_default=sa.text("'plain'"), nullable=False
        ),
        sa.Column("single_answer", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("answer_numbering", sa.String(16), server_default=sa.text("'abc'"), nullable=False),
        sa.Column("numeric_answer", sa.Numeric(18, 6), nullable=True),
        sa.Column("numeric_tolerance", sa.Numeric(18, 6), server_default=sa.text("0"), nullable=True),
        sa.Column("match_pairs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("match_distractors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ordering_sequence", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("category_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column("updated_by", sa.UUID(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by", sa.UUID(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["learning_outcome_id"], ["course_learning_outcomes.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["source_question_id"], ["quiz_questions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["category_id"], ["question_categories.id"], ondelete="SET NULL"),
        sa.CheckConstraint(
            "question_type IN ('multiple_choice', 'true_false', 'short_answer', "
            "'fill_blank', 'code', 'numerical', 'matching', 'ordering')",
            name="ck_quiz_bank_question_type",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'approved', 'archived')", name="ck_quiz_bank_status"
        ),
        sa.CheckConstraint(
            "difficulty IS NULL OR difficulty IN ('easy', 'medium', 'hard')",
            name="ck_quiz_bank_difficulty",
        ),
        sa.CheckConstraint(
            "bloom_level IS NULL OR bloom_level IN "
            "('remember', 'understand', 'apply', 'analyze', 'evaluate', 'create')",
            name="ck_quiz_bank_bloom_level",
        ),
    )
    op.create_index("ix_quiz_bank_course_id", "quiz_question_bank_items", ["course_id"])
    op.create_index("ix_quiz_bank_learning_outcome_id", "quiz_question_bank_items", ["learning_outcome_id"])
    op.create_index("ix_quiz_bank_content_hash", "quiz_question_bank_items", ["content_hash"])
    op.create_index(
        "uq_quiz_bank_live_content",
        "quiz_question_bank_items",
        ["course_id", "content_hash"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL AND status <> 'archived'"),
    )

    op.create_table(
        "quiz_question_bank_options",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("bank_item_id", sa.UUID(), nullable=False),
        sa.Column("option_key", sa.String(5), nullable=False),
        sa.Column("option_text", sa.Text(), nullable=False),
        sa.Column("is_correct", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("option_format", sa.String(10), server_default=sa.text("'plain'"), nullable=False),
        sa.Column("grade_fraction", sa.Numeric(5, 4), nullable=True),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.Column("feedback_format", sa.String(16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column("updated_by", sa.UUID(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by", sa.UUID(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["bank_item_id"], ["quiz_question_bank_items.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("bank_item_id", "option_key", name="uq_quiz_bank_options_key"),
        sa.UniqueConstraint("bank_item_id", "position", name="uq_quiz_bank_options_position"),
    )
    op.create_index("ix_quiz_bank_options_item_id", "quiz_question_bank_options", ["bank_item_id"])

    op.add_column("quiz_questions", sa.Column("imported_from_bank_item_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        "fk_quiz_questions_imported_bank_item",
        "quiz_questions",
        "quiz_question_bank_items",
        ["imported_from_bank_item_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_quiz_questions_imported_from_bank_item_id",
        "quiz_questions",
        ["imported_from_bank_item_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_quiz_questions_imported_from_bank_item_id", table_name="quiz_questions")
    op.drop_constraint(
        "fk_quiz_questions_imported_bank_item", "quiz_questions", type_="foreignkey"
    )
    op.drop_column("quiz_questions", "imported_from_bank_item_id")
    op.drop_index("ix_quiz_bank_options_item_id", table_name="quiz_question_bank_options")
    op.drop_table("quiz_question_bank_options")
    op.drop_index("uq_quiz_bank_live_content", table_name="quiz_question_bank_items")
    op.drop_index("ix_quiz_bank_content_hash", table_name="quiz_question_bank_items")
    op.drop_index("ix_quiz_bank_learning_outcome_id", table_name="quiz_question_bank_items")
    op.drop_index("ix_quiz_bank_course_id", table_name="quiz_question_bank_items")
    op.drop_table("quiz_question_bank_items")
