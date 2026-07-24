"""Phase 11: shared question bank — categories, tags, tag-map + question category_id.

Revision ID: 0055_question_bank_cats
Revises: 0054_quiz_stats_cache
Create Date: 2026-07-24

Additive/reversible DDL only (new tables + nullable/server-defaulted columns) so
the running app — whose ORM does not yet map these — keeps working.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0055_question_bank_cats"
down_revision = "0054_quiz_stats_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "question_categories",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("context_key", sa.String(100), nullable=False),
        sa.Column("parent_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_by", PGUUID(as_uuid=True), nullable=True),
        sa.Column("updated_by", PGUUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by", PGUUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["parent_id"], ["question_categories.id"], ondelete="NO ACTION"),
        sa.UniqueConstraint("context_key", "parent_id", "name", name="uq_question_categories_name"),
    )
    op.create_index("ix_question_categories_context_key", "question_categories", ["context_key"])
    op.create_table(
        "question_tags",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("context_key", sa.String(100), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("context_key", "name", name="uq_question_tags_name"),
    )
    op.create_index("ix_question_tags_context_key", "question_tags", ["context_key"])
    op.create_table(
        "question_tag_map",
        sa.Column("question_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("tag_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["question_id"], ["quiz_questions.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["tag_id"], ["question_tags.id"], ondelete="NO ACTION"),
        sa.PrimaryKeyConstraint("question_id", "tag_id"),
        sa.UniqueConstraint("question_id", "tag_id", name="uq_question_tag_map"),
    )
    op.add_column("quiz_questions", sa.Column("category_id", PGUUID(as_uuid=True), nullable=True))
    op.create_foreign_key("fk_quiz_questions_category", "quiz_questions", "question_categories", ["category_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    op.drop_constraint("fk_quiz_questions_category", "quiz_questions", type_="foreignkey")
    op.drop_column("quiz_questions", "category_id")
    op.drop_table("question_tag_map")
    op.drop_index("ix_question_tags_context_key", table_name="question_tags")
    op.drop_table("question_tags")
    op.drop_index("ix_question_categories_context_key", table_name="question_categories")
    op.drop_table("question_categories")
