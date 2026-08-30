"""Store logical-question groups in the interview question bank.

Revision ID: 0092_iq_bank_variant_groups
Revises: 0091_strict_role_only
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0092_iq_bank_variant_groups"
down_revision = "0091_strict_role_only"
branch_labels = None
depends_on = None


_INDEX_NAME = "ix_interview_question_bank_items_variant_group_id"
_UNIQUE_INDEX_NAME = "uq_iq_bank_live_group_angle"
_CHECK_NAME = "ck_iq_bank_variant_group_angle"


def upgrade() -> None:
    op.add_column(
        "interview_question_bank_items",
        sa.Column("variant_group_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        _INDEX_NAME,
        "interview_question_bank_items",
        ["variant_group_id"],
        unique=False,
    )
    op.create_check_constraint(
        _CHECK_NAME,
        "interview_question_bank_items",
        "variant_group_id IS NULL OR question_type IN "
        "('technical', 'system_design', 'situational', 'behavioral')",
    )
    op.create_index(
        _UNIQUE_INDEX_NAME,
        "interview_question_bank_items",
        ["course_id", "variant_group_id", "question_type"],
        unique=True,
        postgresql_where=sa.text("variant_group_id IS NOT NULL AND deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(_UNIQUE_INDEX_NAME, table_name="interview_question_bank_items")
    op.drop_constraint(_CHECK_NAME, "interview_question_bank_items", type_="check")
    op.drop_index(_INDEX_NAME, table_name="interview_question_bank_items")
    op.drop_column("interview_question_bank_items", "variant_group_id")
