"""Store logical-question groups for multi-angle interview generation.

``all_angles`` yields peer questions that probe one underlying problem through
multiple interviewer angles. Legacy, manual, imported, and role-only questions
remain ungrouped (NULL); no historic relationship is inferred from positions,
outcomes, or prompt similarity.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0088_interview_variant_groups"
down_revision = "0087_lesson_slugs_from_title"
branch_labels = None
depends_on = None


_INDEX_NAME = "ix_interview_questions_variant_group_id"


def upgrade() -> None:
    op.add_column(
        "interview_questions",
        sa.Column(
            "variant_group_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_index(
        _INDEX_NAME,
        "interview_questions",
        ["variant_group_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="interview_questions")
    op.drop_column("interview_questions", "variant_group_id")
