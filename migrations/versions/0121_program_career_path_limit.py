"""Version the Career Path selection limit per Learning Program.

Revision ID: 0121_program_path_limit
Revises: 0120_discussion_direct_reply
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0121_program_path_limit"
down_revision = "0120_discussion_direct_reply"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "learning_program_versions",
        sa.Column(
            "max_career_paths_per_enrollment",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )

    # Preserve the effective limit previously supplied by runtime settings.
    # Organization overrides win over the global override; installations that
    # never configured the hidden setting retain the original default of one.
    op.execute(
        """
        UPDATE learning_program_versions AS version
        SET max_career_paths_per_enrollment = COALESCE(
            (
                SELECT (setting.setting_value_json #>> '{}')::integer
                FROM system_settings AS setting
                JOIN learning_programs AS program
                  ON program.id = version.learning_program_id
                WHERE setting.setting_key =
                      'learning_program.max_career_paths_per_enrollment'
                  AND setting.organization_id = program.organization_id
                LIMIT 1
            ),
            (
                SELECT (setting.setting_value_json #>> '{}')::integer
                FROM system_settings AS setting
                WHERE setting.setting_key =
                      'learning_program.max_career_paths_per_enrollment'
                  AND setting.organization_id IS NULL
                LIMIT 1
            ),
            1
        )
        """
    )
    op.create_check_constraint(
        "ck_learning_program_versions_career_path_limit",
        "learning_program_versions",
        "max_career_paths_per_enrollment BETWEEN 1 AND 10",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_learning_program_versions_career_path_limit",
        "learning_program_versions",
        type_="check",
    )
    op.drop_column(
        "learning_program_versions",
        "max_career_paths_per_enrollment",
    )
