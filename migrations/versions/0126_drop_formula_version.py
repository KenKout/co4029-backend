"""Remove the obsolete career-readiness formula-version stamp.

Revision ID: 0126_drop_career_readiness_formula_version
Revises: 0125_quiz_required_camera
"""

from __future__ import annotations

from alembic import op

revision = "0126_drop_formula_version"
down_revision = "0125_quiz_required_camera"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("career_readiness_snapshots", "formula_version")


def downgrade() -> None:
    # The former provenance values cannot be reconstructed after the drop.
    op.execute(
        "ALTER TABLE career_readiness_snapshots "
        "ADD COLUMN formula_version SMALLINT NOT NULL DEFAULT 1"
    )
