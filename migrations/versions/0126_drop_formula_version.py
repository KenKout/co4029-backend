"""Remove the obsolete career-readiness formula-version stamp.

Revision ID: 0126_drop_formula_version
Revises: 0125_quiz_required_camera
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0126_drop_formula_version"
down_revision = "0125_quiz_required_camera"
branch_labels = None
depends_on = None


def upgrade() -> None:
    rows = op.get_bind().scalar(
        sa.text(
            "SELECT COUNT(*) FROM career_readiness_snapshots "
            "WHERE formula_version IS DISTINCT FROM 2"
        )
    )
    if rows:
        raise RuntimeError(
            f"{rows} readiness snapshot(s) were produced by a formula other than 2. "
            "Dropping the stamp would assert they were not. Resolve the history "
            "question in career-path-progress-formula-cutover-plan.md section 3 first."
        )
    op.drop_column("career_readiness_snapshots", "formula_version")


def downgrade() -> None:
    # The former provenance values cannot be reconstructed after the drop.
    op.execute(
        "ALTER TABLE career_readiness_snapshots "
        "ADD COLUMN formula_version SMALLINT NOT NULL DEFAULT 1"
    )
