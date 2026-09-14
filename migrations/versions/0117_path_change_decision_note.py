"""Separate path-change rejection notes from custom reasons.

Revision ID: 0117_path_change_note
Revises: 0116_career_path_thumbnails
Create Date: 2026-09-14 00:00:00.000000

Predefined rejection categories already express the reason. Historically their
optional dean note was stored in ``decision_reason`` as well. Move that text to
its own nullable column; ``decision_reason`` is then reserved for the custom
reason required by the ``other`` category.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0117_path_change_note"
down_revision = "0116_career_path_thumbnails"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "path_change_requests",
        sa.Column("decision_note", sa.Text(), nullable=True),
    )
    op.execute(
        """
        UPDATE path_change_requests
        SET decision_note = decision_reason,
            decision_reason = NULL
        WHERE decision_reason_code IS NOT NULL
          AND decision_reason_code <> 'other'
          AND decision_reason IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE path_change_requests
        SET decision_reason = decision_note
        WHERE decision_reason IS NULL
          AND decision_note IS NOT NULL
        """
    )
    op.drop_column("path_change_requests", "decision_note")
