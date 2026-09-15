"""Add the deadline-safe required-camera quiz policy.

Revision ID: 0125_quiz_required_camera
Revises: 0124_interview_recording_consent
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0125_quiz_required_camera"
down_revision = "0124_interview_recording_consent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quizzes",
        sa.Column(
            "require_camera",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )


def downgrade() -> None:
    op.drop_column("quizzes", "require_camera")
