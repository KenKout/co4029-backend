"""Guarantee one opening and one closing ceremony turn per interview.

Revision ID: 0026_interview_ceremony_turns
Revises: 0025_interview_security_guard
Create Date: 2026-07-19 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

revision = "0026_interview_ceremony_turns"
down_revision = "0025_interview_security_guard"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX uq_interview_session_messages_ceremony
        ON interview_session_messages (session_id, (metadata_json->>'ceremony_key'))
        WHERE role = 'ai' AND metadata_json ? 'ceremony_key'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_interview_session_messages_ceremony")
