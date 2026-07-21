"""Add action_url to notifications for deep-link navigation.

Revision ID: 0029_notification_action_url
Revises: 0028_interview_onboarding_steps
Create Date: 2026-07-21 00:00:00.000000

Stores a precomputed relative path (e.g. ``/courses/{slug}/quiz/{id}``)
built by the producing feature at creation time, which already holds the
full routing context (course slug, quiz id, ...). The frontend renders it
as a "Take action" link, killing the class of dead-link bugs that arise
when a single ``entity_id`` can't express a nested route.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0029_notification_action_url"
down_revision = "0028_interview_onboarding_steps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "notifications",
        sa.Column("action_url", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("notifications", "action_url")
