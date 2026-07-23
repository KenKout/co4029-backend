"""Add server default uuid_generate_v4() to interview_question_bank_items.id.

Revision ID: 0037_iq_bank_id_def
Revises: 0036_iq_bank
Create Date: 2026-07-23 00:00:00.000000

Migration 0036 created the ``id`` column without a server default, but the
``UUIDPrimaryKeyMixin`` ORM model expects the DB to generate the primary key
(``server_default=uuid_generate_v4()``). Inserts therefore omitted ``id`` and
hit a NotNullViolation. This backfills the server default so the "Add to
question bank" POST (and every other insert) works.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0037_iq_bank_id_def"
down_revision = "0036_iq_bank"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "interview_question_bank_items",
        "id",
        server_default=sa.text("uuid_generate_v4()"),
    )


def downgrade() -> None:
    op.alter_column(
        "interview_question_bank_items",
        "id",
        server_default=None,
    )
