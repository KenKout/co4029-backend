"""Add locale to user_profiles (notification-language support)

Revision ID: 0024_user_profile_locale
Revises: 0023_interview_turn_key_idempotency
Create Date: 2026-07-14 10:00:00.000000

The SPA owns UI language via i18next + localStorage, but server-side
workers (SR due-card scan, remediation dispatch) had no way to learn a
user's language, so every notification ``title``/``body`` was frozen
English. This adds a nullable ``locale`` column on ``user_profiles`` so
the language switch can persist the choice server-side; notification
dispatch reads it to render EN/VI at creation time.

NULL = "not set" -> backend normalizes to 'en' (fallback matches the
frontend ``fallbackLng: "en"``). CHECK constraint enumerates the two
supported locales, mirroring the frontend ``SUPPORTED_LOCALES``; adding
a locale later needs a frontend locale bundle anyway, so a paired
migration is acceptable. No backfill: existing rows stay NULL and keep
today's English behaviour until the user picks a language.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0024_user_profile_locale"
down_revision = "0023_interview_turn_key"
branch_labels = None
depends_on = None

_TABLE = "user_profiles"
_COLUMN = "locale"
_CK = "ck_user_profiles_locale"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(length=5), nullable=True))
    op.create_check_constraint(
        _CK,
        _TABLE,
        "locale IN ('en', 'vi')",
    )


def downgrade() -> None:
    op.drop_constraint(_CK, _TABLE, type_="check")
    op.drop_column(_TABLE, _COLUMN)
