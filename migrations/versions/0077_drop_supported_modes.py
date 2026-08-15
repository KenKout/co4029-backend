"""Drop ``interview_configs.supported_modes``.

Every interview now runs the native LiveKit agent in one room where the
candidate speaks OR types (mic is an in-room toggle, turns ride ``lk.chat``).
The text/hybrid/voice config split — and the lobby mode picker, the
voice-screen runtime and the text-REST transport it selected between — is
gone, so the column that selected it goes too.

``interview_sessions.input_mode`` STAYS: it records what a session actually
ran for analytics, and historical rows keep their value. New sessions are
always written as ``hybrid``.

The check constraint is dropped by LOOKUP rather than by name: deployments
where the table predates the explicit ``ck_*`` naming carry the
PostgreSQL-generated ``interview_configs_supported_modes_check`` instead, and
a hard-coded ``DROP CONSTRAINT`` fails there.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0077_drop_supported_modes"
down_revision = "0076_version_fk_cascade"
branch_labels = None
depends_on = None

_TABLE = "interview_configs"


def _drop_mode_checks(conn: sa.Connection) -> list[str]:
    """Drop every CHECK constraint touching ``supported_modes``; return names."""
    query = sa.text(
        "SELECT conname FROM pg_constraint c "
        "JOIN pg_class t ON t.oid = c.conrelid "
        "JOIN pg_namespace n ON n.oid = t.relnamespace "
        "WHERE c.contype = 'c' AND t.relname = :table AND n.nspname = 'public' "
        "AND pg_get_constraintdef(c.oid) ILIKE '%supported_modes%'"
    )
    names = [row[0] for row in conn.execute(query, {"table": _TABLE})]
    for name in names:
        conn.execute(sa.text(f'ALTER TABLE {_TABLE} DROP CONSTRAINT "{name}"'))
    return names


def upgrade() -> None:
    conn = op.get_bind()
    _drop_mode_checks(conn)
    op.drop_column(_TABLE, "supported_modes")


def downgrade() -> None:
    op.add_column(
        _TABLE,
        sa.Column(
            "supported_modes",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'hybrid'"),
        ),
    )
    op.create_check_constraint(
        "ck_interview_configs_supported_modes",
        _TABLE,
        "supported_modes IN ('voice', 'text', 'hybrid')",
    )

