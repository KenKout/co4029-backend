"""Phase 2: teacher-configurable review visibility matrix (default all-true preserves current behaviour).

Revision ID: 0046_quiz_review_options
Revises: 0045_quiz_regrade_tables
Create Date: 2026-07-24

Adds a per-quiz ``review_options`` JSONB matrix controlling what a student may
see after submitting, per time-window (immediately_after / later_while_open /
after_close). Default is all-true so existing quizzes keep today's always-on
review behaviour until a teacher tightens it.

NOTE: the JSON default contains ``:`` characters (``"key":true``). SQLAlchemy's
``sa.text()`` would parse ``:true`` as a bind parameter and blank it to NULL, so
we deliberately use plain-string ``op.execute`` (no bind-param parsing) for every
statement that embeds the JSON literal.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0046_quiz_review_options"
down_revision = "0045_quiz_regrade_tables"
branch_labels = None
depends_on = None

_ALL_TRUE = (
    '{"immediately_after":{"show_score":true,"show_correctness":true,'
    '"show_correct_answers":true,"show_explanation":true,"show_points":true},'
    '"later_while_open":{"show_score":true,"show_correctness":true,'
    '"show_correct_answers":true,"show_explanation":true,"show_points":true},'
    '"after_close":{"show_score":true,"show_correctness":true,'
    '"show_correct_answers":true,"show_explanation":true,"show_points":true}}'
)


def upgrade() -> None:
    # The JSON literal contains ":true" tokens. BOTH sa.text() and op.execute()
    # route through SQLAlchemy's text parser, which treats ":true" as a bind
    # parameter. Use exec_driver_sql() to hand raw SQL straight to psycopg with
    # NO bind-param parsing.
    bind = op.get_bind()
    op.add_column("quizzes", sa.Column("review_options", JSONB, nullable=True))
    bind.exec_driver_sql(
        "UPDATE quizzes SET review_options = '" + _ALL_TRUE + "'::jsonb "
        "WHERE review_options IS NULL"
    )
    bind.exec_driver_sql(
        "ALTER TABLE quizzes ALTER COLUMN review_options "
        "SET DEFAULT '" + _ALL_TRUE + "'::jsonb"
    )
    bind.exec_driver_sql(
        "ALTER TABLE quizzes ALTER COLUMN review_options SET NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("quizzes", "review_options")
