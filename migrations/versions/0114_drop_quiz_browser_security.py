"""Drop the vestigial ``quizzes.browser_security`` proctoring toggle.

Fullscreen is now MANDATORY for every quiz attempt, exactly as it already was
for every interview — the client gate no longer reads any per-quiz setting.
That leaves this column with no reader anywhere in the system, and a teacher
toggle labelled "Browser security mode" that changed nothing was the same
class of defect as the bug it was retired over: a control that reports a
policy it does not enforce.

The interview side is the precedent. ``interview_configs`` has never had an
equivalent on/off switch — only integrity *weights* — because a proctoring
gate an author can silently disable is not a gate.

Nothing is preserved on the way out. The column only ever carried 'none' or
'securewindow' (``ck_quizzes_browser_security``), and 'securewindow' now
describes the behaviour of EVERY quiz, so there is no distinction left to
keep. The downgrade recreates the column at its original default; a restored
value would be wrong anyway, since the client no longer consults it.

Revision ID: 0114_drop_quiz_browser_security
Revises: 0113_integrity_policy
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0114_drop_quiz_browser_security"
down_revision = "0113_integrity_policy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The CHECK goes first: dropping the column it guards would take it along,
    # but naming it here keeps the intent explicit and the downgrade symmetric.
    op.drop_constraint("ck_quizzes_browser_security", "quizzes", type_="check")
    op.drop_column("quizzes", "browser_security")


def downgrade() -> None:
    op.add_column(
        "quizzes",
        sa.Column(
            "browser_security",
            sa.String(20),
            nullable=False,
            server_default=sa.text("'none'"),
        ),
    )
    op.create_check_constraint(
        "ck_quizzes_browser_security",
        "quizzes",
        "browser_security IN ('none', 'securewindow')",
    )
