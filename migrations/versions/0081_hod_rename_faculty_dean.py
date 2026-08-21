"""Rename the HOD role's display name from "Head of Department" to "Faculty Dean".

Migration 0004 already seeded the catalog on existing databases, so editing
``role_seeds.yaml`` alone does not reach them — this migration updates the
``roles.name`` cell for the ``hod`` role idempotently.

The permission/behaviour keys off the role ``code`` (``hod``), which is
unchanged; only the human-facing display name is renamed.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0081_hod_rename_faculty_dean"
down_revision = "0080_drop_course_level_columns"
branch_labels = None
depends_on = None

_NEW_NAME = "Faculty Dean"
_OLD_NAME = "Head of Department"
_ROLE_CODE = "hod"


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        text(
            "UPDATE roles SET name = :new, updated_at = NOW() "
            "WHERE code = :code AND name = :old"
        ),
        {"new": _NEW_NAME, "code": _ROLE_CODE, "old": _OLD_NAME},
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        text(
            "UPDATE roles SET name = :old, updated_at = NOW() "
            "WHERE code = :code AND name = :new"
        ),
        {"new": _NEW_NAME, "code": _ROLE_CODE, "old": _OLD_NAME},
    )
