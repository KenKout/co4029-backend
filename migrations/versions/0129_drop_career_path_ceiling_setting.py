"""Remove the org ceiling on a program's own career-path limit.

``learning_program.max_career_paths_per_enrollment`` was an organization
guardrail on the per-program limit a manager sets. Its default (10) equalled
the CHECK constraint on
``learning_program_versions.max_career_paths_per_enrollment`` (BETWEEN 1 AND
10), so out of the box it refused nothing while adding a second place the same
number could be rejected — with its own error code and its own message.

The limit that actually protects a student is
``learning_program.max_concurrent_paths_per_student``, which counts paths
across every program. The per-program limits summed, so a student enrolled in
programs capped at 1 and 2 could hold three concurrent paths with each program
inside its own ceiling the whole time; that is the hole the student-wide
setting was added to close, and it is unaffected here.

Rows are deleted rather than left in place because the key is gone from
``SETTINGS_REGISTRY``: the read path only looks up registered keys, and the
write path rejects unregistered ones, so a stored override would be both
invisible and unremovable from the admin UI.

Nothing is dropped from ``learning_program_versions`` — the per-program limit
column and its CHECK constraint stay exactly as they were.

Revision ID: 0129_drop_career_path_ceiling_setting
Revises: 0128_integrity_response_policy
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0129_drop_career_path_ceiling_setting"
down_revision = "0128_integrity_response_policy"
branch_labels = None
depends_on = None

_KEY = "learning_program.max_career_paths_per_enrollment"


def upgrade() -> None:
    bind = op.get_bind()

    # Report what is being discarded. An org that lowered this had a real
    # intent, and the sum-across-programs limit is where that intent now
    # belongs — so the value is worth seeing in the migration log rather than
    # vanishing silently.
    rows = bind.execute(
        sa.text(
            "SELECT organization_id, setting_value_json FROM system_settings "
            "WHERE setting_key = :key"
        ),
        {"key": _KEY},
    ).fetchall()
    for organization_id, value in rows:
        scope = "global" if organization_id is None else f"organization {organization_id}"
        print(f"dropping {_KEY} = {value} ({scope})")  # noqa: T201

    bind.execute(
        sa.text("DELETE FROM system_settings WHERE setting_key = :key"), {"key": _KEY}
    )


def downgrade() -> None:
    # The rows carried per-organization values that cannot be reconstructed.
    # Re-registering the key in SETTINGS_REGISTRY restores the default for
    # every org, which is what an un-set key resolves to anyway.
    pass
