"""Revoke entitlements left behind by a 0131 reconciliation.

``0131`` closed duplicate active attempts by setting ``status = 'switched_out'``
and stamping ``exit_snapshot``. Closing an attempt through the service does
more than that: ``revoke_path_entitlements`` also stamps ``revoked_at`` on the
``course_enrollment_entitlements`` rows the attempt granted, and drops any
course enrollment nothing else still grants. The migration skipped it, so a
reconciled attempt could leave live entitlement rows pointing at an attempt
that has ended.

What that does and does not cost:

* Course ACCESS is not wrongly extended. Both attempts were on the same career
  path, so the surviving attempt grants the same courses through its own
  entitlement rows. The drop below is guarded on no other live entitlement
  existing, which is exactly the surviving attempt's.
* ``release_program_path_access`` is deliberately NOT called. It fires only
  when no other active attempt holds the path, and a reconciliation closes the
  loser precisely because one does.

So the defect is stale rows referencing a dead attempt: an entitlement audit
reads them as live grants, and anything that later revokes the SURVIVOR would
find them and keep access alive past the point it should end.

Written as a follow-up rather than folded into 0131 because 0131 is already
applied. This one is idempotent and scoped to rows 0131 marked, so it is a
no-op in any database that had no duplicates — which includes every
environment where the reconciliation reported nothing.

The SQL is a copy of ``queries.revoke_path_entitlements`` rather than a call
to it: a migration cannot await an AsyncSession, and pinning the statements
here is correct anyway — a migration must keep doing what it did when it ran,
even after that function changes.

Revision ID: 0132_revoke_reconciled
Revises: 0131_student_scoped_active_path
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0132_revoke_reconciled"
down_revision = "0131_student_scoped_active_path"
branch_labels = None
depends_on = None

_RECONCILED_BY = "0131_student_scoped_active_path"


def upgrade() -> None:
    bind = op.get_bind()

    reconciled = bind.execute(
        sa.text(
            """
            SELECT id FROM program_path_attempts
            WHERE exit_snapshot ->> 'reconciled_by' = :marker
            """
        ),
        {"marker": _RECONCILED_BY},
    ).scalars().all()

    if not reconciled:
        print(  # noqa: T201
            "no attempts were reconciled by 0131 in this database; nothing to revoke"
        )
        return

    affected = bind.execute(
        sa.text(
            """
            UPDATE course_enrollment_entitlements
            SET revoked_at = NOW()
            WHERE source_type = 'path_attempt'
              AND source_id = ANY(:attempt_ids)
              AND revoked_at IS NULL
            RETURNING course_enrollment_id
            """
        ),
        {"attempt_ids": list(reconciled)},
    ).scalars().all()

    print(  # noqa: T201
        f"revoked {len(affected)} entitlement row(s) across "
        f"{len(reconciled)} reconciled attempt(s)"
    )

    if not affected:
        return

    # Drop only what nothing else grants. A course the surviving attempt also
    # entitles keeps a live row of its own and is left alone -- the student
    # must not lose access to a course they are still entitled to because the
    # duplicate grant was cleaned up.
    dropped = bind.execute(
        sa.text(
            """
            UPDATE course_enrollments ce
            SET status = 'dropped', dropped_at = NOW(), updated_at = NOW()
            WHERE ce.id = ANY(:enrollment_ids)
              AND ce.status = 'active'
              AND NOT EXISTS (
                  SELECT 1 FROM course_enrollment_entitlements live
                  WHERE live.course_enrollment_id = ce.id
                    AND live.revoked_at IS NULL
              )
            RETURNING ce.id
            """
        ),
        {"enrollment_ids": list(set(affected))},
    ).scalars().all()

    for enrollment_id in dropped:
        print(  # noqa: T201
            f"dropped course enrollment {enrollment_id}: no live entitlement remains"
        )


def downgrade() -> None:
    # Un-revoking would re-grant access on the strength of an attempt that has
    # ended, which is the state this migration exists to remove. The rows keep
    # their revoked_at timestamp and the reason is still on the attempt's
    # exit_snapshot.
    pass
