"""Extend the append-only guard to the assessment audit stores.

Migration 0105 made ``http_audit_log`` and ``system_setting_changes``
append-only at the database level: ``UPDATE`` is refused outright, ``DELETE``
only inside the ``app.audit_maintenance`` scope. Two more stores were already
DOCUMENTED as append-only and had no such guard, so the promise was enforced by
convention only:

* ``assessment_integrity_events`` -- the proctoring log for quiz attempts and
  interview sessions (``assessment_kind`` discriminates). Its whole evidentiary
  value is that a participant cannot retroactively remove a tab-switch, so
  "append-only by docstring" is the weakest possible form of that claim.
* ``quiz_audit_events`` -- the teacher-facing quiz action trail
  (attempt_started / attempt_submitted / attempt_regraded /
  attempt_manually_graded / override_created ...). The router docstring calls it
  "most-recent-first append-only"; nothing stopped an UPDATE.

Reuses ``audit_log_immutable()`` from 0105 rather than defining a second
function, so all four tables share one definition of the rule and a change to
the rule cannot apply to half of them. This migration therefore depends on 0105
having created that function.

Downgrade drops only the two triggers added here; the function stays because
0105 owns it and its other two triggers still need it.
"""

from __future__ import annotations

from alembic import op

revision = "0106_audit_immutability_assess"
down_revision = "0105_audit_immutability"
branch_labels = None
depends_on = None


# Same shape as 0105's APPEND_ONLY_TABLES. Kept as a tuple so the trigger name
# derivation and the downgrade cannot drift apart.
APPEND_ONLY_TABLES: tuple[str, ...] = (
    "assessment_integrity_events",
    "quiz_audit_events",
)


def _trigger(table: str) -> str:
    # Same naming scheme as 0105 so `\d <table>` output reads consistently and
    # tests/support/db_graph.py's catalog probe (which discovers append-only
    # tables by joining pg_trigger to the FUNCTION, not by name) keeps working.
    return f"audit_log_immutable_{table}"


def upgrade() -> None:
    for table in APPEND_ONLY_TABLES:
        # BEFORE, not AFTER: the statement must abort before the heap write.
        op.execute(
            f"CREATE TRIGGER {_trigger(table)} "
            f"BEFORE UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();"
        )


def downgrade() -> None:
    for table in APPEND_ONLY_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {_trigger(table)} ON {table};")
