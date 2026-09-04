"""Enforce audit-trail immutability at the database level (FR-6.7)

Revision ID: 0105_audit_immutability
Revises: 0104_policy_academic_bodies
Create Date: 2026-09-04 00:00:00.000000

Why
---
FR-6.7 requires an *immutable* audit log covering role assignments, sensitive
data changes, and HTTP request events. Every piece of that trail already
exists and is queryable through ``/admin/audit/*`` -- but "immutable" was only
a convention. Migration 0011 called ``http_audit_log`` append-only in a
docstring and 0089 said the same for ``system_setting_changes``; neither
table had anything stopping an ``UPDATE`` or ``DELETE``. An audit record that
the audited party can silently rewrite is not evidence, so the guarantee has
to live below the application, where a compromised service account or a stray
``psql`` session still hits it.

Two distinct shapes of trail need two distinct guards:

1. **Append-only event stores** (``http_audit_log``, ``system_setting_changes``).
   These tables ARE the log. Rows are facts about a moment that has already
   happened, so no row may ever be modified, and rows may only be removed by
   an explicit retention operation.

2. **Audit columns on live entity tables** (``courses``, ``learning_materials``,
   ``users``, ``user_role_assignments`` -- the four tables
   ``/admin/audit/data-changes`` projects). These rows are live data and must
   stay editable; it is the *provenance* that must not move. ``created_at``
   and ``created_by`` record who brought a row into existence and when, and
   nothing may rewrite that after the INSERT. ``updated_at`` / ``updated_by``
   are deliberately NOT frozen -- re-stamping them on every change is exactly
   what migrations 0012 and 0013 exist to do.

Retention escape hatch
----------------------
Blocking ``DELETE`` outright would make retention impossible, and an audit
store nobody can prune eventually takes the database down with it -- so
deletion is gated rather than forbidden. ``audit_log_immutable()`` permits a
``DELETE`` only while the session-local GUC ``app.audit_maintenance`` is set
to ``'on'``, which is what :func:`abridgeai.core.audit.audit_maintenance_scope`
emits as ``SET LOCAL``. The application never sets it, so ordinary request
handling and worker code cannot delete audit rows even by accident; a
retention job (or a test cleaning up its own fixtures) opts in explicitly and
for one transaction only, because ``SET LOCAL`` dies with the transaction.

``UPDATE`` has no escape hatch on either guard. Retention deletes whole rows;
it never edits them, and no legitimate caller rewrites history in place.
"""

from __future__ import annotations

from alembic import op

revision = "0105_audit_immutability"
down_revision = "0104_policy_academic_bodies"
branch_labels = None
depends_on = None


# Append-only event stores: the row IS the audit record.
APPEND_ONLY_TABLES: tuple[str, ...] = (
    "http_audit_log",
    "system_setting_changes",
)

# Live entity tables whose provenance columns back /admin/audit/data-changes.
# Kept in step with ``SUPPORTED_DATA_CHANGE_TABLES`` in
# ``abridgeai/features/admin/queries/audit.py``.
PROVENANCE_TABLES: tuple[str, ...] = (
    "courses",
    "learning_materials",
    "user_role_assignments",
    "users",
)


_APPEND_ONLY_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION audit_log_immutable()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION
            'audit_log_immutable: % is append-only; UPDATE is never permitted',
            TG_TABLE_NAME
            USING ERRCODE = 'restrict_violation';
    END IF;

    -- DELETE: allowed only inside an explicit maintenance scope. NULLIF guards
    -- the empty string a reset backend returns for a never-SET GUC, which is
    -- not NULL and would otherwise compare unequal and read as "not enabled"
    -- only by luck.
    IF COALESCE(NULLIF(current_setting('app.audit_maintenance', true), ''), 'off')
       <> 'on' THEN
        RAISE EXCEPTION
            'audit_log_immutable: % is append-only; DELETE requires the app.audit_maintenance scope (retention only)',
            TG_TABLE_NAME
            USING ERRCODE = 'restrict_violation';
    END IF;

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
"""


# ``to_jsonb(NEW)`` rather than ``NEW.created_by`` so ONE function serves every
# table: plpgsql resolves record fields at runtime, and a direct reference to a
# column the table does not have (``users`` carries no ``created_by`` -- it is
# TimestampMixin-only) would raise at trigger time instead of being skipped.
# ``?`` tests key presence, so the created_by check simply does not apply there.
_PROVENANCE_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION audit_columns_immutable()
RETURNS TRIGGER AS $$
DECLARE
    new_row jsonb := to_jsonb(NEW);
    old_row jsonb := to_jsonb(OLD);
BEGIN
    IF new_row -> 'created_at' IS DISTINCT FROM old_row -> 'created_at' THEN
        RAISE EXCEPTION
            'audit_columns_immutable: %.created_at is immutable (audit provenance)',
            TG_TABLE_NAME
            USING ERRCODE = 'restrict_violation';
    END IF;

    IF old_row ? 'created_by'
       AND new_row -> 'created_by' IS DISTINCT FROM old_row -> 'created_by' THEN
        RAISE EXCEPTION
            'audit_columns_immutable: %.created_by is immutable (audit provenance)',
            TG_TABLE_NAME
            USING ERRCODE = 'restrict_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def _append_only_trigger(table: str) -> str:
    return f"audit_log_immutable_{table}"


def _provenance_trigger(table: str) -> str:
    return f"audit_columns_immutable_{table}"


def upgrade() -> None:
    op.execute(_APPEND_ONLY_FUNCTION_DDL)
    for table in APPEND_ONLY_TABLES:
        # BEFORE, not AFTER: the row must never reach the heap write at all,
        # and BEFORE lets the function abort the statement by raising.
        op.execute(
            f"CREATE TRIGGER {_append_only_trigger(table)} "
            f"BEFORE UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();"
        )

    op.execute(_PROVENANCE_FUNCTION_DDL)
    for table in PROVENANCE_TABLES:
        op.execute(
            f"CREATE TRIGGER {_provenance_trigger(table)} "
            f"BEFORE UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION audit_columns_immutable();"
        )


def downgrade() -> None:
    for table in PROVENANCE_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {_provenance_trigger(table)} ON {table};")
    op.execute("DROP FUNCTION IF EXISTS audit_columns_immutable();")

    for table in APPEND_ONLY_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {_append_only_trigger(table)} ON {table};")
    op.execute("DROP FUNCTION IF EXISTS audit_log_immutable();")
