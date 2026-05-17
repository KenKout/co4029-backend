"""audit_stamp BEFORE INSERT/UPDATE trigger on every AuditedByMixin table (T3)

Revision ID: 0013_audit_stamp_trigger
Revises: 0012_updated_at_trigger
Create Date: 2026-05-17 23:30:00.000000

Why
---
``AuditedByMixin`` adds ``created_by`` and ``updated_by`` user-id columns
that the application stamps from the request actor. T2 (``register_app_actor_listener``)
binds the actor into the Postgres GUC ``app.actor_id`` on every txn begin,
and the ORM-side audit listener (``core/audit/listener.py``) writes those
columns on ORM-managed flushes.

But any code path that bypasses the ORM -- raw ``text("UPDATE ...")``,
bulk ``UPDATE`` from a migration, ``conn.execute(table.update(...))``,
``psql`` admin patches, recursive soft-delete cascades emitted as raw
SQL -- leaves ``created_by`` / ``updated_by`` NULL or stale. The forensic
value of an audit column you cannot trust is zero.

How
---
A single ``audit_stamp()`` plpgsql function plus one
``BEFORE INSERT OR UPDATE FOR EACH ROW`` trigger per ``AuditedByMixin``
table. The function reads ``current_setting('app.actor_id', true)``,
falls back to ``SYSTEM_ACTOR_ID`` (``00000000-0000-0000-0000-000000000001``,
seeded by migration 0004) when no actor is bound (workers, alembic data
migrations), and stamps:

* ``created_by`` -- on INSERT only, only if currently NULL (allows
  app-level override pre-INSERT).
* ``updated_by`` -- on INSERT (if NULL) and on every UPDATE
  (re-stamped unconditionally; this is the audit semantics we want).

Critical correctness rules
--------------------------
* ``NULLIF`` wraps ``current_setting('app.actor_id', true)``: Postgres
  returns ``''`` (empty string), not NULL, on a backend that previously
  SET the GUC then session-reset; raw ``::uuid`` cast on ``''`` raises
  ``invalid input syntax for type uuid``. NULLIF converts ``''`` -> NULL,
  then COALESCE picks ``SYSTEM_ACTOR``.
* ``created_by`` is NEVER overwritten on UPDATE -- only stamped on
  INSERT, only if currently NULL.
* ``updated_by`` on UPDATE: only re-stamped when the caller did NOT
  change it (``NEW.updated_by IS NOT DISTINCT FROM OLD.updated_by``).
  This honours the "do not overwrite an explicit caller value" rule
  while still filling the bypass gap: raw ``UPDATE courses SET title=...``
  (which does not mention ``updated_by``) leaves NEW=OLD so the trigger
  stamps from the GUC; the ORM ``audit_listener`` (which sets
  ``updated_by`` on the in-memory object before flush) yields NEW!=OLD
  so the trigger respects the value. Both layers compose correctly.
* The function REUSES the SYSTEM_ACTOR seeded by ``0004_seed_permission_catalog``;
  no new sentinel is created here.

Inventory
---------
The 30 tables below were enumerated by reflecting ``Base.registry``
and filtering ``issubclass(cls, AuditedByMixin)``. NOTE: the T3 brief
listed 25 tables based on stale research; the live model registry is
the source of truth.

This set is NOT a strict subset of T1's TIMESTAMP_TABLES -- both
include their own respective superset/subset relationship. Notable
TimestampMixin-only tables (no AuditedByMixin) include ``tags``,
``generation_runs``, ``notifications``, ``notification_preferences``,
``users``, ``mfa_factors``, ``processing_jobs``, ``auth_identities``,
``auth_sessions``, ``student_card_state``, ``role_permissions``
(actually CreatedAtMixin-only), ``interview_session_messages``,
``interview_sessions``, ``interview_outcome_evaluations``,
``quiz_attempts``, ``quiz_attempt_answers``.
"""

from __future__ import annotations

from alembic import op

revision = "0013_audit_stamp_trigger"
down_revision = "0012_updated_at_trigger"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# All tables that inherit ``AuditedByMixin`` (i.e. carry ``created_by``
# and ``updated_by``). Sorted alphabetically for grepability and stable
# migration diffs.
# ---------------------------------------------------------------------------
AUDITED_BY_TABLES: tuple[str, ...] = (
    "career_paths",
    "course_enrollments",
    "course_invitation_codes",
    "course_learning_outcomes",
    "courses",
    "interview_configs",
    "interview_outcomes",
    "interview_questions",
    "learning_material_versions",
    "learning_materials",
    "lesson_progress",
    "lesson_resources",
    "lessons",
    "module_items",
    "modules",
    "org_units",
    "organization_domains",
    "organization_memberships",
    "organizations",
    "permissions",
    "quiz_question_options",
    "quiz_questions",
    "quizzes",
    "roles",
    "storage_objects",
    "student_career_enrollments",
    "user_permission_grants",
    "user_profile_links",
    "user_profiles",
    "user_role_assignments",
)


_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION audit_stamp()
RETURNS TRIGGER AS $$
DECLARE
    actor_uuid uuid;
    system_actor uuid := '00000000-0000-0000-0000-000000000001';
BEGIN
    actor_uuid := COALESCE(
        NULLIF(current_setting('app.actor_id', true), '')::uuid,
        system_actor
    );
    IF TG_OP = 'INSERT' THEN
        IF NEW.created_by IS NULL THEN
            NEW.created_by := actor_uuid;
        END IF;
        IF NEW.updated_by IS NULL THEN
            NEW.updated_by := actor_uuid;
        END IF;
    ELSIF TG_OP = 'UPDATE' THEN
        IF NEW.updated_by IS NOT DISTINCT FROM OLD.updated_by THEN
            NEW.updated_by := actor_uuid;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def _trigger_name(table: str) -> str:
    return f"audit_stamp_{table}"


def upgrade() -> None:
    op.execute(_FUNCTION_DDL)
    for table in AUDITED_BY_TABLES:
        op.execute(
            f"CREATE TRIGGER {_trigger_name(table)} "
            f"BEFORE INSERT OR UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION audit_stamp();"
        )


def downgrade() -> None:
    for table in AUDITED_BY_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {_trigger_name(table)} ON {table};")
    op.execute("DROP FUNCTION IF EXISTS audit_stamp();")
