"""updated_at refresh trigger on every TimestampMixin table (T1)

Revision ID: 0012_updated_at_trigger
Revises: 0011_http_audit_log
Create Date: 2026-05-17 22:00:00.000000

Why
---
``TimestampMixin`` declares ``updated_at`` with
``onupdate=text("NOW()")`` -- but that hook only fires when SQLAlchemy
compiles the UPDATE statement. Any code path that bypasses the ORM
(raw ``text("UPDATE ...")``, bulk ``UPDATE`` from a migration,
``conn.execute(table.update(...))`` against the Core table object,
``psql`` admin patches, recursive soft-delete cascades emitted as raw
SQL) leaves ``updated_at`` stale. The forensic value of an audit
column you cannot trust is zero.

How
---
A single ``set_updated_at()`` plpgsql function plus one
``BEFORE UPDATE FOR EACH ROW`` trigger per ``TimestampMixin`` table.
The trigger sets ``NEW.updated_at = NOW()`` unconditionally, so even an
explicit raw ``UPDATE ... SET updated_at = '2020-01-01'`` will be
overwritten -- exactly the behaviour we want for an audit timestamp.

The SQLAlchemy ``onupdate`` hook is intentionally left in place as
harmless redundancy: it does not contradict the trigger (both compute
``NOW()``), it keeps ORM-side ``flush`` snapshots consistent with the
post-trigger row, and it documents intent at the model layer.

Inventory
---------
The 46 tables below were enumerated by reflecting ``Base.registry``
and filtering ``issubclass(cls, TimestampMixin)``. NOTE: the T1
brief listed 34 tables based on stale research; the live model
registry is the source of truth.

Excluded (no ``updated_at`` column):
* ``http_audit_log`` -- append-only, ``CreatedAtMixin``-shaped.
* ``card_reviews``, ``career_path_courses`` (alias ``career_course_items``
  is TimestampMixin and IS included), ``material_engagement``,
  ``quiz_source_lessons``, ``quiz_question_revisions``,
  ``document_chunks``, ``chunking_enrichment_cache``, ``course_tags``,
  ``module_prerequisites``, ``lesson_prerequisites``,
  ``role_permissions``, ``interview_session_questions``,
  ``assessment_integrity_events``, ``mfa_recovery_codes``,
  ``mfa_challenges`` -- all ``CreatedAtMixin``-only.
"""

from __future__ import annotations

from alembic import op

revision = "0012_updated_at_trigger"
down_revision = "0011_http_audit_log"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# All tables that inherit ``TimestampMixin`` (i.e. carry ``updated_at``).
# Sorted alphabetically for grepability and stable migration diffs.
# ---------------------------------------------------------------------------
TIMESTAMP_TABLES: tuple[str, ...] = (
    "auth_identities",
    "auth_sessions",
    "career_paths",
    "course_enrollments",
    "course_invitation_codes",
    "course_learning_outcomes",
    "courses",
    "gap_reports",
    "generation_runs",
    "interview_configs",
    "interview_outcome_evaluations",
    "interview_outcomes",
    "interview_questions",
    "interview_session_messages",
    "interview_sessions",
    "learning_material_versions",
    "learning_materials",
    "lesson_progress",
    "lesson_resources",
    "lessons",
    "mfa_factors",
    "module_items",
    "modules",
    "notification_preferences",
    "notifications",
    "org_units",
    "organization_domains",
    "organization_memberships",
    "organizations",
    "permissions",
    "processing_jobs",
    "quiz_attempt_answers",
    "quiz_attempts",
    "quiz_question_options",
    "quiz_questions",
    "quizzes",
    "roles",
    "storage_objects",
    "student_card_state",
    "student_career_enrollments",
    "tags",
    "user_permission_grants",
    "user_profile_links",
    "user_profiles",
    "user_role_assignments",
    "users",
)


_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def _trigger_name(table: str) -> str:
    return f"update_{table}_updated_at"


def upgrade() -> None:
    op.execute(_FUNCTION_DDL)
    for table in TIMESTAMP_TABLES:
        op.execute(
            f"CREATE TRIGGER {_trigger_name(table)} "
            f"BEFORE UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
        )


def downgrade() -> None:
    for table in TIMESTAMP_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {_trigger_name(table)} ON {table};")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at();")
