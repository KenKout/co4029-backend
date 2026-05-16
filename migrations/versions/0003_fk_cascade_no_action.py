"""fk cascade no action

Revision ID: 0003_fk_cascade_no_action
Revises: 0002_partial_unique_indexes
Create Date: 2026-05-16 22:00:00.000000

Flips every ``ON DELETE CASCADE`` FK whose parent is in the soft-delete
inventory to ``ON DELETE NO ACTION``.

Why
---
Soft-deletable parents (courses, lessons, modules, quizzes, organizations,
interview_configs, learning_materials, etc.) are NEVER hard-deleted on the
happy path: T0.7 installs a global ``deleted_at IS NULL`` filter and T0.15
introduces an app-level recursive soft-delete service. CASCADE rules are
therefore unreachable in normal flow but would silently wipe child rows on
an accidental hard DELETE (admin force-purge, GDPR erasure path).
``NO ACTION`` is the defense: a stray DELETE on a parent will raise an
``IntegrityError`` instead of orphaning / cascading.

How
---
PostgreSQL does not support ``ALTER TABLE ... ALTER CONSTRAINT`` for
referential actions (only ``DEFERRABLE`` / ``INITIALLY DEFERRED`` can be
altered). We DROP each FK and recreate it under the same name with
``ondelete='NO ACTION'``. Column lists, nullability, and DEFERRABLE state
are unchanged — none of the original FKs were declared deferrable.

Source
------
``docs/fk-audit-pre-cascade-fix.md`` (T0.13) — 64 entries in the
``CASCADE → soft-deletable parent`` table. Constraint names taken from
``pg_constraint`` to preserve the live PG names (some are 63-char
truncated autogen names that do not match what 0001_baseline_schema.py
emitted as inline SQL).
"""
from __future__ import annotations

from alembic import op


revision = "0003_fk_cascade_no_action"
down_revision = "0002_partial_unique_indexes"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Conversion inventory
# ---------------------------------------------------------------------------
# Each entry: (from_table, fk_name, [from_columns], to_table, [to_columns])
# Order: by from_table, then constraint_name. Names match the live PG
# ``pg_constraint`` catalog (cross-checked against the T0.13 audit report).
# ---------------------------------------------------------------------------
FKS: tuple[tuple[str, str, list[str], str, list[str]], ...] = (
    # --- bulk_import_jobs ---
    ("bulk_import_jobs", "bulk_import_jobs_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),

    # --- career paths ---
    ("career_course_items", "career_course_items_career_path_id_fkey",
     ["career_path_id"], "career_paths", ["id"]),
    ("career_course_items", "career_course_items_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("career_paths", "career_paths_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),
    ("career_readiness_snapshots", "career_readiness_snapshots_career_path_id_fkey",
     ["career_path_id"], "career_paths", ["id"]),

    # --- courses ---
    ("course_enrollments", "course_enrollments_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("course_invitation_codes", "course_invitation_codes_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("course_learning_outcomes", "course_learning_outcomes_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("course_tags", "course_tags_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("course_tags", "course_tags_tag_id_fkey",
     ["tag_id"], "tags", ["id"]),
    ("courses", "courses_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),

    # --- document chunks (RAG/AI material refs) ---
    ("document_chunks", "document_chunks_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("document_chunks", "document_chunks_lesson_id_fkey",
     ["lesson_id"], "lessons", ["id"]),
    ("document_chunks", "document_chunks_material_version_id_fkey",
     ["material_version_id"], "learning_material_versions", ["id"]),
    ("document_chunks", "document_chunks_module_id_fkey",
     ["module_id"], "modules", ["id"]),

    # --- gap reports ---
    ("gap_reports", "gap_reports_course_id_fkey",
     ["course_id"], "courses", ["id"]),

    # --- generation runs ---
    ("generation_runs", "generation_runs_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("generation_runs", "generation_runs_lesson_id_fkey",
     ["lesson_id"], "lessons", ["id"]),
    ("generation_runs", "generation_runs_module_id_fkey",
     ["module_id"], "modules", ["id"]),

    # --- interview configs / questions / outcomes ---
    ("interview_configs", "interview_configs_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("interview_configs", "interview_configs_module_id_fkey",
     ["module_id"], "modules", ["id"]),
    ("interview_outcome_evaluations", "interview_outcome_evaluations_outcome_id_fkey",
     ["outcome_id"], "interview_outcomes", ["id"]),
    ("interview_outcomes", "interview_outcomes_interview_config_id_fkey",
     ["interview_config_id"], "interview_configs", ["id"]),
    ("interview_questions", "interview_questions_interview_config_id_fkey",
     ["interview_config_id"], "interview_configs", ["id"]),
    ("interview_sessions", "interview_sessions_interview_config_id_fkey",
     ["interview_config_id"], "interview_configs", ["id"]),

    # --- learning materials ---
    ("learning_material_versions", "learning_material_versions_material_id_fkey",
     ["material_id"], "learning_materials", ["id"]),
    ("learning_materials", "learning_materials_lesson_id_fkey",
     ["lesson_id"], "lessons", ["id"]),
    ("lesson_resources", "lesson_resources_lesson_id_fkey",
     ["lesson_id"], "lessons", ["id"]),

    # --- lessons / modules ---
    ("lessons", "lessons_module_id_fkey",
     ["module_id"], "modules", ["id"]),
    ("module_items", "module_items_interview_config_id_fkey",
     ["interview_config_id"], "interview_configs", ["id"]),
    ("module_items", "module_items_lesson_id_fkey",
     ["lesson_id"], "lessons", ["id"]),
    ("module_items", "module_items_module_id_fkey",
     ["module_id"], "modules", ["id"]),
    ("module_items", "module_items_quiz_id_fkey",
     ["quiz_id"], "quizzes", ["id"]),
    ("module_prerequisites", "module_prerequisites_module_id_fkey",
     ["module_id"], "modules", ["id"]),
    ("module_prerequisites", "module_prerequisites_prerequisite_module_id_fkey",
     ["prerequisite_module_id"], "modules", ["id"]),
    ("modules", "modules_course_id_fkey",
     ["course_id"], "courses", ["id"]),

    # --- organization / org units ---
    ("org_units", "org_units_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),
    ("org_units", "org_units_parent_unit_id_fkey",
     ["parent_unit_id"], "org_units", ["id"]),
    ("organization_domains", "organization_domains_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),
    ("organization_memberships", "organization_memberships_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),

    # --- quizzes / questions / answers ---
    ("quiz_attempt_answers", "quiz_attempt_answers_question_id_fkey",
     ["question_id"], "quiz_questions", ["id"]),
    ("quiz_attempts", "quiz_attempts_quiz_id_fkey",
     ["quiz_id"], "quizzes", ["id"]),
    ("quiz_question_options", "quiz_question_options_question_id_fkey",
     ["question_id"], "quiz_questions", ["id"]),
    ("quiz_question_revisions", "quiz_question_revisions_question_id_fkey",
     ["question_id"], "quiz_questions", ["id"]),
    ("quiz_questions", "quiz_questions_quiz_id_fkey",
     ["quiz_id"], "quizzes", ["id"]),
    ("quiz_source_lessons", "quiz_source_lessons_lesson_id_fkey",
     ["lesson_id"], "lessons", ["id"]),
    ("quiz_source_lessons", "quiz_source_lessons_quiz_id_fkey",
     ["quiz_id"], "quizzes", ["id"]),
    ("quizzes", "quizzes_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("quizzes", "quizzes_module_id_fkey",
     ["module_id"], "modules", ["id"]),

    # --- ACL: roles, permissions, role_permissions ---
    ("role_permissions", "role_permissions_permission_id_fkey",
     ["permission_id"], "permissions", ["id"]),
    ("role_permissions", "role_permissions_role_id_fkey",
     ["role_id"], "roles", ["id"]),

    # --- student progress ---
    ("student_career_enrollments", "student_career_enrollments_career_path_id_fkey",
     ["career_path_id"], "career_paths", ["id"]),
    ("student_course_status", "student_course_status_course_id_fkey",
     ["course_id"], "courses", ["id"]),
    ("student_lesson_progress", "student_lesson_progress_lesson_id_fkey",
     ["lesson_id"], "lessons", ["id"]),
    ("student_module_status", "student_module_status_module_id_fkey",
     ["module_id"], "modules", ["id"]),
    ("student_quiz_card_state", "student_quiz_card_state_question_id_fkey",
     ["question_id"], "quiz_questions", ["id"]),

    # --- ACL grants & assignments (custom-named + autogen mix) ---
    ("user_permission_grants", "fk_user_permission_grants_course",
     ["course_id"], "courses", ["id"]),
    ("user_permission_grants", "user_permission_grants_org_unit_id_fkey",
     ["org_unit_id"], "org_units", ["id"]),
    ("user_permission_grants", "user_permission_grants_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),
    ("user_permission_grants", "user_permission_grants_permission_id_fkey",
     ["permission_id"], "permissions", ["id"]),
    ("user_role_assignments", "fk_user_role_assignments_course",
     ["course_id"], "courses", ["id"]),
    ("user_role_assignments", "user_role_assignments_org_unit_id_fkey",
     ["org_unit_id"], "org_units", ["id"]),
    ("user_role_assignments", "user_role_assignments_organization_id_fkey",
     ["organization_id"], "organizations", ["id"]),
    ("user_role_assignments", "user_role_assignments_role_id_fkey",
     ["role_id"], "roles", ["id"]),
)


def upgrade() -> None:
    """Drop each CASCADE FK and recreate as NO ACTION (same name)."""
    for from_table, fk_name, from_cols, to_table, to_cols in FKS:
        op.drop_constraint(fk_name, from_table, type_="foreignkey")
        op.create_foreign_key(
            fk_name,
            from_table,
            to_table,
            from_cols,
            to_cols,
            ondelete="NO ACTION",
        )


def downgrade() -> None:
    """Reverse: drop NO ACTION FK, recreate with CASCADE."""
    for from_table, fk_name, from_cols, to_table, to_cols in reversed(FKS):
        op.drop_constraint(fk_name, from_table, type_="foreignkey")
        op.create_foreign_key(
            fk_name,
            from_table,
            to_table,
            from_cols,
            to_cols,
            ondelete="CASCADE",
        )
