"""partial unique indexes for soft-deletable tables

Revision ID: 0002_partial_unique_indexes
Revises: 0001_baseline_schema
Create Date: 2026-05-16 21:00:00.000000

Converts UNIQUE constraints on soft-deletable tables (those with a
``deleted_at`` column) to partial unique indexes filtered by
``WHERE deleted_at IS NULL``.

Goal: a slug/code/position can be reused after the row that previously
owned it has been soft-deleted (``UPDATE ... SET deleted_at = NOW()``).
Active rows still cannot collide.

Scope rules:
  - Touch only constraints whose parent table is in
    ``SOFT_DELETE_TABLES`` (0001 baseline list — schema-confirmed by
    introspection of ``information_schema.columns`` for ``deleted_at``).
  - Skip PRIMARY KEYs (PG cannot make them partial).
  - Skip UNIQUE constraints on hard-delete tables (``users``,
    ``auth_sessions``, ``mfa_*``, ``tags``, ``career_course_items``,
    ``course_enrollments`` etc. — they have no ``deleted_at``, so a
    ``WHERE deleted_at IS NULL`` filter would error out at index time).
  - Skip FK constraints (handled separately by T0.14).
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_partial_unique_indexes"
down_revision = "0001_baseline_schema"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Conversion inventory
# ---------------------------------------------------------------------------
# Each entry: (table, constraint_name, [columns])
# Constraint names match what PostgreSQL has after 0001 (verified via
# information_schema.table_constraints introspection on the live db).
# ---------------------------------------------------------------------------
CONVERSIONS: tuple[tuple[str, str, list[str]], ...] = (
    # Identity / tenancy / ACL
    ("organizations", "organizations_slug_key", ["slug"]),
    ("organization_domains", "organization_domains_domain_key", ["domain"]),
    ("organization_memberships",
     "organization_memberships_user_id_organization_id_org_unit_i_key",
     ["user_id", "organization_id", "org_unit_id"]),
    ("org_units", "org_units_organization_id_code_key",
     ["organization_id", "code"]),
    ("storage_objects", "storage_objects_bucket_object_key_key",
     ["bucket", "object_key"]),
    ("permissions", "permissions_code_key", ["code"]),
    ("roles", "roles_code_key", ["code"]),
    ("user_role_assignments",
     "user_role_assignments_user_id_role_id_scope_kind_organizati_key",
     ["user_id", "role_id", "scope_kind", "organization_id",
      "org_unit_id", "course_id"]),
    ("user_permission_grants",
     "user_permission_grants_user_id_permission_id_scope_kind_org_key",
     ["user_id", "permission_id", "scope_kind", "organization_id",
      "org_unit_id", "course_id"]),
    # Org / careers / curriculum
    ("career_paths", "career_paths_organization_id_slug_key",
     ["organization_id", "slug"]),
    ("student_career_enrollments",
     "student_career_enrollments_career_path_id_student_id_key",
     ["career_path_id", "student_id"]),
    ("courses", "uq_courses_org_slug", ["organization_id", "slug"]),
    ("course_learning_outcomes",
     "course_learning_outcomes_course_id_position_key",
     ["course_id", "position"]),
    ("modules", "modules_course_id_position_key",
     ["course_id", "position"]),
    ("lessons", "lessons_module_id_slug_key", ["module_id", "slug"]),
    ("lesson_resources", "lesson_resources_lesson_id_position_key",
     ["lesson_id", "position"]),
    ("module_items", "module_items_module_id_position_key",
     ["module_id", "position"]),
    # Learning materials
    ("learning_material_versions",
     "learning_material_versions_material_id_version_no_key",
     ["material_id", "version_no"]),
    # Quizzes
    ("quiz_questions", "quiz_questions_quiz_id_position_key",
     ["quiz_id", "position"]),
    ("quiz_question_options",
     "quiz_question_options_question_id_option_key_key",
     ["question_id", "option_key"]),
    ("quiz_question_options",
     "quiz_question_options_question_id_position_key",
     ["question_id", "position"]),
    # Interviews
    ("interview_outcomes",
     "interview_outcomes_interview_config_id_position_key",
     ["interview_config_id", "position"]),
    ("interview_questions",
     "interview_questions_interview_config_id_position_key",
     ["interview_config_id", "position"]),
)


_PARTIAL_WHERE = sa.text("deleted_at IS NULL")


def upgrade() -> None:
    """Replace each UNIQUE constraint with a partial UNIQUE INDEX of the same name."""
    for table, name, cols in CONVERSIONS:
        op.drop_constraint(name, table, type_="unique")
        op.create_index(
            name,
            table,
            cols,
            unique=True,
            postgresql_where=_PARTIAL_WHERE,
        )


def downgrade() -> None:
    """Reverse: drop partial unique index, restore plain UNIQUE constraint."""
    for table, name, cols in reversed(CONVERSIONS):
        op.drop_index(name, table_name=table)
        op.create_unique_constraint(name, table, cols)
