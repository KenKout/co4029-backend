"""Gap 3: version FKs cascade from the path (0074 follow-up).

``career_path_versions.career_path_id`` and
``career_path_stages.version_id`` were created ``ON DELETE NO ACTION`` in
0074. A hard ``DELETE FROM career_paths`` (test teardowns, cleanup
scripts) therefore fails on the versions row even after the items/stages
rows are gone. Soft-deletes (the service path) are unaffected either way.

Making BOTH links CASCADE lets a hard path delete clean its versions and
their stages — which is exactly what the 0074 items link
(``career_course_items.version_id``) already does. Latched progress
(``student_stage_progress.stage_id``, NO ACTION) and enrollment pins
(``student_career_enrollments.version_id``, NO ACTION) still block the
delete while real student data references it, which is the intended
safety: cleanup teardowns remove latches/enrollments first.
"""

from __future__ import annotations

from alembic import op

revision = "0076_version_fk_cascade"
down_revision = "0075_iv_budgets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "career_path_versions_career_path_id_fkey",
        "career_path_versions",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "career_path_versions_career_path_id_fkey",
        "career_path_versions",
        "career_paths",
        ["career_path_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint(
        "career_path_stages_version_id_fkey",
        "career_path_stages",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "career_path_stages_version_id_fkey",
        "career_path_stages",
        "career_path_versions",
        ["version_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "career_path_stages_version_id_fkey",
        "career_path_stages",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "career_path_stages_version_id_fkey",
        "career_path_stages",
        "career_path_versions",
        ["version_id"],
        ["id"],
        ondelete="NO ACTION",
    )
    op.drop_constraint(
        "career_path_versions_career_path_id_fkey",
        "career_path_versions",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "career_path_versions_career_path_id_fkey",
        "career_path_versions",
        "career_paths",
        ["career_path_id"],
        ["id"],
        ondelete="NO ACTION",
    )
