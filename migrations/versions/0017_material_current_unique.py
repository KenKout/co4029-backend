"""Partial unique index: one current version per material (§C15 backstop)

Revision ID: 0017_material_current_unique
Revises: 0016_quiz_q_imported_from
Create Date: 2026-06-10 15:30:00.000000

``complete_upload`` and the phase-04 ``rollback_material_version`` both
flip ``learning_material_versions.is_current`` with a snapshot-read +
flag-swap (no row lock). Without a DB constraint a concurrent pair can
leave two ``is_current = TRUE`` rows, which makes the learner-side
``get_latest_ready_version`` (``scalar_one_or_none``) raise
``MultipleResultsFound`` → 500, or silently 404 stream URLs when the
two pointer sources diverge.

This partial unique index turns that race into an ``IntegrityError``
that ``flush_or_conflict`` maps to a clean 409 (mapping registered in
``materials/services/authoring/_versions.py``).
"""

from __future__ import annotations

from alembic import op

revision = "0017_material_current_unique"
down_revision = "0016_quiz_q_imported_from"
branch_labels = None
depends_on = None

_INDEX = "uq_learning_material_versions_one_current"


def upgrade() -> None:
    # Defensive de-dup before adding the constraint: keep the newest
    # current row per material, clear the flag on any older duplicates.
    op.execute(
        """
        UPDATE learning_material_versions v
        SET is_current = FALSE
        WHERE is_current = TRUE
          AND deleted_at IS NULL
          AND EXISTS (
            SELECT 1 FROM learning_material_versions newer
            WHERE newer.material_id = v.material_id
              AND newer.is_current = TRUE
              AND newer.deleted_at IS NULL
              AND (newer.version_no > v.version_no)
          )
        """
    )
    op.create_index(
        _INDEX,
        "learning_material_versions",
        ["material_id"],
        unique=True,
        postgresql_where="is_current AND deleted_at IS NULL",
    )


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="learning_material_versions")
