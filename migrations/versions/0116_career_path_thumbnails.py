"""Add an object-storage thumbnail to career paths.

Revision ID: 0116_career_path_thumbnails
Revises: 0115_quiz_integrity_policy
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0116_career_path_thumbnails"
down_revision = "0115_quiz_integrity_policy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "career_paths",
        sa.Column("thumbnail_object_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_career_paths_thumbnail_object_id_storage_objects",
        "career_paths",
        "storage_objects",
        ["thumbnail_object_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_career_paths_thumbnail_object_id_storage_objects",
        "career_paths",
        type_="foreignkey",
    )
    op.drop_column("career_paths", "thumbnail_object_id")
