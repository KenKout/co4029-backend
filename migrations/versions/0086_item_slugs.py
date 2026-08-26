"""Add ``slug`` to curriculum items: quizzes + interview_configs.

Product decision (2026-08-26): every student-facing course item gets a URL
slug so links read like breadcrumbs
(``/courses/<course-slug>/learn/<item-slug>``) instead of exposing UUIDs.
``lessons`` already had one (``uq_lessons_module_slug``); this migration
gives quizzes and interview configs the same treatment:

* ``quizzes.slug`` — unique per module (mirrors lessons), NOT NULL, backfilled
  from the title with ``-1``, ``-2``, … collision suffixes (incrementing from
  1). Backfill is scoped to live rows only (soft-deleted keep their historical
  slug value; they are excluded from uniqueness by the partial index).
* ``interview_configs.slug`` — same shape.

Slug generation rules (shared with the runtime service code in
``abridgeai/core/slug.py``): NFD-fold Vietnamese diacritics to ASCII,
lowercase, collapse non-alphanumerics to single hyphens. The SQL backfill
uses ``translate()`` + ``regexp_replace()`` for the common diacritic range;
the Python runtime path uses unicodedata for full coverage.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0086_item_slugs"
down_revision = "0085_dean_superset_of_manager"
branch_labels = None
depends_on = None

# Each ASCII letter with a diacritic folds to its base letter here. This is
# the pragmatic subset of unicodedata's NFD fold covering Vietnamese plus the
# Latin-1 accented letters that appear in course content.
_FOLD_FROM = (
    "ÀÁÂÃÄÅàáâãäåĀāĂąĄàÇçÈÉÊËèéêëĒēĖėĘęÌÍÎÏìíîïĪīĮį"
    "ÑñÒÓÔÕÖòóôõöŌōŐőØøŒœÙÚÛÜùúûüŪūŮůŰűŲųÝŸýÿŹźŻżŽžĐđŠšČčŘřŽžĚěŇňŤťŮů"
)
_FOLD_TO = (
    "AAAAAAaaaaaaAaAaAAaaCcEEEEeeeeEeEeEeIIIIiiiiIiIi"
    "NnOOOOOoooooOoOoOOoeUuuuuuuuUuUuUuUuYyyyZzZzZzDdSsCcRrZzEnTtUu"
)


def upgrade() -> None:
    op.add_column("quizzes", sa.Column("slug", sa.String(100), nullable=True))
    op.add_column(
        "interview_configs", sa.Column("slug", sa.String(100), nullable=True)
    )

    # 1. Backfill from titles — fold diacritics, lowercase, hyphen-join.
    #    Empty result (title was all symbols) falls back to the table name.
    #    Every live row is re-slugged from its title in one deterministic
    #    pass: park each row on a unique placeholder first so the final
    #    UPDATE cannot collide with a stale value still held by another row,
    #    then number duplicates per (module_id, slug): first keeps the bare
    #    base, later ones get ``-1``, ``-2``, … incrementing from 1.
    for table in ("quizzes", "interview_configs"):
        fallback = "quiz" if table == "quizzes" else "interview"
        op.execute(f"""
            UPDATE {table}
            SET slug = '___reslug_' || id::text
            WHERE deleted_at IS NULL
        """)
        op.execute(f"""
            UPDATE {table}
            SET slug = COALESCE(
                NULLIF(regexp_replace(translate(lower(title), '{_FOLD_FROM}', '{_FOLD_TO}'), '[^a-z0-9]+', '-', 'g'), '-'),
                '{fallback}'
            )
            WHERE deleted_at IS NULL
        """)
        op.execute(f"""
            WITH numbered AS (
                SELECT id, slug,
                       row_number() OVER (
                           PARTITION BY module_id, slug ORDER BY created_at, id
                       ) - 1 AS rn
                FROM {table}
                WHERE deleted_at IS NULL
            )
            UPDATE {table} t
            SET slug = CASE WHEN n.rn = 0 THEN n.slug ELSE n.slug || '-' || n.rn END
            FROM numbered n
            WHERE t.id = n.id
        """)
        # Soft-deleted rows keep a placeholder value: they are outside the
        # partial index, but the column is NOT NULL so they need something.
        op.execute(f"UPDATE {table} SET slug = '___reslug_' || id::text WHERE slug IS NULL")

    # 2. NOT NULL + per-module uniqueness over live rows (partial unique
    #    index mirrors uq_lessons_module_slug so soft-deleted slugs free up).
    op.alter_column("quizzes", "slug", existing_type=sa.String(100), nullable=False)
    op.alter_column("interview_configs", "slug", existing_type=sa.String(100), nullable=False)
    op.create_index(
        "uq_quizzes_module_slug",
        "quizzes",
        ["module_id", "slug"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "uq_interview_configs_module_slug",
        "interview_configs",
        ["module_id", "slug"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_interview_configs_module_slug", table_name="interview_configs")
    op.drop_index("uq_quizzes_module_slug", table_name="quizzes")
    op.drop_column("interview_configs", "slug")
    op.drop_column("quizzes", "slug")
