"""Re-slug live lessons from their current titles (breadcrumb URLs).

Product decision (2026-08-27, continuation of ``0086_item_slugs``): every
student-facing course item carries a URL slug generated from its title so
links read as breadcrumbs (``/courses/<course-slug>/learn/<item-slug>``).
Migration 0086 backfilled quizzes + interview configs; lessons are the
remaining item type. Their slugs predate the shared convention — the FE
used to generate ``<lesson-type>-<timestamp36>`` seeds at creation and
never regenerated them after the default title ("New Reading") was
renamed, so existing slugs do not match titles at all.

This migration re-slugs every live lesson from its CURRENT title using the
very same helpers the runtime uses (:mod:`abridgeai.core.slug`): NFD-fold
Vietnamese diacritics to ASCII, lowercase, hyphen-join, collisions get
``-1``, ``-2``, … incrementing from 1, unique per module. Running the
canonical Python here (instead of a SQL transliteration) guarantees the
backfill output is byte-identical to what authoring would generate today,
including the suffix-skip edge case (a title ``"X 1"`` whose base collides
with another title's suffixed form).

Soft-deleted rows are parked on id placeholders first: ``uq_lessons_module_slug``
(and the legacy ``lessons_module_id_slug_key``) are FULL unique constraints,
so stale soft-deleted slugs must not participate in the re-slug. The
intermediate bare-slug UPDATE of 0086 would have violated the pre-existing
lesson constraint mid-statement (duplicate titles), which is why this
migration assigns the final disambiguated slug in a single pass.

Note: published lessons are re-slugged too — one-time remediation to align
live URLs with the new convention; the API-level immutability rule (a
published lesson's slug cannot change afterwards) is enforced by
``update_lesson``.
"""

from __future__ import annotations

from alembic import op

revision = "0087_lesson_slugs_from_title"
down_revision = "0086_item_slugs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Imported lazily: migrations must not couple the alembic bootstrap to
    # app imports, and core/slug.py is pure (re + unicodedata only).
    from abridgeai.core.slug import slugify, unique_slug

    conn = op.get_bind()

    # 1. Park soft-deleted rows on unique placeholders FIRST: their stale
    #    slugs are still inside the full (module_id, slug) unique
    #    constraints and would collide with a freshly re-slugged live row.
    conn.exec_driver_sql(
        "UPDATE lessons SET slug = '___reslug_' || id::text "
        "WHERE deleted_at IS NOT NULL"
    )

    # 2. Re-slug live rows in deterministic (created_at, id) order so the
    #    first lesson keeping a bare base matches the SQL numbering rule
    #    0086 used. unique_slug skips over suffixed forms already taken, so
    #    the result is collision-free by construction.
    rows = conn.exec_driver_sql(
        "SELECT id, module_id, title FROM lessons "
        "WHERE deleted_at IS NULL "
        "ORDER BY module_id, created_at, id"
    ).fetchall()

    taken_by_module: dict[object, set[str]] = {}
    assignments: list[tuple[str, str]] = []
    for lesson_id, module_id, title in rows:
        taken = taken_by_module.setdefault(module_id, set())
        base = slugify(title or "") or "lesson"
        final_slug = unique_slug(base, taken)
        taken.add(final_slug)
        assignments.append((str(lesson_id), final_slug))

    if assignments:
        # One UPDATE per row — the data volume here is small (a
        # curriculum's lessons), far simpler than a giant CASE and immune
        # to constraint-ordering issues. exec_driver_sql hands the %s
        # params straight to the psycopg driver (no bind parsing).
        conn.exec_driver_sql(
            "UPDATE lessons SET slug = %s WHERE id = %s AND deleted_at IS NULL",
            [(slug, lesson_id) for lesson_id, slug in assignments],
        )


def downgrade() -> None:
    # No automatic inverse: slugs are derived data and the previous
    # (timestamp-style) values are not recoverable in a principled way.
    # Downgrade is intentionally a no-op — the column itself stays.
    pass
