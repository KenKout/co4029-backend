"""Career path stages: staged pathways, lazy enrollment, sticky stage latch.

Revision ID: 0070_career_path_stages
Revises: 0069_sr_ef_ceiling
Create Date: 2026-08-06

Why
---
A career path was a flat ordered list of courses. Managers need *stages*
(groups of courses gated on the previous group) with per-stage unlock and
enforcement policies, plus a "min N optional" rule so a stage can offer
electives.

Steps
-----
1. ``career_path_stages`` — the stage itself, with a PARTIAL unique index on
   ``(career_path_id, position)`` filtered to ``deleted_at IS NULL`` (the
   table is soft-deletable, so a plain unique index would let a deleted
   stage keep reserving its position).
2. ``student_stage_progress`` — append-only latch: "this enrollment
   completed this stage". Deliberately has no ``updated_at`` /
   ``deleted_at`` (see the model docstring); rows are never updated or
   deleted, which is what makes stage completion irreversible.
3. ``career_paths.max_concurrent`` — the attention cap. Path-level, NOT
   stage-level: it is compared against a path-wide count of active course
   enrollments, so a stage-level column would have compared two different
   scopes.
4. ``career_readiness_snapshots.formula_version`` — DEFAULT **1**, not 2.
   The progress formula is gated behind the
   ``careerpath.progress_formula_version`` runtime setting which starts at
   1, so defaulting the column to 2 would mislabel every snapshot written
   before the cutover. ``services/readiness.py`` stamps the value it
   actually used explicitly.
5. ``career_course_items.stage_id`` (nullable) + ``satisfied_by``.
6. Backfill: one synthetic stage per existing path (``title`` NULL so the
   client renders "Stage 1" in the user's locale, ``unlock_policy``
   ``always`` + ``enforcement`` ``advisory`` so no existing path starts
   gating students who were mid-flight), and point every existing course
   item at it. Progress is preserved: the item rows keep their positions.
7. ``stage_id`` → NOT NULL, once every row has one.
8. Swap the unique index from ``(career_path_id, position)`` to
   ``(stage_id, position)``. **NOT partial** — ``career_course_items`` has
   no ``deleted_at`` column (it is a hard-delete link table; migration
   0002's skip-list excludes it), so a ``WHERE deleted_at IS NULL`` filter
   would error out at index-creation time.

``PRIMARY KEY (career_path_id, course_id)`` is deliberately KEPT. It is what
makes "the same course in two stages of one path" structurally impossible;
re-keying on ``(stage_id, course_id)`` would permit exactly that.

Every FK added here is ``ON DELETE NO ACTION``, matching migration 0003
which deliberately flipped every career FK away from CASCADE: CASCADE and
soft delete do not compose (a hard cascade would delete rows the soft-delete
layer believes it still owns).

Reversible: downgrade restores the old unique index, drops the added
columns and tables. The synthetic stages disappear with the table.
"""

from __future__ import annotations

from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0070_career_path_stages"
down_revision = "0069_sr_ef_ceiling"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- 1. stages -------------------------------------------------------
    op.execute(
        """
        CREATE TABLE career_path_stages (
            id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            career_path_id           UUID NOT NULL
                REFERENCES career_paths(id) ON DELETE NO ACTION,
            position                 INT NOT NULL CHECK (position > 0),
            title                    VARCHAR(200),
            description              TEXT,
            min_optional_to_complete INT NOT NULL DEFAULT 0
                CHECK (min_optional_to_complete >= 0),
            unlock_policy            VARCHAR(30) NOT NULL DEFAULT 'after_previous'
                CHECK (unlock_policy IN ('always','after_previous','after_previous_required')),
            enforcement              VARCHAR(20) NOT NULL DEFAULT 'soft'
                CHECK (enforcement IN ('hard','soft','advisory')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_by UUID REFERENCES users(id) ON DELETE SET NULL,
            updated_by UUID REFERENCES users(id) ON DELETE SET NULL,
            deleted_at TIMESTAMPTZ,
            deleted_by UUID REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    # PARTIAL: a soft-deleted stage must not keep reserving its position.
    op.execute(
        """
        CREATE UNIQUE INDEX career_path_stages_path_position_key
            ON career_path_stages (career_path_id, position)
            WHERE deleted_at IS NULL
        """
    )
    op.execute(
        "CREATE INDEX ix_career_path_stages_career_path_id ON career_path_stages (career_path_id)"
    )
    op.execute("CREATE INDEX ix_career_path_stages_deleted_at ON career_path_stages (deleted_at)")

    # -- 2. append-only stage latch --------------------------------------
    op.execute(
        """
        CREATE TABLE student_stage_progress (
            id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            enrollment_id UUID NOT NULL
                REFERENCES student_career_enrollments(id) ON DELETE NO ACTION,
            stage_id      UUID NOT NULL
                REFERENCES career_path_stages(id) ON DELETE NO ACTION,
            completed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (enrollment_id, stage_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_student_stage_progress_enrollment_id "
        "ON student_stage_progress (enrollment_id)"
    )

    # -- 3. attention cap (path-level, see module docstring) -------------
    op.execute("ALTER TABLE career_paths ADD COLUMN max_concurrent INT")
    op.execute(
        "ALTER TABLE career_paths ADD CONSTRAINT career_paths_max_concurrent_check "
        "CHECK (max_concurrent IS NULL OR max_concurrent > 0)"
    )

    # -- 4. snapshot formula stamp (DEFAULT 1 — see docstring) -----------
    op.execute(
        "ALTER TABLE career_readiness_snapshots "
        "ADD COLUMN formula_version SMALLINT NOT NULL DEFAULT 1"
    )

    # -- 5. course item columns ------------------------------------------
    op.execute(
        """
        ALTER TABLE career_course_items
            ADD COLUMN stage_id UUID
                REFERENCES career_path_stages(id) ON DELETE NO ACTION,
            ADD COLUMN satisfied_by VARCHAR(20) NOT NULL DEFAULT 'completion'
                CHECK (satisfied_by IN ('completion','pass'))
        """
    )

    # -- 6. backfill one synthetic stage per path ------------------------
    # 'always' + 'advisory': an existing path must not start gating students
    # who are already mid-flight just because stages now exist.
    op.execute(
        """
        INSERT INTO career_path_stages
            (career_path_id, position, title, description,
             min_optional_to_complete, unlock_policy, enforcement)
        SELECT id, 1, NULL, NULL, 0, 'always', 'advisory'
        FROM career_paths
        """
    )
    op.execute(
        """
        UPDATE career_course_items cci
        SET stage_id = s.id
        FROM career_path_stages s
        WHERE s.career_path_id = cci.career_path_id
          AND s.position = 1
        """
    )

    # -- 7. now that every row has a stage ------------------------------
    op.execute("ALTER TABLE career_course_items ALTER COLUMN stage_id SET NOT NULL")

    # -- 8. re-scope the position uniqueness to the stage ---------------
    op.execute(
        "ALTER TABLE career_course_items "
        "DROP CONSTRAINT career_course_items_career_path_id_position_key"
    )
    # NOT partial: career_course_items has no deleted_at column (0002
    # skip-list), so a WHERE deleted_at IS NULL filter errors at index time.
    op.execute(
        """
        CREATE UNIQUE INDEX career_course_items_stage_position_key
            ON career_course_items (stage_id, position)
        """
    )
    op.execute("CREATE INDEX ix_career_course_items_stage_id ON career_course_items (stage_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_career_course_items_stage_id")
    op.execute("DROP INDEX IF EXISTS career_course_items_stage_position_key")
    op.execute(
        "ALTER TABLE career_course_items "
        "ADD CONSTRAINT career_course_items_career_path_id_position_key "
        'UNIQUE (career_path_id, "position")'
    )
    op.execute(
        "ALTER TABLE career_course_items "
        "DROP COLUMN IF EXISTS satisfied_by, "
        "DROP COLUMN IF EXISTS stage_id"
    )
    op.execute("ALTER TABLE career_readiness_snapshots DROP COLUMN IF EXISTS formula_version")
    op.execute(
        "ALTER TABLE career_paths DROP CONSTRAINT IF EXISTS career_paths_max_concurrent_check"
    )
    op.execute("ALTER TABLE career_paths DROP COLUMN IF EXISTS max_concurrent")
    op.execute("DROP TABLE IF EXISTS student_stage_progress")
    op.execute("DROP TABLE IF EXISTS career_path_stages")
