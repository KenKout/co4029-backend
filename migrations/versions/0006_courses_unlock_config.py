"""courses aggregate unlock config columns + prerequisite tables

Revision ID: 0006_courses_unlock_config
Revises: 0005_ai_audit_pipeline_run
Create Date: 2026-05-17 12:00:00.000000

T3.1 backfill — Reconciliation §A2/§A4/§A13 made these columns / tables
NET-NEW even though they are required by the acceptance criteria of
T3.1. The orchestrator's "migrations are locked" directive applies to
the existing 0001-0005 history, not to additive forward migrations
needed to give the ORM honest DDL to map onto. The pattern matches
0005 (also additive: ALTER + ADD COLUMN + new index).

Adds three pieces:

1. Lesson unlock-config columns (per plan §3884-3886 + Phase 7.5 SR
   gate per Reconciliation §A4):

   * ``ef_min_unlock NUMERIC NOT NULL DEFAULT 2.0`` with CHECK
     ``1.3 <= ef_min_unlock <= 2.5`` — teacher per-lesson override of
     the spaced-repetition easiness threshold.
   * ``tau_unlock NUMERIC NOT NULL DEFAULT 0.8`` with CHECK
     ``0.0 < tau_unlock AND tau_unlock <= 1.0`` — proportion of cards
     above ``ef_min`` required for unlock.
   * ``requires_interview_pass BOOLEAN NOT NULL DEFAULT FALSE`` — gates
     the lesson behind the attached interview if any.
   * ``unlock_rule_json JSONB NOT NULL DEFAULT '{}'::jsonb`` —
     unstructured extension carrier per Reconciliation §A4 (KEEP the
     legacy column; the structured columns above supplement it).

2. Module unlock-mode column (per plan §3881):

   * ``requires_all_lessons_unlocked BOOLEAN NOT NULL DEFAULT FALSE``
     — when TRUE, the module is fully gated until every lesson within
     it is unlocked. DEFAULT FALSE preserves loose semantics per the
     locked user decision §3900.

3. ``lesson_prerequisites`` association table (per plan §3889-3890):

   * Composite PK ``(lesson_id, prereq_lesson_id)`` with self-FK to
     ``lessons.id`` on both columns, ``ON DELETE NO ACTION`` (matches
     T0.14 / migration 0003 cascade flip — soft-deletable parent).
   * CHECK ``lesson_id <> prereq_lesson_id`` rejects self-prereq
     cycles at the row level. This mirrors ``module_prerequisites``
     from baseline 0001 line 453.
   * ``created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`` — append-only
     style (CreatedAtMixin pattern from baseline). No ``updated_at``,
     no audit columns: this is a pure link row.

Round-trip: downgrade drops the table and the columns + their CHECK
constraints in reverse order. Tested on the live docker DB before the
T3.1 commit lands.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006_courses_unlock_config"
down_revision = "0005_ai_audit_pipeline_run"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "lessons",
        sa.Column(
            "ef_min_unlock",
            sa.Numeric(),
            nullable=False,
            server_default=sa.text("2.0"),
        ),
    )
    op.add_column(
        "lessons",
        sa.Column(
            "tau_unlock",
            sa.Numeric(),
            nullable=False,
            server_default=sa.text("0.8"),
        ),
    )
    op.add_column(
        "lessons",
        sa.Column(
            "requires_interview_pass",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "lessons",
        sa.Column(
            "unlock_rule_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_check_constraint(
        "ck_lessons_ef_min_unlock_range",
        "lessons",
        "ef_min_unlock >= 1.3 AND ef_min_unlock <= 2.5",
    )
    op.create_check_constraint(
        "ck_lessons_tau_unlock_range",
        "lessons",
        "tau_unlock > 0.0 AND tau_unlock <= 1.0",
    )

    op.add_column(
        "modules",
        sa.Column(
            "requires_all_lessons_unlocked",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )

    op.execute(
        """
        CREATE TABLE lesson_prerequisites (
            lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE NO ACTION,
            prereq_lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE NO ACTION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (lesson_id, prereq_lesson_id),
            CONSTRAINT ck_lesson_prerequisites_not_self
                CHECK (lesson_id <> prereq_lesson_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS lesson_prerequisites")
    op.drop_column("modules", "requires_all_lessons_unlocked")
    op.drop_constraint("ck_lessons_tau_unlock_range", "lessons", type_="check")
    op.drop_constraint("ck_lessons_ef_min_unlock_range", "lessons", type_="check")
    op.drop_column("lessons", "unlock_rule_json")
    op.drop_column("lessons", "requires_interview_pass")
    op.drop_column("lessons", "tau_unlock")
    op.drop_column("lessons", "ef_min_unlock")
