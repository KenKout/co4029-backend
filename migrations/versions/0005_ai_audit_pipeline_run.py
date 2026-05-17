"""ai_model_calls add pipeline_run_id and stage_name

Revision ID: 0005_ai_model_calls_pipeline_run_stage
Revises: 0004_seed_permission_catalog
Create Date: 2026-05-17 00:00:00.000000

Renames ``pipeline_stage`` → ``stage_name`` (per Reconciliation §B2 — the
column already exists from baseline 0001, so we extend, not duplicate),
adds ``pipeline_run_id UUID NULL`` as a NEW abstract pipeline-lifecycle
identifier, and adds composite index
``ix_ai_model_calls_role_stage_created (role, stage_name, called_at)`` to
support the upcoming AI cost dashboard (T0.27).

Three FK-shaped columns coexist after this migration with distinct
semantics:

* ``pipeline_run_id`` (NEW): abstract pipeline lifecycle UUID. One value
  per logical generation run; links all stages (ideation → generation →
  validation → embedding → extraction → kg_build) under one identifier.
  NOT a foreign key — the lifecycle id is owned by the pipeline
  orchestrator, not a row.
* ``generation_run_id`` (existing): FK → ``generation_runs.id``.
  Identifies the concrete QuizGenerationRun row this call belongs to.
* ``processing_job_id`` (existing): FK → ``processing_jobs.id``.
  Identifies the concrete ProcessingJob row this call belongs to.

Naming note
-----------
Plan §3375 referenced the trailing index column as ``created_at`` but
``ai_model_calls`` uses ``called_at`` since baseline 0001 (there is no
``created_at`` on this table). The composite index keeps the planned
*name* ``ix_ai_model_calls_role_stage_created`` (the QA scenario asserts
that exact name) but indexes the actual ``called_at`` column. The column
order ``(role, stage_name, called_at)`` is chosen for the dashboard's
typical access pattern: ``WHERE role=? AND stage_name=? ORDER BY
called_at DESC``.

Round-trip
----------
The downgrade reverses all three changes: drops the composite index,
drops ``pipeline_run_id``, renames ``stage_name`` back to
``pipeline_stage``, and restores its VARCHAR(40) length. Safe on the
test database which exercises upgrade → downgrade -1 → upgrade.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_ai_audit_pipeline_run"
down_revision = "0004_seed_permission_catalog"
branch_labels = None
depends_on = None


_PIPELINE_RUN_ID_COMMENT = (
    "Abstract pipeline lifecycle UUID. One value per logical generation run; "
    "links all stages (ideation, generation, validation, embedding, extraction, "
    "kg_build) under one identifier. Distinct from generation_run_id (FK to "
    "generation_runs row) and processing_job_id (FK to processing_jobs row)."
)
_GENERATION_RUN_ID_COMMENT = (
    "FK to generation_runs.id. Concrete QuizGenerationRun row this call "
    "belongs to."
)
_PROCESSING_JOB_ID_COMMENT = (
    "FK to processing_jobs.id. Concrete ProcessingJob row this call belongs to."
)
_STAGE_NAME_COMMENT = (
    "Explicit stage label: ideation, generation, validation, embedding, "
    "extraction, kg_build. Renamed from pipeline_stage in 0005."
)


def _comment_sql(column: str, body: str) -> str:
    escaped = body.replace("'", "''")
    return f"COMMENT ON COLUMN ai_model_calls.{column} IS '{escaped}'"


def upgrade() -> None:
    op.alter_column(
        "ai_model_calls",
        "pipeline_stage",
        new_column_name="stage_name",
        existing_type=sa.String(length=40),
        type_=sa.String(length=50),
        existing_nullable=True,
    )

    op.add_column(
        "ai_model_calls",
        sa.Column(
            "pipeline_run_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )

    op.execute(_comment_sql("pipeline_run_id", _PIPELINE_RUN_ID_COMMENT))
    op.execute(_comment_sql("generation_run_id", _GENERATION_RUN_ID_COMMENT))
    op.execute(_comment_sql("processing_job_id", _PROCESSING_JOB_ID_COMMENT))
    op.execute(_comment_sql("stage_name", _STAGE_NAME_COMMENT))

    op.create_index(
        "ix_ai_model_calls_role_stage_created",
        "ai_model_calls",
        ["role", "stage_name", "called_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ai_model_calls_role_stage_created",
        table_name="ai_model_calls",
    )
    op.execute("COMMENT ON COLUMN ai_model_calls.generation_run_id IS NULL")
    op.execute("COMMENT ON COLUMN ai_model_calls.processing_job_id IS NULL")
    op.drop_column("ai_model_calls", "pipeline_run_id")
    op.alter_column(
        "ai_model_calls",
        "stage_name",
        new_column_name="pipeline_stage",
        existing_type=sa.String(length=50),
        type_=sa.String(length=40),
        existing_nullable=True,
    )
