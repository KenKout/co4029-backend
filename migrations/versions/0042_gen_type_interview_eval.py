"""Add 'interview_evaluation' to generation_runs.generation_type check

Revision ID: 0042_gen_type_interview_eval
Revises: 0041_pricing_per_million
Create Date: 2026-07-24 00:00:00.000000

Interview evaluation + gap-report used to run "bare": the LLM calls landed
in ``ai_model_calls`` with NULL ``generation_run_id``/``processing_job_id``,
so they never appeared on the ``/admin/processing`` ops dashboard (which
reads ``generation_runs`` UNION ``processing_jobs``). We now create a
``generation_runs`` row of type ``interview_evaluation`` for each
``evaluate_and_generate_report`` run so the work is tracked like quiz /
interview generation.

This migration only widens the CHECK constraint to admit the new type. The
service change (creating the run) and the backfill of historical rows are
separate; this DDL is the value-safe, reproducible prerequisite.

The baseline DDL created the constraint inline, so Postgres named it
``generation_runs_generation_type_check`` (not the ORM's ``ck_`` name). We
drop by that concrete name and recreate with the widened allow-list.
"""

from __future__ import annotations

from alembic import op

revision = "0042_gen_type_interview_eval"
down_revision = "0041_pricing_per_million"
branch_labels = None
depends_on = None

_TABLE = "generation_runs"
_CONSTRAINT = "generation_runs_generation_type_check"

_OLD_TYPES = "'quiz', 'interview', 'knowledge_graph', 'material_index'"
_NEW_TYPES = "'quiz', 'interview', 'knowledge_graph', 'material_index', 'interview_evaluation'"


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        _TABLE,
        f"generation_type IN ({_NEW_TYPES})",
    )


def downgrade() -> None:
    # Reversible only if no rows use the new type; delete them first so the
    # narrower constraint can be re-applied without a violation.
    op.execute(f"DELETE FROM {_TABLE} WHERE generation_type = 'interview_evaluation'")
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        _TABLE,
        f"generation_type IN ({_OLD_TYPES})",
    )
