"""Relax ck_ai_model_calls_parent_ref for parentless runtime LLM calls

Revision ID: 0018_ai_calls_runtime_parentless
Revises: 0017_material_current_unique
Create Date: 2026-06-27 14:00:00.000000

The baseline 0001 constraint ``ck_ai_model_calls_parent_ref`` demanded
``generation_run_id IS NOT NULL OR processing_job_id IS NOT NULL``. That
holds for the *generation pipeline* (every LLM call belongs to a concrete
generation run or processing job), but it is wrong for *session-runtime*
calls.

The interview follow-up stage (``features/interviews/ai/stages/followup``)
runs synchronously inside ``POST /interview-sessions/{id}/respond`` — not
in the generation pipeline — so it has neither a generation_run nor a
processing_job parent. Its audit rows were therefore rejected by the
constraint, the failure was swallowed by the stage's best-effort
``except`` block, and the poisoned session (``PendingRollbackError``)
then 500'd the whole request on the next flush.

Runtime calls are still attributable: every one carries ``role`` and a
non-null ``stage_name`` (e.g. ``interview_followup``). So we relax the
check to: a row must have SOME attribution — a generation_run, a
processing_job, OR a stage_name. A row with all three null is still a
bug and still rejected.
"""

from __future__ import annotations

from alembic import op

revision = "0018_ai_calls_runtime_parentless"
down_revision = "0017_material_current_unique"
branch_labels = None
depends_on = None

_CONSTRAINT = "ck_ai_model_calls_parent_ref"
_TABLE = "ai_model_calls"

_RELAXED = (
    "generation_run_id IS NOT NULL "
    "OR processing_job_id IS NOT NULL "
    "OR stage_name IS NOT NULL"
)
_STRICT = "generation_run_id IS NOT NULL OR processing_job_id IS NOT NULL"


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(_CONSTRAINT, _TABLE, _RELAXED)


def downgrade() -> None:
    # Clear orphaned runtime rows (no parent, attribution only via
    # stage_name) so the stricter constraint can be re-applied on a
    # round-trip. These are session-runtime audit rows that never
    # existed under the original strict schema.
    op.execute(
        f"""
        DELETE FROM {_TABLE}
        WHERE generation_run_id IS NULL
          AND processing_job_id IS NULL
        """
    )
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(_CONSTRAINT, _TABLE, _STRICT)
