"""One evaluation owner per interview session, plus one gap report per session.

Two structural gaps let two concurrent evaluation jobs corrupt one session's
result. Both are reachable through the normal recovery path, not only under
adversarial timing: ``recover_stalled_evaluations`` enqueues under a per-attempt
job ID (ARQ cannot deduplicate it) while the original job may still be running,
because ``WorkerSettings.job_timeout`` (1200s) overlaps the recovery grace
(900s).

1. No exclusive claim on the evaluation itself
----------------------------------------------
The application-level guard reads ``pass_verdict`` when the job starts and
refuses to overwrite a verdict when it ends, but a full grading pass sits between
those two points, so two jobs that both start while the verdict is NULL both
proceed. The judge is an LLM and is not deterministic, so two passes over the
same transcript can disagree — and the published verdict then belongs to
whichever job commits last. A pass/fail decided by scheduling is a fairness
defect, so exclusivity has to be enforced where the race actually is: the row.

``evaluation_claim_token`` + ``evaluation_claim_expires_at`` support an atomic
conditional UPDATE: claim succeeds only when the session is ungraded AND no
unexpired claim exists. The lease (1800s) is deliberately longer than
``job_timeout``, so a live job always holds a valid lease and never needs to
renew, while a job killed by the timeout has its claim reclaimed automatically.

A partial index covers the reclaim scan: only rows that currently hold a claim
are interesting, which is a vanishing fraction of the table.

2. No uniqueness on a gap report's source session
-------------------------------------------------
``_persist_gap_report`` is read-then-write (SELECT, then INSERT when absent) and
the column had no unique constraint, so two concurrent jobs both saw "no report"
and both inserted. Readers take the newest row, so the stale duplicate stays
invisible while the data is quietly wrong; ``flush_or_conflict`` could not help
because there was no constraint to violate.

The constraint is a UNIQUE INDEX rather than a table constraint because it must
be PARTIAL: quiz-sourced reports leave ``source_interview_session_id`` NULL, and
while Postgres does not treat NULLs as equal (so a plain unique index would also
work today), ``WHERE ... IS NOT NULL`` states the intent and keeps the index
small.

Verified before writing this migration: 144 gap_reports rows, all with a
session reference, zero duplicate ``source_interview_session_id`` values — so the
index builds without a data repair step.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0107_interview_evaluation_claim"
down_revision = "0106_audit_immutability_assess"
branch_labels = None
depends_on = None

_GAP_REPORT_UNIQUE_INDEX = "uq_gap_reports_source_interview_session"
_CLAIM_INDEX = "ix_interview_sessions_evaluation_claim"


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column(
            "evaluation_claim_token",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment=(
                "Owner token of the in-flight evaluation. NULL when unclaimed. "
                "Presented at publish time to prove the lease is still ours."
            ),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "evaluation_claim_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment=(
                "Lease expiry for evaluation_claim_token. Past this instant the "
                "claim is reclaimable (the owning worker is presumed dead)."
            ),
        ),
    )
    # Partial: only claimed rows are ever scanned by the reclaim predicate.
    op.create_index(
        _CLAIM_INDEX,
        "interview_sessions",
        ["evaluation_claim_expires_at"],
        unique=False,
        postgresql_where=sa.text("evaluation_claim_token IS NOT NULL"),
    )
    op.create_index(
        _GAP_REPORT_UNIQUE_INDEX,
        "gap_reports",
        ["source_interview_session_id"],
        unique=True,
        postgresql_where=sa.text("source_interview_session_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_GAP_REPORT_UNIQUE_INDEX, table_name="gap_reports")
    op.drop_index(_CLAIM_INDEX, table_name="interview_sessions")
    op.drop_column("interview_sessions", "evaluation_claim_expires_at")
    op.drop_column("interview_sessions", "evaluation_claim_token")
