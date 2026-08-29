"""Correlation ids on background work.

Investigating a failed ingest means moving between four surfaces — the job, the
AI calls it made, the HTTP request that started it, and the audit trail — and
today an operator does that by copying UUIDs between screens by hand. Nothing
joins a job to the request that enqueued it, and nothing joins either to the
audit log.

``http_audit_log.request_id`` already exists and is returned to the client as
the ``X-Request-ID`` response header. This adds the same id to the two tables
that record work done *because of* a request, which is what turns three
separate lists into one trail (PRD ADM-013/014).

Nullable, and permanently so. Work also arrives from places no request ever
touched — a worker tick, a reaper sweep, a CLI backfill — and a NOT NULL column
would have to be filled with a fabricated id for those, which would be worse
than an honest absence: a correlation id that correlates to nothing is a trap
for whoever follows it. Existing rows stay NULL; this is not backfillable,
because the requests that created them were never recorded against them.

Note on what is deliberately NOT here: ``processing_jobs.organization_id``.
Resolving a job's tenant means walking a different path per ``entity_type``
through a polymorphic ``entity_id`` with no foreign key, and a denormalized
column would need every enqueue site to populate it or it silently rots. The
job-detail query resolves the owner at read time instead, which cannot drift.
A denormalized column is worth adding when org-scoped job *aggregates* are
needed, and should come with the enqueue-site changes in the same commit.

Revision ID: 0090_job_correlation
Revises: 0089_setting_changes
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0090_job_correlation"
down_revision = "0089_setting_changes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "processing_jobs",
        sa.Column("request_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ai_model_calls",
        sa.Column("request_id", UUID(as_uuid=True), nullable=True),
    )
    # Partial indexes: the overwhelming majority of historical rows are NULL and
    # every useful query here is "find the other work from THIS request", so
    # indexing the nulls would cost storage to answer a question nobody asks.
    op.create_index(
        "ix_processing_jobs_request_id",
        "processing_jobs",
        ["request_id"],
        postgresql_where=sa.text("request_id IS NOT NULL"),
    )
    op.create_index(
        "ix_ai_model_calls_request_id",
        "ai_model_calls",
        ["request_id"],
        postgresql_where=sa.text("request_id IS NOT NULL"),
    )
    # Job detail lists the AI calls a job made. The FK exists but was unindexed,
    # so that lookup was a sequential scan of every call ever recorded.
    op.create_index(
        "ix_ai_model_calls_processing_job_id",
        "ai_model_calls",
        ["processing_job_id"],
        postgresql_where=sa.text("processing_job_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ai_model_calls_processing_job_id", table_name="ai_model_calls"
    )
    op.drop_index("ix_ai_model_calls_request_id", table_name="ai_model_calls")
    op.drop_index("ix_processing_jobs_request_id", table_name="processing_jobs")
    op.drop_column("ai_model_calls", "request_id")
    op.drop_column("processing_jobs", "request_id")
