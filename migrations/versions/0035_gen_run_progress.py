"""Add nullable progress_json to generation_runs for live pipeline progress.

Revision ID: 0035_gen_run_progress
Revises: 0034_quiz_q_lo
Create Date: 2026-07-22 00:00:00.000000

Live-progress checkpointing (quiz AI generation UX). Each pipeline stage
writes a small checkpoint here — current stage, step index, total steps,
started/updated timestamps, and an append-only event log — through a
dedicated short-lived session so the status-poll endpoint can surface
real-time progress while the worker runs.

This is a SEPARATE column from ``config_json`` on purpose: the pipeline
owns ``config_json`` and rewrites the whole column at stage boundaries
(full-column overwrite from an in-memory merge), which would clobber any
progress key written concurrently. ``progress_json`` is touched ONLY by
the checkpoint helper, so there is no clobber and the pipeline's
transaction is never perturbed. Nullable + default '{}' so existing rows
and in-flight runs are unaffected.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# NOTE: alembic_version.version_num is varchar(32); keep this id <= 32 chars.
revision = "0035_gen_run_progress"
down_revision = "0034_quiz_q_lo"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generation_runs",
        sa.Column(
            "progress_json",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("generation_runs", "progress_json")
