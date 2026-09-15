"""Durable interview audio-recording lifecycle (LiveKit Egress).

Revision ID: 0123_interview_recordings
Revises: 0122_path_drop_requests

Adds ``interview_recordings`` — one row per interview session (UNIQUE
``session_id``), the queryable recording lifecycle the gap-report replay
feature needs:

  ``egress_id`` UNIQUE — the LiveKit Egress job id both the webhook and the
  reconciliation sweep key on. It is nullable only during the brief claim-to-
  dispatch window; every provider-accepted job has a non-null id.
* ``status`` CHECK (pending/active/complete/failed/cancelled/expired).
* ``storage_object_id`` FK SET NULL — the canonical playback source, mirrored
  onto ``interview_sessions.recording_object_id`` by the same transaction that
  attaches it.
* consent provenance (policy version + timestamp + audio_only scope) — the
  service layer refuses to claim a row without them, and the DB now records
  exactly what the candidate agreed to.
* ``retention_delete_at`` + the ``deleted_at`` tombstone for the 30-day
  retention sweeper: delete the S3 object, clear both FKs, keep the row as
  audit evidence.
* bounded reconciliation state (attempts / last-seen / last-error).

No backfill: the feature ships disabled-by-default and no session has ever
been recorded. ``interview_sessions.recording_object_id`` keeps its role as
the final playback pointer; nothing existing is rewritten.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0123_interview_recordings"
down_revision = "0122_path_drop_requests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interview_recordings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("interview_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "storage_object_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storage_objects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("egress_id", sa.String(length=64), nullable=True),
        sa.Column("room_name", sa.String(length=255), nullable=False),
        sa.Column(
            "status", sa.String(length=20), nullable=False, server_default=sa.text("'pending'")
        ),
        sa.Column(
            "destination_file", sa.String(length=500), nullable=False, server_default=sa.text("''")
        ),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("consent_policy_version", sa.String(length=32), nullable=False),
        sa.Column("consented_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "consent_scope", sa.String(length=20), nullable=False, server_default=sa.text("'audio_only'")
        ),
        sa.Column("retention_delete_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_reconcile_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reconcile_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")
        ),
        sa.UniqueConstraint("session_id", name="uq_interview_recordings_session"),
        sa.UniqueConstraint("egress_id", name="uq_interview_recordings_egress"),
        sa.CheckConstraint(
            "status IN ('pending', 'active', 'complete', 'failed', 'cancelled', 'expired')",
            name="ck_interview_recordings_status",
        ),
        sa.CheckConstraint(
            "consent_scope IN ('audio_only')",
            name="ck_interview_recordings_consent_scope",
        ),
        sa.CheckConstraint(
            "(status = 'expired' AND deleted_at IS NOT NULL AND storage_object_id IS NULL) "
            "OR (status <> 'expired' AND deleted_at IS NULL)",
            name="ck_interview_recordings_tombstone",
        ),
    )
    # The retention sweeper and the reconciliation sweep both scan by status.
    op.create_index("ix_interview_recordings_status", "interview_recordings", ["status"])


def downgrade() -> None:
    # Recordings are deletable by design (retention); nothing here is
    # load-bearing academic record. Drop the table whole.
    op.drop_index("ix_interview_recordings_status", table_name="interview_recordings")
    op.drop_table("interview_recordings")
