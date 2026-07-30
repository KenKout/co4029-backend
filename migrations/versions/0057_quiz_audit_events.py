"""Phase 13: append-only quiz audit-event log.

Revision ID: 0057_quiz_audit_events
Revises: 0056_quiz_access_rules
Create Date: 2026-07-24

Additive/reversible DDL only (new tables + nullable/server-defaulted columns) so
the running app — whose ORM does not yet map these — keeps working.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0057_quiz_audit_events"
down_revision = "0056_quiz_access_rules"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_audit_events",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("event_name", sa.String(64), nullable=False),
        sa.Column("quiz_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("subject_attempt_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("subject_question_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("subject_user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("payload_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["quiz_id"], ["quizzes.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["subject_attempt_id"], ["quiz_attempts.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["subject_question_id"], ["quiz_questions.id"], ondelete="NO ACTION"),
        sa.ForeignKeyConstraint(["subject_user_id"], ["users.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_quiz_audit_events_event_name", "quiz_audit_events", ["event_name"])
    op.create_index("ix_quiz_audit_events_quiz_id", "quiz_audit_events", ["quiz_id"])
    op.create_index("ix_quiz_audit_events_quiz_occurred", "quiz_audit_events", ["quiz_id", "occurred_at"])


def downgrade() -> None:
    op.drop_index("ix_quiz_audit_events_quiz_occurred", table_name="quiz_audit_events")
    op.drop_index("ix_quiz_audit_events_quiz_id", table_name="quiz_audit_events")
    op.drop_index("ix_quiz_audit_events_event_name", table_name="quiz_audit_events")
    op.drop_table("quiz_audit_events")
