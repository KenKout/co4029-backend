"""HTTP audit log table (T0.23)

Revision ID: 0011_http_audit_log
Revises: 0010_spaced_repetition
Create Date: 2026-05-17 19:00:00.000000

T0.23 lands the persistence backing the AuditLogMiddleware. T7.5's
``/admin/audit/http`` router was already shipped with a graceful 503 fallback
that probes ``information_schema.tables`` -- the moment this migration runs,
the endpoint flips from 503 to 200.

The column shape matches what
``abridgeai/features/admin/queries/sql/audit/http_audit.sql`` already SELECTs
(id, user_id, session_id, method, path, status_code, latency_ms, ip_address,
user_agent, created_at). Additional columns (request_id, query_params,
headers_meta, path_params, body_size_bytes) are middleware-private and are
NOT projected by the T7.5 query but persisted for forensic value.

Notes:
* ``id`` defaults to ``uuid_generate_v4()`` to match baseline (uuid-ossp is
  installed; pgcrypto is not).
* No SoftDeleteMixin -- this is an append-only audit log; retention is owned
  by a separate cleanup task (out of scope here).
* No GIN trigram index on ``path`` -- baseline does not enable ``pg_trgm``
  and the T7.5 query uses ``LIKE`` which a btree-on-path covers when the
  pattern is left-anchored.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID

revision = "0011_http_audit_log"
down_revision = "0010_spaced_repetition"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "http_audit_log",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("request_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("method", sa.String(length=10), nullable=False),
        sa.Column("path", sa.String(length=2000), nullable=False),
        sa.Column(
            "query_params",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "headers_meta",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
        sa.Column("ip_address", INET(), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("path_params", JSONB(), nullable=True),
        sa.Column("body_size_bytes", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "method IN ('GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS')",
            name="ck_http_audit_log_method",
        ),
        sa.CheckConstraint(
            "status_code BETWEEN 100 AND 599",
            name="ck_http_audit_log_status_code_range",
        ),
        sa.CheckConstraint(
            "latency_ms >= 0",
            name="ck_http_audit_log_latency_nonneg",
        ),
        sa.CheckConstraint(
            "body_size_bytes IS NULL OR body_size_bytes >= 0",
            name="ck_http_audit_log_body_size_nonneg",
        ),
    )
    op.create_index(
        "ix_http_audit_log_created_at",
        "http_audit_log",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_http_audit_log_user_id_created_at",
        "http_audit_log",
        ["user_id", sa.text("created_at DESC")],
        postgresql_where=sa.text("user_id IS NOT NULL"),
    )
    op.create_index(
        "ix_http_audit_log_path",
        "http_audit_log",
        ["path"],
    )


def downgrade() -> None:
    op.drop_index("ix_http_audit_log_path", table_name="http_audit_log")
    op.drop_index("ix_http_audit_log_user_id_created_at", table_name="http_audit_log")
    op.drop_index("ix_http_audit_log_created_at", table_name="http_audit_log")
    op.drop_table("http_audit_log")
