"""Semantic auth-event log: FR-1.6 as typed events, not inferred HTTP rows.

Until now authentication actions were auditable only as ``http_audit_log``
request records (path + status + user), so "MFA verified" or "an account was
disabled" had to be inferred from the endpoint and status code. That inference
is exactly what FR-1.6 promised NOT to require: "The system shall record
auditable events for authentication actions, including login, logout, MFA
verification, account-security changes, and access-control changes."

``auth_events`` is that typed log. One row per semantic action:

* ``login_succeeded`` / ``login_failed`` / ``logout``
* ``mfa_enrolled`` / ``mfa_enrollment_verified`` / ``mfa_challenge_created`` /
  ``mfa_verified`` / ``mfa_verification_failed`` / ``mfa_disabled`` /
  ``recovery_codes_regenerated``
* ``account_status_changed`` (disable / enable)
* ``role_assigned`` / ``role_revoked`` (access-control changes)

Shape decisions, each deliberate:

* ``user_id`` is SET NULL on user deletion — the trail must survive the
  subject, exactly like ``http_audit_log.user_id``. An audit log that
  disappears with the account it describes can be erased by deleting the
  account.
* ``session_id`` is a BARE uuid with no FK: ``auth_sessions`` is hard-delete
  (CASCADE on the user), and the event must outlive both the session and the
  user.
* ``actor_user_id`` records WHO performed an access-control change (the admin
  granting a role); NULL when the subject acted as themselves (login, logout,
  MFA). Also SET NULL.
* ``organization_id`` SET NULL, nullable: role and status events carry the org
  edge so the security rollup can finally scope them; login/MFA events leave
  it NULL (a Google login has no org until provisioning decides).
* ``detail`` jsonb carries redacted context (reason codes, provider, scope
  kinds) — never codes, tokens or secrets.

Append-only like every other audit store: the 0105 ``audit_log_immutable()``
function is reused, so all audit tables share one definition of the rule and
tests/support/db_graph.py's catalog probe (which joins pg_trigger to the
FUNCTION) discovers this table automatically. The retention sweep gets a
matching window setting (``audit.auth_event_retention_days``, default 90,
registered in the same commit) per the guarded-tables agreement test.

Revision ID: 0110_auth_events
Revises: 0109_one_org_per_user
Create Date: 2026-09-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0110_auth_events"
down_revision = "0109_one_org_per_user"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "auth_events",
        sa.Column(
            "id", sa.UUID(), nullable=False, server_default=sa.text("uuid_generate_v4()")
        ),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("event_type", sa.String(40), nullable=False),
        sa.Column(
            "user_id",
            sa.UUID(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "actor_user_id",
            sa.UUID(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "organization_id",
            sa.UUID(),
            sa.ForeignKey("organizations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Bare uuid on purpose: auth_sessions is hard-delete and the event
        # must outlive it. See the migration docstring.
        sa.Column("session_id", sa.UUID(), nullable=True),
        sa.Column("detail", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "event_type IN ("
            "'login_succeeded', 'login_failed', 'logout', "
            "'mfa_enrolled', 'mfa_enrollment_verified', 'mfa_challenge_created', "
            "'mfa_verified', 'mfa_verification_failed', 'mfa_disabled', "
            "'recovery_codes_regenerated', "
            "'account_status_changed', 'role_assigned', 'role_revoked')",
            name="ck_auth_events_event_type",
        ),
    )
    op.create_index(
        "ix_auth_events_occurred_at", "auth_events", ["occurred_at"]
    )
    op.create_index("ix_auth_events_user_id", "auth_events", ["user_id"])
    op.create_index("ix_auth_events_event_type", "auth_events", ["event_type"])

    # Shared 0105 rule; discovered automatically by db_graph.py's probe.
    op.execute(
        "CREATE TRIGGER audit_log_immutable_auth_events "
        "BEFORE UPDATE OR DELETE ON auth_events "
        "FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_log_immutable_auth_events ON auth_events;")
    op.drop_index("ix_auth_events_event_type", table_name="auth_events")
    op.drop_index("ix_auth_events_user_id", table_name="auth_events")
    op.drop_index("ix_auth_events_occurred_at", table_name="auth_events")
    op.drop_table("auth_events")
