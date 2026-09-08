"""Auth-event recorder (FR-1.6).

One function, one rule: an event is written in the SAME transaction as the
action it describes, so a committed action always has its event and a rolled
back action leaves none. The two exceptions prove the rule:

* ``login_failed`` — the request transaction ROLLS BACK when the service
  raises (``get_db`` never commits), so the event must commit itself before
  the exception propagates. It is also best-effort: an audit failure must
  never replace the 403 the caller should see.
* ``login_succeeded`` on paths that predate a session (none today) — not
  applicable; the success path commits inside ``_issue_tokens`` and the event
  rides that transaction.

Layering: lives in ``identity.services`` because ``users``/``auth_sessions``/
``mfa_*`` are identity's entities, but it is deliberately import-light so the
admin and access-control features can call it without a cycle (it imports no
other feature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.security import utcnow
from abridgeai.features.identity.models import AuthEvent

if TYPE_CHECKING:
    # Indented on purpose: identity services must not import sqlalchemy at
    # module level (tests/unit/test_identity_services.py enforces it —
    # services talk to the DB through the injected session, never the ORM
    # directly). AsyncSession appears only in annotations, and
    # `from __future__ import annotations` keeps those lazy.
    from sqlalchemy.ext.asyncio import AsyncSession

#: Frozen v1 event registry — the only names the recorder accepts. Mirrors the
#: CHECK constraint ``ck_auth_events_event_type``; keep the two in sync.
AUTH_EVENT_TYPES = frozenset(
    {
        "login_succeeded",
        "login_failed",
        "logout",
        "mfa_enrolled",
        "mfa_enrollment_verified",
        "mfa_challenge_created",
        "mfa_verified",
        "mfa_verification_failed",
        "mfa_disabled",
        "recovery_codes_regenerated",
        "account_status_changed",
        "role_assigned",
        "role_revoked",
    }
)


async def record_auth_event(
    db: AsyncSession,
    *,
    event_type: str,
    user_id: UUID | None = None,
    actor_user_id: UUID | None = None,
    organization_id: UUID | None = None,
    session_id: UUID | None = None,
    detail: dict[str, Any] | None = None,
) -> AuthEvent:
    """Append one auth event, participating in the caller's transaction.

    Raises ValueError for an unregistered event type so a typo cannot create
    a silent, unqueryable event class (same rule as the quiz audit recorder).
    """
    if event_type not in AUTH_EVENT_TYPES:
        raise ValueError(f"Unknown auth event type: {event_type}")
    row = AuthEvent(
        event_type=event_type,
        user_id=user_id,
        actor_user_id=actor_user_id,
        organization_id=organization_id,
        session_id=session_id,
        detail=detail or {},
        occurred_at=utcnow(),
    )
    db.add(row)
    return row


async def record_auth_event_standalone(
    event_type: str,
    *,
    user_id: UUID | None = None,
    actor_user_id: UUID | None = None,
    organization_id: UUID | None = None,
    session_id: UUID | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Commit one auth event on its OWN transaction, best-effort.

    For failure paths whose request transaction is about to roll back
    (``login_failed``): the event must outlive the rollback, and an audit
    failure must never mask the caller's error.
    """
    from abridgeai.core.db import get_sessionmaker

    if event_type not in AUTH_EVENT_TYPES:
        raise ValueError(f"Unknown auth event type: {event_type}")
    try:
        async with get_sessionmaker()() as session:
            session.add(
                AuthEvent(
                    event_type=event_type,
                    user_id=user_id,
                    actor_user_id=actor_user_id,
                    organization_id=organization_id,
                    session_id=session_id,
                    detail=detail or {},
                    occurred_at=utcnow(),
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — auditing must never break the caller
        from abridgeai.core.observability.logging import get_logger

        get_logger(__name__).warning(
            "auth_event_standalone_failed",
            event_type=event_type,
            error=str(exc),
            error_type=type(exc).__name__,
        )


__all__ = ["AUTH_EVENT_TYPES", "record_auth_event", "record_auth_event_standalone"]
