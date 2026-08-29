"""Security & access rollup (PRD ADM-020).

Counts, not scores. Every field is a defined quantity an operator can act on
and verify by clicking through; none of them is a severity, a risk rating or a
"review state", because those need alert rules this deployment has not decided
on yet (open decision D-03). Shipping an invented threshold would produce a
number that looks authoritative and is not, which is worse than shipping the
count and letting the operator judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.admin.queries import security as security_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_WINDOW_DAYS = 7


@dataclass(frozen=True)
class SecuritySummary:
    """Windowed security counts plus the point-in-time inventory."""

    as_of: datetime
    window_days: int
    #: 401/403 on the auth surface.
    failed_logins: int
    #: Distinct sources behind those failures. ``None`` when there were none —
    #: "no failures" is not "failures from zero sources".
    distinct_failed_ips: int | None
    #: 403 elsewhere: a signed-in caller reaching for something not theirs.
    denied_requests: int
    role_changes: int
    role_revocations: int
    #: Point-in-time, not windowed: active users holding admin/manager/hod.
    privileged_accounts: int
    active_sessions: int
    #: http_audit_log has no organization edge, so the request-derived counts
    #: stay global even for an org-scoped caller. Reported, not hidden.
    request_scope: str
    #: Role and account figures DO honour the organization filter.
    identity_scope: str


async def summary(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: datetime | None = None,
) -> SecuritySummary:
    evaluated_at = now or datetime.now(tz=UTC)
    row = await security_queries.summary(
        db,
        now=evaluated_at,
        since=evaluated_at - timedelta(days=window_days),
        organization_id=organization_id,
    )
    distinct_ips = row["distinct_failed_ips"]
    return SecuritySummary(
        as_of=row["as_of"],
        window_days=window_days,
        failed_logins=int(row["failed_logins"] or 0),
        distinct_failed_ips=None if distinct_ips is None else int(distinct_ips),
        denied_requests=int(row["denied_requests"] or 0),
        role_changes=int(row["role_changes"] or 0),
        role_revocations=int(row["role_revocations"] or 0),
        privileged_accounts=int(row["privileged_accounts"] or 0),
        active_sessions=int(row["active_sessions"] or 0),
        request_scope="global",
        identity_scope="organization" if organization_id else "global",
    )


__all__ = ["DEFAULT_WINDOW_DAYS", "SecuritySummary", "summary"]
