"""Security & access router -- ``/admin/security`` (PRD ADM-020).

Requires ``audit.read`` OR ``system.administer``: this is the same class of
data as the audit log, and gating it behind a separate permission would leave
the person who reviews audit trails unable to see the summary of what they are
meant to review.

Org-scoping follows :func:`resolve_admin_scope`, but only partly, and the
response says so: ``http_audit_log`` records the acting user without their
organization, so the request-derived counts are always global. A manager sees
their own org's role and account figures beside platform-wide request counts,
labelled as such, rather than a number that silently means something else than
it appears to.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.admin.routers._scope import resolve_admin_scope
from abridgeai.features.admin.services import security as security_service

router = APIRouter(prefix="/admin/security", tags=["admin", "security"])

_REQUIRE_SECURITY = require_any_permission("audit.read", "system.administer")


class SecuritySummaryOut(BaseModel):
    """Security & access counts for the dashboard's Security row.

    Deliberately absent: severity, risk score and review state. Those require
    alert rules that are still an open product decision (D-03), and a
    fabricated severity trains operators to ignore the real one when it lands.
    """

    as_of: datetime
    window_days: int
    failed_logins: int
    #: ``None`` when there were no failures at all — distinct from 0 sources.
    distinct_failed_ips: int | None
    denied_requests: int
    role_changes: int
    role_revocations: int
    privileged_accounts: int
    active_sessions: int
    #: Which filter each family of numbers actually honoured.
    request_scope: str
    identity_scope: str


@router.get("/summary", response_model=SecuritySummaryOut)
async def get_security_summary(
    user: Annotated[CurrentUser, Depends(_REQUIRE_SECURITY)],
    db: Annotated[AsyncSession, Depends(get_db)],
    window_days: Annotated[int, Query(ge=1, le=90)] = (
        security_service.DEFAULT_WINDOW_DAYS
    ),
    organization_id: Annotated[
        UUID | None,
        Query(
            description=(
                "Narrow the role and account figures to one tenant. Honoured "
                "only for system.administer; request counts stay global either "
                "way."
            )
        ),
    ] = None,
) -> SecuritySummaryOut:
    scope = await resolve_admin_scope(db, user)
    if scope is None and organization_id is not None:
        # Reachable only with system.administer — resolve_admin_scope returns
        # None for that permission alone.
        scope = organization_id
    summary = await security_service.summary(
        db, organization_id=scope, window_days=window_days
    )
    return SecuritySummaryOut(**vars(summary))


__all__ = ["router"]
