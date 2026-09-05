"""Manager / faculty-dean dashboard router.

ONE endpoint returning all three Tier-1 sections rather than three endpoints.
The sections are not independent readings: ``counts.courses_blocked`` is the
length of ``blocked_courses``, and ``counts.programs_with_draft`` is derived
from the same program fetch that produces ``programs_needing_attention``. Split
across three requests those numbers could be computed from three different
snapshots, and a tile disagreeing with the table directly beneath it is the kind
of defect nobody reports and everybody stops trusting.

The permission gate is the SAME dependency ``/dept/courses`` uses, so the
population that can already see the org's course list is exactly the population
that can see this page: manager, faculty dean and admin pass; a student or a
plain teacher gets 403.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_any_permission
from abridgeai.features.courses.schemas.management_dashboard import ManagementDashboard
from abridgeai.features.courses.services import (
    management_dashboard as management_dashboard_service,
)

router = APIRouter(prefix="/management", tags=["management-dashboard"])

# Identical to ``routers.assignment._REQUIRE_STAFFING``. Deliberately the same
# set: this page is the decision queue for the courses that list already shows,
# so a caller who can see one must be able to see the other.
_REQUIRE_MANAGEMENT = require_any_permission(
    "course.assign_teacher", "user.role_assign", "system.administer"
)


@router.get("/dashboard", response_model=ManagementDashboard)
async def get_management_dashboard(
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MANAGEMENT)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ManagementDashboard:
    """The caller's decision queue, scoped to their faculty or organization.

    Scope comes from the caller's ROLE ASSIGNMENTS, not from what they author:
    a dean scoped to org units sees their faculties' courses, a manager scoped
    to an organization sees the organization's. The resolved scope is echoed in
    the response so the page can name what it is showing.
    """
    return await management_dashboard_service.get_management_dashboard(
        db, actor=current_user
    )


__all__ = ["router"]
