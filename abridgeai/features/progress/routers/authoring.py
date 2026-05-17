from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.features.access_control.policies import require_course_permission
from abridgeai.features.progress.schemas.authoring import (
    AtRiskListRead,
    RosterProgressRead,
)
from abridgeai.features.progress.services import monitoring

router = APIRouter(prefix="/teacher", tags=["progress-authoring"])


@router.get(
    "/courses/{course_id}/progress/roster",
    response_model=RosterProgressRead,
    dependencies=[Depends(require_course_permission("course_id", "progress.read.cohort"))],
)
async def get_roster_progress(
    course_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RosterProgressRead:
    return await monitoring.get_roster_progress(db, course_id)


@router.get(
    "/courses/{course_id}/progress/at-risk",
    response_model=AtRiskListRead,
    dependencies=[Depends(require_course_permission("course_id", "progress.read.cohort"))],
)
async def get_at_risk_students(
    course_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AtRiskListRead:
    return await monitoring.get_at_risk_students(db, course_id)


__all__ = ["router"]
