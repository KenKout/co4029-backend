from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.career_paths.schemas import (
    CareerPathProgressRead,
    CareerPathPublic,
    MyCareerEnrollmentRead,
)
from abridgeai.features.career_paths.services import enrollment as enrollment_service

router = APIRouter(prefix="/career-paths", tags=["career-paths-learner"])
me_router = APIRouter(prefix="/me/career-enrollments", tags=["career-paths-learner"])


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


@router.get("", response_model=list[CareerPathPublic])
async def list_published_paths(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
) -> list[CareerPathPublic]:
    capped = max(1, min(limit, 100))
    safe_offset = max(0, offset)
    return await enrollment_service.list_published_paths_for_user(
        db,
        user_id=current_user.user_id,
        limit=capped,
        offset=safe_offset,
    )


@router.get("/{slug}", response_model=CareerPathPublic)
async def get_published_path(
    slug: str,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathPublic:
    result = await enrollment_service.get_published_path_for_user(
        db, slug=slug, user_id=current_user.user_id
    )
    if result is None:
        raise _not_found(f"CareerPath {slug!r} not found")
    return result


@me_router.get("", response_model=list[MyCareerEnrollmentRead])
async def list_my_career_enrollments(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MyCareerEnrollmentRead]:
    return await enrollment_service.list_my_career_enrollments(db, current_user.user_id)


@me_router.get(
    "/{career_path_id}/progress",
    response_model=CareerPathProgressRead,
)
async def get_my_career_path_progress(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathProgressRead:
    return await enrollment_service.get_my_path_progress(
        db,
        career_path_id=career_path_id,
        student_id=current_user.user_id,
    )


__all__ = ["me_router", "router"]
