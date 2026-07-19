from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.career_paths.schemas import (
    CareerPathListPage,
    CareerPathProgressRead,
    CareerPathPublic,
    CareerReadinessSnapshotRead,
    MyCareerEnrollmentRead,
)
from abridgeai.features.career_paths.services import enrollment as enrollment_service
from abridgeai.features.career_paths.services import readiness as readiness_service

router = APIRouter(prefix="/career-paths", tags=["career-paths-learner"])
me_router = APIRouter(prefix="/me/career-enrollments", tags=["career-paths-learner"])


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


@router.get("", response_model=CareerPathListPage)
async def list_published_paths(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
    cursor: str | None = None,
) -> CareerPathListPage:
    capped = max(1, min(limit, 100))
    try:
        page = await enrollment_service.list_published_paths_for_user(
            db,
            user_id=current_user.user_id,
            limit=capped,
            cursor=cursor,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_cursor", "message": str(exc)},
        ) from exc
    return CareerPathListPage(items=page.items, next_cursor=page.next_cursor)


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
    result = await enrollment_service.list_my_career_enrollments(db, current_user.user_id)
    await db.commit()
    return result


@me_router.get(
    "/{career_path_id}/progress",
    response_model=CareerPathProgressRead,
)
async def get_my_career_path_progress(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathProgressRead:
    progress = await enrollment_service.get_my_path_progress(
        db,
        career_path_id=career_path_id,
        student_id=current_user.user_id,
    )
    # "Prepared" milestone: mark the enrollment completed once fully done.
    flipped = await enrollment_service.sync_enrollment_completion(
        db,
        career_path_id=career_path_id,
        student_id=current_user.user_id,
        overall_percent=progress.overall_percent,
    )
    if flipped:
        await db.commit()
    return progress


@me_router.get(
    "/{career_path_id}/readiness-history",
    response_model=list[CareerReadinessSnapshotRead],
)
async def get_my_readiness_history(
    career_path_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CareerReadinessSnapshotRead]:
    """Most-recent-first readiness snapshots for the calling student (FR-6.8)."""
    return await readiness_service.get_my_readiness_history(
        db,
        student_id=current_user.user_id,
        career_path_id=career_path_id,
    )


__all__ = ["me_router", "router"]
