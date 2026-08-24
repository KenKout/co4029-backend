from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.career_paths.schemas import (
    CareerPathDetailPublic,
    CareerPathListPage,
    CareerPathProgressRead,
    CareerPathPublic,
    CareerReadinessSnapshotRead,
    MyCareerEnrollmentRead,
    StartCourseResult,
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


@router.get("/{slug}/detail", response_model=CareerPathDetailPublic)
async def get_published_path_detail(
    slug: str,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CareerPathDetailPublic:
    """Published path plus its stage roadmap.

    Separate from ``GET /{slug}`` so the catalog list and the plain detail
    read keep their slim payload; this one is for the screen where a student
    decides whether to commit to a path.

    Registered BEFORE ``/{slug}`` would be irrelevant (different suffix), but
    it must not be shadowed by it — FastAPI matches in declaration order and
    ``/{slug}`` would happily swallow ``detail`` as a slug if it came first.
    """
    result = await enrollment_service.get_published_path_detail_for_user(
        db, slug=slug, user_id=current_user.user_id
    )
    if result is None:
        raise _not_found(f"CareerPath {slug!r} not found")
    return result


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
    """Stage-aware progress for the calling student.

    This GET has TWO write side-effects and must commit unconditionally:

    * ``get_my_path_progress`` writes the append-only stage latch for any
      stage that has just become complete;
    * ``sync_enrollment_completion`` flips the enrollment to ``completed``
      at 100%.

    Committing only when the enrollment flipped (the original behaviour)
    silently rolled the latch back on every other request, so a stage could
    read complete in the response and still be unlatched in the database —
    which then let a manager delete a stage students had actually finished.
    """
    progress = await enrollment_service.get_my_path_progress(
        db,
        career_path_id=career_path_id,
        student_id=current_user.user_id,
    )
    # "Prepared" milestone: mark the enrollment completed once fully done.
    await enrollment_service.sync_enrollment_completion(
        db,
        career_path_id=career_path_id,
        student_id=current_user.user_id,
        overall_percent=progress.overall_percent,
    )
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
    """Most-recent-first readiness snapshots for the calling student (FR-6.8).

    Each point carries ``formula_version``; the chart must segment or annotate
    where it changes rather than drawing one continuous line across formulas.
    """
    return await readiness_service.get_my_readiness_history(
        db,
        student_id=current_user.user_id,
        career_path_id=career_path_id,
    )


@me_router.post(
    "/{career_path_id}/courses/{course_id}/start",
    response_model=StartCourseResult,
    status_code=status.HTTP_201_CREATED,
)
async def start_course_in_path(
    career_path_id: UUID,
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StartCourseResult:
    """Start ONE course of a path the caller is already assigned to.

    Pattern B lazy enrollment, and the documented carve-out to the locked
    "students cannot self-enroll" decision. It is not general self-enrollment:
    the student cannot name an arbitrary course. The server 403s unless the
    course sits in a **stage of a path the caller is already actively enrolled
    in** — so eligibility is derived entirely from a manager-made assignment.

    A LOCKED stage 403s only under ``enforcement='hard'``. ``soft`` and
    ``advisory`` allow the Start and set ``stage_locked_warning`` instead,
    matching what the manager settings UI promises for those levels.

    Idempotent (``created=false`` when an enrollment already existed).
    ``over_concurrency_cap`` is advisory: the attention cap never blocks.
    """
    try:
        result = await enrollment_service.start_course_in_path(
            db,
            career_path_id=career_path_id,
            course_id=course_id,
            student_id=current_user.user_id,
        )
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": str(exc)},
        ) from exc
    await db.commit()
    return result


__all__ = ["me_router", "router"]
