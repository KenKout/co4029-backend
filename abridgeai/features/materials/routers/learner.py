"""Materials learner router (T4.6).

Three GET endpoints under prefix ``/materials`` exposing student-side
catalog reads + a presigned stream URL. Each depends on
:func:`get_current_user` only — visibility is the security boundary
(plan §5053): the ``visible_to_students=TRUE AND processing_status='ready'``
predicate at the query layer is the single access check, and ``None``
returns from the service are mapped to HTTP 404 (NOT 403; plan §5075)
so existence is never leaked.

Routers→services boundary: this module imports
:mod:`features.materials.services.catalog` only, never the queries
package directly. Enforced by the import-linter "Routers do not call
queries directly" contract (T0.4).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.materials.schemas.public import (
    MaterialPublic,
    MaterialStreamUrl,
)
from abridgeai.features.materials.services import catalog as catalog_service
from abridgeai.features.materials.services.catalog import ChunkPreview

router = APIRouter(prefix="/materials", tags=["materials-learner"])


def _not_found(material_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": "material", "id": str(material_id)},
    )


async def _ensure_owning_lesson_unlocked(
    db: AsyncSession, current_user: CurrentUser, lesson_id: UUID
) -> None:
    """FR-4.5 — a material is only readable when its owning lesson is
    unlocked for the student (prerequisites / SR coverage τ / interview
    pass). Mirrors the courses learner-router gate; 403 payload shape is
    identical so the SPA handles both uniformly. Lazy import avoids a
    circular import at app start-up; disabled by
    ``LESSON_GATING_ENFORCED=false``.
    """
    from abridgeai.core.config import get_settings  # noqa: PLC0415

    if not get_settings().lesson_gating_enforced:
        return

    from abridgeai.features.spaced_repetition.api import public as sr_public  # noqa: PLC0415

    unlock = await sr_public.check_lesson_unlock(
        db, student_id=current_user.user_id, lesson_id=lesson_id
    )
    if unlock.eligible:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "lesson_locked",
            "lesson_id": str(lesson_id),
            "current_ratio": unlock.current_ratio,
            "required_ratio": unlock.required_ratio,
            "total_cards": unlock.total_cards,
            "passing_cards": unlock.passing_cards,
            "prerequisites_met": unlock.prereq_lesson_ids_unlocked,
            "interview_pass_required": unlock.interview_pass_required,
            "interview_passed": unlock.interview_passed,
            "next_unlock_estimate": unlock.next_unlock_estimate,
        },
    )


@router.get("/{material_id}", response_model=MaterialPublic)
async def get_material(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialPublic:
    material = await catalog_service.get_visible_material_for_user(
        db, material_id, current_user.user_id
    )
    if material is None:
        raise _not_found(material_id)
    await _ensure_owning_lesson_unlocked(db, current_user, material.lesson_id)
    return material


@router.get("/{material_id}/stream-url", response_model=MaterialStreamUrl)
async def get_material_stream_url(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialStreamUrl:
    material = await catalog_service.get_visible_material_for_user(
        db, material_id, current_user.user_id
    )
    if material is None:
        raise _not_found(material_id)
    await _ensure_owning_lesson_unlocked(db, current_user, material.lesson_id)
    stream = await catalog_service.get_stream_url_for_material(
        db, material_id, current_user.user_id
    )
    if stream is None:
        raise _not_found(material_id)
    return stream


@router.get("/{material_id}/chunks/preview", response_model=list[ChunkPreview])
async def get_material_chunks_preview(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[ChunkPreview]:
    material = await catalog_service.get_visible_material_for_user(
        db, material_id, current_user.user_id
    )
    if material is None:
        raise _not_found(material_id)
    await _ensure_owning_lesson_unlocked(db, current_user, material.lesson_id)
    chunks = await catalog_service.list_visible_chunks_preview(
        db, material_id, current_user.user_id, limit
    )
    if chunks is None:
        raise _not_found(material_id)
    return chunks


__all__ = ["router"]
