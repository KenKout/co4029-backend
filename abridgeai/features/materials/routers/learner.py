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
    return material


@router.get("/{material_id}/stream-url", response_model=MaterialStreamUrl)
async def get_material_stream_url(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialStreamUrl:
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
    chunks = await catalog_service.list_visible_chunks_preview(
        db, material_id, current_user.user_id, limit
    )
    if chunks is None:
        raise _not_found(material_id)
    return chunks


__all__ = ["router"]
