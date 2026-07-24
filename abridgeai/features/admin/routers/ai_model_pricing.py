"""AI model pricing CRUD router — ``/admin/ai/pricing/*``.

Deliberately a *separate* router from ``ai_costs.py`` (the read-only cost
dashboard). That dashboard has a regression guard
(``test_no_hard_block_endpoint_exists``) asserting it exposes no write
endpoints, per an explicit user decision that admins inspect spend but the
dashboard never blocks users on cost. Pricing configuration is an unrelated
concern — it lets admins change how cost is *computed*, not enforce limits
— so it gets its own prefix and permission-gated write routes.

Authorization: reads require ``ai.processing.read`` or
``system.administer``; writes require ``system.administer`` (mutates a
value that feeds every future cost calculation).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_permission,
)
from abridgeai.features.admin.services import ai_model_pricing as pricing_service

router = APIRouter(prefix="/admin/ai/pricing", tags=["admin", "ai", "pricing"])

_REQUIRE_READ = require_any_permission("ai.processing.read", "system.administer")
_REQUIRE_WRITE = require_permission("system.administer")


class ModelPricingOut(BaseModel):
    id: UUID
    model_name: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    notes: str | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime


class ModelPricingCreate(BaseModel):
    model_name: str = Field(min_length=1, max_length=100)
    input_usd_per_1m: float = Field(ge=0)
    output_usd_per_1m: float = Field(ge=0)
    notes: str | None = Field(default=None, max_length=255)


class ModelPricingUpdate(BaseModel):
    input_usd_per_1m: float | None = Field(default=None, ge=0)
    output_usd_per_1m: float | None = Field(default=None, ge=0)
    notes: str | None = Field(default=None, max_length=255)


@router.get("", response_model=list[ModelPricingOut])
async def list_pricing(
    _user: Annotated[CurrentUser, Depends(_REQUIRE_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModelPricingOut]:
    rows = await pricing_service.list_pricing(db)
    return [ModelPricingOut.model_validate(r, from_attributes=True) for r in rows]


@router.post("", response_model=ModelPricingOut, status_code=status.HTTP_201_CREATED)
async def create_pricing(
    body: ModelPricingCreate,
    user: Annotated[CurrentUser, Depends(_REQUIRE_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelPricingOut:
    try:
        row = await pricing_service.create_pricing(
            db,
            model_name=body.model_name,
            input_usd_per_1m=Decimal(str(body.input_usd_per_1m)),
            output_usd_per_1m=Decimal(str(body.output_usd_per_1m)),
            notes=body.notes,
            updated_by=user.user_id,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ModelPricingOut.model_validate(row, from_attributes=True)


@router.patch("/{pricing_id}", response_model=ModelPricingOut)
async def update_pricing(
    pricing_id: UUID,
    body: ModelPricingUpdate,
    user: Annotated[CurrentUser, Depends(_REQUIRE_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelPricingOut:
    try:
        row = await pricing_service.update_pricing(
            db,
            pricing_id=pricing_id,
            input_usd_per_1m=(
                Decimal(str(body.input_usd_per_1m)) if body.input_usd_per_1m is not None else None
            ),
            output_usd_per_1m=(
                Decimal(str(body.output_usd_per_1m)) if body.output_usd_per_1m is not None else None
            ),
            notes=body.notes,
            notes_provided="notes" in body.model_fields_set,
            updated_by=user.user_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ModelPricingOut.model_validate(row, from_attributes=True)


@router.delete("/{pricing_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pricing(
    pricing_id: UUID,
    _user: Annotated[CurrentUser, Depends(_REQUIRE_WRITE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await pricing_service.delete_pricing(db, pricing_id=pricing_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


__all__ = ["router"]
