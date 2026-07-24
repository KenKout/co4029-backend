"""Admin CRUD over ``ai_model_pricing`` (config replacement for PRICE_TABLE).

Each write busts the in-process pricing cache in
``abridgeai.ai.llm.pricing`` so new rates apply to the very next LLM/
embedding call instead of waiting out the cache TTL.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.llm.pricing import invalidate_pricing_cache
from abridgeai.ai.models import AIModelPricing
from abridgeai.core.exceptions import ConflictError, NotFoundError


def _to_dict(row: AIModelPricing) -> dict[str, Any]:
    return {
        "id": row.id,
        "model_name": row.model_name,
        "input_usd_per_1m": float(row.input_usd_per_1m),
        "output_usd_per_1m": float(row.output_usd_per_1m),
        "notes": row.notes,
        "updated_by": row.updated_by,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


async def list_pricing(db: AsyncSession) -> list[dict[str, Any]]:
    rows = (
        (await db.execute(select(AIModelPricing).order_by(AIModelPricing.model_name.asc())))
        .scalars()
        .all()
    )
    return [_to_dict(r) for r in rows]


async def create_pricing(
    db: AsyncSession,
    *,
    model_name: str,
    input_usd_per_1m: Decimal,
    output_usd_per_1m: Decimal,
    notes: str | None,
    updated_by: UUID | None,
) -> dict[str, Any]:
    row = AIModelPricing(
        model_name=model_name,
        input_usd_per_1m=input_usd_per_1m,
        output_usd_per_1m=output_usd_per_1m,
        notes=notes,
        updated_by=updated_by,
    )
    db.add(row)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise ConflictError(f"pricing row for model '{model_name}' already exists") from exc
    await db.commit()
    invalidate_pricing_cache()
    return _to_dict(row)


async def update_pricing(
    db: AsyncSession,
    pricing_id: UUID,
    *,
    input_usd_per_1m: Decimal | None,
    output_usd_per_1m: Decimal | None,
    notes: str | None,
    notes_provided: bool,
    updated_by: UUID | None,
) -> dict[str, Any]:
    row = (
        await db.execute(select(AIModelPricing).where(AIModelPricing.id == pricing_id))
    ).scalar_one_or_none()
    if row is None:
        raise NotFoundError(f"pricing row '{pricing_id}' not found")

    if input_usd_per_1m is not None:
        row.input_usd_per_1m = input_usd_per_1m
    if output_usd_per_1m is not None:
        row.output_usd_per_1m = output_usd_per_1m
    if notes_provided:
        row.notes = notes
    row.updated_by = updated_by

    await db.flush()
    await db.commit()
    invalidate_pricing_cache()
    return _to_dict(row)


async def delete_pricing(db: AsyncSession, pricing_id: UUID) -> None:
    row = (
        await db.execute(select(AIModelPricing).where(AIModelPricing.id == pricing_id))
    ).scalar_one_or_none()
    if row is None:
        raise NotFoundError(f"pricing row '{pricing_id}' not found")
    await db.delete(row)
    await db.commit()
    invalidate_pricing_cache()


__all__ = [
    "create_pricing",
    "delete_pricing",
    "list_pricing",
    "update_pricing",
]
