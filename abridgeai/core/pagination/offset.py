"""Offset pagination with server-side search + whitelisted sort.

The additive companion to :mod:`cursor` — for **page-numbered** data tables
that need a total count, page jumps, and column sort (the DataTable UX).
Cursor pagination stays the right tool for infinite-scroll feeds.

Usage from a service::

    from sqlalchemy import select
    from abridgeai.core.pagination import paginate

    page = await paginate(
        db,
        select(Organization),
        page=page, page_size=page_size,
        search=search, search_columns=[Organization.name, Organization.slug],
        sort=sort, sort_dir=sort_dir,
        sortable={"name": Organization.name, "created_at": Organization.created_at},
        default_order=[Organization.id],  # stable tiebreak
    )

``stmt`` must select a single ORM entity; ``Page.items`` are entity instances.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement, Select

_MAX_PAGE_SIZE = 200


@dataclass(frozen=True)
class Page[T]:
    """Service-layer result of an offset page."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class PageResponse[T](BaseModel):
    """API-layer projection of :class:`Page` (used as a FastAPI ``response_model``)."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


async def paginate(
    db: AsyncSession,
    stmt: Select,
    *,
    page: int,
    page_size: int,
    search: str | None = None,
    search_columns: Sequence[ColumnElement[Any] | Any] = (),
    sort: str | None = None,
    sort_dir: str = "asc",
    sortable: Mapping[str, ColumnElement[Any] | Any] | None = None,
    default_order: Sequence[ColumnElement[Any] | Any] = (),
) -> Page[Any]:
    """Offset-paginate an ORM ``Select`` with optional search + safe sort.

    * ``search`` — case-insensitive substring matched with ``OR`` across
      ``search_columns``; ignored when empty or no columns given.
    * ``sort`` — looked up in the ``sortable`` **whitelist** (api-name →
      column); unknown keys are ignored so callers can never sort by an
      arbitrary/injectable column. ``default_order`` is always appended to
      keep the page boundary stable.
    * ``total`` counts the filtered rows (before limit/offset); pages are
      0-indexed and ``page_size`` is clamped to ``1..200``.
    """
    page = max(0, page)
    page_size = _clamp(page_size, 1, _MAX_PAGE_SIZE)
    sortable = sortable or {}

    term = (search or "").strip()
    if term and search_columns:
        like = f"%{term}%"
        stmt = stmt.where(or_(*(col.ilike(like) for col in search_columns)))

    count_stmt = select(func.count()).select_from(stmt.order_by(None).subquery())
    exec_opts = stmt.get_execution_options()
    if exec_opts:
        count_stmt = count_stmt.execution_options(**exec_opts)
    total = (await db.execute(count_stmt)).scalar_one()

    order_by: list[ColumnElement[Any]] = []
    sort_col = sortable.get(sort) if sort else None
    if sort_col is not None:
        order_by.append(sort_col.desc() if sort_dir == "desc" else sort_col.asc())
    order_by.extend(default_order)
    if order_by:
        stmt = stmt.order_by(None).order_by(*order_by)

    stmt = stmt.limit(page_size).offset(page * page_size)
    items = list((await db.execute(stmt)).scalars().all())

    total_pages = (int(total) + page_size - 1) // page_size
    return Page(
        items=items,
        total=int(total),
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


__all__ = ["Page", "PageResponse", "paginate"]
