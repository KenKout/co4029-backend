"""User-related identity queries (data accessors only).

Each function takes ``db: AsyncSession`` as the first argument and returns the
ORM model (or ``None``). No business logic, no Pydantic conversion. Service
layer (T1.7) composes these and shapes the response.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.pagination import Page, paginate
from abridgeai.features.identity.models import AuthIdentity, User, UserProfile


async def get_user(db: AsyncSession, user_id: UUID) -> User | None:
    return await db.get(User, user_id)


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Fetch a user by email.

    ``users.primary_email`` is ``CITEXT`` (baseline DDL §0001), so postgres
    matches case-insensitively at the type level. We do NOT call ``.lower()``
    on the input — that would inhibit the CITEXT index and is unnecessary.
    """
    result = await db.execute(select(User).where(User.primary_email == email))
    return result.scalar_one_or_none()


async def get_identity_by_provider_subject(
    db: AsyncSession, *, provider: str, provider_subject: str
) -> AuthIdentity | None:
    result = await db.execute(
        select(AuthIdentity).where(
            AuthIdentity.provider == provider,
            AuthIdentity.provider_subject == provider_subject,
        )
    )
    return result.scalar_one_or_none()


async def get_profile(db: AsyncSession, user_id: UUID) -> UserProfile | None:
    return await db.get(UserProfile, user_id)


async def list_users(
    db: AsyncSession,
    *,
    after: UUID | None = None,
    limit: int = 50,
) -> list[User]:
    """Cursor-paginated user listing ordered by ``users.id``.

    ``after`` is exclusive: rows with ``id > after`` are returned. ``None``
    starts at the first row. Caller must clamp ``limit`` (service layer
    handles policy).
    """
    stmt = select(User).order_by(User.id).limit(limit)
    if after is not None:
        stmt = stmt.where(User.id > after)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def search_users(
    db: AsyncSession,
    *,
    status: str | None = None,
    search: str | None = None,
    restrict_ids: Sequence[UUID] | None = None,
    sort: str | None = None,
    sort_dir: str = "asc",
    page: int = 0,
    page_size: int = 25,
) -> Page[User]:
    """Offset page of users with server-side search (email / display name) +
    whitelisted sort. Backs the page-numbered admin table.

    ``display_name`` lives on ``user_profiles``, so we ``outerjoin`` it for the
    search predicate; the SELECT still yields ``User`` rows only (the profile
    is loaded separately by the service, matching :func:`list_users`).

    ``restrict_ids`` — when provided — limits the result to that id set (the
    admin role filter resolves matching ids via access_control and passes them
    here so this query never joins access-control tables directly).
    """
    stmt = select(User).outerjoin(UserProfile, UserProfile.user_id == User.id)
    if status is not None:
        stmt = stmt.where(User.status == status)
    if restrict_ids is not None:
        stmt = stmt.where(User.id.in_(list(restrict_ids)))
    return await paginate(
        db,
        stmt,
        page=page,
        page_size=page_size,
        search=search,
        search_columns=[User.primary_email, UserProfile.display_name],
        sort=sort,
        sort_dir=sort_dir,
        sortable={
            "email": User.primary_email,
            "status": User.status,
            "created_at": User.created_at,
        },
        default_order=[User.id],
    )


async def list_profiles(db: AsyncSession, user_ids: Sequence[UUID]) -> list[UserProfile]:
    if not user_ids:
        return []
    result = await db.execute(select(UserProfile).where(UserProfile.user_id.in_(user_ids)))
    return list(result.scalars().all())
