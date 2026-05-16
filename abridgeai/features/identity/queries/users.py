"""User-related identity queries (data accessors only).

Each function takes ``db: AsyncSession`` as the first argument and returns the
ORM model (or ``None``). No business logic, no Pydantic conversion. Service
layer (T1.7) composes these and shapes the response.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
