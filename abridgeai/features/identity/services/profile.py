from __future__ import annotations

from typing import TYPE_CHECKING

from abridgeai.features.identity.models import User, UserProfile
from abridgeai.features.identity.queries import users as user_queries
from abridgeai.features.identity.schemas import (
    UserProfileRead,
    UserProfileUpdate,
    UserRead,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def serialize_user(user: User, profile: UserProfile | None = None) -> UserRead:
    return UserRead.model_validate(
        {
            "id": user.id,
            "primary_email": user.primary_email,
            "status": user.status,
            "last_login_at": user.last_login_at,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "profile": UserProfileRead.model_validate(profile) if profile else None,
        }
    )


async def get_current_user_read(db: AsyncSession, user: User) -> UserRead:
    profile = await user_queries.get_profile(db, user.id)
    return serialize_user(user, profile)


async def update_profile(
    db: AsyncSession, user: User, payload: UserProfileUpdate
) -> UserProfileRead:
    profile = await user_queries.get_profile(db, user.id)
    if profile is None:
        profile = UserProfile(
            user_id=user.id,
            display_name=payload.display_name or user.primary_email,
        )
        db.add(profile)

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(profile, key, value)

    await db.commit()
    await db.refresh(profile)
    return UserProfileRead.model_validate(profile)


__all__ = ["get_current_user_read", "serialize_user", "update_profile"]
