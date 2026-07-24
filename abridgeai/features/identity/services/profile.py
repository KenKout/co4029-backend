from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from abridgeai.core.config import get_settings
from abridgeai.features.identity.models import StorageObject, User, UserProfile
from abridgeai.features.identity.queries import users as user_queries
from abridgeai.features.identity.schemas import (
    UserProfileRead,
    UserProfileUpdate,
    UserRead,
)
from abridgeai.infrastructure.s3 import create_stream_url, put_object_bytes

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Allowed avatar image types + max size (2 MiB). Small, server-validated blobs.
_AVATAR_MIME_TYPES: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}
_AVATAR_MAX_BYTES = 2 * 1024 * 1024


class AvatarUploadError(ValueError):
    """Raised when an uploaded avatar fails validation (type / size)."""


async def _mint_avatar_url(db: AsyncSession, profile: UserProfile | None) -> str | None:
    """Presigned GET URL for the profile's avatar object (or ``None``)."""

    if profile is None or profile.avatar_object_id is None:
        return None
    storage = await db.get(StorageObject, profile.avatar_object_id)
    if storage is None:
        return None
    try:
        url, _ = await create_stream_url(storage)
    except Exception:  # noqa: BLE001 — never let a storage blip break profile reads
        return None
    return url


async def serialize_user_async(
    db: AsyncSession, user: User, profile: UserProfile | None = None
) -> UserRead:
    """Serialize with a freshly-minted presigned ``avatar_url`` on the profile."""

    avatar_url = await _mint_avatar_url(db, profile)
    profile_payload = None
    if profile is not None:
        profile_payload = UserProfileRead.model_validate(profile).model_copy(
            update={"avatar_url": avatar_url}
        )
    return UserRead.model_validate(
        {
            "id": user.id,
            "primary_email": user.primary_email,
            "status": user.status,
            "last_login_at": user.last_login_at,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "profile": profile_payload,
        }
    )


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
    return await serialize_user_async(db, user, profile)


async def upload_avatar(
    db: AsyncSession, user: User, *, data: bytes, content_type: str
) -> UserRead:
    """Validate + store an avatar image and point the profile at it.

    Uploads the bytes server-side to object storage, records a
    ``storage_objects`` row, and sets ``user_profiles.avatar_object_id``.
    Raises :class:`AvatarUploadError` on an unsupported type or oversized file.
    """

    ext = _AVATAR_MIME_TYPES.get(content_type)
    if ext is None:
        raise AvatarUploadError(
            "unsupported_avatar_type: allowed types are JPEG, PNG, WebP, GIF."
        )
    if len(data) == 0:
        raise AvatarUploadError("empty_avatar: the uploaded file is empty.")
    if len(data) > _AVATAR_MAX_BYTES:
        raise AvatarUploadError("avatar_too_large: images must be 2 MiB or smaller.")

    settings = get_settings()
    bucket = settings.s3_bucket_name or "abridgeai-local"
    object_id = uuid.uuid4()
    object_key = f"avatars/{user.id}/{object_id}.{ext}"

    # Upload bytes first — if storage fails we never touch the DB.
    await put_object_bytes(
        _StorageRef(bucket=bucket, object_key=object_key),
        data,
        content_type=content_type,
    )

    storage = StorageObject(
        id=object_id,
        bucket=bucket,
        object_key=object_key,
        original_filename=f"avatar.{ext}",
        mime_type=content_type,
        size_bytes=len(data),
        uploaded_by=user.id,
        uploaded_at=datetime.now(tz=UTC),
    )
    db.add(storage)

    profile = await user_queries.get_profile(db, user.id)
    if profile is None:
        profile = UserProfile(user_id=user.id, display_name=user.primary_email)
        db.add(profile)
    profile.avatar_object_id = object_id

    await db.commit()
    await db.refresh(profile)
    return await serialize_user_async(db, user, profile)


class _StorageRef:
    """Minimal bucket/key holder matching the s3 ``StorageObject`` protocol."""

    __slots__ = ("bucket", "object_key")

    def __init__(self, *, bucket: str, object_key: str) -> None:
        self.bucket = bucket
        self.object_key = object_key


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
