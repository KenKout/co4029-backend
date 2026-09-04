from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.identity.models import (
    StorageObject,
    User,
    UserProfile,
    UserProfileLink,
)
from abridgeai.features.identity.queries import users as user_queries
from abridgeai.features.identity.schemas import (
    UserProfileLinkIn,
    UserProfileLinkRead,
    UserProfileLinkUpdate,
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

# Upper bound on live links per profile. A profile page shows a handful of
# external destinations; the cap stops a single account turning its profile
# into an unbounded, self-service link farm.
MAX_PROFILE_LINKS = 10


class AvatarUploadError(ValueError):
    """Raised when an uploaded avatar fails validation (type / size)."""


class ProfileLinkError(ValueError):
    """Raised when a profile-link write breaks a service-level rule."""


class ProfileLinkNotFoundError(LookupError):
    """Raised when a link id does not exist on the caller's own profile."""


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
        # Links are a separate table, so ``model_validate(profile)`` cannot
        # reach them — load them here rather than relying on lazy loading,
        # which would emit IO during serialization on an async session.
        links = await user_queries.list_profile_links(db, profile.user_id)
        profile_payload = UserProfileRead.model_validate(profile).model_copy(
            update={
                "avatar_url": avatar_url,
                "links": [UserProfileLinkRead.model_validate(link) for link in links],
            }
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


async def list_links(db: AsyncSession, user: User) -> list[UserProfileLinkRead]:
    """Every external link on the caller's own profile (FR-2.8)."""
    links = await user_queries.list_profile_links(db, user.id)
    return [UserProfileLinkRead.model_validate(link) for link in links]


async def create_link(
    db: AsyncSession, user: User, payload: UserProfileLinkIn
) -> UserProfileLinkRead:
    """Add one external link to the caller's profile.

    Raises :class:`ProfileLinkError` when the profile is already at
    :data:`MAX_PROFILE_LINKS`. Scheme and length of ``url`` are already
    validated by :class:`UserProfileLinkIn`, and ``link_type`` is a Literal
    matching the table's CHECK constraint, so nothing else needs re-checking
    here.
    """
    existing = await user_queries.list_profile_links(db, user.id)
    if len(existing) >= MAX_PROFILE_LINKS:
        raise ProfileLinkError(
            f"too_many_links: a profile may hold at most {MAX_PROFILE_LINKS} links."
        )

    link = UserProfileLink(
        user_id=user.id,
        link_type=payload.link_type,
        url=payload.url,
        label=payload.label,
    )
    db.add(link)
    await db.commit()
    await db.refresh(link)
    return UserProfileLinkRead.model_validate(link)


async def update_link(
    db: AsyncSession,
    user: User,
    *,
    link_id: uuid.UUID,
    payload: UserProfileLinkUpdate,
) -> UserProfileLinkRead:
    """Partially update one of the caller's own links.

    Raises :class:`ProfileLinkNotFoundError` when the id is not a live link on
    THIS user's profile — the ownership check is inside the query, so another
    user's link is indistinguishable from one that never existed.
    """
    link = await user_queries.get_profile_link(db, link_id=link_id, user_id=user.id)
    if link is None:
        raise ProfileLinkNotFoundError(str(link_id))

    # ``exclude_unset`` (not ``exclude_none``): an explicit ``"label": null``
    # is how a caller clears a label, and only the unset check can tell that
    # apart from omitting the field.
    for key, value in payload.model_dump(exclude_unset=True).items():
        if key == "label" or value is not None:
            setattr(link, key, value)

    await db.commit()
    await db.refresh(link)
    return UserProfileLinkRead.model_validate(link)


async def delete_link(db: AsyncSession, user: User, *, link_id: uuid.UUID) -> None:
    """Remove one of the caller's own links.

    Soft-delete, not a physical DELETE: ``user_profile_links`` carries
    ``SoftDeleteMixin``, and the hard-delete guard rejects ``session.delete()``
    on such rows outright. The read-side filter hides the tombstone, so the
    link disappears from every profile read.
    """
    link = await user_queries.get_profile_link(db, link_id=link_id, user_id=user.id)
    if link is None:
        raise ProfileLinkNotFoundError(str(link_id))

    await soft_delete_cascade(db, link, user.id)
    await db.commit()


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


__all__ = [
    "MAX_PROFILE_LINKS",
    "ProfileLinkError",
    "ProfileLinkNotFoundError",
    "create_link",
    "delete_link",
    "get_current_user_read",
    "list_links",
    "serialize_user",
    "update_link",
    "update_profile",
]
