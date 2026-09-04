"""Author identity (display name + presigned avatar) for discussion rows.

Why this module exists
----------------------
The router used to build :class:`DiscussionCommentAuthor` from
``identity_api.get_users_by_ids`` alone, which carries ``display_name`` but
NOT the avatar: ``UserDTO`` has no avatar field, so ``avatar_url`` stayed
``None`` on every comment and the client always fell back to initials. The
avatar column lives on ``user_profiles.avatar_object_id`` and has to be turned
into a short-lived presigned GET URL before it can reach a response body — the
raw bucket/key must never be serialized (same rule as the course roster).

Batching
--------
One query for the profiles and one ``StorageObject`` lookup per DISTINCT
avatar object, not per row: a thread where the same person wrote twenty
comments mints one URL, not twenty. Callers pass every id they need resolved
(comment authors plus topic authors) in a single call.

Failure policy
--------------
A storage blip degrades to ``avatar_url=None`` (initials fallback) instead of
failing the discussion read — the same guard
``identity.services.profile._mint_avatar_url`` and
``courses.services.authoring.list_course_roster`` apply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from abridgeai.core.observability import get_logger
from abridgeai.features.discussions.schemas import DiscussionCommentAuthor
from abridgeai.features.identity.api import public as identity_api
from abridgeai.features.identity.models import StorageObject, UserProfile
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)


async def _avatar_urls(
    db: AsyncSession, avatar_object_ids: set[UUID]
) -> dict[UUID, str]:
    """Presigned GET URL per storage object id; unresolvable ids are omitted."""
    if not avatar_object_ids:
        return {}
    storage_rows = (
        await db.execute(
            select(StorageObject).where(StorageObject.id.in_(avatar_object_ids))
        )
    ).scalars()
    urls: dict[UUID, str] = {}
    for storage in storage_rows:
        try:
            url, _ = await create_stream_url(storage)
        except Exception:  # noqa: BLE001 — a storage blip must not break the thread
            _logger.warning(
                "discussions.avatar_presign_failed",
                exc_info=True,
                storage_object_id=str(storage.id),
            )
            continue
        urls[storage.id] = url
    return urls


async def resolve_authors(
    db: AsyncSession, user_ids: list[UUID]
) -> dict[UUID, DiscussionCommentAuthor]:
    """``{user_id: author}`` with display name + presigned avatar, batched.

    Ids with no user row are absent from the result; the caller substitutes a
    bare author carrying just the id (the display name then falls back to the
    client's "unknown author" string).
    """
    distinct = {uid for uid in user_ids if uid is not None}
    if not distinct:
        return {}

    users = await identity_api.get_users_by_ids(db, list(distinct))

    profile_rows = (
        await db.execute(
            select(UserProfile.user_id, UserProfile.avatar_object_id).where(
                UserProfile.user_id.in_(distinct)
            )
        )
    ).all()
    avatar_object_by_user = {
        row.user_id: row.avatar_object_id
        for row in profile_rows
        if row.avatar_object_id is not None
    }
    urls = await _avatar_urls(db, set(avatar_object_by_user.values()))

    resolved: dict[UUID, DiscussionCommentAuthor] = {}
    for user_id in distinct:
        dto = users.get(user_id)
        if dto is None:
            continue
        object_id = avatar_object_by_user.get(user_id)
        resolved[user_id] = DiscussionCommentAuthor(
            id=user_id,
            display_name=dto.display_name,
            avatar_url=urls.get(object_id) if object_id is not None else None,
        )
    return resolved


def author_or_bare(
    resolved: dict[UUID, DiscussionCommentAuthor], user_id: UUID | None
) -> DiscussionCommentAuthor | None:
    """Author for ``user_id``, or a bare id-only author when unresolved.

    ``None`` in / ``None`` out: a topic whose ``created_by`` is NULL (author
    account hard-deleted) renders with no author block rather than a fake one.
    """
    if user_id is None:
        return None
    return resolved.get(user_id) or DiscussionCommentAuthor(id=user_id)


__all__ = ["author_or_bare", "resolve_authors"]
