"""Learner-side material catalog reads + presigned stream URL minting (T4.6).

Composes :mod:`features.materials.queries.published` and
:mod:`features.materials.queries.chunks`, gates each call behind the
same visibility predicate, and serializes results into the public
schemas owned by :mod:`features.materials.schemas.public`.

Visibility-as-security-boundary (plan §5053 / §5075): the
``visible_to_students=TRUE AND processing_status='ready'`` predicate
is the only access check; routers return 404 (not 403) on misses so
existence is never leaked.

Stream-URL TTL: capped at one hour for learner downloads (plan §5051)
even when the global ``s3_url_ttl_seconds`` is configured higher —
authoring downloads can stay long-lived but a learner link with a
multi-hour TTL is too easy to exfiltrate.

Service / SQLAlchemy boundary: this module imports ``AsyncSession`` only
under :data:`TYPE_CHECKING`; runtime imports stay in ``queries/``. Mirrors
:mod:`features.identity.services.login` and
:mod:`features.courses.services.catalog` and keeps the import-linter
"Services do not touch SQLAlchemy directly" contract green.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from abridgeai.core.config import get_settings
from abridgeai.features.materials.queries.chunks import (
    get_stream_target_for_material,
    list_chunks_preview,
)
from abridgeai.features.materials.queries.published import get_visible_material
from abridgeai.features.materials.schemas.public import (
    MaterialPublic,
    MaterialStreamUrl,
)
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Plan §5051: learner stream URLs cap at one hour even if
# ``settings.s3_url_ttl_seconds`` is configured higher (authoring downloads
# may run longer; a learner link is too easy to exfiltrate / forward).
_LEARNER_STREAM_TTL_CAP_SECONDS = 3600


class ChunkPreview(BaseModel):
    """Narrow ``DocumentChunk`` projection for learner-side preview.

    Reserved for the quiz-UI source-attribution surface (plan §5052).
    Excludes the embedding vector (1536 floats per row would dwarf the
    payload) and the LLM-enrichment metadata (authoring-only).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    chunk_index: int
    chunk_type: str
    content: str


def _escape_filename(filename: str) -> str:
    """Quote-escape a filename for ``Content-Disposition``.

    A bare filename containing ``"`` would terminate the quoted-string
    early and let the remainder bleed into other header parameters
    (header injection). Backslash-escape ``"`` and ``\\`` per RFC 6266 §4.1
    quoted-string semantics, and strip CR / LF entirely (no defensible
    use in a filename, easy CRLF-injection vector if interpolated).
    """
    return filename.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


async def get_visible_material_for_user(
    db: AsyncSession, material_id: UUID, user_id: UUID
) -> MaterialPublic | None:
    """Return the public DTO for ``material_id`` if visible to ``user_id``.

    ``user_id`` is currently unused — visibility is enforced solely by the
    query layer's ``visible_to_students=TRUE AND processing_status='ready'``
    predicate. The arg is reserved for the Phase 7 enrollments check
    ("only students enrolled in the owning course see the material");
    keeping the slot present means routers don't change when that gate
    lands.
    """
    del user_id
    material = await get_visible_material(db, material_id)
    return None if material is None else MaterialPublic.model_validate(material)


async def get_stream_url_for_material(
    db: AsyncSession, material_id: UUID, user_id: UUID
) -> MaterialStreamUrl | None:
    """Mint a presigned GET URL for a learner streaming a visible material.

    Returns ``None`` (router maps to 404) when the material is invisible,
    soft-deleted, draft, mid-pipeline, or its current version has no
    resolvable storage object.

    The TTL is :data:`_LEARNER_STREAM_TTL_CAP_SECONDS` or the global
    ``settings.s3_url_ttl_seconds``, whichever is smaller. The
    ``Content-Disposition`` header forces the browser into download mode
    rather than inline streaming for non-media types and tags the file
    with the material's title (filename-quoted to block header injection).
    """
    del user_id
    target = await get_stream_target_for_material(db, material_id)
    if target is None:
        return None

    settings = get_settings()
    ttl = min(settings.s3_url_ttl_seconds, _LEARNER_STREAM_TTL_CAP_SECONDS)
    safe_title = _escape_filename(target.title)
    url, _ = await create_stream_url(
        target,
        response_headers={"Content-Disposition": f'attachment; filename="{safe_title}"'},
    )
    expires_at = datetime.now(tz=UTC) + timedelta(seconds=ttl)
    return MaterialStreamUrl(
        url=url,
        expires_at=expires_at,
        material_version_id=target.material_version_id,
    )


async def list_visible_chunks_preview(
    db: AsyncSession, material_id: UUID, user_id: UUID, limit: int
) -> list[ChunkPreview] | None:
    """Return the first ``limit`` chunks for a visible material (or ``None``).

    ``None`` signals the material is not visible; the router maps it to
    404 to preserve the no-existence-leak contract. An empty list
    surfaces normally for visible-but-not-yet-chunked materials.
    """
    del user_id
    material = await get_visible_material(db, material_id)
    if material is None:
        return None
    chunks = await list_chunks_preview(db, material_id, limit=limit)
    return [ChunkPreview.model_validate(chunk) for chunk in chunks]


__all__ = [
    "ChunkPreview",
    "get_stream_url_for_material",
    "get_visible_material_for_user",
    "list_visible_chunks_preview",
]
