"""Request-body DTOs for the materials feature (T4.2).

Used by the teacher-facing upload + metadata-edit endpoints (T4.5 /
T4.6) and validated at the API boundary BEFORE the service layer
runs. ``extra="forbid"`` surfaces typos in API payloads early — the
same guard T3.2 (courses) adopted via ``_StrictRequest``.

Reconciliation §C14 — storage object metadata
----------------------------------------------
:class:`MaterialUploadComplete` accepts ``checksum_sha256`` as part of
the post-S3-PUT register-version flow. The checksum lives on the
``storage_objects`` row (per §C14), not on ``learning_material_versions``;
the service (T4.5) propagates it onto the storage row. The 64-hex
format is enforced here so malformed checksums fail fast at the API
boundary.
"""

from __future__ import annotations

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from abridgeai.features.materials.schemas.public import MaterialTypeLiteral

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


class _StrictRequest(BaseModel):
    """Reject unknown keys to surface typos in API payloads early."""

    model_config = ConfigDict(extra="forbid")


class MaterialUploadInit(_StrictRequest):
    """Body for ``POST /teacher/lessons/{id}/materials``.

    Returns the presigned PUT URL + ``storage_object_id`` so the client
    can upload bytes directly to S3. ``material_type`` is optional —
    omitted callers let the backend auto-detect from ``content_type``
    (MIME). Both ``size_bytes`` and ``filename`` are required for the
    backend to choose the bucket layout and validate against per-org
    upload quotas.
    """

    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=100)
    size_bytes: int = Field(ge=0)
    material_type: MaterialTypeLiteral | None = None
    title: str = Field(max_length=255)


class MaterialUploadComplete(_StrictRequest):
    """Body for ``POST /teacher/materials/{id}/versions/complete``.

    Called by the client AFTER the S3 PUT lands. The service (T4.5)
    flips ``is_current`` on the new version, sets
    ``LearningMaterial.current_version_id``, and propagates
    ``checksum_sha256`` onto the ``storage_objects`` row.
    """

    storage_object_id: UUID
    checksum_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("checksum_sha256")
    @classmethod
    def _validate_checksum_format(cls, value: str) -> str:
        if not _SHA256_HEX.match(value):
            raise ValueError("checksum_sha256 must be 64 lowercase or uppercase hex chars")
        return value


class MaterialUpdate(_StrictRequest):
    """Partial update for material metadata (PATCH /teacher/materials/{id})."""

    title: str | None = Field(default=None, max_length=255)
    visible_to_students: bool | None = None
    ai_processing_enabled: bool | None = None


__all__ = [
    "MaterialUpdate",
    "MaterialUploadComplete",
    "MaterialUploadInit",
]
