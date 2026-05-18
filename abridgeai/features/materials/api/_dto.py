"""Pydantic DTOs returned by :mod:`abridgeai.features.materials.api.public`.

Cross-feature consumers receive these instead of ORM models so they
cannot accidentally mutate persistent state, lazy-load attributes in
async context, or depend on relationship configuration that is
intentionally minimal across the feature boundary.

All DTOs are immutable (`model_config['frozen'] = True`) and use
``from_attributes=True`` so :func:`pydantic.BaseModel.model_validate`
works directly against ORM instances.

The DTOs purposely do NOT mirror the full ORM column set — only the
fields the inventoried consumers in Wave 5 actually read are exposed.
Adding fields is a forward-compatible change; removing fields is not.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class StorageBlobInfoDTO(BaseModel):
    """Minimal view onto a learning material's S3 blob.

    Wraps ``storage_objects.{bucket, object_key, size_bytes}`` for cross-
    feature consumers without exposing the identity-owned ``StorageObject``
    ORM. Issued by :func:`api.public.get_storage_object_size_and_key` and
    :func:`api.public.get_storage_blob_for_version`.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    bucket: str
    object_key: str
    size_bytes: int


class MaterialContextDTO(BaseModel):
    """Material + its lesson / module / course context.

    Returned by :func:`api.public.get_material_with_lesson_context` and
    :func:`api.public.resolve_chunks_to_materials`. The "course slug"
    field powers deep-link generation in the SR remediation notification
    card (Wave 5 T30).
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    material_id: UUID
    material_title: str
    material_type: str
    current_version_id: UUID | None
    lesson_id: UUID
    lesson_title: str
    module_id: UUID
    course_id: UUID
    course_slug: str


class DocumentChunkDTO(BaseModel):
    """One ``document_chunks`` row, denormalized FKs included.

    Returned by :func:`api.public.get_document_chunks_by_material`. The
    ``metadata`` field is the JSONB blob; consumers treat it as opaque.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    chunk_id: UUID
    material_version_id: UUID
    course_id: UUID
    module_id: UUID
    lesson_id: UUID
    chunk_index: int
    chunk_type: str
    content: str
    content_hash: str
    metadata: dict[str, Any]


class ProcessingJobDTO(BaseModel):
    """Status snapshot of a ``processing_jobs`` row.

    Returned by :func:`api.public.get_processing_job_status`. The job
    table is hard-deleted (no ``deleted_at``) so soft-delete filtering
    does not apply.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    job_id: UUID
    entity_type: str
    entity_id: UUID
    job_type: str
    status: str
    progress_percent: int
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    retry_count: int


__all__ = [
    "DocumentChunkDTO",
    "MaterialContextDTO",
    "ProcessingJobDTO",
    "StorageBlobInfoDTO",
]
