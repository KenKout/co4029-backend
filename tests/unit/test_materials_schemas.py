from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from abridgeai.features.materials.schemas import (
    MaterialAuthoring,
    MaterialPublic,
    MaterialStreamUrl,
    MaterialUpdate,
    MaterialUploadComplete,
    MaterialUploadInit,
    MaterialVersionAuthoring,
    ProcessingProgress,
)

_PUBLIC_FORBIDDEN = frozenset(
    {
        "processing_status",
        "processing_error",
        "internal_notes",
        "deleted_at",
        "deleted_by",
        "created_by",
        "updated_by",
        "uploaded_by",
        "current_version_id",
        "extracted_metadata",
        "ai_processing_enabled",
        "visible_to_students",
        "version_count",
        "latest_version",
    }
)


def test_authoring_inherits_public() -> None:
    assert issubclass(MaterialAuthoring, MaterialPublic)


def test_public_excludes_internal_fields() -> None:
    leaked = _PUBLIC_FORBIDDEN & set(MaterialPublic.model_fields.keys())
    assert not leaked, f"MaterialPublic leaks internal fields: {leaked}"


def test_authoring_includes_authoring_fields() -> None:
    keys = set(MaterialAuthoring.model_fields.keys())
    assert {
        "visible_to_students",
        "ai_processing_enabled",
        "current_version_id",
        "version_count",
        "latest_version",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    } <= keys


def test_orm_compat_material_public() -> None:
    obj = SimpleNamespace(
        id=uuid4(),
        lesson_id=uuid4(),
        title="Intro to Linear Algebra",
        material_type="pdf",
    )
    dto = MaterialPublic.model_validate(obj)
    assert dto.title == "Intro to Linear Algebra"
    assert dto.material_type == "pdf"


def test_status_literal_narrowing_processing_status() -> None:
    with pytest.raises(ValidationError):
        MaterialVersionAuthoring(
            id=uuid4(),
            material_id=uuid4(),
            version_no=1,
            storage_object_id=uuid4(),
            is_current=True,
            processing_status="invalid",  # type: ignore[arg-type]
            uploaded_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )


def test_material_type_literal_narrowing() -> None:
    with pytest.raises(ValidationError):
        MaterialPublic(
            id=uuid4(),
            lesson_id=uuid4(),
            title="bad type",
            material_type="html",  # type: ignore[arg-type]
        )


def test_upload_init_optional_material_type() -> None:
    payload = MaterialUploadInit(
        filename="x.pdf",
        content_type="application/pdf",
        size_bytes=100,
        title="x",
        material_type=None,
    )
    assert payload.material_type is None


def test_upload_init_strict_extras_rejected() -> None:
    with pytest.raises(ValidationError):
        MaterialUploadInit.model_validate(
            {
                "filename": "x.pdf",
                "content_type": "application/pdf",
                "size_bytes": 100,
                "title": "x",
                "unexpected_field": "boom",
            }
        )


def test_upload_complete_validates_checksum_format() -> None:
    good = "a" * 64
    payload = MaterialUploadComplete(storage_object_id=uuid4(), checksum_sha256=good)
    assert payload.checksum_sha256 == good

    with pytest.raises(ValidationError):
        MaterialUploadComplete(storage_object_id=uuid4(), checksum_sha256="not-hex")
    with pytest.raises(ValidationError):
        MaterialUploadComplete(storage_object_id=uuid4(), checksum_sha256="a" * 63)
    with pytest.raises(ValidationError):
        MaterialUploadComplete(storage_object_id=uuid4(), checksum_sha256="g" * 64)


def test_processing_progress_percent_bounds() -> None:
    ok = ProcessingProgress(
        material_id=uuid4(),
        version_id=uuid4(),
        processing_status="extracting",
        progress_percent=50,
    )
    assert ok.progress_percent == 50

    with pytest.raises(ValidationError):
        ProcessingProgress(
            material_id=uuid4(),
            version_id=uuid4(),
            processing_status="extracting",
            progress_percent=101,
        )
    with pytest.raises(ValidationError):
        ProcessingProgress(
            material_id=uuid4(),
            version_id=uuid4(),
            processing_status="extracting",
            progress_percent=-1,
        )


def test_material_stream_url_orm_compat() -> None:
    expires = datetime.now(UTC)
    obj = SimpleNamespace(url="https://s3/...", expires_at=expires)
    dto = MaterialStreamUrl.model_validate(obj)
    assert dto.url == "https://s3/..."
    assert dto.expires_at == expires


def test_material_update_all_optional() -> None:
    empty = MaterialUpdate()
    assert empty.title is None
    assert empty.visible_to_students is None
    assert empty.ai_processing_enabled is None

    with pytest.raises(ValidationError):
        MaterialUpdate.model_validate({"unknown_field": True})
