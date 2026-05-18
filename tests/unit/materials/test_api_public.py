from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.materials.api import public
from abridgeai.features.materials.api._dto import (
    DocumentChunkDTO,
    MaterialContextDTO,
    ProcessingJobDTO,
    StorageBlobInfoDTO,
)


_EXPECTED_FUNCTIONS = (
    "get_storage_blob_for_version",
    "get_storage_size_and_key",
    "get_material_with_lesson_context",
    "resolve_chunks_to_materials",
    "get_document_chunks_by_material",
    "get_processing_job_status",
)


_EXPECTED_DTOS = (
    StorageBlobInfoDTO,
    MaterialContextDTO,
    DocumentChunkDTO,
    ProcessingJobDTO,
)


def test_all_expected_functions_in_dunder_all() -> None:
    exported = set(public.__all__)
    for name in _EXPECTED_FUNCTIONS:
        assert name in exported, f"{name} missing from __all__"


@pytest.mark.parametrize("name", _EXPECTED_FUNCTIONS)
def test_function_is_async_coroutine(name: str) -> None:
    fn = getattr(public, name)
    assert inspect.iscoroutinefunction(fn), f"{name} must be `async def`"


@pytest.mark.parametrize("name", _EXPECTED_FUNCTIONS)
def test_first_positional_param_is_db(name: str) -> None:
    fn = getattr(public, name)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    assert params, f"{name} has no parameters"
    assert params[0].name == "db", (
        f"{name} first positional must be `db: AsyncSession`, got {params[0].name!r}"
    )


def test_get_storage_size_and_key_signature() -> None:
    sig = inspect.signature(public.get_storage_size_and_key)
    params = sig.parameters
    assert "material_id" in params
    assert params["material_id"].kind is inspect.Parameter.KEYWORD_ONLY
    hints = get_type_hints(public.get_storage_size_and_key)
    assert hints["material_id"] is UUID
    assert hints["return"] == (tuple[int, str] | None)


def test_get_storage_blob_for_version_signature() -> None:
    sig = inspect.signature(public.get_storage_blob_for_version)
    assert list(sig.parameters) == ["db", "version_id"]
    hints = get_type_hints(public.get_storage_blob_for_version)
    assert hints["version_id"] is UUID
    assert hints["return"] == (StorageBlobInfoDTO | None)


def test_get_material_with_lesson_context_signature() -> None:
    sig = inspect.signature(public.get_material_with_lesson_context)
    assert list(sig.parameters) == ["db", "material_id"]
    hints = get_type_hints(public.get_material_with_lesson_context)
    assert hints["material_id"] is UUID
    assert hints["return"] == (MaterialContextDTO | None)


def test_resolve_chunks_to_materials_signature() -> None:
    sig = inspect.signature(public.resolve_chunks_to_materials)
    assert list(sig.parameters) == ["db", "chunk_ids"]
    hints = get_type_hints(public.resolve_chunks_to_materials)
    assert hints["chunk_ids"] == Iterable[UUID]
    assert hints["return"] == dict[UUID, MaterialContextDTO]


def test_get_document_chunks_by_material_signature() -> None:
    hints = get_type_hints(public.get_document_chunks_by_material)
    assert hints["material_id"] is UUID
    assert hints["return"] == list[DocumentChunkDTO]


def test_get_processing_job_status_signature() -> None:
    hints = get_type_hints(public.get_processing_job_status)
    assert hints["job_id"] is UUID
    assert hints["return"] == (ProcessingJobDTO | None)


@pytest.mark.parametrize("dto", _EXPECTED_DTOS)
def test_dtos_are_frozen_and_from_attributes(dto: type) -> None:
    cfg = dto.model_config
    assert cfg.get("frozen") is True, f"{dto.__name__} must be frozen"
    assert cfg.get("from_attributes") is True, f"{dto.__name__} must allow ORM attribute hydration"


def test_storage_blob_dto_field_set() -> None:
    assert set(StorageBlobInfoDTO.model_fields) == {"bucket", "object_key", "size_bytes"}


def test_material_context_dto_field_set() -> None:
    assert set(MaterialContextDTO.model_fields) == {
        "material_id",
        "material_title",
        "material_type",
        "current_version_id",
        "lesson_id",
        "lesson_title",
        "module_id",
        "course_id",
        "course_slug",
    }


def test_document_chunk_dto_field_set() -> None:
    assert set(DocumentChunkDTO.model_fields) == {
        "chunk_id",
        "material_version_id",
        "course_id",
        "module_id",
        "lesson_id",
        "chunk_index",
        "chunk_type",
        "content",
        "content_hash",
        "metadata",
    }


def test_processing_job_dto_field_set() -> None:
    assert set(ProcessingJobDTO.model_fields) == {
        "job_id",
        "entity_type",
        "entity_id",
        "job_type",
        "status",
        "progress_percent",
        "started_at",
        "finished_at",
        "error_message",
        "retry_count",
    }


def test_storage_blob_dto_round_trip() -> None:
    src = SimpleAttrs(bucket="b1", object_key="k1", size_bytes=42)
    dto = StorageBlobInfoDTO.model_validate(src)
    assert dto.bucket == "b1"
    assert dto.object_key == "k1"
    assert dto.size_bytes == 42


def test_material_context_dto_immutable() -> None:
    dto = MaterialContextDTO(
        material_id=uuid4(),
        material_title="T",
        material_type="pdf",
        current_version_id=None,
        lesson_id=uuid4(),
        lesson_title="L",
        module_id=uuid4(),
        course_id=uuid4(),
        course_slug="c",
    )
    with pytest.raises((TypeError, ValueError)):
        dto.material_title = "other"


class SimpleAttrs:
    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_db_with_first_row(row: object | None) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.first = MagicMock(return_value=row)
    db.execute = AsyncMock(return_value=result)
    return db


def _make_db_with_mappings(rows: list[dict]) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    mappings_proxy = MagicMock()
    mappings_proxy.all = MagicMock(return_value=rows)
    result.mappings = MagicMock(return_value=mappings_proxy)
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_get_material_with_lesson_context_found() -> None:
    material_id = uuid4()
    lesson_id = uuid4()
    module_id = uuid4()
    course_id = uuid4()
    version_id = uuid4()
    row = SimpleAttrs(
        material_id=material_id,
        material_title="Intro PDF",
        material_type="pdf",
        current_version_id=version_id,
        lesson_id=lesson_id,
        lesson_title="Lesson 1",
        module_id=module_id,
        course_id=course_id,
        course_slug="cs101",
    )
    db = _make_db_with_first_row(row)
    dto = await public.get_material_with_lesson_context(db, material_id)
    assert dto is not None
    assert dto.material_id == material_id
    assert dto.current_version_id == version_id
    assert dto.lesson_title == "Lesson 1"
    assert dto.course_slug == "cs101"


@pytest.mark.asyncio
async def test_get_material_with_lesson_context_not_found() -> None:
    db = _make_db_with_first_row(None)
    dto = await public.get_material_with_lesson_context(db, uuid4())
    assert dto is None


@pytest.mark.asyncio
async def test_get_material_with_lesson_context_soft_deleted_returns_none() -> None:
    db = _make_db_with_first_row(None)
    dto = await public.get_material_with_lesson_context(db, uuid4())
    assert dto is None


@pytest.mark.asyncio
async def test_get_material_with_lesson_context_query_filters_soft_delete() -> None:
    db = _make_db_with_first_row(None)
    await public.get_material_with_lesson_context(db, uuid4())
    rendered = str(db.execute.call_args.args[0])
    assert "lm.deleted_at IS NULL" in rendered
    assert "l.deleted_at IS NULL" in rendered
    assert "m.deleted_at IS NULL" in rendered
    assert "c.deleted_at IS NULL" in rendered


@pytest.mark.asyncio
async def test_resolve_chunks_to_materials_empty_input_skips_db() -> None:
    db = AsyncMock()
    db.execute = AsyncMock()
    out = await public.resolve_chunks_to_materials(db, [])
    assert out == {}
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_chunks_to_materials_dedups_by_material_id() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    material_id = uuid4()
    lesson_id = uuid4()
    module_id = uuid4()
    course_id = uuid4()
    rows = [
        {
            "material_id": material_id,
            "material_title": "M",
            "material_type": "pdf",
            "current_version_id": uuid4(),
            "lesson_id": lesson_id,
            "lesson_title": "L",
            "module_id": module_id,
            "course_id": course_id,
            "course_slug": "cs101",
            "chunk_id": chunk_a,
        },
    ]
    db = _make_db_with_mappings(rows)
    result = await public.resolve_chunks_to_materials(db, [chunk_a, chunk_b])
    assert set(result.keys()) == {material_id}
    assert result[material_id].course_slug == "cs101"
    bound = db.execute.call_args.args[1]
    assert bound["chunk_ids"] == [str(chunk_a), str(chunk_b)]


@pytest.mark.asyncio
async def test_resolve_chunks_to_materials_returns_typed_dto() -> None:
    material_id = uuid4()
    rows = [
        {
            "material_id": material_id,
            "material_title": "M",
            "material_type": "pdf",
            "current_version_id": None,
            "lesson_id": uuid4(),
            "lesson_title": "L",
            "module_id": uuid4(),
            "course_id": uuid4(),
            "course_slug": "cs101",
            "chunk_id": uuid4(),
        },
    ]
    db = _make_db_with_mappings(rows)
    out = await public.resolve_chunks_to_materials(db, [uuid4()])
    assert isinstance(out[material_id], MaterialContextDTO)
    assert out[material_id].current_version_id is None


@pytest.mark.asyncio
async def test_resolve_chunks_query_filters_soft_delete_chain() -> None:
    db = _make_db_with_mappings([])
    await public.resolve_chunks_to_materials(db, [uuid4()])
    rendered = str(db.execute.call_args.args[0])
    for guard in (
        "lm.deleted_at IS NULL",
        "lmv.deleted_at IS NULL",
        "l.deleted_at IS NULL",
        "m.deleted_at IS NULL",
        "c.deleted_at IS NULL",
    ):
        assert guard in rendered, f"missing soft-delete guard: {guard}"


@pytest.mark.asyncio
async def test_get_storage_size_and_key_found() -> None:
    row = SimpleAttrs(bucket="bk", object_key="materials/x.pdf", size_bytes=2048)
    db = _make_db_with_first_row(row)
    out = await public.get_storage_size_and_key(db, material_id=uuid4())
    assert out == (2048, "materials/x.pdf")


@pytest.mark.asyncio
async def test_get_storage_size_and_key_not_found() -> None:
    db = _make_db_with_first_row(None)
    out = await public.get_storage_size_and_key(db, material_id=uuid4())
    assert out is None


@pytest.mark.asyncio
async def test_get_storage_blob_for_version_found() -> None:
    row = SimpleAttrs(bucket="bk", object_key="materials/v.pdf", size_bytes=10)
    db = _make_db_with_first_row(row)
    out = await public.get_storage_blob_for_version(db, uuid4())
    assert out is not None
    assert out.bucket == "bk"
    assert out.size_bytes == 10


@pytest.mark.asyncio
async def test_get_storage_blob_for_version_not_found() -> None:
    db = _make_db_with_first_row(None)
    out = await public.get_storage_blob_for_version(db, uuid4())
    assert out is None


@pytest.mark.asyncio
async def test_get_processing_job_status_found() -> None:
    job_id = uuid4()
    entity_id = uuid4()
    job = SimpleAttrs(
        id=job_id,
        entity_type="material_version",
        entity_id=entity_id,
        job_type="full_pipeline",
        status="running",
        progress_percent=42,
        started_at=None,
        finished_at=None,
        error_message=None,
        retry_count=0,
    )
    db = AsyncMock()
    db.get = AsyncMock(return_value=job)
    out = await public.get_processing_job_status(db, job_id)
    assert out is not None
    assert out.job_id == job_id
    assert out.entity_id == entity_id
    assert out.status == "running"
    assert out.progress_percent == 42


@pytest.mark.asyncio
async def test_get_processing_job_status_not_found() -> None:
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)
    out = await public.get_processing_job_status(db, uuid4())
    assert out is None


@pytest.mark.asyncio
async def test_get_document_chunks_by_material_maps_each_row() -> None:
    chunk_id = uuid4()
    version_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    lesson_id = uuid4()
    chunk_orm = SimpleAttrs(
        id=chunk_id,
        material_version_id=version_id,
        course_id=course_id,
        module_id=module_id,
        lesson_id=lesson_id,
        chunk_index=0,
        chunk_type="pdf",
        content="hello",
        content_hash="abc",
        metadata_json={"k": "v"},
    )
    scalars_proxy = MagicMock()
    scalars_proxy.all = MagicMock(return_value=[chunk_orm])
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars_proxy)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    out = await public.get_document_chunks_by_material(db, uuid4())
    assert len(out) == 1
    dto = out[0]
    assert dto.chunk_id == chunk_id
    assert dto.material_version_id == version_id
    assert dto.metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_get_document_chunks_by_material_empty() -> None:
    scalars_proxy = MagicMock()
    scalars_proxy.all = MagicMock(return_value=[])
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars_proxy)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    out = await public.get_document_chunks_by_material(db, uuid4())
    assert out == []


def test_module_docstring_documents_public_contract() -> None:
    doc = (public.__doc__ or "").lower()
    assert "cross-feature" in doc
    assert "soft-delete" in doc
