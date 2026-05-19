"""Materials ingestion orchestrator (T4.4).

Composes Phase 2 primitives into a 5-stage ingest for one
``LearningMaterialVersion``:

  Stage 1  extract   — ``dispatch_extractor(mime).extract(path)``
  Stage 2  chunk     — branch by ``ExtractedContent.source_type``
  Stage 3  embed     — ``EmbeddingClient.embed`` (one audit row per call)
  Stage 4  persist   — write ``DocumentChunk`` rows with denormalized FKs
  Stage 5  kg        — optional, gated by ``settings.knowledge_graph_enabled``

Source-type → chunker dispatch table::

    audio | video    -> TimestampAwareChunker (preserves segment timestamps)
    image            -> TokenAwareChunker(max_tokens=10_000) (single chunk)
    pdf | docx |
    pptx | html |
    text | code |
    mock             -> SemanticChunker when llm_gateway provided
                        else TokenAwareChunker(max_tokens=500)

Commit discipline (Reconciliation §B2 + T1.7 / T3.5 pattern):
  the orchestrator FLUSHES; the caller (worker, CLI) commits. On any
  exception the failure-state writes are flushed, then the exception is
  re-raised — the worker's outer transaction handler decides whether to
  commit the failure row or roll back.

S3 decoupling (plan §4814-4815):
  ``source_path`` provided  → worker downloaded already, pipeline reuses
  ``source_path`` is ``None`` → pipeline calls ``download_to_temp`` itself
                                inside a ``TemporaryDirectory()`` context

``pipeline_run_id`` is threaded through every LLM / embedding call so
``ai_model_calls`` rows for one ingest share a single grouping key.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.chunking import (
    EnrichedChunk,
    SemanticChunker,
    TimestampAwareChunker,
    TokenAwareChunker,
)
from abridgeai.ai.chunking.base import RawChunk
from abridgeai.ai.chunking.cache import ChunkingCache
from abridgeai.ai.extraction import (
    ExtractedContent,
    UnsupportedMimeError,
    dispatch_extractor,
    maybe_local_mock_extractor,
)
from abridgeai.ai.knowledge_graph import (
    build_knowledge_graph_for_material_version,
)
from abridgeai.ai.llm import EmbeddingClient, LLMGateway
from abridgeai.ai.models import ProcessingJob
from abridgeai.core.config import get_settings
from abridgeai.features.materials.models import (
    DocumentChunk,
    LearningMaterial,
    LearningMaterialVersion,
)
from abridgeai.features.materials.queries.processing import get_latest_processing_job
from abridgeai.infrastructure.s3 import download_to_temp

if TYPE_CHECKING:
    from abridgeai.infrastructure.neo4j import KnowledgeGraphClient

logger = logging.getLogger(__name__)


_TIMESTAMP_SOURCES: frozenset[str] = frozenset({"audio", "video"})
_IMAGE_SOURCES: frozenset[str] = frozenset({"image"})
_TEXT_SOURCES: frozenset[str] = frozenset({"pdf", "docx", "pptx", "html", "text", "code", "mock"})
_DEFAULT_TEXT_TOKENS = 500
_IMAGE_CHUNK_TOKENS = 10_000
_ERROR_FIELD_LIMIT = 5000


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True)
class _IngestContext:
    """Resolved Course→Module→Lesson→Material chain plus the storage row.

    Carries every identifier the chunk-persist + KG stages need without
    forcing the orchestrator to chase joins multiple times. Built once at
    pipeline start.
    """

    version: LearningMaterialVersion
    material: LearningMaterial
    lesson_id: UUID
    module_id: UUID
    course_id: UUID
    course_title: str
    module_title: str
    lesson_title: str
    storage_bucket: str | None
    storage_object_key: str | None
    storage_mime_type: str | None
    storage_object_id: UUID | None


@dataclass
class _PersistedChunk:
    """View of a freshly-flushed ``DocumentChunk`` row needed by Stage 5."""

    id: UUID
    chunk_index: int
    content: str
    material_version_id: UUID


async def _load_context(
    db: AsyncSession,
    material_version_id: UUID,
) -> _IngestContext:
    row = (
        await db.execute(
            text(
                """
                SELECT
                    lm.id            AS material_id,
                    lm.title         AS material_title,
                    lm.material_type AS material_type,
                    lm.lesson_id     AS lesson_id,
                    l.title          AS lesson_title,
                    m.id             AS module_id,
                    m.title          AS module_title,
                    m.course_id      AS course_id,
                    c.title          AS course_title,
                    so.id            AS storage_object_id,
                    so.bucket        AS storage_bucket,
                    so.object_key    AS storage_object_key,
                    so.mime_type     AS storage_mime_type
                FROM learning_material_versions lmv
                JOIN learning_materials lm ON lm.id = lmv.material_id
                JOIN lessons l            ON l.id = lm.lesson_id
                JOIN modules m            ON m.id = l.module_id
                JOIN courses c            ON c.id = m.course_id
                LEFT JOIN storage_objects so ON so.id = lmv.storage_object_id
                WHERE lmv.id = :version_id
                """
            ),
            {"version_id": material_version_id},
        )
    ).first()
    if row is None:
        raise LookupError(f"learning_material_version {material_version_id} not found")

    version = await db.get(LearningMaterialVersion, material_version_id)
    material = await db.get(LearningMaterial, row.material_id)
    if version is None or material is None:
        raise LookupError(f"learning_material_version {material_version_id} disappeared mid-load")

    return _IngestContext(
        version=version,
        material=material,
        lesson_id=row.lesson_id,
        module_id=row.module_id,
        course_id=row.course_id,
        course_title=row.course_title or "",
        module_title=row.module_title or "",
        lesson_title=row.lesson_title or "",
        storage_bucket=row.storage_bucket,
        storage_object_key=row.storage_object_key,
        storage_mime_type=row.storage_mime_type,
        storage_object_id=row.storage_object_id,
    )


async def _ensure_processing_job(
    db: AsyncSession,
    material_version_id: UUID,
) -> ProcessingJob:
    job = await get_latest_processing_job(db, material_version_id)
    if job is None:
        job = ProcessingJob(
            entity_type="material_version",
            entity_id=material_version_id,
            job_type="full_pipeline",
            status="pending",
        )
        db.add(job)
        await db.flush()
    return job


def _resolve_extractor_mime(ctx: _IngestContext) -> str:
    """Pick the MIME the dispatch path keys on.

    Prefers the storage-object MIME (set by the upload-init service from
    the client's Content-Type) and falls back to ``material.material_type``
    so fixtures or backfill that pre-date storage rows still resolve.
    """
    if ctx.storage_mime_type:
        return ctx.storage_mime_type
    return ctx.material.material_type or ""


async def _run_extraction(
    db: AsyncSession,
    ctx: _IngestContext,
    *,
    source_path: Path,
) -> ExtractedContent:
    mime = _resolve_extractor_mime(ctx)
    try:
        extractor = dispatch_extractor(mime)
    except UnsupportedMimeError:
        fallback = maybe_local_mock_extractor(mime)
        if fallback is None:
            raise
        extractor = fallback

    return await extractor.extract(str(source_path))


async def _run_chunking(
    extracted: ExtractedContent,
    *,
    db: AsyncSession,
    llm_gateway: LLMGateway | None,
    pipeline_run_id: UUID,
    parent_job_id: UUID,
    document_title: str,
) -> list[RawChunk]:
    source_type = extracted.source_type
    if source_type in _TIMESTAMP_SOURCES:
        return TimestampAwareChunker().chunk(extracted)
    if source_type in _IMAGE_SOURCES:
        return TokenAwareChunker(max_tokens=_IMAGE_CHUNK_TOKENS).chunk(extracted)
    if source_type in _TEXT_SOURCES:
        if llm_gateway is None:
            return TokenAwareChunker(max_tokens=_DEFAULT_TEXT_TOKENS).chunk(extracted)
        cache = ChunkingCache(db)
        enriched = await SemanticChunker().chunk(
            extracted,
            llm_gateway=llm_gateway,  # type: ignore[arg-type]  # LLMGateway satisfies LLMGatewayProto structurally; mypy flags role: LLMRole vs role: object
            db=db,
            cache=cache,
            document_title=document_title,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
        )
        return list(enriched)
    return TokenAwareChunker(max_tokens=_DEFAULT_TEXT_TOKENS).chunk(extracted)


def _build_chunk_metadata(
    raw: RawChunk,
    *,
    ctx: _IngestContext,
) -> dict[str, Any]:
    base = dict(raw.metadata or {})
    if isinstance(raw, EnrichedChunk) and raw.semantic_metadata:
        base["semantic"] = dict(raw.semantic_metadata)
    base.setdefault("source_type", ctx.material.material_type)
    if ctx.storage_object_id is not None:
        base["storage_object_id"] = str(ctx.storage_object_id)
    if ctx.storage_bucket:
        base["storage_bucket"] = ctx.storage_bucket
    if ctx.storage_object_key:
        base["storage_object_key"] = ctx.storage_object_key
    base["course_title"] = ctx.course_title
    base["module_title"] = ctx.module_title
    base["lesson_title"] = ctx.lesson_title
    base["material_title"] = ctx.material.title
    return base


async def _persist_chunks(
    db: AsyncSession,
    ctx: _IngestContext,
    raw_chunks: list[RawChunk],
    embeddings: list[list[float]],
) -> list[_PersistedChunk]:
    if len(raw_chunks) != len(embeddings):
        raise ValueError(
            f"chunk/embedding length mismatch: {len(raw_chunks)} chunks vs "
            f"{len(embeddings)} embeddings"
        )

    rows: list[DocumentChunk] = []
    for raw, embedding in zip(raw_chunks, embeddings, strict=True):
        content_hash = hashlib.sha256(raw.content.encode("utf-8")).hexdigest()
        rows.append(
            DocumentChunk(
                course_id=ctx.course_id,
                module_id=ctx.module_id,
                lesson_id=ctx.lesson_id,
                material_version_id=ctx.version.id,
                chunk_index=raw.chunk_index,
                chunk_type=ctx.material.material_type,
                content=raw.content,
                metadata_json=_build_chunk_metadata(raw, ctx=ctx),
                embedding=embedding if any(v != 0.0 for v in embedding) else None,
                content_hash=content_hash,
            )
        )
    db.add_all(rows)
    await db.flush()
    return [
        _PersistedChunk(
            id=row.id,
            chunk_index=row.chunk_index,
            content=row.content,
            material_version_id=row.material_version_id,
        )
        for row in rows
    ]


@dataclass
class _Hierarchy:
    """``HierarchyPayload``-shaped view for the KG builder.

    The KG primitive accepts any object with the right attributes (T2.7
    Protocol); a frozen dataclass keeps the orchestrator free of an ORM
    relationship import on Course / Module / Lesson.
    """

    course_id: UUID
    course_title: str
    module_id: UUID
    module_title: str
    lesson_id: UUID
    lesson_title: str
    material_id: UUID
    material_title: str
    material_type: str


def _hierarchy_from_context(ctx: _IngestContext) -> _Hierarchy:
    return _Hierarchy(
        course_id=ctx.course_id,
        course_title=ctx.course_title,
        module_id=ctx.module_id,
        module_title=ctx.module_title,
        lesson_id=ctx.lesson_id,
        lesson_title=ctx.lesson_title,
        material_id=ctx.material.id,
        material_title=ctx.material.title,
        material_type=ctx.material.material_type,
    )


async def _capture_failure(
    db: AsyncSession,
    *,
    ctx: _IngestContext | None,
    job: ProcessingJob | None,
    stage_label: str,
    exc: BaseException,
) -> None:
    """Mark version + job failed and flush. Caller decides commit vs rollback.

    ``stage_label`` becomes part of ``processing_error`` so operators can
    tell which stage tripped from the polling endpoint without trawling
    logs. ``ctx`` may be ``None`` when the failure happened before the
    chain was loaded — we still write the job error in that case.
    """
    detail = f"{stage_label}: {exc!r}"
    if ctx is not None:
        ctx.version.processing_status = "failed"
        ctx.version.processing_error = detail[:_ERROR_FIELD_LIMIT]
    if job is not None:
        job.status = "failed"
        job.error_message = str(exc)[:_ERROR_FIELD_LIMIT]
        job.finished_at = _utcnow()
    try:
        await db.flush()
    except Exception:
        logger.exception(
            "failed to flush failure-state for material_version %s during stage %s",
            ctx.version.id if ctx is not None else "<unloaded>",
            stage_label,
        )


async def _run_stages(
    db: AsyncSession,
    ctx: _IngestContext,
    job: ProcessingJob,
    *,
    source_path: Path,
    pipeline_run_id: UUID,
    kg_client: KnowledgeGraphClient | None,
    llm_gateway: LLMGateway | None,
    embedding_client: EmbeddingClient | None,
) -> None:
    settings = get_settings()
    embed_client = embedding_client or EmbeddingClient(settings)

    stage_label = "extraction"
    try:
        ctx.version.processing_status = "extracting"
        ctx.version.processing_error = None
        job.status = "running"
        job.started_at = job.started_at or _utcnow()
        job.progress_percent = 10
        await db.flush()

        extracted = await _run_extraction(db, ctx, source_path=source_path)

        stage_label = "chunking"
        ctx.version.processing_status = "chunking"
        job.progress_percent = 30
        await db.flush()

        raw_chunks = await _run_chunking(
            extracted,
            db=db,
            llm_gateway=llm_gateway,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=job.id,
            document_title=ctx.material.title,
        )

        stage_label = "embedding"
        ctx.version.processing_status = "embedding"
        job.progress_percent = 60
        await db.flush()

        if not raw_chunks:
            ctx.version.extracted_metadata = dict(ctx.version.extracted_metadata or {}) | {
                **dict(extracted.metadata or {}),
                "chunk_count": 0,
                "knowledge_graph": {
                    "enabled": settings.knowledge_graph_enabled,
                    "concept_count": 0,
                    "relationship_count": 0,
                },
            }
            ctx.version.processing_status = "ready"
            ctx.version.processed_at = _utcnow()
            job.status = "completed"
            job.progress_percent = 100
            job.finished_at = _utcnow()
            await db.flush()
            return

        embeddings = await embed_client.embed(
            [c.content for c in raw_chunks],
            db=db,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=job.id,
        )

        stage_label = "persist"
        ctx.version.processing_status = "enriching"
        job.progress_percent = 80
        persisted = await _persist_chunks(db, ctx, raw_chunks, embeddings)

        kg_summary: dict[str, Any] = {
            "enabled": settings.knowledge_graph_enabled,
            "concept_count": 0,
            "relationship_count": 0,
        }
        if settings.knowledge_graph_enabled and kg_client is not None and llm_gateway is not None:
            stage_label = "kg_build"
            ctx.version.processing_status = "building_kg"
            job.progress_percent = 95
            await db.flush()
            summary = await build_knowledge_graph_for_material_version(
                ctx.version.id,
                list(persisted),
                hierarchy=_hierarchy_from_context(ctx),
                pipeline_run_id=pipeline_run_id,
                db=db,
                kg_client=kg_client,
                llm_gateway=llm_gateway,
                parent_job_id=job.id,
            )
            kg_summary = {
                "enabled": summary.enabled,
                "concept_count": summary.concept_count,
                "relationship_count": summary.relationship_count,
            }

        ctx.version.extracted_metadata = dict(ctx.version.extracted_metadata or {}) | {
            **dict(extracted.metadata or {}),
            "chunk_count": len(persisted),
            "knowledge_graph": kg_summary,
        }
        ctx.version.processing_status = "ready"
        ctx.version.processed_at = _utcnow()
        job.status = "completed"
        job.progress_percent = 100
        job.finished_at = _utcnow()
        await db.flush()

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        await _capture_failure(db, ctx=ctx, job=job, stage_label=stage_label, exc=exc)
        raise


async def run_material_ingest(
    db: AsyncSession,
    material_version_id: UUID,
    pipeline_run_id: UUID,
    *,
    source_path: Path | None = None,
    kg_client: KnowledgeGraphClient | None = None,
    llm_gateway: LLMGateway | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> None:
    """Run the 5-stage materials ingestion pipeline for one version.

    When ``source_path`` is provided (worker context per T4.7) the pipeline
    reads that local file directly and never touches S3. When it is
    ``None`` the pipeline opens its own ``TemporaryDirectory()`` and calls
    ``download_to_temp`` itself; the temp dir is cleaned up on exit.

    The orchestrator FLUSHES; the caller commits. On any exception the
    failure-state writes (``version.processing_status='failed'``,
    ``job.status='failed'``) are flushed and the exception is re-raised so
    the worker can log + decide commit vs rollback.
    """
    ctx: _IngestContext | None = None
    job: ProcessingJob | None = None
    try:
        ctx = await _load_context(db, material_version_id)
        job = await _ensure_processing_job(db, material_version_id)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        await _capture_failure(db, ctx=ctx, job=job, stage_label="bootstrap", exc=exc)
        raise

    if source_path is not None:
        await _run_stages(
            db,
            ctx,
            job,
            source_path=source_path,
            pipeline_run_id=pipeline_run_id,
            kg_client=kg_client,
            llm_gateway=llm_gateway,
            embedding_client=embedding_client,
        )
        return

    if ctx.storage_bucket is None or ctx.storage_object_key is None:
        await _capture_failure(
            db,
            ctx=ctx,
            job=job,
            stage_label="bootstrap",
            exc=ValueError("storage_object missing for material version"),
        )
        raise ValueError(
            f"learning_material_version {material_version_id} has no storage_object; "
            "cannot run ingestion without an explicit source_path"
        )

    storage_view = _StorageView(
        bucket=ctx.storage_bucket,
        object_key=ctx.storage_object_key,
    )
    with TemporaryDirectory(prefix="abridgeai-ingest-") as tmpdir:
        local_path = await download_to_temp(storage_view, Path(tmpdir))
        await _run_stages(
            db,
            ctx,
            job,
            source_path=local_path,
            pipeline_run_id=pipeline_run_id,
            kg_client=kg_client,
            llm_gateway=llm_gateway,
            embedding_client=embedding_client,
        )


@dataclass
class _StorageView:
    """Duck-typed ``StorageObject`` for ``download_to_temp``.

    ``infrastructure/s3.StorageObject`` is a Protocol; tests can pass a
    dataclass and the worker passes the ORM row. Building one here lets
    the orchestrator call S3 without importing the identity feature's
    ``StorageObject`` ORM class (which would tie features together).
    """

    bucket: str
    object_key: str


__all__ = ["run_material_ingest"]
