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
from abridgeai.ai.chunking.contextual import build_contextual_text
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
from abridgeai.ai.preprocessing.dedup import link_semantic_duplicates
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.features.materials.ingestion.preprocess import run_preprocess_stage
from abridgeai.features.materials.ingestion.progress import clear_progress, publish_progress
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
_TEXT_SOURCES: frozenset[str] = frozenset(
    {"pdf", "docx", "pptx", "xlsx", "html", "text", "code", "mock"}
)
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
    organization_id: UUID
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
                    c.organization_id AS organization_id,
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
        organization_id=row.organization_id,
        course_title=row.course_title or "",
        module_title=row.module_title or "",
        lesson_title=row.lesson_title or "",
        storage_bucket=row.storage_bucket,
        storage_object_key=row.storage_object_key,
        storage_mime_type=row.storage_mime_type,
        storage_object_id=row.storage_object_id,
    )


# A job in one of these states is over. Reusing its row for a new run buries
# the previous run's outcome and mixes two runs' cost rows under one id.
_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})


async def _ensure_processing_job(
    db: AsyncSession,
    material_version_id: UUID,
) -> ProcessingJob:
    """Return the in-flight job for this version, or start a new one.

    Only a job that has not finished is reused — that is the case this exists
    for: an ARQ retry, or the reaper re-enqueueing an ingest whose worker died,
    must attach to the row already tracking that attempt rather than opening a
    second one.

    A *finished* job is never reused. It used to be, and a reprocess then
    inherited the original row: ``started_at`` stayed pinned to the first run
    ever (it is only set ``or _utcnow()``), the previous run's outcome was
    overwritten, and every ``ai_model_calls`` row from every reprocess
    accumulated under one ``processing_job_id`` — so per-run cost could not be
    read back, and "has my reprocess finished?" could not be answered from the
    table at all.
    """
    job = await get_latest_processing_job(db, material_version_id)
    if job is None or job.status in _TERMINAL_JOB_STATUSES:
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
    llm_gateway: LLMGateway | None = None,
) -> ExtractedContent:
    mime = _resolve_extractor_mime(ctx)
    try:
        # Media extractors (audio/image/video) need db + gateway injected to
        # write STT/vision audit rows and reach the providers; dispatch picks
        # only the kwargs each constructor accepts, so no-arg extractors
        # (pdf/docx/…) are unaffected.
        extractor = dispatch_extractor(mime, db=db, gateway=llm_gateway)
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
        # Pass the process sessionmaker so Stage-C enrichment runs each window
        # CONCURRENTLY inside its own short-lived session (audit + cache writes
        # are isolated per coroutine). This restores parallelism without the
        # shared-session flush race that used to hang multi-window PDFs. The
        # per-window sessions only touch ``ai_model_calls`` /
        # ``chunking_enrichment_cache`` — no lock conflict with the main ``db``
        # transaction (which holds the version/job rows).
        enriched = await SemanticChunker().chunk(
            extracted,
            llm_gateway=llm_gateway,  # type: ignore[arg-type]  # LLMGateway satisfies LLMGatewayProto structurally; mypy flags role: LLMRole vs role: object
            db=db,
            cache=cache,
            document_title=document_title,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
            session_factory=get_sessionmaker(),
        )
        return list(enriched)
    return TokenAwareChunker(max_tokens=_DEFAULT_TEXT_TOKENS).chunk(extracted)


def _build_chunk_metadata(
    raw: RawChunk,
    *,
    ctx: _IngestContext,
    canonical_hash: str | None = None,
) -> dict[str, Any]:
    base = dict(raw.metadata or {})
    if canonical_hash is not None:
        # Near-duplicate of an earlier chunk (typically a recap slide). It is
        # KEPT and stays retrievable — a recap is often phrased closest to how
        # the exam question is worded, which makes it the best anchor even
        # though it is a poor answer body. Retrieval collapses the pair at
        # query time via this pointer.
        base["canonical_chunk_hash"] = canonical_hash
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

    # Idempotency guard: purge any pre-existing chunks for this version before
    # inserting. Ingestion can legitimately re-run on the SAME version — ARQ
    # auto-retries a failed job (max_tries=3), and a re-enqueue that doesn't go
    # through ``reprocess_material`` (which purges) would otherwise trip the
    # ``document_chunks_material_version_id_chunk_index_key`` UNIQUE constraint
    # on ``(material_version_id, chunk_index)`` and fail the whole run with a
    # duplicate-key error — leaving the document stuck in ``pending``. Deleting
    # first makes persist a clean overwrite, so retries converge instead of
    # dead-locking on their own partial output.
    await db.execute(
        text("DELETE FROM document_chunks WHERE material_version_id = :vid"),
        {"vid": ctx.version.id},
    )
    await db.flush()

    # Link near-duplicates to their first occurrence. Annotation only: nothing
    # is dropped, so an over-eager threshold costs a redundant retrieval hit
    # rather than a missing lecture slide.
    duplicate_links = link_semantic_duplicates(embeddings)
    hashes = [hashlib.sha256(c.content.encode("utf-8")).hexdigest() for c in raw_chunks]

    rows: list[DocumentChunk] = []
    zero_vector_indices: list[int] = []
    for position, (raw, embedding) in enumerate(zip(raw_chunks, embeddings, strict=True)):
        content_hash = hashes[position]
        canonical_index = duplicate_links.get(position)
        canonical_hash = hashes[canonical_index] if canonical_index is not None else None
        # Detect zero-vector embeddings (silent provider failure mode —
        # the LAN gateway has been observed returning all-zero arrays
        # on transient backend hiccups). Persisting these as NULL was
        # the original bug in this module: vector_search filters
        # ``WHERE embedding IS NOT NULL`` so a NULL row is invisible
        # to retrieval, and the quiz pipeline silently runs on BM25
        # scraps and hallucinates ungrounded questions. Fail loudly
        # instead so reprocess can fix it on the next attempt.
        if not any(v != 0.0 for v in embedding):
            zero_vector_indices.append(raw.chunk_index)
            continue
        rows.append(
            DocumentChunk(
                course_id=ctx.course_id,
                module_id=ctx.module_id,
                lesson_id=ctx.lesson_id,
                material_version_id=ctx.version.id,
                chunk_index=raw.chunk_index,
                chunk_type=ctx.material.material_type,
                content=raw.content,
                metadata_json=_build_chunk_metadata(
                    raw, ctx=ctx, canonical_hash=canonical_hash
                ),
                embedding=embedding,
                content_hash=content_hash,
            )
        )

    if zero_vector_indices:
        raise RuntimeError(
            f"embedding provider returned zero vectors for chunks "
            f"{zero_vector_indices} of material_version {ctx.version.id}; "
            f"refusing to persist NULL embeddings (vector_search filters "
            f"NULL rows, leaving downstream quiz/RAG pipelines with no "
            f"retrievable content). Reprocess to retry."
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

    organization_id: UUID
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
        organization_id=ctx.organization_id,
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
        await publish_progress(
            ctx.version.id, status="extracting", percent=10, stage_label=stage_label
        )

        extracted = await _run_extraction(db, ctx, source_path=source_path, llm_gateway=llm_gateway)

        # Stage 1b — preprocessing. Drops blank pages, strips running
        # headers/footers and page numbers, tags cover/instructor/TOC/
        # reference/closing pages, groups slide decks, normalizes unicode and
        # de-hyphenates, and OCRs image-only pages. ``processing_status``
        # deliberately stays "extracting": that column carries a 9-value CHECK
        # constraint, and ``stage_label`` (free-form) is what the UI surfaces.
        stage_label = "preprocessing"
        job.progress_percent = 20
        await db.flush()
        await publish_progress(
            ctx.version.id, status="extracting", percent=20, stage_label=stage_label
        )
        extracted, _preprocess_report = await run_preprocess_stage(
            extracted,
            db=db,
            settings=settings,
            source_path=source_path,
            llm_gateway=llm_gateway,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=job.id,
            material_version_id=ctx.version.id,
            course_id=ctx.course_id,
            mode=getattr(ctx.material, "preprocess_mode", "full") or "full",
        )

        stage_label = "chunking"
        ctx.version.processing_status = "chunking"
        job.progress_percent = 30
        await db.flush()
        await publish_progress(
            ctx.version.id, status="chunking", percent=30, stage_label=stage_label
        )

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
        await publish_progress(
            ctx.version.id, status="embedding", percent=60, stage_label=stage_label
        )

        # Drop empty / whitespace-only chunks before embedding. The embedding
        # API rejects an empty input string with HTTP 400 and fails the WHOLE
        # batch, so a single blank chunk (silent audio segment, blank video
        # frame, blank PDF page) would otherwise crash an entire ingest. Guard
        # on the contextual embed input, not just raw content, so a chunk that
        # is only a "[Topic: …]" prefix with no body is also dropped.
        raw_chunks = [c for c in raw_chunks if (c.content or "").strip()]

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
            await clear_progress(ctx.version.id)
            return

        # Anthropic Contextual Retrieval: prepend Stage C section_title +
        # context_sentence onto each chunk before embedding so the vector
        # captures topic context, not just bare paragraph text. Reduces
        # retrieval failure ~35% on BEIR-style benchmarks. ``content``
        # itself is unchanged in DB — only the embedder input is
        # contextualized.
        embed_inputs = [build_contextual_text(c) for c in raw_chunks]
        embeddings = await embed_client.embed(
            embed_inputs,
            db=db,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=job.id,
        )

        stage_label = "persist"
        ctx.version.processing_status = "enriching"
        job.progress_percent = 80
        await db.flush()
        await publish_progress(
            ctx.version.id, status="enriching", percent=80, stage_label=stage_label
        )
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
            await publish_progress(
                ctx.version.id, status="building_kg", percent=95, stage_label=stage_label
            )

            # KG build is one sequential LLM call per chunk — on a big doc it
            # runs for minutes at a fixed 95%. Emit live sub-progress per
            # chunk (mapped into the 95→99% band, with a "done/total" detail)
            # so the teacher can see it's still working, not frozen.
            version_id_for_progress = ctx.version.id

            async def _kg_progress(done: int, total: int) -> None:
                pct = 95 + int(4 * done / total) if total else 95
                await publish_progress(
                    version_id_for_progress,
                    status="building_kg",
                    percent=min(99, pct),
                    stage_label="kg_build",
                    detail=f"{done}/{total}",
                )

            summary = await build_knowledge_graph_for_material_version(
                ctx.version.id,
                list(persisted),
                hierarchy=_hierarchy_from_context(ctx),
                pipeline_run_id=pipeline_run_id,
                db=db,
                kg_client=kg_client,
                llm_gateway=llm_gateway,
                parent_job_id=job.id,
                on_progress=_kg_progress,
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
        await clear_progress(ctx.version.id)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        await _capture_failure(db, ctx=ctx, job=job, stage_label=stage_label, exc=exc)
        # Surface the failure to the live-progress channel so the UI flips
        # to "failed" immediately instead of waiting on the DB commit.
        await publish_progress(
            ctx.version.id, status="failed", percent=job.progress_percent, stage_label=stage_label
        )
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
