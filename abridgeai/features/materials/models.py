"""Materials feature ORM models.

Ports the legacy ``backend/app/models/ai_processing.py`` declarations onto
the new ``abridgeai.core.db.Base`` and the canonical mixins. The source of
truth for column names, types, and constraints is
``migrations/versions/0001_baseline_schema.py`` — not the legacy ORM.

Mixin policy (Reconciliation §A13 + draft "Soft-delete eligibility"):

* ``LearningMaterial`` and ``LearningMaterialVersion`` are listed in
  ``SOFT_DELETE_TABLES`` in the baseline migration. They receive
  ``UUIDPrimaryKeyMixin`` + ``TimestampMixin`` + ``AuditedByMixin`` +
  ``SoftDeleteMixin`` so the audit columns the migration added line up
  with the ORM metadata.
* ``DocumentChunk`` is derived data, deleted-and-rebuilt on re-ingest
  (plan §4647 / §4653 / Reconciliation §C11). ``UUIDPrimaryKeyMixin``
  + ``CreatedAtMixin`` only — no soft delete, no ``updated_at``.
* ``ProcessingJob`` is an audit-style timeline of background work. It
  keeps the historical ``TimestampMixin`` schema (the baseline DDL has
  ``updated_at`` so workers can bump it on status transitions) but is
  not in ``SOFT_DELETE_TABLES`` — no audit / soft-delete cols.
* ``ChunkingEnrichmentCache`` (Reconciliation §B9 / §C12) is an LLM
  output cache keyed by ``(content_hash, prompt_version)``. Append-only,
  rebuilt on prompt-version bump — ``UUIDPrimaryKeyMixin`` +
  ``CreatedAtMixin`` only.

Reconciliation §C10 — VARCHAR + CHECK, not PG enum
--------------------------------------------------
``material_type`` and ``processing_status`` are mapped as
``Mapped[str] = mapped_column(String(N), ...)`` with the value set
enforced by ``CheckConstraint`` in ``__table_args__``. The legacy
backend used PG enums; the new schema deliberately does not so we
can extend value sets without ``ALTER TYPE`` migrations.

Reconciliation §C11 — denormalized chunk FKs
---------------------------------------------
``DocumentChunk`` carries ``course_id`` / ``module_id`` / ``lesson_id``
directly even though they are reachable via ``material_version_id ->
material_id -> lesson_id``. The denormalization powers the lesson-scoped
``vector_search`` ANN query without three joins per row. The columns are
NOT NULL and are kept in sync by the ingestion pipeline (T4.4).

Reconciliation §C15 — dual is-current invariant
------------------------------------------------
Both ``LearningMaterial.current_version_id`` and
``LearningMaterialVersion.is_current`` exist intentionally. The first
is a denormalized fast-path lookup; the second is the per-version flag.
Keeping them coherent is the responsibility of the upload-complete
service (T4.5) — out of scope for the model port.

Forward references
------------------
``GenerationRun`` is intentionally NOT declared here. It will land in
``features/ai/models.py`` (T5.x) alongside ``AIModelCall``. Tables
referenced by ORM relationships across feature boundaries use string FK
references so the import-linter independence contract stays green.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import (
    Vector,  # type: ignore[import-not-found,import-untyped,unused-ignore]
)
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import Base
from abridgeai.core.db.mixins import (
    PGUUID,
    AuditedByMixin,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

# ---------------------------------------------------------------------------
# Baseline-DDL value sets (Reconciliation §C10 — VARCHAR + CHECK, not PG enum)
# ---------------------------------------------------------------------------
# Mirror the CHECK constraint text from
# ``migrations/versions/0001_baseline_schema.py`` byte-for-byte so the ORM
# refuses values the DB would reject.
_MATERIAL_TYPE_CHECK = (
    "material_type IN ('video', 'pdf', 'code', 'audio', 'image', 'docx', 'pptx', 'xlsx', 'text')"
)
_PROCESSING_STATUS_CHECK = (
    "processing_status IN "
    "('pending', 'extracting', 'chunking', 'enriching', 'embedding', "
    "'building_kg', 'ready', 'failed', 'cancelled')"
)
_CHUNK_TYPE_CHECK = (
    "chunk_type IN ('video', 'pdf', 'code', 'audio', 'image', 'docx', 'pptx', 'xlsx', 'text')"
)
_PROCESSING_JOB_ENTITY_CHECK = (
    "entity_type IN ('material_version', 'lesson', 'quiz', 'interview_config', 'generation_run')"
)
_PROCESSING_JOB_TYPE_CHECK = (
    "job_type IN "
    "('transcribe', 'parse_document', 'parse_code', 'chunk', 'embed', "
    "'extract_entities', 'extract_relations', 'build_graph', 'full_pipeline', "
    "'generate_quiz', 'generate_interview_questions')"
)
_PROCESSING_JOB_STATUS_CHECK = (
    "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')"
)


# ---------------------------------------------------------------------------
# Learning materials & versions
# ---------------------------------------------------------------------------


class LearningMaterial(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    Base,
):
    """A teacher-uploaded asset attached to a lesson.

    The ``current_version_id`` column is the denormalized fast-path to
    the version flagged ``is_current = TRUE``; the dual-source invariant
    is owned by the upload-complete service (Reconciliation §C15).
    """

    __tablename__ = "learning_materials"
    __table_args__ = (
        CheckConstraint(_MATERIAL_TYPE_CHECK, name="learning_materials_material_type_check"),
    )

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    material_type: Mapped[str] = mapped_column(String(20), nullable=False)
    ai_processing_enabled: Mapped[bool] = mapped_column(nullable=False, server_default=text("TRUE"))
    visible_to_students: Mapped[bool] = mapped_column(nullable=False, server_default=text("TRUE"))
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_material_versions.id", ondelete="SET NULL"),
        nullable=True,
    )


class LearningMaterialVersion(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    Base,
):
    """One immutable revision of a ``LearningMaterial``.

    A new row is inserted on every re-upload; the application code flips
    ``is_current`` to point at the latest ready version. Long-running AI
    work is tracked via ``processing_status`` (9-state pipeline per
    Reconciliation §C10).
    """

    __tablename__ = "learning_material_versions"
    __table_args__ = (
        UniqueConstraint(
            "material_id",
            "version_no",
            name="learning_material_versions_material_id_version_no_key",
        ),
        CheckConstraint(
            _PROCESSING_STATUS_CHECK,
            name="learning_material_versions_processing_status_check",
        ),
    )

    material_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_materials.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    storage_object_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("storage_objects.id", ondelete="RESTRICT"),
        nullable=False,
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    is_current: Mapped[bool] = mapped_column(nullable=False, server_default=text("FALSE"))
    processing_status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# DocumentChunk — RAG corpus row
# ---------------------------------------------------------------------------


class DocumentChunk(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """One vectorised chunk of a learning-material version.

    Append-only and rebuilt on every re-ingest (plan §4647 / §4653) so
    no ``SoftDeleteMixin`` and no ``updated_at``. Carries denormalized
    ``course_id`` / ``module_id`` / ``lesson_id`` per Reconciliation
    §C11 to power the lesson-scoped ANN query without joins.

    The DB column for the JSONB blob is named ``metadata`` (per the
    baseline DDL); the Python attribute is renamed to ``metadata_json``
    because ``metadata`` is reserved on ``DeclarativeBase``.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "material_version_id",
            "chunk_index",
            name="document_chunks_material_version_id_chunk_index_key",
        ),
        CheckConstraint(_CHUNK_TYPE_CHECK, name="document_chunks_chunk_type_check"),
        CheckConstraint("chunk_index >= 0", name="document_chunks_chunk_index_check"),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    material_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_material_versions.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536),
        nullable=True,
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


# ---------------------------------------------------------------------------
# ProcessingJob — background-work timeline
# ---------------------------------------------------------------------------


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """ARQ-driven background job for ingestion / RAG / generation work.

    Polymorphic via ``(entity_type, entity_id)`` rather than a hard FK to
    a single parent table — the same job machinery handles material
    versions, lessons, quizzes, interview configs, and generation runs.
    Hard-delete (no soft-delete columns) because finished/failed jobs are
    audit history, not soft-deletable content.
    """

    __tablename__ = "processing_jobs"
    __table_args__ = (
        CheckConstraint(_PROCESSING_JOB_ENTITY_CHECK, name="processing_jobs_entity_type_check"),
        CheckConstraint(_PROCESSING_JOB_TYPE_CHECK, name="processing_jobs_job_type_check"),
        CheckConstraint(_PROCESSING_JOB_STATUS_CHECK, name="processing_jobs_status_check"),
        CheckConstraint(
            "progress_percent BETWEEN 0 AND 100",
            name="processing_jobs_progress_percent_check",
        ),
        CheckConstraint("retry_count >= 0", name="processing_jobs_retry_count_check"),
    )

    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))


# ---------------------------------------------------------------------------
# ChunkingEnrichmentCache — Reconciliation §B9 / §C12
# ---------------------------------------------------------------------------


class ChunkingEnrichmentCache(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Cached output of the LLM-enrichment chunking stage.

    Keyed by ``(content_hash, prompt_version)``. ``content_hash`` is
    SHA-256 of the raw window text (post Stage A token-aware split,
    pre Stage C LLM call). ``prompt_version`` is bumped any time the
    enrichment prompt changes so old entries auto-invalidate. Stored
    output is the verbatim parsed LLM JSON; a hit short-circuits the
    LLM call on re-uploads of identical content.
    """

    __tablename__ = "chunking_enrichment_cache"
    __table_args__ = (
        UniqueConstraint(
            "content_hash",
            "prompt_version",
            name="uq_chunking_enrichment_cache_hash_prompt",
        ),
    )

    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    output_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)


__all__ = [
    "ChunkingEnrichmentCache",
    "DocumentChunk",
    "LearningMaterial",
    "LearningMaterialVersion",
    "ProcessingJob",
]
