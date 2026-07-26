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
    HALFVEC,  # type: ignore[import-not-found,import-untyped,unused-ignore]
)
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
    # Per-material escape hatch for the preprocessing cascade.
    #   full           - the whole cascade (default)
    #   normalize_only - unicode + de-hyphenation only; these are never
    #                    destructive, so this is the safe setting for an
    #                    unusual document the filters misread
    #   off            - raw extraction straight to chunking
    preprocess_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'full'")
    )
    visible_to_students: Mapped[bool] = mapped_column(nullable=False, server_default=text("TRUE"))
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_material_versions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ``foreign_keys`` disambiguates against ``LearningMaterial.current_version_id``.
    versions: Mapped[list[LearningMaterialVersion]] = relationship(
        back_populates="material",
        foreign_keys="[LearningMaterialVersion.material_id]",
        cascade="save-update, merge, refresh-expire, expunge",
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

    material: Mapped[LearningMaterial] = relationship(
        back_populates="versions",
        foreign_keys=[material_id],
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="version",
        cascade="save-update, merge, refresh-expire, expunge",
    )


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
        HALFVEC(3072),
        nullable=True,
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    version: Mapped[LearningMaterialVersion] = relationship(back_populates="chunks")


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


class LessonKnowledgeGraphCurated(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    Base,
):
    """Teacher-authored, publishable knowledge graph for one lesson.

    Deliberately SEPARATE from the AI-generated concept graph in Neo4j
    (which stays read-only, behind ``knowledge_graph_enabled``). This is
    the graph the teacher curates (CRUD nodes / edges / node detail) and
    publishes to the student reading-lesson view. See migration 0062.

    Two JSON snapshots:

    * ``draft_json`` — the working copy the teacher edits freely. Shape:
      ``{"nodes": [{id,label,type,definition,weight}],
         "edges": [{source,target,relation}]}``.
    * ``published_json`` — snapshot students see. NULL until first publish;
      re-publishing overwrites it with the current draft.

    ``primary_node_id`` (the single required centre node) is a top-level
    column, not buried in JSON, so the "exactly one primary" rule is
    enforceable/queryable. ``published_primary_node_id`` captures the
    primary at publish time (the draft's primary may drift before the next
    publish).
    """

    __tablename__ = "lesson_knowledge_graphs"
    __table_args__ = (
        UniqueConstraint("lesson_id", name="uq_lesson_knowledge_graphs_lesson"),
    )

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    draft_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    primary_node_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    published_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    published_primary_node_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# ---------------------------------------------------------------------------
# MaterialPreprocessQuarantine — reversible noise filtering
# ---------------------------------------------------------------------------


class MaterialPreprocessQuarantine(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One unit the preprocessing cascade removed or de-prioritized.

    Deliberately NOT rebuilt on re-ingest, unlike ``DocumentChunk``. This
    table holds teacher decisions (``teacher_action``), and those must
    outlive a reprocess — a teacher who restores a wrongly-dropped page
    should not have to restore it again after every re-upload. The
    preprocessing stage upserts on ``(material_version_id, unit_kind,
    ordinal)`` and preserves the ``teacher_action*`` columns.

    Line-level removals are aggregated per pattern per document: a footer
    repeated on 300 pages is ONE row with ``occurrences=299``, not 300 rows.
    """

    __tablename__ = "material_preprocess_quarantine"
    __table_args__ = (
        UniqueConstraint(
            "material_version_id",
            "unit_kind",
            "ordinal",
            name="material_preprocess_quarantine_unit_key",
        ),
        CheckConstraint(
            "unit_kind IN ('page', 'line', 'chunk')",
            name="material_preprocess_quarantine_unit_kind_check",
        ),
        CheckConstraint(
            "teacher_action IS NULL OR teacher_action IN ('restore', 'confirm')",
            name="material_preprocess_quarantine_teacher_action_check",
        ),
    )

    material_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_material_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalized so a course-wide "what did the filter eat" query needs no
    # joins — the same rationale as DocumentChunk's denormalized FKs (§C11).
    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    unit_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    occurrences: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    rule_name: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    rule_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_stage: Mapped[str] = mapped_column(String(16), nullable=False)
    teacher_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    teacher_action_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    teacher_action_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


__all__ = [
    "ChunkingEnrichmentCache",
    "DocumentChunk",
    "LearningMaterial",
    "LearningMaterialVersion",
    "LessonKnowledgeGraphCurated",
    "MaterialPreprocessQuarantine",
]
