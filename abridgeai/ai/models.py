"""Cross-feature ORM models for AI pipeline plumbing.

Backs three tables that one feature writes and several read:

* ``ai_model_calls``   - LLM/embedding audit rows (written by ``ai.llm.audit``)
* ``processing_jobs``  - ARQ job timeline (written by materials/quizzes/interviews workers)
* ``generation_runs``  - AI pipeline run header (written by quizzes/interviews services)

Lives outside ``abridgeai.features`` so cross-feature consumers can import
the ORM classes directly without violating the ``Features are independent``
import-linter contract. Feature ownership of the *write* path is unchanged.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import PGUUID, Base, UUIDPrimaryKeyMixin
from abridgeai.core.db.mixins import TimestampMixin

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


class AIModelCall(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "ai_model_calls"
    __table_args__ = (
        # Pipeline calls attribute via a generation_run/processing_job parent;
        # session-runtime calls (e.g. interview_followup) have no parent and
        # attribute via stage_name instead. A row with no attribution at all
        # is still a bug. Relaxed in migration 0018.
        CheckConstraint(
            "generation_run_id IS NOT NULL "
            "OR processing_job_id IS NOT NULL "
            "OR stage_name IS NOT NULL",
            name="ck_ai_model_calls_parent_ref",
        ),
        CheckConstraint(
            "operation IN ('chat_completion', 'embedding')",
            name="ck_ai_model_calls_operation",
        ),
    )

    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("generation_runs.id", ondelete="CASCADE"),
        index=True,
    )
    processing_job_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("processing_jobs.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str | None] = mapped_column(String(30))
    tier: Mapped[str | None] = mapped_column(String(20))
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    stage_name: Mapped[str | None] = mapped_column(String(50))
    operation: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("'chat_completion'")
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(255))
    request_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """ARQ-driven background job for ingestion / RAG / generation work.

    Polymorphic via ``(entity_type, entity_id)`` rather than a hard FK to a
    single parent table - the same job machinery handles material versions,
    lessons, quizzes, interview configs, and generation runs. Hard-delete
    (no soft-delete columns) because finished/failed jobs are audit history,
    not soft-deletable content.
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


class GenerationRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Parent record for AI generation pipeline runs.

    FK columns ``course_id`` / ``module_id`` / ``lesson_id`` / ``requested_by``
    point at feature-owned tables; FKs are declared as DDL strings so this
    module does not import any feature models.
    """

    __tablename__ = "generation_runs"
    __table_args__ = (
        CheckConstraint(
            "generation_type IN ('quiz', 'interview', 'knowledge_graph', 'material_index')",
            name="ck_generation_runs_type",
        ),
        CheckConstraint(
            "source_scope_kind IN ('lesson', 'module', 'course')",
            name="ck_generation_runs_scope",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_generation_runs_status",
        ),
    )

    generation_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_scope_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    course_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
    )
    module_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="NO ACTION"),
    )
    lesson_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
    )
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    config_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    dedup_key: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


__all__ = ["AIModelCall", "GenerationRun", "ProcessingJob"]
