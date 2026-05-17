"""SQLAlchemy mapping for the ``ai_model_calls`` audit table.

Owned by the LLM integration module because only ``LLMGateway`` and
``EmbeddingClient`` write rows (via ``audit.write_ai_model_call``). Pipeline
call sites must not construct ``AIModelCall`` directly.

T2.4 extends this table with ``pipeline_run_id`` + renames ``pipeline_stage``
to ``stage_name``; this module reflects only the baseline schema (see
``migrations/versions/0001_baseline_schema.py``).
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


class AIModelCall(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "ai_model_calls"
    __table_args__ = (
        CheckConstraint(
            "generation_run_id IS NOT NULL OR processing_job_id IS NOT NULL",
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
    pipeline_stage: Mapped[str | None] = mapped_column(String(40))
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


__all__ = ["AIModelCall"]
