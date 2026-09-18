"""Quiz reporting, metrics, and audit-event ORM models.

Kept separate from the core quiz-taking models so the quizzes feature stays
under the repository's per-file size cap. These tables have no relationships
back into the Python classes in ``models.py``; their foreign keys remain the
same database-level contracts.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import (
    PGUUID,
    Base,
    CreatedAtMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class QuizStatisticsCache(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "quiz_statistics_cache"

    quiz_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quizzes.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="NO ACTION"),
    )
    facility_index: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    discrimination_index: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QuizAuditEvent(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "quiz_audit_events"

    event_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    quiz_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quizzes.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    subject_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_attempts.id", ondelete="NO ACTION"),
    )
    subject_question_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="NO ACTION"),
    )
    subject_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


__all__ = ["QuizAuditEvent", "QuizStatisticsCache"]
