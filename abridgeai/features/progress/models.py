from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    CreatedAtMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class LessonProgress(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, Base):
    __tablename__ = "lesson_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "lesson_id", name="uq_lesson_progress_user_lesson"),
        CheckConstraint(
            "status IN ('not_started', 'in_progress', 'completed')",
            name="ck_lesson_progress_status",
        ),
        CheckConstraint(
            "completion_percent >= 0 AND completion_percent <= 100",
            name="ck_lesson_progress_completion_percent",
        ),
        CheckConstraint(
            "total_time_seconds >= 0",
            name="ck_lesson_progress_total_time_nonneg",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'not_started'")
    )
    completion_percent: Mapped[Decimal] = mapped_column(
        Numeric(5, 2), nullable=False, server_default=text("0")
    )
    last_activity_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_time_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )


class MaterialEngagement(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "material_engagement"
    __table_args__ = (
        CheckConstraint(
            "engagement_seconds >= 0",
            name="ck_material_engagement_seconds_nonneg",
        ),
        CheckConstraint(
            "scroll_position_percent IS NULL "
            "OR (scroll_position_percent >= 0 AND scroll_position_percent <= 100)",
            name="ck_material_engagement_scroll_range",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    material_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("learning_material_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    engagement_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    scroll_position_percent: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


__all__ = ["LessonProgress", "MaterialEngagement"]
