from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    Base,
    CreatedAtMixin,
    UUIDPrimaryKeyMixin,
)
from abridgeai.features.access_control.models import (
    CareerPath,
    StudentCareerEnrollment,
)


class CareerPathCourse(CreatedAtMixin, Base):
    __tablename__ = "career_course_items"
    __table_args__ = (
        UniqueConstraint(
            "career_path_id",
            "position",
            name="career_course_items_career_path_id_position_key",
        ),
        CheckConstraint("position > 0", name="career_course_items_position_check"),
    )

    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_paths.id", ondelete="CASCADE"),
        primary_key=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    is_required: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("TRUE"))

    career_path: Mapped[CareerPath] = relationship()


class CareerReadinessSnapshot(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Point-in-time readiness score for a (student, career path) pair.

    FR-6.8: written nightly by the readiness cron (and on demand) so
    managers can aggregate pathway-level outcomes over time. Maps the
    baseline-DDL ``career_readiness_snapshots`` table (migration 0001);
    ``readiness_score`` is the 0–100 pathway completion aggregate from
    :func:`features.career_paths.services.enrollment.get_my_path_progress`.
    """

    __tablename__ = "career_readiness_snapshots"
    __table_args__ = (
        CheckConstraint(
            "readiness_score BETWEEN 0 AND 100",
            name="career_readiness_snapshots_readiness_score_check",
        ),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    career_path_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("career_paths.id", ondelete="CASCADE"),
        nullable=False,
    )
    readiness_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


__all__ = [
    "CareerPath",
    "CareerPathCourse",
    "CareerReadinessSnapshot",
    "StudentCareerEnrollment",
]
