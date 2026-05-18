from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    Base,
    CreatedAtMixin,
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


__all__ = ["CareerPath", "CareerPathCourse", "StudentCareerEnrollment"]
