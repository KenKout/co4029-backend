"""Throwaway ORM models for testing the recursive soft-delete service.

Phase 3 (T3.x) builds the canonical ``features/courses/models.py`` with
real Course, Module, Lesson classes. Until then, this module gives the
T0.15 integration tests a real-DB substrate to exercise relationship
traversal against.

Bound to the existing ``courses`` / ``modules`` / ``lessons`` /
``org_units`` tables created by ``0001_baseline_schema.py``. We declare
only the columns the tests touch + mixin columns, and rely on the
production schema for everything else.

DELETE THIS FILE when ``features/courses/models.py`` lands.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from abridgeai.core.db.mixins import (
    AuditedByMixin,
    PGUUID,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class _TestBase(DeclarativeBase):
    pass


class _UserStub(_TestBase):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)


class _Course(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    _TestBase,
):
    __tablename__ = "courses"

    organization_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")

    modules: Mapped[list[_Module]] = relationship(
        back_populates="course",
        cascade="all",
    )


class _Module(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    _TestBase,
):
    __tablename__ = "modules"

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")

    course: Mapped[_Course] = relationship(back_populates="modules")
    lessons: Mapped[list[_Lesson]] = relationship(
        back_populates="module",
        cascade="all",
    )


class _Lesson(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    _TestBase,
):
    __tablename__ = "lessons"

    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id"),
        nullable=False,
    )
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    lesson_type: Mapped[str] = mapped_column(String(30), nullable=False, default="video")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")

    module: Mapped[_Module] = relationship(back_populates="lessons")


class _OrgUnit(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    SoftDeleteMixin,
    _TestBase,
):
    __tablename__ = "org_units"

    organization_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    parent_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("org_units.id"),
        nullable=True,
    )
    unit_type: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    children: Mapped[list[_OrgUnit]] = relationship(
        back_populates="parent",
        remote_side="_OrgUnit.parent_unit_id",
        foreign_keys=[parent_unit_id],
        post_update=True,
    )
    parent: Mapped[_OrgUnit | None] = relationship(
        back_populates="children",
        remote_side="_OrgUnit.id",
        foreign_keys=[parent_unit_id],
        post_update=True,
    )


__all__ = ["_Course", "_Lesson", "_Module", "_OrgUnit", "_TestBase"]
