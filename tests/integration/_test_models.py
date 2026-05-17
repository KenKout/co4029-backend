"""Throwaway ORM models for testing the recursive soft-delete service.

Phase 3 (T3.x) builds the canonical ``features/courses/models.py`` with
real Course, Module, Lesson classes. Until then, this module gives the
T0.15 / T0.17 integration tests a real-DB substrate to exercise
relationship traversal against.

Bound to the existing ``courses`` / ``modules`` / ``lessons`` /
``module_items`` / ``lesson_resources`` / ``org_units`` tables created
by ``0001_baseline_schema.py``. We declare only the columns the tests
touch + mixin columns, and rely on the production schema for everything
else. ``auth_sessions`` is exercised via raw SQL in the hard-delete
boundary test — no ORM model is needed because the test only verifies
non-touch.

KEEP THIS FILE. The production ``features/courses/models.py`` is intentionally
relationship-less per the ``Features are independent`` import-linter contract —
no ``relationship()`` declarations means cross-feature ORM walks can't fan out
across feature boundaries. ``soft_delete_cascade`` (core/db/recursive_delete.py)
nevertheless walks ``mapper.relationships`` ONETOMANY edges, so the cascade-walker
tests need an ORM substrate that DOES expose those edges. This file is that
substrate. Production code uses ``soft_delete_cascade`` only on leaf objects
(LessonResource has no ORM children); the walker is fully exercised here.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from abridgeai.core.db.mixins import (
    PGUUID,
    AuditedByMixin,
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
    items: Mapped[list[_ModuleItem]] = relationship(
        back_populates="module",
        cascade="all",
        foreign_keys="[_ModuleItem.module_id]",
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
    resources: Mapped[list[_LessonResource]] = relationship(
        back_populates="lesson",
        cascade="all",
    )


class _ModuleItem(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    _TestBase,
):
    __tablename__ = "module_items"

    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id"),
        nullable=False,
    )
    item_type: Mapped[str] = mapped_column(String(20), nullable=False)
    lesson_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id"),
        nullable=True,
    )
    position: Mapped[int] = mapped_column(nullable=False)

    module: Mapped[_Module] = relationship(
        back_populates="items",
        foreign_keys=[module_id],
    )


class _LessonResource(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    _TestBase,
):
    __tablename__ = "lesson_resources"

    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(20), nullable=False, default="link")
    position: Mapped[int] = mapped_column(nullable=False)

    lesson: Mapped[_Lesson] = relationship(back_populates="resources")


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


class _AuthSession(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    _TestBase,
):
    """Hard-delete table — intentionally NO SoftDeleteMixin.

    Used by ``test_cascade_does_not_touch_hard_delete_table`` to prove
    the cascade walker stops at non-SoftDeleteMixin instances and never
    fabricates a ``deleted_at`` column on tables that don't have one.
    """

    __tablename__ = "auth_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    refresh_token_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    expires_at: Mapped[uuid.UUID] = mapped_column(
        # actually TIMESTAMPTZ; declared loosely since we never SELECT this column
        # in tests (only INSERT via raw SQL with explicit value). Mapping kept
        # minimal so SQLAlchemy can hydrate id/user_id round-trips.
        String,
        nullable=False,
    )


__all__ = [
    "_AuthSession",
    "_Course",
    "_Lesson",
    "_LessonResource",
    "_Module",
    "_ModuleItem",
    "_OrgUnit",
    "_TestBase",
]
