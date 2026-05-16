from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import ForeignKey, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from abridgeai.core.db.mixins import SoftDeleteMixin
from abridgeai.core.db.soft_delete import register_soft_delete_filter

register_soft_delete_filter()


class _Base(DeclarativeBase):
    pass


class _UsersStub(_Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)


class _Course(SoftDeleteMixin, _Base):
    __tablename__ = "_filter_test_course"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(100))


class _Module(SoftDeleteMixin, _Base):
    __tablename__ = "_filter_test_module"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("_filter_test_course.id"))
    name: Mapped[str] = mapped_column(String(100))


class _Tag(_Base):
    __tablename__ = "_filter_test_tag"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(100))


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://")
    _Base.metadata.create_all(eng)
    yield eng
    _Base.metadata.drop_all(eng)
    eng.dispose()


def _soft_delete(session: Session, instance) -> None:
    instance.deleted_at = datetime.now(UTC)
    instance.deleted_by = uuid.uuid4()
    session.commit()


def test_default_hides(engine):
    with Session(engine) as s:
        c = _Course(title="Hidden")
        s.add(c)
        s.commit()
        _soft_delete(s, c)

        rows = s.execute(select(_Course)).scalars().all()
        assert rows == []


def test_opt_out_shows(engine):
    with Session(engine) as s:
        c = _Course(title="Visible to admin")
        s.add(c)
        s.commit()
        _soft_delete(s, c)

        rows = (
            s.execute(select(_Course).execution_options(include_deleted=True))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].title == "Visible to admin"
        assert rows[0].deleted_at is not None


def test_no_mixin_unaffected(engine):
    with Session(engine) as s:
        s.add_all([_Tag(label="alpha"), _Tag(label="beta")])
        s.commit()

        rows = s.execute(select(_Tag)).scalars().all()
        assert {r.label for r in rows} == {"alpha", "beta"}


def test_join_filter(engine):
    with Session(engine) as s:
        live = _Course(title="Live")
        dead = _Course(title="Dead")
        s.add_all([live, dead])
        s.commit()

        s.add_all(
            [
                _Module(course_id=live.id, name="m-live"),
                _Module(course_id=dead.id, name="m-orphan"),
            ]
        )
        s.commit()
        _soft_delete(s, dead)

        rows = (
            s.execute(
                select(_Course, _Module).join(_Module, _Module.course_id == _Course.id)
            )
            .all()
        )
        assert len(rows) == 1
        course, module = rows[0]
        assert course.title == "Live"
        assert module.name == "m-live"

        all_rows = (
            s.execute(
                select(_Course, _Module)
                .join(_Module, _Module.course_id == _Course.id)
                .execution_options(include_deleted=True)
            )
            .all()
        )
        assert len(all_rows) == 2
