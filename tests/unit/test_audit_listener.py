from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from abridgeai.core.audit import current_actor_var, register_audit_listener
from abridgeai.core.db import AuditedByMixin
from abridgeai.workers.actor import set_worker_actor


register_audit_listener()


class _Base(DeclarativeBase):
    pass


class _StubUser(_Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)


class _Doc(AuditedByMixin, _Base):
    __tablename__ = "test_audit_doc"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(default="")


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite://")
    _Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_http_actor_populates(session: Session) -> None:
    actor = uuid.uuid4()
    token = current_actor_var.set(actor)
    try:
        doc = _Doc(pk=1, title="hello")
        session.add(doc)
        session.commit()
        session.refresh(doc)
        assert doc.created_by == actor
        assert doc.updated_by == actor
    finally:
        current_actor_var.reset(token)


def test_http_update_sets_updated_by(session: Session) -> None:
    creator = uuid.uuid4()
    updater = uuid.uuid4()

    token = current_actor_var.set(creator)
    try:
        doc = _Doc(pk=2, title="v1")
        session.add(doc)
        session.commit()
        session.refresh(doc)
        assert doc.created_by == creator
        assert doc.updated_by == creator
    finally:
        current_actor_var.reset(token)

    token = current_actor_var.set(updater)
    try:
        doc.title = "v2"
        session.commit()
        session.refresh(doc)
    finally:
        current_actor_var.reset(token)

    assert doc.created_by == creator
    assert doc.updated_by == updater


def test_worker_set_actor(session: Session) -> None:
    actor = uuid.uuid4()
    token = set_worker_actor(actor)
    try:
        doc = _Doc(pk=3, title="from-worker")
        session.add(doc)
        session.commit()
        session.refresh(doc)
        assert doc.created_by == actor
        assert doc.updated_by == actor
    finally:
        current_actor_var.reset(token)


def test_system_null_actor(session: Session) -> None:
    assert current_actor_var.get() is None
    doc = _Doc(pk=4, title="system")
    session.add(doc)
    session.commit()
    session.refresh(doc)
    assert doc.created_by is None
    assert doc.updated_by is None
