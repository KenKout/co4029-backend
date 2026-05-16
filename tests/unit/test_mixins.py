from __future__ import annotations

from sqlalchemy import Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from abridgeai.core.db import (
    AuditedByMixin,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class _Base(DeclarativeBase):
    pass


class _AllMixins(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    _Base,
):
    __tablename__ = "test_all_columns"


class _OnlyTimestamp(TimestampMixin, _Base):
    __tablename__ = "test_only_timestamp"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True)


class _OnlySoftDelete(SoftDeleteMixin, _Base):
    __tablename__ = "test_only_soft_delete"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True)


class _OnlyAudited(AuditedByMixin, _Base):
    __tablename__ = "test_only_audited"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True)


class _OnlyCreatedAt(CreatedAtMixin, _Base):
    __tablename__ = "test_only_created_at"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True)


def test_all_columns_present() -> None:
    cols = {c.name for c in _AllMixins.__table__.columns}
    expected = {
        "id",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
    assert expected <= cols, f"missing: {expected - cols}"


def test_onupdate_set() -> None:
    assert _OnlyTimestamp.__table__.c.updated_at.onupdate is not None


def test_soft_delete_index() -> None:
    assert _OnlySoftDelete.__table__.c.deleted_at.index is True


def test_audited_by_fk_target() -> None:
    fks = list(_OnlyAudited.__table__.c.created_by.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "users.id"


def test_created_at_only_no_updated_at() -> None:
    cols = {c.name for c in _OnlyCreatedAt.__table__.columns}
    assert "created_at" in cols
    assert "updated_at" not in cols
