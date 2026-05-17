from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session

from abridgeai.core.audit import register_audit_listener
from abridgeai.core.cache.invalidator import register_cache_invalidator
from abridgeai.core.config import get_settings
from abridgeai.core.db.hard_delete_guard import register_hard_delete_guard
from abridgeai.core.db.mixins import (
    PGUUID,
    AuditedByMixin,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from abridgeai.core.db.soft_delete import register_soft_delete_filter


class Base(DeclarativeBase):
    pass


register_audit_listener()


register_soft_delete_filter()


register_cache_invalidator()


register_hard_delete_guard(Session)


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
            pool_recycle=settings.db_pool_recycle_seconds,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False, autoflush=False)
    return _sessionmaker


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def close_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


__all__ = [
    "AuditedByMixin",
    "Base",
    "CreatedAtMixin",
    "PGUUID",
    "SoftDeleteMixin",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "close_db",
    "get_db",
    "get_engine",
    "get_sessionmaker",
    "register_soft_delete_filter",
]
