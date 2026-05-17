from collections.abc import AsyncIterator
from weakref import WeakSet

from sqlalchemy import event, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session

from abridgeai.core.audit import register_audit_listener
from abridgeai.core.audit.context import current_actor_var
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


_SET_APP_ACTOR_SQL = text("SELECT set_config('app.actor_id', :v, true)")
_REGISTERED_ENGINES: WeakSet[AsyncEngine] = WeakSet()


def _set_app_actor_on_begin(conn: Connection) -> None:
    actor = current_actor_var.get()
    if actor is None:
        return
    conn.execute(_SET_APP_ACTOR_SQL, {"v": str(actor)})


def register_app_actor_listener(engine: AsyncEngine) -> None:
    """Bind ``app.actor_id`` GUC on every transaction begin (idempotent).

    The companion to ``current_actor_var``: the HTTP layer sets the
    ContextVar in ``get_current_user``; this listener propagates it to
    PostgreSQL via ``set_config('app.actor_id', :v, true)`` so audit
    triggers (T3) can read it from ``current_setting``. The third arg
    ``true`` makes the GUC transaction-local, so it cannot leak across
    pooled connections after rollback/commit.
    """
    if engine in _REGISTERED_ENGINES:
        return
    event.listen(engine.sync_engine, "begin", _set_app_actor_on_begin)
    _REGISTERED_ENGINES.add(engine)


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
        register_app_actor_listener(_engine)
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
