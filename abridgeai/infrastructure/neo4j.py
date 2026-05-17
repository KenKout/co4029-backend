"""Neo4j async driver lifecycle.

Driver-layer module: factory + singleton + context manager + the slim
``KnowledgeGraphClient`` (just session access). All Cypher queries and KG
business logic live in :mod:`abridgeai.ai.knowledge_graph`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from neo4j import AsyncDriver, AsyncGraphDatabase

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import AppError

if TYPE_CHECKING:
    from neo4j import AsyncSession


class KnowledgeGraphDisabledError(AppError):
    """Raised when Neo4j is requested but ``knowledge_graph_enabled`` is False
    (or required credentials are missing). Distinct from a connection error:
    this means the feature is intentionally off, not that Neo4j is down."""


_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    global _driver
    settings = get_settings()
    if not settings.knowledge_graph_enabled:
        raise KnowledgeGraphDisabledError("Knowledge graph is not enabled")
    if not settings.neo4j_password:
        raise KnowledgeGraphDisabledError(
            "NEO4J_PASSWORD is required when knowledge graph is enabled"
        )
    if not settings.neo4j_uri:
        raise KnowledgeGraphDisabledError("NEO4J_URI is required when knowledge graph is enabled")
    if _driver is None:
        password = settings.neo4j_password.get_secret_value()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        )
    return _driver


async def close_neo4j() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
    _driver = None


def _reset_for_tests() -> None:
    """Test hook: drop cached driver without awaiting close."""
    global _driver
    _driver = None


@asynccontextmanager
async def graph_client() -> AsyncIterator[KnowledgeGraphClient]:
    yield KnowledgeGraphClient(get_neo4j_driver())


class KnowledgeGraphClient:
    """Thin handle over an :class:`AsyncDriver`.

    All Cypher logic lives in :mod:`abridgeai.ai.knowledge_graph`; this class
    just exposes ``session()`` so callers don't import ``neo4j`` directly.
    """

    def __init__(self, driver: AsyncDriver) -> None:
        self.driver = driver

    def session(self) -> AsyncSession:
        return self.driver.session()

    async def aclose(self) -> None:
        await self.driver.close()


__all__ = [
    "KnowledgeGraphClient",
    "KnowledgeGraphDisabledError",
    "close_neo4j",
    "get_neo4j_driver",
    "graph_client",
]
