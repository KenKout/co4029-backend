from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from pydantic import SecretStr

from abridgeai.core.config import get_settings
from abridgeai.infrastructure import neo4j as neo4j_mod
from abridgeai.infrastructure.neo4j import (
    KnowledgeGraphClient,
    KnowledgeGraphDisabledError,
    close_neo4j,
    get_neo4j_driver,
    graph_client,
)


@pytest.fixture(autouse=True)
def _reset_state() -> Any:
    neo4j_mod._reset_for_tests()
    get_settings.cache_clear()
    yield
    neo4j_mod._reset_for_tests()
    get_settings.cache_clear()


def _stub_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    uri: str = "",
    password: str | None = None,
) -> None:
    """Force ``neo4j.get_settings`` to a controlled stub.

    ``delenv`` alone is insufficient: pydantic ``Settings`` loads values from
    the on-disk ``.env`` (which sets KNOWLEDGE_GRAPH_ENABLED=true and a
    password), so an env-only approach never exercises the disabled / missing
    branches. Patching the settings accessor the module actually calls makes the
    test independent of the developer's .env.
    """
    stub = SimpleNamespace(
        knowledge_graph_enabled=enabled,
        neo4j_uri=uri,
        neo4j_user="neo4j",
        neo4j_password=SecretStr(password) if password is not None else None,
        neo4j_max_connection_pool_size=50,
    )
    monkeypatch.setattr(neo4j_mod, "get_settings", lambda: stub)


def test_disabled_skips_init(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, enabled=False)

    with pytest.raises(KnowledgeGraphDisabledError, match="not enabled"):
        get_neo4j_driver()


def test_enabled_no_password_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, enabled=True, uri="bolt://localhost:7687", password=None)

    with pytest.raises(KnowledgeGraphDisabledError, match="NEO4J_PASSWORD"):
        get_neo4j_driver()


def test_enabled_no_uri_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_settings(monkeypatch, enabled=True, uri="", password="secret")

    with pytest.raises(KnowledgeGraphDisabledError, match="NEO4J_URI"):
        get_neo4j_driver()


def test_singleton_caches_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    sentinel = MagicMock(name="AsyncDriver")
    factory = MagicMock(return_value=sentinel)
    monkeypatch.setattr(neo4j_mod.AsyncGraphDatabase, "driver", factory)

    first = get_neo4j_driver()
    second = get_neo4j_driver()

    assert first is sentinel
    assert second is sentinel
    factory.assert_called_once()
    args, kwargs = factory.call_args
    assert args[0] == "bolt://localhost:7687"
    assert kwargs["auth"] == ("neo4j", "secret")
    assert kwargs["max_connection_pool_size"] == 50


@pytest.mark.asyncio
async def test_close_neo4j_resets_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    closed = MagicMock()

    class _StubDriver:
        async def close(self) -> None:
            closed()

    monkeypatch.setattr(
        neo4j_mod.AsyncGraphDatabase, "driver", MagicMock(return_value=_StubDriver())
    )
    get_neo4j_driver()
    await close_neo4j()
    closed.assert_called_once()
    assert neo4j_mod._driver is None


@pytest.mark.asyncio
async def test_graph_client_yields_kg_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    sentinel = MagicMock(name="AsyncDriver")
    monkeypatch.setattr(neo4j_mod.AsyncGraphDatabase, "driver", MagicMock(return_value=sentinel))

    async with graph_client() as kg:
        assert isinstance(kg, KnowledgeGraphClient)
        assert kg.driver is sentinel
