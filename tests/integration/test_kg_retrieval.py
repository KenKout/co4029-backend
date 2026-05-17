from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from abridgeai.ai.knowledge_graph import (
    Concept,
    ConceptRelationship,
    KGContext,
    lesson_concept_graph,
    lesson_concepts,
    retrieve_kg_context_for_anchors,
)
from abridgeai.ai.knowledge_graph import retrieval as retrieval_mod
from abridgeai.core.config import get_settings


class _FakeResult:
    def __init__(self, records: list[dict[str, Any]] | dict[str, Any] | None) -> None:
        self._records = records

    def __aiter__(self) -> _FakeResult:
        if isinstance(self._records, list):
            self._iter = iter(self._records)
        else:
            self._iter = iter([])
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def single(self) -> dict[str, Any] | None:
        if isinstance(self._records, dict):
            return self._records
        if isinstance(self._records, list) and self._records:
            return self._records[0]
        return None


class _FakeSession:
    def __init__(self, response: list[dict[str, Any]] | dict[str, Any] | None) -> None:
        self._response = response
        self.last_query: str | None = None
        self.last_params: dict[str, Any] | None = None

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def run(self, query: str, **params: Any) -> _FakeResult:
        self.last_query = query
        self.last_params = params
        return _FakeResult(self._response)


class _FakeKGClient:
    def __init__(self, response: list[dict[str, Any]] | dict[str, Any] | None) -> None:
        self._session = _FakeSession(response)

    def session(self) -> _FakeSession:
        return self._session

    async def aclose(self) -> None:
        return None


@pytest.fixture
def _enable_kg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j-test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_lesson_concepts_returns_concept_dataclasses() -> None:
    client = _FakeKGClient(
        [
            {
                "name": "Binary Search",
                "type": "Concept",
                "definition": "A search algorithm.",
                "confidence": 0.9,
            },
            {
                "name": "Sorted Array",
                "type": "Concept",
                "definition": None,
                "confidence": 0.5,
            },
        ]
    )

    concepts = await lesson_concepts(client, uuid4())

    assert len(concepts) == 2
    assert all(isinstance(c, Concept) for c in concepts)
    assert concepts[0].name == "Binary Search"
    assert concepts[0].confidence == 0.9
    assert concepts[1].definition is None


@pytest.mark.asyncio
async def test_lesson_concept_graph_parses_nodes_and_edges() -> None:
    response = {
        "nodes": [
            {
                "id": "binary search",
                "label": "Binary Search",
                "type": "Concept",
                "definition": "A search algorithm.",
            },
            {
                "id": "sorted array",
                "label": "Sorted Array",
                "type": "Concept",
                "definition": None,
            },
        ],
        "edges": [
            {
                "source": "sorted array",
                "target": "binary search",
                "relation": "PREREQUISITE_OF",
                "evidence": "Binary search requires sorted input.",
                "confidence": 0.88,
            }
        ],
    }
    client = _FakeKGClient(response)

    nodes, edges = await lesson_concept_graph(client, uuid4())

    assert {n.name for n in nodes} == {"Binary Search", "Sorted Array"}
    assert len(edges) == 1
    assert edges[0] == ConceptRelationship(
        source="sorted array",
        target="binary search",
        relation="PREREQUISITE_OF",
        evidence="Binary search requires sorted input.",
        confidence=0.88,
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("_enable_kg")
async def test_retrieve_with_depth_returns_anchor_chain() -> None:
    response = {
        "nodes": [
            {"id": "python", "label": "Python", "type": "Concept", "definition": None},
            {"id": "oop", "label": "OOP", "type": "Concept", "definition": None},
            {
                "id": "inheritance",
                "label": "Inheritance",
                "type": "Concept",
                "definition": None,
            },
        ],
        "edges": [
            {
                "source": "python",
                "target": "oop",
                "relation": "RELATED_TO",
                "evidence": None,
                "confidence": 0.9,
            },
            {
                "source": "oop",
                "target": "inheritance",
                "relation": "RELATED_TO",
                "evidence": None,
                "confidence": 0.85,
            },
        ],
    }
    client = _FakeKGClient(response)

    context = await retrieve_kg_context_for_anchors(
        ["Python"],
        depth=2,
        client=client,
    )

    assert isinstance(context, KGContext)
    assert context.enabled is True
    assert {c.name for c in context.concepts} == {"Python", "OOP", "Inheritance"}
    assert len(context.related) == 2
    assert context.prerequisites == []
    assert client.session().last_params == {"names": ["python"]}


@pytest.mark.asyncio
async def test_retrieve_anchor_disabled_returns_empty_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KNOWLEDGE_GRAPH_ENABLED", raising=False)
    get_settings.cache_clear()
    context = await retrieve_kg_context_for_anchors(["Python"])
    assert context.is_empty
    assert context.enabled is False


@pytest.mark.asyncio
@pytest.mark.usefixtures("_enable_kg")
async def test_retrieve_anchor_empty_input_short_circuits() -> None:
    context = await retrieve_kg_context_for_anchors([])
    assert context.is_empty
    assert context.enabled is False


@pytest.mark.asyncio
@pytest.mark.usefixtures("_enable_kg")
async def test_retrieve_anchor_caps_results() -> None:
    nodes = [
        {"id": f"c{i}", "label": f"Concept-{i}", "type": "Concept", "definition": None}
        for i in range(retrieval_mod.MAX_CONCEPTS + 5)
    ]
    edges = [
        {
            "source": f"c{i}",
            "target": f"c{i + 1}",
            "relation": "RELATED_TO" if i % 2 else "PREREQUISITE_OF",
            "evidence": None,
            "confidence": 0.5,
        }
        for i in range(retrieval_mod.MAX_RELATIONSHIPS + 5)
    ]
    client = _FakeKGClient({"nodes": nodes, "edges": edges})
    context = await retrieve_kg_context_for_anchors(
        ["Concept-0"],
        depth=2,
        client=client,
    )
    assert len(context.concepts) <= retrieval_mod.MAX_CONCEPTS
    assert len(context.prerequisites) <= retrieval_mod.MAX_RELATIONSHIPS
    assert len(context.related) <= retrieval_mod.MAX_RELATIONSHIPS


def test_concept_from_record_skips_invalid() -> None:
    assert retrieval_mod._concept_from_record({"label": None, "name": None}) is None
    assert retrieval_mod._edge_from_record({"source": None, "target": "x"}) is None
    assert retrieval_mod._to_float("not-a-number") is None
    assert retrieval_mod._to_float(0.5) == 0.5
