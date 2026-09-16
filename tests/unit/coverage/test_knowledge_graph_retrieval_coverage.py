from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.ai.knowledge_graph import retrieval
from abridgeai.ai.knowledge_graph.schemas import Concept, ConceptRelationship


class _AsyncRecords:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self._records = records

    def __aiter__(self) -> _AsyncRecords:
        self._iterator = iter(self._records)
        return self

    async def __anext__(self) -> dict[str, object]:
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _SessionContext:
    def __init__(self, session: object) -> None:
        self.session = session

    async def __aenter__(self) -> object:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


def _client_with_result(result: object) -> tuple[object, AsyncMock]:
    run = AsyncMock(return_value=result)
    session = SimpleNamespace(run=run)
    return SimpleNamespace(session=lambda: _SessionContext(session)), run


@pytest.mark.asyncio
async def test_lesson_concepts_filters_invalid_names() -> None:
    client, run = _client_with_result(
        _AsyncRecords(
            [
                {
                    "name": "Normalization",
                    "type": None,
                    "definition": "Reduce redundancy",
                    "confidence": 0.9,
                },
                {"name": None},
            ]
        )
    )

    concepts = await retrieval.lesson_concepts(client, uuid4())

    assert concepts == [
        Concept(
            name="Normalization",
            definition="Reduce redundancy",
            confidence=0.9,
        )
    ]
    assert "lesson_id" in run.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("preview", [True, False])
async def test_lesson_graph_parses_nodes_edges_and_clamps_options(preview: bool) -> None:
    record = {
        "nodes": [
            {"id": "a", "label": "A", "type": "Skill", "confidence": "bad"},
            {"id": "invalid"},
        ],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "relation": "PREREQUISITE_OF",
                "confidence": 1,
            },
            {"source": 1, "target": "b"},
        ],
    }
    result = SimpleNamespace(single=AsyncMock(return_value=record))
    client, run = _client_with_result(result)

    if preview:
        concepts, edges = await retrieval.lesson_concept_graph_preview(
            client, "lesson", limit=999
        )
        assert run.await_args.kwargs["limit"] == 60
    else:
        concepts, edges = await retrieval.lesson_concept_graph(client, "lesson", depth=0)
        assert "*1..1" in run.await_args.args[0]

    assert concepts == [Concept(name="A", type="Skill")]
    assert edges == [
        ConceptRelationship(
            source="a",
            target="b",
            relation="PREREQUISITE_OF",
            confidence=1.0,
        )
    ]


@pytest.mark.asyncio
async def test_lesson_graph_returns_empty_for_missing_record() -> None:
    result = SimpleNamespace(single=AsyncMock(return_value=None))
    client, _run = _client_with_result(result)
    assert await retrieval.lesson_concept_graph_preview(client, "lesson") == ([], [])
    assert await retrieval.lesson_concept_graph(client, "lesson") == ([], [])


def test_context_fold_deduplicates_bounds_and_normalizes_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval, "MAX_CONCEPTS", 1)
    monkeypatch.setattr(retrieval, "MAX_RELATIONSHIPS", 1)
    record = {
        "nodes": [
            {"id": "sql", "label": "SQL", "confidence": 0.8},
            {"id": "sql", "label": "Duplicate"},
            {"id": "db", "label": "Database"},
            {"id": "broken"},
        ],
        "edges": [
            {"source": "sql", "target": "db", "relation": "PREREQUISITE_OF"},
            {"source": "sql", "target": "db", "relation": "PREREQUISITE_OF"},
            {"source": "db", "target": "sql", "relation": "UNKNOWN"},
            {"source": None, "target": "sql"},
        ],
    }

    context = retrieval._context_from_result(record)

    assert [concept.name for concept in context.concepts] == ["SQL"]
    assert len(context.prerequisites) == 1
    assert context.related[0].relation == "RELATED_TO"
    assert retrieval._context_from_result(None).is_empty


@pytest.mark.asyncio
async def test_anchor_context_guards_feature_and_queries_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retrieval,
        "get_settings",
        lambda: SimpleNamespace(knowledge_graph_enabled=False),
    )
    assert not (await retrieval.retrieve_kg_context_for_anchors([], org_id=uuid4())).enabled
    assert not (
        await retrieval.retrieve_kg_context_for_anchors(["SQL"], org_id=uuid4())
    ).enabled

    monkeypatch.setattr(
        retrieval,
        "get_settings",
        lambda: SimpleNamespace(knowledge_graph_enabled=True),
    )
    result = SimpleNamespace(single=AsyncMock(return_value={"nodes": [], "edges": []}))
    client, run = _client_with_result(result)
    context = await retrieval.retrieve_kg_context_for_anchors(
        [" SQL ", ""], org_id="org", depth=99, client=client
    )
    assert context.enabled
    assert run.await_args.kwargs == {"names": ["sql"], "org_id": "org"}
    assert "*0..3" in run.await_args.args[0]


@pytest.mark.asyncio
async def test_lesson_id_context_queries_seed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retrieval,
        "get_settings",
        lambda: SimpleNamespace(knowledge_graph_enabled=True),
    )
    result = SimpleNamespace(single=AsyncMock(return_value={"nodes": [], "edges": []}))
    client, run = _client_with_result(result)

    context = await retrieval.retrieve_kg_context_for_lesson_ids(
        [None, "lesson-1"], org_id="org", depth=-2, client=client
    )

    assert context.enabled
    assert run.await_args.kwargs == {"lesson_ids": ["lesson-1"], "org_id": "org"}
    assert "*0..1" in run.await_args.args[0]
    assert not (
        await retrieval.retrieve_kg_context_for_lesson_ids([], org_id="org", client=client)
    ).enabled


@pytest.mark.asyncio
async def test_legacy_lesson_aggregate_deduplicates_and_buckets_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retrieval,
        "get_settings",
        lambda: SimpleNamespace(knowledge_graph_enabled=True),
    )
    prereq = ConceptRelationship("A", "B", "PREREQUISITE_OF")
    related = ConceptRelationship("B", "A", "RELATED_TO")
    graphs = [
        ([Concept("SQL")], [prereq]),
        ([Concept("sql"), Concept("Database")], [prereq, related]),
    ]
    monkeypatch.setattr(
        retrieval,
        "lesson_concept_graph",
        AsyncMock(side_effect=graphs),
    )

    context = await retrieval.retrieve_kg_context_for_lessons(
        ["lesson-1", "lesson-2"], client=object()
    )

    assert [concept.name for concept in context.concepts] == ["SQL", "Database"]
    assert context.prerequisites == [prereq]
    assert context.related == [related]


@pytest.mark.asyncio
async def test_owned_client_disabled_error_degrades_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retrieval,
        "get_settings",
        lambda: SimpleNamespace(knowledge_graph_enabled=True),
    )

    class _DisabledContext:
        async def __aenter__(self) -> object:
            raise retrieval.KnowledgeGraphDisabledError("disabled")

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(retrieval, "graph_client", lambda: _DisabledContext())

    assert not (
        await retrieval.retrieve_kg_context_for_anchors(["SQL"], org_id="org")
    ).enabled
    assert not (
        await retrieval.retrieve_kg_context_for_lesson_ids(["lesson"], org_id="org")
    ).enabled
    assert not (
        await retrieval.retrieve_kg_context_for_lessons(["lesson"])
    ).enabled
