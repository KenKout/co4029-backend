from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest

from abridgeai.ai.knowledge_graph import (
    KG_BUILD_STAGE_NAME,
    Concept,
    KGSummary,
    build_knowledge_graph_for_material_version,
)
from abridgeai.ai.knowledge_graph import builder as builder_mod
from abridgeai.ai.llm.gateway import LLMResult
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.core.config import get_settings
from abridgeai.infrastructure.neo4j import KnowledgeGraphClient


@dataclass
class _Chunk:
    id: UUID
    chunk_index: int
    content: str
    material_version_id: UUID


@dataclass
class _Hierarchy:
    organization_id: UUID
    course_id: UUID
    course_title: str
    module_id: UUID
    module_title: str
    lesson_id: UUID
    lesson_title: str
    material_id: UUID
    material_title: str
    material_type: str


class _FakeSession:
    def __init__(self, captured: list[dict[str, Any]]) -> None:
        self._captured = captured

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def execute_write(self, fn: Any, *args: Any) -> None:
        self._captured.append({"fn_name": getattr(fn, "__name__", "<lambda>"), "args": args})


class _FakeKGClient:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    def session(self) -> _FakeSession:
        return _FakeSession(self.upserts)

    async def aclose(self) -> None:
        return None


@dataclass
class _RecordedCall:
    role: LLMRole
    stage_name: str | None
    pipeline_run_id: UUID | None
    parent_job_id: UUID | None
    user_prompt: str


class _FakeGateway:
    """Captures gateway kwargs and returns a canned KG payload.

    Avoids touching the real ORM audit path -- that's covered by T2.4.
    Here we verify the builder THREADS pipeline_run_id + stage_name through
    every per-chunk gateway invocation.
    """

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or {
            "entities": [
                {
                    "name": "Binary Search",
                    "type": "Concept",
                    "definition": "A search algorithm.",
                    "confidence": 0.92,
                },
                {
                    "name": "Sorted Array",
                    "type": "Concept",
                    "definition": "An ordered array.",
                    "confidence": 0.85,
                },
            ],
            "relationships": [
                {
                    "source": "Sorted Array",
                    "target": "Binary Search",
                    "relation": "PREREQUISITE_OF",
                    "evidence": "Binary search requires sorted input.",
                    "confidence": 0.88,
                }
            ],
        }
        self.calls: list[_RecordedCall] = []

    async def generate_json(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        db: Any,
        stage_name: str | None = None,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
        parent_run_id: UUID | None = None,
    ) -> LLMResult:
        self.calls.append(
            _RecordedCall(
                role=role,
                stage_name=stage_name,
                pipeline_run_id=pipeline_run_id,
                parent_job_id=parent_job_id,
                user_prompt=user_prompt,
            )
        )
        return LLMResult(
            role=role,
            tier="small",
            model_name="fake-model",
            base_url="https://fake.test/v1",
            stage_name=stage_name,
            pipeline_run_id=pipeline_run_id,
            request_payload={},
            response_payload={},
            content_json=self._payload,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_input_tokens=None,
            latency_ms=12,
            estimated_cost_usd=Decimal("0.000010"),
        )


def _make_chunk(
    material_version_id: UUID,
    *,
    idx: int = 0,
    content: str | None = None,
) -> _Chunk:
    return _Chunk(
        id=uuid4(),
        chunk_index=idx,
        content=content or "Binary search is an algorithm that requires a sorted array as input.",
        material_version_id=material_version_id,
    )


def _make_hierarchy() -> _Hierarchy:
    return _Hierarchy(
        organization_id=uuid4(),
        course_id=uuid4(),
        course_title="Algorithms",
        module_id=uuid4(),
        module_title="Searching",
        lesson_id=uuid4(),
        lesson_title="Binary Search",
        material_id=uuid4(),
        material_title="Slides",
        material_type="document",
    )


@pytest.fixture
def _kg_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j-test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_neo4j_client_is_slim() -> None:
    for removed in ("upsert_chunk_graph", "lesson_concepts", "lesson_concept_graph"):
        assert not hasattr(KnowledgeGraphClient, removed), (
            f"KnowledgeGraphClient should no longer expose {removed} -- moved to ai/knowledge_graph"
        )
    assert hasattr(KnowledgeGraphClient, "session")
    assert hasattr(KnowledgeGraphClient, "aclose")


@pytest.mark.asyncio
async def test_build_kg_disabled_returns_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "false")
    get_settings.cache_clear()

    fake_client = _FakeKGClient()
    fake_gateway = _FakeGateway()
    material_version_id = uuid4()
    chunk = _make_chunk(material_version_id)

    summary = await build_knowledge_graph_for_material_version(
        material_version_id,
        [chunk],
        hierarchy=_make_hierarchy(),
        pipeline_run_id=uuid4(),
        db=None,
        kg_client=fake_client,
        llm_gateway=fake_gateway,
    )

    assert summary == KGSummary(concept_count=0, relationship_count=0, enabled=False)
    assert fake_client.upserts == []
    assert fake_gateway.calls == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("_kg_env")
async def test_build_kg_no_chunks_short_circuits() -> None:
    fake_client = _FakeKGClient()
    fake_gateway = _FakeGateway()
    summary = await build_knowledge_graph_for_material_version(
        uuid4(),
        [],
        hierarchy=_make_hierarchy(),
        pipeline_run_id=uuid4(),
        db=None,
        kg_client=fake_client,
        llm_gateway=fake_gateway,
    )
    assert summary == KGSummary(concept_count=0, relationship_count=0, enabled=True)
    assert fake_client.upserts == []
    assert fake_gateway.calls == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("_kg_env")
async def test_build_kg_creates_concepts() -> None:
    material_version_id = uuid4()
    chunk = _make_chunk(material_version_id)
    fake_client = _FakeKGClient()
    fake_gateway = _FakeGateway()

    summary = await build_knowledge_graph_for_material_version(
        material_version_id,
        [chunk],
        hierarchy=_make_hierarchy(),
        pipeline_run_id=uuid4(),
        db=None,
        kg_client=fake_client,
        llm_gateway=fake_gateway,
    )

    assert summary.enabled is True
    assert summary.concept_count == 2
    assert summary.relationship_count == 1
    assert len(fake_client.upserts) == 1

    upsert = fake_client.upserts[0]
    hierarchy_arg, chunk_arg, concepts_arg, relationships_arg = upsert["args"]
    assert hierarchy_arg["course_title"] == "Algorithms"
    assert hierarchy_arg["lesson_title"] == "Binary Search"
    assert hierarchy_arg["material_type"] == "document"
    assert chunk_arg["chunk_id"] == str(chunk.id)
    assert chunk_arg["chunk_index"] == 0
    assert {c["name"] for c in concepts_arg} == {"Binary Search", "Sorted Array"}
    assert relationships_arg[0]["relation"] == "PREREQUISITE_OF"
    assert relationships_arg[0]["source"] == "Sorted Array"
    assert relationships_arg[0]["target"] == "Binary Search"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_kg_env")
async def test_audit_pipeline_run_id() -> None:
    material_version_id = uuid4()
    pipeline_run_id = uuid4()
    parent_job_id = uuid4()
    chunks = [_make_chunk(material_version_id, idx=i) for i in range(3)]
    fake_client = _FakeKGClient()
    fake_gateway = _FakeGateway()

    summary = await build_knowledge_graph_for_material_version(
        material_version_id,
        chunks,
        hierarchy=_make_hierarchy(),
        pipeline_run_id=pipeline_run_id,
        db=None,
        kg_client=fake_client,
        llm_gateway=fake_gateway,
        parent_job_id=parent_job_id,
    )

    assert summary.enabled is True
    assert len(fake_gateway.calls) == 3
    for call in fake_gateway.calls:
        assert call.role is LLMRole.KG_EXTRACTION
        assert call.stage_name == KG_BUILD_STAGE_NAME
        assert call.pipeline_run_id == pipeline_run_id
        assert call.parent_job_id == parent_job_id


@pytest.mark.asyncio
@pytest.mark.usefixtures("_kg_env")
async def test_build_kg_skips_mismatched_material_version() -> None:
    material_version_id = uuid4()
    other_material_version_id = uuid4()
    matching_chunk = _make_chunk(material_version_id, idx=0)
    foreign_chunk = _make_chunk(other_material_version_id, idx=1)

    fake_client = _FakeKGClient()
    fake_gateway = _FakeGateway()
    summary = await build_knowledge_graph_for_material_version(
        material_version_id,
        [matching_chunk, foreign_chunk],
        hierarchy=_make_hierarchy(),
        pipeline_run_id=uuid4(),
        db=None,
        kg_client=fake_client,
        llm_gateway=fake_gateway,
    )

    assert len(fake_gateway.calls) == 1
    assert len(fake_client.upserts) == 1
    assert summary.enabled is True


def test_normalize_kg_payload_drops_dangling_relationships() -> None:
    payload = {
        "entities": [{"name": "Python"}],
        "relationships": [
            {"source": "Python", "target": "Ghost", "relation": "RELATED_TO"},
            {"source": "Python", "target": "Python", "relation": "RELATED_TO"},
        ],
    }
    concepts, relationships = builder_mod._normalize_kg_payload(payload)
    assert {c["name"] for c in concepts} == {"Python"}
    assert len(relationships) == 1
    assert relationships[0]["target"] == "Python"


def test_normalize_kg_payload_handles_invalid_payload() -> None:
    concepts, relationships = builder_mod._normalize_kg_payload({"oops": True})
    assert concepts == []
    assert relationships == []
    concepts, relationships = builder_mod._normalize_kg_payload([])
    assert concepts == []
    assert relationships == []


def test_concept_dataclass_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    concept = Concept(name="Python", type="Concept", definition=None, confidence=None)
    with pytest.raises(FrozenInstanceError):
        concept.name = "Java"
