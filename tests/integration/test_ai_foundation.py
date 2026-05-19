"""AI foundation end-to-end smoke (T2.10).

Composes every Phase 2 primitive in the order a Phase 4-5 ingestion /
quiz pipeline will: extract -> chunk -> embed -> store -> retrieve ->
diversify. Mocks only what crosses an external boundary (LLM HTTP,
embedder API, Neo4j); the chunker, vector_search, MMR, and audit writer
all run real. This file is the Phase 2 exit gate.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from abridgeai.ai.chunking import TokenAwareChunker
from abridgeai.ai.extraction import EXTRACTOR_REGISTRY, ExtractedContent, dispatch_extractor
from abridgeai.ai.knowledge_graph import KGContext, retrieve_kg_context_for_anchors
from abridgeai.ai.llm.gateway import LLMGateway
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.ai.retrieval import mmr_diversify, vector_search
from abridgeai.core.config import get_settings

_EMBEDDING_DIM = 3072


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _vec_literal(values: list[float]) -> str:
    padded = list(values) + [0.0] * (_EMBEDDING_DIM - len(values))
    return "[" + ",".join(repr(float(v)) for v in padded[:_EMBEDDING_DIM]) + "]"


def _vec(values: list[float]) -> list[float]:
    padded = list(values) + [0.0] * (_EMBEDDING_DIM - len(values))
    return [float(v) for v in padded[:_EMBEDDING_DIM]]


def _fake_embed_one(content: str) -> list[float]:
    h = abs(hash(content))
    a = (h % 100) / 100.0
    b = ((h >> 7) % 100) / 100.0
    c = ((h >> 13) % 100) / 100.0
    return _vec([a, b, c, 0.1])


def _make_synthetic_pdf_bytes() -> bytes:
    import fitz  # type: ignore[import-untyped,unused-ignore]

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "The quick brown fox jumps over the lazy dog. " * 20)
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20,
    )
    raw: bytes = doc.tobytes()
    doc.close()
    return raw


def _stub_chat_completion_body(content_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-small",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(content_payload)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


async def _reflect_audit_parent_tables(engine: AsyncEngine) -> None:
    from abridgeai.core.db import Base

    needed = {"generation_runs", "processing_jobs"}
    if needed.issubset(Base.metadata.tables):
        return
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.reflect(
                bind=sync_conn, only=tuple(needed - set(Base.metadata.tables))
            )
        )


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    await _reflect_audit_parent_tables(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def material_scope(engine: AsyncEngine):
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    storage_obj_id = uuid.uuid4()
    material_id = uuid.uuid4()
    version_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {
                "id": org_id,
                "slug": f"ai-smoke-{org_id.hex[:8]}",
                "name": "AI Foundation Smoke Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"ai-smoke-{user_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": user_id,
                "slug": f"course-{course_id.hex[:8]}",
                "title": "Smoke Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :course, :title, :pos)"
            ),
            {"id": module_id, "course": course_id, "title": "M", "pos": 1},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title) "
                "VALUES (:id, :module, :slug, :title)"
            ),
            {
                "id": lesson_id,
                "module": module_id,
                "slug": f"l-{lesson_id.hex[:6]}",
                "title": "Smoke Lesson",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, :bucket, :key, :mime)"
            ),
            {
                "id": storage_obj_id,
                "bucket": "test",
                "key": f"smoke/{storage_obj_id.hex}",
                "mime": "application/pdf",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :lesson, :title, 'pdf')"
            ),
            {"id": material_id, "lesson": lesson_id, "title": "Smoke Material"},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions "
                "(id, material_id, storage_object_id, version_no, processing_status) "
                "VALUES (:id, :mat, :obj, 1, 'ready')"
            ),
            {"id": version_id, "mat": material_id, "obj": storage_obj_id},
        )

    yield {
        "org_id": org_id,
        "user_id": user_id,
        "course_id": course_id,
        "module_id": module_id,
        "lesson_id": lesson_id,
        "storage_obj_id": storage_obj_id,
        "material_id": material_id,
        "version_id": version_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM document_chunks WHERE material_version_id = :v"),
            {"v": version_id},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = :id"),
            {"id": version_id},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :id"),
            {"id": material_id},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": storage_obj_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


@pytest_asyncio.fixture
async def audit_scope(engine: AsyncEngine):
    user_id = uuid.uuid4()
    run_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"audit-{user_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO generation_runs "
                "(id, generation_type, source_scope_kind, status, requested_by) "
                "VALUES (:id, 'quiz', 'lesson', 'running', :uid)"
            ),
            {"id": run_id, "uid": user_id},
        )

    yield {"user_id": user_id, "generation_run_id": run_id}

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM ai_model_calls WHERE generation_run_id = :id"),
            {"id": run_id},
        )
        await conn.execute(
            text("DELETE FROM generation_runs WHERE id = :id"),
            {"id": run_id},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})


@pytest.fixture
def llm_test_settings(monkeypatch: pytest.MonkeyPatch):
    """Fake LLM_API_KEY + respx-mockable base URL.

    Plan §3823 forbids hard-coding real API keys; we use a placeholder
    plus respx so the gateway path runs deterministically without ever
    hitting a real provider.
    """
    monkeypatch.setenv("LLM_API_KEY", "sk-test-foundation")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.foundation-test.local/v1")
    monkeypatch.setenv("LLM_MODEL_SMALL", "test-small")
    monkeypatch.setenv("LLM_MODEL_STANDARD", "test-standard")
    monkeypatch.setenv("LLM_MODEL_LARGE", "test-large")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "5")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


async def test_full_pipeline_extract_chunk_embed_store_retrieve_diversify(
    engine: AsyncEngine,
    material_scope: dict[str, uuid.UUID],
) -> None:
    pdf_bytes = _make_synthetic_pdf_bytes()

    extractor = dispatch_extractor("application/pdf")
    extracted = await extractor.extract(pdf_bytes)
    assert isinstance(extracted, ExtractedContent)
    assert extracted.source_type == "pdf"
    assert extracted.text.strip()

    raw_chunks = TokenAwareChunker(max_tokens=80, overlap_tokens=10).chunk(extracted)
    assert len(raw_chunks) >= 2, f"Expected >=2 chunks from synthetic PDF, got {len(raw_chunks)}"

    embeddings = [_fake_embed_one(c.content) for c in raw_chunks]
    assert all(len(e) == _EMBEDDING_DIM for e in embeddings)

    chunk_ids: list[uuid.UUID] = []
    async with engine.begin() as conn:
        for idx, (raw, emb) in enumerate(zip(raw_chunks, embeddings, strict=True)):
            chunk_id = uuid.uuid4()
            await conn.execute(
                text(
                    "INSERT INTO document_chunks "
                    "(id, course_id, module_id, lesson_id, material_version_id, "
                    " chunk_index, chunk_type, content, content_hash, embedding) "
                    "VALUES (:id, :course, :module, :lesson, :ver, :idx, 'pdf', "
                    "        :content, :hash, CAST(:emb AS halfvec))"
                ),
                {
                    "id": chunk_id,
                    "course": material_scope["course_id"],
                    "module": material_scope["module_id"],
                    "lesson": material_scope["lesson_id"],
                    "ver": material_scope["version_id"],
                    "idx": idx,
                    "content": raw.content,
                    "hash": chunk_id.hex,
                    "emb": _vec_literal(emb),
                },
            )
            chunk_ids.append(chunk_id)

    target_idx = 0
    query_emb = embeddings[target_idx]

    async with AsyncSession(engine) as session:
        retrieved = await vector_search(
            session,
            query_emb,
            course_id=material_scope["course_id"],
            top_k=len(chunk_ids),
            include_embeddings=True,
        )
    assert len(retrieved) == len(chunk_ids)
    assert retrieved[0].chunk_id == chunk_ids[target_idx]
    assert retrieved[0].embedding is not None
    assert len(retrieved[0].embedding) == _EMBEDDING_DIM

    diversified = mmr_diversify(retrieved, top_k=2, lambda_diversity=0.5)
    assert len(diversified) == 2
    assert diversified[0].chunk_id == chunk_ids[target_idx]


async def test_audit_pipeline_run_id_threaded_through_llm_call(
    engine: AsyncEngine,
    audit_scope: dict[str, uuid.UUID],
    llm_test_settings: Any,
) -> None:
    pipeline_run_id = uuid.uuid4()
    parent_run_id = audit_scope["generation_run_id"]

    gateway = LLMGateway(llm_test_settings)

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(
            "https://api.foundation-test.local/v1/chat/completions"
        ).return_value = httpx.Response(200, json=_stub_chat_completion_body({"answer": "ok"}))

        async with AsyncSession(engine) as session:
            result = await gateway.generate_json(
                role=LLMRole.EXTRACTION,
                system_prompt="You are a helpful extractor.",
                user_prompt='Return {"answer": "ok"}.',
                db=session,
                stage_name="extraction",
                pipeline_run_id=pipeline_run_id,
                parent_run_id=parent_run_id,
            )
            await session.commit()

    assert result.role is LLMRole.EXTRACTION
    assert result.stage_name == "extraction"
    assert result.pipeline_run_id == pipeline_run_id
    assert result.content_json == {"answer": "ok"}

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT pipeline_run_id, stage_name, role, status "
                    "FROM ai_model_calls WHERE pipeline_run_id = :rid"
                ),
                {"rid": pipeline_run_id},
            )
        ).all()

    assert len(rows) == 1
    row = rows[0]
    assert row.pipeline_run_id == pipeline_run_id
    assert row.stage_name == "extraction"
    assert row.role == "extraction"
    assert row.status == "success"


async def test_kg_extraction_role_audited_with_kg_build_stage(
    engine: AsyncEngine,
    audit_scope: dict[str, uuid.UUID],
    llm_test_settings: Any,
) -> None:
    from abridgeai.ai.knowledge_graph import KG_BUILD_STAGE_NAME

    pipeline_run_id = uuid.uuid4()
    parent_run_id = audit_scope["generation_run_id"]
    gateway = LLMGateway(llm_test_settings)

    payload = {
        "entities": [{"name": "Binary Search", "type": "Concept", "confidence": 0.9}],
        "relationships": [],
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(
            "https://api.foundation-test.local/v1/chat/completions"
        ).return_value = httpx.Response(200, json=_stub_chat_completion_body(payload))

        async with AsyncSession(engine) as session:
            await gateway.generate_json(
                role=LLMRole.KG_EXTRACTION,
                system_prompt="Extract concepts.",
                user_prompt="Binary search needs a sorted array.",
                db=session,
                stage_name=KG_BUILD_STAGE_NAME,
                pipeline_run_id=pipeline_run_id,
                parent_run_id=parent_run_id,
            )
            await session.commit()

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT role, stage_name, status FROM ai_model_calls "
                    "WHERE pipeline_run_id = :rid"
                ),
                {"rid": pipeline_run_id},
            )
        ).all()

    assert len(rows) == 1
    assert rows[0].role == "kg_extraction"
    assert rows[0].stage_name == KG_BUILD_STAGE_NAME == "kg_build"
    assert rows[0].status == "success"


class _StubResult:
    def __init__(self, record: dict[str, Any] | None) -> None:
        self._record = record

    async def single(self) -> dict[str, Any] | None:
        return self._record


class _StubSession:
    def __init__(self, response: dict[str, Any] | None) -> None:
        self._response = response
        self.last_query: str | None = None
        self.last_params: dict[str, Any] | None = None

    async def __aenter__(self) -> _StubSession:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def run(self, query: str, **params: Any) -> _StubResult:
        self.last_query = query
        self.last_params = params
        return _StubResult(self._response)


class _StubKGClient:
    def __init__(self, response: dict[str, Any] | None) -> None:
        self._session = _StubSession(response)

    def session(self) -> _StubSession:
        return self._session

    async def aclose(self) -> None:
        return None


async def test_retrieve_kg_context_for_anchors_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    get_settings.cache_clear()
    try:
        client = _StubKGClient(
            {
                "nodes": [
                    {
                        "id": "binary search",
                        "label": "Binary Search",
                        "type": "Concept",
                        "definition": None,
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
                        "confidence": 0.9,
                    }
                ],
            }
        )

        ctx = await retrieve_kg_context_for_anchors(["Binary Search"], depth=2, client=client)

        assert isinstance(ctx, KGContext)
        assert ctx.enabled is True
        assert {c.name for c in ctx.concepts} == {"Binary Search", "Sorted Array"}
        assert len(ctx.prerequisites) == 1
        assert ctx.related == []
        assert client.session().last_params == {"names": ["binary search"]}
    finally:
        get_settings.cache_clear()


def test_no_god_files_under_abridgeai_ai() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ai_dir = repo_root / "abridgeai" / "ai"
    assert ai_dir.is_dir(), f"expected {ai_dir} to exist"

    too_big: list[tuple[str, int]] = []
    for path in sorted(ai_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        loc = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        if loc > 500:
            too_big.append((str(path.relative_to(repo_root)), loc))

    assert too_big == [], f"Files exceeding 500 LOC under abridgeai/ai/: {too_big}"


def test_extraction_registry_has_expected_mimes() -> None:
    expected = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain",
        "text/markdown",
        "text/html",
        "text/x-python",
        "audio/mpeg",
        "video/mp4",
        "image/png",
    }
    missing = expected - set(EXTRACTOR_REGISTRY)
    assert missing == set(), f"Missing extractor registrations for: {missing}"
    assert len(EXTRACTOR_REGISTRY) >= 10
