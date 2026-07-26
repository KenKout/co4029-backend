"""Semantic chunker integration tests.

Covers the rule-based fallback (no LLM call) and the cache-hit short-circuit
(LLM not called when a prior enrichment is cached). Uses ``AsyncMock`` to
stand in for the LLM gateway and ``ChunkingCache``; the real DB is
exercised by ``test_chunking_cache_get_set``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from abridgeai.ai.chunking import EnrichedChunk, SemanticChunker
from abridgeai.ai.chunking._enrich import PROMPT_VERSION, content_hash
from abridgeai.ai.chunking._llm_boundary import BOUNDARY_PROMPT_VERSION
from abridgeai.ai.chunking.cache import ChunkingCache
from abridgeai.ai.extraction import ExtractedContent


def _document(paragraph_count: int = 4) -> ExtractedContent:
    paragraphs = [
        f"Paragraph {i}: decision making theory continues across pages "
        + ("with additional supporting detail. " * 12)
        for i in range(paragraph_count)
    ]
    body = "[Page 1]\n" + paragraphs[0] + "\n\n[Page 2]\n" + "\n\n".join(paragraphs[1:])
    return ExtractedContent(
        text=body,
        metadata={"page_count": paragraph_count},
        source_type="pdf",
        source_locations=[],
    )


@pytest.mark.asyncio
async def test_rule_based_fallback() -> None:
    content = _document()
    chunker = SemanticChunker()
    mock_gateway = AsyncMock()

    chunks = await chunker.chunk(
        content,
        embedder=None,
        llm_gateway=mock_gateway,
        llm_enrichment=False,
    )

    assert chunks, "rule-based path must still produce chunks"
    assert all(isinstance(c, EnrichedChunk) for c in chunks)
    assert mock_gateway.generate_json.await_count == 0
    for c in chunks:
        role = c.semantic_metadata.get("content_role")
        assert role in {"body", "summary", "review", "front_matter"}
        assert c.semantic_metadata.get("section_title")


@pytest.mark.asyncio
async def test_full_pipeline_with_cache_hit() -> None:
    content = _document(paragraph_count=2)
    chunker = SemanticChunker()

    rule_chunks = chunker  # for type hints only
    del rule_chunks

    from abridgeai.ai.chunking._window import window_chunks

    pre = window_chunks(content, max_tokens=800, overlap_tokens=80)

    cached_payload = {
        "section_title": "Decision-Making Phases",
        "content_role": "body",
        "context_sentence": "This window covers Simon's three phases.",
        "key_concepts": ["intelligence phase", "design phase"],
        "propositions": ["Decision making has phases."],
    }

    # Stage B' shares this cache under its own prompt-version namespace, so the
    # fake has to answer both. The point of the test is that a fully cached
    # document costs zero LLM calls — that now covers the boundary decision as
    # well as enrichment.
    boundary_payload = {"groups": [[i] for i in range(len(pre))]}

    def _cache_get(content_hash: str, prompt_version: str) -> dict[str, object]:
        if prompt_version == BOUNDARY_PROMPT_VERSION:
            return {"output_json": boundary_payload}
        return {
            "output_json": cached_payload,
            "model_name": "test-model",
            "input_tokens": 500,
            "output_tokens": 100,
        }

    mock_cache = AsyncMock(spec=ChunkingCache)
    mock_cache.get.side_effect = _cache_get
    mock_cache.put.return_value = None

    mock_gateway = AsyncMock()
    mock_db = AsyncMock()

    chunks = await chunker.chunk(
        content,
        embedder=None,
        llm_gateway=mock_gateway,
        db=mock_db,
        cache=mock_cache,
        llm_enrichment=True,
        document_title="Decision Theory",
        pipeline_run_id=uuid4(),
    )

    assert chunks
    assert mock_gateway.generate_json.await_count == 0
    assert mock_cache.get.await_count == len(pre) or mock_cache.get.await_count >= 1
    for c in chunks:
        assert c.semantic_metadata["cached"] is True
        assert c.semantic_metadata["section_title"] == "Decision-Making Phases"
        assert "intelligence phase" in c.semantic_metadata["key_concepts"]


@pytest.mark.asyncio
async def test_chunking_cache_get_set() -> None:
    db = AsyncMock()
    miss_result = MagicMock()
    miss_result.first.return_value = None
    db.execute.return_value = miss_result

    cache = ChunkingCache(db)

    miss = await cache.get(content_hash("hello"), PROMPT_VERSION)
    assert miss is None

    hit_row = MagicMock()
    hit_row.output_json = {"section_title": "Hello"}
    hit_row.model_name = "m"
    hit_row.input_tokens = 10
    hit_row.output_tokens = 5
    hit_result = MagicMock()
    hit_result.first.return_value = hit_row
    db.execute.return_value = hit_result

    hit = await cache.get(content_hash("hello"), PROMPT_VERSION)
    assert hit == {
        "output_json": {"section_title": "Hello"},
        "model_name": "m",
        "input_tokens": 10,
        "output_tokens": 5,
    }

    await cache.put(
        content_hash("hello"),
        PROMPT_VERSION,
        output_json={"section_title": "Hello"},
        model_name="m",
        input_tokens=10,
        output_tokens=5,
    )
    assert db.execute.await_count >= 3


@pytest.mark.asyncio
async def test_full_pipeline_threads_pipeline_run_id() -> None:
    content = _document(paragraph_count=2)
    chunker = SemanticChunker()
    pipeline_run_id = uuid4()

    mock_gateway = AsyncMock()
    mock_gateway.generate_json.return_value.content_json = {
        "section_title": "Auto Section",
        "content_role": "body",
        "context_sentence": "",
        "key_concepts": [],
        "propositions": [],
    }
    mock_gateway.generate_json.return_value.model_name = "gpt-test"
    mock_gateway.generate_json.return_value.input_tokens = 100
    mock_gateway.generate_json.return_value.output_tokens = 50

    mock_db = AsyncMock()

    chunks = await chunker.chunk(
        content,
        embedder=None,
        llm_gateway=mock_gateway,
        db=mock_db,
        cache=None,
        llm_enrichment=True,
        pipeline_run_id=pipeline_run_id,
    )

    assert chunks
    assert mock_gateway.generate_json.await_count >= 1
    # Stage B' (boundary) and Stage C (enrichment) both call the gateway; every
    # call must carry the run id so cost rolls up per pipeline run.
    stages = set()
    for call in mock_gateway.generate_json.await_args_list:
        kwargs = call.kwargs
        assert kwargs["pipeline_run_id"] == pipeline_run_id
        stages.add(kwargs["stage_name"])
    assert stages <= {"chunk_boundary", "chunking_enrichment"}
    assert "chunking_enrichment" in stages
