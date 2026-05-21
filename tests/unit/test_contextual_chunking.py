"""Tests for ``abridgeai.ai.chunking.contextual``.

The contextual prepend helper is the linchpin of Phase 2 of the
contextual retrieval upgrade: it produces the embedder input, which
is where Anthropic's contextual retrieval technique earns its -35%
retrieval failure improvement. These tests pin down the contract:

  - Stage C output flows into the prefix
  - Both ``EnrichedChunk.semantic_metadata`` and ``metadata['semantic']``
    are accepted (covers freshly-produced and rehydrated chunks)
  - Missing enrichment falls back to ``content`` unchanged
  - Prefix budget is enforced (max_prefix_tokens cap)
"""

from __future__ import annotations

import pytest

from abridgeai.ai.chunking.base import EnrichedChunk, RawChunk
from abridgeai.ai.chunking.contextual import build_contextual_text
from abridgeai.ai.chunking.token_aware import count_tokens


def test_contextual_prepend_uses_enriched_chunk_metadata() -> None:
    chunk = EnrichedChunk(
        chunk_index=0,
        content="The system handles 1M requests per second.",
        metadata={"page": 7},
        embedding=None,
        semantic_metadata={
            "section_title": "Performance Architecture",
            "context_sentence": (
                "This section describes the load characteristics of the "
                "production deployment."
            ),
        },
    )
    result = build_contextual_text(chunk)
    assert result.startswith("[Topic: Performance Architecture] ")
    assert "load characteristics" in result
    assert "1M requests per second" in result


def test_contextual_prepend_reads_nested_metadata_for_rehydrated_chunk() -> None:
    """Chunks rebuilt from ``DocumentChunk.metadata_json`` arrive as
    ``RawChunk`` with semantic data flattened under
    ``metadata['semantic']``. The helper must read from that path too.
    """
    chunk = RawChunk(
        chunk_index=0,
        content="Embeddings live in pgvector.",
        metadata={
            "page": 3,
            "semantic": {
                "section_title": "Storage Layer",
                "context_sentence": "The vector index is partitioned by tenant.",
            },
        },
    )
    result = build_contextual_text(chunk)
    assert "[Topic: Storage Layer]" in result
    assert "vector index is partitioned" in result
    assert result.endswith("Embeddings live in pgvector.")


def test_contextual_prepend_returns_content_when_no_enrichment() -> None:
    chunk = RawChunk(chunk_index=0, content="bare content", metadata={})
    assert build_contextual_text(chunk) == "bare content"


def test_contextual_prepend_returns_content_when_metadata_semantic_blank() -> None:
    """Empty strings in semantic metadata count as missing — don't emit a
    naked ``[Topic: ]`` tag."""
    chunk = RawChunk(
        chunk_index=0,
        content="bare content",
        metadata={"semantic": {"section_title": "", "context_sentence": "  "}},
    )
    assert build_contextual_text(chunk) == "bare content"


def test_contextual_prepend_uses_title_only_when_context_missing() -> None:
    chunk = RawChunk(
        chunk_index=0,
        content="content",
        metadata={"semantic": {"section_title": "Heading"}},
    )
    result = build_contextual_text(chunk)
    assert result == "[Topic: Heading] content"


def test_contextual_prepend_uses_context_only_when_title_missing() -> None:
    chunk = RawChunk(
        chunk_index=0,
        content="content",
        metadata={"semantic": {"context_sentence": "Helper context."}},
    )
    result = build_contextual_text(chunk)
    assert result == "Helper context. content"


def test_contextual_prepend_truncates_oversized_prefix() -> None:
    """A verbose context sentence must be truncated so it can't crowd out
    chunk content in the embedder's input window."""
    long_ctx = "Lorem ipsum dolor sit amet. " * 200  # roughly 1200 tokens
    chunk = RawChunk(
        chunk_index=0,
        content="real content",
        metadata={
            "semantic": {
                "section_title": "Topic",
                "context_sentence": long_ctx,
            },
        },
    )
    result = build_contextual_text(chunk, max_prefix_tokens=50)
    # Body content still present
    assert result.endswith("real content")
    # Prefix portion fits the budget — measure the portion before content
    prefix_part = result[: result.rfind("real content")].rstrip()
    assert count_tokens(prefix_part) <= 50


def test_enriched_chunk_metadata_takes_precedence_over_nested() -> None:
    """When both sources are present, ``semantic_metadata`` (typed) wins
    over ``metadata['semantic']`` (rehydrated). Prevents drift if one
    side gets stale during a partial reprocess."""
    chunk = EnrichedChunk(
        chunk_index=0,
        content="content",
        metadata={
            "semantic": {
                "section_title": "Stale Title",
                "context_sentence": "Stale context.",
            },
        },
        embedding=None,
        semantic_metadata={
            "section_title": "Fresh Title",
            "context_sentence": "Fresh context.",
        },
    )
    result = build_contextual_text(chunk)
    assert "Fresh Title" in result
    assert "Fresh context." in result
    assert "Stale Title" not in result


@pytest.mark.parametrize(
    "metadata",
    [
        {"semantic": None},
        {"semantic": "not-a-dict"},
        {"semantic": []},
    ],
)
def test_contextual_prepend_handles_malformed_semantic(metadata: dict[str, object]) -> None:
    """Defensive: stale or test-fixture chunks may have a non-dict
    ``semantic`` entry. The helper falls back to ``content`` rather than
    crashing."""
    chunk = RawChunk(chunk_index=0, content="content", metadata=metadata)
    assert build_contextual_text(chunk) == "content"
