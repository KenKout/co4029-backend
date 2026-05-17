"""Token-aware chunker tests."""

from __future__ import annotations

import pytest

from abridgeai.ai.chunking import RawChunk, TokenAwareChunker
from abridgeai.ai.chunking.token_aware import count_tokens
from abridgeai.ai.extraction import ExtractedContent


def _make(text: str, source_type: str = "pdf") -> ExtractedContent:
    return ExtractedContent(
        text=text,
        metadata={},
        source_type=source_type,
        source_locations=[],
    )


def test_token_chunker_respects_budget() -> None:
    body = ("This is a sentence about decision making theory. " * 800).strip()
    content = _make(body, "text")

    chunker = TokenAwareChunker(max_tokens=500, overlap_tokens=50)
    chunks = chunker.chunk(content)

    assert chunks, "expected at least one chunk for 10000-char input"
    assert all(isinstance(c, RawChunk) for c in chunks)
    for c in chunks:
        assert count_tokens(c.content) <= 500, (
            f"chunk {c.chunk_index} exceeds 500 tokens: {count_tokens(c.content)}"
        )


def test_token_chunker_overlap() -> None:
    paragraphs = [
        "Paragraph " + str(i) + " " + ("alpha beta gamma delta epsilon zeta eta theta " * 30)
        for i in range(6)
    ]
    body = "\n\n".join(paragraphs)
    content = _make(body, "text")

    chunker = TokenAwareChunker(max_tokens=300, overlap_tokens=60)
    chunks = chunker.chunk(content)
    assert len(chunks) >= 2, "expected multiple chunks for overlap test"

    first_tail_words = chunks[0].content.split()[-30:]
    second_head_words = chunks[1].content.split()[:80]
    overlap = [w for w in first_tail_words if w in second_head_words]
    assert overlap, "adjacent chunks should share overlap tokens"


def test_token_chunker_section_markers() -> None:
    body = "[Page 1]\nIntroduction to learners.\n\n[Page 2]\nSecond page details."
    content = _make(body, "pdf")

    chunks = TokenAwareChunker(max_tokens=500, overlap_tokens=10).chunk(content)
    assert len(chunks) == 2
    assert any(c.metadata.get("page") == 1 for c in chunks)
    assert any(c.metadata.get("page") == 2 for c in chunks)


def test_token_chunker_empty_input() -> None:
    chunks = TokenAwareChunker().chunk(_make(""))
    assert chunks == []


def test_token_chunker_validates_constructor() -> None:
    with pytest.raises(ValueError):
        TokenAwareChunker(max_tokens=0)
    with pytest.raises(ValueError):
        TokenAwareChunker(max_tokens=100, overlap_tokens=100)
