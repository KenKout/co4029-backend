"""Chunking dataclasses + Protocol.

Three primitives shape every downstream consumer:

* ``RawChunk`` — output of Stage A (rule-based windowing) and of cheap
  chunkers like ``TokenAwareChunker`` / ``TimestampAwareChunker``. Carries
  text + position + a free-form metadata bag (page, paragraph, timestamp
  range, etc.).
* ``EnrichedChunk`` — output of Stage C (LLM enrichment) and of
  ``SemanticChunker``. Adds an embedding and a structured semantic
  metadata dict (heading, summary, key terms, glue group id).
* ``Chunker`` — structural Protocol every chunker satisfies. Takes an
  ``ExtractedContent`` from ``ai.extraction`` and returns ``list[RawChunk]``.

The Protocol is ``runtime_checkable`` so tests can assert conformance via
``isinstance``; production dispatch is by source-type / file-format match,
not isinstance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from abridgeai.ai.extraction import ExtractedContent


@dataclass(frozen=True)
class RawChunk:
    """Single chunk of text with positional metadata.

    ``metadata`` is intentionally untyped so each chunker can attach what
    makes sense for its source (``page``, ``paragraph``, ``line_start``,
    ``timestamp_start_ms``, etc). Downstream code keys off the presence of
    a field, never its absence.
    """

    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnrichedChunk(RawChunk):
    """RawChunk plus an embedding and Stage-C semantic metadata.

    ``embedding`` is ``None`` when no embedder was supplied (rule-based-only
    pipeline). ``semantic_metadata`` carries window-level attributes the
    LLM extracted: section title, role hint, context sentence, key
    concepts, propositions, glue group id, etc. The structure mirrors what
    the legacy 5-stage pipeline persisted under ``section_*`` keys but is
    no longer flattened into ``metadata``.
    """

    embedding: list[float] | None = None
    semantic_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    """Structural contract every chunker satisfies.

    Implementations may be ``async def chunk(...)`` (semantic / timestamp
    chunkers do I/O) or ``def chunk(...)`` (token chunker is pure CPU).
    The Protocol is permissive on awaitability — callers branch on whether
    the result is a coroutine.
    """

    def chunk(self, content: ExtractedContent, **opts: Any) -> list[RawChunk]: ...  # noqa: ANN401 -- forwarded chunker kwargs


__all__ = ["Chunker", "EnrichedChunk", "RawChunk"]
