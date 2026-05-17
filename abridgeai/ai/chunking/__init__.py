"""Public API for the chunking package.

Three concrete chunkers compose with the cache:

* ``TokenAwareChunker`` — pure-CPU, tiktoken-budget split. Cheap; used for
  text materials and as the Stage-A engine of ``SemanticChunker``.
* ``SemanticChunker`` — three-stage orchestrator: rule-based windowing →
  embedding glue → LLM enrichment. Returns ``list[EnrichedChunk]``.
* ``TimestampAwareChunker`` — splits audio/video transcripts on silence
  gaps so a chunk never spans more than ``silence_gap_ms`` of dead air.

``ChunkingCache`` wraps the ``chunking_enrichment_cache`` table and is
keyed by ``(content_hash, prompt_version)``. Without it every re-ingest
re-pays for the LLM enrichment.
"""

from abridgeai.ai.chunking.base import Chunker, EnrichedChunk, RawChunk
from abridgeai.ai.chunking.cache import ChunkingCache
from abridgeai.ai.chunking.semantic import SemanticChunker
from abridgeai.ai.chunking.timestamp_aware import TimestampAwareChunker
from abridgeai.ai.chunking.token_aware import TokenAwareChunker

__all__ = [
    "Chunker",
    "ChunkingCache",
    "EnrichedChunk",
    "RawChunk",
    "SemanticChunker",
    "TimestampAwareChunker",
    "TokenAwareChunker",
]
