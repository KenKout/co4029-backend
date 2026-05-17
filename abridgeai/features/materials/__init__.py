"""Materials feature public re-exports.

Per plan T4.1 / Reconciliation §B9 §C10-§C12 §C15: 5 ORM models cover
the upload + ingestion + retrieval lifecycle. ``GenerationRun`` is
intentionally absent — it ports to ``features/ai/models.py`` in T5.x.
"""

from abridgeai.features.materials.models import (
    ChunkingEnrichmentCache,
    DocumentChunk,
    LearningMaterial,
    LearningMaterialVersion,
    ProcessingJob,
)

__all__ = [
    "ChunkingEnrichmentCache",
    "DocumentChunk",
    "LearningMaterial",
    "LearningMaterialVersion",
    "ProcessingJob",
]
