"""Materials feature public re-exports.

Per plan T4.1 / Reconciliation §B9 §C10-§C12 §C15: 4 ORM models cover
the upload + ingestion + retrieval lifecycle. ``ProcessingJob`` and
``GenerationRun`` live in :mod:`abridgeai.ai.models` (cross-feature
plumbing, not materials-owned).
"""

from abridgeai.ai.models import ProcessingJob
from abridgeai.features.materials.models import (
    ChunkingEnrichmentCache,
    DocumentChunk,
    LearningMaterial,
    LearningMaterialVersion,
)

__all__ = [
    "ChunkingEnrichmentCache",
    "DocumentChunk",
    "LearningMaterial",
    "LearningMaterialVersion",
    "ProcessingJob",
]
