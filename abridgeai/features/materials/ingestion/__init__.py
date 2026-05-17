"""Materials ingestion pipeline (T4.4).

Public re-export of :func:`run_material_ingest` — the orchestrator that
composes Phase 2 primitives (extraction, chunking, embedding, KG) into a
five-stage ingest for one ``LearningMaterialVersion``.

The pipeline is S3-decoupled: workers (T4.7) hand it a local file path so
the worker owns S3 download lifecycle. CLI / backfill callers may omit
``source_path`` to have the pipeline fetch the StorageObject itself.
"""

from abridgeai.features.materials.ingestion.pipeline import run_material_ingest

__all__ = ["run_material_ingest"]
