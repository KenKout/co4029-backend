"""Processing-progress DTO for the materials feature (T4.2).

Used by the status-polling endpoint (T4.6) and the future websocket
firehose. Decoupled from :mod:`.public` and :mod:`.authoring` because
the status payload is shared by both audiences (teachers see live
progress while uploading; learners may see "still processing" hints
on materials they have early access to). The DTO carries no audit /
soft-delete columns — it is a transient projection over
``LearningMaterialVersion`` + the active ``ProcessingJob`` row.

Reconciliation §C10 — the ``processing_status`` Literal mirrors the
9-state CHECK constraint from the baseline DDL byte-for-byte.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ProcessingStatusLiteral = Literal[
    "pending",
    "extracting",
    "chunking",
    "embedding",
    "enriching",
    "building_kg",
    "ready",
    "failed",
    "cancelled",
]


class ProcessingProgress(BaseModel):
    """Live-progress slice for a single material version.

    Composed by the service layer (T4.5) from the version row plus the
    most-recent ``ProcessingJob`` for that version. ``progress_percent``
    is bound to ``[0, 100]`` to match the
    ``processing_jobs_progress_percent_check`` DB constraint.
    """

    model_config = ConfigDict(from_attributes=True)

    material_id: UUID
    version_id: UUID
    processing_status: ProcessingStatusLiteral
    progress_percent: int = Field(ge=0, le=100)
    latest_log_line: str | None = None
    error_message: str | None = None


__all__ = [
    "ProcessingProgress",
    "ProcessingStatusLiteral",
]
