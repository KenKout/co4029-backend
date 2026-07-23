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


class LessonProcessingSummary(BaseModel):
    """Aggregate processing-status counts for every material under a lesson.

    Used by the teacher's lesson-manage page to render a single roll-up
    badge ("3 of 5 processed") instead of polling every material's
    progress separately.
    """

    lesson_id: UUID
    materials_total: int
    versions_total: int
    pending_versions: int
    processing_versions: int
    completed_versions: int
    failed_versions: int


class KGNode(BaseModel):
    """A single concept node in the lesson knowledge-graph preview.

    ``id`` is the normalized concept name (stable key for edges); ``label``
    is the human display name. ``weight`` is the mention count across the
    lesson's chunks — the UI sizes the node by it.
    """

    id: str
    label: str
    type: str = "Concept"
    definition: str | None = None
    weight: int = 1


class KGEdge(BaseModel):
    """A directed relationship between two concept nodes in the preview."""

    source: str
    target: str
    relation: Literal["PREREQUISITE_OF", "RELATED_TO"] = "RELATED_TO"


class LessonKnowledgeGraph(BaseModel):
    """Bounded concept graph for one lesson, for the teacher AI-Hub viz.

    ``enabled`` is False when the KG feature is off (UI shows a disabled
    hint rather than an empty graph). ``total_concepts`` is the full
    lesson concept count so the UI can say "showing top 24 of 830".
    """

    lesson_id: UUID
    enabled: bool
    nodes: list[KGNode] = []
    edges: list[KGEdge] = []
    total_concepts: int = 0


__all__ = [
    "KGEdge",
    "KGNode",
    "LessonKnowledgeGraph",
    "LessonProcessingSummary",
    "ProcessingProgress",
    "ProcessingStatusLiteral",
]
