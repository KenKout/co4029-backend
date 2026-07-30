"""Public re-exports for the materials-feature schema module (T4.2).

Four concerns split across sibling files:

* :mod:`.public`     — student-facing DTOs (no audit / processing
  metadata; only ready materials surface).
* :mod:`.authoring`  — teacher-facing DTOs that inherit from Public
  and widen with audit + soft-delete + version metadata + the
  ``visible_to_students`` / ``ai_processing_enabled`` toggles.
* :mod:`.status`     — :class:`ProcessingProgress` slice shared by
  the polling endpoint (T4.6) and future websocket firehose.
* :mod:`.request`    — request bodies for upload init / complete /
  metadata update flows.
"""

from __future__ import annotations

from abridgeai.features.materials.schemas.authoring import (
    MaterialAuthoring,
    MaterialVersionAuthoring,
)
from abridgeai.features.materials.schemas.curated_kg import (
    CuratedKGDraft,
    CuratedKGDraftSave,
    CuratedKGEdge,
    CuratedKGGraph,
    CuratedKGNode,
    CuratedKGPublished,
    CuratedKGRelation,
)
from abridgeai.features.materials.schemas.public import (
    MaterialPublic,
    MaterialStreamUrl,
    MaterialTypeLiteral,
)
from abridgeai.features.materials.schemas.request import (
    MaterialLinkExisting,
    MaterialUpdate,
    MaterialUploadComplete,
    MaterialUploadInit,
)
from abridgeai.features.materials.schemas.status import (
    KGEdge,
    KGNode,
    LessonKnowledgeGraph,
    LessonProcessingSummary,
    ProcessingProgress,
    ProcessingStatusLiteral,
)

__all__ = [
    "MaterialAuthoring",
    "MaterialLinkExisting",
    "MaterialPublic",
    "MaterialStreamUrl",
    "MaterialTypeLiteral",
    "MaterialUpdate",
    "MaterialUploadComplete",
    "MaterialUploadInit",
    "MaterialVersionAuthoring",
    "KGEdge",
    "KGNode",
    "LessonKnowledgeGraph",
    "LessonProcessingSummary",
    "ProcessingProgress",
    "ProcessingStatusLiteral",
    "CuratedKGDraft",
    "CuratedKGDraftSave",
    "CuratedKGEdge",
    "CuratedKGGraph",
    "CuratedKGNode",
    "CuratedKGPublished",
    "CuratedKGRelation",
]
