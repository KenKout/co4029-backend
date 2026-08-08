"""Teacher-curated, publishable knowledge-graph DTOs.

Distinct from the AI-generated preview schema in :mod:`.status`
(``KGNode`` / ``KGEdge`` / ``LessonKnowledgeGraph``), which projects the
read-only Neo4j concept graph. These DTOs back the teacher CRUD + publish
surface persisted in the ``lesson_knowledge_graphs`` Postgres table.

The hard product rule — exactly ONE primary (centre) node — is enforced in
:class:`CuratedKGDraftSave` at the API boundary so a malformed draft can never
be saved. The service layer keeps ``primary_node_id`` as a top-level column in
sync with the validated payload.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

CuratedKGRelation = Literal["PREREQUISITE_OF", "RELATED_TO"]


class CuratedKGNode(BaseModel):
    """One teacher-authored concept node.

    ``id`` is a stable client-assigned key (slug or uuid string) used to wire
    edges; ``label`` is the display name. ``is_primary`` marks the single
    centre node. ``weight`` sizes the node in the viewer (default 1).
    """

    id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=300)
    type: str = Field(default="Concept", max_length=50)
    definition: str | None = Field(default=None, max_length=4000)
    weight: int = Field(default=1, ge=1, le=100)
    is_primary: bool = False


class CuratedKGEdge(BaseModel):
    """A directed relationship between two curated nodes (by node id)."""

    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    relation: CuratedKGRelation = "RELATED_TO"


class CuratedKGGraph(BaseModel):
    """A full curated graph payload (nodes + edges)."""

    nodes: list[CuratedKGNode] = []
    edges: list[CuratedKGEdge] = []


class CuratedKGDraftSave(BaseModel):
    """Request body for saving the teacher's draft.

    Enforces every invariant the store depends on, so the service can persist
    the payload verbatim:

    * exactly ONE node has ``is_primary=True`` (the required centre node);
    * node ids are unique;
    * every edge references existing node ids and isn't a self-loop.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: list[CuratedKGNode] = []
    edges: list[CuratedKGEdge] = []

    @model_validator(mode="after")
    def _check_invariants(self) -> CuratedKGDraftSave:
        if not self.nodes:
            raise ValueError("A knowledge graph must have at least one node")

        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Node ids must be unique")

        primaries = [n for n in self.nodes if n.is_primary]
        if len(primaries) != 1:
            raise ValueError(
                "Exactly one node must be marked primary "
                f"(found {len(primaries)})"
            )

        id_set = set(ids)
        for e in self.edges:
            if e.source == e.target:
                raise ValueError(f"Edge cannot be a self-loop: {e.source}")
            if e.source not in id_set:
                raise ValueError(f"Edge source '{e.source}' is not a node id")
            if e.target not in id_set:
                raise ValueError(f"Edge target '{e.target}' is not a node id")
        return self

    @property
    def primary_node_id(self) -> str:
        """The (validated) single primary node id."""
        return next(n.id for n in self.nodes if n.is_primary)


class CuratedKGDraft(BaseModel):
    """Teacher-facing read of the draft graph + publish state.

    ``seeded`` is True when the draft was auto-populated from the AI concept
    graph on first open (so the UI can show a "seeded from AI" hint). ``exists``
    is False when no curated graph row exists yet for the lesson.
    ``seeded_placeholder`` is True when the seed was the fallback single
    "Main concept" node (AI graph off / empty / unreachable) — such a draft
    is NOT publishable: it has no real content and would show students a
    meaningless one-node graph.
    """

    model_config = ConfigDict(from_attributes=True)

    lesson_id: UUID
    exists: bool = True
    seeded: bool = False
    seeded_placeholder: bool = False
    nodes: list[CuratedKGNode] = []
    edges: list[CuratedKGEdge] = []
    primary_node_id: str | None = None
    is_published: bool = False
    published_at: datetime | None = None
    # True when the draft differs from the published snapshot (unpublished
    # edits pending). Always True when never published but the draft is non-empty.
    has_unpublished_changes: bool = False


class CuratedKGPublished(BaseModel):
    """Student-facing read of the PUBLISHED graph only.

    ``published`` is False (with empty lists) when the teacher has never
    published a graph for this lesson — the student UI hides the panel.
    """

    lesson_id: UUID
    published: bool = False
    nodes: list[CuratedKGNode] = []
    edges: list[CuratedKGEdge] = []
    primary_node_id: str | None = None
    published_at: datetime | None = None


__all__ = [
    "CuratedKGDraft",
    "CuratedKGDraftSave",
    "CuratedKGEdge",
    "CuratedKGGraph",
    "CuratedKGNode",
    "CuratedKGPublished",
    "CuratedKGRelation",
]
