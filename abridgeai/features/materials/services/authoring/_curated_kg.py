"""Teacher-curated knowledge-graph service (get / save-draft / publish).

Owns the ``lesson_knowledge_graphs`` Postgres table — the teacher-authored,
publishable KG that is SEPARATE from the AI-generated Neo4j concept graph. The
teacher edits a private ``draft`` freely and clicks Publish to snapshot it into
``published_json`` for the student reading-lesson view; editing the draft
afterward never affects the live student view until the next publish.

Follows the materials-service conventions: this module references
:class:`AsyncSession` only under ``TYPE_CHECKING`` where possible, flushes (the
router commits), and keeps HTTP concerns out (the router maps exceptions).

First-open seeding
------------------
When no curated row exists yet, :func:`get_or_seed_draft` builds a starting
draft from the AI concept graph (top concepts + relations) so the teacher
curates from a populated canvas rather than a blank one. The heaviest AI
concept becomes the initial primary node. When the AI KG is disabled / empty /
unreachable, we seed a single placeholder primary node instead so the
"exactly one primary" invariant always holds from the first save.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select

from abridgeai.features.materials.models import LessonKnowledgeGraphCurated
from abridgeai.features.materials.schemas.curated_kg import (
    CuratedKGDraft,
    CuratedKGDraftSave,
    CuratedKGEdge,
    CuratedKGNode,
    CuratedKGPublished,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _graph_from_json(payload: dict[str, Any] | None) -> tuple[list[CuratedKGNode], list[CuratedKGEdge]]:
    """Parse a stored ``{nodes, edges}`` JSON blob into typed DTOs.

    Tolerant of missing keys / malformed rows (returns what parses) so a
    corrupt draft never 500s the editor — the teacher can just re-save.
    """
    if not payload:
        return [], []
    nodes: list[CuratedKGNode] = []
    for raw in payload.get("nodes", []) or []:
        try:
            nodes.append(CuratedKGNode.model_validate(raw))
        except Exception:  # noqa: BLE001 -- skip an unparseable node, keep the rest
            continue
    edges: list[CuratedKGEdge] = []
    node_ids = {n.id for n in nodes}
    for raw in payload.get("edges", []) or []:
        try:
            edge = CuratedKGEdge.model_validate(raw)
        except Exception:  # noqa: BLE001
            continue
        # Drop dangling edges (a node it references was removed).
        if edge.source in node_ids and edge.target in node_ids:
            edges.append(edge)
    return nodes, edges


def _json_from_graph(nodes: list[CuratedKGNode], edges: list[CuratedKGEdge]) -> dict[str, Any]:
    return {
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }


def _is_placeholder_graph(nodes: list[CuratedKGNode]) -> bool:
    """True when ``nodes`` is exactly the fallback one-node seed.

    The seed produced when the AI graph is off/empty/unreachable is a single
    "Main concept" primary node. Publishing it shows students a meaningless
    one-node graph, so publish must refuse it — the UI hides the action based
    on ``seeded_placeholder``, this is the server-side backstop for a draft
    that was saved and then published directly via API.
    """
    return len(nodes) == 1 and nodes[0].id == "primary" and nodes[0].label == "Main concept"


async def _get_row(
    db: AsyncSession, lesson_id: UUID
) -> LessonKnowledgeGraphCurated | None:
    stmt = select(LessonKnowledgeGraphCurated).where(
        LessonKnowledgeGraphCurated.lesson_id == lesson_id,
        LessonKnowledgeGraphCurated.deleted_at.is_(None),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _seed_nodes_from_ai(
    db: AsyncSession, lesson_id: UUID
) -> tuple[list[CuratedKGNode], list[CuratedKGEdge], bool]:
    """Build a starting draft from the AI concept graph, if available.

    Returns a single placeholder primary node when the AI KG is off / empty /
    unreachable, so the caller always gets a valid one-primary seed. The
    third element flags that placeholder case: publishing such a draft would
    show students a meaningless one-node "Main concept" graph, so the caller
    must refuse to publish it (the UI also hides the publish action).
    """
    from abridgeai.features.materials.services.authoring._reads import (  # noqa: PLC0415
        get_lesson_knowledge_graph,
    )

    placeholder = (
        [CuratedKGNode(id="primary", label="Main concept", type="Concept", weight=10, is_primary=True)],
        [],
        True,
    )
    try:
        ai = await get_lesson_knowledge_graph(db, lesson_id, limit=24)
    except Exception:  # noqa: BLE001 -- AI KG is best-effort seed material
        return placeholder
    if not ai.enabled or not ai.nodes:
        return placeholder

    # The AI nodes come weight-descending; the heaviest is the initial primary.
    nodes: list[CuratedKGNode] = []
    for i, n in enumerate(ai.nodes):
        nodes.append(
            CuratedKGNode(
                id=n.id,
                label=n.label,
                type=n.type or "Concept",
                definition=n.definition,
                weight=max(1, min(100, n.weight)),
                is_primary=(i == 0),
            )
        )
    node_ids = {n.id for n in nodes}
    edges = [
        CuratedKGEdge(source=e.source, target=e.target, relation=e.relation)
        for e in ai.edges
        if e.source in node_ids and e.target in node_ids and e.source != e.target
    ]
    return nodes, edges, False


def _has_unpublished_changes(row: LessonKnowledgeGraphCurated) -> bool:
    """True when the draft differs from the published snapshot."""
    if row.published_json is None:
        # Never published: pending changes exist if the draft has any nodes.
        return bool((row.draft_json or {}).get("nodes"))
    return row.draft_json != row.published_json


async def get_or_seed_draft(db: AsyncSession, lesson_id: UUID) -> CuratedKGDraft:
    """Return the teacher's draft, seeding from the AI KG on first open.

    Does NOT persist the seed — the seeded graph is returned as an in-memory
    starting point and only written when the teacher explicitly saves. This
    keeps "open the editor" a pure read and avoids creating rows for lessons a
    teacher merely glances at.
    """
    row = await _get_row(db, lesson_id)
    if row is not None:
        nodes, edges = _graph_from_json(row.draft_json)
        return CuratedKGDraft(
            lesson_id=lesson_id,
            exists=True,
            seeded=False,
            nodes=nodes,
            edges=edges,
            primary_node_id=row.primary_node_id,
            is_published=row.published_json is not None,
            published_at=row.published_at,
            has_unpublished_changes=_has_unpublished_changes(row),
        )

    # No row yet — seed from AI (not persisted until save).
    nodes, edges, seeded_placeholder = await _seed_nodes_from_ai(db, lesson_id)
    primary = next((n.id for n in nodes if n.is_primary), None)
    return CuratedKGDraft(
        lesson_id=lesson_id,
        exists=False,
        seeded=True,
        seeded_placeholder=seeded_placeholder,
        nodes=nodes,
        edges=edges,
        primary_node_id=primary,
        is_published=False,
        published_at=None,
        has_unpublished_changes=True,
    )


async def save_draft(
    db: AsyncSession,
    lesson_id: UUID,
    payload: CuratedKGDraftSave,
    actor_id: UUID | None,
) -> CuratedKGDraft:
    """Persist the teacher's draft (upsert by lesson).

    The payload is pre-validated (exactly one primary, unique ids, valid
    edges), so we store it verbatim and mirror ``primary_node_id`` to the
    top-level column. Does NOT touch the published snapshot — publishing is a
    separate explicit action.
    """
    draft_json = _json_from_graph(payload.nodes, payload.edges)
    row = await _get_row(db, lesson_id)
    if row is None:
        row = LessonKnowledgeGraphCurated(
            lesson_id=lesson_id,
            draft_json=draft_json,
            primary_node_id=payload.primary_node_id,
            created_by=actor_id,
            updated_by=actor_id,
        )
        db.add(row)
    else:
        row.draft_json = draft_json
        row.primary_node_id = payload.primary_node_id
        row.updated_by = actor_id
    await db.flush()
    await db.refresh(row)

    return CuratedKGDraft(
        lesson_id=lesson_id,
        exists=True,
        seeded=False,
        nodes=payload.nodes,
        edges=payload.edges,
        primary_node_id=row.primary_node_id,
        is_published=row.published_json is not None,
        published_at=row.published_at,
        has_unpublished_changes=_has_unpublished_changes(row),
    )


async def publish(
    db: AsyncSession,
    lesson_id: UUID,
    actor_id: UUID | None,
) -> CuratedKGDraft:
    """Snapshot the current draft into the published slot (students see this).

    Publishing an empty / never-saved draft is a no-op guarded by the caller
    (router returns 409 when there's nothing to publish). A placeholder draft
    (the single "Main concept" seed produced when the AI graph is empty) is
    refused the same way: it carries no real content, so publishing it would
    show students a meaningless one-node graph — exactly what happened before
    the guard existed (a lesson could be "published with an empty KG" before
    any material was uploaded). Copies draft_json → published_json and stamps
    published_at.
    """
    row = await _get_row(db, lesson_id)
    if row is None or not (row.draft_json or {}).get("nodes"):
        raise CuratedKGEmptyError(
            "Cannot publish an empty knowledge graph — save a draft with at "
            "least one primary node first"
        )
    nodes, _edges = _graph_from_json(row.draft_json)
    if _is_placeholder_graph(nodes):
        raise CuratedKGEmptyError(
            "Cannot publish a placeholder knowledge graph — upload and process "
            "material first so the graph has real concepts to show"
        )
    row.published_json = dict(row.draft_json)
    row.published_primary_node_id = row.primary_node_id
    row.published_at = datetime.now(UTC)
    row.updated_by = actor_id
    await db.flush()
    await db.refresh(row)

    nodes, edges = _graph_from_json(row.draft_json)
    return CuratedKGDraft(
        lesson_id=lesson_id,
        exists=True,
        seeded=False,
        nodes=nodes,
        edges=edges,
        primary_node_id=row.primary_node_id,
        is_published=True,
        published_at=row.published_at,
        has_unpublished_changes=False,
    )


async def unpublish(
    db: AsyncSession,
    lesson_id: UUID,
    actor_id: UUID | None,
) -> CuratedKGDraft:
    """Roll back a publish: clear the student-visible snapshot.

    The inverse of :func:`publish`. Publish is one-way today, so a graph
    published by mistake (or one whose material was later deleted) stays on
    the student reading view forever with no way to remove it. This clears
    ``published_json`` / ``published_primary_node_id`` / ``published_at``;
    the draft is untouched, so the teacher can re-publish after fixing it.
    Students immediately see the knowledge-map panel disappear
    (``published=False`` hides it). No-op when nothing is published.
    """
    row = await _get_row(db, lesson_id)
    if row is None or row.published_json is None:
        return CuratedKGDraft(
            lesson_id=lesson_id,
            exists=row is not None,
            seeded=False,
            nodes=[],
            edges=[],
            primary_node_id=row.primary_node_id if row is not None else None,
            is_published=False,
            published_at=None,
            has_unpublished_changes=False,
        )
    row.published_json = None
    row.published_primary_node_id = None
    row.published_at = None
    row.updated_by = actor_id
    await db.flush()
    await db.refresh(row)

    nodes, edges = _graph_from_json(row.draft_json)
    return CuratedKGDraft(
        lesson_id=lesson_id,
        exists=True,
        seeded=False,
        nodes=nodes,
        edges=edges,
        primary_node_id=row.primary_node_id,
        is_published=False,
        published_at=None,
        has_unpublished_changes=bool(nodes),
    )


async def get_published(db: AsyncSession, lesson_id: UUID) -> CuratedKGPublished:
    """Student-facing read of the PUBLISHED graph only.

    Returns ``published=False`` with empty lists when the teacher has never
    published for this lesson, so the student UI can hide the panel.
    """
    row = await _get_row(db, lesson_id)
    if row is None or row.published_json is None:
        return CuratedKGPublished(lesson_id=lesson_id, published=False)
    nodes, edges = _graph_from_json(row.published_json)
    return CuratedKGPublished(
        lesson_id=lesson_id,
        published=True,
        nodes=nodes,
        edges=edges,
        primary_node_id=row.published_primary_node_id,
        published_at=row.published_at,
    )


class CuratedKGEmptyError(Exception):
    """Raised when publishing a graph that has no nodes."""


__all__ = [
    "CuratedKGEmptyError",
    "get_or_seed_draft",
    "get_published",
    "publish",
    "save_draft",
    "unpublish",
]
