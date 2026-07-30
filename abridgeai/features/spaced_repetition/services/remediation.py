"""Remediation notification dispatch for failed SR cards (T7.5.10).

Builds a deep-link notification when a student fails a card. Walks the
question's source chunks outward through the knowledge graph, resolves
related chunks back to material versions, composes deep-link URLs
(``?p=<page>`` for documents, ``?t=<seconds>`` for media, ``#anchor``
for HTML), and dispatches a ``spaced_repetition`` notification via
:func:`abridgeai.features.notifications.services.dispatch.send_notification`.

BUG-2: the entrypoint is awaited by the caller **after** ``db.commit()``
returns (see :mod:`._events`). A rolled-back ``CardReview`` therefore
never reaches this code -- no ghost notifications.

KG boundary
-----------
The knowledge graph lives in Neo4j. Two narrow helpers
(``_concepts_for_chunks`` and ``_chunks_for_concepts``) wrap the Cypher
calls so the dispatcher itself stays focused on orchestration. Both
gracefully return empty results when the KG is disabled or the driver
raises -- in those cases dispatch is silently skipped, matching the
"no notification with empty resources" rule from plan §7.5.10.

Cross-feature reads
-------------------
Postgres reads (``quiz_questions``, ``quizzes``, ``courses``,
``learning_materials``, ``learning_material_versions``,
``document_chunks``, ``lessons``) all go through raw
``sqlalchemy.text(...)`` -- no foreign feature ORM models are imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.ai.knowledge_graph.retrieval import retrieve_kg_context_for_anchors
from abridgeai.ai.knowledge_graph.tenancy import organization_id_for_course
from abridgeai.core.observability import get_logger
from abridgeai.features.notifications.services.dispatch import send_notification
from abridgeai.infrastructure.neo4j import (
    KnowledgeGraphDisabledError,
    graph_client,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_logger = get_logger(__name__)

_MAX_RESOURCES = 3
_NOTIFICATION_CATEGORY = "spaced_repetition"


@dataclass(frozen=True)
class _ResolvedResource:
    material_id: UUID
    material_version_id: UUID
    material_title: str
    material_type: str
    lesson_id: UUID
    lesson_title: str
    course_slug: str
    chunk_id: UUID
    chunk_index: int
    source_metadata: dict[str, Any]
    deep_link: str


def build_deep_link(
    *,
    course_slug: str,
    lesson_id: UUID,
    material_id: UUID,
    material_type: str,
    source_location: dict[str, Any],
) -> str:
    """Compose a deep-link URL into a learning resource.

    Per plan §7.5.10:

    * ``audio`` / ``video`` → ``?t=<seconds>`` (timestamp_start_ms // 1000)
    * ``document`` / ``pdf`` / ``slides`` / ``pptx`` / ``docx`` /
      ``xlsx`` → ``?p=<page>``
    * ``html`` → ``#<anchor>``
    * fallback → bare ``base`` URL
    """
    base = f"/courses/{course_slug}/lessons/{lesson_id}/resources/{material_id}"
    media_kinds = {"audio", "video"}
    page_kinds = {"document", "pdf", "slides", "pptx", "docx", "xlsx"}

    if material_type in media_kinds:
        ts_ms = source_location.get("timestamp_start_ms")
        if ts_ms is None:
            ts_ms = source_location.get("timestamp_ms")
        if ts_ms is not None:
            try:
                seconds = int(ts_ms) // 1000
            except (TypeError, ValueError):
                return base
            return f"{base}?t={seconds}"
        return base

    if material_type in page_kinds:
        page = source_location.get("page")
        if page is None:
            page = source_location.get("page_number")
        if page is not None:
            try:
                page_int = int(page)
            except (TypeError, ValueError):
                return base
            return f"{base}?p={page_int}"
        return base

    if material_type == "html":
        anchor = source_location.get("anchor")
        if isinstance(anchor, str) and anchor:
            return f"{base}#{anchor.lstrip('#')}"
        return base

    return base


async def _load_question_context(db: AsyncSession, *, question_id: UUID) -> dict[str, Any] | None:
    """Resolve question → quiz → course → first source-lesson via raw SQL.

    Returns ``None`` when the question cannot be resolved -- not an
    error, just nothing to remediate.
    """
    result = await db.execute(
        text(
            """
            SELECT
                qq.id              AS question_id,
                qq.source_refs     AS source_refs,
                q.id               AS quiz_id,
                q.title            AS quiz_title,
                q.course_id        AS course_id,
                q.module_id        AS module_id,
                c.slug             AS course_slug,
                qsl.lesson_id      AS lesson_id
            FROM quiz_questions qq
            JOIN quizzes q ON q.id = qq.quiz_id
            JOIN courses c ON c.id = q.course_id
            LEFT JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
            WHERE qq.id = :qid
            ORDER BY qsl.lesson_id
            LIMIT 1
            """
        ),
        {"qid": str(question_id)},
    )
    row = result.mappings().first()
    if row is None:
        return None
    return dict(row)


def _extract_chunk_ids(source_refs: object) -> list[UUID]:
    """Pull chunk UUIDs out of the heterogeneous ``source_refs`` JSON.

    Tolerates list-of-dict (canonical, ``chunk_id`` key), list-of-str
    (bare UUIDs), and the legacy ``id`` key for forward-compat with
    older fixtures.
    """
    if not isinstance(source_refs, list):
        return []
    out: list[UUID] = []
    for entry in source_refs:
        candidate: Any = None
        if isinstance(entry, str):
            candidate = entry
        elif isinstance(entry, dict):
            candidate = entry.get("chunk_id") or entry.get("id")
        if candidate is None:
            continue
        try:
            out.append(UUID(str(candidate)))
        except (TypeError, ValueError):
            continue
    return out


async def _concepts_for_chunks(chunk_ids: list[UUID]) -> list[str]:
    """Neo4j: chunks → concept names via ``MENTIONS_CONCEPT``.

    Returns empty list when the KG is disabled or unreachable so the
    dispatcher quietly skips rather than crashing the after-commit hook.
    """
    if not chunk_ids:
        return []
    query = """
        MATCH (chunk:Chunk)-[:MENTIONS_CONCEPT]->(concept:Concept)
        WHERE chunk.id IN $chunk_ids
        RETURN DISTINCT concept.name AS name
    """
    string_ids = [str(cid) for cid in chunk_ids]
    try:
        async with graph_client() as client, client.session() as session:
            result = await session.run(query, chunk_ids=string_ids)
            records = [dict(record) async for record in result]
    except KnowledgeGraphDisabledError:
        return []
    except Exception as exc:  # pragma: no cover -- graceful degradation when Neo4j is down
        _logger.debug("remediation_kg_seed_lookup_failed", error=str(exc))
        return []
    return [str(record["name"]) for record in records if record.get("name")]


async def _chunks_for_concepts(
    concept_names: list[str],
    *,
    exclude: set[str],
    org_id: UUID,
) -> list[UUID]:
    """Neo4j: concept names → chunk UUIDs that mention any of them.

    Excludes the original seed chunks so we surface *related* material,
    not the same one the student just flunked.

    ``org_id`` scopes both hops. Concept names are generic ("normalization",
    "recursion"), so an unscoped match walks into other tenants' chunks. The
    downstream Postgres join does filter by course, but relying on that means
    the leak is one refactor away — and until then this query reads and
    discards another customer's graph on every card failure.
    """
    if not concept_names:
        return []
    query = """
        MATCH (chunk:Chunk {org_id: $org_id})-[:MENTIONS_CONCEPT]->(concept:Concept)
        WHERE toLower(concept.name) IN $names
          AND concept.org_id = $org_id
          AND NOT chunk.id IN $exclude
        RETURN DISTINCT chunk.id AS chunk_id
    """
    lowered = [n.lower() for n in concept_names if n]
    try:
        async with graph_client() as client, client.session() as session:
            result = await session.run(
                query,
                names=lowered,
                exclude=list(exclude),
                org_id=str(org_id),
            )
            records = [dict(record) async for record in result]
    except KnowledgeGraphDisabledError:
        return []
    except Exception as exc:  # pragma: no cover -- graceful degradation when Neo4j is down
        _logger.debug("remediation_kg_chunk_lookup_failed", error=str(exc))
        return []
    out: list[UUID] = []
    for record in records:
        raw = record.get("chunk_id")
        if raw is None:
            continue
        try:
            out.append(UUID(str(raw)))
        except (TypeError, ValueError):
            continue
    return out


async def _resolve_chunks_to_materials(
    db: AsyncSession,
    *,
    chunk_ids: list[UUID],
    course_id: UUID,
) -> list[_ResolvedResource]:
    """Resolve chunk UUIDs → material rows + deep-links via raw SQL.

    Scoped to ``course_id``, de-duped by ``material_id`` -- one
    deep-link per material is enough for the notification card.
    """
    if not chunk_ids:
        return []
    rows = await db.execute(
        text(
            """
            SELECT
                dc.id                     AS chunk_id,
                dc.chunk_index            AS chunk_index,
                dc.metadata               AS chunk_metadata,
                dc.lesson_id              AS lesson_id,
                dc.material_version_id    AS material_version_id,
                lmv.material_id           AS material_id,
                lm.title                  AS material_title,
                lm.material_type          AS material_type,
                l.title                   AS lesson_title,
                c.slug                    AS course_slug
            FROM document_chunks dc
            JOIN learning_material_versions lmv ON lmv.id = dc.material_version_id
            JOIN learning_materials lm ON lm.id = lmv.material_id
            JOIN lessons l ON l.id = dc.lesson_id
            JOIN courses c ON c.id = dc.course_id
            WHERE dc.id = ANY(:ids)
              AND dc.course_id = :course_id
            ORDER BY dc.created_at DESC
            """
        ),
        {
            "ids": [str(cid) for cid in chunk_ids],
            "course_id": str(course_id),
        },
    )

    resolved: list[_ResolvedResource] = []
    seen_materials: set[UUID] = set()
    for row in rows.mappings().all():
        material_id = row["material_id"]
        if isinstance(material_id, str):
            material_id = UUID(material_id)
        if material_id in seen_materials:
            continue
        seen_materials.add(material_id)

        chunk_id = row["chunk_id"]
        if isinstance(chunk_id, str):
            chunk_id = UUID(chunk_id)
        lesson_id = row["lesson_id"]
        if isinstance(lesson_id, str):
            lesson_id = UUID(lesson_id)
        version_id = row["material_version_id"]
        if isinstance(version_id, str):
            version_id = UUID(version_id)

        metadata = row["chunk_metadata"] or {}
        if not isinstance(metadata, dict):
            metadata = {}

        material_type = str(row["material_type"] or "")
        course_slug = str(row["course_slug"] or "")
        deep_link = build_deep_link(
            course_slug=course_slug,
            lesson_id=lesson_id,
            material_id=material_id,
            material_type=material_type,
            source_location=metadata,
        )
        resolved.append(
            _ResolvedResource(
                material_id=material_id,
                material_version_id=version_id,
                material_title=str(row["material_title"] or ""),
                material_type=material_type,
                lesson_id=lesson_id,
                lesson_title=str(row["lesson_title"] or ""),
                course_slug=course_slug,
                chunk_id=chunk_id,
                chunk_index=int(row["chunk_index"] or 0),
                source_metadata=metadata,
                deep_link=deep_link,
            )
        )
        if len(resolved) >= _MAX_RESOURCES:
            break
    return resolved


def _compose_payload(
    *,
    missed_concepts: list[str],
    resources: list[_ResolvedResource],
    locale: str | None,
) -> tuple[str, str]:
    """Render notification ``title`` + ``body`` from concepts + resources.

    The Notification model has no ``payload`` JSONB column (see
    ``features/notifications/models.py``), so the primary deep-link is
    embedded in the body text together with up to two follow-up links.
    The frontend learner inbox renders the body as Markdown, so plain
    paths suffice as link targets.

    Copy is localized to the recipient's ``locale`` ('en' | 'vi') via the
    notifications ``messages`` module (cross-feature import authorised in
    pyproject). Resource labels stay verbatim (material/lesson titles);
    only the surrounding sentences translate.
    """
    from abridgeai.features.notifications import messages

    primary_resource = resources[0].material_title if resources else None
    resource_links = [
        (
            resource.material_title or resource.lesson_title or "Resource",
            resource.deep_link,
        )
        for resource in resources
    ]
    title = messages.remediation_title(
        missed_concepts=missed_concepts,
        primary_resource=primary_resource,
        locale=locale,
    )
    body = messages.remediation_body(resource_links=resource_links, locale=locale)
    return title, body


async def dispatch_remediation_for_card_failure(
    db: AsyncSession,
    *,
    student_id: UUID,
    question_id: UUID,
    quiz_attempt_id: UUID | None,  # noqa: ARG001 -- carried for future telemetry hooks
    arq_pool: object | None = None,
) -> None:
    """Build deep-link notification when student fails a card.

    Steps (per plan §7.5.10):

    1. Resolve question → quiz → course → first source-lesson.
    2. Read ``question.source_refs`` (chunk UUIDs).
    3. Look up concepts mentioned by the seed chunks (Neo4j).
    4. Walk KG outward via :func:`retrieve_kg_context_for_anchors`.
    5. Resolve related concept names back to chunks (Neo4j).
    6. Resolve chunks to materials in the same course (Postgres).
    7. Skip dispatch if no resources resolved.
    8. Send notification via :func:`send_notification`.
    """
    context = await _load_question_context(db, question_id=question_id)
    if context is None:
        _logger.debug(
            "remediation_skipped_no_context",
            question_id=str(question_id),
            student_id=str(student_id),
        )
        return

    course_slug = context.get("course_slug")
    course_id_raw = context.get("course_id")
    if not course_slug or not course_id_raw:
        _logger.debug(
            "remediation_skipped_no_course",
            question_id=str(question_id),
        )
        return
    course_id = course_id_raw if isinstance(course_id_raw, UUID) else UUID(str(course_id_raw))

    seed_chunk_ids = _extract_chunk_ids(context.get("source_refs"))
    if not seed_chunk_ids:
        _logger.debug(
            "remediation_skipped_no_source_refs",
            question_id=str(question_id),
        )
        return

    seed_concepts = await _concepts_for_chunks(seed_chunk_ids)
    if not seed_concepts:
        _logger.debug(
            "remediation_skipped_no_seed_concepts",
            question_id=str(question_id),
        )
        return

    org_id = await organization_id_for_course(db, course_id)
    if org_id is None:
        _logger.debug(
            "remediation_skipped_no_org",
            question_id=str(question_id),
            course_id=str(course_id),
        )
        return

    kg_context = await retrieve_kg_context_for_anchors(
        seed_concepts, org_id=org_id, depth=2
    )
    related_names = [c.name for c in kg_context.concepts if c.name]
    if not related_names:
        _logger.debug(
            "remediation_skipped_empty_kg",
            question_id=str(question_id),
        )
        return

    seed_chunk_str = {str(cid) for cid in seed_chunk_ids}
    related_chunk_ids = await _chunks_for_concepts(
        related_names, exclude=seed_chunk_str, org_id=org_id
    )
    if not related_chunk_ids:
        _logger.debug(
            "remediation_skipped_no_related_chunks",
            question_id=str(question_id),
        )
        return

    resources = await _resolve_chunks_to_materials(
        db, chunk_ids=related_chunk_ids, course_id=course_id
    )
    if not resources:
        _logger.debug(
            "remediation_skipped_no_resources",
            question_id=str(question_id),
        )
        return

    # Cross-feature read: recipient's preferred locale ('en' | 'vi') so the
    # notification copy is rendered in their language at creation time. The
    # identity public API is authorised by the api.public wildcard carve-out.
    from abridgeai.features.identity.api.public import get_user_locale

    locale = await get_user_locale(db, student_id)
    title, body = _compose_payload(
        missed_concepts=seed_concepts, resources=resources, locale=locale
    )

    # Precomputed deep-link (Option B): the producer holds the routing
    # context (course slug) so it builds the exact relative path once. The
    # learner "continue learning" route is the valid navigable target — the
    # per-material resources route does not exist on the client, so the body
    # links (build_deep_link) land nowhere; action_url is the reliable path.
    action_url = f"/courses/{course_slug}/learn"

    await send_notification(
        db,
        recipient_user_id=student_id,
        notification_type=_NOTIFICATION_CATEGORY,
        title=title,
        body=body,
        entity_type="quiz_question",
        entity_id=question_id,
        action_url=action_url,
        arq_pool=arq_pool,
    )
    _logger.info(
        "remediation_dispatched",
        question_id=str(question_id),
        student_id=str(student_id),
        resource_count=len(resources),
        primary_deep_link=resources[0].deep_link,
    )


__all__ = [
    "build_deep_link",
    "dispatch_remediation_for_card_failure",
]
