"""Authoring read-side: list / get / update / processing-progress.

Composed by ``GET`` endpoints in
:mod:`features.materials.routers.authoring`. Mutating writes
(``init_upload``, ``complete_upload``, ``reprocess_material``,
``soft_delete_material``) live in :mod:`._upload` and :mod:`._versions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from abridgeai.ai.models import ProcessingJob
from abridgeai.core.config import get_settings
from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.security import CurrentUser
from abridgeai.features.materials.models import LearningMaterial, LearningMaterialVersion
from abridgeai.features.materials.workers.enqueue import enqueue_material_ingest
from abridgeai.features.materials.queries import (
    get_authoring_stream_target_for_material,
    get_latest_processing_job,
    get_lesson_processing_summary,
    get_material_for_authoring,
    list_all_materials,
)
from abridgeai.features.materials.schemas import (
    LessonProcessingSummary,
    MaterialAuthoring,
    MaterialLinkExisting,
    MaterialStreamUrl,
    MaterialUpdate,
    ProcessingProgress,
)
from abridgeai.features.materials.schemas.status import (
    KGEdge,
    KGNode,
    LessonKnowledgeGraph,
)
from abridgeai.features.materials.services.authoring._common import (
    present_version,
    require_material,
)
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_authoring_materials(
    db: AsyncSession, lesson_id: UUID, *, include_archived: bool = False
) -> list[MaterialAuthoring]:
    materials = await list_all_materials(db, lesson_id, include_archived=include_archived)
    return [await _present_material(db, m) for m in materials]


async def get_authoring_material(db: AsyncSession, material_id: UUID) -> MaterialAuthoring | None:
    material = await get_material_for_authoring(db, material_id)
    if material is None:
        return None
    return await _present_material(db, material)


async def _present_material(db: AsyncSession, material: LearningMaterial) -> MaterialAuthoring:
    """Hydrate :class:`MaterialAuthoring` (latest version + counts)."""
    from sqlalchemy import func, select  # noqa: PLC0415  -- localised raw escape hatch

    version_count = (
        await db.execute(
            select(func.count(LearningMaterialVersion.id)).where(
                LearningMaterialVersion.material_id == material.id
            )
        )
    ).scalar_one()
    latest_version: LearningMaterialVersion | None = None
    if material.current_version_id is not None:
        latest_version = await db.get(LearningMaterialVersion, material.current_version_id)

    return MaterialAuthoring.model_validate(material).model_copy(
        update={
            "version_count": int(version_count),
            "latest_version": present_version(latest_version)
            if latest_version is not None
            else None,
        }
    )


async def get_processing_progress(db: AsyncSession, material_id: UUID) -> ProcessingProgress | None:
    """Return the latest version's progress slice (or ``None``).

    Prefers the live Redis progress snapshot the worker publishes at each
    stage transition. The ingest runs in one long DB transaction that only
    commits at the end, so the DB ``processing_jobs.progress_percent`` /
    ``version.processing_status`` stay frozen at their pre-run values for
    the whole run (MVCC). Redis is written outside that transaction, so it
    reflects the real, live stage. When no live key exists (run finished,
    never started, or Redis down) we fall back to the authoritative DB row.
    """
    from abridgeai.features.materials.ingestion.progress import read_progress  # noqa: PLC0415

    material = await get_material_for_authoring(db, material_id)
    if material is None or material.current_version_id is None:
        return None
    version = await db.get(LearningMaterialVersion, material.current_version_id)
    if version is None:
        return None
    job = await get_latest_processing_job(db, version.id)
    db_percent = int(job.progress_percent) if job is not None else 0
    db_status = version.processing_status

    live = await read_progress(version.id)
    if live is not None:
        # A live key means a run is in flight NOW; it is strictly more
        # current than the DB row (which stays frozen at its pre-run
        # snapshot until the ingest's final commit — and on a *reprocess*
        # that snapshot is the previous run's ready/100). So trust the live
        # values verbatim rather than max()-ing against the stale DB, which
        # would otherwise show e.g. "embedding 100%". The key is cleared on
        # clean completion, at which point we fall through to the DB row.
        # Fold any sub-progress detail (e.g. "42/85" from the per-chunk KG
        # build) onto the stage label so a long-running looping stage shows
        # a live running count instead of a frozen percent.
        stage_label = live.get("stage_label")
        detail = live.get("detail")
        log_line = f"{stage_label} · {detail}" if stage_label and detail else (stage_label or detail)
        return ProcessingProgress(
            material_id=material.id,
            version_id=version.id,
            processing_status=live.get("status") or db_status,
            progress_percent=max(0, min(100, int(live.get("percent", db_percent)))),
            latest_log_line=log_line,
            error_message=version.processing_error,
        )

    return ProcessingProgress(
        material_id=material.id,
        version_id=version.id,
        processing_status=db_status,
        progress_percent=db_percent,
        latest_log_line=None,
        error_message=version.processing_error,
    )


async def update_material(
    db: AsyncSession,
    material_id: UUID,
    payload: MaterialUpdate,
    actor: CurrentUser,
) -> MaterialAuthoring:
    del actor  # audit listener writes updated_by
    material = await require_material(db, material_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(material, key, value)
    await flush_or_conflict(db)
    await db.refresh(material)
    return await _present_material(db, material)


async def get_authoring_stream_url(db: AsyncSession, material_id: UUID) -> MaterialStreamUrl | None:
    """Mint a presigned GET URL for a teacher previewing a material.

    Authoring sibling of
    :func:`features.materials.services.catalog.get_stream_url_for_material`.
    Skips the learner ``visible_to_students`` and ``processing_status='ready'``
    gates so a teacher can review hidden / mid-pipeline materials. Returns
    ``None`` (router maps to 404) when the material is missing, soft-deleted,
    or has no current version with a resolvable storage object.
    """
    target = await get_authoring_stream_target_for_material(db, material_id)
    if target is None:
        return None
    settings = get_settings()
    safe_title = target.title.replace('"', "")
    url, _ = await create_stream_url(
        target,
        response_headers={"Content-Disposition": f'attachment; filename="{safe_title}"'},
    )
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    expires_at = datetime.now(tz=UTC) + timedelta(seconds=settings.s3_url_ttl_seconds)
    return MaterialStreamUrl(
        url=url,
        expires_at=expires_at,
        material_version_id=target.material_version_id,
    )


async def get_lesson_processing_summary_view(
    db: AsyncSession, lesson_id: UUID
) -> LessonProcessingSummary:
    """Aggregate processing-status counts across every material under a lesson.

    Wraps :func:`get_lesson_processing_summary` with a typed DTO. Returns
    a row of zeroes for a lesson with no materials so the SPA can render
    the empty-state without an extra null check.
    """
    counts = await get_lesson_processing_summary(db, lesson_id)
    return LessonProcessingSummary(lesson_id=lesson_id, **counts)


async def get_lesson_knowledge_graph(
    db: AsyncSession, lesson_id: UUID, *, limit: int = 24
) -> LessonKnowledgeGraph:
    """Bounded concept graph for a lesson, for the teacher AI-Hub viz.

    Reads the top-``limit`` most-mentioned concepts (+ edges among them)
    from Neo4j via :func:`lesson_concept_graph_preview`. Degrades
    gracefully: when the KG feature is disabled or Neo4j is unreachable,
    returns ``enabled`` accordingly with empty lists rather than raising,
    so the SPA renders a hint instead of erroring. ``db`` is currently
    unused (KG lives in Neo4j) but kept in the signature for consistency
    with the other read-service functions and future SQL-side gating.
    """
    del db  # KG data lives in Neo4j, not Postgres
    from abridgeai.ai.knowledge_graph import lesson_concept_graph_preview  # noqa: PLC0415
    from abridgeai.ai.knowledge_graph.retrieval import lesson_concepts  # noqa: PLC0415
    from abridgeai.infrastructure.neo4j import (  # noqa: PLC0415
        KnowledgeGraphDisabledError,
        graph_client,
    )

    settings = get_settings()
    if not settings.knowledge_graph_enabled:
        return LessonKnowledgeGraph(lesson_id=lesson_id, enabled=False)

    try:
        async with graph_client() as client:
            concepts, relationships = await lesson_concept_graph_preview(
                client, lesson_id, limit=limit
            )
            # Full concept count so the UI can say "top N of M".
            total = len(await lesson_concepts(client, lesson_id))
    except KnowledgeGraphDisabledError:
        return LessonKnowledgeGraph(lesson_id=lesson_id, enabled=False)
    except Exception:  # noqa: BLE001 -- Neo4j down: degrade, don't 500
        return LessonKnowledgeGraph(lesson_id=lesson_id, enabled=True)

    # Concepts come back ordered by centrality (mention count DESC), so the
    # first node is the most-mentioned. We stamp a descending weight from
    # that ordering (N, N-1, …, 1) — enough for the UI to size nodes by
    # relative importance without threading the raw count through the
    # shared Concept dataclass.
    n = len(concepts)
    nodes = [
        KGNode(
            id=c.name.lower(),
            label=c.name,
            type=c.type,
            definition=c.definition,
            weight=n - i,
        )
        for i, c in enumerate(concepts)
    ]
    node_ids = {n.id for n in nodes}
    edges = [
        KGEdge(
            source=r.source,
            target=r.target,
            relation="PREREQUISITE_OF" if r.relation == "PREREQUISITE_OF" else "RELATED_TO",
        )
        for r in relationships
        if r.source in node_ids and r.target in node_ids
    ]
    return LessonKnowledgeGraph(
        lesson_id=lesson_id,
        enabled=True,
        nodes=nodes,
        edges=edges,
        total_concepts=total,
    )


async def link_existing_material(
    db: AsyncSession,
    lesson_id: UUID,
    payload: MaterialLinkExisting,
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> MaterialAuthoring:
    """Create a material record linked to an already-uploaded storage object.

    Bug-fix (material-ingestion "pending forever"): previously this always
    stamped ``processing_status='pending'`` but NEVER created a
    ``ProcessingJob`` or enqueued ingestion — so the version sat in
    "pending" forever with nothing scheduled to move it, and no viewable
    rendition was ever produced. Now:

    * ``ai_processing_enabled=True`` → create a ``ProcessingJob`` and
      enqueue ``ingest_material_version_task`` (via the resilient
      :func:`enqueue_material_ingest`, which never silently no-ops). The
      version legitimately starts ``pending`` because a job is now queued.
    * ``ai_processing_enabled=False`` → leave the version ``cancelled``
      (a terminal, non-misleading state) so it doesn't masquerade as an
      in-flight job the teacher is waiting on. The AI Hub's "Enable AI"
      action re-enables + reprocesses to kick off ingestion later.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    material_type = payload.material_type or "other"

    material = LearningMaterial(
        lesson_id=lesson_id,
        title=payload.title,
        material_type=material_type,
        ai_processing_enabled=payload.ai_processing_enabled,
        visible_to_students=payload.visible_to_students,
    )
    db.add(material)
    await flush_or_conflict(db)
    await db.refresh(material)

    # "pending" only when a job is actually being scheduled; otherwise the
    # version is parked in a terminal state so it never looks stuck.
    initial_status = "pending" if payload.ai_processing_enabled else "cancelled"
    version = LearningMaterialVersion(
        material_id=material.id,
        storage_object_id=payload.storage_object_id,
        version_no=1,
        is_current=True,
        processing_status=initial_status,
        uploaded_by=actor.user_id,
        uploaded_at=datetime.now(tz=UTC),
    )
    db.add(version)
    await flush_or_conflict(db)
    await db.refresh(version)

    material.current_version_id = version.id
    await flush_or_conflict(db)
    await db.refresh(material)

    if payload.ai_processing_enabled:
        pipeline_run_id = uuid4()
        job = ProcessingJob(
            entity_type="material_version",
            entity_id=version.id,
            job_type="full_pipeline",
            status="pending",
        )
        db.add(job)
        await flush_or_conflict(db)
        await enqueue_material_ingest(
            arq_pool,
            actor_id=actor.user_id,
            material_version_id=version.id,
            pipeline_run_id=pipeline_run_id,
        )

    return await _present_material(db, material)
