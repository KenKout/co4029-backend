"""Knowledge-graph rebuild from current materials (T9.2).

Walks every ``LearningMaterialVersion`` row whose ``processing_status='ready'``
and re-runs the per-chunk KG extraction through
:func:`abridgeai.ai.knowledge_graph.build_knowledge_graph_for_material_version`.
The Cypher writes are MERGE-keyed on ``Concept.name_norm`` so re-running is
additive — the script never wipes Neo4j.

Operational pattern (Phase 9 cutover):

* **Pre-flight** — count ``Concept`` nodes in Neo4j (skipped on ``--dry-run``).
* **Walk** — stream candidates from Postgres ordered by ``uploaded_at DESC``.
* **Rebuild** — run up to ``--workers`` versions concurrently
  (``asyncio.Semaphore``); failures are logged and skipped, never aborting
  the run.
* **Budget guard** — after each version completes, sum
  ``ai_model_calls.estimated_cost_usd`` for this run's ``pipeline_run_id``
  set; abort with exit code 2 if ``--budget-usd`` is exceeded.
* **Post-flight** — recount nodes; report processed / failed / total cost
  / pre / post / delta_pct. Exit code 0 if delta within ±5% (Metis
  acceptance), 1 otherwise.

Invocation::

    cd backend-new && uv run python scripts/rebuild_knowledge_graph.py \\
        --workers 2 --max-materials 5 --dry-run

Idempotency follows from the upstream Cypher MERGE on ``name_norm``; we
never DELETE or DETACH DELETE — re-running is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from abridgeai.ai.knowledge_graph import build_knowledge_graph_for_material_version
from abridgeai.ai.knowledge_graph.builder import EnrichedChunk, HierarchyPayload
from abridgeai.ai.llm.gateway import LLMGateway
from abridgeai.core.db import close_db, get_sessionmaker
from abridgeai.core.observability import configure_structlog, get_logger
from abridgeai.features.materials.models import (
    DocumentChunk,
    LearningMaterialVersion,
)
from abridgeai.infrastructure.neo4j import (
    KnowledgeGraphClient,
    close_neo4j,
    get_neo4j_driver,
)

# Module-level pipeline_run namespace UUID — every per-version run gets a
# distinct UUID inside this script invocation; the namespace simply makes
# log scrubbing easier (search for ``rebuild_kg_run_id``).
DELTA_TOLERANCE_PCT: float = 5.0
EXIT_OK = 0
EXIT_DELTA_OUT_OF_TOLERANCE = 1
EXIT_BUDGET_EXCEEDED = 2

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency-injection seams (tests swap these)
# ---------------------------------------------------------------------------


BuilderFn = Callable[
    ...,
    Awaitable[Any],
]
"""Signature-compatible with :func:`build_knowledge_graph_for_material_version`."""


ConceptCountFn = Callable[[], Awaitable[int]]
"""Returns the number of ``:Concept`` nodes currently in Neo4j."""


KGClientFactory = Callable[[], KnowledgeGraphClient]
"""Synchronous factory returning a ``KnowledgeGraphClient``."""


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


@dataclass
class RebuildArgs:
    """Parsed CLI arguments. Decoupled from ``argparse.Namespace`` for typing."""

    workers: int
    max_materials: int | None
    dry_run: bool
    since: date | None
    material_id: UUID | None
    budget_usd: Decimal | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rebuild_knowledge_graph",
        description=(
            "Rebuild the Neo4j knowledge graph from every ready "
            "LearningMaterialVersion. Idempotent (MERGE on name_norm)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent rebuild tasks (default 1). Each worker holds one "
        "LLMGateway slot; raise carefully under provider rate limits.",
    )
    parser.add_argument(
        "--max-materials",
        type=int,
        default=None,
        help="Cap on number of versions processed (default unbounded).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidate version IDs and exit without LLM/Neo4j writes.",
    )
    parser.add_argument(
        "--since",
        type=_parse_date,
        default=None,
        help="Only include versions with uploaded_at >= YYYY-MM-DD.",
    )
    parser.add_argument(
        "--material-id",
        type=UUID,
        default=None,
        help="Restrict to one specific learning_material_versions.id (debugging).",
    )
    parser.add_argument(
        "--budget-usd",
        type=Decimal,
        default=None,
        help="Abort if estimated total cost (USD) exceeds this value after any version completes.",
    )
    return parser


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_args(argv: list[str] | None = None) -> RebuildArgs:
    parsed = _build_parser().parse_args(argv)
    if parsed.workers < 1:
        raise SystemExit(f"--workers must be >= 1 (got {parsed.workers})")
    if parsed.max_materials is not None and parsed.max_materials < 1:
        raise SystemExit(f"--max-materials must be >= 1 (got {parsed.max_materials})")
    return RebuildArgs(
        workers=parsed.workers,
        max_materials=parsed.max_materials,
        dry_run=bool(parsed.dry_run),
        since=parsed.since,
        material_id=parsed.material_id,
        budget_usd=parsed.budget_usd,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    version_id: UUID
    material_id: UUID
    uploaded_at: datetime


async def _discover_candidates(
    db: AsyncSession,
    args: RebuildArgs,
) -> list[_Candidate]:
    """Return ready, non-deleted versions ordered newest-first."""
    stmt = (
        select(
            LearningMaterialVersion.id,
            LearningMaterialVersion.material_id,
            LearningMaterialVersion.uploaded_at,
        )
        .where(LearningMaterialVersion.processing_status == "ready")
        .where(LearningMaterialVersion.deleted_at.is_(None))
        .order_by(LearningMaterialVersion.uploaded_at.desc())
    )
    if args.material_id is not None:
        stmt = stmt.where(LearningMaterialVersion.id == args.material_id)
    if args.since is not None:
        stmt = stmt.where(LearningMaterialVersion.uploaded_at >= args.since)
    if args.max_materials is not None:
        stmt = stmt.limit(args.max_materials)

    result = await db.execute(stmt)
    return [
        _Candidate(version_id=row[0], material_id=row[1], uploaded_at=row[2])
        for row in result.all()
    ]


# ---------------------------------------------------------------------------
# Hierarchy + chunks loaders
# ---------------------------------------------------------------------------


_HIERARCHY_SQL = text(
    """
    SELECT
        c.id            AS course_id,
        c.title         AS course_title,
        m.id            AS module_id,
        m.title         AS module_title,
        l.id            AS lesson_id,
        l.title         AS lesson_title,
        lm.id           AS material_id,
        lm.title        AS material_title,
        lm.material_type AS material_type
    FROM learning_material_versions lmv
    JOIN learning_materials lm ON lm.id = lmv.material_id
    JOIN lessons l             ON l.id = lm.lesson_id
    JOIN modules m             ON m.id = l.module_id
    JOIN courses c             ON c.id = m.course_id
    WHERE lmv.id = :version_id
    """
)


@dataclass(frozen=True)
class _Hierarchy:
    course_id: UUID
    course_title: str
    module_id: UUID
    module_title: str
    lesson_id: UUID
    lesson_title: str
    material_id: UUID
    material_title: str
    material_type: str


async def _load_hierarchy(db: AsyncSession, version_id: UUID) -> _Hierarchy:
    row = (await db.execute(_HIERARCHY_SQL, {"version_id": version_id})).first()
    if row is None:
        raise LookupError(f"learning_material_version {version_id} not found")
    return _Hierarchy(
        course_id=row.course_id,
        course_title=row.course_title or "",
        module_id=row.module_id,
        module_title=row.module_title or "",
        lesson_id=row.lesson_id,
        lesson_title=row.lesson_title or "",
        material_id=row.material_id,
        material_title=row.material_title or "",
        material_type=row.material_type or "",
    )


@dataclass(frozen=True)
class _ChunkView:
    id: UUID
    chunk_index: int
    content: str
    material_version_id: UUID


async def _load_chunks(db: AsyncSession, version_id: UUID) -> list[_ChunkView]:
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.material_version_id,
        )
        .where(DocumentChunk.material_version_id == version_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    result = await db.execute(stmt)
    return [
        _ChunkView(
            id=row[0],
            chunk_index=row[1],
            content=row[2],
            material_version_id=row[3],
        )
        for row in result.all()
    ]


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


async def _running_cost_usd(
    db: AsyncSession,
    pipeline_run_ids: list[UUID],
) -> Decimal:
    """Sum ``ai_model_calls.estimated_cost_usd`` for the given runs.

    Returns ``Decimal('0')`` when nothing has been spent yet (or no rows
    were found). Casts ``UUID`` → ``str`` because asyncpg/psycopg-async
    bind parameters require explicit casting on SOME drivers when using
    raw text + array; using ``= ANY(:ids)`` keeps the path
    parameterised.
    """
    if not pipeline_run_ids:
        return Decimal("0")
    rows = await db.execute(
        text(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) AS total "
            "FROM ai_model_calls "
            "WHERE pipeline_run_id = ANY(:ids)"
        ),
        {"ids": [str(rid) for rid in pipeline_run_ids]},
    )
    row = rows.one()
    total = row.total
    if total is None:
        return Decimal("0")
    return Decimal(str(total))


# ---------------------------------------------------------------------------
# Neo4j ``Concept`` count helper (default ConceptCountFn)
# ---------------------------------------------------------------------------


async def _default_concept_count() -> int:
    driver = get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run("MATCH (n:Concept) RETURN count(n) AS c")
        record = await result.single()
        if record is None:
            return 0
        return int(record["c"])


# ---------------------------------------------------------------------------
# Per-version rebuild
# ---------------------------------------------------------------------------


@dataclass
class _RunStats:
    processed: int = 0
    failed: int = 0
    failed_ids: list[UUID] = field(default_factory=list)
    pipeline_run_ids: list[UUID] = field(default_factory=list)


async def _rebuild_one(
    sessionmaker: async_sessionmaker[AsyncSession],
    candidate: _Candidate,
    *,
    builder: BuilderFn,
    kg_client: KnowledgeGraphClient,
    llm_gateway: LLMGateway,
    pipeline_run_id: UUID,
) -> None:
    """Run one rebuild inside its own session/commit so failures don't poison the txn.

    We open a fresh ``AsyncSession`` per version so a failure rolls back
    only that version's audit rows; the next version starts clean.
    """
    async with sessionmaker() as db:
        hierarchy = await _load_hierarchy(db, candidate.version_id)
        chunks = await _load_chunks(db, candidate.version_id)
        # KG builder accepts EnrichedChunk + HierarchyPayload Protocols; our
        # frozen dataclasses match structurally.
        await builder(
            candidate.version_id,
            list(chunks),
            hierarchy=hierarchy,
            pipeline_run_id=pipeline_run_id,
            db=db,
            kg_client=kg_client,
            llm_gateway=llm_gateway,
            parent_job_id=None,
        )
        await db.commit()


# Mypy / ruff: structural-typing assertion. ``_ChunkView`` and ``_Hierarchy``
# satisfy the KG builder's ``EnrichedChunk`` / ``HierarchyPayload`` Protocols
# at runtime; the cast helpers below make that explicit for static analysis.
def _as_enriched(chunk: _ChunkView) -> EnrichedChunk:  # pragma: no cover
    return chunk  # type: ignore[return-value]


def _as_hierarchy(h: _Hierarchy) -> HierarchyPayload:  # pragma: no cover
    return h  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _Report:
    total_candidates: int
    processed: int
    failed: int
    failed_ids: list[UUID]
    total_cost_usd: Decimal
    pre_count: int | None
    post_count: int | None
    delta_pct: float | None
    budget_exceeded: bool
    dry_run: bool


def _delta_pct(pre: int, post: int) -> float:
    if pre == 0:
        return 0.0 if post == 0 else 100.0
    return ((post - pre) / pre) * 100.0


def _format_report(report: _Report) -> str:
    lines = [
        "=" * 70,
        "Knowledge graph rebuild — final report",
        "=" * 70,
        f"  candidates:        {report.total_candidates}",
        f"  processed:         {report.processed}",
        f"  failed:            {report.failed}",
        f"  total cost (USD):  {report.total_cost_usd}",
    ]
    if report.dry_run:
        lines.append("  mode:              dry-run (no Neo4j / LLM writes)")
    else:
        lines.append(
            f"  pre Concept count: {report.pre_count if report.pre_count is not None else 'n/a'}"
        )
        lines.append(
            f"  post Concept count:{report.post_count if report.post_count is not None else 'n/a'}"
        )
        if report.delta_pct is not None:
            lines.append(f"  delta:             {report.delta_pct:+.2f}%")
    if report.budget_exceeded:
        lines.append("  status:            BUDGET EXCEEDED — aborted")
    if report.failed_ids:
        lines.append("  failed version ids:")
        for vid in report.failed_ids[:20]:
            lines.append(f"    - {vid}")
        if len(report.failed_ids) > 20:
            lines.append(f"    ... ({len(report.failed_ids) - 20} more)")
    lines.append("=" * 70)
    return "\n".join(lines)


@dataclass
class _BudgetState:
    exceeded: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def _check_budget(
    sessionmaker: async_sessionmaker[AsyncSession],
    stats: _RunStats,
    budget: _BudgetState,
    *,
    budget_usd: Decimal,
    log: structlog.stdlib.BoundLogger,
) -> None:
    async with budget.lock:
        if budget.exceeded:
            return
        async with sessionmaker() as cost_db:
            cost = await _running_cost_usd(cost_db, list(stats.pipeline_run_ids))
        if cost > budget_usd:
            budget.exceeded = True
            log.warning(
                "rebuild_kg_budget_exceeded",
                cost_usd=str(cost),
                budget_usd=str(budget_usd),
            )


async def _run_one_worker(
    cand: _Candidate,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    builder: BuilderFn,
    kg_client: KnowledgeGraphClient,
    gateway: LLMGateway,
    semaphore: asyncio.Semaphore,
    stats: _RunStats,
    budget: _BudgetState,
    budget_usd: Decimal | None,
    log: structlog.stdlib.BoundLogger,
) -> None:
    if budget.exceeded:
        return
    async with semaphore:
        if budget.exceeded:
            return
        run_id = uuid4()
        try:
            await _rebuild_one(
                sessionmaker,
                cand,
                builder=builder,
                kg_client=kg_client,
                llm_gateway=gateway,
                pipeline_run_id=run_id,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            stats.failed += 1
            stats.failed_ids.append(cand.version_id)
            log.exception(
                "rebuild_kg_version_failed",
                version_id=str(cand.version_id),
                error=str(exc),
            )
            return
        stats.processed += 1
        stats.pipeline_run_ids.append(run_id)
        log.info(
            "rebuild_kg_version_done",
            version_id=str(cand.version_id),
            processed=stats.processed,
            failed=stats.failed,
        )
        if budget_usd is not None:
            await _check_budget(sessionmaker, stats, budget, budget_usd=budget_usd, log=log)


def _empty_report(*, total: int, dry_run: bool) -> _Report:
    return _Report(
        total_candidates=total,
        processed=0,
        failed=0,
        failed_ids=[],
        total_cost_usd=Decimal("0"),
        pre_count=None,
        post_count=None,
        delta_pct=None,
        budget_exceeded=False,
        dry_run=dry_run,
    )


def _emit_dry_run(candidates: list[_Candidate]) -> _Report:
    sys.stdout.write(f"[rebuild_kg] DRY RUN — would process {len(candidates)} version(s):\n")
    for cand in candidates:
        sys.stdout.write(
            f"  - {cand.version_id} (material={cand.material_id}, "
            f"uploaded_at={cand.uploaded_at.isoformat()})\n"
        )
    report = _empty_report(total=len(candidates), dry_run=True)
    sys.stdout.write(_format_report(report) + "\n")
    return report


async def run(
    args: RebuildArgs,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    builder: BuilderFn,
    kg_client_factory: KGClientFactory | None,
    concept_count_fn: ConceptCountFn,
    llm_gateway: LLMGateway | None = None,
) -> tuple[int, _Report]:
    """Drive the rebuild. Returns ``(exit_code, report)``.

    Splitting argument parsing from execution lets tests construct
    ``RebuildArgs`` directly and inject ``builder`` / ``concept_count_fn``
    fakes without spawning subprocesses.
    """
    log = _logger.bind(workers=args.workers, dry_run=args.dry_run)

    async with sessionmaker() as db:
        candidates = await _discover_candidates(db, args)

    log.info(
        "rebuild_kg_candidates_loaded",
        total=len(candidates),
        max_materials=args.max_materials,
        since=args.since.isoformat() if args.since is not None else None,
        material_id=str(args.material_id) if args.material_id else None,
    )

    if args.dry_run:
        return EXIT_OK, _emit_dry_run(candidates)

    if not candidates:
        log.warning("rebuild_kg_no_candidates")
        report = _empty_report(total=0, dry_run=False)
        sys.stdout.write(_format_report(report) + "\n")
        return EXIT_OK, report

    pre_count = await concept_count_fn()
    log.info("rebuild_kg_pre_count", concept_count=pre_count)

    if kg_client_factory is None:
        raise RuntimeError("kg_client_factory is required when not in dry-run mode")
    kg_client = kg_client_factory()
    gateway = llm_gateway or LLMGateway()

    stats = _RunStats()
    budget = _BudgetState()
    semaphore = asyncio.Semaphore(args.workers)

    await asyncio.gather(
        *(
            _run_one_worker(
                c,
                sessionmaker=sessionmaker,
                builder=builder,
                kg_client=kg_client,
                gateway=gateway,
                semaphore=semaphore,
                stats=stats,
                budget=budget,
                budget_usd=args.budget_usd,
                log=log,
            )
            for c in candidates
        )
    )

    post_count = await concept_count_fn()
    log.info("rebuild_kg_post_count", concept_count=post_count)

    async with sessionmaker() as cost_db:
        total_cost = await _running_cost_usd(cost_db, stats.pipeline_run_ids)

    delta = _delta_pct(pre_count, post_count)
    report = _Report(
        total_candidates=len(candidates),
        processed=stats.processed,
        failed=stats.failed,
        failed_ids=list(stats.failed_ids),
        total_cost_usd=total_cost,
        pre_count=pre_count,
        post_count=post_count,
        delta_pct=delta,
        budget_exceeded=budget.exceeded,
        dry_run=False,
    )
    sys.stdout.write(_format_report(report) + "\n")

    if budget.exceeded:
        return EXIT_BUDGET_EXCEEDED, report
    if abs(delta) > DELTA_TOLERANCE_PCT and stats.processed > 0:
        log.warning(
            "rebuild_kg_delta_out_of_tolerance",
            delta_pct=delta,
            tolerance_pct=DELTA_TOLERANCE_PCT,
        )
        return EXIT_DELTA_OUT_OF_TOLERANCE, report
    return EXIT_OK, report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _default_kg_client_factory() -> KnowledgeGraphClient:
    return KnowledgeGraphClient(get_neo4j_driver())


async def _async_main(argv: list[str] | None) -> int:
    args = parse_args(argv)
    configure_structlog()
    sm = get_sessionmaker()

    builder: BuilderFn = build_knowledge_graph_for_material_version
    factory: KGClientFactory | None = None if args.dry_run else _default_kg_client_factory

    try:
        exit_code, _ = await run(
            args,
            sessionmaker=sm,
            builder=builder,
            kg_client_factory=factory,
            concept_count_fn=_default_concept_count,
        )
    finally:
        # Best-effort cleanup. Failures to close are logged at the driver
        # level; we do NOT mask the rebuild's exit code.
        try:
            await close_neo4j()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            structlog.get_logger(__name__).exception("rebuild_kg_close_neo4j_failed")
        try:
            await close_db()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            structlog.get_logger(__name__).exception("rebuild_kg_close_db_failed")

    return exit_code


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
