"""Career readiness snapshot service (FR-6.8).

Readiness v1 = the pathway completion aggregate already computed by
:func:`services.enrollment.get_my_path_progress` (``overall_percent``,
0–100). Snapshots persist that value into ``career_readiness_snapshots``
so managers get historical, org-scoped aggregates with per-student
drill-down — without per-interview rubric details (FR-5.7).

Weighting beyond completion (quiz pass rate, interview outcomes) is a
documented follow-up; keep the formula in ONE place (here) when it
evolves.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import AppError
from abridgeai.core.observability import get_logger
from abridgeai.features.career_paths.queries import authoring as authoring_queries
from abridgeai.features.career_paths.queries import readiness as readiness_queries
from abridgeai.features.career_paths.queries import student as student_queries
from abridgeai.features.career_paths.schemas.authoring import (
    PathReadinessOverview,
    StudentReadinessRead,
)
from abridgeai.features.career_paths.schemas.public import CareerReadinessSnapshotRead
from abridgeai.features.career_paths.services import enrollment as enrollment_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

_TWO_PLACES = Decimal("0.01")


async def compute_readiness_score(
    db: AsyncSession, *, career_path_id: UUID, student_id: UUID
) -> tuple[Decimal, int]:
    """``(score, formula_version)`` for one (student, path) pair.

    Returns the version alongside the score so the caller stamps the snapshot
    with the formula that ACTUALLY produced it rather than a column default.
    """
    progress = await enrollment_service.get_my_path_progress(
        db, career_path_id=career_path_id, student_id=student_id
    )
    score = Decimal(str(progress.overall_percent)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    bounded = max(Decimal("0"), min(Decimal("100"), score))
    return bounded, progress.formula_version


async def snapshot_enrollment(
    db: AsyncSession, *, career_path_id: UUID, student_id: UUID
) -> Decimal:
    """Compute + persist one snapshot; returns the stored score.

    ``formula_version`` is written EXPLICITLY from the version that produced
    the score, never left to the column default. The column defaults to 1 and
    the setting starts at 1, but relying on that coupling is exactly the bug
    that would mislabel every snapshot in the window between deploying
    formula 2 and flipping the setting.

    Also flips the enrollment to ``completed`` ("prepared") once the score
    reaches 100 — a backstop to the synchronous writer, not the primary path.
    """
    score, formula_version = await compute_readiness_score(
        db, career_path_id=career_path_id, student_id=student_id
    )
    # Gap 3: the snapshot records which version produced the score — the
    # student's pin, or the latest published version when unenrolled.
    enrollment = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    version_id = enrollment.version_id if enrollment is not None else None
    if version_id is None:
        published = await authoring_queries.get_published_version(db, career_path_id)
        if published is None:
            raise AppError(
                f"Career path {career_path_id} has no published version to snapshot against"
            )
        version_id = published.id
    await readiness_queries.insert_snapshot(
        db,
        student_id=student_id,
        career_path_id=career_path_id,
        version_id=version_id,
        readiness_score=score,
        formula_version=formula_version,
    )
    await enrollment_service.sync_enrollment_completion(
        db,
        career_path_id=career_path_id,
        student_id=student_id,
        overall_percent=float(score),
    )
    return score


async def snapshot_all_active_enrollments(db: AsyncSession) -> int:
    """Snapshot every active career enrollment; returns count written.

    Per-pair failures are logged and skipped so one bad enrollment never
    aborts the nightly batch.
    """
    pairs = await readiness_queries.list_active_enrollment_pairs(db)
    written = 0
    for student_id, career_path_id, _version_id in pairs:
        try:
            # SAVEPOINT per pair: a DB-level error (FK violation etc.)
            # rolls back only this pair instead of poisoning the session
            # and losing the whole nightly batch at final commit.
            async with db.begin_nested():
                await snapshot_enrollment(db, career_path_id=career_path_id, student_id=student_id)
            written += 1
        except Exception:
            logger.exception(
                "career_readiness.snapshot_failed",
                student_id=str(student_id),
                career_path_id=str(career_path_id),
            )
    return written


async def get_my_readiness_history(
    db: AsyncSession, *, student_id: UUID, career_path_id: UUID, limit: int = 30
) -> list[CareerReadinessSnapshotRead]:
    snapshots = await readiness_queries.list_my_snapshots(
        db, student_id=student_id, career_path_id=career_path_id, limit=limit
    )
    return [CareerReadinessSnapshotRead.model_validate(s) for s in snapshots]


async def get_path_readiness_overview(
    db: AsyncSession, career_path_id: UUID
) -> PathReadinessOverview:
    """Latest snapshot per actively-enrolled student + path average.

    Drill-down carries score/email only — never per-interview rubric
    details (FR-6.8 exclusion).
    """
    rows = await readiness_queries.latest_snapshots_for_path(db, career_path_id)
    students = [StudentReadinessRead.model_validate(row) for row in rows]
    average = float(sum(s.readiness_score for s in students) / len(students)) if students else None
    return PathReadinessOverview(
        career_path_id=career_path_id,
        average_score=average,
        student_count=len(students),
        students=students,
    )


__all__ = [
    "compute_readiness_score",
    "get_my_readiness_history",
    "get_path_readiness_overview",
    "snapshot_all_active_enrollments",
    "snapshot_enrollment",
]
