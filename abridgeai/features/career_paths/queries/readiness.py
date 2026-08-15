"""Query layer for career readiness snapshots (FR-6.8).

All SQLAlchemy access for the readiness feature lives here so the
service layer stays inside the "Services do not touch SQLAlchemy"
import-linter contract. The latest-per-student read uses Postgres
``DISTINCT ON``; the email join follows the raw-SQL precedent of
``queries/student.py`` (cross-feature ``users`` read).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, text

from abridgeai.features.career_paths.models import (
    CareerReadinessSnapshot,
    StudentCareerEnrollment,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def insert_snapshot(
    db: AsyncSession,
    *,
    student_id: UUID,
    career_path_id: UUID,
    version_id: UUID,
    readiness_score: Decimal,
    formula_version: int = 1,
) -> CareerReadinessSnapshot:
    """Append one readiness snapshot.

    ``formula_version`` must be passed explicitly by the caller with the
    version that actually produced ``readiness_score``. The default of 1
    matches the column default and the setting's default, but a caller that
    relies on it during a cutover would mislabel its snapshots.

    ``version_id`` (Gap 3) records WHICH version of the path the score
    measures — a score's meaning is version-dependent.
    """
    snapshot = CareerReadinessSnapshot(
        student_id=student_id,
        career_path_id=career_path_id,
        version_id=version_id,
        readiness_score=readiness_score,
        formula_version=formula_version,
    )
    db.add(snapshot)
    await db.flush()
    return snapshot


async def list_active_enrollment_pairs(db: AsyncSession) -> list[tuple[UUID, UUID, UUID]]:
    """All ``(student_id, career_path_id, version_id)`` triples with an active
    enrollment — the version pin so snapshots record which route produced
    the score (Gap 3)."""
    stmt = select(
        StudentCareerEnrollment.student_id,
        StudentCareerEnrollment.career_path_id,
        StudentCareerEnrollment.version_id,
    ).where(
        StudentCareerEnrollment.status == "active",
        StudentCareerEnrollment.deleted_at.is_(None),
    )
    rows = (await db.execute(stmt)).all()
    return [(row.student_id, row.career_path_id, row.version_id) for row in rows]


async def list_my_snapshots(
    db: AsyncSession,
    *,
    student_id: UUID,
    career_path_id: UUID,
    limit: int = 30,
) -> list[CareerReadinessSnapshot]:
    """Most-recent-first snapshot history for one (student, path) pair."""
    stmt = (
        select(CareerReadinessSnapshot)
        .where(
            CareerReadinessSnapshot.student_id == student_id,
            CareerReadinessSnapshot.career_path_id == career_path_id,
        )
        .order_by(CareerReadinessSnapshot.captured_at.desc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


async def latest_snapshots_for_path(db: AsyncSession, career_path_id: UUID) -> list[dict[str, Any]]:
    """Latest snapshot per actively-enrolled student, with email.

    Joins ``users`` for ``primary_email`` (raw-SQL cross-feature read,
    same precedent as the roster queries). Students enrolled after the
    last cron run simply have no row yet — callers treat absence as
    "no snapshot", not as score 0.
    """
    result = await db.execute(
        text(
            "SELECT DISTINCT ON (s.student_id) "
            "       s.student_id, u.primary_email, s.readiness_score, s.captured_at "
            "FROM career_readiness_snapshots s "
            "JOIN users u ON u.id = s.student_id "
            "JOIN student_career_enrollments e "
            "  ON e.student_id = s.student_id "
            " AND e.career_path_id = s.career_path_id "
            " AND e.status = 'active' "
            " AND e.deleted_at IS NULL "
            "WHERE s.career_path_id = :path_id "
            "ORDER BY s.student_id, s.captured_at DESC, s.id DESC"
        ),
        {"path_id": str(career_path_id)},
    )
    return [
        {
            "student_id": row.student_id,
            "student_email": row.primary_email,
            "readiness_score": row.readiness_score,
            "captured_at": row.captured_at,
        }
        for row in result.all()
    ]


__all__ = [
    "insert_snapshot",
    "latest_snapshots_for_path",
    "list_active_enrollment_pairs",
    "list_my_snapshots",
]
