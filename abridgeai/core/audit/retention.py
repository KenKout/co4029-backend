"""Retention sweep for the append-only audit stores.

Why this exists
---------------
Migration 0105 made the audit tables append-only and 0106 extended that to the
assessment stores, which settled half of the problem: nothing can rewrite
history. The other half was still open — nothing DELETED anything either. The
HTTP audit middleware writes one row per non-skipped request and no cron job
touched ``http_audit_log``, so it grew without bound (measured at 127k rows /
87 MB after ~3.5 months on a single dev box). "Immutable" and "prunable" are
both requirements; ``core.audit.maintenance.audit_maintenance`` was written for
exactly this caller and had none in production until now.

Per-table windows, not one global knob: an HTTP request row is operational
noise with a short useful life, while a proctoring signal backs an integrity
review and may need to outlast an appeal. Each window is a runtime setting so an
operator retunes it without a deploy, and ``0`` means "keep forever" so the
sweep can be turned off per table without editing code.

Deliberately global-scope settings: retention is a property of the deployment's
storage and compliance posture, not of one tenant, and the tables are not all
organization-scoped anyway (``http_audit_log`` has no ``organization_id``).

Safety
------
* Deletion happens inside :func:`audit_maintenance`, which the trigger requires.
  The grant is ``SET LOCAL`` so it dies with the transaction.
* Batched with a bounded loop. A first prune on a long-lived table can span
  hundreds of thousands of rows; one unbounded ``DELETE`` would hold locks and
  bloat WAL for minutes. Each batch is its own transaction, so an interrupted
  sweep leaves consistent state and the next run continues.
* Table names come from this module's own constant, never from a caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text

from abridgeai.core.audit.maintenance import audit_maintenance
from abridgeai.core.observability import get_logger
from abridgeai.core.runtime_settings import resolve_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_logger = get_logger(__name__)

# Rows removed per statement. Large enough that a big backlog clears in a
# reasonable number of round trips, small enough that each transaction is short.
_BATCH_SIZE = 5_000

# Hard ceiling on batches per table per run, so a pathological backlog cannot
# turn a nightly job into an all-night job. Whatever is left is picked up
# tomorrow: 5k * 200 = 1M rows per table per run.
_MAX_BATCHES = 200


@dataclass(frozen=True)
class RetainedTable:
    """One append-only table and the setting that governs its window."""

    table: str
    timestamp_column: str
    setting_key: str


# Every table guarded by ``audit_log_immutable`` (migrations 0105 + 0106).
# Adding a table here without a corresponding trigger would still work, but the
# reverse — a guarded table with no entry here — is the bug this list exists to
# prevent, and ``test_audit_retention`` asserts the two agree.
RETAINED_TABLES: tuple[RetainedTable, ...] = (
    RetainedTable(
        table="http_audit_log",
        timestamp_column="created_at",
        setting_key="audit.http_log_retention_days",
    ),
    RetainedTable(
        table="assessment_integrity_events",
        timestamp_column="created_at",
        setting_key="audit.integrity_event_retention_days",
    ),
    RetainedTable(
        table="quiz_audit_events",
        timestamp_column="created_at",
        setting_key="audit.quiz_event_retention_days",
    ),
)


async def _prune_one(
    session_factory: async_sessionmaker[AsyncSession],
    spec: RetainedTable,
    retention_days: int,
) -> int:
    """Delete rows older than the window in bounded batches. Returns the count.

    ``ctid`` sub-select rather than ``DELETE ... LIMIT`` (which Postgres does not
    support) — it also avoids depending on a primary key, which
    ``assessment_integrity_events`` does have but which is not needed here.
    """
    deleted = 0
    for _ in range(_MAX_BATCHES):
        async with session_factory() as session, session.begin():
            # The scope must be entered in the SAME transaction as the
            # delete: SET LOCAL reverts at commit.
            await audit_maintenance(session)
            result = await session.execute(
                text(
                    f"DELETE FROM {spec.table} "  # noqa: S608 -- constant from RETAINED_TABLES
                    f"WHERE ctid IN ("
                    f"  SELECT ctid FROM {spec.table} "
                    f"  WHERE {spec.timestamp_column} < NOW() - CAST(:days AS interval) "
                    f"  LIMIT {_BATCH_SIZE}"
                    f")"
                ),
                {"days": f"{retention_days} days"},
            )
        batch = result.rowcount or 0
        deleted += batch
        if batch < _BATCH_SIZE:
            break
    else:
        # Loop exhausted without a short batch: more remains than one run
        # removes. Worth surfacing — either the window was just shortened
        # dramatically, or write volume outpaces the sweep.
        _logger.warning(
            "audit.retention_batch_ceiling_reached",
            table=spec.table,
            deleted=deleted,
            max_batches=_MAX_BATCHES,
        )
    return deleted


async def prune_audit_logs(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, int]:
    """Prune every retained audit table. Returns ``{table: rows_deleted}``.

    Each table is independent: a failure on one is logged and the rest still
    run, because a single unprunable table must not stop the database from being
    kept in shape. A window of ``0`` skips the table entirely (keep forever).
    """
    async with session_factory() as session:
        settings = await resolve_settings(session, None)

    results: dict[str, int] = {}
    for spec in RETAINED_TABLES:
        days = int(settings[spec.setting_key])
        if days <= 0:
            results[spec.table] = 0
            continue
        try:
            results[spec.table] = await _prune_one(session_factory, spec, days)
        except Exception:
            _logger.exception("audit.retention_failed", table=spec.table)
            results[spec.table] = 0
    return results


__all__ = ["RETAINED_TABLES", "RetainedTable", "prune_audit_logs"]
