"""Live-progress checkpointing for quiz AI generation pipelines.

The generation pipeline runs inside one ARQ worker task on a single
``AsyncSession`` that does not commit until the run finishes (success or
failure). That means the status-poll endpoint sees nothing but
``status='running'`` for the whole run — no stage, no step, no timing.

This module fixes that WITHOUT perturbing the pipeline's transaction. A
checkpoint is written through a DEDICATED short-lived session with a
targeted ``UPDATE ... SET progress_json = :payload WHERE id = :run_id``
that commits immediately and closes. Because it:

* touches a SEPARATE column (``progress_json``, migration 0035) that the
  pipeline never writes, there is no read-modify-write clobber against
  the pipeline's wholesale ``config_json`` rewrites;
* uses its own session/transaction, a checkpoint commit can land while
  the pipeline's main session is mid-flight, so a poller sees progress
  in real time;
* swallows its own errors, a telemetry write can never fail a real
  generation run.

Payload shape (stable public contract, surfaced by
``QuizGenerationRunRead``)::

    {
      "current_stage": "generation",   # machine key, see STAGES
      "stage_index": 3,                 # 1-based position of current stage
      "total_stages": 6,               # len(STAGES) for this pipeline
      "updated_at": "2026-07-22T...Z", # last checkpoint time
      "events": [                       # append-only, capped
        {"stage": "retrieval", "at": "...Z", "detail": "42 chunks"},
        ...
      ]
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability.logging import get_logger
from abridgeai.core.security import utcnow

if TYPE_CHECKING:
    pass

_logger = get_logger(__name__)

# Ordered stage keys for each pipeline flavour. The stepper UI derives a
# stepped percentage from ``stage_index / total_stages``. Keep these in
# sync with the actual stage call order in full.py / coverage.py /
# regenerate.py.
FULL_STAGES: tuple[str, ...] = (
    "retrieval",
    "ideation",
    "generation",
    "validation",
    "dedup",
    "persistence",
)
COVERAGE_STAGES: tuple[str, ...] = (
    "outline",
    "ideation",
    "generation",
    "validation",
    "dedup",
    "persistence",
)
REGENERATE_STAGES: tuple[str, ...] = (
    "retrieval",
    "generation",
    "validation",
    "persistence",
)

# Cap the append-only event log so a pathological run can't grow the row
# unbounded. The UI shows newest-last; older events beyond the cap are
# dropped from the head.
_MAX_EVENTS = 40


async def record_stage(
    run_id: UUID,
    *,
    stages: tuple[str, ...],
    current_stage: str,
    detail: str | None = None,
) -> None:
    """Write one progress checkpoint for ``run_id`` and commit immediately.

    Best-effort: any exception is logged and swallowed so progress
    telemetry can never fail (or roll back) a live generation run.

    Parameters
    ----------
    run_id
        ``generation_runs.id`` to update.
    stages
        The ordered stage tuple for this pipeline (one of the module
        constants). Drives ``total_stages`` and the ``stage_index`` lookup.
    current_stage
        The stage now starting. Must be a member of ``stages``.
    detail
        Optional human-readable note appended to the event log (e.g.
        "42 chunks", "12 candidates").
    """
    try:
        stage_index = stages.index(current_stage) + 1
    except ValueError:
        # Unknown stage key — record it at an unknown index rather than
        # raising; a mislabelled checkpoint shouldn't crash the run.
        stage_index = 0

    now = utcnow()
    now_iso = now.isoformat()
    event = {"stage": current_stage, "at": now_iso}
    if detail is not None:
        event["detail"] = detail

    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as db:
            # Read the current event list (separate session, so no clobber
            # of the pipeline's config_json writes), append, cap, write back.
            row = (
                await db.execute(
                    text("SELECT progress_json FROM generation_runs WHERE id = :id"),
                    {"id": run_id},
                )
            ).first()
            existing: dict[str, Any] = (row[0] if row and isinstance(row[0], dict) else {}) or {}
            events = list(existing.get("events") or [])
            events.append(event)
            if len(events) > _MAX_EVENTS:
                events = events[-_MAX_EVENTS:]

            payload: dict[str, Any] = {
                "current_stage": current_stage,
                "stage_index": stage_index,
                "total_stages": len(stages),
                "updated_at": now_iso,
                "events": events,
            }
            await db.execute(
                text(
                    "UPDATE generation_runs "
                    "SET progress_json = CAST(:payload AS jsonb) "
                    "WHERE id = :id"
                ),
                {"payload": _json_dumps(payload), "id": run_id},
            )
            await db.commit()
    except Exception as exc:  # noqa: BLE001 -- telemetry must never fail a run
        _logger.warning(
            "generation_progress_checkpoint_failed",
            generation_run_id=str(run_id),
            current_stage=current_stage,
            error=str(exc),
        )


async def record_config_patch(run_id: UUID, patch: dict[str, Any]) -> None:
    """Shallow-merge ``patch`` into ``generation_runs.config_json`` and commit.

    Why this exists (row-lock avoidance)
    ------------------------------------
    The pipeline used to accumulate metadata by mutating the shared-session
    ``run.config_json`` object directly (e.g. retrieval stats, the final
    pipeline summary). Because the session is ``autoflush=False`` and only
    commits when the whole run finishes, that dirty ``run`` row got flushed
    as an ``UPDATE generation_runs ...`` by the *next* explicit ``flush()``
    (an LLM audit-row write) — and the row lock it took was then held for the
    entire duration of the following LLM call. A worker restart mid-call
    orphaned that transaction ``idle in transaction`` with the lock still
    held, wedging the run (and any recovery) forever.

    Writing patches through a DEDICATED short-lived session that commits
    immediately keeps the pipeline's long-lived transaction from ever
    locking the ``generation_runs`` row across an LLM call. Read-modify-write
    (not a blind overwrite) so concurrent progress checkpoints and the
    original request config are preserved; the merge is shallow, matching
    the previous ``run.config_json | {key: value}`` semantics.

    Best-effort on the telemetry side, but note: unlike ``record_stage``
    this carries real run metadata, so failures are logged at WARNING. It
    still never raises — a metadata write must not fail a live run.
    """
    if not patch:
        return
    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as db:
            row = (
                await db.execute(
                    text("SELECT config_json FROM generation_runs WHERE id = :id"),
                    {"id": run_id},
                )
            ).first()
            existing: dict[str, Any] = (row[0] if row and isinstance(row[0], dict) else {}) or {}
            merged = existing | patch
            await db.execute(
                text(
                    "UPDATE generation_runs "
                    "SET config_json = CAST(:payload AS jsonb), updated_at = NOW() "
                    "WHERE id = :id"
                ),
                {"payload": _json_dumps(merged), "id": run_id},
            )
            await db.commit()
    except Exception as exc:  # noqa: BLE001 -- metadata write must never fail a run
        _logger.warning(
            "generation_config_patch_failed",
            generation_run_id=str(run_id),
            keys=sorted(patch.keys()),
            error=str(exc),
        )


def _json_dumps(payload: dict[str, Any]) -> str:
    import json  # noqa: PLC0415

    return json.dumps(payload)


__all__ = [
    "COVERAGE_STAGES",
    "FULL_STAGES",
    "REGENERATE_STAGES",
    "record_config_patch",
    "record_stage",
]
