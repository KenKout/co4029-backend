"""Backfill ``interview_questions.embedding`` for rows that never got one.

Why this exists
---------------
``interview_questions.embedding`` was added by migration ``0063_iq_embedding``
as nullable, and only two code paths ever populated it: ``add_question`` and
``update_question``. The AI generation pipeline (``ai/pipelines/generation.py``)
inserted its rows directly and did not, so every question a teacher generated
landed with ``embedding = NULL``.

Because ``shortlist_similar_questions`` filters on ``embedding IS NOT NULL``,
those rows were invisible to duplicate detection: a bank that had never been
hand-edited answered "not a duplicate" to everything, while the feature looked
enabled. The pipeline gap is fixed at the source; this script repairs the rows
created before the fix.

Usage
-----
    # from backend/, venv active
    python -m scripts.backfill_interview_question_embeddings --dry-run
    python -m scripts.backfill_interview_question_embeddings
    python -m scripts.backfill_interview_question_embeddings --config-id <uuid>

Requires ``INTERVIEW_DEDUP_ENABLED=true`` — the shared helper is gated on it, so
with the flag off there is nothing to backfill for. Costs one embedding call per
``--batch-size`` questions. Idempotent: only ``embedding IS NULL`` rows are read,
so a re-run after a partial failure picks up exactly what is left.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.config import get_settings
from abridgeai.core.db import close_db, get_sessionmaker
from abridgeai.features.interviews.dedup import store_question_embeddings

logger = logging.getLogger("backfill_iq_embeddings")

# One provider call per batch. Kept modest so a single failure loses little work
# and so we stay well inside provider per-request input limits.
DEFAULT_BATCH_SIZE = 32


async def _fetch_missing(
    db: AsyncSession, *, config_id: UUID | None, limit: int
) -> list[tuple[UUID, str]]:
    """Oldest-first page of questions with no embedding."""
    sql = (
        "SELECT id, prompt_text FROM interview_questions "
        "WHERE embedding IS NULL AND deleted_at IS NULL "
        "  AND prompt_text IS NOT NULL AND btrim(prompt_text) <> '' "
    )
    params: dict[str, object] = {"limit": limit}
    if config_id is not None:
        sql += "  AND interview_config_id = :config_id "
        params["config_id"] = config_id
    sql += "ORDER BY created_at LIMIT :limit"
    rows = (await db.execute(text(sql), params)).all()
    return [(row[0], row[1]) for row in rows]


async def _count_missing(db: AsyncSession, *, config_id: UUID | None) -> int:
    sql = (
        "SELECT count(*) FROM interview_questions "
        "WHERE embedding IS NULL AND deleted_at IS NULL "
        "  AND prompt_text IS NOT NULL AND btrim(prompt_text) <> '' "
    )
    params: dict[str, object] = {}
    if config_id is not None:
        sql += "  AND interview_config_id = :config_id"
        params["config_id"] = config_id
    return int((await db.execute(text(sql), params)).scalar_one())


async def run(
    *, config_id: UUID | None, batch_size: int, dry_run: bool
) -> tuple[int, int]:
    """Returns ``(attempted, stored)``."""
    settings = get_settings()
    if not settings.interview_dedup_enabled:
        logger.error(
            "INTERVIEW_DEDUP_ENABLED is false — the embedding helper is gated on "
            "it and would be a no-op. Enable the flag before backfilling."
        )
        return (0, 0)

    sessionmaker = get_sessionmaker()
    attempted = 0
    stored = 0

    async with sessionmaker() as db:
        total = await _count_missing(db, config_id=config_id)
        logger.info("questions missing an embedding: %d", total)
        if total == 0:
            return (0, 0)
        if dry_run:
            rows = await _fetch_missing(db, config_id=config_id, limit=total)
            for qid, prompt in rows:
                logger.info("would embed %s  %s", qid, prompt[:70])
            return (len(rows), 0)

    while True:
        # A fresh session per batch: one bad batch can't poison a long run, and
        # each batch commits on its own so progress survives an interruption.
        async with sessionmaker() as db:
            rows = await _fetch_missing(db, config_id=config_id, limit=batch_size)
            if not rows:
                break
            attempted += len(rows)
            written = await store_question_embeddings(
                db,
                question_ids=[qid for qid, _ in rows],
                prompt_texts=[prompt for _, prompt in rows],
            )
            await db.commit()
            stored += written
            logger.info("batch: %d/%d embedded (running total %d)", written, len(rows), stored)
            if written == 0:
                # store_question_embeddings swallows provider errors and returns
                # 0. Without this guard the same page would be re-read forever.
                logger.error(
                    "batch stored 0 embeddings — provider likely failing; stopping. "
                    "Re-run once the provider is healthy."
                )
                break

    return (attempted, stored)


async def _main_async(
    *, config_id: UUID | None, batch_size: int, dry_run: bool
) -> tuple[int, int]:
    """Run the backfill, always disposing the engine so the process can exit."""
    try:
        return await run(
            config_id=config_id, batch_size=batch_size, dry_run=dry_run
        )
    finally:
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-id",
        type=UUID,
        default=None,
        help="Restrict to one interview config (default: every config).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be embedded; make no provider call and no write.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    attempted, stored = asyncio.run(
        _main_async(
            config_id=args.config_id,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )
    logger.info("done: attempted=%d stored=%d", attempted, stored)
    # Non-zero when work was found but nothing landed, so a wrapper/cron notices.
    return 0 if (stored > 0 or attempted == 0 or args.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())
