"""Re-grade one interview session through the production evaluation path.

For sessions whose verdict was produced by a defective grader (e.g. the
voice-transcript eligibility bug where ``_is_candidate_answer`` dropped every
``record_turn`` row): reset the published verdict, then re-enqueue the ARQ
evaluation job — the worker re-runs outcome verdicts, rubric scoring and the
gap report, upserting over the stale rows (both persistence writers are
ON CONFLICT DO UPDATE).

Refuses to run when the session holds a live evaluation claim, so a regrade
can never stomp a job that is currently grading.

Usage (from ``backend/`` with the app env active):

    .venv/bin/python scripts/regrade_interview_session.py <session_id> [--yes]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from uuid import UUID

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from abridgeai.core.config import get_settings

_EVALUATION_TASK = "evaluate_interview_session_task"


async def regrade(session_id: UUID, *, execute: bool) -> int:
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT status, pass_verdict, student_id, "
                        "evaluation_claim_token, evaluation_claim_expires_at "
                        "FROM interview_sessions WHERE id = :id"
                    ),
                    {"id": session_id},
                )
            ).mappings().first()
        if row is None:
            print(f"session {session_id} not found")
            return 1
        if row["evaluation_claim_token"] is not None and (
            row["evaluation_claim_expires_at"] is None
            or row["evaluation_claim_expires_at"] > datetime.now(UTC)
        ):
            print(
                "REFUSED: session holds a live evaluation claim "
                f"(expires {row['evaluation_claim_expires_at']}) — a job may be grading now"
            )
            return 1
        if row["pass_verdict"] is None:
            print(
                "REFUSED: no published verdict to correct — use the normal "
                "evaluation/recovery flow instead"
            )
            return 1

        print(
            f"session={session_id}\n"
            f"  status={row['status']}  pass_verdict={row['pass_verdict']}\n"
            f"  will reset pass_verdict + claim, then enqueue {_EVALUATION_TASK}"
        )
        if not execute:
            print("dry-run: nothing changed (pass --yes to execute)")
            return 0

        async with engine.begin() as conn:
            reset = await conn.execute(
                text(
                    "UPDATE interview_sessions SET pass_verdict = NULL, "
                    "evaluation_claim_token = NULL, "
                    "evaluation_claim_expires_at = NULL WHERE id = :id"
                ),
                {"id": session_id},
            )
            if reset.rowcount != 1:
                print("reset failed: session row vanished mid-run")
                return 1

        pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
        try:
            job_id = f"interview-evaluation:{session_id}:regrade-{int(time.time())}"
            job = await pool.enqueue_job(
                _EVALUATION_TASK,
                row["student_id"],
                session_id,
                _job_id=job_id,
            )
        finally:
            await pool.aclose()
        if job is None:
            print(f"enqueue REFUSED by redis for {job_id} (duplicate id?) — reset already applied")
            return 1
        print(f"regrade enqueued: job_id={job_id}")
        print("watch: pm2 logs abridgeai-worker --lines 50")
        return 0
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", type=UUID)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="execute the reset + enqueue (default is a dry-run)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(regrade(args.session_id, execute=args.yes)))


if __name__ == "__main__":
    main()
