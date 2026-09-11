"""Remediation: audit/backfill historical typed-turn receipts with NULL linkage.

Plan §1 "Data impact": typed receipts persisted BEFORE the linkage fix carry
``session_question_id IS NULL`` while their ``metadata_json->>'bank_question_id'``
names the question they answer. The evaluator filters on the linkage column,
so those answers were acked/applied but never graded.

Safety contract:
- DRY-RUN by default — ``--apply`` is required to write.
- Backfills the LINK only; never re-grades anything.
- Also exports the sessions whose zero/unanswered evaluation may have been
  distorted by the missing link, for OPERATOR review before any manual
  requeue. The export is a report, not an action.

Usage (from co4029/backend):
    uv run python scripts/backfill_typed_receipt_linkage.py            # dry-run
    uv run python scripts/backfill_typed_receipt_linkage.py --apply    # write links
    ... --json out/report.json                                         # export review list

Requires DATABASE_URL (reads the dev/prod DB the service uses).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


def _database_url() -> str:
    """SQLAlchemy-async URL from the environment (docker-compose default)."""
    import os  # noqa: PLC0415

    url = os.environ.get("DATABASE_URL")
    if not url:
        url = "postgresql+psycopg://abridgeai:***@localhost:5433/abridgeai"
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


async def _run(*, apply: bool, json_out: str | None) -> int:
    from sqlalchemy import text  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import (  # noqa: PLC0415
        async_sessionmaker,
        create_async_engine,
    )

    engine = create_async_engine(_database_url())
    maker = async_sessionmaker(engine)

    unlinked: list[dict[str, Any]] = []
    distorted_sessions: set[str] = set()

    async with maker() as db:
        rows = (
            await db.execute(
                text(
                    "SELECT id, session_id, created_at, "
                    "       metadata_json->>'turn_key' AS turn_key, "
                    "       metadata_json->>'turn_state' AS turn_state, "
                    "       metadata_json->>'bank_question_id' AS bank_question_id "
                    "FROM interview_session_messages "
                    "WHERE role = 'user' "
                    "  AND session_question_id IS NULL "
                    "  AND metadata_json->>'source' = 'native_agent' "
                    "  AND metadata_json->>'turn_state' = 'applied' "
                    "ORDER BY created_at"
                )
            )
        ).mappings().all()

        for row in rows:
            bank_q = row["bank_question_id"]
            session_id = row["session_id"]
            link: str | None = None
            if bank_q is not None:
                link = (
                    await db.execute(
                        text(
                            "SELECT id FROM interview_session_questions "
                            "WHERE session_id = :s AND interview_question_id = :q "
                            "ORDER BY sequence_no LIMIT 1"
                        ),
                        {"s": session_id, "q": bank_q},
                    )
                ).scalar_one_or_none()
            unlinked.append(
                {
                    "message_id": str(row["id"]),
                    "session_id": str(session_id),
                    "turn_key": row["turn_key"],
                    "turn_state": row["turn_state"],
                    "bank_question_id": bank_q,
                    "proposed_session_question_id": str(link) if link else None,
                    "backfillable": link is not None,
                }
            )
            if link is not None:
                # A session whose applied answer was invisible to the grader.
                # Export for OPERATOR review — this script never re-grades.
                distorted_sessions.add(str(session_id))

    for item in unlinked:
        if apply and item["backfillable"]:
            async with maker() as db:
                await db.execute(
                    text(
                        "UPDATE interview_session_messages "
                        "SET session_question_id = :link WHERE id = :mid "
                        "  AND session_question_id IS NULL"
                    ),
                    {"link": item["proposed_session_question_id"], "mid": item["message_id"]},
                )
                await db.commit()
                item["applied"] = True

    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "unlinked_applied_receipts": len(unlinked),
        "backfillable": sum(1 for i in unlinked if i["backfillable"]),
        "sessions_for_operator_review": sorted(distorted_sessions),
        "items": unlinked,
    }
    rendered = json.dumps(report, indent=2)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"report written to {json_out}")
    else:
        print(rendered)

    await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the missing links (default: dry-run, report only)",
    )
    parser.add_argument("--json", dest="json_out", default=None, help="write the report as JSON")
    args = parser.parse_args()
    try:
        return asyncio.run(_run(apply=args.apply, json_out=args.json_out))
    except Exception as exc:  # noqa: BLE001 - operator-facing script
        print(f"remediation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
