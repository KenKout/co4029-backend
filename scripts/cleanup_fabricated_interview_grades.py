#!/usr/bin/env python
"""Remove grades fabricated for interviews the candidate never started.

Background
----------
``submit_session`` used to gate evaluation on::

    has_content = user_message_count > 0 or reason != "timed_out"

The ``or`` made ANY non-timeout submit gradeable regardless of content, so a
candidate who quit during onboarding and hit submit was marked ``completed`` and
sent to the judge. The judge then produced outcome verdicts, a ``pass_verdict``
and a ``GapReport`` from onboarding chatter alone ("Yes, that's me." / "The audio
is clear."), sometimes from a completely empty transcript.

The code path is fixed (see ``services/taking.py`` and
``services/evaluation.py``), but rows written before the fix are still in the
database: teachers see them as real results, and the stalled-evaluation sweep
would re-queue them if the verdict were merely nulled.

What counts as fabricated
-------------------------
``assessment_started_at IS NULL`` — the run never left onboarding. This is a
provable signal, not a heuristic: ``taking.record_answer`` refuses to record an
answer until ``onboarding_stage == 'completed'``, so a NULL here means no answer
can exist. The script additionally asserts zero question-linked user turns and
refuses to touch any session that has even one, so a bad signal cannot cause
data loss.

Practice runs are skipped: they are ungraded by design and their NULL verdict is
load-bearing elsewhere.

What it changes, per affected session
-------------------------------------
1. Deletes its ``gap_reports`` rows.
2. Deletes its ``interview_outcome_evaluations`` rows.
3. Sets ``pass_verdict = NULL``.
4. Sets ``status = 'abandoned'``.

Step 4 matters and is not cosmetic. Leaving ``status='completed'`` with a NULL
verdict puts the row into two "terminal but ungraded" scans forever:
``list_sessions_with_stalled_evaluation`` would re-enqueue grading (recreating
exactly what we just deleted), and the teacher dashboard's "needs grading"
widget would count it permanently.

What it does NOT change
-----------------------
``attempt_number``, and the student's consumed-attempt count. ``abandoned`` is
already one of ``_TERMINAL_SESSION_STATUSES``, so these runs count against
``max_attempts`` either way and this script cannot hand the attempt back.
Restoring attempts is a separate, per-student decision — deliberately out of
scope here.

Usage
-----
Dry run (default — prints the plan, writes nothing)::

    python scripts/cleanup_fabricated_interview_grades.py

Apply, after reading the dry run::

    python scripts/cleanup_fabricated_interview_grades.py --apply

Both modes print the same per-session detail so the two can be diffed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from sqlalchemy import text  # noqa: E402

from abridgeai.core.db import close_db, get_sessionmaker  # noqa: E402

# Candidate rows. The two NOT EXISTS clauses are belt and braces: the first is
# the signal, the second refuses to touch anything that has a real answer.
_SELECT_AFFECTED = text("""
    SELECT s.id                                              AS session_id,
           s.status,
           s.pass_verdict,
           s.student_id,
           s.interview_config_id,
           s.onboarding_stage,
           s.attempt_number,
           s.started_at,
           (SELECT count(*) FROM gap_reports g
             WHERE g.source_interview_session_id = s.id)     AS gap_reports,
           (SELECT count(*) FROM interview_outcome_evaluations e
             WHERE e.session_id = s.id)                      AS outcome_evals,
           (SELECT count(*) FROM interview_session_messages m
             WHERE m.session_id = s.id AND m.role = 'user')  AS user_msgs_any
    FROM interview_sessions s
    WHERE s.assessment_started_at IS NULL
      AND (
            s.pass_verdict IS NOT NULL
            OR EXISTS (SELECT 1 FROM gap_reports g
                       WHERE g.source_interview_session_id = s.id)
            OR EXISTS (SELECT 1 FROM interview_outcome_evaluations e
                       WHERE e.session_id = s.id)
          )
      AND NOT EXISTS (SELECT 1 FROM interview_session_messages m
                      WHERE m.session_id = s.id
                        AND m.role = 'user'
                        AND m.session_question_id IS NOT NULL)
    ORDER BY s.started_at
""")

# Independent safety net: anything with a real answer must never be in scope.
_ASSERT_NO_REAL_ANSWERS = text("""
    SELECT count(*) AS n
    FROM interview_sessions s
    WHERE s.assessment_started_at IS NULL
      AND EXISTS (SELECT 1 FROM interview_session_messages m
                  WHERE m.session_id = s.id
                    AND m.role = 'user'
                    AND m.session_question_id IS NOT NULL)
""")

_DELETE_GAP_REPORTS = text("DELETE FROM gap_reports WHERE source_interview_session_id = ANY(:ids)")
_DELETE_OUTCOME_EVALS = text(
    "DELETE FROM interview_outcome_evaluations WHERE session_id = ANY(:ids)"
)
_RESET_SESSIONS = text("""
    UPDATE interview_sessions
       SET pass_verdict = NULL,
           status = 'abandoned'
     WHERE id = ANY(:ids)
""")


def _print_plan(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("Nothing to clean: no fabricated grades found.")
        return

    print(f"{len(rows)} session(s) graded without ever starting the assessment:\n")
    for r in rows:
        print(f"  session      {r['session_id']}")
        print(f"    status     {r['status']}  ->  abandoned")
        print(f"    verdict    {r['pass_verdict']}  ->  NULL")
        print(f"    stage      {r['onboarding_stage']} (attempt #{r['attempt_number']})")
        print(f"    started    {r['started_at']}")
        print(
            f"    deleting   {r['gap_reports']} gap report(s), "
            f"{r['outcome_evals']} outcome evaluation(s)"
        )
        print(
            f"    transcript {r['user_msgs_any']} student message(s), "
            f"0 of them answering a question"
        )
        print()

    print(
        f"TOTAL: {sum(r['gap_reports'] for r in rows)} gap report(s), "
        f"{sum(r['outcome_evals'] for r in rows)} outcome evaluation(s), "
        f"{len(rows)} session row(s) reset."
    )
    print(
        "\nNOT changed: attempt_number, and each student's consumed-attempt count "
        "('abandoned' is already terminal, so these runs count against "
        "max_attempts either way)."
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the changes. Without this flag nothing is modified.",
    )
    args = parser.parse_args()

    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            rows = [dict(r) for r in (await db.execute(_SELECT_AFFECTED)).mappings().all()]
            _print_plan(rows)

            if not rows:
                return 0

            # Guard: if ANY never-started session has a real answer, the signal
            # is wrong and we must not delete anything anywhere.
            leaked = int((await db.execute(_ASSERT_NO_REAL_ANSWERS)).mappings().one()["n"])
            if leaked:
                print(
                    f"\nABORT: {leaked} session(s) have assessment_started_at NULL yet "
                    "contain question-linked answers. The signal this cleanup relies on "
                    "does not hold — investigate before deleting anything.",
                    file=sys.stderr,
                )
                return 2

            if not args.apply:
                print("\nDRY RUN — nothing written. Re-run with --apply to perform it.")
                return 0

            ids = [r["session_id"] for r in rows]
            deleted_reports = (await db.execute(_DELETE_GAP_REPORTS, {"ids": ids})).rowcount
            deleted_evals = (await db.execute(_DELETE_OUTCOME_EVALS, {"ids": ids})).rowcount
            reset_sessions = (await db.execute(_RESET_SESSIONS, {"ids": ids})).rowcount
            await db.commit()

            print(
                f"\nAPPLIED: deleted {deleted_reports} gap report(s), "
                f"{deleted_evals} outcome evaluation(s); reset {reset_sessions} session(s)."
            )

            remaining = [dict(r) for r in (await db.execute(_SELECT_AFFECTED)).mappings().all()]
            if remaining:
                print(
                    f"WARNING: {len(remaining)} session(s) still match the query "
                    "after the write — re-run and investigate.",
                    file=sys.stderr,
                )
                return 1
            print("Verified: no fabricated grades remain.")
            return 0
    finally:
        await close_db()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
