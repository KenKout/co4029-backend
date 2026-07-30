#!/usr/bin/env python
"""Post-hoc interview quality report (offline diagnostics).

Runs three metrics over a FINISHED interview session and prints a report. All
three are read-only and out-of-band — nothing here can affect a live session.

    calibration   rule-based, FREE. Did the interviewer's runtime belief about
                  outcome coverage match the post-session evaluator's verdicts?
                  Over-confidence is the costly direction: the interview stopped
                  probing an outcome the evaluator later judged unmet.

    contingency   LLM-judged. Do follow-up questions actually build on what the
                  student just said, or are they scripted? An interview with no
                  follow-ups at all is reported as such.

    leading       LLM-judged. Do questions hand the student part of the answer?
                  This is a VALIDITY check for exam settings, not a style note.

Contingency and leading make one LLM call per interviewer utterance, so both
require an explicit ``--limit`` ceiling and are opt-in:

    # free, no LLM calls
    python scripts/interview_quality_report.py --session <uuid>

    # add the judged metrics, capped at 15 utterances each
    python scripts/interview_quality_report.py --session <uuid> \
        --contingency --leading --limit 15

    # machine-readable
    python scripts/interview_quality_report.py --session <uuid> --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from sqlalchemy import select  # noqa: E402

from abridgeai.core.db import close_db, get_sessionmaker  # noqa: E402
from abridgeai.features.interviews.models import InterviewSession  # noqa: E402
from abridgeai.features.interviews.quality import compute_calibration  # noqa: E402
from abridgeai.features.interviews.quality.judges import (  # noqa: E402
    judge_contingency,
    judge_leading,
)
from abridgeai.features.interviews.quality.loaders import (  # noqa: E402
    load_outcome_labels,
    load_runtime_coverage,
    load_transcript,
    load_verdicts,
)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _short(text: str | None, width: int = 70) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


async def _run(args: argparse.Namespace) -> int:
    session_uuid = UUID(args.session)
    sm = get_sessionmaker()
    report: dict[str, Any] = {"session_id": str(session_uuid)}
    labels: dict[str, str] = {}
    try:
        async with sm() as db:
            session = (
                await db.execute(select(InterviewSession).where(InterviewSession.id == session_uuid))
            ).scalar_one_or_none()
            if session is None:
                print(f"session {session_uuid} not found", file=sys.stderr)
                return 2

            # ── calibration (free) ────────────────────────────────────────────
            coverage = await load_runtime_coverage(db, session_uuid)
            verdicts = await load_verdicts(db, session_uuid)
            labels = await load_outcome_labels(db, session.interview_config_id)
            calib = compute_calibration(
                session_id=str(session_uuid),
                runtime_coverage=coverage,
                verdicts=verdicts,
            )
            report["calibration"] = calib.to_dict()

            transcript = await load_transcript(db, session_uuid)
            report["transcript_turns"] = len(transcript)

            # ── judged metrics (cost money; opt-in) ───────────────────────────
            if args.contingency:
                cont = await judge_contingency(
                    db,
                    session_id=str(session_uuid),
                    turns=transcript,
                    limit=args.limit,
                )
                report["contingency"] = cont.to_dict()
            if args.leading:
                lead = await judge_leading(
                    db,
                    session_id=str(session_uuid),
                    turns=transcript,
                    limit=args.limit,
                )
                report["leading"] = lead.to_dict()
    finally:
        await close_db()

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    _print_human(report, labels=labels)
    return 0


def _print_human(report: dict[str, Any], *, labels: dict[str, str]) -> None:
    print(f"\n=== interview quality report: {report['session_id']} ===")
    print(f"transcript turns: {report.get('transcript_turns', 0)}")

    calib = report["calibration"]
    print("\n-- calibration (runtime belief vs evaluator verdict) --")
    if calib["scored_outcomes"] == 0:
        print(
            "  no comparable outcomes (session not evaluated yet, or the "
            "adaptive orchestrator never ran for it)"
        )
    else:
        print(f"  scored outcomes : {calib['scored_outcomes']}")
        print(f"  agreement rate  : {_fmt(calib['agreement_rate'])}")
        print(f"  over-confidence : {_fmt(calib['over_confidence_rate'])}", end="")
        print("   <- interview stopped probing an outcome the evaluator failed")
        for oid in calib["over_confident"]:
            print(f"    ! over-confident: {_short(labels.get(oid, oid))}")
        for oid in calib["under_confident"]:
            print(f"    . under-confident (wasted turns): {_short(labels.get(oid, oid))}")

    if "contingency" in report:
        cont = report["contingency"]
        print("\n-- contingency (do follow-ups build on the answer?) --")
        if cont["no_followups"]:
            print("  NO follow-ups in this session — scripted question-bank behaviour")
        else:
            print(f"  judged     : {cont['judged']}   mean: {_fmt(cont['mean_score'])} / 5")
            print(f"  weak (<3)  : {cont['weak_count']}")
            print(f"  fabricated : {cont['fabricated_premise_count']}  (invented a premise)")
            for turn in cont["turns"]:
                if turn["score"] < 3 or turn["fabricated_premise"]:
                    print(f"    ! [{turn['score']}/5] {_short(turn['explanation'])}")
        for err in cont["errors"]:
            print(f"    (judge error, excluded) {err}")

    if "leading" in report:
        lead = report["leading"]
        print("\n-- leading questions (validity: did we hand over the answer?) --")
        print(f"  judged       : {lead['judged']}   mean: {_fmt(lead['mean_score'])} / 5")
        print(f"  leading (<3) : {lead['leading_count']}")
        print(f"  contaminated : {lead['contaminated_count']}  (answer echoed the question)")
        for turn in lead["turns"]:
            if turn["score"] < 3:
                leaked = _short(turn["leaked_content"], 50)
                print(f"    ! [{turn['score']}/5] leaked: {leaked or '(see note)'}")
                print(f"        {_short(turn['explanation'])}")
        for err in lead["errors"]:
            print(f"    (judge error, excluded) {err}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="interview session UUID")
    parser.add_argument(
        "--contingency",
        action="store_true",
        help="run the contingency judge (1 LLM call per follow-up)",
    )
    parser.add_argument(
        "--leading",
        action="store_true",
        help="run the leading-question judge (1 LLM call per interviewer turn)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="max utterances judged per metric (cost ceiling; default 20)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
