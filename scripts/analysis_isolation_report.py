"""Measure what the analysis split actually buys, offline and deterministically.

Runs no LLM and touches no database, so it is free, repeatable, and CI-safe —
the same design as ``security_replay.py``, and for the same reason: a number you
cannot reproduce is a number you cannot defend.

What it measures
----------------
Two things, kept separate because they are different strengths of claim.

1. **Structural isolation** — the guarantee. In ``off`` mode the rubric-bearing
   prompt receives the candidate's answer verbatim and unbounded, so every
   attack payload reaches it: 100%, by construction. In ``enforce`` mode that
   prompt receives no raw text at all — only up to ``MAX_CLAIMS`` paraphrases of
   ``MAX_CLAIM_CHARS`` each. This is not a detection rate that can regress; it
   follows from which fields are in which prompt, and
   ``tests/unit/test_interview_analysis_split.py`` pins it.

2. **Verbatim-echo containment** — the honest residual. The structural bound
   does not stop an extractor that paraphrases an injection *faithfully*. So we
   ask the worst case: if the extractor echoed a payload verbatim into a claim,
   would the boundary screen drop it?

An important caveat this report makes explicit rather than hiding: the boundary
screen calls the SAME rules engine as the turn-level guard. For a verbatim echo,
anything the turn guard missed the screen also misses. The screen therefore adds
little on the normal path — its real value is the separable-evidence path
(``taking.py:_record_separable_evidence``), where the text being analysed has
*already* tripped the guard, so a verbatim echo is caught by construction.

The load-bearing win is (1), not (2). Reporting (2) alongside it is what keeps
the claim defensible.

Usage::

    python scripts/analysis_isolation_report.py --report
    python scripts/analysis_isolation_report.py --baseline docs/analysis-isolation-baseline.json
    python scripts/analysis_isolation_report.py --diff docs/analysis-isolation-baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abridgeai.features.interviews.orchestrator.claim_filter import (  # noqa: E402
    filter_claims,
)
from abridgeai.features.interviews.orchestrator.extraction import (  # noqa: E402
    MAX_CLAIM_CHARS,
    MAX_CLAIMS,
    AnswerClaims,
    Claim,
)
from abridgeai.features.interviews.orchestrator.intent import (  # noqa: E402
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.security_logic import (  # noqa: E402
    assess_by_rules,
)

_REDTEAM_FIXTURE = Path(__file__).with_name("fixtures") / "interview_security_redteam.json"
# Suites whose cases are attacks. ``controls`` and ``adaptive`` are benign
# behaviour checks and would distort an attack-containment rate.
_ATTACK_SUITES = ("security", "semantic_security", "multiturn_security", "rubric_exfiltration")


def _dummy_intent() -> IntentClassification:
    return IntentClassification(
        intent=StudentIntent.ANSWER, confidence=0.9, rationale="report", source="report"
    )


def _survives_boundary(payload: str) -> bool:
    """Would a verbatim echo of ``payload`` reach the matcher?

    Exercises the real :func:`filter_claims`, not a reimplementation of it, so
    the number tracks the shipped code.
    """
    claims = AnswerClaims(
        intent=_dummy_intent(),
        claims=[Claim(text=payload[:MAX_CLAIM_CHARS])],
    )
    return len(filter_claims(claims).claims) > 0


def _load_attack_cases(fixture: Path) -> list[dict[str, Any]]:
    data = json.loads(fixture.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for suite in _ATTACK_SUITES:
        for row in data.get(suite, []):
            if isinstance(row, dict) and isinstance(row.get("text"), str):
                cases.append({"suite": suite, **row})
    return cases


def build_report(fixture: Path = _REDTEAM_FIXTURE) -> dict[str, Any]:
    cases = _load_attack_cases(fixture)
    if not cases:
        raise RuntimeError(f"no attack cases found in {fixture}")

    caught_by_turn_guard = 0
    echo_contained = 0
    leaked: list[str] = []
    per_suite: dict[str, dict[str, int]] = {}

    for case in cases:
        suite = case["suite"]
        bucket = per_suite.setdefault(suite, {"total": 0, "turn_guard": 0, "echo_contained": 0})
        bucket["total"] += 1

        text = case["text"]
        if assess_by_rules(text).detected:
            caught_by_turn_guard += 1
            bucket["turn_guard"] += 1
        if not _survives_boundary(text):
            echo_contained += 1
            bucket["echo_contained"] += 1
        else:
            leaked.append(str(case.get("name") or case.get("id") or "<unnamed>"))

    total = len(cases)
    return {
        "total_attack_cases": total,
        "structural": {
            # These two are definitional, not measured — see the module docstring.
            "raw_answer_reaches_rubric_prompt_off_mode_pct": 100.0,
            "raw_answer_reaches_rubric_prompt_enforce_mode_pct": 0.0,
            "max_attacker_chars_reaching_rubric_prompt_enforce": MAX_CLAIMS * MAX_CLAIM_CHARS,
            "max_claims": MAX_CLAIMS,
            "max_claim_chars": MAX_CLAIM_CHARS,
        },
        "verbatim_echo": {
            "caught_by_turn_guard": caught_by_turn_guard,
            "caught_by_turn_guard_pct": round(100.0 * caught_by_turn_guard / total, 2),
            "contained_at_boundary": echo_contained,
            "contained_at_boundary_pct": round(100.0 * echo_contained / total, 2),
            "would_survive_verbatim_echo": sorted(leaked),
        },
        "per_suite": per_suite,
    }


def _print_report(report: dict[str, Any]) -> None:
    structural = report["structural"]
    echo = report["verbatim_echo"]
    print(f"attack cases: {report['total_attack_cases']}")
    print()
    print("structural isolation (definitional, pinned by unit test):")
    print(
        "  raw answer reaches the rubric-bearing prompt:  "
        f"off={structural['raw_answer_reaches_rubric_prompt_off_mode_pct']}%  "
        f"enforce={structural['raw_answer_reaches_rubric_prompt_enforce_mode_pct']}%"
    )
    print(
        "  attacker-influenced chars able to reach it:    "
        f"unbounded -> {structural['max_attacker_chars_reaching_rubric_prompt_enforce']} "
        f"({structural['max_claims']} x {structural['max_claim_chars']}, paraphrased)"
    )
    print()
    print("verbatim-echo containment (worst case, same rules engine as the turn guard):")
    print(
        f"  caught by turn guard:      {echo['caught_by_turn_guard']:>3d}"
        f"  ({echo['caught_by_turn_guard_pct']}%)"
    )
    print(
        f"  contained at boundary:     {echo['contained_at_boundary']:>3d}"
        f"  ({echo['contained_at_boundary_pct']}%)"
    )
    if echo["would_survive_verbatim_echo"]:
        print(f"  would survive a verbatim echo ({len(echo['would_survive_verbatim_echo'])}):")
        for name in echo["would_survive_verbatim_echo"]:
            print(f"    - {name}")
    print()
    print("per suite (total / turn-guard / boundary):")
    for suite, counts in sorted(report["per_suite"].items()):
        print(
            f"  {suite:<22s} {counts['total']:>3d} / {counts['turn_guard']:>3d}"
            f" / {counts['echo_contained']:>3d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print the report (default)")
    parser.add_argument("--baseline", type=Path, help="write the report to PATH as JSON")
    parser.add_argument(
        "--diff", type=Path, help="compare against a baseline JSON and exit 1 on drift"
    )
    parser.add_argument("--fixtures", type=Path, default=_REDTEAM_FIXTURE)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report = build_report(args.fixtures)

    if args.baseline:
        args.baseline.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"baseline written: {args.baseline}")
        return 0

    if args.diff:
        previous = json.loads(args.diff.read_text(encoding="utf-8"))
        if previous == report:
            print("no drift")
            return 0
        print("DRIFT against baseline:")
        for key in ("structural", "verbatim_echo", "per_suite", "total_attack_cases"):
            if previous.get(key) != report.get(key):
                print(f"  {key}:")
                print(f"    baseline: {json.dumps(previous.get(key), sort_keys=True)}")
                print(f"    current:  {json.dumps(report.get(key), sort_keys=True)}")
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
