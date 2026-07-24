#!/usr/bin/env python
"""Shadow-mode replay harness for the interview security guard (Phase 0.3).

Runs the full red-team corpus through the deterministic security functions and
prints, per case, the current decision (category + block). Two modes:

* ``--report`` (default): print a coverage summary — per-category recall over
  ``covered`` cases, benign false-positive rate, and how many ``gap`` cases the
  current code already catches. This is the number the hardening phases move.

* ``--baseline <path>``: write the current decisions to a JSON snapshot.
* ``--diff <path>``: compare current decisions against a saved snapshot and
  print only the cases whose decision changed (added blocks, removed blocks,
  category shifts). Use this before promoting a rules/prompt version: change
  rules -> replay -> diff against the committed baseline -> review -> ship.

The harness never invokes the live LLM classifier; it exercises the
deterministic ``assess_by_rules`` path plus the ``is_ambiguous_security_text``
routing gate (which decides whether a case WOULD reach the classifier). That
keeps replay fast, free, and reproducible in CI.

Usage::

    python scripts/security_replay.py --report
    python scripts/security_replay.py --baseline tests/unit/interviews/security/baseline_decisions.json
    python scripts/security_replay.py --diff  tests/unit/interviews/security/baseline_decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make both the security functions and the corpus loader importable.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tests" / "unit" / "interviews" / "security"))

from abridgeai.features.interviews.orchestrator.security_logic import (  # noqa: E402
    assess_by_rules,
    is_ambiguous_security_text,
)
from corpus_loader import load_corpus  # noqa: E402


def _decision_for(text: str) -> dict[str, Any]:
    """Deterministic decision snapshot for one utterance."""
    result = assess_by_rules(text)
    return {
        "detected": result.detected,
        "category": result.category.value,
        "should_block": result.should_block,
        "ambiguous_gate": is_ambiguous_security_text(text),
    }


def _all_decisions() -> dict[str, dict[str, Any]]:
    corpus = load_corpus()
    return {case.id: _decision_for(case.text) for case in corpus.input_cases}


def _report() -> int:
    corpus = load_corpus()
    per_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    benign_total = benign_fp = known_fp = 0
    gap_total = gap_caught = 0

    for case in corpus.input_cases:
        result = assess_by_rules(case.text)
        if case.status == "gap":
            gap_total += 1
            if case.classifier_only:
                if is_ambiguous_security_text(case.text):
                    gap_caught += 1
            elif result.detected and result.category.value == case.expected_category:
                gap_caught += 1
            continue
        if case.expected_category == "benign":
            benign_total += 1
            if result.detected:
                if getattr(case, "baseline_fp", False):
                    known_fp += 1
                else:
                    benign_fp += 1
                per_cat[result.category.value]["fp"] += 1
        elif result.detected and result.category.value == case.expected_category:
            per_cat[case.expected_category]["tp"] += 1
        else:
            per_cat[case.expected_category]["fn"] += 1

    print("=" * 64)
    print("SECURITY REPLAY — coverage report")
    print("=" * 64)
    print(f"{'category':<28}{'tp':>5}{'fp':>5}{'fn':>5}{'recall':>9}")
    print("-" * 64)
    for cat in sorted(per_cat):
        d = per_cat[cat]
        denom = d["tp"] + d["fn"]
        recall = (d["tp"] / denom) if denom else float("nan")
        print(f"{cat:<28}{d['tp']:>5}{d['fp']:>5}{d['fn']:>5}{recall:>9.2f}")
    print("-" * 64)
    rate = (benign_fp / benign_total) if benign_total else 0.0
    print(f"benign false-positive rate: {benign_fp}/{benign_total} = {rate:.2%}")
    print(f"known baseline false positives (tracked): {known_fp}")
    print(f"gap cases caught at baseline: {gap_caught}/{gap_total}")
    print("=" * 64)
    # Non-zero exit if an untracked benign FP appears — CI-friendly.
    return 1 if benign_fp else 0


def _write_baseline(path: Path) -> int:
    decisions = _all_decisions()
    path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(decisions)} decisions to {path}")
    return 0


def _diff(path: Path) -> int:
    if not path.exists():
        print(f"baseline snapshot not found: {path}", file=sys.stderr)
        return 2
    saved = json.loads(path.read_text(encoding="utf-8"))
    current = _all_decisions()
    changed = 0
    for case_id in sorted(set(saved) | set(current)):
        before = saved.get(case_id)
        after = current.get(case_id)
        if before != after:
            changed += 1
            print(f"\nΔ {case_id}")
            print(f"    before: {before}")
            print(f"    after:  {after}")
    if not changed:
        print("no decision changes vs baseline snapshot")
    else:
        print(f"\n{changed} case(s) changed decision")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interview security replay harness")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--report", action="store_true", help="print coverage report (default)")
    group.add_argument("--baseline", type=Path, help="write current decisions to a JSON snapshot")
    group.add_argument("--diff", type=Path, help="diff current decisions against a snapshot")
    args = parser.parse_args()

    if args.baseline:
        return _write_baseline(args.baseline)
    if args.diff:
        return _diff(args.diff)
    return _report()


if __name__ == "__main__":
    raise SystemExit(main())
