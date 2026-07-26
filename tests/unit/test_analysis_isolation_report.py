"""Guard the isolation-report script and its committed baseline.

The report is the number this work is defended with, so it needs the same
protection as ``security_replay.py``'s baseline: if the corpus grows or the
boundary changes, the committed baseline must be refreshed deliberately rather
than drifting silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analysis_isolation_report import build_report  # noqa: E402

_BASELINE = _REPO_ROOT / "docs" / "analysis-isolation-baseline.json"


def test_report_matches_committed_baseline() -> None:
    """No silent drift. Refresh with ``--baseline`` when the change is intended."""
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert build_report() == baseline, (
        "isolation report drifted from docs/analysis-isolation-baseline.json.\n"
        "If the change is intended, refresh it:\n"
        "  .venv/bin/python scripts/analysis_isolation_report.py "
        "--baseline docs/analysis-isolation-baseline.json"
    )


def test_structural_claim_is_stated_as_a_bound_not_a_rate() -> None:
    """The headline claim must stay definitional.

    Under enforce, zero raw answer text reaches the rubric-bearing prompt, and
    what does reach it is bounded. Both come from prompt composition, not from
    detection succeeding — if either of these numbers ever becomes something
    other than 100/0, the split has been broken rather than merely weakened.
    """
    structural = build_report()["structural"]
    assert structural["raw_answer_reaches_rubric_prompt_off_mode_pct"] == 100.0
    assert structural["raw_answer_reaches_rubric_prompt_enforce_mode_pct"] == 0.0
    assert structural["max_attacker_chars_reaching_rubric_prompt_enforce"] == (
        structural["max_claims"] * structural["max_claim_chars"]
    )


def test_report_covers_the_analysis_stage_suite() -> None:
    """The suite this work added must actually be measured.

    A report that silently stopped reading ``rubric_exfiltration`` would show a
    flattering containment rate built on the easy cases only.
    """
    report = build_report()
    assert "rubric_exfiltration" in report["per_suite"]
    assert report["per_suite"]["rubric_exfiltration"]["total"] >= 7


def test_report_rejects_an_empty_fixture(tmp_path: Path) -> None:
    """A fixture with no attack cases must fail loudly, not report 0/0.

    Percentages over an empty corpus would be a division error or a vacuous
    100%; either way the number would be meaningless.
    """
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"fixture_version": "test", "controls": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no attack cases"):
        build_report(empty)
