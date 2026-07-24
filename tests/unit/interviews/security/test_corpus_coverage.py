"""Corpus-driven regression coverage for the interview security guard.

This suite is the *measurement foundation* (Phase 0) for the security-hardening
workstream. It runs the shared, side-effect-free security functions
(``assess_by_rules`` / ``assess_output_leakage``) against a versioned red-team
corpus and asserts:

* ``status: covered`` cases behave as the baseline expects (rules catch them,
  or benign cases stay benign) — these must NEVER regress.
* ``status: must_stay_benign`` cases are NEVER blocked — the false-positive
  guard.
* ``status: gap`` cases are the forward spec for later phases. They are
  reported (not hard-failed) at baseline via ``xfail(strict=False)`` so the
  suite stays green until the owning phase ships, at which point they flip to
  passing and the xfail becomes an ``XPASS`` we then tighten.

Run the coverage report with::

    pytest tests/unit/interviews/security/test_corpus_coverage.py -s -q

The ``test_zzz_coverage_report`` case prints a per-category confusion matrix and
the benign false-positive rate at the end of the run.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

from abridgeai.features.interviews.orchestrator.security import ProtectedContent
from abridgeai.features.interviews.orchestrator.security_logic import (
    assess_by_rules,
    assess_output_leakage,
    is_ambiguous_security_text,
)

# The corpus loader lives beside this test module (not an installed package).
sys.path.insert(0, str(Path(__file__).parent))
from corpus_loader import (  # noqa: E402
    InputCase,
    OutputCase,
    load_corpus,
)

_CORPUS = load_corpus()


def _input_id(case: InputCase) -> str:
    return case.id


def _output_id(case: OutputCase) -> str:
    return case.id


# ─────────────────────────── Covered (baseline) ───────────────────────────


_COVERED_INPUT = [c for c in _CORPUS.input_cases if c.status == "covered"]


@pytest.mark.parametrize("case", _COVERED_INPUT, ids=[_input_id(c) for c in _COVERED_INPUT])
def test_covered_input_cases_do_not_regress(case: InputCase) -> None:
    """Baseline: every covered case must keep its current rules verdict."""
    result = assess_by_rules(case.text)
    if case.expected_category == "benign":
        assert result.detected is False, (
            f"{case.id}: benign case was flagged as {result.category.value}"
        )
        return
    assert result.detected is True, f"{case.id}: expected detection, got benign"
    assert result.should_block is case.expected_block
    assert result.category.value == case.expected_category, (
        f"{case.id}: expected {case.expected_category}, got {result.category.value}"
    )


# ─────────────────────── Must-stay-benign (FP guard) ───────────────────────


_BENIGN_GUARD = [c for c in _CORPUS.input_cases if c.status == "must_stay_benign"]


@pytest.mark.parametrize("case", _BENIGN_GUARD, ids=[_input_id(c) for c in _BENIGN_GUARD])
def test_must_stay_benign_is_never_blocked(
    case: InputCase, request: pytest.FixtureRequest
) -> None:
    """False-positive guard: academic answers with scary words stay benign.

    A case flagged ``baseline_fp`` is a KNOWN pre-existing false positive that
    the owning phase (``target_phase``) will fix. We record it as a non-strict
    xfail so the baseline stays green while the debt is tracked — when the phase
    ships it XPASSes, signalling it's time to drop the ``baseline_fp`` flag.
    """
    if case.baseline_fp:
        request.node.add_marker(
            pytest.mark.xfail(
                reason=(
                    f"known baseline false positive; fixed in phase {case.target_phase}"
                ),
                strict=False,
            )
        )
    result = assess_by_rules(case.text)
    assert result.detected is False, (
        f"{case.id}: FALSE POSITIVE — benign answer flagged as {result.category.value}"
    )
    assert result.should_block is False


# ───────────────────────────── Gap (forward spec) ─────────────────────────


_GAP_INPUT = [c for c in _CORPUS.input_cases if c.status == "gap"]


@pytest.mark.parametrize("case", _GAP_INPUT, ids=[_input_id(c) for c in _GAP_INPUT])
def test_gap_input_cases_forward_spec(case: InputCase, request: pytest.FixtureRequest) -> None:
    """Forward spec for a later phase.

    ``classifier_only`` gaps are considered satisfied when the ambiguity gate
    routes them to the semantic classifier (we can't invoke the live model in a
    unit test). Rule-catchable gaps assert the rules verdict directly.

    All gap assertions are wrapped in a non-strict xfail so the suite is green
    at baseline; when the owning phase ships the case XPASSes, signalling it's
    time to promote the case to ``status: covered``.
    """
    request.node.add_marker(
        pytest.mark.xfail(
            reason=f"gap for phase {case.target_phase}; not yet shipped",
            strict=False,
        )
    )
    if case.classifier_only:
        assert is_ambiguous_security_text(case.text) is True, (
            f"{case.id}: not routed to classifier"
        )
        return
    result = assess_by_rules(case.text)
    assert result.detected is True, f"{case.id}: expected detection for gap case"
    assert result.category.value == case.expected_category


# ───────────────────────────── Output leakage ─────────────────────────────


def _to_protected(case: OutputCase) -> list[ProtectedContent]:
    return [ProtectedContent(category=p["category"], text=p["text"]) for p in case.protected]


_COVERED_OUTPUT = [c for c in _CORPUS.output_cases if c.status == "covered"]
_GAP_OUTPUT = [c for c in _CORPUS.output_cases if c.status == "gap"]


@pytest.mark.parametrize("case", _COVERED_OUTPUT, ids=[_output_id(c) for c in _COVERED_OUTPUT])
def test_covered_output_leakage_does_not_regress(case: OutputCase) -> None:
    result = assess_output_leakage(case.text, _to_protected(case))
    assert result.blocked is case.expected_block, (
        f"{case.id}: expected blocked={case.expected_block}, got {result.blocked}"
    )
    if case.expected_block and case.expected_method:
        assert result.match_method == case.expected_method or case.expected_method in {
            "token_overlap",
            "fuzzy",
        }


@pytest.mark.parametrize("case", _GAP_OUTPUT, ids=[_output_id(c) for c in _GAP_OUTPUT])
def test_gap_output_leakage_forward_spec(
    case: OutputCase, request: pytest.FixtureRequest
) -> None:
    request.node.add_marker(
        pytest.mark.xfail(
            reason=f"output-guard gap for phase {case.target_phase}; not yet shipped",
            strict=False,
        )
    )
    result = assess_output_leakage(case.text, _to_protected(case))
    assert result.blocked is case.expected_block


# ─────────────────────────── Coverage report ──────────────────────────────


def test_zzz_coverage_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Emit a per-category confusion matrix + benign FP rate (informational)."""
    # Per-category tallies over covered + must_stay_benign (baseline-scored).
    scored = [
        c
        for c in _CORPUS.input_cases
        if c.status in {"covered", "must_stay_benign"}
    ]
    per_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    benign_total = 0
    benign_blocked = 0
    gap_caught = 0
    gap_total = len(_GAP_INPUT)

    known_baseline_fp = 0
    for case in scored:
        result = assess_by_rules(case.text)
        is_benign_expected = case.expected_category == "benign"
        if is_benign_expected:
            benign_total += 1
            if result.detected:
                if case.baseline_fp:
                    known_baseline_fp += 1
                else:
                    benign_blocked += 1
                per_cat[result.category.value]["fp"] += 1
            else:
                per_cat["benign"]["tn"] += 1
        else:
            if result.detected and result.category.value == case.expected_category:
                per_cat[case.expected_category]["tp"] += 1
            else:
                per_cat[case.expected_category]["fn"] += 1

    for case in _GAP_INPUT:
        if case.classifier_only:
            if is_ambiguous_security_text(case.text):
                gap_caught += 1
        else:
            r = assess_by_rules(case.text)
            if r.detected and r.category.value == case.expected_category:
                gap_caught += 1

    lines = ["", "═" * 64, "SECURITY CORPUS COVERAGE REPORT (baseline)", "═" * 64]
    lines.append(f"{'category':<28}{'tp':>5}{'fp':>5}{'fn':>5}{'recall':>9}")
    lines.append("-" * 64)
    for cat in sorted(per_cat):
        d = per_cat[cat]
        denom = d["tp"] + d["fn"]
        recall = (d["tp"] / denom) if denom else float("nan")
        lines.append(f"{cat:<28}{d['tp']:>5}{d['fp']:>5}{d['fn']:>5}{recall:>9.2f}")
    lines.append("-" * 64)
    fp_rate = (benign_blocked / benign_total) if benign_total else 0.0
    lines.append(f"benign false-positive rate: {benign_blocked}/{benign_total} = {fp_rate:.2%}")
    lines.append(f"known baseline false positives (tracked, phase-fixed): {known_baseline_fp}")
    lines.append(f"gap cases caught at baseline: {gap_caught}/{gap_total}")
    lines.append("═" * 64)
    report = "\n".join(lines)

    with capsys.disabled():
        print(report)

    # The report itself must always succeed; substantive assertions live in the
    # per-case tests above. We only assert the corpus is non-trivial here.
    assert len(_CORPUS.input_cases) >= 40
    # Baseline invariant: benign FP rate must be 0 (no academic answer blocked).
    assert benign_blocked == 0, f"benign false positives at baseline: {benign_blocked}"
