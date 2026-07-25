"""Guard: the term-explanation feature must not define the question's own subject.

Regression being locked in
--------------------------
A live session (004a8900) was audited by the persona-adherence judge and flagged
``declared_answer``. Tracing it: the question was "Compare and contrast fact
tables and factless fact tables in a dimensional model." The candidate, stuck,
asked "Could you explain the term 'Fact tables'?" and the ``explain_current_term``
feature dutifully DEFINED fact tables — i.e. handed over exactly what the
question asked them to produce.

``explain_current_term`` is a legitimate vocabulary-help feature; the fix is not
to remove it but to refuse ONLY when the requested term IS the question's graded
subject, while still explaining genuinely peripheral terms (e.g. a context noun
like "dimensional model" in the same question).
"""

from __future__ import annotations

import pytest

from abridgeai.features.interviews.orchestrator.assistance_logic import (
    question_subjects,
    term_is_question_subject,
)


# ── The exact bug: the subject term is refused ───────────────────────────────

_BUG_Q = "Compare and contrast fact tables and factless fact tables in a dimensional model."


@pytest.mark.parametrize(
    "term",
    [
        "fact tables",
        "Fact tables",
        "fact table",  # singular ≈ plural via containment
        "factless fact tables",
        "the fact tables",  # leading article stripped
    ],
)
def test_subject_terms_are_flagged(term: str) -> None:
    assert term_is_question_subject(term, _BUG_Q) is True


# ── The feature still works: peripheral/context terms are explained ──────────


@pytest.mark.parametrize(
    "term",
    [
        "dimensional model",  # trailing context, NOT the graded subject
        "business process",  # not even in the question
    ],
)
def test_context_terms_are_not_flagged(term: str) -> None:
    assert term_is_question_subject(term, _BUG_Q) is False


# ── Subject extraction ───────────────────────────────────────────────────────


def test_extracts_both_subjects_and_drops_context() -> None:
    subjects = question_subjects(_BUG_Q)
    assert "fact tables" in subjects
    assert "factless fact tables" in subjects
    assert "dimensional model" not in subjects


def test_define_single_subject() -> None:
    subjects = question_subjects("Define idempotency.")
    assert subjects == ["idempotency"]


def test_what_is_form() -> None:
    assert term_is_question_subject("normalization", "What is normalization?") is True


def test_vietnamese_definition_question() -> None:
    q = "Định nghĩa chuẩn hóa cơ sở dữ liệu."
    assert term_is_question_subject("chuẩn hóa cơ sở dữ liệu", q) is True


# ── Non-definition questions never trigger the guard ─────────────────────────


def test_plain_question_has_no_subjects() -> None:
    # A question that does not ask for a definition/comparison should yield no
    # subjects, so ANY term the candidate asks about is safe to explain.
    q = "Why might a team choose a star schema for reporting workloads?"
    assert question_subjects(q) == []
    assert term_is_question_subject("star schema", q) is False


def test_empty_inputs_are_safe() -> None:
    assert question_subjects("") == []
    assert term_is_question_subject("", "Define X.") is False
    assert term_is_question_subject("x", "") is False


def test_single_char_term_never_flagged() -> None:
    # Guards against a stray letter matching via containment.
    assert term_is_question_subject("a", "Define arrays.") is False
