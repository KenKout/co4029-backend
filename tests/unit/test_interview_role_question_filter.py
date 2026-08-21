"""Tests for :mod:`orchestrator.role_question_filter`.

Locks the role→type mapping and the HARD-filter semantics (plus the
coverage-preserving fallback and the no-filter degrade) before it is wired
into ``adaptive.py``.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.interviewer_identity import InterviewerRole
from abridgeai.features.interviews.orchestrator.role_question_filter import (
    filter_candidates_by_role,
    preferred_type,
)
from abridgeai.features.interviews.orchestrator.selection import CandidateQuestion


def _c(qid: str, oid: str | None, qtype: str) -> CandidateQuestion:
    return CandidateQuestion(
        question_id=qid,
        linked_outcome_id=oid,
        question_type=qtype,
        difficulty=None,
        position=None,
    )


def test_preferred_type_mapping():
    assert preferred_type(InterviewerRole.BACKEND_TECH_LEAD) == "technical"
    assert preferred_type(InterviewerRole.STAFF_ENGINEER) == "system_design"
    assert preferred_type(InterviewerRole.ENG_MANAGER) == "situational"
    assert preferred_type(InterviewerRole.HR_SCREENER) == "behavioral"
    assert preferred_type(InterviewerRole.GENERIC_ASSISTANT) is None


def test_generic_assistant_no_filter():
    cands = [_c("a", "o1", "technical"), _c("b", "o1", "situational")]
    assert filter_candidates_by_role(cands, InterviewerRole.GENERIC_ASSISTANT) == cands


def test_hard_filter_keeps_only_preferred():
    cands = [
        _c("a", "o1", "technical"),
        _c("b", "o1", "situational"),
        _c("c", "o2", "technical"),
        _c("d", "o2", "behavioral"),
    ]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a", "c"}


def test_outcome_without_preferred_keeps_fallback():
    cands = [_c("a", "o1", "technical"), _c("b", "o2", "situational")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a", "b"}  # b kept as coverage fallback


def test_empty_pool_degrades_to_all():
    cands = [_c("b", "o1", "situational")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert out == cands


def test_no_outcome_non_preferred_is_dropped():
    # A question with no linked outcome is not a coverage fallback and does not
    # match the role's type, so a hard filter drops it.
    cands = [_c("a", "o1", "technical"), _c("b", None, "behavioral")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a"}
