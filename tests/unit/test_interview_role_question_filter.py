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


def _c(qid: str, oid: str | None, qtype: str, group: str | None = None) -> CandidateQuestion:
    return CandidateQuestion(
        question_id=qid,
        linked_outcome_id=oid,
        question_type=qtype,
        difficulty=None,
        position=None,
        variant_group_id=group,
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
    # Grouped candidates: only the role's preferred type survives, plus
    # per-outcome fallbacks for grouped questions of uncovered outcomes.
    cands = [
        _c("a", "o1", "technical", group="g1"),
        _c("b", "o1", "situational", group="g1"),
        _c("c", "o2", "technical", group="g1"),
        _c("d", "o2", "behavioral", group="g1"),
    ]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a", "c"}


def test_ungrouped_non_preferred_without_outcome_is_kept():
    # Ungrouped questions are always askable by any role — even a
    # non-preferred type with no linked outcome survives the filter.
    cands = [_c("a", "o1", "technical"), _c("b", None, "behavioral")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a", "b"}


def test_ungrouped_questions_survive_strict_mode():
    # Strict only drops non-preferred GROUPED fallbacks; ungrouped stays.
    cands = [
        _c("g-tech", "o1", "technical", group="g1"),
        _c("g-sit", "o2", "situational", group="g1"),
        _c("loose", "o2", "behavioral"),
    ]
    out = filter_candidates_by_role(
        cands, InterviewerRole.BACKEND_TECH_LEAD, strict=True
    )
    assert {c.question_id for c in out} == {"g-tech", "loose"}


def test_strict_mode_drops_grouped_fallback():
    # Strict: a grouped question of an uncovered outcome keeps no fallback.
    cands = [
        _c("a", "o1", "technical", group="g1"),
        _c("b", "o2", "situational", group="g1"),
    ]
    out = filter_candidates_by_role(
        cands, InterviewerRole.BACKEND_TECH_LEAD, strict=True
    )
    assert {c.question_id for c in out} == {"a"}
    # Same pool without strict restores the grouped fallback.
    relaxed = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in relaxed} == {"a", "b"}


def test_outcome_without_preferred_keeps_fallback():
    cands = [_c("a", "o1", "technical"), _c("b", "o2", "situational")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a", "b"}  # b kept as coverage fallback


def test_empty_pool_degrades_to_all():
    cands = [_c("b", "o1", "situational")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert out == cands


def test_strict_generic_assistant_still_no_filter():
    cands = [_c("a", "o1", "technical")]
    assert (
        filter_candidates_by_role(
            cands, InterviewerRole.GENERIC_ASSISTANT, strict=True
        )
        == cands
    )
