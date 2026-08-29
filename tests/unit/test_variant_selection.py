"""Tests for :mod:`orchestrator.variant_selection`.

Locks the role-aware logical-question collapse: named roles keep their
preferred angle, the generic role keeps one stable pseudo-random member
per group (deterministic per session seed), and ungrouped/malformed
groups pass through untouched.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.interviewer_identity import InterviewerRole
from abridgeai.features.interviews.orchestrator.selection import CandidateQuestion
from abridgeai.features.interviews.orchestrator.variant_selection import select_logical_variants

ANGLES = ("technical", "system_design", "situational", "behavioral")


def _group(
    group_id: str,
    qids: list[str],
    qtypes: tuple[str, ...] = ANGLES,
) -> list[CandidateQuestion]:
    assert len(qids) == len(qtypes)
    return [
        CandidateQuestion(
            question_id=qid,
            linked_outcome_id="o1",
            question_type=qtype,
            difficulty="medium",
            position=index,
            variant_group_id=group_id,
        )
        for index, (qid, qtype) in enumerate(zip(qids, qtypes, strict=True))
    ]


def _ungrouped(qid: str, qtype: str = "technical") -> CandidateQuestion:
    return CandidateQuestion(
        question_id=qid,
        linked_outcome_id="o1",
        question_type=qtype,
        difficulty="medium",
        position=1,
        variant_group_id=None,
    )


def test_named_role_keeps_only_preferred_angle():
    group = _group("g1", ["g1t", "g1s", "g1sit", "g1b"])
    out = select_logical_variants(
        group, role=InterviewerRole.BACKEND_TECH_LEAD, session_seed="s1"
    )
    assert [c.question_id for c in out] == ["g1t"]

    out = select_logical_variants(
        group, role=InterviewerRole.HR_SCREENER, session_seed="s1"
    )
    assert [c.question_id for c in out] == ["g1b"]


def test_generic_keeps_exactly_one_stable_member_per_group():
    group = _group("g1", ["g1t", "g1s", "g1sit", "g1b"])
    first = select_logical_variants(
        group, role=InterviewerRole.GENERIC_ASSISTANT, session_seed="s1"
    )
    again = select_logical_variants(
        group, role=InterviewerRole.GENERIC_ASSISTANT, session_seed="s1"
    )
    assert len(first) == 1
    assert first[0].question_id == again[0].question_id


def test_generic_pick_varies_across_session_seeds():
    group = _group("g1", ["g1t", "g1s", "g1sit", "g1b"])
    picks = {
        select_logical_variants(
            group, role=InterviewerRole.GENERIC_ASSISTANT, session_seed=seed
        )[0].question_id
        for seed in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")
    }
    assert len(picks) >= 2  # not stuck on one angle


def test_malformed_group_passes_through_unchanged():
    partial = _group("g1", ["g1t", "g1s", "g1sit"], qtypes=ANGLES[:3])
    out = select_logical_variants(
        partial, role=InterviewerRole.BACKEND_TECH_LEAD, session_seed="s1"
    )
    assert len(out) == 3  # legacy behavior: no collapse on partial groups


def test_ungrouped_candidates_untouched():
    pool = [_ungrouped("u1"), _ungrouped("u2", qtype="behavioral")]
    out = select_logical_variants(
        pool, role=InterviewerRole.BACKEND_TECH_LEAD, session_seed="s1"
    )
    assert {c.question_id for c in out} == {"u1", "u2"}


def test_group_collapse_coexists_with_ungrouped():
    group = _group("g1", ["g1t", "g1s", "g1sit", "g1b"])
    pool = [*group, _ungrouped("u1")]
    out = select_logical_variants(
        pool, role=InterviewerRole.GENERIC_ASSISTANT, session_seed="s1"
    )
    assert len(out) == 2  # one from the group + the ungrouped question
    assert "u1" in {c.question_id for c in out}
