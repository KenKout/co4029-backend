"""Regression tests for LLMRole enum and ROLE_TO_TIER mapping.

Per Reconciliation §B1, LLMRole values are APPEND-ONLY and must never be
reordered. These tests pin the new Phase 6 interview/gap-report roles and
their tier assignments.
"""

from __future__ import annotations

from abridgeai.ai.llm.roles import ROLE_TO_TIER, LLMRole


def test_interview_roles_appended() -> None:
    assert LLMRole.INTERVIEW_GENERATION.value == "interview_generation"
    assert LLMRole.INTERVIEW_VALIDATION.value == "interview_validation"
    assert LLMRole.INTERVIEW_FOLLOWUP.value == "interview_followup"
    assert LLMRole.INTERVIEW_EVALUATION.value == "interview_evaluation"
    assert LLMRole.GAP_REPORT_GENERATION.value == "gap_report_generation"


def test_interview_roles_tier_mapping() -> None:
    assert ROLE_TO_TIER[LLMRole.INTERVIEW_GENERATION] == "large"
    assert ROLE_TO_TIER[LLMRole.INTERVIEW_VALIDATION] == "standard"
    assert ROLE_TO_TIER[LLMRole.INTERVIEW_FOLLOWUP] == "small"
    assert ROLE_TO_TIER[LLMRole.INTERVIEW_EVALUATION] == "large"
    assert ROLE_TO_TIER[LLMRole.GAP_REPORT_GENERATION] == "large"


def test_existing_roles_not_reordered() -> None:
    expected_prefix = [
        "extraction",
        "enrichment",
        "ideation",
        "generation",
        "validation",
        "embedding",
        "chunking_enrichment",
        "kg_extraction",
        "stt",
        "vision",
    ]
    actual = [r.value for r in LLMRole]
    assert actual[: len(expected_prefix)] == expected_prefix
