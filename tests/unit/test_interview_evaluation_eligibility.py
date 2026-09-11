"""Evaluation eligibility: only APPLIED typed receipts are gradeable evidence.

Plan §1 (regression tests, "tests/integration/test_interview_evaluation_stage.py"):
a typed receipt that is still ``received`` or ``failed`` must NOT reach the
rubric/outcome prompts, while ordinary voice/REST user rows keep their old
eligibility. The predicate under test is the SINGLE shared one
(``logic._is_candidate_answer``) that both the evaluation stage and
``_list_candidate_answers`` route through.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from abridgeai.features.interviews.ai.stages.evaluation.logic import (
    _candidate_response_text,
    _is_candidate_answer,
)


def _message(
    *,
    role: str = "user",
    content: str = "recursion terminates at the base case",
    metadata: dict[str, Any] | None = None,
    session_question_id: Any = None,
) -> Any:
    """A duck-typed InterviewSessionMessage (the predicate only uses getattr)."""

    class _Msg:
        pass

    msg: Any = _Msg()
    msg.role = role  # type: ignore[attr-defined]
    msg.content_text = content  # type: ignore[attr-defined]
    msg.metadata_json = metadata  # type: ignore[attr-defined]
    msg.session_question_id = session_question_id  # type: ignore[attr-defined]
    return msg


def _receipt(turn_state: str) -> dict[str, Any]:
    return {"source": "native_agent", "kind": "answer", "turn_key": "tk", "turn_state": turn_state}


class TestTypedReceiptStates:
    def test_applied_receipt_is_evidence(self) -> None:
        assert _is_candidate_answer(_message(metadata=_receipt("applied"))) is True

    def test_received_receipt_is_not_yet_evidence(self) -> None:
        # Durable but the fold hasn't committed — grading it would double-count
        # a turn that may still fail.
        assert _is_candidate_answer(_message(metadata=_receipt("received"))) is False

    def test_failed_receipt_is_not_evidence(self) -> None:
        assert _is_candidate_answer(_message(metadata=_receipt("failed"))) is False

    def test_missing_turn_state_is_not_evidence(self) -> None:
        meta: dict[str, Any] = {"source": "native_agent", "kind": "answer", "turn_key": "tk"}
        assert _is_candidate_answer(_message(metadata=meta)) is False


class TestNonReceiptRowsKeepOldBehavior:
    def test_voice_or_rest_user_row_is_evidence(self) -> None:
        assert _is_candidate_answer(_message(metadata={"kind": "answer"})) is True

    def test_plain_user_row_without_metadata_is_evidence(self) -> None:
        assert _is_candidate_answer(_message(metadata=None)) is True

    def test_ai_rows_are_never_evidence(self) -> None:
        assert _is_candidate_answer(_message(role="ai", metadata=None)) is False

    def test_non_native_agent_source_is_ordinary_row(self) -> None:
        # A different 'source' value is not a typed receipt — old rules apply.
        rest_meta: dict[str, Any] = {"source": "rest", "turn_state": "received"}
        assert _is_candidate_answer(_message(metadata=rest_meta)) is True


class TestCandidateText:
    def test_receipt_text_is_its_content(self) -> None:
        msg = _message(content="the answer", metadata=_receipt("applied"))
        assert _candidate_response_text(msg) == "the answer"

    def test_received_receipt_text_still_reads_but_predicate_gates(self) -> None:
        # The text helper stays content-only; eligibility is the predicate's job.
        msg = _message(content="the answer", metadata=_receipt("received"))
        assert _candidate_response_text(msg) == "the answer"
        assert _is_candidate_answer(msg) is False


class TestLinkageContract:
    def test_receipt_metadata_carries_bank_question_id(self) -> None:
        """The receipt row records the bank question for remediation scripts."""
        from abridgeai.features.interviews.realtime.native_typed_turn import (
            typed_receipt_metadata,
        )

        bank_q = uuid4()
        meta = typed_receipt_metadata(turn_key="tk-1", bank_question_id=bank_q)
        assert meta["source"] == "native_agent"
        assert meta["bank_question_id"] == str(bank_q)
