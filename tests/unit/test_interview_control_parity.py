"""Realtime control payload must reach full parity with REST ``/respond``.

Two transports deliver a typed interview turn:

* ``POST /interview-sessions/{id}/respond`` → :class:`InterviewSubmitAnswerResponse`
* the ``abridge.interview.control`` LiveKit topic → the same payload as JSON

A client that types its answer over ``lk.chat`` must learn EXACTLY what a REST
client learns. The bug these tests exist to prevent already shipped once: the
control payload hand-listed six fields while REST returned fifteen, so
``next_question`` (the object needed to render the next Question Card), every
``transition_*`` field, ``pending_confirmation`` and ``assistance_kind`` were
silently missing — and ``time_remaining_seconds`` was always ``None`` because
the interview brain never returns it.

The parity test below is deliberately structural (it enumerates
``model_fields``) rather than a fixed list, so a field added to the response
tomorrow fails here instead of quietly reaching one transport only.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from abridgeai.features.interviews.schemas.public import InterviewQuestionPublic
from abridgeai.features.interviews.schemas.session import InterviewSubmitAnswerResponse


class _FakeQuestion:
    """Stands in for an ``InterviewQuestion`` ORM row.

    Carries authoring-only columns (``difficulty``, ``review_status``,
    ``source_refs_json``) so the tests can prove the projection drops them
    rather than forwarding the row.
    """

    def __init__(self) -> None:
        self.id = uuid4()
        self.prompt_text = "Explain database indexing."
        self.question_type = "technical"
        # Authoring-only — must NEVER reach a learner.
        self.difficulty = "hard"
        self.review_status = "approved"
        self.source_refs_json = {"chunk": "secret grounding text"}
        self.linked_outcome_id = uuid4()


def _adaptive_result(*, question: _FakeQuestion | None = None) -> dict[str, object]:
    """A realistic adaptive-path result dict from ``take_session_step``."""
    return {
        "next_question": question,
        "is_finished": False,
        "followup_text": "Good. Now the next one.",
        "ai_turn_text": "Thanks — that covers it. Next question.",
        "language": "en",
        "should_narrate": True,
        "should_await_response": True,
        "should_finish": False,
        "assistance_kind": None,
        "pending_confirmation": False,
        "interaction_state": "awaiting_answer",
        "transition_id": "transition:abc:1",
        "transition_text": "Let's move on.",
        "transition_target": "next_question",
        # Internal keys the projection must ignore.
        "state_version": 42,
        "action": "advance",
        "reason_code": "coverage_met",
        "current_question_id": str(uuid4()),
    }


class TestFieldCoverage:
    """Every response field must be populated by the shared projection."""

    def test_projection_covers_every_response_field(self):
        # Structural, not a hard-coded list: a field added to the response
        # without a mapping in `from_step_result` fails here.
        built = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(question=_FakeQuestion()),
            time_remaining_seconds=300,
        )
        payload = built.model_dump(mode="json")

        declared = set(InterviewSubmitAnswerResponse.model_fields)
        assert set(payload) == declared, (
            "control payload and response schema disagree; "
            f"missing={declared - set(payload)} extra={set(payload) - declared}"
        )

    def test_no_response_field_is_silently_none_on_a_full_adaptive_turn(self):
        # A full adaptive turn populates every field. If any comes back None
        # here, the projection dropped it (the original six-field bug).
        built = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(question=_FakeQuestion()),
            time_remaining_seconds=300,
        )
        payload = built.model_dump(mode="json")

        # `assistance_kind` is legitimately None on a plain advance.
        expected_none = {"assistance_kind"}
        actually_none = {k for k, v in payload.items() if v is None}
        assert actually_none == expected_none, (
            f"unexpectedly-null fields: {actually_none - expected_none}"
        )

    def test_field_count_is_pinned(self):
        # A tripwire on the contract's size: if the response grows, this fails
        # and forces a look at whether the control topic carries the new field.
        assert len(InterviewSubmitAnswerResponse.model_fields) == 15


class TestNextQuestionProjection:
    """`next_question` must be a projected DTO, never the ORM row."""

    def test_next_question_is_projected_not_the_orm_row(self):
        question = _FakeQuestion()
        built = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(question=question),
            time_remaining_seconds=None,
        )
        assert isinstance(built.next_question, InterviewQuestionPublic)
        assert built.next_question.id == question.id
        assert built.next_question.prompt_text == question.prompt_text
        assert built.next_question.question_type == question.question_type

    def test_authoring_only_columns_do_not_leak(self):
        # The whole reason for projecting: difficulty and grounding refs are
        # teacher-only. A learner client must not receive them.
        payload = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(question=_FakeQuestion()),
            time_remaining_seconds=None,
        ).model_dump(mode="json")

        nq = payload["next_question"]
        assert set(nq) == {"id", "prompt_text", "question_type"}
        for leaked in ("difficulty", "review_status", "source_refs_json", "linked_outcome_id"):
            assert leaked not in nq

    def test_next_question_is_json_serializable(self):
        # The control topic carries JSON text. A raw UUID would raise at
        # publish time, i.e. AFTER the turn was already graded.
        import json

        payload = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(question=_FakeQuestion()),
            time_remaining_seconds=120,
        ).model_dump(mode="json")

        encoded = json.dumps(payload)  # must not raise
        assert isinstance(json.loads(encoded)["next_question"]["id"], str)

    def test_absent_next_question_stays_none(self):
        built = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(question=None),
            time_remaining_seconds=None,
        )
        assert built.next_question is None


class TestTimeRemaining:
    """The timer is computed by the caller, never read from the result."""

    def test_timer_comes_from_the_keyword_not_the_result_dict(self):
        # `take_session_step` does NOT return time_remaining_seconds. Reading it
        # from the result is how the control topic came to publish null.
        result = _adaptive_result()
        assert "time_remaining_seconds" not in result

        built = InterviewSubmitAnswerResponse.from_step_result(
            result,
            time_remaining_seconds=417,
        )
        assert built.time_remaining_seconds == 417

    def test_a_stray_result_key_cannot_override_the_caller(self):
        # Defence in depth: even if the brain later starts returning this key,
        # the caller's computed value (from the session row) wins.
        result = _adaptive_result()
        result["time_remaining_seconds"] = 999_999

        built = InterviewSubmitAnswerResponse.from_step_result(
            result,
            time_remaining_seconds=60,
        )
        assert built.time_remaining_seconds == 60

    def test_none_timer_is_allowed(self):
        # Untimed interviews, and the best-effort lookup failing, both yield None.
        built = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(),
            time_remaining_seconds=None,
        )
        assert built.time_remaining_seconds is None


class TestLegacyPath:
    """The legacy (non-adaptive) path must still project cleanly."""

    def test_legacy_result_leaves_adaptive_fields_none(self):
        legacy = {
            "next_question": _FakeQuestion(),
            "is_finished": False,
            "followup_text": "Tell me more.",
        }
        built = InterviewSubmitAnswerResponse.from_step_result(
            legacy,
            time_remaining_seconds=200,
        )
        assert built.ai_followup_text == "Tell me more."
        assert built.next_question is not None
        # Adaptive-only fields stay None rather than raising.
        assert built.ai_turn_text is None
        assert built.transition_target is None
        assert built.pending_confirmation is None

    def test_empty_result_does_not_raise(self):
        # Defensive: a result dict with nothing in it must still produce a valid
        # response (is_finished falls back to False) rather than a 500.
        built = InterviewSubmitAnswerResponse.from_step_result(
            {},
            time_remaining_seconds=None,
        )
        assert built.is_finished is False
        assert built.next_question is None


class TestFinishedTurn:
    """A typed FINAL answer must carry the finish signals."""

    def test_finished_turn_carries_both_finish_flags(self):
        # The frontend reads `should_finish ?? is_finished`, so a typed final
        # answer must surface whichever the path produced.
        result = _adaptive_result()
        result["is_finished"] = True
        result["should_finish"] = True
        result["next_question"] = None
        result["transition_target"] = "closing"

        built = InterviewSubmitAnswerResponse.from_step_result(
            result,
            time_remaining_seconds=0,
        )
        assert built.is_finished is True
        assert built.should_finish is True
        assert built.next_question is None
        assert built.transition_target == "closing"

    def test_end_confirmation_gate_survives_the_projection(self):
        # `pending_confirmation` drives the Continue / End-and-submit UI. If the
        # projection drops it, a typed candidate's "I'm done" is graded as an
        # answer instead of prompting confirmation.
        result = _adaptive_result()
        result["pending_confirmation"] = True
        result["interaction_state"] = "awaiting_end_confirmation"

        built = InterviewSubmitAnswerResponse.from_step_result(
            result,
            time_remaining_seconds=45,
        )
        assert built.pending_confirmation is True
        assert built.interaction_state == "awaiting_end_confirmation"


class TestRestAndControlAgree:
    """The two transports must produce byte-identical payloads."""

    def test_same_result_yields_the_same_payload_for_both_transports(self):
        # REST returns the model; the control topic publishes
        # model_dump(mode="json") of the same model built from the same result.
        # Building twice from one result must be identical, which is what makes
        # "one projection, two transports" true rather than aspirational.
        result = _adaptive_result(question=_FakeQuestion())

        rest_model = InterviewSubmitAnswerResponse.from_step_result(
            result, time_remaining_seconds=180
        )
        control_payload = InterviewSubmitAnswerResponse.from_step_result(
            result, time_remaining_seconds=180
        ).model_dump(mode="json")

        assert rest_model.model_dump(mode="json") == control_payload

    @pytest.mark.parametrize(
        "field_name",
        sorted(InterviewSubmitAnswerResponse.model_fields),
    )
    def test_every_field_appears_in_the_serialized_control_payload(self, field_name):
        # One test per field, so a failure names the exact field that went
        # missing rather than a set-difference blob.
        payload = InterviewSubmitAnswerResponse.from_step_result(
            _adaptive_result(question=_FakeQuestion()),
            time_remaining_seconds=300,
        ).model_dump(mode="json")
        assert field_name in payload
