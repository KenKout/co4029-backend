"""Text protocol: inbound attribute validation and control-event encoding.

No LiveKit, no DB — this module is pure so the wire contract can be pinned
exactly. These are the guards that stand between an untrusted browser and
``take_session_step``.
"""

from __future__ import annotations

import json

import pytest

from abridgeai.features.interviews.realtime import text_protocol as tp


class TestParseInboundAttributes:
    def test_plain_text_defaults_to_answer(self):
        turn = tp.parse_inbound_attributes("Paris is the capital", None)
        assert turn.text == "Paris is the capital"
        assert turn.turn_action == "answer"
        assert turn.turn_key is None

    def test_strips_surrounding_whitespace(self):
        assert tp.parse_inbound_attributes("  hi  ", {}).text == "hi"

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_rejects_blank_text(self, blank):
        with pytest.raises(tp.InboundTurnError) as exc:
            tp.parse_inbound_attributes(blank, None)
        assert exc.value.rejection is tp.TurnRejection.EMPTY_TEXT

    def test_rejects_oversized_text(self):
        with pytest.raises(tp.InboundTurnError) as exc:
            tp.parse_inbound_attributes("x" * (tp.MAX_TEXT_CHARS + 1), None)
        assert exc.value.rejection is tp.TurnRejection.TEXT_TOO_LONG

    def test_accepts_text_at_exactly_the_cap(self):
        # Boundary: a long-but-real answer must survive.
        turn = tp.parse_inbound_attributes("x" * tp.MAX_TEXT_CHARS, None)
        assert len(turn.text) == tp.MAX_TEXT_CHARS

    @pytest.mark.parametrize("action", sorted(tp.VALID_TURN_ACTIONS))
    def test_accepts_every_valid_turn_action(self, action):
        turn = tp.parse_inbound_attributes("q", {tp.ATTR_TURN_ACTION: action})
        assert turn.turn_action == action

    def test_rejects_unknown_turn_action_rather_than_downgrading(self):
        # THE point of this validator. Silently coercing an unknown action to
        # "answer" is the scoring bug the protocol exists to prevent: a "hint"
        # request treated as an answer gets graded as one.
        with pytest.raises(tp.InboundTurnError) as exc:
            tp.parse_inbound_attributes("give me a hint", {tp.ATTR_TURN_ACTION: "hintt"})
        assert exc.value.rejection is tp.TurnRejection.INVALID_TURN_ACTION

    def test_rejects_turn_action_case_variants(self):
        # The brain's comparison is exact; accepting "Answer" here would pass a
        # value downstream that matches no branch.
        with pytest.raises(tp.InboundTurnError):
            tp.parse_inbound_attributes("x", {tp.ATTR_TURN_ACTION: "Answer"})

    def test_empty_turn_action_attribute_is_treated_as_absent(self):
        turn = tp.parse_inbound_attributes("x", {tp.ATTR_TURN_ACTION: ""})
        assert turn.turn_action == "answer"

    def test_accepts_uuid_turn_key(self):
        key = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        assert tp.parse_inbound_attributes("x", {tp.ATTR_TURN_KEY: key}).turn_key == key

    def test_accepts_client_fallback_turn_key_shape(self):
        # The frontend falls back to `tk-<ts>-<rand>` when crypto.randomUUID is
        # unavailable, so that shape must validate too.
        key = "tk-1738368000000-a1b2c3"
        assert tp.parse_inbound_attributes("x", {tp.ATTR_TURN_KEY: key}).turn_key == key

    @pytest.mark.parametrize(
        "bad_key",
        [
            "short",  # under 8 chars
            "x" * 129,  # over 128
            "has spaces in it",
            "semi;colon;injection",
            "../../etc/passwd",
            "drop table interviews--",
        ],
    )
    def test_rejects_malformed_turn_key(self, bad_key):
        # A turn_key reaches a DB column and structured logs. Bounding its shape
        # stops it being used to smuggle content into either.
        with pytest.raises(tp.InboundTurnError) as exc:
            tp.parse_inbound_attributes("x", {tp.ATTR_TURN_KEY: bad_key})
        assert exc.value.rejection is tp.TurnRejection.INVALID_TURN_KEY

    def test_absent_turn_key_is_allowed(self):
        assert tp.parse_inbound_attributes("x", {}).turn_key is None

    def test_client_supplied_identity_attributes_are_ignored(self):
        # The security requirement: session_id / student_id must NEVER be taken
        # from the client. They are not part of the parsed result at all, so a
        # client sending them cannot influence which session is graded.
        turn = tp.parse_inbound_attributes(
            "x",
            {
                "session_id": "00000000-0000-0000-0000-000000000000",
                "student_id": "11111111-1111-1111-1111-111111111111",
                "user_id": "attacker",
                tp.ATTR_TURN_ACTION: "answer",
            },
        )
        assert not hasattr(turn, "session_id")
        assert not hasattr(turn, "student_id")
        assert turn.turn_action == "answer"


class TestControlEvent:
    def test_accepted_event_has_seq_but_no_state_version(self):
        # ACCEPTED is emitted before the brain runs, so there is no brain version
        # to report yet — and the key must be absent, not null-with-meaning.
        ev = tp.ControlEvent(status=tp.ControlStatus.ACCEPTED, turn_key="tk-abcdefgh", seq=1)
        payload = json.loads(ev.to_json())
        assert payload["status"] == "accepted"
        assert payload["seq"] == 1
        assert "state_version" not in payload

    def test_completed_event_carries_both_counters_and_state(self):
        ev = tp.ControlEvent(
            status=tp.ControlStatus.COMPLETED,
            turn_key="tk-abcdefgh",
            seq=4,
            state_version=7,
            state={"is_finished": False, "next_question_text": "Next?"},
        )
        payload = json.loads(ev.to_json())
        assert payload["seq"] == 4
        assert payload["state_version"] == 7
        assert payload["state"]["next_question_text"] == "Next?"

    def test_rejected_event_names_the_reason(self):
        ev = tp.ControlEvent(
            status=tp.ControlStatus.REJECTED,
            turn_key=None,
            seq=2,
            rejection=tp.TurnRejection.TURN_IN_FLIGHT,
        )
        payload = json.loads(ev.to_json())
        assert payload["rejection"] == "turn_in_flight"
        assert payload["turn_key"] is None

    def test_failed_event_carries_only_an_error_class(self):
        # Never a raw exception message: those can contain prompt text, SQL, or
        # provider detail that must not reach a browser.
        ev = tp.ControlEvent(
            status=tp.ControlStatus.FAILED,
            turn_key="tk-abcdefgh",
            seq=3,
            error_class="TimeoutError",
        )
        payload = json.loads(ev.to_json())
        assert payload["error_class"] == "TimeoutError"
        assert "detail" not in payload
        assert "message" not in payload

    def test_json_is_deterministic(self):
        # Sorted keys + compact separators: two identical events must serialise
        # byte-identically so a client can dedupe on the raw payload if it wants.
        kwargs = {
            "status": tp.ControlStatus.ACCEPTED,
            "turn_key": "tk-abcdefgh",
            "seq": 1,
        }
        assert tp.ControlEvent(**kwargs).to_json() == tp.ControlEvent(**kwargs).to_json()

    def test_topics_are_namespaced_and_distinct(self):
        # Control must not collide with an `lk.*` topic the SDK owns now or later.
        assert tp.TOPIC_CHAT == "lk.chat"
        assert tp.TOPIC_TRANSCRIPTION == "lk.transcription"
        assert not tp.TOPIC_CONTROL.startswith("lk.")
        assert len({tp.TOPIC_CHAT, tp.TOPIC_TRANSCRIPTION, tp.TOPIC_CONTROL}) == 3

    def test_valid_turn_actions_match_the_rest_contract(self):
        # If `take_session_step` gains an action, this test is the tripwire that
        # says the text transport was not updated with it.
        expected = {"answer", "repeat", "clarify", "explain_term", "hint"}
        assert expected == tp.VALID_TURN_ACTIONS
