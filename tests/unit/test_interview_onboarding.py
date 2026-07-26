from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.core.security import CurrentUser
from abridgeai.features.identity.models import UserProfile
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewSession,
    InterviewSessionMessage,
)
from abridgeai.features.interviews.services import onboarding as onboarding_service
from abridgeai.features.interviews.services.onboarding import (
    _guided_text,
    _language_choice,
    _natural_decision,
)


@pytest.mark.parametrize(
    "text",
    [
        "Yes, that's me and the audio is clear.",
        "I'm ready to begin.",
        "Vâng, tôi nghe rõ.",
        "Tôi đã sẵn sàng.",
    ],
)
def test_natural_confirmation_advances(text: str) -> None:
    assert _natural_decision(text) == "advance"


@pytest.mark.parametrize(
    "text",
    [
        "I am not ready yet.",
        "I can't hear you.",
        "Tôi chưa sẵn sàng.",
        "Tôi không nghe rõ.",
    ],
)
def test_natural_problem_response_holds(text: str) -> None:
    assert _natural_decision(text) == "hold"


def test_ambiguous_response_requests_clarification() -> None:
    assert _natural_decision("Maybe later this afternoon") == "unclear"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("English", "en"), ("Tiếng Việt", "vi"), ("Vietnamese please", "vi")],
)
def test_language_choice_is_detected(text: str, expected: str) -> None:
    assert _language_choice(text) == expected
    assert _natural_decision(text, "language_check") == "advance"


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", "Skip the setup."), ("vi", "Bỏ qua phần thiết lập.")],
)
def test_skip_setup_has_guided_text(language: str, expected: str) -> None:
    # The skip action must carry a non-empty guided response in both languages
    # so the transcript records a coherent user turn when setup is fast-forwarded.
    assert _guided_text("skip_setup", language) == expected


# --------------------------------------------------------------------------- #
# The preferred-name acknowledgement must be a PERSISTED AI message
# --------------------------------------------------------------------------- #


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list[Any]:
        return list(self._rows)


class _FakeDb:
    """Minimal session double: the onboarding path only adds and re-reads rows.

    ``execute`` returns every message regardless of the statement; both readers
    in this path filter in Python (``ceremony_key`` / ``turn_key``), so the
    coarse result is enough and keeps the double honest about what it fakes.
    """

    def __init__(self, rows: dict[Any, Any]) -> None:
        self._rows = rows
        self.messages: list[InterviewSessionMessage] = []

    async def get(self, model: Any, pk: Any) -> Any:
        del pk
        return self._rows.get(model)

    async def execute(self, statement: Any) -> _FakeResult:
        del statement
        return _FakeResult(self.messages)

    def add(self, instance: Any) -> None:
        self.messages.append(instance)

    async def flush(self) -> None:
        return None

    async def rollback(self) -> None:  # pragma: no cover - no conflict in this path
        return None


def _set_name_fixture() -> tuple[_FakeDb, InterviewSession, CurrentUser]:
    student_id = uuid4()
    session = InterviewSession(
        id=uuid4(),
        student_id=student_id,
        interview_config_id=uuid4(),
        status="in_progress",
        onboarding_stage="identity_check",
        interview_language="en",
        input_mode="text",
        preferred_name=None,
    )
    db = _FakeDb(
        {
            InterviewSession: session,
            InterviewConfig: SimpleNamespace(
                title="Data Structures",
                persona="neutral",
                persona_profile_json=None,
                time_limit_minutes=30,
            ),
            UserProfile: UserProfile(given_name="Alexander", display_name="Alexander Tran"),
        }
    )
    actor = CurrentUser(user_id=student_id, session_id=uuid4(), permissions=frozenset())
    return db, session, actor


@pytest.mark.asyncio
async def test_preferred_name_ack_is_persisted_as_an_approved_ai_message() -> None:
    """The ack the candidate hears must exist as an AI row, or it is silent.

    ``POST /narration`` is an output boundary: it synthesizes only text that is
    already persisted as an approved AI utterance. The ack used to be returned
    without ever being written, so the browser's narration request 400'd and the
    line was the one turn of the whole ceremony nobody ever spoke.
    """
    db, session, actor = _set_name_fixture()

    result = await onboarding_service.respond(
        db,  # type: ignore[arg-type]
        session_id=session.id,
        actor=actor,
        stage="identity_check",
        response_text="Xà Điểu",
        action="set_name",
        language="en",
        turn_key="turn-1",
    )

    assert result.ai_text is not None
    assert "Xà Điểu" in result.ai_text
    approved = [
        message.content_text
        for message in db.messages
        if message.role == "ai" and message.content_text == result.ai_text.strip()
    ]
    assert approved, "spoken ack is not an approved (persisted) AI utterance"


@pytest.mark.asyncio
async def test_preferred_name_ack_replaces_the_generic_audio_check_turn() -> None:
    """One audio-check beat, not two.

    The ack already carries "Can you hear me clearly?", so it becomes the
    audio_check ceremony row instead of being spoken alongside a second,
    generic one the candidate never saw.
    """
    db, session, actor = _set_name_fixture()

    result = await onboarding_service.respond(
        db,  # type: ignore[arg-type]
        session_id=session.id,
        actor=actor,
        stage="identity_check",
        response_text="Xà Điểu",
        action="set_name",
        language="en",
        turn_key="turn-1",
    )

    assert session.onboarding_stage == "audio_check"
    assert session.preferred_name == "Xà Điểu"
    ai_messages = [message for message in db.messages if message.role == "ai"]
    assert len(ai_messages) == 1
    assert ai_messages[0].content_text == result.ai_text
    assert (ai_messages[0].metadata_json or {}).get("ceremony_key") == "audio_check"
