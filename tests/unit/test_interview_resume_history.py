"""Regression coverage for learner transcript restoration on resume."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from abridgeai.features.interviews.routers.learner import _build_session_history


def _message(
    *,
    role: str,
    text: str,
    created_at: datetime,
    session_question_id: object | None = None,
    metadata: dict[str, str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        role=role,
        content_text=text,
        created_at=created_at,
        session_question_id=session_question_id,
        metadata_json=metadata or {},
    )


def test_resume_history_restores_onboarding_questions_and_answers_in_order() -> None:
    started_at = datetime(2026, 7, 19, 10, 0, tzinfo=UTC)
    assessment_started_at = started_at + timedelta(minutes=5)
    session = SimpleNamespace(
        onboarding_stage="completed",
        assessment_started_at=assessment_started_at,
    )

    first_id = uuid4()
    second_id = uuid4()
    first_session_question = SimpleNamespace(
        id=first_id,
        sequence_no=1,
        asked_at=started_at,
    )
    second_session_question = SimpleNamespace(
        id=second_id,
        sequence_no=2,
        asked_at=assessment_started_at + timedelta(seconds=65),
    )
    first_question = SimpleNamespace(
        prompt_text="Explain dependency injection.",
        question_type="technical",
    )
    second_question = SimpleNamespace(
        prompt_text="When would you use it?",
        question_type="situational",
    )

    messages = [
        _message(
            role="ai",
            text="Welcome. Please confirm your identity.",
            created_at=started_at,
            metadata={"ceremony_key": "candidate_confirmation"},
        ),
        _message(
            role="user",
            text="Yes, that is me.",
            created_at=started_at + timedelta(minutes=1),
        ),
        _message(
            role="ai",
            text="Let us begin.",
            created_at=assessment_started_at,
            metadata={"ceremony_key": "ready_transition"},
        ),
        _message(
            role="user",
            text="It supplies dependencies from outside a class.",
            created_at=assessment_started_at + timedelta(seconds=60),
            session_question_id=first_id,
        ),
    ]

    history = _build_session_history(
        session,
        [
            (first_session_question, first_question),
            (second_session_question, second_question),
        ],
        messages,
    )

    assert [(turn.role, turn.kind, turn.content_text) for turn in history] == [
        ("ai", "opening", "Welcome. Please confirm your identity."),
        ("user", "answer", "Yes, that is me."),
        ("ai", "transition", "Let us begin."),
        ("ai", "question", "Explain dependency injection."),
        ("user", "answer", "It supplies dependencies from outside a class."),
        ("ai", "question", "When would you use it?"),
    ]
    assert [turn.elapsed_seconds for turn in history] == [None, None, 0, 0, 60, 65]
    assert history[3].question_type == "technical"
    assert history[5].question_type == "situational"


def test_resume_history_keeps_greeting_questions_hidden_until_assessment() -> None:
    created_at = datetime(2026, 7, 19, 10, 0, tzinfo=UTC)
    session = SimpleNamespace(
        onboarding_stage="audio_check",
        assessment_started_at=None,
    )
    session_question = SimpleNamespace(id=uuid4(), sequence_no=1, asked_at=created_at)
    question = SimpleNamespace(prompt_text="Hidden technical question", question_type="technical")
    messages = [
        _message(
            role="ai",
            text="Can you hear me clearly?",
            created_at=created_at,
            metadata={"ceremony_key": "audio_check"},
        )
    ]

    history = _build_session_history(session, [(session_question, question)], messages)

    assert [turn.content_text for turn in history] == ["Can you hear me clearly?"]
    assert history[0].elapsed_seconds is None


def test_resume_history_preserves_clarification_and_hint_kinds() -> None:
    started = datetime(2026, 7, 20, 10, 0, tzinfo=UTC)
    question_id = uuid4()
    session = SimpleNamespace(onboarding_stage="completed", assessment_started_at=started)
    session_question = SimpleNamespace(id=question_id, sequence_no=1, asked_at=started)
    question = SimpleNamespace(prompt_text="Compare A and B.", question_type="technical")
    messages = [
        _message(
            role="user",
            text="Could you clarify this question?",
            created_at=started + timedelta(seconds=2),
            session_question_id=question_id,
            metadata={"kind": "clarification"},
        ),
        _message(
            role="ai",
            text="Put another way, describe similarities and differences.",
            created_at=started + timedelta(seconds=3),
            session_question_id=question_id,
            metadata={"kind": "clarification"},
        ),
        _message(
            role="ai",
            text="Organize the response around purpose and usage.",
            created_at=started + timedelta(seconds=4),
            session_question_id=question_id,
            metadata={"kind": "hint"},
        ),
    ]

    history = _build_session_history(session, [(session_question, question)], messages)

    assert [turn.kind for turn in history] == [
        "question",
        "clarification",
        "clarification",
        "hint",
    ]
