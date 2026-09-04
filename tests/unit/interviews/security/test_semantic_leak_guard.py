"""Phase 3.1 — the semantic output guard, and why it records instead of blocking.

The load-bearing facts, in the order they matter:

1. **The grey-zone pre-filter decides whether anything is spent at all.** Turns
   that share no subject with a secret must return no candidates, so an ordinary
   interview pays for no embedding call.
2. **The allowed question is subtracted first.** The assigned question is related
   to its own model answer by construction, so failing to subtract it makes asking
   the question look like leaking the answer. This was a real bug.
3. **Failure is never a student's problem.** If the embedding service is down the
   verdict is "no leak", flagged ``degraded``. Blocking on an infrastructure fault
   would turn an outage into an assessment penalty.
4. **A crossed threshold is recorded, not enforced.** Measured on the live model,
   paraphrased leaks (0.496-0.816) overlap legitimate interviewer turns
   (0.034-0.588), so no cutoff separates them. See SECURITY_BASELINE.md.

The embedding client is stubbed here: these tests pin the wiring and the policy,
not the model's numbers. The threshold measurements live in the baseline doc and
were taken against the real service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest

from abridgeai.features.interviews.orchestrator.security import ProtectedContent
from abridgeai.features.interviews.orchestrator.semantic_leak import (
    SEMANTIC_LEAK_THRESHOLD,
    SemanticLeakAssessment,
    assess_semantic_leakage,
    grey_zone_leak_candidates,
)

_UUID_A = UUID("00000000-0000-0000-0000-0000000000a1")
_UUID_B = UUID("00000000-0000-0000-0000-0000000000b2")


@dataclass(frozen=True)
class _Settings:
    """Only the two flags ``guard_student_output`` reads."""

    mode: str
    semantic: bool

    @property
    def interview_security_guard_mode(self) -> str:
        return self.mode

    @property
    def interview_security_semantic_guard_enabled(self) -> bool:
        return self.semantic


def _async_return(value: Any) -> Any:
    async def _inner(*_: Any, **__: Any) -> Any:
        return value

    return _inner


def _semantic_hit(*, category: str) -> SemanticLeakAssessment:
    return SemanticLeakAssessment(
        blocked=True, protected_content_category=category, similarity=0.71
    )


QUESTION = "How does a database guarantee atomicity when a process fails unexpectedly?"
MODEL_ANSWER = (
    "A transaction uses write-ahead logging to preserve atomicity during "
    "unexpected process failures."
)
RUBRIC = (
    "Candidate must discuss isolation levels, compare optimistic versus pessimistic "
    "concurrency control, and evaluate durability trade-offs."
)


def _protected(*, include_question: bool = True) -> list[ProtectedContent]:
    items = [ProtectedContent(category="model_answer", text=MODEL_ANSWER)]
    if include_question:
        items.append(ProtectedContent(category="allowed_question_text", text=QUESTION))
    return items


class _StubEmbedder:
    """Returns fixed vectors so similarity is exact and offline."""

    def __init__(self, *vectors: list[float]) -> None:
        self._vectors = list(vectors)
        self.calls = 0

    async def embed(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls += 1
        assert len(self._vectors) == len(texts), (
            f"stub configured for {len(self._vectors)} texts, asked for {len(texts)}"
        )
        return self._vectors


class _FailingEmbedder:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    async def embed(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls += 1
        raise self._error


# ───────────────────── grey-zone pre-filter (cost control) ─────────────────────


def test_unrelated_turn_costs_nothing() -> None:
    """No shared subject means no candidate, so no embedding call is made."""
    assert grey_zone_leak_candidates("Thank you, that concludes the interview.", _protected()) == []


def test_a_paraphrase_is_offered_for_semantic_review() -> None:
    reworded = (
        "Think about how a durable log records each change before the commit happens, "
        "which is what keeps the transaction atomic when the process dies unexpectedly."
    )
    candidates = grey_zone_leak_candidates(reworded, _protected())
    assert [c.category for c in candidates] == ["model_answer"]


def test_the_allowed_question_is_subtracted_before_comparing() -> None:
    """Regression: asking the assigned question must not look like a leak.

    The question and its model answer are about the same thing by construction, so
    without subtracting the allowed text the guard flags the interviewer for doing
    its job.
    """
    with_question = grey_zone_leak_candidates(QUESTION, _protected(include_question=True))
    without_question = grey_zone_leak_candidates(QUESTION, _protected(include_question=False))
    assert with_question == [], "the allowed question was not subtracted"
    assert without_question, "sanity: the unsubtracted comparison does flag it"


def test_allowed_question_text_is_never_itself_a_candidate() -> None:
    for candidate in grey_zone_leak_candidates(
        "Let me ask about transactions and atomicity in a database again.", _protected()
    ):
        assert candidate.category != "allowed_question_text"


def test_short_turns_are_not_probed() -> None:
    assert grey_zone_leak_candidates("Yes, exactly.", _protected()) == []


# ───────────────────────── semantic comparison ─────────────────────────


@pytest.mark.asyncio
async def test_no_candidates_means_no_model_call() -> None:
    embedder = _StubEmbedder()
    result = await assess_semantic_leakage(
        None, proposed="anything", candidates=[], client=embedder
    )
    assert result.blocked is False
    assert embedder.calls == 0, "spent an embedding call with nothing to compare"


@pytest.mark.asyncio
async def test_identical_meaning_crosses_the_recording_threshold() -> None:
    embedder = _StubEmbedder([1.0, 0.0], [1.0, 0.0])
    result = await assess_semantic_leakage(
        None,
        proposed="reworded secret",
        candidates=[ProtectedContent(category="model_answer", text=MODEL_ANSWER)],
        client=embedder,
    )
    assert result.blocked is True
    assert result.protected_content_category == "model_answer"
    assert result.similarity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_orthogonal_meaning_does_not() -> None:
    embedder = _StubEmbedder([1.0, 0.0], [0.0, 1.0])
    result = await assess_semantic_leakage(
        None,
        proposed="unrelated",
        candidates=[ProtectedContent(category="model_answer", text=MODEL_ANSWER)],
        client=embedder,
    )
    assert result.blocked is False
    assert result.similarity == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_the_strongest_candidate_supplies_the_category() -> None:
    # proposal, then two secrets: the second is the closer match.
    embedder = _StubEmbedder([1.0, 0.0], [0.0, 1.0], [1.0, 0.0])
    result = await assess_semantic_leakage(
        None,
        proposed="reworded",
        candidates=[
            ProtectedContent(category="model_answer", text=MODEL_ANSWER),
            ProtectedContent(category="rubric_text", text=RUBRIC),
        ],
        client=embedder,
    )
    assert result.blocked is True
    assert result.protected_content_category == "rubric_text"


@pytest.mark.asyncio
async def test_everything_is_embedded_in_one_call() -> None:
    embedder = _StubEmbedder([1.0, 0.0], [0.0, 1.0], [0.0, 1.0])
    await assess_semantic_leakage(
        None,
        proposed="p",
        candidates=[
            ProtectedContent(category="model_answer", text=MODEL_ANSWER),
            ProtectedContent(category="rubric_text", text=RUBRIC),
        ],
        client=embedder,
    )
    assert embedder.calls == 1, "candidates should be batched, not embedded one by one"


# ───────────────────────── failure is non-blocking ─────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [OSError("connection reset"), ValueError("bad payload")]
)
async def test_an_embedding_failure_never_costs_the_student_a_turn(error: Exception) -> None:
    embedder = _FailingEmbedder(error)
    result = await assess_semantic_leakage(
        None,
        proposed="reworded secret",
        candidates=[ProtectedContent(category="model_answer", text=MODEL_ANSWER)],
        client=embedder,
    )
    assert result.blocked is False
    assert result.degraded is True, "a skipped check must be distinguishable from a clean pass"


@pytest.mark.asyncio
async def test_a_malformed_vector_count_degrades_instead_of_guessing() -> None:
    class _ShortEmbedder:
        async def embed(self, texts: list[str], **_: Any) -> list[list[float]]:
            return [[1.0, 0.0]]  # one vector for two texts

    result = await assess_semantic_leakage(
        None,
        proposed="reworded secret",
        candidates=[ProtectedContent(category="model_answer", text=MODEL_ANSWER)],
        client=_ShortEmbedder(),
    )
    assert result.blocked is False
    assert result.degraded is True


# ──────────────── the guard call site: records, never substitutes ────────────────


@pytest.mark.asyncio
async def test_guard_records_a_semantic_hit_without_altering_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with the stage enabled and the guard enforcing, the text is untouched.

    This is the load-bearing policy: the measured populations overlap, so a
    semantic hit is evidence for review, never grounds to replace a genuine
    interview question with a fallback.
    """
    from abridgeai.features.interviews.orchestrator.security import (
        SecurityAction,
        SecurityAssessment,
        SecurityCategory,
    )
    from abridgeai.features.interviews.services import security as service

    monkeypatch.setattr(
        service, "get_settings", lambda: _Settings(mode="enforce", semantic=True)
    )
    monkeypatch.setattr(
        service,
        "protected_content_for_config",
        _async_return(
            [
                ProtectedContent(category="model_answer", text=MODEL_ANSWER),
                ProtectedContent(category="allowed_question_text", text=QUESTION),
            ]
        ),
    )
    # Force the pre-filter to offer a candidate: this test is about what the call
    # site does with a hit, not about which turns qualify for one (covered above).
    monkeypatch.setattr(
        service,
        "grey_zone_leak_candidates",
        lambda *_, **__: [ProtectedContent(category="model_answer", text=MODEL_ANSWER)],
    )
    monkeypatch.setattr(
        service,
        "assess_semantic_leakage",
        _async_return(
            _semantic_hit(category="model_answer"),
        ),
    )
    recorded: list[dict[str, Any]] = []

    async def _record(_db: Any, **kwargs: Any) -> None:
        recorded.append(kwargs)

    monkeypatch.setattr(service, "record_security_event", _record)

    proposed = (
        "Think about how a durable log records each change before the commit happens."
    )
    result = await service.guard_student_output(
        None,
        session_id=_UUID_A,
        config_id=_UUID_B,
        turn_key="turn-1",
        proposed_text=proposed,
        fallback_text="I can repeat or clarify the current question.",
        allowed_question_ids=[],
        assessment=SecurityAssessment(
            category=SecurityCategory.BENIGN,
            detected=False,
            confidence=1.0,
            should_block=False,
            should_record_academic_evidence=False,
            response_key=None,
            normalized_fingerprint=None,
            source="rules",
        ),
        action=SecurityAction.ALLOW,
        attempt_count=0,
    )

    assert result.text == proposed, "a semantic hit must not substitute the fallback"
    assert result.output_fallback_used is False
    assert result.output_leakage_blocked is False
    assert len(recorded) == 1, "the hit must still be persisted for review"
    assert recorded[0]["fallback_status"] is False
    assert str(recorded[0]["turn_id"]).startswith("semantic:"), (
        "semantic hits need a distinct turn_id so they do not collide with the "
        "lexical event for the same turn"
    )


@pytest.mark.asyncio
async def test_guard_skips_the_stage_entirely_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from abridgeai.features.interviews.orchestrator.security import (
        SecurityAction,
        SecurityAssessment,
        SecurityCategory,
    )
    from abridgeai.features.interviews.services import security as service

    monkeypatch.setattr(
        service, "get_settings", lambda: _Settings(mode="enforce", semantic=False)
    )
    monkeypatch.setattr(
        service,
        "protected_content_for_config",
        _async_return([ProtectedContent(category="model_answer", text=MODEL_ANSWER)]),
    )
    called = False

    async def _should_not_run(*_: Any, **__: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(service, "assess_semantic_leakage", _should_not_run)

    result = await service.guard_student_output(
        None,
        session_id=_UUID_A,
        config_id=_UUID_B,
        turn_key="turn-1",
        proposed_text="Think about how a durable log records each change before commit.",
        fallback_text="fallback",
        allowed_question_ids=[],
        assessment=SecurityAssessment(
            category=SecurityCategory.BENIGN,
            detected=False,
            confidence=1.0,
            should_block=False,
            should_record_academic_evidence=False,
            response_key=None,
            normalized_fingerprint=None,
            source="rules",
        ),
        action=SecurityAction.ALLOW,
        attempt_count=0,
    )
    assert called is False, "disabled stage must not be reached"
    assert result.output_leakage_blocked is False


# ───────────────────────── the policy itself ─────────────────────────


def test_the_stage_is_opt_in() -> None:
    """Default OFF: a record-only stage must not add a model call by default.

    Enabling it is a deployment decision — the audit signal is worth an embedding
    call on ~12% of turns only where someone wants that trail.
    """
    from abridgeai.core.config import Settings

    assert Settings().interview_security_semantic_guard_enabled is False


def test_threshold_reflects_the_measured_overlap() -> None:
    """Pins the documented value so a silent tightening is visible in review.

    Measured leaks span 0.496-0.816 and benign interviewer turns 0.034-0.588, so a
    threshold at or below the benign ceiling would flag normal turns and one much
    higher would flag nothing. Any change needs fresh measurements — see
    SECURITY_BASELINE.md.
    """
    assert pytest.approx(0.60) == SEMANTIC_LEAK_THRESHOLD
    assert SEMANTIC_LEAK_THRESHOLD > 0.588, "would record legitimate interviewer turns"
    assert SEMANTIC_LEAK_THRESHOLD < 0.816, "would record nothing at all"
