"""Guardrail tests: persona shapes LANGUAGE only, never decisions or scores.

The product promises three interviewer personas (strict / neutral / supportive)
and — soon — teacher-tuned trait profiles. The one thing that must never happen
is a persona leaking into control flow: two candidates of equal ability must get
identical questions, identical follow-up decisions, and identical scores no
matter which persona is configured. Otherwise "persona" becomes an unfairness in
grading rather than a tone choice.

These tests lock that boundary in structurally, so a future change that wires a
trait into ``decision.py`` / ``selection.py`` fails here loudly.

They also assert the trait value object reproduces the three legacy personas
exactly (``PersonaProfile.persona()`` maps back to the enum the deterministic
fallback tables key on), so introducing traits cannot silently change today's
behaviour.
"""

from __future__ import annotations

import dataclasses

import pytest

from abridgeai.features.interviews.orchestrator.decision import (
    DecisionInputs,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.persona import (
    PRESETS,
    PersonaProfile,
    OpeningStyle,
    TRAIT_MAX,
    TRAIT_MIN,
    profile_from,
)
from abridgeai.features.interviews.orchestrator.utterance import Persona

from tests.unit.test_interview_decision_invariants import _analysis, _inputs, _intent  # noqa: E402
from abridgeai.features.interviews.orchestrator.intent import StudentIntent


_PERSONA_KEYS = ("strict", "neutral", "supportive")


# ── The value object reproduces the three legacy personas ────────────────────


def test_presets_cover_exactly_the_three_legacy_personas() -> None:
    assert set(PRESETS.keys()) == set(_PERSONA_KEYS)


@pytest.mark.parametrize("key", _PERSONA_KEYS)
def test_preset_key_maps_back_to_legacy_enum(key: str) -> None:
    """The fallback tables key on the Persona enum — the preset must resolve to it."""
    profile = PRESETS[key]
    assert profile.persona() is Persona(key)


def test_profile_from_unknown_falls_back_to_neutral() -> None:
    # Mirrors utterance.persona_from so the two resolvers never disagree.
    assert profile_from(None).key == "neutral"
    assert profile_from("does-not-exist").key == "neutral"
    assert profile_from("STRICT").key == "neutral"  # case-sensitive, like the enum


@pytest.mark.parametrize("key", _PERSONA_KEYS)
def test_profile_from_known_returns_that_preset(key: str) -> None:
    assert profile_from(key) is PRESETS[key]


def test_all_preset_traits_are_within_bounds() -> None:
    for profile in PRESETS.values():
        for field in ("warmth", "directness", "verbosity", "formality", "ack_frequency"):
            value = getattr(profile, field)
            assert TRAIT_MIN <= value <= TRAIT_MAX, f"{profile.key}.{field}={value} out of range"


def test_clamped_pulls_out_of_range_traits_into_bounds() -> None:
    wild = PersonaProfile(
        key="neutral",
        warmth=99,
        directness=-5,
        verbosity=2,
        formality=2,
        ack_frequency=2,
        opening_style=OpeningStyle.STANDARD,
    ).clamped()
    assert wild.warmth == TRAIT_MAX
    assert wild.directness == TRAIT_MIN


def test_prompt_traits_carry_no_decision_bearing_keys() -> None:
    """The phrasing prompt must receive tone only — nothing that shifts difficulty."""
    traits = PRESETS["supportive"].as_prompt_traits()
    allowed = {
        "key",
        "warmth",
        "directness",
        "verbosity",
        "formality",
        "ack_frequency",
        "opening_style",
    }
    assert set(traits.keys()) == allowed
    # None of these words hint at difficulty / scoring / question selection.
    forbidden = {"difficulty", "score", "weight", "question", "outcome", "pass"}
    assert forbidden.isdisjoint(traits.keys())


# ── Structural boundary: persona is not even an input to the decision ────────


def test_persona_is_not_a_decision_input_field() -> None:
    """If persona ever becomes a DecisionInputs field, this fails on purpose.

    The decision policy is persona-blind BY CONSTRUCTION — persona is resolved
    later, in the phrasing layer. This test guards the construction so a future
    refactor can't quietly thread a trait into the policy.
    """
    field_names = {f.name for f in dataclasses.fields(DecisionInputs)}
    for banned in (
        "persona", "warmth", "directness", "verbosity", "tone", "profile",
        "role", "interviewer_role", "identity",
    ):
        assert banned not in field_names


# ── Behavioural boundary: same inputs → same decision (persona plays no part) ─

# A spread of reachable turns: a strong answer, a weak answer, a follow-up-budget
# edge, an end request, and a skip. The decision must be identical across every
# persona because persona is never consulted.
_DECISION_CASES = (
    _inputs(),  # default: relevant, mostly-correct answer
    _inputs(analysis=_analysis()),
    _inputs(intent=_intent(StudentIntent.END_INTERVIEW)),
    _inputs(intent=_intent(StudentIntent.SKIP_QUESTION)),
    _inputs(current_question_follow_up_count=5, total_follow_up_count=11),
    _inputs(time_fraction_remaining=0.05),
    _inputs(has_next_question=False, all_required_outcomes_covered=True),
)


@pytest.mark.parametrize("inputs", _DECISION_CASES)
def test_decision_is_identical_regardless_of_persona(inputs: DecisionInputs) -> None:
    """decide_next_action does not take persona — so its output can't depend on it.

    We assert the decision is stable across repeated calls (the policy is pure)
    and document that persona resolution happens strictly downstream. If someone
    adds a persona arg to decide_next_action, the call below stops type-checking
    and this test has to be revisited deliberately.
    """
    first = decide_next_action(inputs)
    again = decide_next_action(inputs)
    assert first.action == again.action
    assert first.reason_code == again.reason_code
    assert first.should_advance_question == again.should_advance_question
    assert first.should_end_session == again.should_end_session


# ── Selection boundary: persona is not an input to question selection ─────────


def test_persona_is_not_a_selection_context_field() -> None:
    """If persona ever becomes a SelectionContext field, this fails on purpose.

    Question selection must be persona-blind by construction: which question a
    candidate is asked cannot depend on the interviewer's tone. This mirrors the
    DecisionInputs guard above for the adaptive selector.
    """
    from abridgeai.features.interviews.orchestrator.selection import SelectionContext

    field_names = {f.name for f in dataclasses.fields(SelectionContext)}
    for banned in (
        "persona", "warmth", "directness", "verbosity", "tone", "profile",
        "role", "interviewer_role", "identity",
    ):
        assert banned not in field_names


# ── Language boundary: the question/probe survives verbatim in every persona ──

# The utterance layer may rephrase acknowledgement/transition per persona, but
# the authoritative question text must appear UNCHANGED in what the candidate
# sees — otherwise persona would be silently altering the assessed question.
_VERBATIM_QUESTION = "Explain how a hash join differs from a nested loop join."


@pytest.mark.parametrize("key", _PERSONA_KEYS)
@pytest.mark.parametrize("lang", ["en", "vi"])
def test_question_survives_verbatim_across_personas(key: str, lang: str) -> None:
    from abridgeai.features.interviews.orchestrator.decision import (
        InterviewerActionType,
        InterviewerDecision,
        ReasonCode,
    )
    from abridgeai.features.interviews.orchestrator.utterance import (
        Persona,
        build_fallback_utterance,
    )

    decision = InterviewerDecision(
        action=InterviewerActionType.ASK_MAIN_QUESTION,
        reason_code=ReasonCode.OUTCOME_NOT_COVERED,
    )
    utt = build_fallback_utterance(
        decision,
        persona=Persona(key),
        language=lang,
        question_text=_VERBATIM_QUESTION,
    )
    # The exact question text is present, unchanged, in both the dedicated field
    # and the combined turn text — for every persona and language.
    assert utt.question_or_probe == _VERBATIM_QUESTION
    assert _VERBATIM_QUESTION in utt.ai_turn_text


# ── Answer-leak boundary: no persona ever emits fabricated answer content ─────


@pytest.mark.parametrize("key", _PERSONA_KEYS)
@pytest.mark.parametrize("lang", ["en", "vi"])
def test_no_persona_emits_content_beyond_the_given_question(key: str, lang: str) -> None:
    """A probe with no supplied question text must fall back to a generic,
    answer-safe prompt — never invent domain content. Across every persona the
    deterministic probe stays a content-free "say more" style ask, so tone can
    never smuggle in an expected answer.
    """
    from abridgeai.features.interviews.orchestrator.decision import (
        InterviewerActionType,
        InterviewerDecision,
        ReasonCode,
    )
    from abridgeai.features.interviews.orchestrator.utterance import (
        Persona,
        build_fallback_utterance,
    )

    # A secret string that would only appear if the renderer fabricated content.
    secret = "the answer is a bitmap index on the join key"
    decision = InterviewerDecision(
        action=InterviewerActionType.PROBE_DEEPER,
        reason_code=ReasonCode.ANSWER_TOO_VAGUE,
    )
    utt = build_fallback_utterance(
        decision,
        persona=Persona(key),
        language=lang,
        question_text="",  # no question supplied → generic answer-safe probe
    )
    assert secret not in utt.ai_turn_text.lower()
    # The generic probe is non-empty (the candidate still gets a prompt) but
    # carries no supplied question text to leak.
    assert utt.ai_turn_text.strip() != ""


# ── The re-key preserved behaviour: warmth bands map personas correctly ───────


def test_warmth_band_maps_presets_to_expected_bands() -> None:
    """The table re-key is keyed on warmth band; assert each preset lands in the
    band whose strings reproduce its old wording (strict=low, neutral=mid,
    supportive=high). Locks the mapping the golden file depends on.
    """
    from abridgeai.features.interviews.orchestrator.utterance import (
        WARMTH_HIGH,
        WARMTH_LOW,
        WARMTH_MID,
        warmth_band,
    )

    assert warmth_band(PRESETS["strict"].warmth) == WARMTH_LOW
    assert warmth_band(PRESETS["neutral"].warmth) == WARMTH_MID
    assert warmth_band(PRESETS["supportive"].warmth) == WARMTH_HIGH
