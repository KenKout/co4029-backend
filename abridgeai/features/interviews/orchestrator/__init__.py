"""Adaptive Interview Orchestrator (staged rollout).

Gated behind ``Settings.adaptive_interviewer_enabled``. Nothing here is wired
into the live request path yet — each slice is built and tested in isolation
before it drives anything, and the orchestrator always falls back to the
existing sequential ``take_session_step`` flow on failure.

Slices landed:
* Phase 1 — persistent runtime state (``state``, ``repository``).
* Phase 2 — student intent classification (``intent``, ``intent_logic``).
* Phase 3 — structured answer analysis (``analysis``, ``analysis_logic``).
* Perception pipeline (``pipeline``) ties intent + analysis together and
  persists provisional signals into the runtime state (idempotent,
  version-guarded). Not yet consumed by REST/LiveKit.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Completeness,
    Correctness,
    EvidenceType,
    OutcomeEvidence,
    ProbeType,
    Relevance,
    Specificity,
)
from abridgeai.features.interviews.orchestrator.decision import (
    AcknowledgementStyle,
    DecisionInputs,
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
    decide_next_action,
)
from abridgeai.features.interviews.orchestrator.intent import (
    NON_ACADEMIC_INTENTS,
    IntentClassification,
    StudentIntent,
)
from abridgeai.features.interviews.orchestrator.mapping import canonical_step_result
from abridgeai.features.interviews.orchestrator.pipeline import (
    ACADEMIC_INTENTS,
    PerceptionResult,
    perceive_turn,
)
from abridgeai.features.interviews.orchestrator.selection import (
    CandidateQuestion,
    ScoredCandidate,
    SelectionContext,
    score_candidate,
    select_next_question,
    sequential_pick,
)
from abridgeai.features.interviews.orchestrator.state import (
    CandidateSignals,
    CoverageStatus,
    InterviewPhase,
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.orchestrator.utterance import (
    Persona,
    Utterance,
    build_fallback_utterance,
    persona_from,
)

__all__ = [
    "Persona",
    "Utterance",
    "build_fallback_utterance",
    "canonical_step_result",
    "persona_from",
    "ACADEMIC_INTENTS",
    "NON_ACADEMIC_INTENTS",
    "AcknowledgementStyle",
    "AnswerAnalysis",
    "CandidateQuestion",
    "CandidateSignals",
    "Completeness",
    "Correctness",
    "CoverageStatus",
    "DecisionInputs",
    "EvidenceType",
    "IntentClassification",
    "InterviewPhase",
    "InterviewRuntimeStateData",
    "InterviewerActionType",
    "InterviewerDecision",
    "OutcomeCoverageState",
    "OutcomeEvidence",
    "PerceptionResult",
    "ProbeType",
    "ReasonCode",
    "Relevance",
    "ScoredCandidate",
    "SelectionContext",
    "Specificity",
    "StudentIntent",
    "decide_next_action",
    "perceive_turn",
    "score_candidate",
    "select_next_question",
    "sequential_pick",
]
