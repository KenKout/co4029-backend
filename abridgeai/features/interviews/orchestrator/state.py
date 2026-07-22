"""Adaptive-interviewer runtime state schema + (de)serialization (Phase 1).

This is the typed, transport-agnostic shape of the orchestrator's per-session
memory. It is persisted as JSONB on ``InterviewRuntimeState.state_json`` (plus
a few hot columns: ``phase``, ``state_version``, ``last_turn_idempotency_key``).

Design notes
------------
* Pure data + (de)serialization only — NO DB access, NO LLM calls, NO policy.
  Keeping this module side-effect-free makes it trivially unit-testable and
  reusable from the REST path, the LiveKit bridge, and the evaluator.
* Forward/backward compatible: ``from_dict`` tolerates missing keys (older rows)
  by falling back to defaults, and ignores unknown keys (newer rows read by
  older code). This is what makes lazy initialisation of pre-existing sessions
  safe — an empty ``{}`` deserializes to a valid default state.
* ``version`` inside the payload is the schema version of THIS structure (bump
  when the shape changes incompatibly); it is distinct from the DB
  ``state_version`` optimistic-lock counter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# Current schema version of the serialized state payload. Bump on incompatible
# shape changes so ``from_dict`` can migrate old payloads if ever needed.
STATE_SCHEMA_VERSION = 5


class InterviewPhase(str, Enum):  # noqa: UP042 -- StrEnum changes value coercion; match codebase convention
    OPENING = "opening"
    WARMUP = "warmup"
    CORE = "core"
    DEEP_PROBE = "deep_probe"
    CLOSING = "closing"
    COMPLETED = "completed"


class CoverageStatus(str, Enum):  # noqa: UP042 -- StrEnum changes value coercion; match codebase convention
    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class InteractionState(str, Enum):  # noqa: UP042 -- match codebase convention
    """Per-turn interaction lifecycle (Slice 4).

    This is a SEPARATE axis from ``InterviewPhase``. ``InterviewPhase`` tracks
    *progress* through the interview (opening → core → closing → completed);
    ``InteractionState`` tracks what the current *turn* is waiting on. Keeping
    them distinct avoids overloading ``phase`` — e.g. an end-confirmation can be
    pending (``CONFIRMING_END``) while the phase is still ``CORE``.

    Most turns sit in ``AWAITING_ANSWER``. ``CONFIRMING_END`` is the only state
    that currently changes control flow (it gates the confirm/cancel handling);
    the others are reserved for richer client rendering and future slices, and
    default deserialization is tolerant so older rows load as ``AWAITING_ANSWER``.
    """

    ASKING = "asking"
    AWAITING_ANSWER = "awaiting_answer"
    ANALYZING = "analyzing"
    CONFIRMING_END = "confirming_end"
    CLOSING = "closing"
    COMPLETED = "completed"


@dataclass
class OutcomeCoverageState:
    """Provisional (runtime) coverage of a single learning outcome.

    This is guidance for interview decisions ONLY — never the final verdict.
    The post-session evaluator independently re-judges the transcript.
    """

    outcome_id: str
    # Raw count of evidence items attributed to this outcome (audit / traceability).
    evidence_count: int = 0
    # Weighted provisional coverage (Slice 2): confident supporting evidence adds
    # 2, confident partial support adds 1, contradiction/insufficient/low-confidence
    # add 0. An outcome is provisionally sufficient at COVERAGE_SUFFICIENT_POINTS.
    # This — not the raw evidence_count — drives runtime selection sufficiency.
    coverage_points: int = 0
    provisional_score: float | None = None
    confidence: float = 0.0
    status: CoverageStatus = CoverageStatus.NOT_STARTED
    last_updated_at: str | None = None
    supporting_turn_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    # Bounded log of the candidate's own prior claims about this outcome (Slice
    # 9, v2). Fed back into answer analysis so the interviewer can spot a
    # cross-turn contradiction ("earlier you said X"). Their words only — never
    # rubric/answer content. Bounded to the last few by the writer.
    claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeCoverageState:
        evidence_count = int(data.get("evidence_count", 0))
        # Backfill for sessions persisted before weighted coverage existed: when
        # coverage_points is absent, seed it from the raw count so a resumed
        # in-flight session keeps its already-earned coverage instead of
        # resetting every outcome to "uncovered".
        coverage_points = int(data.get("coverage_points", evidence_count) or 0)
        return cls(
            outcome_id=str(data.get("outcome_id", "")),
            evidence_count=evidence_count,
            coverage_points=coverage_points,
            provisional_score=data.get("provisional_score"),
            confidence=float(data.get("confidence", 0.0)),
            status=_coverage_status(data.get("status")),
            last_updated_at=data.get("last_updated_at"),
            supporting_turn_ids=list(data.get("supporting_turn_ids", []) or []),
            missing_evidence=list(data.get("missing_evidence", []) or []),
            claims=list(data.get("claims", []) or []),
        )


@dataclass
class CandidateSignals:
    requested_repeat: bool = False
    requested_clarification: bool = False
    requested_skip: bool = False
    appeared_off_topic: bool = False
    appeared_uncertain: bool = False
    technical_issue_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CandidateSignals:
        data = data or {}
        return cls(
            requested_repeat=bool(data.get("requested_repeat", False)),
            requested_clarification=bool(data.get("requested_clarification", False)),
            requested_skip=bool(data.get("requested_skip", False)),
            appeared_off_topic=bool(data.get("appeared_off_topic", False)),
            appeared_uncertain=bool(data.get("appeared_uncertain", False)),
            technical_issue_detected=bool(data.get("technical_issue_detected", False)),
        )


@dataclass
class InterviewRuntimeStateData:
    """The full serialized orchestrator state payload (stored in state_json)."""

    phase: InterviewPhase = InterviewPhase.OPENING

    started_at: str | None = None
    remaining_time_seconds: int | None = None

    current_question_id: str | None = None
    current_outcome_id: str | None = None

    asked_question_ids: list[str] = field(default_factory=list)
    skipped_question_ids: list[str] = field(default_factory=list)
    completed_question_ids: list[str] = field(default_factory=list)

    current_question_follow_up_count: int = 0
    total_follow_up_count: int = 0

    # Phase-dwell tracking (Slice 7). ``turns_in_phase`` counts turns spent in
    # the CURRENT phase (reset to 0 on any phase change); the phase policy uses
    # it to decide when to advance OPENING → WARMUP → CORE. ``warmup_turns_target``
    # is how many warmup turns to run before entering CORE (authorable later).
    turns_in_phase: int = 0
    warmup_turns_target: int = 1

    outcome_coverage: dict[str, OutcomeCoverageState] = field(default_factory=dict)

    last_student_intent: dict[str, Any] | None = None
    last_answer_analysis: dict[str, Any] | None = None

    consecutive_weak_answers: int = 0
    consecutive_strong_answers: int = 0

    # Per-turn interaction lifecycle (Slice 4) — a SEPARATE axis from ``phase``.
    # ``pending_confirmation`` is True only while an end-confirmation is awaiting
    # the candidate's yes/no; it gates the confirm/cancel branch and is cleared
    # on either resolution.
    interaction_state: InteractionState = InteractionState.AWAITING_ANSWER
    pending_confirmation: bool = False

    candidate_signals: CandidateSignals = field(default_factory=CandidateSignals)

    # Bounded prompt-injection security state. Raw student content is never
    # stored here; only counters, enums, a short fingerprint, and a turn key.
    security_assessment_count: int = 0
    security_attempt_count: int = 0
    consecutive_security_attempts: int = 0
    repeated_security_attempt_count: int = 0
    output_leakage_prevented_count: int = 0
    security_fallback_count: int = 0
    last_security_category: str | None = None
    last_security_action: str | None = None
    last_security_fingerprint: str | None = None
    last_security_turn_key: str | None = None
    security_warning_issued: bool = False
    session_security_flagged: bool = False

    # Schema version of this payload (NOT the DB optimistic-lock counter).
    version: int = STATE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "started_at": self.started_at,
            "remaining_time_seconds": self.remaining_time_seconds,
            "current_question_id": self.current_question_id,
            "current_outcome_id": self.current_outcome_id,
            "asked_question_ids": list(self.asked_question_ids),
            "skipped_question_ids": list(self.skipped_question_ids),
            "completed_question_ids": list(self.completed_question_ids),
            "current_question_follow_up_count": self.current_question_follow_up_count,
            "total_follow_up_count": self.total_follow_up_count,
            "turns_in_phase": self.turns_in_phase,
            "warmup_turns_target": self.warmup_turns_target,
            "outcome_coverage": {k: v.to_dict() for k, v in self.outcome_coverage.items()},
            "last_student_intent": self.last_student_intent,
            "last_answer_analysis": self.last_answer_analysis,
            "consecutive_weak_answers": self.consecutive_weak_answers,
            "consecutive_strong_answers": self.consecutive_strong_answers,
            "interaction_state": self.interaction_state.value,
            "pending_confirmation": self.pending_confirmation,
            "candidate_signals": self.candidate_signals.to_dict(),
            "security_assessment_count": self.security_assessment_count,
            "security_attempt_count": self.security_attempt_count,
            "consecutive_security_attempts": self.consecutive_security_attempts,
            "repeated_security_attempt_count": self.repeated_security_attempt_count,
            "output_leakage_prevented_count": self.output_leakage_prevented_count,
            "security_fallback_count": self.security_fallback_count,
            "last_security_category": self.last_security_category,
            "last_security_action": self.last_security_action,
            "last_security_fingerprint": self.last_security_fingerprint,
            "last_security_turn_key": self.last_security_turn_key,
            "security_warning_issued": self.security_warning_issued,
            "session_security_flagged": self.session_security_flagged,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> InterviewRuntimeStateData:
        """Deserialize, tolerating missing keys (old rows) + unknown keys (new).

        An empty dict yields a valid default state — this is exactly what makes
        lazy initialisation of a pre-existing session safe.
        """
        data = data or {}
        coverage_raw = data.get("outcome_coverage", {}) or {}
        coverage = {
            str(k): OutcomeCoverageState.from_dict(v)
            for k, v in coverage_raw.items()
            if isinstance(v, dict)
        }
        return cls(
            phase=_phase(data.get("phase")),
            started_at=data.get("started_at"),
            remaining_time_seconds=data.get("remaining_time_seconds"),
            current_question_id=data.get("current_question_id"),
            current_outcome_id=data.get("current_outcome_id"),
            asked_question_ids=list(data.get("asked_question_ids", []) or []),
            skipped_question_ids=list(data.get("skipped_question_ids", []) or []),
            completed_question_ids=list(data.get("completed_question_ids", []) or []),
            current_question_follow_up_count=int(data.get("current_question_follow_up_count", 0)),
            total_follow_up_count=int(data.get("total_follow_up_count", 0)),
            turns_in_phase=int(data.get("turns_in_phase", 0)),
            warmup_turns_target=int(data.get("warmup_turns_target", 1)),
            outcome_coverage=coverage,
            last_student_intent=data.get("last_student_intent"),
            last_answer_analysis=data.get("last_answer_analysis"),
            consecutive_weak_answers=int(data.get("consecutive_weak_answers", 0)),
            consecutive_strong_answers=int(data.get("consecutive_strong_answers", 0)),
            interaction_state=_interaction_state(data.get("interaction_state")),
            pending_confirmation=bool(data.get("pending_confirmation", False)),
            candidate_signals=CandidateSignals.from_dict(data.get("candidate_signals")),
            security_assessment_count=max(0, int(data.get("security_assessment_count", 0))),
            security_attempt_count=max(0, int(data.get("security_attempt_count", 0))),
            consecutive_security_attempts=max(0, int(data.get("consecutive_security_attempts", 0))),
            repeated_security_attempt_count=max(
                0, int(data.get("repeated_security_attempt_count", 0))
            ),
            output_leakage_prevented_count=max(
                0, int(data.get("output_leakage_prevented_count", 0))
            ),
            security_fallback_count=max(0, int(data.get("security_fallback_count", 0))),
            last_security_category=_optional_short_string(data.get("last_security_category")),
            last_security_action=_optional_short_string(data.get("last_security_action")),
            last_security_fingerprint=_optional_short_string(data.get("last_security_fingerprint")),
            last_security_turn_key=_optional_short_string(data.get("last_security_turn_key")),
            security_warning_issued=bool(data.get("security_warning_issued", False)),
            session_security_flagged=bool(data.get("session_security_flagged", False)),
            version=int(data.get("version", STATE_SCHEMA_VERSION)),
        )


def _phase(value: object) -> InterviewPhase:
    try:
        return InterviewPhase(value)
    except (ValueError, TypeError):
        return InterviewPhase.OPENING


def _interaction_state(value: object) -> InteractionState:
    try:
        return InteractionState(value)
    except (ValueError, TypeError):
        return InteractionState.AWAITING_ANSWER


def _coverage_status(value: object) -> CoverageStatus:
    try:
        return CoverageStatus(value)
    except (ValueError, TypeError):
        return CoverageStatus.NOT_STARTED


def _optional_short_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value[:255]
    return None


__all__ = [
    "STATE_SCHEMA_VERSION",
    "CandidateSignals",
    "CoverageStatus",
    "InteractionState",
    "InterviewPhase",
    "InterviewRuntimeStateData",
    "OutcomeCoverageState",
]
