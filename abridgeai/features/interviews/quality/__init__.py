"""Post-hoc interview quality metrics (offline; never in the request path).

Three metrics ported conceptually from SALT-NLP/SparkMe's evaluation harness,
adapted for an *assessment* setting rather than qualitative research:

``calibration``
    Rule-based, no LLM. Compares the interview's own runtime belief about
    outcome coverage (``interview_runtime_states.state_json``, self-reported by
    the adaptive selector) against the post-session evaluator's independent
    verdicts (``interview_outcome_evaluations.verdict_met``). Answers: "when the
    interviewer decided an outcome had enough evidence and moved on, was it
    right?" Over-confidence is the dangerous direction — it means a student was
    passed over on an outcome the evaluator later judged unmet.

``contingency``
    LLM-as-judge. Does each follow-up genuinely build on what the student just
    said, or is it a scripted next question? SparkMe's rubric scores an
    interview that never follows up at 1, which is exactly the failure mode a
    question-bank-driven interviewer falls into.

``leading``
    LLM-as-judge. Does an interviewer turn introduce content, embed a
    presupposition, or otherwise hand the student part of the answer? This is a
    *validity* metric, not a quality one: in an exam setting a leading question
    inflates measured coverage while invalidating the measurement.

Everything here is read-only and runs out-of-band (CLI / analysis), so it can
never affect a live session.
"""

from __future__ import annotations

from abridgeai.features.interviews.quality.calibration import (
    CalibrationReport,
    OutcomeCalibration,
    compute_calibration,
)
from abridgeai.features.interviews.quality.transcript import (
    TranscriptTurn,
    build_qa_pairs,
    followup_pairs,
)

__all__ = [
    "CalibrationReport",
    "OutcomeCalibration",
    "TranscriptTurn",
    "build_qa_pairs",
    "compute_calibration",
    "followup_pairs",
]
