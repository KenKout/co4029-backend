"""Gap-Report DTOs (T6.2).

Back the gap-analysis endpoints (T6.12):

* ``GET /students/{id}/gap-reports/{report_id}`` — student-facing
  read. Response: :class:`GapReportRead`.
* ``GET /teacher/gap-reports/{report_id}`` — teacher-facing read.
  Response: :class:`GapReportAuthoringRead`.

Both DTOs project the
:class:`~abridgeai.features.interviews.models.GapReport` row plus the
``report_json`` JSONB payload (per §A13 baseline keeps the analysis
in JSONB rather than dedicated columns — the schema layer flattens
the keys the UI needs and leaves the rest in
``raw_evaluation_json`` on the authoring DTO).

Security invariant
------------------

* :class:`GapReportRead` (student) MUST NOT carry:

  - ``raw_evaluation_json`` — per-response LLM rationale.
  - ``teacher_summary`` — author-only commentary.
  - ``source_quiz_attempt_id`` / ``source_interview_session_id`` —
    leaks the comparison anchors (a student doesn't need the
    cross-link; just the takeaways).

* :class:`GapReportAuthoringRead` re-introduces all of the above via
  inheritance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StudyPlanItem(BaseModel):
    """One actionable remediation step on the student-facing study plan."""

    model_config = ConfigDict(from_attributes=True)

    topic: str
    lesson_id: UUID | None = None
    suggested_resources: list[str] = []


class GapReportRead(BaseModel):
    """Student-facing projection of a ``GapReport`` row.

    Carries the human-readable discrepancy summary and the actionable
    study plan. FR-5.7 invariant: the student sees only the pass/fail
    verdict plus qualitative remediation — NOT numeric rubric scores.
    ``per_criterion_breakdown`` (criterion-level mean scores, e.g.
    ``technical_accuracy: 3.2``) is therefore teacher-only and lives on
    :class:`GapReportAuthoringRead`. Internal LLM rationale likewise
    stays teacher-side.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    student_id: UUID
    course_id: UUID
    module_id: UUID | None = None
    discrepancy_summary: str | None = None
    study_plan: list[StudyPlanItem] = []
    generated_at: datetime


class GapReportAuthoringRead(GapReportRead):
    """Teacher-facing projection of a ``GapReport`` row.

    Inherits the student-facing schema and re-introduces:

    * ``per_criterion_breakdown`` — criterion-level mean rubric scores
      (e.g. ``technical_accuracy: 3.2``). Teacher-only per FR-5.7: the
      student view exposes pass/fail + qualitative remediation only, so
      numeric rubric detail is surfaced here rather than on
      :class:`GapReportRead`.
    * ``raw_evaluation_json`` — per-response LLM verdicts (the
      backing :class:`~abridgeai.features.interviews.models.InterviewOutcomeEvaluation`
      rows joined into one payload).
    * ``teacher_summary`` — instructor-authored commentary.
    * ``source_quiz_attempt_id`` / ``source_interview_session_id`` —
      anchor links so the teacher can drill into the original attempts.
    """

    per_criterion_breakdown: dict[str, Any] = {}
    # Qualitative analysis surfaced from ``report_json``: short criterion-tagged
    # bullet phrases (e.g. "technical_accuracy: Cited specific bounds"). These
    # are the judge's per-criterion notes — the "why" behind the mean scores.
    strengths: list[str] = []
    weaknesses: list[str] = []
    # Quantitative rollup surfaced from ``internal_summary_json``: the weighted
    # session total (0-100), outcomes met/total, and answered/total/unanswered
    # question counts — the numbers that contextualize the per-criterion means.
    score_summary: dict[str, Any] = {}
    # Per-criterion rubric weights (sum to 1.0) so the teacher sees how much each
    # criterion contributes to the total — resolved from the interview config.
    rubric_weights: dict[str, float] = {}
    raw_evaluation_json: dict[str, Any] = {}
    teacher_summary: str | None = None
    source_quiz_attempt_id: UUID | None = None
    source_interview_session_id: UUID | None = None
    # Human-readable context so the teacher view isn't a wall of UUIDs. These
    # are resolved server-side (student display name, interview config title)
    # and are read-only projections, never persisted on the GapReport row.
    student_name: str | None = None
    interview_title: str | None = None


class GapReportNotesUpdate(BaseModel):
    """Teacher edit of the instructor-authored ``teacher_summary`` note."""

    model_config = ConfigDict(extra="forbid")

    teacher_summary: str | None = Field(default=None, max_length=5000)


__all__ = [
    "GapReportAuthoringRead",
    "GapReportNotesUpdate",
    "GapReportRead",
    "StudyPlanItem",
]
