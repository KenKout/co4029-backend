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

from pydantic import BaseModel, ConfigDict


class StudyPlanItem(BaseModel):
    """One actionable remediation step on the student-facing study plan."""

    model_config = ConfigDict(from_attributes=True)

    topic: str
    lesson_id: UUID | None = None
    suggested_resources: list[str] = []


class GapReportRead(BaseModel):
    """Student-facing projection of a ``GapReport`` row.

    Carries the human-readable discrepancy summary, the actionable
    study plan, and a per-criterion breakdown (outcome → verdict).
    Internal LLM rationale stays in
    :class:`GapReportAuthoringRead`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    student_id: UUID
    course_id: UUID
    module_id: UUID | None = None
    discrepancy_summary: str | None = None
    study_plan: list[StudyPlanItem] = []
    per_criterion_breakdown: dict[str, Any] = {}
    generated_at: datetime


class GapReportAuthoringRead(GapReportRead):
    """Teacher-facing projection of a ``GapReport`` row.

    Inherits the student-facing schema and re-introduces:

    * ``raw_evaluation_json`` — per-response LLM verdicts (the
      backing :class:`~abridgeai.features.interviews.models.InterviewOutcomeEvaluation`
      rows joined into one payload).
    * ``teacher_summary`` — instructor-authored commentary.
    * ``source_quiz_attempt_id`` / ``source_interview_session_id`` —
      anchor links so the teacher can drill into the original attempts.
    """

    raw_evaluation_json: dict[str, Any] = {}
    teacher_summary: str | None = None
    source_quiz_attempt_id: UUID | None = None
    source_interview_session_id: UUID | None = None


__all__ = [
    "GapReportAuthoringRead",
    "GapReportRead",
    "StudyPlanItem",
]
