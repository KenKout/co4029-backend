"""Manager / faculty-dean dashboard DTOs (Tier-1 sections).

The manager-side counterpart to the teacher dashboard's
:class:`~abridgeai.features.courses.schemas.authoring.TeacherDashboardStats` /
:class:`~abridgeai.features.courses.schemas.authoring.PriorityTask` /
:class:`~abridgeai.features.courses.schemas.authoring.CourseHealthRow`, and it
is deliberately a SEPARATE set of types rather than a reuse of them.

Two reasons the types are not shared:

* ``PriorityTask.kind`` is a closed ``Literal`` of six TEACHER work kinds
  (student risk, pending questions, calibration, materials, overdue reviews).
  A manager's queue is about publish-readiness and program governance, which
  are different kinds of work; widening that Literal would let a teacher-facing
  consumer receive a variant it has no branch for.
* The teacher rows are scoped to courses the caller AUTHORS. These are scoped
  to an organization or faculty, so the same field name would carry a different
  meaning — a subtle way for two dashboards to disagree.

Every "cannot publish" verdict in :class:`BlockedCourseRow` is reproduced from
``assignment.get_course_readiness``'s conjunction over BATCHED data, so the
queue and the publish endpoint's 409 can never disagree.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

#: Why a course is in the blocked queue. Manager-side kinds only — deliberately
#: NOT an extension of ``PriorityTask.kind`` (see the module docstring).
BlockedCourseReasonCode = Literal[
    "no_gradeable_content",
    "no_learning_outcomes",
    "understaffed",
    "archived",
]


class BlockedCourseRow(BaseModel):
    """One course that cannot currently be published, worst first.

    A queue, not a report: only courses whose ``can_publish`` is False appear,
    because listing finished work buries the work that is not finished.

    ``blocks_required_stage`` is the highest severity signal on the page. A
    REQUIRED course with no gradeable unit does not merely fail to complete —
    it locks its stage and every stage behind it, for every student on that
    path. It is ranked above draft-vs-published for that reason.

    The raw counts travel with the verdict so the UI can show the ratio behind
    it ("1 of 2 teachers") instead of only a colour. ``reason`` is the same
    verdict as a human-readable sentence: a severity that exists only as a
    colour or a flag is unreadable to a screen reader and unactionable to a
    manager who cannot guess which of four gates failed.
    """

    model_config = ConfigDict(extra="forbid")

    course_id: UUID
    title: str
    slug: str
    status: Literal["draft", "published", "archived"]
    faculty_id: UUID | None = None
    organization_id: UUID

    # --- The publish gate, reproduced exactly ------------------------------
    #: ``teacher_count >= min_teachers OR status != 'draft'``. The staffing
    #: minimum is a FIRST-publish gate: an already-published course is
    #: grandfathered, so it must not appear understaffed forever.
    staffing_ok: bool
    teacher_count: int = 0
    #: Resolved ONCE per organization via ``courses.min_teachers_per_course``,
    #: not per course — every row in one organization carries the same value.
    min_teachers: int = 0
    gradeable_unit_count: int = 0
    learning_outcome_count: int = 0

    #: Highest severity: no gradeable unit AND the course is REQUIRED on at
    #: least one career path.
    blocks_required_stage: bool = False
    #: Machine-readable companion to ``reason``, in the same order, so the SPA
    #: can render per-gate chips without parsing the sentence.
    reason_codes: list[BlockedCourseReasonCode] = Field(default_factory=list)
    #: Human-readable list of what is missing, e.g. "No gradeable content; no
    #: learning outcomes". Never empty — a row only exists because something
    #: is missing.
    reason: str


class ProgramAttentionRow(BaseModel):
    """One learning program with unreleased or unreviewed work.

    Emitted when the program has an open draft version OR at least one open
    path-change request, i.e. it is mid-revise or someone is waiting on a
    decision.

    ``open_path_change_request_count`` counts OPEN statuses ONLY (``pending``
    plus ``in_progress``), matching the management list card. The per-program
    ``GET /management/learning-programs/{id}/path-change-requests`` drill-down
    returns EVERY status, so it will legitimately show more rows than this
    number — that is by design, not drift. Do not mix the two.
    """

    model_config = ConfigDict(extra="forbid")

    program_id: UUID
    name: str
    slug: str
    status: str
    organization_id: UUID
    faculty_id: UUID
    student_count: int = 0
    has_draft_version: bool = False
    open_path_change_request_count: int = 0
    #: Human-readable summary of why the row is here, same rule as
    #: ``BlockedCourseRow.reason``.
    reason: str


class ManagementDashboardCounts(BaseModel):
    """Headline counts, derived entirely from the two sections above.

    Adds no queries: every field is computed from the course and program data
    already fetched for sections A and B. A tile that needed its own query
    could drift from the table underneath it.

    ``courses_blocked`` is the length of the blocked queue, so the tile and
    the table can never disagree.
    """

    model_config = ConfigDict(extra="forbid")

    courses_total: int = 0
    courses_draft: int = 0
    courses_published: int = 0
    #: Courses whose ``can_publish`` is False — equals ``len(blocked_courses)``.
    courses_blocked: int = 0
    programs_total: int = 0
    programs_with_draft: int = 0

    #: DEAN-ONLY. Total open path-change requests across in-scope programs.
    #: ``None`` — not 0 — for a caller without ``learning_program.switch.review``
    #: (a manager). ``None`` means "not your job"; 0 would read as "no work
    #: waiting", which is a different and possibly false claim.
    open_path_change_requests: int | None = None


class ManagementDashboard(BaseModel):
    """The whole manager / faculty-dean dashboard in one payload.

    One response so the page is one round trip; see the router docstring for
    why it is not split per section.

    ``scope_kind`` / ``organization_id`` / ``org_unit_id`` are echoed back so
    the SPA can label the page truthfully ("Faculty of X" vs the whole
    organization) instead of guessing from the caller's role. A dashboard that
    silently shows a narrower or wider set than its heading claims is the
    failure this feature exists to fix.
    """

    model_config = ConfigDict(extra="forbid")

    scope_kind: Literal["course", "org_unit", "organization", "global"]
    organization_id: UUID | None = None
    org_unit_id: UUID | None = None
    #: True when the caller holds ``learning_program.switch.review``. Tells the
    #: SPA to render the review panel at all, so it need not infer intent from
    #: ``open_path_change_requests`` being null.
    can_review_path_changes: bool = False

    counts: ManagementDashboardCounts
    #: Server-sorted, worst first: ``blocks_required_stage`` desc, then draft
    #: before published, then title asc. The client must never re-rank — two
    #: surfaces ordering the same queue differently is how a manager loses
    #: track of which item is actually most urgent.
    blocked_courses: list[BlockedCourseRow] = Field(default_factory=list)
    #: Server-sorted: open requests desc (someone is waiting), then draft
    #: before not, then name asc.
    programs_needing_attention: list[ProgramAttentionRow] = Field(default_factory=list)


__all__ = [
    "BlockedCourseReasonCode",
    "BlockedCourseRow",
    "ManagementDashboard",
    "ManagementDashboardCounts",
    "ProgramAttentionRow",
]
