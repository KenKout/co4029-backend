"""Manager / faculty-dean dashboard composition (Tier-1 decision queue).

Why this module exists separately from the teacher dashboard
-----------------------------------------------------------
The teacher dashboard resolves its course set through
``courses.services.authoring._list_authorable_courses`` = courses the caller
OWNS union courses they are ASSIGNED to teach. A manager or a faculty dean holds
``course.create``, so they PASS the teacher dashboard's permission gate and
receive ``200 OK`` — with their own authored course set, which for a manager is
normally EMPTY.

That is the failure this module exists to avoid: it fails OPEN, producing a
dashboard that renders perfectly and reports nothing. So the scope here is
resolved from the caller's ROLE ASSIGNMENTS (faculty for a dean, organization
for a manager), reusing the same primitive the ``/dept/courses`` staffing list
already uses, and the scope actually applied is echoed back in the response so
the SPA can label the page truthfully instead of guessing.

Parity with the publish gate
----------------------------
Every "cannot publish" verdict reproduces the conjunction in
``assignment.get_course_readiness`` exactly, over BATCHED data. That function
documents its ``can_publish`` as the same condition ``publish_course`` gates on,
so a checklist that disagrees with the 409 is worse than no checklist: the
manager trusts the green tick and blames the button. The four batched queries in
``queries.authoring`` carry the per-course predicates verbatim for the same
reason.

Cost
----
``get_course_readiness`` is per-course and issues ~6 reads; fanning it over an
organization would be 6xN on the critical path. This composes the same answer
from five batched queries plus one settings read, regardless of course count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.runtime_settings import resolve_setting
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.courses.queries import authoring as authoring_queries
from abridgeai.features.courses.schemas.management_dashboard import (
    BlockedCourseReasonCode,
    BlockedCourseRow,
    ManagementDashboard,
    ManagementDashboardCounts,
    ProgramAttentionRow,
)
from abridgeai.features.courses.services import assignment as assignment_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser
    from abridgeai.features.courses.schemas.authoring import CourseAuthoring

#: Permission that makes a caller a path-change REVIEWER (Faculty Dean). A
#: manager does not hold it, and the distinction is surfaced rather than hidden:
#: see ``ManagementDashboardCounts.open_path_change_requests``.
_REVIEW_PERMISSION = "learning_program.switch.review"

#: Most-specific-wins ordering, mirroring ``routers.assignment._SCOPE_PRIORITY``.
#: Kept in step with it deliberately: two surfaces that resolve "my scope"
#: differently would show a dean two different course sets.
_SCOPE_PRIORITY: dict[str, int] = {
    "course": 1,
    "org_unit": 2,
    "organization": 3,
    "global": 4,
}


async def _resolve_scope(
    db: AsyncSession, user_id: UUID
) -> tuple[str, UUID | None, UUID | None, list[UUID]]:
    """``(scope_kind, organization_id, org_unit_id, faculty_ids)`` for a caller.

    Same resolution as ``routers.assignment._resolve_caller_scope``, plus the
    full faculty list: a dean may hold ``org_unit`` assignments on several
    faculties, and showing only the most specific one would silently hide the
    rest of their remit.
    """
    assignments = await access_control_api.get_role_assignments_for_user(db, user_id)
    if not assignments:
        return ("global", None, None, [])
    most_specific = min(assignments, key=lambda a: _SCOPE_PRIORITY.get(a.scope_kind, 5))
    faculty_ids = list(
        {
            UUID(str(row.org_unit_id))
            for row in assignments
            if row.scope_kind == "org_unit" and row.org_unit_id is not None
        }
    )
    return (
        most_specific.scope_kind,
        most_specific.organization_id,
        most_specific.org_unit_id,
        faculty_ids,
    )


async def _courses_in_scope(
    db: AsyncSession, scope_kind: str, organization_id: UUID | None, faculty_ids: list[UUID]
) -> list[CourseAuthoring]:
    """Courses the caller governs — NOT the courses they personally author."""
    if scope_kind == "org_unit" and faculty_ids:
        return await assignment_service.list_courses_in_faculties(db, faculty_ids)
    # organization / global / course fall through to the org list. A ``global``
    # caller (admin) passing organization_id=None sees every course, which is
    # what the equivalent /dept/courses branch does.
    return await assignment_service.list_courses_for_organization(db, organization_id)


def _describe(
    *,
    units: int,
    outcomes: int,
    staffing_ok: bool,
    teacher_count: int,
    min_teachers: int,
    archived: bool,
) -> tuple[list[BlockedCourseReasonCode], str]:
    """Machine codes + one human sentence for why a course cannot publish.

    Both shapes ship together and in the same order: the codes let the SPA
    render a chip per failed gate, while the sentence is what a screen reader
    and a manager in a hurry actually read. A severity expressed only as a
    colour is unactionable — it does not say WHICH of four gates failed.
    """
    codes: list[BlockedCourseReasonCode] = []
    parts: list[str] = []
    if archived:
        codes.append("archived")
        parts.append("Course is archived")
    if units == 0:
        codes.append("no_gradeable_content")
        parts.append("No gradeable content (no published lesson, quiz or interview)")
    if outcomes == 0:
        codes.append("no_learning_outcomes")
        parts.append("No learning outcomes")
    if not staffing_ok:
        codes.append("understaffed")
        parts.append(f"Understaffed ({teacher_count} of {min_teachers} teachers)")
    return codes, "; ".join(parts)


async def _blocked_courses(
    db: AsyncSession, courses: list[CourseAuthoring], organization_id: UUID | None
) -> list[BlockedCourseRow]:
    """Courses whose ``can_publish`` is False, worst first.

    A queue, not a report: publishable courses are filtered out, because listing
    finished work buries the work that is not finished.
    """
    if not courses:
        return []
    course_ids = [course.id for course in courses]

    units_by_course = await authoring_queries.gradeable_unit_count_for_courses(db, course_ids)
    teachers_by_course = await authoring_queries.teacher_count_for_courses(db, course_ids)
    # NOT len(course.outcomes): the list endpoints leave that field empty, so
    # deriving the count from the DTO would accuse every course in the
    # organization of having no outcomes.
    outcomes_by_course = await authoring_queries.outcome_count_for_courses(db, course_ids)
    required_path_ids = await authoring_queries.required_path_course_ids(db, course_ids)

    # Resolved ONCE per organization, not per course: it is an org-level policy
    # and a per-course read would be N settings lookups for one value.
    min_teachers = int(
        await resolve_setting(db, "courses.min_teachers_per_course", organization_id)
    )

    rows: list[BlockedCourseRow] = []
    for course in courses:
        units = units_by_course.get(course.id, 0)
        outcomes = outcomes_by_course.get(course.id, 0)
        teacher_count = teachers_by_course.get(course.id, 0)
        archived = course.status == "archived"
        # The staffing minimum is a FIRST-publish gate: an already-published
        # course is grandfathered, so raising the minimum must not retroactively
        # mark live courses understaffed.
        staffing_ok = teacher_count >= min_teachers or course.status != "draft"
        can_publish = units > 0 and outcomes > 0 and not archived and staffing_ok
        if can_publish:
            continue
        codes, reason = _describe(
            units=units,
            outcomes=outcomes,
            staffing_ok=staffing_ok,
            teacher_count=teacher_count,
            min_teachers=min_teachers,
            archived=archived,
        )
        rows.append(
            BlockedCourseRow(
                course_id=course.id,
                title=course.title,
                slug=course.slug,
                status=course.status,
                faculty_id=course.faculty_id,
                organization_id=course.organization_id,
                staffing_ok=staffing_ok,
                teacher_count=teacher_count,
                min_teachers=min_teachers,
                gradeable_unit_count=units,
                learning_outcome_count=outcomes,
                blocks_required_stage=units == 0 and course.id in required_path_ids,
                reason_codes=codes,
                reason=reason,
            )
        )

    # Sorted HERE, not in the client: a required course with no gradeable unit
    # locks its stage and every stage behind it, so it outranks everything else.
    # Draft before published next (a draft is the one still fixable pre-launch),
    # then title for a stable order.
    rows.sort(
        key=lambda r: (
            not r.blocks_required_stage,
            r.status != "draft",
            r.title.casefold(),
        )
    )
    return rows


def _program_reason(*, has_draft: bool, open_requests: int) -> str:
    parts: list[str] = []
    if open_requests > 0:
        parts.append(
            f"{open_requests} pathway-change request"
            f"{'s' if open_requests != 1 else ''} awaiting a decision"
        )
    if has_draft:
        parts.append("Unpublished draft version")
    return "; ".join(parts)


async def get_management_dashboard(
    db: AsyncSession, *, actor: CurrentUser
) -> ManagementDashboard:
    """Compose the manager / faculty-dean decision queue.

    One payload, so the page is one round trip and every tile is derived from
    the same snapshot as the table beneath it.
    """
    scope_kind, organization_id, org_unit_id, faculty_ids = await _resolve_scope(
        db, actor.user_id
    )
    courses = await _courses_in_scope(db, scope_kind, organization_id, faculty_ids)
    blocked = await _blocked_courses(db, courses, organization_id)

    # get_active_permissions honours role-assignment validity windows, so a
    # dean whose assignment has expired stops being treated as a reviewer.
    permissions = {
        perm.code for perm in await access_control_api.get_active_permissions(db, actor.user_id)
    }
    can_review = _REVIEW_PERMISSION in permissions

    programs: list[ProgramAttentionRow] = []
    programs_total = 0
    programs_with_draft = 0
    open_requests_total = 0
    if organization_id is not None:
        from abridgeai.features.learning_programs.api import public as programs_api  # noqa: PLC0415

        program_dtos = await programs_api.list_program_governance_rows(
            db, organization_id=organization_id, actor=actor
        )
        programs_total = len(program_dtos)
        for dto in program_dtos:
            # OPEN statuses only (pending + in_progress). The per-program
            # drill-down returns EVERY status and will legitimately show more
            # rows than this count — by design, not drift.
            open_requests = int(dto.get("path_change_request_count") or 0)
            has_draft = bool(dto.get("has_draft_version"))
            open_requests_total += open_requests
            if has_draft:
                programs_with_draft += 1
            if not has_draft and open_requests == 0:
                continue
            programs.append(
                ProgramAttentionRow(
                    program_id=dto["id"],
                    name=dto["name"],
                    slug=dto["slug"],
                    status=dto["status"],
                    organization_id=dto["organization_id"],
                    faculty_id=dto["faculty_id"],
                    student_count=int(dto.get("student_count") or 0),
                    has_draft_version=has_draft,
                    open_path_change_request_count=open_requests,
                    reason=_program_reason(
                        has_draft=has_draft, open_requests=open_requests
                    ),
                )
            )
        # Someone waiting on a decision outranks a draft nobody is blocked on.
        programs.sort(
            key=lambda p: (
                -p.open_path_change_request_count,
                not p.has_draft_version,
                p.name.casefold(),
            )
        )

    counts = ManagementDashboardCounts(
        courses_total=len(courses),
        courses_draft=sum(1 for c in courses if c.status == "draft"),
        courses_published=sum(1 for c in courses if c.status == "published"),
        courses_blocked=len(blocked),
        programs_total=programs_total,
        programs_with_draft=programs_with_draft,
        # None, not 0, for a caller who cannot review: 0 claims "no work
        # waiting", which for a manager is a different and possibly false
        # statement than "this is not your queue".
        open_path_change_requests=open_requests_total if can_review else None,
    )

    return ManagementDashboard(
        scope_kind=scope_kind,  # type: ignore[arg-type]
        organization_id=organization_id,
        org_unit_id=org_unit_id,
        can_review_path_changes=can_review,
        counts=counts,
        blocked_courses=blocked,
        programs_needing_attention=programs,
    )


__all__ = ["get_management_dashboard"]
