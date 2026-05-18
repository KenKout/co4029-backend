"""Sub-resource permission wrappers for the interviews authoring router (T6.12 / FIX-SEC-1).

Closes the same perimeter gap T3.7 / T4.5 / T5.14 closed for courses /
materials / quizzes: every authoring endpoint walks UP from the path-
param sub-resource id to the owning ``course_id``, then runs the
standard owner-or-grant check that mirrors
:func:`features.access_control.policies.require_course_permission`.

Three factories cover the interview authoring + session path-param
shapes:

* :func:`require_interview_authoring_access` — resolves
  ``config_id → course_id`` via the ``interview_configs.course_id``
  column directly.
* :func:`require_question_authoring_access` — resolves
  ``question_id → interview_configs → courses``; if a sibling
  ``config_id`` path parameter is also present it is cross-checked so
  a request that smuggles a foreign ``config_id`` gets a 404 (no
  information leak).
* :func:`require_session_owner_access` — student-side perimeter:
  asserts ``session.student_id == current_user.user_id`` (also enforced
  by the service layer; this is defence-in-depth at the HTTP boundary).

Why a separate module: keeping these wrappers out of ``authoring.py``
makes the security perimeter independently auditable — the FIX-SEC-1
grep guard asserts every authoring endpoint depends on a wrapper from
this file (or :func:`require_course_permission`), never on a bare
:func:`get_current_user`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.policies import can_manage_course
from abridgeai.features.courses.api import public as courses_public
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewQuestion,
    InterviewSession,
)

_DEFAULT_AUTHORING_PERMS: tuple[str, ...] = ("course.update",)

SubResourceDependency = Callable[..., Awaitable[CurrentUser]]


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


def _permission_denied(*, codes: tuple[str, ...], course_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "permission_denied",
            "required": list(codes),
            "scope": "course",
            "course_id": str(course_id),
        },
    )


def _forbidden_session(session_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "forbidden",
            "resource": "interview_session",
            "id": str(session_id),
        },
    )


async def _check_course_permission(
    db: AsyncSession,
    current_user: CurrentUser,
    course_id: UUID,
    owner_user_id: UUID,
    codes: tuple[str, ...],
) -> CurrentUser:
    """Owner short-circuit + per-code ``can_manage_course`` lookup."""
    if owner_user_id == current_user.user_id:
        return current_user
    for code in codes:
        if await can_manage_course(db, current_user.user_id, course_id, manage_perm=code):
            return current_user
    raise _permission_denied(codes=codes, course_id=course_id)


def require_interview_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``config_id → course_id`` and enforces course perms.

    The path parameter MUST be named ``config_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        config_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        config = (
            await db.execute(select(InterviewConfig).where(InterviewConfig.id == config_id))
        ).scalar_one_or_none()
        if config is None:
            raise _not_found("interview_config", config_id)
        course = await courses_public.get_course_by_id(db, config.course_id)
        if course is None:
            raise _not_found("interview_config", config_id)
        return await _check_course_permission(
            db, current_user, course.id, course.owner_user_id, codes
        )

    return dependency


def require_question_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``question_id → interview_config → course`` and enforces course perms.

    The path parameter MUST be named ``question_id``. If a sibling
    ``config_id`` path parameter is also present it is cross-checked
    against the question's parent — a request smuggling a foreign
    ``config_id`` is rejected with 404 to avoid leaking existence.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        request: Request,
        question_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        question = (
            await db.execute(select(InterviewQuestion).where(InterviewQuestion.id == question_id))
        ).scalar_one_or_none()
        if question is None:
            raise _not_found("interview_question", question_id)
        config = (
            await db.execute(
                select(InterviewConfig).where(InterviewConfig.id == question.interview_config_id)
            )
        ).scalar_one_or_none()
        if config is None:
            raise _not_found("interview_question", question_id)
        path_config = request.path_params.get("config_id")
        if path_config is not None and str(path_config) != str(config.id):
            raise _not_found("interview_question", question_id)
        course = await courses_public.get_course_by_id(db, config.course_id)
        if course is None:
            raise _not_found("interview_question", question_id)
        return await _check_course_permission(
            db, current_user, course.id, course.owner_user_id, codes
        )

    return dependency


def require_session_owner_access() -> SubResourceDependency:
    """Walks ``session_id → student_id`` and enforces ownership.

    The path parameter MUST be named ``session_id``. Returns the
    calling user when ``session.student_id == current_user.user_id``
    and 404 / 403 otherwise. The 404 path is reserved for missing
    sessions (existence not leaked); 403 fires for cross-user reads.

    Defence-in-depth: the ``services.taking._assert_owns_session``
    helper raises :class:`ForbiddenError` for the same condition.
    Here we close the perimeter at the HTTP boundary so endpoints
    that don't go through ``services.taking`` (e.g. simple ``GET``
    reads) still get the check.
    """

    async def dependency(
        session_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
        session = result.scalar_one_or_none()
        if session is None:
            raise _not_found("interview_session", session_id)
        if session.student_id != current_user.user_id:
            raise _forbidden_session(session_id)
        return current_user

    return dependency


def require_session_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``session_id → interview_config → course`` and enforces course perms.

    The path parameter MUST be named ``session_id``. Used by the
    teacher-side gap-report endpoint so the FIX-SEC-1 grep guard does
    not fire on a bare ``Depends(get_current_user)`` for an
    authoring-tagged route.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        session_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        session = (
            await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
        ).scalar_one_or_none()
        if session is None:
            raise _not_found("interview_session", session_id)
        config = (
            await db.execute(
                select(InterviewConfig).where(InterviewConfig.id == session.interview_config_id)
            )
        ).scalar_one_or_none()
        if config is None:
            raise _not_found("interview_session", session_id)
        course = await courses_public.get_course_by_id(db, config.course_id)
        if course is None:
            raise _not_found("interview_session", session_id)
        return await _check_course_permission(
            db, current_user, course.id, course.owner_user_id, codes
        )

    return dependency


__all__ = [
    "SubResourceDependency",
    "require_interview_authoring_access",
    "require_question_authoring_access",
    "require_session_authoring_access",
    "require_session_owner_access",
]
