"""Sub-resource permission wrappers for the quizzes authoring router (T5.14 / FIX-SEC-1).

Closes the same perimeter gap T3.7 / T4.5 closed for courses / materials:
every authoring endpoint walks UP from the path-param sub-resource id to
the owning ``course_id``, then runs the standard owner-or-grant check
that mirrors :func:`features.access_control.policies.require_course_permission`.

Two factories cover the quiz authoring path-param shapes:

* :func:`require_quiz_authoring_access` -- resolves ``quiz_id -> course_id``
  via the ``quizzes.course_id`` column directly.
* :func:`require_question_authoring_access` -- resolves
  ``question_id -> quiz -> course`` (quiz_questions ORM, then quiz ORM); a
  question's URL also carries ``quiz_id``, which we cross-check so a
  request that smuggles a foreign ``quiz_id`` gets a 404 (no information
  leak).

Why a separate module: keeping these wrappers out of ``authoring.py``
makes the security perimeter independently auditable -- the FIX-SEC-1
grep guard asserts every authoring endpoint depends on a wrapper from
this file (or :func:`require_course_permission`), never on a bare
:func:`get_current_user`.

Cross-feature reads route through :mod:`features.courses.api.public` so
the ``Features are independent`` import-linter contract holds without
per-edge ``ignore_imports`` entries (T35).
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
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.quizzes.models import Quiz, QuizQuestion

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


async def _resolve_quiz_course(db: AsyncSession, quiz_id: UUID) -> tuple[UUID, UUID] | None:
    """Return ``(course_id, owner_user_id)`` for a non-deleted quiz.

    The quizzes feature owns ``Quiz``, so the ORM read is feature-local;
    the courses-side ``owner_user_id`` is read through
    :mod:`courses.api.public` so this module never imports the courses
    ORM. Soft-deletion on ``Quiz`` is auto-filtered by the T0.7 listener.
    """
    quiz = (await db.execute(select(Quiz.course_id).where(Quiz.id == quiz_id))).scalar_one_or_none()
    if quiz is None:
        return None
    course = await courses_api.get_course_by_id(db, quiz)
    if course is None:
        return None
    return course.id, course.owner_user_id


def require_quiz_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``quiz_id -> course_id`` and enforces course perms.

    The path parameter MUST be named ``quiz_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        quiz_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await _resolve_quiz_course(db, quiz_id)
        if resolved is None:
            raise _not_found("quiz", quiz_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


def require_question_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``question_id -> quiz -> course`` and enforces course perms.

    The path parameter MUST be named ``question_id``. If a sibling
    ``quiz_id`` path parameter is also present it is cross-checked
    against the question's parent -- a request smuggling a foreign
    ``quiz_id`` is rejected with 404 to avoid leaking existence.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        request: Request,
        question_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        question_quiz = (
            await db.execute(select(QuizQuestion.quiz_id).where(QuizQuestion.id == question_id))
        ).scalar_one_or_none()
        if question_quiz is None:
            raise _not_found("quiz_question", question_id)
        path_quiz = request.path_params.get("quiz_id")
        if path_quiz is not None and str(path_quiz) != str(question_quiz):
            raise _not_found("quiz_question", question_id)
        resolved = await _resolve_quiz_course(db, question_quiz)
        if resolved is None:
            raise _not_found("quiz_question", question_id)
        course_id, owner_user_id = resolved
        return await _check_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


__all__ = [
    "SubResourceDependency",
    "require_question_authoring_access",
    "require_quiz_authoring_access",
]
