"""Sub-resource permission wrappers for the quizzes authoring router (T5.14 / FIX-SEC-1).

Closes the same perimeter gap T3.7 / T4.5 closed for courses / materials:
every authoring endpoint walks UP from the path-param sub-resource id to
the owning ``course_id``, then runs the standard owner-or-grant check
that mirrors :func:`features.access_control.policies.require_course_permission`.

Two factories cover the quiz authoring path-param shapes:

* :func:`require_quiz_authoring_access` — resolves ``quiz_id → course_id``
  via the ``quizzes.course_id`` column directly.
* :func:`require_question_authoring_access` — resolves
  ``question_id → quiz → course`` (joins quiz_questions → quizzes); a
  question's URL also carries ``quiz_id``, which we cross-check so a
  request that smuggles a foreign ``quiz_id`` gets a 404 (no information
  leak).

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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.policies import can_manage_course

_DEFAULT_AUTHORING_PERMS: tuple[str, ...] = ("course.update",)

SubResourceDependency = Callable[..., Awaitable[CurrentUser]]


_QUIZ_TO_COURSE_SQL = text(
    """
    SELECT q.course_id     AS course_id,
           c.owner_user_id AS owner_user_id
    FROM quizzes q
    JOIN courses c ON c.id = q.course_id
    WHERE q.id = :quiz_id
      AND q.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)

_QUESTION_TO_COURSE_SQL = text(
    """
    SELECT q.course_id     AS course_id,
           q.id            AS quiz_id,
           c.owner_user_id AS owner_user_id
    FROM quiz_questions qq
    JOIN quizzes q  ON q.id = qq.quiz_id
    JOIN courses c  ON c.id = q.course_id
    WHERE qq.id = :question_id
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
      AND c.deleted_at IS NULL
    """
)


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


def require_quiz_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``quiz_id → course_id`` and enforces course perms.

    The path parameter MUST be named ``quiz_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        quiz_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(_QUIZ_TO_COURSE_SQL, {"quiz_id": quiz_id})
        row = result.mappings().one_or_none()
        if row is None:
            raise _not_found("quiz", quiz_id)
        return await _check_course_permission(
            db, current_user, row["course_id"], row["owner_user_id"], codes
        )

    return dependency


def require_question_authoring_access(
    *perm_codes: str,
) -> SubResourceDependency:
    """Walks ``question_id → quiz → course`` and enforces course perms.

    The path parameter MUST be named ``question_id``. If a sibling
    ``quiz_id`` path parameter is also present it is cross-checked
    against the question's parent — a request smuggling a foreign
    ``quiz_id`` is rejected with 404 to avoid leaking existence.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERMS

    async def dependency(
        request: Request,
        question_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        result = await db.execute(_QUESTION_TO_COURSE_SQL, {"question_id": question_id})
        row = result.mappings().one_or_none()
        if row is None:
            raise _not_found("quiz_question", question_id)
        path_quiz = request.path_params.get("quiz_id")
        if path_quiz is not None and str(path_quiz) != str(row["quiz_id"]):
            raise _not_found("quiz_question", question_id)
        return await _check_course_permission(
            db, current_user, row["course_id"], row["owner_user_id"], codes
        )

    return dependency


__all__ = [
    "SubResourceDependency",
    "require_question_authoring_access",
    "require_quiz_authoring_access",
]
