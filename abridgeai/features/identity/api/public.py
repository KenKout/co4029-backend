"""Typed cross-feature read API for the identity feature.

Sibling features (admin, courses, enrollments, materials, sr) import
from this module instead of touching identity-owned tables directly.
Reads return Pydantic DTOs (the immutable contract); ORM models and
all credential-bearing rows (``AuthIdentity``, ``AuthSession``,
``MfaFactor``, ``MfaRecoveryCode``, ``MfaChallenge``) stay private.

Soft-delete: ``UserProfile`` carries ``SoftDeleteMixin`` and inherits
the global ``with_loader_criteria`` filter automatically (see
``core/db/soft_delete.py``). ``User`` is intentionally HARD-delete
(plan §A13: credential-adjacent rows do not get soft-delete columns),
so soft-deletion is checked here against the joined profile row when
present, and fall-through ``status='inactive'`` semantics are the
service layer's concern.

Security contract: the DTOs returned here MUST NEVER include
``password_hash`` (the project is OAuth-only, no such column today),
``secret_encrypted``/``mfa_secret``, ``refresh_token_hash``, or raw
``auth_identities``/``mfa_factors`` rows. ``tests/unit/identity/
test_api_public.py::test_dto_excludes_secrets`` enforces this.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.identity.models import AuthSession, User, UserProfile

from ._dto import UserDTO, UserProfileDTO


def _user_to_dto(user: User, display_name: str | None) -> UserDTO:
    return UserDTO.model_validate(
        {
            "id": user.id,
            "primary_email": user.primary_email,
            "display_name": display_name,
            "status": user.status,
            "created_at": user.created_at,
        }
    )


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> UserDTO | None:
    stmt = (
        select(User, UserProfile.display_name)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .where(User.id == user_id)
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    user, display_name = row
    return _user_to_dto(user, display_name)


async def get_users_by_ids(db: AsyncSession, user_ids: Sequence[UUID]) -> dict[UUID, UserDTO]:
    """Fetch many users in one round-trip; missing ids are absent from the dict.

    Empty input short-circuits to an empty dict so callers can pass an
    unconditionally-built id list. Callers receive only rows that exist
    — non-matching ids are NOT mapped to ``None`` (admin pages already
    filter by membership before calling).
    """
    if not user_ids:
        return {}
    stmt = (
        select(User, UserProfile.display_name)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .where(User.id.in_(user_ids))
    )
    rows = (await db.execute(stmt)).all()
    return {user.id: _user_to_dto(user, display_name) for user, display_name in rows}


async def get_user_profile(db: AsyncSession, user_id: UUID) -> UserProfileDTO | None:
    stmt = select(UserProfile).where(UserProfile.user_id == user_id)
    profile = (await db.execute(stmt)).scalar_one_or_none()
    return UserProfileDTO.model_validate(profile) if profile else None


async def get_user_locale(db: AsyncSession, user_id: UUID) -> str:
    """Return the user's normalized preferred locale ('en' | 'vi').

    Reads ``user_profiles.locale``. Missing profile, NULL, or any
    unrecognized value falls back to ``'en'`` -- matching the frontend
    i18next ``fallbackLng``. Sibling features (notifications dispatch,
    spaced-repetition workers) call this to render notification copy in
    the recipient's language at creation time.
    """
    stmt = select(UserProfile.locale).where(UserProfile.user_id == user_id)
    value = (await db.execute(stmt)).scalar_one_or_none()
    if value == "vi":
        return "vi"
    return "en"


async def get_active_session_count(db: AsyncSession, user_id: UUID) -> int:
    """Count non-revoked sessions for a user (admin dashboard signal).

    "Active" here mirrors ``queries/sessions.py``: ``revoked_at IS NULL``
    only. Expiry policy (``expires_at`` comparison, grace windows) is the
    caller's decision and lives in the service layer.
    """
    stmt = (
        select(func.count(AuthSession.id))
        .where(AuthSession.user_id == user_id)
        .where(AuthSession.revoked_at.is_(None))
    )
    return int((await db.execute(stmt)).scalar_one())


async def find_or_create_student(
    db: AsyncSession,
    *,
    email: str,
    organization_id: UUID,
    actor_id: UUID,
    given_name: str | None = None,
    family_name: str | None = None,
    display_name: str | None = None,
) -> tuple[UUID, bool]:
    """Resolve ``email`` to a user id, creating the account if it is new.

    Returns ``(user_id, created)``. Backs bulk student import, where a roster
    file mixes people who already have accounts with people who do not, and
    the importer must not care which is which.

    An EXISTING user is returned untouched — no profile overwrite, no role
    change. A roster file is not authority over an account that already
    exists: silently renaming someone or re-scoping their role because a
    spreadsheet disagreed is the kind of write nobody asks for.

    A NEW user is created through the same admin-invite path a manual invite
    uses (active account + profile + org-scoped ``student`` role), so imported
    students are indistinguishable from invited ones.
    """
    from abridgeai.features.identity.queries import users as user_queries  # noqa: PLC0415
    from abridgeai.features.identity.schemas import UserCreate  # noqa: PLC0415
    from abridgeai.features.identity.services import admin as admin_service  # noqa: PLC0415

    existing = await user_queries.get_user_by_email(db, email)
    if existing is not None:
        return UUID(str(existing.id)), False

    created = await admin_service.create_user_account(
        db,
        payload=UserCreate(
            primary_email=email,
            given_name=given_name,
            family_name=family_name,
            display_name=display_name,
            organization_id=organization_id,
            role_code="student",
        ),
        actor_id=actor_id,
    )
    return UUID(str(created.id)), True


__all__ = [
    "find_or_create_student",
    "UserDTO",
    "UserProfileDTO",
    "get_active_session_count",
    "get_user_by_id",
    "get_user_locale",
    "get_user_profile",
    "get_users_by_ids",
]
