from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.identity.models import (
    AuthSession,
    MfaFactor,
    User,
    UserProfile,
)
from abridgeai.features.identity.schemas import (
    MfaTotpVerifyRequest,
    MfaVerifyRequest,
)
from abridgeai.features.identity.services import login, mfa, profile, session
from abridgeai.infrastructure.google_oauth import GoogleProfile


def _make_user(*, status: str = "active") -> User:
    user = User(primary_email="alice@example.com", status=status)
    user.id = uuid4()
    user.last_login_at = datetime.now(UTC)
    user.created_at = datetime.now(UTC)
    user.updated_at = datetime.now(UTC)
    return user


def _make_session(user_id, *, revoked: bool = False, expired: bool = False) -> AuthSession:
    s = AuthSession(
        user_id=user_id,
        refresh_token_hash="sha256:dummy",  # noqa: S106
        expires_at=datetime.now(UTC) - timedelta(seconds=1)
        if expired
        else datetime.now(UTC) + timedelta(days=1),
    )
    s.id = uuid4()
    s.revoked_at = datetime.now(UTC) if revoked else None
    s.mfa_verified_at = None
    s.created_at = datetime.now(UTC)
    s.updated_at = datetime.now(UTC)
    return s


def _make_db() -> MagicMock:
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    db.get = AsyncMock()
    return db


def test_serialize_user_pure_function() -> None:
    user = _make_user()
    prof = UserProfile(user_id=user.id, display_name="Alice")
    prof.given_name = "Alice"
    prof.family_name = None
    prof.avatar_object_id = None
    prof.bio = None

    result = profile.serialize_user(user, prof)

    assert result.id == user.id
    assert result.primary_email == "alice@example.com"
    assert result.status == "active"
    assert result.profile is not None
    assert result.profile.display_name == "Alice"


def test_serialize_user_without_profile() -> None:
    user = _make_user()
    result = profile.serialize_user(user, None)
    assert result.profile is None
    assert result.primary_email == "alice@example.com"


@pytest.mark.asyncio
async def test_upload_avatar_rejects_unsupported_type() -> None:
    db = _make_db()
    user = _make_user()
    with pytest.raises(profile.AvatarUploadError, match="unsupported_avatar_type"):
        await profile.upload_avatar(
            db, user, data=b"\x00\x01\x02", content_type="application/pdf"
        )
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_avatar_rejects_empty_file() -> None:
    db = _make_db()
    user = _make_user()
    with pytest.raises(profile.AvatarUploadError, match="empty_avatar"):
        await profile.upload_avatar(db, user, data=b"", content_type="image/png")
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_avatar_rejects_oversized_file() -> None:
    db = _make_db()
    user = _make_user()
    too_big = b"\x00" * (2 * 1024 * 1024 + 1)
    with pytest.raises(profile.AvatarUploadError, match="avatar_too_large"):
        await profile.upload_avatar(db, user, data=too_big, content_type="image/png")
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_login_handle_callback_rejects_unprovisioned_email() -> None:
    """Invite-only OAuth: brand-new emails get HTTP 403 (ForbiddenError)."""
    from abridgeai.core.exceptions import ForbiddenError

    db = _make_db()

    google_profile = GoogleProfile(
        subject="google-sub-123",
        email="newuser@example.com",
        given_name="New",
        family_name="User",
        display_name="New User",
    )

    with (
        patch(
            "abridgeai.features.identity.services.login.fetch_google_profile",
            new=AsyncMock(return_value=google_profile),
        ),
        patch(
            "abridgeai.features.identity.services.login.user_queries.get_identity_by_provider_subject",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "abridgeai.features.identity.services.login.user_queries.get_user_by_email",
            new=AsyncMock(return_value=None),
        ),
        pytest.raises(ForbiddenError, match="not registered"),
    ):
        await login.handle_google_callback(db, code="oauth-code")


@pytest.mark.asyncio
async def test_login_handle_callback_links_identity_for_preprovisioned_user() -> None:
    """Admin-created user without identity yet: link AuthIdentity, issue tokens."""
    db = _make_db()

    google_profile = GoogleProfile(
        subject="google-sub-123",
        email="alice@example.com",
        given_name="Alice",
        family_name="Doe",
        display_name="Alice Doe",
    )
    existing_user = _make_user()

    captured: list[object] = []

    def fake_add(obj: object) -> None:
        captured.append(obj)
        if not getattr(obj, "id", None):
            obj.id = uuid4()
        if not getattr(obj, "created_at", None):
            obj.created_at = datetime.now(UTC)
            obj.updated_at = datetime.now(UTC)

    db.add.side_effect = fake_add

    with (
        patch(
            "abridgeai.features.identity.services.login.fetch_google_profile",
            new=AsyncMock(return_value=google_profile),
        ),
        patch(
            "abridgeai.features.identity.services.login.user_queries.get_identity_by_provider_subject",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "abridgeai.features.identity.services.login.user_queries.get_user_by_email",
            new=AsyncMock(return_value=existing_user),
        ),
        patch(
            "abridgeai.features.identity.services.login.user_queries.get_profile",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "abridgeai.features.identity.services.login.session_queries.user_has_verified_mfa",
            new=AsyncMock(return_value=False),
        ),
    ):
        result = await login.handle_google_callback(db, code="oauth-code")

    assert result.access_token
    assert result.refresh_token
    assert result.user.primary_email == "alice@example.com"

    added_types = {type(o).__name__ for o in captured}
    assert "User" not in added_types
    assert "UserProfile" in added_types
    assert "AuthIdentity" in added_types
    assert "AuthSession" in added_types


@pytest.mark.asyncio
async def test_session_refresh_invalid_token_raises_unauthorized() -> None:
    from abridgeai.core.exceptions import UnauthorizedError

    db = _make_db()
    with (
        patch(
            "abridgeai.features.identity.services.session.session_queries.get_session_by_refresh_hash",
            new=AsyncMock(return_value=None),
        ),
        pytest.raises(UnauthorizedError),
    ):
        await session.refresh_tokens(db, "any-bad-token")


@pytest.mark.asyncio
async def test_session_logout_marks_session_revoked() -> None:
    db = _make_db()
    user = _make_user()
    auth_session = _make_session(user.id)

    with patch(
        "abridgeai.features.identity.services.session.session_queries.get_session_by_refresh_hash",
        new=AsyncMock(return_value=auth_session),
    ):
        await session.logout(db, refresh_token="some-refresh-token")  # noqa: S106

    assert auth_session.revoked_at is not None
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_mfa_enroll_totp_creates_factor_with_encrypted_secret() -> None:
    db = _make_db()
    user = _make_user()

    captured: list[MfaFactor] = []

    def fake_add(obj: object) -> None:
        if isinstance(obj, MfaFactor):
            obj.id = uuid4()
            obj.created_at = datetime.now(UTC)
            obj.updated_at = datetime.now(UTC)
            captured.append(obj)

    db.add.side_effect = fake_add

    result = await mfa.enroll_totp(db, user)

    assert len(captured) == 1
    factor = captured[0]
    assert factor.user_id == user.id
    assert factor.factor_type == "totp"
    assert factor.secret_encrypted
    assert factor.secret_encrypted != result.secret
    assert "otpauth://totp/" in result.otpauth_url
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_mfa_verify_challenge_with_invalid_challenge_raises() -> None:
    from abridgeai.core.exceptions import UnauthorizedError

    db = _make_db()
    user = _make_user()
    auth_session = _make_session(user.id)
    payload = MfaVerifyRequest(challenge_id=uuid4(), code="123456")

    with (
        patch(
            "abridgeai.features.identity.services.mfa.mfa_queries.get_challenge_by_id",
            new=AsyncMock(return_value=None),
        ),
        pytest.raises(UnauthorizedError),
    ):
        await mfa.verify_mfa_challenge(db, user, auth_session, payload)


@pytest.mark.asyncio
async def test_mfa_verify_totp_enrollment_unknown_factor_raises() -> None:
    from abridgeai.core.exceptions import NotFoundError

    db = _make_db()
    user = _make_user()
    auth_session = _make_session(user.id)
    payload = MfaTotpVerifyRequest(factor_id=uuid4(), code="123456")

    with (
        patch(
            "abridgeai.features.identity.services.mfa.mfa_queries.get_factor_by_id",
            new=AsyncMock(return_value=None),
        ),
        pytest.raises(NotFoundError),
    ):
        await mfa.verify_totp_enrollment(db, user, auth_session, payload)


def test_services_have_no_sqlalchemy_imports() -> None:
    services_dir = (
        Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "identity" / "services"
    )
    offenders: list[tuple[str, int, str]] = []
    for py_file in services_dir.glob("*.py"):
        for lineno, raw in enumerate(py_file.read_text().splitlines(), start=1):
            stripped = raw.lstrip()
            if stripped.startswith("from sqlalchemy") or stripped.startswith("import sqlalchemy"):
                if raw.startswith(" ") or raw.startswith("\t"):
                    continue
                offenders.append((py_file.name, lineno, raw))
    assert not offenders, f"Service modules must not import sqlalchemy: {offenders}"


def test_services_have_no_cross_feature_imports() -> None:
    services_dir = (
        Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "identity" / "services"
    )
    forbidden_prefixes = (
        "from abridgeai.features.access_control",
        "from abridgeai.features.courses",
        "from abridgeai.features.assessments",
        "from app.routes.users",
    )
    offenders: list[tuple[str, int, str]] = []
    for py_file in services_dir.glob("*.py"):
        for lineno, raw in enumerate(py_file.read_text().splitlines(), start=1):
            stripped = raw.lstrip()
            if any(stripped.startswith(p) for p in forbidden_prefixes):
                offenders.append((py_file.name, lineno, raw))
    assert not offenders, f"Service modules must stay feature-local: {offenders}"
