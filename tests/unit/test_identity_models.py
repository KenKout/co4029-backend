from __future__ import annotations

from abridgeai.core.db.mixins import SoftDeleteMixin
from abridgeai.features.identity.models import (
    AuthIdentity,
    AuthSession,
    MfaChallenge,
    MfaFactor,
    MfaRecoveryCode,
    StorageObject,
    User,
    UserProfile,
    UserProfileLink,
)


def test_identity_models_importable() -> None:
    expected_tables = {
        User: "users",
        AuthIdentity: "auth_identities",
        AuthSession: "auth_sessions",
        MfaFactor: "mfa_factors",
        MfaRecoveryCode: "mfa_recovery_codes",
        MfaChallenge: "mfa_challenges",
        UserProfile: "user_profiles",
        UserProfileLink: "user_profile_links",
        StorageObject: "storage_objects",
    }
    for model, tablename in expected_tables.items():
        assert model.__tablename__ == tablename


def test_user_no_password_field() -> None:
    cols = {c.name for c in User.__table__.columns}
    assert "password_hash" not in cols
    assert "password" not in cols


def test_credential_models_no_softdelete_mixin() -> None:
    credential_models = (
        User,
        AuthIdentity,
        AuthSession,
        MfaFactor,
        MfaRecoveryCode,
        MfaChallenge,
    )
    for model in credential_models:
        assert not issubclass(model, SoftDeleteMixin), (
            f"{model.__name__} must not inherit SoftDeleteMixin"
        )
        cols = {c.name for c in model.__table__.columns}
        assert "deleted_at" not in cols, f"{model.__name__} has deleted_at"
        assert "deleted_by" not in cols, f"{model.__name__} has deleted_by"


def test_profile_has_softdelete() -> None:
    cols = {c.name for c in UserProfile.__table__.columns}
    assert "deleted_at" in cols
    assert "deleted_by" in cols
    assert issubclass(UserProfile, SoftDeleteMixin)
