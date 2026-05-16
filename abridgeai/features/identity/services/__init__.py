"""Identity feature service layer.

Split by concern (login / session / mfa / profile) per the "Per-Feature
Module Layout" draft. Services delegate ALL DB access to
``abridgeai.features.identity.queries.*`` — they never import
``sqlalchemy`` at runtime, enforced by import-linter contract #1
("Services do not touch SQLAlchemy directly"). Type annotations that
need ``AsyncSession`` go through a ``TYPE_CHECKING`` guard.
"""

from __future__ import annotations

from . import login, mfa, profile, session
from .login import handle_google_callback
from .mfa import (
    create_mfa_challenge,
    enroll_totp,
    regenerate_recovery_codes,
    verify_mfa_challenge,
    verify_totp_enrollment,
)
from .profile import get_current_user_read, serialize_user, update_profile
from .session import logout, refresh_tokens

__all__ = [
    "create_mfa_challenge",
    "enroll_totp",
    "get_current_user_read",
    "handle_google_callback",
    "login",
    "logout",
    "mfa",
    "profile",
    "refresh_tokens",
    "regenerate_recovery_codes",
    "serialize_user",
    "session",
    "update_profile",
    "verify_mfa_challenge",
    "verify_totp_enrollment",
]
