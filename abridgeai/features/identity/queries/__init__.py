"""Identity feature queries (repository layer).

This package is the SQLAlchemy boundary for the ``identity`` feature: every
function takes ``db: AsyncSession`` and returns ORM models. Service code
(``features.identity.services``) calls these accessors; routers never do.

Per the locked "no mechanism split" decision (plan §13 + Reconciliation),
files are flat-by-resource (``users``, ``sessions``, ``mfa``) — no
``orm/`` / ``raw/`` subdirectory.
"""

from abridgeai.features.identity.queries.mfa import (
    get_active_challenge,
    get_verified_totp_factor,
)
from abridgeai.features.identity.queries.sessions import (
    get_session_by_refresh_hash,
    user_has_verified_mfa,
)
from abridgeai.features.identity.queries.users import (
    get_profile,
    get_profile_link,
    get_user,
    get_user_by_email,
    list_profile_links,
    list_profiles,
    list_users,
)

__all__ = [
    "get_active_challenge",
    "get_profile",
    "get_profile_link",
    "get_session_by_refresh_hash",
    "get_user",
    "get_user_by_email",
    "get_verified_totp_factor",
    "list_profile_links",
    "list_profiles",
    "list_users",
    "user_has_verified_mfa",
]
