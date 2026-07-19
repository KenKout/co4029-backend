"""Unit tests for ``features.identity.api.public`` (T24).

DB-free: asserts public surface shape (importable, signatures, DTO
contract, no-secrets invariant). Behavior over real rows is covered by
integration tests in Wave 5 when consumers migrate onto this module.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints
from uuid import UUID

import pytest

from abridgeai.features.identity.api import public
from abridgeai.features.identity.api._dto import UserDTO, UserProfileDTO

_EXPECTED_FUNCTIONS = (
    "get_user_by_id",
    "get_users_by_ids",
    "get_user_profile",
    "get_active_session_count",
)

_EXPECTED_DTO_NAMES = ("UserDTO", "UserProfileDTO")

_FORBIDDEN_FIELDS = frozenset(
    {
        "password_hash",
        "password",
        "mfa_secret",
        "secret_encrypted",
        "refresh_token_hash",
        "auth_identities",
        "mfa_factors",
        "mfa_recovery_codes",
    }
)


def test_all_expected_symbols_in_dunder_all() -> None:
    exported = set(public.__all__)
    for name in _EXPECTED_FUNCTIONS:
        assert name in exported, f"{name} missing from __all__"
    for name in _EXPECTED_DTO_NAMES:
        assert name in exported, f"{name} missing from __all__"


@pytest.mark.parametrize("name", _EXPECTED_FUNCTIONS)
def test_function_is_async_coroutine(name: str) -> None:
    fn = getattr(public, name)
    assert inspect.iscoroutinefunction(fn), f"{name} must be `async def`"


@pytest.mark.parametrize("name", _EXPECTED_FUNCTIONS)
def test_first_positional_param_is_db(name: str) -> None:
    fn = getattr(public, name)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    assert params, f"{name} has no parameters"
    assert params[0].name == "db", (
        f"{name} first positional must be `db: AsyncSession`, got {params[0].name!r}"
    )


def test_get_user_by_id_signature() -> None:
    sig = inspect.signature(public.get_user_by_id)
    assert list(sig.parameters) == ["db", "user_id"]
    hints = get_type_hints(public.get_user_by_id)
    assert hints["user_id"] is UUID
    assert hints["return"] == (UserDTO | None)


def test_get_users_by_ids_signature() -> None:
    sig = inspect.signature(public.get_users_by_ids)
    assert list(sig.parameters) == ["db", "user_ids"]
    hints = get_type_hints(public.get_users_by_ids)
    assert hints["return"] == dict[UUID, UserDTO]


def test_get_user_profile_signature() -> None:
    hints = get_type_hints(public.get_user_profile)
    assert hints["user_id"] is UUID
    assert hints["return"] == (UserProfileDTO | None)


def test_get_active_session_count_returns_int() -> None:
    hints = get_type_hints(public.get_active_session_count)
    assert hints["user_id"] is UUID
    assert hints["return"] is int


def test_dtos_are_frozen_and_from_attributes() -> None:
    for dto in (UserDTO, UserProfileDTO):
        cfg = dto.model_config
        assert cfg.get("frozen") is True, f"{dto.__name__} must be frozen"
        assert cfg.get("from_attributes") is True, (
            f"{dto.__name__} must allow ORM attribute hydration"
        )


def test_dto_excludes_secrets() -> None:
    for dto in (UserDTO, UserProfileDTO):
        leaked = _FORBIDDEN_FIELDS & set(dto.model_fields)
        assert not leaked, f"{dto.__name__} leaks secret fields: {sorted(leaked)}"


def test_user_dto_has_only_public_fields() -> None:
    fields = set(UserDTO.model_fields)
    assert fields == {"id", "primary_email", "display_name", "status", "created_at"}


def test_user_profile_dto_has_only_public_fields() -> None:
    fields = set(UserProfileDTO.model_fields)
    assert fields == {
        "user_id",
        "display_name",
        "given_name",
        "family_name",
        "avatar_object_id",
        "bio",
        "locale",
    }


def test_dtos_drop_extra_fields() -> None:
    payload = {
        "id": UUID("00000000-0000-0000-0000-000000000001"),
        "primary_email": "u@example.com",
        "display_name": "U",
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z",
        "password_hash": "should-be-stripped",
        "mfa_secret": "should-be-stripped",
    }
    dto = UserDTO.model_validate(payload)
    assert not hasattr(dto, "password_hash")
    assert not hasattr(dto, "mfa_secret")


def test_dtos_are_immutable() -> None:
    dto = UserDTO(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        primary_email="u@example.com",
        display_name="U",
        status="active",
        created_at="2024-01-01T00:00:00Z",  # type: ignore[arg-type]
    )
    with pytest.raises((TypeError, ValueError)):
        dto.status = "suspended"  # type: ignore[misc]


def test_module_docstring_documents_security_contract() -> None:
    doc = (public.__doc__ or "").lower()
    assert "security" in doc
    assert "cross-feature" in doc


def test_dto_module_docstring_documents_security_contract() -> None:
    from abridgeai.features.identity.api import _dto

    doc = (_dto.__doc__ or "").lower()
    assert "password_hash" in doc
    assert "mfa_secret" in doc or "secret_encrypted" in doc
