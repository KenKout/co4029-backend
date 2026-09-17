"""Unit tests for ``features.identity.api.public`` (T24).

DB-free throughout. The first half asserts the public surface's shape --
importable, signatures, DTO contract, no-secrets invariant. The second half
covers behaviour, with the session stubbed: the projections and fallbacks
these functions apply are decisions of their own, and the sibling features
that call them inherit whatever they decide.
"""

from __future__ import annotations

import inspect
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import AsyncMock
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


# ---------------------------------------------------------------------------
# Behaviour
#
# The session is a stub. What is under test is what each function does with
# the rows it gets back, which is where a cross-feature caller can be handed
# something other than what it asked for.
# ---------------------------------------------------------------------------


def _row_result(rows: list) -> SimpleNamespace:
    """Stand in for a SQLAlchemy Result over ``rows``."""
    return SimpleNamespace(
        all=lambda: list(rows),
        one_or_none=lambda: rows[0] if rows else None,
        scalar_one_or_none=lambda: rows[0] if rows else None,
        scalar_one=lambda: rows[0],
    )


def _db(*results: SimpleNamespace) -> SimpleNamespace:
    """A session that returns each result in turn."""
    queue = list(results)

    async def _execute(_stmt):
        return queue.pop(0) if queue else _row_result([])

    return SimpleNamespace(execute=_execute, add=lambda _obj: None)


def _user_row(user_id: UUID, *, email: str = "a@test.local", status: str = "active"):
    return SimpleNamespace(
        id=user_id,
        primary_email=email,
        status=status,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class TestFetchingOneUser:
    async def test_a_user_with_a_profile_carries_its_display_name(self) -> None:
        user_id = uuid.uuid4()
        db = _db(_row_result([(_user_row(user_id), "Nguyen Van A")]))

        dto = await public.get_user_by_id(db, user_id)

        assert dto is not None
        assert dto.id == user_id
        assert dto.display_name == "Nguyen Van A"

    async def test_a_user_without_a_profile_still_resolves(self) -> None:
        """The join is an OUTER join for a reason: a freshly invited account
        has no profile row yet, and returning ``None`` for it would make the
        person look deleted to every sibling feature.
        """
        user_id = uuid.uuid4()
        db = _db(_row_result([(_user_row(user_id), None)]))

        dto = await public.get_user_by_id(db, user_id)

        assert dto is not None
        assert dto.display_name is None
        assert dto.primary_email == "a@test.local"

    async def test_an_unknown_user_is_none(self) -> None:
        assert await public.get_user_by_id(_db(_row_result([])), uuid.uuid4()) is None


class TestFetchingManyUsers:
    async def test_an_empty_request_never_reaches_the_database(self) -> None:
        """Callers build the id list unconditionally, so the short-circuit is
        what stops an ``IN ()`` on every page that happens to have no rows.
        """
        touched = False

        async def _execute(_stmt):
            nonlocal touched
            touched = True
            return _row_result([])

        db = SimpleNamespace(execute=_execute)

        assert await public.get_users_by_ids(db, []) == {}
        assert touched is False

    async def test_the_result_is_keyed_by_user_id(self) -> None:
        first, second = uuid.uuid4(), uuid.uuid4()
        db = _db(
            _row_result(
                [
                    (_user_row(first, email="one@test.local"), "One"),
                    (_user_row(second, email="two@test.local"), None),
                ]
            )
        )

        users = await public.get_users_by_ids(db, [first, second])

        assert set(users) == {first, second}
        assert users[first].display_name == "One"
        assert users[second].display_name is None

    async def test_an_id_with_no_row_is_absent_rather_than_none(self) -> None:
        """Documented deliberately: callers already filter by membership
        before asking, so a missing id means the caller's own list was
        stale. Mapping it to ``None`` would push a null into every consumer
        that iterates the dict.
        """
        present, missing = uuid.uuid4(), uuid.uuid4()
        db = _db(_row_result([(_user_row(present), "Present")]))

        users = await public.get_users_by_ids(db, [present, missing])

        assert set(users) == {present}
        assert missing not in users


class TestTheLocaleThatChoosesANotificationsLanguage:
    """Read at creation time by notifications and the SR workers.

    Whatever this returns is the language a student reads their reminder
    in, so every unknown case has to land on something renderable rather
    than on a value the copy tables have no entry for.
    """

    async def test_vietnamese_is_honoured(self) -> None:
        assert await public.get_user_locale(_db(_row_result(["vi"])), uuid.uuid4()) == "vi"

    async def test_english_is_honoured(self) -> None:
        assert await public.get_user_locale(_db(_row_result(["en"])), uuid.uuid4()) == "en"

    @pytest.mark.parametrize("stored", [None, "", "fr", "VI", "vi-VN", "en-GB"])
    async def test_anything_else_falls_back_to_english(self, stored: str | None) -> None:
        """Matching the frontend's i18next ``fallbackLng``.

        The comparison is exact, so a regional tag or a capitalised value
        reads as English rather than as its own language -- pinned as the
        current contract, since a student whose profile says ``vi-VN``
        would today be written to in English.
        """
        assert await public.get_user_locale(_db(_row_result([stored])), uuid.uuid4()) == "en"

    async def test_a_user_with_no_profile_row_falls_back_to_english(self) -> None:
        assert await public.get_user_locale(_db(_row_result([])), uuid.uuid4()) == "en"


class TestCountingActiveSessions:
    async def test_the_count_is_returned_as_an_int(self) -> None:
        """The admin dashboard renders it directly."""
        assert await public.get_active_session_count(_db(_row_result([3])), uuid.uuid4()) == 3

    async def test_a_user_with_no_sessions_counts_zero(self) -> None:
        assert await public.get_active_session_count(_db(_row_result([0])), uuid.uuid4()) == 0


class TestResolvingARosterEmailToAnAccount:
    """Bulk student import, where a roster file mixes people who already
    have accounts with people who do not.

    The rule that matters is what happens to the ones who do.
    """

    async def test_an_existing_account_is_returned_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A spreadsheet is not authority over an account that exists.

        Silently renaming someone, or re-scoping their role, because a
        roster file disagreed is a write nobody asked for -- and the person
        it happens to is the last to find out. So an existing user comes
        back as-is and the creation path is not entered at all.
        """
        from abridgeai.features.identity.queries import users as user_queries
        from abridgeai.features.identity.services import admin as admin_service

        existing_id = uuid.uuid4()
        monkeypatch.setattr(
            user_queries,
            "get_user_by_email",
            AsyncMock(return_value=SimpleNamespace(id=existing_id)),
        )
        create = AsyncMock()
        monkeypatch.setattr(admin_service, "create_user_account", create)

        user_id, created = await public.find_or_create_student(
            object(),
            email="existing@test.local",
            organization_id=uuid.uuid4(),
            actor_id=uuid.uuid4(),
            given_name="Overwritten",
            family_name="ByRoster",
            display_name="Not Their Choice",
        )

        assert user_id == existing_id
        assert created is False
        create.assert_not_awaited()

    async def test_a_new_email_is_invited_as_an_org_scoped_student(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Imported students go through the same path a manual invite uses,
        so they end up indistinguishable from invited ones rather than as a
        second class of account with its own quirks.
        """
        from abridgeai.features.identity.queries import users as user_queries
        from abridgeai.features.identity.services import admin as admin_service

        new_id, org_id, actor_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        monkeypatch.setattr(user_queries, "get_user_by_email", AsyncMock(return_value=None))
        create = AsyncMock(return_value=SimpleNamespace(id=new_id))
        monkeypatch.setattr(admin_service, "create_user_account", create)

        user_id, created = await public.find_or_create_student(
            object(),
            email="new@test.local",
            organization_id=org_id,
            actor_id=actor_id,
            given_name="New",
            family_name="Student",
        )

        assert (user_id, created) == (new_id, True)
        payload = create.await_args.kwargs["payload"]
        assert payload.primary_email == "new@test.local"
        assert payload.organization_id == org_id
        assert payload.role_code == "student", (
            "an import must not be a way to mint an account with any other role"
        )
        assert create.await_args.kwargs["actor_id"] == actor_id

    async def test_the_created_flag_is_what_the_importer_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The import summary tells the manager how many people were newly
        invited versus already present, which is the number they check
        before sending anyone a welcome mail.
        """
        from abridgeai.features.identity.queries import users as user_queries
        from abridgeai.features.identity.services import admin as admin_service

        monkeypatch.setattr(
            user_queries,
            "get_user_by_email",
            AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())),
        )
        monkeypatch.setattr(admin_service, "create_user_account", AsyncMock())

        _, created = await public.find_or_create_student(
            object(),
            email="x@test.local",
            organization_id=uuid.uuid4(),
            actor_id=uuid.uuid4(),
        )
        assert created is False


class TestStagingAStorageObject:
    def test_the_row_is_added_without_exposing_the_model(self) -> None:
        """Sibling features upload files but must not import the identity
        ORM class, so the add happens here and they pass plain values.
        """
        added: list = []
        db = SimpleNamespace(add=added.append)
        object_id, uploader = uuid.uuid4(), uuid.uuid4()
        uploaded_at = datetime(2026, 9, 1, tzinfo=UTC)

        public.add_storage_object(
            db,
            object_id=object_id,
            bucket="materials",
            object_key="lessons/week-1.pdf",
            original_filename="Week 1.pdf",
            mime_type="application/pdf",
            size_bytes=2048,
            uploaded_by=uploader,
            uploaded_at=uploaded_at,
        )

        assert len(added) == 1
        row = added[0]
        assert row.id == object_id
        assert (row.bucket, row.object_key) == ("materials", "lessons/week-1.pdf")
        assert row.uploaded_by == uploader

    def test_it_stages_rather_than_commits(self) -> None:
        """Not a coroutine: it only calls ``session.add``, so the row joins
        whatever transaction the caller is already in and is theirs to
        commit or roll back with the rest of their work.
        """
        assert not inspect.iscoroutinefunction(public.add_storage_object)


class TestResolvingStorageTargets:
    async def test_ids_map_to_their_bucket_and_key(self) -> None:
        first, second = uuid.uuid4(), uuid.uuid4()
        db = _db(
            _row_result(
                [(first, "materials", "a/one.pdf"), (second, "avatars", "b/two.png")]
            )
        )

        targets = await public.get_storage_object_targets(db, [first, second])

        assert targets == {
            first: ("materials", "a/one.pdf"),
            second: ("avatars", "b/two.png"),
        }

    async def test_an_empty_id_list_never_reaches_the_database(self) -> None:
        touched = False

        async def _execute(_stmt):
            nonlocal touched
            touched = True
            return _row_result([])

        assert await public.get_storage_object_targets(
            SimpleNamespace(execute=_execute), []
        ) == {}
        assert touched is False


class TestFetchingAProfile:
    async def test_a_missing_profile_is_none(self) -> None:
        assert await public.get_user_profile(_db(_row_result([])), uuid.uuid4()) is None
