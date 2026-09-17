"""The self-service ``/users/me`` surface.

Every endpoint here is self-scoped, so there is no permission check to
test -- a person may always read and edit their own profile. What is left
is a small set of decisions that each fix a specific way the SPA would
otherwise break.

**One endpoint sits in front of the MFA gate.** ``GET /users/me`` uses the
pre-MFA dependency so the login page can show the person their own name and
avatar while asking for the second factor. Everything else stays behind the
full gate, and that asymmetry is deliberate rather than an oversight -- so
there is a source-level test that it stays exactly one endpoint wide.

**The avatar arrives as a raw body.** No multipart wrapper, so the MIME
type comes from the ``Content-Type`` header, which browsers freely decorate
with parameters. Without the strip, ``image/png; charset=binary`` is not
``image/png`` and a perfectly good upload is refused.

**A foreign link id answers 404, not 403.** The lookup is scoped by owner,
so a link that belongs to someone else is never confirmed to exist.

The profile service is mocked; it has its own suite.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from abridgeai.features.identity.routers import me as me_router
from abridgeai.features.identity.services.profile import (
    AvatarUploadError,
    ProfileLinkError,
    ProfileLinkNotFoundError,
)


def _actor() -> SimpleNamespace:
    return SimpleNamespace(user_id=uuid4())


def _db(user: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(get=AsyncMock(return_value=user if user is not None else object()))


def _request(body: bytes = b"", content_type: str | None = None) -> SimpleNamespace:
    headers = {} if content_type is None else {"content-type": content_type}
    return SimpleNamespace(body=AsyncMock(return_value=body), headers=headers)


class TestLoadingTheCaller:
    async def test_a_token_whose_user_is_gone_is_unauthorized(self) -> None:
        """The account was deleted while a valid token was still in the wild.

        401 rather than 404: the caller is not missing a resource, their
        credential no longer identifies anyone, and the header tells the SPA
        to re-authenticate rather than showing an error page.
        """
        db = SimpleNamespace(get=AsyncMock(return_value=None))

        with pytest.raises(HTTPException) as raised:
            await me_router._load_user(db, _actor())

        assert raised.value.status_code == 401
        assert raised.value.headers == {"WWW-Authenticate": "Bearer"}

    async def test_the_caller_is_looked_up_by_their_own_token(self) -> None:
        """Nothing here takes a user id from the path or the body, which is
        what makes the whole router self-scoped by construction."""
        user = object()
        db = SimpleNamespace(get=AsyncMock(return_value=user))
        actor = _actor()

        assert await me_router._load_user(db, actor) is user
        assert db.get.await_args.args[1] == actor.user_id


class TestTheAvatarUpload:
    @pytest.fixture
    def upload(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        stub = AsyncMock(return_value="user-read")
        monkeypatch.setattr(me_router.profile_service, "upload_avatar", stub)
        return stub

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("image/png", "image/png"),
            ("image/png; charset=binary", "image/png"),
            ("IMAGE/PNG", "image/png"),
            ("  image/webp  ", "image/webp"),
            ("image/jpeg;q=0.9", "image/jpeg"),
        ],
    )
    async def test_the_content_type_is_normalised(
        self, upload: AsyncMock, header: str, expected: str
    ) -> None:
        """Browsers decorate this header freely.

        Passed through raw, ``image/png; charset=binary`` fails the type
        check and the person is told their PNG is an unsupported format.
        """
        await me_router.upload_my_avatar(_request(b"bytes", header), _actor(), _db())

        assert upload.await_args.kwargs["content_type"] == expected

    async def test_a_request_with_no_content_type_gets_a_refusable_default(
        self, upload: AsyncMock
    ) -> None:
        """``application/octet-stream`` is not on the allow-list, so a body
        with no declared type is refused by the service rather than guessed
        at here."""
        await me_router.upload_my_avatar(_request(b"bytes"), _actor(), _db())

        assert upload.await_args.kwargs["content_type"] == "application/octet-stream"

    async def test_the_raw_body_is_what_gets_stored(self, upload: AsyncMock) -> None:
        """No multipart wrapper: the bytes on the wire are the image."""
        await me_router.upload_my_avatar(_request(b"\x89PNG-data", "image/png"), _actor(), _db())

        assert upload.await_args.kwargs["data"] == b"\x89PNG-data"

    @pytest.mark.parametrize(
        "reason",
        ["unsupported_avatar_type: allowed types are JPEG, PNG, WebP, GIF.",
         "avatar_too_large: images must be 2 MiB or smaller."],
    )
    async def test_a_rejected_image_is_unprocessable_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch, reason: str
    ) -> None:
        """422 rather than 400: the request is well-formed, the content is
        not acceptable -- and the service's sentence is passed through
        because it names the limit the person has to work within.
        """
        monkeypatch.setattr(
            me_router.profile_service,
            "upload_avatar",
            AsyncMock(side_effect=AvatarUploadError(reason)),
        )

        with pytest.raises(HTTPException) as raised:
            await me_router.upload_my_avatar(_request(b"x", "image/png"), _actor(), _db())

        assert raised.value.status_code == 422
        assert raised.value.detail == reason


class TestTheProfileLinks:
    async def test_hitting_the_cap_is_unprocessable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            me_router.profile_service,
            "create_link",
            AsyncMock(side_effect=ProfileLinkError("too_many_links: at most 10 links.")),
        )

        with pytest.raises(HTTPException) as raised:
            await me_router.create_my_link(object(), _actor(), _db())

        assert raised.value.status_code == 422
        assert "too_many_links" in raised.value.detail

    @pytest.mark.parametrize("endpoint", ["update_my_link", "delete_my_link"])
    async def test_a_foreign_link_is_not_found_rather_than_forbidden(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        """The service scopes the lookup by owner, so it cannot tell a
        foreign id from a nonexistent one -- and answering 403 would confirm
        that someone else's link exists.
        """
        method = endpoint.split("_")[0] + "_link"
        monkeypatch.setattr(
            me_router.profile_service,
            method,
            AsyncMock(side_effect=ProfileLinkNotFoundError("nope")),
        )
        link_id = uuid4()

        with pytest.raises(HTTPException) as raised:
            if endpoint == "update_my_link":
                await me_router.update_my_link(link_id, object(), _actor(), _db())
            else:
                await me_router.delete_my_link(link_id, _actor(), _db())

        assert raised.value.status_code == 404
        assert raised.value.detail == {
            "error": "not_found",
            "resource": "profile_link",
            "id": str(link_id),
        }

    async def test_a_deleted_link_answers_no_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """204 with an explicit Response rather than a ``None`` return, so
        the body really is empty."""
        monkeypatch.setattr(me_router.profile_service, "delete_link", AsyncMock())

        response = await me_router.delete_my_link(uuid4(), _actor(), _db())

        assert response.status_code == 204
        assert response.body == b""


class TestTheRoleList:
    async def test_role_codes_are_de_duplicated_across_scopes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One person can hold ``teacher`` on four courses.

        The hook this feeds only asks "is this person a teacher?", so four
        copies would make the answer no different and the payload four times
        the size.
        """
        monkeypatch.setattr(
            me_router.access_control_api,
            "get_role_assignments_for_user",
            AsyncMock(
                return_value=[
                    SimpleNamespace(role_code="teacher"),
                    SimpleNamespace(role_code="teacher"),
                    SimpleNamespace(role_code="student"),
                    SimpleNamespace(role_code="teacher"),
                ]
            ),
        )

        assert await me_router.read_my_roles(_actor(), _db()) == ["teacher", "student"]

    async def test_the_first_occurrence_order_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A set would de-duplicate too, and would reorder the list between
        requests -- which the SPA renders directly.
        """
        monkeypatch.setattr(
            me_router.access_control_api,
            "get_role_assignments_for_user",
            AsyncMock(
                return_value=[
                    SimpleNamespace(role_code="student"),
                    SimpleNamespace(role_code="manager"),
                    SimpleNamespace(role_code="teacher"),
                ]
            ),
        )

        assert await me_router.read_my_roles(_actor(), _db()) == [
            "student",
            "manager",
            "teacher",
        ]

    async def test_someone_with_no_active_assignment_gets_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            me_router.access_control_api,
            "get_role_assignments_for_user",
            AsyncMock(return_value=[]),
        )

        assert await me_router.read_my_roles(_actor(), _db()) == []


class TestReadingYourOwnIdentity:
    @pytest.fixture
    def read_world(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        read = SimpleNamespace(organization_id=None, organization_name=None)
        monkeypatch.setattr(
            me_router, "get_current_user_read", AsyncMock(return_value=read)
        )
        return {"read": read}

    async def test_the_primary_organization_is_filled_in(
        self, monkeypatch: pytest.MonkeyPatch, read_world: dict[str, Any]
    ) -> None:
        """It used to answer ``organization_id=None`` for everyone.

        Anything org-scoped in the SPA -- the org-unit tree and the scope
        filters built on it -- then had no way to learn which organization
        the caller belongs to without an admin-only lookup they may not be
        allowed to make.
        """
        org_id = uuid4()
        monkeypatch.setattr(
            me_router.access_control_api,
            "get_user_primary_org",
            AsyncMock(return_value=SimpleNamespace(id=org_id, name="Faculty of CSE")),
        )

        result = await me_router.read_me(_actor(), _db())

        assert result.organization_id == org_id
        assert result.organization_name == "Faculty of CSE"

    async def test_someone_with_no_organization_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, read_world: dict[str, Any]
    ) -> None:
        """A platform admin belongs to no organization, and overwriting the
        fields with a placeholder would make them look like they did."""
        monkeypatch.setattr(
            me_router.access_control_api, "get_user_primary_org", AsyncMock(return_value=None)
        )

        result = await me_router.read_me(_actor(), _db())

        assert result.organization_id is None
        assert result.organization_name is None


def _router_source() -> str:
    return (
        Path(me_router.__file__).read_text(encoding="utf-8")
    )


def test_only_the_identity_read_sits_in_front_of_the_mfa_gate() -> None:
    """A deliberate, single exception.

    ``GET /users/me`` runs pre-MFA so the login page can show the person
    their own name and avatar while asking for the second factor. Every
    other endpoint here writes or reveals something and must stay behind the
    full gate, so a second use of the pre-MFA dependency is a hole rather
    than a convenience.
    """
    source = _router_source()
    assert source.count("Depends(get_current_user_pre_mfa)") == 1

    signature = inspect.signature(me_router.read_me)
    annotation = str(signature.parameters["current_user"].annotation)
    assert "get_current_user_pre_mfa" in annotation


def test_every_other_endpoint_requires_the_full_gate() -> None:
    """Counted rather than spot-checked: a new endpoint added without any
    dependency at all would be reachable by an unauthenticated caller.
    """
    source = _router_source()
    endpoints = source.count("@router.") + source.count("@me_root_router.")
    gated = source.count("Depends(get_current_user)") + source.count(
        "Depends(get_current_user_pre_mfa)"
    )
    assert gated == endpoints, (
        f"{endpoints} endpoints but {gated} authentication dependencies"
    )
