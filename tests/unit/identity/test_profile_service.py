"""A person editing their own profile: avatar, links, display fields.

Three things here are worth more than their line count.

The **avatar upload** is the one place the backend accepts raw bytes from a
browser and writes them to object storage itself. It validates type and
size before doing so, and it uploads before it touches the database -- so a
storage failure leaves no ``storage_objects`` row pointing at an object
that was never written.

The **link list** is a small self-service surface with two rules that exist
for different reasons: a cap, so a profile cannot become an unbounded link
farm, and an ownership check folded into the lookup, so someone else's link
id is indistinguishable from one that never existed.

The **PATCH semantics** differ between links and the profile itself, and
the difference is not cosmetic: on a link, an explicit ``"label": null``
clears the label, which only ``exclude_unset`` can tell apart from omitting
the field. The profile update drops every ``None``, so the same gesture
does nothing there. Both behaviours are pinned below as they stand.

Storage and the query layer are mocked.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.identity.services import profile as profile_service
from abridgeai.features.identity.services.profile import (
    AvatarUploadError,
    ProfileLinkError,
    ProfileLinkNotFoundError,
)


def _user() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        primary_email="student@test.local",
        status="active",
        last_login_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _db() -> SimpleNamespace:
    return SimpleNamespace(
        add=lambda _row: None,
        commit=AsyncMock(),
        refresh=AsyncMock(),
        get=AsyncMock(return_value=None),
    )


class TestTheAvatarIsValidatedBeforeAnythingIsWritten:
    """The only endpoint that takes raw bytes from a browser.

    Each refusal names what is allowed rather than just refusing, because
    the person on the other end is choosing a file, not debugging an API.
    """

    @pytest.mark.parametrize(
        "content_type",
        ["application/pdf", "image/svg+xml", "text/html", "image/tiff", ""],
    )
    async def test_an_unsupported_image_type_is_refused(self, content_type: str) -> None:
        """``image/svg+xml`` is the one worth noticing: an SVG is a document
        that can carry script, so it is not on the list even though it is an
        image.
        """
        with pytest.raises(AvatarUploadError, match="unsupported_avatar_type"):
            await profile_service.upload_avatar(
                _db(), _user(), data=b"x", content_type=content_type
            )

    @pytest.mark.parametrize(
        "content_type", ["image/jpeg", "image/png", "image/webp", "image/gif"]
    )
    async def test_the_four_supported_types_pass_validation(
        self, monkeypatch: pytest.MonkeyPatch, content_type: str
    ) -> None:
        monkeypatch.setattr(profile_service, "put_object_bytes", AsyncMock())
        monkeypatch.setattr(profile_service, "serialize_user_async", AsyncMock())
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            profile_service,
            "get_settings",
            lambda: SimpleNamespace(s3_bucket_name="abridgeai"),
        )

        await profile_service.upload_avatar(
            _db(), _user(), data=b"bytes", content_type=content_type
        )

    async def test_an_empty_file_is_refused(self) -> None:
        """A zero-byte upload is a browser or network mishap, not a picture.

        Stored, it would render as a broken image on every page the person
        appears on, with nothing saying why.
        """
        with pytest.raises(AvatarUploadError, match="empty_avatar"):
            await profile_service.upload_avatar(
                _db(), _user(), data=b"", content_type="image/png"
            )

    async def test_an_oversized_image_is_refused_with_the_limit(self) -> None:
        """The avatar is served inline on every roster row, so the cap is
        about what other people have to download, not about disk."""
        too_big = b"x" * (2 * 1024 * 1024 + 1)

        with pytest.raises(AvatarUploadError, match="2 MiB"):
            await profile_service.upload_avatar(
                _db(), _user(), data=too_big, content_type="image/png"
            )

    async def test_exactly_the_limit_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound is inclusive; an off-by-one here rejects a file the UI
        just told the person was fine."""
        monkeypatch.setattr(profile_service, "put_object_bytes", AsyncMock())
        monkeypatch.setattr(profile_service, "serialize_user_async", AsyncMock())
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            profile_service,
            "get_settings",
            lambda: SimpleNamespace(s3_bucket_name="abridgeai"),
        )

        await profile_service.upload_avatar(
            _db(), _user(), data=b"x" * (2 * 1024 * 1024), content_type="image/png"
        )

    async def test_a_refused_upload_never_reaches_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        put = AsyncMock()
        monkeypatch.setattr(profile_service, "put_object_bytes", put)

        for data, content_type in ((b"x", "application/pdf"), (b"", "image/png")):
            with pytest.raises(AvatarUploadError):
                await profile_service.upload_avatar(
                    _db(), _user(), data=data, content_type=content_type
                )

        put.assert_not_awaited()


class TestTheUploadHappensBeforeTheDatabaseRow:
    async def test_a_storage_failure_leaves_no_row_behind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ordering is the whole point of the comment on that line.

        Writing the row first and uploading second would leave a
        ``storage_objects`` row and an ``avatar_object_id`` pointing at an
        object that does not exist -- a profile whose avatar 404s forever,
        with the database insisting it is there.
        """
        monkeypatch.setattr(
            profile_service,
            "put_object_bytes",
            AsyncMock(side_effect=ConnectionError("bucket unreachable")),
        )
        monkeypatch.setattr(
            profile_service,
            "get_settings",
            lambda: SimpleNamespace(s3_bucket_name="abridgeai"),
        )
        added: list[Any] = []
        db = SimpleNamespace(
            add=added.append, commit=AsyncMock(), refresh=AsyncMock(), get=AsyncMock()
        )

        with pytest.raises(ConnectionError):
            await profile_service.upload_avatar(
                db, _user(), data=b"png bytes", content_type="image/png"
            )

        assert added == []
        db.commit.assert_not_awaited()

    async def test_the_object_key_is_namespaced_per_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two people uploading at once must not be able to collide, and an
        operator looking at the bucket should be able to tell whose avatar
        an object is without joining back to the database.
        """
        captured: dict[str, Any] = {}

        async def _put(ref: Any, data: bytes, *, content_type: str) -> None:
            captured["bucket"] = ref.bucket
            captured["key"] = ref.object_key
            captured["content_type"] = content_type

        monkeypatch.setattr(profile_service, "put_object_bytes", _put)
        monkeypatch.setattr(profile_service, "serialize_user_async", AsyncMock())
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            profile_service,
            "get_settings",
            lambda: SimpleNamespace(s3_bucket_name="abridgeai"),
        )
        user = _user()

        await profile_service.upload_avatar(
            _db(), user, data=b"bytes", content_type="image/webp"
        )

        assert captured["key"].startswith(f"avatars/{user.id}/")
        assert captured["key"].endswith(".webp"), "the extension follows the declared type"
        assert captured["content_type"] == "image/webp"

    async def test_a_profile_is_created_for_someone_who_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An account invited but never edited has no profile row yet, and
        uploading a picture should not be the thing that fails."""
        monkeypatch.setattr(profile_service, "put_object_bytes", AsyncMock())
        monkeypatch.setattr(profile_service, "serialize_user_async", AsyncMock())
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            profile_service,
            "get_settings",
            lambda: SimpleNamespace(s3_bucket_name="abridgeai"),
        )
        added: list[Any] = []
        db = SimpleNamespace(
            add=added.append, commit=AsyncMock(), refresh=AsyncMock(), get=AsyncMock()
        )
        user = _user()

        await profile_service.upload_avatar(
            db, user, data=b"bytes", content_type="image/png"
        )

        profiles = [row for row in added if hasattr(row, "avatar_object_id")]
        assert len(profiles) == 1
        assert profiles[0].display_name == user.primary_email, (
            "seeded from the email so the row is never nameless"
        )


class TestReadingAProfileSurvivesAStorageOutage:
    async def test_a_failing_presign_yields_no_avatar_rather_than_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A profile read happens on every page that shows a person.

        Letting a storage blip propagate would turn a missing thumbnail
        into a failed page load across the whole application.
        """
        monkeypatch.setattr(
            profile_service,
            "create_stream_url",
            AsyncMock(side_effect=ConnectionError("s3 down")),
        )
        db = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace()))
        profile = SimpleNamespace(avatar_object_id=uuid4())

        assert await profile_service._mint_avatar_url(db, profile) is None

    async def test_a_profile_with_no_avatar_needs_no_storage_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        presign = AsyncMock()
        monkeypatch.setattr(profile_service, "create_stream_url", presign)

        assert await profile_service._mint_avatar_url(_db(), None) is None
        assert (
            await profile_service._mint_avatar_url(
                _db(), SimpleNamespace(avatar_object_id=None)
            )
            is None
        )
        presign.assert_not_awaited()

    async def test_a_dangling_storage_pointer_yields_no_avatar(self) -> None:
        """The row was deleted from under the profile."""
        db = SimpleNamespace(get=AsyncMock(return_value=None))
        profile = SimpleNamespace(avatar_object_id=uuid4())

        assert await profile_service._mint_avatar_url(db, profile) is None


class TestTheProfileLinkList:
    async def test_a_profile_may_not_exceed_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A profile page shows a handful of destinations. Without a cap it
        is a self-service place to publish an unbounded list of links.
        """
        monkeypatch.setattr(
            profile_service.user_queries,
            "list_profile_links",
            AsyncMock(return_value=[object()] * profile_service.MAX_PROFILE_LINKS),
        )

        with pytest.raises(ProfileLinkError, match="too_many_links"):
            await profile_service.create_link(
                _db(), _user(), SimpleNamespace(link_type="github", url="https://x", label=None)
            )

    async def test_the_refusal_names_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            profile_service.user_queries,
            "list_profile_links",
            AsyncMock(return_value=[object()] * profile_service.MAX_PROFILE_LINKS),
        )

        with pytest.raises(ProfileLinkError, match=str(profile_service.MAX_PROFILE_LINKS)):
            await profile_service.create_link(
                _db(), _user(), SimpleNamespace(link_type="github", url="https://x", label=None)
            )

    async def test_one_below_the_cap_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An off-by-one here costs the person their last slot."""
        monkeypatch.setattr(
            profile_service.user_queries,
            "list_profile_links",
            AsyncMock(return_value=[object()] * (profile_service.MAX_PROFILE_LINKS - 1)),
        )
        monkeypatch.setattr(
            profile_service.UserProfileLinkRead, "model_validate", lambda row: row
        )
        added: list[Any] = []
        db = SimpleNamespace(add=added.append, commit=AsyncMock(), refresh=AsyncMock())

        await profile_service.create_link(
            db,
            _user(),
            SimpleNamespace(link_type="github", url="https://github.com/x", label="Code"),
        )

        assert len(added) == 1
        assert added[0].url == "https://github.com/x"

    @pytest.mark.parametrize("operation", ["update", "delete"])
    async def test_someone_elses_link_is_indistinguishable_from_a_missing_one(
        self, monkeypatch: pytest.MonkeyPatch, operation: str
    ) -> None:
        """The ownership filter lives inside the lookup, so both cases
        return nothing and both raise the same error.

        Splitting them would let anyone probe which link ids exist on other
        people's profiles by watching which error came back.
        """
        lookup = AsyncMock(return_value=None)
        monkeypatch.setattr(profile_service.user_queries, "get_profile_link", lookup)
        link_id = uuid4()
        user = _user()

        with pytest.raises(ProfileLinkNotFoundError):
            if operation == "update":
                await profile_service.update_link(
                    _db(), user, link_id=link_id, payload=SimpleNamespace(model_dump=dict)
                )
            else:
                await profile_service.delete_link(_db(), user, link_id=link_id)

        assert lookup.await_args.kwargs["user_id"] == user.id, (
            "the caller's own id is part of the lookup, not checked afterwards"
        )

    async def test_deleting_a_link_tombstones_it_rather_than_removing_the_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``user_profile_links`` carries the soft-delete mixin, and the
        hard-delete guard rejects ``session.delete()`` on such rows outright,
        so a physical delete would not merely be wrong -- it would raise.
        """
        link = SimpleNamespace(id=uuid4())
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile_link", AsyncMock(return_value=link)
        )
        cascade = AsyncMock()
        monkeypatch.setattr(profile_service, "soft_delete_cascade", cascade)
        db = SimpleNamespace(commit=AsyncMock())
        user = _user()

        await profile_service.delete_link(db, user, link_id=link.id)

        cascade.assert_awaited_once_with(db, link, user.id)
        db.commit.assert_awaited_once_with()


class TestPatchSemanticsOnALink:
    """``exclude_unset``, not ``exclude_none`` -- and the difference shows.

    A caller clears a label by sending ``"label": null``. Only the unset
    check can tell that apart from omitting the field, which must leave the
    existing label alone.
    """

    async def test_an_explicit_null_label_clears_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        link = SimpleNamespace(id=uuid4(), label="Old label", url="https://x")
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile_link", AsyncMock(return_value=link)
        )
        monkeypatch.setattr(
            profile_service.UserProfileLinkRead, "model_validate", lambda row: row
        )

        await profile_service.update_link(
            SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock()),
            _user(),
            link_id=link.id,
            payload=SimpleNamespace(model_dump=lambda exclude_unset=False: {"label": None}),
        )

        assert link.label is None

    async def test_an_omitted_label_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        link = SimpleNamespace(id=uuid4(), label="Keep me", url="https://x")
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile_link", AsyncMock(return_value=link)
        )
        monkeypatch.setattr(
            profile_service.UserProfileLinkRead, "model_validate", lambda row: row
        )

        await profile_service.update_link(
            SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock()),
            _user(),
            link_id=link.id,
            payload=SimpleNamespace(
                model_dump=lambda exclude_unset=False: {"url": "https://new"}
            ),
        )

        assert link.label == "Keep me"
        assert link.url == "https://new"

    async def test_a_null_url_is_ignored_rather_than_clearing_the_link(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only ``label`` is nullable. A link with no destination is not a
        link, so a null there is dropped rather than applied.
        """
        link = SimpleNamespace(id=uuid4(), label="L", url="https://keep")
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile_link", AsyncMock(return_value=link)
        )
        monkeypatch.setattr(
            profile_service.UserProfileLinkRead, "model_validate", lambda row: row
        )

        await profile_service.update_link(
            SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock()),
            _user(),
            link_id=link.id,
            payload=SimpleNamespace(model_dump=lambda exclude_unset=False: {"url": None}),
        )

        assert link.url == "https://keep"


class TestPatchSemanticsOnTheProfile:
    async def test_a_null_field_is_dropped_rather_than_clearing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unlike the link path, the profile update has no nullable carve-out.

        Every ``None`` is skipped, so there is currently no way to clear a
        profile field through this endpoint -- pinned as it stands, because
        the asymmetry with ``update_link`` is easy to change by accident in
        either direction.
        """
        profile = SimpleNamespace(user_id=uuid4(), display_name="Nguyen Van A", bio="Hello")
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile", AsyncMock(return_value=profile)
        )
        monkeypatch.setattr(
            profile_service.UserProfileRead, "model_validate", lambda row: row
        )

        await profile_service.update_profile(
            SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock()),
            _user(),
            SimpleNamespace(
                model_dump=lambda exclude_unset=False: {"bio": None, "display_name": "New Name"}
            ),
        )

        assert profile.bio == "Hello", "the null was skipped"
        assert profile.display_name == "New Name"

    async def test_a_missing_profile_is_created_with_a_usable_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to the email so the row is never nameless -- a blank
        display name would render as an empty space wherever the person
        appears.
        """
        monkeypatch.setattr(
            profile_service.user_queries, "get_profile", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            profile_service.UserProfileRead, "model_validate", lambda row: row
        )
        added: list[Any] = []
        user = _user()

        await profile_service.update_profile(
            SimpleNamespace(add=added.append, commit=AsyncMock(), refresh=AsyncMock()),
            user,
            SimpleNamespace(model_dump=lambda exclude_unset=False: {}),
        )

        assert len(added) == 1
        assert added[0].display_name == user.primary_email


def test_the_avatar_limit_is_two_mebibytes() -> None:
    """Named in the refusal message the person reads, so the two must agree."""
    assert profile_service._AVATAR_MAX_BYTES == 2 * 1024 * 1024


def test_svg_is_not_an_allowed_avatar_type() -> None:
    """An SVG is a document that can carry script, and it would be served
    from the same origin family as the rest of the application."""
    assert "image/svg+xml" not in profile_service._AVATAR_MIME_TYPES
