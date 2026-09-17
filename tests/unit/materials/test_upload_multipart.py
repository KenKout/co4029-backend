"""The multipart upload lifecycle, after the first batch of part URLs.

A large material is uploaded straight to object storage in numbered parts.
The backend never sees the bytes; it mints presigned URLs, and then S3 is
told which parts arrived. That makes three things load-bearing here, none
of which produce a useful error when they go wrong.

**Part numbers must line up.** S3 numbers parts from 1 and reassembles by
number. The second batch is presigned from ``part_from``, so an off-by-one
produces URLs the client happily uploads to and a ``CompleteMultipartUpload``
that fails much later with a message about a part nobody can find.

**A version id is always checked against the material in the path.** The
router's permission dependency guards the *material*, so a version that
belongs to a different one would let a caller act on another course's
upload through their own material's URL. Each of these functions is a
separate write path, so each carries its own check.

**An abandoned upload is marked cancelled.** Left at ``pending`` it looks
in-flight forever -- the same eternal spinner the ingest reaper exists to
clean up, arrived at by a different route.

Storage and the query layer are mocked; no S3, no database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.features.materials.services.authoring import _upload


def _version(material_id: UUID, version_id: UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=version_id or uuid4(), material_id=material_id, processing_status="pending"
    )


def _storage_view() -> SimpleNamespace:
    return SimpleNamespace(bucket="abridgeai", object_key="materials/week-1.pdf")


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A version that belongs to the material in the path."""
    material_id = uuid4()
    version = _version(material_id)

    monkeypatch.setattr(_upload, "require_version", AsyncMock(return_value=version))
    monkeypatch.setattr(
        _upload, "resolve_storage_view", AsyncMock(return_value=_storage_view())
    )
    monkeypatch.setattr(_upload, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(
        _upload,
        "_get_settings",
        lambda: SimpleNamespace(
            s3_url_ttl_seconds=3600,
            s3_bucket_name="abridgeai",
            aws_region="us-east-1",
            aws_endpoint_url=None,
            aws_public_endpoint_url=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
        ),
    )

    return {
        "db": SimpleNamespace(),
        "material_id": material_id,
        "version": version,
    }


class TestTheSecondBatchOfPartUrls:
    async def test_the_batch_starts_where_the_caller_asked(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """S3 reassembles by part number, so these have to continue the
        sequence the first batch started -- not restart at 1."""
        presign = AsyncMock(return_value=[(11, "url-11"), (12, "url-12")])
        monkeypatch.setattr(_upload, "_presign_existing_multipart", presign)

        result = await _upload.fetch_multipart_parts(
            world["db"],
            world["material_id"],
            world["version"].id,
            "upload-id",
            part_from=11,
            part_count=2,
        )

        assert presign.await_args.kwargs["start"] == 11
        assert presign.await_args.kwargs["count"] == 2
        assert [p.part_number for p in result.parts] == [11, 12]
        assert [p.url for p in result.parts] == ["url-11", "url-12"]

    async def test_the_upload_id_identifies_which_upload_to_extend(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        presign = AsyncMock(return_value=[(1, "u")])
        monkeypatch.setattr(_upload, "_presign_existing_multipart", presign)

        await _upload.fetch_multipart_parts(
            world["db"],
            world["material_id"],
            world["version"].id,
            "the-upload-id",
            part_from=1,
            part_count=1,
        )

        assert presign.await_args.kwargs["upload_id"] == "the-upload-id"

    async def test_the_urls_carry_an_expiry_the_client_can_plan_around(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """A browser uploading a large file needs to know when to come back
        for fresh URLs rather than discovering it mid-part."""
        monkeypatch.setattr(
            _upload, "_presign_existing_multipart", AsyncMock(return_value=[(1, "u")])
        )
        before = datetime.now(tz=UTC)

        result = await _upload.fetch_multipart_parts(
            world["db"],
            world["material_id"],
            world["version"].id,
            "upload-id",
            part_from=1,
            part_count=1,
        )

        assert result.expires_at > before
        assert result.expires_at.tzinfo is not None

    @pytest.mark.parametrize("part_from", [0, -1])
    async def test_a_part_number_below_one_is_refused(
        self, world: dict[str, Any], part_from: int
    ) -> None:
        """S3 numbers parts from 1. Zero would be accepted here and rejected
        by S3 on upload, after the client had already sent the bytes.
        """
        with pytest.raises(AppError, match="part_from must be >= 1"):
            await _upload.fetch_multipart_parts(
                world["db"],
                world["material_id"],
                world["version"].id,
                "upload-id",
                part_from=part_from,
                part_count=1,
            )

    @pytest.mark.parametrize("part_count", [0, -5])
    async def test_asking_for_no_parts_is_refused(
        self, world: dict[str, Any], part_count: int
    ) -> None:
        with pytest.raises(AppError, match="part_count must be in"):
            await _upload.fetch_multipart_parts(
                world["db"],
                world["material_id"],
                world["version"].id,
                "upload-id",
                part_from=1,
                part_count=part_count,
            )

    async def test_the_batch_size_is_capped(self, world: dict[str, Any]) -> None:
        """Each URL is a signing operation and the response carries them
        all; an unbounded request is a way to make one call expensive.
        """
        with pytest.raises(AppError, match=str(_upload._common.MULTIPART_FIRST_BATCH_CAP)):
            await _upload.fetch_multipart_parts(
                world["db"],
                world["material_id"],
                world["version"].id,
                "upload-id",
                part_from=1,
                part_count=_upload._common.MULTIPART_FIRST_BATCH_CAP + 1,
            )

    async def test_exactly_the_cap_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """The bound is inclusive; off by one here costs the client a round
        trip on every full batch."""
        cap = _upload._common.MULTIPART_FIRST_BATCH_CAP
        monkeypatch.setattr(
            _upload,
            "_presign_existing_multipart",
            AsyncMock(return_value=[(n, f"u{n}") for n in range(1, cap + 1)]),
        )

        result = await _upload.fetch_multipart_parts(
            world["db"],
            world["material_id"],
            world["version"].id,
            "upload-id",
            part_from=1,
            part_count=cap,
        )

        assert len(result.parts) == cap

    async def test_the_arguments_are_checked_before_anything_is_looked_up(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """A bad request should cost nothing: no version read, no signing."""
        lookup = AsyncMock()
        monkeypatch.setattr(_upload, "require_version", lookup)

        with pytest.raises(AppError):
            await _upload.fetch_multipart_parts(
                world["db"],
                world["material_id"],
                world["version"].id,
                "upload-id",
                part_from=0,
                part_count=1,
            )

        lookup.assert_not_awaited()


class TestAVersionMustBelongToItsMaterial:
    """The router guards the material in the path, not the version.

    Every one of these is a separate write path against object storage, so
    a caller pairing their own material id with someone else's version
    would reach it through a URL the permission check approved.
    """

    @pytest.fixture
    def foreign(self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]) -> dict[str, Any]:
        monkeypatch.setattr(
            _upload, "require_version", AsyncMock(return_value=_version(uuid4()))
        )
        return world

    async def test_fetching_more_parts_is_refused(self, foreign: dict[str, Any]) -> None:
        with pytest.raises(NotFoundError, match="does not belong to material"):
            await _upload.fetch_multipart_parts(
                foreign["db"],
                foreign["material_id"],
                uuid4(),
                "upload-id",
                part_from=1,
                part_count=1,
            )

    async def test_completing_is_refused(self, foreign: dict[str, Any]) -> None:
        with pytest.raises(NotFoundError, match="does not belong to material"):
            await _upload.complete_multipart(
                foreign["db"], foreign["material_id"], uuid4(), "upload-id", []
            )

    async def test_aborting_is_refused(self, foreign: dict[str, Any]) -> None:
        """The one that would otherwise destroy someone else's upload."""
        with pytest.raises(NotFoundError, match="does not belong to material"):
            await _upload.abort_multipart(
                foreign["db"], foreign["material_id"], uuid4(), "upload-id"
            )

    async def test_the_refusal_is_not_found_rather_than_forbidden(
        self, foreign: dict[str, Any]
    ) -> None:
        """A mismatch must not confirm that the version exists to someone
        pairing ids to find out."""
        with pytest.raises(NotFoundError):
            await _upload.abort_multipart(
                foreign["db"], foreign["material_id"], uuid4(), "upload-id"
            )


class TestFinishingTheUpload:
    async def test_the_parts_the_client_reports_are_what_s3_is_told(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """S3 verifies each ETag against the part it stored, so the pairing
        has to survive the hop between our DTO and theirs intact --
        swapping two numbers assembles the file in the wrong order.
        """
        complete = AsyncMock()
        monkeypatch.setattr(_upload, "complete_multipart_upload", complete)
        parts = [
            SimpleNamespace(part_number=1, etag="etag-one"),
            SimpleNamespace(part_number=2, etag="etag-two"),
        ]

        await _upload.complete_multipart(
            world["db"], world["material_id"], world["version"].id, "upload-id", parts
        )

        _view, upload_id, s3_parts = complete.await_args.args
        assert upload_id == "upload-id"
        assert [(p.part_number, p.etag) for p in s3_parts] == [(1, "etag-one"), (2, "etag-two")]

    async def test_an_abandoned_upload_is_marked_cancelled(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """``cancelled`` is terminal and ``pending`` is a promise.

        Left pending, an abandoned upload shows the teacher a spinner over
        work nobody is doing -- and the ingest reaper would eventually try
        to recover an upload that never happened.
        """
        monkeypatch.setattr(_upload, "abort_multipart_upload", AsyncMock())

        await _upload.abort_multipart(
            world["db"], world["material_id"], world["version"].id, "upload-id"
        )

        assert world["version"].processing_status == "cancelled"

    async def test_the_abort_reaches_storage_before_the_row_is_flushed(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Marking the row first and failing the abort would leave S3
        holding the parts it bills for, with nothing recording that they
        exist.
        """
        order: list[str] = []
        monkeypatch.setattr(
            _upload,
            "abort_multipart_upload",
            AsyncMock(side_effect=lambda *a: order.append("s3")),
        )
        monkeypatch.setattr(
            _upload, "flush_or_conflict", AsyncMock(side_effect=lambda *a: order.append("db"))
        )

        await _upload.abort_multipart(
            world["db"], world["material_id"], world["version"].id, "upload-id"
        )

        assert order == ["s3", "db"]


class TestPresigningAgainstAnExistingUpload:
    async def test_missing_credentials_are_refused_before_any_signing(
        self, world: dict[str, Any]
    ) -> None:
        """Signing with no key produces a URL that looks valid and 403s on
        use, which is a far harder failure to read than refusing here.
        """
        from abridgeai.infrastructure.errors import S3NotConfiguredError

        with pytest.raises(S3NotConfiguredError):
            await _upload._presign_existing_multipart(
                _storage_view(), upload_id="u", start=1, count=1
            )


class TestTheSimpleUploadUrl:
    """The non-AI path: a lesson resource that bypasses the pipeline."""

    @pytest.fixture
    def simple(self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]) -> dict[str, Any]:
        object_id = uuid4()
        create = AsyncMock(return_value=object_id)
        set_key = AsyncMock()
        mint = AsyncMock(return_value=("https://signed", datetime.now(tz=UTC)))
        monkeypatch.setattr(_upload, "create_storage_object", create)
        monkeypatch.setattr(_upload, "set_storage_object_key", set_key)
        monkeypatch.setattr(_upload, "create_upload_url", mint)
        return {
            **world,
            "object_id": object_id,
            "create": create,
            "set_key": set_key,
            "mint": mint,
            "actor": SimpleNamespace(user_id=uuid4()),
        }

    async def _request(self, simple: dict[str, Any], **over: Any):
        kwargs: dict[str, Any] = {
            "original_filename": "Week 1.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 2048,
            "actor": simple["actor"],
        }
        kwargs.update(over)
        return await _upload.request_simple_upload(simple["db"], **kwargs)

    async def test_the_key_is_namespaced_by_the_new_object_id(
        self, simple: dict[str, Any]
    ) -> None:
        """The row is created first precisely so its id can namespace the
        key: two teachers uploading ``Week 1.pdf`` to the same bucket would
        otherwise overwrite each other.
        """
        object_id, _view, _url, _expires = await self._request(simple)

        assert object_id == simple["object_id"]
        key = simple["set_key"].await_args.args[2]
        assert key == f"resources/{simple['object_id']}/Week 1.pdf"

    async def test_the_presigned_url_is_for_the_key_that_was_recorded(
        self, simple: dict[str, Any]
    ) -> None:
        """A URL pointing somewhere other than the stored key uploads the
        file to an object nothing will ever read."""
        _id, view, _url, _expires = await self._request(simple)

        assert view.object_key == simple["set_key"].await_args.args[2]
        assert simple["mint"].await_args.args[0].object_key == view.object_key

    async def test_the_declared_content_type_is_signed_in(
        self, simple: dict[str, Any]
    ) -> None:
        await self._request(simple, mime_type="application/pdf")

        assert simple["mint"].await_args.kwargs["content_type"] == "application/pdf"
        assert simple["create"].await_args.kwargs["content_type"] == "application/pdf"

    @pytest.mark.parametrize(("given", "stored"), [(None, 0), (-5, 0), (0, 0), (4096, 4096)])
    async def test_a_nonsense_size_is_clamped_to_zero(
        self, simple: dict[str, Any], given: int | None, stored: int
    ) -> None:
        """The size is what the client claims before uploading anything.

        A negative value would reach a column that cannot hold it, and
        ``None`` arrives whenever the browser cannot determine the size --
        neither should fail the request, because the real size is verified
        on completion anyway.
        """
        await self._request(simple, size_bytes=given)

        assert simple["create"].await_args.kwargs["size_bytes"] == stored

    async def test_the_uploader_is_recorded(self, simple: dict[str, Any]) -> None:
        await self._request(simple)

        assert simple["create"].await_args.kwargs["uploaded_by"] == simple["actor"].user_id
        assert simple["create"].await_args.kwargs["original_filename"] == "Week 1.pdf"
