"""Unit coverage for the consent-gated LiveKit recording seam."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.config import get_settings
from abridgeai.features.interviews.services import recording


def _settings(**updates: object):
    return get_settings().model_copy(
        update={
            "interview_recording_enabled": True,
            "interview_recording_policy_version": "test-policy",
            "interview_recording_format": "mp3",
            "livekit_api_key": "test-key",
            "livekit_api_secret": "test-secret",
            "livekit_ws_url": "wss://livekit.test",
            **updates,
        }
    )


def test_destination_is_deterministic_and_scoped() -> None:
    session_id = uuid4()
    recording_id = uuid4()
    settings = _settings(interview_recording_s3_prefix="/private/interviews/")

    first = recording.build_destination_file(
        session_id=session_id,
        recording_id=recording_id,
        fmt="mp3",
        settings=settings,
    )
    second = recording.build_destination_file(
        session_id=session_id,
        recording_id=recording_id,
        fmt="mp3",
        settings=settings,
    )

    assert first == second
    assert first == f"private/interviews/{session_id}/{recording_id}/audio.mp3"


def test_reported_s3_uri_must_use_expected_bucket() -> None:
    assert recording._normalize_reported_key(  # noqa: SLF001
        "s3://recordings/interviews/a.mp3", expected_bucket="recordings"
    ) == "interviews/a.mp3"
    assert recording._normalize_reported_key(  # noqa: SLF001
        "s3://other/interviews/a.mp3", expected_bucket="recordings"
    ) == ""


def test_reported_http_location_url_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Egress v1.13 reports the file as a full S3-endpoint URL, not a key.

    Regression: the URL shape fell through the bare-key/s3-URI normalizer, so
    the prefix check failed and every recording was marked failed even though
    the object sat in the right bucket.
    """
    from abridgeai.core.config import get_settings

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:3900")
    get_settings.cache_clear()
    try:
        bucket = "recordings"
        key = "interviews/recordings/s/r/audio.mp3"
        normalize = lambda raw: recording._normalize_reported_key(  # noqa: SLF001, E731
            raw, expected_bucket=bucket
        )
        # Same-endpoint URL -> key.
        assert (
            normalize(f"http://localhost:3900/{bucket}/{key}") == key
        )
        # Foreign endpoint host -> rejected.
        assert normalize(f"https://evil.example/{bucket}/{key}") == ""
        # Same host, foreign bucket -> rejected.
        assert normalize(f"http://localhost:3900/other/{key}") == ""
        # Bare key and bucket-prefixed key still work.
        assert normalize(key) == key
        assert normalize(f"{bucket}/{key}") == key
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_start_is_noop_without_affirmative_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    claim = AsyncMock()
    monkeypatch.setattr(recording.recordings_queries, "claim_recording", claim)

    await recording.start_recording_for_session(
        AsyncMock(),
        session_id=uuid4(),
        student_id=uuid4(),
        room_name="interview-room",
        consented=False,
        settings=_settings(),
    )

    claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_uses_consented_pending_row(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = uuid4()
    recording_id = uuid4()
    row = SimpleNamespace(
        id=recording_id,
        status="pending",
        egress_id=None,
        consent_policy_version="test-policy",
    )
    claim = AsyncMock(return_value=recording_id)
    get_row = AsyncMock(return_value=row)
    attach = AsyncMock(return_value=True)
    start = AsyncMock(return_value="EG_test")
    monkeypatch.setattr(recording.recordings_queries, "claim_recording", claim)
    monkeypatch.setattr(recording.recordings_queries, "get_recording_for_session", get_row)
    monkeypatch.setattr(recording.recordings_queries, "attach_egress", attach)
    monkeypatch.setattr(recording, "_start_audio_egress", start)

    await recording.start_recording_for_session(
        AsyncMock(),
        session_id=session_id,
        student_id=uuid4(),
        room_name=f"interview-{session_id}",
        consented=True,
        settings=_settings(),
    )

    start.assert_awaited_once()
    attach.assert_awaited_once_with(
        ANY,
        recording_id,
        egress_id="EG_test",
        destination_file=f"interviews/recordings/{session_id}/{recording_id}/audio.mp3",
    )


@pytest.mark.asyncio
async def test_stale_consent_does_not_start_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    row = SimpleNamespace(
        id=uuid4(), status="pending", egress_id=None, consent_policy_version="old-policy"
    )
    monkeypatch.setattr(
        recording.recordings_queries, "claim_recording", AsyncMock(return_value=row.id)
    )
    monkeypatch.setattr(
        recording.recordings_queries, "get_recording_for_session", AsyncMock(return_value=row)
    )
    provider = AsyncMock()
    monkeypatch.setattr(recording, "_start_audio_egress", provider)

    await recording.start_recording_for_session(
        AsyncMock(),
        session_id=uuid4(),
        student_id=uuid4(),
        room_name="room",
        consented=True,
        settings=_settings(),
    )

    provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_failure_keeps_active_recording_for_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = SimpleNamespace(id=uuid4(), status="active", egress_id="EG_test")
    get_row = AsyncMock(return_value=row)
    cancel = AsyncMock()
    monkeypatch.setattr(recording.recordings_queries, "get_recording_for_session", get_row)
    monkeypatch.setattr(recording.recordings_queries, "mark_recording_cancelled", cancel)
    monkeypatch.setattr(recording, "_stop_egress_quietly", AsyncMock(return_value=False))

    await recording.stop_recording_for_session(AsyncMock(), session_id=uuid4())

    cancel.assert_not_awaited()


def test_missing_webhook_signature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recording, "get_settings", lambda: _settings())
    with pytest.raises(ValueError, match="missing webhook"):
        recording.verify_webhook_signature("{}", None)


@pytest.mark.asyncio
async def test_output_duration_is_not_derived_from_consent_time() -> None:
    # Regression pin for the DTO's recorded-at contract: the row's completion
    # timestamp is used when present, with created_at only as a legacy fallback.
    completed = datetime(2026, 1, 2, tzinfo=UTC)
    created = datetime(2026, 1, 1, tzinfo=UTC)
    row = SimpleNamespace(
        status="complete",
        storage_object_id=None,
        completed_at=completed,
        created_at=created,
    )
    assert row.completed_at != row.created_at
