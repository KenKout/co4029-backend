from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from abridgeai.ai.extraction import (
    CODE_MIMES,
    EXTRACTOR_REGISTRY,
    AudioExtractor,
    CodeExtractor,
    ExtractedContent,
    HtmlExtractor,
    ImageExtractor,
    LocalMockExtractor,
    SourceLocation,
    TextExtractor,
    VideoExtractor,
    dispatch_extractor,
    maybe_local_mock_extractor,
)
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.core.config import Settings
from abridgeai.core.exceptions import AppError


@pytest.mark.asyncio
async def test_text_extractor_decodes_utf8() -> None:
    raw = "héllo wörld\nsecond line".encode()
    result = await TextExtractor().extract(raw)
    assert "héllo wörld" in result.text
    assert "second line" in result.text
    assert result.source_type == "text"
    assert result.source_locations[0].line_start == 1


@pytest.mark.asyncio
async def test_text_extractor_latin1_fallback() -> None:
    raw = "café".encode("latin-1")
    result = await TextExtractor().extract(raw)
    assert result.text
    assert result.source_type == "text"


@pytest.mark.asyncio
async def test_text_extractor_strips_utf8_bom() -> None:
    raw = b"\xef\xbb\xbfhello"
    result = await TextExtractor().extract(raw)
    assert result.text == "hello"


def test_code_extractor_registers_python() -> None:
    extractor = dispatch_extractor("text/x-python")
    assert isinstance(extractor, CodeExtractor)


def test_code_extractor_registers_at_least_23_mimes() -> None:
    assert len(CODE_MIMES) >= 23
    for mime in CODE_MIMES:
        assert EXTRACTOR_REGISTRY[mime] is CodeExtractor


@pytest.mark.asyncio
async def test_code_extractor_returns_code_source_type() -> None:
    raw = b"def hello():\n    return 'world'\n"
    result = await CodeExtractor().extract(raw)
    assert "def hello" in result.text
    assert result.source_type == "code"
    assert result.metadata["line_count"] >= 2


@pytest.mark.asyncio
async def test_html_extractor_strips_tags() -> None:
    raw = b"<html><body><h1>Title</h1><p>Body</p></body></html>"
    result = await HtmlExtractor().extract(raw)
    assert "Title" in result.text
    assert "Body" in result.text
    assert "<" not in result.text
    assert result.source_type == "html"


@pytest.mark.asyncio
async def test_html_extractor_extracts_headings() -> None:
    raw = b"<html><h1>One</h1><h2>Two</h2><p>Body</p></html>"
    result = await HtmlExtractor().extract(raw)
    assert "[H1] One" in result.text
    assert "[H2] Two" in result.text
    assert result.metadata["block_count"] == 3
    assert len(result.source_locations) == 3


@pytest.mark.asyncio
async def test_html_extractor_strips_scripts_and_styles() -> None:
    raw = b"<html><head><style>body{}</style></head><body><script>x=1</script><p>Visible</p></body></html>"
    result = await HtmlExtractor().extract(raw)
    assert "x=1" not in result.text
    assert "body{}" not in result.text
    assert "Visible" in result.text


@pytest.mark.asyncio
async def test_image_extractor_dispatches_tesseract() -> None:
    settings = Settings(image_ocr_provider="tesseract", image_ocr_lang="eng+vie")
    extractor = ImageExtractor(settings=settings)
    fake_data = {
        "text": ["Hello", "", "World"],
        "left": [10, 0, 80],
        "top": [20, 0, 20],
        "width": [40, 0, 50],
        "height": [15, 0, 15],
    }
    with (
        patch("abridgeai.ai.extraction.image.pytesseract", create=True) as mock_pyt,
        patch("abridgeai.ai.extraction.image.Image", create=True) as mock_image,
    ):
        mock_image.open.return_value.size = (200, 100)
        mock_pyt.image_to_data.return_value = fake_data
        mock_pyt.Output.DICT = "DICT"
        result = await extractor.extract(b"fake-png-bytes")
    assert "Hello" in result.text
    assert "World" in result.text
    assert result.metadata["ocr_provider"] == "tesseract"
    assert len(result.source_locations) == 2
    assert result.source_locations[0].bbox is not None
    # The configured OCR language is threaded through to pytesseract verbatim
    # (bilingual EN/VI corpus needs the 'vie' traineddata to be requested).
    assert mock_pyt.image_to_data.call_args.kwargs["lang"] == "eng+vie"


@pytest.mark.asyncio
async def test_image_extractor_dispatches_llm_vision() -> None:
    settings = Settings(image_ocr_provider="llm_vision", llm_api_key="test")
    fake_result = MagicMock()
    fake_result.content_json = {"text": "Hello from vision"}
    fake_result.model_name = "gpt-4o-vision"
    gateway = MagicMock()
    gateway.generate_json = AsyncMock(return_value=fake_result)
    db = MagicMock()
    extractor = ImageExtractor(settings=settings, gateway=gateway, db=db)
    with patch(
        "abridgeai.ai.extraction.image._image_meta",
        return_value=((800, 600), "image/png"),
    ):
        result = await extractor.extract(b"fake-png-bytes")
    gateway.generate_json.assert_awaited_once()
    call_kwargs = gateway.generate_json.await_args.kwargs
    assert call_kwargs["role"] is LLMRole.VISION
    # The image is sent as a proper base64 data URL via image_data_url — not
    # pasted into the prompt string — so the vision model actually sees it.
    assert call_kwargs["image_data_url"].startswith("data:image/png;base64,")
    assert "fake-png-bytes" not in call_kwargs["user_prompt"]
    assert "Hello from vision" in result.text
    assert result.metadata["ocr_provider"] == "llm_vision"
    assert result.source_locations[0].bbox == (0.0, 0.0, 800.0, 600.0)


@pytest.mark.asyncio
async def test_image_extractor_unknown_provider_raises() -> None:
    settings = Settings()
    object.__setattr__(settings, "image_ocr_provider", "bogus")
    extractor = ImageExtractor(settings=settings)
    with pytest.raises(ValueError, match="Unknown image_ocr_provider"):
        await extractor.extract(b"x")


@dataclass
class _FakeWhisperResponse:
    status_code: int = 200
    payload: dict[str, Any] = field(
        default_factory=lambda: {
            "text": "hello world",
            "language": "en",
            "duration": 1.5,
            "segments": [
                {"start": 0.0, "end": 0.7, "text": "hello"},
                {"start": 0.7, "end": 1.5, "text": "world"},
            ],
        }
    )
    text: str = "ok"

    def json(self) -> dict[str, Any]:
        return self.payload


@pytest.mark.asyncio
async def test_audio_extractor_uses_whisper_api() -> None:
    settings = Settings(
        audio_extraction_local=False,
        whisper_model="whisper-1",
        llm_api_key="sk-test",
    )
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    extractor = AudioExtractor(settings=settings, db=db)

    fake_response = _FakeWhisperResponse()

    async def fake_post(*args: Any, **kwargs: Any) -> _FakeWhisperResponse:
        return fake_response

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_response)

    with patch("abridgeai.ai.extraction.audio.httpx.AsyncClient", return_value=fake_client):
        result = await extractor.extract(b"fake-wav-bytes")

    assert "hello" in result.text
    assert "world" in result.text
    assert result.metadata["stt_provider"] == "whisper_api"
    assert result.metadata["segment_count"] == 2
    assert result.source_locations[0].timestamp_start_ms == 0
    assert result.source_locations[1].timestamp_end_ms == 1500
    fake_client.post.assert_awaited_once()
    db.add.assert_called_once()


@pytest.mark.asyncio
async def test_audio_extractor_local_path_missing_dep_errors() -> None:
    settings = Settings(audio_extraction_local=True, llm_api_key="sk-test")
    extractor = AudioExtractor(settings=settings)
    with patch.dict("sys.modules", {"faster_whisper": None}), pytest.raises(AppError) as excinfo:
        await extractor.extract(b"x")
    assert "faster-whisper" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_audio_extractor_whisper_api_error_writes_failed_audit() -> None:
    settings = Settings(
        audio_extraction_local=False,
        whisper_model="whisper-1",
        llm_api_key="sk-test",
    )
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    extractor = AudioExtractor(settings=settings, db=db)

    fake_response = _FakeWhisperResponse(status_code=500)
    fake_response.text = "internal error"

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_response)

    with (
        patch("abridgeai.ai.extraction.audio.httpx.AsyncClient", return_value=fake_client),
        pytest.raises(AppError),
    ):
        await extractor.extract(b"x")
    db.add.assert_called_once()


@pytest.mark.asyncio
async def test_video_extractor_pipeline_orchestration(tmp_path: Any) -> None:
    settings = Settings(
        ffmpeg_path="ffmpeg",
        video_frame_sample_fps=1.0,
        image_ocr_provider="tesseract",
    )

    audio_extractor = MagicMock()
    audio_extractor.extract = AsyncMock(
        return_value=ExtractedContent(
            text="hello\nworld",
            metadata={"segment_count": 2},
            source_type="audio",
            source_locations=[
                SourceLocation(timestamp_start_ms=0, timestamp_end_ms=500),
                SourceLocation(timestamp_start_ms=500, timestamp_end_ms=1500),
            ],
        )
    )

    image_extractor = MagicMock()
    image_extractor.extract = AsyncMock(
        side_effect=[
            ExtractedContent(
                text="frame-0 OCR",
                metadata={},
                source_type="image",
                source_locations=[SourceLocation(bbox=(0.0, 0.0, 100.0, 100.0))],
            ),
            ExtractedContent(
                text="frame-1 OCR",
                metadata={},
                source_type="image",
                source_locations=[SourceLocation(bbox=(0.0, 0.0, 100.0, 100.0))],
            ),
        ]
    )

    fake_outputs_workdir: dict[str, list[str]] = {}

    def _touch(path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(b"")

    async def fake_split(*, ffmpeg_path: str, input_path: str, workdir: str, fps: float) -> Any:
        from abridgeai.ai.extraction.video import _FfmpegOutputs

        audio_path = f"{workdir}/audio.wav"
        frame_paths = [f"{workdir}/frame_0001.jpg", f"{workdir}/frame_0002.jpg"]
        for path in [audio_path, *frame_paths]:
            await asyncio.to_thread(_touch, path)
        fake_outputs_workdir["paths"] = frame_paths
        return _FfmpegOutputs(audio_path=audio_path, frame_paths=frame_paths)

    extractor = VideoExtractor(
        settings=settings,
        audio_extractor=audio_extractor,
        image_extractor=image_extractor,
    )
    with patch("abridgeai.ai.extraction.video._split_audio_and_frames", side_effect=fake_split):
        result = await extractor.extract(b"fake-mp4-bytes")

    audio_extractor.extract.assert_awaited_once()
    assert image_extractor.extract.await_count == 2
    assert "frame-0 OCR" in result.text
    assert "frame-1 OCR" in result.text
    assert "[Audio @" in result.text
    assert "[Frame OCR @" in result.text
    assert result.source_type == "video"
    assert result.metadata["frame_count"] == 2


@pytest.mark.asyncio
async def test_local_mock_fallback_returns_for_local_env() -> None:
    settings = Settings(environment="local")
    extractor = maybe_local_mock_extractor("application/x-foo", settings)
    assert isinstance(extractor, LocalMockExtractor)
    result = await extractor.extract(b"")
    assert "[mock content for application/x-foo]" in result.text
    assert result.source_type == "mock"
    assert result.metadata["mock"] is True


def test_local_mock_returns_none_for_non_local_env() -> None:
    settings = Settings(environment="production", jwt_secret_key="x" * 64)
    assert maybe_local_mock_extractor("application/x-foo", settings) is None


def test_multimodal_mimes_registered() -> None:
    expected = {
        "text/plain": TextExtractor,
        "text/markdown": TextExtractor,
        "text/html": HtmlExtractor,
        "image/png": ImageExtractor,
        "image/jpeg": ImageExtractor,
        "image/webp": ImageExtractor,
        "image/gif": ImageExtractor,
        "audio/wav": AudioExtractor,
        "audio/mpeg": AudioExtractor,
        "audio/m4a": AudioExtractor,
        "audio/flac": AudioExtractor,
        "audio/ogg": AudioExtractor,
        "video/mp4": VideoExtractor,
        "video/quicktime": VideoExtractor,
        "video/x-matroska": VideoExtractor,
        "video/webm": VideoExtractor,
    }
    for mime, klass in expected.items():
        assert EXTRACTOR_REGISTRY[mime] is klass


def test_llm_role_appends_stt_and_vision() -> None:
    assert LLMRole.STT.value == "stt"
    assert LLMRole.VISION.value == "vision"
    values = [r.value for r in LLMRole]
    assert values.index("kg_extraction") < values.index("stt") < values.index("vision")
