"""End-to-end smoke tests: each extractor consumes its synthetic fixture (T4.8).

One test per (fixture, extractor) pair. Tests gate on optional binaries
(``tesseract``, ``ffmpeg``) and optional Python modules
(``faster_whisper``) so the suite skips cleanly in minimal environments.
The fixtures themselves are produced by
``tests/fixtures/multimodal/generate_fixtures.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from abridgeai.ai.extraction import (
    AudioExtractor,
    HtmlExtractor,
    ImageExtractor,
    SourceLocation,
    VideoExtractor,
    dispatch_extractor,
)
from abridgeai.core.config import Settings

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "multimodal"

PDF_PATH = FIXTURES_DIR / "sample.pdf"
DOCX_PATH = FIXTURES_DIR / "sample.docx"
PPTX_PATH = FIXTURES_DIR / "sample.pptx"
WAV_PATH = FIXTURES_DIR / "sample.wav"
MP4_PATH = FIXTURES_DIR / "sample.mp4"
PNG_PATH = FIXTURES_DIR / "text-image.png"
HTML_PATH = FIXTURES_DIR / "sample.html"

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
HTML_MIME = "text/html"
PNG_MIME = "image/png"


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.mark.asyncio
async def test_pdf_extractor_consumes_sample_pdf() -> None:
    assert PDF_PATH.exists(), "Run generate_fixtures.py to produce sample.pdf"
    extractor = dispatch_extractor(PDF_MIME)
    with PDF_PATH.open("rb") as fh:
        result = await extractor.extract(fh)
    assert "Hello World" in result.text
    assert "Page 2 content" in result.text
    assert result.source_type == "pdf"
    assert result.metadata["page_count"] == 2
    assert len(result.source_locations) == 2


@pytest.mark.asyncio
async def test_docx_extractor_consumes_sample_docx() -> None:
    assert DOCX_PATH.exists(), "Run generate_fixtures.py to produce sample.docx"
    extractor = dispatch_extractor(DOCX_MIME)
    with DOCX_PATH.open("rb") as fh:
        result = await extractor.extract(fh)
    assert "Test Fixture" in result.text
    assert "Hello World" in result.text
    assert result.source_type == "docx"
    assert result.metadata["paragraph_count"] >= 2


@pytest.mark.asyncio
async def test_pptx_extractor_consumes_sample_pptx() -> None:
    assert PPTX_PATH.exists(), "Run generate_fixtures.py to produce sample.pptx"
    extractor = dispatch_extractor(PPTX_MIME)
    with PPTX_PATH.open("rb") as fh:
        result = await extractor.extract(fh)
    assert "Slide 1" in result.text
    assert "Slide 2" in result.text
    assert "Hello World" in result.text
    assert result.source_type == "pptx"
    assert result.metadata["slide_count"] == 2


@pytest.mark.asyncio
async def test_html_extractor_consumes_sample_html() -> None:
    assert HTML_PATH.exists(), "Run generate_fixtures.py to produce sample.html"
    extractor = dispatch_extractor(HTML_MIME)
    assert isinstance(extractor, HtmlExtractor)
    with HTML_PATH.open("rb") as fh:
        result = await extractor.extract(fh)
    assert "Title" in result.text
    assert "Body" in result.text
    assert result.source_type == "html"


@pytest.mark.asyncio
async def test_image_extractor_consumes_text_image() -> None:
    assert PNG_PATH.exists(), "Run generate_fixtures.py to produce text-image.png"
    settings = Settings(image_ocr_provider="tesseract")
    extractor = ImageExtractor(settings=settings)
    raw = PNG_PATH.read_bytes()

    if _tesseract_available():
        result = await extractor.extract(raw)
        assert result.source_type == "image"
        assert result.metadata["ocr_provider"] == "tesseract"
        assert "Hello" in result.text or "World" in result.text
        return

    fake_data = {
        "text": ["Hello", "World"],
        "left": [10, 80],
        "top": [30, 30],
        "width": [60, 60],
        "height": [20, 20],
    }
    with (
        patch("abridgeai.ai.extraction.image.pytesseract", create=True) as mock_pyt,
        patch("abridgeai.ai.extraction.image.Image", create=True) as mock_image,
    ):
        mock_image.open.return_value.size = (200, 100)
        mock_pyt.image_to_data.return_value = fake_data
        mock_pyt.Output.DICT = "DICT"
        result = await extractor.extract(raw)
    assert "Hello" in result.text
    assert "World" in result.text
    assert result.metadata["ocr_provider"] == "tesseract"


@pytest.mark.asyncio
async def test_audio_extractor_consumes_sample_wav() -> None:
    assert WAV_PATH.exists(), "Run generate_fixtures.py to produce sample.wav"
    raw = WAV_PATH.read_bytes()

    settings = Settings(
        audio_extraction_local=False,
        whisper_model="whisper-1",
        llm_api_key="sk-test",
    )
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    extractor = AudioExtractor(settings=settings, db=db)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.text = "ok"
    fake_response.json = MagicMock(
        return_value={
            "text": "synthetic tone",
            "language": "en",
            "duration": 5.0,
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "synthetic"},
                {"start": 2.5, "end": 5.0, "text": "tone"},
            ],
        }
    )
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_response)

    with patch("abridgeai.ai.extraction.audio.httpx.AsyncClient", return_value=fake_client):
        result = await extractor.extract(raw)

    fake_client.post.assert_awaited_once()
    files_arg: dict[str, Any] = fake_client.post.await_args.kwargs["files"]
    _, payload_bytes, _ = files_arg["file"]
    assert payload_bytes == raw
    assert "synthetic" in result.text
    assert result.metadata["stt_provider"] == "whisper_api"
    assert result.metadata["segment_count"] == 2


@pytest.mark.asyncio
async def test_video_extractor_consumes_sample_mp4() -> None:
    if not MP4_PATH.exists():
        pytest.skip("sample.mp4 missing - install ffmpeg and re-run generate_fixtures.py")
    if not _ffmpeg_available():
        pytest.skip("ffmpeg binary not on PATH")

    settings = Settings(
        ffmpeg_path="ffmpeg",
        video_frame_sample_fps=1.0,
        image_ocr_provider="tesseract",
    )

    audio_extractor = MagicMock()
    audio_extractor.extract = AsyncMock(
        return_value=__import__(
            "abridgeai.ai.extraction", fromlist=["ExtractedContent"]
        ).ExtractedContent(
            text="audio segment one",
            metadata={"segment_count": 1},
            source_type="audio",
            source_locations=[SourceLocation(timestamp_start_ms=0, timestamp_end_ms=5000)],
        )
    )
    image_extractor = MagicMock()
    image_extractor.extract = AsyncMock(
        return_value=__import__(
            "abridgeai.ai.extraction", fromlist=["ExtractedContent"]
        ).ExtractedContent(
            text="frame text",
            metadata={},
            source_type="image",
            source_locations=[SourceLocation(bbox=(0.0, 0.0, 100.0, 100.0))],
        )
    )

    extractor = VideoExtractor(
        settings=settings,
        audio_extractor=audio_extractor,
        image_extractor=image_extractor,
    )
    with MP4_PATH.open("rb") as fh:
        result = await extractor.extract(fh)

    assert result.source_type == "video"
    audio_extractor.extract.assert_awaited()
    assert image_extractor.extract.await_count >= 1
    assert "audio segment" in result.text or "frame text" in result.text
