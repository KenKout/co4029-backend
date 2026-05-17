"""ENVIRONMENT=local mock fallback per Reconciliation §B10.

When the configured environment is ``local`` and ``dispatch_extractor`` raises
``UnsupportedMimeError``, the ingestion pipeline can call
``maybe_local_mock_extractor(mime, settings)`` to obtain a stand-in that
returns a deterministic stub ``ExtractedContent``. This lets developers run
the pipeline end-to-end without ffmpeg / tesseract / Whisper credentials.

Returns ``None`` when the fallback should NOT be used (any non-local
environment), so callers can re-raise the original ``UnsupportedMimeError``.
"""

from __future__ import annotations

from typing import BinaryIO

from abridgeai.ai.extraction.base import ExtractedContent, MaterialExtractor, SourceLocation
from abridgeai.core.config import Settings, get_settings


class LocalMockExtractor:
    def __init__(self, mime: str) -> None:
        self._mime = mime

    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        del source
        text = (
            f"[mock content for {self._mime}]\n"
            "This is placeholder text emitted by the local-environment mock "
            "extractor. Configure a real extractor in production environments."
        )
        return ExtractedContent(
            text=text,
            metadata={"mock": True, "mime_type": self._mime},
            source_type="mock",
            source_locations=[SourceLocation(line_start=1, line_end=text.count("\n") + 1)],
        )


def maybe_local_mock_extractor(
    mime: str, settings: Settings | None = None
) -> MaterialExtractor | None:
    resolved = settings or get_settings()
    if resolved.environment.lower() != "local":
        return None
    return LocalMockExtractor(mime)


__all__ = ["LocalMockExtractor", "maybe_local_mock_extractor"]
