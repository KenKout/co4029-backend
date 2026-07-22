"""Image extractor with dual OCR backends.

Two providers are selected by ``settings.image_ocr_provider``:

* ``tesseract`` — local OCR via the ``pytesseract`` wrapper around the
  system ``tesseract`` binary. Returns plain text and bounding-box
  ``SourceLocation`` records when the ``image_to_data`` call surfaces
  per-word geometry.
* ``llm_vision`` — calls ``LLMGateway.generate_json`` with
  ``LLMRole.VISION`` and a proper OpenAI-style multimodal image part
  (a base64 ``data:`` URL passed as an ``image_url`` content block, so
  the model actually sees the image). Used in API-only deployments where
  shipping a tesseract binary is undesirable. Output is a single
  ``SourceLocation`` covering the whole image.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import TYPE_CHECKING, BinaryIO

import pytesseract
from PIL import Image

from abridgeai.ai.extraction.base import ExtractedContent, SourceLocation
from abridgeai.ai.extraction.registry import register_extractor
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.core.config import Settings, get_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm.gateway import LLMGateway


_VISION_SYSTEM_PROMPT = (
    "You are an OCR engine. Extract every legible text element from the image "
    "verbatim. Preserve line breaks. Return JSON of shape "
    '{"text": "<extracted>"}.'
)
_VISION_USER_PROMPT = "Return all readable text from this image."


def _read_source(source: BinaryIO | bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as fh:
            return fh.read()
    return source.read()


def _tesseract_extract(raw: bytes, *, lang: str) -> ExtractedContent:
    image = Image.open(io.BytesIO(raw))
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words = data.get("text") or []
    lefts = data.get("left") or []
    tops = data.get("top") or []
    widths = data.get("width") or []
    heights = data.get("height") or []

    parts: list[str] = []
    locations: list[SourceLocation] = []
    for word, left, top, width, height in zip(words, lefts, tops, widths, heights, strict=False):
        token = (word or "").strip()
        if not token:
            continue
        parts.append(token)
        locations.append(
            SourceLocation(
                bbox=(float(left), float(top), float(left + width), float(top + height)),
            )
        )

    text = " ".join(parts).strip()
    return ExtractedContent(
        text=text,
        metadata={
            "ocr_provider": "tesseract",
            "image_size": image.size,
            "token_count": len(parts),
        },
        source_type="image",
        source_locations=locations,
    )


def _image_meta(raw: bytes) -> tuple[tuple[int, int], str]:
    """Return ``((width, height), mime)`` for the raw image bytes.

    The MIME drives the data-URL prefix sent to the vision model. Falls back
    to ``image/png`` when Pillow can't map the format (rare; keeps the data
    URL well-formed rather than emitting an empty subtype).
    """
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        fmt = (image.format or "PNG").lower()
        mime = f"image/{'jpeg' if fmt in ('jpg', 'jpeg') else fmt}"
        return (int(width), int(height)), mime


@register_extractor("image/png")
@register_extractor("image/jpeg")
@register_extractor("image/webp")
@register_extractor("image/gif")
class ImageExtractor:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        gateway: LLMGateway | None = None,
        db: AsyncSession | None = None,
        stage_name: str = "extraction",
    ) -> None:
        self._settings = settings or get_settings()
        self._gateway = gateway
        self._db = db
        self._stage_name = stage_name

    async def extract(self, source: BinaryIO | bytes | str) -> ExtractedContent:
        raw = await asyncio.to_thread(_read_source, source)
        provider = self._settings.image_ocr_provider
        if provider == "tesseract":
            return await asyncio.to_thread(
                _tesseract_extract, raw, lang=self._settings.image_ocr_lang
            )
        if provider == "llm_vision":
            return await self._llm_vision_extract(raw)
        raise ValueError(f"Unknown image_ocr_provider: {provider!r}")

    async def _llm_vision_extract(self, raw: bytes) -> ExtractedContent:
        if self._gateway is None or self._db is None:
            raise RuntimeError(
                "ImageExtractor configured for llm_vision but no LLMGateway / "
                "AsyncSession was injected; pass gateway= and db= when "
                "instantiating outside the registry dispatch path."
            )
        encoded = await asyncio.to_thread(base64.b64encode, raw)
        size, mime = await asyncio.to_thread(_image_meta, raw)
        # Proper OpenAI-style multimodal image part: the model receives a
        # base64 data URL as a real image_url content block, not text. The
        # earlier implementation pasted base64 into the prompt string, so the
        # model never actually saw the image.
        data_url = f"data:{mime};base64,{encoded.decode('ascii')}"
        result = await self._gateway.generate_json(
            role=LLMRole.VISION,
            system_prompt=_VISION_SYSTEM_PROMPT,
            user_prompt=_VISION_USER_PROMPT,
            db=self._db,
            stage_name=self._stage_name,
            image_data_url=data_url,
        )
        content = result.content_json
        text = ""
        if isinstance(content, dict):
            value = content.get("text")
            if isinstance(value, str):
                text = value
        return ExtractedContent(
            text=text.strip(),
            metadata={
                "ocr_provider": "llm_vision",
                "image_size": size,
                "model_name": result.model_name,
            },
            source_type="image",
            source_locations=[SourceLocation(bbox=(0.0, 0.0, float(size[0]), float(size[1])))],
        )


__all__ = ["ImageExtractor"]
