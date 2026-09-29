"""Course / career-path thumbnail pipeline: encode-on-upload, stable delivery.

Lighthouse flagged thumbnail delivery as the dominant LCP cost on
/dashboard (~1.8 MiB est. savings): uploads were stored verbatim (no
resize), and every read minted a *fresh* presigned URL — a URL that
differs per request can never be reused by the browser cache, and the
responses carried no ``Cache-Control`` header either.

This module owns the two halves of the fix, shared by the ``courses``
and ``career_paths`` features. It lives in ``infrastructure`` rather
than under either feature because the import-linter independence
contract forbids cross-feature imports outside ``api.public``, and a
thumbnail codec is not a feature-level read API.

**Encode-on-upload** (:func:`encode_thumbnail`) — re-encode the image
with Pillow to WebP, longest side ≤ 1280 px (the dashboard renders at
~330 CSS px, so 1280 covers 3× DPR with margin), quality 82, effort 6.
Only down-scaling happens: a small upload is re-encoded but never
enlarged. GIF passes through untouched — it is the one allowed type
whose animation a single-frame re-encode would destroy.

**Stable delivery** (:func:`mint_thumbnail_url`) — one presigned GET
URL per storage object, memoised in Redis for just under the presign
TTL and served with ``Cache-Control: public, max-age=86400`` so the
browser keeps its copy for a day and, within that window, never
re-requests the object at all.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from abridgeai.core.cache import THUMBNAIL_URL, get_json, set_json
from abridgeai.core.config import get_settings
from abridgeai.infrastructure.s3 import create_stream_url

#: Longest-side cap. Never enlarged past, only shrunk towards.
THUMBNAIL_MAX_DIMENSION = 1280
#: WebP encoder knobs: 82 is visually lossless-ish for photos at card size;
#: method 6 is the slowest/best compression (upload is a one-off).
THUMBNAIL_WEBP_QUALITY = 82
THUMBNAIL_WEBP_METHOD = 6
#: 24 h browser caching. Deliberately longer than the presign TTL: within
#: max-age the browser serves from its own cache and never re-requests,
#: so the 1 h signature expiry never bites. A re-upload creates a NEW
#: object key (per-upload uuid), hence a new URL — no stale-cache window.
THUMBNAIL_CACHE_CONTROL = "public, max-age=86400"
#: Safety margin so the cached URL is always younger than its signature.
_TTL_SAFETY_SECONDS = 60

_WEBP_MAGIC = b"RIFF"


@dataclass(frozen=True)
class EncodedThumbnail:
    """Bytes ready for ``put_object_bytes`` plus the metadata to record."""

    data: bytes
    content_type: str
    extension: str


class ThumbnailDecodeError(ValueError):
    """The uploaded bytes are not an image Pillow can decode."""


def is_webp(data: bytes) -> bool:
    """Cheap container check: RIFF header + WEBP tag."""
    return len(data) >= 12 and data[:4] == _WEBP_MAGIC and data[8:12] == b"WEBP"


def encode_thumbnail(data: bytes, *, content_type: str) -> EncodedThumbnail:
    """Re-encode an uploaded thumbnail as size-constrained WebP.

    Callers apply their 5 MiB cap to ``data`` BEFORE calling here, so the
    decode only ever sees size-bounded input; Pillow's own
    ``MAX_IMAGE_PIXELS`` decompression-bomb guard covers the pixel
    dimension on top of that.

    Raises :class:`ThumbnailDecodeError` when the bytes are not a
    decodable image — callers map that to their existing 422 upload
    error (the MIME allowlist can be spoofed by a mislabelled body).
    """
    if content_type == "image/gif":
        # GIF passes through verbatim: re-encoding would flatten the
        # animation to its first frame, and GIF thumbnails are rare.
        return EncodedThumbnail(data=data, content_type="image/gif", extension="gif")

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()  # force the decode now — corrupt bytes raise here
            # Honour EXIF orientation before resizing: a phone portrait
            # stored rotated would otherwise be shrunk along the wrong axis.
            oriented = ImageOps.exif_transpose(image)
            has_alpha = oriented.mode in ("RGBA", "LA", "PA") or "transparency" in oriented.info
            if oriented.mode not in ("RGB", "RGBA"):
                oriented = oriented.convert("RGBA" if has_alpha else "RGB")
            # thumbnail() only ever shrinks (never enlarges) and preserves
            # aspect, so a 3000×2000 JPEG lands at 1280×853 while a
            # 400×300 upload keeps its pixels.
            oriented.thumbnail(
                (THUMBNAIL_MAX_DIMENSION, THUMBNAIL_MAX_DIMENSION),
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            oriented.save(
                buffer,
                format="WEBP",
                quality=THUMBNAIL_WEBP_QUALITY,
                method=THUMBNAIL_WEBP_METHOD,
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ThumbnailDecodeError(
            "invalid_thumbnail: the uploaded file is not a decodable image."
        ) from exc

    return EncodedThumbnail(data=buffer.getvalue(), content_type="image/webp", extension="webp")


def thumbnail_cache_ttl() -> int:
    """Cache TTL, clamped below the presign TTL.

    Serving a cached URL after its signature expires hands the browser a
    dead link, so the entry must always expire before the URL does
    (same clamp the materials ``PRESIGNED_URL`` key intends by its
    3000 s < 3600 s defaults).
    """
    url_ttl = get_settings().s3_url_ttl_seconds
    return max(1, min(THUMBNAIL_URL.ttl_seconds, url_ttl - _TTL_SAFETY_SECONDS))


@dataclass
class _ThumbnailTarget:
    """Duck-typed storage target for :func:`create_stream_url`."""

    bucket: str
    object_key: str


async def mint_thumbnail_url(bucket: str, object_key: str) -> str:
    """Presigned thumbnail GET URL, stable for the cache TTL.

    A presigned URL embeds the minting timestamp in its signature, so
    minting per request yields a *different* URL every time and defeats
    the browser cache regardless of ``Cache-Control``. Memoising one URL
    per object key (which embeds the per-upload uuid) means consecutive
    list/detail responses hand back the same string and the browser
    reuses its cached bytes. A re-upload creates a new key, so there is
    no stale entry to invalidate.

    Redis failures are swallowed by :mod:`core.cache.json_store` — an
    outage degrades to per-request minting (the pre-fix behaviour) and
    never breaks a read.
    """
    key = THUMBNAIL_URL.format(object_key=object_key)
    namespace = THUMBNAIL_URL.pattern

    cached = await get_json(key, namespace=namespace)
    if isinstance(cached, str) and cached:
        return cached

    url, _ = await create_stream_url(
        _ThumbnailTarget(bucket=bucket, object_key=object_key),
        response_headers={"Cache-Control": THUMBNAIL_CACHE_CONTROL},
    )
    await set_json(key, url, ttl=thumbnail_cache_ttl(), namespace=namespace)
    return url


__all__ = [
    "EncodedThumbnail",
    "THUMBNAIL_CACHE_CONTROL",
    "THUMBNAIL_MAX_DIMENSION",
    "THUMBNAIL_WEBP_METHOD",
    "THUMBNAIL_WEBP_QUALITY",
    "ThumbnailDecodeError",
    "encode_thumbnail",
    "is_webp",
    "mint_thumbnail_url",
    "thumbnail_cache_ttl",
]
