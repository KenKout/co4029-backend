"""Re-encode EXISTING course / career-path thumbnails in Garage to WebP.

Upload-time re-encoding (:mod:`abridgeai.infrastructure.thumbnails`)
only fixes NEW uploads; every thumbnail already in Garage keeps its
original bytes (often a multi-MiB JPEG/PNG) until someone re-uploads.
This script rewrites those objects in place using the SAME encoder, so
the whole fleet converges on ≤1280px WebP and the dashboard LCP fix
applies to existing courses too.

    uv run --no-sync python scripts/backfill_thumbnails.py            # dry-run
    uv run --no-sync python scripts/backfill_thumbnails.py --apply

Per object (storage_objects rows under the ``course-thumbnails/`` and
``career-path-thumbnails/`` prefixes — the exact prefixes the upload
code writes):

* fetch the bytes from Garage via the existing ``download_to_temp``;
* SKIP when already compliant (image/webp AND ≤1280px) — the run is
  idempotent, a second invocation is a no-op;
* SKIP GIFs (passthrough policy: re-encoding would flatten animation);
* otherwise re-encode with :func:`encode_thumbnail`, PUT the bytes back
  under the SAME key with ContentType ``image/webp``, and update the
  row's ``mime_type`` / ``size_bytes`` to match the new bytes.

The object key keeps its original ``.jpg``/``.png`` extension: keys are
immutable identity (presigned URLs, the Redis URL cache, and any
external references point at them); the authoritative type is the
Content-Type metadata, which is rewritten.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import tempfile
from pathlib import Path
from uuid import UUID

from PIL import Image, UnidentifiedImageError
from sqlalchemy import text

from abridgeai.core.db import get_sessionmaker
from abridgeai.infrastructure.s3 import download_to_temp, put_object_bytes
from abridgeai.infrastructure.thumbnails import (
    THUMBNAIL_MAX_DIMENSION,
    ThumbnailDecodeError,
    encode_thumbnail,
    is_webp,
)

# The prefixes the upload services write (courses/services/authoring.py
# `course-thumbnails/{course_id}/...` and career_paths/services/authoring.py
# `career-path-thumbnails/{path_id}/...`).
_LIST_SQL = text(
    """
    SELECT id, bucket, object_key, mime_type, size_bytes
    FROM storage_objects
    WHERE deleted_at IS NULL
      AND (object_key LIKE 'course-thumbnails/%'
           OR object_key LIKE 'career-path-thumbnails/%')
    ORDER BY object_key
    """
)

_UPDATE_SQL = text(
    """
    UPDATE storage_objects
    SET mime_type = :mime_type, size_bytes = :size_bytes
    WHERE id = :object_id
    """
)


class _Target:
    """Duck-typed storage target for the s3 helpers (bucket + key)."""

    def __init__(self, *, bucket: str, object_key: str) -> None:
        self.bucket = bucket
        self.object_key = object_key


def _probe(data: bytes) -> tuple[str | None, tuple[int, int] | None]:
    """Cheap Pillow sniff of (format, (w, h)); ``(None, None)`` if undecodable."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.format, image.size
    except (UnidentifiedImageError, OSError, ValueError):
        return None, None


async def _fetch_bytes(target: _Target) -> bytes:
    """Download an object to a scratch dir and read it back as bytes."""
    workdir = Path(tempfile.mkdtemp(prefix="backfill-thumbnails-"))
    try:
        path = await download_to_temp(target, workdir)
        return path.read_bytes()
    finally:
        for leftover in workdir.iterdir():
            leftover.unlink(missing_ok=True)
        workdir.rmdir()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually rewrite objects and DB rows (default: dry-run report)",
    )
    args = parser.parse_args()

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        rows = (await session.execute(_LIST_SQL)).mappings().all()
        if not rows:
            print("No storage_objects rows under the thumbnail prefixes.")
            return 0

        already_ok = gifs = failures = 0
        candidates = 0
        bytes_before = bytes_after = 0

        print(f"{'object_key':70} {'status':12} {'old':>16} {'new':>16}")
        for row in rows:
            object_id: UUID = row["id"]
            target = _Target(bucket=row["bucket"], object_key=row["object_key"])
            label = row["object_key"]
            if len(label) > 69:
                label = "…" + label[-68:]

            try:
                data = await _fetch_bytes(target)
            except Exception as exc:  # noqa: BLE001 -- report and keep going
                failures += 1
                print(f"{label:70} {'FETCH-FAIL':12} {exc!s:.46}")
                continue

            # Decide GIF by the actual container, not the DB mime_type —
            # the upload policy passes GIFs through, so the backfill must
            # never flatten an animation regardless of what the row says.
            fmt, dims = _probe(data)
            if dims is None:
                failures += 1
                print(f"{label:70} {'DECODE-FAIL':12}")
                continue

            if fmt == "GIF":
                gifs += 1
                print(f"{label:70} {'skip-gif':12}")
                continue

            old_summary = f"{dims[0]}x{dims[1]} {len(data) // 1024}KiB"
            if is_webp(data) and max(dims) <= THUMBNAIL_MAX_DIMENSION:
                already_ok += 1
                print(f"{label:70} {'skip-ok':12} {old_summary:>16}")
                continue

            # Header probing (above) can pass on a body that fails a full
            # decode — browsers tolerate truncated JPEG tails that strict
            # Pillow does not. Such objects keep serving as-is; rewriting
            # them would only trade a working (if bloated) image for an
            # error, so they are reported and skipped.
            try:
                encoded = encode_thumbnail(data, content_type=row["mime_type"] or "")
            except ThumbnailDecodeError:
                failures += 1
                print(f"{label:70} {'DECODE-FAIL':12} {old_summary:>16}")
                continue
            new_dims = _probe(encoded.data)[1] or (0, 0)
            new_summary = f"{new_dims[0]}x{new_dims[1]} {len(encoded.data) // 1024}KiB"
            print(f"{label:70} {'REENCODE':12} {old_summary:>16} {new_summary:>16}")

            candidates += 1
            bytes_before += len(data)
            bytes_after += len(encoded.data)
            if not args.apply:
                continue

            await put_object_bytes(target, encoded.data, content_type=encoded.content_type)
            await session.execute(
                _UPDATE_SQL,
                {
                    "object_id": object_id,
                    "mime_type": encoded.content_type,
                    "size_bytes": len(encoded.data),
                },
            )

        print(
            f"\n{len(rows)} object(s): {already_ok} already compliant, "
            f"{gifs} gif passthrough, {failures} failure(s), "
            f"{candidates} re-encoded this run."
        )
        if candidates and bytes_before:
            saved = 100 * (bytes_before - bytes_after) / bytes_before
            print(
                f"Bytes: {bytes_before // 1024} KiB -> {bytes_after // 1024} KiB "
                f"({saved:.1f}% smaller)."
            )
        if not args.apply:
            print("\nDry-run: nothing written. Re-run with --apply to execute.")
        else:
            await session.commit()
            print("\nCommitted object rewrites and storage_objects updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
