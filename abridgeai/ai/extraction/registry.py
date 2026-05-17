"""MIME-string -> extractor class registry.

Each extractor module decorates its class with ``@register_extractor("mime/type")``
which side-effect populates ``EXTRACTOR_REGISTRY``. Callers in the ingestion
pipeline use ``dispatch_extractor(mime)`` to instantiate the right extractor
without having to ``import`` every format.

A MIME with no registered extractor raises ``UnsupportedMimeError``. The
ingestion pipeline catches this to decide between rejecting the upload and
falling back to a best-effort handler (e.g. the ``ENVIRONMENT=local`` mock).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from abridgeai.ai.extraction.base import MaterialExtractor

EXTRACTOR_REGISTRY: dict[str, type[MaterialExtractor]] = {}

_E = TypeVar("_E", bound=type[MaterialExtractor])


class UnsupportedMimeError(Exception):
    """Raised by ``dispatch_extractor`` when no extractor handles ``mime``."""


def register_extractor(mime: str) -> Callable[[_E], _E]:
    """Class decorator that binds an extractor to a MIME type.

    The decorator returns the class unchanged; only side effect is registry
    insertion. Re-registering the same MIME silently overwrites — useful for
    test patching, harmless in production because each module registers once
    on first import.
    """

    def _bind(cls: _E) -> _E:
        EXTRACTOR_REGISTRY[mime] = cls
        return cls

    return _bind


def dispatch_extractor(mime: str) -> MaterialExtractor:
    """Instantiate the extractor registered for ``mime``.

    Raises ``UnsupportedMimeError`` if no handler is registered. Importing the
    extraction package (``abridgeai.ai.extraction``) ensures all built-in
    extractors have registered themselves before the first dispatch.
    """
    cls = EXTRACTOR_REGISTRY.get(mime)
    if cls is None:
        raise UnsupportedMimeError(f"No extractor registered for MIME type: {mime}")
    return cls()
