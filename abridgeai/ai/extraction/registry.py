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

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from abridgeai.ai.extraction.base import MaterialExtractor

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm.gateway import LLMGateway
    from abridgeai.core.config import Settings

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


def dispatch_extractor(
    mime: str,
    *,
    db: AsyncSession | None = None,
    gateway: LLMGateway | None = None,
    settings: Settings | None = None,
    stage_name: str = "extraction",
) -> MaterialExtractor:
    """Instantiate the extractor registered for ``mime``.

    Raises ``UnsupportedMimeError`` if no handler is registered. Importing the
    extraction package (``abridgeai.ai.extraction``) ensures all built-in
    extractors have registered themselves before the first dispatch.

    Extractors have heterogeneous constructors: text/pdf/docx/pptx/html/xlsx
    take no args, while audio/image/video need some subset of
    ``db`` / ``gateway`` / ``settings`` / ``stage_name`` to write audit rows
    and reach the vision/STT providers. Rather than hard-code which extractor
    wants what, we inspect the constructor signature and pass only the
    keyword args it actually accepts. This keeps the media extractors working
    through the ingestion pipeline (which previously called ``cls()`` with no
    args and tripped their "needs a db/gateway" guards at runtime).
    """
    cls = EXTRACTOR_REGISTRY.get(mime)
    if cls is None:
        raise UnsupportedMimeError(f"No extractor registered for MIME type: {mime}")

    available: dict[str, Any] = {
        "db": db,
        "gateway": gateway,
        "settings": settings,
        "stage_name": stage_name,
    }
    try:
        params = inspect.signature(cls.__init__).parameters
    except (ValueError, TypeError):
        return cls()
    kwargs = {name: value for name, value in available.items() if name in params}
    return cls(**kwargs)
