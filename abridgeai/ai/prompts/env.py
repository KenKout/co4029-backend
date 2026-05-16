"""Jinja2 environment factory for LLM prompt templates.

Design notes
------------
- ``autoescape=False``: templates render LLM prompts (plain text), not HTML.
  HTML escaping would corrupt prompt content (e.g. turning ``<`` into ``&lt;``).
- ``undefined=StrictUndefined``: missing template variables raise
  ``jinja2.UndefinedError`` at render time. Silent fallback to empty strings
  hides typos and degrades prompt quality without warning.
- ``keep_trailing_newline=True``: preserves the final newline authors put in
  ``.j2`` files; many LLM APIs are sensitive to trailing whitespace.

The cached singleton returned by :func:`get_jinja_env` is used when callers
have an absolute-from-package-root template path. The more common pattern —
caller-relative resolution from a stage's ``prompts/`` subdir — is handled by
:func:`abridgeai.ai.prompts.loader.render_prompt`, which builds a fresh
``Environment`` scoped to the caller's package directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jinja2

# Root of the installed ``abridgeai`` package. The cached environment uses this
# as its search path so callers may reference templates by package-relative
# paths (e.g. ``ai/prompts/shared/header.j2``).
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_jinja_env() -> jinja2.Environment:
    """Return the module-level Jinja2 environment singleton.

    Templates are resolved relative to the ``abridgeai`` package root.
    """
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=[str(_PACKAGE_ROOT)]),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
