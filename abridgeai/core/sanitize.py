"""Server-side sanitization for rich quiz content (Phase 3).

``plain``   -> returned verbatim (FE renders as escaped text, never parsed).
``html``    -> nh3-sanitized HTML.
``markdown``-> the raw markdown string is nh3-sanitized so any embedded HTML is
              cleaned, while markdown syntax and ``$..$`` math delimiters survive
              (nh3 leaves non-tag text untouched). Markdown->HTML rendering is
              done on the FE.

Every write path that accepts a rich field calls :func:`sanitize_rich_content`
with the field's format discriminator so dangerous markup can never be stored.
"""

from __future__ import annotations

import nh3

# Allowed tags cover formatting, lists, tables, code, links, images, and the
# span/div wrappers KaTeX/markdown emit.
_ALLOWED_TAGS: set[str] = {
    "p",
    "br",
    "hr",
    "span",
    "div",
    "strong",
    "b",
    "em",
    "i",
    "u",
    "s",
    "sub",
    "sup",
    "mark",
    "small",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "blockquote",
    "pre",
    "code",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "a",
    "img",
}

_ALLOWED_ATTRS: dict[str, set[str]] = {
    "a": {"href", "title", "target"},  # 'rel' managed by link_rel below
    "img": {"src", "alt", "title", "width", "height"},
    "span": {"class", "style"},
    "div": {"class", "style"},
    "code": {"class"},
    "pre": {"class"},
    "td": {"colspan", "rowspan", "align"},
    "th": {"colspan", "rowspan", "align"},
}

# Only http(s) URLs; blocks javascript:, etc.
_ALLOWED_URL_SCHEMES: set[str] = {"http", "https"}


def sanitize_rich_content(value: str | None, *, fmt: str | None) -> str | None:
    """Return sanitized content for the given format discriminator.

    ``plain`` (or None fmt) is returned verbatim — the client renders it as
    escaped text. ``markdown``/``html`` are nh3-cleaned.
    """
    if value is None:
        return None
    if fmt in (None, "plain"):
        return value
    if fmt in ("markdown", "html"):
        return nh3.clean(
            value,
            tags=_ALLOWED_TAGS,
            attributes=_ALLOWED_ATTRS,
            url_schemes=_ALLOWED_URL_SCHEMES,
            link_rel="noopener noreferrer nofollow",
        )
    return value


__all__ = ["sanitize_rich_content"]
