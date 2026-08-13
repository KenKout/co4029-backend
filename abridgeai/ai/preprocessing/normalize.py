"""Unicode normalization + conservative de-hyphenation.

Nothing in the extraction or chunking layer normalizes text today: a
verified round-trip of ``'The decision sup-\\nport system uses ﬁve\\xa0models'``
through ``TokenAwareChunker`` comes out byte-identical — hard hyphen, ``ﬁ``
ligature and NBSP all intact. Each of those costs retrieval quality: the
ligature makes ``five`` unmatchable by lexical search, the NBSP splits a
token, and ``sup-\\nport`` tokenizes as two fragments.

Two deliberate choices:

**NFC, never NFKC.** NFKC is the usual reflex and it is wrong for course
material: it rewrites ``x²`` to ``x2`` and ``½`` to ``1⁄2``, silently
corrupting the STEM notation this product exists to teach. NFC composes
diacritics (which matters for the Vietnamese corpus) without touching
compatibility characters. Ligatures and fullwidth forms — the two
compatibility classes that are pure noise — are folded explicitly instead.

**De-hyphenation is conservative.** The naive ``(\\w+)-\\s*\\n\\s*(\\w+)``
join measures only ~67% balanced accuracy against a trained classifier's
~92% (Hernæs 2019), and its errors destroy real compounds. The rules below
trade recall for precision, and the document's own vocabulary is used as a
free classifier: if the author writes ``decision-support`` intact anywhere,
a line-broken ``decision-\\nsupport`` keeps its hyphen.
"""

from __future__ import annotations

import re
import unicodedata

# Compatibility folds we DO want. Applied explicitly because NFKC would also
# flatten superscripts, subscripts, fractions and math alphanumerics.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

# Zero-width and formatting characters that carry no meaning in extracted
# text but break tokenization and exact-match search.
_ZERO_WIDTH = dict.fromkeys(
    [
        "­",  # soft hyphen
        "​",  # zero-width space
        "‌",  # zero-width non-joiner
        "‍",  # zero-width joiner
        "⁠",  # word joiner
        "﻿",  # BOM / zero-width no-break space
    ],
    "",
)

# Space-like characters that should behave as a plain space.
_SPACES = dict.fromkeys(
    [
        " ",  # no-break space
        " ",  # figure space
        " ",  # thin space
        " ",  # narrow no-break space
        "　",  # ideographic space
    ],
    " ",
)

_LINE_SEPARATORS = {" ": "\n", " ": "\n"}

_TRANSLATION = str.maketrans({**_LIGATURES, **_ZERO_WIDTH, **_SPACES, **_LINE_SEPARATORS})

_FULLWIDTH_RE = re.compile(r"[！-～]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)

# A hyphen at end-of-line followed by a continuation word.
_LINE_HYPHEN_RE = re.compile(r"(\w{2,})-[ \t]*\n[ \t]*(\w+)")
# Hyphenated compounds the author wrote intact somewhere in the document.
_INTACT_COMPOUND_RE = re.compile(r"\b(\w{2,})-(\w+)\b")


def _fold_fullwidth(match: re.Match[str]) -> str:
    """Map U+FF01–U+FF5E to their ASCII equivalents (offset 0xFEE0)."""
    return chr(ord(match.group(0)) - 0xFEE0)


def _strip_controls(text: str) -> str:
    """Drop Unicode category C* except newline and tab.

    Extractors occasionally emit stray control bytes from broken text
    layers; they render as boxes downstream and poison exact-match search.
    """
    return "".join(
        ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )


def normalize_text(text: str) -> str:
    """NFC + explicit compatibility folds + whitespace collapse."""
    if not text:
        return text
    out = unicodedata.normalize("NFC", text)
    out = out.translate(_TRANSLATION)
    out = _FULLWIDTH_RE.sub(_fold_fullwidth, out)
    out = _strip_controls(out)
    out = _MULTI_SPACE_RE.sub(" ", out)
    out = _TRAILING_WS_RE.sub("", out)
    return _MULTI_NEWLINE_RE.sub("\n\n", out)


def sanitize_json_value(value: object) -> object:
    """Recursively strip control characters so ``value`` is JSONB-safe.

    PostgreSQL's ``jsonb`` rejects ``\u0000`` and psycopg raises
    ``UntranslatableCharacter`` when a broken PDF text layer or OCR output
    smuggles one into ``extracted_metadata``. Dicts and lists are mutated in
    place (so a frozen ``ExtractedContent.metadata`` dict stays usable);
    strings pass through :func:`_strip_controls`.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = sanitize_json_value(item)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = sanitize_json_value(item)
        return value
    if isinstance(value, str):
        return _strip_controls(value)
    return value


def build_hyphen_vocab(text: str) -> set[str]:
    """Collect hyphenated compounds the document writes intact.

    Used as a free, document-internal classifier: a compound that appears
    unbroken elsewhere is an authored hyphen, not a line-break artefact.
    """
    return {
        f"{m.group(1).lower()}-{m.group(2).lower()}" for m in _INTACT_COMPOUND_RE.finditer(text)
    }


def dehyphenate(text: str, vocab: set[str] | None = None) -> tuple[str, int]:
    """Join words split across a line break. Returns ``(text, join_count)``.

    A join happens only when ALL of these hold, because every one of them
    is a case where the naive regex is known to be wrong:

    * the head is >= 2 chars — ``a-\\nsymmetric`` is more likely a bullet
    * the tail starts lowercase — ``Anh-\\nMinh`` is a name, ``co-\\nOp`` a title
    * neither fragment contains a digit — ``ISO-\\n9001`` is an identifier
    * the head is not capitalized — protects proper nouns mid-sentence
    * the compound is not written intact elsewhere in the document
    """
    if not text:
        return text, 0
    vocab = vocab if vocab is not None else build_hyphen_vocab(text)
    joins = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal joins
        head, tail = match.group(1), match.group(2)
        if not tail[:1].islower():
            return match.group(0)
        if head[:1].isupper():
            return match.group(0)
        if any(ch.isdigit() for ch in head) or any(ch.isdigit() for ch in tail):
            return match.group(0)
        if f"{head.lower()}-{tail.lower()}" in vocab:
            return match.group(0)
        joins += 1
        return f"{head}{tail}"

    return _LINE_HYPHEN_RE.sub(_replace, text), joins


__all__ = ["build_hyphen_vocab", "dehyphenate", "normalize_text", "sanitize_json_value"]
