"""Token-aware chunker.

Pure-CPU split that respects a tiktoken token budget rather than character
counts. Ported from ``backend/app/ai/haystack/components/chunkers.py`` with
the budget rephrased from ``max_chars`` to ``max_tokens``: char-based
splitting under-counted CJK / Vietnamese / LaTeX-heavy text and over-flowed
real LLM context windows. ``cl100k_base`` (GPT-4 / GPT-4o family) is the
canonical encoder; for non-OpenAI providers it slightly over-counts but
never under-counts, which is the safe direction.

The chunker first splits on heading / ``[Page N]`` / ``[Slide N]`` markers
to preserve natural section boundaries, then within each section slides a
``max_tokens`` window with ``overlap_tokens`` of overlap to give downstream
embedding / LLM stages enough context across boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from abridgeai.ai.chunking.base import RawChunk

if TYPE_CHECKING:
    from abridgeai.ai.extraction import ExtractedContent

_SECTION_MARKER_RE = re.compile(
    r"^(?P<heading>#{1,3}\s+.+)|^\[Page (?P<page>\d+)\]$|^\[Slide (?P<slide>\d+)\]$",
    re.MULTILINE,
)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_TIKTOKEN_ENCODING_NAME = "cl100k_base"

_tokenizer_cache: Any = None


def _get_tokenizer() -> Any:  # noqa: ANN401 -- tiktoken Encoding has no public type
    global _tokenizer_cache  # noqa: PLW0603 -- module-level lazy init pattern
    if _tokenizer_cache is None:
        import tiktoken

        _tokenizer_cache = tiktoken.get_encoding(_TIKTOKEN_ENCODING_NAME)
    return _tokenizer_cache


def count_tokens(text: str) -> int:
    """Count cl100k_base tokens, falling back to a 4-char heuristic."""
    if not text:
        return 0
    try:
        return len(_get_tokenizer().encode(text, disallowed_special=()))
    except Exception:  # noqa: BLE001 -- tiktoken cache download blocked, fallback heuristic
        return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` cl100k_base tokens."""
    if max_tokens <= 0 or not text:
        return ""
    try:
        enc = _get_tokenizer()
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        decoded: str = enc.decode(tokens[:max_tokens])
        return decoded
    except Exception:  # noqa: BLE001 -- same robustness guarantee as count_tokens
        approx = max_tokens * 4
        return text[:approx]


@dataclass(frozen=True)
class _Section:
    heading: str
    body: str
    location_metadata: dict[str, Any] = field(default_factory=dict)


class TokenAwareChunker:
    """Section-aware, token-budget split.

    Stable across runs for the same input. Configurable via constructor or
    keyword overrides on ``chunk()``.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        section_context: str = "",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._section_context = section_context

    def chunk(self, content: ExtractedContent, **opts: Any) -> list[RawChunk]:  # noqa: ANN401 -- forwarded chunker kwargs
        max_tokens = int(opts.get("max_tokens", self._max_tokens))
        overlap_tokens = int(opts.get("overlap_tokens", self._overlap_tokens))
        section_context = str(opts.get("section_context", self._section_context))

        text = (content.text or "").strip()
        if not text:
            return []

        sections = _split_into_sections(text)
        chunks: list[RawChunk] = []
        for section in sections:
            heading_label = section.heading
            if heading_label and section_context:
                ctx = f"{section_context} > {heading_label}"
            elif heading_label:
                ctx = heading_label
            else:
                ctx = section_context
            for piece in _split_section_by_tokens(
                section.body,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            ):
                metadata: dict[str, Any] = {
                    "source_type": content.source_type,
                    "token_count": count_tokens(piece),
                }
                if ctx:
                    metadata["section"] = ctx
                metadata.update(section.location_metadata)
                chunks.append(
                    RawChunk(
                        content=piece.strip(),
                        chunk_index=len(chunks),
                        metadata=metadata,
                    )
                )
        return chunks


def _split_into_sections(text: str) -> list[_Section]:
    matches = list(_SECTION_MARKER_RE.finditer(text))
    if not matches:
        return [_Section("", text)]

    sections: list[_Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(_Section("", preamble))

    for i, match in enumerate(matches):
        location: dict[str, Any] = {}
        if match.group("heading"):
            heading = match.group("heading").lstrip("#").strip()
        elif match.group("page"):
            page = int(match.group("page"))
            heading = f"Page {page}"
            location = {"page": page}
        else:
            slide = int(match.group("slide"))
            heading = f"Slide {slide}"
            location = {"slide": slide}
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append(_Section(heading, body, location))
    return sections


def _split_section_by_tokens(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if not text.strip():
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            joined = "\n\n".join(current)
            chunks.append(joined)
            tail = _tail_by_tokens(joined, overlap_tokens)
            current = [tail, para] if tail else [para]
            current_tokens = count_tokens("\n\n".join(current))
        else:
            current.append(para)
            current_tokens += para_tokens
    if current:
        chunks.append("\n\n".join(current))

    flushed: list[str] = []
    for chunk in chunks:
        if count_tokens(chunk) > max_tokens:
            flushed.extend(
                _split_by_sentence(chunk, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            )
        else:
            flushed.append(chunk)

    capped: list[str] = []
    for chunk in flushed:
        if count_tokens(chunk) <= max_tokens:
            capped.append(chunk)
        else:
            capped.extend(_hard_split_by_tokens(chunk, max_tokens=max_tokens))
    return capped


def _split_by_sentence(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = _SENTENCE_END_RE.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_tokens = count_tokens(sentence)
        if current and current_tokens + s_tokens > max_tokens:
            joined = " ".join(current)
            chunks.append(joined)
            tail = _tail_by_tokens(joined, overlap_tokens)
            current = [tail, sentence] if tail else [sentence]
            current_tokens = count_tokens(" ".join(current))
        else:
            current.append(sentence)
            current_tokens += s_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _hard_split_by_tokens(text: str, *, max_tokens: int) -> list[str]:
    """Last-resort encoder split for one giant unbroken sentence."""
    enc = _get_tokenizer()
    try:
        tokens = enc.encode(text, disallowed_special=())
    except Exception:  # noqa: BLE001 -- char fallback when tiktoken unavailable
        approx = max_tokens * 4
        return [text[i : i + approx] for i in range(0, len(text), approx)]
    return [enc.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _tail_by_tokens(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    try:
        enc = _get_tokenizer()
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= overlap_tokens:
            return text
        decoded: str = enc.decode(tokens[-overlap_tokens:])
        return decoded
    except Exception:  # noqa: BLE001 -- char fallback to preserve overlap roughly
        approx = overlap_tokens * 4
        return text[-approx:]


__all__ = ["TokenAwareChunker", "count_tokens", "truncate_to_tokens"]
