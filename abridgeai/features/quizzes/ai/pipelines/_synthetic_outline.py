"""Synthetic outline fallback for the full pipeline.

When the caller does not supply a structured outline + budget (e.g.
manual generation mode without prior outline build), construct a single
synthetic section containing every retrieved chunk so ideation has
something to anchor templates against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from abridgeai.ai.retrieval import ChunkWithDistance


class _SyntheticSection:
    __slots__ = ("chunk_ids", "content_role", "depth", "id", "page_range", "preview", "title")

    def __init__(self, *, id: str, chunk_ids: list[str]) -> None:  # noqa: A002
        self.id = id
        self.chunk_ids = chunk_ids
        self.title = "Lesson material"
        self.depth = 1
        self.page_range = (1, 1)
        self.content_role = "core"
        self.preview = ""


class _SyntheticOutline:
    __slots__ = ("sections",)

    def __init__(self, *, sections: list[_SyntheticSection]) -> None:
        self.sections = sections


def resolve_outline_inputs(
    outlines: list[Any] | None,
    budget: dict[str, int] | None,
    chunks: list[ChunkWithDistance],
    template_target: int,
) -> tuple[list[Any], dict[str, int]]:
    """Pass through caller-supplied outline/budget or synthesize a single
    section spanning every chunk."""
    if outlines and budget:
        return outlines, budget
    section_id = "synthetic-section"
    chunk_ids = [str(chunk.chunk_id) for chunk in chunks]
    section = _SyntheticSection(id=section_id, chunk_ids=chunk_ids)
    return [_SyntheticOutline(sections=[section])], {section_id: template_target}


__all__ = ["resolve_outline_inputs"]
