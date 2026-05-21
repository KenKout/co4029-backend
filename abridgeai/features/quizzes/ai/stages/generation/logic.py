"""Quiz generation stage orchestrator (T5.6).

Ports ``_run_generation`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:758-816``.

Composes the T2.1 LLMGateway (``LLMRole.GENERATION``,
``stage_name="generation"``), the T0.10 Jinja2 prompt loader, and the
sibling :mod:`parsers` module. Stages do not implement HTTP / audit
logic themselves: the gateway writes one ``ai_model_calls`` row per
call and rolls audit up to ``pipeline_run_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.quizzes.ai.stages.generation.parsers import (
    GeneratedQuestion,
    parse_generation_response,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.knowledge_graph.schemas import KGContext
    from abridgeai.ai.retrieval import ChunkWithDistance


_STAGE_NAME = "generation"


async def generate_questions(
    title: str,
    config: dict[str, Any],
    chunks: list[ChunkWithDistance],
    templates: list[dict[str, Any]],
    kg_context: KGContext | None,
    db: AsyncSession,
    *,
    pipeline_run_id: UUID,
    parent_run_id: UUID | None = None,
    previous_questions: list[str] | None = None,
    gateway: LLMGateway | None = None,
) -> list[GeneratedQuestion]:
    """Run one GENERATION LLM call and return parsed questions.

    Parameters
    ----------
    title
        Quiz title — surfaced verbatim in the user prompt.
    config
        Run config dict (``difficulty``, ``question_types``,
        ``focus_topics``, ``avoid_topics``, ``extra_instructions``).
    chunks
        Source chunks selected by the T5.4 retrieval stage.
    templates
        Per-question templates from the T5.5 ideation stage. May be
        empty for the legacy "no templates" path; the prompt then asks
        the LLM to produce ``config["question_count"]`` questions.
    kg_context
        Knowledge-graph block from retrieval; ``None`` when KG is off.
    db
        Async session — passed through to the gateway for audit rows.
    pipeline_run_id
        Parent pipeline-run id; threaded into ``ai_model_calls`` so
        cost attribution rolls up to the parent generation run.
    previous_questions
        Optional list of previously-emitted question stems. When
        supplied (regeneration path), they are rendered inline so the
        LLM avoids producing conceptually identical questions.
    gateway
        Inject a custom :class:`LLMGateway` (test seam).

    Returns
    -------
    list[GeneratedQuestion]
        Parsed + Pydantic-validated questions. Bad LLM rows are dropped
        in :func:`parse_generation_response`; an empty list signals
        "every question failed validation".
    """

    user_prompt = render_prompt(
        "prompts/user.j2",
        title=title,
        difficulty=config.get("difficulty") or "medium",
        question_types=config.get("question_types") or ["multiple_choice"],
        focus_topics=list(config.get("focus_topics") or []),
        avoid_topics=list(config.get("avoid_topics") or []),
        extra_instructions=(config.get("extra_instructions") or "").strip(),
        kg_section=_render_kg_section(kg_context),
        templates=templates or [],
        previous_questions=[
            stem.strip()
            for stem in (previous_questions or [])
            if isinstance(stem, str) and stem.strip()
        ][-20:],
        chunks_block=_render_chunks(chunks),
    )
    system_prompt = render_prompt("prompts/system.j2")

    client = gateway or LLMGateway()
    result = await client.generate_json(
        role=LLMRole.GENERATION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        db=db,
        stage_name=_STAGE_NAME,
        pipeline_run_id=pipeline_run_id,
        parent_run_id=parent_run_id,
    )
    return parse_generation_response(result.content_json)


def _render_chunks(chunks: list[ChunkWithDistance]) -> str:
    """Render retrieved chunks as a labelled prompt block.

    Mirrors the legacy ``_render_chunks`` (prompts/quiz.py:431-472) but
    consumes ``ChunkWithDistance`` from the T2.9 retrieval primitives
    instead of the legacy ``RetrievedChunk`` dataclass.
    """
    if not chunks:
        return "No indexed chunks were found."
    rendered: list[str] = []
    for chunk in chunks:
        rendered.append(f"[{chunk.chunk_id}]\ntext: {chunk.content}")
    return "\n\n".join(rendered)


def _render_kg_section(kg_context: KGContext | None) -> str:
    """Render KGContext as plain-text prompt block; empty string when off."""
    if kg_context is None or kg_context.is_empty:
        return ""
    lines: list[str] = ["Knowledge graph concepts (from indexed lesson materials):"]
    for concept in kg_context.concepts:
        definition = concept.definition or "(no definition recorded)"
        lines.append(f"- {concept.name}: {definition}")
    if kg_context.prerequisites:
        lines.append("")
        lines.append("Prerequisite chains (source must be learned before target):")
        for edge in kg_context.prerequisites:
            evidence = f" — {edge.evidence}" if edge.evidence else ""
            lines.append(f"- {edge.source} → {edge.target}{evidence}")
    if kg_context.related:
        lines.append("")
        lines.append("Related concepts:")
        for edge in kg_context.related:
            lines.append(f"- {edge.source} ↔ {edge.target}")
    return "\n".join(lines)


__all__ = ["generate_questions"]
