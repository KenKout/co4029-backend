"""Quiz ideation stage orchestrator (T5.5).

Ports ``_ideate_for_outline`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:567-700``. One LLM
call maps a lesson outline + per-section budget onto section-tagged quiz
templates. Composes ``render_prompt`` (T0.10) + ``LLMGateway.generate_json``
(T2.1) + ``parse_ideation_response`` (parsers.py). Audit row tagged
``stage_name="ideation"`` is written by the gateway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.quizzes.ai.stages.ideation.parsers import (
    Template,
    parse_ideation_response,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_DEFAULT_BLOOM_ORDER = ("remember", "understand", "apply", "analyze", "evaluate", "create")


def _default_bloom_distribution(count: int) -> dict[str, int]:  # noqa: C901 -- verbatim port of legacy heuristic (FR-7)
    """Spread ``count`` questions across Bloom levels (FR-7).

    Verbatim port from ``backend/app/ai/haystack/prompts/quiz.py:253``.
    """
    count = max(1, int(count))
    if count == 1:
        return {"understand": 1}
    if count == 2:
        return {"understand": 1, "apply": 1}
    if count == 3:
        return {"remember": 1, "understand": 1, "apply": 1}
    if count == 4:
        return {"remember": 1, "understand": 2, "apply": 1}
    if count == 5:
        return {"remember": 1, "understand": 2, "apply": 2}
    if count <= 7:
        return {"remember": 1, "understand": 2, "apply": 3, "analyze": count - 6}
    if count <= 10:
        return {"remember": 2, "understand": 3, "apply": 3, "analyze": count - 8}
    if count <= 14:
        remainder = count - 3
        u = remainder // 3 + (1 if remainder % 3 >= 1 else 0)
        a = remainder // 3 + (1 if remainder % 3 >= 2 else 0)
        az = remainder // 3
        return {
            "remember": 2,
            "understand": u,
            "apply": a,
            "analyze": az,
            "evaluate": 1,
        }
    base: dict[str, int] = {
        "remember": max(2, count // 8),
        "understand": max(3, count // 4),
        "apply": max(3, count // 4),
        "analyze": max(2, count // 6),
        "evaluate": max(1, count // 10),
    }
    used = sum(base.values())
    if used < count:
        base["create"] = count - used
    elif used > count:
        while sum(base.values()) > count:
            top_key = max(base, key=lambda k: base[k])
            base[top_key] -= 1
            if base[top_key] == 0:
                del base[top_key]
    return base


def _render_outline_for_prompt(outlines: list[Any], budget: dict[str, int]) -> str:
    """Render lesson outlines as one labelled block per section."""
    blocks: list[str] = []
    for outline in outlines:
        for sec in outline.sections:
            chunk_ids = ", ".join(str(c)[:8] for c in sec.chunk_ids[:6])
            block = (
                f"[{sec.id} / depth={sec.depth} / "
                f'"{sec.title}" / pages {sec.page_range[0]}-{sec.page_range[1]} / '
                f"role={sec.content_role} / chunks: {chunk_ids}]\n"
                f"preview: {sec.preview[:400]}\n"
                f"budget: {budget.get(sec.id, 0)}"
            )
            blocks.append(block)
    return "\n\n".join(blocks)


def _redistribute_chunk_anchors_within_section(
    templates: list[Template],
    sections_by_id: dict[str, Any],
) -> list[Template]:
    """Reassign collisions on primary chunk_id within a section.

    Templates processed in input order. First template to claim a chunk
    wins. Subsequent templates targeting the same section + same primary
    chunk get reassigned to the section's first unused chunk_id when
    one exists; otherwise the original chunk is kept.
    """
    used_per_section: dict[str, set[str]] = {}
    out: list[Template] = []
    for tmpl in templates:
        section_id = tmpl.section_id
        chunk_ids = list(tmpl.source_chunk_ids)
        if not section_id or not chunk_ids:
            out.append(tmpl)
            continue

        section = sections_by_id.get(section_id)
        if section is None:
            out.append(tmpl)
            continue

        used = used_per_section.setdefault(section_id, set())
        primary = chunk_ids[0]

        if primary not in used:
            used.add(primary)
            out.append(tmpl)
            continue

        section_chunk_ids = [str(c) for c in section.chunk_ids]
        unused = [cid for cid in section_chunk_ids if cid not in used]
        if not unused:
            used.add(primary)
            out.append(tmpl)
            continue

        new_primary = unused[0]
        used.add(new_primary)
        new_chunk_ids = [new_primary, *chunk_ids[1:]]
        out.append(tmpl.model_copy(update={"source_chunk_ids": new_chunk_ids}))
    return out


def _filter_templates_to_outline(
    templates: list[Template],
    sections_by_id: dict[str, Any],
) -> list[Template]:
    """Drop templates whose section_id / chunk_ids don't match the outline."""
    cleaned: list[Template] = []
    for tmpl in templates:
        section = sections_by_id.get(tmpl.section_id)
        if section is None:
            continue
        valid_chunks = {str(cid) for cid in section.chunk_ids}
        kept_chunks = [cid for cid in tmpl.source_chunk_ids if cid in valid_chunks]
        if not kept_chunks:
            kept_chunks = [str(section.chunk_ids[0])] if section.chunk_ids else []
        if not kept_chunks:
            continue
        cleaned.append(tmpl.model_copy(update={"source_chunk_ids": kept_chunks}))
    return cleaned


async def ideate_for_outline(
    db: AsyncSession,
    run: Any,  # noqa: ANN401 -- GenerationRun ORM lives in legacy backend until T5.x port
    title: str,
    config: dict[str, Any],
    outlines: list[Any],
    budget: dict[str, int],
    *,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> list[Template]:
    """Map outline + budget to section-tagged templates via one LLM call.

    ``run.id`` flows as ``parent_run_id`` on the audit row;
    ``pipeline_run_id`` matches the parent pipeline run (T2.4). Returned
    templates are validated, outline-filtered, and chunk-deduped within
    each section. Inject ``gateway`` in tests.
    """
    sections_by_id: dict[str, Any] = {
        sec.id: sec for outline in outlines for sec in outline.sections
    }
    template_count = sum(budget.values())
    outline_text = _render_outline_for_prompt(outlines, budget)

    bloom_distribution = config.get("bloom_distribution") or {}
    if not bloom_distribution:
        bloom_distribution = _default_bloom_distribution(template_count)

    system_prompt = render_prompt("prompts/system.j2")
    user_prompt = render_prompt(
        "prompts/user.j2",
        title=title,
        difficulty=config.get("difficulty") or "mixed",
        bloom_distribution=bloom_distribution,
        template_count=template_count,
        budget=dict(budget),
        outline_text=outline_text,
        question_types=list(config.get("question_types") or ["multiple_choice"]),
        focus_topics=list(config.get("focus_topics") or []),
        avoid_topics=list(config.get("avoid_topics") or []),
        extra_instructions=(config.get("extra_instructions") or "").strip()
        if config.get("extra_instructions")
        else "",
    )

    client = gateway or LLMGateway()
    parent_run_id = getattr(run, "id", None)
    result = await client.generate_json(
        role=LLMRole.IDEATION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        db=db,
        stage_name="ideation",
        pipeline_run_id=pipeline_run_id,
        parent_run_id=parent_run_id,
    )

    templates = parse_ideation_response(result.content_json)
    cleaned = _filter_templates_to_outline(templates, sections_by_id)
    return _redistribute_chunk_anchors_within_section(cleaned, sections_by_id)


__all__ = [
    "_default_bloom_distribution",
    "_redistribute_chunk_anchors_within_section",
    "_render_outline_for_prompt",
    "ideate_for_outline",
]
