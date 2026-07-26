"""DB/LLM-wired preprocessing stage for the materials ingestion pipeline.

``ai/preprocessing`` holds the pure algorithms and knows nothing about
sessions, gateways or files. This module supplies the two side-effecting
collaborators it declares as Protocols:

* ``_PdfPageOcr``   — renders an image-only PDF page and reads it back with a
  vision model on ``LLMRole.PAGE_OCR`` (SMALL tier)
* ``_LlmPageAdjudicator`` — labels the narrow band of pages the deterministic
  rules could not settle, on ``LLMRole.PAGE_CLASSIFICATION`` (SMALL tier)

Both are gated by settings and both FAIL OPEN: any exception, timeout or
malformed response leaves the page exactly as the deterministic tiers left
it. A gateway outage must degrade retrieval quality, never silently delete a
teacher's content.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.ai.llm.roles import LLMRole
from abridgeai.ai.preprocessing import PreprocessConfig, run_preprocessing
from abridgeai.features.materials.queries.preprocess import list_restored_pages

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.extraction import ExtractedContent
    from abridgeai.ai.llm.gateway import LLMGateway
    from abridgeai.ai.preprocessing import PreprocessReport
    from abridgeai.core.config import Settings

logger = logging.getLogger(__name__)

_OCR_SYSTEM_PROMPT = (
    "You are an OCR engine for lecture slides and textbook pages. Transcribe "
    "every legible text element verbatim, preserving reading order and line "
    "breaks. Describe any chart, diagram or figure in one short sentence "
    "prefixed with '[Figure] ' so its meaning is searchable. Do not summarize, "
    "interpret or add commentary. Return JSON of shape {\"text\": \"<content>\"}. "
    "If the page carries no legible content, return {\"text\": \"\"}."
)
_OCR_USER_PROMPT = "Transcribe this page."

_CLASSIFY_SYSTEM_PROMPT = (
    "You label pages of teaching material so a retrieval system can "
    "de-prioritize administrative filler. Labels:\n"
    "- body: teachable subject matter (definitions, explanations, examples, "
    "exercises, data)\n"
    "- front_matter: cover/title page, instructor or syllabus block, course "
    "admin, closing 'thank you / questions' slide\n"
    "- summary: agenda, outline, table of contents, recap of other slides\n"
    "- reference: bibliography or citation list\n"
    "- divider: a bare section-break title with no content\n\n"
    "Prefer 'body' whenever you are unsure — wrongly de-prioritizing real "
    "teaching content is far worse than keeping filler. Return JSON of shape "
    '{"pages": [{"page": <int>, "label": "<label>", "confidence": <0..1>}]}.'
)


class _PdfPageOcr:
    """Renders one PDF page to PNG and reads it back with a vision model."""

    def __init__(
        self,
        *,
        source_path: Path,
        db: AsyncSession,
        gateway: LLMGateway,
        dpi: int,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
    ) -> None:
        self._source_path = source_path
        self._db = db
        self._gateway = gateway
        self._dpi = dpi
        self._pipeline_run_id = pipeline_run_id
        self._parent_job_id = parent_job_id
        self._doc: Any | None = None

    def _render(self, page_number: int) -> bytes | None:
        try:
            import fitz  # type: ignore[import-untyped,unused-ignore]
        except ImportError:
            return None
        if self._doc is None:
            self._doc = fitz.open(str(self._source_path))
        index = page_number - 1
        if index < 0 or index >= self._doc.page_count:
            return None
        pixmap = self._doc.load_page(index).get_pixmap(dpi=self._dpi)
        data: bytes = pixmap.tobytes("png")
        return data

    async def ocr_page(self, page_number: int) -> str | None:
        raw = await asyncio.to_thread(self._render, page_number)
        if not raw:
            return None
        encoded = await asyncio.to_thread(base64.b64encode, raw)
        data_url = f"data:image/png;base64,{encoded.decode('ascii')}"
        result = await self._gateway.generate_json(
            role=LLMRole.PAGE_OCR,
            system_prompt=_OCR_SYSTEM_PROMPT,
            user_prompt=_OCR_USER_PROMPT,
            db=self._db,
            stage_name="preprocessing",
            pipeline_run_id=self._pipeline_run_id,
            parent_job_id=self._parent_job_id,
            image_data_url=data_url,
        )
        content = result.content_json
        if isinstance(content, dict):
            # Not named ``text`` — that is ``sqlalchemy.text`` at module scope.
            value = content.get("text")
            if isinstance(value, str):
                return value
        return None

    def close(self) -> None:
        if self._doc is not None:
            try:
                self._doc.close()
            finally:
                self._doc = None


class _LlmPageAdjudicator:
    """Labels ambiguous pages in batches of 10."""

    def __init__(
        self,
        *,
        db: AsyncSession,
        gateway: LLMGateway,
        pipeline_run_id: UUID | None = None,
        parent_job_id: UUID | None = None,
    ) -> None:
        self._db = db
        self._gateway = gateway
        self._pipeline_run_id = pipeline_run_id
        self._parent_job_id = parent_job_id

    async def classify(self, pages: list[tuple[int, str]]) -> dict[int, tuple[str, float]]:
        payload = [
            {"page": page, "text": body[:400], "word_count": len(body.split())}
            for page, body in pages
        ]
        result = await self._gateway.generate_json(
            role=LLMRole.PAGE_CLASSIFICATION,
            system_prompt=_CLASSIFY_SYSTEM_PROMPT,
            user_prompt=json.dumps({"pages": payload}, ensure_ascii=False),
            db=self._db,
            stage_name="preprocessing",
            pipeline_run_id=self._pipeline_run_id,
            parent_job_id=self._parent_job_id,
        )
        content = result.content_json
        if not isinstance(content, dict):
            return {}
        out: dict[int, tuple[str, float]] = {}
        for row in content.get("pages") or []:
            if not isinstance(row, dict):
                continue
            try:
                page = int(row["page"])
                label = str(row["label"])
                confidence = float(row.get("confidence") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            out[page] = (label, confidence)
        return out


def config_from_settings(
    settings: Settings,
    *,
    mode: str = "full",
) -> PreprocessConfig:
    """Build the cascade config, honouring the per-material escape hatch.

    ``mode`` comes from ``learning_materials.preprocess_mode``:

    * ``off`` — raw extraction goes straight to chunking
    * ``normalize_only`` — unicode folds + de-hyphenation and nothing else.
      Those two are never destructive, which makes this the right setting for
      a document the filters misread: the teacher keeps every page while still
      getting ligatures and line-break hyphens repaired.
    * ``full`` — the whole cascade
    """
    if mode == "off":
        return PreprocessConfig(enabled=False)
    if mode == "normalize_only":
        return PreprocessConfig(
            enabled=settings.preprocess_enabled,
            normalize=True,
            dehyphenation=settings.preprocess_dehyphenation,
            blankness=False,
            running_marks=False,
            page_roles=False,
            deck_detection=False,
            ocr_enabled=False,
            llm_adjudication=False,
        )
    return PreprocessConfig(
        enabled=settings.preprocess_enabled,
        dehyphenation=settings.preprocess_dehyphenation,
        running_marks=settings.preprocess_running_marks,
        page_roles=settings.preprocess_page_roles,
        deck_detection=settings.preprocess_deck_detection,
        ocr_enabled=settings.preprocess_ocr_enabled,
        ocr_max_pages=settings.preprocess_ocr_max_pages,
        llm_adjudication=settings.preprocess_llm_adjudication,
        llm_min_confidence=settings.preprocess_llm_min_confidence,
    )


async def run_preprocess_stage(
    extracted: ExtractedContent,
    *,
    db: AsyncSession,
    settings: Settings,
    source_path: Path | None = None,
    llm_gateway: LLMGateway | None = None,
    pipeline_run_id: UUID | None = None,
    parent_job_id: UUID | None = None,
    material_version_id: UUID | None = None,
    course_id: UUID | None = None,
    mode: str = "full",
) -> tuple[ExtractedContent, PreprocessReport | None]:
    """Run the preprocessing cascade over freshly extracted content.

    Returns ``(content, report)``. On ANY failure the ORIGINAL content is
    returned unchanged — preprocessing is a quality improvement, never a
    correctness dependency, so it must not be able to fail an ingest.
    """
    config = config_from_settings(settings, mode=mode)
    if not config.enabled:
        return extracted, None

    ocr: _PdfPageOcr | None = None
    adjudicator: _LlmPageAdjudicator | None = None

    if (
        config.ocr_enabled
        and llm_gateway is not None
        and source_path is not None
        and extracted.source_type == "pdf"
    ):
        ocr = _PdfPageOcr(
            source_path=source_path,
            db=db,
            gateway=llm_gateway,
            dpi=settings.preprocess_ocr_dpi,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
        )

    if config.llm_adjudication and llm_gateway is not None:
        adjudicator = _LlmPageAdjudicator(
            db=db,
            gateway=llm_gateway,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
        )

    protected_pages: set[int] = set()
    if material_version_id is not None:
        try:
            protected_pages = await list_restored_pages(db, material_version_id)
        except Exception:
            logger.exception("preprocess: could not load teacher restores; proceeding without")

    try:
        content, report = await run_preprocessing(
            extracted,
            config=config,
            ocr=ocr,
            adjudicator=adjudicator,
            protected_pages=protected_pages,
        )
    except Exception:
        logger.exception(
            "preprocessing failed for source_type=%s; continuing with raw extraction",
            extracted.source_type,
        )
        return extracted, None
    finally:
        if ocr is not None:
            ocr.close()

    if material_version_id is not None and course_id is not None:
        await _persist_quarantine(
            db,
            report,
            material_version_id=material_version_id,
            course_id=course_id,
        )

    logger.info(
        "preprocess: pages %d->%d dropped=%d ocr=%d lines_stripped=%d deck=%s roles=%s",
        report.page_count_in,
        report.page_count_out,
        report.pages_dropped,
        report.pages_ocr_routed,
        report.lines_stripped,
        report.is_deck,
        report.role_counts,
    )
    return content, report


# Decisions worth an audit row. ``TAG_ROLE`` is included because
# de-prioritizing a page to front_matter/summary is a retrieval-visible call a
# teacher may want to overturn, even though no text was removed.
_QUARANTINED_ACTIONS = frozenset(
    {"drop_page", "strip_lines", "exclude_retrieval", "tag_role"}
)
# Backstop so one pathological document cannot write thousands of rows.
_MAX_QUARANTINE_ROWS = 500


async def _persist_quarantine(
    db: AsyncSession,
    report: PreprocessReport,
    *,
    material_version_id: UUID,
    course_id: UUID,
) -> None:
    """Upsert the cascade's decisions, PRESERVING teacher actions.

    ``ON CONFLICT ... DO UPDATE`` deliberately never touches
    ``teacher_action`` / ``teacher_action_by`` / ``teacher_action_at``: a
    reprocess re-derives what the rules decided, but a teacher's restore is
    an input to the next run, not an output of it.

    Best-effort. The audit trail is worth less than the ingest, so a failure
    here is logged and swallowed rather than rolled back into the pipeline.
    """
    rows: list[dict[str, Any]] = []
    for ordinal, decision in enumerate(report.decisions):
        if decision.action.value not in _QUARANTINED_ACTIONS:
            continue
        if len(rows) >= _MAX_QUARANTINE_ROWS:
            logger.warning(
                "preprocess: quarantine capped at %d rows for material_version %s",
                _MAX_QUARANTINE_ROWS,
                material_version_id,
            )
            break
        rows.append(
            {
                "material_version_id": material_version_id,
                "course_id": course_id,
                "unit_kind": "line" if decision.action.value == "strip_lines" else "page",
                "page_number": decision.page_number,
                "ordinal": ordinal,
                "content": decision.content[:4000],
                "occurrences": decision.occurrences,
                "rule_name": decision.rule_name[:64],
                "reason_code": decision.reason.value[:48],
                "action": decision.action.value[:24],
                "rule_score": decision.score,
                "detector_stage": decision.stage[:16],
            }
        )

    if not rows:
        return

    try:
        await db.execute(
            text(
                """
                INSERT INTO material_preprocess_quarantine (
                    material_version_id, course_id, unit_kind, page_number,
                    ordinal, content, occurrences, rule_name, reason_code,
                    action, rule_score, detector_stage
                ) VALUES (
                    :material_version_id, :course_id, :unit_kind, :page_number,
                    :ordinal, :content, :occurrences, :rule_name, :reason_code,
                    :action, :rule_score, :detector_stage
                )
                ON CONFLICT (material_version_id, unit_kind, ordinal)
                DO UPDATE SET
                    page_number = EXCLUDED.page_number,
                    content = EXCLUDED.content,
                    occurrences = EXCLUDED.occurrences,
                    rule_name = EXCLUDED.rule_name,
                    reason_code = EXCLUDED.reason_code,
                    action = EXCLUDED.action,
                    rule_score = EXCLUDED.rule_score,
                    detector_stage = EXCLUDED.detector_stage,
                    updated_at = now()
                """
            ),
            rows,
        )
    except Exception:
        logger.exception(
            "preprocess: failed to persist quarantine rows for material_version %s",
            material_version_id,
        )


__all__ = ["config_from_settings", "run_preprocess_stage"]
