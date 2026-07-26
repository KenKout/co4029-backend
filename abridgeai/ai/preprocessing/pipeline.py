"""Preprocessing cascade orchestrator.

Sequences the tiers over ``PageUnit``s and returns a NEW frozen
``ExtractedContent`` plus a ``PreprocessReport``. Ordering is load-bearing:

1. **normalize** first, so every later comparison runs on clean text (a
   NBSP or a ligature would otherwise defeat the running-mark key match)
2. **emptiness** before running marks, so dropped pages do not skew the
   header/footer frequency denominator
3. **running marks** before role classification, so a stripped running
   header is not mistaken for the page's title line
4. **deck detection** before grouping, since grouping is deck-only
5. **OCR / LLM** last — they are the only paid steps, and by this point the
   candidate set is as small as the deterministic tiers can make it

Page-level outcomes reach the chunker through
``ExtractedContent.metadata["page_roles"]``: ``ai/chunking/_window`` already
receives the whole ``ExtractedContent``, so no new plumbing is needed and
the chunk metadata contract is unchanged for every other source type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from abridgeai.ai.extraction.base import ExtractedContent
from abridgeai.ai.preprocessing.base import (
    ROLE_BODY,
    Action,
    Decision,
    PageFacts,
    PageUnit,
    PreprocessReport,
    ReasonCode,
)
from abridgeai.ai.preprocessing.blankness import classify_emptiness
from abridgeai.ai.preprocessing.deck import assign_topic_groups, detect_deck
from abridgeai.ai.preprocessing.normalize import build_hyphen_vocab, dehyphenate, normalize_text
from abridgeai.ai.preprocessing.page_roles import classify_page_role
from abridgeai.ai.preprocessing.paging import join_pages, split_pages
from abridgeai.ai.preprocessing.running_marks import strip_running_marks

logger = logging.getLogger(__name__)

# Source types that carry page geometry and boilerplate worth filtering.
# Transcripts (audio/video) and images are excluded: a transcript has no
# pages, and an image is already a single OCR'd unit.
PAGED_SOURCE_TYPES = frozenset({"pdf", "pptx"})
# ``code`` and ``xlsx`` are deliberately absent. Whitespace collapsing is the
# whole point of the normalize tier and it is destructive for both: leading
# indentation is semantics in Python, and a spreadsheet's cell layout is
# carried by the extractor's own spacing. Neither format has cover pages,
# running headers or slide decks, so the rest of the cascade has nothing to
# offer them either.
TEXT_SOURCE_TYPES = frozenset({"pdf", "pptx", "docx", "html", "text"})


class PageOcr(Protocol):
    """Renders one page and returns its text, or ``None`` on failure."""

    async def ocr_page(self, page_number: int) -> str | None: ...


class PageAdjudicator(Protocol):
    """Classifies ambiguous pages. Returns ``{page: (role, confidence)}``."""

    async def classify(self, pages: list[tuple[int, str]]) -> dict[int, tuple[str, float]]: ...


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool = True
    normalize: bool = True
    dehyphenation: bool = True
    blankness: bool = True
    running_marks: bool = True
    page_roles: bool = True
    deck_detection: bool = True
    ocr_enabled: bool = True
    # Advisory threshold: exceeding it logs, it does not truncate. See
    # :func:`_run_ocr`.
    ocr_max_pages: int = 30
    llm_adjudication: bool = False
    llm_min_confidence: float = 0.8


async def run_preprocessing(
    content: ExtractedContent,
    *,
    config: PreprocessConfig | None = None,
    ocr: PageOcr | None = None,
    adjudicator: PageAdjudicator | None = None,
    protected_pages: set[int] | None = None,
) -> tuple[ExtractedContent, PreprocessReport]:
    """Run the cascade. Returns ``(new_content, report)``.

    ``ExtractedContent`` is frozen, so a new instance is returned; the
    caller swaps it into the pipeline in place of the extractor's output.
    """
    config = config or PreprocessConfig()
    report = PreprocessReport(enabled=config.enabled)

    if not config.enabled or content.source_type not in TEXT_SOURCE_TYPES:
        report.enabled = False
        return content, report

    report.chars_in = len(content.text)

    facts_by_page = {
        int(raw.get("page_number") or 0): PageFacts.from_dict(raw)
        for raw in (content.metadata.get("pages") or [])
        if isinstance(raw, dict)
    }
    units = split_pages(content.text, facts_by_page)
    report.page_count_in = len([u for u in units if u.marker])

    is_deck = _run_deterministic_tiers(units, content, config, report)

    if config.ocr_enabled and ocr is not None:
        await _run_ocr(units, ocr, report, max_pages=config.ocr_max_pages)

    if config.llm_adjudication and adjudicator is not None:
        await _run_adjudication(units, adjudicator, report, config.llm_min_confidence)

    # Teacher overrides win over every tier, including the LLM. Applied last
    # so nothing downstream can re-drop a page a human explicitly restored.
    if protected_pages:
        _apply_teacher_restores(units, protected_pages, report)

    text = join_pages(units)
    report.chars_out = len(text)
    report.page_count_out = len([u for u in units if u.marker and not u.dropped and u.body.strip()])
    report.pages_dropped = len([u for u in units if u.dropped])
    for unit in units:
        report.decisions.extend(unit.decisions)
        report.role_counts[unit.role] = report.role_counts.get(unit.role, 0) + 1

    metadata = dict(content.metadata)
    metadata["page_roles"] = _page_role_map(units)
    metadata["preprocess"] = report.as_dict()
    if is_deck:
        metadata["deck"] = {"is_deck": True, "score": report.deck_score}

    # Surviving pages only — a dropped page's SourceLocation would dangle.
    kept_pages = {u.page_number for u in units if not u.dropped and u.page_number is not None}
    locations = [
        loc for loc in content.source_locations if loc.page is None or loc.page in kept_pages
    ]

    return (
        replace(content, text=text, metadata=metadata, source_locations=locations),
        report,
    )


def _normalize_units(
    units: list[PageUnit],
    full_text: str,
    config: PreprocessConfig,
    report: PreprocessReport,
) -> None:
    """Tier 1a: unicode folds + conservative de-hyphenation.

    The hyphen vocabulary is built once over the WHOLE document: a compound
    the author writes intact on page 12 must protect the line-broken copy on
    page 4, which a per-page view cannot see.
    """
    vocab = build_hyphen_vocab(full_text)
    for unit in units:
        unit.body = normalize_text(unit.body)
        if config.dehyphenation:
            unit.body, joins = dehyphenate(unit.body, vocab)
            report.hyphen_joins += joins


def _run_deterministic_tiers(
    units: list[PageUnit],
    content: ExtractedContent,
    config: PreprocessConfig,
    report: PreprocessReport,
) -> bool:
    """Tiers 0-2: free, CPU-only, no network. Returns ``is_deck``."""
    if config.normalize:
        _normalize_units(units, content.text, config, report)

    if config.blankness:
        for unit in units:
            classify_emptiness(unit)

    if config.running_marks:
        report.lines_stripped = strip_running_marks(units)

    if config.blankness:
        # Second pass. A diagram slide whose only text was the running header
        # measured 5 words at extraction time — just enough to clear the
        # near-empty threshold and never reach the OCR tier. Now that the
        # boilerplate is gone the page reads as what it is: image-only.
        for unit in units:
            classify_emptiness(unit)

    is_deck = False
    if config.deck_detection and content.source_type in PAGED_SOURCE_TYPES:
        is_deck, score = detect_deck(units, content.metadata)
        report.is_deck, report.deck_score = is_deck, score
        if is_deck:
            assign_topic_groups(units)

    if config.page_roles:
        total = len([u for u in units if u.marker])
        for unit in units:
            classify_page_role(unit, total_pages=total)

    return is_deck


async def _run_ocr(
    units: list[PageUnit],
    ocr: PageOcr,
    report: PreprocessReport,
    *,
    max_pages: int,
) -> None:
    """OCR every image-only page the blankness tier routed here.

    ``max_pages`` no longer truncates. It used to: past the cap the remaining
    pages were dropped with a warning, on the reasoning that a document
    needing that much OCR is a scan and the vision bill should be stopped.
    That trade was wrong in the one direction that matters — the pages routed
    here are by definition the ones with no text layer, so truncating did not
    degrade the document, it *deleted* the back half of it, and nothing
    downstream could tell the difference between "page 40 had no content" and
    "page 40 was over budget". A cost ceiling is not worth a silent hole in
    the middle of a course.

    The value is still logged when exceeded so an unusually OCR-heavy upload
    is visible in the worker log rather than merely expensive.
    """
    targets = [u for u in units if u.needs_ocr and u.page_number is not None]
    if not targets:
        return
    if max_pages and len(targets) > max_pages:
        logger.warning(
            "preprocess: %d pages need OCR (advisory threshold %d). OCR-ing all of "
            "them — this document is probably a scan, so expect the vision spend "
            "to scale with its length.",
            len(targets),
            max_pages,
        )

    for unit in targets:
        page_number = unit.page_number
        if page_number is None:  # pragma: no cover - filtered above
            continue
        try:
            text = await ocr.ocr_page(page_number)
        except Exception:
            # Fail open: a vision outage must never become silent content loss.
            logger.exception("preprocess: OCR failed for page %s", unit.page_number)
            continue
        if not text or not text.strip():
            continue
        unit.body = (unit.body + "\n" + text.strip()).strip()
        unit.needs_ocr = False
        unit.flag("ocr_recovered")
        report.pages_ocr_routed += 1
        unit.record(
            Decision(
                action=Action.ROUTE_OCR,
                reason=ReasonCode.IMAGE_ONLY_NEEDS_OCR,
                rule_name="llm_page_ocr",
                page_number=page_number,
                content=text[:200],
                stage="llm",
            )
        )


# Pages the deterministic tier could not settle: exactly one boilerplate
# signal fired, or the page sits at either end of the document in the word
# band where a cover and a content slide look alike.
_AMBIGUOUS_MIN_WORDS = 12
_AMBIGUOUS_MAX_WORDS = 60
_AMBIGUOUS_HEAD_PAGES = 3
_AMBIGUOUS_TAIL_PAGES = 2
_ADJUDICATION_BATCH = 10

_ADJUDICATED_ROLES = {"body", "front_matter", "summary", "reference", "divider"}


def _is_ambiguous(unit: PageUnit, total_pages: int) -> bool:
    if unit.dropped or unit.role != ROLE_BODY or not unit.page_number:
        return False
    words = len(unit.body.split())
    if not (_AMBIGUOUS_MIN_WORDS <= words <= _AMBIGUOUS_MAX_WORDS):
        return False
    head = unit.page_number <= _AMBIGUOUS_HEAD_PAGES
    tail = unit.page_number > total_pages - _AMBIGUOUS_TAIL_PAGES
    return head or tail


async def _run_adjudication(
    units: list[PageUnit],
    adjudicator: PageAdjudicator,
    report: PreprocessReport,
    min_confidence: float,
) -> None:
    total = len([u for u in units if u.marker])
    candidates = [u for u in units if _is_ambiguous(u, total)]
    if not candidates:
        return

    by_page = {u.page_number: u for u in candidates if u.page_number is not None}
    batch: list[tuple[int, str]] = [(p, u.body) for p, u in by_page.items()]

    for start in range(0, len(batch), _ADJUDICATION_BATCH):
        window = batch[start : start + _ADJUDICATION_BATCH]
        try:
            verdicts = await adjudicator.classify(window)
        except Exception:
            # Fail open to ``body``: an LLM outage must not drop content.
            logger.exception("preprocess: page adjudication failed for batch at %d", start)
            continue
        for page, (role, confidence) in verdicts.items():
            unit = by_page.get(page)
            if unit is None or role not in _ADJUDICATED_ROLES or role == ROLE_BODY:
                continue
            if confidence < min_confidence:
                continue
            unit.role = role
            unit.flag("llm_adjudicated")
            report.llm_adjudicated += 1
            unit.record(
                Decision(
                    action=Action.TAG_ROLE,
                    reason=ReasonCode.LLM_ADJUDICATED,
                    rule_name="llm_page_class",
                    page_number=page,
                    content=unit.body[:200],
                    score=confidence,
                    stage="llm",
                )
            )


def _apply_teacher_restores(
    units: list[PageUnit],
    protected_pages: set[int],
    report: PreprocessReport,
) -> None:
    """Un-drop and un-exclude pages a teacher restored.

    The role tag is reset to ``body`` as well: leaving a restored page tagged
    ``front_matter`` would keep it capped at 25% of the candidate pool by
    ``role_filter``, which is a quieter version of the same suppression the
    teacher just overturned.
    """
    restored = 0
    for unit in units:
        if unit.page_number not in protected_pages:
            continue
        if unit.dropped or unit.retrieval_excluded or unit.role != ROLE_BODY:
            unit.dropped = False
            unit.retrieval_excluded = False
            unit.role = ROLE_BODY
            unit.flag("teacher_restored")
            restored += 1
    if restored:
        report.role_counts["teacher_restored"] = restored


def _page_role_map(units: list[PageUnit]) -> dict[str, dict[str, object]]:
    """Page -> chunk-metadata hints, consumed by ``ai/chunking/_window``.

    Keys are strings because this dict round-trips through JSONB.
    """
    out: dict[str, dict[str, object]] = {}
    for unit in units:
        if unit.page_number is None or unit.dropped:
            continue
        entry: dict[str, object] = {"role": unit.role}
        if unit.retrieval_excluded:
            entry["retrieval_excluded"] = True
        if unit.noise_flags:
            entry["noise_flags"] = list(unit.noise_flags)
        if unit.topic_group_id is not None:
            entry["topic_group_id"] = unit.topic_group_id
        if unit.slide_title:
            entry["slide_title"] = unit.slide_title
        out[str(unit.page_number)] = entry
    return out


__all__ = [
    "PAGED_SOURCE_TYPES",
    "PageAdjudicator",
    "PageOcr",
    "PreprocessConfig",
    "run_preprocessing",
]
