"""Repeated header/footer detection and page-number stripping.

A footer repeated on all 60 slides of a lecture deck is currently emitted
60 times: ``pdf.py`` skips a page only when it is 100% empty and never
looks across pages, so the university name, the course code banner and the
slide number all become embedded content competing with the actual lecture.

The detector is a port of Lin, X., *"Header and footer extraction by
page-association"*, SPIE DRR-X vol. 5010 (2003), as implemented by ISPRAS
dedoc, with its constants preserved: 4 candidate slots at each end of the
page, position weights, a page-association step of 2 (so alternating
recto/verso running heads still associate), digit-and-roman normalization
before comparison, a 0.5 similarity gate and a popularity gate on top.

Two additions over dedoc, both learned from the failure modes of the naive
version:

* **Geometric gate.** A candidate must sit in the page's margin band.
  pymupdf4llm moved its own top-level ``margins`` default from ``(0,50,0,50)``
  to ``0`` precisely because a blind band ate real content — so the band
  narrows candidates, it never authorizes a delete on its own.
* **Page 1 is excluded** from both the frequency denominator and from
  stripping. A cover page has no running head and poisons the frequency
  test, and the first-page footer is usually bibliographic front matter.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from abridgeai.ai.preprocessing.base import Action, Decision, PageUnit, ReasonCode

_SLOT_COUNT = 4
# Lin 2003 position weights: the outermost lines are the most reliable
# running-mark positions, confidence decays inward.
_HEAD_WEIGHTS = (1.0, 1.0, 0.85, 0.75)
_TAIL_WEIGHTS = (0.75, 0.85, 1.0, 1.0)

_SIMILARITY_GATE = 0.5
_POPULARITY_STEP2 = 0.4
_POPULARITY_STEP1 = 0.7
_MIN_PAGES = 3

# Margin band: 8% of page height, floored at pymupdf4llm's legacy 50pt.
# Scales for a 540pt landscape deck (43pt -> floored to 50) and A4 (67pt).
_BAND_RATIO = 0.08
_BAND_FLOOR = 50.0

_ROMAN_RE = re.compile(r"\b[IVXLCDM]+\.?\b|\b[ivxlcdm]+\.?\b")
_DIGITS_RE = re.compile(r"\d+")
_AT_COLLAPSE_RE = re.compile(r"@+")
_WS_RE = re.compile(r"\s+")

_PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$"),
    re.compile(
        r"^\s*(page|p\.|pg\.?|trang)\s*\d{1,4}(\s*(of|/|\||-)\s*\d{1,4})?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*\d{1,4}\s*(of|/)\s*\d{1,4}\s*$", re.IGNORECASE),
)
_ROMAN_ONLY_RE = re.compile(r"^\s*[ivxlcdm]{1,7}\s*$", re.IGNORECASE)

# Inverted jusText: a line that reads like a sentence is content, however
# often it repeats. jusText classifies by stopword density precisely because
# boilerplate (cover pages, TOCs, instructor blocks) has near-zero density.
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by with
    as is are was were be been being it its we you they he she not no do does did have has had
    will would can could should may might must về của và là các một những cho trong với khi
    được không này đó thì mà nếu như để từ tại theo trên dưới""".split()  # noqa: SIM905 - a 60-word literal list is far less readable
)
_SENTENCE_MIN_WORDS = 8
_SENTENCE_MIN_STOPWORDS = 2


def normalize_key(line: str) -> str:
    """Collapse digit/roman variants so ``Page 3 of 12`` == ``Page 4 of 12``."""
    key = _ROMAN_RE.sub("@", line)
    key = _DIGITS_RE.sub("@", key)
    key = _AT_COLLAPSE_RE.sub("@", key)
    return _WS_RE.sub(" ", key).strip().lower()


def is_sentence_like(line: str) -> bool:
    """True when the line reads as prose and must never be dropped."""
    words = line.split()
    if len(words) < _SENTENCE_MIN_WORDS:
        return False
    stops = sum(1 for w in words if w.strip(".,;:!?()[]").lower() in _STOPWORDS)
    return stops >= _SENTENCE_MIN_STOPWORDS


def is_page_number(line: str, *, page_has_other_content: bool) -> bool:
    """Match bare page numbers, ``Page 3 of 12``, ``Trang 5``, roman numerals.

    ``page_has_other_content`` gates the bare-roman rule: a slide whose only
    content is the literal ``IV`` is a section divider, not a folio.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if any(p.match(stripped) for p in _PAGE_NUMBER_PATTERNS):
        return True
    return bool(page_has_other_content and _ROMAN_ONLY_RE.match(stripped))


def _band(unit: PageUnit) -> float:
    if unit.facts is None or not unit.facts.height:
        return 0.0
    return max(_BAND_RATIO * unit.facts.height, _BAND_FLOOR)


def _in_margin_band(unit: PageUnit, line_text: str) -> bool:
    """True when the line sits in the top/bottom band, or geometry is absent.

    Without ``facts`` (non-PDF sources) there is no geometry to gate on, so
    the slot position alone carries the decision — matching dedoc, which is
    text-only.
    """
    band = _band(unit)
    if band <= 0 or unit.facts is None or not unit.facts.lines:
        return True
    height = unit.facts.height
    target = line_text.strip()
    for ln in unit.facts.lines:
        if ln.text.strip() == target:
            return ln.y0 <= band or ln.y1 >= (height - band)
    return True


def _slots(body: str) -> list[tuple[int, str]]:
    """Return ``(slot_index, line)`` for the first and last 4 non-empty lines.

    Slot indices are ``0..3`` from the top and ``-4..-1`` from the bottom.
    """
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return []
    out: list[tuple[int, str]] = []
    for i, line in enumerate(lines[:_SLOT_COUNT]):
        out.append((i, line))
    # Tail slots are filled even when the page has fewer than 8 lines, so a
    # line may occupy both a head and a tail slot. dedoc assumes dense prose
    # pages; a slide holds 2-3 lines, which put its footer in a HEAD slot and
    # left it unassociated with the same footer sitting at slot -1 on every
    # dense page. Overlapping the bands just gives a candidate two chances to
    # associate — the popularity gate still decides whether it dies.
    tail = lines[-_SLOT_COUNT :]
    for i, line in enumerate(tail):
        out.append((i - len(tail), line))
    return out


def _weight(slot: int) -> float:
    return _HEAD_WEIGHTS[slot] if slot >= 0 else _TAIL_WEIGHTS[slot + _SLOT_COUNT]


def strip_running_marks(units: list[PageUnit]) -> int:
    """Strip running headers/footers and page numbers. Returns lines removed.

    Operates on ``unit.body`` only — markers are held separately by
    ``paging.PageUnit`` and are structurally out of reach.
    """
    live = [u for u in units if not u.dropped]
    if len(live) < _MIN_PAGES:
        return _strip_page_numbers_only(live)

    # Page 1 is excluded from the association test but still gets bare
    # page-number stripping below.
    body_pages = [u for u in live if u.page_number != 1]
    if len(body_pages) < _MIN_PAGES:
        return _strip_page_numbers_only(live)

    step = 2 if len(body_pages) > 5 else 1
    popularity_gate = _POPULARITY_STEP2 if step == 2 else _POPULARITY_STEP1

    slot_maps: list[dict[int, str]] = [dict(_slots(u.body)) for u in body_pages]

    # Lin 2003 page association: a slot is a running-mark slot when the
    # weighted mean similarity between page i and page i+step clears 0.5.
    running_slots: set[int] = set()
    for slot in [*range(_SLOT_COUNT), *range(-_SLOT_COUNT, 0)]:
        pairs = 0
        total = 0.0
        for i in range(len(slot_maps) - step):
            a, b = slot_maps[i].get(slot), slot_maps[i + step].get(slot)
            if a is None or b is None:
                continue
            total += SequenceMatcher(None, normalize_key(a), normalize_key(b)).ratio()
            pairs += 1
        if pairs and (total / pairs) * _weight(slot) > _SIMILARITY_GATE:
            running_slots.add(slot)

    # Popularity gate: how many pages carry this exact normalized key in a
    # running slot. One-off lines that happen to look similar do not survive.
    key_pages: dict[str, set[int]] = {}
    for idx, slot_map in enumerate(slot_maps):
        for slot, line in slot_map.items():
            if slot not in running_slots:
                continue
            key_pages.setdefault(normalize_key(line), set()).add(idx)

    doomed = {
        key
        for key, pages in key_pages.items()
        if key and len(pages) / len(body_pages) > popularity_gate
    }

    removed = 0
    for unit, slot_map in zip(body_pages, slot_maps, strict=True):
        targets = {
            line
            for slot, line in slot_map.items()
            if slot in running_slots
            and normalize_key(line) in doomed
            and not is_sentence_like(line)
            and _in_margin_band(unit, line)
        }
        if targets:
            removed += _remove_lines(
                unit,
                targets,
                reason=ReasonCode.RUNNING_FOOTER,
                rule_name="running_mark_lin2003",
                score=len(key_pages.get(normalize_key(next(iter(targets))), ())) / len(body_pages),
            )

    removed += _strip_page_numbers_only(live)
    return removed


def _strip_page_numbers_only(units: list[PageUnit]) -> int:
    removed = 0
    for unit in units:
        lines = [ln for ln in unit.body.splitlines() if ln.strip()]
        if not lines:
            continue
        # Only the outer slots can hold a folio; a bare number mid-page is
        # far more likely to be a list item or an answer key.
        candidates = {ln for _, ln in _slots(unit.body)}
        has_other = len(lines) > 1
        targets = {
            ln
            for ln in candidates
            if is_page_number(ln, page_has_other_content=has_other)
            and _in_margin_band(unit, ln)
        }
        if targets:
            removed += _remove_lines(
                unit,
                targets,
                reason=ReasonCode.PAGE_NUMBER,
                rule_name="page_number",
            )
    return removed


def _remove_lines(
    unit: PageUnit,
    targets: set[str],
    *,
    reason: ReasonCode,
    rule_name: str,
    score: float | None = None,
) -> int:
    kept: list[str] = []
    removed = 0
    for line in unit.body.splitlines():
        if line in targets:
            removed += 1
            continue
        kept.append(line)
    if not removed:
        return 0
    unit.body = "\n".join(kept).strip()
    unit.record(
        Decision(
            action=Action.STRIP_LINES,
            reason=reason,
            rule_name=rule_name,
            page_number=unit.page_number,
            content=" | ".join(sorted(targets))[:300],
            score=score,
            stage="statistical",
            occurrences=removed,
        )
    )
    return removed


__all__ = [
    "is_page_number",
    "is_sentence_like",
    "normalize_key",
    "strip_running_marks",
]
