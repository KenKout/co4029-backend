"""Phase 3.1 — catch protected content the AI reworded instead of quoted.

The lexical output guard compares strings. A paraphrase keeps the meaning and
throws away the wording, so it slides under every string threshold: measured
against the corpus, the reworded model answer scores **0.295** similarity to the
secret it leaks, while a genuinely benign reply about the same topic scores
**0.619**. No lexical cutoff separates those two, which is why this stage exists
and why it compares *meaning* via embeddings.

Cost control, and why the grey zone is a pre-filter rather than a decision:

``grey_zone_leak_candidates`` in ``security_logic`` answers only "are these texts
about the same subject?". That is cheap and runs on every guarded turn. This
module runs solely for the phrases it returns, so an ordinary interview turn pays
nothing extra. When it does run, one embedding call covers the proposal plus every
candidate secret in a single batch.

Failure is deliberately non-blocking. If the embedding call fails, times out, or
returns a malformed payload, the verdict is "no semantic leak" and the lexical
result stands. Refusing a student's turn because an internal service was
unavailable would convert an infrastructure fault into an assessment penalty, and
this stage is a *second* line of defence — exact and fuzzy leaks are already
blocked without it.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.core.exceptions import AppError
from abridgeai.features.interviews.orchestrator.security_logic import (
    FUZZY_LEAK_THRESHOLD,
    WHITESPACE_RE,
    normalize_input,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.orchestrator.security import ProtectedContent

logger = logging.getLogger(__name__)

# Cosine similarity above which a proposal is RECORDED as a probable paraphrase.
#
# Measured on the live embedding model over 6 hand-written paraphrase leaks and 20
# legitimate interviewer turns about the same subject. The two populations
# OVERLAP: benign runs 0.034-0.588 (top: "let me repeat the question: how does a
# transaction stay atomic across a crash?") and leaks run 0.496-0.816. There is no
# cutoff that catches every paraphrase without also refusing a real interview
# question, which is why this stage records rather than blocks — see the comment
# at the call site in ``services/security.py``.
#
# 0.60 sits just above the benign ceiling: it keeps the audit trail dominated by
# genuine paraphrases (4 of 6 in the sample) instead of drowning it in normal
# turns. Raising it loses recall; lowering it makes the signal useless. Any change
# needs fresh measurements, not intuition, so this is not a runtime setting.
SEMANTIC_LEAK_THRESHOLD = 0.60

# A hard ceiling on candidates embedded in one call. The grey-zone filter rarely
# returns more than a couple; this only bounds a pathological config with hundreds
# of near-identical outcomes.
_MAX_CANDIDATES = 8

# Phase 3.1 grey zone floor. Measured, not guessed: the paraphrased model answer
# scores only 0.295 lexical similarity to its secret while a benign reply scores
# 0.619, so lexical similarity cannot separate them at any threshold. It can only
# say "these texts are about the same subject", which is all this pre-filter is
# for — the embedding comparison decides.
_SEMANTIC_GREY_FLOOR = 0.15
# Below this length neither side carries enough content to compare meaningfully.
_SEMANTIC_MIN_LEN = 40
# Absolute floor alongside the ratio: one shared word is a coincidence.
#
# Two is low, and measurement is why. A good paraphrase REPLACES the vocabulary, so
# the reworded model answer shares only "transaction" and "process" (2 of 11
# content words) with the secret it leaks — while the reworded rubric shares 8 of
# 14. A floor of 3 would have filtered out the very case this phase targets. The
# cost of being generous here is bounded: a candidate only buys a place in one
# batched embedding call, and the embedding decides.
_MIN_SHARED_CONTENT_WORDS = 2

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Words too common to indicate a shared subject. Small on purpose — this only has
# to stop "the/that/is" from counting as evidence, not to be a real stoplist.
_STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
        "in", "into", "is", "it", "its", "me", "must", "my", "no", "not", "of", "on", "or",
        "should", "so", "such", "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "to", "too", "very", "was", "were", "what", "when", "where",
        "which", "who", "will", "with", "would", "you", "your", "về", "của", "cho", "là",
        "các", "những", "một", "và", "hoặc", "nếu", "khi", "nào", "bạn", "tôi", "em", "mình",
        "không", "có", "được", "ở", "trong", "trên", "với", "từ", "đến", "thì", "mà", "rồi",
        "đã", "đang", "sẽ", "phải", "cần", "rất", "quá", "cũng", "chỉ",
    )
)


@dataclass(frozen=True)
class SemanticLeakAssessment:
    """Result of the semantic comparison. Never carries the matched text."""

    # "This crossed the recording threshold", NOT "refuse this turn". The caller
    # deliberately does not enforce on it; see SEMANTIC_LEAK_THRESHOLD.
    blocked: bool
    protected_content_category: str | None = None
    similarity: float | None = None
    # True when the check could not run (call failed / bad payload). Lets the
    # caller distinguish "no leak" from "not actually verified".
    degraded: bool = False


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


async def assess_semantic_leakage(
    db: AsyncSession,
    *,
    proposed: str,
    candidates: list[ProtectedContent],
    organization_id: UUID | None = None,
    pipeline_run_id: UUID | None = None,
    client: EmbeddingClient | None = None,
) -> SemanticLeakAssessment:
    """Compare ``proposed`` against ``candidates`` by meaning.

    ``candidates`` comes from ``grey_zone_leak_candidates``; an empty list short
    circuits without any model call.
    """
    if not candidates:
        return SemanticLeakAssessment(blocked=False)

    considered = candidates[:_MAX_CANDIDATES]
    embedder = client or EmbeddingClient()
    texts = [proposed, *(item.text for item in considered)]

    try:
        vectors = await embedder.embed(
            texts,
            db=db,
            organization_id=organization_id,
            pipeline_run_id=pipeline_run_id,
        )
    except (AppError, OSError, ValueError):
        # Non-blocking by design: an unavailable embedding service must not cost
        # a student their turn. Logged without the texts — one of them is secret.
        logger.warning(
            "interview.security.semantic_leak_check_failed",
            extra={"candidate_count": len(considered)},
            exc_info=True,
        )
        return SemanticLeakAssessment(blocked=False, degraded=True)

    if len(vectors) != len(texts):
        logger.warning(
            "interview.security.semantic_leak_vector_count_mismatch",
            extra={"expected": len(texts), "received": len(vectors)},
        )
        return SemanticLeakAssessment(blocked=False, degraded=True)

    proposed_vector = vectors[0]
    best_similarity = 0.0
    best_category: str | None = None
    for item, vector in zip(considered, vectors[1:], strict=True):
        similarity = _cosine(proposed_vector, vector)
        if similarity > best_similarity:
            best_similarity = similarity
            best_category = item.category

    if best_similarity >= SEMANTIC_LEAK_THRESHOLD:
        return SemanticLeakAssessment(
            blocked=True,
            protected_content_category=best_category,
            similarity=best_similarity,
        )
    return SemanticLeakAssessment(blocked=False, similarity=best_similarity)


def grey_zone_leak_candidates(
    proposed: str,
    protected_content: Iterable[ProtectedContent],
) -> list[ProtectedContent]:
    """Protected phrases that are *near* the fuzzy threshold but under it.

    Phase 3.1. A paraphrase keeps the meaning and loses the wording, so it lands
    below ``FUZZY_LEAK_THRESHOLD`` and ``assess_output_leakage`` clears it. This
    reports the phrases worth a semantic (embedding) second look, so the caller
    pays for that comparison only when lexical similarity already suggests the
    proposal is circling one specific secret.

    Empty result means "no reason to spend an embedding call" — the common case.
    """
    supplied = list(protected_content)
    proposed_norm = normalize_input(proposed)
    # Subtract the question the interviewer is ALLOWED to say, exactly as
    # ``assess_output_leakage`` does. Without this the guard measures the question
    # text against the secret answer and finds them related — they are, by
    # construction — so asking the assigned question would look like a leak.
    for allowed in supplied:
        if allowed.category == "allowed_question_text":
            allowed_norm = normalize_input(allowed.text)
            if allowed_norm:
                proposed_norm = proposed_norm.replace(allowed_norm, " ")
    proposed_norm = WHITESPACE_RE.sub(" ", proposed_norm).strip()
    if len(proposed_norm) < _SEMANTIC_MIN_LEN:
        return []
    proposed_tokens = {t for t in _TOKEN_RE.findall(proposed_norm) if t not in _STOPWORDS}
    candidates: list[ProtectedContent] = []
    matcher = SequenceMatcher(None, "", proposed_norm, autojunk=False)
    for item in supplied:
        if item.category in {"allowed_question_text", "internal_prompt_marker"}:
            continue
        secret_norm = normalize_input(item.text)
        if len(secret_norm) < _SEMANTIC_MIN_LEN:
            continue
        # Content-word overlap decides, not character similarity. SequenceMatcher
        # rates "thank you, that concludes the interview" at 0.338 against a secret
        # about write-ahead logging purely on shared letters, which would buy an
        # embedding call on almost every turn. Sharing the SUBJECT means sharing
        # uncommon words.
        secret_tokens = {t for t in _TOKEN_RE.findall(secret_norm) if t not in _STOPWORDS}
        if not secret_tokens:
            continue
        shared = len(proposed_tokens & secret_tokens)
        if shared < _MIN_SHARED_CONTENT_WORDS:
            continue
        if shared / len(secret_tokens) < _SEMANTIC_GREY_FLOOR:
            continue
        # Anything at or above the lexical threshold is already blocked outright by
        # ``assess_output_leakage``; this stage exists for what falls short of it.
        matcher.set_seq1(secret_norm)
        if matcher.ratio() >= FUZZY_LEAK_THRESHOLD:
            continue
        candidates.append(item)
    return candidates


__all__ = [
    "SEMANTIC_LEAK_THRESHOLD",
    "SemanticLeakAssessment",
    "assess_semantic_leakage",
    "grey_zone_leak_candidates",
]
