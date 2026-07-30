"""Role-aware retrieval filter (FR-11 port from legacy quiz pipeline).

Caps ``summary`` / ``review`` / ``front_matter`` chunks at ``floor(limit/4)``
and lets ``body`` chunks fill the rest. Ports the legacy
``app/ai/haystack/components/retrievers.py:_split_by_role`` helper which
was lost during the Phase-3 retrieval rewrite.

Why this exists
---------------
Embedding queries that use a topic-style anchor ("data warehouse",
"decision support") tend to pull back the slide deck's title page, the
table of contents, and the "Review questions" recap slide as their top
hits — the cover slide repeats the deck title verbatim, the TOC lists
every chapter heading, and the recap mirrors body content with extra
prefix words. Without this filter, MMR + rerank then dutifully feed
those chunks into the generator prompt, and the LLM writes questions
about *the document's structure* ("In the second excerpt, the outline
begins with the numeral ___") rather than the subject matter.

The cap is applied AFTER pool merge but BEFORE MMR / rerank — every
downstream stage sees a body-priority candidate set. When the corpus
genuinely lacks body chunks (rare for real materials, common for
synthetic test fixtures), summary chunks fall back in to fill the
quota so we still surface ``limit`` total candidates.

This module intentionally has zero retrieval-pipeline imports — it is
a pure list transformation that takes any object exposing
``metadata['content_role']`` (None tolerated) and returns the same
type. Callers in the quiz / interview retrieval stages wire it in.
"""

# Roles that get capped. ``body`` is the only "teachable content" role;
# everything else is administrivia or summary recap.
_DEPRIORITIZED_ROLES: frozenset[str] = frozenset(
    {"summary", "review", "front_matter", "reference", "divider"}
)

# Default cap ratio: keep at most floor(limit / N) deprioritized chunks.
# Legacy used ``limit // 4``; we keep that constant so behaviour matches
# the legacy pipeline for the same ``limit``.
_DEPRIORITIZED_RATIO = 4


def split_by_role[T](
    chunks: list[T],
    limit: int,
    *,
    deprioritized_ratio: int = _DEPRIORITIZED_RATIO,
) -> list[T]:
    """Cap deprioritized-role chunks at ``floor(limit / deprioritized_ratio)``.

    Parameters
    ----------
    chunks
        Pool of candidates. Each item must expose a ``metadata`` attribute
        that is either ``None`` or a dict carrying ``content_role``.
    limit
        Final desired pool size. The function returns at most ``limit``
        items, with body chunks taking priority.
    deprioritized_ratio
        Cap divisor: at most ``floor(limit / deprioritized_ratio)``
        chunks of role summary/review/front_matter survive the filter
        when body chunks are abundant. ``4`` (legacy default) reserves
        25% of the pool for non-body content so ToC/recap chunks aren't
        completely starved when the lesson genuinely uses them.

    Returns
    -------
    list[T]
        Body-priority subset, length ≤ ``limit``. Body chunks come
        first (preserving the input order), then up to the cap of
        deprioritized chunks. When body alone cannot fill the body
        quota, the gap is back-filled with extra deprioritized chunks
        so callers still get ``limit`` total when the pool is large
        enough.

    Notes
    -----
    Input order is preserved within each bucket — the caller is
    responsible for sorting the pool by relevance before passing it in.
    Items whose ``metadata`` is ``None`` or missing ``content_role`` are
    treated as ``body`` (the safe default — don't cap something we
    can't classify).
    """
    if not chunks:
        return []
    if limit <= 0:
        return []
    if deprioritized_ratio < 1:
        # Defensive: ratio must be ≥1 to be meaningful. Treat any
        # bogus value as "no cap" rather than blow up.
        return chunks[:limit]

    body: list[T] = []
    deprioritized: list[T] = []
    for chunk in chunks:
        role = _role_of(chunk)
        if role in _DEPRIORITIZED_ROLES:
            deprioritized.append(chunk)
        else:
            body.append(chunk)

    # Cap deprioritized chunks at floor(limit / ratio), but never reserve more
    # slots than there are deprioritized chunks to fill — otherwise an all-body
    # pool would be capped below ``limit`` (body is the GOOD content and must be
    # allowed to fill the whole pool when nothing needs deprioritizing).
    cap = max(1, limit // deprioritized_ratio)
    deprioritized_taken = deprioritized[:cap]

    # Body fills every slot the (capped) deprioritized chunks did not claim.
    body_taken = body[: limit - len(deprioritized_taken)]

    # Backfill: if body is short, take extra deprioritized beyond the cap so the
    # caller still gets ``limit`` candidates when the pool is large enough.
    extra_quota = limit - len(body_taken) - len(deprioritized_taken)
    if extra_quota > 0:
        deprioritized_taken.extend(
            deprioritized[cap : cap + extra_quota]
        )

    return body_taken + deprioritized_taken


def _role_of(chunk: object) -> str:
    """Read the chunk's content role defensively, defaulting to body.

    Prefers ``metadata['semantic']['content_role']`` (written by the
    chunking enrichment LLM, which reads the slide and knows a recap from
    a definition) over top-level ``metadata['content_role']`` (rule-based
    classifier). The two disagree often enough to matter: on a real
    lecture deck the closing "Summary" slide and the "Review questions"
    slide both come out of the rule classifier as ``body`` while the LLM
    labels them ``summary`` / ``review``. Reading only the top level left
    exactly the two slides this module exists to cap sitting in the pool
    uncapped, which is the failure described in the module docstring.

    This mirrors ``features/quizzes/ai/outline.py::_chunk_role`` — the
    same precedence, so a chunk cannot be ``review`` for quiz coverage
    allocation and ``body`` for retrieval.

    Tolerates dataclass instances (any object with a ``metadata`` attr),
    raw mappings, and missing/None metadata. Returns a lowercase role
    string; an unknown / malformed value is normalised to ``body`` so
    we never accidentally cap a chunk we can't classify.
    """
    md = getattr(chunk, "metadata", None)
    if not isinstance(md, dict):
        return "body"

    semantic = md.get("semantic")
    if isinstance(semantic, dict):
        semantic_role = semantic.get("content_role")
        if isinstance(semantic_role, str) and semantic_role.strip():
            return semantic_role.strip().lower()

    raw = md.get("content_role")
    if not isinstance(raw, str):
        return "body"
    return raw.strip().lower() or "body"


__all__ = ["split_by_role"]
