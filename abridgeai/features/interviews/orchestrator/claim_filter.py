"""Deterministic boundary between the quarantined extractor and the matcher.

This module is the security mechanism of the analysis split. Everything else is
plumbing: the extractor (:mod:`orchestrator.extraction`) sees the raw answer but
holds nothing worth stealing, the matcher (:mod:`orchestrator.matching`) holds
the rubric but never sees raw text, and *this* is the only thing that crosses
between them. It is plain Python with no LLM call, which is the whole point —
the isolation property must not depend on a model behaving.

What it does
------------
Every claim the extractor produced is screened before the matcher can see it:

1. Re-enforce the schema caps. The parser already truncates, but a cap that is
   only enforced where the data is parsed is a cap that moves when the parser
   moves.
2. Run the existing rules engine (:func:`assess_by_rules`) over each claim.
   A claim that trips an injection pattern is dropped, not sanitised —
   there is no reliable way to neutralise an instruction in-place, and the
   matcher does not need that claim to do its job.
3. Report how many were dropped, so shadow mode can measure the boundary.

What it does not do
-------------------
It cannot certify that a surviving claim is free of adversarial intent. The
rules engine has known gaps (tracked in ``tests/unit/interviews/security/``),
and a paraphrased injection that reads like subject matter will pass. The claim
that holds is narrower and worth stating precisely: **raw student text never
reaches the rubric-bearing prompt**, and what does reach it is bounded in
length and count and has been screened by the same rules that guard the turn
itself. That is defence in depth over the classifier, not a replacement for it.

Note the asymmetry with the turn-level guard: there, a detection blocks the
whole turn and the student is told. Here a detection silently drops one claim,
because by this point the turn has already been admitted as academic — the
student gets a slightly worse analysis, not an error.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.extraction import (
    MAX_CLAIM_CHARS,
    MAX_CLAIMS,
    AnswerClaims,
    Claim,
)
from abridgeai.features.interviews.orchestrator.security_logic import assess_by_rules


def filter_claims(claims: AnswerClaims) -> AnswerClaims:
    """Screen extracted claims before they may travel to the rubric stage.

    Returns a new :class:`AnswerClaims` with unsafe claims removed and
    ``dropped_claim_count`` set. Never raises: a screening failure must not
    cost the student their turn, and the safe direction on error is to drop
    the claim rather than forward it.
    """
    kept: list[Claim] = []
    dropped = 0
    for claim in claims.claims[:MAX_CLAIMS]:
        if _is_safe(claim.text):
            kept.append(
                Claim(text=claim.text[:MAX_CLAIM_CHARS], kind=claim.kind)
                if len(claim.text) > MAX_CLAIM_CHARS
                else claim
            )
        else:
            dropped += 1

    claims.claims = kept
    claims.dropped_claim_count = dropped
    return claims


def _is_safe(text: str) -> bool:
    """True when the rules engine sees nothing adversarial in ``text``.

    Uses the turn-level rules with no ``last_category``: cross-turn escalation
    state belongs to the conversation, not to a fragment of one answer, and
    feeding it here would let an earlier flagged turn suppress this turn's
    legitimate claims.
    """
    if not text.strip():
        return False
    try:
        return not assess_by_rules(text).detected
    except Exception:  # noqa: BLE001 -- screening must fail closed, never raise
        return False


__all__ = ["filter_claims"]
