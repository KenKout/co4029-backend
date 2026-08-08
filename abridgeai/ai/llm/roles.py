"""Role registry, tier mapping, and per-call binding resolution.

A "role" is what a piece of the pipeline does (e.g. extracting concepts from a
chunk, generating a quiz question). A "tier" is the size class of the model
(``small`` / ``standard`` / ``large``). Roles map to tiers in code; tiers map
to concrete model names via env-driven settings.

The ``binding_for(role, settings)`` resolver returns a ``ModelBinding`` that
the gateway / embedding client uses to make one upstream call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from abridgeai.core.config import Settings


class LLMRole(str, Enum):  # noqa: UP042 - StrEnum changes value coercion; preserve verbatim per plan §3051
    """The logical roles AI calls can play in our pipelines.

    Stored verbatim in ``ai_model_calls.role`` for cost attribution.
    Per Reconciliation §B1 values are APPEND-ONLY — never reorder.
    """

    EXTRACTION = "extraction"
    ENRICHMENT = "enrichment"
    IDEATION = "ideation"
    GENERATION = "generation"
    VALIDATION = "validation"
    EMBEDDING = "embedding"
    CHUNKING_ENRICHMENT = "chunking_enrichment"
    KG_EXTRACTION = "kg_extraction"
    STT = "stt"
    VISION = "vision"
    INTERVIEW_GENERATION = "interview_generation"
    INTERVIEW_VALIDATION = "interview_validation"
    INTERVIEW_FOLLOWUP = "interview_followup"
    INTERVIEW_EVALUATION = "interview_evaluation"
    GAP_REPORT_GENERATION = "gap_report_generation"
    INTERVIEW_INTENT = "interview_intent"
    INTERVIEW_ANALYSIS = "interview_analysis"
    INTERVIEW_SECURITY = "interview_security"
    INTERVIEW_PERSONA_ADHERENCE = "interview_persona_adherence"
    # Offline post-hoc quality judging (contingency / leading questions). Never
    # runs in a live session — only from the quality CLI — but needs its own role
    # so its spend is attributable and never mixed into live interview costs.
    INTERVIEW_QUALITY_JUDGE = "interview_quality_judge"
    # Question-bank duplicate detection. Unlike the quality judge above, this DOES
    # run with someone waiting (a teacher saving or generating questions), so it
    # belongs in INTERACTIVE_LLM_ROLES for the tighter timeout.
    INTERVIEW_DEDUP = "interview_dedup"
    # Ingestion preprocessing. PAGE_OCR reads back an image-only PDF page via a
    # vision model; PAGE_CLASSIFICATION adjudicates the narrow band of pages the
    # deterministic boilerplate rules could not settle. Both are
    # batch/background — nobody is waiting.
    #
    # PAGE_OCR sits on STANDARD, not small. The claim that "OCR-ing a slide is
    # not a reasoning task" holds for a page of prose and fails for the pages
    # that actually get routed here, which are by definition the ones with no
    # text layer: decision matrices, framework diagrams, case-study columns. On
    # the small tier those came back with mangled tokens (``r:ontrol``,
    # ``assura\ce``) and, worse, with table cells read in the wrong order — one
    # 3x3 matrix had an entry migrate into the wrong cell, which does not look
    # like an error downstream, it looks like course content. Reading a grid
    # and keeping row/column association IS a reasoning task, so this trades
    # cost against silent corruption of the densest pages.
    #
    # Note the spend is NOT bounded per document: ``preprocess_ocr_max_pages``
    # is advisory since truncating it meant dropping pages. A long scan will
    # cost proportionally. ``LLM_MODEL_PAGE_OCR`` overrides the tier per
    # deployment if that trade needs revisiting.
    PAGE_OCR = "page_ocr"
    PAGE_CLASSIFICATION = "page_classification"
    # Quarantined answer extraction. Sees a student's raw answer and the current
    # question and NOTHING else — no rubric, no outcomes, no model answer — so
    # that the rubric-bearing analysis call never has to hold untrusted text.
    # Interactive (the student is waiting) and small tier: paraphrasing a turn
    # into a handful of bounded claims is not a reasoning task, and it replaces
    # the intent call rather than adding to the per-turn budget.
    INTERVIEW_EXTRACTION = "interview_extraction"
    # Fast sufficiency probe. The ONLY analysis call left on the blocking turn
    # path: it answers one question — did this answer move the current outcome
    # toward sufficiency — in a ~30-token verdict, and the full evidence
    # extraction (INTERVIEW_ANALYSIS) is reconciled in afterwards off the turn
    # path. Interactive (the candidate is waiting) and SMALL tier, deliberately:
    # on this deployment the standard model spends 700-1000 hidden reasoning
    # tokens per call, and a one-bit judgement does not need them.
    INTERVIEW_SUFFICIENCY = "interview_sufficiency"
    # Chunk boundary decision (chunking Stage B'). Reads a list of window
    # digests — never the full text — and returns which consecutive windows
    # form one chunk. Batch/background. Small tier: judging "does this slide
    # continue the previous one" from a title and an opening line is a
    # comprehension task the lite models handle, and the volume is one call per
    # multi-window topic group, not one per window.
    CHUNK_BOUNDARY = "chunk_boundary"


# Default role -> tier mapping. Embedding intentionally excluded — it has its
# own EMBEDDING_* config and never participates in the LLM tier system.
# STT and VISION map to "standard" since hosted Whisper / multimodal models
# are not partitioned into our small/large tiers; per-role overrides
# (``llm_model_stt`` / ``llm_model_vision``) carry the concrete model name.
ROLE_TO_TIER: dict[LLMRole, Literal["small", "standard", "large"]] = {
    LLMRole.EXTRACTION: "small",
    LLMRole.ENRICHMENT: "small",
    LLMRole.IDEATION: "standard",
    LLMRole.GENERATION: "large",
    LLMRole.VALIDATION: "standard",
    LLMRole.CHUNKING_ENRICHMENT: "small",
    LLMRole.KG_EXTRACTION: "small",
    LLMRole.STT: "standard",
    LLMRole.VISION: "standard",
    LLMRole.INTERVIEW_GENERATION: "large",
    LLMRole.INTERVIEW_VALIDATION: "standard",
    LLMRole.INTERVIEW_FOLLOWUP: "small",
    LLMRole.INTERVIEW_EVALUATION: "large",
    LLMRole.GAP_REPORT_GENERATION: "large",
    # Intent is a fast, latency-sensitive classification (student is waiting) —
    # small tier. Answer analysis needs more reasoning (rubric/evidence) but is
    # still a runtime call, so standard tier balances quality vs latency.
    LLMRole.INTERVIEW_INTENT: "small",
    LLMRole.INTERVIEW_ANALYSIS: "standard",
    LLMRole.INTERVIEW_SECURITY: "small",
    # Persona-adherence judge runs OFFLINE (post-session, over a stored
    # transcript) — no student is waiting, so it is deliberately NOT in
    # INTERACTIVE_LLM_ROLES. Standard tier: it needs to reason over a whole
    # transcript but is a diagnostic, not a scoring gate.
    LLMRole.INTERVIEW_PERSONA_ADHERENCE: "standard",
    # Post-hoc quality judges (contingency / leading questions) are offline
    # diagnostics like persona adherence — nobody is waiting, so they stay out of
    # INTERACTIVE_LLM_ROLES. Standard tier: judging one Q&A exchange needs real
    # reading comprehension, but these never gate a grade.
    LLMRole.INTERVIEW_QUALITY_JUDGE: "standard",
    # A same-or-different judgement on two short texts; "standard" is ample and
    # keeps a per-question check affordable during bulk generation.
    LLMRole.INTERVIEW_DEDUP: "standard",
    # Ingestion preprocessing — see the role docstrings. PAGE_OCR is standard
    # tier because the pages it sees are the diagram/matrix ones a text
    # extractor could not read at all; PAGE_CLASSIFICATION stays small because
    # labelling a cover page really is a lookup.
    LLMRole.PAGE_OCR: "standard",
    LLMRole.PAGE_CLASSIFICATION: "small",
    # Quarantined extraction — see the role docstring. Small tier: it subsumes
    # the intent call, which was already small, and holds no protected content.
    LLMRole.INTERVIEW_EXTRACTION: "small",
    # Fast sufficiency probe — see the role docstring. Small tier is the whole
    # point: it exists to get a one-bit coverage signal off the reasoning-token
    # path so the candidate is not waiting on the full evidence extraction.
    LLMRole.INTERVIEW_SUFFICIENCY: "small",
    # Boundary decision over window digests — see the role docstring.
    LLMRole.CHUNK_BOUNDARY: "small",
}


# Roles that run in an interactive pipeline stage where a user is actively
# waiting on a spinner (quiz + interview ideation/generation/validation, plus
# the latency-sensitive interview runtime roles). These use the tighter
# ``llm_interactive_timeout_seconds`` instead of the 600s batch timeout, so a
# stalled LAN endpoint can't hang the worker (and the user) for ten minutes.
# Batch/background roles (extraction, enrichment, kg_extraction) intentionally
# keep the long ``llm_timeout_seconds``.
INTERACTIVE_LLM_ROLES: frozenset[LLMRole] = frozenset(
    {
        LLMRole.IDEATION,
        LLMRole.GENERATION,
        LLMRole.VALIDATION,
        LLMRole.INTERVIEW_GENERATION,
        LLMRole.INTERVIEW_VALIDATION,
        LLMRole.INTERVIEW_DEDUP,
        LLMRole.INTERVIEW_FOLLOWUP,
        LLMRole.INTERVIEW_INTENT,
        LLMRole.INTERVIEW_ANALYSIS,
        LLMRole.INTERVIEW_SECURITY,
        LLMRole.INTERVIEW_EXTRACTION,
        LLMRole.INTERVIEW_SUFFICIENCY,
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    """Inline HTTP-429 retry policy for a single call.

    Resolved from runtime settings (``ai.rate_limit_*``) with env/default
    fallback, so an admin can tune 429 backoff without a deploy. Only HTTP 429
    is retried inline; every other failure propagates to the job-level retry.
    """

    max_attempts: int
    base_delay_s: float
    max_delay_s: float


# Code defaults, kept in sync with the ``ai.rate_limit_*`` registry specs.
# Used when a caller resolves no override (e.g. scripts constructing a binding
# straight from Settings without a db session).
DEFAULT_RETRY_POLICY = RetryPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=30.0)


@dataclass(frozen=True)
class ModelBinding:
    """One concrete (endpoint, key, model) tuple resolved for a single call.

    Frozen so a binding can be safely shared across a request.
    """

    role: LLMRole
    tier: str | None
    base_url: str
    api_key: str
    model: str
    extra_headers: dict[str, str]
    timeout_s: float
    retry: RetryPolicy = DEFAULT_RETRY_POLICY


def binding_for(
    role: LLMRole,
    settings: Settings,
    *,
    timeout_override_s: float | None = None,
    retry_override: RetryPolicy | None = None,
) -> ModelBinding:
    """Resolve a ``ModelBinding`` for ``role`` against ``settings``.

    Precedence for chat-completion roles:
      1. Per-role override ``LLM_MODEL_<ROLE>`` (e.g. ``LLM_MODEL_GENERATION``)
      2. Tier model ``LLM_MODEL_<TIER>`` where tier comes from ``ROLE_TO_TIER``
      3. Default-tier model ``LLM_MODEL_<settings.llm_default_tier>``

    Embedding bypasses the tier system and reads ``EMBEDDING_*`` settings.

    ``timeout_override_s`` / ``retry_override`` let a caller that has resolved
    the DB-backed ``ai.*`` runtime settings (org-aware) inject them. When
    ``None`` the binding falls back to the env-based ``Settings`` timeout and
    the code-default retry policy, so callers without a db session (scripts,
    startup probes) keep working unchanged.

    Raises:
        ConfigError: when the required API key is unset.
    """
    from abridgeai.ai.llm.errors import ConfigError

    retry = retry_override or DEFAULT_RETRY_POLICY

    if role is LLMRole.EMBEDDING:
        if not settings.embedding_api_key:
            raise ConfigError("EMBEDDING_API_KEY is required to call the embedding endpoint")
        return ModelBinding(
            role=role,
            tier=None,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            extra_headers=_resolve_extra_headers(settings),
            timeout_s=(
                timeout_override_s
                if timeout_override_s is not None
                else settings.embedding_timeout_seconds
            ),
            retry=retry,
        )

    if not settings.llm_api_key:
        raise ConfigError("LLM_API_KEY is required to call the chat-completions endpoint")

    per_role_override = getattr(settings, f"llm_model_{role.value}", None)
    tier = ROLE_TO_TIER.get(role, settings.llm_default_tier)
    tier_model = getattr(settings, f"llm_model_{tier}")
    model = per_role_override or tier_model

    # Interactive stages (user waiting on a spinner) get the tighter timeout so
    # a stalled endpoint fails fast and the job-level retry engages, instead of
    # hanging the worker for the full batch timeout. Batch roles keep the long
    # one for large-context ingest calls. A caller-supplied override (resolved
    # from runtime settings) wins over the env-based Settings value.
    if timeout_override_s is not None:
        timeout_s = timeout_override_s
    else:
        timeout_s = (
            settings.llm_interactive_timeout_seconds
            if role in INTERACTIVE_LLM_ROLES
            else settings.llm_timeout_seconds
        )

    return ModelBinding(
        role=role,
        tier=tier,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        extra_headers=_resolve_extra_headers(settings),
        timeout_s=timeout_s,
        retry=retry,
    )


def _resolve_extra_headers(settings: Settings) -> dict[str, str]:
    """Return the parsed-and-whitelisted extra headers from Settings.

    Settings runs the FR-12 whitelist at parse time and stores the cleaned
    dict on ``llm_extra_headers``. Always returns a dict (possibly empty).
    """
    cleaned = getattr(settings, "llm_extra_headers", None)
    if isinstance(cleaned, dict):
        return dict(cleaned)
    return {}


async def resolve_binding_overrides(
    db: AsyncSession,
    role: LLMRole,
    *,
    organization_id: UUID | None = None,
) -> tuple[float, RetryPolicy]:
    """Resolve the DB-backed ``ai.*`` runtime settings into a (timeout, retry)
    pair for ``role``, honouring org overrides.

    Returns ``(timeout_s, RetryPolicy)`` ready to hand to :func:`binding_for`
    as ``timeout_override_s`` / ``retry_override``. The timeout picks the
    interactive vs batch value the same way the env-based path does — an
    interactive role gets ``ai.llm_interactive_timeout_seconds``, embedding
    gets ``ai.embedding_timeout_seconds``, everything else the batch timeout.

    Never raises: :func:`resolve_settings` already degrades to env + defaults
    on a database error, so the AI call path stays up even if the settings
    table is unreachable.
    """
    from abridgeai.core.runtime_settings import (
        resolve_settings,  # noqa: PLC0415 - avoid import cycle
    )

    resolved = await resolve_settings(db, organization_id)

    if role is LLMRole.EMBEDDING:
        timeout_s = float(resolved["ai.embedding_timeout_seconds"])
    elif role in INTERACTIVE_LLM_ROLES:
        timeout_s = float(resolved["ai.llm_interactive_timeout_seconds"])
    else:
        timeout_s = float(resolved["ai.llm_timeout_seconds"])

    retry = RetryPolicy(
        max_attempts=int(resolved["ai.rate_limit_max_attempts"]),
        base_delay_s=float(resolved["ai.rate_limit_base_delay_seconds"]),
        max_delay_s=float(resolved["ai.rate_limit_max_delay_seconds"]),
    )
    return timeout_s, retry
