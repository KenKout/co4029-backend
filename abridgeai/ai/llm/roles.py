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
        LLMRole.INTERVIEW_FOLLOWUP,
        LLMRole.INTERVIEW_INTENT,
        LLMRole.INTERVIEW_ANALYSIS,
        LLMRole.INTERVIEW_SECURITY,
    }
)


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


def binding_for(role: LLMRole, settings: Settings) -> ModelBinding:
    """Resolve a ``ModelBinding`` for ``role`` against ``settings``.

    Precedence for chat-completion roles:
      1. Per-role override ``LLM_MODEL_<ROLE>`` (e.g. ``LLM_MODEL_GENERATION``)
      2. Tier model ``LLM_MODEL_<TIER>`` where tier comes from ``ROLE_TO_TIER``
      3. Default-tier model ``LLM_MODEL_<settings.llm_default_tier>``

    Embedding bypasses the tier system and reads ``EMBEDDING_*`` settings.

    Raises:
        ConfigError: when the required API key is unset.
    """
    from abridgeai.ai.llm.errors import ConfigError

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
            timeout_s=settings.embedding_timeout_seconds,
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
    # one for large-context ingest calls.
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
