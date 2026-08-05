"""Catalogue of runtime-tunable settings.

Every knob an administrator may change without a deploy is declared here once:
its type, bounds, default, and the environment variable it falls back to. The
registry is the single source of truth for the admin API's validation, the
admin UI's form rendering, and the resolver in
:mod:`abridgeai.core.runtime_settings`.

Why a registry rather than free-form key/value
----------------------------------------------
``system_settings`` stores JSON, so without a schema an admin can write
``{"max_tokens": "lots"}`` and the failure surfaces hours later inside an
ingest worker. Declaring bounds here means a bad value is rejected at the API
boundary with a message naming the field.

It also fixes a quieter problem. ``CHUNKING_MAX_TOKENS``,
``CHUNKING_SEMANTIC_GLUE_THRESHOLD``, ``CHUNKING_LLM_PARALLELISM`` and the rest
sat in ``.env`` and were read by NOTHING — the values actually in force were
the hard-coded defaults in ``SemanticChunker.__init__``. Anyone tuning that
file was changing a dead variable. Every entry below names its ``env_var``, and
:mod:`abridgeai.core.runtime_settings` reads it, so the file finally means what
it appears to mean.

Sensitivity
-----------
Only non-sensitive operational knobs belong here. Credentials, model
endpoints and signing keys stay in the environment: this table is editable
through an HTTP API by anyone holding ``system.administer``, and the values are
returned in plaintext to the UI. ``SETTINGS_REGISTRY`` is intentionally a
closed allowlist — a key absent from it cannot be written through the admin
API at all, so adding a knob is a code review rather than an API call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SettingType = Literal["bool", "int", "float"]

# Groups drive the section headings in the admin UI.
SettingGroup = Literal[
    "ai",
    "chunking",
    "preprocessing",
    "knowledge_graph",
    "retrieval",
    "notifications",
    "spaced_repetition",
]


@dataclass(frozen=True)
class SettingSpec:
    """One tunable value.

    ``env_var`` is the fallback consulted when neither an org row nor a global
    row exists. ``default`` is the last resort and must match the behaviour the
    code had before the setting existed, so introducing a knob never changes
    what a deployment does until someone sets it.
    """

    key: str
    group: SettingGroup
    type: SettingType
    default: bool | int | float
    label: str
    description: str
    env_var: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    # Set when changing the value only takes effect on the next ingest, so the
    # UI can say so rather than implying a live change.
    requires_reprocess: bool = False


_SPECS: tuple[SettingSpec, ...] = (
    # -- ai (model call timeouts + rate-limit retry) ----------------------
    # These override the code defaults that used to live only in
    # ``core.config.Settings`` (timeouts) and ``ai/llm/client.py`` (the 429
    # retry constants). env_var names match the existing Settings env vars so a
    # deployment's current ``.env`` keeps working as the env-layer fallback.
    SettingSpec(
        key="ai.llm_timeout_seconds",
        group="ai",
        type="float",
        default=60.0,
        minimum=1.0,
        maximum=1200.0,
        env_var="LLM_TIMEOUT_SECONDS",
        label="LLM timeout — batch (seconds)",
        description=(
            "Per-call timeout for background/batch LLM roles (extraction, "
            "enrichment, KG extraction, OCR). Raise it for long large-context "
            "ingest calls; lower it to fail a stalled endpoint sooner."
        ),
    ),
    SettingSpec(
        key="ai.llm_interactive_timeout_seconds",
        group="ai",
        type="float",
        default=240.0,
        minimum=1.0,
        maximum=1200.0,
        env_var="LLM_INTERACTIVE_TIMEOUT_SECONDS",
        label="LLM timeout — interactive (seconds)",
        description=(
            "Per-call timeout for interactive stages where a user waits on a "
            "spinner (quiz + interview ideation / generation / validation and "
            "the interview runtime roles). Kept tighter than the batch timeout "
            "so a stalled endpoint can't hang the user for the full batch limit."
        ),
    ),
    SettingSpec(
        key="ai.embedding_timeout_seconds",
        group="ai",
        type="float",
        default=30.0,
        minimum=1.0,
        maximum=600.0,
        env_var="EMBEDDING_TIMEOUT_SECONDS",
        label="Embedding timeout (seconds)",
        description="Per-call timeout for embedding requests.",
    ),
    SettingSpec(
        key="ai.rate_limit_max_attempts",
        group="ai",
        type="int",
        default=4,
        minimum=1,
        maximum=10,
        env_var="LLM_RATE_LIMIT_MAX_ATTEMPTS",
        label="Rate-limit (429) max attempts",
        description=(
            "Total attempts for an LLM/embedding call that keeps returning "
            "HTTP 429 (rate-limited): 1 initial try plus (N-1) retries. Only "
            "429 is retried inline; all other errors propagate to the "
            "job-level retry. 1 disables inline 429 retry."
        ),
    ),
    SettingSpec(
        key="ai.rate_limit_base_delay_seconds",
        group="ai",
        type="float",
        default=1.0,
        minimum=0.0,
        maximum=60.0,
        env_var="LLM_RATE_LIMIT_BASE_DELAY_SECONDS",
        label="Rate-limit backoff base delay (seconds)",
        description=(
            "Base delay for exponential backoff between 429 retries when the "
            "provider sends no Retry-After header. Delay = base * 2^(attempt-1) "
            "plus jitter, capped by the max delay below."
        ),
    ),
    SettingSpec(
        key="ai.rate_limit_max_delay_seconds",
        group="ai",
        type="float",
        default=30.0,
        minimum=1.0,
        maximum=300.0,
        env_var="LLM_RATE_LIMIT_MAX_DELAY_SECONDS",
        label="Rate-limit backoff max delay (seconds)",
        description=(
            "Upper bound on any single wait between 429 retries, applied to "
            "both the exponential backoff and a provider-supplied Retry-After."
        ),
    ),
    # -- chunking ---------------------------------------------------------
    SettingSpec(
        key="chunking.max_tokens",
        group="chunking",
        type="int",
        default=800,
        minimum=100,
        maximum=4000,
        env_var="CHUNKING_MAX_TOKENS",
        label="Window size (tokens)",
        description=(
            "Upper bound on a single rule-based window before the boundary "
            "stage groups them. Larger windows carry more context per "
            "embedding but retrieve less precisely."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.overlap_tokens",
        group="chunking",
        type="int",
        default=80,
        minimum=0,
        maximum=500,
        env_var="CHUNKING_OVERLAP_TOKENS",
        label="Window overlap (tokens)",
        description=(
            "Tokens repeated between adjacent windows so a fact split across a boundary survives "
            "in one of them."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.llm_boundary_enabled",
        group="chunking",
        type="bool",
        default=True,
        env_var="CHUNKING_LLM_BOUNDARY_ENABLED",
        label="LLM decides chunk boundaries",
        description=(
            "Let a lite model group consecutive windows that develop one idea, "
            "instead of merging by embedding similarity. Off falls back to the "
            "similarity glue."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.llm_boundary_batch_size",
        group="chunking",
        type="int",
        default=40,
        minimum=5,
        maximum=200,
        env_var="CHUNKING_LLM_BOUNDARY_BATCH_SIZE",
        label="Boundary batch size (windows)",
        description=(
            "Windows shown to the boundary model per call. The limit is the "
            "model's accuracy over a long list, not the context window."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.llm_boundary_max_carry",
        group="chunking",
        type="int",
        default=8,
        minimum=1,
        maximum=50,
        env_var="CHUNKING_LLM_BOUNDARY_MAX_CARRY",
        label="Boundary carry-over cap (windows)",
        description=(
            "Most windows that may be withheld from a batch and re-shown in the "
            "next one. Bounds the trailing group whose continuation the model "
            "could not see."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.glue_threshold",
        group="chunking",
        type="float",
        default=0.72,
        minimum=0.0,
        maximum=1.0,
        env_var="CHUNKING_SEMANTIC_GLUE_THRESHOLD",
        label="Similarity glue threshold",
        description=(
            "Cosine similarity above which adjacent windows merge. Only "
            "consulted on the fallback path — the LLM boundary stage supersedes it."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.max_window_tokens",
        group="chunking",
        type="int",
        default=2000,
        minimum=200,
        maximum=8000,
        env_var="CHUNKING_SEMANTIC_GLUE_MAX_WINDOW_TOKENS",
        label="Merged chunk ceiling (tokens)",
        description="No merge may produce a chunk larger than this, whichever stage proposes it.",
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.min_window_tokens",
        group="chunking",
        type="int",
        default=30,
        minimum=0,
        maximum=500,
        env_var="CHUNKING_SEMANTIC_GLUE_MIN_WINDOW_TOKENS",
        label="Tiny-window absorb floor (tokens)",
        description=(
            "Windows below this are folded into a same-role neighbour so slide-divider artefacts "
            "never become their own chunk."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="chunking.parallelism",
        group="chunking",
        type="int",
        default=4,
        minimum=1,
        maximum=16,
        env_var="CHUNKING_LLM_PARALLELISM",
        label="Enrichment concurrency",
        description="Chunks enriched concurrently, each in its own DB session.",
    ),
    SettingSpec(
        key="chunking.llm_enrichment_enabled",
        group="chunking",
        type="bool",
        default=True,
        env_var="CHUNKING_LLM_ENRICHMENT_ENABLED",
        label="LLM enrichment",
        description=(
            "Section title, context sentence, key concepts and propositions per "
            "chunk. Off degrades retrieval sharply — the context sentence is "
            "what the embedding is built from."
        ),
        requires_reprocess=True,
    ),
    # -- preprocessing ----------------------------------------------------
    SettingSpec(
        key="preprocess.ocr_enabled",
        group="preprocessing",
        type="bool",
        default=True,
        env_var="PREPROCESS_OCR_ENABLED",
        label="OCR image-only pages",
        description=(
            "Read back pages that carry a figure but no text layer. Off loses those pages "
            "entirely."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="preprocess.ocr_dpi",
        group="preprocessing",
        type="int",
        default=150,
        minimum=72,
        maximum=400,
        env_var="PREPROCESS_OCR_DPI",
        label="OCR render DPI",
        description=(
            "Resolution the page is rasterised at before the vision call. Higher helps dense "
            "tables and costs more."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="preprocess.ocr_advisory_max_pages",
        group="preprocessing",
        type="int",
        default=30,
        minimum=0,
        maximum=5000,
        env_var="PREPROCESS_OCR_MAX_PAGES",
        label="OCR page warning threshold",
        description=(
            "Logs a warning past this many OCR pages in one document. Advisory "
            "only — it does not truncate, because dropping pages deleted "
            "content rather than saving money. 0 silences the warning."
        ),
    ),
    SettingSpec(
        key="preprocess.figure_page_max_words",
        group="preprocessing",
        type="int",
        default=20,
        minimum=0,
        maximum=200,
        env_var="PREPROCESS_FIGURE_PAGE_MAX_WORDS",
        label="Figure-slide word ceiling",
        description=(
            "A page under this many words with real image coverage is treated "
            "as a figure and sent to OCR. Raise it if titled diagram slides are "
            "still coming through empty."
        ),
        requires_reprocess=True,
    ),
    SettingSpec(
        key="preprocess.figure_area_min",
        group="preprocessing",
        type="float",
        default=0.12,
        minimum=0.0,
        maximum=1.0,
        env_var="PREPROCESS_FIGURE_AREA_MIN",
        label="Figure coverage floor",
        description=(
            "Fraction of the page an image must cover to count as content rather than decoration."
        ),
        requires_reprocess=True,
    ),
    # -- knowledge graph --------------------------------------------------
    SettingSpec(
        key="knowledge_graph.enabled",
        group="knowledge_graph",
        type="bool",
        default=True,
        env_var="KNOWLEDGE_GRAPH_ENABLED",
        label="Build the knowledge graph",
        description=(
            "Extract concepts and relationships per chunk into Neo4j. Off skips the stage and its "
            "cost."
        ),
        requires_reprocess=True,
    ),
    # -- retrieval --------------------------------------------------------
    SettingSpec(
        key="retrieval.deprioritized_ratio",
        group="retrieval",
        type="int",
        default=4,
        minimum=1,
        maximum=20,
        env_var="RETRIEVAL_DEPRIORITIZED_RATIO",
        label="Summary/review cap divisor",
        description=(
            "At most floor(limit / N) summary, review or front-matter chunks "
            "reach the generator. Lower values admit more of them."
        ),
    ),
    # -- notifications ----------------------------------------------------
    SettingSpec(
        key="notifications.sr_reminder_cooldown_hours",
        group="notifications",
        type="int",
        default=24,
        minimum=1,
        maximum=168,
        env_var="SR_REMINDER_COOLDOWN_HOURS",
        label="Spaced-repetition reminder cooldown (hours)",
        description=(
            "Minimum gap between spaced-repetition due-card reminders for the "
            "same student. The scan still runs hourly to catch new backlog "
            "promptly, but a student who already got a reminder inside this "
            "window is skipped — so at the default of 24 they get at most one "
            "review reminder per day no matter how many cards are overdue."
        ),
    ),
    # -- spaced_repetition ------------------------------------------------
    SettingSpec(
        key="spaced_repetition.daily_review_cap",
        group="spaced_repetition",
        type="int",
        default=200,
        minimum=0,
        maximum=1000,
        env_var="SR_DAILY_REVIEW_CAP",
        label="Daily review cap (cards/day)",
        description=(
            "Maximum spaced-repetition cards a student is served for review in "
            "one day, so a large backlog stays a finishable daily goal instead "
            "of an unbounded wall. Cards already reviewed today count toward the "
            "cap. This bounds the REVIEW QUEUE only — it never changes lesson "
            "unlock eligibility or retention scoring, so progression is "
            "unaffected. Set to 0 for no cap (serve the whole backlog)."
        ),
    ),
)


SETTINGS_REGISTRY: dict[str, SettingSpec] = {spec.key: spec for spec in _SPECS}


class SettingValidationError(ValueError):
    """A value that does not satisfy its spec."""


def coerce_and_validate(key: str, value: Any) -> bool | int | float:  # noqa: ANN401
    """Return ``value`` coerced to the spec's type, or raise.

    Booleans are checked before ints on purpose: ``bool`` is a subclass of
    ``int`` in Python, so ``isinstance(True, int)`` is true and an unguarded
    int branch would silently accept ``True`` as ``1`` for a numeric setting.
    """
    spec = SETTINGS_REGISTRY.get(key)
    if spec is None:
        raise SettingValidationError(f"Unknown setting {key!r}")

    if spec.type == "bool":
        if not isinstance(value, bool):
            raise SettingValidationError(f"{key} must be true or false")
        return value

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SettingValidationError(f"{key} must be a number")

    number: int | float = int(value) if spec.type == "int" else float(value)
    if spec.type == "int" and float(value) != float(number):
        raise SettingValidationError(f"{key} must be a whole number")
    if spec.minimum is not None and number < spec.minimum:
        raise SettingValidationError(f"{key} must be >= {spec.minimum}")
    if spec.maximum is not None and number > spec.maximum:
        raise SettingValidationError(f"{key} must be <= {spec.maximum}")
    return number


__all__ = [
    "SETTINGS_REGISTRY",
    "SettingGroup",
    "SettingSpec",
    "SettingType",
    "SettingValidationError",
    "coerce_and_validate",
]
