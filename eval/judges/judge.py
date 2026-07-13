"""LLM-as-judge entrypoint (T8.3).

One LLM call per ``judge_response`` (single-criterion scoring) or two
LLM calls per ``judge_pairwise`` (position-swap bias control).

Talks to the upstream provider via ``OpenAICompatibleClient`` directly
rather than ``LLMGateway`` because:
  * the gateway requires an ``AsyncSession`` to write ``ai_model_calls``
    audit rows; eval runs do not (and must not) touch production audit;
  * the gateway resolves a model from per-role tier settings, but the
    judge model is chosen at CLI time and may differ from any tier.

Cost is computed via ``abridgeai.ai.llm.pricing.compute_cost``; models
absent from the price table record ``cost_usd=0.0`` (the eval runner
already enforces a hard $ ceiling separately).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from abridgeai.ai.llm.client import OpenAICompatibleClient
from abridgeai.ai.llm.pricing import compute_cost_static
from abridgeai.ai.llm.roles import LLMRole, ModelBinding
from abridgeai.core.config import Settings, get_settings

_PROMPT_DIR: Final[Path] = Path(__file__).parent / "prompts"

_CAPABILITY_TO_TEMPLATE: Final[dict[str, str]] = {
    "quiz_generation": "judge_quiz.j2",
    "interview_generation": "judge_interview.j2",
    "gap_report": "judge_gap_report.j2",
}

_PAIRWISE_TEMPLATE: Final[str] = "judge_pairwise.j2"

_JUSTIFICATION_MAX_CHARS: Final[int] = 1000

_PARSE_FAILED_PREFIX: Final[str] = "JUDGE_PARSE_FAILED: "

_JSON_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class JudgeScore:
    criterion_id: str
    score: float
    justification: str
    confidence: float
    judge_model: str
    cost_usd: float


@dataclass(frozen=True)
class PairwiseVerdict:
    criterion_id: str
    winner: Literal["a", "b", "tie"]
    justification: str
    confidence: float
    judge_model: str
    cost_usd: float


def _build_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPT_DIR)),
        undefined=StrictUndefined,
        autoescape=False,  # noqa: S701 - prompts emit raw text to an LLM, not HTML
        keep_trailing_newline=True,
    )


_TEMPLATE_ENV: Final[Environment] = _build_environment()


def _load_template(name: str) -> Template:
    return _TEMPLATE_ENV.get_template(name)


def _template_for_capability(capability: str) -> Template:
    try:
        name = _CAPABILITY_TO_TEMPLATE[capability]
    except KeyError as exc:
        raise ValueError(
            f"unknown scenario capability: {capability!r}; "
            f"expected one of {sorted(_CAPABILITY_TO_TEMPLATE)}"
        ) from exc
    return _load_template(name)


def _build_judge_binding(judge_model: str, settings: Settings) -> ModelBinding:
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is required to run the eval judge; set it in the env")
    return ModelBinding(
        role=LLMRole.VALIDATION,
        tier=None,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=judge_model,
        extra_headers=getattr(settings, "llm_extra_headers", {}) or {},
        timeout_s=settings.llm_timeout_seconds,
    )


def _extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    fence = _JSON_FENCE_RE.search(raw)
    if fence is not None:
        try:
            value = json.loads(fence.group(1))
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value

    if cleaned:
        try:
            decoded, _ = json.JSONDecoder().raw_decode(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"no JSON object found in response: {exc}") from exc
        if isinstance(decoded, dict):
            return decoded

    raise ValueError("no JSON object found in response")


def _clamp_score(value: object, score_min: int, score_max: int) -> float:
    score = float(value)  # type: ignore[arg-type]
    if score < score_min:
        return float(score_min)
    if score > score_max:
        return float(score_max)
    return score


def _clamp_confidence(value: object) -> float:
    confidence = float(value)  # type: ignore[arg-type]
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _truncate_justification(text: str) -> str:
    return text[:_JUSTIFICATION_MAX_CHARS]


async def _call_judge_llm(
    *,
    judge_model: str,
    prompt: str,
    settings: Settings,
) -> tuple[str, float]:
    """POST one chat-completion request and return ``(content, cost_usd)``.

    Isolated for ease of monkey-patching in tests. Returns the raw text
    content (NOT parsed) plus the estimated USD cost; cost is 0.0 when
    the model is not present in the static price table — the eval
    runner's budget check is the source of truth.
    """
    binding = _build_judge_binding(judge_model, settings)
    client = OpenAICompatibleClient(binding)
    messages = [{"role": "user", "content": prompt}]
    response_body, _latency_ms = await client.chat_completions_json(
        messages, response_format={"type": "json_object"}
    )
    try:
        content_str = response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"judge response had unexpected shape: {exc}") from exc
    usage = response_body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    cost_decimal = compute_cost_static(judge_model, input_tokens, output_tokens)
    cost_usd = float(cost_decimal) if cost_decimal is not None else 0.0
    return str(content_str), cost_usd


async def judge_response(
    *,
    judge_model: str,
    scenario_capability: str,
    criterion_id: str,
    criterion_description: str,
    candidate_response: str,
    source_context: str | None = None,
    score_scale: tuple[int, int] = (1, 5),
    settings: Settings | None = None,
) -> JudgeScore:
    """Score one criterion of one candidate response.

    On parse failure or LLM error: returns a ``JudgeScore`` with
    ``score=score_min``, ``confidence=0.0``, and a justification
    prefixed with ``JUDGE_PARSE_FAILED:`` (raw response truncated to
    ``_JUSTIFICATION_MAX_CHARS``). The runner can detect these from the
    prefix.
    """
    settings = settings or get_settings()
    score_min, score_max = score_scale
    template = _template_for_capability(scenario_capability)
    prompt = template.render(
        criterion_description=criterion_description,
        candidate_response=candidate_response,
        source_context=source_context,
        score_min=score_min,
        score_max=score_max,
    )

    cost_usd = 0.0
    try:
        raw, cost_usd = await _call_judge_llm(
            judge_model=judge_model, prompt=prompt, settings=settings
        )
    except Exception as exc:  # noqa: BLE001 - eval must always return a JudgeScore
        return JudgeScore(
            criterion_id=criterion_id,
            score=float(score_min),
            justification=_truncate_justification(f"{_PARSE_FAILED_PREFIX}{exc}"),
            confidence=0.0,
            judge_model=judge_model,
            cost_usd=cost_usd,
        )

    try:
        data = _extract_json_object(raw)
        score = _clamp_score(data["score"], score_min, score_max)
        justification = _truncate_justification(str(data["justification"]))
        confidence = _clamp_confidence(data.get("confidence", 0.5))
    except (KeyError, ValueError, TypeError):
        return JudgeScore(
            criterion_id=criterion_id,
            score=float(score_min),
            justification=_truncate_justification(f"{_PARSE_FAILED_PREFIX}{raw}"),
            confidence=0.0,
            judge_model=judge_model,
            cost_usd=cost_usd,
        )

    return JudgeScore(
        criterion_id=criterion_id,
        score=score,
        justification=justification,
        confidence=confidence,
        judge_model=judge_model,
        cost_usd=cost_usd,
    )


_VALID_PAIRWISE_WINNERS: Final[frozenset[str]] = frozenset({"a", "b", "tie"})

_SWAP_TRANSLATION: Final[dict[str, Literal["a", "b", "tie"]]] = {
    "a": "b",
    "b": "a",
    "tie": "tie",
}


@dataclass(frozen=True)
class _PairwiseRun:
    winner: Literal["a", "b", "tie"]
    justification: str
    confidence: float
    cost_usd: float
    parsed: bool


async def _run_pairwise_once(
    *,
    judge_model: str,
    template: Template,
    criterion_description: str,
    candidate_a: str,
    candidate_b: str,
    source_context: str | None,
    settings: Settings,
) -> _PairwiseRun:
    prompt = template.render(
        criterion_description=criterion_description,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        source_context=source_context,
    )
    cost_usd = 0.0
    try:
        raw, cost_usd = await _call_judge_llm(
            judge_model=judge_model, prompt=prompt, settings=settings
        )
    except Exception as exc:  # noqa: BLE001 - eval must always return a verdict
        return _PairwiseRun(
            winner="tie",
            justification=_truncate_justification(f"{_PARSE_FAILED_PREFIX}{exc}"),
            confidence=0.0,
            cost_usd=cost_usd,
            parsed=False,
        )

    try:
        data = _extract_json_object(raw)
        winner_raw = str(data["winner"]).lower()
        if winner_raw not in _VALID_PAIRWISE_WINNERS:
            raise ValueError(f"invalid winner: {winner_raw!r}")
        winner: Literal["a", "b", "tie"] = winner_raw  # type: ignore[assignment]
        justification = _truncate_justification(str(data["justification"]))
        confidence = _clamp_confidence(data.get("confidence", 0.5))
    except (KeyError, ValueError, TypeError):
        return _PairwiseRun(
            winner="tie",
            justification=_truncate_justification(f"{_PARSE_FAILED_PREFIX}{raw}"),
            confidence=0.0,
            cost_usd=cost_usd,
            parsed=False,
        )

    return _PairwiseRun(
        winner=winner,
        justification=justification,
        confidence=confidence,
        cost_usd=cost_usd,
        parsed=True,
    )


async def judge_pairwise(
    *,
    judge_model: str,
    scenario_capability: str,
    criterion_id: str,
    criterion_description: str,
    candidate_a: str,
    candidate_b: str,
    source_context: str | None = None,
    settings: Settings | None = None,
) -> PairwiseVerdict:
    """Position-swap pairwise comparison for bias control.

    Runs the pairwise prompt twice — once with (a, b) and once with
    (b, a). If both runs agree (after re-mapping the swapped run back
    to original positions) the agreed winner is final; if they
    disagree, the verdict is ``tie``. Either run failing to parse also
    yields ``tie`` so a noisy judge cannot push false confidence.
    """
    settings = settings or get_settings()
    if scenario_capability not in _CAPABILITY_TO_TEMPLATE:
        raise ValueError(
            f"unknown scenario capability: {scenario_capability!r}; "
            f"expected one of {sorted(_CAPABILITY_TO_TEMPLATE)}"
        )
    template = _load_template(_PAIRWISE_TEMPLATE)

    first = await _run_pairwise_once(
        judge_model=judge_model,
        template=template,
        criterion_description=criterion_description,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        source_context=source_context,
        settings=settings,
    )
    second = await _run_pairwise_once(
        judge_model=judge_model,
        template=template,
        criterion_description=criterion_description,
        candidate_a=candidate_b,
        candidate_b=candidate_a,
        source_context=source_context,
        settings=settings,
    )

    cost_usd = first.cost_usd + second.cost_usd
    if not first.parsed or not second.parsed:
        merged = " | ".join(part for part in (first.justification, second.justification) if part)
        return PairwiseVerdict(
            criterion_id=criterion_id,
            winner="tie",
            justification=_truncate_justification(merged),
            confidence=0.0,
            judge_model=judge_model,
            cost_usd=cost_usd,
        )

    second_in_original_frame = _SWAP_TRANSLATION[second.winner]
    if first.winner == second_in_original_frame:
        final_winner = first.winner
        confidence = (first.confidence + second.confidence) / 2.0
    else:
        final_winner = "tie"
        confidence = 0.0

    merged = " | ".join(part for part in (first.justification, second.justification) if part)
    return PairwiseVerdict(
        criterion_id=criterion_id,
        winner=final_winner,
        justification=_truncate_justification(merged),
        confidence=confidence,
        judge_model=judge_model,
        cost_usd=cost_usd,
    )
