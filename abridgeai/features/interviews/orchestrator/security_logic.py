"""Rules-first prompt-injection assessment and deterministic output guarding.

Student content is normalized and classified as data. This module never
executes, decodes, evaluates, or follows instructions found in that content.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.orchestrator.security import (
    OutputLeakageAssessment,
    ProtectedContent,
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
    SecurityResponsePolicy,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


SECURITY_STAGE_NAME = "interview_security"

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")

# Common Greek/Cyrillic look-alikes used in prompt-obfuscation attempts. This
# is intentionally small and deterministic; NFKC handles compatibility forms.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "ɑ": "a",
        "Α": "a",
        "А": "a",
        "е": "e",
        "Ε": "e",
        "Е": "e",
        "і": "i",
        "Ι": "i",
        "І": "i",
        "ο": "o",
        "о": "o",
        "Ο": "o",
        "О": "o",
        "р": "p",
        "Ρ": "p",
        "Р": "p",
        "с": "c",
        "С": "c",
        "ѕ": "s",
        "Ѕ": "s",
        "х": "x",
        "Χ": "x",
        "Х": "x",
        "у": "y",
        "Υ": "y",
        "У": "y",
        "м": "m",
        "М": "m",
        "т": "t",
        "Т": "t",
    }
)

_REQUEST = (
    r"(?:show|print|list|reveal|give|provide|tell|display|dump|expose|translate|share|"
    r"hãy|cho|in|liệt\s*kê|tiết\s*lộ|hiển\s*thị|cung\s*cấp|nói|dịch|chia\s*sẻ)"
)
_FUTURE = (
    r"(?:remaining|future|next|unasked|hidden|all)\s+(?:interview\s+)?questions?|"
    r"question\s*bank|every\s+(?:remaining\s+)?question|"
    r"(?:tất\s*cả|còn\s*lại|tiếp\s*theo|chưa\s*được\s*hỏi|ẩn)\s+(?:các\s+)?câu\s*hỏi|"
    r"ngân\s*hàng\s*câu\s*hỏi"
)
_ANSWER_KEY = (
    r"(?:ideal|perfect|model|reference|expected|best)\s+answers?|answer\s*key|"
    r"perfect\s+response|(?:đáp\s*án|câu\s*trả\s*lời)\s*(?:mẫu|hoàn\s*hảo|lý\s*tưởng|đúng)|"
    r"đáp\s*án"
)
_RUBRIC = (
    r"rubric|grading\s+(?:criteria|logic|weights?)|scoring\s+(?:criteria|logic|weights?)|"
    r"passing\s+threshold|outcome\s+coverage|expected\s+evidence|common\s+misconceptions|"
    r"tiêu\s+chí\s+chấm(?:\s*điểm)?|trọng\s+số|logic\s+chấm|ngưỡng\s+(?:đạt|qua)|"
    r"minh\s+chứng\s+mong\s+đợi|độ\s+bao\s+phủ"
)
_SYSTEM = (
    r"system\s*prompt|developer\s*prompt|hidden\s+instructions?|internal\s+prompt|"
    r"prompt\s+hệ\s+thống|chỉ\s+dẫn\s+ẩn|hướng\s+dẫn\s+nội\s+bộ|lời\s+nhắc\s+hệ\s+thống"
)
_HIDDEN_STATE = (
    r"internal\s+(?:state|decision|rationale|reasoning|control\s*flow)|"
    r"tool\s+(?:definitions?|arguments?)|candidate\s+question\s+scores?|remembered\s+words?|"
    r"trạng\s+thái\s+nội\s+bộ|lý\s+do\s+nội\s+bộ|định\s+nghĩa\s+công\s+cụ|"
    r"đối\s+số\s+công\s+cụ|các\s+từ\s+đã\s+nhớ"
)
_OVERRIDE = (
    r"ignore\s+(?:all\s+)?(?:your\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)|"
    r"disregard\s+(?:your\s+)?(?:instructions?|rules?)|forget\s+(?:your\s+)?(?:instructions?|rules?)|"
    r"override\s+(?:your\s+)?(?:instructions?|rules?)|do\s+not\s+follow\s+(?:your\s+)?rules|"
    r"bỏ\s+qua\s+(?:tất\s+cả\s+)?(?:các\s+)?(?:chỉ\s+dẫn|hướng\s+dẫn|quy\s+tắc)\s+(?:trước|trên)|"
    r"quên\s+(?:các\s+)?(?:chỉ\s+dẫn|quy\s+tắc)|ghi\s+đè\s+(?:chỉ\s+dẫn|quy\s+tắc)"
)
_ROLEPLAY = (
    r"(?:pretend|act|role[ -]?play)\s+(?:that\s+)?(?:you\s+are|as)|"
    r"i\s+am\s+(?:the\s+)?(?:course\s+)?(?:administrator|teacher|developer)|"
    r"debug(?:ging)?\s+session|teacher\s+mode|admin\s+mode|"
    r"(?:giả\s+vờ|đóng\s+vai)\s+(?:là\s+)?|tôi\s+là\s+(?:quản\s+trị\s+viên|giảng\s+viên|giáo\s+viên)|"
    r"chế\s+độ\s+(?:gỡ\s+lỗi|giáo\s+viên|quản\s+trị)"
)
_STANDALONE_ROLEPLAY_BYPASS = (
    r"pretend\s+(?:this|it)\s+is\s+(?:a\s+)?debug(?:ging)?\s+session|"
    r"act\s+as\s+(?:the\s+)?teacher\s+instead\s+of\s+(?:the\s+)?interviewer|"
    r"switch\s+to\s+(?:teacher|administrator|developer|debug)\s+mode|"
    r"giả\s+vờ\s+(?:đây|nó)\s+là\s+(?:một\s+)?phiên\s+gỡ\s+lỗi|"
    r"đóng\s+vai\s+(?:giáo\s+viên|giảng\s+viên)\s+thay\s+vì\s+người\s+phỏng\s+vấn|"
    r"chuyển\s+sang\s+chế\s+độ\s+(?:giáo\s+viên|quản\s+trị|gỡ\s+lỗi)"
)
_MULTITURN_COLLECTION = (
    r"(?:remember|memorize|collect|extract|copy)\s+(?:the\s+)?(?:first|last)\s+"
    r"(?:word|letter)\s+(?:of|from)\s+(?:each|every|all).*(?:remaining|future|unasked|hidden)\s+question|"
    r"(?:hãy\s+)?(?:nhớ|ghi\s+nhớ|thu\s+thập|trích)\s+(?:từ|chữ)\s+(?:đầu|cuối).*(?:mỗi|tất\s+cả).*"
    r"(?:câu\s+hỏi\s+còn\s+lại|câu\s+hỏi\s+chưa\s+được\s+hỏi|câu\s+hỏi\s+ẩn)"
)
_GRADING = (
    r"(?:give|set|assign|change)\s+(?:me|my\s+score)?\s*(?:to\s+)?100|"
    r"(?:give|award|assign|set)\s+(?:me\s+|my\s+)?(?:a\s+)?perfect\s+(?:score|grade)\b|"
    r"(?:give|award|assign)\s+me\s+full\s+marks?\b|"
    r"mark\s+(?:this|my)\s+answer\s+(?:as\s+)?correct|"
    r"do\s+not\s+record\s+(?:that\s+)?i\s+skipped|"
    r"(?:cho|đặt|chấm)\s+(?:tôi|em|mình)?\s*(?:điểm\s*)?100|"
    r"đặt\s+điểm\s+(?:của\s+)?(?:tôi|em|mình)\s+(?:thành|là)\s+100|"
    r"chấm\s+(?:câu\s+trả\s+lời\s+)?(?:này\s+)?đúng|đừng\s+ghi\s+(?:lại|nhận).*(?:bỏ\s+qua|không\s+trả\s+lời)"
)
_CROSS_SESSION = (
    r"(?:another|other)\s+students?(?:'s|')?\s+(?:answers?|scores?|data|information)|"
    r"(?:answers?|scores?|data|information)\s+(?:from|of)\s+(?:another|other)\s+students?|"
    r"(?:câu\s+trả\s+lời|điểm|dữ\s+liệu|thông\s+tin)\s+(?:của\s+)?(?:sinh\s+viên|học\s+sinh)\s+khác"
)

_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])")
_HEX_RE = re.compile(r"(?:(?:0x)?[0-9a-f]{2}[\s:,_-]*){12,}", re.IGNORECASE)

_INTERNAL_MARKERS = (
    "system prompt",
    "developer prompt",
    "tool definitions",
    "tool arguments",
    "candidate question scores",
    "internal decision rationale",
    "expected evidence",
    "common misconceptions",
    "security_policy_version",
    "security_rules_version",
    "security_prompt_version",
    "output_guard_version",
)

_HIGH_CONFIDENCE_INTERNAL_OUTPUT_RE = re.compile(
    r"(?:my|the|our)\s+(?:system|developer)\s+prompt\s+(?:is|says|contains|:)|"
    r"(?:tool\s+(?:definitions?|arguments?)|candidate\s+question\s+scores?)\s+(?:are|:)|"
    r"(?:internal\s+decision\s+rationale|expected\s+evidence|common\s+misconceptions)\s+"
    r"(?:is|are|includes?|:)|"
    r"(?:rubric|scoring|grading|outcome).{0,30}(?:weights?|threshold).{0,20}(?:is|are|:|\d)|"
    r"security_(?:policy|rules|prompt)_version|output_guard_version",
    re.IGNORECASE,
)


def normalize_input(value: str) -> str:
    """Normalize without decoding or executing any embedded content."""
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = _ZERO_WIDTH_RE.sub("", normalized).translate(_HOMOGLYPHS).casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalized_fingerprint(value: str) -> str | None:
    normalized = normalize_input(value)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _compact(value: str) -> str:
    return "".join(ch for ch in normalize_input(value) if ch.isalnum())


def _contains_request_for(text: str, target: str) -> bool:
    return bool(
        re.search(rf"{_REQUEST}.{{0,80}}(?:{target})", text, re.IGNORECASE)
        or re.search(rf"(?:{target}).{{0,80}}{_REQUEST}", text, re.IGNORECASE)
    )


def _rule_category(  # noqa: C901 -- ordered security rules are intentionally explicit
    value: str,
    *,
    last_category: SecurityCategory | None = None,
) -> SecurityCategory:
    text = normalize_input(value)
    compact = _compact(text)
    if not text:
        return SecurityCategory.BENIGN

    if re.search(_CROSS_SESSION, text, re.IGNORECASE):
        return SecurityCategory.CROSS_SESSION_DATA_REQUEST
    if re.search(_MULTITURN_COLLECTION, text, re.IGNORECASE):
        return SecurityCategory.FUTURE_QUESTION_REQUEST
    if _contains_request_for(text, _FUTURE) or any(
        marker in compact
        for marker in (
            "showallremainingquestions",
            "listallremainingquestions",
            "revealquestionbank",
            "tatcacauhoiconlai",
            "nganhangcauhoi",
        )
    ):
        return SecurityCategory.FUTURE_QUESTION_REQUEST
    if _contains_request_for(text, _ANSWER_KEY) or any(
        marker in compact for marker in ("givemetheidealanswer", "showanswerkey", "chotoidapan")
    ):
        return SecurityCategory.ANSWER_KEY_REQUEST
    if _contains_request_for(text, _RUBRIC) or any(
        marker in compact
        for marker in ("showthegradingrubric", "rubricandscoringweights", "tieuchichamdiem")
    ):
        return SecurityCategory.RUBRIC_EXFILTRATION
    if _contains_request_for(text, _SYSTEM) or any(
        marker in compact
        for marker in ("printyoursystemprompt", "showsystemprompt", "dichchidanan")
    ):
        return SecurityCategory.SYSTEM_PROMPT_REQUEST
    if re.search(_GRADING, text, re.IGNORECASE):
        return SecurityCategory.GRADING_MANIPULATION
    if _contains_request_for(text, _HIDDEN_STATE):
        return SecurityCategory.HIDDEN_STATE_REQUEST
    if re.search(_OVERRIDE, text, re.IGNORECASE):
        return SecurityCategory.INSTRUCTION_OVERRIDE
    if re.search(_STANDALONE_ROLEPLAY_BYPASS, text, re.IGNORECASE):
        return SecurityCategory.ROLEPLAY_BYPASS
    if re.search(_ROLEPLAY, text, re.IGNORECASE) and (
        re.search(r"reveal|show|print|give|provide|bypass|ignore|tiết\s*lộ|cho|bỏ\s+qua", text)
        or re.search(_FUTURE + "|" + _SYSTEM + "|" + _RUBRIC + "|" + _ANSWER_KEY, text)
    ):
        return SecurityCategory.ROLEPLAY_BYPASS
    encoded_hint = re.search(r"base64|hex(?:adecimal)?|rot13|encoded?|mã\s+hóa|giải\s+mã", text)
    if encoded_hint and (
        _BASE64_RE.search(value) or _HEX_RE.search(value) or re.search(_REQUEST, text)
    ):
        return SecurityCategory.ENCODED_EXFILTRATION
    if (_BASE64_RE.search(value) or _HEX_RE.search(value)) and re.search(
        r"decode|execute|follow|read|giải\s*mã|thực\s*hiện|làm\s+theo", text
    ):
        return SecurityCategory.ENCODED_EXFILTRATION
    if (
        last_category is not None
        and last_category is not SecurityCategory.BENIGN
        and re.search(
            r"(?:now|then)\s+(?:give|show|tell|print).*(?:remembered|them|it)|"
            r"(?:bây\s+giờ|tiếp\s+theo).*(?:cho|nói|in).*(?:đã\s+nhớ|chúng|nó)",
            text,
        )
    ):
        return SecurityCategory.HIDDEN_STATE_REQUEST
    return SecurityCategory.BENIGN


def assess_by_rules(
    value: str,
    *,
    last_category: SecurityCategory | None = None,
) -> SecurityAssessment:
    category = _rule_category(value, last_category=last_category)
    detected = category is not SecurityCategory.BENIGN
    return SecurityAssessment(
        category=category,
        detected=detected,
        confidence=0.98 if detected else 0.0,
        should_block=detected,
        should_record_academic_evidence=not detected,
        response_key=(f"security.{category.value}" if detected else None),
        normalized_fingerprint=normalized_fingerprint(value),
        source="rules",
    )


def is_ambiguous_security_text(value: str) -> bool:
    text = normalize_input(value)
    if len(text) < 8:
        return False
    suspicious = re.search(
        r"hidden|secret|instruction|prompt|rubric|answer\s*key|question\s*bank|"
        r"administrator|teacher\s+mode|debug|bypass|translate|decode|internal|"
        r"perfect\s+(?:score|grade)|full\s+marks?|grading|score\s*(?:to|of)\s*\d+|"
        r"ẩn|bí\s+mật|chỉ\s+dẫn|đáp\s+án|ngân\s+hàng\s+câu\s+hỏi|quản\s+trị|gỡ\s+lỗi|nội\s+bộ",
        text,
    )
    requestish = re.search(
        r"\b(?:can|could|would|please|show|tell|give|print|act|pretend|hãy|cho|vui\s+lòng|giả\s+vờ)\b",
        text,
    )
    return bool(suspicious and requestish)


def security_classifier_system_prompt() -> str:
    """Render the classifier prompt relative to this trusted module."""
    return render_prompt("prompts/security_system.j2")


async def assess_security(
    db: AsyncSession,
    *,
    student_utterance: str,
    last_category: SecurityCategory | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> SecurityAssessment:
    """Rules first; call a semantic classifier only for ambiguous text."""
    rules = assess_by_rules(student_utterance, last_category=last_category)
    if rules.detected or not is_ambiguous_security_text(student_utterance):
        return rules

    try:
        system_prompt = security_classifier_system_prompt()
        user_prompt = json.dumps(
            {
                "student_utterance": student_utterance,
                "prior_security_category": last_category.value if last_category else None,
            },
            ensure_ascii=False,
        )
        gateway = gateway or LLMGateway()
        async with db.begin_nested():
            llm_result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_SECURITY,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=SECURITY_STAGE_NAME,
                pipeline_run_id=pipeline_run_id,
            )
        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
        parsed = _parse_semantic_assessment(payload, student_utterance)
        if parsed is not None:
            return parsed
    except Exception:  # noqa: S110 -- deterministic fallback deliberately hides model errors
        pass

    return SecurityAssessment(
        # This branch is reached only for text that deterministic rules already
        # deemed ambiguous and security-shaped. Enforcement fails closed while
        # shadow mode still observes without changing student-facing behavior.
        category=SecurityCategory.INSTRUCTION_OVERRIDE,
        detected=True,
        confidence=0.65,
        should_block=True,
        should_record_academic_evidence=False,
        response_key="security.classifier_fallback",
        normalized_fingerprint=rules.normalized_fingerprint,
        source="fallback",
        classifier_failed=True,
    )


def _parse_semantic_assessment(
    payload: Mapping[str, Any] | None,
    original: str,
) -> SecurityAssessment | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        category = SecurityCategory(str(payload.get("category", "")).strip().lower())
    except ValueError:
        return None
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return None
    detected = category is not SecurityCategory.BENIGN and confidence >= 0.65
    return SecurityAssessment(
        category=category if detected else SecurityCategory.BENIGN,
        detected=detected,
        confidence=confidence,
        should_block=detected,
        should_record_academic_evidence=not detected,
        response_key=(f"security.{category.value}" if detected else None),
        normalized_fingerprint=normalized_fingerprint(original),
        source="classifier",
    )


def choose_security_action(
    assessment: SecurityAssessment,
    *,
    consecutive_attempts: int,
    max_consecutive_attempts: int,
    response_policy: SecurityResponsePolicy,
    guard_mode: str,
) -> SecurityAction:
    """The classifier describes semantics; code alone chooses control flow."""
    if not assessment.detected or guard_mode in {"off", "shadow"}:
        return SecurityAction.ALLOW
    next_count = consecutive_attempts + 1
    threshold = max(2, max_consecutive_attempts)
    if response_policy is SecurityResponsePolicy.END_AND_FLAG and next_count >= threshold:
        return SecurityAction.END_AND_FLAG
    if response_policy is not SecurityResponsePolicy.CONTINUE_AND_LOG and next_count >= 2:
        return SecurityAction.WARN_AND_REDIRECT
    return SecurityAction.REFUSE_AND_REDIRECT


def should_flag_session(*, attempt_count: int, max_consecutive_attempts: int) -> bool:
    return attempt_count >= max(2, max_consecutive_attempts)


def is_explicit_end_request(value: str) -> bool:
    text = normalize_input(value)
    return bool(
        re.search(
            r"\b(?:end|stop|finish|quit)\s+(?:the\s+|this\s+)?(?:interview|session)\b|"
            r"\bi(?:'|’)m\s+done\b|"
            r"\b(?:kết\s+thúc|dừng|thoát)\s+(?:buổi\s+|cuộc\s+)?phỏng\s+vấn\b|"
            r"\b(?:tôi|mình|em)\s+muốn\s+(?:dừng|kết\s+thúc)\b",
            text,
        )
    )


def requested_current_question_action(value: str) -> SecurityAction | None:
    """Recognize narrow repeat/clarification requests before academic analysis."""
    text = normalize_input(value)
    if re.search(_FUTURE + "|" + _SYSTEM + "|" + _RUBRIC + "|" + _ANSWER_KEY, text):
        return None
    repeat = re.search(
        r"(?:please\s+)?repeat\s+(?:the\s+)?current\s+question|"
        r"(?:can|could)\s+you\s+(?:please\s+)?repeat\s+(?:that|it|the\s+question)|"
        r"say\s+(?:that|the\s+question)\s+again|"
        r"(?:vui\s+lòng\s+)?nhắc\s+lại\s+câu\s+hỏi\s+(?:hiện\s+tại|này)|"
        r"(?:bạn\s+)?có\s+thể\s+nhắc\s+lại\s+(?:câu\s+hỏi|được\s+không)",
        text,
    )
    if repeat:
        return SecurityAction.REPEAT_CURRENT_QUESTION
    clarify = re.search(
        r"(?:can|could)\s+you\s+clarify\s+(?:the\s+)?current\s+question|"
        r"what\s+does\s+.{1,60}\s+mean\s+in\s+(?:this|the)\s+question|"
        r"i\s+don(?:'|’)t\s+understand\s+(?:the\s+)?(?:wording|question)|"
        r"clarify\s+what\s+(?:the\s+)?current\s+question\s+is\s+asking|"
        r"(?:giải\s+thích|làm\s+rõ)\s+(?:câu\s+hỏi\s+)?(?:hiện\s+tại|này)|"
        r"(?:tôi|mình|em)\s+không\s+hiểu\s+(?:cách\s+diễn\s+đạt|câu\s+hỏi)",
        text,
    )
    if clarify:
        return SecurityAction.CLARIFY_CURRENT_QUESTION
    return None


def extract_separable_academic_text(value: str) -> str | None:
    """Remove independently malicious segments; never transform/decode them."""
    segments = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(value or "")]
    safe = [
        segment
        for segment in segments
        if segment and _rule_category(segment) is SecurityCategory.BENIGN
    ]
    joined = " ".join(safe).strip()
    if len(joined) >= 30 and len(_TOKEN_RE.findall(joined)) >= 6:
        return joined
    return None


def safe_security_response(
    action: SecurityAction,
    *,
    language: str,
    current_question: str | None,
    custom_refusal: str | None = None,
) -> str:
    """Deterministic persona-neutral EN/VI response for security actions."""
    vi = (language or "en").lower().startswith("vi")
    question = (current_question or "").strip()
    if action is SecurityAction.REPEAT_CURRENT_QUESTION:
        return question or (
            "Hiện chưa có câu hỏi nào để nhắc lại."
            if vi
            else "There is no current question to repeat."
        )
    if action is SecurityAction.CLARIFY_CURRENT_QUESTION:
        prefix = (
            "Tôi có thể làm rõ cách diễn đạt của câu hỏi hiện tại:"
            if vi
            else "I can clarify the wording of the current question:"
        )
        return f"{prefix} {question}".strip()
    if action is SecurityAction.END_AND_FLAG:
        return (
            "Buổi phỏng vấn đã kết thúc theo chính sách bảo mật của nền tảng."
            if vi
            else "The interview has ended under the platform security policy."
        )
    if action is SecurityAction.WARN_AND_REDIRECT:
        base = (
            "Tôi không thể hỗ trợ yêu cầu đó. Vui lòng trả lời câu hỏi hiện tại; "
            "các yêu cầu lặp lại có thể được ghi nhận để xem xét bảo mật."
            if vi
            else "I can’t help with that request. Please answer the current question; "
            "repeated requests may be recorded for security review."
        )
    else:
        base = custom_refusal.strip() if custom_refusal and custom_refusal.strip() else (
            "Tôi không thể cung cấp các câu hỏi chưa được hỏi, đáp án hoặc tiêu chí "
            "chấm điểm. Tôi có thể nhắc lại hoặc giải thích câu hỏi hiện tại."
            if vi
            else "I can’t provide hidden interview questions, answers, or grading criteria. "
            "I can repeat or clarify the current question."
        )
    if question and action in {
        SecurityAction.REFUSE_AND_REDIRECT,
        SecurityAction.WARN_AND_REDIRECT,
    }:
        label = "Câu hỏi hiện tại:" if vi else "Current question:"
        return f"{base} {label} {question}"
    return base


def safe_closing(language: str) -> str:
    return (
        "Cảm ơn bạn. Buổi phỏng vấn kết thúc tại đây."
        if (language or "en").lower().startswith("vi")
        else "Thank you. That concludes the interview."
    )


def assess_output_leakage(  # noqa: C901 -- ordered leak checks remain auditable
    proposed: str,
    protected_content: Iterable[ProtectedContent],
) -> OutputLeakageAssessment:
    """Compare a proposed learner utterance with server-side protected text."""
    proposed_norm = normalize_input(proposed)
    if not proposed_norm:
        return OutputLeakageAssessment(blocked=False)
    supplied_content = list(protected_content)
    guardable_norm = proposed_norm
    for allowed in supplied_content:
        if allowed.category == "allowed_question_text":
            allowed_norm = normalize_input(allowed.text)
            if allowed_norm:
                guardable_norm = guardable_norm.replace(allowed_norm, " ")
    guardable_norm = _WHITESPACE_RE.sub(" ", guardable_norm).strip()
    proposed_tokens = set(_TOKEN_RE.findall(guardable_norm))
    if _HIGH_CONFIDENCE_INTERNAL_OUTPUT_RE.search(guardable_norm):
        return OutputLeakageAssessment(
            True,
            "internal_control_flow",
            normalized_fingerprint(guardable_norm),
            "internal_marker",
        )

    corpus = [item for item in supplied_content if item.category != "allowed_question_text"] + [
        ProtectedContent(category="internal_prompt_marker", text=marker)
        for marker in _INTERNAL_MARKERS
    ]
    for item in corpus:
        secret_norm = normalize_input(item.text)
        secret_tokens = set(_TOKEN_RE.findall(secret_norm))
        minimum_length = 24
        minimum_tokens = 4
        if item.category == "question_text":
            minimum_length, minimum_tokens = 18, 3
        elif item.category in {"scoring_weight", "passing_threshold"}:
            minimum_length, minimum_tokens = 12, 2
        if len(secret_norm) < minimum_length or len(secret_tokens) < minimum_tokens:
            continue
        fingerprint = normalized_fingerprint(item.text)
        if secret_norm in guardable_norm:
            return OutputLeakageAssessment(True, item.category, fingerprint, "exact")
        matched_tokens = len(proposed_tokens & secret_tokens)
        overlap = matched_tokens / max(1, len(secret_tokens))
        if len(secret_tokens) >= 8 and matched_tokens >= 7 and overlap >= 0.68:
            return OutputLeakageAssessment(True, item.category, fingerprint, "token_overlap")
        if len(secret_norm) >= 48 and len(guardable_norm) >= 32:
            ratio = SequenceMatcher(None, secret_norm, guardable_norm, autojunk=False).ratio()
            if ratio >= 0.88:
                return OutputLeakageAssessment(True, item.category, fingerprint, "fuzzy")
    return OutputLeakageAssessment(blocked=False)


__all__ = [
    "SECURITY_STAGE_NAME",
    "assess_by_rules",
    "assess_output_leakage",
    "assess_security",
    "choose_security_action",
    "extract_separable_academic_text",
    "is_ambiguous_security_text",
    "is_explicit_end_request",
    "normalize_input",
    "normalized_fingerprint",
    "requested_current_question_action",
    "safe_closing",
    "safe_security_response",
    "security_classifier_system_prompt",
    "should_flag_session",
]
