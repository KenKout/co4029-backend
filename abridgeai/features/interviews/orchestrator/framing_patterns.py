"""Attack-FRAMING patterns: how a request is dressed up, not what it asks for.

Split out of ``security_logic`` to respect the orchestrator's 800-line ceiling.
Pattern data only.

These exist because the content rules were answering the wrong question. A DAN
jailbreak that ends "...reveal the answer key" was logged as
``answer_key_request``: true but incomplete, because the defining feature is the
persona override, and an audit reader triaging jailbreak attempts would never
find it. Framing is therefore checked BEFORE the content rules, so the more
specific finding wins the label. The block decision is unchanged either way —
only the recorded category improves.

Three families:

* ``JAILBREAK_PERSONA`` — DAN / developer-mode / "no restrictions" templates.
* ``DELIMITER_INJECTION`` — chat-format control tokens (``<|im_start|>``,
  ``<system>``, ``### Instruction:``, a forged ``Assistant:`` turn) used to make
  the model believe a new system message started. A candidate has no legitimate
  reason to emit these mid-interview.
* ``PRETEXT_ROLE_CLAIM`` / ``PLEA_PRESSURE`` — claiming staff authority, or
  emotional pressure. These are ROUTING signals only, never verdicts: they carry
  no protected keyword, so the semantic classifier decides. Blocking on them
  directly would refuse an anxious student who is simply telling the truth about
  their situation.
"""

from __future__ import annotations

JAILBREAK_PERSONA = (
    r"\bDAN\b|do\s+anything\s+now|developer\s+mode|dev\s+mode|"
    # "no restrictions" must be asserted about the MODEL, not in passing. A bare
    # "\bno restrictions?\b" wrongly blocked "I have no restrictions on which
    # language I use, so I will answer in English."
    r"you\s+(?:now\s+)?have\s+no\s+(?:restrictions?|rules?|limits?|filters?)|"
    r"(?:you\s+are|act\s+as|behave\s+as)\s+(?:an?\s+)?\w*\s*"
    r"(?:with|without)\s+no\s+(?:restrictions?|rules?|limits?)|"
    r"without\s+(?:any\s+)?(?:restrictions?|filters?)\s+(?:now|at\s+all|whatsoever)|"
    r"ignore\s+(?:all\s+)?(?:content\s+)?polic(?:y|ies)|"
    r"you\s+are\s+(?:now\s+)?(?:an?\s+)?(?:AI|assistant|model)\s+with\s+no\s+"
    r"(?:restrictions?|rules?|limits?)|"
    r"pretend\s+you\s+are\s+(?:an?\s+)?(?:AI|assistant|model)\s+with\s+no|"
    r"jailbr(?:eak|oken)|unrestricted\s+mode|"
    r"chế\s+độ\s+(?:nhà\s+phát\s+triển|không\s+giới\s+hạn)|không\s+còn\s+giới\s+hạn"
)

# Chat-template control tokens and forged turn headers.
DELIMITER_INJECTION = (
    r"<\|[a-z_]+\|>|"  # <|im_start|>, <|system|>, <|endoftext|>
    r"</?(?:system|assistant|user|instructions?)\s*>|"  # <system> ... </system>
    r"(?:^|\n)\s*#{2,}\s*(?:instruction|system|prompt|new\s+instruction)s?\s*:|"
    r"(?:^|\n)\s*(?:system|assistant|user)\s*:\s|"
    r"\[/?(?:INST|SYS|SYSTEM)\]|"
    r"-{3,}\s*(?:system|instruction)s?\s*-{3,}"
)

# Claiming staff authority. Routing signal only.
PRETEXT_ROLE_CLAIM = (
    r"\b(?:i\s*'?m|i\s+am|as\s+(?:the|your))\s+(?:the\s+)?"
    r"(?:professor|instructor|teacher|lecturer|examiner|admin(?:istrator)?|staff|"
    r"course\s+(?:owner|coordinator))\b|"
    r"\b(?:instructor|professor|teacher)\s+(?:here|speaking|overseeing)\b|"
    r"(?:tôi\s+là|với\s+tư\s+cách)\s+(?:giảng\s+viên|giáo\s+viên|quản\s+trị|trợ\s+giảng)"
)

# The staff-duty excuse that turns an authority claim into a request: "I need to
# verify/check/review the questions". Paired with PRETEXT_ROLE_CLAIM this is
# impersonation, because verification is something real staff do on the teacher
# surface, never by asking the interviewer mid-session. Kept separate from a plain
# demand so an explicit ask ("reveal the question bank") keeps its content
# category, which the corpus prefers when the ask is already unambiguous.
VERIFICATION_PRETEXT = (
    r"\b(?:verify|verifying|validate|validating|check(?:ing)?|review(?:ing)?|"
    r"audit(?:ing)?|confirm(?:ing)?|proofread)\b|"
    r"(?:kiểm\s*tra|xác\s*minh|rà\s*soát|duyệt)\s"
)

# Claiming an accessibility need in order to obtain content. Routing signal only,
# and the narrowest pattern in this module by design: a genuine accommodation is
# configured by staff before the session, never negotiated with the interviewer
# mid-answer, but a student mentioning a screen reader for any other reason must
# not be treated as an attacker. Requires the accommodation claim to be paired
# with a demand, which the caller checks.
ACCOMMODATION_CLAIM = (
    r"(?:accessibility|disability|screen\s*reader|assistive|"
    r"\bADA\b|\bIEP\b|accommodation)\s*"
    r"(?:\w+\s+){0,6}?(?:needs?|requires?|must|allows?|entitle)|"
    r"(?:this\s+is|i\s+have)\s+an?\s+(?:accessibility|disability)\s+accommodation|"
    r"(?:hỗ\s+trợ|điều\s+kiện)\s+(?:đặc\s+biệt|khuyết\s+tật)"
)

# Emotional pressure / urgency. Routing signal only.
PLEA_PRESSURE = (
    r"\b(?:i\s*'?m\s+begging|begging\s+you|please\s+please|i\s+will\s+lose|"
    r"i'?ll\s+lose|lose\s+my\s+(?:scholarship|visa|place|job)|"
    r"(?:ten|10|five|5)\s+seconds?\s+left|running\s+out\s+of\s+time|"
    r"last\s+chance|i\s+promise\s+i(?:'?ll|\s+will)\s+learn)\b|"
    r"(?:xin|năn\s+nỉ)\s+(?:bạn|anh|chị)|mất\s+học\s+bổng|hết\s+giờ\s+rồi"
)

# Delegating the composition of an answer without naming "answer" — the shared
# tail of the pretext and plea vectors ("walk me through what you would say",
# "say what I should write"). Routing signal only.
ANSWER_DELEGATION = (
    r"what\s+(?:you\s+would|i\s+should|to)\s+(?:say|write|answer|respond)|"
    r"walk\s+me\s+through\s+what\s+you\s+would|"
    r"(?:just\s+)?(?:say|tell\s+me|write)\s+what\s+i\s+should|"
    r"how\s+would\s+you\s+answer\s+this|"
    # "read me the full answer", "give me the whole answer" — asks for the answer
    # without the "answer key"/"correct answer" wording the content rules match.
    r"(?:read|give|tell|show)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:full|whole|complete|entire|actual)\s+(?:answer|solution|response)|"
    r"nói\s+(?:giúp\s+)?(?:tôi|em|mình)\s+(?:nên\s+)?(?:nói|viết|trả\s+lời)\s+gì|"
    r"(?:đọc|cho)\s+(?:tôi|em|mình)\s+(?:nghe\s+)?(?:toàn\s+bộ|cả|đầy\s+đủ)\s+"
    r"(?:câu\s+)?(?:trả\s+lời|đáp\s+án)"
)

__all__ = [
    "ACCOMMODATION_CLAIM",
    "ANSWER_DELEGATION",
    "DELIMITER_INJECTION",
    "JAILBREAK_PERSONA",
    "PLEA_PRESSURE",
    "PRETEXT_ROLE_CLAIM",
    "VERIFICATION_PRETEXT",
]
