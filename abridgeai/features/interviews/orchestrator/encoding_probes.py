"""Reversible decoders used to judge obfuscated prompt-injection attempts.

Phase 1.3 of the interview security hardening. A candidate can hide a protected
request behind a trivial transform — rot13, percent-encoding, leetspeak,
reversal — and the rules layer, which matches plain text, sees nothing.

Two absolute constraints on everything in this module:

* **Nothing decoded is ever executed, echoed, logged, or sent to a model.** A
  decoded string exists only long enough to be matched against the protected
  concepts and is then dropped. The verdict is the only thing that escapes.
* **Every transform is total and side-effect-free.** No eval, no imports, no
  network, no filesystem. Worst case a decode produces gibberish that matches
  nothing, which is indistinguishable from "no attempt found".

Deliberately NOT here: anything lossy or ambiguous enough to fire on ordinary
prose. Naming an encoding is not an attempt — "0xffffff in hexadecimal notation"
is a legitimate answer — so the caller pairs these probes with act-on-it intent
and only treats a decode as evidence when it lands on a protected concept.
"""

from __future__ import annotations

import binascii
import codecs
import re
from urllib.parse import unquote

# Concepts that make a decoded payload an exfiltration attempt rather than
# noise. Kept narrow on purpose: these are the platform's own protected assets,
# not general vocabulary, so a false decode almost never lands here by accident.
_PROTECTED_INTENT_RE = re.compile(
    r"system\s*prompt|developer\s*prompt|answer\s*key|"
    r"(?:model|ideal|correct)\s+answers?|"
    r"grading\s+(?:criteria|rubric|weights?)|scoring\s+(?:criteria|weights?)|rubric|"
    r"remaining\s+questions?|question\s*bank|hidden\s+questions?|"
    r"internal\s+(?:state|instructions?|rationale)|tool\s+definitions?",
    re.IGNORECASE,
)

# Leetspeak is decoded by folding digits back to letters. Only the unambiguous
# substitutions are included; 1->l and 5->s style collisions are accepted
# because the result is only ever pattern-matched, never shown to anyone.
_LEET_TABLE = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})


_MIN_PAYLOAD_LEN = 8
_MAX_INPUT_LEN = 4000


def _rot13(value: str) -> str:
    return codecs.encode(value, "rot13")


def _percent_decoded(value: str) -> str:
    # errors="replace" so a malformed escape degrades to a replacement char
    # instead of raising — a broken payload is simply not an attempt.
    return unquote(value, errors="replace")


def leet_folded(value: str) -> str:
    """Fold leetspeak digits back to letters.

    Public because it is the one transform that yields readable prose, so the
    caller can recover the SEMANTIC category of a leetspeak request — a folded
    "5y5t3m pr0mpt" is still a system-prompt request — instead of flattening
    everything to "encoded". Still just a character translation.
    """
    return value.translate(_LEET_TABLE)


def _reversed_text(value: str) -> str:
    return value[::-1]


def _hex_decoded(value: str) -> str:
    """Decode long runs of hex bytes; ignore anything that is not clean hex."""
    out: list[str] = []
    for run in re.findall(r"(?:[0-9a-fA-F]{2}[\s:,_-]*){8,}", value):
        compact = re.sub(r"[^0-9a-fA-F]", "", run)
        if len(compact) % 2:
            compact = compact[:-1]
        try:
            out.append(bytes.fromhex(compact).decode("utf-8", errors="replace"))
        except (ValueError, binascii.Error):
            continue
    return " ".join(out)


def decoded_variants(value: str) -> list[str]:
    """Every candidate plaintext for ``value``, cheapest transform first.

    Order is irrelevant to correctness — the caller matches all of them — but
    keeping it stable makes the behaviour reproducible in tests.
    """
    if not value or len(value) < _MIN_PAYLOAD_LEN:
        return []
    # Bound the work: the guard runs synchronously on every turn, and a decoder
    # sweep over an unbounded transcript is a denial-of-service surface.
    probe = value[:_MAX_INPUT_LEN]
    variants = [
        _rot13(probe),
        _percent_decoded(probe),
        leet_folded(probe),
        _reversed_text(probe),
        _hex_decoded(probe),
    ]
    return [v for v in variants if v and v != probe]


def decodes_to_protected_request(value: str) -> bool:
    """True when some reversible decode of ``value`` asks for protected content.

    This is the whole public contract: a boolean. The decoded text never leaves
    this module.
    """
    return any(_PROTECTED_INTENT_RE.search(variant) for variant in decoded_variants(value))


# Greek/Cyrillic look-alikes used to smuggle keywords past ASCII matching.
# Intentionally small and deterministic; NFKC already handles compatibility
# forms. Lives here with the other canonicalisation data — ``normalize_input``
# in security_logic applies it before any rule runs.
HOMOGLYPHS = str.maketrans(
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


__all__ = ["HOMOGLYPHS", "decoded_variants", "decodes_to_protected_request", "leet_folded"]
