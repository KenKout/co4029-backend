"""Interview PERSONA ADHERENCE stage (Phase 4 — persona measurement).

Public API:
    audit_persona_adherence(...) — offline tone-only auditor entry point
    PersonaAdherence             — stage output dataclass
    parse_persona_adherence(...) — pure parser (testable in isolation)
    unavailable()                — sentinel when no usable judgement exists
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.stages.persona_adherence.logic import (
    PERSONA_ADHERENCE_STAGE_NAME,
    audit_persona_adherence,
)
from abridgeai.features.interviews.ai.stages.persona_adherence.parsers import (
    PersonaAdherence,
    parse_persona_adherence,
    unavailable,
)

__all__ = [
    "PERSONA_ADHERENCE_STAGE_NAME",
    "PersonaAdherence",
    "audit_persona_adherence",
    "parse_persona_adherence",
    "unavailable",
]
