"""Interviewer professional identity — who the interviewer is, not how they sound.

Why this is separate from :mod:`orchestrator.persona`
-----------------------------------------------------
``persona`` answers *how does the interviewer sound* (warmth, directness,
verbosity…). It could not answer *who is the interviewer*, so every session was
conducted by a nameless "virtual interview assistant". Teachers reported the
result reads as software rather than a person, which matches the published
finding on AI mock interviews: candidates rate the probing and time pressure as
realistic while still describing the experience as "structured software".

Identity is a separate axis because it composes with tone rather than replacing
it: a supportive engineering manager and a strict engineering manager are both
coherent people. Keeping the axes apart also keeps ``persona``'s three-value
CHECK constraint and its preset invariants untouched.

Hard boundary — identical to persona's
--------------------------------------
Everything here shapes LANGUAGE ONLY. No field may be read by ``decision.py``,
``selection.py``, difficulty targeting, or the rubric. Two candidates of equal
ability MUST receive identical decisions and scores whichever interviewer they
meet. Letting the interviewer change *what is asked* would move the assessment
from structured toward unstructured, where measured validity is materially
lower and scores start reflecting the interviewer as much as the candidate.

Deliberately absent: domain knowledge
-------------------------------------
A preset carries a name, a title and a register hint. It carries NO subject
matter, and nothing here may generate a first-person anecdote ("when I scaled
Postgres we hit lock contention…"). Two of this repo's own instruments explain
why: ``quality/prompts/leading_system.j2`` scores "introduces domain content the
student had not produced" at 1-2 out of 5, and ``assess_output_leakage`` runs
``SequenceMatcher`` over the whole utterance, so *longer* interviewer prose
dilutes the ratio and weakens the leak check. War stories cost accuracy and
safety at once.

Storage
-------
The chosen role lives under the ``interviewer_role`` key of the existing
``interview_configs.persona_profile_json`` blob — no migration. ``opening_style``
already establishes that a non-numeric enum belongs there, and migration 0060's
own note says the JSON schema is meant to stay "editable without a migration".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InterviewerRole(str, Enum):  # noqa: UP042 -- match codebase Enum convention
    """Professional role the interviewer presents as.

    ``GENERIC_ASSISTANT`` is the default and reproduces the pre-identity wording
    byte for byte, so every existing config is unaffected until a teacher opts in.
    """

    GENERIC_ASSISTANT = "generic_assistant"
    BACKEND_TECH_LEAD = "backend_tech_lead"
    STAFF_ENGINEER = "staff_engineer"
    ENG_MANAGER = "eng_manager"
    HR_SCREENER = "hr_screener"


@dataclass(frozen=True)
class InterviewerIdentity:
    """A resolved interviewer identity. Language-shaping fields only.

    ``name`` is deliberately absent for :attr:`InterviewerRole.GENERIC_ASSISTANT`
    (``None``), which is what preserves the original "I'm aBridgeAI's virtual
    interview assistant" sentence unchanged.
    """

    role: InterviewerRole
    #: Given name the interviewer introduces themselves by. ``None`` → no name.
    name: str | None
    #: Professional title, per language. Rendered into the opening sentence.
    title_en: str
    title_vi: str
    #: One short clause on register, handed to the phrasing model as guidance.
    #: Never subject matter — see the module docstring.
    register_en: str
    register_vi: str

    def title(self, language: str | None) -> str:
        return self.title_vi if _is_vi(language) else self.title_en

    def register(self, language: str | None) -> str:
        return self.register_vi if _is_vi(language) else self.register_en

    def is_named(self) -> bool:
        """True when this identity introduces itself by name and title."""
        return self.name is not None


def _is_vi(language: str | None) -> bool:
    """Prefix match, mirroring ``ceremony.normalize_language``.

    Kept local rather than imported so this module stays free of service-layer
    dependencies, matching how ``utterance.py`` carries its own ``_lang``.
    """
    return bool(language) and str(language).strip().lower().startswith("vi")


# ── Presets ──────────────────────────────────────────────────────────────────
# Names are common Vietnamese given names: the learner population is Vietnamese
# and an English name would read as a foreign interviewer in a VI session while
# adding nothing in an EN one. Titles are translated because a role rendered in
# English inside a Vietnamese sentence code-switches mid-utterance, which the
# persona-adherence judge is liable to read as tone drift.
PRESETS: dict[str, InterviewerIdentity] = {
    InterviewerRole.GENERIC_ASSISTANT.value: InterviewerIdentity(
        role=InterviewerRole.GENERIC_ASSISTANT,
        name=None,
        title_en="virtual interview assistant",
        title_vi="trợ lý phỏng vấn ảo",
        register_en="neutral and procedural",
        register_vi="trung lập và theo quy trình",
    ),
    InterviewerRole.BACKEND_TECH_LEAD.value: InterviewerIdentity(
        role=InterviewerRole.BACKEND_TECH_LEAD,
        name="Minh",
        title_en="backend tech lead",
        title_vi="tech lead backend",
        register_en="practical and implementation-minded",
        register_vi="thực tế, thiên về triển khai",
    ),
    InterviewerRole.STAFF_ENGINEER.value: InterviewerIdentity(
        role=InterviewerRole.STAFF_ENGINEER,
        name="Quân",
        title_en="staff engineer",
        title_vi="staff engineer",
        register_en="precise, interested in reasoning and trade-offs",
        register_vi="chính xác, quan tâm tới lập luận và đánh đổi",
    ),
    InterviewerRole.ENG_MANAGER.value: InterviewerIdentity(
        role=InterviewerRole.ENG_MANAGER,
        name="Hà",
        title_en="engineering manager",
        title_vi="engineering manager",
        register_en="structured, attentive to how clearly things are explained",
        register_vi="có cấu trúc, chú ý cách diễn đạt",
    ),
    InterviewerRole.HR_SCREENER.value: InterviewerIdentity(
        role=InterviewerRole.HR_SCREENER,
        name="Lan",
        title_en="talent partner",
        title_vi="chuyên viên tuyển dụng",
        register_en="friendly and brisk",
        register_vi="thân thiện và gọn gàng",
    ),
}

_DEFAULT_ROLE = InterviewerRole.GENERIC_ASSISTANT.value

#: The JSON key the chosen role is stored under inside ``persona_profile_json``.
ROLE_JSON_KEY = "interviewer_role"


def identity_from(value: str | None) -> InterviewerIdentity:
    """Resolve a role label to its preset.

    Unknown or missing labels fall back to the generic assistant — i.e. to
    today's behaviour — mirroring how :func:`persona.profile_from` treats an
    unrecognised persona. Returns the shared preset object, so callers must
    treat it as immutable (the dataclass is frozen).
    """
    if not isinstance(value, str):
        return PRESETS[_DEFAULT_ROLE]
    return PRESETS.get(value.strip(), PRESETS[_DEFAULT_ROLE])


def identity_from_config(persona_profile_json: dict | None) -> InterviewerIdentity:
    """Resolve the interviewer identity for a config.

    Defensive by construction, like :func:`persona.profile_from_config`: a
    malformed or absent blob can never raise and can never yield anything but a
    valid preset. Worst case the teacher's choice is ignored and the generic
    assistant stands, which is the pre-feature behaviour.
    """
    if not isinstance(persona_profile_json, dict):
        return PRESETS[_DEFAULT_ROLE]
    return identity_from(persona_profile_json.get(ROLE_JSON_KEY))


def as_prompt_identity(identity: InterviewerIdentity, language: str | None) -> dict[str, str]:
    """Serialise identity for the phrasing prompt.

    Only presentational fields, resolved to one language so the model is never
    handed a translation choice. Nothing decision-bearing appears here — the
    persona invariant test asserts that property over the combined trait payload.
    """
    return {
        "role": identity.role.value,
        "name": identity.name or "",
        "title": identity.title(language),
        "register": identity.register(language),
    }


__all__ = [
    "PRESETS",
    "ROLE_JSON_KEY",
    "InterviewerIdentity",
    "InterviewerRole",
    "as_prompt_identity",
    "identity_from",
    "identity_from_config",
]
