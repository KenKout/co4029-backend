"""End-to-end voice harness CLI (semi-automated smoke test).

Drives the ENTIRE voice interview path without a human at a microphone:

    create voice session → mint token (dispatches agent) → join room →
    agent speaks first question (captured) → publish student answer clip
    (agent Whisper STT transcribes) → adaptive brain decides → agent speaks
    (captured) → … → finish.

Student "answers" are synthesized with the SAME gateway TTS the agent uses, so
they are real spoken audio the agent can transcribe. Every agent utterance is
captured to a WAV file under the output dir for inspection.

WHAT THIS COVERS
  * token mint + agent dispatch + room join
  * agent STT of real spoken audio
  * orchestration bridge → adaptive brain (flag on) → canonical result
  * agent TTS produced and delivered as non-empty audio, in EN or VI
  * per-turn language threading (via Accept-Language → dispatch metadata)

WHAT THIS DOES NOT COVER
  * subjective audio quality / naturalness (a human must still listen)
  * true acoustic barge-in timing (approximated: we can start a clip while the
    agent is still speaking, but echo-cancellation / VAD behaviour on real mic
    hardware is not reproduced)

USAGE
    # flag must be on for adaptive; harness will warn if it is off
    ADAPTIVE_INTERVIEWER_ENABLED=true INTERVIEW_VOICE_ENABLED=true \
      .venv/bin/python -m scripts.voice_harness.run_harness \
        --config-id <interview_config_id> \
        --student-id <user_id> \
        --language en \
        --answers "A fact table stores measurements" "It links to dimensions" \
        --out ./voice-harness-out

If --config-id is omitted the harness provisions a throwaway config + 2
outcomes + 2 approved questions and a student, runs, then leaves them (use
--cleanup to delete afterwards).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as `python -m scripts.voice_harness.run_harness` from backend root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text  # noqa: E402

from abridgeai.core.config import get_settings  # noqa: E402
from abridgeai.core.db import get_sessionmaker  # noqa: E402
from abridgeai.features.interviews.services import real_time as rt_service  # noqa: E402
from abridgeai.features.interviews.services.narration import synthesize_speech  # noqa: E402
from scripts.voice_harness import audio_io  # noqa: E402
from scripts.voice_harness.room_client import HarnessRoom  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("voice_harness")

# Default hard ceiling on a single scenario (join + all turns + capture). Beyond
# this the scenario is aborted and cleanup still runs. Overridable per call.
DEFAULT_SCENARIO_TIMEOUT_S = 240.0

# Bump when the machine-readable result shape changes in a breaking way, so
# downstream consumers (dashboards, CI parsers) can guard on it.
RESULT_SCHEMA_VERSION = 1


class HarnessError(RuntimeError):
    """Raised for a setup/credential failure the caller should surface."""


@dataclass
class HarnessTimeline:
    """Wall-clock UTC (ISO-8601) event timeline for one scenario.

    Every field here is CLIENT-OBSERVABLE from the synthetic student's vantage
    point — room lifecycle, when we published student audio, and when agent
    audio arrived. Fields the harness cannot observe from outside the agent
    (true STT-final time, decision-complete time INSIDE the agent) are left
    null and documented as agent-internal; the DB-derived proxies live in
    ``LatencyMetrics`` with explicit ``*_source`` notes.
    """

    room_joined_at: str | None = None
    agent_joined_at: str | None = None
    # First student answer published (end-of-speech of turn 1) and the last one.
    first_student_audio_started_at: str | None = None
    last_student_audio_ended_at: str | None = None
    # First agent audio frame ever, and after the LAST student turn (the reply
    # whose latency we can honestly measure client-side).
    agent_audio_first_frame_at: str | None = None
    agent_audio_first_frame_after_last_prompt_at: str | None = None
    agent_audio_ended_at: str | None = None
    disconnected_at: str | None = None
    # AI-turn DB row timestamps (created_at) — a proxy for "decision committed".
    ai_turn_committed_at: list[str] = field(default_factory=list)


@dataclass
class LatencyMetrics:
    """Derived latencies (seconds). Each carries a ``*_source`` note making
    clear whether it is CLIENT-observable or a DB-derived proxy. Aggressive
    thresholds are NOT enforced yet — these are baseline-collection only.
    """

    # end-of-last-student-speech → the AI turn's DB commit for that turn. This
    # is the MOST HONEST turn-latency proxy: it folds STT + decision + phrasing
    # up to persistence. Source is the DB row created_at, so it excludes
    # TTS-audio playout but is not fooled by the continuously-streamed track.
    end_of_speech_to_decision_committed_s: float | None = None
    # end-of-last-student-speech → first agent audio frame observed after it.
    # CAVEAT: the agent publishes a CONTINUOUS audio track (frames never stop
    # between utterances), so this catches the next streamed frame — NOT
    # reliably the reply onset. Kept for completeness but NOT a trustworthy
    # turn-latency signal; prefer end_of_speech_to_decision_committed_s.
    end_of_speech_to_agent_audio_s: float | None = None
    # Total agent speaking span (first frame ever → last frame).
    agent_audio_span_s: float | None = None
    # Room join → agent audio track subscribed.
    room_join_to_agent_join_s: float | None = None
    # Whole scenario wall-clock (room join → disconnect).
    total_scenario_s: float | None = None
    notes: dict[str, str] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Machine-readable outcome of one harness scenario run.

    Serialized to JSON (``--json`` / ``result.json``) so CI / pytest can assert
    on it without scraping logs.
    """

    ok: bool
    language: str
    session_id: str | None = None
    config_id: str | None = None
    # Authoritative "did the ADAPTIVE brain actually run" signal: the adaptive
    # orchestrator writes exactly one interview_runtime_states row per session;
    # the legacy path writes none.
    runtime_state_count: int = 0
    adaptive_ran: bool = False
    message_count: int = 0
    adaptive_actions: list[str] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    # Selected next-question ids, in order (advance actions only). A proxy for
    # what the adaptive selector chose to ask next.
    selected_question_ids: list[str] = field(default_factory=list)
    agent_utterance_count: int = 0
    captured_audio_seconds: float = 0.0
    peak_amplitude: int = 0
    # Schema version for the machine-readable result (guard for consumers).
    result_schema_version: int = RESULT_SCHEMA_VERSION
    # Diagnostic event timeline + derived latencies (baseline collection only;
    # no aggressive thresholds enforced yet).
    timeline: HarnessTimeline = field(default_factory=HarnessTimeline)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    # ── Strict adaptive verification (--require-adaptive / REQUIRE_ADAPTIVE_VOICE) ──
    # adaptive_required: strict mode was requested for this run.
    # adaptive_requirement_met: ALL strict criteria held (only meaningful when
    #   adaptive_required is True). When strict mode is off this stays True so it
    #   never spuriously fails a non-strict run.
    # fallback_count: adaptive AI turns that degraded to the deterministic
    #   fallback phrasing (utterance_status == "fallback"). A run where EVERY
    #   turn fell back is not a genuine adaptive exercise.
    # state_version: the runtime-state optimistic-lock counter (None if no row).
    # decision_count: adaptive AI turns carrying a structured action decision.
    adaptive_required: bool = False
    adaptive_requirement_met: bool = True
    fallback_count: int = 0
    state_version: int | None = None
    decision_count: int = 0
    # Security verification is sourced from persisted redacted audit events,
    # never inferred from refusal wording or audio transcription text.
    required_security_blocks: int = 0
    security_requirement_met: bool = True
    security_assessment_count: int = 0
    security_blocked_count: int = 0
    security_repeated_attempt_count: int = 0
    output_leakage_blocked_count: int = 0
    session_security_flagged: bool = False
    security_categories: list[str] = field(default_factory=list)
    security_actions: list[str] = field(default_factory=list)
    # Flag state as seen by THIS process at scenario start vs. after any
    # agent-flag management was restored. Disambiguates the old, misleading
    # single "adaptive flag: False" line.
    adaptive_flag_during_run: bool = False
    adaptive_flag_after_restoration: bool = False
    db_cleanup_status: str = "not_attempted"
    livekit_cleanup_status: str = "not_attempted"
    audio_dir: str | None = None
    audio_kept: bool = False
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


async def _provision_throwaway(db) -> tuple[uuid.UUID, uuid.UUID]:  # noqa: ANN001
    """Create a minimal published voice-capable config + student. Returns (config_id, student_id)."""
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()
    suffix = org_id.hex[:8]
    await db.execute(
        text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
        {"id": org_id, "slug": f"vh-{suffix}", "name": "Voice Harness Org"},
    )
    for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
        await db.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": uid, "email": f"vh-{label}-{suffix}@test.local"},
        )
    await db.execute(
        text(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
            "VALUES (:id, :org, :owner, :slug, 'VH Course', 'published')"
        ),
        {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"vh-course-{suffix}"},
    )
    await db.execute(
        text(
            "INSERT INTO modules (id, course_id, title, position, status) "
            "VALUES (:id, :course, 'Module', 1, 'published')"
        ),
        {"id": module_id, "course": course_id},
    )
    await db.execute(
        text(
            "INSERT INTO interview_configs "
            "(id, course_id, module_id, title, status, supported_modes, persona, created_by) "
            "VALUES (:id, :course, :module, 'VH Interview', 'published', 'voice', 'neutral', :t)"
        ),
        {"id": config_id, "course": course_id, "module": module_id, "t": teacher_id},
    )
    for i in range(2):
        oid = uuid.uuid4()
        await db.execute(
            text(
                "INSERT INTO interview_outcomes "
                "(id, interview_config_id, position, outcome_text, outcome_type, importance_weight) "
                "VALUES (:id, :cfg, :pos, :txt, 'knowledge', :w)"
            ),
            {"id": oid, "cfg": config_id, "pos": i + 1, "txt": f"Outcome {i + 1}", "w": 3 + i},
        )
        await db.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, linked_outcome_id, position, question_type, "
                "prompt_text, difficulty, review_status, ai_generated) "
                "VALUES (:id, :cfg, :oid, :pos, 'conceptual', :prompt, 'mid_level', 'approved', TRUE)"
            ),
            {
                "id": uuid.uuid4(),
                "cfg": config_id,
                "oid": oid,
                "pos": i + 1,
                "prompt": f"Question {i + 1}: explain a data warehousing concept?",
            },
        )
    await db.commit()
    logger.info("provisioned throwaway config=%s student=%s", config_id, student_id)
    return config_id, student_id


async def _create_voice_session(db, config_id: uuid.UUID, student_id: uuid.UUID) -> uuid.UUID:  # noqa: ANN001
    """Create an in_progress voice session with its first question attached."""
    session_id = uuid.uuid4()
    await db.execute(
        text(
            "INSERT INTO interview_sessions "
            "(id, interview_config_id, student_id, attempt_number, status, input_mode) "
            "VALUES (:id, :cfg, :s, 1, 'in_progress', 'voice')"
        ),
        {"id": session_id, "cfg": config_id, "s": student_id},
    )
    first_q = (
        await db.execute(
            text(
                "SELECT id FROM interview_questions WHERE interview_config_id = :c "
                "ORDER BY position LIMIT 1"
            ),
            {"c": config_id},
        )
    ).scalar_one()
    await db.execute(
        text(
            "INSERT INTO interview_session_questions "
            "(id, session_id, interview_question_id, sequence_no) VALUES (:id, :sid, :qid, 1)"
        ),
        {"id": uuid.uuid4(), "sid": session_id, "qid": first_q},
    )
    # Persist the deterministic room name so the token endpoint & agent agree.
    await db.execute(
        text("UPDATE interview_sessions SET livekit_room_name = :r WHERE id = :id"),
        {"r": rt_service.build_room_name(session_id), "id": session_id},
    )
    await db.commit()
    return session_id


async def _cleanup_throwaway(db, config_id: uuid.UUID) -> None:  # noqa: ANN001
    """Delete a throwaway config's full FK chain (child → parent order).

    Only safe for harness-provisioned data (``vh-`` orgs). Walks the FK graph
    explicitly because ``organizations`` → ``courses`` is NO ACTION, not CASCADE.
    """
    # Resolve the owning org/course/module from the config so we can unwind fully.
    row = (
        await db.execute(
            text(
                "SELECT c.id AS course_id, c.organization_id AS org_id, "
                "c.owner_user_id AS teacher_id, cf.module_id "
                "FROM interview_configs cf JOIN courses c ON c.id = cf.course_id "
                "WHERE cf.id = :cfg"
            ),
            {"cfg": config_id},
        )
    ).first()
    if row is None:
        return
    course_id, org_id, teacher_id, module_id = (
        row.course_id,
        row.org_id,
        row.teacher_id,
        row.module_id,
    )
    # Capture harness-created students before their sessions are deleted.
    student_ids = [
        r.student_id
        for r in (
            await db.execute(
                text(
                    "SELECT DISTINCT student_id FROM interview_sessions "
                    "WHERE interview_config_id = :c"
                ),
                {"c": config_id},
            )
        ).all()
    ]
    # Interview subtree (sessions → messages/questions/runtime state, then bank).
    await db.execute(
        text(
            "DELETE FROM interview_runtime_states WHERE session_id IN "
            "(SELECT id FROM interview_sessions WHERE interview_config_id = :c)"
        ),
        {"c": config_id},
    )
    for tbl in (
        "interview_session_messages",
        "interview_session_questions",
    ):
        await db.execute(
            text(
                f"DELETE FROM {tbl} WHERE session_id IN "  # noqa: S608 - fixed table names
                "(SELECT id FROM interview_sessions WHERE interview_config_id = :c)"
            ),
            {"c": config_id},
        )
    await db.execute(
        text("DELETE FROM interview_sessions WHERE interview_config_id = :c"), {"c": config_id}
    )
    await db.execute(
        text("DELETE FROM interview_questions WHERE interview_config_id = :c"), {"c": config_id}
    )
    await db.execute(
        text("DELETE FROM interview_outcomes WHERE interview_config_id = :c"), {"c": config_id}
    )
    await db.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
    await db.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
    await db.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
    # Users are resolved before deleting courses/sessions. Delete only those
    # exact harness-created ids; never infer a broad organization-wide target.
    user_ids = [teacher_id, *student_ids]
    for user_id in user_ids:
        await db.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})
    await db.execute(
        text("DELETE FROM organizations WHERE id = :o AND slug LIKE 'vh-%'"), {"o": org_id}
    )
    await db.commit()
    logger.info("cleaned up throwaway config=%s (org=%s)", config_id, org_id)


def _require_credentials(settings: Any) -> None:  # noqa: ANN401 - Settings duck-typed
    """Fail fast with a clear message if LiveKit or model creds are missing.

    A live voice run needs BOTH the LiveKit room infra (to join + dispatch the
    agent) and the OpenAI-compatible gateway (STT for the student clips, TTS for
    both the student clips and the agent's replies). We check them explicitly so
    a missing secret fails here instead of deep inside an SDK call.
    """
    missing: list[str] = []
    if not settings.livekit_ws_url:
        missing.append("LIVEKIT_WS_URL")
    if not settings.livekit_api_key:
        missing.append("LIVEKIT_API_KEY")
    if not settings.livekit_api_secret:
        missing.append("LIVEKIT_API_SECRET")
    if not (settings.llm_base_url or "").strip():
        missing.append("LLM_BASE_URL")
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY (used for STT + TTS)")
    if not settings.interview_voice_enabled:
        missing.append("INTERVIEW_VOICE_ENABLED=true")
    if missing:
        raise HarnessError(
            "missing required credentials/flags for a live voice run: "
            + ", ".join(missing)
        )


async def _collect_db_signals(session_id: uuid.UUID) -> dict[str, Any]:
    """Read the authoritative DB signals for a finished scenario.

    Returns message rows (role + text + adaptive action, when present) and the
    runtime-state row count (>0 iff the adaptive brain actually ran).
    """
    async with get_sessionmaker()() as db:
        msg_rows = (
            await db.execute(
                text(
                    "SELECT role, "
                    "metadata_json->>'kind' AS kind, "
                    "metadata_json->>'action' AS action, "
                    "metadata_json->>'security_action' AS security_action, "
                    "metadata_json->>'security_category' AS security_category, "
                    "metadata_json->>'utterance_status' AS utterance_status, "
                    "left(content_text, 200) AS txt, "
                    "created_at "
                    "FROM interview_session_messages "
                    "WHERE session_id = :s ORDER BY created_at"
                ),
                {"s": session_id},
            )
        ).all()
        # state_version is the authoritative optimistic-lock counter; NULL row
        # (no adaptive run) → None.
        rt_row = (
            await db.execute(
                text(
                    "SELECT count(*) AS n, max(state_version) AS ver "
                    "FROM interview_runtime_states WHERE session_id = :s"
                ),
                {"s": session_id},
            )
        ).one()
        rt_count = int(rt_row.n)
        state_version = int(rt_row.ver) if rt_row.ver is not None else None
        # Selected next-question ids, in ask order (what the adaptive selector
        # chose to ask). sequence_no orders the session's question ledger.
        sq_rows = (
            await db.execute(
                text(
                    "SELECT interview_question_id FROM interview_session_questions "
                    "WHERE session_id = :s AND interview_question_id IS NOT NULL "
                    "ORDER BY sequence_no"
                ),
                {"s": session_id},
            )
        ).all()
        security_row = (
            await db.execute(
                text(
                    "SELECT "
                    "count(*) FILTER (WHERE event_type = 'interview.security.assessed') "
                    "AS assessed, "
                    "count(*) FILTER (WHERE event_type = 'interview.security.blocked') "
                    "AS blocked, "
                    "count(*) FILTER (WHERE event_type = 'interview.security.repeated_attempt') "
                    "AS repeated_attempt, "
                    "count(*) FILTER (WHERE event_type = "
                    "'interview.security.output_leakage_blocked') AS output_leakage, "
                    "array_remove(array_agg(DISTINCT category), NULL) AS categories, "
                    "array_remove(array_agg(DISTINCT action), NULL) AS actions "
                    "FROM interview_security_events WHERE session_id = :s"
                ),
                {"s": session_id},
            )
        ).one()
        session_flagged = bool(
            (
                await db.execute(
                    text(
                        "SELECT session_security_flagged FROM interview_sessions "
                        "WHERE id = :s"
                    ),
                    {"s": session_id},
                )
            ).scalar_one()
        )
    transcript = [
        {
            "role": r.role,
            "action": r.action or "",
            "security_action": r.security_action or "",
            "security_category": r.security_category or "",
            "text": r.txt or "",
        }
        for r in msg_rows
    ]
    actions = [r["action"] for r in transcript if r["action"]]
    # Decisions + fallbacks come from the persisted AI-turn metadata (the source
    # of truth), NOT from utterance text. An adaptive AI turn is one tagged
    # kind="adaptive"; it carries a structured action and an utterance_status
    # that is "llm" (genuine phrasing) or "fallback" (deterministic template).
    adaptive_ai = [r for r in msg_rows if r.kind == "adaptive"]
    decision_count = sum(1 for r in adaptive_ai if r.action)
    fallback_count = sum(1 for r in adaptive_ai if r.utterance_status == "fallback")
    # AI-turn commit timestamps (created_at) — a DB-side proxy for when each
    # adaptive decision was persisted. ISO-8601 UTC for cross-comparability.
    ai_turn_committed_at = [
        r.created_at.isoformat() for r in msg_rows if r.role == "ai" and r.created_at
    ]
    return {
        "transcript": transcript,
        "message_count": len(transcript),
        "runtime_state_count": rt_count,
        "adaptive_actions": actions,
        "state_version": state_version,
        "decision_count": decision_count,
        "fallback_count": fallback_count,
        "selected_question_ids": [str(r.interview_question_id) for r in sq_rows],
        "ai_turn_committed_at": ai_turn_committed_at,
        "security_assessment_count": int(security_row.assessed or 0),
        "security_blocked_count": int(security_row.blocked or 0),
        "security_repeated_attempt_count": int(security_row.repeated_attempt or 0),
        "output_leakage_blocked_count": int(security_row.output_leakage or 0),
        "session_security_flagged": session_flagged,
        "security_categories": sorted(str(v) for v in (security_row.categories or [])),
        "security_actions": sorted(str(v) for v in (security_row.actions or [])),
    }


def _iso(dt: Any) -> str | None:  # noqa: ANN401 - datetime | None
    return dt.isoformat() if dt is not None else None


def _delta_s(later: Any, earlier: Any) -> float | None:  # noqa: ANN401
    """Seconds between two datetimes, or None if either is missing/negative."""
    if later is None or earlier is None:
        return None
    delta = (later - earlier).total_seconds()
    return round(delta, 3) if delta >= 0 else None


def _build_timeline_and_latency(
    room: Any,  # noqa: ANN401 - HarnessRoom
    signals: dict[str, Any],
) -> tuple[HarnessTimeline, LatencyMetrics]:
    """Assemble the observable event timeline + derived latencies.

    Client-observable events come from the room object; the AI-turn commit
    times come from the DB (a proxy for decision-complete). STT-final and the
    agent's INTERNAL decision-complete instants are not observable from outside
    the agent, so they stay null and are called out in ``notes``.
    """
    cap = room.capture
    first_turn = room.student_turns[0] if room.student_turns else None
    last_turn = room.student_turns[-1] if room.student_turns else None

    timeline = HarnessTimeline(
        room_joined_at=_iso(room.room_joined_at),
        agent_joined_at=_iso(room.agent_joined_at),
        first_student_audio_started_at=_iso(first_turn["started_at"]) if first_turn else None,
        last_student_audio_ended_at=_iso(last_turn["ended_at"]) if last_turn else None,
        agent_audio_first_frame_at=_iso(cap.first_frame_at),
        agent_audio_first_frame_after_last_prompt_at=_iso(cap.first_frame_after_prompt_at),
        agent_audio_ended_at=_iso(cap.last_frame_at),
        disconnected_at=_iso(room.disconnected_at),
        ai_turn_committed_at=list(signals.get("ai_turn_committed_at", [])),
    )

    # Honest turn-latency proxy: end of the last student turn → the LAST AI-turn
    # DB commit (created_at). Parse the last ai_turn_committed_at ISO string.
    last_ai_commit_dt: Any = None
    committed = signals.get("ai_turn_committed_at", [])
    if committed:
        with contextlib.suppress(Exception):
            last_ai_commit_dt = datetime.fromisoformat(committed[-1])
    last_speech_end = last_turn["ended_at"] if last_turn else None

    latency = LatencyMetrics(
        end_of_speech_to_decision_committed_s=_delta_s(last_ai_commit_dt, last_speech_end),
        end_of_speech_to_agent_audio_s=_delta_s(
            cap.first_frame_after_prompt_at,
            last_speech_end,
        ),
        agent_audio_span_s=_delta_s(cap.last_frame_at, cap.first_frame_at),
        room_join_to_agent_join_s=_delta_s(room.agent_joined_at, room.room_joined_at),
        total_scenario_s=_delta_s(room.disconnected_at, room.room_joined_at),
        notes={
            "end_of_speech_to_decision_committed_s": (
                "DB-PROXY (preferred turn latency): end of the LAST published "
                "student turn → last AI-turn row created_at. Folds STT + decision "
                "+ phrasing up to persistence; excludes TTS-audio playout."
            ),
            "end_of_speech_to_agent_audio_s": (
                "UNRELIABLE: the agent publishes a CONTINUOUS audio track, so "
                "'first frame after prompt' catches the next streamed frame "
                "(~ms), NOT the reply onset. Do not trust as turn latency; "
                "prefer end_of_speech_to_decision_committed_s."
            ),
            "stt_final_at": "agent-internal; not observable from the harness (null).",
            "decision_completed_at": (
                "agent-internal; ai_turn_committed_at (DB created_at) is the "
                "closest observable proxy."
            ),
            "agent_audio_span_s": "CLIENT-observable: first→last agent frame across the session.",
        },
    )
    return timeline, latency


async def _delete_livekit_room(room_name: str, settings: Any) -> str:  # noqa: ANN401
    """Best-effort delete the LiveKit room so no server-side room lingers.

    Returns a status string for the machine-readable result. Never raises — a
    cleanup failure must not mask the scenario outcome.
    """
    try:
        from livekit import api  # noqa: PLC0415

        lk = api.LiveKitAPI(
            url=settings.livekit_ws_url,
            api_key=settings.livekit_api_key.get_secret_value()
            if hasattr(settings.livekit_api_key, "get_secret_value")
            else settings.livekit_api_key,
            api_secret=settings.livekit_api_secret.get_secret_value()
            if hasattr(settings.livekit_api_secret, "get_secret_value")
            else settings.livekit_api_secret,
        )
        try:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
        finally:
            await lk.aclose()
        logger.info("deleted LiveKit room=%s", room_name)
        return "deleted"
    except Exception as exc:  # noqa: BLE001 - best-effort, report status only
        logger.warning("LiveKit room cleanup failed for %s: %s", room_name, exc)
        return f"failed: {exc}"


def _peak_amplitude(pcm) -> int:  # noqa: ANN001 - np.ndarray
    import numpy as np  # noqa: PLC0415

    if pcm is None or len(pcm) == 0:
        return 0
    return int(np.abs(pcm.astype(np.int64)).max())


def _strict_adaptive_ok(result: ScenarioResult) -> bool:
    """Decide whether a run genuinely exercised the adaptive brain.

    Source of truth is PERSISTED state + decision metadata (never the spoken
    utterance text). ALL of the following must hold:

    * a runtime-state row exists (``runtime_state_count >= 1``);
    * the adaptive path ran (``adaptive_ran``);
    * at least one structured adaptive action was persisted
      (``decision_count >= 1`` and ``adaptive_actions`` non-empty);
    * the runtime-state version is non-null (state actually advanced);
    * NOT every persisted adaptive turn was a deterministic fallback — a run
      where every turn fell back is not a real adaptive exercise.
    """
    if result.runtime_state_count < 1 or not result.adaptive_ran:
        return False
    if result.decision_count < 1 or not result.adaptive_actions:
        return False
    if result.state_version is None:
        return False
    # If there were decisions and ALL of them fell back, reject: no genuine
    # adaptive phrasing happened on any turn.
    return not (result.decision_count > 0 and result.fallback_count >= result.decision_count)


def _strict_failure_reason(result: ScenarioResult) -> str:
    """Human-actionable explanation of why strict adaptive verification failed.

    The most common cause by far is the voice-agent process running with the
    adaptive feature flag disabled, so lead with that.
    """
    if result.runtime_state_count < 1 or not result.adaptive_ran:
        return (
            "strict adaptive verification FAILED: no interview_runtime_states row "
            "was created, so the adaptive brain never ran. The most likely cause is "
            "that the LiveKit voice-agent process (pm2 abridgeai-interview-agent) was "
            "started with the adaptive feature flag DISABLED. Restart it with "
            "ADAPTIVE_INTERVIEWER_ENABLED=true AND "
            "ADAPTIVE_INTERVIEWER_VOICE_ENABLED=true, then re-run."
        )
    if result.decision_count < 1 or not result.adaptive_actions:
        return (
            "strict adaptive verification FAILED: a runtime-state row exists but no "
            "structured adaptive action/decision was persisted on any AI turn."
        )
    if result.state_version is None:
        return "strict adaptive verification FAILED: runtime-state version is null."
    return (
        "strict adaptive verification FAILED: every adaptive turn used the "
        f"deterministic fallback ({result.fallback_count}/{result.decision_count}); "
        "no genuine adaptive phrasing occurred (check the LLM gateway)."
    )


def _security_requirement_ok(result: ScenarioResult) -> bool:
    """Check the persisted minimum block count requested by the caller."""
    if result.required_security_blocks <= 0:
        return True
    return (
        result.security_assessment_count >= result.required_security_blocks
        and result.security_blocked_count >= result.required_security_blocks
    )


def _security_failure_reason(result: ScenarioResult) -> str:
    return (
        "security verification FAILED: required at least "
        f"{result.required_security_blocks} assessed+blocked voice turn(s), but persisted "
        f"assessed={result.security_assessment_count}, blocked={result.security_blocked_count}. "
        "Confirm INTERVIEW_SECURITY_GUARD_MODE=enforce in both backend and "
        "abridgeai-interview-agent, then inspect STT transcripts for recognition errors."
    )


async def _drive_scenario(
    *,
    settings: Any,  # noqa: ANN401
    result: ScenarioResult,
    session_id: uuid.UUID,
    student_id: uuid.UUID,
    language: str,
    answers: list[str],
    out_dir: Path,
    agent_timeout: float,
    turn_gap: float,
    reply_wait: float,
    barge_in: bool,
    room_holder: dict[str, Any],
) -> None:
    """The actual live interaction. Mutates ``result`` in place. Raises on hard
    failure (missing agent) so the caller's finally-block still runs cleanup.

    ``room_holder`` receives the ``HarnessRoom`` under key ``"room"`` as soon as
    it is created, so the caller can build the timeline from its observed
    timestamps even if the scenario later times out or errors."""
    token_resp = rt_service.mint_participant_token(
        session_id=session_id,
        student_id=student_id,
        room_name=rt_service.build_room_name(session_id),
        settings=settings,
        language=language,
    )

    logger.info("synthesizing %d student answer clip(s) via gateway TTS…", len(answers))
    answer_pcms = []
    for i, ans in enumerate(answers):
        mp3 = await synthesize_speech(ans, persona="neutral", settings=settings)
        pcm = audio_io.decode_to_pcm48k_mono(mp3)
        answer_pcms.append(pcm)
        logger.info("  clip %d: %.2fs (%r)", i + 1, audio_io.pcm_duration_seconds(pcm), ans[:40])

    room = HarnessRoom(url=token_resp.url, token=token_resp.token)
    room_holder["room"] = room  # expose for timeline even on timeout/error
    await room.connect()
    try:
        if not await room.wait_for_agent(timeout=agent_timeout):
            raise HarnessError(f"agent never joined within {agent_timeout}s")
        logger.info("agent joined; waiting for first question audio…")
        await asyncio.sleep(turn_gap)

        for i, pcm in enumerate(answer_pcms):
            logger.info("→ speaking student answer %d/%d", i + 1, len(answer_pcms))
            if barge_in and i == 0:
                logger.info("  (barge-in mode: starting during agent speech)")
            else:
                await asyncio.sleep(turn_gap)
            await room.speak_pcm(pcm)
            await asyncio.sleep(reply_wait)

        # Allow a final closing utterance to arrive.
        await asyncio.sleep(reply_wait)

        utterances = await room.capture.finalize()
        sr = room.capture.sample_rate
    finally:
        await room.disconnect()

    logger.info("captured %d agent utterance(s) @ %d Hz", len(utterances), sr)
    total_samples = 0
    peak = 0
    for i, u in enumerate(utterances):
        path = out_dir / f"agent_utterance_{i + 1:02d}.wav"
        audio_io.write_wav(str(path), u, sample_rate=sr)
        dur = audio_io.pcm_duration_seconds(u, sample_rate=sr)
        total_samples += len(u)
        peak = max(peak, _peak_amplitude(u))
        logger.info("  wrote %s (%.2fs)", path.name, dur)

    result.agent_utterance_count = len(utterances)
    result.captured_audio_seconds = round(total_samples / float(sr), 3) if sr else 0.0
    result.peak_amplitude = peak


async def run_scenario(
    *,
    language: str = "en",
    answers: list[str] | None = None,
    out_dir: str = "./voice-harness-out",
    config_id: str | None = None,
    student_id: str | None = None,
    agent_timeout: float = 30.0,
    turn_gap: float = 9.0,
    reply_wait: float = 12.0,
    barge_in: bool = False,
    cleanup_db: bool = True,
    cleanup_livekit: bool = True,
    delete_audio_on_success: bool = False,
    scenario_timeout_s: float = DEFAULT_SCENARIO_TIMEOUT_S,
    require_adaptive: bool = False,
    require_security_blocks: int = 0,
) -> ScenarioResult:
    """Run one live voice scenario end-to-end and return a machine-readable result.

    This is the canonical entry point invoked by BOTH the CLI (``main``) and the
    pytest wrapper — no harness logic is duplicated. Cleanup (DB + LiveKit room)
    always runs in ``finally``. Audio is kept on failure; optionally deleted on
    success.
    """
    settings = get_settings()
    answers = answers or ["A fact table stores quantitative measurements for analysis."]
    adaptive_during = bool(getattr(settings, "adaptive_interviewer_enabled", False))
    result = ScenarioResult(
        ok=False,
        language=language,
        adaptive_required=require_adaptive,
        required_security_blocks=max(0, require_security_blocks),
        adaptive_flag_during_run=adaptive_during,
        adaptive_flag_after_restoration=adaptive_during,
        audio_dir=str(Path(out_dir).resolve()),
    )

    # Fail fast on missing creds BEFORE we provision anything.
    _require_credentials(settings)
    if not adaptive_during:
        logger.warning(
            "adaptive_interviewer_enabled is OFF in THIS process — note the agent "
            "process governs the live path; runtime_state_count in the result is the "
            "authoritative 'did adaptive run' signal."
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sessionmaker = get_sessionmaker()
    provisioned = False
    prov_config_id: uuid.UUID | None = None
    session_uuid: uuid.UUID | None = None
    room_name: str | None = None
    room_holder: dict[str, Any] = {}
    signals: dict[str, Any] = {}

    try:
        async with sessionmaker() as db:
            if config_id and student_id:
                cfg_uuid = uuid.UUID(config_id)
                stu_uuid = uuid.UUID(student_id)
            else:
                cfg_uuid, stu_uuid = await _provision_throwaway(db)
                provisioned = True
                prov_config_id = cfg_uuid
            session_uuid = await _create_voice_session(db, cfg_uuid, stu_uuid)

        result.config_id = str(cfg_uuid)
        result.session_id = str(session_uuid)
        room_name = rt_service.build_room_name(session_uuid)
        logger.info("voice session=%s (config=%s)", session_uuid, cfg_uuid)

        # Hard scenario timeout wraps the entire live interaction.
        await asyncio.wait_for(
            _drive_scenario(
                settings=settings,
                result=result,
                session_id=session_uuid,
                student_id=stu_uuid,
                language=language,
                answers=answers,
                out_dir=out_path,
                agent_timeout=agent_timeout,
                turn_gap=turn_gap,
                reply_wait=reply_wait,
                barge_in=barge_in,
                room_holder=room_holder,
            ),
            timeout=scenario_timeout_s,
        )

        # Authoritative DB signals (adaptive ran? how many turns?).
        signals = await _collect_db_signals(session_uuid)
        result.transcript = signals.get("transcript", [])
        result.message_count = signals.get("message_count", 0)
        result.runtime_state_count = signals.get("runtime_state_count", 0)
        result.adaptive_actions = signals.get("adaptive_actions", [])
        result.adaptive_ran = signals.get("runtime_state_count", 0) > 0
        result.state_version = signals.get("state_version")
        result.decision_count = signals.get("decision_count", 0)
        result.fallback_count = signals.get("fallback_count", 0)
        result.selected_question_ids = signals.get("selected_question_ids", [])
        result.security_assessment_count = signals.get("security_assessment_count", 0)
        result.security_blocked_count = signals.get("security_blocked_count", 0)
        result.security_repeated_attempt_count = signals.get(
            "security_repeated_attempt_count", 0
        )
        result.output_leakage_blocked_count = signals.get(
            "output_leakage_blocked_count", 0
        )
        result.session_security_flagged = signals.get("session_security_flagged", False)
        result.security_categories = signals.get("security_categories", [])
        result.security_actions = signals.get("security_actions", [])

        # Baseline verdict: audio produced in both directions AND turns persisted.
        result.ok = (
            result.agent_utterance_count >= 1
            and result.captured_audio_seconds > 0
            and result.message_count >= 1
        )

        # Strict adaptive verification (layered on top). The source of truth is
        # PERSISTED state + decision metadata — never the utterance text.
        if result.adaptive_required:
            result.adaptive_requirement_met = _strict_adaptive_ok(result)
            if not result.adaptive_requirement_met:
                result.ok = False
                if result.error is None:
                    result.error = _strict_failure_reason(result)
        result.security_requirement_met = _security_requirement_ok(result)
        if not result.security_requirement_met:
            result.ok = False
            if result.error is None:
                result.error = _security_failure_reason(result)
    except TimeoutError:
        result.error = f"scenario exceeded hard timeout of {scenario_timeout_s}s"
        logger.error(result.error)
    except HarnessError as exc:
        result.error = str(exc)
        logger.error("harness error: %s", exc)
    except Exception as exc:  # noqa: BLE001 - report, don't crash before cleanup
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("scenario failed")
    finally:
        # Re-read the flag as this process now sees it (unchanged by us — we do
        # not manage the agent flag here; documented in the summary line).
        result.adaptive_flag_after_restoration = bool(
            getattr(get_settings(), "adaptive_interviewer_enabled", False)
        )
        # ── Diagnostic timeline + latencies (best-effort; populated even if the
        # scenario timed out or errored, from whatever the room observed) ─────
        room_obj = room_holder.get("room")
        if room_obj is not None:
            with contextlib.suppress(Exception):
                result.timeline, result.latency = _build_timeline_and_latency(
                    room_obj, signals
                )
        # ── DB cleanup (only throwaway data we provisioned) ──────────────────
        if provisioned and prov_config_id is not None and cleanup_db:
            try:
                async with sessionmaker() as db:
                    await _cleanup_throwaway(db, prov_config_id)
                result.db_cleanup_status = "cleaned"
            except Exception as exc:  # noqa: BLE001
                result.db_cleanup_status = f"failed: {exc}"
                logger.warning("DB cleanup failed: %s", exc)
        elif provisioned and not cleanup_db:
            result.db_cleanup_status = "kept (cleanup_db=False)"
        elif not provisioned:
            result.db_cleanup_status = "n/a (used existing config)"
        # ── LiveKit room cleanup ─────────────────────────────────────────────
        if room_name and cleanup_livekit:
            result.livekit_cleanup_status = await _delete_livekit_room(room_name, settings)
        elif room_name:
            result.livekit_cleanup_status = "kept (cleanup_livekit=False)"
        # ── Audio artifacts: keep on failure, optionally delete on success ────
        if result.ok and delete_audio_on_success:
            with contextlib.suppress(Exception):
                shutil.rmtree(out_path, ignore_errors=True)
            result.audio_kept = False
        else:
            result.audio_kept = True

    return result


def _print_summary(result: ScenarioResult) -> None:
    print(f"\n=== HARNESS {'PASS' if result.ok else 'FAIL'} ===")
    print(f"session_id                     : {result.session_id}")
    print(f"language                       : {result.language}")
    print(f"persisted messages             : {result.message_count}")
    print(f"runtime-state rows             : {result.runtime_state_count}")
    print(f"adaptive ran (rows>0)          : {result.adaptive_ran}")
    print(f"adaptive actions               : {result.adaptive_actions}")
    print(f"state version                  : {result.state_version}")
    print(f"decision count                 : {result.decision_count}")
    print(f"fallback count                 : {result.fallback_count}")
    print(f"adaptive REQUIRED (strict)     : {result.adaptive_required}")
    print(f"adaptive requirement met       : {result.adaptive_requirement_met}")
    print(f"security blocks REQUIRED       : {result.required_security_blocks}")
    print(f"security requirement met       : {result.security_requirement_met}")
    print(f"security assessed              : {result.security_assessment_count}")
    print(f"security blocked               : {result.security_blocked_count}")
    print(f"security repeated attempts     : {result.security_repeated_attempt_count}")
    print(f"output leakage blocked         : {result.output_leakage_blocked_count}")
    print(f"session security flagged       : {result.session_security_flagged}")
    print(f"security categories            : {result.security_categories}")
    print(f"security actions               : {result.security_actions}")
    print(f"agent utterances               : {result.agent_utterance_count}")
    print(f"captured audio (s)             : {result.captured_audio_seconds}")
    print(f"peak amplitude                 : {result.peak_amplitude}")
    print(f"adaptive flag DURING run       : {result.adaptive_flag_during_run}")
    print(f"adaptive flag AFTER restoration: {result.adaptive_flag_after_restoration}")
    print(f"db cleanup                     : {result.db_cleanup_status}")
    print(f"livekit cleanup                : {result.livekit_cleanup_status}")
    print(f"audio kept                     : {result.audio_kept} ({result.audio_dir})")
    print(f"result schema version          : {result.result_schema_version}")
    print(f"selected question ids          : {result.selected_question_ids}")
    lat = result.latency
    print(f"latency room→agent join (s)    : {lat.room_join_to_agent_join_s}")
    print(f"latency end-of-speech→decision : {lat.end_of_speech_to_decision_committed_s} (DB proxy, preferred)")
    print(f"latency agent audio span (s)   : {lat.agent_audio_span_s}")
    print(f"latency total scenario (s)     : {lat.total_scenario_s}")
    if result.error:
        print(f"error                          : {result.error}")


async def run(args: argparse.Namespace) -> int:
    result = await run_scenario(
        language=args.language,
        answers=args.answers,
        out_dir=args.out,
        config_id=args.config_id,
        student_id=args.student_id,
        agent_timeout=args.agent_timeout,
        turn_gap=args.turn_gap,
        reply_wait=args.reply_wait,
        barge_in=args.barge_in,
        cleanup_db=args.cleanup,
        cleanup_livekit=args.cleanup,
        delete_audio_on_success=args.delete_audio_on_success,
        scenario_timeout_s=args.scenario_timeout,
        require_adaptive=args.require_adaptive,
        require_security_blocks=args.require_security_blocks,
    )
    _print_summary(result)
    if args.json:
        # The result JSON must survive --delete-audio-on-success (which removes
        # the capture dir). Ensure its parent exists so a json path INSIDE the
        # audio dir still writes rather than crashing after cleanup.
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(result.to_json(), encoding="utf-8")
        logger.info("wrote machine-readable result to %s", args.json)
    return 0 if result.ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Semi-automated voice interview harness")
    p.add_argument("--config-id", help="Existing interview_config_id (published, voice-capable)")
    p.add_argument("--student-id", help="Existing user id to act as the student")
    p.add_argument("--language", default="en", choices=["en", "vi"], help="Utterance language")
    p.add_argument(
        "--answers",
        nargs="+",
        default=["A fact table stores quantitative measurements for analysis."],
        help="Student answer texts (each becomes one spoken turn)",
    )
    p.add_argument("--out", default="./voice-harness-out", help="Output dir for captured WAVs")
    p.add_argument("--agent-timeout", type=float, default=30.0, help="Seconds to wait for agent join")
    p.add_argument("--turn-gap", type=float, default=4.0, help="Seconds to let the agent speak before answering")
    p.add_argument("--reply-wait", type=float, default=8.0, help="Seconds to wait for agent reply after an answer")
    p.add_argument("--barge-in", action="store_true", help="Start the first answer during agent speech")
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete throwaway DB data AND the LiveKit room after the run "
        "(only DB data the harness itself provisioned)",
    )
    p.add_argument(
        "--delete-audio-on-success",
        action="store_true",
        help="Delete the captured WAV dir when the run PASSes (kept on failure regardless)",
    )
    p.add_argument(
        "--scenario-timeout",
        type=float,
        default=DEFAULT_SCENARIO_TIMEOUT_S,
        help=f"Hard ceiling (s) on the whole scenario (default {DEFAULT_SCENARIO_TIMEOUT_S})",
    )
    p.add_argument("--json", help="Write the machine-readable ScenarioResult JSON to this path")
    p.add_argument(
        "--require-adaptive",
        action="store_true",
        default=os.getenv("REQUIRE_ADAPTIVE_VOICE") == "1",
        help="Strict mode: FAIL the run unless the adaptive brain genuinely ran "
        "(runtime-state row + persisted decision/action + non-null version + not "
        "all-fallback). Defaults on when REQUIRE_ADAPTIVE_VOICE=1.",
    )
    p.add_argument(
        "--require-security-blocks",
        type=int,
        default=0,
        help="FAIL unless at least this many redacted security assessed+blocked "
        "events were persisted (use with spoken injection fixtures in enforce mode)",
    )
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
