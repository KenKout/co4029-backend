"""Verifies the BEFORE UPDATE trigger from migration 0012 fires on every
``TimestampMixin`` table when an UPDATE is issued via raw SQL (the path
that bypasses SQLAlchemy's ``onupdate=NOW()`` hook).

Strategy: INSERT a minimum row, then issue ``UPDATE <t> SET updated_at =
'2020-01-01' WHERE <pk> = ...`` (deliberately a stale literal -- the
trigger MUST overwrite it with NOW()). Re-read and assert the stored
value is not the stale literal.

Each test case runs inside a transaction that rolls back on exit, so
seeded fixture rows leave no residue.
"""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from abridgeai.core.config import get_settings


def _load_timestamp_tables() -> tuple[str, ...]:
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "0012_updated_at_trigger.py"
    )
    spec = importlib.util.spec_from_file_location("_mig_0012", migration_path)
    assert spec is not None, f"cannot load {migration_path}"
    assert spec.loader is not None, f"cannot load {migration_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.TIMESTAMP_TABLES)


TIMESTAMP_TABLES = _load_timestamp_tables()


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


PrimaryKey = dict[str, Any]
RowFactory = Callable[[AsyncConnection], Awaitable[PrimaryKey]]


async def _insert_user(conn: AsyncConnection) -> PrimaryKey:
    uid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
        {"id": uid, "email": f"trig-{uid.hex[:10]}@test.local"},
    )
    return {"id": uid}


async def _insert_organization(conn: AsyncConnection) -> PrimaryKey:
    oid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
        {"id": oid, "slug": f"trig-{oid.hex[:10]}", "name": "trig org"},
    )
    return {"id": oid}


async def _new_user(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_user(conn))["id"]


async def _new_organization(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_organization(conn))["id"]


async def _insert_org_unit(conn: AsyncConnection) -> PrimaryKey:
    org_id = await _new_organization(conn)
    uid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
            # 0094_flat_faculties: ck_org_units_live_faculty_root requires
            # every LIVE unit to be a top-level faculty.
            "VALUES (:id, :org, 'faculty', :name, :code)"
        ),
        {"id": uid, "org": org_id, "name": "trig unit", "code": f"u-{uid.hex[:8]}"},
    )
    return {"id": uid}


async def _insert_course(conn: AsyncConnection) -> PrimaryKey:
    org_id = await _new_organization(conn)
    owner_id = await _new_user(conn)
    cid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
            "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
        ),
        {
            "id": cid,
            "org": org_id,
            "owner": owner_id,
            "slug": f"c-{cid.hex[:10]}",
            "title": "trig course",
        },
    )
    return {"id": cid}


async def _new_course(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_course(conn))["id"]


async def _insert_module(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    mid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO modules (id, course_id, position, title) VALUES (:id, :cid, 1, :title)"),
        {"id": mid, "cid": course_id, "title": "trig module"},
    )
    return {"id": mid}


async def _new_module(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_module(conn))["id"]


async def _insert_lesson(conn: AsyncConnection) -> PrimaryKey:
    module_id = await _new_module(conn)
    lid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO lessons (id, module_id, slug, title) VALUES (:id, :mid, :slug, :title)"),
        {
            "id": lid,
            "mid": module_id,
            "slug": f"l-{lid.hex[:10]}",
            "title": "trig lesson",
        },
    )
    return {"id": lid}


async def _new_lesson(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_lesson(conn))["id"]


async def _insert_quiz(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    module_id = await _new_module(conn)
    qid = uuid.uuid4()
    await conn.execute(
        text(
                "INSERT INTO quizzes (id, course_id, module_id, title, slug) VALUES (:id, :cid, :mid, :title, 'slug-' || uuid_generate_v4()::text);"
            ),
        {"id": qid, "cid": course_id, "mid": module_id, "title": "trig quiz"},
    )
    return {"id": qid}


async def _new_quiz(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_quiz(conn))["id"]


async def _insert_quiz_question(conn: AsyncConnection) -> PrimaryKey:
    quiz_id = await _new_quiz(conn)
    qid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO quiz_questions "
            "(id, quiz_id, position, question_type, prompt_text) "
            "VALUES (:id, :quiz, 1, 'multiple_choice', :prompt)"
        ),
        {"id": qid, "quiz": quiz_id, "prompt": "trig prompt?"},
    )
    return {"id": qid}


async def _new_quiz_question(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_quiz_question(conn))["id"]


async def _insert_quiz_attempt(conn: AsyncConnection) -> PrimaryKey:
    quiz_id = await _new_quiz(conn)
    student_id = await _new_user(conn)
    aid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO quiz_attempts "
            "(id, quiz_id, student_id, attempt_number, status) "
            "VALUES (:id, :quiz, :student, 1, 'in_progress')"
        ),
        {"id": aid, "quiz": quiz_id, "student": student_id},
    )
    return {"id": aid}


async def _insert_career_path(conn: AsyncConnection) -> PrimaryKey:
    org_id = await _new_organization(conn)
    cpid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO career_paths (id, organization_id, slug, name) "
            "VALUES (:id, :org, :slug, :name)"
        ),
        {"id": cpid, "org": org_id, "slug": f"cp-{cpid.hex[:10]}", "name": "trig path"},
    )
    return {"id": cpid}


async def _new_career_path(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_career_path(conn))["id"]


async def _insert_interview_config(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    module_id = await _new_module(conn)
    iid = uuid.uuid4()
    await conn.execute(
        text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, slug) VALUES (:id, :cid, :mid, :title, 'slug-' || uuid_generate_v4()::text);"
            ),
        {"id": iid, "cid": course_id, "mid": module_id, "title": "trig ic"},
    )
    return {"id": iid}


async def _new_interview_config(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_interview_config(conn))["id"]


async def _insert_interview_outcome(conn: AsyncConnection) -> PrimaryKey:
    config_id = await _new_interview_config(conn)
    oid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO interview_outcomes "
            "(id, interview_config_id, position, outcome_text, outcome_type) "
            "VALUES (:id, :cfg, 1, :txt, 'knowledge')"
        ),
        {"id": oid, "cfg": config_id, "txt": "trig outcome text"},
    )
    return {"id": oid}


async def _new_interview_outcome(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_interview_outcome(conn))["id"]


async def _insert_interview_session(conn: AsyncConnection) -> PrimaryKey:
    config_id = await _new_interview_config(conn)
    student_id = await _new_user(conn)
    sid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO interview_sessions "
            "(id, interview_config_id, student_id, attempt_number, status, input_mode) "
            "VALUES (:id, :cfg, :student, 1, 'in_progress', 'text')"
        ),
        {"id": sid, "cfg": config_id, "student": student_id},
    )
    return {"id": sid}


async def _new_interview_session(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_interview_session(conn))["id"]


async def _insert_learning_material(conn: AsyncConnection) -> PrimaryKey:
    lesson_id = await _new_lesson(conn)
    mid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO learning_materials "
            "(id, lesson_id, title, material_type) "
            "VALUES (:id, :lid, :title, 'text')"
        ),
        {"id": mid, "lid": lesson_id, "title": "trig material"},
    )
    return {"id": mid}


async def _new_learning_material(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_learning_material(conn))["id"]


async def _insert_storage_object(conn: AsyncConnection) -> PrimaryKey:
    sid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO storage_objects "
            "(id, bucket, object_key, uploaded_at) "
            "VALUES (:id, 'test', :key, NOW())"
        ),
        {"id": sid, "key": f"trig/{sid.hex[:10]}"},
    )
    return {"id": sid}


async def _new_storage_object(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_storage_object(conn))["id"]


async def _insert_learning_material_version(conn: AsyncConnection) -> PrimaryKey:
    material_id = await _new_learning_material(conn)
    storage_object_id = await _new_storage_object(conn)
    vid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO learning_material_versions "
            "(id, material_id, storage_object_id, version_no) "
            "VALUES (:id, :mid, :sid, 1)"
        ),
        {"id": vid, "mid": material_id, "sid": storage_object_id},
    )
    return {"id": vid}


async def _insert_role(conn: AsyncConnection) -> PrimaryKey:
    rid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO roles (id, code, name) VALUES (:id, :code, :name)"),
        {"id": rid, "code": f"r-{rid.hex[:10]}", "name": "trig role"},
    )
    return {"id": rid}


async def _new_role(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_role(conn))["id"]


async def _insert_permission(conn: AsyncConnection) -> PrimaryKey:
    pid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO permissions (id, code, name) VALUES (:id, :code, :name)"),
        {"id": pid, "code": f"p-{pid.hex[:10]}", "name": "trig perm"},
    )
    return {"id": pid}


async def _new_permission(conn: AsyncConnection) -> uuid.UUID:
    return (await _insert_permission(conn))["id"]


async def _insert_user_role_assignment(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    role_id = await _new_role(conn)
    org_id = await _new_organization(conn)
    aid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO user_role_assignments "
            "(id, user_id, role_id, scope_kind, organization_id) "
            "VALUES (:id, :uid, :rid, 'organization', :org)"
        ),
        {"id": aid, "uid": user_id, "rid": role_id, "org": org_id},
    )
    return {"id": aid}


async def _insert_user_permission_grant(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    perm_id = await _new_permission(conn)
    org_id = await _new_organization(conn)
    gid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO user_permission_grants "
            "(id, user_id, permission_id, scope_kind, organization_id) "
            "VALUES (:id, :uid, :pid, 'organization', :org)"
        ),
        {"id": gid, "uid": user_id, "pid": perm_id, "org": org_id},
    )
    return {"id": gid}


async def _insert_organization_domain(conn: AsyncConnection) -> PrimaryKey:
    org_id = await _new_organization(conn)
    did = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO organization_domains (id, organization_id, domain) "
            "VALUES (:id, :org, :domain)"
        ),
        {"id": did, "org": org_id, "domain": f"d-{did.hex[:10]}.test"},
    )
    return {"id": did}


async def _insert_organization_membership(conn: AsyncConnection) -> PrimaryKey:
    org_id = await _new_organization(conn)
    user_id = await _new_user(conn)
    mid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO organization_memberships "
            "(id, organization_id, user_id, status) "
            "VALUES (:id, :org, :uid, 'active')"
        ),
        {"id": mid, "org": org_id, "uid": user_id},
    )
    return {"id": mid}


async def _insert_user_profile(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    await conn.execute(
        text("INSERT INTO user_profiles (user_id, display_name) VALUES (:uid, :name)"),
        {"uid": user_id, "name": "trig profile"},
    )
    return {"user_id": user_id}


async def _insert_user_profile_link(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    lid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO user_profile_links (id, user_id, link_type, url) "
            "VALUES (:id, :uid, 'website', :url)"
        ),
        {"id": lid, "uid": user_id, "url": f"https://t-{lid.hex[:10]}.test"},
    )
    return {"id": lid}


async def _insert_auth_identity(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    aid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO auth_identities "
            "(id, user_id, provider, provider_subject) "
            "VALUES (:id, :uid, 'google', :sub)"
        ),
        {"id": aid, "uid": user_id, "sub": f"sub-{aid.hex[:12]}"},
    )
    return {"id": aid}


async def _insert_auth_session(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    sid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO auth_sessions "
            "(id, user_id, refresh_token_hash, expires_at) "
            "VALUES (:id, :uid, :tok, NOW() + interval '1 day')"
        ),
        {"id": sid, "uid": user_id, "tok": f"tok-{sid.hex}"},
    )
    return {"id": sid}


async def _insert_mfa_factor(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    fid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO mfa_factors "
            "(id, user_id, factor_type, secret_encrypted) "
            "VALUES (:id, :uid, 'totp', :secret)"
        ),
        {"id": fid, "uid": user_id, "secret": f"enc-{fid.hex}"},
    )
    return {"id": fid}


async def _insert_tag(conn: AsyncConnection) -> PrimaryKey:
    tid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO tags (id, slug, name) VALUES (:id, :slug, :name)"),
        {"id": tid, "slug": f"t-{tid.hex[:10]}", "name": "trig tag"},
    )
    return {"id": tid}


async def _insert_course_enrollment(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    student_id = await _new_user(conn)
    eid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO course_enrollments "
            "(id, course_id, student_id, status) "
            "VALUES (:id, :cid, :sid, 'active')"
        ),
        {"id": eid, "cid": course_id, "sid": student_id},
    )
    return {"id": eid}


async def _insert_course_invitation_code(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    org_id = await _new_organization(conn)
    iid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO course_invitation_codes "
            "(id, course_id, organization_id, code) "
            "VALUES (:id, :cid, :org, :code)"
        ),
        {"id": iid, "cid": course_id, "org": org_id, "code": f"INV-{iid.hex[:12]}"},
    )
    return {"id": iid}


async def _insert_course_learning_outcome(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    oid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO course_learning_outcomes "
            "(id, course_id, position, outcome_text) "
            "VALUES (:id, :cid, 1, :txt)"
        ),
        {"id": oid, "cid": course_id, "txt": "trig outcome text"},
    )
    return {"id": oid}


async def _insert_lesson_resource(conn: AsyncConnection) -> PrimaryKey:
    lesson_id = await _new_lesson(conn)
    rid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO lesson_resources "
            "(id, lesson_id, title, resource_type, position) "
            "VALUES (:id, :lid, :title, 'link', 1)"
        ),
        {"id": rid, "lid": lesson_id, "title": "trig resource"},
    )
    return {"id": rid}


async def _insert_module_item(conn: AsyncConnection) -> PrimaryKey:
    module_id = await _new_module(conn)
    lesson_id = await _new_lesson(conn)
    iid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO module_items "
            "(id, module_id, item_type, position, lesson_id) "
            "VALUES (:id, :mid, 'lesson', 1, :lid)"
        ),
        {"id": iid, "mid": module_id, "lid": lesson_id},
    )
    return {"id": iid}


async def _insert_lesson_progress(conn: AsyncConnection) -> PrimaryKey:
    lesson_id = await _new_lesson(conn)
    user_id = await _new_user(conn)
    pid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO lesson_progress "
            "(id, lesson_id, user_id, status) "
            "VALUES (:id, :lid, :uid, 'not_started')"
        ),
        {"id": pid, "lid": lesson_id, "uid": user_id},
    )
    return {"id": pid}


async def _insert_processing_job(conn: AsyncConnection) -> PrimaryKey:
    jid = uuid.uuid4()
    target_id = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO processing_jobs "
            "(id, entity_type, entity_id, job_type) "
            "VALUES (:id, 'material_version', :eid, 'parse_document')"
        ),
        {"id": jid, "eid": target_id},
    )
    return {"id": jid}


async def _insert_quiz_question_option(conn: AsyncConnection) -> PrimaryKey:
    question_id = await _new_quiz_question(conn)
    oid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO quiz_question_options "
            "(id, question_id, option_key, option_text, is_correct, position) "
            "VALUES (:id, :qid, 'A', :txt, false, 1)"
        ),
        {"id": oid, "qid": question_id, "txt": "trig option"},
    )
    return {"id": oid}


async def _insert_quiz_attempt_answer(conn: AsyncConnection) -> PrimaryKey:
    attempt_pk = await _insert_quiz_attempt(conn)
    question_id = await _new_quiz_question(conn)
    aid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO quiz_attempt_answers "
            "(id, attempt_id, question_id, is_correct) "
            "VALUES (:id, :aid, :qid, false)"
        ),
        {"id": aid, "aid": attempt_pk["id"], "qid": question_id},
    )
    return {"id": aid}


async def _insert_generation_run(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    actor_id = await _new_user(conn)
    rid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO generation_runs "
            "(id, generation_type, source_scope_kind, course_id, requested_by) "
            "VALUES (:id, 'quiz', 'course', :cid, :uid)"
        ),
        {"id": rid, "cid": course_id, "uid": actor_id},
    )
    return {"id": rid}


async def _insert_interview_question(conn: AsyncConnection) -> PrimaryKey:
    config_id = await _new_interview_config(conn)
    qid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO interview_questions "
            "(id, interview_config_id, question_type, prompt_text) "
            "VALUES (:id, :cfg, 'conceptual', :p)"
        ),
        {"id": qid, "cfg": config_id, "p": "trig iq?"},
    )
    return {"id": qid}


async def _insert_interview_session_message(conn: AsyncConnection) -> PrimaryKey:
    session_id = await _new_interview_session(conn)
    mid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO interview_session_messages "
            "(id, session_id, role, content_text) "
            "VALUES (:id, :sid, 'ai', :txt)"
        ),
        {"id": mid, "sid": session_id, "txt": "trig msg"},
    )
    return {"id": mid}


async def _insert_interview_outcome_evaluation(conn: AsyncConnection) -> PrimaryKey:
    session_id = await _new_interview_session(conn)
    outcome_id = await _new_interview_outcome(conn)
    eid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO interview_outcome_evaluations "
            "(id, session_id, outcome_id, verdict_met) "
            "VALUES (:id, :sid, :oid, false)"
        ),
        {"id": eid, "sid": session_id, "oid": outcome_id},
    )
    return {"id": eid}


async def _insert_gap_report(conn: AsyncConnection) -> PrimaryKey:
    course_id = await _new_course(conn)
    student_id = await _new_user(conn)
    rid = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO gap_reports (id, course_id, student_id) VALUES (:id, :cid, :uid)"),
        {"id": rid, "cid": course_id, "uid": student_id},
    )
    return {"id": rid}


async def _insert_notification(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    nid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO notifications "
            "(id, user_id, category, title, body) "
            "VALUES (:id, :uid, 'system', :title, :body)"
        ),
        {"id": nid, "uid": user_id, "title": "trig title", "body": "trig body"},
    )
    return {"id": nid}


async def _insert_notification_preference(conn: AsyncConnection) -> PrimaryKey:
    user_id = await _new_user(conn)
    pid = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO notification_preferences "
            "(id, user_id, category, channel, enabled) "
            "VALUES (:id, :uid, 'system', 'in_app', TRUE)"
        ),
        {"id": pid, "uid": user_id},
    )
    return {"id": pid}


async def _insert_student_card_state(conn: AsyncConnection) -> PrimaryKey:
    question_id = await _new_quiz_question(conn)
    student_id = await _new_user(conn)
    await conn.execute(
        text("INSERT INTO student_card_state (student_id, question_id) VALUES (:uid, :qid)"),
        {"uid": student_id, "qid": question_id},
    )
    return {"student_id": student_id, "question_id": question_id}


async def _insert_student_career_enrollment(conn: AsyncConnection) -> PrimaryKey:
    career_path_id = await _new_career_path(conn)
    student_id = await _new_user(conn)
    eid = uuid.uuid4()
    version_id = (
        await conn.execute(
            text(
                "INSERT INTO career_path_versions "
                "(id, career_path_id, version_no, status) "
                "VALUES (gen_random_uuid(), :cpid, 1, 'draft') "
                "RETURNING id"
            ),
            {"cpid": career_path_id},
        )
    ).scalar_one()
    await conn.execute(
        text(
            "INSERT INTO student_career_enrollments "
            "(id, career_path_id, version_id, student_id, status) "
            "VALUES (:id, :cpid, :vid, :uid, 'active')"
        ),
        {"id": eid, "cpid": career_path_id, "vid": version_id, "uid": student_id},
    )
    return {"id": eid}


ROW_FACTORIES: dict[str, RowFactory] = {
    "auth_identities": _insert_auth_identity,
    "auth_sessions": _insert_auth_session,
    "career_paths": _insert_career_path,
    "course_enrollments": _insert_course_enrollment,
    "course_invitation_codes": _insert_course_invitation_code,
    "course_learning_outcomes": _insert_course_learning_outcome,
    "courses": _insert_course,
    "gap_reports": _insert_gap_report,
    "generation_runs": _insert_generation_run,
    "interview_configs": _insert_interview_config,
    "interview_outcome_evaluations": _insert_interview_outcome_evaluation,
    "interview_outcomes": _insert_interview_outcome,
    "interview_questions": _insert_interview_question,
    "interview_session_messages": _insert_interview_session_message,
    "interview_sessions": _insert_interview_session,
    "learning_material_versions": _insert_learning_material_version,
    "learning_materials": _insert_learning_material,
    "lesson_progress": _insert_lesson_progress,
    "lesson_resources": _insert_lesson_resource,
    "lessons": _insert_lesson,
    "mfa_factors": _insert_mfa_factor,
    "module_items": _insert_module_item,
    "modules": _insert_module,
    "notification_preferences": _insert_notification_preference,
    "notifications": _insert_notification,
    "org_units": _insert_org_unit,
    "organization_domains": _insert_organization_domain,
    "organization_memberships": _insert_organization_membership,
    "organizations": _insert_organization,
    "permissions": _insert_permission,
    "processing_jobs": _insert_processing_job,
    "quiz_attempt_answers": _insert_quiz_attempt_answer,
    "quiz_attempts": _insert_quiz_attempt,
    "quiz_question_options": _insert_quiz_question_option,
    "quiz_questions": _insert_quiz_question,
    "quizzes": _insert_quiz,
    "roles": _insert_role,
    "storage_objects": _insert_storage_object,
    "student_card_state": _insert_student_card_state,
    "student_career_enrollments": _insert_student_career_enrollment,
    "tags": _insert_tag,
    "user_permission_grants": _insert_user_permission_grant,
    "user_profile_links": _insert_user_profile_link,
    "user_profiles": _insert_user_profile,
    "user_role_assignments": _insert_user_role_assignment,
    "users": _insert_user,
}


def _where_clause(pk: PrimaryKey) -> str:
    return " AND ".join(f"{col} = :pk_{col}" for col in pk)


def _where_params(pk: PrimaryKey) -> dict[str, Any]:
    return {f"pk_{col}": val for col, val in pk.items()}


@pytest.mark.parametrize("table", sorted(TIMESTAMP_TABLES))
async def test_trigger_overwrites_stale_updated_at(engine: AsyncEngine, table: str) -> None:
    factory = ROW_FACTORIES[table]

    async with engine.connect() as conn:
        trans = await conn.begin()
        try:
            pk = await factory(conn)
            where = _where_clause(pk)
            params = _where_params(pk)

            stale = datetime(2020, 1, 1, tzinfo=UTC)
            await conn.execute(
                text(f"UPDATE {table} SET updated_at = :stale WHERE {where}"),  # noqa: S608  # test fixture: table names from code-controlled list
                {"stale": stale, **params},
            )

            after_row = await conn.execute(
                text(f"SELECT updated_at FROM {table} WHERE {where}"),  # noqa: S608  # test fixture: table names from code-controlled list
                params,
            )
            after = after_row.scalar_one()
            assert isinstance(after, datetime)

            assert after != stale, (
                f"Trigger did not fire on {table}: stale literal {stale!r} stuck in updated_at"
            )
            assert after.year >= 2025, f"Trigger on {table} produced suspicious value: {after!r}"
        finally:
            await trans.rollback()


def test_factory_coverage_matches_migration_inventory() -> None:
    expected = set(TIMESTAMP_TABLES)
    covered = set(ROW_FACTORIES.keys())
    missing = expected - covered
    extra = covered - expected
    assert not missing, f"ROW_FACTORIES missing tables: {sorted(missing)}"
    assert not extra, f"ROW_FACTORIES has extra tables: {sorted(extra)}"
