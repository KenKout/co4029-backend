"""baseline schema

Revision ID: 0001_baseline_schema
Revises:
Create Date: 2026-05-16 19:51:13.821894

Ports backend/db/schema.sql (53KB hand-managed schema) into a single Alembic
baseline. Hybrid strategy:
  1. op.execute() raw DDL for the bulk of schema.sql (extensions + types + tables
     + indexes + deferred FKs).
  2. op.add_column() for net-new audit columns (created_by/updated_by/deleted_*)
     on every soft-deletable table.

Deltas vs. schema.sql (per Reconciliation supplement):
  - DROP study_groups + study_group_members entirely (Reconciliation §B8/§D4).
  - Add `description TEXT` column to career_paths (§B6).
  - Extend learning_material_versions.processing_status CHECK to include
    'enriching' (§C10, 8 values total).
  - Add audit columns: created_by (where missing), updated_by, deleted_at,
    deleted_by on soft-deletable content tables.
  - notifications gets only `updated_by` (no soft-delete columns) per §B5.
  - Identity/credentials/jobs/audit/event tables stay HARD-delete.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "0001_baseline_schema"
down_revision = None
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Soft-delete eligibility (Reconciliation §A13 + draft "Soft-delete eligibility")
# ---------------------------------------------------------------------------
# Tables that receive: created_by (if missing), updated_by, deleted_at, deleted_by
SOFT_DELETE_TABLES: tuple[str, ...] = (
    "organizations",
    "org_units",
    "organization_memberships",
    "organization_domains",
    "user_profiles",
    "user_profile_links",
    "storage_objects",
    "permissions",
    "roles",
    "user_role_assignments",
    "user_permission_grants",
    "career_paths",
    "student_career_enrollments",
    "courses",
    "course_learning_outcomes",
    "modules",
    "lessons",
    "lesson_resources",
    "module_items",
    "learning_materials",
    "learning_material_versions",
    "quizzes",
    "quiz_questions",
    "quiz_question_options",
    "interview_configs",
    "interview_outcomes",
    "interview_questions",
)

# Tables that already have a created_by column → skip created_by addition
# (cross-checked against schema.sql lines 323/414/585/643/722/973)
TABLES_WITH_EXISTING_CREATED_BY: frozenset[str] = frozenset(
    {
        "course_invitation_codes",
        "quizzes",
        "quiz_question_revisions",
        "interview_configs",
        "bulk_import_jobs",
    }
)


# ---------------------------------------------------------------------------
# Bulk schema (verbatim port of backend/db/schema.sql with the documented edits)
# ---------------------------------------------------------------------------
BASELINE_DDL = r"""
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "citext";


-- =========================================================
-- Identity, tenancy, and access control
-- =========================================================

CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'inactive', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE org_units (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    parent_unit_id UUID REFERENCES org_units(id) ON DELETE CASCADE,
    unit_type VARCHAR(20) NOT NULL
        CHECK (unit_type IN ('faculty', 'department', 'office', 'program', 'campus', 'other')),
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (organization_id, code)
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    primary_email CITEXT NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'invited', 'inactive', 'suspended')),
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE storage_objects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bucket VARCHAR(100) NOT NULL,
    object_key VARCHAR(500) NOT NULL,
    original_filename VARCHAR(255),
    mime_type VARCHAR(100),
    size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    uploaded_by UUID REFERENCES users(id) ON DELETE SET NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (bucket, object_key)
);

CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    given_name VARCHAR(100),
    family_name VARCHAR(100),
    display_name VARCHAR(200) NOT NULL,
    avatar_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    bio TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE user_profile_links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    link_type VARCHAR(30) NOT NULL
        CHECK (link_type IN ('website', 'github', 'linkedin', 'portfolio', 'other')),
    url TEXT NOT NULL,
    label VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE organization_domains (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    domain CITEXT NOT NULL UNIQUE,
    auto_provision BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE organization_memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    org_unit_id UUID REFERENCES org_units(id) ON DELETE SET NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'invited', 'inactive', 'suspended', 'left')),
    student_code VARCHAR(50),
    employee_code VARCHAR(50),
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    left_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, organization_id, org_unit_id)
);

CREATE TABLE auth_identities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(20) NOT NULL CHECK (provider IN ('google')),
    provider_subject VARCHAR(255) NOT NULL,
    provider_email CITEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, provider_subject)
);

CREATE TABLE auth_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    mfa_verified_at TIMESTAMPTZ,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE mfa_factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    factor_type VARCHAR(20) NOT NULL CHECK (factor_type IN ('totp')),
    secret_encrypted TEXT NOT NULL,
    verified_at TIMESTAMPTZ,
    disabled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE mfa_recovery_codes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_id UUID NOT NULL REFERENCES mfa_factors(id) ON DELETE CASCADE,
    code_hash VARCHAR(255) NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (factor_id, code_hash)
);

CREATE TABLE mfa_challenges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    factor_id UUID NOT NULL REFERENCES mfa_factors(id) ON DELETE CASCADE,
    session_id UUID REFERENCES auth_sessions(id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    is_system_role BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE role_permissions (
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE user_role_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    scope_kind VARCHAR(20) NOT NULL
        CHECK (scope_kind IN ('global', 'organization', 'org_unit', 'course')),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    org_unit_id UUID REFERENCES org_units(id) ON DELETE CASCADE,
    course_id UUID,
    granted_by UUID REFERENCES users(id) ON DELETE SET NULL,
    active_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, role_id, scope_kind, organization_id, org_unit_id, course_id),
    CHECK (
        (scope_kind = 'global' AND organization_id IS NULL AND org_unit_id IS NULL AND course_id IS NULL) OR
        (scope_kind = 'organization' AND organization_id IS NOT NULL AND org_unit_id IS NULL AND course_id IS NULL) OR
        (scope_kind = 'org_unit' AND organization_id IS NOT NULL AND org_unit_id IS NOT NULL AND course_id IS NULL) OR
        (scope_kind = 'course' AND organization_id IS NOT NULL AND course_id IS NOT NULL)
    )
);

CREATE TABLE user_permission_grants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    scope_kind VARCHAR(20) NOT NULL
        CHECK (scope_kind IN ('global', 'organization', 'org_unit', 'course')),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    org_unit_id UUID REFERENCES org_units(id) ON DELETE CASCADE,
    course_id UUID,
    granted_by UUID REFERENCES users(id) ON DELETE SET NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, permission_id, scope_kind, organization_id, org_unit_id, course_id),
    CHECK (
        (scope_kind = 'global' AND organization_id IS NULL AND org_unit_id IS NULL AND course_id IS NULL) OR
        (scope_kind = 'organization' AND organization_id IS NOT NULL AND org_unit_id IS NULL AND course_id IS NULL) OR
        (scope_kind = 'org_unit' AND organization_id IS NOT NULL AND org_unit_id IS NOT NULL AND course_id IS NULL) OR
        (scope_kind = 'course' AND organization_id IS NOT NULL AND course_id IS NOT NULL)
    )
);


-- =========================================================
-- Organization, careers, courses, and curriculum
-- =========================================================

CREATE TABLE career_paths (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    org_unit_id UUID REFERENCES org_units(id) ON DELETE SET NULL,
    slug VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,  -- Reconciliation §B6: net-new column for path overview
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (organization_id, slug)
);

CREATE TABLE student_career_enrollments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    career_path_id UUID NOT NULL REFERENCES career_paths(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'completed', 'dropped')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (career_path_id, student_id)
);

CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE courses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    org_unit_id UUID REFERENCES org_units(id) ON DELETE SET NULL,
    owner_user_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    slug VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'archived')),
    level VARCHAR(20)
        CHECK (level IN ('beginner', 'intermediate', 'advanced')),
    thumbnail_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    estimated_minutes INT CHECK (estimated_minutes IS NULL OR estimated_minutes > 0),
    expected_completion_days INT CHECK (expected_completion_days IS NULL OR expected_completion_days > 0),
    enrollment_cap INT CHECK (enrollment_cap IS NULL OR enrollment_cap > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_courses_org_slug UNIQUE (organization_id, slug)
);

CREATE TABLE career_course_items (
    career_path_id UUID NOT NULL REFERENCES career_paths(id) ON DELETE CASCADE,
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    position INT NOT NULL CHECK (position > 0),
    is_required BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (career_path_id, course_id),
    UNIQUE (career_path_id, position)
);

CREATE TABLE course_tags (
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    tag_id UUID NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (course_id, tag_id)
);

CREATE TABLE course_learning_outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    position INT NOT NULL CHECK (position > 0),
    outcome_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (course_id, position)
);

CREATE TABLE course_invitation_codes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    code VARCHAR(64) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ,
    max_uses INT CHECK (max_uses IS NULL OR max_uses > 0),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE course_enrollments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'completed', 'dropped', 'waitlisted')),
    waitlist_position INT CHECK (waitlist_position IS NULL OR waitlist_position > 0),
    source VARCHAR(20) NOT NULL DEFAULT 'self_enroll'
        CHECK (source IN ('self_enroll', 'invite_code', 'admin_bulk', 'manager_bulk', 'manual')),
    invitation_code_id UUID REFERENCES course_invitation_codes(id) ON DELETE SET NULL,
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    dropped_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (course_id, student_id)
);

CREATE TABLE modules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    position INT NOT NULL CHECK (position > 0),
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'archived')),
    estimated_minutes INT CHECK (estimated_minutes IS NULL OR estimated_minutes > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (course_id, position)
);

CREATE TABLE module_prerequisites (
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    prerequisite_module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (module_id, prerequisite_module_id),
    CHECK (module_id <> prerequisite_module_id)
);

CREATE TABLE lessons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    slug VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    summary TEXT,
    notes_markdown TEXT,
    primary_material_id UUID,
    lesson_type VARCHAR(30) NOT NULL DEFAULT 'video'
        CHECK (lesson_type IN ('video', 'reading')),
    difficulty VARCHAR(20)
        CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
    estimated_minutes INT CHECK (estimated_minutes IS NULL OR estimated_minutes > 0),
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (module_id, slug)
);

CREATE TABLE lesson_resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    resource_type VARCHAR(20) NOT NULL
        CHECK (resource_type IN ('pdf', 'zip', 'mp4', 'xlsx', 'pptx', 'docx', 'link', 'other')),
    storage_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    position INT NOT NULL CHECK (position > 0),
    visible_to_students BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (lesson_id, position)
);


-- =========================================================
-- (study_groups + study_group_members DROPPED per Reconciliation §B8/§D4)
-- =========================================================


-- =========================================================
-- Learning materials, AI processing, and retrieval corpus
-- =========================================================

CREATE TABLE learning_materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    material_type VARCHAR(20) NOT NULL
        CHECK (material_type IN ('video', 'pdf', 'code', 'audio', 'image', 'docx', 'pptx', 'xlsx', 'text')),
    ai_processing_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    visible_to_students BOOLEAN NOT NULL DEFAULT TRUE,
    current_version_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE learning_material_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    material_id UUID NOT NULL REFERENCES learning_materials(id) ON DELETE CASCADE,
    storage_object_id UUID NOT NULL REFERENCES storage_objects(id) ON DELETE RESTRICT,
    version_no INT NOT NULL CHECK (version_no > 0),
    is_current BOOLEAN NOT NULL DEFAULT FALSE,
    -- Reconciliation §C10: 8-state pipeline; 'enriching' added to original 7
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'extracting', 'chunking', 'enriching', 'embedding', 'building_kg', 'ready', 'failed', 'cancelled')),
    processing_error TEXT,
    extracted_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    uploaded_by UUID REFERENCES users(id) ON DELETE SET NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (material_id, version_no)
);

CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(30) NOT NULL
        CHECK (entity_type IN ('material_version', 'lesson', 'quiz', 'interview_config', 'generation_run')),
    entity_id UUID NOT NULL,
    job_type VARCHAR(40) NOT NULL
        CHECK (job_type IN (
            'transcribe', 'parse_document', 'parse_code', 'chunk', 'embed',
            'extract_entities', 'extract_relations', 'build_graph', 'full_pipeline',
            'generate_quiz', 'generate_interview_questions'
        )),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress_percent INT NOT NULL DEFAULT 0 CHECK (progress_percent BETWEEN 0 AND 100),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INT NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE generation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    generation_type VARCHAR(30) NOT NULL
        CHECK (generation_type IN ('quiz', 'interview', 'knowledge_graph', 'material_index')),
    source_scope_kind VARCHAR(20) NOT NULL
        CHECK (source_scope_kind IN ('lesson', 'module', 'course')),
    course_id UUID REFERENCES courses(id) ON DELETE CASCADE,
    module_id UUID REFERENCES modules(id) ON DELETE CASCADE,
    lesson_id UUID REFERENCES lessons(id) ON DELETE CASCADE,
    requested_by UUID REFERENCES users(id) ON DELETE SET NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    dedup_key VARCHAR(255),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE ai_model_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    generation_run_id UUID REFERENCES generation_runs(id) ON DELETE CASCADE,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    role VARCHAR(30),
    tier VARCHAR(20),
    pipeline_stage VARCHAR(40),
    operation VARCHAR(30) NOT NULL DEFAULT 'chat_completion',
    model_name VARCHAR(100) NOT NULL,
    base_url VARCHAR(255),
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_payload JSONB,
    input_tokens INT CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INT CHECK (output_tokens IS NULL OR output_tokens >= 0),
    total_tokens INT CHECK (total_tokens IS NULL OR total_tokens >= 0),
    cached_input_tokens INT CHECK (cached_input_tokens IS NULL OR cached_input_tokens >= 0),
    estimated_cost_usd NUMERIC(12, 6) CHECK (estimated_cost_usd IS NULL OR estimated_cost_usd >= 0),
    latency_ms INT CHECK (latency_ms IS NULL OR latency_ms >= 0),
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failed')),
    error_message TEXT,
    called_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_ai_model_calls_parent_ref
        CHECK (generation_run_id IS NOT NULL OR processing_job_id IS NOT NULL),
    CONSTRAINT ck_ai_model_calls_operation
        CHECK (operation IN ('chat_completion', 'embedding'))
);

CREATE INDEX ix_ai_model_calls_called_at_role
    ON ai_model_calls (called_at, role);
CREATE INDEX ix_ai_model_calls_model_called_at
    ON ai_model_calls (model_name, called_at);
CREATE INDEX ix_ai_model_calls_operation_called_at
    ON ai_model_calls (operation, called_at);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    material_version_id UUID NOT NULL REFERENCES learning_material_versions(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL CHECK (chunk_index >= 0),
    chunk_type VARCHAR(20) NOT NULL
        CHECK (chunk_type IN ('video', 'pdf', 'code', 'audio', 'image', 'docx', 'pptx', 'xlsx', 'text')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536),
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (material_version_id, chunk_index)
);


-- =========================================================
-- Quiz authoring, review, and attempts
-- =========================================================

CREATE TABLE quizzes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'archived')),
    time_limit_seconds INT CHECK (time_limit_seconds IS NULL OR time_limit_seconds > 0),
    passing_score_percent NUMERIC(5, 2) NOT NULL DEFAULT 70.00 CHECK (passing_score_percent BETWEEN 0 AND 100),
    allow_retakes BOOLEAN NOT NULL DEFAULT TRUE,
    max_attempts INT CHECK (max_attempts IS NULL OR max_attempts > 0),
    cooldown_hours INT CHECK (cooldown_hours IS NULL OR cooldown_hours >= 0),
    shuffle_questions BOOLEAN NOT NULL DEFAULT FALSE,
    shuffle_options BOOLEAN NOT NULL DEFAULT FALSE,
    show_hints BOOLEAN NOT NULL DEFAULT TRUE,
    initial_ef NUMERIC(4, 2) CHECK (initial_ef IS NULL OR initial_ef > 0),
    min_ef_for_unlock NUMERIC(4, 2) CHECK (min_ef_for_unlock IS NULL OR min_ef_for_unlock > 0),
    coverage_threshold NUMERIC(5, 2) CHECK (coverage_threshold IS NULL OR coverage_threshold BETWEEN 0 AND 100),
    reminders_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    generation_instructions TEXT,
    generation_run_id UUID REFERENCES generation_runs(id) ON DELETE SET NULL,
    published_at TIMESTAMPTZ,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE quiz_source_lessons (
    quiz_id UUID NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (quiz_id, lesson_id)
);

CREATE TABLE quiz_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    quiz_id UUID NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    position INT NOT NULL CHECK (position > 0),
    question_type VARCHAR(30) NOT NULL
        CHECK (question_type IN ('mcq', 'true_false', 'short_answer', 'fill_blank', 'code')),
    prompt_text TEXT NOT NULL,
    hint_text TEXT,
    explanation TEXT,
    difficulty VARCHAR(20)
        CHECK (difficulty IN ('easy', 'medium', 'hard')),
    bloom_level VARCHAR(20)
        CHECK (bloom_level IN ('remember', 'understand', 'apply', 'analyze', 'evaluate', 'create')),
    review_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'edited', 'rejected')),
    expected_response_ms INT NOT NULL CHECK (expected_response_ms > 0),
    expected_ef_ceiling NUMERIC(4, 2) CHECK (expected_ef_ceiling IS NULL OR expected_ef_ceiling > 0),
    source_refs_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    original_generated_payload JSONB,
    reviewed_by UUID REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (quiz_id, position)
);

CREATE TABLE quiz_question_options (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_id UUID NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    option_key VARCHAR(5) NOT NULL,
    option_text TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL DEFAULT FALSE,
    position INT NOT NULL CHECK (position > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (question_id, option_key),
    UNIQUE (question_id, position)
);

CREATE TABLE quiz_question_revisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_id UUID NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    revision_no INT NOT NULL CHECK (revision_no > 0),
    source_kind VARCHAR(20) NOT NULL CHECK (source_kind IN ('ai', 'teacher')),
    payload_json JSONB NOT NULL,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (question_id, revision_no)
);

CREATE TABLE quiz_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    quiz_id UUID NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    attempt_number INT NOT NULL CHECK (attempt_number > 0),
    status VARCHAR(20) NOT NULL DEFAULT 'in_progress'
        CHECK (status IN ('in_progress', 'submitted', 'graded', 'abandoned', 'expired')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    time_taken_seconds INT CHECK (time_taken_seconds IS NULL OR time_taken_seconds >= 0),
    score_points NUMERIC(10, 2) CHECK (score_points IS NULL OR score_points >= 0),
    score_percent NUMERIC(5, 2) CHECK (score_percent IS NULL OR score_percent BETWEEN 0 AND 100),
    passed BOOLEAN,
    idempotency_key UUID UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (quiz_id, student_id, attempt_number)
);

CREATE TABLE quiz_attempt_answers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    attempt_id UUID NOT NULL REFERENCES quiz_attempts(id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    selected_option_id UUID REFERENCES quiz_question_options(id) ON DELETE SET NULL,
    answer_text TEXT,
    is_correct BOOLEAN NOT NULL,
    hint_used BOOLEAN NOT NULL DEFAULT FALSE,
    response_time_ms INT CHECK (response_time_ms IS NULL OR response_time_ms >= 0),
    points_awarded NUMERIC(8, 2) NOT NULL DEFAULT 0,
    sm2_q_value NUMERIC(4, 2),
    ef_before NUMERIC(4, 2),
    ef_after NUMERIC(4, 2),
    interval_before_days INT,
    interval_after_days INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (attempt_id, question_id)
);

CREATE TABLE student_quiz_card_state (
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    current_ef NUMERIC(4, 2) NOT NULL CHECK (current_ef > 0),
    current_interval_days INT NOT NULL CHECK (current_interval_days >= 0),
    repetition_count INT NOT NULL DEFAULT 0 CHECK (repetition_count >= 0),
    next_due_at TIMESTAMPTZ,
    last_attempt_id UUID REFERENCES quiz_attempts(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (student_id, question_id)
);


-- =========================================================
-- Interview configuration, sessions, and evaluation
-- =========================================================

CREATE TABLE interview_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'archived')),
    max_attempts INT CHECK (max_attempts IS NULL OR max_attempts > 0),
    min_outcomes_to_pass INT CHECK (min_outcomes_to_pass IS NULL OR min_outcomes_to_pass > 0),
    lock_quiz_ef_until_pass BOOLEAN NOT NULL DEFAULT FALSE,
    time_limit_minutes INT CHECK (time_limit_minutes IS NULL OR time_limit_minutes > 0),
    persona VARCHAR(20) CHECK (persona IN ('strict', 'neutral', 'supportive')),
    supported_modes VARCHAR(20) NOT NULL DEFAULT 'hybrid'
        CHECK (supported_modes IN ('voice', 'text', 'hybrid')),
    supplementary_instructions TEXT,
    generation_run_id UUID REFERENCES generation_runs(id) ON DELETE SET NULL,
    published_at TIMESTAMPTZ,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE interview_outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interview_config_id UUID NOT NULL REFERENCES interview_configs(id) ON DELETE CASCADE,
    position INT NOT NULL CHECK (position > 0),
    outcome_text TEXT NOT NULL,
    outcome_type VARCHAR(20) NOT NULL
        CHECK (outcome_type IN ('knowledge', 'skill', 'attitude')),
    importance_weight INT NOT NULL DEFAULT 1 CHECK (importance_weight BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (interview_config_id, position)
);

CREATE TABLE interview_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interview_config_id UUID NOT NULL REFERENCES interview_configs(id) ON DELETE CASCADE,
    linked_outcome_id UUID REFERENCES interview_outcomes(id) ON DELETE SET NULL,
    position INT CHECK (position IS NULL OR position > 0),
    question_type VARCHAR(30) NOT NULL
        CHECK (question_type IN ('conceptual', 'behavioral', 'technical', 'situational', 'system_design')),
    prompt_text TEXT NOT NULL,
    difficulty VARCHAR(20)
        CHECK (difficulty IN ('junior', 'mid_level', 'senior')),
    review_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'edited', 'rejected')),
    ai_generated BOOLEAN NOT NULL DEFAULT TRUE,
    source_refs_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    reviewed_by UUID REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (interview_config_id, position)
);

CREATE TABLE interview_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interview_config_id UUID NOT NULL REFERENCES interview_configs(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    attempt_number INT NOT NULL CHECK (attempt_number > 0),
    status VARCHAR(20) NOT NULL DEFAULT 'in_progress'
        CHECK (status IN ('in_progress', 'completed', 'timed_out', 'abandoned', 'failed')),
    input_mode VARCHAR(20) NOT NULL
        CHECK (input_mode IN ('voice', 'text', 'hybrid')),
    livekit_room_name VARCHAR(255),
    livekit_session_ref VARCHAR(255),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    resume_deadline_at TIMESTAMPTZ,
    transcript_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    recording_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    pass_verdict BOOLEAN,
    internal_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (interview_config_id, student_id, attempt_number)
);

CREATE TABLE interview_session_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    interview_question_id UUID REFERENCES interview_questions(id) ON DELETE SET NULL,
    sequence_no INT NOT NULL CHECK (sequence_no > 0),
    asked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, sequence_no)
);

CREATE TABLE interview_session_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    session_question_id UUID REFERENCES interview_session_questions(id) ON DELETE SET NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('ai', 'user', 'system')),
    content_text TEXT,
    audio_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    latency_ms INT CHECK (latency_ms IS NULL OR latency_ms >= 0),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE interview_outcome_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    outcome_id UUID NOT NULL REFERENCES interview_outcomes(id) ON DELETE CASCADE,
    verdict_met BOOLEAN NOT NULL,
    hidden_reasoning TEXT,
    evidence_excerpt TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, outcome_id)
);

CREATE TABLE gap_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id UUID REFERENCES modules(id) ON DELETE SET NULL,
    source_quiz_attempt_id UUID REFERENCES quiz_attempts(id) ON DELETE SET NULL,
    source_interview_session_id UUID REFERENCES interview_sessions(id) ON DELETE SET NULL,
    student_summary TEXT,
    teacher_summary TEXT,
    report_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =========================================================
-- Curriculum ordering after quiz/interview entities exist
-- =========================================================

CREATE TABLE module_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    item_type VARCHAR(20) NOT NULL CHECK (item_type IN ('lesson', 'quiz', 'interview')),
    lesson_id UUID REFERENCES lessons(id) ON DELETE CASCADE,
    quiz_id UUID REFERENCES quizzes(id) ON DELETE CASCADE,
    interview_config_id UUID REFERENCES interview_configs(id) ON DELETE CASCADE,
    position INT NOT NULL CHECK (position > 0),
    unlock_rule_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (module_id, position),
    CHECK (
        (item_type = 'lesson' AND lesson_id IS NOT NULL AND quiz_id IS NULL AND interview_config_id IS NULL) OR
        (item_type = 'quiz' AND lesson_id IS NULL AND quiz_id IS NOT NULL AND interview_config_id IS NULL) OR
        (item_type = 'interview' AND lesson_id IS NULL AND quiz_id IS NULL AND interview_config_id IS NOT NULL)
    )
);


-- =========================================================
-- Progress, notifications, gamification, and admin monitoring
-- =========================================================

CREATE TABLE student_lesson_progress (
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'not_started'
        CHECK (status IN ('not_started', 'in_progress', 'completed')),
    progress_percent NUMERIC(5, 2) NOT NULL DEFAULT 0 CHECK (progress_percent BETWEEN 0 AND 100),
    time_spent_seconds INT NOT NULL DEFAULT 0 CHECK (time_spent_seconds >= 0),
    last_position_seconds INT CHECK (last_position_seconds IS NULL OR last_position_seconds >= 0),
    last_opened_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (student_id, lesson_id)
);

CREATE TABLE student_module_status (
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    lesson_completion_ratio NUMERIC(5, 2) NOT NULL DEFAULT 0 CHECK (lesson_completion_ratio BETWEEN 0 AND 100),
    quiz_passed BOOLEAN NOT NULL DEFAULT FALSE,
    interview_passed BOOLEAN NOT NULL DEFAULT FALSE,
    unlocked_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    last_computed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (student_id, module_id)
);

CREATE TABLE student_course_status (
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    progress_percent NUMERIC(5, 2) NOT NULL DEFAULT 0 CHECK (progress_percent BETWEEN 0 AND 100),
    final_grade VARCHAR(20),
    retention_estimate NUMERIC(5, 2) CHECK (retention_estimate IS NULL OR retention_estimate BETWEEN 0 AND 100),
    review_compliance_rate NUMERIC(5, 2) CHECK (review_compliance_rate IS NULL OR review_compliance_rate BETWEEN 0 AND 100),
    at_risk_level VARCHAR(20) NOT NULL DEFAULT 'none'
        CHECK (at_risk_level IN ('none', 'low', 'medium', 'high')),
    last_activity_at TIMESTAMPTZ,
    last_computed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (student_id, course_id)
);

CREATE TABLE career_readiness_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    career_path_id UUID NOT NULL REFERENCES career_paths(id) ON DELETE CASCADE,
    readiness_score NUMERIC(5, 2) NOT NULL CHECK (readiness_score BETWEEN 0 AND 100),
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(30) NOT NULL
        CHECK (category IN ('course_updates', 'ai_recommendations', 'spaced_repetition', 'lesson_unlock', 'interview_result', 'course_announcement', 'system')),
    channel VARCHAR(20) NOT NULL
        CHECK (channel IN ('email', 'in_app')),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, category, channel)
);

CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(30) NOT NULL
        CHECK (category IN ('spaced_repetition', 'lesson_unlock', 'interview_result', 'course_announcement', 'system')),
    entity_type VARCHAR(30),
    entity_id UUID,
    title VARCHAR(255) NOT NULL,
    body TEXT NOT NULL,
    scheduled_for TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    read_at TIMESTAMPTZ,
    delivery_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (delivery_status IN ('pending', 'sent', 'failed', 'cancelled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE feature_flags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    updated_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE system_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE bulk_import_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    file_object_id UUID REFERENCES storage_objects(id) ON DELETE SET NULL,
    import_type VARCHAR(20) NOT NULL CHECK (import_type IN ('users', 'enrollments')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE bulk_import_rows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bulk_import_job_id UUID NOT NULL REFERENCES bulk_import_jobs(id) ON DELETE CASCADE,
    row_number INT NOT NULL CHECK (row_number > 0),
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processed', 'failed', 'skipped')),
    error_message TEXT,
    resolved_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (bulk_import_job_id, row_number)
);

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    actor_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    actor_role_assignment_id UUID REFERENCES user_role_assignments(id) ON DELETE SET NULL,
    action_code VARCHAR(100) NOT NULL,
    target_type VARCHAR(50),
    target_id UUID,
    target_label VARCHAR(255),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE assessment_integrity_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_kind VARCHAR(20) NOT NULL CHECK (assessment_kind IN ('quiz', 'interview')),
    quiz_attempt_id UUID REFERENCES quiz_attempts(id) ON DELETE CASCADE,
    interview_session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type VARCHAR(30) NOT NULL
        CHECK (event_type IN ('focus_lost', 'tab_switch', 'fullscreen_exit', 'warning_issued', 'reconnect', 'disconnect')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (assessment_kind = 'quiz' AND quiz_attempt_id IS NOT NULL AND interview_session_id IS NULL) OR
        (assessment_kind = 'interview' AND interview_session_id IS NOT NULL AND quiz_attempt_id IS NULL)
    )
);


-- =========================================================
-- Deferred foreign keys for cyclical references
-- =========================================================

ALTER TABLE lessons
    ADD CONSTRAINT fk_lessons_primary_material
    FOREIGN KEY (primary_material_id) REFERENCES learning_materials(id) ON DELETE SET NULL;

ALTER TABLE learning_materials
    ADD CONSTRAINT fk_learning_materials_current_version
    FOREIGN KEY (current_version_id) REFERENCES learning_material_versions(id) ON DELETE SET NULL;

ALTER TABLE user_role_assignments
    ADD CONSTRAINT fk_user_role_assignments_course
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE;

ALTER TABLE user_permission_grants
    ADD CONSTRAINT fk_user_permission_grants_course
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE;


-- =========================================================
-- Indexes
-- =========================================================

CREATE INDEX idx_org_units_organization ON org_units (organization_id);
CREATE INDEX idx_storage_objects_uploaded_by ON storage_objects (uploaded_by);
CREATE INDEX idx_organization_memberships_user ON organization_memberships (user_id);
CREATE INDEX idx_organization_memberships_org ON organization_memberships (organization_id);
CREATE INDEX idx_auth_identities_user ON auth_identities (user_id);
CREATE INDEX idx_auth_sessions_user ON auth_sessions (user_id);
CREATE INDEX idx_mfa_factors_user ON mfa_factors (user_id);
CREATE INDEX idx_role_permissions_permission ON role_permissions (permission_id);
CREATE INDEX idx_user_role_assignments_user ON user_role_assignments (user_id);
CREATE INDEX idx_user_role_assignments_scope ON user_role_assignments (scope_kind, organization_id, org_unit_id, course_id);
CREATE INDEX idx_user_permission_grants_user ON user_permission_grants (user_id);
CREATE INDEX idx_courses_organization ON courses (organization_id);
CREATE INDEX idx_course_enrollments_student ON course_enrollments (student_id);
CREATE INDEX idx_modules_course ON modules (course_id);
CREATE INDEX idx_lessons_module ON lessons (module_id);
-- (idx_study_groups_organization DROPPED with the table)
CREATE INDEX idx_learning_materials_lesson ON learning_materials (lesson_id);
CREATE INDEX idx_learning_material_versions_material ON learning_material_versions (material_id);
CREATE INDEX idx_learning_material_versions_status ON learning_material_versions (processing_status);
CREATE INDEX idx_processing_jobs_entity ON processing_jobs (entity_type, entity_id);
CREATE INDEX idx_processing_jobs_status ON processing_jobs (status);
CREATE INDEX idx_generation_runs_status ON generation_runs (status);
CREATE INDEX idx_generation_runs_scope ON generation_runs (course_id, module_id, lesson_id);
CREATE UNIQUE INDEX uq_generation_runs_active_dedup
    ON generation_runs (dedup_key)
    WHERE dedup_key IS NOT NULL AND status IN ('pending', 'running');
CREATE INDEX idx_ai_model_calls_generation_run ON ai_model_calls (generation_run_id);
CREATE INDEX idx_ai_model_calls_processing_job ON ai_model_calls (processing_job_id);
CREATE INDEX idx_document_chunks_course ON document_chunks (course_id);
CREATE INDEX idx_document_chunks_module ON document_chunks (module_id);
CREATE INDEX idx_document_chunks_lesson ON document_chunks (lesson_id);
CREATE INDEX idx_document_chunks_material_version ON document_chunks (material_version_id);
CREATE INDEX idx_document_chunks_metadata ON document_chunks USING gin (metadata);
CREATE INDEX idx_document_chunks_content_fts
    ON document_chunks USING gin (to_tsvector('english', content));
CREATE INDEX idx_document_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_quizzes_module ON quizzes (module_id);
CREATE INDEX idx_quiz_questions_quiz ON quiz_questions (quiz_id);
CREATE INDEX idx_quiz_attempts_quiz_student ON quiz_attempts (quiz_id, student_id);
CREATE INDEX idx_quiz_attempt_answers_attempt ON quiz_attempt_answers (attempt_id);
CREATE INDEX idx_interview_configs_module ON interview_configs (module_id);
CREATE INDEX idx_interview_questions_config ON interview_questions (interview_config_id);
CREATE INDEX idx_interview_sessions_student ON interview_sessions (student_id);
CREATE INDEX idx_interview_session_messages_session ON interview_session_messages (session_id);
CREATE INDEX idx_gap_reports_student_course ON gap_reports (student_id, course_id);
CREATE INDEX idx_module_items_module ON module_items (module_id);
CREATE INDEX idx_student_lesson_progress_lesson ON student_lesson_progress (lesson_id);
CREATE INDEX idx_student_module_status_module ON student_module_status (module_id);
CREATE INDEX idx_student_course_status_course ON student_course_status (course_id);
CREATE INDEX idx_career_readiness_student ON career_readiness_snapshots (student_id);
CREATE INDEX idx_notifications_user_created ON notifications (user_id, created_at DESC);
CREATE INDEX idx_notifications_user_unread ON notifications (user_id) WHERE read_at IS NULL;
CREATE INDEX idx_bulk_import_jobs_org ON bulk_import_jobs (organization_id);
CREATE INDEX idx_bulk_import_rows_job ON bulk_import_rows (bulk_import_job_id);
CREATE INDEX idx_audit_logs_actor ON audit_logs (actor_user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs (created_at DESC);
CREATE INDEX idx_assessment_integrity_student_created ON assessment_integrity_events (student_id, created_at DESC);


-- ============================================================================
-- Chunking enrichment cache (5-stage chunking pipeline, Stage C)
-- ============================================================================
CREATE TABLE chunking_enrichment_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) NOT NULL,
    prompt_version VARCHAR(20) NOT NULL,
    model_name VARCHAR(100),
    output_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_chunking_enrichment_cache_hash_prompt
        UNIQUE (content_hash, prompt_version)
);
CREATE INDEX ix_chunking_enrichment_cache_content_hash
    ON chunking_enrichment_cache (content_hash);
"""


# Tables created by the bulk DDL above (used by downgrade in reverse order).
ALL_TABLES: tuple[str, ...] = (
    # Identity / tenancy / ACL
    "organizations",
    "org_units",
    "users",
    "storage_objects",
    "user_profiles",
    "user_profile_links",
    "organization_domains",
    "organization_memberships",
    "auth_identities",
    "auth_sessions",
    "mfa_factors",
    "mfa_recovery_codes",
    "mfa_challenges",
    "permissions",
    "roles",
    "role_permissions",
    "user_role_assignments",
    "user_permission_grants",
    # Org / careers / curriculum
    "career_paths",
    "student_career_enrollments",
    "tags",
    "courses",
    "career_course_items",
    "course_tags",
    "course_learning_outcomes",
    "course_invitation_codes",
    "course_enrollments",
    "modules",
    "module_prerequisites",
    "lessons",
    "lesson_resources",
    # Learning materials / AI
    "learning_materials",
    "learning_material_versions",
    "processing_jobs",
    "generation_runs",
    "ai_model_calls",
    "document_chunks",
    # Quizzes
    "quizzes",
    "quiz_source_lessons",
    "quiz_questions",
    "quiz_question_options",
    "quiz_question_revisions",
    "quiz_attempts",
    "quiz_attempt_answers",
    "student_quiz_card_state",
    # Interviews
    "interview_configs",
    "interview_outcomes",
    "interview_questions",
    "interview_sessions",
    "interview_session_questions",
    "interview_session_messages",
    "interview_outcome_evaluations",
    "gap_reports",
    # Curriculum ordering (after quiz/interview)
    "module_items",
    # Progress / notifications / admin
    "student_lesson_progress",
    "student_module_status",
    "student_course_status",
    "career_readiness_snapshots",
    "notification_preferences",
    "notifications",
    "feature_flags",
    "system_settings",
    "bulk_import_jobs",
    "bulk_import_rows",
    "audit_logs",
    "assessment_integrity_events",
    # AI cache
    "chunking_enrichment_cache",
)


def _add_audit_columns(table: str, *, with_soft_delete: bool, with_created_by: bool) -> None:
    """Add net-new audit columns to an existing table.

    All FKs to ``users.id`` use ``ON DELETE SET NULL`` so deleting a user
    does not cascade-wipe content rows they touched.
    """
    if with_created_by:
        op.add_column(
            table,
            sa.Column(
                "created_by",
                UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
    op.add_column(
        table,
        sa.Column(
            "updated_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    if with_soft_delete:
        op.add_column(
            table,
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.add_column(
            table,
            sa.Column(
                "deleted_by",
                UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        op.create_index(
            f"ix_{table}_deleted_at",
            table,
            ["deleted_at"],
        )


def upgrade() -> None:
    # 1. Bulk schema (extensions + tables + indexes + deferred FKs)
    op.execute(BASELINE_DDL)

    # 2. Audit columns on soft-deletable tables
    for table in SOFT_DELETE_TABLES:
        _add_audit_columns(
            table,
            with_soft_delete=True,
            with_created_by=table not in TABLES_WITH_EXISTING_CREATED_BY,
        )

    # 3. notifications gets only updated_by (Reconciliation §B5: NO SoftDeleteMixin)
    op.add_column(
        "notifications",
        sa.Column(
            "updated_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    # Drop tables in reverse-creation order with CASCADE so FKs unwind cleanly.
    for table in reversed(ALL_TABLES):
        op.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')

    # Extensions go last. Don't drop citext/vector/uuid-ossp blindly: another DB
    # in the same cluster may depend on them. Use IF EXISTS but no CASCADE.
    op.execute('DROP EXTENSION IF EXISTS "vector"')
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
    op.execute('DROP EXTENSION IF EXISTS "citext"')
