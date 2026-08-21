# Role-Conditioned Question Selection (HARD filter) — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement task-by-task, two-stage review per task.

**Goal:** Khi config interview có `interviewer_role` (tech lead / staff engineer / eng manager / hr screener), session sẽ chỉ hỏi các câu hỏi có `question_type` khớp với role đó. Để làm được, generation có thể sinh nhiều variant câu hỏi (mỗi variant một "góc hỏi"), và selection lọc theo role.

**Architecture:** Reuse 100% trường hiện có — `InterviewQuestion.question_type` = "góc hỏi" (angle), `InterviewConfig.persona_profile_json["interviewer_role"]` = role. **Không migration, không công nghệ mới.** Thêm 1 module thuần `role_question_filter.py` (mapping role→type + pre-filter candidate) và 1 sub-flag `adaptive_v2_role_question_filter_enabled`.

**Tech stack:** FastAPI + SQLAlchemy (async), LLMGateway, Jinja2 prompts, ARQ worker, flag-gating theo `core/config.py::_V2_SUBFLAGS`.

---

## Fairness note (đọc trước khi code — đây là gate đã chốt với owner)

`interviewer_role` là **config-scoped** (1 role cố định per config, lưu trong `persona_profile_json`), nên mọi candidate cùng config gặp **cùng** role → **cùng** selection → invariant *"2 candidate ngang sức = cùng điểm"* **KHÔNG bị phá**. Khác hẳn trường hợp role biến thiên per-candidate mà evidence validity cấm (xem `references/interview-structure-validity-evidence.md`: "question selection by role → REJECT" là cho role per-session).

Nhưng đây là **đổi "what is assessed"** (nội dung câu hỏi), không phải "đổi cách hỏi". Vì vậy bắt buộc:
1. **Flag-gate default OFF** — khi OFF, engine byte-for-byte v1.
2. **UI phải nói rõ** chọn role giờ đổi NỘI DUNG (không chỉ giọng) — FE phase.
3. **Đo lường** bằng quality harness để chứng minh role đổi hành vi (không chỉ assert).

**Guardrail kỹ thuật (giữ nguyên fairness boundary):** role KHÔNG được đưa vào `SelectionContext` hay `DecisionInputs` (giữ `test_interview_persona_invariants.py` xanh). Role chỉ là **pre-filter trên list candidate** ở call-site (`adaptive.py`), *trước* khi gọi `select_next_question` — scorer bên trong vẫn role-blind.

---

## Design decisions

### Đã chốt với owner (vòng 2)
- **HARD filter**: role chọn ĐÚNG type (không soft-bias, không giữ mix).
- "1 phần"/"góc hỏi" = `question_type` (enum DB 5 giá trị).
- **D5 (owner sửa):** teacher VẪN gõ `question_count = N` (số câu logic). Khi `all_angles`, hệ sinh **4N** câu (4 variant mỗi câu logic). UI hiện note dưới ô count: *"sẽ sinh 4× (4N câu) — 1 variant tương ứng mỗi role"*.
- **D4 (owner sửa):** thêm field chiến lược gen **2 option**:
  - **`all_angles`** (option 1): gen N × 4 variant (full bank, đổi role không cần regenerate).
  - **`role_only`** (option 2): gen N câu đúng type của role hiện tại (rẻ, nhưng đổi role phải regenerate).

### Proposed defaults còn lại (owner xác nhận trước khi code Phase 2+)
| # | Decision | Đề xuất |
|---|---|---|
| D1 | Role→type mapping (1:1) | `backend_tech_lead→technical`, `staff_engineer→system_design`, `eng_manager→situational`, `hr_screener→behavioral`, `generic_assistant→None` (no filter) |
| D2 | Bộ 4 angles (cố định) | `technical, system_design, situational, behavioral` = 4 type của 4 role non-generic (widen generation từ 3 → 4 type, thêm `system_design`; bỏ `conceptual`) |
| D3 | Vị trí field `variant_strategy` | FORM value trong `generation_runs.config_json` (cạnh `question_count`, `target_outcome_ids`), KHÔNG phải supplementary |
| D4 | Trigger | `variant_strategy` ∈ `all_angles` / `role_only`; vắng mặt → legacy (1 câu/position, no variant) |
| D5 | `role_only` cần role set trước | nếu role = generic (no type) khi gen `role_only` → degrade về legacy mixed |
| D6 | Fallback khi bank thiếu variant khớp role | giữ 1 câu non-matching của outcome đó (không block coverage). No-op trên bank variant chuẩn |
| D7 | Gate selection | sub-flag `adaptive_v2_role_question_filter_enabled` (default OFF) |

---

## Data model — KHÔNG migration

- `interview_questions` đã có `question_type` (CHECK 5 giá trị) + `linked_outcome_id` + `position` (UNIQUE per config). Nhiều variant cùng câu logic = nhiều row khác `question_type`, khác `position`. Không cần cột mới.
- Role đã lưu ở `persona_profile_json["interviewer_role"]` (migration 0060 precedent: JSONB editable không migration).
- `variant_strategy` là form value trong `generation_runs.config_json` (precedent: `question_count` + `target_outcome_ids` đã là form value đọc qua `run.config_json`, KHÔNG phải `supplementary_instructions`).

> **Grouping (optional, defer):** không cần cột `variant_group_id` cho core filter — filter chỉ cần "giữ câu đúng type", không cần biết nhóm. Nhóm chỉ cần cho (a) dedup exemption và (b) FE review grouping. Defer, dùng `position`-slot nếu cần sau.

---

## Phase 1 — Role→type mapping module (backend, thuần, TDD)

**Files:**
- Create: `backend/abridgeai/features/interviews/orchestrator/role_question_filter.py`
- Test: `backend/tests/unit/test_interview_role_question_filter.py`

**Step 1 — RED.** Viết test:
```python
from abridgeai.features.interviews.orchestrator.interviewer_identity import InterviewerRole
from abridgeai.features.interviews.orchestrator.role_question_filter import (
    preferred_type, filter_candidates_by_role,
)
from abridgeai.features.interviews.orchestrator.selection import CandidateQuestion

def _c(qid, oid, qtype):
    return CandidateQuestion(question_id=qid, linked_outcome_id=oid,
                             question_type=qtype, difficulty=None, position=None)

def test_preferred_type_mapping():
    assert preferred_type(InterviewerRole.BACKEND_TECH_LEAD) == "technical"
    assert preferred_type(InterviewerRole.STAFF_ENGINEER) == "system_design"
    assert preferred_type(InterviewerRole.ENG_MANAGER) == "situational"
    assert preferred_type(InterviewerRole.HR_SCREENER) == "behavioral"
    assert preferred_type(InterviewerRole.GENERIC_ASSISTANT) is None

def test_generic_assistant_no_filter():
    cands = [_c("a", "o1", "technical"), _c("b", "o1", "situational")]
    assert filter_candidates_by_role(cands, InterviewerRole.GENERIC_ASSISTANT) == cands

def test_hard_filter_keeps_only_preferred():
    cands = [_c("a", "o1", "technical"), _c("b", "o1", "situational"),
             _c("c", "o2", "technical"), _c("d", "o2", "behavioral")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a", "c"}

def test_outcome_without_preferred_keeps_fallback():
    cands = [_c("a", "o1", "technical"), _c("b", "o2", "situational")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert {c.question_id for c in out} == {"a", "b"}  # b kept as fallback

def test_empty_pool_degrades_to_all():
    cands = [_c("b", "o1", "situational")]
    out = filter_candidates_by_role(cands, InterviewerRole.BACKEND_TECH_LEAD)
    assert out == cands
```

**Step 2 — GREEN.** Implement:
```python
_ROLE_PREFERRED_TYPE: dict[InterviewerRole, str | None] = {
    InterviewerRole.BACKEND_TECH_LEAD: "technical",
    InterviewerRole.STAFF_ENGINEER: "system_design",
    InterviewerRole.ENG_MANAGER: "situational",
    InterviewerRole.HR_SCREENER: "behavioral",
    InterviewerRole.GENERIC_ASSISTANT: None,
}

def preferred_type(role: InterviewerRole) -> str | None:
    return _ROLE_PREFERRED_TYPE.get(role)

def filter_candidates_by_role(
    candidates: list[CandidateQuestion], role: InterviewerRole
) -> list[CandidateQuestion]:
    preferred = preferred_type(role)
    if preferred is None:
        return candidates
    kept = [c for c in candidates if c.question_type == preferred]
    if not kept:
        return candidates
    kept_outcomes = {c.linked_outcome_id for c in kept}
    fallback = [c for c in candidates if c.linked_outcome_id not in kept_outcomes]
    return kept + fallback
```

**Step 3 — Verify.** `uv run --no-sync pytest backend/tests/unit/test_interview_role_question_filter.py -v` → PASS. `uv run --no-sync ruff check <file>` sạch. Commit.

---

## Phase 2 — Wire pre-filter vào `adaptive.py` (gated, invariant test)

**Files:**
- Modify: `backend/abridgeai/features/interviews/orchestrator/adaptive.py` (call-site, ~line 253)
- Modify: `backend/tests/unit/test_interview_persona_invariants.py` (thêm guardrail mới)

**Step 1 — RED (invariant).** Khẳng định role KHÔNG vào `SelectionContext`/`DecisionInputs` (đã có cho persona; bổ sung khẳng định role filter là pre-filter ngoài scorer). Thêm test "flag OFF → candidate list bất biến".

**Step 2 — Thread.** Trong `run_adaptive_turn`, sau `load_candidates`:
```python
if role_question_filter_enabled:
    resolved_identity = identity_from_config(getattr(config, "persona_profile_json", None))
    candidates = filter_candidates_by_role(candidates, resolved_identity.role)
```
(Không đổi `SelectionContext`, không đổi `select_next_question`.)

**Step 3 — Verify.** Full interview sweep + `test_interview_persona_invariants.py` xanh. Commit.

> `identity_from_config` hiện resolve ở ~line 389 (chỉ cho utterance). Giờ resolve thêm 1 lần cho filter (pure, rẻ). Không move resolution utterance hiện tại.

---

## Phase 3 — Flag trong `core/config.py` + thread từ `taking.py`

**Files:**
- Modify: `backend/abridgeai/core/config.py` (~line 394-418): thêm `adaptive_v2_role_question_filter_enabled: bool = False` + entry `"role_question_filter": "adaptive_v2_role_question_filter_enabled"` vào `_V2_SUBFLAGS`.
- Modify: `backend/abridgeai/features/interviews/services/taking.py` (resolve flag → `run_adaptive_turn`).
- Modify: `backend/abridgeai/features/interviews/orchestrator/adaptive.py` (thêm `role_question_filter_enabled` kwarg vào signature + call).
- Modify: `backend/tests/integration/test_interview_adaptive_step.py` (thêm flag vào `_settings_v2` helper + `model_copy`).

**Pitfall (flag-gating reference):** grep đếm anchor trước — các kwarg `*_enabled` đã có ~7 chỗ (resolution, live-advance, legacy path, `_try_adaptive_step` sig+call, `_run_shadow_step` sig+call, `run_adaptive_turn` sig). Thêm đủ từng chỗ, scripted-replace **longest-indent-first**, rồi `grep -n role_question_filter_enabled` eyeball indent + count.

**Verify:** `uv run --no-sync pytest backend/tests/ -k "interview" -q` + ruff + lint-imports. Commit.

---

## Phase 4 — Generation: field `variant_strategy` (2 option)

**Files:**
- Modify: `backend/abridgeai/features/interviews/ai/stages/generation/parsers.py` (widen `_VALID_TYPES` + `InterviewQuestionType` thêm `system_design`).
- Modify: `backend/abridgeai/features/interviews/ai/stages/generation/resolve.py` (thêm `resolve_variant_strategy(run_config_json) -> "all_angles" | "role_only" | None`).
- Modify: `backend/abridgeai/features/interviews/ai/stages/generation/logic.py` (`generate_interview_questions`: nhận `variant_strategy` + `role_type`, build prompt tương ứng + target count).
- Modify: `backend/abridgeai/features/interviews/ai/stages/generation/prompts/user.j2` + `system.j2` (variant-mode block + định nghĩa `system_design`).
- Modify: `backend/abridgeai/features/interviews/ai/pipelines/generation.py` (đọc `variant_strategy` từ `state.config_json`, tính `target_count`, resolve `role_type` từ `identity_from_config` cho `role_only`).
- Modify: `backend/abridgeai/features/interviews/ai/stages/validation/` (xác nhận validator chấp nhận `system_design` — grep trước khi sửa).

**Target count theo strategy:**
- `all_angles`: `target_count = question_count × 4` (N câu logic × 4 variant).
- `role_only`: `target_count = question_count` (N câu, đúng type role).
- `None` (legacy): `target_count = question_count` (hành vi cũ, type mix 60/30/10).

**Prompt (all_angles):** "Write N questions, each assessed against one outcome. For EACH of those N questions, write 4 variants — one `technical`, one `system_design`, one `situational`, one `behavioral` — all on the SAME topic and all carrying the SAME `linked_outcome_id`." Output 4N rows. Difficulty/expected_depth gán theo **chỉ số câu logic** (i=1..N), 4 variant cùng câu logic chia sẻ cùng difficulty (band theo i, không theo raw position — vì candidate chỉ thấy N câu sau filter).

**Prompt (role_only):** "Write N questions, ALL of type `<role_type>`." Output N rows.

**`role_only` + role generic** → degrade về legacy (mixed).

**Pitfall — dedup:** `store_question_embeddings` chạy trên mọi câu. 4 variant cùng câu logic (cùng topic, khác góc) có thể bị `dedup/shortlist.py` cờ là dup. Grep ngưỡng similarity; nếu loại nhầm, thêm exemption "cùng `linked_outcome_id` thì không tính dup" (hoặc defer — dedup chỉ cờ để teacher review, không auto-xoá).

**Pitfall — prompt vocabulary:** widen enum → sửa CẢ `system.j2` enum string (`question_type` = exactly one of technical/behavioral/situational/system_design), nếu không `system_design` dead-on-arrival.

**Verify:** pytest generation stage + full sweep. Ruff. Commit.

---

## Phase 5 — Integration + measurement

**Files:**
- Modify: `backend/tests/integration/test_interview_adaptive_step.py` (smoke: flag ON vẫn drive adaptive).
- Add: script đo bằng `features/interviews/quality/` — chạy cùng transcript qua 2 role khác nhau, assert câu hỏi được chọn khác type. Chứng minh role ĐỔI hành vi (không chỉ assert unit).

**Verify:** full backend suite + lint-imports + mypy. Commit + push + `pm2 restart abridgeai-backend`. Runtime-verify flag resolves True (load settings + `adaptive_v2_feature_enabled` cho 3 mode).

---

## Phase 6 — FE (follow-on, backend xong mới làm)

**Files (frontend repo):**
- Generation form (interview config): thêm radio `variant_strategy` (`all_angles` / `role_only`) + giữ ô `question_count`. Khi `all_angles`, hiện note dưới ô count: *"Sẽ sinh 4× (4N câu) — 1 variant tương ứng mỗi role interviewer"*.
- Khi chọn non-generic role: disclosure *"role này đổi NỘI DUNG câu hỏi (chỉ hỏi <type>)"*.
- Question bank review: group variant theo `linked_outcome_id` (32 câu vs 8 câu → cần grouping).
- i18n EN+VI. `npx eslint` từng file sửa.

> **Defer được nếu cần** — backend tự đứng được. FE chỉ là disclosure + form + review UX.

---

## Pitfalls tổng hợp

1. **Đừng đưa role vào scorer.** Pre-filter ngoài `selection.py`; `SelectionContext`/`DecisionInputs` role-blind. Guardrail test assert ABSENCE.
2. **Flag-off = byte-identical v1.** `filter_candidates_by_role` với `GENERIC_ASSISTANT`/flag OFF trả về nguyên list.
3. **Prompt text là bản copy vocabulary.** Widen enum → sửa CẢ `system.j2` enum string.
4. **Backfill loop sai target count** nếu `resolve_variant_strategy` + công thức count không dùng chung ở pipeline + logic.py.
5. **Dedup giết variant cùng topic.**
6. **7 anchor khi thread flag** (đếm, không nhớ).
7. **LOC caps:** `adaptive.py` ~590/580, `decision.py` ~730/800. Pre-filter vào module mới `role_question_filter.py`, không nhồi vào `adaptive.py`.
8. **Difficulty theo câu logic, không raw position** (all_angles) — nếu không, band easy/medium/hard vỡ trên 4N row.

---

## Verification (user-redo-test)

- Config role = tech lead + `all_angles` + N=8 → bank 32 câu → session chỉ hỏi 8 câu `technical`.
- Config role = tech lead + `role_only` → bank 8 câu technical → session hỏi 8 technical (filter no-op).
- Config role = generic assistant → hành vi cũ hoàn toàn (flag OFF lẫn ON).
- Flag OFF + role = tech lead → không filter (byte-identical v1).
- Outcome thiếu variant khớp role → không treo (fallback 1 câu).
- Đổi role sau khi gen `role_only` → filter degrade về no-filter (câu vẫn là type cũ) — chấp nhận theo trade-off đã chốt, yêu cầu regenerate.
