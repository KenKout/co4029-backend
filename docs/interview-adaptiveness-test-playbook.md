# Interview Adaptiveness — Manual Test Playbook (Production)

Generated for manual QA of the v2 realism features. All features are flag-gated;
this playbook covers what is **currently LIVE** on production (text/hybrid/voice),
plus the two deployed-but-OFF features so you know what should NOT happen.

## Currently LIVE (as of enablement)
- Slice 15 — self-correction recognition
- Slice 16 — confident-but-wrong challenge
- Slice 17 — rambling redirect
- Slice 20 — communication polish (time-pressure + recovery framing)
- Slice 19A — frustration de-escalation
- Slice 7 — phase progression (opening → warmup → core → deep-probe → closing)
- Slice 8 — depth probe on strong answers
- Slice 13 — rich closing sequence

## Deployed but OFF (should NOT trigger)
- Slice 18 — outcome backtracking
- Slice 19B — mid-interview question deferral
- Slice 9 — cross-turn contradiction
- Slice 10 — affect tone lead-ins
- Slice 11 — hint ladder
- Slice 12 — per-outcome difficulty

---

## How to confirm what the engine actually decided

Every adaptive turn emits a decision event. While testing, tail the backend log
and watch the `action=` field. In a separate terminal on the production host:

```bash
# Live-follow decision events, ANSI-stripped, showing action + reason_code
tail -f /root/.pm2/logs/abridgeai-backend-out.log \
  | sed -u 's/\x1b\[[0-9;]*m//g' \
  | grep --line-buffered 'event_type=voice.decision' \
  | grep --line-buffered -oE 'action=[^ ]+ .*reason_code=[^ ]+'
```

For the voice path, also tail `abridgeai-interview-agent-out.log`.

The action names to look for:
| Feature | action= | reason_code= |
|---|---|---|
| 19A frustration | `deescalate` | `candidate_frustrated` |
| 16 confident-wrong | `challenge_reasoning` | `confident_but_wrong_challenge` |
| 17 rambling | `redirect_to_topic` | `rambling_redirect` |
| 8 depth probe (extend) | `extend_answer` | `strong_answer_depth_probe` |
| 8 depth probe (edge) | `probe_edge_case` | `strong_answer_depth_probe` |
| 13 closing: reflection | `prompt_self_reflection` | `closing_self_reflection` |
| 13 closing: invite Qs | `invite_candidate_questions` | `closing_invite_questions` |
| 13 closing: answer Q | `answer_candidate_question` | `closing_answered_question` |
| 19B deferral (OFF) | `defer_candidate_question` | `candidate_question_deferred` |

Slice 15 (self-correction), Slice 20 (comms polish) and Slice 7 (phases) have NO
distinct action — they are tone/difficulty/flow changes on otherwise-normal turns,
so verify them by reading the interviewer's reply text (and the sequence of turns),
not a single log action.

The daily cron report (09:00 UTC, this channel) also counts these by action.

---

## TEST 1 — Frustration de-escalation (Slice 19A)  [DETERMINISTIC — most reliable]

**What to type** (any of these; EN or VI):
- "This is pointless."
- "I give up."
- "This is a waste of time."
- "What's the point of this."
- "I can't do this."
- (VI) "Chán quá đi." / "Thôi tôi bỏ cuộc." / "Mất thời gian quá." / "Cái này vô nghĩa."

**Expected interviewer behavior:**
- A warm, non-penalizing acknowledgement, roughly:
  > "That's completely okay — take a breath. There's no penalty here; let's take it one step at a time."
  (VI: "Không sao đâu — bạn cứ bình tĩnh. Không có điểm trừ gì cả; chúng ta cứ đi từng bước một nhé.")
- The **same question is repeated** — you are NOT advanced to the next question.
- It is **not scored** and does **not** consume the follow-up budget (you can express
  frustration repeatedly without being pushed forward).

**Log check:** `action=deescalate reason_code=candidate_frustrated`

**Negative test:** an answer that merely mentions difficulty must NOT trigger it, e.g.
"This is a hard topic, but a fact table stores measurable business events." → should be
treated as a normal answer (high-precision rules avoid hijacking real answers).

---

## TEST 2 — Confident-but-wrong challenge (Slice 16)  [LLM-analysis dependent]

**What to type:** a **specific, confident, but factually wrong** answer to a question.
The key is: relevant + specific + stated assertively + incorrect. Example (adapt to
your question): if asked "What does a database index do?", answer confidently:
> "An index makes writes faster because it stores a compressed copy of the whole table in RAM, so every INSERT is O(1)."
(Specific, confident, and wrong.)

**Expected interviewer behavior:**
- Instead of quietly moving on, it **challenges your reasoning** — a corrective but
  non-shaming follow-up asking you to reconsider/defend, e.g. "Let's dig into that —
  are you sure inserts get faster when you add an index? Walk me through why."
- You stay on the same question (a probe, not an advance).

**Log check:** `action=challenge_reasoning reason_code=confident_but_wrong_challenge`

**Note:** if the analyzer reads your answer as vague/low-confidence or already
recommends a different probe, that other probe wins (by design). Be specific + assertive
to trigger the challenge.

---

## TEST 3 — Rambling redirect (Slice 17)  [affect dependent]

**What to type:** a **long (aim for 60+ words), on-topic, low-substance** answer that
meanders without saying much. Example:
> "Yeah so I think this is really interesting and there are a lot of angles to consider,
> like there's the historical context and also the practical side, and honestly it depends
> on a lot of things, and different people have different opinions, and when you really
> think about it there's so much to unpack here, it's a big topic and I could talk about
> it for a while because it connects to many other things I've studied over the years."

**Expected interviewer behavior:**
- A gentle steer back to focus, e.g. "Let's focus in a little —" then redirect to the
  current question, rather than advancing.

**Log check:** `action=redirect_to_topic reason_code=rambling_redirect`

**Note:** a long answer that IS substantive should NOT trigger it — length alone isn't
enough; the affect signal keys on low substance.

---

## TEST 4 — Self-correction recognition (Slice 15)  [LLM-analysis dependent, subtle]

**What to type:** an answer where you **correct yourself mid-answer**:
> "A primary key can be null — wait, no, actually a primary key can never be null, that's
> a NOT NULL uniqueness constraint. I mixed it up with a foreign key for a second."

**Expected interviewer behavior:**
- A **positive** acknowledgement that credits the self-correction (rather than a neutral
  or corrective tone), e.g. "Nice catch correcting yourself —".
- It will **not** then probe you about the contradiction you already resolved.

**How to verify:** this is a tone/acknowledgement change (no distinct action in the log).
Read the interviewer's opening acknowledgement — it should feel rewarding, not corrective,
and it should NOT re-ask about the thing you just fixed.

---

## TEST 5 — Communication polish (Slice 20)  [state dependent]

Two sub-behaviors, each needs a real condition:

**5a — Recovery framing:** give **two weak/incorrect answers in a row** (e.g. "I don't
know" then a clearly wrong attempt). On the turn after the streak, the interviewer should
prepend an encouraging lead-in:
> "No problem — let's take a fresh, straightforward one."
(VI: "Không sao — mình thử một câu nhẹ nhàng, rõ ràng hơn nhé.")

**5b — Time-pressure:** only fires when **≤20% of the interview's time remains**. If your
test config is timed and you're near the end, the interviewer prepends:
> "We're a little short on time, so let's prioritise."
(VI: "Chúng ta còn hơi ít thời gian, nên hãy tập trung vào điểm chính.")

**Precedence:** if both conditions AND an affect lead-in would apply at once, only ONE
lead-in shows, in priority order **recovery > time-pressure > affect**. You'll never see
them stacked.

**How to verify:** read the lead-in sentence at the start of the interviewer's reply.
No distinct log action (tone only).

---

## TEST 7 — Phase progression (Slice 7)  [flow/difficulty — no single log action]

Phases shape the whole session arc: OPENING → WARMUP → CORE → DEEP_PROBE → CLOSING,
with difficulty biased **down** in warmup and **up** in deep-probe.

**How to test:** run a full interview from the start and observe the shape:
- The **first turn or two feel gentle** (warmup): easier phrasing, eased-in difficulty.
- The **middle** is the main body (core): normal difficulty, outcome-driven probing.
- Once all required outcomes are covered AND time+budget remain, it enters **deep-probe**
  (harder follow-ups — see TEST 8), rather than closing immediately.
- When outcomes are covered and time is low, it heads to **closing** (see TEST 9).

**How to verify:** there is no single action to grep — phases is a difficulty/flow bias.
Confirm by the *sequence*: warmup questions should feel easier than deep-probe ones, and
the session should not jump straight from core to a hard probe without covering outcomes
first. (In the decision events you'll see difficulty/`phase` shift across turns.)

**Note:** phases underpins TEST 8 (deep-probe is a phase) and TEST 9 (closing is a phase),
so the cleanest way to exercise all three is one complete interview end-to-end.

---

## TEST 8 — Depth probe on a strong answer (Slice 8)  [LLM-analysis dependent]

**What to type:** a **genuinely strong, complete, correct** answer to a question — the
opposite of the confident-wrong test. Give a precise, well-structured, accurate response
that fully covers what was asked.

**Expected interviewer behavior:**
- Instead of advancing to the next question, it **digs for your ceiling** with a harder
  follow-up on the SAME topic — either asking you to extend/generalize your answer, or
  posing an edge case:
  - extend: "Good — can you push that further / what would change at scale?"
  - edge case: "Solid. Now what about the case where <boundary condition>?"
- This consumes the follow-up budget, so it won't loop forever — after the budget/time is
  spent it advances normally.

**Log check:** `action=extend_answer` or `action=probe_edge_case`
(both with `reason_code=strong_answer_depth_probe`)

**Note:** only strong answers trigger this. A partial/weak/mixed answer will get normal
handling (or a different probe). It also only fires when follow-up budget + time remain,
and (with phases on) is most likely once required outcomes are covered — i.e. in the
deep-probe phase. If it doesn't fire, give a clearly excellent answer earlier in the topic.

---

## TEST 9 — Rich closing sequence (Slice 13)  [flow — multiple closing turns]

**How to test:** complete an interview normally (cover the outcomes, or run down the time /
ask to end). Instead of a single abrupt "that concludes the interview," you should get a
short **multi-turn closing sequence**:

1. **Self-reflection prompt** — e.g. "Before we wrap up: looking back, what's one thing you
   feel went well, and one you'd approach differently?"
   (`action=prompt_self_reflection`)
2. **Invite candidate questions** — e.g. "Thank you for sharing that. Is there anything
   you'd like to ask me?" (`action=invite_candidate_questions`)
3. If you ask something, an **answer-safe reply** that never leaks rubric/answers, e.g.
   "That's a good question. I can't share evaluation details here, but your instructor will
   follow up." (`action=answer_candidate_question`)
4. **Graceful sign-off** — the final close.

**Log check:** you should see the closing actions in sequence:
`prompt_self_reflection` → `invite_candidate_questions` → (`answer_candidate_question`) →
`begin_closing` / `close_interview`.

**Note:** this adds 2–3 turns to the end of every completed interview. That's expected — it's
the graceful wind-down replacing the old one-shot cutoff.

---

## TEST 6 — Confirm the OFF features do NOT trigger

**6a — Question deferral (19B, OFF):** mid-interview, type "Can I ask you a question?"
- Expected NOW (off): treated as a normal turn / falls through — you should NOT see a
  "good question, let's come back to that at the end" deferral.
- Log check: you should NOT see `action=defer_candidate_question`.
- **Exception:** during the **closing** phase, asking a question IS answered (that's Slice 13
  rich closing, TEST 9, which is ON) — you'll see `action=answer_candidate_question`. The
  19B "defer" behavior is specifically the *mid-interview* case, which stays off.

**6b — Backtracking (18, OFF):** this is invisible in normal use and only affects question
selection ordering; nothing specific to type. No user-visible test.

**6c — Cross-turn (9), affect (10), hint ladder (11), per-outcome difficulty (12) — all OFF:**
- Cross-turn: contradicting something you said several turns earlier should NOT be flagged
  as a cross-turn contradiction (only within-answer contradictions are caught).
- Affect: a nervous/terse tone should NOT add a reassuring lead-in *by itself* (note: comms
  polish recovery/time-pressure lead-ins ARE on — TEST 5 — so don't confuse the two).
- Hint ladder: repeated hint requests will NOT escalate through distinct nudge→structural→
  direct levels; you get the standard single hint behavior.
- Per-outcome difficulty: difficulty tracks the global streak, not per-topic competence.

---

## Reliability summary
- **TEST 1 (frustration)** — deterministic, will fire every time on the listed phrases. Best smoke test.
- **TESTS 2–4** — depend on the LLM analyzer; fire reliably when the answer clearly matches the shape, but not guaranteed on borderline answers. Retry with a clearer example if it doesn't trigger.
- **TEST 5** — needs the real state condition (weak streak / low time).
- **TEST 6** — negative tests; confirm nothing happens.

## Rollback (if anything misbehaves)
On the production host:
```bash
cd /root/co4029/backend
# disable a single feature (example: frustration):
#   set ADAPTIVE_V2_FRUSTRATION_DEESCALATION_ENABLED=false in .env
# or kill ALL v2 realism features at once:
#   set ADAPTIVE_INTERVIEWER_V2_ENABLED=false in .env
pm2 restart abridgeai-backend --update-env
```
Backups of .env are at ~/co4029/backend/.env.bak.* (timestamped).
