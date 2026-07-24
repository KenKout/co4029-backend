# Interview Session — Toàn bộ script (adaptiveness)

> Bản ghi đầy đủ mọi câu thoại (EN + VI) mà AI interviewer có thể nói, theo từng
> case. Trích nguyên văn từ code — nguồn sự thật:
> - `orchestrator/utterance.py` — câu thoại adaptive (ack / transition / probe / hint / lead-in)
> - `orchestrator/decision.py` + `decision_types.py` — action nào được chọn khi nào
> - `services/ceremony.py` — câu onboarding + closing (setup/kết thúc)
>
> Persona chỉ định hình **giọng điệu** (strict / neutral / supportive), KHÔNG bao
> giờ đổi logic chấm điểm. Mọi template đều song ngữ; `language` chọn EN/VI, mặc
> định EN cho ngôn ngữ lạ.
>
> Cấu trúc mỗi lượt Ak = `acknowledgement + transition + question_or_probe`, ghép
> bằng `_combine` (nối 1 khoảng trắng, bỏ phần rỗng).

---

## PHẦN A — ONBOARDING / CEREMONY (trước khi tính giờ)

Thứ tự stage: `identity_check → audio_check → language_check → preparation →
readiness(briefing) → completed(ready_transition)`.

### A1. Opening / xác nhận danh tính (`opening_text`)
- **EN:** `Hi{, name}. It's nice to meet you. I'm aBridgeAI's virtual interview assistant, and I'll guide your "{title}" technical course interview today. First, can you confirm that I'm speaking with {name}?`
- **VI:** `Xin chào{ name}. Rất vui được gặp bạn. Tôi là trợ lý phỏng vấn ảo của aBridgeAI và sẽ hướng dẫn buổi phỏng vấn kỹ thuật "{title}" hôm nay. Trước tiên, bạn có thể xác nhận tôi đang trao đổi với {name} không?`

### A2. Khi user nói "không phải tên tôi" (`ask_preferred_name_text`)
- **EN:** `No problem. What should I call you?`
- **VI:** `Không sao. Vậy tôi nên gọi bạn là gì?`

### A3. Sau khi user nhập tên (`preferred_name_ack_text`)
- **EN:** `Got it — I'll call you {name} from now on. Can you hear me clearly?`
- **VI:** `Đã rõ, từ giờ tôi sẽ gọi bạn là {name}. Bạn có nghe rõ tôi không?`

### A4. Kiểm tra âm thanh (`audio_check_text`)
- **EN:** `Thank you. Can you hear me clearly?`
- **VI:** `Cảm ơn bạn. Bạn có nghe rõ tôi không?`

### A5. Chọn ngôn ngữ (`language_check_text`)
- **EN:** `Great. Which language would you like to use for this interview: English or Vietnamese?`
- **VI:** `Tốt rồi. Bạn muốn sử dụng ngôn ngữ nào cho buổi phỏng vấn: tiếng Anh hay tiếng Việt?`

### A6. Chuẩn bị (`preparation_text`)
- **EN:** `We'll continue in English. Before I explain the interview structure, do you need a moment to prepare or adjust your setup?`
- **VI:** `Chúng ta sẽ tiếp tục bằng tiếng Việt. Trước khi tôi giải thích cấu trúc buổi phỏng vấn, bạn có cần một chút thời gian để chuẩn bị hoặc điều chỉnh thiết bị không?`

### A7. Briefing / sẵn sàng (`briefing_text`) — có biến `{duration}` + `{answer_mode}`
- **`duration`**: có giới hạn → `This interview will take up to {N} minutes.` / `Buổi phỏng vấn kéo dài tối đa {N} phút.` — không giới hạn → `This interview has no fixed overall time limit.` / `Buổi phỏng vấn này không có giới hạn tổng thời gian cố định.`
- **`answer_mode`**: voice → `You'll answer by speaking.` / `Bạn sẽ trả lời bằng giọng nói.` — hybrid → `You may type or speak your answers.` / `Bạn có thể nhập hoặc nói câu trả lời.` — text → `You'll type your answers.` / `Bạn sẽ nhập câu trả lời.`
- **EN:** `Thank you. {duration} The "{title}" interview will assess your answers against the current module criteria, and I'll ask one technical question at a time. {answer_mode} You may ask me to repeat or clarify a question at any time. There is no separate per-question limit; the overall timer begins only when you confirm. Are you ready to begin?`
- **VI:** `Cảm ơn bạn. {duration} Buổi "{title}" sẽ đánh giá câu trả lời theo các tiêu chí của mô-đun hiện tại và tôi sẽ hỏi từng câu kỹ thuật một. {answer_mode} Bạn có thể yêu cầu tôi nhắc lại hoặc làm rõ câu hỏi bất cứ lúc nào. Không có giới hạn riêng cho từng câu; đồng hồ tổng chỉ bắt đầu khi bạn xác nhận sẵn sàng. Bạn đã sẵn sàng bắt đầu chưa?`

### A8. Bắt đầu phỏng vấn (`ready_transition_text`)
- **EN:** `Great—the introduction is complete. Let's begin. Here is your first question.`
- **VI:** `Tuyệt vời—phần chào hỏi đã hoàn tất. Chúng ta bắt đầu nhé. Đây là câu hỏi đầu tiên.`

### A9. Retry khi câu trả lời onboarding không hợp lệ (`onboarding_retry_text`)
| Stage | EN | VI |
|---|---|---|
| identity_check | `I couldn't confirm that. Please let me know whether I'm speaking with the right candidate.` | `Tôi chưa xác nhận được. Vui lòng cho tôi biết tôi có đang trao đổi đúng người không.` |
| audio_check | `No problem. Adjust your audio if needed, then let me know when you can hear me clearly.` | `Không sao. Hãy điều chỉnh âm thanh nếu cần, sau đó cho tôi biết khi bạn nghe rõ.` |
| language_check | `Please choose English or Vietnamese for the interview.` | `Vui lòng chọn tiếng Anh hoặc tiếng Việt cho buổi phỏng vấn.` |
| preparation | `No problem. Take another moment and let me know when you'd like to continue.` | `Không sao. Hãy dành thêm một chút thời gian và cho tôi biết khi bạn muốn tiếp tục.` |
| (mặc định) | `No problem. Take another moment, and let me know when you're ready to begin.` | `Không sao. Hãy dành thêm một chút thời gian. Hãy cho tôi biết khi bạn sẵn sàng bắt đầu.` |

---

## PHẦN B — ACKNOWLEDGEMENT (mở đầu lượt, theo style × persona)

Chọn theo `acknowledgement_style` mà decision gán. Style `NONE` → không có ack.

| Style | Persona | EN | VI |
|---|---|---|---|
| NEUTRAL | strict | `Noted.` | `Đã ghi nhận.` |
| NEUTRAL | neutral | `Thank you.` | `Cảm ơn bạn.` |
| NEUTRAL | supportive | `Thanks for sharing that.` | `Cảm ơn bạn đã chia sẻ.` |
| POSITIVE | strict | `Good.` | `Tốt.` |
| POSITIVE | neutral | `That's helpful.` | `Điều đó hữu ích.` |
| POSITIVE | supportive | `That's a solid start, well done.` | `Một khởi đầu tốt, làm tốt lắm.` |
| CORRECTIVE | strict | `I understand your reasoning.` | `Tôi hiểu lập luận của bạn.` |
| CORRECTIVE | neutral | `I understand your reasoning.` | `Tôi hiểu cách bạn nghĩ.` |
| CORRECTIVE | supportive | `I see your direction.` | `Tôi thấy hướng bạn đang đi.` |

---

## PHẦN C — TRANSITION (chuyển câu hỏi)

> ⚠️ Đã bỏ "Thank you." mở đầu (ack Phần B đã có lời cảm ơn) — tránh double "Thank you. Thank you.".

### C1. Chuyển sang câu tiếp theo (`_TRANSITION`)
| Persona | EN | VI |
|---|---|---|
| strict | `Let's move on to the next question.` | `Chúng ta sang câu hỏi tiếp theo.` |
| neutral | `Now let's move on to the next question.` | `Bây giờ chúng ta chuyển sang câu hỏi tiếp theo.` |
| supportive | `Now let's move on to the next question together.` | `Bây giờ chúng ta cùng chuyển sang câu hỏi tiếp theo nhé.` |

### C2. Chuyển tiếp câu hỏi CUỐI (`_FINAL_QUESTION_TRANSITION`)
> ⚠️ Cũng đã bỏ "Thank you." (câu closing ngay sau đó có "Thank you" rồi).
| Persona | EN | VI |
|---|---|---|
| strict | `That was the final question.` | `Đó là câu hỏi cuối cùng.` |
| neutral | `That was the final question.` | `Đó là câu hỏi cuối cùng.` |
| supportive | `That was the final question — well done.` | `Đó là câu hỏi cuối cùng — bạn đã làm rất tốt.` |

---

## PHẦN D — PROBE / FOLLOW-UP (đào sâu câu trả lời)

### D0. KHI NÀO probe/follow-up diễn ra

Probe là hành vi **giữa phỏng vấn**, khi học viên vừa **trả lời thật** một câu (KHÔNG
phải yêu cầu như xin hint/lặp lại/skip). Nó chạy ở **rule 11** trong `decide_next_action`,
nên chỉ kích hoạt sau khi vượt qua mọi cửa ưu tiên cao hơn.

**Điều kiện bắt buộc (cả 3 phải đúng):**
1. Bộ phân tích khuyến nghị một probe — `analysis.recommended_probe_type != NONE`
   (câu trả lời mơ hồ / thiếu ví dụ / phủ mục tiêu một phần / có mâu thuẫn…).
2. Chưa cạn quota follow-up — chưa đạt **2/câu** và chưa đạt **12/toàn phiên**.
3. Thời gian chưa thấp — còn **> 20%** (`_LOW_TIME_FRACTION = 0.2`).

Thiếu bất kỳ điều nào → KHÔNG probe, mà **advance** sang câu kế / **đóng** phiên.

**Bị chặn trước (ưu tiên cao hơn probe) — nếu trúng thì probe không chạy:**
- Rule 1–8: học viên đang *yêu cầu* (repeat / clarify / hint / more time / skip /
  cannot-answer / off-topic / đòi kết thúc).
- Rule 9: thời gian chạm ngưỡng đóng (≤ 10%) → wrap up.
- Rule 10: hết quota follow-up → advance.

**Ánh xạ tình huống → loại probe** (bộ phân tích chọn `ProbeType`):

| Câu trả lời thế nào | ProbeType → action |
|---|---|
| Mơ hồ, chung chung | CLARIFICATION → clarify |
| Thiếu ví dụ cụ thể | ASK_FOR_EXAMPLE |
| Phủ một phần, cần đào lập luận | PROBE_REASONING → probe deeper |
| Cần thử thách giả định | CHALLENGE_ASSUMPTION → challenge reasoning |
| Cần bàn đánh đổi | EXPLORE_TRADEOFF |
| Mâu thuẫn với ý trước | RESOLVE_CONTRADICTION |

**Probe nâng cao (chạy SAU rule 11, chỉ khi bật feature + còn quota + còn giờ):**
- 11.5 **Depth probe** — câu trả lời **mạnh**, phase CORE/DEEP_PROBE → EXTEND_ANSWER
  (CORE) hoặc PROBE_EDGE_CASE (DEEP_PROBE): đào tìm "trần" năng lực thay vì advance.
- 11.6 **Confident-but-wrong challenge** — trả lời **tự tin nhưng sai** → CHALLENGE_REASONING.
- 11.7 **Rambling redirect** — trả lời **dài dòng, đúng chủ đề nhưng ít chất** → REDIRECT_TO_TOPIC.

Mỗi probe **tiêu 1 quota follow-up** (giữ loop protection). Xem đầy đủ thứ tự ở Phần J.

### D1. Signpost dẫn vào probe (`_probe_signpost`)
- **Hint (PROVIDE_NEUTRAL_HINT):** EN `Here's a small hint to guide you.` / VI `Đây là một gợi ý nhỏ để bạn định hướng.`
- **Depth probe (EXTEND_ANSWER / PROBE_EDGE_CASE — sau câu trả lời TỐT):** EN `That's a strong answer — let's go further.` / VI `Đó là một câu trả lời tốt — chúng ta hãy đi xa hơn.`
- **Follow-up thường (theo persona):**
  | Persona | EN | VI |
  |---|---|---|
  | strict | `Let's dig into that.` | `Chúng ta hãy đi sâu hơn.` |
  | neutral | `Let me follow up on that.` | `Tôi muốn hỏi thêm về điều đó.` |
  | supportive | `Thanks — let me follow up on that.` | `Cảm ơn bạn — tôi muốn hỏi thêm một chút.` |

### D2. Probe chung khi không có text cụ thể (`_generic_probe`)
| Action | EN | VI |
|---|---|---|
| ASK_FOR_EXAMPLE | `Could you give a concrete example?` | `Bạn có thể cho một ví dụ cụ thể không?` |
| PROBE_DEEPER | `Could you explain your reasoning further?` | `Bạn có thể giải thích rõ hơn lập luận của mình không?` |
| CHALLENGE_REASONING | `What makes you confident in that?` | `Điều gì khiến bạn tự tin về điều đó?` |
| EXPLORE_TRADEOFF | `What trade-offs would that involve?` | `Điều đó sẽ có những đánh đổi gì?` |
| RESOLVE_CONTRADICTION | `Earlier you said something that seems different — can you reconcile the two?` | `Trước đó bạn nói điều có vẻ khác — bạn có thể dung hòa hai ý đó không?` |
| CLARIFY_WITHOUT_REVEALING_ANSWER | `Which part of the question would you like me to rephrase?` | `Bạn muốn tôi diễn đạt lại phần nào của câu hỏi?` |
| PROVIDE_NEUTRAL_HINT | `A small hint: organize your answer around the main concepts in the question and how they relate.` | `Gợi ý nhỏ: hãy sắp xếp câu trả lời theo các khái niệm chính trong câu hỏi và mối quan hệ giữa chúng.` |
| REFRAME_QUESTION | `Let me put the question another way.` | `Để tôi diễn đạt câu hỏi theo cách khác.` |
| EXTEND_ANSWER | `That's solid — can you generalize it or extend it to a broader case?` | `Rất tốt — bạn có thể khái quát hóa hoặc mở rộng nó cho một trường hợp rộng hơn không?` |
| PROBE_EDGE_CASE | `Where might that break down — what edge cases or failure modes should we consider?` | `Nó có thể thất bại ở đâu — có trường hợp biên hay tình huống lỗi nào cần cân nhắc không?` |
| (fallback) | `Could you say more?` | (dùng EN) |

---

## PHẦN E — HINT LADDER (gợi ý leo thang, `_HINT_LADDER`)

Không bao giờ tiết lộ đáp án. Level clamp ở bậc cuối.
| Level | EN | VI |
|---|---|---|
| 0 | `A small hint: organize your answer around the main concepts in the question and how they relate.` | `Gợi ý nhỏ: hãy sắp xếp câu trả lời theo các khái niệm chính trong câu hỏi và mối quan hệ giữa chúng.` |
| 1 | `A bigger hint: break the question into its parts and address each one in turn — start with the definition, then the 'why', then an example.` | `Gợi ý rõ hơn: hãy chia câu hỏi thành các phần và trả lời lần lượt — bắt đầu từ định nghĩa, rồi đến 'tại sao', rồi một ví dụ.` |
| 2+ | `Let's approach it together: pick the single most central idea, state it plainly, then explain one consequence of it. You don't need the whole answer at once.` | `Chúng ta cùng tiếp cận nhé: chọn ý trọng tâm nhất, nêu rõ ràng, rồi giải thích một hệ quả của nó. Bạn không cần trả lời hết ngay.` |

---

## PHẦN F — REFRAME SIGNPOST (diễn đạt lại, `_REFRAME_SIGNPOSTS`)

Đổi theo `reframe_count` để không lặp nguyên văn.
| Lần | EN | VI |
|---|---|---|
| 0 | `Of course. Let me rephrase the question.` | `Tất nhiên. Để tôi diễn đạt lại câu hỏi.` |
| 1 | `Let me put it a different way.` | `Để tôi nói theo một cách khác.` |
| 2+ | `Here's another way to think about what I'm asking.` | `Đây là một cách khác để hiểu câu hỏi của tôi.` |

---

## PHẦN G — CÂU THOẠI THEO ACTION (self-contained, `_fallback_parts`)

| Action | EN | VI |
|---|---|---|
| REPEAT_QUESTION | `Of course. I'll repeat the question.` | `Tất nhiên. Tôi sẽ nhắc lại câu hỏi.` |
| REDIRECT_TO_TOPIC | `Let's refocus on the question.` | `Chúng ta hãy tập trung lại vào câu hỏi.` |
| OFFER_BRIEF_PAUSE | `Take a moment to think — I'll wait.` | `Bạn cứ suy nghĩ một chút — tôi sẽ đợi.` |
| HANDLE_TECHNICAL_ISSUE | `No problem — take your time to sort that out, then we'll continue.` | `Không sao — bạn cứ xử lý vấn đề đó, rồi chúng ta tiếp tục.` |
| REQUEST_END_CONFIRMATION | `Just to confirm — would you like to end and submit for grading, or continue the interview?` | `Xin xác nhận — bạn muốn kết thúc và nộp bài để chấm điểm, hay tiếp tục buổi phỏng vấn?` |
| CANCEL_END | `No problem — let's continue.` | `Không sao — chúng ta tiếp tục nhé.` |
| BEGIN_CLOSING / CLOSE_INTERVIEW (one-shot) | `Thank you. That concludes the interview.` | `Cảm ơn bạn. Buổi phỏng vấn kết thúc tại đây.` |
| PROMPT_SELF_REFLECTION | `Before we wrap up: looking back on the interview, what's one thing you feel went well, and one you'd approach differently?` | `Trước khi kết thúc: nhìn lại buổi phỏng vấn, bạn thấy điều gì mình đã làm tốt, và điều gì bạn sẽ làm khác đi?` |
| INVITE_CANDIDATE_QUESTIONS | `Thank you for sharing that. Is there anything you'd like to ask me?` | `Cảm ơn bạn đã chia sẻ. Bạn có muốn hỏi tôi điều gì không?` |
| ANSWER_CANDIDATE_QUESTION | `That's a good question. I can't share the evaluation details here, but your instructor will follow up with feedback and results.` | `Đó là một câu hỏi hay. Tôi không thể chia sẻ chi tiết đánh giá ở đây, nhưng giảng viên của bạn sẽ phản hồi kèm kết quả sau.` |
| DEESCALATE | `That's completely okay — take a breath. There's no penalty here; let's take it one step at a time.` | `Không sao đâu — bạn cứ bình tĩnh. Không có điểm trừ gì cả; chúng ta cứ đi từng bước một nhé.` |
| DEFER_CANDIDATE_QUESTION | `Good question — let's come back to that at the end. For now, let's stay with the current one.` | `Câu hỏi hay — mình sẽ quay lại cuối buổi nhé. Bây giờ, chúng ta tiếp tục với câu hiện tại.` |

---

## PHẦN H — LEAD-IN THEO CẢM XÚC / BỐI CẢNH (prepend vào ack — TONE ONLY)

Chỉ thêm MỘT lead-in, ưu tiên: **recovery > time_pressure > affect**.

### H1. Affect (`_AFFECT_LEAD_IN`)
| Affect | EN | VI |
|---|---|---|
| nervous | `No rush — you're doing fine.` | `Bạn cứ từ từ — bạn đang làm tốt mà.` |
| rambling | `Let's focus in a little.` | `Chúng ta hãy tập trung lại một chút.` |
| terse | `Feel free to expand.` | `Bạn cứ trình bày thêm nhé.` |

### H2. Time pressure (`_TIME_PRESSURE_LEAD_IN`)
- **EN:** `We're a little short on time, so let's prioritise.`
- **VI:** `Chúng ta còn hơi ít thời gian, nên hãy tập trung vào điểm chính.`

### H3. Recovery — sau chuỗi câu yếu (`_RECOVERY_LEAD_IN`)
- **EN:** `No problem — let's take a fresh, straightforward one.`
- **VI:** `Không sao — mình thử một câu nhẹ nhàng, rõ ràng hơn nhé.`

---

## PHẦN I — CLOSING (kết thúc, `closing_text`) — 3 lý do

Có biến `{, name}` và `{title}`. Supportive thêm "for your time and effort".
- **middle theo reason:**
  - natural → EN `That concludes your "{title}" interview.` / VI `Buổi phỏng vấn "{title}" đến đây là kết thúc.`
  - ended_early → EN `We'll end your "{title}" interview here.` / VI `Chúng ta sẽ kết thúc sớm buổi phỏng vấn "{title}" tại đây.`
  - timed_out → EN `The time for your "{title}" interview has ended.` / VI `Thời gian cho buổi phỏng vấn "{title}" đã kết thúc.`
- **EN:** `Thank you{ for your time and effort}{, name}. {middle} Your submitted responses have been recorded for evaluation. Goodbye.`
- **VI:** `Cảm ơn bạn{ name}{ đã dành thời gian và nỗ lực}. {middle} Các câu trả lời đã gửi của bạn đã được ghi nhận để đánh giá. Tạm biệt.`

---

## PHẦN J — LOGIC CHỌN ACTION (decision.py) — case nào gọi câu nào

Thứ tự ưu tiên (cao → thấp) trong `decide_next_action`:

0. **Rich closing đang chạy** (phase CLOSING + bật) → chạy sub-sequence: self-reflection → invite questions → answer/​sign-off (Phần G).
1–8. **Request / non-answer** (`_decide_from_intent_request`):
   - FRUSTRATED (nếu bật) → DEESCALATE, giữ nguyên câu, không chấm.
   - ASK_INTERVIEWER_QUESTION ngoài closing (nếu bật) → DEFER_CANDIDATE_QUESTION.
   - Đang chờ xác nhận kết thúc (`pending_confirmation`): confirm/end → BEGIN_CLOSING; khác → CANCEL_END (quay lại câu hiện tại).
   - END_INTERVIEW lần đầu → REQUEST_END_CONFIRMATION (KHÔNG đóng ngay).
   - TECHNICAL_ISSUE → HANDLE_TECHNICAL_ISSUE; ASK_TO_REPEAT → REPEAT_QUESTION; ASK_FOR_CLARIFICATION → CLARIFY; ASK_FOR_HINT → PROVIDE_NEUTRAL_HINT; ASK_FOR_MORE_TIME → OFFER_BRIEF_PAUSE. (Không chấm điểm.)
   - SKIP_QUESTION → SKIP_QUESTION (đánh dấu bỏ qua, advance).
   - CANNOT_ANSWER → advance, ghi "bằng chứng không đủ".
   - OFF_TOPIC → REDIRECT_TO_TOPIC (lần đầu), lần sau → advance.
9. **Hết giờ** (`time_fraction_remaining ≤ closing_time_fraction=0.1`) → advance/close, ghi bằng chứng.
10. **Hết quota follow-up** (2/câu hoặc 12/phiên) → advance/close.
11. **Analysis đề nghị probe** (còn quota + còn giờ) → probe tương ứng (Phần D).
   - 11.5 Câu trả lời TỐT + bật depth_probe → EXTEND_ANSWER (CORE) / PROBE_EDGE_CASE (DEEP_PROBE).
   - 11.6 Tự tin nhưng SAI + bật → CHALLENGE_REASONING (style CORRECTIVE).
   - 11.7 Lan man on-topic + bật → REDIRECT_TO_TOPIC.
12. **Còn lại** → advance sang câu tiếp (TRANSITION_TOPIC, Phần C1). Nếu mọi outcome bắt buộc đã đủ → BEGIN_CLOSING. Nếu không còn câu → BEGIN_CLOSING.
   - Nếu tự sửa lỗi (self-correction, bật) → nâng ack lên POSITIVE.

**Giới hạn loop:** `DEFAULT_MAX_FOLLOWUPS_PER_QUESTION = 2`, `DEFAULT_MAX_TOTAL_FOLLOWUPS = 12`.
**Ngưỡng thời gian:** `_LOW_TIME_FRACTION = 0.2` (ngừng đào sâu), `closing_time_fraction = 0.1` (bắt đầu đóng).

---

## GHI CHÚ QUAN TRỌNG

- **LLM chỉ diễn đạt lại (phrasing)** — không đổi action/reason/advance/close/scoring. Nếu LLM lỗi → dùng thẳng template deterministic trên đây (`build_fallback_utterance`).
- **Không bao giờ tiết lộ đáp án** ở mọi probe/hint (requirement #8).
- **Persona = giọng điệu**, không đổi logic chấm hay đậu/rớt.
- Ack neutral mở đầu bằng "Thank you." → các transition đã được bỏ "Thank you." để không double (fix 2025-07).
