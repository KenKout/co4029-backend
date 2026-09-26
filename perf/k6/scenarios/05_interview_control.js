// 05_interview_control.js — AI voice-interview control plane.
// Covers the lifecycle the previous evaluation skipped entirely (AI pipelines):
//   start session (LLM opening turn) → mint WARM LiveKit token →
//   onboarding state machine → poll session state → [optional LLM turns via
//   /respond] → finish (queues the evaluation job).
//
// ⚠ COST WARNING: session start, each /respond turn and finish all hit the
// LLM gateway (and finish enqueues a grading job). Defaults are tiny on
// purpose — raise only if you accept the token spend. This does NOT
// load-test LiveKit media (WebRTC), only the HTTP control plane.

import { sleep } from 'k6';

import { INTERVIEW_CONFIG_ID, INTERVIEW_TURNS, VUS, summaryTrendStats } from '../config.js';
import { apiPost } from '../lib/client.js';
import {
  completeOnboarding,
  finishInterviewSession,
  mintWarmToken,
  pollSessionState,
  startInterviewSession,
} from '../lib/interview_journey.js';
import { handleSummary } from '../lib/report.js';

export const options = {
  summaryTrendStats,
  scenarios: {
    interview: {
      executor: 'per-vu-iterations',
      vus: Math.min(VUS, 5),
      iterations: Number(__ENV.INTERVIEW_ITERS || 1),
      maxDuration: __ENV.DURATION || '10m',
      tags: { scenario: 'interview_control' },
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'],
    'http_req_duration{name:POST /api/v1/interview-configs/{config_id}/sessions}': ['p(95)<10000'],
    'http_req_duration{name:GET /api/v1/interview-sessions/{id}}': ['p(95)<500'],
  },
};

export default function () {
  const session = startInterviewSession();
  if (!session) return;
  const sid = session.id;

  mintWarmToken(sid); // allowed during onboarding — the cheap mint path
  completeOnboarding(sid, session.onboardingStage);

  pollSessionState(sid, 3, 2);

  for (let t = 0; t < INTERVIEW_TURNS; t++) {
    const turn = apiPost(
      `/api/v1/interview-sessions/${sid}/respond`,
      { text: 'I would start by normalizing the schema, then denormalize the star.' },
      'POST /api/v1/interview-sessions/{id}/respond',
    );
    if (turn.status >= 400) {
      console.warn('respond failed ' + turn.status + ': ' + String(turn.body).slice(0, 200));
      break;
    }
    sleep(2);
  }

  finishInterviewSession(sid);
}

export { handleSummary };
