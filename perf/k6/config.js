// Central configuration for the k6 load-test suite.
// Every value can be overridden via environment variables (see .env.example).

export const BASE_URL = (__ENV.BASE_URL || 'https://abridgeai.tech').replace(/\/$/, '');

// Pre-obtained JWT access token (Google OAuth is out of scope for load tests,
// mirroring the methodology from the specialized-project evaluation).
export const ACCESS_TOKEN = __ENV.ACCESS_TOKEN || '';

// Multi-user mode: mint N synthetic students on the server (see
// bin/mint_loadtest_users.py) and point USERS_FILE at the produced JSON.
// Each VU then authenticates as its own student, which unlocks true
// concurrent quiz attempts (the one-active-attempt-per-user guard no longer
// collides) and realistic per-user SR/progress traffic.
let loadtestUsers = [];
if (__ENV.USERS_FILE) {
  try {
    loadtestUsers = JSON.parse(open(__ENV.USERS_FILE));
  } catch (err) {
    throw new Error(`USERS_FILE=${__ENV.USERS_FILE} could not be read: ${err}`);
  }
}
export const LOADTEST_USERS = loadtestUsers;

// Business ids probed on the deployment (Data Warehouses & DSS course).
export const QUIZ_ID = __ENV.QUIZ_ID || 'd3e2fe10-19b7-482d-bb68-37a793a3f4fa';
export const COURSE_ID = __ENV.COURSE_ID || 'e5ba475d-140e-4a82-9a6b-b283b8e760ac';
export const INTERVIEW_CONFIG_ID =
  __ENV.INTERVIEW_CONFIG_ID || 'caec31d5-cb16-486d-b4fa-b9b522d9bee6';

// Traffic shaping.
export const VUS = Number(__ENV.VUS || 50);
export const DURATION = __ENV.DURATION || '2m';
export const THINK_TIME_S = Number(__ENV.THINK_TIME_S || 2);

// Quiz journey tuning.
export const HEARTBEAT_EVERY_N_ANSWERS = Number(__ENV.HEARTBEAT_EVERY_N_ANSWERS || 2);

// Spaced-repetition journey tuning.
export const SR_CARDS_PER_ITER = Number(__ENV.SR_CARDS_PER_ITER || 3);

// Interview journey tuning. The interview pipeline hits the LLM gateway on
// session start / respond / finish — keep VUS and iterations tiny unless you
// explicitly accept the token cost.
export const INTERVIEW_TURNS = Number(__ENV.INTERVIEW_TURNS || 0);

export function authHeaders() {
  return {
    Authorization: `Bearer ${currentAccessToken()}`,
    'Content-Type': 'application/json',
  };
}

export function currentAccessToken() {
  if (LOADTEST_USERS.length) {
    // 1-based __VU → stable per-VU user (wraps if VUs outnumber users).
    return LOADTEST_USERS[(__VU - 1 + LOADTEST_USERS.length) % LOADTEST_USERS.length].access_token;
  }
  if (ACCESS_TOKEN) {
    return ACCESS_TOKEN;
  }
  throw new Error(
    'No token: set ACCESS_TOKEN, or mint users (bin/mint_loadtest_users.py) ' +
      'and export USERS_FILE=/abs/path/.state/users.json',
  );
}

// Default latency/error targets evaluated against NFR-2.1 (tune per scenario).
export const defaultThresholds = {
  http_req_failed: ['rate<0.01'],
  http_req_duration: ['p(50)<400', 'p(95)<1000'],
};

// Ensure handleSummary receives count + p99 for the per-endpoint table
// (k6 v2 defaults to avg/min/med/max/p(90)/p(95) only).
export const summaryTrendStats = ['med', 'p(90)', 'p(95)', 'p(99)', 'max', 'count'];
