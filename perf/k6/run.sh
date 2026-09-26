#!/usr/bin/env bash
# Convenience runner: ensures a fresh access token, then executes a scenario.
#
#   ./run.sh 01_smoke
#   ./run.sh 02_browse_mix                    # VUS=50 DURATION=2m defaults
#   VUS=100 DURATION=5m ./run.sh 03_quiz_take
#   ./run.sh 06_stress_ramp                   # RAMP_PROFILE="50:30s,100:1m,150:1m,200:1m"
#
# Also installs k6 locally (./bin/k6) on first run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO="$1"; shift || true

# ── k6 binary ────────────────────────────────────────────────────────────────
if ! command -v k6 >/dev/null 2>&1; then
  if [[ ! -x "$HERE/bin/k6" ]]; then
    mkdir -p "$HERE/bin"
    echo ">> downloading k6 ..."
    VER="$(curl -sS -m 20 https://api.github.com/repos/grafana/k6/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"].lstrip("v"))')"
    curl -sS -m 120 -L "https://github.com/grafana/k6/releases/download/v${VER}/k6-v${VER}-linux-amd64.tar.gz" \
      | tar -xz -C "$HERE/bin" --strip-components=1 "k6-v${VER}-linux-amd64/k6"
    chmod +x "$HERE/bin/k6"
  fi
  export PATH="$HERE/bin:$PATH"
fi

# ── auth ─────────────────────────────────────────────────────────────────────
# Multi-user mode: .state/users.json (minted by bin/mint_loadtest_users.py)
# takes precedence — 30-day tokens, no refresh needed.
if [[ -z "${ACCESS_TOKEN:-}" && -z "${USERS_FILE:-}" && -f "$HERE/.state/users.json" ]]; then
  export USERS_FILE="$HERE/.state/users.json"
  echo ">> multi-user mode: USERS_FILE=$USERS_FILE ($(python3 -c "import json;print(len(json.load(open('$USERS_FILE'))))") users)"
fi

if [[ -z "${ACCESS_TOKEN:-}" && -z "${USERS_FILE:-}" ]]; then
  echo ">> no ACCESS_TOKEN / users.json — refreshing via refresh-token.sh"
  ACCESS_TOKEN="$("$HERE/refresh-token.sh")"
  export ACCESS_TOKEN
elif [[ -n "${ACCESS_TOKEN:-}" && -z "${USERS_FILE:-}" ]]; then
  # Refresh only if the provided token is expired/nearly expired (<60s left).
  EXP="$(printf '%s' "$ACCESS_TOKEN" | cut -d. -f2 | tr '_-' '/+' | python3 -c '
import base64,sys
s = sys.stdin.read().strip()
s += "=" * (-len(s) % 4)
import json; print(json.loads(base64.b64decode(s)).get("exp", 0))')"
  NOW="$(date +%s)"
  if (( EXP - NOW < 60 )); then
    echo ">> ACCESS_TOKEN expired — refreshing"
    ACCESS_TOKEN="$("$HERE/refresh-token.sh")"
    export ACCESS_TOKEN
  fi
fi

echo ">> running scenarios/$SCENARIO.js $*"
exec k6 run "$HERE/scenarios/$SCENARIO.js" "$@"
