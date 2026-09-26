#!/usr/bin/env bash
# Mint a fresh access token via POST /api/v1/auth/refresh.
#
# The backend ROTATES refresh tokens on every refresh, so the token that was
# used must never be reused. This script keeps the latest one in
# .state/refresh_token (chmod 600) and seeds it from $REFRESH_TOKEN the first
# time. Prints the fresh access token on stdout.
#
# Usage: REFRESH_TOKEN=xxx ./refresh-token.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$HERE/.state"
STATE_FILE="$STATE_DIR/refresh_token"
BASE_URL="${BASE_URL:-https://abridgeai.tech}"

RT="${REFRESH_TOKEN:-}"
if [[ -z "$RT" && -f "$STATE_FILE" ]]; then
  RT="$(cat "$STATE_FILE")"
fi
if [[ -z "$RT" ]]; then
  echo "error: set REFRESH_TOKEN env (first run) — it will be stored in $STATE_FILE afterwards" >&2
  exit 1
fi

mkdir -p "$STATE_DIR"

RESP="$(curl -sS -m 20 -X POST "$BASE_URL/api/v1/auth/refresh" \
  -H 'Content-Type: application/json' \
  -d "{\"refresh_token\": \"$RT\"}")"

ACCESS="$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("access_token",""))')"
NEW_RT="$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("refresh_token",""))')"

if [[ -z "$ACCESS" || -z "$NEW_RT" ]]; then
  echo "error: refresh failed: $RESP" >&2
  exit 1
fi

printf '%s' "$NEW_RT" > "$STATE_FILE"
chmod 600 "$STATE_FILE"
printf '%s' "$ACCESS"
