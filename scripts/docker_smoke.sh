#!/usr/bin/env bash
# Smoke test of the built container image (used by CI and `make docker-smoke`).
#
# Runs the image with the offline `hash` embedder and a test API key, then checks:
#   1. GET /health returns 200 within HEALTH_TIMEOUT_S;
#   2. POST /ask without a key is rejected with 401 (auth gate is closed);
#   3. POST /ask with the key answers a golden question with >= 1 source, which
#      exercises the lazy corpus ingestion inside the image;
#   4. the Dockerfile HEALTHCHECK reports `healthy`.
#
# No Ollama is reachable inside the container, so generation fails fast
# (connection refused) and the agent falls back to its extractive answer.
set -euo pipefail

IMAGE="${1:-anchora:ci}"
PORT="${PORT:-8000}"
API_KEY="ci-smoke"
HEALTH_TIMEOUT_S=30
HEALTHY_TIMEOUT_S=60
ASK_TIMEOUT_S=30
QUESTION="What is the deadline for a public agency to respond to an access to information request?"
BASE="http://127.0.0.1:${PORT}"

cid=$(docker run -d -p "${PORT}:8000" \
  -e ANCHORA_EMBED_PROVIDER=hash \
  -e ANCHORA_API_KEY="${API_KEY}" \
  "${IMAGE}")

cleanup() {
  status=$?
  if [ "${status}" -ne 0 ]; then
    echo "--- container logs ---"
    docker logs "${cid}" || true
  fi
  docker rm -f "${cid}" >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

echo "waiting for /health (up to ${HEALTH_TIMEOUT_S}s)"
deadline=$((SECONDS + HEALTH_TIMEOUT_S))
until [ "$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health")" = "200" ]; do
  [ "${SECONDS}" -lt "${deadline}" ] || fail "/health did not return 200 in ${HEALTH_TIMEOUT_S}s"
  sleep 1
done
echo "ok: /health 200"

body=$(printf '{"question": "%s"}' "${QUESTION}")

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/ask" \
  -H 'content-type: application/json' -d "${body}")
[ "${code}" = "401" ] || fail "POST /ask without key returned ${code}, expected 401"
echo "ok: /ask without key 401"

resp=$(mktemp)
code=$(curl -s -o "${resp}" -w '%{http_code}' --max-time "${ASK_TIMEOUT_S}" -X POST "${BASE}/ask" \
  -H 'content-type: application/json' -H "x-api-key: ${API_KEY}" -d "${body}")
[ "${code}" = "200" ] || fail "POST /ask with key returned ${code}: $(cat "${resp}")"
sources=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["sources"]))' "${resp}")
[ "${sources}" -ge 1 ] || fail "POST /ask returned no sources: $(cat "${resp}")"
echo "ok: /ask with key 200, ${sources} source(s)"

echo "waiting for HEALTHCHECK (up to ${HEALTHY_TIMEOUT_S}s)"
deadline=$((SECONDS + HEALTHY_TIMEOUT_S))
until [ "$(docker inspect --format '{{.State.Health.Status}}' "${cid}")" = "healthy" ]; do
  [ "${SECONDS}" -lt "${deadline}" ] || fail "container not healthy in ${HEALTHY_TIMEOUT_S}s"
  sleep 1
done
echo "ok: container healthy"
