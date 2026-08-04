#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec-brpc-wrapper}"
WRAPPER_DEPLOYMENT="${WRAPPER_DEPLOYMENT:-brpc-burst-wrapper}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
REQUESTS="${REQUESTS:-3}"
USER_ID="${USER_ID:-5}"
SIZE="${SIZE:-8}"
SCENE_ID="${SCENE_ID:-home_feed}"
OUT_DIR="${OUT_DIR:-/tmp/pairec-brpc-wrapper-smoke/$(date +%Y%m%d-%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$SIZE" =~ ^[1-9][0-9]*$ ]] || die "SIZE must be positive"
for command in kubectl curl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: ${command}"
done

mkdir -p "$OUT_DIR"
STARTED_AT="$(date --iso-8601=seconds)"

pod_state() {
  local deployment="$1"
  kubectl -n "$NAMESPACE" get pod -l "app=${deployment}" -o json | python3 -c '
import json, sys
pods = json.load(sys.stdin).get("items", [])
ready = []
for pod in pods:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if pod.get("status", {}).get("phase") == "Running" and statuses and all(s.get("ready") for s in statuses):
        ready.append((pod, statuses))
if len(ready) != 1:
    raise SystemExit(f"expected one ready pod, found {len(ready)}")
pod, statuses = ready[0]
print("pod=" + pod["metadata"]["name"])
print("uid=" + pod["metadata"]["uid"])
print("restarts=" + str(sum(s.get("restartCount", 0) for s in statuses)))
'
}

for deployment in "$PAIREC_DEPLOYMENT" "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT"; do
  pod_state "$deployment" >"${OUT_DIR}/${deployment}.before"
done

SERVICE_IP="$(kubectl -n "$NAMESPACE" get service "$PAIREC_DEPLOYMENT" \
  -o jsonpath='{.spec.clusterIP}')"
test -n "$SERVICE_IP" && test "$SERVICE_IP" != "None" \
  || die "service/${PAIREC_DEPLOYMENT} has no ClusterIP"
PAIREC_URL="http://${SERVICE_IP}:18080/api/recommend"

printf 'index\thttp_code\te2e_ms\trequest_id\titem_count\n' >"${OUT_DIR}/requests.tsv"
for index in $(seq 1 "$REQUESTS"); do
  body_file="${OUT_DIR}/response-${index}.json"
  metrics_file="${OUT_DIR}/curl-${index}.txt"
  curl -sS "$PAIREC_URL" \
    -H 'Content-Type: application/json' \
    -d "{\"scene_id\":\"${SCENE_ID}\",\"uid\":\"${USER_ID}\",\"size\":${SIZE}}" \
    -o "$body_file" \
    -w '%{http_code}\t%{time_total}\n' >"$metrics_file"

  read -r http_code e2e_seconds <"$metrics_file"
  read -r response_code request_id item_count < <(
    python3 - "$body_file" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data.get("code"), data.get("request_id", ""), len(data.get("items", [])))
PY
  )
  test "$http_code" = "200" || die "request ${index} HTTP status=${http_code}"
  test "$response_code" = "200" || die "request ${index} response code=${response_code}"
  test "$item_count" = "$SIZE" || die "request ${index} items=${item_count}, expected=${SIZE}"
  test -n "$request_id" || die "request ${index} has no request_id"
  e2e_ms="$(python3 -c 'import sys; print(f"{float(sys.argv[1]) * 1000:.3f}")' "$e2e_seconds")"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$index" "$http_code" "$e2e_ms" "$request_id" "$item_count" \
    | tee -a "${OUT_DIR}/requests.tsv"
done

for deployment in "$PAIREC_DEPLOYMENT" "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT"; do
  pod_state "$deployment" >"${OUT_DIR}/${deployment}.after"
  cmp -s "${OUT_DIR}/${deployment}.before" "${OUT_DIR}/${deployment}.after" \
    || die "deployment/${deployment} pod identity or restart count changed"
  pod="$(awk -F= '$1 == "pod" {print $2}' "${OUT_DIR}/${deployment}.after")"
  kubectl -n "$NAMESPACE" logs "$pod" --since-time="$STARTED_AT" \
    >"${OUT_DIR}/${deployment}.log" 2>&1
done

while IFS=$'\t' read -r index _ _ request_id _; do
  test "$index" = "index" && continue
  grep -F "request_id=${request_id}" "${OUT_DIR}/${PAIREC_DEPLOYMENT}.log" >/dev/null \
    || die "PaiRec trace missing request_id=${request_id}"
  grep -F "request_id=${request_id}" "${OUT_DIR}/${WRAPPER_DEPLOYMENT}.log" >/dev/null \
    || die "Wrapper trace missing request_id=${request_id}"
  grep -F "request_id=${request_id}" "${OUT_DIR}/${INFERENCE_DEPLOYMENT}.log" >/dev/null \
    || die "inference trace missing request_id=${request_id}"
done <"${OUT_DIR}/requests.tsv"

if grep -F 'from=cache' "${OUT_DIR}/${PAIREC_DEPLOYMENT}.log" >/dev/null; then
  die "PaiRec cache hit detected while cache must be disabled"
fi
if grep -Eqi 'fallback|brpc request failed|Segmentation|core dumped|Out of memory' \
    "${OUT_DIR}/${PAIREC_DEPLOYMENT}.log" \
    "${OUT_DIR}/${WRAPPER_DEPLOYMENT}.log" \
    "${OUT_DIR}/${INFERENCE_DEPLOYMENT}.log"; then
  die "fallback, RPC failure, or crash marker detected"
fi

echo "PAIREC_BRPC_WRAPPER_SMOKE_OK requests=${REQUESTS} user_id=${USER_ID} size=${SIZE}"
echo "endpoint=${PAIREC_URL}"
echo "output_dir=${OUT_DIR}"
