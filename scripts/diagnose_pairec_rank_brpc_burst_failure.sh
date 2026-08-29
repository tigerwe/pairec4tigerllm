#!/usr/bin/env bash
# Collect one Rank BRPC burst failure without sending traffic or changing deployments.
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_APP="${PAIREC_APP:-pairec-brpc-observed-wrapper}"
RANK_WRAPPER_APP="${RANK_WRAPPER_APP:-deepfm-rank-burst-wrapper}"
RANK_BACKEND_APP="${RANK_BACKEND_APP:-deepfm-rank-brpc-worker1}"
POST_HOP1_APP="${POST_HOP1_APP:-post-rank-hop1}"
POST_HOP2_APP="${POST_HOP2_APP:-post-rank-hop2}"
REQUEST_ID="${REQUEST_ID:-${1:-}}"
RUN_ROOT="${RUN_ROOT:-${2:-}}"
LOG_SINCE="${LOG_SINCE:-60m}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/pairec-rank-brpc-burst-diagnostic}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d-%H%M%S)}"

die() { echo "ERROR: $*" >&2; exit 1; }
for command in kubectl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
[[ -z "$RUN_ROOT" || -d "$RUN_ROOT" ]] || die "RUN_ROOT does not exist: $RUN_ROOT"
mkdir -p "$OUTPUT_DIR"

discover_failed_request() {
  local root="$1"
  python3 - "$root" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
candidates=[]
for path in root.rglob("response-*.json"):
    try:
        data=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        continue
    if data.get("code") != 200 and data.get("request_id"):
        candidates.append((path.stat().st_mtime,str(data["request_id"])))
print(max(candidates)[1] if candidates else "")
PY
}

if [[ -z "$REQUEST_ID" && -n "$RUN_ROOT" ]]; then
  REQUEST_ID="$(discover_failed_request "$RUN_ROOT")"
fi

collect_app_logs() {
  local app="$1" preferred_container="$2" output="$3"
  : >"$output"
  mapfile -t pods < <(kubectl -n "$NAMESPACE" get pods -l "app=$app" \
    --sort-by=.metadata.creationTimestamp -o name 2>/dev/null || true)
  for pod_ref in "${pods[@]}"; do
    [[ -n "$pod_ref" ]] || continue
    local pod="${pod_ref#pod/}" containers container restart_count
    containers="$(kubectl -n "$NAMESPACE" get pod "$pod" \
      -o jsonpath='{.spec.containers[*].name}' 2>/dev/null || true)"
    for container in $containers; do
      [[ -z "$preferred_container" || "$container" = "$preferred_container" ]] || continue
      {
        echo "===== pod=$pod container=$container current ====="
        kubectl -n "$NAMESPACE" logs "$pod" -c "$container" \
          --since="$LOG_SINCE" --timestamps 2>&1 || true
        restart_count="$(kubectl -n "$NAMESPACE" get pod "$pod" \
          -o "jsonpath={.status.containerStatuses[?(@.name=='$container')].restartCount}" \
          2>/dev/null || echo 0)"
        if [[ "${restart_count:-0}" != 0 ]]; then
          echo "===== pod=$pod container=$container previous ====="
          kubectl -n "$NAMESPACE" logs "$pod" -c "$container" \
            --previous --timestamps 2>&1 || true
        fi
      } >>"$output"
    done
  done
}

echo "== Collect Rank BRPC failure evidence =="
collect_app_logs "$PAIREC_APP" pairec "$OUTPUT_DIR/pairec.log"
collect_app_logs "$RANK_WRAPPER_APP" rank-burst-wrapper "$OUTPUT_DIR/rank-wrapper.log"
collect_app_logs "$RANK_WRAPPER_APP" rank-kvc-burst-wrapper "$OUTPUT_DIR/rank-kvc.log"
collect_app_logs "$RANK_BACKEND_APP" adapter "$OUTPUT_DIR/rank-adapter.log"
collect_app_logs "$RANK_BACKEND_APP" backend "$OUTPUT_DIR/rank-backend.log"
collect_app_logs "$POST_HOP1_APP" post-rank-hop1 "$OUTPUT_DIR/post-rank-hop1.log"
collect_app_logs "$POST_HOP2_APP" post-rank-hop2 "$OUTPUT_DIR/post-rank-hop2.log"

if [[ -z "$REQUEST_ID" ]]; then
  REQUEST_ID="$(python3 - "$OUTPUT_DIR/pairec.log" <<'PY'
import json,sys
latest=""
for line in open(sys.argv[1],errors="replace"):
    pos=line.find("{")
    if pos<0: continue
    try: event=json.loads(line[pos:])
    except json.JSONDecodeError: continue
    if event.get("event")=="deepfm_rank_error" and event.get("request_id"):
        latest=str(event["request_id"])
print(latest)
PY
)"
fi
[[ -n "$REQUEST_ID" ]] \
  || die "set REQUEST_ID or RUN_ROOT containing a failed warmup/response-*.json"

for source in pairec rank-wrapper rank-kvc rank-adapter rank-backend post-rank-hop1 post-rank-hop2; do
  grep -F "$REQUEST_ID" "$OUTPUT_DIR/${source}.log" \
    >"$OUTPUT_DIR/${source}-request.log" 2>/dev/null || true
done

if [[ -n "$RUN_ROOT" ]]; then
  find "$RUN_ROOT" -type f -print >"$OUTPUT_DIR/run-files.txt"
  grep -R -F "$REQUEST_ID" "$RUN_ROOT" \
    >"$OUTPUT_DIR/run-request.log" 2>/dev/null || true
else
  : >"$OUTPUT_DIR/run-files.txt"
  : >"$OUTPUT_DIR/run-request.log"
fi

apps="${PAIREC_APP},${RANK_WRAPPER_APP},${RANK_BACKEND_APP},${POST_HOP1_APP},${POST_HOP2_APP}"
kubectl -n "$NAMESPACE" get pods -l "app in ($apps)" -o wide \
  >"$OUTPUT_DIR/pods.txt" 2>&1 || true
kubectl -n "$NAMESPACE" get pods -l "app in ($apps)" -o json \
  >"$OUTPUT_DIR/pods.json" 2>&1 || true
kubectl -n "$NAMESPACE" get deployment \
  "$PAIREC_APP" "$RANK_WRAPPER_APP" "$RANK_BACKEND_APP" \
  "$POST_HOP1_APP" "$POST_HOP2_APP" -o yaml \
  >"$OUTPUT_DIR/deployments.yaml" 2>&1 || true
kubectl -n "$NAMESPACE" get service \
  "$PAIREC_APP" "$RANK_WRAPPER_APP" "$RANK_BACKEND_APP" \
  "$POST_HOP1_APP" "$POST_HOP2_APP" -o wide \
  >"$OUTPUT_DIR/services.txt" 2>&1 || true
kubectl -n "$NAMESPACE" get endpoints \
  "$PAIREC_APP" "$RANK_WRAPPER_APP" "$RANK_BACKEND_APP" \
  "$POST_HOP1_APP" "$POST_HOP2_APP" -o wide \
  >"$OUTPUT_DIR/endpoints.txt" 2>&1 || true
kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp \
  >"$OUTPUT_DIR/events.txt" 2>&1 || true

for app in "$PAIREC_APP" "$RANK_WRAPPER_APP" "$RANK_BACKEND_APP" \
  "$POST_HOP1_APP" "$POST_HOP2_APP"; do
  mapfile -t pods < <(kubectl -n "$NAMESPACE" get pods -l "app=$app" -o name 2>/dev/null || true)
  for pod_ref in "${pods[@]}"; do
    pod="${pod_ref#pod/}"
    kubectl -n "$NAMESPACE" describe pod "$pod" \
      >"$OUTPUT_DIR/describe-${pod}.txt" 2>&1 || true
  done
done

python3 scripts/summarize_pairec_rank_brpc_failure.py \
  --request-id "$REQUEST_ID" \
  --pairec "$OUTPUT_DIR/pairec-request.log" \
  --wrapper "$OUTPUT_DIR/rank-wrapper-request.log" \
  --rank-kvc "$OUTPUT_DIR/rank-kvc-request.log" \
  --adapter "$OUTPUT_DIR/rank-adapter-request.log" \
  --backend "$OUTPUT_DIR/rank-backend-request.log" \
  --hop1 "$OUTPUT_DIR/post-rank-hop1-request.log" \
  --hop2 "$OUTPUT_DIR/post-rank-hop2-request.log" \
  --artifacts "$OUTPUT_DIR/run-request.log" \
  --output "$OUTPUT_DIR/summary.json" \
  | tee "$OUTPUT_DIR/summary.txt"

echo
echo "== Request evidence tails =="
for source in pairec rank-wrapper rank-kvc rank-adapter rank-backend post-rank-hop1 post-rank-hop2 run; do
  echo "-- $source --"
  if [[ -s "$OUTPUT_DIR/${source}-request.log" ]]; then
    tail -20 "$OUTPUT_DIR/${source}-request.log"
  else
    echo "REQUEST_SPECIFIC_LOG_NOT_FOUND"
  fi
done
echo
echo "output_dir=$OUTPUT_DIR"
echo "PAIREC_RANK_BRPC_BURST_DIAGNOSTIC_COMPLETE"
