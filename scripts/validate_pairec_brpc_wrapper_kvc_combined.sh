#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
KVC_CONCURRENCY=${KVC_CONCURRENCY:-10}
KVC_PRESSURE_KEY_COUNT=${KVC_PRESSURE_KEY_COUNT:-9}
KVC_OBJECT_SIZE=${KVC_OBJECT_SIZE:-3670016}
KVC_PRESSURE_LEAD_US=${KVC_PRESSURE_LEAD_US:-1000}
WRAPPER_CONCURRENCY=${WRAPPER_CONCURRENCY:-1}
REQUESTS=${REQUESTS:-3}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
BURST_POOL_SIZE=${BURST_POOL_SIZE:-$WRAPPER_CONCURRENCY}
BURST_ACTIVE_CONNECTIONS=${BURST_ACTIVE_CONNECTIONS:-$WRAPPER_CONCURRENCY}
BUSINESS_PAYLOAD_BYTES=${BUSINESS_PAYLOAD_BYTES:-102400}
BRPC_PRESSURE_PAYLOAD_BYTES=${BRPC_PRESSURE_PAYLOAD_BYTES:-102400}
WRAPPER_ENDPOINT=${WRAPPER_ENDPOINT:-192.168.100.11:18103}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-kvc-combined/$(date +%Y%m%d-%H%M%S)-c${KVC_CONCURRENCY}-n${REQUESTS}}
KVC_OVERLAY_BACKUP=${KVC_OVERLAY_BACKUP:-$OUTPUT_DIR/kvc-deployment-before.json}
WRAPPER_OUTPUT_DIR=${WRAPPER_OUTPUT_DIR:-$OUTPUT_DIR/wrapper}
CONTENTION_OUTPUT_DIR=${CONTENTION_OUTPUT_DIR:-$OUTPUT_DIR/contention}
MIN_ROOT_AVAILABLE_KB=${MIN_ROOT_AVAILABLE_KB:-5242880}
EXPECTED_ONBOARDS_MIN=${EXPECTED_ONBOARDS_MIN:-2}
EXPECTED_ONBOARDS_MAX=${EXPECTED_ONBOARDS_MAX:-2}
KVC_SUSTAINED_PRESSURE=${KVC_SUSTAINED_PRESSURE:-0}
KVC_SUSTAINED_MAX_DURATION_MS=${KVC_SUSTAINED_MAX_DURATION_MS:-5000}
KVC_SUSTAINED_MAX_LOOPS=${KVC_SUSTAINED_MAX_LOOPS:-1000}
KVC_INPROCESS_PRESSURE=${KVC_INPROCESS_PRESSURE:-0}
PRIME_REQUESTS=${PRIME_REQUESTS:-195}
BUILD_PAIREC_IMAGE=${BUILD_PAIREC_IMAGE:-0}
IMPORT_PAIREC_IMAGE=${IMPORT_PAIREC_IMAGE:-0}
RANK_BURST_ENABLED=${RANK_BURST_ENABLED:-0}
RANK_BURST_CONCURRENCY=${RANK_BURST_CONCURRENCY:-1}
RANK_BURST_POOL_SIZE=${RANK_BURST_POOL_SIZE:-$RANK_BURST_CONCURRENCY}
RANK_BURST_PAYLOAD_BYTES=${RANK_BURST_PAYLOAD_BYTES:-102400}
RANK_BUSINESS_PAYLOAD_BYTES=${RANK_BUSINESS_PAYLOAD_BYTES:-102400}
RANK_BURST_PRESSURE_TIMEOUT_MS=${RANK_BURST_PRESSURE_TIMEOUT_MS:-5000}
RANK_ENDPOINT_OVERRIDE=${RANK_ENDPOINT_OVERRIDE:-}
RANK_DEPLOYMENT=${RANK_DEPLOYMENT:-}
if [[ -z "$RANK_DEPLOYMENT" ]]; then
  if [[ "$RANK_BURST_ENABLED" = 1 ]]; then
    RANK_DEPLOYMENT=deepfm-rank-burst-wrapper
  else
    RANK_DEPLOYMENT=deepfm-rank-brpc
  fi
fi
RANK_SERVICE=${RANK_SERVICE:-$RANK_DEPLOYMENT}
RANK_PORT=${RANK_PORT:-18211}
RANK_COMPLETION_TIMEOUT_SECONDS=${RANK_COMPLETION_TIMEOUT_SECONDS:-30}
LOG_SINCE_LOOKBACK_SECONDS=${LOG_SINCE_LOOKBACK_SECONDS:-60}

die() { echo "ERROR: $*" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
[[ "$KVC_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] && (( KVC_CONCURRENCY <= 256 )) \
  || die "KVC_CONCURRENCY must be between 1 and 256"
[[ "$WRAPPER_CONCURRENCY" =~ ^(1|1000)$ ]] || die "WRAPPER_CONCURRENCY must be 1 or 1000"
[[ "$BUSINESS_PAYLOAD_BYTES" =~ ^[0-9]+$ ]] && (( BUSINESS_PAYLOAD_BYTES <= 1048576 )) \
  || die "BUSINESS_PAYLOAD_BYTES must be in [0,1048576]"
[[ "$BRPC_PRESSURE_PAYLOAD_BYTES" =~ ^[1-9][0-9]*$ ]] \
  && (( BRPC_PRESSURE_PAYLOAD_BYTES <= 1048576 )) \
  || die "BRPC_PRESSURE_PAYLOAD_BYTES must be in [1,1048576]"
[[ "$BUSINESS_PAYLOAD_BYTES" = 102400 ]] \
  || die "one-shot BRPC matrix requires BUSINESS_PAYLOAD_BYTES=102400"
[[ "$KVC_PRESSURE_KEY_COUNT" =~ ^[0-9]+$ ]] || die "KVC_PRESSURE_KEY_COUNT must be non-negative"
(( KVC_PRESSURE_KEY_COUNT <= KVC_CONCURRENCY - 1 )) \
  || die "KVC_PRESSURE_KEY_COUNT must not exceed KVC pressure lanes"
[[ "$KVC_OBJECT_SIZE" =~ ^[1-9][0-9]*$ ]] || die "KVC_OBJECT_SIZE must be positive"
[[ "$KVC_PRESSURE_LEAD_US" =~ ^[0-9]+$ ]] && (( KVC_PRESSURE_LEAD_US <= 1000000 )) \
  || die "KVC_PRESSURE_LEAD_US must be between 0 and 1000000"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "WARMUP_REQUESTS must be non-negative"
[[ "$KVC_SUSTAINED_PRESSURE" = 0 || "$KVC_SUSTAINED_PRESSURE" = 1 ]] \
  || die "KVC_SUSTAINED_PRESSURE must be 0 or 1"
[[ "$KVC_SUSTAINED_MAX_DURATION_MS" =~ ^[1-9][0-9]*$ ]] \
  || die "KVC_SUSTAINED_MAX_DURATION_MS must be positive"
[[ "$KVC_SUSTAINED_MAX_LOOPS" =~ ^[1-9][0-9]*$ ]] \
  || die "KVC_SUSTAINED_MAX_LOOPS must be positive"
[[ "$KVC_INPROCESS_PRESSURE" = 0 || "$KVC_INPROCESS_PRESSURE" = 1 ]] \
  || die "KVC_INPROCESS_PRESSURE must be 0 or 1"
if [[ "$KVC_INPROCESS_PRESSURE" = 1 ]]; then
  (( KVC_CONCURRENCY == 32 )) || die "in-process pressure requires KVC_CONCURRENCY=32"
  (( KVC_PRESSURE_KEY_COUNT >= 1 && KVC_PRESSURE_KEY_COUNT <= 31 )) \
    || die "in-process pressure requires KVC_PRESSURE_KEY_COUNT between 1 and 31"
  (( KVC_OBJECT_SIZE == 3670016 )) || die "in-process pressure requires KVC_OBJECT_SIZE=3670016"
  KVC_SUSTAINED_PRESSURE=1
  (( KVC_SUSTAINED_MAX_DURATION_MS >= 2000 )) \
    || die "in-process sustained pressure requires at least 2000ms max duration"
  (( KVC_SUSTAINED_MAX_LOOPS >= 100 )) \
    || die "in-process sustained pressure requires at least 100 loops"
fi
KVC_BARRIER_TIMEOUT_MS=5
[[ "$KVC_INPROCESS_PRESSURE" = 0 ]] || KVC_BARRIER_TIMEOUT_MS=100
[[ "$PRIME_REQUESTS" =~ ^[1-9][0-9]*$ ]] \
  || die "PRIME_REQUESTS must be positive"
for flag in "$BUILD_PAIREC_IMAGE" "$IMPORT_PAIREC_IMAGE" "$RANK_BURST_ENABLED"; do
  [[ "$flag" = 0 || "$flag" = 1 ]] || die "boolean flags must be 0 or 1"
done
[[ "$RANK_BURST_CONCURRENCY" =~ ^(1|1000)$ ]] \
  || die "RANK_BURST_CONCURRENCY must be 1 or 1000"
[[ "$RANK_BURST_POOL_SIZE" =~ ^[1-9][0-9]*$ ]] \
  && (( RANK_BURST_POOL_SIZE >= RANK_BURST_CONCURRENCY && RANK_BURST_POOL_SIZE <= 1000 )) \
  || die "RANK_BURST_POOL_SIZE must be in [rank_concurrency,1000]"
[[ "$RANK_BURST_PAYLOAD_BYTES" = 102400 ]] \
  || die "combined Rank pressure requires RANK_BURST_PAYLOAD_BYTES=102400"
[[ "$RANK_BUSINESS_PAYLOAD_BYTES" = 102400 ]] \
  || die "combined Rank pressure requires RANK_BUSINESS_PAYLOAD_BYTES=102400"
[[ "$RANK_BURST_PRESSURE_TIMEOUT_MS" =~ ^[1-9][0-9]*$ ]] \
  || die "RANK_BURST_PRESSURE_TIMEOUT_MS must be positive"
[[ "$RANK_COMPLETION_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || die "RANK_COMPLETION_TIMEOUT_SECONDS must be positive"
[[ "$LOG_SINCE_LOOKBACK_SECONDS" =~ ^[0-9]+$ ]] \
  && (( LOG_SINCE_LOOKBACK_SECONDS <= 3600 )) \
  || die "LOG_SINCE_LOOKBACK_SECONDS must be in [0,3600]"
if [[ "$RANK_BURST_ENABLED" = 1 ]]; then
  [[ "$RANK_ENDPOINT_OVERRIDE" == *:* ]] \
    || die "Rank burst combination requires direct RANK_ENDPOINT_OVERRIDE=host:port"
fi

restore_kvc() {
  if [[ -f "$KVC_OVERLAY_BACKUP" ]]; then
    NAMESPACE="$NAMESPACE" DEPLOYMENT=inference-brpc-trtllm \
      BACKUP_FILE="$KVC_OVERLAY_BACKUP" \
      bash scripts/deploy_f14_kvc_burst_overlay.sh restore || true
  fi
}
trap restore_kvc EXIT INT TERM

echo "== Apply KVC c${KVC_CONCURRENCY} overlay =="
NAMESPACE="$NAMESPACE" DEPLOYMENT=inference-brpc-trtllm \
  BACKUP_FILE="$KVC_OVERLAY_BACKUP" CONCURRENCY="$KVC_CONCURRENCY" \
  PRESSURE_KEY_COUNT="$KVC_PRESSURE_KEY_COUNT" OBJECT_SIZE="$KVC_OBJECT_SIZE" \
  PRESSURE_LEAD_US="$KVC_PRESSURE_LEAD_US" \
  BARRIER_TIMEOUT_MS="$KVC_BARRIER_TIMEOUT_MS" \
  KVC_BURST_ENABLED=1 \
  KVC_BURST_VERBOSE=1 KVC_BURST_INITIAL_ARMED=0 \
  SUSTAINED_PRESSURE="$KVC_SUSTAINED_PRESSURE" \
  SUSTAINED_MAX_DURATION_MS="$KVC_SUSTAINED_MAX_DURATION_MS" \
  SUSTAINED_MAX_LOOPS="$KVC_SUSTAINED_MAX_LOOPS" \
  INPROCESS_PRESSURE="$KVC_INPROCESS_PRESSURE" \
  bash scripts/deploy_f14_kvc_burst_overlay.sh apply \
  | tee "$OUTPUT_DIR/kvc-overlay.log"

echo "== Deploy preconnected BRPC Wrapper full chain =="
set +e
NAMESPACE="$NAMESPACE" REQUESTS=1 WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  WRAPPER_ENDPOINT="$WRAPPER_ENDPOINT" \
  BURST_CONCURRENCY="$WRAPPER_CONCURRENCY" BURST_POOL_SIZE="$BURST_POOL_SIZE" \
  BURST_ACTIVE_CONNECTIONS="$BURST_ACTIVE_CONNECTIONS" \
  BURST_PAYLOAD_BYTES="$BRPC_PRESSURE_PAYLOAD_BYTES" \
  BUSINESS_PAYLOAD_BYTES="$BUSINESS_PAYLOAD_BYTES" \
  BUILD_PAIREC_IMAGE="$BUILD_PAIREC_IMAGE" IMPORT_PAIREC_IMAGE="$IMPORT_PAIREC_IMAGE" \
  RANK_DEPLOYMENT="$RANK_DEPLOYMENT" RANK_SERVICE="$RANK_SERVICE" RANK_PORT="$RANK_PORT" \
  RANK_ENDPOINT_OVERRIDE="$RANK_ENDPOINT_OVERRIDE" \
  RANK_BURST_ENABLED="$RANK_BURST_ENABLED" \
  RANK_BURST_CONCURRENCY="$RANK_BURST_CONCURRENCY" \
  RANK_BURST_POOL_SIZE="$RANK_BURST_POOL_SIZE" \
  RANK_BURST_PAYLOAD_BYTES="$RANK_BURST_PAYLOAD_BYTES" \
  RANK_BUSINESS_PAYLOAD_BYTES="$RANK_BUSINESS_PAYLOAD_BYTES" \
  RANK_BURST_PRESSURE_TIMEOUT_MS="$RANK_BURST_PRESSURE_TIMEOUT_MS" \
  LOG_SINCE_LOOKBACK_SECONDS="$LOG_SINCE_LOOKBACK_SECONDS" \
  OUTPUT_DIR="$WRAPPER_OUTPUT_DIR" \
  bash scripts/deploy_and_validate_pairec_brpc_wrapper_full.sh \
  | tee "$OUTPUT_DIR/wrapper-console.log"
wrapper_code=${PIPESTATUS[0]}
set -e
[[ "$wrapper_code" -eq 0 ]] || die "BRPC Wrapper full-chain validation failed: exit=$wrapper_code"

if [[ "$RANK_BURST_ENABLED" = 1 ]]; then
  RANK_POD="$(kubectl -n "$NAMESPACE" get pod -l "app=$RANK_DEPLOYMENT" \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
  [[ -n "$RANK_POD" ]] || die "Rank Wrapper Pod not found for app=$RANK_DEPLOYMENT"
  kubectl -n "$NAMESPACE" get pod "$RANK_POD" \
    -o jsonpath='{.spec.containers[*].name}' \
    | tr ' ' '\n' | grep -Fxq rank-burst-wrapper \
    || die "app=$RANK_DEPLOYMENT pod=$RANK_POD does not contain rank-burst-wrapper"
  echo "PAIREC_COMBINED_RANK_WRAPPER_READY deployment=$RANK_DEPLOYMENT pod=$RANK_POD"
fi

echo "== Run KVC contention through the deployed BRPC Wrapper =="
LOG_SINCE_AT="$(date --date="${LOG_SINCE_LOOKBACK_SECONDS} seconds ago" --iso-8601=seconds)"
echo "log_since_at=$LOG_SINCE_AT lookback_seconds=$LOG_SINCE_LOOKBACK_SECONDS"
set +e
NAMESPACE="$NAMESPACE" REPEATS="$REQUESTS" MODE=baseline \
  KVC_BURST_CONTAINER=kvc-burst-wrapper KVC_BURST_REQUIRE_COMPLETE=1 \
  KVC_BURST_PRESTART_PRESSURE=0 \
  KVC_BURST_DYNAMIC_ARM=1 KVC_BURST_PRESSURE_KEY_COUNT="$KVC_PRESSURE_KEY_COUNT" \
  EXPECTED_OFFLOADS=3 EXPECTED_ONBOARDS=2 STRICT_COUNTS=1 \
  EXPECTED_ONBOARDS_MIN="$EXPECTED_ONBOARDS_MIN" \
  EXPECTED_ONBOARDS_MAX="$EXPECTED_ONBOARDS_MAX" \
  REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 PRIME_REQUESTS="$PRIME_REQUESTS" \
  REQUIRE_FULL_CHAIN_BRPC_ATTRIBUTION=1 \
  RESET_INFERENCE_BEFORE_ROUND=1 RESET_INFERENCE_MODE=pod-recreate \
  MIN_ROOT_AVAILABLE_KB="$MIN_ROOT_AVAILABLE_KB" \
  PAIREC_TARGET=deploy/pairec-brpc-observed-wrapper \
  BRPC_TARGET=deployment/inference-brpc-trtllm \
  OUT_DIR="$CONTENTION_OUTPUT_DIR" \
  bash scripts/benchmark_brpc_kvc_contention.sh \
  | tee "$OUTPUT_DIR/contention-console.log"
contention_code=${PIPESTATUS[0]}
set -e
[[ "$contention_code" -eq 0 ]] || die "KVC contention through Wrapper failed: exit=$contention_code"

PAIREC_RANK_MEASURED_LOG="$OUTPUT_DIR/pairec-rank-measured.log"
RANK_WRAPPER_MEASURED_LOG="$OUTPUT_DIR/rank-wrapper-measured.log"
: >"$PAIREC_RANK_MEASURED_LOG"
: >"$RANK_WRAPPER_MEASURED_LOG"
if [[ "$RANK_BURST_ENABLED" = 1 ]]; then
  echo "== Wait for measured Rank burst completion =="
  PAIREC_POD="$(kubectl -n "$NAMESPACE" get pod -l app=pairec-brpc-observed-wrapper \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
  RANK_POD="$(kubectl -n "$NAMESPACE" get pod -l "app=$RANK_DEPLOYMENT" \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
  [[ -n "$PAIREC_POD" && -n "$RANK_POD" ]] \
    || die "PaiRec or Rank Wrapper Pod not found"
  mapfile -t RANK_REQUEST_IDS < <(python3 - "$CONTENTION_OUTPUT_DIR/result.json" <<'PY'
import json,pathlib,sys
result=json.load(open(sys.argv[1]))
for row in result.get("rows",[]):
    replay=pathlib.Path(row["summary_path"]).parent
    print(json.load(open(replay / "summary.json"))["request_id"])
PY
  )
  [[ "${#RANK_REQUEST_IDS[@]}" -eq "$REQUESTS" ]] \
    || die "expected $REQUESTS measured Rank request ids, got ${#RANK_REQUEST_IDS[@]}"
  rank_deadline=$((SECONDS + RANK_COMPLETION_TIMEOUT_SECONDS))
  while true; do
    kubectl -n "$NAMESPACE" logs "$PAIREC_POD" -c pairec \
      --since-time="$LOG_SINCE_AT" --timestamps >"$PAIREC_RANK_MEASURED_LOG"
    if python3 - "$PAIREC_RANK_MEASURED_LOG" "${RANK_REQUEST_IDS[@]}" <<'PY'
import json,pathlib,sys
expected=set(sys.argv[2:]); found=set()
for line in pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines():
    pos=line.find("{")
    if pos < 0: continue
    try: event=json.loads(line[pos:])
    except json.JSONDecodeError: continue
    if event.get("event")=="pairec_rank_brpc_burst_complete":
        found.add(event.get("request_id"))
raise SystemExit(0 if expected <= found else 1)
PY
    then
      break
    fi
    (( SECONDS < rank_deadline )) \
      || die "timed out waiting for measured Rank pressure completion"
    sleep 0.2
  done
  kubectl -n "$NAMESPACE" logs "$RANK_POD" -c rank-burst-wrapper \
    --since-time="$LOG_SINCE_AT" --timestamps >"$RANK_WRAPPER_MEASURED_LOG"
  echo "PAIREC_COMBINED_RANK_BURST_DRAINED requests=${#RANK_REQUEST_IDS[@]} concurrency=$RANK_BURST_CONCURRENCY"
fi

WRAPPER_POD="$(kubectl -n "$NAMESPACE" get pod -l app=brpc-burst-wrapper \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
[[ -n "$WRAPPER_POD" ]] || die "BRPC Wrapper Pod not found"
kubectl -n "$NAMESPACE" logs "$WRAPPER_POD" -c brpc-burst-wrapper \
  --since-time="$LOG_SINCE_AT" >"$OUTPUT_DIR/wrapper-measured.log" 2>&1 || true

python3 - "$CONTENTION_OUTPUT_DIR/result.json" "$OUTPUT_DIR/wrapper-measured.log" \
  "$OUTPUT_DIR/summary.json" "$WRAPPER_CONCURRENCY" "$KVC_CONCURRENCY" \
  "$EXPECTED_ONBOARDS_MIN" "$EXPECTED_ONBOARDS_MAX" "$KVC_OBJECT_SIZE" \
  "$KVC_PRESSURE_KEY_COUNT" "$KVC_SUSTAINED_PRESSURE" "$KVC_INPROCESS_PRESSURE" \
  "$BUSINESS_PAYLOAD_BYTES" "$BRPC_PRESSURE_PAYLOAD_BYTES" \
  "$PAIREC_RANK_MEASURED_LOG" "$RANK_WRAPPER_MEASURED_LOG" \
  "$RANK_BURST_ENABLED" "$RANK_BURST_CONCURRENCY" \
  "$RANK_BUSINESS_PAYLOAD_BYTES" "$RANK_BURST_PAYLOAD_BYTES" \
  "$WRAPPER_OUTPUT_DIR/rank-endpoint.txt" "$BURST_POOL_SIZE" <<'PY'
import json, math, pathlib, statistics, sys
contention = json.load(open(sys.argv[1]))
wrapper_log = pathlib.Path(sys.argv[2]).read_text(errors="replace")
wrapper_concurrency = int(sys.argv[4])
kvc_concurrency = int(sys.argv[5])
expected_onboards_min = int(sys.argv[6])
expected_onboards_max = int(sys.argv[7])
kvc_object_size = int(sys.argv[8])
kvc_pressure_key_count = int(sys.argv[9])
kvc_sustained_pressure = sys.argv[10] == "1"
kvc_inprocess_pressure = sys.argv[11] == "1"
business_payload_bytes = int(sys.argv[12])
brpc_pressure_payload_bytes = int(sys.argv[13])
rank_pairec_log_path = pathlib.Path(sys.argv[14])
rank_wrapper_log = pathlib.Path(sys.argv[15]).read_text(errors="replace")
rank_burst_enabled = sys.argv[16] == "1"
rank_burst_concurrency = int(sys.argv[17])
rank_business_payload_bytes = int(sys.argv[18])
rank_pressure_payload_bytes = int(sys.argv[19])
brpc_burst_pool_size = int(sys.argv[21])
rank_endpoint = {}
for line in pathlib.Path(sys.argv[20]).read_text(errors="replace").splitlines():
    if "=" in line:
        key,value=line.split("=",1); rank_endpoint[key]=value
if rank_burst_enabled:
    assert rank_endpoint.get("rank_endpoint"), rank_endpoint
    assert rank_endpoint.get("rank_endpoint_source") == "override", rank_endpoint
valid = contention.get("valid_repeats") == contention.get("expected_repeats")
rows = []

def json_events(path):
    events = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        pos = line.find("{")
        if pos < 0:
            continue
        try:
            events.append(json.loads(line[pos:]))
        except json.JSONDecodeError:
            pass
    return events

def one(events, name, request_id):
    values = [event for event in events
              if event.get("event") == name and event.get("request_id") == request_id]
    assert len(values) == 1, (request_id, name, values)
    return values[0]

def log_value(line, key):
    prefix = key + "="
    for token in line.split():
        if token.startswith(prefix):
            return token[len(prefix):]
    raise AssertionError((key, line))

def optional_log_value(line, key, default):
    try:
        return log_value(line, key)
    except AssertionError:
        return default

def overlap_ms(left_start, left_end, right_start, right_end):
    return max(0, min(left_end, right_end) - max(left_start, right_start)) / 1e6

rank_pairec_events = json_events(rank_pairec_log_path) if rank_pairec_log_path.is_file() else []

for row in contention.get("rows", []):
    assert row.get("brpc_attribution_complete") is True, row
    replay = pathlib.Path(row["summary_path"]).parent
    trace = json.load(open(replay / "summary.json"))
    request_id = trace["request_id"]
    pairec_events = json_events(replay / "pairec_stdout.log")
    pipeline = one(pairec_events, "pipeline_trace_complete", request_id)
    wrapper_burst_start = one(
        pairec_events, "pairec_brpc_burst_start", request_id)
    business = one(pairec_events, "pairec_brpc_burst_business_complete", request_id)
    wrapper_burst = one(pairec_events, "pairec_brpc_burst_complete", request_id)
    assert pipeline["status"] == "ok" and pipeline["valid"] is True, pipeline
    spans = {span["name"]: span for span in pipeline["spans"]}
    for name in ("vector_recall", "generative_recall", "deepfm_rank", "rerank"):
        assert spans[name]["status"] == "ok", (name, spans[name])
    for name in ("vector_recall", "generative_recall", "deepfm_rank"):
        assert spans[name]["protocol"] == "brpc", (name, spans[name])
    pressure_requests = max(wrapper_concurrency - 1, 0)
    assert business["concurrency"] == wrapper_concurrency, business
    assert business["business_success"], business
    assert wrapper_burst["concurrency"] == wrapper_concurrency, wrapper_burst
    assert wrapper_burst["burst_valid"], wrapper_burst
    assert wrapper_burst["pressure_requests"] == pressure_requests, wrapper_burst
    assert wrapper_burst["pressure_success"] == pressure_requests, wrapper_burst
    assert wrapper_burst["pressure_errors"] == 0, wrapper_burst
    assert (wrapper_burst_start["pressure_payload_bytes"]
            == brpc_pressure_payload_bytes), wrapper_burst_start
    kvc_events = [event for event in trace["kvc_proxy_events"]
                  if event.get("event") == "kvc_burst_complete"
                  and event.get("request_id") == request_id]
    assert len(kvc_events) == 1, (request_id, kvc_events)
    kvc = kvc_events[0]
    assert kvc["concurrency"] == kvc_concurrency and kvc["valid"] is True, kvc
    assert kvc["object_size_bytes"] == kvc_object_size, kvc
    assert kvc["pressure_key_count"] == kvc_pressure_key_count, kvc
    pressure_lanes = max(kvc_concurrency - 1, 0)
    assert kvc["pressure_lanes"] == pressure_lanes, kvc
    assert kvc["pressure_success"] == pressure_lanes, kvc
    assert kvc["pressure_errors"] == 0, kvc
    expected_engine = "inprocess-shared-client" if kvc_inprocess_pressure else "sidecar-exclusive-clients"
    assert kvc.get("pressure_engine") == expected_engine, kvc
    if kvc_inprocess_pressure:
        assert kvc.get("shared_client_errors") == 0, kvc
    required_pressure_first = math.ceil(pressure_lanes * .95)
    assert kvc["pressure_started_before_business"] >= required_pressure_first, kvc
    assert kvc["pressure_inflight_at_business_start"] >= required_pressure_first, kvc
    assert kvc["business_submit_rank"] >= required_pressure_first + 1, kvc
    if kvc_inprocess_pressure:
        assert kvc["business_submit_rank"] == 32, kvc
        assert kvc["pressure_inflight_at_business_start"] >= required_pressure_first, kvc
        assert kvc["pressure_completed_before_business"] == 0, kvc
        assert kvc["business_get_count"] == 2, kvc
        assert kvc["business_get_success_count"] == 2, kvc
        assert kvc["business_get_1_ms"] > 0, kvc
        assert kvc["business_get_2_ms"] > 0, kvc
        assert kvc["pressure_active_at_second_get_start"] >= required_pressure_first, kvc
        assert kvc["pressure_active_at_stop"] >= required_pressure_first, kvc
        assert kvc["pressure_completed_at_stop"] >= kvc["pressure_completed_at_second_get_start"], kvc
        assert kvc["pressure_completions_after_stop"] > 0, kvc
        assert kvc["pressure_tail_after_stop_ms"] > 0, kvc
        assert kvc["pressure_stop_epoch_ns"] >= kvc["business_get_2_end_epoch_ns"], kvc
        assert kvc["pressure_last_end_epoch_ns"] >= kvc["pressure_stop_epoch_ns"], kvc
        assert kvc["business_lifecycle_done"] is True, kvc
        assert kvc["business_lifecycle_done_epoch_ns"] >= kvc["business_get_2_end_epoch_ns"], kvc
    assert kvc.get("sustained_enabled", False) is kvc_sustained_pressure, kvc
    if kvc_sustained_pressure and pressure_lanes > 0:
        assert kvc["sustained_errors"] == 0, kvc
    exact = trace["datasystem_request_complete"]
    executor = trace["trt_executor_request_completions"]
    assert exact["set_count"] == 3, exact
    assert expected_onboards_min <= exact["get_count"] <= expected_onboards_max, exact
    assert len(executor) == 1, executor
    if kvc_inprocess_pressure:
        assert exact["get_count"] == 2, exact
        assert kvc["add_token_observation_count"] == exact["add_token_count"], (kvc, exact)
    matches = [line for line in wrapper_log.splitlines()
               if "method=Recommend" in line and f"request_id={request_id}" in line]
    payload_tokens = (
        f" front_payload_bytes={business_payload_bytes} ",
        " backend_payload_bytes=0 ",
    )
    if (len(matches) != 1 or " code=200 " not in matches[0]
            or not all(token in f" {matches[0]} " for token in payload_tokens)):
        valid = False
    wrapper_active_health_at_start = -1
    wrapper_max_active_health = -1
    wrapper_health_calls_at_start = -1
    wrapper_health_calls_during_recommend = -1
    wrapper_health_payload_bytes_during_recommend = -1
    wrapper_health_calls_during_backend = -1
    wrapper_health_payload_bytes_during_backend = -1
    wrapper_pressure_drain_required = -1
    wrapper_pressure_drain_success = -1
    wrapper_pressure_drain_ms = 0.0
    wrapper_pressure_drain_quiet_ms = 0.0
    brpc_pressure_stop_signal_delay_ms = 0.0
    if len(matches) == 1:
        wrapper_active_health_at_start = int(
            optional_log_value(matches[0], "active_health_at_start", 0))
        wrapper_max_active_health = int(
            optional_log_value(matches[0], "max_active_health", 0))
        wrapper_health_calls_at_start = int(
            optional_log_value(matches[0], "health_calls_at_start", 0))
        wrapper_health_calls_during_recommend = int(
            optional_log_value(matches[0], "health_calls_during_recommend", 0))
        wrapper_health_payload_bytes_during_recommend = int(
            optional_log_value(
                matches[0], "health_payload_bytes_during_recommend", 0))
        wrapper_health_calls_during_backend = int(
            optional_log_value(matches[0], "health_calls_during_backend", 0))
        wrapper_health_payload_bytes_during_backend = int(
            optional_log_value(matches[0], "health_payload_bytes_during_backend", 0))
        wrapper_pressure_drain_required = int(
            optional_log_value(matches[0], "pressure_drain_required", 0))
        wrapper_pressure_drain_success = int(
            optional_log_value(matches[0], "pressure_drain_success", 1))
        wrapper_pressure_drain_ms = float(
            optional_log_value(matches[0], "pressure_drain_ms", 0.0))
        wrapper_pressure_drain_quiet_ms = float(
            optional_log_value(matches[0], "pressure_drain_quiet_ms", 0.0))
        # One-shot pressure uses the original zero-filled Health payload and
        # therefore must never enter the coordinated external drain path.
        assert wrapper_pressure_drain_required == 0, matches[0]
        assert wrapper_pressure_drain_ms == 0.0, matches[0]
    rank_metrics = {
        "rank_business_client_ms": 0.0,
        "rank_front_brpc_ms": 0.0,
        "rank_service_ms": 0.0,
        "rank_wrapper_backend_rpc_ms": 0.0,
        "rank_pressure_p95_ms": 0.0,
        "rank_pressure_max_active": 0.0,
        "rank_pressure_start_skew_us": 0.0,
        "rank_pressure_tail_after_business_ms": 0.0,
        "rank_pressure_rerank_overlap_ms": 0.0,
        "rank_pressure_pipeline_overlap_ms": 0.0,
        "rank_inference_to_business_gap_ms": 0.0,
        "rank_wrapper_health_calls_during_rank": 0.0,
        "rank_wrapper_health_payload_bytes_during_rank": 0.0,
        "rank_pressure_requests": 0.0,
        "rank_pressure_success": 0.0,
        "rank_pressure_errors": 0.0,
    }
    if rank_burst_enabled:
        rank_own = [event for event in rank_pairec_events
                    if event.get("request_id") == request_id]
        def rank_one(name):
            values = [event for event in rank_own if event.get("event") == name]
            assert len(values) == 1, (request_id, name, values)
            return values[0]
        rank_start_event = rank_one("pairec_rank_brpc_burst_start")
        rank_business = rank_one("pairec_rank_brpc_burst_business_complete")
        rank_complete = rank_one("pairec_rank_brpc_burst_complete")
        rank_service = rank_one("deepfm_rank_complete")
        rerank_event = rank_one("source_quota_rerank_complete")
        assert rank_start_event["concurrency"] == rank_burst_concurrency, rank_start_event
        assert rank_start_event["armed_workers"] == rank_burst_concurrency, rank_start_event
        assert rank_start_event["business_payload_bytes"] == rank_business_payload_bytes, rank_start_event
        assert rank_start_event["pressure_payload_bytes"] == rank_pressure_payload_bytes, rank_start_event
        assert rank_business["business_success"] and rank_business["trace_valid"], rank_business
        assert rank_business["business_payload_bytes"] == rank_business_payload_bytes, rank_business
        expected_rank_pressure = rank_burst_concurrency - 1
        assert rank_complete["pressure_requests"] == expected_rank_pressure, rank_complete
        assert rank_complete["pressure_success"] == expected_rank_pressure, rank_complete
        assert rank_complete["pressure_errors"] == 0, rank_complete
        assert rank_complete["burst_valid"] and rank_complete["trace_valid"], rank_complete
        assert rank_complete["connected_sessions"] == rank_burst_concurrency, rank_complete
        if rank_burst_concurrency > 1:
            assert rank_complete["pressure_overlap_business"] > 0, rank_complete
        assert rank_service["candidate_count"] == 50, rank_service
        assert rank_service["reordered"] is True, rank_service
        rank_wrapper_matches = [line for line in rank_wrapper_log.splitlines()
                                if "[brpc-rank-burst-wrapper] method=Rank" in line
                                and f"request_id={request_id}" in line]
        assert len(rank_wrapper_matches) == 1, (request_id, rank_wrapper_matches)
        rank_wrapper = rank_wrapper_matches[0]
        assert int(log_value(rank_wrapper, "code")) == 200, rank_wrapper
        assert int(log_value(rank_wrapper, "front_payload_bytes")) == rank_business_payload_bytes, rank_wrapper
        assert int(log_value(rank_wrapper, "backend_payload_bytes")) == 0, rank_wrapper
        rank_business_start = int(rank_business["rank_business_start_epoch_ns"])
        rank_business_end = int(rank_business["rank_business_end_epoch_ns"])
        pressure_start = int(rank_complete.get("rank_pressure_first_start_epoch_ns") or rank_business_start)
        pressure_end = int(rank_complete.get("rank_pressure_last_end_epoch_ns") or rank_business_end)
        generative = spans["generative_recall"]
        inference_end = int(pipeline["start_epoch_ns"]) + 1000 * (
            int(generative["start_offset_us"]) + int(generative["duration_us"]))
        assert inference_end <= rank_business_start, (generative, rank_business)
        rank_metrics = {
            "rank_business_client_ms": float(rank_business["business_client_wall_ms"]),
            "rank_front_brpc_ms": float(rank_business["front_brpc_estimate_ms"]),
            "rank_service_ms": float(rank_business["service_total_ms"]),
            "rank_wrapper_backend_rpc_ms": float(log_value(rank_wrapper, "backend_rpc_ms")),
            "rank_pressure_p95_ms": float(rank_complete["pressure_latency_p95_ms"]),
            "rank_pressure_max_active": float(rank_complete["max_active_workers"]),
            "rank_pressure_start_skew_us": float(rank_complete["start_skew_us"]),
            "rank_pressure_tail_after_business_ms": float(rank_complete["pressure_tail_after_business_ms"]),
            "rank_pressure_rerank_overlap_ms": overlap_ms(
                pressure_start, pressure_end, int(rerank_event["start_epoch_ns"]),
                int(rerank_event["end_epoch_ns"])),
            "rank_pressure_pipeline_overlap_ms": overlap_ms(
                pressure_start, pressure_end, rank_business_end,
                int(pipeline["end_epoch_ns"])),
            "rank_inference_to_business_gap_ms": (
                rank_business_start - inference_end) / 1e6,
            "rank_wrapper_health_calls_during_rank": float(
                log_value(rank_wrapper, "health_calls_during_rank")),
            "rank_wrapper_health_payload_bytes_during_rank": float(
                log_value(rank_wrapper, "health_payload_bytes_during_rank")),
            "rank_pressure_requests": float(rank_complete["pressure_requests"]),
            "rank_pressure_success": float(rank_complete["pressure_success"]),
            "rank_pressure_errors": float(rank_complete["pressure_errors"]),
        }
        if rank_burst_concurrency > 1:
            assert rank_metrics["rank_wrapper_health_calls_during_rank"] > 0, rank_wrapper
            assert rank_metrics["rank_wrapper_health_payload_bytes_during_rank"] > 0, rank_wrapper
    coordination_ms = float(kvc["coordination_wait_ms"])
    client_e2e_actual_ms = float(trace["client"]["client_e2e_ms"])
    runner_actual_ms = executor[0]["runner_us"] / 1000.0
    rows.append({
        "request_id": request_id,
        "client_e2e_ms": client_e2e_actual_ms,
        "client_e2e_adjusted_ms": max(0.0, client_e2e_actual_ms - coordination_ms),
        "pairec_total_ms": pipeline["pairec_total_us"] / 1000.0,
        "pairec_total_adjusted_ms": pipeline["pairec_total_us"] / 1000.0,
        "vector_recall_ms": spans["vector_recall"]["duration_us"] / 1000.0,
        "generative_recall_ms": spans["generative_recall"]["duration_us"] / 1000.0,
        "deepfm_rank_ms": spans["deepfm_rank"]["duration_us"] / 1000.0,
        "rerank_ms": spans["rerank"]["duration_us"] / 1000.0,
        "front_brpc_ms": business["business_front_brpc_ms"],
        "wrapper_total_ms": business["wrapper_total_ms"],
        "wrapper_total_adjusted_ms": business["wrapper_total_ms"],
        "backend_brpc_ms": business["wrapper_backend_brpc_ms"],
        "runner_ms": runner_actual_ms,
        "runner_adjusted_ms": max(0.0, runner_actual_ms - coordination_ms),
        "brpc_pressure_ready_wait_ms": float(row.get("brpc_pressure_ready_wait_ms", 0.0)),
        "brpc_pressure_qps": float(row.get("brpc_pressure_qps", 0.0)),
        "brpc_pressure_gbps": float(row.get("brpc_pressure_gbps", 0.0)),
        "brpc_pressure_max_active": float(row.get("brpc_pressure_max_active", 0)),
        "brpc_pressure_first_round_completed": float(
            row.get("brpc_pressure_first_round_completed", 0)),
        "brpc_pressure_second_round_started": float(
            row.get("brpc_pressure_second_round_started", 0)),
        "brpc_pressure_active_at_ready": float(
            row.get("brpc_pressure_active_at_ready", 0)),
        "wrapper_external_active_health_at_start": float(
            wrapper_active_health_at_start),
        "wrapper_external_max_active_health": float(wrapper_max_active_health),
        "wrapper_external_health_calls_at_start": float(
            wrapper_health_calls_at_start),
        "wrapper_external_health_calls_during_recommend": float(
            wrapper_health_calls_during_recommend),
        "wrapper_external_health_payload_bytes_during_recommend": float(
            wrapper_health_payload_bytes_during_recommend),
        "wrapper_external_health_calls_during_backend": float(
            wrapper_health_calls_during_backend),
        "wrapper_external_health_payload_bytes_during_backend": float(
            wrapper_health_payload_bytes_during_backend),
        "brpc_pressure_drain_ms": wrapper_pressure_drain_ms,
        "brpc_pressure_drain_quiet_ms": wrapper_pressure_drain_quiet_ms,
        "brpc_pressure_stop_signal_delay_ms": brpc_pressure_stop_signal_delay_ms,
        "kvc_business_get_ms": kvc["business_get_ms"],
        "kvc_business_get_1_ms": float(kvc.get("business_get_1_ms", 0.0)),
        "kvc_business_get_2_ms": float(kvc.get("business_get_2_ms", 0.0)),
        "kvc_pressure_p99_ms": kvc["pressure_get_p99_ms"],
        "kvc_barrier_ms": kvc["barrier_wait_ms"],
        "kvc_pressure_first_wait_ms": kvc["pressure_first_wait_ms"],
        "kvc_pressure_lead_wait_ms": kvc["pressure_lead_wait_ms"],
        "kvc_coordination_wait_ms": kvc["coordination_wait_ms"],
        "kvc_pressure_tail_after_stop_ms": float(
            kvc.get("pressure_tail_after_stop_ms", 0.0)),
        "kvc_pressure_stop_signal_delay_ms": float(
            kvc.get("pressure_stop_signal_delay_us", 0.0)) / 1000.0,
        "kvc_pressure_stop_to_first_add_token_ms": float(
            kvc.get("pressure_stop_to_first_add_token_ms", 0.0)),
        "kvc_add_token_window_ms": float(kvc.get("add_token_window_ms", 0.0)),
        "kvc_pressure_add_token_overlap_ms": float(
            kvc.get("pressure_add_token_overlap_ms", 0.0)),
        "kvc_pressure_active_at_stop": float(kvc.get("pressure_active_at_stop", 0)),
        "kvc_pressure_active_at_second_get_start": float(
            kvc.get("pressure_active_at_second_get_start", 0)),
        "kvc_pressure_completed_at_second_get_start": float(
            kvc.get("pressure_completed_at_second_get_start", 0)),
        "kvc_pressure_completed_at_stop": float(
            kvc.get("pressure_completed_at_stop", 0)),
        "kvc_pressure_completed_gets": float(kvc.get("pressure_completed_gets", 0)),
        "kvc_pressure_completions_after_stop": float(
            kvc.get("pressure_completions_after_stop", 0)),
        "kvc_pressure_active_at_first_add_token": float(
            kvc.get("pressure_active_at_first_add_token", 0)),
        "kvc_pressure_active_at_last_add_token": float(
            kvc.get("pressure_active_at_last_add_token", 0)),
        "kvc_add_token_observation_count": float(
            kvc.get("add_token_observation_count", 0)),
        "kvc_business_submit_rank": float(kvc.get("business_submit_rank", 0)),
        "kvc_pressure_inflight_at_business_start": float(
            kvc.get("pressure_inflight_at_business_start", 0)),
        "kvc_sustained_loop_gets": float(kvc.get("sustained_loop_gets", 0)),
        "kvc_sustained_window_ms": float(kvc.get("sustained_window_ms", 0.0)),
        "datasystem_get_ms": exact["get_us"] / 1000.0,
        "datasystem_set_ms": exact["set_us"] / 1000.0,
        "native_add_token_ms": exact["add_token_us"] / 1000.0,
        "native_kv_update_ms": exact["kv_update_us"] / 1000.0,
        "native_executor_queue_ms": exact["executor_queue_us"] / 1000.0,
        "native_add_sequence_ms": exact["add_sequence_us"] / 1000.0,
        "native_prefill_gap_ms": exact["prefill_gap_us"] / 1000.0,
        "native_decode_gap_ms": exact["decode_gap_us"] / 1000.0,
        "native_finalization_gap_ms": exact["finalization_gap_us"] / 1000.0,
        "native_remove_sequence_ms": exact["remove_sequence_us"] / 1000.0,
        "native_lifecycle_ms": exact["native_lifecycle_us"] / 1000.0,
        "native_accounted_ms": exact["native_accounted_us"] / 1000.0,
        "wrapper_pressure_p95_ms": wrapper_burst["pressure_latency_p95_ms"],
        "wrapper_max_active": wrapper_burst["max_active_workers"],
        "wrapper_log_matches": len(matches),
        **rank_metrics,
    })

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (pos - low)

metric_names = [name for name in rows[0] if name not in {"request_id", "wrapper_log_matches"}]
metrics = {}
for name in metric_names:
    values = [float(row[name]) for row in rows]
    metrics[name] = {"avg": statistics.fmean(values), "p50": percentile(values, .5),
                     "p95": percentile(values, .95), "p99": percentile(values, .99),
                     "max": max(values)}
ds_worker_samples = []
ds_worker_threadpool = []
nic_burst_samples = []
for row in contention.get("rows", []):
    round_dir = pathlib.Path(row["summary_path"]).parent.parent
    metrics_path = round_dir / "datasystem-worker-metrics.json"
    error_path = round_dir / "datasystem-worker-metrics.error"
    if metrics_path.is_file():
        ds_worker_samples.append(json.loads(metrics_path.read_text()))
    elif error_path.is_file():
        ds_worker_samples.append({"error": error_path.read_text().strip()})
    tp_path = round_dir / "worker-threadpool.after.json"
    tp_error = round_dir / "worker-threadpool.error"
    if tp_path.is_file():
        ds_worker_threadpool.append(json.loads(tp_path.read_text()).get("peak", {}))
    elif tp_error.is_file():
        ds_worker_threadpool.append({"error": tp_error.read_text().strip()})
    nic_path = pathlib.Path(row["summary_path"]).parent / "nic-burst.json"
    nic_error = pathlib.Path(row["summary_path"]).parent / "nic-burst.error"
    if nic_path.is_file():
        nic_burst_samples.append(json.loads(nic_path.read_text()))
    elif nic_error.is_file():
        nic_burst_samples.append({"error": nic_error.read_text().strip()})
result = {
    "classification": (f"PAIREC_BRPC_WRAPPER_C{wrapper_concurrency}_KVC_C{kvc_concurrency}_OK"
                       if valid else "PAIREC_BRPC_WRAPPER_KVC_COMBINED_FAIL"),
    "wrapper_concurrency": wrapper_concurrency,
    "business_payload_bytes": business_payload_bytes,
    "brpc_pressure_payload_bytes": brpc_pressure_payload_bytes,
    "brpc_pressure_model": "synchronized_one_shot_burst",
    "embedded_burst_concurrency": wrapper_concurrency,
    "brpc_burst_pool_size": brpc_burst_pool_size,
    "brpc_pressure_enabled": wrapper_concurrency > 1,
    "brpc_pressure_concurrency": max(wrapper_concurrency - 1, 0),
    "brpc_pressure_ready_min_active": 0,
    "kvc_concurrency": kvc_concurrency,
    "kvc_object_size_bytes": kvc_object_size,
    "kvc_pressure_key_count": kvc_pressure_key_count,
    "kvc_sustained_pressure": kvc_sustained_pressure,
    "rank_burst_enabled": rank_burst_enabled,
    "rank_burst_concurrency": rank_burst_concurrency if rank_burst_enabled else 0,
    "rank_business_payload_bytes": rank_business_payload_bytes if rank_burst_enabled else 0,
    "rank_pressure_payload_bytes": rank_pressure_payload_bytes if rank_burst_enabled else 0,
    "rank_endpoint": rank_endpoint.get("rank_endpoint", ""),
    "rank_endpoint_source": rank_endpoint.get("rank_endpoint_source", ""),
    "expected_onboards_min": expected_onboards_min,
    "expected_onboards_max": expected_onboards_max,
    "contention": contention,
    "samples": rows,
    "metrics": metrics,
    "datasystem_worker_samples": ds_worker_samples,
    "datasystem_worker_threadpool": ds_worker_threadpool,
    "nic_burst_samples": nic_burst_samples,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
print(f"classification={result['classification']}")
print(f"requests={len(rows)}")
print("metric avg_ms p50_ms p95_ms p99_ms max_ms")
for name in metric_names:
    item = metrics[name]
    print(f"{name} {item['avg']:.3f} {item['p50']:.3f} {item['p95']:.3f} {item['p99']:.3f} {item['max']:.3f}")
print(f"summary_json={sys.argv[3]}")
if not valid:
    raise SystemExit(1)
PY

echo "output_dir=$OUTPUT_DIR"
