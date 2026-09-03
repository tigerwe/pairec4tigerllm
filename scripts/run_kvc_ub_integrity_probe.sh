#!/usr/bin/env bash
set -euo pipefail

KVC_UB_PROBE_BIN=${KVC_UB_PROBE_BIN:-/opt/pairec-kvc-burst/bin/kvc_ub_integrity_probe}
WORKER_HOST=${WORKER_HOST:-141.62.33.105}
WORKER_PORT=${WORKER_PORT:-32501}
OBJECT_SIZE=${OBJECT_SIZE:-3670016}
ITERATIONS=${ITERATIONS:-100}
CONCURRENCY=${CONCURRENCY:-1}
DS_UB_DEV_NAME=${DS_UB_DEV_NAME:-bonding_dev_1}
DS_UB_DEV_EID=${DS_UB_DEV_EID:-0}
LOG_DIR=${LOG_DIR:-/home/zcx/kvc-ub-poc/integrity-client-log}
OUTPUT_DIR=${OUTPUT_DIR:-/home/zcx/kvc-ub-poc/integrity-evidence}
PREFIX=${PREFIX:-PairecKvcUbIntegrity_$(date +%Y%m%d_%H%M%S)_$$}
START_FILE=${START_FILE:-}
ACCESS_LOG_TIMEOUT_SECONDS=${ACCESS_LOG_TIMEOUT_SECONDS:-30}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "$KVC_UB_PROBE_BIN" ]] || die "probe binary is missing: $KVC_UB_PROBE_BIN"
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || die "ITERATIONS must be a positive integer"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] && (( CONCURRENCY <= 256 )) \
  || die "CONCURRENCY must be between 1 and 256"
[[ "$OBJECT_SIZE" =~ ^[1-9][0-9]*$ ]] || die "OBJECT_SIZE must be a positive integer"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

export DS_UB_DEV_NAME DS_UB_DEV_EID
export DATASYSTEM_CLIENT_LOG_DIR="$LOG_DIR"
export DATASYSTEM_LOG_MONITOR_ENABLE=true
export DATASYSTEM_CLIENT_LOG_WITHOUT_PID=true

REPORT="$OUTPUT_DIR/${PREFIX}-probe.log"
ACCESS_EVIDENCE="$OUTPUT_DIR/${PREFIX}-access.log"

probe_args=(
  --host "$WORKER_HOST" \
  --port "$WORKER_PORT" \
  --object_size "$OBJECT_SIZE" \
  --iterations "$ITERATIONS" \
  --concurrency "$CONCURRENCY" \
  --prefix "$PREFIX" \
  --cleanup true
)
[[ -z "$START_FILE" ]] || probe_args+=(--start_file "$START_FILE")
"$KVC_UB_PROBE_BIN" "${probe_args[@]}" | tee "$REPORT"

if [[ "$CONCURRENCY" -eq 1 ]]; then
  grep -Eq "^KVC_UB_INTEGRITY_PASS( concurrency=1 object_size=$OBJECT_SIZE operations=$ITERATIONS)?$" "$REPORT" \
    || die "probe did not report integrity success"
else
  grep -q "^KVC_UB_INTEGRITY_PASS concurrency=$CONCURRENCY object_size=$OBJECT_SIZE operations=$((ITERATIONS * CONCURRENCY))$" "$REPORT" \
    || die "probe did not report c$CONCURRENCY integrity success"
fi

deadline=$((SECONDS + ACCESS_LOG_TIMEOUT_SECONDS))
while :; do
  : >"$ACCESS_EVIDENCE"
  shopt -s nullglob
  access_logs=("$LOG_DIR"/ds_client_access*.log*)
  shopt -u nullglob
  if ((${#access_logs[@]} > 0)); then
    grep -hF "$PREFIX" "${access_logs[@]}" >"$ACCESS_EVIDENCE" || true
  fi
  observed=$(awk '/DS_KV_CLIENT_(SET|GET)/ {count++} END {print count + 0}' "$ACCESS_EVIDENCE")
  ((observed >= ITERATIONS * CONCURRENCY * 2)) && break
  ((SECONDS >= deadline)) && break
  sleep 1
done

set_ub=$(awk '/DS_KV_CLIENT_SET/ && /transportType:UB/ {count++} END {print count + 0}' "$ACCESS_EVIDENCE")
get_ub=$(awk '/DS_KV_CLIENT_GET/ && /transportType:UB/ {count++} END {print count + 0}' "$ACCESS_EVIDENCE")
tcp=$(awk '/DS_KV_CLIENT_(SET|GET)/ && /transportType:TCP/ {count++} END {print count + 0}' "$ACCESS_EVIDENCE")
total=$(awk '/DS_KV_CLIENT_(SET|GET)/ {count++} END {print count + 0}' "$ACCESS_EVIDENCE")

expected_operations=$((ITERATIONS * CONCURRENCY))
[[ "$set_ub" -eq "$expected_operations" ]] \
  || die "expected $expected_operations UB Set records, observed $set_ub"
[[ "$get_ub" -eq "$expected_operations" ]] \
  || die "expected $expected_operations UB Get records, observed $get_ub"
[[ "$tcp" -eq 0 ]] || die "observed $tcp TCP fallback records"
[[ "$total" -eq $((expected_operations * 2)) ]] \
  || die "expected $((expected_operations * 2)) Set/Get access records, observed $total"

echo "KVC_UB_ACCESS_LOG_PASS prefix=$PREFIX concurrency=$CONCURRENCY object_size=$OBJECT_SIZE set_ub=$set_ub get_ub=$get_ub tcp=$tcp"
echo "probe_report=$REPORT"
echo "access_evidence=$ACCESS_EVIDENCE"
echo "KVC_UB_POC_PASS"
