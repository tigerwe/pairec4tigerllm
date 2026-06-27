#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
BRPC_TARGET="${BRPC_TARGET:-deployment/inference-brpc-trtllm}"
BRPC_CONTAINER="${BRPC_CONTAINER:-brpc-inference}"
BRPC_PORT="${BRPC_PORT:-18100}"
ENDPOINT="${ENDPOINT:-}"

USER_ID="${USER_ID:-go_brpc_probe}"
TOPK="${TOPK:-1}"
REQUESTS="${REQUESTS:-20}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
MAX_RETRIES="${MAX_RETRIES:-1}"
HISTORY_SOURCE="${HISTORY_SOURCE:-synthetic}"
UIDS="${UIDS:-}"
USER_FEATURES_PATH="${USER_FEATURES_PATH:-data/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-data/tenrec/processed/semantic_id_map.json}"
HISTORY_MAX_LENGTH="${HISTORY_MAX_LENGTH:-20}"
VARY_USER_ID="${VARY_USER_ID:-true}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/go_brpc_probe_kvc_latency/${RUN_ID}}"
PROBE_LOG="${OUT_DIR}/go_brpc_probe.log"
SERVER_LOG="${OUT_DIR}/brpc_inference.log"
SUMMARY_TXT="${OUT_DIR}/summary.txt"
SUMMARY_JSON="${OUT_DIR}/summary.json"

SERVER_LOG_PID=""

log() {
  printf '\n== %s ==\n' "$*"
}

cleanup() {
  if [ -n "$SERVER_LOG_PID" ] && kill -0 "$SERVER_LOG_PID" >/dev/null 2>&1; then
    kill "$SERVER_LOG_PID" >/dev/null 2>&1 || true
    wait "$SERVER_LOG_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command: $1" >&2
    exit 1
  fi
}

resolve_endpoint() {
  if [ -n "$ENDPOINT" ]; then
    return
  fi
  local service_ip
  service_ip="$(kubectl -n "$NAMESPACE" get svc "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
  if [ -z "$service_ip" ] || [ "$service_ip" = "None" ]; then
    echo "ERROR: service ${NAMESPACE}/${INFERENCE_SERVICE} has no ClusterIP" >&2
    exit 1
  fi
  ENDPOINT="${service_ip}:${BRPC_PORT}"
}

start_server_log_collector() {
  : >"$SERVER_LOG"
  kubectl -n "$NAMESPACE" logs --tail=0 -f "$BRPC_TARGET" -c "$BRPC_CONTAINER" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_LOG_PID="$!"
  sleep 2
}

stop_server_log_collector() {
  cleanup
  SERVER_LOG_PID=""
}

run_probe() {
  set +e
  GOPROXY="${GOPROXY:-off}" \
  GOSUMDB="${GOSUMDB:-off}" \
  go run -mod=vendor ./scripts/probe_go_brpc_client.go \
    --endpoint="$ENDPOINT" \
    --method=recommend \
    --user_id="$USER_ID" \
    --topk="$TOPK" \
    --requests="$REQUESTS" \
    --timeout_ms="$TIMEOUT_MS" \
    --max_retries="$MAX_RETRIES" \
    --history_source="$HISTORY_SOURCE" \
    --uids="$UIDS" \
    --user_features_path="$USER_FEATURES_PATH" \
    --semantic_map_path="$SEMANTIC_MAP_PATH" \
    --history_max_length="$HISTORY_MAX_LENGTH" \
    --vary_user_id="$VARY_USER_ID" \
    >"$PROBE_LOG" 2>&1
  local code="$?"
  set -e
  cat "$PROBE_LOG"
  echo "$code" >"${OUT_DIR}/probe.exit_code"
  return "$code"
}

summarize() {
  python3 - "$PROBE_LOG" "$SERVER_LOG" "$SUMMARY_JSON" <<'PY' | tee "$SUMMARY_TXT"
import json
import math
import re
import statistics
import sys
from collections import defaultdict

probe_log_path, server_log_path, summary_json = sys.argv[1:4]

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""

def parse_kv(line):
    fields = {}
    for part in line.replace("\t", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value.rstrip(".,")
    return fields

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * quantile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight

def summarize_values(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "avg": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "p9999": percentile(values, 0.9999),
        "max": max(values),
    }

def print_metric(name, values, digits=3):
    stats = summarize_values(values)
    if not stats:
        print(f"  {name:<30} (no samples)")
        return None
    print(
        f"  {name:<30} count={int(stats['count']):>5} "
        f"avg={stats['avg']:.{digits}f} p50={stats['p50']:.{digits}f} "
        f"p95={stats['p95']:.{digits}f} p99={stats['p99']:.{digits}f} "
        f"p9999={stats['p9999']:.{digits}f} max={stats['max']:.{digits}f}"
    )
    return stats

probe_lines = read_text(probe_log_path).splitlines()
server_lines = read_text(server_log_path).splitlines()

probe_events = [
    parse_kv(line)
    for line in probe_lines
    if line.startswith("recommend ok ")
]
server_events = [
    parse_kv(line)
    for line in server_lines
    if "[brpc-inference]" in line and "method=Recommend" in line
]
ds_events = [
    parse_kv(line)
    for line in server_lines
    if "[Datasystem][TRACE]" in line
]

server_by_request_id = {
    event.get("request_id"): event
    for event in server_events
    if event.get("request_id")
}

paired = []
pairing_mode = "none"
if server_by_request_id:
    for event in probe_events:
        match = server_by_request_id.get(event.get("request_id"))
        if match:
            paired.append((event, match))
    pairing_mode = "request_id"
elif len(probe_events) == len(server_events):
    paired = list(zip(probe_events, server_events))
    pairing_mode = "serial_order"

probe_rpc_ms = [as_float(event.get("latency_ms")) for event in probe_events]
probe_inference_ms = [as_float(event.get("inference_ms")) for event in probe_events]
server_ms = [as_float(event.get("latency_ms")) for event in server_events]
comm_est_ms = []
for probe_event, server_event in paired:
    rpc_ms = as_float(probe_event.get("latency_ms"))
    srv_ms = as_float(server_event.get("latency_ms"))
    if rpc_ms is not None and srv_ms is not None:
        comm_est_ms.append(max(0.0, rpc_ms - srv_ms))

offloads = [event for event in ds_events if event.get("op") == "offload"]
onboards = [event for event in ds_events if event.get("op") == "onboard"]
probe_count = len(probe_events)
offload_per_probe = (len(offloads) / probe_count) if probe_count else None
onboard_per_probe = (len(onboards) / probe_count) if probe_count else None
datasystem_per_probe = (len(ds_events) / probe_count) if probe_count else None

def values(events, key):
    return [as_float(event.get(key)) for event in events]

print("Go brpc probe latency summary")
print(f"  probe_recommend_events={len(probe_events)}")
print(f"  brpc_server_events={len(server_events)}")
print(f"  datasystem_events={len(ds_events)} offload={len(offloads)} onboard={len(onboards)}")
print(f"  residual_pairing={pairing_mode}")
if pairing_mode == "none":
    print("  WARN: cannot pair probe latency with server latency.")

print("\n== brpc/TCP ==")
probe_rpc_stats = print_metric("go_probe_brpc_rpc_ms", probe_rpc_ms, 3)
probe_inference_stats = print_metric("go_probe_inference_ms", probe_inference_ms, 3)
server_stats = print_metric("server_method_ms", server_ms, 3)
comm_stats = print_metric("brpc_comm_est_ms", comm_est_ms, 3)
print("  note: brpc_comm_est_ms = Go probe brpc RPC wall-clock - C++ server method latency.")
print("        It includes Go encode/decode, brpc framing, TCP/CNI/kube-proxy, and server-side time outside the measured method.")

print("\n== call counts per brpc request ==")
print("  brpc_calls_per_probe          1.000")
if offload_per_probe is None:
    print("  offload_set_per_probe         (no samples)")
    print("  onboard_get_per_probe         (no samples)")
    print("  datasystem_access_per_probe   (no samples)")
else:
    print(f"  offload_set_per_probe         {offload_per_probe:.3f}")
    print(f"  onboard_get_per_probe         {onboard_per_probe:.3f}")
    print(f"  datasystem_access_per_probe   {datasystem_per_probe:.3f}")
print("  note: onboard_get_per_probe is the observed KVC Get count per brpc Recommend request.")

print("\n== KVC/DataSystem per block ==")
offload_total = print_metric("offload.total_ms", values(offloads, "total_ms"), 3)
offload_set = print_metric("offload.set_ms", values(offloads, "set_ms"), 3)
offload_d2h = print_metric("offload.d2h_ms", values(offloads, "d2h_ms"), 3)
onboard_total = print_metric("onboard.total_ms", values(onboards, "total_ms"), 3)
onboard_get = print_metric("onboard.get_ms", values(onboards, "get_ms"), 3)
onboard_h2d = print_metric("onboard.h2d_ms", values(onboards, "h2d_ms"), 3)

summary = {
    "counts": {
        "probe_recommend_events": len(probe_events),
        "brpc_server_events": len(server_events),
        "datasystem_events": len(ds_events),
        "offload_events": len(offloads),
        "onboard_events": len(onboards),
        "brpc_calls_per_probe": 1.0 if probe_count else None,
        "offload_set_per_probe": offload_per_probe,
        "onboard_get_per_probe": onboard_per_probe,
        "datasystem_access_per_probe": datasystem_per_probe,
    },
    "pairing_mode": pairing_mode,
    "brpc": {
        "go_probe_brpc_rpc_ms": probe_rpc_stats,
        "go_probe_inference_ms": probe_inference_stats,
        "server_method_ms": server_stats,
        "brpc_comm_est_ms": comm_stats,
    },
    "datasystem": {
        "offload_total_ms": offload_total,
        "offload_set_ms": offload_set,
        "offload_d2h_ms": offload_d2h,
        "onboard_total_ms": onboard_total,
        "onboard_get_ms": onboard_get,
        "onboard_h2d_ms": onboard_h2d,
    },
}
with open(summary_json, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
print(f"\nwrote_json={summary_json}")
PY
}

mkdir -p "$OUT_DIR"
require_command kubectl
require_command python3
require_command go
resolve_endpoint

log "Output directory"
echo "$OUT_DIR"

log "Topology"
kubectl -n "$NAMESPACE" get pods -o wide | grep -E 'inference-brpc-trtllm|datasystem-pool' \
  | tee "${OUT_DIR}/pods.txt" || true
kubectl -n "$NAMESPACE" get svc "$INFERENCE_SERVICE" -o wide \
  | tee "${OUT_DIR}/service.txt"
echo "endpoint=${ENDPOINT}" | tee "${OUT_DIR}/endpoint.txt"
{
  echo "history_source=${HISTORY_SOURCE}"
  echo "uids=${UIDS}"
  echo "user_features_path=${USER_FEATURES_PATH}"
  echo "semantic_map_path=${SEMANTIC_MAP_PATH}"
  echo "history_max_length=${HISTORY_MAX_LENGTH}"
} | tee "${OUT_DIR}/request_source.txt"

log "Start inference log collector"
start_server_log_collector

log "Run Go brpc probe"
set +e
run_probe
PROBE_CODE="$?"
set -e
sleep 3
stop_server_log_collector

log "Summarize brpc/KVC latency"
summarize

if [ "$PROBE_CODE" -ne 0 ]; then
  echo "ERROR: Go brpc probe failed with code ${PROBE_CODE}" >&2
  exit "$PROBE_CODE"
fi

log "Artifacts"
find "$OUT_DIR" -maxdepth 1 -type f | sort
echo "Go brpc probe benchmark finished. OUT_DIR=${OUT_DIR}"
