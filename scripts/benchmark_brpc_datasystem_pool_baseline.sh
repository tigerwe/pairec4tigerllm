#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec}"
BRPC_TARGET="${BRPC_TARGET:-deployment/inference-brpc-trtllm}"
BRPC_CONTAINER="${BRPC_CONTAINER:-brpc-inference}"
BRPC_SERVER="${BRPC_SERVER:-10.96.15.101:18100}"

PAIREC_URL_WAS_SET=0
if [ "${PAIREC_URL+x}" = "x" ]; then
  PAIREC_URL_WAS_SET=1
fi
LOCAL_PORT="${LOCAL_PORT:-18080}"
PAIREC_URL="${PAIREC_URL:-http://127.0.0.1:${LOCAL_PORT}/api/recommend}"
UIDS="${UIDS:-6312,130,2184,7494}"
SCENE_ID="${SCENE_ID:-home_feed}"

BRPC_REQUESTS="${BRPC_REQUESTS:-20}"
TOPK="${TOPK:-10}"

E2E_REQUESTS="${E2E_REQUESTS:-100}"
E2E_REPEAT_REQUESTS="${E2E_REPEAT_REQUESTS:-50}"
E2E_CONCURRENCY="${E2E_CONCURRENCY:-10}"
E2E_SIZE="${E2E_SIZE:-1}"
E2E_TIMEOUT="${E2E_TIMEOUT:-30}"
E2E_WARMUP="${E2E_WARMUP:-0}"

RUN_QUALITY="${RUN_QUALITY:-1}"
QUALITY_SIZE="${QUALITY_SIZE:-10}"
QUALITY_REQUESTS="${QUALITY_REQUESTS:-100}"
QUALITY_REPEAT_REQUESTS="${QUALITY_REPEAT_REQUESTS:-50}"
QUALITY_CONCURRENCY="${QUALITY_CONCURRENCY:-${E2E_CONCURRENCY}}"

START_PORT_FORWARD="${START_PORT_FORWARD:-1}"
STRICT_E2E="${STRICT_E2E:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/pairec_brpc_datasystem_pool_baseline/${RUN_ID}}"

PORT_FORWARD_PID=""
PAIREC_LOG_PID=""
TRT_LOG_PID=""
CURRENT_PAIREC_LOG=""
CURRENT_TRT_LOG=""

log() {
  printf '\n== %s ==\n' "$*"
}

cleanup_pid() {
  local pid="${1:-}"
  if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

stop_log_collectors() {
  cleanup_pid "$PAIREC_LOG_PID"
  cleanup_pid "$TRT_LOG_PID"
  PAIREC_LOG_PID=""
  TRT_LOG_PID=""
}

cleanup() {
  stop_log_collectors
  cleanup_pid "$PORT_FORWARD_PID"
}

trap cleanup EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command: $1" >&2
    exit 1
  fi
}

find_free_local_port() {
  local start_port="$1"
  python3 - "$start_port" <<'PY'
import socket
import sys

start = int(sys.argv[1])

def can_bind(host, port, family):
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()

for port in range(start, start + 200):
    ok4 = can_bind("127.0.0.1", port, socket.AF_INET)
    try:
        ok6 = can_bind("::1", port, socket.AF_INET6)
    except OSError:
        ok6 = True
    if ok4 and ok6:
        print(port)
        raise SystemExit(0)

raise SystemExit(f"no free local port in range [{start}, {start + 199}]")
PY
}

prepare_port_forward_endpoint() {
  if [ "$PAIREC_URL_WAS_SET" = "1" ]; then
    return
  fi

  local selected_port
  selected_port="$(find_free_local_port "$LOCAL_PORT")"
  if [ "$selected_port" != "$LOCAL_PORT" ]; then
    echo "local port ${LOCAL_PORT} is busy; using ${selected_port}"
    LOCAL_PORT="$selected_port"
  fi
  PAIREC_URL="http://127.0.0.1:${LOCAL_PORT}/api/recommend"
}

start_port_forward() {
  if [ "$START_PORT_FORWARD" != "1" ]; then
    echo "skip port-forward: START_PORT_FORWARD=$START_PORT_FORWARD"
    return
  fi

  log "Start PaiRec port-forward"
  prepare_port_forward_endpoint
  kubectl -n "$NAMESPACE" port-forward "$PAIREC_TARGET" "${LOCAL_PORT}:18080" \
    >"${OUT_DIR}/pairec_port_forward.log" 2>&1 &
  PORT_FORWARD_PID="$!"
  sleep 3
  if ! kill -0 "$PORT_FORWARD_PID" >/dev/null 2>&1; then
    echo "ERROR: port-forward exited. Log:" >&2
    cat "${OUT_DIR}/pairec_port_forward.log" >&2 || true
    exit 1
  fi
  echo "port-forward pid=$PORT_FORWARD_PID url=$PAIREC_URL"
}

start_log_collectors() {
  local label="$1"
  stop_log_collectors
  CURRENT_PAIREC_LOG="${OUT_DIR}/pairec_${label}.log"
  CURRENT_TRT_LOG="${OUT_DIR}/server_${label}.log"
  : >"$CURRENT_PAIREC_LOG"
  : >"$CURRENT_TRT_LOG"

  # --tail=0 is important: kubectl logs -f otherwise dumps historical logs first,
  # which would inflate brpc/KV call counts for this benchmark run.
  kubectl -n "$NAMESPACE" logs --tail=0 -f "$PAIREC_TARGET" \
    >"$CURRENT_PAIREC_LOG" 2>&1 &
  PAIREC_LOG_PID="$!"

  kubectl -n "$NAMESPACE" logs --tail=0 -f "$BRPC_TARGET" -c "$BRPC_CONTAINER" \
    >"$CURRENT_TRT_LOG" 2>&1 &
  TRT_LOG_PID="$!"

  sleep 2
}

summarize_trt_log() {
  local label="$1"
  local trt_log="$2"
  local json_output="${OUT_DIR}/summary_${label}.json"
  local text_output="${OUT_DIR}/summary_${label}.txt"

  python3 - "$label" "$trt_log" "$json_output" <<'PY' | tee "$text_output"
import json
import math
import re
import sys
from collections import Counter, defaultdict

label, log_path, json_output = sys.argv[1:4]

def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight

def summarize(values):
    values = [float(v) for v in values]
    if not values:
        return {}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "p9999": percentile(values, 0.9999),
        "max": max(values),
    }

def parse_kv_fields(line):
    fields = {}
    for part in line.replace("\t", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value.rstrip(".,")
    return fields

try:
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
except FileNotFoundError:
    lines = []

brpc_re = re.compile(r"\[brpc-inference\]\s+method=Recommend\b")
brpc_latencies = []
brpc_codes = Counter()
item_counts = Counter()
for line in lines:
    if not brpc_re.search(line):
        continue
    fields = parse_kv_fields(line)
    if "latency_ms" in fields:
        try:
            brpc_latencies.append(float(fields["latency_ms"]))
        except ValueError:
            pass
    if "code" in fields:
        brpc_codes[fields["code"]] += 1
    if "items" in fields:
        item_counts[fields["items"]] += 1

ds_events = defaultdict(list)
for line in lines:
    if "[Datasystem][TRACE]" not in line:
        continue
    fields = parse_kv_fields(line)
    op = fields.get("op")
    if op:
        ds_events[op].append(fields)

ds_metric_fields = {
    "offload": ["create_ms", "d2h_ms", "set_ms", "total_ms"],
    "onboard": ["get_ms", "h2d_ms", "total_ms"],
}
ds_metrics = {}
for op, metric_fields in ds_metric_fields.items():
    metrics = {}
    for metric in metric_fields:
        values = []
        for event in ds_events.get(op, []):
            try:
                values.append(float(event[metric]))
            except (KeyError, ValueError):
                pass
        metrics[metric] = summarize(values)
    ds_metrics[op] = {
        "events": len(ds_events.get(op, [])),
        "metrics": metrics,
    }

brpc_calls = len(brpc_latencies)
offload_events = len(ds_events.get("offload", []))
onboard_events = len(ds_events.get("onboard", []))
summary = {
    "label": label,
    "brpc_calls": brpc_calls,
    "brpc_codes": dict(brpc_codes),
    "brpc_latency_ms": summarize(brpc_latencies),
    "item_distribution": dict(sorted(item_counts.items(), key=lambda item: int(item[0]))),
    "datasystem": ds_metrics,
    "offload_per_brpc": (offload_events / brpc_calls) if brpc_calls else None,
    "onboard_per_brpc": (onboard_events / brpc_calls) if brpc_calls else None,
}

with open(json_output, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"summary label={label}")
print(f"  brpc_calls={brpc_calls}")
if brpc_codes:
    print("  brpc_codes=" + ",".join(f"{k}:{v}" for k, v in sorted(brpc_codes.items())))
if brpc_latencies:
    latency = summary["brpc_latency_ms"]
    print(
        "  brpc_latency_ms "
        f"avg={latency['avg']:.1f} p50={latency['p50']:.1f} "
        f"p95={latency['p95']:.1f} p99={latency['p99']:.1f} "
        f"p9999={latency['p9999']:.1f} max={latency['max']:.1f}"
    )
if item_counts:
    print("  item_distribution=" + ",".join(
        f"items_{k}:{v}" for k, v in sorted(item_counts.items(), key=lambda item: int(item[0]))
    ))
print(f"  offload_events={offload_events}")
print(f"  onboard_events={onboard_events}")
if brpc_calls:
    print(f"  offload_per_brpc={offload_events / brpc_calls:.3f}")
    print(f"  onboard_per_brpc={onboard_events / brpc_calls:.3f}")
for op in ("offload", "onboard"):
    op_summary = ds_metrics[op]
    print(f"  {op}: events={op_summary['events']}")
    for metric, values in op_summary["metrics"].items():
        if not values:
            continue
        print(
            f"    {metric} avg={values['avg']:.3f} p50={values['p50']:.3f} "
            f"p95={values['p95']:.3f} p99={values['p99']:.3f} "
            f"p9999={values['p9999']:.3f} max={values['max']:.3f}"
        )
print(f"  wrote_json={json_output}")
PY
}

run_e2e_benchmark() {
  local label="$1"
  local size="$2"
  local requests="$3"
  local repeat_requests="$4"
  local concurrency="$5"
  local output_txt="${OUT_DIR}/e2e_${label}.txt"
  local output_json="${OUT_DIR}/e2e_${label}.json"

  log "Run E2E benchmark: ${label}"
  start_log_collectors "$label"

  set +e
  python3 scripts/benchmark_e2e_latency.py \
    --url "$PAIREC_URL" \
    --uids "$UIDS" \
    --requests "$requests" \
    --repeat-requests "$repeat_requests" \
    --warmup "$E2E_WARMUP" \
    --concurrency "$concurrency" \
    --size "$size" \
    --scene-id "$SCENE_ID" \
    --timeout "$E2E_TIMEOUT" \
    --pairec-log "$CURRENT_PAIREC_LOG" \
    --trt-log "$CURRENT_TRT_LOG" \
    --json-output "$output_json" \
    >"$output_txt" 2>&1
  local code="$?"
  set -e

  stop_log_collectors
  cat "$output_txt"

  log "Trace summary: ${label}"
  summarize_trt_log "$label" "$CURRENT_TRT_LOG"

  echo "$code" >"${OUT_DIR}/e2e_${label}.exit_code"
  if [ "$code" -ne 0 ]; then
    echo "WARN: E2E benchmark ${label} exited with code ${code}"
    if [ "$STRICT_E2E" = "1" ]; then
      exit "$code"
    fi
  fi
}

mkdir -p "$OUT_DIR"
require_command kubectl
require_command python3

log "Output directory"
echo "$OUT_DIR"

log "Cluster pods"
kubectl -n "$NAMESPACE" get pods -o wide | tee "${OUT_DIR}/pods.txt"

log "Services"
kubectl -n "$NAMESPACE" get svc pairec inference-brpc-trtllm -o wide | tee "${OUT_DIR}/services.txt"

log "PaiRec brpc config"
kubectl -n "$NAMESPACE" exec "$PAIREC_TARGET" -- \
  sh -c 'grep -n "brpc_endpoint" /app/configs/pairec_config.json || true' \
  | tee "${OUT_DIR}/pairec_brpc_config.txt"

log "Functional smoke"
kubectl -n "$NAMESPACE" exec "$PAIREC_TARGET" -- \
  wget -q -O - \
  --header='Content-Type: application/json' \
  --post-data='{"scene_id":"home_feed","uid":"6312","size":10}' \
  http://127.0.0.1:18080/api/recommend \
  | tee "${OUT_DIR}/functional_smoke.json"
echo

log "Recent brpc server request log"
kubectl -n "$NAMESPACE" logs "$BRPC_TARGET" -c "$BRPC_CONTAINER" --since=2m \
  | grep "method=Recommend" \
  | tail -20 \
  | tee "${OUT_DIR}/recent_brpc_recommend.log" || true

log "Direct brpc/TCP serial smoke"
set +e
TARGET="$BRPC_TARGET" \
CONTAINER="$BRPC_CONTAINER" \
SERVER="$BRPC_SERVER" \
REQUESTS="$BRPC_REQUESTS" \
TOPK="$TOPK" \
bash scripts/test_brpc_native_inference_smoke.sh \
  >"${OUT_DIR}/brpc_direct_smoke.txt" 2>&1
BRPC_SMOKE_CODE="$?"
set -e
cat "${OUT_DIR}/brpc_direct_smoke.txt"
echo "$BRPC_SMOKE_CODE" >"${OUT_DIR}/brpc_direct_smoke.exit_code"
if [ "$BRPC_SMOKE_CODE" -ne 0 ]; then
  echo "ERROR: direct brpc/TCP smoke failed with code ${BRPC_SMOKE_CODE}" >&2
  exit "$BRPC_SMOKE_CODE"
fi

start_port_forward

run_e2e_benchmark \
  "system_size${E2E_SIZE}_c${E2E_CONCURRENCY}" \
  "$E2E_SIZE" \
  "$E2E_REQUESTS" \
  "$E2E_REPEAT_REQUESTS" \
  "$E2E_CONCURRENCY"

if [ "$RUN_QUALITY" = "1" ]; then
  run_e2e_benchmark \
    "quality_size${QUALITY_SIZE}_c${QUALITY_CONCURRENCY}" \
    "$QUALITY_SIZE" \
    "$QUALITY_REQUESTS" \
    "$QUALITY_REPEAT_REQUESTS" \
    "$QUALITY_CONCURRENCY"
fi

log "Artifacts"
find "$OUT_DIR" -maxdepth 1 -type f | sort

cat >"${OUT_DIR}/README.txt" <<EOF
PaiRec brpc + DataSystem pool baseline artifacts

Run ID: ${RUN_ID}
Namespace: ${NAMESPACE}
PaiRec target: ${PAIREC_TARGET}
brpc target: ${BRPC_TARGET}
brpc server: ${BRPC_SERVER}
PaiRec URL: ${PAIREC_URL}
UIDs: ${UIDS}

Main files:
  pods.txt
  services.txt
  functional_smoke.json
  brpc_direct_smoke.txt
  e2e_system_size${E2E_SIZE}_c${E2E_CONCURRENCY}.txt/json
  summary_system_size${E2E_SIZE}_c${E2E_CONCURRENCY}.txt/json

Optional quality files are present when RUN_QUALITY=1.
EOF

echo "Baseline finished. OUT_DIR=${OUT_DIR}"
