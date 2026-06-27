#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
BRPC_PORT="${BRPC_PORT:-18100}"
ENDPOINT="${ENDPOINT:-}"
REQUESTS="${REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-1}"
PAYLOAD_BYTES="${PAYLOAD_BYTES:-102400}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
MAX_RETRIES="${MAX_RETRIES:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/go_brpc_payload_latency/${RUN_ID}}"
PROBE_LOG="${OUT_DIR}/go_brpc_payload_probe.log"
SUMMARY_TXT="${OUT_DIR}/summary.txt"
SUMMARY_JSON="${OUT_DIR}/summary.json"

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

summarize() {
  python3 - "$PROBE_LOG" "$SUMMARY_JSON" <<'PY' | tee "$SUMMARY_TXT"
import json
import math
import statistics
import sys

probe_log_path, summary_json = sys.argv[1:3]

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

def stats(values):
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

with open(probe_log_path, "r", encoding="utf-8", errors="replace") as handle:
    lines = handle.read().splitlines()

events = [parse_kv(line) for line in lines if line.startswith("health ok ")]
summary_events = [parse_kv(line) for line in lines if line.startswith("summary ")]
latencies = [as_float(event.get("latency_ms")) for event in events]
payloads = sorted({int(event.get("payload_bytes", "0")) for event in events})
concurrency = None
if summary_events:
    concurrency = summary_events[-1].get("concurrency")
latency_stats = stats(latencies)

print("Go brpc payload latency summary")
print(f"  health_events={len(events)}")
if concurrency:
    print(f"  concurrency={concurrency}")
print(f"  payload_bytes={','.join(str(v) for v in payloads) if payloads else 'unknown'}")
if latency_stats:
    print(
        "  health_rpc_ms "
        f"count={latency_stats['count']:>5} "
        f"avg={latency_stats['avg']:.3f} "
        f"p50={latency_stats['p50']:.3f} "
        f"p95={latency_stats['p95']:.3f} "
        f"p99={latency_stats['p99']:.3f} "
        f"p9999={latency_stats['p9999']:.3f} "
        f"max={latency_stats['max']:.3f}"
    )
else:
    print("  health_rpc_ms (no samples)")
print("  note: Health carries the padding to the brpc inference service and skips TRT/KVC.")
print("        The server ignores the payload field; this is a brpc/TCP + protobuf/framework latency probe.")

summary = {
    "counts": {"health_events": len(events)},
    "concurrency": int(concurrency) if concurrency else None,
    "payload_bytes": payloads,
    "brpc": {"health_rpc_ms": latency_stats},
}
with open(summary_json, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
print(f"wrote_json={summary_json}")
PY
}

mkdir -p "$OUT_DIR"
require_command kubectl
require_command go
require_command python3
resolve_endpoint

echo "Go brpc payload latency benchmark"
echo "  endpoint:      ${ENDPOINT}"
echo "  requests:      ${REQUESTS}"
echo "  concurrency:   ${CONCURRENCY}"
echo "  payload_bytes: ${PAYLOAD_BYTES}"
echo "  out_dir:       ${OUT_DIR}"
echo

set +e
GOPROXY="${GOPROXY:-off}" \
GOSUMDB="${GOSUMDB:-off}" \
go run -mod=vendor ./scripts/probe_go_brpc_client.go \
  --endpoint="$ENDPOINT" \
  --method=health \
  --requests="$REQUESTS" \
  --concurrency="$CONCURRENCY" \
  --payload_bytes="$PAYLOAD_BYTES" \
  --timeout_ms="$TIMEOUT_MS" \
  --max_retries="$MAX_RETRIES" \
  >"$PROBE_LOG" 2>&1
PROBE_CODE="$?"
set -e

cat "$PROBE_LOG"
echo "$PROBE_CODE" >"${OUT_DIR}/probe.exit_code"

echo
summarize

if [ "$PROBE_CODE" -ne 0 ]; then
  echo "ERROR: Go brpc payload probe failed with code ${PROBE_CODE}" >&2
  exit "$PROBE_CODE"
fi

echo
find "$OUT_DIR" -maxdepth 1 -type f | sort
echo "Go brpc payload benchmark finished. OUT_DIR=${OUT_DIR}"
