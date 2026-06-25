#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec}"
BRPC_TARGET="${BRPC_TARGET:-deployment/inference-brpc-trtllm}"
BRPC_CONTAINER="${BRPC_CONTAINER:-brpc-inference}"

USER_ID="${USER_ID:-6312}"
SIZE="${SIZE:-1}"
SCENE_ID="${SCENE_ID:-home_feed}"
TIMEOUT="${TIMEOUT:-30}"

PAIREC_URL_WAS_SET=0
if [ "${PAIREC_URL+x}" = "x" ]; then
  PAIREC_URL_WAS_SET=1
fi
LOCAL_PORT="${LOCAL_PORT:-18080}"
PAIREC_URL="${PAIREC_URL:-http://127.0.0.1:${LOCAL_PORT}/api/recommend}"
START_PORT_FORWARD="${START_PORT_FORWARD:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/pairec_single_request_trace/${RUN_ID}}"

PORT_FORWARD_PID=""
PAIREC_LOG_PID=""
TRT_LOG_PID=""
PAIREC_LOG="${OUT_DIR}/pairec_stdout.log"
TRT_LOG="${OUT_DIR}/brpc_trtllm.log"
PAIREC_EXTRA_LOG="${OUT_DIR}/pairec_request_trace.log"
RESPONSE_JSON="${OUT_DIR}/response.json"
CLIENT_JSON="${OUT_DIR}/client.json"
SUMMARY_JSON="${OUT_DIR}/summary.json"
SUMMARY_TXT="${OUT_DIR}/summary.txt"

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

cleanup() {
  cleanup_pid "$PAIREC_LOG_PID"
  cleanup_pid "$TRT_LOG_PID"
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
  sleep 2
  if ! kill -0 "$PORT_FORWARD_PID" >/dev/null 2>&1; then
    echo "ERROR: port-forward exited. Log:" >&2
    cat "${OUT_DIR}/pairec_port_forward.log" >&2 || true
    exit 1
  fi
  echo "port-forward pid=$PORT_FORWARD_PID url=$PAIREC_URL"
}

start_log_collectors() {
  log "Start log collectors"
  : >"$PAIREC_LOG"
  : >"$TRT_LOG"

  kubectl -n "$NAMESPACE" logs --tail=0 -f "$PAIREC_TARGET" \
    >"$PAIREC_LOG" 2>&1 &
  PAIREC_LOG_PID="$!"

  kubectl -n "$NAMESPACE" logs --tail=0 -f "$BRPC_TARGET" -c "$BRPC_CONTAINER" \
    >"$TRT_LOG" 2>&1 &
  TRT_LOG_PID="$!"
  sleep 2
}

send_request() {
  log "Send one PaiRec request"
  python3 - "$PAIREC_URL" "$USER_ID" "$SIZE" "$SCENE_ID" "$TIMEOUT" "$CLIENT_JSON" "$RESPONSE_JSON" <<'PY'
import json
import sys
import time
import urllib.request

url, uid, size, scene_id, timeout, client_json, response_json = sys.argv[1:8]
payload = {
    "scene_id": scene_id,
    "uid": uid,
    "size": int(size),
}
data = json.dumps(payload).encode("utf-8")
request = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
started = time.perf_counter()
try:
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        body_text = response.read().decode("utf-8", errors="replace")
        http_status = response.status
except Exception as exc:  # noqa: BLE001 - keep the artifact explicit.
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = {
        "ok": False,
        "error": str(exc),
        "client_e2e_ms": elapsed_ms,
        "payload": payload,
    }
    with open(client_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with open(response_json, "w", encoding="utf-8") as handle:
        json.dump({}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1)

elapsed_ms = (time.perf_counter() - started) * 1000
body = json.loads(body_text)
client = {
    "ok": body.get("code") == 200,
    "http_status": http_status,
    "client_e2e_ms": elapsed_ms,
    "payload": payload,
    "response_code": body.get("code"),
    "response_msg": body.get("msg", ""),
    "request_id": body.get("request_id", ""),
    "item_count": len(body.get("items") or []),
}
with open(client_json, "w", encoding="utf-8") as handle:
    json.dump(client, handle, ensure_ascii=False, indent=2)
with open(response_json, "w", encoding="utf-8") as handle:
    json.dump(body, handle, ensure_ascii=False, indent=2)
print(json.dumps(client, ensure_ascii=False, indent=2))
PY
}

collect_pairec_request_trace() {
  local request_id="$1"
  : >"$PAIREC_EXTRA_LOG"
  if [ -z "$request_id" ]; then
    return
  fi

  log "Collect PaiRec request trace from pod files"
  kubectl -n "$NAMESPACE" exec "$PAIREC_TARGET" -- \
    sh -c "REQ='${request_id}'; for f in /tmp/*INFO* /tmp/pairec* /tmp/*log; do [ -f \"\$f\" ] && grep -h \"\$REQ\" \"\$f\"; done 2>/dev/null || true" \
    >"$PAIREC_EXTRA_LOG" 2>/dev/null || true
  if [ -s "$PAIREC_EXTRA_LOG" ]; then
    cat "$PAIREC_EXTRA_LOG"
  else
    echo "(no requestId=${request_id} trace line found in PaiRec pod /tmp files)"
  fi
}

summarize() {
  log "Single request summary"
  python3 - "$CLIENT_JSON" "$RESPONSE_JSON" "$PAIREC_LOG" "$PAIREC_EXTRA_LOG" "$TRT_LOG" "$SUMMARY_JSON" <<'PY' | tee "$SUMMARY_TXT"
import json
import sys

client_json, response_json, pairec_log_path, pairec_extra_path, trt_log_path, summary_json = sys.argv[1:7]

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}

def parse_kv_fields(line):
    fields = {}
    for part in line.replace("\t", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value.rstrip(".,")
    return fields

def numeric(fields, key):
    try:
        return float(fields.get(key, 0))
    except (TypeError, ValueError):
        return 0.0

client = read_json(client_json)
response = read_json(response_json)
pairec_log = read_text(pairec_log_path)
pairec_extra = read_text(pairec_extra_path)
trt_log = read_text(trt_log_path)
request_id = str(client.get("request_id") or response.get("request_id") or "")

pairec_lines = []
for line in (pairec_log + "\n" + pairec_extra).splitlines():
    if request_id and request_id in line:
        pairec_lines.append(line)

generative_trace = {}
recommend_trace = {}
for line in pairec_lines:
    fields = parse_kv_fields(line)
    if fields.get("module") == "GenerativeRecall":
        generative_trace = fields
    elif fields.get("module") == "RecommendTrace":
        recommend_trace = fields

brpc_lines = [
    line for line in trt_log.splitlines()
    if "[brpc-inference]" in line and "method=Recommend" in line
]
brpc_events = [parse_kv_fields(line) for line in brpc_lines]

ds_events = []
for line in trt_log.splitlines():
    if "[Datasystem][TRACE]" not in line:
        continue
    fields = parse_kv_fields(line)
    fields["_raw"] = line
    ds_events.append(fields)

offloads = [event for event in ds_events if event.get("op") == "offload"]
onboards = [event for event in ds_events if event.get("op") == "onboard"]

summary = {
    "request_id": request_id,
    "client": client,
    "response_code": response.get("code"),
    "response_msg": response.get("msg", ""),
    "response_size": response.get("size"),
    "brpc_calls": len(brpc_events),
    "brpc_events": brpc_events,
    "kvc_access": {
        "offload_count": len(offloads),
        "onboard_count": len(onboards),
        "total_count": len(ds_events),
        "offload_events": offloads,
        "onboard_events": onboards,
    },
    "pairec_generative_trace": generative_trace,
    "pairec_recommend_trace": recommend_trace,
}

with open(summary_json, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"request_id={request_id}")
print(f"response code={response.get('code')} msg={response.get('msg','')} size={response.get('size')}")
print(f"E2E client_ms={client.get('client_e2e_ms', 0):.3f}")
print(f"brpc_calls={len(brpc_events)}")
for index, event in enumerate(brpc_events, start=1):
    print(
        f"  brpc[{index}] code={event.get('code','')} items={event.get('items','')} "
        f"latency_ms={event.get('latency_ms','')} backend={event.get('backend','')}"
    )

if generative_trace:
    print("PaiRec GenerativeRecall stages:")
    for key in (
        "from", "protocol", "cost", "cache_ms", "history_ms", "convert_ms",
        "rpc_ms", "brpc_ms", "http_ms", "items_ms", "http_overhead_ms",
        "inference_svc_ms", "tr_total_ms", "tr_infer_ms", "tr_prompt_ms",
        "tr_runner_ms", "tr_parse_ms", "tr_pad_ms", "tr_backend_total_ms",
        "tr_map_ms", "tr_kv_lookup_ms", "tr_kv_write_ms",
        "tr_result_cache_lookup_ms", "tr_result_cache_ds_lookup_ms",
        "tr_result_cache_write_submit_ms",
    ):
        if key in generative_trace:
            print(f"  {key}={generative_trace[key]}")
else:
    print("PaiRec GenerativeRecall stages: not found in captured logs")

if recommend_trace:
    print("PaiRec RecommendTrace stages:")
    for key in (
        "total_ms", "user_feature_ms", "recall_ms", "filter_ms",
        "general_rank_ms", "feature_ms", "rank_ms",
        "pipeline_wait_ms", "merge_ms", "sort_ms",
    ):
        if key in recommend_trace:
            print(f"  {key}={recommend_trace[key]}")
else:
    print("PaiRec RecommendTrace stages: not found in captured logs")

print("KVC/DataSystem access:")
print(f"  offload_count={len(offloads)}")
print(f"  onboard_count={len(onboards)}")
print(f"  total_count={len(ds_events)}")

for index, event in enumerate(offloads, start=1):
    print(
        f"  offload[{index}] key={event.get('key','')} size_bytes={event.get('size_bytes','')} "
        f"create_ms={event.get('create_ms','')} d2h_ms={event.get('d2h_ms','')} "
        f"set_ms={event.get('set_ms','')} total_ms={event.get('total_ms','')}"
    )
for index, event in enumerate(onboards, start=1):
    print(
        f"  onboard[{index}] key={event.get('key','')} size_bytes={event.get('size_bytes','')} "
        f"get_ms={event.get('get_ms','')} h2d_ms={event.get('h2d_ms','')} "
        f"total_ms={event.get('total_ms','')}"
    )

if brpc_events:
    brpc_ms = sum(numeric(event, "latency_ms") for event in brpc_events)
    print(f"derived non_brpc_client_overhead_ms={client.get('client_e2e_ms', 0) - brpc_ms:.3f}")

print(f"wrote_json={summary_json}")
PY
}

mkdir -p "$OUT_DIR"
require_command kubectl
require_command python3

log "Output directory"
echo "$OUT_DIR"

log "Environment"
kubectl -n "$NAMESPACE" get pods -o wide | grep -E 'pairec|inference-brpc-trtllm|datasystem-pool' \
  | tee "${OUT_DIR}/pods.txt" || true
kubectl -n "$NAMESPACE" get svc pairec inference-brpc-trtllm -o wide \
  | tee "${OUT_DIR}/services.txt"

start_port_forward
start_log_collectors
send_request
sleep 3
cleanup_pid "$PAIREC_LOG_PID"
cleanup_pid "$TRT_LOG_PID"
PAIREC_LOG_PID=""
TRT_LOG_PID=""

REQUEST_ID="$(python3 - "$CLIENT_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle).get("request_id", ""))
PY
)"

collect_pairec_request_trace "$REQUEST_ID"
summarize

log "Artifacts"
find "$OUT_DIR" -maxdepth 1 -type f | sort
echo "Single request trace finished. OUT_DIR=${OUT_DIR}"
