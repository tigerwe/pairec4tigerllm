#!/usr/bin/env bash
set -euo pipefail

CLIENT_BIN="${CLIENT_BIN:-/opt/pairec-brpc-ub-recommend-known-good/bin/post_rank_qualification_client}"
SERVER="${SERVER:-127.0.0.1:18311}"
TRANSPORT="${TRANSPORT:-ub}"
REQUESTS="${REQUESTS:-3}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
COMPLETION_TIMEOUT_SECONDS="${COMPLETION_TIMEOUT_SECONDS:-30}"
START_FILE="${START_FILE:-}"
START_WAIT_TIMEOUT_MS="${START_WAIT_TIMEOUT_MS:-60000}"
HOP1_LOG="${HOP1_LOG:-}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/root/brpc-ub-post-rank/evidence}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$CLIENT_BIN" ]] || die "qualification client is not executable: $CLIENT_BIN"
[[ "$TRANSPORT" = tcp || "$TRANSPORT" = ub ]] || die "TRANSPORT must be tcp or ub"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ -n "$HOP1_LOG" && -f "$HOP1_LOG" ]] || die "HOP1_LOG must name the active Hop1 log"
ulimit -n 1048576
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/urma${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$EVIDENCE_DIR"
run_id="$(date +%Y%m%d-%H%M%S)-$$"
client_log="$EVIDENCE_DIR/post-rank-client-$TRANSPORT-$run_id.log"
summary="$EVIDENCE_DIR/post-rank-$TRANSPORT-$run_id.json"

client_args=(
  --server="$SERVER"
  --requests="$REQUESTS"
  --timeout_ms="$TIMEOUT_MS"
  --payload_bytes=102400
  --request_prefix="post-rank-$TRANSPORT-$run_id"
)
if [[ -n "$START_FILE" ]]; then
  client_args+=(--start_file="$START_FILE" --start_wait_timeout_ms="$START_WAIT_TIMEOUT_MS")
fi

"$CLIENT_BIN" "${client_args[@]}" 2>&1 | tee "$client_log"

deadline=$((SECONDS + COMPLETION_TIMEOUT_SECONDS))
while true; do
  complete_count="$(grep -F '"event":"pairec_post_rank_hop2_brpc_burst_complete"' "$HOP1_LOG" \
    | grep -Fc "post-rank-$TRANSPORT-$run_id-" || true)"
  (( complete_count >= REQUESTS )) && break
  (( SECONDS < deadline )) || die "timed out waiting for Hop1 asynchronous pressure completion"
  sleep 1
done

python3 - "$client_log" "$HOP1_LOG" "$summary" \
  "$TRANSPORT" "$REQUESTS" <<'PY'
import json, pathlib, sys

client_path, hop1_path, output_path, transport, expected_text = sys.argv[1:]
expected = int(expected_text)

def events(path):
    result = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            item = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        result.append(item)
    return result

client = events(client_path)
hop1 = events(hop1_path)
ready = [item for item in hop1 if item.get("event") == "pairec_post_rank_hop2_brpc_burst_ready"]
assert ready, "Hop1 c1000 ready event is missing"
assert any(
    item.get("transport") == transport
    and item.get("connected_sessions") == 1000
    and item.get("armed_workers") == 1000
    and item.get("payload_bytes") == 102400
    for item in ready
), "Hop1 does not have a matching c1000/100KiB ready event"
requests = [item for item in client if item.get("event") == "post_rank_qualification_request"]
assert len(requests) == expected, (len(requests), expected)
ids = {item["request_id"] for item in requests}
assert len(ids) == expected and all(item.get("valid") is True for item in requests)
assert all(item.get("payload_bytes") == 102400 and item.get("candidate_count") == 50 for item in requests)

def indexed(source, name):
    items = [item for item in source if item.get("event") == name and item.get("request_id") in ids]
    assert len(items) == expected, (name, len(items), expected)
    return {item["request_id"]: item for item in items}

business = indexed(hop1, "pairec_post_rank_hop2_brpc_burst_business_complete")
complete = indexed(hop1, "pairec_post_rank_hop2_brpc_burst_complete")
for request_id in ids:
    b = business[request_id]
    c = complete[request_id]
    assert b.get("transport") == transport and c.get("transport") == transport
    assert b.get("concurrency") == 1000 and c.get("concurrency") == 1000
    assert b.get("business_success") is True
    assert b.get("pressure_started_at_business_start", 0) >= 950
    assert c.get("pressure_requests") == 999 and c.get("pressure_success") == 999
    assert c.get("pressure_errors") == 0 and c.get("burst_valid") is True
    assert c.get("pressure_overlap_business", 0) > 0

all_text = pathlib.Path(hop1_path).read_text(errors="replace")
assert "bthread_setspecific is called on invalid bthread_key_t" not in all_text
ub_evidence = None
if transport == "ub":
    assert "PAIREC_POST_RANK_UB_TRACE_KEYS_READY" in all_text, "UB trace keys were not initialized"
    assert "PAIREC_POST_RANK_UB_GLOBAL_ENABLED" in all_text, "global ubsocket_enable was not enabled"
    assert "Use Bonding:" in all_text, "UB Bonding data-plane initialization evidence is missing"
    assert "bind jetty success" in all_text, "UB peer binding evidence is missing"
    ub_evidence = {
        "trace_keys_ready": True,
        "global_enabled": True,
        "bonding": True,
        "bind_jetty": True,
    }

result = {
    "classification": "PAIREC_POST_RANK_UB_C1000_100K_OK" if transport == "ub" else "PAIREC_POST_RANK_TCP_C1000_100K_OK",
    "transport": transport,
    "requests": expected,
    "concurrency": 1000,
    "business_requests": 1,
    "pressure_requests": 999,
    "payload_bytes_per_lane": 102400,
    "ub_evidence": ub_evidence,
    "samples": requests,
}
pathlib.Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
print(f"classification={result['classification']}")
print(f"requests={expected} concurrency=1000 payload_bytes=102400")
PY

echo "evidence=$summary"
