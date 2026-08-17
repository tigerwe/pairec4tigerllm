#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
KVC_CONCURRENCY=${KVC_CONCURRENCY:-10}
KVC_PRESSURE_KEY_COUNT=${KVC_PRESSURE_KEY_COUNT:-9}
REQUESTS=${REQUESTS:-3}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
BURST_POOL_SIZE=${BURST_POOL_SIZE:-10000}
BURST_ACTIVE_CONNECTIONS=${BURST_ACTIVE_CONNECTIONS:-1}
BURST_PAYLOAD_BYTES=${BURST_PAYLOAD_BYTES:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-kvc-combined/$(date +%Y%m%d-%H%M%S)-c${KVC_CONCURRENCY}-n${REQUESTS}}
KVC_OVERLAY_BACKUP=${KVC_OVERLAY_BACKUP:-$OUTPUT_DIR/kvc-deployment-before.json}
WRAPPER_OUTPUT_DIR=${WRAPPER_OUTPUT_DIR:-$OUTPUT_DIR/wrapper}
KVC_LOG=${KVC_LOG:-$OUTPUT_DIR/kvc-burst.log}
KVC_LOG_WAIT_SECONDS=${KVC_LOG_WAIT_SECONDS:-45}

die() { echo "ERROR: $*" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
[[ "$KVC_CONCURRENCY" =~ ^(1|10|100)$ ]] || die "KVC_CONCURRENCY must be 1, 10, or 100"
[[ "$KVC_PRESSURE_KEY_COUNT" =~ ^[0-9]+$ ]] || die "KVC_PRESSURE_KEY_COUNT must be non-negative"
(( KVC_PRESSURE_KEY_COUNT <= KVC_CONCURRENCY - 1 )) \
  || die "KVC_PRESSURE_KEY_COUNT must not exceed KVC pressure lanes"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "WARMUP_REQUESTS must be non-negative"

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
  PRESSURE_KEY_COUNT="$KVC_PRESSURE_KEY_COUNT" KVC_BURST_ENABLED=1 \
  KVC_BURST_VERBOSE=1 KVC_BURST_INITIAL_ARMED=1 \
  bash scripts/deploy_f14_kvc_burst_overlay.sh apply \
  | tee "$OUTPUT_DIR/kvc-overlay.log"

echo "== Run preconnected BRPC Wrapper full chain =="
set +e
NAMESPACE="$NAMESPACE" REQUESTS="$REQUESTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  BURST_CONCURRENCY=1 BURST_POOL_SIZE="$BURST_POOL_SIZE" \
  BURST_ACTIVE_CONNECTIONS="$BURST_ACTIVE_CONNECTIONS" \
  BURST_PAYLOAD_BYTES="$BURST_PAYLOAD_BYTES" \
  BUILD_PAIREC_IMAGE=0 IMPORT_PAIREC_IMAGE=0 \
  OUTPUT_DIR="$WRAPPER_OUTPUT_DIR" \
  bash scripts/deploy_and_validate_pairec_brpc_wrapper_full.sh \
  | tee "$OUTPUT_DIR/wrapper-console.log"
wrapper_code=${PIPESTATUS[0]}
set -e
[[ "$wrapper_code" -eq 0 ]] || die "BRPC Wrapper full-chain validation failed: exit=$wrapper_code"

INFERENCE_POD="$(kubectl -n "$NAMESPACE" get pod -l app=inference-brpc-trtllm \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
[[ -n "$INFERENCE_POD" ]] || die "inference Pod not found"

deadline=$((SECONDS + KVC_LOG_WAIT_SECONDS))
while true; do
  kubectl -n "$NAMESPACE" logs "$INFERENCE_POD" -c kvc-burst-wrapper \
    --since-time="$(date --iso-8601=seconds -d "-$((KVC_LOG_WAIT_SECONDS + 60)) seconds")" \
    >"$KVC_LOG" 2>&1 || true
  if python3 - "$WRAPPER_OUTPUT_DIR/requests.tsv" "$KVC_LOG" <<'PY'
import csv, json, pathlib, sys
ids = {row["request_id"] for row in csv.DictReader(open(sys.argv[1]), delimiter="\t")}
events = []
for line in pathlib.Path(sys.argv[2]).read_text(errors="replace").splitlines():
    pos = line.find("{")
    if pos < 0:
        continue
    try:
        item = json.loads(line[pos:])
    except json.JSONDecodeError:
        continue
    if item.get("event") == "kvc_burst_complete":
        events.append(item)
by_id = {}
for item in events:
    by_id.setdefault(item.get("request_id"), []).append(item)
if all(len(by_id.get(request_id, [])) == 1 for request_id in ids):
    raise SystemExit(0)
raise SystemExit(1)
PY
  then
    break
  fi
  (( SECONDS < deadline )) || die "KVC completion did not arrive for every Wrapper request"
  sleep 1
done

python3 - "$WRAPPER_OUTPUT_DIR/requests.tsv" "$WRAPPER_OUTPUT_DIR/summary.json" "$KVC_LOG" "$OUTPUT_DIR/summary.json" "$KVC_CONCURRENCY" "$KVC_PRESSURE_KEY_COUNT" <<'PY'
import csv, json, pathlib, sys
requests_path, wrapper_path, kvc_path, output_path = sys.argv[1:5]
expected_concurrency = int(sys.argv[5])
expected_keys = int(sys.argv[6])
requests = list(csv.DictReader(open(requests_path), delimiter="\t"))
wrapper = json.load(open(wrapper_path))
events = []
for line in pathlib.Path(kvc_path).read_text(errors="replace").splitlines():
    pos = line.find("{")
    if pos < 0:
        continue
    try:
        item = json.loads(line[pos:])
    except json.JSONDecodeError:
        continue
    if item.get("event") == "kvc_burst_complete":
        events.append(item)
by_id = {}
for item in events:
    by_id.setdefault(item.get("request_id"), []).append(item)
rows = []
for row in requests:
    request_id = row["request_id"]
    matches = by_id.get(request_id, [])
    assert len(matches) == 1, (request_id, matches)
    kvc = matches[0]
    assert kvc["concurrency"] == expected_concurrency, kvc
    assert kvc["pressure_key_count"] == expected_keys, kvc
    assert kvc["valid"] is True and kvc["failure"] == 0, kvc
    assert kvc["pressure_success"] == expected_concurrency - 1, kvc
    assert kvc["pressure_errors"] == 0, kvc
    rows.append({"request_id": request_id, "e2e_ms": float(row["e2e_ms"]), "kvc": kvc})
result = {
    "classification": f"PAIREC_BRPC_WRAPPER_KVC_COMBINED_C{expected_concurrency}_OK",
    "requests": len(rows),
    "wrapper_summary": wrapper,
    "kvc_samples": rows,
}
pathlib.Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
print(f"classification={result['classification']}")
print(f"requests={len(rows)}")
for row in rows:
    kvc = row["kvc"]
    print("request", row["request_id"], "e2e_ms=%.3f" % row["e2e_ms"],
          "business_get_ms=%.3f" % kvc["business_get_ms"],
          "pressure_p99_ms=%.3f" % kvc["pressure_get_p99_ms"],
          "barrier_ms=%.3f" % kvc["barrier_wait_ms"])
print(f"summary_json={output_path}")
print(result["classification"])
PY

echo "output_dir=$OUTPUT_DIR"
