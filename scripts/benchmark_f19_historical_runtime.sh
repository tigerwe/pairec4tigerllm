#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-inference-brpc-trtllm}"
CONTAINER="${CONTAINER:-brpc-inference}"
APP_LABEL="${APP_LABEL:-app=inference-brpc-trtllm}"
PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec-brpc-observed}"
HISTORICAL_RUNTIME_DIR="${HISTORICAL_RUNTIME_DIR:-/home/zcx/pairec-f19-historical-runtime}"
HISTORICAL_POD_DIR="${HISTORICAL_POD_DIR:-/opt/pairec-f19-historical}"
TRTLLM_RUNTIME_PATH="${TRTLLM_RUNTIME_PATH:-/TensorRT-LLM/cpp/build/tensorrt_llm/libtensorrt_llm.so}"
SAMPLES="${SAMPLES:-5}"
USER_ID="${USER_ID:-6312}"
SIZE="${SIZE:-1}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/f19-historical-runtime-ab/$RUN_ID}"

ORIGINAL_JSON="$OUTPUT_DIR/deployment-original.json"
RESTORE_PATCH="$OUTPUT_DIR/restore-patch.json"
HISTORICAL_PATCH="$OUTPUT_DIR/historical-patch.json"
DEPLOYMENT_CHANGED=0

die() { echo "ERROR: $*" >&2; exit 1; }

current_pod() {
  kubectl -n "$NAMESPACE" get pod -l "$APP_LABEL" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}'
}

wait_for_rollout() {
  kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" \
    --timeout="$ROLLOUT_TIMEOUT"
}

restore_current_runtime() {
  if [[ "$DEPLOYMENT_CHANGED" != 1 ]]; then return; fi
  echo "== Restore current F19 runtime =="
  kubectl -n "$NAMESPACE" patch deployment "$DEPLOYMENT" \
    --type=merge --patch "$(cat "$RESTORE_PATCH")"
  wait_for_rollout
  DEPLOYMENT_CHANGED=0
}

cleanup() {
  status=$?
  set +e
  restore_current_runtime
  restore_status=$?
  set -e
  if (( status == 0 && restore_status != 0 )); then status=$restore_status; fi
  exit "$status"
}
trap cleanup EXIT

run_samples() {
  local mode="$1"
  local require_exact="$2"
  local mode_dir="$OUTPUT_DIR/$mode"
  mkdir -p "$mode_dir"
  for index in $(seq 1 "$SAMPLES"); do
    echo "== $mode sample $index/$SAMPLES =="
    OUT_DIR="$mode_dir/sample-$index" \
    USER_ID="$USER_ID" \
    SIZE="$SIZE" \
    PAIREC_TARGET="$PAIREC_TARGET" \
    BRPC_TARGET="deployment/$DEPLOYMENT" \
    REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION="$require_exact" \
      bash scripts/trace_single_brpc_datasystem_request.sh
  done
}

summarize() {
  python3 - "$OUTPUT_DIR" "$SAMPLES" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
samples = int(sys.argv[2])
result = {"samples_per_mode": samples, "modes": {}}

def number(fields, key):
    try:
        return float(fields.get(key, 0))
    except (TypeError, ValueError):
        return 0.0

for mode in ("historical", "current"):
    rows = []
    for path in sorted((root / mode).glob("sample-*/summary.json")):
        payload = json.loads(path.read_text())
        trace = payload.get("pairec_generative_trace") or {}
        diagnosis = payload.get("trt_latency_diagnosis") or {}
        rows.append({
            "request_id": payload.get("request_id"),
            "client_ms": number(payload.get("client") or {}, "client_e2e_ms"),
            "runner_ms": number(trace, "tr_runner_ms"),
            "output_tokens": number(trace, "tr_output_tokens")
                or number(diagnosis, "output_token_count")
                or number(payload, "legacy_native_output_token_count"),
            "runner_per_token_ms": number(trace, "tr_runner_per_token_ms")
                or number(diagnosis, "runner_per_output_token_ms"),
            "prefill_ms": number(diagnosis, "prefill_gap_us") / 1000.0,
            "decode_ms": number(diagnosis, "decode_gap_us") / 1000.0,
        })
        if rows[-1]["runner_per_token_ms"] <= 0 and rows[-1]["output_tokens"] > 0:
            rows[-1]["runner_per_token_ms"] = round(
                rows[-1]["runner_ms"] / rows[-1]["output_tokens"], 6)
    if len(rows) != samples:
        raise SystemExit(f"{mode}: expected {samples} summaries, got {len(rows)}")
    missing_tokens = [row["request_id"] for row in rows if row["output_tokens"] <= 0]
    if missing_tokens:
        raise SystemExit(
            f"{mode}: output token count missing for requests: "
            + ",".join(str(value) for value in missing_tokens)
        )
    warm = rows[1:] if len(rows) > 1 else rows
    result["modes"][mode] = {
        "rows": rows,
        "cold": rows[0],
        "warm_avg": {
            key: round(statistics.mean(row[key] for row in warm), 6)
            for key in ("client_ms", "runner_ms", "output_tokens",
                        "runner_per_token_ms", "prefill_ms", "decode_ms")
        },
    }

historical = result["modes"]["historical"]["warm_avg"]
current = result["modes"]["current"]["warm_avg"]
result["warm_current_minus_historical"] = {
    key: round(current[key] - historical[key], 6)
    for key in historical
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

print("mode warm_runner_ms warm_output_tokens warm_runner_per_token_ms warm_prefill_ms warm_decode_ms")
for mode in ("historical", "current"):
    row = result["modes"][mode]["warm_avg"]
    print(mode, row["runner_ms"], row["output_tokens"],
          row["runner_per_token_ms"], row["prefill_ms"], row["decode_ms"])
print("delta", json.dumps(result["warm_current_minus_historical"], sort_keys=True))
print(f"summary_json={root / 'summary.json'}")
PY
}

for command in kubectl python3 sha256sum grep seq; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
[[ "$SAMPLES" =~ ^[0-9]+$ ]] && (( SAMPLES >= 2 )) \
  || die "SAMPLES must be an integer >= 2"
mkdir -p "$OUTPUT_DIR"
kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json >"$ORIGINAL_JSON"

python3 - "$ORIGINAL_JSON" "$RESTORE_PATCH" "$HISTORICAL_PATCH" \
    "$CONTAINER" "$HISTORICAL_RUNTIME_DIR" "$HISTORICAL_POD_DIR" \
    "$TRTLLM_RUNTIME_PATH" <<'PY'
import json
import sys
from pathlib import Path

source, restore_out, historical_out, container_name, host_dir, pod_dir, trt_path = sys.argv[1:]
deployment = json.loads(Path(source).read_text())
template = deployment["spec"]["template"]
Path(restore_out).write_text(json.dumps({"spec": {"template": template}}, indent=2) + "\n")

container = next(c for c in template["spec"]["containers"] if c["name"] == container_name)
env = {item["name"]: item for item in container.get("env", [])}
env["TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION"] = {
    "name": "TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION", "value": "0"}
env["PAIREC_REQUIRE_NATIVE_DATASYSTEM_ATTRIBUTION"] = {
    "name": "PAIREC_REQUIRE_NATIVE_DATASYSTEM_ATTRIBUTION", "value": "0"}
env["TLLM_LOG_LEVEL"] = {"name": "TLLM_LOG_LEVEL", "value": "DEBUG"}

f19_names = {"f19-runtime-bin", "f19-runtime-lib", "f19-trtllm-file",
             "f19-historical-bin", "f19-historical-trt"}
container["command"] = [f"{pod_dir}/bin/brpc_inference_server"]
container["env"] = list(env.values())
container["volumeMounts"] = [
    mount for mount in container.get("volumeMounts", [])
    if mount["name"] not in f19_names
] + [
    {"name": "f19-historical-bin", "mountPath": f"{pod_dir}/bin", "readOnly": True},
    {"name": "f19-historical-trt", "mountPath": trt_path, "readOnly": True},
]
template["spec"]["volumes"] = [
    volume for volume in template["spec"].get("volumes", [])
    if volume["name"] not in f19_names
] + [
    {"name": "f19-historical-bin", "hostPath": {
        "path": f"{host_dir}/bin", "type": "Directory"}},
    {"name": "f19-historical-trt", "hostPath": {
        "path": f"{host_dir}/lib/libtensorrt_llm.so", "type": "File"}},
]
patch = {"spec": {"template": template}}
Path(historical_out).write_text(json.dumps(patch, indent=2) + "\n")
PY

echo "== Historical runtime files on worker1 hostPath =="
POD="$(current_pod)"
NODE="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.nodeName}')"
echo "current_pod=$POD node=$NODE historical_runtime=$HISTORICAL_RUNTIME_DIR"
[[ "$NODE" == "worker1" ]] || die "inference Pod must run on worker1, got $NODE"

echo "== Deploy historical gateway and TRT library as a pair =="
kubectl -n "$NAMESPACE" patch deployment "$DEPLOYMENT" \
  --type=merge --patch "$(cat "$HISTORICAL_PATCH")"
DEPLOYMENT_CHANGED=1
wait_for_rollout
run_samples historical 0

restore_current_runtime
run_samples current 1
summarize
echo "F19_HISTORICAL_RUNTIME_AB_OK output_dir=$OUTPUT_DIR"
