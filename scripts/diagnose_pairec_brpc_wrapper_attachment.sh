#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
WRAPPER_DEPLOYMENT="${WRAPPER_DEPLOYMENT:-brpc-burst-wrapper}"
WRAPPER_ENDPOINT="${WRAPPER_ENDPOINT:-192.168.100.11:18103}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"
PAIREC_IMAGE="${PAIREC_IMAGE:-docker.io/library/pairec-server:k8s-arm64-brpc-v1}"
WRAPPER_BUILD_IMAGE="${WRAPPER_BUILD_IMAGE:-docker.io/library/pairec-brpc-gateway:k8s-arm64-attachment-v1}"
BUILD_COMPONENTS="${BUILD_COMPONENTS:-1}"
REQUESTS="${REQUESTS:-100}"
QUALIFICATION_REQUESTS="${QUALIFICATION_REQUESTS:-10}"
USER_ID="${USER_ID:-6312}"
DETERMINISTIC_TRT_TOP_K="${DETERMINISTIC_TRT_TOP_K:-1}"
MAX_RUNNER_P99_DELTA_MS="${MAX_RUNNER_P99_DELTA_MS:-10}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-attachment/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}"
WORKER_SSH="${WORKER_SSH:-root@192.168.100.11}"
WORKER_PAUSE_IMAGE="${WORKER_PAUSE_IMAGE:-docker.io/library/pause-aarch64:3.8}"
WORKER_PAUSE_ARCHIVE="${WORKER_PAUSE_ARCHIVE:-/home/zcx/pause-aarch64-3.8.tar}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$BUILD_COMPONENTS" = 0 || "$BUILD_COMPONENTS" = 1 ]] || die "BUILD_COMPONENTS must be 0 or 1"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$QUALIFICATION_REQUESTS" =~ ^[0-9]+$ ]] || die "QUALIFICATION_REQUESTS must be non-negative"
[[ "$DETERMINISTIC_TRT_TOP_K" = 1 ]] || die "DETERMINISTIC_TRT_TOP_K must be 1"
for command in ctr docker kubectl python3 ssh; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
mkdir -p "$OUTPUT_DIR"

ctr_k8s() {
  if (( EUID == 0 )); then ctr -n k8s.io "$@"; else sudo -n ctr -n k8s.io "$@"; fi
}

ensure_worker_pause_image() {
  echo "== Ensure worker1 Kubernetes sandbox image =="
  ssh "$WORKER_SSH" bash -s -- "$WORKER_PAUSE_IMAGE" "$WORKER_PAUSE_ARCHIVE" <<'REMOTE'
set -euo pipefail
image="$1"; archive="$2"
ctr_k8s() {
  if (( EUID == 0 )); then ctr -n k8s.io "$@"; else sudo -n ctr -n k8s.io "$@"; fi
}
if ! ctr_k8s images list | awk 'NR > 1 {print $1}' | grep -Fxq "$image"; then
  test -f "$archive"
  ctr_k8s images import "$archive" >/dev/null
fi
ctr_k8s images list | awk 'NR > 1 {print $1}' | grep -Fxq "$image"
echo "K8S_WORKER_SANDBOX_READY image=$image"
REMOTE
}

if [[ "$BUILD_COMPONENTS" = 1 ]]; then
  echo "== Build and import PaiRec attachment client =="
  bash scripts/build_pairec_binary_image.sh "$PAIREC_IMAGE"
  docker save "$PAIREC_IMAGE" | ctr_k8s images import -

  echo "== Build and distribute attachment-aware Wrapper =="
  bash scripts/build_brpc_gateway_image.sh "$WRAPPER_BUILD_IMAGE"
  IMAGE="$WRAPPER_BUILD_IMAGE" \
    bash scripts/ship_brpc_burst_wrapper_binary_to_worker.sh
  ensure_worker_pause_image
  kubectl -n "$NAMESPACE" rollout restart "deployment/$WRAPPER_DEPLOYMENT"
  kubectl -n "$NAMESPACE" rollout status "deployment/$WRAPPER_DEPLOYMENT" \
    --timeout="$ROLLOUT_TIMEOUT"
fi

DEPLOYMENT_BEFORE="$OUTPUT_DIR/inference-deployment-before.json"
DETERMINISTIC_PATCH="$OUTPUT_DIR/inference-top-k-deterministic-patch.json"
RESTORE_PATCH="$OUTPUT_DIR/inference-top-k-restore-patch.json"
kubectl -n "$NAMESPACE" get deployment "$INFERENCE_DEPLOYMENT" -o json >"$DEPLOYMENT_BEFORE"
python3 - "$DEPLOYMENT_BEFORE" "$INFERENCE_CONTAINER" "$DETERMINISTIC_TRT_TOP_K" \
  "$DETERMINISTIC_PATCH" "$RESTORE_PATCH" <<'PY'
import json,pathlib,sys
source,container_name,top_k,deterministic_path,restore_path=sys.argv[1:]
deployment=json.load(open(source)); containers=deployment["spec"]["template"]["spec"]["containers"]
matches=[(i,c) for i,c in enumerate(containers) if c["name"]==container_name]
assert len(matches)==1,matches
index,container=matches[0]; original=list(container.get("args",[]))
positions=[i for i,arg in enumerate(original) if arg.startswith("--trt_top_k=")]
assert len(positions)==1,positions
deterministic=list(original); deterministic[positions[0]]=f"--trt_top_k={top_k}"
path=f"/spec/template/spec/containers/{index}/args"; changed=deterministic!=original
pathlib.Path(deterministic_path).write_text(json.dumps([{"op":"replace","path":path,"value":deterministic}] if changed else [])+"\n")
pathlib.Path(restore_path).write_text(json.dumps([{"op":"replace","path":path,"value":original}] if changed else [])+"\n")
print(f"original_trt_top_k={original[positions[0]].split('=',1)[1]} diagnostic_trt_top_k={top_k} patch_required={str(changed).lower()}")
PY

DETERMINISTIC_APPLIED=0
cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT INT TERM
  if [[ "$DETERMINISTIC_APPLIED" = 1 ]]; then
    echo "== Restore original TRT sampling configuration =="
    kubectl -n "$NAMESPACE" patch deployment "$INFERENCE_DEPLOYMENT" --type=json \
      -p "$(cat "$RESTORE_PATCH")" >"$OUTPUT_DIR/inference-config-restore.log" 2>&1 \
      || cleanup_failed=1
    if [[ "$cleanup_failed" = 0 ]]; then
      kubectl -n "$NAMESPACE" rollout status "deployment/$INFERENCE_DEPLOYMENT" \
        --timeout="$ROLLOUT_TIMEOUT" >>"$OUTPUT_DIR/inference-config-restore.log" 2>&1 \
        || cleanup_failed=1
    fi
  fi
  if [[ "$cleanup_failed" = 1 ]]; then
    echo "ERROR: TRT configuration restore failed: $OUTPUT_DIR/inference-config-restore.log" >&2
    [[ "$status" != 0 ]] || status=1
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$(cat "$DETERMINISTIC_PATCH")" != "[]" ]]; then
  kubectl -n "$NAMESPACE" patch deployment "$INFERENCE_DEPLOYMENT" --type=json \
    -p "$(cat "$DETERMINISTIC_PATCH")"
  DETERMINISTIC_APPLIED=1
  kubectl -n "$NAMESPACE" rollout status "deployment/$INFERENCE_DEPLOYMENT" \
    --timeout="$ROLLOUT_TIMEOUT"
fi

run_transport() {
  local transport="$1"
  echo "== Payload transport: $transport =="
  NAMESPACE="$NAMESPACE" \
  WRAPPER_DEPLOYMENT="$WRAPPER_DEPLOYMENT" WRAPPER_ENDPOINT="$WRAPPER_ENDPOINT" \
  INFERENCE_DEPLOYMENT="$INFERENCE_DEPLOYMENT" \
  BURST_PAYLOAD_TRANSPORT="$transport" \
  REQUESTS="$REQUESTS" QUALIFICATION_REQUESTS="$QUALIFICATION_REQUESTS" USER_ID="$USER_ID" \
  MAX_RUNNER_P99_DELTA_MS="$MAX_RUNNER_P99_DELTA_MS" \
  BUILD_PAIREC_IMAGE=0 IMPORT_PAIREC_IMAGE=0 \
  OUTPUT_DIR="$OUTPUT_DIR/$transport" \
    bash scripts/diagnose_pairec_brpc_wrapper_runner_interference.sh \
    | tee "$OUTPUT_DIR/$transport.log"
}

run_transport protobuf
run_transport attachment

python3 - "$OUTPUT_DIR/protobuf/summary.json" "$OUTPUT_DIR/attachment/summary.json" \
  "$OUTPUT_DIR/comparison.json" "$MAX_RUNNER_P99_DELTA_MS" <<'PY'
import json,pathlib,sys
protobuf=json.load(open(sys.argv[1])); attachment=json.load(open(sys.argv[2])); limit=float(sys.argv[4])
assert protobuf["payload_transport"]=="protobuf",protobuf
assert attachment["payload_transport"]=="attachment",attachment
p_total=float(protobuf["runner_p99_total_delta_ms"]); a_total=float(attachment["runner_p99_total_delta_ms"])
p_payload=float(protobuf["runner_p99_payload_increment_ms"]); a_payload=float(attachment["runner_p99_payload_increment_ms"])
result={
 "classification":"PAIREC_BRPC_WRAPPER_ATTACHMENT_COMPARISON",
 "protobuf_runner_p99_total_delta_ms":p_total,
 "attachment_runner_p99_total_delta_ms":a_total,
 "total_delta_improvement_ms":p_total-a_total,
 "protobuf_runner_p99_payload_increment_ms":p_payload,
 "attachment_runner_p99_payload_increment_ms":a_payload,
 "payload_increment_improvement_ms":p_payload-a_payload,
 "runner_p99_limit_ms":limit,
 "attachment_runner_gate_passed":a_total<=limit,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False))
PY

echo "summary_json=$OUTPUT_DIR/comparison.json"
echo "PAIREC_BRPC_WRAPPER_ATTACHMENT_DIAGNOSIS_COMPLETE"
