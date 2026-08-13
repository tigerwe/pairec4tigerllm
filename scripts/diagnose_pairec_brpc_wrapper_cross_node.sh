#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"
COLOCATED_WRAPPER_DEPLOYMENT="${COLOCATED_WRAPPER_DEPLOYMENT:-brpc-burst-wrapper}"
COLOCATED_WRAPPER_ENDPOINT="${COLOCATED_WRAPPER_ENDPOINT:-192.168.100.11:18103}"
CROSS_NODE_WRAPPER_DEPLOYMENT="${CROSS_NODE_WRAPPER_DEPLOYMENT:-brpc-burst-wrapper-master}"
CROSS_NODE_WRAPPER_ENDPOINT="${CROSS_NODE_WRAPPER_ENDPOINT:-192.168.100.12:18104}"
DETERMINISTIC_TRT_TOP_K="${DETERMINISTIC_TRT_TOP_K:-1}"
REQUESTS="${REQUESTS:-100}"
QUALIFICATION_REQUESTS="${QUALIFICATION_REQUESTS:-10}"
USER_ID="${USER_ID:-6312}"
MAX_RUNNER_P99_DELTA_MS="${MAX_RUNNER_P99_DELTA_MS:-10}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-cross-node/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}"
WORKER_SSH="${WORKER_SSH:-root@192.168.100.11}"
WORKER_PAUSE_IMAGE="${WORKER_PAUSE_IMAGE:-docker.io/library/pause-aarch64:3.8}"
WORKER_PAUSE_ARCHIVE="${WORKER_PAUSE_ARCHIVE:-/home/zcx/pause-aarch64-3.8.tar}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$QUALIFICATION_REQUESTS" =~ ^[0-9]+$ ]] || die "QUALIFICATION_REQUESTS must be non-negative"
[[ "$DETERMINISTIC_TRT_TOP_K" = 1 ]] || die "DETERMINISTIC_TRT_TOP_K must be 1"
for command in kubectl python3 ssh; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
mkdir -p "$OUTPUT_DIR"

ready_pod() {
  local app="$1"
  kubectl -n "$NAMESPACE" get pods -l "app=$app" -o json | python3 -c '
import json,sys
pods=[]
for pod in json.load(sys.stdin).get("items",[]):
 status=pod.get("status",{}); containers=status.get("containerStatuses",[])
 if (not pod["metadata"].get("deletionTimestamp") and status.get("phase")=="Running"
     and containers and all(item.get("ready") for item in containers)):
  pods.append((pod["metadata"].get("creationTimestamp",""),pod["metadata"]["name"]))
assert pods,f"no ready pod for {sys.argv[1]}"
print(max(pods)[1])
' "$app"
}

if kubectl -n "$NAMESPACE" get deployment "$CROSS_NODE_WRAPPER_DEPLOYMENT" >/dev/null 2>&1; then
  die "diagnostic deployment already exists: $CROSS_NODE_WRAPPER_DEPLOYMENT"
fi
DETERMINISTIC_APPLIED=0
RESTORE_PATCH="$OUTPUT_DIR/inference-top-k-restore-patch.json"
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
  echo "== Remove diagnostic cross-node Wrapper =="
  kubectl -n "$NAMESPACE" delete deployment "$CROSS_NODE_WRAPPER_DEPLOYMENT" \
    --ignore-not-found >"$OUTPUT_DIR/master-wrapper-cleanup.log" 2>&1 \
    || cleanup_failed=1
  if [[ "$cleanup_failed" = 1 ]]; then
    echo "ERROR: diagnostic cleanup failed; inspect $OUTPUT_DIR/*cleanup.log and *restore.log" >&2
    [[ "$status" != 0 ]] || status=1
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

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

echo "== Deploy cross-node Wrapper on master =="
NAMESPACE="$NAMESPACE" DEPLOYMENT="$CROSS_NODE_WRAPPER_DEPLOYMENT" \
WRAPPER_ENDPOINT="$CROSS_NODE_WRAPPER_ENDPOINT" \
  bash scripts/k8s_apply_brpc_burst_wrapper_master.sh \
  | tee "$OUTPUT_DIR/master-wrapper-deploy.log"

INFERENCE_POD="$(ready_pod "$INFERENCE_DEPLOYMENT")"
COLOCATED_POD="$(ready_pod "$COLOCATED_WRAPPER_DEPLOYMENT")"
CROSS_NODE_POD="$(ready_pod "$CROSS_NODE_WRAPPER_DEPLOYMENT")"
INFERENCE_NODE="$(kubectl -n "$NAMESPACE" get pod "$INFERENCE_POD" -o jsonpath='{.spec.nodeName}')"
COLOCATED_NODE="$(kubectl -n "$NAMESPACE" get pod "$COLOCATED_POD" -o jsonpath='{.spec.nodeName}')"
CROSS_NODE="$(kubectl -n "$NAMESPACE" get pod "$CROSS_NODE_POD" -o jsonpath='{.spec.nodeName}')"
[[ "$COLOCATED_NODE" = "$INFERENCE_NODE" ]] \
  || die "baseline Wrapper is not colocated with inference: $COLOCATED_NODE != $INFERENCE_NODE"
[[ "$CROSS_NODE" != "$INFERENCE_NODE" ]] \
  || die "cross-node Wrapper is still colocated with inference: $CROSS_NODE"
[[ "$CROSS_NODE" = master ]] || die "cross-node Wrapper is not on master: $CROSS_NODE"
echo "BRPC_WRAPPER_CROSS_NODE_TOPOLOGY_OK inference=$INFERENCE_NODE colocated=$COLOCATED_NODE cross_node=$CROSS_NODE"
COLOCATED_WRAPPER_SHA="$(kubectl -n "$NAMESPACE" exec "$COLOCATED_POD" -c brpc-burst-wrapper -- \
  sha256sum /opt/pairec-brpc/bin/brpc_burst_wrapper | awk '{print $1}')"
CROSS_NODE_WRAPPER_SHA="$(kubectl -n "$NAMESPACE" exec "$CROSS_NODE_POD" -c brpc-burst-wrapper -- \
  sha256sum /opt/pairec-brpc/bin/brpc_burst_wrapper | awk '{print $1}')"
[[ -n "$COLOCATED_WRAPPER_SHA" && "$COLOCATED_WRAPPER_SHA" = "$CROSS_NODE_WRAPPER_SHA" ]] \
  || die "Wrapper binary mismatch: colocated=$COLOCATED_WRAPPER_SHA cross_node=$CROSS_NODE_WRAPPER_SHA"
echo "BRPC_WRAPPER_BINARY_MATCH_OK sha256=$COLOCATED_WRAPPER_SHA"

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
path=f"/spec/template/spec/containers/{index}/args"
changed=deterministic!=original
pathlib.Path(deterministic_path).write_text(json.dumps([{"op":"replace","path":path,"value":deterministic}] if changed else [])+"\n")
pathlib.Path(restore_path).write_text(json.dumps([{"op":"replace","path":path,"value":original}] if changed else [])+"\n")
print(f"original_trt_top_k={original[positions[0]].split('=',1)[1]} diagnostic_trt_top_k={top_k} patch_required={str(changed).lower()}")
PY

if [[ "$(cat "$DETERMINISTIC_PATCH")" != "[]" ]]; then
  echo "== Apply deterministic TRT sampling =="
  kubectl -n "$NAMESPACE" patch deployment "$INFERENCE_DEPLOYMENT" --type=json \
    -p "$(cat "$DETERMINISTIC_PATCH")"
  DETERMINISTIC_APPLIED=1
  kubectl -n "$NAMESPACE" rollout status "deployment/$INFERENCE_DEPLOYMENT" \
    --timeout="$ROLLOUT_TIMEOUT"
fi

run_topology() {
  local name="$1" deployment="$2" endpoint="$3"
  echo "== Topology: $name deployment=$deployment endpoint=$endpoint =="
  NAMESPACE="$NAMESPACE" \
  INFERENCE_DEPLOYMENT="$INFERENCE_DEPLOYMENT" \
  WRAPPER_DEPLOYMENT="$deployment" \
  WRAPPER_ENDPOINT="$endpoint" \
  REQUESTS="$REQUESTS" \
  QUALIFICATION_REQUESTS="$QUALIFICATION_REQUESTS" \
  USER_ID="$USER_ID" \
  MAX_RUNNER_P99_DELTA_MS="$MAX_RUNNER_P99_DELTA_MS" \
  BUILD_PAIREC_IMAGE=0 IMPORT_PAIREC_IMAGE=0 \
  OUTPUT_DIR="$OUTPUT_DIR/$name" \
    bash scripts/diagnose_pairec_brpc_wrapper_runner_interference.sh \
    | tee "$OUTPUT_DIR/$name.log"
}

run_topology colocated "$COLOCATED_WRAPPER_DEPLOYMENT" "$COLOCATED_WRAPPER_ENDPOINT"
run_topology cross_node "$CROSS_NODE_WRAPPER_DEPLOYMENT" "$CROSS_NODE_WRAPPER_ENDPOINT"

python3 - "$OUTPUT_DIR/colocated/summary.json" "$OUTPUT_DIR/cross_node/summary.json" \
  "$OUTPUT_DIR/comparison.json" "$MAX_RUNNER_P99_DELTA_MS" <<'PY'
import json,pathlib,sys
colocated=json.load(open(sys.argv[1])); cross=json.load(open(sys.argv[2])); limit=float(sys.argv[4])
col_delta=float(colocated["runner_p99_total_delta_ms"]); cross_delta=float(cross["runner_p99_total_delta_ms"])
result={
 "classification":"PAIREC_BRPC_WRAPPER_CROSS_NODE_COMPARISON",
 "colocated_runner_p99_total_delta_ms":col_delta,
 "cross_node_runner_p99_total_delta_ms":cross_delta,
 "improvement_ms":col_delta-cross_delta,
 "runner_p99_limit_ms":limit,
 "cross_node_runner_gate_passed":cross_delta<=limit,
 "colocated_runner_p99_payload_increment_ms":float(colocated["runner_p99_payload_increment_ms"]),
 "cross_node_runner_p99_payload_increment_ms":float(cross["runner_p99_payload_increment_ms"]),
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False))
PY

echo "summary_json=$OUTPUT_DIR/comparison.json"
echo "PAIREC_BRPC_WRAPPER_CROSS_NODE_DIAGNOSIS_COMPLETE"
