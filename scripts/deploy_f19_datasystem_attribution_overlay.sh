#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-apply}"
NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-inference-brpc-trtllm}"
CONTAINER="${CONTAINER:-brpc-inference}"
APP_LABEL="${APP_LABEL:-app=inference-brpc-trtllm}"
HOST_RUNTIME_DIR="${HOST_RUNTIME_DIR:-/home/zcx/pairec-f19-runtime}"
POD_RUNTIME_DIR="${POD_RUNTIME_DIR:-/opt/pairec-f19}"
EXPECTED_GATEWAY_SHA256="${EXPECTED_GATEWAY_SHA256:-39d85875eae04648aed42cebdaaf105000b3eb3244eb6d5c07feface8c3bc751}"
EXPECTED_TRTLLM_SHA256="${EXPECTED_TRTLLM_SHA256:-c6461918d88e742fea78b02d7dcaa3b9dcea30d5c6d0db3ff02cee20c438ccff}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"
RUN_EXACT_SMOKE="${RUN_EXACT_SMOKE:-1}"
PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec-brpc-observed}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/f19-datasystem-overlay/$(date +%Y%m%d-%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash scripts/deploy_f19_datasystem_attribution_overlay.sh [apply|verify|rollback]

apply     Mount the worker1 F19 runtime, verify it with attribution disabled,
          enable strict attribution, and run one exact request smoke by default.
verify    Verify the currently running Pod without changing the Deployment.
rollback  Roll the Deployment back one revision and wait for it to become ready.

Important environment overrides:
  HOST_RUNTIME_DIR, EXPECTED_GATEWAY_SHA256, EXPECTED_TRTLLM_SHA256
  RUN_EXACT_SMOKE=0, PAIREC_TARGET, NAMESPACE, DEPLOYMENT
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

deployment_json() {
  kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json
}

current_pod() {
  kubectl -n "$NAMESPACE" get pod -l "$APP_LABEL" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}'
}

wait_for_rollout() {
  if ! kubectl -n "$NAMESPACE" rollout status \
      "deployment/$DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"; then
    kubectl -n "$NAMESPACE" get pod -l "$APP_LABEL" -o wide || true
    pod="$(current_pod 2>/dev/null || true)"
    if [[ -n "$pod" ]]; then
      kubectl -n "$NAMESPACE" describe "pod/$pod" | tail -120 || true
      kubectl -n "$NAMESPACE" logs "$pod" -c "$CONTAINER" --tail=200 || true
    fi
    die "rollout failed; use '$0 rollback' after collecting evidence"
  fi
}

health_check() {
  local pod="$1"
  local output
  output="$(kubectl -n "$NAMESPACE" exec "$pod" -c "$CONTAINER" -- \
    /opt/pairec-brpc/bin/brpc_recommend_client \
      --server=127.0.0.1:18100 \
      --method=health \
      --requests=1 \
      --timeout_ms=3000 \
      --max_retry=0)"
  echo "$output"
  grep -q 'health ok' <<<"$output" || die "BRPC Health did not succeed"
}

verify_runtime() {
  local require_attribution="$1"
  local pod process_path hashes ldd_output gateway_hash trtllm_hash

  pod="$(current_pod)"
  [[ -n "$pod" ]] || die "no Pod found for label $APP_LABEL"
  kubectl -n "$NAMESPACE" get pod "$pod" -o wide

  process_path="$(kubectl -n "$NAMESPACE" exec "$pod" -c "$CONTAINER" -- \
    readlink /proc/1/exe)"
  echo "process=$process_path"
  [[ "$process_path" == "$POD_RUNTIME_DIR/bin/brpc_inference_server" ]] \
    || die "unexpected PID 1 executable: $process_path"

  hashes="$(kubectl -n "$NAMESPACE" exec "$pod" -c "$CONTAINER" -- \
    sha256sum \
      "$POD_RUNTIME_DIR/bin/brpc_inference_server" \
      "$POD_RUNTIME_DIR/lib/libtensorrt_llm.so")"
  echo "$hashes"
  gateway_hash="$(awk 'NR==1 {print $1}' <<<"$hashes")"
  trtllm_hash="$(awk 'NR==2 {print $1}' <<<"$hashes")"
  [[ "$gateway_hash" == "$EXPECTED_GATEWAY_SHA256" ]] \
    || die "gateway hash mismatch: $gateway_hash"
  [[ "$trtllm_hash" == "$EXPECTED_TRTLLM_SHA256" ]] \
    || die "TensorRT-LLM hash mismatch: $trtllm_hash"

  ldd_output="$(kubectl -n "$NAMESPACE" exec "$pod" -c "$CONTAINER" -- \
    sh -c 'unset LD_PRELOAD; ldd "$1"' sh \
      "$POD_RUNTIME_DIR/bin/brpc_inference_server")"
  echo "$ldd_output" | grep -E 'tensorrt_llm|not found' || true
  ! grep -q 'not found' <<<"$ldd_output" || die "F19 gateway has missing runtime libraries"
  grep -Fq "libtensorrt_llm.so => $POD_RUNTIME_DIR/lib/libtensorrt_llm.so" \
    <<<"$ldd_output" || die "gateway did not load the F19 TensorRT-LLM overlay"

  health_check "$pod"

  if [[ "$require_attribution" == 1 ]]; then
    kubectl -n "$NAMESPACE" logs "$pod" -c "$CONTAINER" \
      | grep -F '"event":"datasystem_attribution_ready"' \
      || die "native DataSystem attribution startup marker is missing"
  fi

  echo "F19_RUNTIME_VERIFY_OK attribution_required=$require_attribution pod=$pod"
}

apply_overlay() {
  local deployment_file patch_file current_ld new_ld

  mkdir -p "$OUTPUT_DIR"
  deployment_file="$OUTPUT_DIR/deployment-before.yaml"
  patch_file="$OUTPUT_DIR/overlay-patch.json"
  kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o yaml >"$deployment_file"
  deployment_json >"$OUTPUT_DIR/deployment-before.json"

  current_ld="$(python3 - "$CONTAINER" "$OUTPUT_DIR/deployment-before.json" <<'PY'
import json, pathlib, sys
container_name, path = sys.argv[1:]
deployment = json.loads(pathlib.Path(path).read_text())
container = next(
    item for item in deployment["spec"]["template"]["spec"]["containers"]
    if item["name"] == container_name
)
values = [item.get("value", "") for item in container.get("env", [])
          if item["name"] == "LD_LIBRARY_PATH"]
if len(values) != 1 or not values[0]:
    raise SystemExit("expected one non-empty LD_LIBRARY_PATH")
print(values[0])
PY
)"
  case ":$current_ld:" in
    *:"$POD_RUNTIME_DIR/lib":*) new_ld="$current_ld" ;;
    *) new_ld="$POD_RUNTIME_DIR/lib:$current_ld" ;;
  esac

  python3 - "$CONTAINER" "$HOST_RUNTIME_DIR" "$POD_RUNTIME_DIR" "$new_ld" \
      "$patch_file" <<'PY'
import json, pathlib, sys
container, host_runtime, pod_runtime, ld_library_path, output = sys.argv[1:]
patch = {
    "spec": {"template": {"spec": {
        "containers": [{
            "name": container,
            "command": [f"{pod_runtime}/bin/brpc_inference_server"],
            "env": [
                {"name": "LD_LIBRARY_PATH", "value": ld_library_path},
                {"name": "TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION", "value": "0"},
                {"name": "TRTLLM_DATASYSTEM_ATTRIBUTION_TTL_SECONDS", "value": "30"},
                {"name": "PAIREC_REQUIRE_NATIVE_DATASYSTEM_ATTRIBUTION", "value": "0"},
            ],
            "volumeMounts": [
                {"name": "f19-runtime-bin", "mountPath": f"{pod_runtime}/bin", "readOnly": True},
                {"name": "f19-runtime-lib", "mountPath": f"{pod_runtime}/lib", "readOnly": True},
            ],
        }],
        "volumes": [
            {"name": "f19-runtime-bin", "hostPath": {
                "path": f"{host_runtime}/bin", "type": "Directory"}},
            {"name": "f19-runtime-lib", "hostPath": {
                "path": f"{host_runtime}/lib", "type": "Directory"}},
        ],
    }}}}
}
pathlib.Path(output).write_text(json.dumps(patch, indent=2) + "\n")
PY

  echo "== Apply F19 runtime overlay with attribution disabled =="
  echo "backup=$deployment_file"
  echo "patch=$patch_file"
  kubectl -n "$NAMESPACE" patch deployment "$DEPLOYMENT" \
    --type=strategic --patch "$(cat "$patch_file")"
  wait_for_rollout
  verify_runtime 0

  echo "== Enable strict native DataSystem attribution =="
  kubectl -n "$NAMESPACE" set env "deployment/$DEPLOYMENT" \
    TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION=1 \
    TRTLLM_DATASYSTEM_ATTRIBUTION_TTL_SECONDS=30 \
    PAIREC_REQUIRE_NATIVE_DATASYSTEM_ATTRIBUTION=1
  wait_for_rollout
  verify_runtime 1

  if [[ "$RUN_EXACT_SMOKE" == 1 ]]; then
    echo "== Run one exact request-id attribution smoke =="
    NAMESPACE="$NAMESPACE" \
    PAIREC_TARGET="$PAIREC_TARGET" \
    BRPC_TARGET="deployment/$DEPLOYMENT" \
    REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 \
      bash scripts/trace_single_brpc_datasystem_request.sh
  fi

  kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o yaml \
    >"$OUTPUT_DIR/deployment-after.yaml"
  echo "F19_DATASYSTEM_ATTRIBUTION_OVERLAY_OK"
  echo "output_dir=$OUTPUT_DIR"
}

[[ "$RUN_EXACT_SMOKE" == 0 || "$RUN_EXACT_SMOKE" == 1 ]] \
  || die "RUN_EXACT_SMOKE must be 0 or 1"

case "$ACTION" in
  apply)
    for command in kubectl python3 sha256sum awk grep; do
      require_command "$command"
    done
    apply_overlay
    ;;
  verify)
    for command in kubectl sha256sum awk grep; do
      require_command "$command"
    done
    verify_runtime 1
    ;;
  rollback)
    require_command kubectl
    kubectl -n "$NAMESPACE" rollout undo "deployment/$DEPLOYMENT"
    wait_for_rollout
    echo "F19_DATASYSTEM_ATTRIBUTION_ROLLBACK_OK"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    die "unsupported action: $ACTION"
    ;;
esac
