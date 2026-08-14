#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-apply}
NAMESPACE=${NAMESPACE:-pairec}
DEPLOYMENT=${DEPLOYMENT:-inference-brpc-trtllm}
INFERENCE_CONTAINER=${INFERENCE_CONTAINER:-brpc-inference}
SIDECAR_CONTAINER=${SIDECAR_CONTAINER:-kvc-burst-wrapper}
APP_LABEL=${APP_LABEL:-app=inference-brpc-trtllm}
HOST_RUNTIME_DIR=${HOST_RUNTIME_DIR:-/home/zcx/pairec-f19-runtime}
POD_RUNTIME_DIR=${POD_RUNTIME_DIR:-/opt/pairec-f19}
CONCURRENCY=${CONCURRENCY:-1}
OBJECT_SIZE=${OBJECT_SIZE:-3670016}
BARRIER_TIMEOUT_MS=${BARRIER_TIMEOUT_MS:-5}
DS_ENDPOINT=${DS_ENDPOINT:-192.168.100.12:18482}
KVC_BURST_ENABLED=${KVC_BURST_ENABLED:-1}
MEASURE_DISABLED=${MEASURE_DISABLED:-0}
BACKUP_FILE=${BACKUP_FILE:-/tmp/f14-kvc-burst-deployment-before.json}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-10m}

die() { echo "ERROR: $*" >&2; exit 1; }

current_pod() {
  kubectl -n "$NAMESPACE" get pod -l "$APP_LABEL" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}'
}

wait_rollout() {
  kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT" || {
    pod=$(current_pod 2>/dev/null || true)
    [[ -z "$pod" ]] || kubectl -n "$NAMESPACE" describe "pod/$pod" | tail -120 || true
    die "rollout failed"
  }
}

verify() {
  pod=$(current_pod)
  [[ -n "$pod" ]] || die "inference Pod not found"
  kubectl -n "$NAMESPACE" get pod "$pod" -o wide
  kubectl -n "$NAMESPACE" exec "$pod" -c "$INFERENCE_CONTAINER" -- \
    grep -aFq KVC_BURST_CONTROL_PATH \
      "$POD_RUNTIME_DIR/lib/libtensorrt_llm.so" \
    || die "KVC proxy capability marker is missing from TensorRT-LLM"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$SIDECAR_CONTAINER" -- \
    test -s /run/pairec-kvc-burst/ready \
    || die "KVC burst sidecar is not ready"
  ready=$(kubectl -n "$NAMESPACE" exec "$pod" -c "$SIDECAR_CONTAINER" -- \
    cat /run/pairec-kvc-burst/ready)
  echo "$ready"
  grep -Fq "concurrency=$CONCURRENCY" <<<"$ready" \
    || die "sidecar concurrency mismatch"
  kubectl -n "$NAMESPACE" logs "$pod" -c "$SIDECAR_CONTAINER" --tail=200 \
    | grep -F '"event":"kvc_burst_ready"' | tail -1
  echo "F14_KVC_BURST_OVERLAY_VERIFY_OK pod=$pod concurrency=$CONCURRENCY enabled=$KVC_BURST_ENABLED"
}

apply_overlay() {
  [[ "$CONCURRENCY" =~ ^(1|10|100)$ ]] || die "CONCURRENCY must be 1, 10, or 100"
  [[ "$KVC_BURST_ENABLED" = 0 || "$KVC_BURST_ENABLED" = 1 ]] \
    || die "KVC_BURST_ENABLED must be 0 or 1"
  [[ "$MEASURE_DISABLED" = 0 || "$MEASURE_DISABLED" = 1 ]] \
    || die "MEASURE_DISABLED must be 0 or 1"
  [[ "$DS_ENDPOINT" == *:* ]] || die "DS_ENDPOINT must be host:port"
  ds_host=${DS_ENDPOINT%:*}
  ds_port=${DS_ENDPOINT##*:}

  if [[ ! -f "$BACKUP_FILE" ]]; then
    kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json >"$BACKUP_FILE"
    echo "backup=$BACKUP_FILE"
  fi
  deployment_json=$(mktemp /tmp/f14-kvc-deployment.XXXXXX.json)
  patch_json=$(mktemp /tmp/f14-kvc-patch.XXXXXX.json)
  kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json >"$deployment_json"
  python3 - "$deployment_json" "$patch_json" "$INFERENCE_CONTAINER" "$SIDECAR_CONTAINER" \
      "$HOST_RUNTIME_DIR" "$POD_RUNTIME_DIR" "$CONCURRENCY" "$OBJECT_SIZE" \
      "$BARRIER_TIMEOUT_MS" "$ds_host" "$ds_port" "$KVC_BURST_ENABLED" "$MEASURE_DISABLED" <<'PY'
import json, pathlib, sys
(deployment_path, output_path, inference_name, sidecar_name, host_runtime,
 pod_runtime, concurrency, object_size, barrier_timeout, ds_host, ds_port,
 enabled, measure_disabled) = sys.argv[1:]
deployment = json.loads(pathlib.Path(deployment_path).read_text())
inference = next(c for c in deployment["spec"]["template"]["spec"]["containers"]
                 if c["name"] == inference_name)
image = inference["image"]
runtime_env_names = {
    "HOST_IP",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
}
sidecar_env = [
    entry for entry in inference.get("env", [])
    if entry["name"] in runtime_env_names or entry["name"].startswith("DATASYSTEM_")
]
preload = next((entry.get("value", "") for entry in sidecar_env
                if entry["name"] == "LD_PRELOAD"), "")
if not preload:
    raise RuntimeError(
        "inference LD_PRELOAD is empty; KVC sidecar requires the DataSystem ARM GPU runtime preload chain"
    )
sidecar_preload = " ".join(
    token for token in preload.split() if "libnvidia-ml.so" not in token
)
required_preloads = ("block_ds_consumer.so", "stub_gpu.so", "libabseil_dll.so")
missing_preloads = [name for name in required_preloads if name not in sidecar_preload]
if missing_preloads:
    raise RuntimeError(
        "inference LD_PRELOAD lacks required KVC sidecar libraries: " + ",".join(missing_preloads)
    )
for index, entry in enumerate(sidecar_env):
    if entry["name"] == "LD_PRELOAD":
        sidecar_env[index] = {**entry, "value": sidecar_preload}
        break
patch = {"spec": {"template": {"metadata": {"annotations": {
    "pairec.io/f14-kvc-burst-generation": str(__import__("time").time_ns())
}}, "spec": {
    "containers": [
        {"name": inference_name,
         "env": [
             {"name": "KVC_BURST_ENABLED", "value": enabled},
             {"name": "KVC_BURST_MEASURE_DISABLED", "value": measure_disabled},
             {"name": "KVC_BURST_VERBOSE", "value": "0"},
             {"name": "KVC_BURST_CONTROL_PATH", "value": "/run/pairec-kvc-burst/control"},
         ],
         "volumeMounts": [{"name": "kvc-burst-control", "mountPath": "/run/pairec-kvc-burst"}]},
        {"name": sidecar_name, "image": image, "imagePullPolicy": "IfNotPresent",
         "command": [f"{pod_runtime}/bin/kvc_burst_wrapper"],
         "args": [f"--host={ds_host}", f"--port={ds_port}", f"--concurrency={concurrency}",
                  f"--object_size={object_size}", f"--barrier_timeout_ms={barrier_timeout}",
                  "--prefix=PairecKvcBurstV2", "--control_path=/run/pairec-kvc-burst/control",
                  "--ready_file=/run/pairec-kvc-burst/ready", "--cleanup_keys=true"],
         "env": sidecar_env,
         "resources": {"requests": {"cpu": "4", "memory": "1Gi"},
                       "limits": {"memory": "4Gi"}},
         "volumeMounts": [
             {"name": "f19-runtime-bin", "mountPath": f"{pod_runtime}/bin", "readOnly": True},
             {"name": "kvc-burst-control", "mountPath": "/run/pairec-kvc-burst"}],
         "startupProbe": {"exec": {"command": ["sh", "-c", "test -s /run/pairec-kvc-burst/ready"]},
                          "periodSeconds": 2, "failureThreshold": 150},
         "readinessProbe": {"exec": {"command": ["sh", "-c", "test -s /run/pairec-kvc-burst/ready"]},
                            "periodSeconds": 5, "failureThreshold": 2},
         "livenessProbe": {"exec": {"command": ["sh", "-c", "test -s /run/pairec-kvc-burst/ready"]},
                           "periodSeconds": 15, "failureThreshold": 3}},
    ],
    "volumes": [
        {"name": "f19-runtime-bin", "hostPath": {"path": f"{host_runtime}/bin", "type": "Directory"}},
        {"name": "kvc-burst-control", "emptyDir": {"medium": "Memory", "sizeLimit": "1Mi"}},
    ],
}}}}
pathlib.Path(output_path).write_text(json.dumps(patch))
PY
  kubectl -n "$NAMESPACE" patch deployment "$DEPLOYMENT" \
    --type=strategic --patch "$(cat "$patch_json")"
  rm -f "$deployment_json" "$patch_json"
  wait_rollout
  verify
  echo "F14_KVC_BURST_OVERLAY_APPLY_OK"
}

restore() {
  [[ -f "$BACKUP_FILE" ]] || die "backup not found: $BACKUP_FILE"
  restore_file=$(mktemp /tmp/f14-kvc-restore.XXXXXX.json)
  python3 - "$BACKUP_FILE" "$restore_file" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
value = json.loads(source.read_text())
value.pop("status", None)
metadata = value["metadata"]
for key in ("creationTimestamp", "generation", "managedFields", "resourceVersion", "uid"):
    metadata.pop(key, None)
pathlib.Path(output).write_text(json.dumps(value))
PY
  kubectl -n "$NAMESPACE" replace --force -f "$restore_file"
  rm -f "$restore_file"
  wait_rollout
  rm -f "$BACKUP_FILE"
  echo "F14_KVC_BURST_OVERLAY_RESTORE_OK"
}

case "$ACTION" in
  apply) apply_overlay ;;
  verify) verify ;;
  restore) restore ;;
  *) die "usage: $0 [apply|verify|restore]" ;;
esac
