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
PRESSURE_KEY_COUNT=${PRESSURE_KEY_COUNT:-0}
OBJECT_SIZE=${OBJECT_SIZE:-3670016}
BARRIER_TIMEOUT_MS=${BARRIER_TIMEOUT_MS:-5}
PRESSURE_LEAD_US=${PRESSURE_LEAD_US:-1000}
DS_ENDPOINT=${DS_ENDPOINT:-192.168.100.12:18482}
KVC_BURST_ENABLED=${KVC_BURST_ENABLED:-1}
MEASURE_DISABLED=${MEASURE_DISABLED:-0}
KVC_BURST_VERBOSE=${KVC_BURST_VERBOSE:-0}
KVC_BURST_INITIAL_ARMED=${KVC_BURST_INITIAL_ARMED:-1}
SUSTAINED_PRESSURE=${SUSTAINED_PRESSURE:-0}
SUSTAINED_MAX_DURATION_MS=${SUSTAINED_MAX_DURATION_MS:-5000}
SUSTAINED_MAX_LOOPS=${SUSTAINED_MAX_LOOPS:-1000}
INPROCESS_PRESSURE=${INPROCESS_PRESSURE:-0}
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
    grep -aFq PAIREC_KVC_BURST_PROXY_V5 \
      "$POD_RUNTIME_DIR/lib/libtensorrt_llm.so" \
    || die "KVC proxy V5 capability marker is missing from TensorRT-LLM"
  if [[ "$INPROCESS_PRESSURE" = 1 ]]; then
    kubectl -n "$NAMESPACE" exec "$pod" -c "$INFERENCE_CONTAINER" -- \
      grep -aFq PAIREC_KVC_INPROCESS_SUSTAINED_C32_V1 \
        "$POD_RUNTIME_DIR/lib/libtensorrt_llm.so" \
      || die "in-process sustained KVC c32 capability marker is missing from TensorRT-LLM"
  fi
  kubectl -n "$NAMESPACE" exec "$pod" -c "$SIDECAR_CONTAINER" -- \
    test -s /run/pairec-kvc-burst/ready \
    || die "KVC burst sidecar is not ready"
  ready=$(kubectl -n "$NAMESPACE" exec "$pod" -c "$SIDECAR_CONTAINER" -- \
    cat /run/pairec-kvc-burst/ready)
  echo "$ready"
  grep -Fq "concurrency=$CONCURRENCY" <<<"$ready" \
    || die "sidecar concurrency mismatch"
  ready_event=$(kubectl -n "$NAMESPACE" logs "$pod" -c "$SIDECAR_CONTAINER" --tail=200 \
    | grep -F '"event":"kvc_burst_ready"' | tail -1)
  echo "$ready_event"
  grep -Fq "\"object_size_bytes\":$OBJECT_SIZE" <<<"$ready_event" \
    || die "sidecar object size mismatch: expected=$OBJECT_SIZE"
  grep -Fq '"version":5' <<<"$ready_event" \
    || die "sidecar control protocol mismatch: expected version=5"
  expected_engine=sidecar-exclusive-clients
  [[ "$INPROCESS_PRESSURE" = 0 ]] || expected_engine=inprocess-shared-client
  grep -Fq "\"pressure_engine\":\"$expected_engine\"" <<<"$ready_event" \
    || die "sidecar pressure engine mismatch: expected=$expected_engine"
  grep -Fq "\"pressure_lead_us\":$PRESSURE_LEAD_US" <<<"$ready_event" \
    || die "sidecar pressure lead mismatch: expected=$PRESSURE_LEAD_US"
  if [[ "$SUSTAINED_PRESSURE" = 1 ]]; then
    grep -Fq '"sustained_pressure":true' <<<"$ready_event" \
      || die "sidecar sustained pressure mismatch: expected enabled"
    grep -Fq "\"sustained_max_duration_ms\":$SUSTAINED_MAX_DURATION_MS" <<<"$ready_event" \
      || die "sidecar sustained max duration mismatch: expected=$SUSTAINED_MAX_DURATION_MS"
    grep -Fq "\"sustained_max_loops\":$SUSTAINED_MAX_LOOPS" <<<"$ready_event" \
      || die "sidecar sustained max loops mismatch: expected=$SUSTAINED_MAX_LOOPS"
  fi
  echo "F14_KVC_BURST_OVERLAY_VERIFY_OK pod=$pod concurrency=$CONCURRENCY object_size_bytes=$OBJECT_SIZE enabled=$KVC_BURST_ENABLED"
}

apply_overlay() {
  [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] && (( CONCURRENCY <= 256 )) \
    || die "CONCURRENCY must be between 1 and 256"
  [[ "$PRESSURE_KEY_COUNT" =~ ^[0-9]+$ ]] || die "PRESSURE_KEY_COUNT must be non-negative"
  (( PRESSURE_KEY_COUNT <= CONCURRENCY - 1 )) \
    || die "PRESSURE_KEY_COUNT must not exceed pressure lanes"
  [[ "$OBJECT_SIZE" =~ ^[1-9][0-9]*$ ]] || die "OBJECT_SIZE must be positive"
  [[ "$PRESSURE_LEAD_US" =~ ^[0-9]+$ ]] && (( PRESSURE_LEAD_US <= 1000000 )) \
    || die "PRESSURE_LEAD_US must be between 0 and 1000000"
  [[ "$KVC_BURST_ENABLED" = 0 || "$KVC_BURST_ENABLED" = 1 ]] \
    || die "KVC_BURST_ENABLED must be 0 or 1"
  [[ "$MEASURE_DISABLED" = 0 || "$MEASURE_DISABLED" = 1 ]] \
    || die "MEASURE_DISABLED must be 0 or 1"
  [[ "$KVC_BURST_VERBOSE" = 0 || "$KVC_BURST_VERBOSE" = 1 ]] \
    || die "KVC_BURST_VERBOSE must be 0 or 1"
  [[ "$KVC_BURST_INITIAL_ARMED" = 0 || "$KVC_BURST_INITIAL_ARMED" = 1 ]] \
    || die "KVC_BURST_INITIAL_ARMED must be 0 or 1"
  [[ "$SUSTAINED_PRESSURE" = 0 || "$SUSTAINED_PRESSURE" = 1 ]] \
    || die "SUSTAINED_PRESSURE must be 0 or 1"
  [[ "$SUSTAINED_MAX_DURATION_MS" =~ ^[1-9][0-9]*$ ]] \
    || die "SUSTAINED_MAX_DURATION_MS must be positive"
  [[ "$SUSTAINED_MAX_LOOPS" =~ ^[1-9][0-9]*$ ]] \
    || die "SUSTAINED_MAX_LOOPS must be positive"
  [[ "$INPROCESS_PRESSURE" = 0 || "$INPROCESS_PRESSURE" = 1 ]] \
    || die "INPROCESS_PRESSURE must be 0 or 1"
  if [[ "$INPROCESS_PRESSURE" = 1 ]]; then
    (( CONCURRENCY == 32 )) || die "in-process pressure requires CONCURRENCY=32"
    (( PRESSURE_KEY_COUNT >= 1 && PRESSURE_KEY_COUNT <= 31 )) \
      || die "in-process pressure requires PRESSURE_KEY_COUNT between 1 and 31"
    (( OBJECT_SIZE == 3670016 )) || die "in-process pressure requires OBJECT_SIZE=3670016"
    [[ "$KVC_BURST_INITIAL_ARMED" = 0 ]] || die "in-process pressure requires dynamic arm"
    [[ "$SUSTAINED_PRESSURE" = 1 ]] || die "in-process pressure must be sustained"
  fi
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
      "$HOST_RUNTIME_DIR" "$POD_RUNTIME_DIR" "$CONCURRENCY" "$PRESSURE_KEY_COUNT" "$OBJECT_SIZE" \
      "$BARRIER_TIMEOUT_MS" "$PRESSURE_LEAD_US" "$ds_host" "$ds_port" "$KVC_BURST_ENABLED" "$MEASURE_DISABLED" \
      "$KVC_BURST_VERBOSE" "$KVC_BURST_INITIAL_ARMED" \
      "$SUSTAINED_PRESSURE" "$SUSTAINED_MAX_DURATION_MS" "$SUSTAINED_MAX_LOOPS" \
      "$INPROCESS_PRESSURE" <<'PY'
import json, pathlib, sys
(deployment_path, output_path, inference_name, sidecar_name, host_runtime,
 pod_runtime, concurrency, pressure_key_count, object_size, barrier_timeout, pressure_lead_us,
 ds_host, ds_port,
 enabled, measure_disabled, verbose, initially_armed,
 sustained_pressure, sustained_max_duration_ms, sustained_max_loops,
 inprocess_pressure) = sys.argv[1:]
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
             {"name": "KVC_BURST_VERBOSE", "value": verbose},
             {"name": "KVC_BURST_CONTROL_PATH", "value": "/run/pairec-kvc-burst/control"},
             {"name": "PAIREC_KVC_INPROCESS_BURST", "value": inprocess_pressure},
             {"name": "PAIREC_KVC_INPROCESS_PRESSURE_PREFIX", "value": "PairecKvcBurstV2"},
         ],
         "volumeMounts": [{"name": "kvc-burst-control", "mountPath": "/run/pairec-kvc-burst"}]},
        {"name": sidecar_name, "image": image, "imagePullPolicy": "IfNotPresent",
         "command": [f"{pod_runtime}/bin/kvc_burst_wrapper"],
         "args": [f"--host={ds_host}", f"--port={ds_port}", f"--concurrency={concurrency}",
                  f"--pressure_key_count={pressure_key_count}",
                  f"--object_size={object_size}", f"--barrier_timeout_ms={barrier_timeout}",
                  f"--pressure_lead_us={pressure_lead_us}",
                  "--prefix=PairecKvcBurstV2", "--control_path=/run/pairec-kvc-burst/control",
                  "--ready_file=/run/pairec-kvc-burst/ready", "--cleanup_keys=true",
                  f"--initially_armed={initially_armed}",
                  f"--sustained_pressure={'true' if sustained_pressure == '1' else 'false'}",
                  f"--sustained_max_duration_ms={sustained_max_duration_ms}",
                  f"--sustained_max_loops={sustained_max_loops}",
                  f"--inprocess_pressure={'true' if inprocess_pressure == '1' else 'false'}"],
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
  current_file=$(mktemp /tmp/f14-kvc-current.XXXXXX.json)
  restore_patch=$(mktemp /tmp/f14-kvc-restore-patch.XXXXXX.json)
  kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json >"$current_file"
  python3 - "$current_file" "$restore_patch" "$INFERENCE_CONTAINER" "$SIDECAR_CONTAINER" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:3])
inference_name, sidecar_name = sys.argv[3:]
deployment = json.loads(source.read_text())
pod_spec = deployment["spec"]["template"]["spec"]
containers = []
for container in pod_spec.get("containers", []):
    if container["name"] == sidecar_name:
        continue
    if container["name"] == inference_name:
        container["env"] = [
            entry for entry in container.get("env", [])
            if not entry["name"].startswith("KVC_BURST_")
            and not entry["name"].startswith("PAIREC_KVC_INPROCESS_")
        ]
        container["volumeMounts"] = [
            mount for mount in container.get("volumeMounts", [])
            if mount["name"] != "kvc-burst-control"
        ]
    containers.append(container)
volumes = [
    volume for volume in pod_spec.get("volumes", [])
    if volume["name"] != "kvc-burst-control"
]
patch = {"spec": {"template": {
    "metadata": {"annotations": {"pairec.io/f14-kvc-burst-generation": None}},
    "spec": {"containers": containers, "volumes": volumes},
}}}
output.write_text(json.dumps(patch))
PY
  kubectl -n "$NAMESPACE" patch deployment "$DEPLOYMENT" \
    --type=merge --patch "$(cat "$restore_patch")"
  rm -f "$current_file" "$restore_patch"
  wait_rollout
  rm -f "$BACKUP_FILE"
  pod=$(current_pod)
  containers=$(kubectl -n "$NAMESPACE" get pod "$pod" \
    -o jsonpath='{range .spec.containers[*]}{.name}{"\n"}{end}')
  if grep -Fxq "$SIDECAR_CONTAINER" <<<"$containers"; then
    die "F14 sidecar remains after restore in pod $pod"
  fi
  echo "F14_KVC_BURST_OVERLAY_RESTORE_OK"
}

case "$ACTION" in
  apply) apply_overlay ;;
  verify) verify ;;
  restore) restore ;;
  *) die "usage: $0 [apply|verify|restore]" ;;
esac
