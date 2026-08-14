#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
DEPLOYMENT=${DEPLOYMENT:-inference-brpc-trtllm}
APP_LABEL=${APP_LABEL:-app=inference-brpc-trtllm}
INFERENCE_CONTAINER=${INFERENCE_CONTAINER:-brpc-inference}
SIDECAR_CONTAINER=${SIDECAR_CONTAINER:-kvc-burst-wrapper}
POD_RUNTIME_DIR=${POD_RUNTIME_DIR:-/opt/pairec-f19}
DS_ENDPOINT=${DS_ENDPOINT:-192.168.100.12:18482}
SINCE=${SINCE:-20m}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/f14-kvc-burst-diagnostic/$(date +%Y%m%d-%H%M%S)}

mkdir -p "$OUTPUT_DIR"

section() {
  printf '\n== %s ==\n' "$1"
}

latest_pod() {
  kubectl -n "$NAMESPACE" get pod -l "$APP_LABEL" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
    | tail -n 1
}

POD=${POD:-$(latest_pod)}
if [[ -z "$POD" ]]; then
  echo "ERROR: no Pod found for label $APP_LABEL in namespace $NAMESPACE" >&2
  exit 1
fi

echo "== F14 KVC burst CrashLoop diagnostic target =="
echo "namespace=$NAMESPACE deployment=$DEPLOYMENT pod=$POD"
echo "inference_container=$INFERENCE_CONTAINER sidecar_container=$SIDECAR_CONTAINER"
echo "datasystem_endpoint=$DS_ENDPOINT log_since=$SINCE"
echo "output_dir=$OUTPUT_DIR"

kubectl -n "$NAMESPACE" get pod "$POD" -o json >"$OUTPUT_DIR/pod.json"
kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json >"$OUTPUT_DIR/deployment.json"

section "Pod and container states"
kubectl -n "$NAMESPACE" get pod "$POD" -o wide | tee "$OUTPUT_DIR/pod.txt"
python3 - "$OUTPUT_DIR/pod.json" <<'PY' | tee "$OUTPUT_DIR/container-statuses.txt"
import json
import pathlib
import sys

pod = json.loads(pathlib.Path(sys.argv[1]).read_text())
for status in pod.get("status", {}).get("containerStatuses", []):
    print(f"container={status['name']}")
    print(f"  ready={status.get('ready')} restart_count={status.get('restartCount', 0)}")
    print(f"  state={json.dumps(status.get('state', {}), separators=(',', ':'))}")
    print(f"  last_state={json.dumps(status.get('lastState', {}), separators=(',', ':'))}")
PY

section "Pod events"
kubectl -n "$NAMESPACE" get events \
  --field-selector "involvedObject.kind=Pod,involvedObject.name=$POD" \
  --sort-by=.lastTimestamp \
  -o custom-columns='LAST:.lastTimestamp,TYPE:.type,REASON:.reason,MESSAGE:.message' \
  2>&1 | tee "$OUTPUT_DIR/events.txt" || true

section "Container command, environment, resources, and mounts"
python3 - "$OUTPUT_DIR/pod.json" <<'PY' | tee "$OUTPUT_DIR/container-specs.txt"
import json
import pathlib
import sys

pod = json.loads(pathlib.Path(sys.argv[1]).read_text())
for container in pod["spec"].get("containers", []):
    print(f"container={container['name']}")
    print(f"  image={container.get('image', '')}")
    print(f"  command={json.dumps(container.get('command', []), separators=(',', ':'))}")
    print(f"  args={json.dumps(container.get('args', []), separators=(',', ':'))}")
    env = {entry["name"]: entry.get("value", "<valueFrom>") for entry in container.get("env", [])}
    print(f"  env={json.dumps(env, sort_keys=True, separators=(',', ':'))}")
    print(f"  resources={json.dumps(container.get('resources', {}), separators=(',', ':'))}")
    print(f"  mounts={json.dumps(container.get('volumeMounts', []), separators=(',', ':'))}")
PY

section "Current and previous logs"
for container in "$INFERENCE_CONTAINER" "$SIDECAR_CONTAINER"; do
  echo "-- $container current --"
  kubectl -n "$NAMESPACE" logs "$POD" -c "$container" --since="$SINCE" \
    >"$OUTPUT_DIR/${container}-current.log" 2>&1 || true
  tail -200 "$OUTPUT_DIR/${container}-current.log"

  echo "-- $container previous --"
  kubectl -n "$NAMESPACE" logs "$POD" -c "$container" --previous --tail=500 \
    >"$OUTPUT_DIR/${container}-previous.log" 2>&1 || true
  tail -300 "$OUTPUT_DIR/${container}-previous.log"
done

section "Sidecar binary and runtime dependencies"
kubectl -n "$NAMESPACE" exec "$POD" -c "$INFERENCE_CONTAINER" -- sh -c "
  binary='$POD_RUNTIME_DIR/bin/kvc_burst_wrapper'
  echo binary=\"\$binary\"
  ls -lh \"\$binary\" || exit 20
  if command -v ldd >/dev/null 2>&1; then
    ldd \"\$binary\"
  else
    echo 'ldd unavailable'
  fi
  echo '-- shared control directory --'
  ls -la /run/pairec-kvc-burst || true
" >"$OUTPUT_DIR/runtime-check.txt" 2>&1 || true
cat "$OUTPUT_DIR/runtime-check.txt"

section "DataSystem TCP reachability from inference container"
ds_host=${DS_ENDPOINT%:*}
ds_port=${DS_ENDPOINT##*:}
kubectl -n "$NAMESPACE" exec "$POD" -c "$INFERENCE_CONTAINER" -- \
  env DS_HOST="$ds_host" DS_PORT="$ds_port" sh -c '
    if command -v nc >/dev/null 2>&1; then
      nc -vz -w 3 "$DS_HOST" "$DS_PORT"
    elif command -v timeout >/dev/null 2>&1 && command -v bash >/dev/null 2>&1; then
      timeout 3 bash -c ": </dev/tcp/${DS_HOST}/${DS_PORT}"
    else
      echo "SKIPPED: nc or timeout+bash is unavailable"
      exit 3
    fi
  ' >"$OUTPUT_DIR/datasystem-tcp.txt" 2>&1 || true
cat "$OUTPUT_DIR/datasystem-tcp.txt"

section "Diagnosis"
python3 - "$OUTPUT_DIR" "$SIDECAR_CONTAINER" <<'PY' | tee "$OUTPUT_DIR/summary.txt"
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
sidecar = sys.argv[2]
pod = json.loads((root / "pod.json").read_text())
statuses = {s["name"]: s for s in pod.get("status", {}).get("containerStatuses", [])}
status = statuses.get(sidecar, {})
last = status.get("lastState", {}).get("terminated", {})
current = status.get("state", {})
exit_code = last.get("exitCode")
reason = last.get("reason", "")

parts = []
for name in (f"{sidecar}-previous.log", f"{sidecar}-current.log", "runtime-check.txt", "datasystem-tcp.txt"):
    path = root / name
    if path.exists():
        parts.append(path.read_text(errors="replace"))
evidence = "\n".join(parts)
lower = evidence.lower()

if reason == "OOMKilled" or exit_code == 137:
    classification = "KVC_BURST_SIDECAR_OOM"
    next_action = "inspect the sidecar memory limit and prefill allocation before increasing it"
elif re.search(r"error while loading shared libraries|cannot open shared object|\.so[^\n]*=>\s*not found", lower):
    classification = "KVC_BURST_SIDECAR_DYNAMIC_LIBRARY_FAILURE"
    next_action = "fix LD_LIBRARY_PATH/rpath or mount the missing DataSystem runtime library"
elif exit_code == 2 or "unknown argument" in lower or "invalid argument" in lower:
    classification = "KVC_BURST_SIDECAR_ARGUMENT_FAILURE"
    next_action = "compare the deployed sidecar args with the built binary CLI"
elif any(token in lower for token in ("prefill", "verify pressure", "pressure key", "mcreate", "mset")):
    classification = "KVC_BURST_DATASYSTEM_PREFILL_FAILURE"
    next_action = "check the first explicit prefill/verify error and DataSystem endpoint health"
elif any(token in lower for token in ("connection refused", "connect failed", "timed out", "timeout")):
    classification = "KVC_BURST_DATASYSTEM_CONNECTIVITY_FAILURE"
    next_action = "verify routing and TCP reachability to the configured DataSystem endpoint"
elif any(token in lower for token in ("mmap", "shm", "control_path", "control mapping")):
    classification = "KVC_BURST_SHARED_CONTROL_FAILURE"
    next_action = "verify the shared tmpfs mount and control file ABI initialization"
elif exit_code not in (None, 0):
    classification = "KVC_BURST_SIDECAR_PROCESS_FAILURE"
    next_action = "use the previous sidecar log and exit code to locate the first fatal initialization step"
elif not status:
    classification = "KVC_BURST_SIDECAR_STATUS_MISSING"
    next_action = "verify that the overlay added the expected sidecar container"
elif current.get("running") and status.get("ready"):
    classification = "KVC_BURST_SIDECAR_HEALTHY_NOW"
    next_action = "the sidecar recovered; audit previous logs and restart count before validation"
else:
    classification = "KVC_BURST_SIDECAR_NOT_READY"
    next_action = "inspect current logs and readiness file creation"

print(f"classification={classification}")
print(f"pod={pod['metadata']['name']}")
print(f"sidecar_ready={status.get('ready', False)}")
print(f"sidecar_restart_count={status.get('restartCount', 0)}")
print(f"sidecar_last_reason={reason or 'none'}")
print(f"sidecar_last_exit_code={exit_code if exit_code is not None else 'none'}")
print(f"next_action={next_action}")
print(f"output_dir={root}")
PY

echo "F14_KVC_BURST_CRASHLOOP_DIAGNOSTIC_COMPLETE"
