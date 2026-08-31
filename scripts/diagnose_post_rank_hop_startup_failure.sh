#!/usr/bin/env bash
# Collect and classify a post-rank Hop startup failure without sending traffic.
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
TARGET_ROLE="${TARGET_ROLE:-hop1}"
HOP1_APP="${HOP1_APP:-post-rank-hop1}"
HOP2_APP="${HOP2_APP:-post-rank-hop2}"
BINARY_PATH="${BINARY_PATH:-/home/zcx/bin/brpc_post_rank_hop}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-post-rank-hop-diagnostic/$(date +%Y%m%d-%H%M%S)}"

die() { echo "ERROR: $*" >&2; exit 1; }
for command in kubectl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
[[ "$TARGET_ROLE" = hop1 || "$TARGET_ROLE" = hop2 ]] \
  || die "TARGET_ROLE must be hop1 or hop2"
mkdir -p "$OUTPUT_DIR"

latest_pod() {
  kubectl -n "$NAMESPACE" get pods -l "app=$1" -o json | python3 -c '
import json,sys
pods=json.load(sys.stdin).get("items",[])
if pods:
    print(max((p["metadata"].get("creationTimestamp",""),p["metadata"]["name"]) for p in pods)[1])
' "$1"
}

collect_pod() {
  local app="$1" container="$2" prefix="$3" pod restarts
  pod="$(latest_pod "$app")"
  if [[ -z "$pod" ]]; then
    printf '{"metadata":{"name":""},"status":{}}\n' >"$OUTPUT_DIR/${prefix}-pod.json"
    printf 'POD_NOT_FOUND app=%s\n' "$app" >"$OUTPUT_DIR/${prefix}-describe.txt"
    printf 'POD_NOT_FOUND app=%s\n' "$app" >"$OUTPUT_DIR/${prefix}-current.log"
    : >"$OUTPUT_DIR/${prefix}-previous.log"
    : >"$OUTPUT_DIR/${prefix}-container-resources.txt"
    return
  fi
  kubectl -n "$NAMESPACE" get pod "$pod" -o json >"$OUTPUT_DIR/${prefix}-pod.json"
  kubectl -n "$NAMESPACE" describe pod "$pod" >"$OUTPUT_DIR/${prefix}-describe.txt" 2>&1 || true
  kubectl -n "$NAMESPACE" logs "$pod" -c "$container" --timestamps --tail=500 \
    >"$OUTPUT_DIR/${prefix}-current.log" 2>&1 || true
  restarts="$(kubectl -n "$NAMESPACE" get pod "$pod" \
    -o "jsonpath={.status.containerStatuses[?(@.name=='$container')].restartCount}" \
    2>/dev/null || echo 0)"
  if [[ "${restarts:-0}" != 0 ]]; then
    kubectl -n "$NAMESPACE" logs "$pod" -c "$container" --previous --timestamps --tail=500 \
      >"$OUTPUT_DIR/${prefix}-previous.log" 2>&1 || true
  else
    : >"$OUTPUT_DIR/${prefix}-previous.log"
  fi
  kubectl -n "$NAMESPACE" exec "$pod" -c "$container" -- sh -c '
    echo "== limits =="; cat /proc/1/limits 2>/dev/null || true
    echo "== status =="; grep -E "^(Name|State|VmPeak|VmSize|VmRSS|Threads):" /proc/1/status 2>/dev/null || true
    echo "== cgroup v2 =="
    for name in pids.current pids.max memory.current memory.max memory.events; do
      test ! -r "/sys/fs/cgroup/$name" || { echo "-- $name"; cat "/sys/fs/cgroup/$name"; }
    done
    echo "== cgroup v1 =="
    for name in pids/pids.current pids/pids.max memory/memory.usage_in_bytes memory/memory.limit_in_bytes memory/memory.failcnt; do
      test ! -r "/sys/fs/cgroup/$name" || { echo "-- $name"; cat "/sys/fs/cgroup/$name"; }
    done
  ' >"$OUTPUT_DIR/${prefix}-container-resources.txt" 2>&1 || true
  printf '%s\n' "$pod"
}

HOP1_POD="$(collect_pod "$HOP1_APP" post-rank-hop1 hop1)"
HOP2_POD="$(collect_pod "$HOP2_APP" post-rank-hop2 hop2)"

kubectl -n "$NAMESPACE" get pods -l "app in ($HOP1_APP,$HOP2_APP)" -o wide \
  >"$OUTPUT_DIR/pods.txt" 2>&1 || true
kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp \
  >"$OUTPUT_DIR/events.txt" 2>&1 || true

if command -v ss >/dev/null 2>&1; then
  ss -ltnp >"$OUTPUT_DIR/listeners.txt" 2>&1 || true
else
  : >"$OUTPUT_DIR/listeners.txt"
fi

if [[ -f "$BINARY_PATH" ]]; then
  sha256sum "$BINARY_PATH" >"$OUTPUT_DIR/host-binary-sha256.txt" 2>&1 || true
  if command -v strings >/dev/null 2>&1; then
    strings "$BINARY_PATH" | grep -E \
      'PAIREC_SOURCE_COMMIT|pressure_start_quorum|pressure_start_timeout_ms|business_backend|pressure_backend' \
      >"$OUTPUT_DIR/host-binary-capabilities.txt" 2>&1 || true
  else
    printf 'strings command unavailable\n' >"$OUTPUT_DIR/host-binary-capabilities.txt"
  fi
else
  printf 'BINARY_NOT_FOUND path=%s\n' "$BINARY_PATH" >"$OUTPUT_DIR/host-binary-sha256.txt"
  : >"$OUTPUT_DIR/host-binary-capabilities.txt"
fi

{
  echo "threads_max=$(cat /proc/sys/kernel/threads-max 2>/dev/null || echo unavailable)"
  echo "pid_max=$(cat /proc/sys/kernel/pid_max 2>/dev/null || echo unavailable)"
  echo "host_threads=$(ps -eLf 2>/dev/null | wc -l || echo unavailable)"
  echo "ulimit_user_processes=$(ulimit -u 2>/dev/null || echo unavailable)"
  echo "ulimit_open_files=$(ulimit -n 2>/dev/null || echo unavailable)"
  command -v free >/dev/null 2>&1 && free -b || true
} >"$OUTPUT_DIR/host-resources.txt"

python3 - "$OUTPUT_DIR/tcp-probes.json" <<'PY'
import json,socket,sys
result={}
for port in (18311,18312,18313):
    sock=socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect(("127.0.0.1",port))
        result[str(port)]=True
    except OSError:
        result[str(port)]=False
    finally:
        sock.close()
open(sys.argv[1],"w").write(json.dumps(result,indent=2)+"\n")
PY

python3 - "$TARGET_ROLE" "$OUTPUT_DIR" "$HOP1_POD" "$HOP2_POD" <<'PY'
import json,pathlib,re,sys

role,root,hop1_pod,hop2_pod=sys.argv[1:]
root=pathlib.Path(root)
prefix="hop1" if role=="hop1" else "hop2"
pod=json.loads((root/f"{prefix}-pod.json").read_text())
logs="\n".join((root/f"{prefix}-{kind}.log").read_text(errors="replace")
               for kind in ("previous","current"))
resources=(root/f"{prefix}-container-resources.txt").read_text(errors="replace")
capabilities=(root/"host-binary-capabilities.txt").read_text(errors="replace")
listeners=(root/"listeners.txt").read_text(errors="replace")
tcp=json.loads((root/"tcp-probes.json").read_text())
statuses={x.get("name"):x for x in pod.get("status",{}).get("containerStatuses",[])}
name="post-rank-hop1" if role=="hop1" else "post-rank-hop2"
status=statuses.get(name,{})
terminated=status.get("lastState",{}).get("terminated",{})
waiting=status.get("state",{}).get("waiting",{})
reason=terminated.get("reason","")
exit_code=terminated.get("exitCode")
signal=terminated.get("signal")
restart_count=status.get("restartCount",0)

classification="UNKNOWN_POST_RANK_HOP_STARTUP_FAILURE"
confidence="low"
detail="startup evidence does not match a known boundary"
next_action="inspect the preserved previous log before changing timeouts or concurrency"
connected=None

match=re.search(r"post-rank hop preconnect failed: connected=(\d+)/1000",logs)
thread_failure=any(token in logs for token in (
    "Resource temporarily unavailable","std::system_error","system_error","pthread_create",
    "pressure worker thread creation failed","Cannot allocate memory"))
address_in_use=("Address already in use" in logs or "EADDRINUSE" in logs)
target_port="18311" if role=="hop1" else "18312"
target_port_listening=re.search(rf":{target_port}\b",listeners) is not None
pod_name=pod.get("metadata",{}).get("name","")
containers=pod.get("spec",{}).get("containers",[])
container=next((item for item in containers if item.get("name")==name),{})
pod_args=container.get("args",[])
condition_messages=" ".join(str(item.get("message","")) for item in
    pod.get("status",{}).get("conditions",[]))

if not pod_name:
    classification="POST_RANK_HOP_POD_NOT_CREATED"
    confidence="high"
    detail=f"no {role} Pod exists after the Deployment was applied"
    next_action="inspect diagnostic events.txt for admission, selector, or controller errors"
elif waiting.get("reason") in ("ErrImagePull","ImagePullBackOff","InvalidImageName"):
    classification="POST_RANK_HOP_IMAGE_PULL_FAILURE"
    confidence="high"
    detail=f"{role} container cannot pull its runtime image: {waiting.get('message','')}"
    next_action="repair the exact runtime image reference; Pod retries cannot create a missing image"
elif "Unschedulable" in condition_messages:
    classification="POST_RANK_HOP_UNSCHEDULABLE"
    confidence="high"
    detail=f"{role} cannot be scheduled: {condition_messages}"
    next_action="repair nodeSelector or requested resources before retrying the rollout"
elif exit_code==2:
    required=("pressure_start_quorum","pressure_start_timeout_ms",
              "business_backend","pressure_backend") if role=="hop1" else ()
    missing=[token for token in required if token not in capabilities]
    classification="POST_RANK_HOP_BINARY_ARGUMENT_CONTRACT_MISMATCH"
    confidence="high"
    detail=f"{role} exited with code 2 during argument parsing"
    if missing:
        detail+=f"; host binary lacks capabilities: {','.join(missing)}"
    next_action="build the current commit with BUILD_HOP_IMAGE=1 and redeploy; restarting this stale binary cannot recover"
elif reason=="OOMKilled" or exit_code==137:
    classification="POST_RANK_HOP_STARTUP_OOM"
    confidence="high"
    detail=f"{role} was OOM-killed before becoming Ready"
    next_action="increase the real memory limit only after checking thread count and RSS evidence"
elif thread_failure:
    classification="POST_RANK_HOP1_THREAD_RESOURCE_EXHAUSTION" if role=="hop1" else "POST_RANK_HOP_THREAD_RESOURCE_EXHAUSTION"
    confidence="high"
    lane=re.search(r"thread creation failed: lane=(\d+)",logs)
    detail="native pressure worker creation exhausted PID/thread/memory resources"
    if lane:
        detail+=f" at lane {lane.group(1)}"
    next_action="inspect hop1-container-resources.txt and remove the PID/thread limit; repeated Pod restarts cannot fix it"
elif "post-rank business channel preconnect failed" in logs:
    classification="POST_RANK_HOP2_BUSINESS_ENDPOINT_UNREACHABLE"
    confidence="high"
    detail="Hop-1 could not preconnect its dedicated business channel to Hop-2:18312"
    next_action="repair Hop-2 business listener or route before deploying Hop-1"
elif "post-rank hop preconnect startup timed out" in logs:
    classification="POST_RANK_HOP1_PRECONNECT_TIMEOUT"
    confidence="high"
    detail="Hop-1 did not finish all c1000 session preconnects before the startup deadline"
    next_action="inspect Hop-2 Health capacity and failed lanes; do not increase the timeout without connection evidence"
elif match:
    connected=int(match.group(1))
    classification="POST_RANK_HOP1_PRECONNECT_PARTIAL"
    confidence="high"
    detail=f"Hop-1 connected only {connected}/1000 dedicated sessions"
    next_action="inspect host file descriptors/PIDs and Hop-2 refusal logs for the missing sessions"
elif address_in_use or ("Failed to start post-rank" in logs and target_port_listening):
    classification="POST_RANK_HOST_PORT_CONFLICT"
    confidence="high"
    detail=f"hostNetwork port {target_port} was occupied when {role} started"
    next_action="identify the listener owner and wait for port release before applying the Deployment"
elif role=="hop1" and exit_code not in (None,0) and tcp.get("18312") and tcp.get("18313") and not tcp.get("18311"):
    classification="POST_RANK_HOP1_PRECONNECT_INITIALIZATION_FAILURE"
    confidence="medium"
    detail="Hop-2 listeners are reachable and 18311 is free, but Hop-1 exited before listening"
    next_action="use the first fatal line in hop1-previous.log; rebuild with lane-level startup diagnostics if absent"

summary={
    "classification":classification,"confidence":confidence,"target_role":role,
    "target_pod":hop1_pod if role=="hop1" else hop2_pod,
    "hop1_pod":hop1_pod,"hop2_pod":hop2_pod,"reason":detail,
    "next_action":next_action,"restart_count":restart_count,"exit_code":exit_code,
    "signal":signal,"terminated_reason":reason,"connected_sessions":connected,
    "tcp_listening":tcp,
    "pod_args":pod_args,
    "host_binary_capabilities":capabilities.splitlines(),
    "container_resource_evidence":str(root/f"{prefix}-container-resources.txt"),
}
(root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
for key in ("classification","confidence","target_role","target_pod","reason",
            "restart_count","exit_code","signal","terminated_reason","connected_sessions",
            "next_action"):
    print(f"{key}={summary[key]}")
print(f"summary_json={root/'summary.json'}")
PY

echo "== Previous target log tail =="
tail -120 "$OUTPUT_DIR/${TARGET_ROLE}-previous.log"
echo "output_dir=$OUTPUT_DIR"
echo "PAIREC_POST_RANK_HOP_STARTUP_DIAGNOSTIC_COMPLETE"
