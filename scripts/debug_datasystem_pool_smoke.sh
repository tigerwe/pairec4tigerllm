#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
ETCD_ADDRESS="${ETCD_ADDRESS:-141.61.91.188:12379}"
CLUSTER_NAME="${CLUSTER_NAME:-pairec}"
HOST_ID_ENV_NAME="${HOST_ID_ENV_NAME:-HOST_IP}"
AFFINITY_POLICY="${AFFINITY_POLICY:-RANDOM}"
EXPECT_MIN_WORKERS="${EXPECT_MIN_WORKERS:-${EXPECT_MIN_WORKKERS:-2}}"
KUBECTL_TIMEOUT="${KUBECTL_TIMEOUT:-20s}"
POD="${POD:-}"

log() {
  printf '\n== %s ==\n' "$*"
}

run_kubectl() {
  timeout "$KUBECTL_TIMEOUT" kubectl "$@"
}

if [ -z "$POD" ]; then
  POD="$(kubectl -n "$NAMESPACE" get pod -l app=datasystem-pool-worker \
    -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' \
    | head -n 1)"
fi

if [ -z "$POD" ]; then
  echo "ERROR: no running datasystem-pool-worker pod found" >&2
  kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/part-of=datasystem-pool -o wide >&2
  exit 1
fi

log "pool pods"
kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/part-of=datasystem-pool -o wide

log "selected worker pod"
kubectl -n "$NAMESPACE" get pod "$POD" -o wide

log "exec channel"
run_kubectl -n "$NAMESPACE" exec "$POD" -c datasystem-worker -- \
  bash -lc 'echo exec-ok; cat /etc/hostname 2>/dev/null || true; env | grep -E "^(HOST_IP|ETCD|CLUSTER|LD_PRELOAD)=" || true'

log "python import"
run_kubectl -n "$NAMESPACE" exec -i "$POD" -c datasystem-worker -- \
  python - <<'PY'
print("python-start", flush=True)
import pkgutil
import yr.datasystem as ds
print("datasystem-import-ok", flush=True)
print("datasystem-file=", getattr(ds, "__file__", "<unknown>"), flush=True)
try:
    from yr.datasystem import KVClient, ServiceAffinityPolicy, ServiceDiscovery, ServiceDiscoveryOptions
except ImportError as first_error:
    print("top-level-service-discovery-import-failed=", repr(first_error), flush=True)
    try:
        from yr.datasystem import KVClient
        from yr.datasystem.service_discovery import ServiceAffinityPolicy, ServiceDiscovery, ServiceDiscoveryOptions
    except Exception as second_error:
        print("service-discovery-submodule-import-failed=", repr(second_error), flush=True)
        print("service-discovery-symbols=", [name for name in dir(ds) if "Service" in name or "Discovery" in name], flush=True)
        if hasattr(ds, "__path__"):
            print("datasystem-submodules=", sorted(m.name for m in pkgutil.iter_modules(ds.__path__)), flush=True)
        raise
print("symbols-ok", flush=True)
PY

log "etcd tcp connectivity"
ETCD_HOST="${ETCD_ADDRESS%:*}"
ETCD_PORT="${ETCD_ADDRESS##*:}"
run_kubectl -n "$NAMESPACE" exec "$POD" -c datasystem-worker -- \
  env ETCD_HOST="$ETCD_HOST" ETCD_PORT="$ETCD_PORT" \
  bash -lc 'bash -c "</dev/tcp/${ETCD_HOST}/${ETCD_PORT}" && echo etcd-port-ok || { echo etcd-port-fail; exit 1; }'

log "service discovery"
run_kubectl -n "$NAMESPACE" exec -i "$POD" -c datasystem-worker -- \
  env \
    ETCD_ADDRESS="$ETCD_ADDRESS" \
    CLUSTER_NAME="$CLUSTER_NAME" \
    HOST_ID_ENV_NAME="$HOST_ID_ENV_NAME" \
    AFFINITY_POLICY="$AFFINITY_POLICY" \
    EXPECT_MIN_WORKERS="$EXPECT_MIN_WORKERS" \
    python - <<'PY'
import os

print("sd-start", flush=True)
try:
    from yr.datasystem import ServiceAffinityPolicy, ServiceDiscovery, ServiceDiscoveryOptions
except ImportError:
    from yr.datasystem.service_discovery import ServiceAffinityPolicy, ServiceDiscovery, ServiceDiscoveryOptions

policy_name = os.environ["AFFINITY_POLICY"].upper()
policy = {
    "PREFERRED_SAME_NODE": ServiceAffinityPolicy.PREFERRED_SAME_NODE,
    "PREFERRED": ServiceAffinityPolicy.PREFERRED_SAME_NODE,
    "REQUIRED_SAME_NODE": ServiceAffinityPolicy.REQUIRED_SAME_NODE,
    "REQUIRED": ServiceAffinityPolicy.REQUIRED_SAME_NODE,
    "RANDOM": ServiceAffinityPolicy.RANDOM,
}[policy_name]

opts = ServiceDiscoveryOptions()
opts.etcd_address = os.environ["ETCD_ADDRESS"]
opts.cluster_name = os.environ["CLUSTER_NAME"]
opts.host_id_env_name = os.environ["HOST_ID_ENV_NAME"]
opts.affinity_policy = policy

sd = ServiceDiscovery(opts)
print("before-init", flush=True)
sd.init()
print("after-init", flush=True)

workers = set()
for i in range(8):
    print(f"before-select {i}", flush=True)
    status, ip, port, is_same_node = sd.select_worker()
    if status.is_error():
        raise RuntimeError(status.to_string())
    worker = f"{ip}:{port}"
    workers.add(worker)
    print(f"selected {worker} same_node={is_same_node}", flush=True)

expect_min = int(os.environ["EXPECT_MIN_WORKERS"])
print("discovered_workers=", sorted(workers), flush=True)
if len(workers) < expect_min:
    raise RuntimeError(f"expected at least {expect_min} workers, got {len(workers)}")
PY

log "kv set/get"
run_kubectl -n "$NAMESPACE" exec -i "$POD" -c datasystem-worker -- \
  env \
    ETCD_ADDRESS="$ETCD_ADDRESS" \
    CLUSTER_NAME="$CLUSTER_NAME" \
    HOST_ID_ENV_NAME="$HOST_ID_ENV_NAME" \
    AFFINITY_POLICY="$AFFINITY_POLICY" \
    python - <<'PY'
import os
import time
import uuid

print("kv-start", flush=True)
try:
    from yr.datasystem import KVClient, ServiceAffinityPolicy, ServiceDiscovery, ServiceDiscoveryOptions
except ImportError:
    from yr.datasystem import KVClient
    from yr.datasystem.service_discovery import ServiceAffinityPolicy, ServiceDiscovery, ServiceDiscoveryOptions

policy_name = os.environ["AFFINITY_POLICY"].upper()
policy = {
    "PREFERRED_SAME_NODE": ServiceAffinityPolicy.PREFERRED_SAME_NODE,
    "PREFERRED": ServiceAffinityPolicy.PREFERRED_SAME_NODE,
    "REQUIRED_SAME_NODE": ServiceAffinityPolicy.REQUIRED_SAME_NODE,
    "REQUIRED": ServiceAffinityPolicy.REQUIRED_SAME_NODE,
    "RANDOM": ServiceAffinityPolicy.RANDOM,
}[policy_name]

opts = ServiceDiscoveryOptions()
opts.etcd_address = os.environ["ETCD_ADDRESS"]
opts.cluster_name = os.environ["CLUSTER_NAME"]
opts.host_id_env_name = os.environ["HOST_ID_ENV_NAME"]
opts.affinity_policy = policy

sd = ServiceDiscovery(opts)
sd.init()
client = KVClient("", 0, service_discovery=sd, enable_cross_node_connection=True)
print("before-client-init", flush=True)
client.init()
print("after-client-init", flush=True)

key = "pairec:ds-pool-debug:" + uuid.uuid4().hex
value = ("ok:" + str(time.time())).encode()
client.set(key, value)
print("after-set", key, flush=True)
got = client.get([key])
if got != [value]:
    raise RuntimeError(f"unexpected get result: {got!r}")
print("kv set/get ok key=", key, flush=True)
PY

log "debug finished"
