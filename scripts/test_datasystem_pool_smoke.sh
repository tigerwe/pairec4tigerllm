#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
ETCD_ADDRESS="${ETCD_ADDRESS:-141.61.91.188:12379}"
CLUSTER_NAME="${CLUSTER_NAME:-pairec}"
HOST_ID_ENV_NAME="${HOST_ID_ENV_NAME:-HOST_IP}"
AFFINITY_POLICY="${AFFINITY_POLICY:-RANDOM}"
EXPECT_MIN_WORKERS="${EXPECT_MIN_WORKERS:-${EXPECT_MIN_WORKKERS:-2}}"

POD="$(kubectl -n "$NAMESPACE" get pod -l app=datasystem-pool-worker \
  -o jsonpath='{.items[0].metadata.name}')"

if [ -z "$POD" ]; then
  echo "ERROR: no datasystem-pool-worker pod found" >&2
  exit 1
fi

kubectl -n "$NAMESPACE" exec -i "$POD" -c datasystem-worker -- \
  env \
    ETCD_ADDRESS="$ETCD_ADDRESS" \
    CLUSTER_NAME="$CLUSTER_NAME" \
    HOST_ID_ENV_NAME="$HOST_ID_ENV_NAME" \
    AFFINITY_POLICY="$AFFINITY_POLICY" \
    EXPECT_MIN_WORKERS="$EXPECT_MIN_WORKERS" \
    python - <<'PY'
import os
import time
import uuid

from yr.datasystem import KVClient, ServiceAffinityPolicy, ServiceDiscovery, ServiceDiscoveryOptions

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

workers = set()
last = None
for _ in range(32):
    status, ip, port, is_same_node = sd.select_worker()
    if status.is_error():
        raise RuntimeError(status.to_string())
    last = (ip, port, is_same_node)
    workers.add(f"{ip}:{port}")

expect_min = int(os.environ["EXPECT_MIN_WORKERS"])
print("discovered_workers=", sorted(workers))
print("last_selected=", last)
if len(workers) < expect_min:
    raise RuntimeError(f"expected at least {expect_min} workers, got {len(workers)}")

client = KVClient("", 0, service_discovery=sd, enable_cross_node_connection=True)
client.init()
key = "pairec:ds-pool-smoke:" + uuid.uuid4().hex
value = ("ok:" + str(time.time())).encode()
client.set(key, value)
got = client.get([key])
if got != [value]:
    raise RuntimeError(f"unexpected get result: {got!r}")
print("kv set/get ok key=", key)
PY
