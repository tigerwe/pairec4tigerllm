#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
MANIFEST="${MANIFEST:-k8s/deployment-brpc-return-pressure-sinks-master.yaml}"
HOST_BIN="${HOST_BIN:-/home/zcx/bin/brpc_return_pressure_sink}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"

die() { echo "ERROR: $*" >&2; exit 1; }

test -f "$MANIFEST" || die "manifest not found: $MANIFEST"
test -x "$HOST_BIN" || die "return pressure sink binary is missing: $HOST_BIN"
command -v kubectl >/dev/null 2>&1 || die "kubectl is required"

expected_sha="$(sha256sum "$HOST_BIN" | awk '{print $1}')"
echo "sink_sha256=$expected_sha"
kubectl apply -f "$MANIFEST"
for deployment in generation-return-pressure-sink rank-return-pressure-sink; do
  kubectl -n "$NAMESPACE" rollout restart "deployment/$deployment"
  kubectl -n "$NAMESPACE" rollout status "deployment/$deployment" --timeout="$ROLLOUT_TIMEOUT"
  pod="$(kubectl -n "$NAMESPACE" get pod -l "app=$deployment" -o jsonpath='{.items[0].metadata.name}')"
  test -n "$pod" || die "pod not found for $deployment"
  mounted_sha="$(kubectl -n "$NAMESPACE" exec "$pod" -- env -u LD_PRELOAD sha256sum /proc/1/exe | awk '{print $1}')"
  test "$mounted_sha" = "$expected_sha" || die "$deployment running binary checksum mismatch"
  qos="$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.status.qosClass}')"
  test "$qos" = Guaranteed || die "$deployment must have Guaranteed QoS, got $qos"
  kubectl -n "$NAMESPACE" exec "$pod" -- env -u LD_PRELOAD sh -c '
    grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/1/status
    test ! -f /sys/fs/cgroup/cpu.stat || cat /sys/fs/cgroup/cpu.stat
  '
done

kubectl -n "$NAMESPACE" get pods \
  -l 'app in (generation-return-pressure-sink,rank-return-pressure-sink)' -o wide
echo "PAIREC_RETURN_PRESSURE_SINKS_READY endpoints=141.61.91.189:18301,141.61.91.189:18302"
