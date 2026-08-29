#!/usr/bin/env bash
# Disable both reverse BRPC stages and remove their master-side Sinks.
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-10m}
SINK_DELETE_TIMEOUT=${SINK_DELETE_TIMEOUT:-120s}

die() { echo "ERROR: $*" >&2; exit 1; }

for command in kubectl python3; do
  command -v "$command" >/dev/null 2>&1 || die "$command is required"
done

disable_reverse_endpoint() {
  local deployment=$1
  local container=$2
  local patch

  echo "== Disable reverse BRPC: deployment/$deployment container=$container =="
  kubectl -n "$NAMESPACE" get deployment "$deployment" >/dev/null 2>&1 \
    || die "deployment not found: $deployment"

  patch=$(
    kubectl -n "$NAMESPACE" get deployment "$deployment" -o json \
      | python3 -c '
import json
import sys

container_name = sys.argv[1]
deployment = json.load(sys.stdin)
containers = deployment["spec"]["template"]["spec"]["containers"]
container = next(
    (item for item in containers if item.get("name") == container_name), None
)
if container is None:
    raise RuntimeError("container not found: " + container_name)

args = list(container.get("args", []))
indexes = [
    index for index, value in enumerate(args)
    if value.startswith("--reverse_burst_endpoint=")
]
if len(indexes) != 1:
    raise RuntimeError(
        "expected exactly one reverse_burst_endpoint argument, got "
        + repr(indexes)
    )
args[indexes[0]] = "--reverse_burst_endpoint="

print(json.dumps({
    "spec": {"template": {"spec": {"containers": [{
        "name": container_name,
        "args": args,
    }]}}}
}))
' "$container"
  )

  kubectl -n "$NAMESPACE" patch deployment "$deployment" \
    --type=strategic -p "$patch"
  kubectl -n "$NAMESPACE" rollout status "deployment/$deployment" \
    --timeout="$ROLLOUT_TIMEOUT"

  kubectl -n "$NAMESPACE" get deployment "$deployment" -o json \
    | python3 -c '
import json
import sys

container_name = sys.argv[1]
deployment = json.load(sys.stdin)
container = next(
    item
    for item in deployment["spec"]["template"]["spec"]["containers"]
    if item.get("name") == container_name
)
values = [
    value for value in container.get("args", [])
    if value.startswith("--reverse_burst_endpoint=")
]
if values != ["--reverse_burst_endpoint="]:
    raise RuntimeError("reverse BRPC remains enabled: " + repr(values))
print(container_name + " reverse_burst_endpoint=<empty>")
' "$container"
}

disable_reverse_endpoint brpc-burst-wrapper brpc-burst-wrapper
disable_reverse_endpoint deepfm-rank-burst-wrapper rank-burst-wrapper

echo "== Delete reverse BRPC pressure Sinks =="
for sink in generation-return-pressure-sink rank-return-pressure-sink; do
  kubectl -n "$NAMESPACE" delete deployment "$sink" \
    --ignore-not-found --wait=true --timeout="$SINK_DELETE_TIMEOUT"
  if kubectl -n "$NAMESPACE" get pods -l "app=$sink" -o name \
      | grep -q .; then
    kubectl -n "$NAMESPACE" wait --for=delete pod -l "app=$sink" \
      --timeout="$SINK_DELETE_TIMEOUT"
  fi
done

echo "== Recovered Wrapper status =="
kubectl -n "$NAMESPACE" get pods \
  -l 'app in (brpc-burst-wrapper,deepfm-rank-burst-wrapper)' -o wide

echo "PAIREC_REVERSE_BRPC_ENVIRONMENT_RECOVERED"
