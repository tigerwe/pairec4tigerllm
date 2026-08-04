#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-c1}}"
NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-pairec-brpc-wrapper}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$MODE" in
  c1)
    CONCURRENCY=1
    CONFIG_NAME=pairec-config-brpc-wrapper-c1
    CONFIG_MANIFEST=k8s/configmap-pairec-brpc-wrapper-c1.yaml
    ;;
  c1000)
    CONCURRENCY=1000
    CONFIG_NAME=pairec-config-brpc-wrapper-c1000
    CONFIG_MANIFEST=k8s/configmap-pairec-brpc-wrapper-c1000.yaml
    ;;
  *) die "mode must be c1 or c1000" ;;
esac

for command in kubectl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: ${command}"
done

echo "== Apply burst mode ${MODE} =="
kubectl apply -f "$CONFIG_MANIFEST"
kubectl -n "$NAMESPACE" patch deployment "$DEPLOYMENT" --type=strategic \
  -p "{\"spec\":{\"template\":{\"spec\":{\"volumes\":[{\"name\":\"config-volume\",\"configMap\":{\"name\":\"${CONFIG_NAME}\"}}]}}}}"
kubectl -n "$NAMESPACE" rollout restart "deployment/${DEPLOYMENT}"
kubectl -n "$NAMESPACE" rollout status "deployment/${DEPLOYMENT}" --timeout="$ROLLOUT_TIMEOUT"

POD="$(kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')"
test -n "$POD" || die "running pod not found"

echo "== Verify mounted configuration =="
actual="$(kubectl -n "$NAMESPACE" exec "$POD" -- python3 -c '
import json
config = json.load(open("/app/configs/pairec_config.json", encoding="utf-8"))
algo = json.loads(config["RecallConfs"][0]["RecallAlgo"])
print(algo["brpc_burst_concurrency"], algo["brpc_burst_payload_bytes"],
      str(algo["brpc_burst_preconnect"]).lower(), str(algo["brpc_fallback_to_http"]).lower(),
      config["RecallConfs"][0]["CacheTime"])
')"
test "$actual" = "$CONCURRENCY 102400 true false 0" \
  || die "unexpected runtime configuration: ${actual}"

echo "== Verify strict startup preconnect =="
ready_line="$(kubectl -n "$NAMESPACE" logs "$POD" | grep '"event":"pairec_brpc_burst_ready"' | tail -1 || true)"
test -n "$ready_line" || die "burst ready event missing"
python3 - "$ready_line" "$CONCURRENCY" <<'PY'
import json, sys
event = json.loads(sys.argv[1][sys.argv[1].find("{"):])
expected = int(sys.argv[2])
assert event["concurrency"] == expected, event
assert event["connected_sessions"] == expected, event
assert event["payload_bytes"] == 102400, event
PY

echo "PAIREC_BRPC_BURST_MODE_READY mode=${MODE} concurrency=${CONCURRENCY}"
echo "pod=${POD}"
echo "next=EXPECTED_CONCURRENCY=${CONCURRENCY} REQUESTS=3 bash scripts/benchmark_pairec_brpc_burst.sh"
