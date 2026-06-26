#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_CONFIGMAP="${PAIREC_CONFIGMAP:-pairec-config}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
BRPC_PORT="${BRPC_PORT:-18100}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-5m}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to patch the ConfigMap safely" >&2
  exit 1
fi

service_ip="$(kubectl -n "$NAMESPACE" get svc "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
if [ -z "$service_ip" ] || [ "$service_ip" = "None" ]; then
  echo "ERROR: service ${NAMESPACE}/${INFERENCE_SERVICE} has no ClusterIP" >&2
  exit 1
fi

endpoint="${service_ip}:${BRPC_PORT}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
configmap_json="${tmpdir}/configmap.json"

echo "Patching ${NAMESPACE}/${PAIREC_CONFIGMAP}"
echo "  brpc_endpoint -> ${endpoint}"

kubectl -n "$NAMESPACE" get configmap "$PAIREC_CONFIGMAP" -o json > "$configmap_json"

python3 - "$configmap_json" "$endpoint" <<'PY'
import json
import re
import sys

path, endpoint = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)

data = obj.setdefault("data", {})
config = data.get("pairec_config.json")
if not config:
    raise SystemExit("ConfigMap does not contain data['pairec_config.json']")

pattern = r'(\\?"brpc_endpoint\\?"\s*:\s*\\?")[^"\\]*(\\?")'
updated, count = re.subn(pattern, r"\g<1>" + endpoint + r"\2", config, count=1)
if count != 1:
    raise SystemExit("failed to find brpc_endpoint in pairec_config.json")

data["pairec_config.json"] = updated
with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False)
PY

kubectl replace -f "$configmap_json"

echo
echo "Restarting ${NAMESPACE}/${PAIREC_DEPLOYMENT} to reload ConfigMap"
kubectl -n "$NAMESPACE" rollout restart "deployment/${PAIREC_DEPLOYMENT}"
kubectl -n "$NAMESPACE" rollout status "deployment/${PAIREC_DEPLOYMENT}" --timeout="$WAIT_TIMEOUT"

echo
echo "Current brpc_endpoint:"
kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- \
  sh -lc 'grep -n "brpc_endpoint" /app/configs/pairec_config.json || true'
