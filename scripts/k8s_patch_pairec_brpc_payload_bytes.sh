#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_CONFIGMAP="${PAIREC_CONFIGMAP:-pairec-config}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec}"
BRPC_PAYLOAD_BYTES="${BRPC_PAYLOAD_BYTES:-102400}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-5m}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to patch the ConfigMap safely" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
configmap_json="${tmpdir}/configmap.json"

echo "Patching ${NAMESPACE}/${PAIREC_CONFIGMAP}"
echo "  RecallAlgo.brpc_payload_bytes -> ${BRPC_PAYLOAD_BYTES}"

kubectl -n "$NAMESPACE" get configmap "$PAIREC_CONFIGMAP" -o json > "$configmap_json"

python3 - "$configmap_json" "$BRPC_PAYLOAD_BYTES" <<'PY'
import json
import sys

path, payload_bytes = sys.argv[1], int(sys.argv[2])
if payload_bytes < 0:
    raise SystemExit("brpc_payload_bytes must be >= 0")

with open(path, "r", encoding="utf-8") as handle:
    obj = json.load(handle)

data = obj.setdefault("data", {})
config_text = data.get("pairec_config.json")
if not config_text:
    raise SystemExit("ConfigMap does not contain data['pairec_config.json']")

config = json.loads(config_text)
recall_confs = config.get("RecallConfs") or []
if not recall_confs:
    raise SystemExit("pairec_config.json does not contain RecallConfs")

updated = False
for recall_conf in recall_confs:
    if recall_conf.get("Name") != "generative_recall":
        continue
    algo_text = recall_conf.get("RecallAlgo") or "{}"
    algo = json.loads(algo_text)
    algo["brpc_payload_bytes"] = payload_bytes
    recall_conf["RecallAlgo"] = json.dumps(algo, ensure_ascii=False, separators=(",", ":"))
    updated = True
    break

if not updated:
    raise SystemExit("failed to find RecallConfs entry name=generative_recall")

data["pairec_config.json"] = json.dumps(config, ensure_ascii=False, indent=2)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(obj, handle, ensure_ascii=False)
PY

kubectl replace -f "$configmap_json"

echo
echo "Restarting ${NAMESPACE}/${PAIREC_DEPLOYMENT} to reload ConfigMap"
kubectl -n "$NAMESPACE" rollout restart "deployment/${PAIREC_DEPLOYMENT}"
kubectl -n "$NAMESPACE" rollout status "deployment/${PAIREC_DEPLOYMENT}" --timeout="$WAIT_TIMEOUT"

echo
echo "Current brpc_payload_bytes:"
kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- \
  sh -lc 'grep -n "brpc_payload_bytes" /app/configs/pairec_config.json || true'
