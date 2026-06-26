#!/usr/bin/env bash
set -euo pipefail

SOURCE_NODE="${SOURCE_NODE:-141.61.91.188}"
WORKSPACE="${WORKSPACE:-/home/zcx/workspace/pairec4tigerllm}"
FEATURE_FILE="${FEATURE_FILE:-data/user_features.json}"
CHECK_USER="${CHECK_USER:-6312}"
NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec}"
RESTART_PAIREC="${RESTART_PAIREC:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-5m}"

src="${SOURCE_NODE}:${WORKSPACE}/${FEATURE_FILE}"
dst="${WORKSPACE}/${FEATURE_FILE}"
backup="${dst}.bak.$(date +%Y%m%d-%H%M%S)"

echo "Sync PaiRec fallback feature file"
echo "  source: ${src}"
echo "  target: ${dst}"
echo

if [ ! -d "$(dirname "$dst")" ]; then
  echo "ERROR: target directory does not exist: $(dirname "$dst")" >&2
  exit 1
fi

if [ -f "$dst" ]; then
  echo "Backup current target:"
  echo "  ${backup}"
  cp -a "$dst" "$backup"
else
  echo "WARN: target file does not exist, no backup created"
fi

echo
echo "Copying feature file from ${SOURCE_NODE} ..."
scp "$src" "$dst"

echo
echo "Target file after sync:"
ls -lh "$dst"

if command -v python3 >/dev/null 2>&1; then
  python3 - "$dst" "$CHECK_USER" <<'PY'
import json
import sys

path, user = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"user_count={len(data)}")
print(f"check_user={user} present={user in data}")
PY
else
  echo "python3 not found; skip user count check"
fi

if [ "$RESTART_PAIREC" = "1" ]; then
  echo
  echo "Restarting ${NAMESPACE}/${PAIREC_DEPLOYMENT} because fallback features are cached in process"
  kubectl -n "$NAMESPACE" rollout restart "deployment/${PAIREC_DEPLOYMENT}"
  kubectl -n "$NAMESPACE" rollout status "deployment/${PAIREC_DEPLOYMENT}" --timeout="$WAIT_TIMEOUT"
fi

echo
echo "Verify with:"
echo "  kubectl -n ${NAMESPACE} exec deploy/${PAIREC_DEPLOYMENT} -- wget -q -O - --header='Content-Type: application/json' --post-data='{\"scene_id\":\"home_feed\",\"uid\":\"${CHECK_USER}\",\"size\":10}' http://127.0.0.1:18080/api/recommend"
echo "  kubectl -n ${NAMESPACE} logs deploy/inference-brpc-trtllm -c brpc-inference --since=2m | grep 'method=Recommend'"
