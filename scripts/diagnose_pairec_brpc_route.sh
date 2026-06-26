#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"
DATASYSTEM_LABEL="${DATASYSTEM_LABEL:-app.kubernetes.io/part-of=datasystem-pool}"
SCENE_ID="${SCENE_ID:-home_feed}"
USER_ID="${USER_ID:-6312}"
SIZE="${SIZE:-10}"
SINCE="${SINCE:-5m}"
RUN_REQUEST="${RUN_REQUEST:-1}"
RUN_DIRECT_BRPC_SMOKE="${RUN_DIRECT_BRPC_SMOKE:-1}"

section() {
  printf '\n== %s ==\n' "$1"
}

run_allow_fail() {
  printf '+ %s\n' "$*"
  "$@" || true
}

section "topology pods"
run_allow_fail kubectl -n "$NAMESPACE" get pods -o wide

section "datasystem pool pods"
run_allow_fail kubectl -n "$NAMESPACE" get pods -l "$DATASYSTEM_LABEL" -o wide

section "services and endpoints"
run_allow_fail kubectl -n "$NAMESPACE" get svc "$PAIREC_DEPLOYMENT" "$INFERENCE_DEPLOYMENT" -o wide
run_allow_fail kubectl -n "$NAMESPACE" get endpoints "$PAIREC_DEPLOYMENT" "$INFERENCE_DEPLOYMENT" -o wide

section "deployment placement"
run_allow_fail kubectl -n "$NAMESPACE" get deploy "$PAIREC_DEPLOYMENT" "$INFERENCE_DEPLOYMENT" -o wide
echo
echo "PaiRec nodeName:"
kubectl -n "$NAMESPACE" get deploy "$PAIREC_DEPLOYMENT" \
  -o jsonpath='{.spec.template.spec.nodeName}{"\n"}' 2>/dev/null || true
echo "Inference nodeName:"
kubectl -n "$NAMESPACE" get deploy "$INFERENCE_DEPLOYMENT" \
  -o jsonpath='{.spec.template.spec.nodeName}{"\n"}' 2>/dev/null || true

section "inference datasystem env and args"
echo "DataSystem env:"
kubectl -n "$NAMESPACE" get deploy "$INFERENCE_DEPLOYMENT" \
  -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}{"="}{.value}{"\n"}{end}' 2>/dev/null \
  | grep -E '^DATASYSTEM_|^HOST_IP=|^LD_PRELOAD=' || true
echo
echo "Args:"
kubectl -n "$NAMESPACE" get deploy "$INFERENCE_DEPLOYMENT" \
  -o jsonpath='{range .spec.template.spec.containers[0].args[*]}{@}{"\n"}{end}' 2>/dev/null || true

section "pairec brpc config"
run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- \
  sh -lc 'grep -n "brpc_endpoint" /app/configs/pairec_config.json || true'

section "pairec DNS and TCP checks"
run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- sh -lc '
set +e
echo "-- /etc/resolv.conf --"
cat /etc/resolv.conf || true

endpoint=$(sed -n '\''s/.*\\*"brpc_endpoint\\*"[[:space:]]*:[[:space:]]*\\*"\([^"\\]*\).*/\1/p'\'' /app/configs/pairec_config.json | head -1)
host=${endpoint%:*}
port=${endpoint##*:}

echo "brpc_endpoint=${endpoint}"
echo "brpc_host=${host}"
echo "brpc_port=${port}"

echo "-- nslookup short --"
if command -v nslookup >/dev/null 2>&1; then
  nslookup "$host" || true
  nslookup "${host}.pairec.svc.cluster.local" || true
else
  echo "nslookup not found"
fi

echo "-- tcp check --"
if command -v nc >/dev/null 2>&1; then
  nc -vz -w 3 "$host" "$port" || true
elif command -v telnet >/dev/null 2>&1; then
  echo quit | telnet "$host" "$port" || true
else
  echo "nc/telnet not found in PaiRec image; skipping raw TCP check"
fi
'

if [ "$RUN_DIRECT_BRPC_SMOKE" = "1" ]; then
  section "direct brpc smoke inside inference pod"
  if [ -x scripts/test_brpc_native_inference_smoke.sh ]; then
    TARGET="deployment/${INFERENCE_DEPLOYMENT}" \
      CONTAINER="$INFERENCE_CONTAINER" \
      SERVER="127.0.0.1:18100" \
      REQUESTS=1 \
      TOPK="$SIZE" \
      bash scripts/test_brpc_native_inference_smoke.sh || true
  else
    echo "scripts/test_brpc_native_inference_smoke.sh not found or not executable"
  fi
fi

if [ "$RUN_REQUEST" = "1" ]; then
  section "send one PaiRec request"
  request_json="{\"scene_id\":\"${SCENE_ID}\",\"uid\":\"${USER_ID}\",\"size\":${SIZE}}"
  echo "request=${request_json}"
  run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- \
    wget -q -O - \
      --header='Content-Type: application/json' \
      --post-data="$request_json" \
      http://127.0.0.1:18080/api/recommend
  echo
else
  echo
  echo "Skipped request because RUN_REQUEST=${RUN_REQUEST}"
fi

section "recent inference Recommend logs"
kubectl -n "$NAMESPACE" logs "deploy/${INFERENCE_DEPLOYMENT}" -c "$INFERENCE_CONTAINER" --since="$SINCE" 2>/dev/null \
  | grep -E 'method=Recommend|brpc inference server listening|Init KvCache|Rank 0 is using GPU|with host endpoint|with ServiceDiscovery|\[Datasystem\]\[TRACE\]' \
  || echo "no matching inference logs in --since=${SINCE}"

section "recent PaiRec logs"
run_allow_fail kubectl -n "$NAMESPACE" logs "deploy/${PAIREC_DEPLOYMENT}" --since="$SINCE" --tail=200

section "PaiRec recall debug file"
run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- \
  sh -lc 'tail -240 /tmp/recall_debug.log 2>/dev/null || echo "/tmp/recall_debug.log not found"'

section "quick interpretation hints"
cat <<EOF
- If Service endpoints are empty, inference is not Ready or selector is wrong.
- If nslookup fails in PaiRec, use the Service ClusterIP in brpc_endpoint or fix CoreDNS.
- If direct brpc smoke succeeds but PaiRec request does not produce method=Recommend, the break is PaiRec config/client/DNS/TCP.
- If inference logs show method=Recommend but PaiRec returns code=299, the issue is recommendation quality/item count, not brpc routing.
- If inference logs show Init KvCache host=141.61.91.189, the current inference -> DataSystem endpoint is 189.
EOF
