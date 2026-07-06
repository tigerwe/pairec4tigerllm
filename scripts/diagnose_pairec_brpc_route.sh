#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"
PAIREC_POD_LABEL="${PAIREC_POD_LABEL:-app=${PAIREC_DEPLOYMENT}}"
INFERENCE_POD_LABEL="${INFERENCE_POD_LABEL:-app=${INFERENCE_DEPLOYMENT}}"
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

latest_pod_by_label() {
  local label="$1"
  local pods
  pods="$(kubectl -n "$NAMESPACE" get pod -l "$label" --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)"
  printf '%s\n' "$pods" | tail -n 1
}

PAIREC_POD="$(latest_pod_by_label "$PAIREC_POD_LABEL")"
INFERENCE_POD="$(latest_pod_by_label "$INFERENCE_POD_LABEL")"
INFERENCE_SVC_IP="$(kubectl -n "$NAMESPACE" get svc "$INFERENCE_DEPLOYMENT" \
  -o jsonpath='{.spec.clusterIP}' 2>/dev/null || true)"
INFERENCE_ENDPOINT_IP="$(kubectl -n "$NAMESPACE" get endpoints "$INFERENCE_DEPLOYMENT" \
  -o jsonpath='{range .subsets[*].addresses[*]}{.ip}{"\n"}{end}' 2>/dev/null | head -n 1 || true)"

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

section "runtime pod identity"
echo "PaiRec latest pod: ${PAIREC_POD:-<none>} label=${PAIREC_POD_LABEL}"
echo "Inference latest pod: ${INFERENCE_POD:-<none>} label=${INFERENCE_POD_LABEL}"
echo "Inference service clusterIP: ${INFERENCE_SVC_IP:-<none>}"
echo "Inference endpoint podIP: ${INFERENCE_ENDPOINT_IP:-<none>}"
if [ -n "$INFERENCE_POD" ]; then
  echo
  run_allow_fail kubectl -n "$NAMESPACE" get pod "$INFERENCE_POD" -o wide
  echo
  echo "Inference image/runtime identity:"
  kubectl -n "$NAMESPACE" get pod "$INFERENCE_POD" \
    -o jsonpath='image={.spec.containers[0].image}{"\n"}imageID={.status.containerStatuses[0].imageID}{"\n"}containerID={.status.containerStatuses[0].containerID}{"\n"}restartCount={.status.containerStatuses[0].restartCount}{"\n"}state={.status.containerStatuses[0].state}{"\n"}lastState={.status.containerStatuses[0].lastState}{"\n"}' \
    2>/dev/null || true
  echo
  echo "Inference command and args:"
  kubectl -n "$NAMESPACE" get pod "$INFERENCE_POD" \
    -o jsonpath='{range .spec.containers[0].command[*]}{.}{"\n"}{end}{range .spec.containers[0].args[*]}{.}{"\n"}{end}' \
    2>/dev/null || true
  echo
  echo "Inference brpc/app-related volume mounts:"
  kubectl -n "$NAMESPACE" get pod "$INFERENCE_POD" -o yaml 2>/dev/null \
    | sed -n '/containers:/,/volumes:/p' \
    | grep -E 'brpc-dev-bin|/opt/pairec-brpc/bin/brpc_inference_server|/app/trt_engines|/app/exported|/app/data|mountPath:|name:' \
    || true
fi

section "brpc binary overlay check"
overlay_matches="$(kubectl -n "$NAMESPACE" get deploy "$INFERENCE_DEPLOYMENT" -o yaml 2>/dev/null \
  | grep -nE 'brpc-dev-bin|/opt/pairec-brpc/bin/brpc_inference_server|pairec-brpc-dev' || true)"
if [ -n "$overlay_matches" ]; then
  cat <<EOF
WARNING: deployment still contains a brpc binary overlay. This can make the pod run
an old hostPath binary even when image/imageID points to a newly built image.
Remove the brpc-dev-bin volumeMount/volume before trusting image-based checks.
EOF
  printf '%s\n' "$overlay_matches"
else
  echo "No brpc-dev-bin hostPath overlay found in deployment/${INFERENCE_DEPLOYMENT}."
fi

section "inference datasystem env and args"
echo "DataSystem env:"
kubectl -n "$NAMESPACE" get deploy "$INFERENCE_DEPLOYMENT" \
  -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}{"="}{.value}{"\n"}{end}' 2>/dev/null \
  | grep -E '^DATASYSTEM_|^HOST_IP=|^LD_PRELOAD=' || true
echo
echo "Args:"
kubectl -n "$NAMESPACE" get deploy "$INFERENCE_DEPLOYMENT" \
  -o jsonpath='{range .spec.template.spec.containers[0].args[*]}{@}{"\n"}{end}' 2>/dev/null || true

section "inference brpc binary self-check"
if [ -n "$INFERENCE_POD" ]; then
  run_allow_fail kubectl -n "$NAMESPACE" exec "$INFERENCE_POD" -c "$INFERENCE_CONTAINER" -- sh -lc '
set +e
bin=/opt/pairec-brpc/bin/brpc_inference_server
echo "-- binary path --"
ls -l "$bin" || true
echo "-- strings markers --"
if command -v strings >/dev/null 2>&1; then
  strings "$bin" | grep -E "trt_datasystem_mget_probe|TrtllmDatasystemSetGet|trtllm_cpp_datasystem_set_get" || true
else
  echo "strings not found"
fi
echo "-- help flag check --"
"$bin" --help 2>&1 | grep -E "trt_datasystem_mget_probe|Usage:" || true
'
else
  echo "No inference pod found; skipped binary self-check."
fi

section "pairec brpc config"
run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- \
  sh -lc 'grep -n "brpc_endpoint" /app/configs/pairec_config.json || true'

section "pairec DNS and TCP checks"
run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- \
  env INFERENCE_SERVICE="$INFERENCE_DEPLOYMENT" \
    NAMESPACE="$NAMESPACE" \
    INFERENCE_SVC_IP="$INFERENCE_SVC_IP" \
    INFERENCE_ENDPOINT_IP="$INFERENCE_ENDPOINT_IP" \
    sh -lc '
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

tcp_check() {
  h="$1"
  p="$2"
  label="$3"
  if [ -z "$h" ] || [ -z "$p" ] || [ "$h" = "$p" ]; then
    echo "-- ${label}: skipped, empty host/port"
    return
  fi
  echo "-- ${label}: ${h}:${p} --"
  if command -v nc >/dev/null 2>&1; then
    nc -vz -w 3 "$h" "$p" || true
  elif command -v telnet >/dev/null 2>&1; then
    echo quit | telnet "$h" "$p" || true
  elif command -v bash >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    timeout 3 bash -c ":</dev/tcp/${h}/${p}" && echo "tcp ok" || echo "tcp failed"
  else
    echo "nc/telnet/bash+timeout not found in PaiRec image; skipping raw TCP check"
  fi
}

echo "-- tcp check matrix --"
tcp_check "$host" "$port" "configured brpc_endpoint"
tcp_check "$INFERENCE_SERVICE" "18100" "service short name"
tcp_check "${INFERENCE_SERVICE}.${NAMESPACE}.svc.cluster.local" "18100" "service fqdn"
tcp_check "$INFERENCE_SVC_IP" "18100" "service clusterIP"
tcp_check "$INFERENCE_ENDPOINT_IP" "18100" "endpoint podIP"
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

section "recent inference brpc and datasystem logs"
kubectl -n "$NAMESPACE" logs "deploy/${INFERENCE_DEPLOYMENT}" -c "$INFERENCE_CONTAINER" --since="$SINCE" 2>/dev/null \
  | grep -E 'Unknown argument|method=Recommend|method=TrtllmDatasystemSetGet|trtllm_datasystem_set_get_probe|brpc inference server listening|Init KvCache|Rank 0 is using GPU|with host endpoint|with ServiceDiscovery|\[Datasystem\]\[TRACE\]' \
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
- If imageID has the right binary markers but logs still say "Unknown argument",
  check for a brpc-dev-bin hostPath overlay that replaces the image binary.
- If direct brpc smoke succeeds but PaiRec request does not produce method=Recommend, the break is PaiRec config/client/DNS/TCP.
- If PaiRec trace shows brpc_calls=0 and inference has no method=Recommend log, PaiRec did not reach the C++ server.
- If TCP is open but PaiRec still does not produce method=Recommend, run:
  bash scripts/test_go_brpc_client_probe.sh
- If inference logs show method=Recommend but PaiRec returns code=299, the issue is recommendation quality/item count, not brpc routing.
- If inference logs show method=TrtllmDatasystemSetGet, the real trtllm_cpp request executed the Set/Get probe.
- If inference logs show Init KvCache host=141.61.91.189, the current inference -> DataSystem endpoint is 189.
EOF
