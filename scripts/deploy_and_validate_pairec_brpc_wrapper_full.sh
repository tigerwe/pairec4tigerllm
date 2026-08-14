#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="pairec-brpc-observed-wrapper"
PAIREC_CONFIGMAP="pairec-config-brpc-observed-wrapper"
WRAPPER_DEPLOYMENT="${WRAPPER_DEPLOYMENT:-brpc-burst-wrapper}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
VECTOR_DEPLOYMENT="${VECTOR_DEPLOYMENT:-vector-recall-brpc}"
RANK_DEPLOYMENT="${RANK_DEPLOYMENT:-deepfm-rank-brpc}"
WRAPPER_ENDPOINT="${WRAPPER_ENDPOINT:-192.168.100.11:18103}"
BURST_CONCURRENCY="${BURST_CONCURRENCY:-1}"
BURST_POOL_SIZE="${BURST_POOL_SIZE:-$BURST_CONCURRENCY}"
BURST_ACTIVE_CONNECTIONS="${BURST_ACTIVE_CONNECTIONS:-$BURST_CONCURRENCY}"
BURST_CPU_SHARDS="${BURST_CPU_SHARDS:-[]}"
BURST_PAYLOAD_BYTES="${BURST_PAYLOAD_BYTES:-102400}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
QUALIFICATION_REQUESTS="${QUALIFICATION_REQUESTS:-0}"
REQUESTS="${REQUESTS:-3}"
USER_ID="${USER_ID:-1}"
SCENE_ID="${SCENE_ID:-home_feed}"
SIZE="${SIZE:-10}"
DEEPFM_MODEL_ROLE="${DEEPFM_MODEL_ROLE:-engineering}"
BUILD_PAIREC_IMAGE="${BUILD_PAIREC_IMAGE:-1}"
IMPORT_PAIREC_IMAGE="${IMPORT_PAIREC_IMAGE:-1}"
PAIREC_IMAGE="${PAIREC_IMAGE:-docker.io/library/pairec-server:k8s-arm64-brpc-v1}"
ENSURE_PAUSE_IMAGE="${ENSURE_PAUSE_IMAGE:-1}"
PAUSE_IMAGE="${PAUSE_IMAGE:-docker.io/library/pause-aarch64:3.8}"
PAUSE_ARCHIVE="${PAUSE_ARCHIVE:-/home/zcx/pause-aarch64-3.8.tar}"
PAUSE_FALLBACK_ARCHIVE="${PAUSE_FALLBACK_ARCHIVE:-/home/zcx/master-runtime-images.tar}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/pairec_config.brpc_wrapper_full.json}"
PAIREC_MANIFEST="${PAIREC_MANIFEST:-k8s/deployment-pairec-brpc-observed-wrapper.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-full/$(date +%Y%m%d-%H%M%S)-c${BURST_CONCURRENCY}-n${REQUESTS}}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"
COMPLETION_TIMEOUT_SECONDS="${COMPLETION_TIMEOUT_SECONDS:-30}"
SERVICE_READY_TIMEOUT_SECONDS="${SERVICE_READY_TIMEOUT_SECONDS:-60}"
CPU_THROTTLED_PERIOD_LIMIT_PCT="${CPU_THROTTLED_PERIOD_LIMIT_PCT:-5}"
CPU_THROTTLED_GATE_MIN_PERIODS="${CPU_THROTTLED_GATE_MIN_PERIODS:-100}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "$BURST_CONCURRENCY" =~ ^(1|1000)$ ]] \
  || die "BURST_CONCURRENCY must be 1 or 1000"
[[ "$BURST_POOL_SIZE" =~ ^[1-9][0-9]*$ ]] && (( BURST_POOL_SIZE <= 10000 )) \
  || die "BURST_POOL_SIZE must be in [1,10000]"
[[ "$BURST_ACTIVE_CONNECTIONS" =~ ^[1-9][0-9]*$ ]] && \
  (( BURST_ACTIVE_CONNECTIONS <= 1000 && BURST_ACTIVE_CONNECTIONS <= BURST_POOL_SIZE )) \
  || die "BURST_ACTIVE_CONNECTIONS must be in [1,min(1000,pool_size)]"
[[ "$BURST_PAYLOAD_BYTES" =~ ^[0-9]+$ ]] && (( BURST_PAYLOAD_BYTES <= 1048576 )) \
  || die "BURST_PAYLOAD_BYTES must be in [0,1048576]"
python3 -c 'import json,sys; value=json.loads(sys.argv[1]); assert isinstance(value,list); assert all(isinstance(x,int) and x>=0 for x in value); assert len(value)==len(set(value))' \
  "$BURST_CPU_SHARDS" || die "BURST_CPU_SHARDS must be a JSON array of unique non-negative CPU IDs"
[[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "WARMUP_REQUESTS must be non-negative"
[[ "$QUALIFICATION_REQUESTS" =~ ^[0-9]+$ ]] || die "QUALIFICATION_REQUESTS must be non-negative"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$SIZE" =~ ^[1-9][0-9]*$ ]] || die "SIZE must be positive"
[[ "$SERVICE_READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || die "SERVICE_READY_TIMEOUT_SECONDS must be positive"
[[ "$CPU_THROTTLED_PERIOD_LIMIT_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "CPU_THROTTLED_PERIOD_LIMIT_PCT must be numeric"
[[ "$CPU_THROTTLED_GATE_MIN_PERIODS" =~ ^[1-9][0-9]*$ ]] \
  || die "CPU_THROTTLED_GATE_MIN_PERIODS must be positive"
for flag in "$BUILD_PAIREC_IMAGE" "$IMPORT_PAIREC_IMAGE" "$ENSURE_PAUSE_IMAGE"; do
  [[ "$flag" = 0 || "$flag" = 1 ]] || die "boolean flags must be 0 or 1"
done
for command in kubectl curl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
if [[ "$BUILD_PAIREC_IMAGE" = 1 || "$IMPORT_PAIREC_IMAGE" = 1 || \
      "$ENSURE_PAUSE_IMAGE" = 1 ]]; then
  command -v docker >/dev/null 2>&1 || die "missing command: docker"
fi
if [[ "$IMPORT_PAIREC_IMAGE" = 1 || "$ENSURE_PAUSE_IMAGE" = 1 ]]; then
  command -v ctr >/dev/null 2>&1 || die "missing command: ctr"
fi
test -f "$CONFIG_TEMPLATE" || die "missing config template: $CONFIG_TEMPLATE"
test -f "$PAIREC_MANIFEST" || die "missing manifest: $PAIREC_MANIFEST"
mkdir -p "$OUTPUT_DIR"

ctr_k8s() {
  if (( EUID == 0 )); then
    ctr -n k8s.io "$@"
    return
  fi
  command -v sudo >/dev/null 2>&1 || die "ctr requires root and sudo is unavailable"
  sudo -n ctr -n k8s.io "$@"
}

ctr_k8s_has_image() {
  local listing
  listing="$(ctr_k8s images list)"
  awk 'NR > 1 {print $1}' <<<"$listing" | grep -Fxq "$1"
}

ensure_pause_image() {
  [[ "$ENSURE_PAUSE_IMAGE" = 1 ]] || return
  local archive import_log

  echo "== Ensure Kubernetes sandbox image in k8s.io containerd =="
  if ctr_k8s_has_image "$PAUSE_IMAGE"; then
    echo "K8S_PAUSE_IMAGE_PRESENT image=$PAUSE_IMAGE"
    return
  fi

  import_log="$OUTPUT_DIR/pause-image-import.log"
  for archive in "$PAUSE_ARCHIVE" "$PAUSE_FALLBACK_ARCHIVE"; do
    [[ -f "$archive" ]] || continue
    echo "Trying sandbox archive: $archive"

    if ctr_k8s images import "$archive" >"$import_log" 2>&1; then
      cat "$import_log"
    else
      echo "WARN: direct ctr import failed for $archive" >&2
      cat "$import_log" >&2
      if docker load -i "$archive" >>"$import_log" 2>&1 && \
          docker image inspect "$PAUSE_IMAGE" >/dev/null 2>&1 && \
          docker save "$PAUSE_IMAGE" | ctr_k8s images import - >>"$import_log" 2>&1; then
        echo "Repacked sandbox image through Docker: $archive"
      else
        echo "WARN: Docker repack failed for $archive" >&2
        tail -80 "$import_log" >&2
        continue
      fi
    fi

    if ctr_k8s_has_image "$PAUSE_IMAGE"; then
      echo "K8S_PAUSE_IMAGE_IMPORTED image=$PAUSE_IMAGE archive=$archive"
      return
    fi
    echo "WARN: archive did not create exact sandbox tag: $archive" >&2
  done

  die "sandbox image $PAUSE_IMAGE is missing; neither $PAUSE_ARCHIVE nor $PAUSE_FALLBACK_ARCHIVE produced the required tag (log: $import_log)"
}

ready_pod() {
  local app="$1"
  kubectl -n "$NAMESPACE" get pods -l "app=$app" -o json | python3 -c '
import json, sys
pods=[]
for pod in json.load(sys.stdin).get("items", []):
    status=pod.get("status", {})
    containers=status.get("containerStatuses", [])
    if (not pod.get("metadata", {}).get("deletionTimestamp") and
            status.get("phase")=="Running" and containers and
            all(item.get("ready") for item in containers)):
        pods.append((pod["metadata"].get("creationTimestamp", ""), pod["metadata"]["name"]))
if not pods:
    raise SystemExit(f"no ready pod for app={sys.argv[1]}")
print(max(pods)[1])
' "$app"
}

pod_state() {
  local app="$1" pod
  pod="$(ready_pod "$app")"
  kubectl -n "$NAMESPACE" get pod "$pod" -o json | python3 -c '
import json, sys
pod=json.load(sys.stdin)
statuses=pod["status"].get("containerStatuses", [])
print("pod=" + pod["metadata"]["name"])
print("uid=" + pod["metadata"]["uid"])
print("restarts=" + str(sum(item.get("restartCount", 0) for item in statuses)))
'
}

# Repair CRI before inspecting dependencies: an existing dependency may also be
# waiting for a sandbox when this script is used to recover the cluster.
ensure_pause_image

for app in "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT" \
  "$VECTOR_DEPLOYMENT" "$RANK_DEPLOYMENT"; do
  ready_pod "$app" >/dev/null || die "dependency is not ready: $app"
done

WRAPPER_HOST="${WRAPPER_ENDPOINT%:*}"
WRAPPER_PORT="${WRAPPER_ENDPOINT##*:}"
VECTOR_IP="$(kubectl -n "$NAMESPACE" get service vector-recall-brpc -o jsonpath='{.spec.clusterIP}')"
RANK_IP="$(kubectl -n "$NAMESPACE" get service deepfm-rank-brpc -o jsonpath='{.spec.clusterIP}')"
[[ -n "$WRAPPER_HOST" && -n "$WRAPPER_PORT" && -n "$VECTOR_IP" && -n "$RANK_IP" ]] \
  || die "dependency endpoint is empty"

echo "== Full-chain BRPC Wrapper configuration =="
echo "wrapper_endpoint=$WRAPPER_ENDPOINT burst_concurrency=$BURST_CONCURRENCY pool_size=$BURST_POOL_SIZE active_connections=$BURST_ACTIVE_CONNECTIONS cpu_shards=$BURST_CPU_SHARDS payload_bytes=$BURST_PAYLOAD_BYTES"
echo "vector_endpoint=${VECTOR_IP}:18201 rank_endpoint=${RANK_IP}:18211"
echo "deployment=$PAIREC_DEPLOYMENT output_dir=$OUTPUT_DIR"

if [[ "$BUILD_PAIREC_IMAGE" = 1 ]]; then
  echo "== Build PaiRec binary image =="
  bash scripts/build_pairec_binary_image.sh "$PAIREC_IMAGE"
fi
if [[ "$IMPORT_PAIREC_IMAGE" = 1 ]]; then
  echo "== Import PaiRec image into k8s.io containerd =="
  docker image inspect "$PAIREC_IMAGE" >/dev/null || die "missing image: $PAIREC_IMAGE"
  docker save "$PAIREC_IMAGE" | ctr_k8s images import -
fi

echo "== Render strict full-chain configuration =="
python3 - "$CONFIG_TEMPLATE" "$OUTPUT_DIR/pairec_config.json" \
  "$WRAPPER_ENDPOINT" "$BURST_CONCURRENCY" "${VECTOR_IP}:18201" \
  "${RANK_IP}:18211" "$DEEPFM_MODEL_ROLE" "$BURST_POOL_SIZE" \
  "$BURST_ACTIVE_CONNECTIONS" "$BURST_CPU_SHARDS" "$BURST_PAYLOAD_BYTES" <<'PY'
import json, pathlib, sys
source,target,wrapper,concurrency,vector,rank,role,pool,active,cpu_shards,payload_bytes=sys.argv[1:]
text=pathlib.Path(source).read_text()
for old,new in {
    "__WRAPPER_ENDPOINT__": wrapper,
    "__BURST_CONCURRENCY__": concurrency,
    "__VECTOR_ENDPOINT__": vector,
    "__RANK_ENDPOINT__": rank,
    "__DEEPFM_MODEL_ROLE__": role,
    "__BURST_POOL_SIZE__": pool,
    "__BURST_ACTIVE_CONNECTIONS__": active,
    "__BURST_CPU_SHARDS__": cpu_shards,
    "__BURST_PAYLOAD_BYTES__": payload_bytes,
}.items():
    text=text.replace(old,new)
assert "__" not in text
config=json.loads(text)
recalls={item["Name"]:json.loads(item["RecallAlgo"]) for item in config["RecallConfs"]}
gen=recalls["generative_recall"]
assert gen["protocol"]=="brpc" and gen["brpc_endpoint"]==wrapper
assert gen["brpc_fallback_to_http"] is False and gen["max_retries"]==0
assert gen["brpc_burst_enabled"] is True
assert int(concurrency) in (1,1000)
assert gen["brpc_burst_concurrency"]==int(concurrency)
assert gen["brpc_burst_pool_size"]==int(pool)
assert gen["brpc_burst_active_connections"]==int(active)
assert gen["brpc_burst_cpu_shards"]==json.loads(cpu_shards)
assert gen["brpc_burst_preconnect"] is True
assert gen["brpc_burst_payload_bytes"]==int(payload_bytes)
assert recalls["milvus_recall"]["brpc_endpoint"]==vector
assert config["UserDefineConfs"]["DeepFMRankSorts"][0]["brpc_endpoint"]==rank
rerank=config["UserDefineConfs"]["RerankConfs"][0]
assert rerank["fail_closed"] is True and rerank["minimum_generative"]==1
pathlib.Path(target).write_text(json.dumps(config, indent=2)+"\n")
print("PAIREC_BRPC_WRAPPER_FULL_CONFIG_OK")
PY
kubectl -n "$NAMESPACE" create configmap "$PAIREC_CONFIGMAP" \
  --from-file="pairec_config.json=$OUTPUT_DIR/pairec_config.json" \
  --dry-run=client -o yaml | kubectl apply -f -

python3 - "$PAIREC_MANIFEST" "$OUTPUT_DIR/pairec.yaml" \
  "$WRAPPER_HOST" "$WRAPPER_PORT" "$VECTOR_IP" "$RANK_IP" <<'PY'
import pathlib, sys
source,target,wrapper_host,wrapper_port,vector_host,rank_host=sys.argv[1:]
text=pathlib.Path(source).read_text()
for old,new in {
    "__WRAPPER_HOST__":wrapper_host, "__WRAPPER_PORT__":wrapper_port,
    "__VECTOR_HOST__":vector_host, "__RANK_HOST__":rank_host,
}.items():
    text=text.replace(old,new)
assert "__" not in text
pathlib.Path(target).write_text(text)
PY

echo "== Deploy isolated full-chain Wrapper instance =="
kubectl apply -f "$OUTPUT_DIR/pairec.yaml"
kubectl -n "$NAMESPACE" set image "deployment/$PAIREC_DEPLOYMENT" "pairec=$PAIREC_IMAGE"
kubectl -n "$NAMESPACE" rollout restart "deployment/$PAIREC_DEPLOYMENT"
kubectl -n "$NAMESPACE" rollout status "deployment/$PAIREC_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"

PAIREC_POD="$(ready_pod "$PAIREC_DEPLOYMENT")"
WRAPPER_POD="$(ready_pod "$WRAPPER_DEPLOYMENT")"
INFERENCE_POD="$(ready_pod "$INFERENCE_DEPLOYMENT")"
SERVICE_IP="$(kubectl -n "$NAMESPACE" get service "$PAIREC_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}')"
[[ -n "$SERVICE_IP" && "$SERVICE_IP" != None ]] || die "PaiRec service has no ClusterIP"
PAIREC_URL="http://${SERVICE_IP}:18080/api/recommend"

echo "== Wait for isolated PaiRec Service endpoint =="
service_deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
while true; do
  ready_addresses="$(kubectl -n "$NAMESPACE" get endpoints "$PAIREC_DEPLOYMENT" \
    -o jsonpath='{range .subsets[*].addresses[*]}{.ip}{"\n"}{end}' 2>/dev/null || true)"
  if [[ -n "$ready_addresses" ]] && \
      curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 \
        "http://${SERVICE_IP}:18080/ping" | grep -q success; then
    break
  fi
  (( SECONDS < service_deadline )) \
    || die "service/$PAIREC_DEPLOYMENT did not become reachable"
  sleep 1
done
echo "PAIREC_BRPC_WRAPPER_FULL_SERVICE_READY endpoint=${SERVICE_IP}:18080"

echo "== Verify preconnected c${BURST_CONCURRENCY} sessions =="
ready_line="$(kubectl -n "$NAMESPACE" logs "$PAIREC_POD" -c pairec \
  | grep -F '"event":"pairec_brpc_burst_ready"' | tail -1 || true)"
[[ -n "$ready_line" ]] || die "pairec_brpc_burst_ready is missing"
python3 - "$ready_line" "$BURST_ACTIVE_CONNECTIONS" "$BURST_POOL_SIZE" \
  "$BURST_PAYLOAD_BYTES" <<'PY'
import json, sys
event=json.loads(sys.argv[1][sys.argv[1].index("{"):])
expected=int(sys.argv[2])
assert event["concurrency"]==expected, event
assert event["active_connections"]==expected, event
assert event["connected_sessions"]==event["pool_size"]==int(sys.argv[3]), event
assert event["payload_bytes"]==int(sys.argv[4]), event
connections=event["shard_connections"]
assert max(connections)-min(connections)<=1,event
PY
echo "PAIREC_BRPC_WRAPPER_PRECONNECTED_OK connected_sessions=$BURST_POOL_SIZE active_connections=$BURST_ACTIVE_CONNECTIONS payload_bytes=$BURST_PAYLOAD_BYTES"

if (( QUALIFICATION_REQUESTS > 0 )); then
  echo "== Qualify deterministic workload user before measurement: $QUALIFICATION_REQUESTS requests =="
  mkdir -p "$OUTPUT_DIR/qualification"
  for index in $(seq 1 "$QUALIFICATION_REQUESTS"); do
    response="$OUTPUT_DIR/qualification/response-${index}.json"
    curl --noproxy '*' -fsS --connect-timeout 2 --max-time 10 \
      "$PAIREC_URL" -H 'Content-Type: application/json' \
      -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":$SIZE}" \
      -o "$response"
    python3 - "$response" "$SIZE" <<'PY'
import json,sys
data=json.load(open(sys.argv[1])); size=int(sys.argv[2]); items=data.get("items",[])
assert data.get("code")==200,data
assert len(items)==size and len({item["item_id"] for item in items})==size,data
sources=[item.get("retrieve_id") for item in items]
generative=sources.count("generative_recall")
assert 1<=generative<=min(2,size),data
assert sources==["milvus_recall"]*(size-generative)+["generative_recall"]*generative,data
PY
  done
  echo "PAIREC_BRPC_WRAPPER_WORKLOAD_QUALIFIED user_id=$USER_ID requests=$QUALIFICATION_REQUESTS"
fi

if (( WARMUP_REQUESTS > 0 )); then
  echo "== Warm up full-chain Wrapper instance: $WARMUP_REQUESTS requests (excluded) =="
  mkdir -p "$OUTPUT_DIR/warmup"
  WARMUP_REQUEST_IDS=()
  for index in $(seq 1 "$WARMUP_REQUESTS"); do
    response="$OUTPUT_DIR/warmup/response-${index}.json"
    curl --noproxy '*' -fsS --connect-timeout 2 --max-time 10 \
      "$PAIREC_URL" -H 'Content-Type: application/json' \
      -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":$SIZE}" \
      -o "$response"
    warmup_request_id="$(python3 - "$response" "$SIZE" <<'PY'
import json, sys
data=json.load(open(sys.argv[1])); size=int(sys.argv[2]); items=data.get("items", [])
assert data.get("code")==200, data
assert len(items)==size and len({item["item_id"] for item in items})==size, data
sources=[item.get("retrieve_id") for item in items]
generative=sources.count("generative_recall")
assert 1 <= generative <= min(2,size), data
assert sources==["milvus_recall"]*(size-generative)+["generative_recall"]*generative, data
print(data["request_id"])
PY
    )"
    WARMUP_REQUEST_IDS+=("$warmup_request_id")
    echo "warmup request_id=$warmup_request_id"
  done
  echo "== Wait for warmup pressure completion =="
  warmup_deadline=$((SECONDS + COMPLETION_TIMEOUT_SECONDS))
  for warmup_request_id in "${WARMUP_REQUEST_IDS[@]}"; do
    while ! kubectl -n "$NAMESPACE" logs "$PAIREC_POD" -c pairec 2>/dev/null \
        | grep -F '"event":"pairec_brpc_burst_complete"' \
        | grep -F "\"request_id\":\"${warmup_request_id}\"" >/dev/null; do
      (( SECONDS < warmup_deadline )) \
        || die "warmup burst completion timed out for request_id=$warmup_request_id"
      sleep 0.1
    done
  done
  echo "PAIREC_BRPC_WRAPPER_FULL_WARMUP_OK requests=$WARMUP_REQUESTS"
fi

for app in "$PAIREC_DEPLOYMENT" "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT" \
  "$VECTOR_DEPLOYMENT" "$RANK_DEPLOYMENT"; do
  pod_state "$app" >"$OUTPUT_DIR/${app}.before"
done

collect_cpu_stat() {
  local app="$1" container="$2" output="$3" pod
  pod="$(ready_pod "$app")"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$container" -- /bin/sh -ec \
    'if test -f /sys/fs/cgroup/cpu.stat; then cat /sys/fs/cgroup/cpu.stat; else cat /sys/fs/cgroup/cpu/cpu.stat; fi' \
    >"$output"
}

RESOURCE_TARGETS=(
  "$PAIREC_DEPLOYMENT:pairec"
  "$WRAPPER_DEPLOYMENT:brpc-burst-wrapper"
  "$INFERENCE_DEPLOYMENT:brpc-inference"
)
for tuple in "${RESOURCE_TARGETS[@]}"; do
  IFS=: read -r app container <<<"$tuple"
  collect_cpu_stat "$app" "$container" "$OUTPUT_DIR/${app}-${container}.cpu.before"
done

echo "== Run strict full-chain c${BURST_CONCURRENCY} pressure: $REQUESTS requests =="
STARTED_AT="$(date --iso-8601=seconds)"
WORKLOAD_STARTED_AT="$(date +%s.%N)"
printf 'index\te2e_ms\trequest_id\n' >"$OUTPUT_DIR/requests.tsv"
for index in $(seq 1 "$REQUESTS"); do
  response="$OUTPUT_DIR/response-${index}.json"
  seconds="$(curl --noproxy '*' -sS --connect-timeout 2 --max-time 10 \
    "$PAIREC_URL" -H 'Content-Type: application/json' \
    -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":$SIZE}" \
    -o "$response" -w '%{time_total}')"
  request_id="$(python3 - "$response" "$SIZE" <<'PY'
import json, sys
data=json.load(open(sys.argv[1])); size=int(sys.argv[2]); items=data.get("items", [])
assert data.get("code")==200, data
assert len(items)==size and len({item["item_id"] for item in items})==size, data
sources=[item.get("retrieve_id") for item in items]
generative=sources.count("generative_recall")
assert 1 <= generative <= min(2,size), data
assert sources==["milvus_recall"]*(size-generative)+["generative_recall"]*generative, data
print(data["request_id"])
PY
)"
  e2e_ms="$(python3 -c 'import sys; print(round(float(sys.argv[1])*1000,3))' "$seconds")"
  printf '%s\t%s\t%s\n' "$index" "$e2e_ms" "$request_id" | tee -a "$OUTPUT_DIR/requests.tsv"
done
WORKLOAD_FINISHED_AT="$(date +%s.%N)"
WORKLOAD_ELAPSED_SECONDS="$(python3 -c \
  'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' \
  "$WORKLOAD_STARTED_AT" "$WORKLOAD_FINISHED_AT")"
echo "workload elapsed_seconds=$WORKLOAD_ELAPSED_SECONDS requests=$REQUESTS"

deadline=$((SECONDS + COMPLETION_TIMEOUT_SECONDS))
while true; do
  kubectl -n "$NAMESPACE" logs "$PAIREC_POD" -c pairec --since-time="$STARTED_AT" \
    >"$OUTPUT_DIR/pairec.log"
  kubectl -n "$NAMESPACE" logs "$WRAPPER_POD" -c brpc-burst-wrapper --since-time="$STARTED_AT" \
    >"$OUTPUT_DIR/wrapper.log"
  kubectl -n "$NAMESPACE" logs "$INFERENCE_POD" -c brpc-inference --since-time="$STARTED_AT" \
    >"$OUTPUT_DIR/inference.log"
  if python3 - "$OUTPUT_DIR/requests.tsv" "$OUTPUT_DIR/pairec.log" \
      "$OUTPUT_DIR/inference.log" <<'PY'
import csv,json,pathlib,sys
ids={row["request_id"] for row in csv.DictReader(open(sys.argv[1]),delimiter="\t")}
def events(path):
 out=[]
 for line in pathlib.Path(path).read_text(errors="replace").splitlines():
  pos=line.find("{")
  if pos<0: continue
  try: out.append(json.loads(line[pos:]))
  except json.JSONDecodeError: pass
 return out
p=events(sys.argv[2]); i=events(sys.argv[3])
for rid in ids:
 for name in ("pairec_brpc_burst_start","pairec_brpc_burst_business_complete",
              "pairec_brpc_burst_complete","pipeline_trace_complete",
              "deepfm_rank_complete","source_quota_rerank_complete"):
  assert sum(e.get("event")==name and e.get("request_id")==rid for e in p)==1
 assert sum(e.get("event")=="datasystem_request_complete" and e.get("request_id")==rid for e in i)==1
 assert sum(e.get("event")=="trt_executor_request_complete" and e.get("request_id")==rid for e in i)==1
PY
  then break; fi
  (( SECONDS < deadline )) || die "timed out waiting for request-level completion events"
  sleep 1
done

echo "== Validate request-id contracts and latency evidence =="
python3 - "$OUTPUT_DIR/requests.tsv" "$OUTPUT_DIR/pairec.log" \
  "$OUTPUT_DIR/wrapper.log" "$OUTPUT_DIR/inference.log" "$OUTPUT_DIR/summary.json" \
  "$BURST_ACTIVE_CONNECTIONS" "$WORKLOAD_ELAPSED_SECONDS" "$BURST_POOL_SIZE" <<'PY'
import csv,json,math,pathlib,statistics,sys
requests_path,pairec_path,wrapper_path,inference_path,output,concurrency,elapsed,pool=sys.argv[1:]
expected=int(concurrency)
pool=int(pool)
elapsed=float(elapsed)
rows=list(csv.DictReader(open(requests_path),delimiter="\t"))
ids=[row["request_id"] for row in rows]
def json_events(path):
 out=[]
 for line in pathlib.Path(path).read_text(errors="replace").splitlines():
  pos=line.find("{")
  if pos<0: continue
  try: out.append(json.loads(line[pos:]))
  except json.JSONDecodeError: pass
 return out
p=json_events(pairec_path); native=json_events(inference_path)
wrapper_text=pathlib.Path(wrapper_path).read_text(errors="replace")
samples=[]
rank_reordered_count=0
for row in rows:
 rid=row["request_id"]
 by_name={}
 for event in p:
  if event.get("request_id")==rid: by_name.setdefault(event.get("event"),[]).append(event)
 def one(name):
  values=by_name.get(name,[]); assert len(values)==1,(rid,name,values); return values[0]
 start=one("pairec_brpc_burst_start")
 business=one("pairec_brpc_burst_business_complete")
 complete=one("pairec_brpc_burst_complete")
 pipeline=one("pipeline_trace_complete")
 rank=one("deepfm_rank_complete")
 rerank=one("source_quota_rerank_complete")
 assert start["concurrency"]==start["active_connections"]==expected,start
 assert start["selected_sessions"]==expected,start
 assert start["connected_sessions"]==start["pool_size"]==pool,start
 assert complete["armed_workers"]==expected,complete
 assert complete["pressure_requests"]==expected-1,complete
 assert complete["pressure_success"]==expected-1 and complete["pressure_errors"]==0,complete
 assert complete["pool_size"]==pool and complete["active_connections"]==expected,complete
 assert max(complete["shard_requests"])-min(complete["shard_requests"])<=1,complete
 assert sum(complete["shard_requests"])==expected,complete
 assert complete["shard_requests"]==complete["shard_success"],complete
 assert min(complete["shard_requests"])>0,complete
 assert max(complete["shard_bytes"])-min(complete["shard_bytes"])<=204800,complete
 assert business["business_success"] and business["trace_valid"],business
 assert complete["business_success"] and complete["trace_valid"] and complete["burst_valid"],complete
 assert business["wrapper_total_ms"]>0 and business["wrapper_backend_rpc_ms"]>0,business
 assert pipeline["status"]=="ok" and pipeline["valid"] is True,pipeline
 spans={span["name"]:span for span in pipeline["spans"]}
 for name in ("generative_recall","vector_recall","deepfm_rank"):
  assert spans[name]["protocol"]=="brpc" and spans[name]["status"]=="ok",spans[name]
 assert spans["rerank"]["status"]=="ok",spans["rerank"]
 assert rank["candidate_count"]==50 and rank["service_total_ms"]>0,rank
 rank_reordered_count += int(rank["reordered"] is True)
 assert rerank["status"]=="ok" and rerank["generative_selected"]>=1,rerank
 wrapper_lines=[line for line in wrapper_text.splitlines()
                if "[brpc-burst-wrapper] method=Recommend" in line and f"request_id={rid}" in line]
 assert len(wrapper_lines)==1,(rid,wrapper_lines)
 assert " code=200 " in wrapper_lines[0],wrapper_lines[0]
 ds=[e for e in native if e.get("event")=="datasystem_request_complete" and e.get("request_id")==rid]
 executor=[e for e in native if e.get("event")=="trt_executor_request_complete" and e.get("request_id")==rid]
 assert len(ds)==len(executor)==1,(rid,ds,executor)
 ds=ds[0]; executor=executor[0]
 assert ds.get("attribution_complete") is True and ds.get("phase_timing_complete") is True,ds
 for field in ("get_failed_count","set_failed_count","pending_count","unknown_count","phase_unknown_count"):
  assert int(ds.get(field,-1))==0,(field,ds)
 assert int(ds["native_closure_error_us"])<=100,ds
 assert int(executor["gateway_closure_error_us"])<=100,executor
 samples.append({
  "request_id":rid,"client_e2e_ms":float(row["e2e_ms"]),
  "front_brpc_ms":business["business_front_brpc_ms"],
  "wrapper_total_ms":business["wrapper_total_ms"],
  "wrapper_backend_rpc_ms":business["wrapper_backend_rpc_ms"],
  "runner_ms":business["business_runner_generate_ms"],
  "burst_total_ms":complete["burst_total_ms"],
  "max_active_workers":complete["max_active_workers"],
  "start_skew_us":complete["start_skew_us"],
  "pressure_latency_p95_ms":complete["pressure_latency_p95_ms"],
  "datasystem_get_count":ds["get_count"],"datasystem_get_ms":ds["get_us"]/1000,
  "datasystem_set_count":ds["set_count"],"datasystem_set_ms":ds["set_us"]/1000,
 })
assert rank_reordered_count>0,"DeepFM did not reorder any pressure request"
classification=f"PAIREC_BRPC_WRAPPER_FULL_C{expected}_OK"
def percentile(values,q):
 values=sorted(values); pos=(len(values)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
 return values[lo] if lo==hi else values[lo]+(values[hi]-values[lo])*(pos-lo)
metric_names=("client_e2e_ms","front_brpc_ms","wrapper_total_ms",
              "wrapper_backend_rpc_ms","runner_ms","burst_total_ms",
              "max_active_workers","start_skew_us","pressure_latency_p95_ms")
metrics={}
for name in metric_names:
 values=[float(sample[name]) for sample in samples]
 metrics[name]={"count":len(values),"avg":statistics.fmean(values),
                "p50":percentile(values,.5),"p95":percentile(values,.95),
                "p99":percentile(values,.99),"max":max(values)}
summary={"classification":classification,"concurrency":expected,"pool_size":pool,
         "requests":len(samples),"elapsed_seconds":elapsed,
         "throughput_rps":len(samples)/elapsed,
         "rank_reordered_count":rank_reordered_count,"samples":samples}
summary["metrics"]=metrics
pathlib.Path(output).write_text(json.dumps(summary,indent=2)+"\n")
print("request e2e_ms front_brpc_ms wrapper_ms backend_rpc_ms runner_ms burst_ms active skew_us pressure_p95_ms ds_get ds_set")
for s in samples:
 print(s["request_id"],s["client_e2e_ms"],round(s["front_brpc_ms"],3),
       round(s["wrapper_total_ms"],3),round(s["wrapper_backend_rpc_ms"],3),
       round(s["runner_ms"],3),round(s["burst_total_ms"],3),s["max_active_workers"],
       s["start_skew_us"],round(s["pressure_latency_p95_ms"],3),
       s["datasystem_get_count"],s["datasystem_set_count"])
print("metric count avg p50 p95 p99 max")
for name in metric_names:
 item=metrics[name]
 print(f"{name} {item['count']} {item['avg']:.3f} {item['p50']:.3f} {item['p95']:.3f} {item['p99']:.3f} {item['max']:.3f}")
print(f"throughput_rps={summary['throughput_rps']:.6f}")
PY

if grep -Eqi 'fallback to HTTP|fallback_to_http[^a-zA-Z0-9]+true|brpc request failed|Segmentation|core dumped|Out of memory' \
    "$OUTPUT_DIR/pairec.log" "$OUTPUT_DIR/wrapper.log" "$OUTPUT_DIR/inference.log"; then
  die "fallback, BRPC failure, or crash marker detected"
fi
for app in "$PAIREC_DEPLOYMENT" "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT" \
  "$VECTOR_DEPLOYMENT" "$RANK_DEPLOYMENT"; do
  pod_state "$app" >"$OUTPUT_DIR/${app}.after"
  cmp -s "$OUTPUT_DIR/${app}.before" "$OUTPUT_DIR/${app}.after" \
    || die "pod identity or restart count changed: $app"
done

echo "== CPU throttling gates =="
for tuple in "${RESOURCE_TARGETS[@]}"; do
  IFS=: read -r app container <<<"$tuple"
  collect_cpu_stat "$app" "$container" "$OUTPUT_DIR/${app}-${container}.cpu.after"
  python3 - "$OUTPUT_DIR/${app}-${container}.cpu.before" \
    "$OUTPUT_DIR/${app}-${container}.cpu.after" "$app/$container" \
    "$CPU_THROTTLED_PERIOD_LIMIT_PCT" "$CPU_THROTTLED_GATE_MIN_PERIODS" <<'PY'
import pathlib,sys
def parse(path):
 result={}
 for line in pathlib.Path(path).read_text().splitlines():
  fields=line.split()
  if len(fields)==2: result[fields[0]]=int(fields[1])
 return result
before,after=parse(sys.argv[1]),parse(sys.argv[2])
periods=max(0,after.get("nr_periods",0)-before.get("nr_periods",0))
throttled=max(0,after.get("nr_throttled",0)-before.get("nr_throttled",0))
ratio=100.0*throttled/max(periods,1)
print(f"resource={sys.argv[3]} periods={periods} throttled={throttled} throttled_period_pct={ratio:.3f}")
if periods<int(sys.argv[5]):
 print(f"resource_gate=diagnostic reason=periods_below_{sys.argv[5]}")
else:
 assert ratio<=float(sys.argv[4]),f"CPU throttling gate failed for {sys.argv[3]}: {ratio:.3f}%"
PY
done

echo "== Summary =="
CLASSIFICATION="PAIREC_BRPC_WRAPPER_FULL_C${BURST_CONCURRENCY}_OK"
echo "classification=$CLASSIFICATION"
echo "endpoint=$PAIREC_URL"
echo "warmup_requests=$WARMUP_REQUESTS"
echo "requests=$REQUESTS"
echo "output_dir=$OUTPUT_DIR"
echo "$CLASSIFICATION"
