#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
NODE="${NODE:-worker1}"
RANK_DEPLOYMENT="${RANK_DEPLOYMENT:-deepfm-rank-brpc-worker1}"
WRAPPER_DEPLOYMENT="${WRAPPER_DEPLOYMENT:-deepfm-rank-burst-wrapper}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"
ROLLBACK_RANK_DEPLOYMENT="${ROLLBACK_RANK_DEPLOYMENT:-deepfm-rank-brpc}"
RANK_MANIFEST="${RANK_MANIFEST:-k8s/deployment-deepfm-rank-brpc-worker1.yaml}"
WRAPPER_MANIFEST="${WRAPPER_MANIFEST:-k8s/deployment-deepfm-rank-burst-wrapper-worker1.yaml}"
BACKEND_REPO_DIR="${BACKEND_REPO_DIR:-/home/zcx/workspace/pairec4tigerllm-f19}"
DEEPFM_MODEL_DIR="${DEEPFM_MODEL_DIR:-/home/zcx/workspace/pairec4tigerllm/deepfm_out}"
DEEPFM_MODEL_ROLE="${DEEPFM_MODEL_ROLE:-engineering}"
WRAPPER_HOST_BIN="${WRAPPER_HOST_BIN:-/home/zcx/bin/brpc_rank_burst_wrapper}"
RANK_ADAPTER_HOST_BIN="${RANK_ADAPTER_HOST_BIN:-/home/zcx/bin/brpc_deepfm_rank_adapter}"
PIPELINE_CLIENT_HOST_BIN="${PIPELINE_CLIENT_HOST_BIN:-/home/zcx/bin/brpc_pipeline_client}"
RANK_KVC_BURST_HOST_BIN="${RANK_KVC_BURST_HOST_BIN:-/home/zcx/bin/kvc_burst_wrapper}"
RANK_KVC_RUNTIME_HOST_DIR="${RANK_KVC_RUNTIME_HOST_DIR:-/home/zcx/rank-kvc-runtime}"
RANK_KVC_RUNTIME_POD_DIR="${RANK_KVC_RUNTIME_POD_DIR:-/opt/pairec-rank-runtime}"
RANK_KVC_CONCURRENCY="${RANK_KVC_CONCURRENCY:-32}"
RANK_KVC_PRESSURE_KEY_COUNT="${RANK_KVC_PRESSURE_KEY_COUNT:-4}"
RANK_REVERSE_BURST_ENDPOINT="${RANK_REVERSE_BURST_ENDPOINT:-}"
WORKER_SSH="${WORKER_SSH:-root@192.168.100.11}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/deepfm-rank-burst-worker1/$(date +%Y%m%d-%H%M%S)}"

die() { echo "ERROR: $*" >&2; exit 1; }
ready_pod() {
  kubectl -n "$NAMESPACE" get pods -l "app=$1" -o json | python3 -c '
import json,sys
ready=[]
for pod in json.load(sys.stdin).get("items",[]):
 status=pod.get("status",{}); containers=status.get("containerStatuses",[])
 expected=len(pod.get("spec",{}).get("containers",[]))
 if (status.get("phase")=="Running" and expected > 0 and
     len(containers)==expected and all(x.get("ready") for x in containers)):
  ready.append((pod["metadata"].get("creationTimestamp",""),pod["metadata"]["name"]))
assert ready, "no ready pod for app="+sys.argv[1]
print(max(ready)[1])
' "$1"
}

for command in kubectl python3 scp ssh sha256sum; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
[[ "$DEEPFM_MODEL_ROLE" = engineering || "$DEEPFM_MODEL_ROLE" = production_candidate ]] \
  || die "invalid DEEPFM_MODEL_ROLE=$DEEPFM_MODEL_ROLE"
[[ "$RANK_KVC_CONCURRENCY" = 1 || "$RANK_KVC_CONCURRENCY" = 32 ]] \
  || die "RANK_KVC_CONCURRENCY must be 1 or 32"
if [[ "$RANK_KVC_CONCURRENCY" = 1 ]]; then
  [[ "$RANK_KVC_PRESSURE_KEY_COUNT" = 0 ]] \
    || die "Rank KVC c1 requires RANK_KVC_PRESSURE_KEY_COUNT=0"
else
  [[ "$RANK_KVC_PRESSURE_KEY_COUNT" = 4 ]] \
    || die "Rank KVC c32 requires RANK_KVC_PRESSURE_KEY_COUNT=4"
fi
if [[ -n "$RANK_REVERSE_BURST_ENDPOINT" && \
      ! "$RANK_REVERSE_BURST_ENDPOINT" =~ ^[^:[:space:]]+:[0-9]+$ ]]; then
  die "RANK_REVERSE_BURST_ENDPOINT must be empty or host:port"
fi
test -f "$RANK_MANIFEST" || die "missing manifest: $RANK_MANIFEST"
test -f "$WRAPPER_MANIFEST" || die "missing manifest: $WRAPPER_MANIFEST"
mkdir -p "$OUTPUT_DIR"

echo "== Worker1 and wrapper binary preflight =="
kubectl get node "$NODE" >/dev/null
ssh "$WORKER_SSH" "test -x '$WRAPPER_HOST_BIN' && sha256sum '$WRAPPER_HOST_BIN'" \
  | tee "$OUTPUT_DIR/wrapper-host.sha256"
ssh "$WORKER_SSH" "test -x '$RANK_ADAPTER_HOST_BIN' && sha256sum '$RANK_ADAPTER_HOST_BIN'" \
  | tee "$OUTPUT_DIR/rank-adapter-host.sha256"
ssh "$WORKER_SSH" "test -x '$PIPELINE_CLIENT_HOST_BIN' && sha256sum '$PIPELINE_CLIENT_HOST_BIN'" \
  | tee "$OUTPUT_DIR/pipeline-client-host.sha256"
ssh "$WORKER_SSH" "test -x '$RANK_KVC_BURST_HOST_BIN' && sha256sum '$RANK_KVC_BURST_HOST_BIN'" \
  | tee "$OUTPUT_DIR/rank-kvc-burst-host.sha256"
ssh "$WORKER_SSH" "test -d '$BACKEND_REPO_DIR'" \
  || die "worker1 backend repository is missing: $BACKEND_REPO_DIR"
for artifact in deepfm_best.pt feature_vocab.json user_profiles.json item_categories.json; do
  ssh "$WORKER_SSH" "test -s '$DEEPFM_MODEL_DIR/$artifact'" \
    || die "worker1 DeepFM model artifact is missing or empty: $DEEPFM_MODEL_DIR/$artifact"
done
echo "backend_repo_dir=$BACKEND_REPO_DIR model_dir=$DEEPFM_MODEL_DIR model_role=$DEEPFM_MODEL_ROLE" \
  | tee "$OUTPUT_DIR/worker1-paths.txt"

echo "== Deploy isolated worker1 Rank backend =="
python3 - "$RANK_MANIFEST" "$OUTPUT_DIR/rank.yaml" "$BACKEND_REPO_DIR" \
  "$DEEPFM_MODEL_DIR" "$DEEPFM_MODEL_ROLE" <<'PY'
import pathlib,sys
source,target,repo,model,role=sys.argv[1:]
text=pathlib.Path(source).read_text()
for old,new in {"__BACKEND_REPO_DIR__":repo,"__DEEPFM_MODEL_DIR__":model,
                "__DEEPFM_MODEL_ROLE__":role}.items(): text=text.replace(old,new)
assert "__" not in text
pathlib.Path(target).write_text(text)
PY
kubectl apply -f "$OUTPUT_DIR/rank.yaml"
# The adapter and pipeline client are hostPath File mounts. Binary shipping uses
# atomic rename, so existing pods keep the old inode until they are recreated.
kubectl -n "$NAMESPACE" rollout restart "deployment/$RANK_DEPLOYMENT"
kubectl -n "$NAMESPACE" rollout status "deployment/$RANK_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
RANK_POD="$(ready_pod "$RANK_DEPLOYMENT")"
RANK_IP="$(kubectl -n "$NAMESPACE" get service "$RANK_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}')"
[[ -n "$RANK_IP" && "$RANK_IP" != None ]] || die "worker1 Rank service has no ClusterIP"

echo "== Deploy Rank burst wrapper =="
INFERENCE_POD="$(ready_pod "$INFERENCE_DEPLOYMENT")"
INFERENCE_RUNTIME_ENV="$OUTPUT_DIR/inference-runtime.env"
kubectl -n "$NAMESPACE" exec "$INFERENCE_POD" -c "$INFERENCE_CONTAINER" -- env \
  >"$INFERENCE_RUNTIME_ENV"
INFERENCE_HOST_IP="$(kubectl -n "$NAMESPACE" get pod "$INFERENCE_POD" \
  -o jsonpath='{.status.hostIP}')"
[[ -n "$INFERENCE_HOST_IP" ]] || die "inference Pod has no status.hostIP: $INFERENCE_POD"
printf 'HOST_IP=%s\n' "$INFERENCE_HOST_IP" >>"$INFERENCE_RUNTIME_ENV"
ssh "$WORKER_SSH" "mkdir -p '$RANK_KVC_RUNTIME_HOST_DIR'"
for library in block_ds_consumer.so stub_gpu.so; do
  local_library="$OUTPUT_DIR/$library"
  remote_library="$RANK_KVC_RUNTIME_HOST_DIR/$library"
  kubectl -n "$NAMESPACE" exec "$INFERENCE_POD" -c "$INFERENCE_CONTAINER" -- \
    cat "/opt/pairec/lib/$library" >"$local_library"
  [[ -s "$local_library" ]] || die "empty inference runtime library: $library"
  local_sha="$(sha256sum "$local_library" | awk '{print $1}')"
  scp "$local_library" "${WORKER_SSH}:${remote_library}.part"
  remote_sha="$(ssh "$WORKER_SSH" "sha256sum '${remote_library}.part'" | awk '{print $1}')"
  [[ "$local_sha" = "$remote_sha" ]] \
    || die "Rank KVC runtime library checksum mismatch: $library"
  ssh "$WORKER_SSH" \
    "chmod 0755 '${remote_library}.part' && mv -f '${remote_library}.part' '$remote_library'"
  echo "rank_kvc_runtime_library=$remote_library sha256=$local_sha"
done
python3 - "$WRAPPER_MANIFEST" "$OUTPUT_DIR/wrapper.yaml" "${RANK_IP}:18211" \
  "$RANK_KVC_CONCURRENCY" "$RANK_KVC_PRESSURE_KEY_COUNT" \
  "$INFERENCE_RUNTIME_ENV" "$RANK_KVC_RUNTIME_POD_DIR" \
  "$RANK_KVC_RUNTIME_HOST_DIR" "$RANK_REVERSE_BURST_ENDPOINT" <<'PY'
import json,pathlib,re,sys
text=pathlib.Path(sys.argv[1]).read_text()
for old,new in {
    "__RANK_BACKEND_ENDPOINT__":sys.argv[3],
    "__RANK_KVC_CONCURRENCY__":sys.argv[4],
    "__RANK_KVC_PRESSURE_KEY_COUNT__":sys.argv[5],
    "__RANK_KVC_RUNTIME_POD_DIR__":sys.argv[7],
    "__RANK_KVC_RUNTIME_HOST_DIR__":sys.argv[8],
}.items(): text=text.replace(old,new)
reverse_pattern=r"(?m)^(\s*-\s+--reverse_burst_endpoint=).*$"
if len(re.findall(reverse_pattern, text)) != 1:
    raise RuntimeError("expected exactly one reverse_burst_endpoint argument")
text=re.sub(reverse_pattern, lambda match: match.group(1) + sys.argv[9], text)
runtime={}
for line in pathlib.Path(sys.argv[6]).read_text().splitlines():
    name,separator,value=line.partition("=")
    if separator: runtime[name]=value
preload_tokens=[]
for token in re.split(r"[\s:]+", runtime.get("LD_PRELOAD", "")):
    if not token or "libnvidia-ml.so" in token: continue
    name=pathlib.PurePosixPath(token).name
    if name in ("block_ds_consumer.so", "stub_gpu.so"):
        token=str(pathlib.PurePosixPath(sys.argv[7]) / name)
    preload_tokens.append(token)
runtime["LD_PRELOAD"]=" ".join(preload_tokens)
required_preloads=("block_ds_consumer.so", "stub_gpu.so", "libabseil_dll.so")
missing=[name for name in required_preloads if name not in runtime["LD_PRELOAD"]]
if missing:
    raise RuntimeError("inference LD_PRELOAD lacks required Rank KVC libraries: "
                       + ",".join(missing))
required_env=("HOST_IP", "LD_LIBRARY_PATH", "LD_PRELOAD")
missing_env=[name for name in required_env if not runtime.get(name)]
if missing_env:
    raise RuntimeError("inference runtime environment is empty: " + ",".join(missing_env))
runtime_names={"HOST_IP", "LD_LIBRARY_PATH", "LD_PRELOAD", "NVIDIA_DRIVER_CAPABILITIES"}
runtime_names.update(name for name in runtime if name.startswith("DATASYSTEM_"))
inherited=[(name,runtime[name]) for name in sorted(runtime_names) if runtime.get(name)]
indent="            "
rendered=[]
for name,value in inherited:
    rendered.append(indent + "- name: " + json.dumps(name))
    rendered.append(indent + "  value: " + json.dumps(value))
marker=indent + "# __RANK_KVC_RUNTIME_ENV__"
if text.count(marker) != 2:
    raise RuntimeError("expected two Rank KVC runtime environment markers")
text=text.replace(marker, "\n".join(rendered))
assert "__" not in text
pathlib.Path(sys.argv[2]).write_text(text, encoding="utf-8")
print("Rank KVC runtime environment inherited from inference Pod; LD_PRELOAD="
      + runtime["LD_PRELOAD"])
print("Rank reverse BRPC endpoint=" + (sys.argv[9] or "<disabled>"))
PY
kubectl apply -f "$OUTPUT_DIR/wrapper.yaml"
kubectl -n "$NAMESPACE" rollout restart "deployment/$WRAPPER_DEPLOYMENT"
kubectl -n "$NAMESPACE" rollout status "deployment/$WRAPPER_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
WRAPPER_POD="$(ready_pod "$WRAPPER_DEPLOYMENT")"

echo "== Model identity against retained master rollback Rank =="
ROLLBACK_POD="$(ready_pod "$ROLLBACK_RANK_DEPLOYMENT")"
for artifact in deepfm_best.pt feature_vocab.json user_profiles.json item_categories.json; do
  old="$(kubectl -n "$NAMESPACE" exec "$ROLLBACK_POD" -c backend -- sha256sum "/models/deepfm/$artifact" | awk '{print $1}')"
  new="$(kubectl -n "$NAMESPACE" exec "$RANK_POD" -c backend -- sha256sum "/models/deepfm/$artifact" | awk '{print $1}')"
  echo "$artifact old=$old new=$new" | tee -a "$OUTPUT_DIR/model-identity.txt"
  [[ "$old" = "$new" ]] || die "model identity mismatch: $artifact"
done

echo "== CPU placement diagnostics (non-blocking) =="
for pod in "$RANK_POD" "$WRAPPER_POD"; do
  qos="$(kubectl -n "$NAMESPACE" get pod "$pod" \
    -o jsonpath='{.status.qosClass}' 2>/dev/null || true)"
  qos="${qos:-unavailable}"
  echo "pod=$pod qos=$qos" | tee -a "$OUTPUT_DIR/cpu-isolation.txt"
done
python3 - "$NAMESPACE" "$RANK_POD" "$WRAPPER_POD" "$INFERENCE_POD" \
  "$OUTPUT_DIR/cpu-isolation.json" <<'PY'
import json,subprocess,sys
namespace,rank_pod,wrapper_pod,inference_pod,output=sys.argv[1:]
targets=[(rank_pod,"adapter","rank_adapter"),(rank_pod,"backend","rank_backend"),
         (wrapper_pod,"rank-burst-wrapper","rank_wrapper")]
errors=[]
discovery=subprocess.run(
 ["kubectl","-n",namespace,"get","pod",inference_pod,"-o","json"],
 text=True,capture_output=True)
if discovery.returncode == 0:
 raw=json.loads(discovery.stdout)
 for item in raw["spec"]["containers"]:
  targets.append((inference_pod,item["name"],"inference_"+item["name"]))
else:
 errors.append({"component":"inference_discovery",
                "error":discovery.stderr.strip() or "kubectl get pod failed"})
def expand(value):
 out=set()
 for part in value.split(","):
  ends=part.split("-",1); lo=int(ends[0]); hi=int(ends[-1]); out.update(range(lo,hi+1))
 return out
rows={}
for pod,container,name in targets:
 completed=subprocess.run(
  ["kubectl","-n",namespace,"exec",pod,"-c",container,"--",
   "sh","-c","grep Cpus_allowed_list /proc/1/status"],
  text=True,capture_output=True)
 if completed.returncode != 0:
  error=completed.stderr.strip() or "CPU placement command failed"
  rows[name]={"available":False,"pod":pod,"container":container,"error":error}
  errors.append({"component":name,"error":error})
  continue
 value=completed.stdout.split(":",1)[1].strip()
 rows[name]={"available":True,"cpulist":value,"cpus":sorted(expand(value))}
rank_names=["rank_adapter","rank_backend","rank_wrapper"]
inference_names=[name for name in rows if name.startswith("inference_")]
conflicts=[]
for index,left in enumerate(rank_names):
 for right in rank_names[index+1:]+inference_names:
  if "cpus" not in rows.get(left,{}) or "cpus" not in rows.get(right,{}): continue
  overlap=sorted(set(rows[left]["cpus"]) & set(rows[right]["cpus"]))
  if overlap: conflicts.append({"left":left,"right":right,"overlap":overlap})
result={"valid":not conflicts and not errors,"components":rows,
        "conflicts":conflicts,"errors":errors}
open(output,"w").write(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
PY
CPU_ISOLATION_VALID="$(python3 -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["valid"])).lower())' \
  "$OUTPUT_DIR/cpu-isolation.json")"
echo "cpu_isolation_valid=$CPU_ISOLATION_VALID" | tee -a "$OUTPUT_DIR/cpu-isolation.txt"
if [[ "$CPU_ISOLATION_VALID" != true ]]; then
  echo "WARNING: Rank CPU sets overlap with Rank or inference/KVC containers; continuing with CPU isolation as a diagnostic only" \
    | tee -a "$OUTPUT_DIR/cpu-isolation.txt" >&2
fi

echo "== Wrapper binary identity and local Health =="
expected="$(awk '{print $1}' "$OUTPUT_DIR/wrapper-host.sha256")"
mounted="$(kubectl -n "$NAMESPACE" exec "$WRAPPER_POD" -c rank-burst-wrapper -- sha256sum /proc/1/exe | awk '{print $1}')"
[[ "$expected" = "$mounted" ]] || die "running Rank wrapper binary does not match worker host binary"
expected="$(awk '{print $1}' "$OUTPUT_DIR/rank-adapter-host.sha256")"
mounted="$(kubectl -n "$NAMESPACE" exec "$RANK_POD" -c adapter -- sha256sum /proc/1/exe | awk '{print $1}')"
[[ "$expected" = "$mounted" ]] || die "running Rank adapter binary does not match worker host binary"
expected="$(awk '{print $1}' "$OUTPUT_DIR/pipeline-client-host.sha256")"
mounted="$(kubectl -n "$NAMESPACE" exec "$RANK_POD" -c adapter -- sha256sum /opt/pairec-brpc/bin/brpc_pipeline_client | awk '{print $1}')"
[[ "$expected" = "$mounted" ]] || die "mounted pipeline client does not match worker host binary"
expected="$(awk '{print $1}' "$OUTPUT_DIR/rank-kvc-burst-host.sha256")"
mounted="$(kubectl -n "$NAMESPACE" exec "$WRAPPER_POD" -c rank-kvc-burst-wrapper -- sha256sum /proc/1/exe | awk '{print $1}')"
[[ "$expected" = "$mounted" ]] || die "running Rank KVC sidecar does not match worker host binary"
kubectl -n "$NAMESPACE" exec "$WRAPPER_POD" -c rank-kvc-burst-wrapper -- \
  test -s /run/pairec-rank-kvc-burst/ready \
  || die "Rank KVC sidecar is not ready"
kubectl -n "$NAMESPACE" logs "$WRAPPER_POD" -c rank-burst-wrapper --tail=200 \
  | grep -F '"event":"rank_kvc_preflight_complete"' \
  | tail -1 | tee "$OUTPUT_DIR/rank-kvc-preflight.json"
kubectl -n "$NAMESPACE" exec "$WRAPPER_POD" -c rank-burst-wrapper -- \
  /opt/pairec-brpc/bin/brpc_pipeline_client --server=127.0.0.1:18213 --service=rank --timeout_ms=1000

echo "DEEPFM_RANK_BURST_WORKER1_DEPLOYMENT_OK"
echo "RANK_KVC_WRAPPER_READY concurrency=$RANK_KVC_CONCURRENCY object_size_bytes=8388608 shared_client=true"
echo "rank_pod=$RANK_POD wrapper_pod=$WRAPPER_POD rank_wrapper_endpoint=$(kubectl -n "$NAMESPACE" get service "$WRAPPER_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}'):18213"
echo "output_dir=$OUTPUT_DIR"
