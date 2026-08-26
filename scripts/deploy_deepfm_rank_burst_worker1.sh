#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
NODE="${NODE:-worker1}"
RANK_DEPLOYMENT="${RANK_DEPLOYMENT:-deepfm-rank-brpc-worker1}"
WRAPPER_DEPLOYMENT="${WRAPPER_DEPLOYMENT:-deepfm-rank-burst-wrapper}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
ROLLBACK_RANK_DEPLOYMENT="${ROLLBACK_RANK_DEPLOYMENT:-deepfm-rank-brpc}"
RANK_MANIFEST="${RANK_MANIFEST:-k8s/deployment-deepfm-rank-brpc-worker1.yaml}"
WRAPPER_MANIFEST="${WRAPPER_MANIFEST:-k8s/deployment-deepfm-rank-burst-wrapper-worker1.yaml}"
BACKEND_REPO_DIR="${BACKEND_REPO_DIR:-/home/zcx/workspace/pairec4tigerllm-f19}"
DEEPFM_MODEL_DIR="${DEEPFM_MODEL_DIR:-/home/zcx/workspace/pairec4tigerllm/deepfm_out}"
DEEPFM_MODEL_ROLE="${DEEPFM_MODEL_ROLE:-engineering}"
WRAPPER_HOST_BIN="${WRAPPER_HOST_BIN:-/home/zcx/bin/brpc_rank_burst_wrapper}"
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
 if status.get("phase")=="Running" and containers and all(x.get("ready") for x in containers):
  ready.append((pod["metadata"].get("creationTimestamp",""),pod["metadata"]["name"]))
assert ready, "no ready pod for app="+sys.argv[1]
print(max(ready)[1])
' "$1"
}

for command in kubectl python3 ssh sha256sum; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
[[ "$DEEPFM_MODEL_ROLE" = engineering || "$DEEPFM_MODEL_ROLE" = production_candidate ]] \
  || die "invalid DEEPFM_MODEL_ROLE=$DEEPFM_MODEL_ROLE"
test -f "$RANK_MANIFEST" || die "missing manifest: $RANK_MANIFEST"
test -f "$WRAPPER_MANIFEST" || die "missing manifest: $WRAPPER_MANIFEST"
mkdir -p "$OUTPUT_DIR"

echo "== Worker1 and wrapper binary preflight =="
kubectl get node "$NODE" >/dev/null
ssh "$WORKER_SSH" "test -x '$WRAPPER_HOST_BIN' && sha256sum '$WRAPPER_HOST_BIN'" \
  | tee "$OUTPUT_DIR/wrapper-host.sha256"
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
kubectl -n "$NAMESPACE" rollout status "deployment/$RANK_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
RANK_POD="$(ready_pod "$RANK_DEPLOYMENT")"
RANK_IP="$(kubectl -n "$NAMESPACE" get service "$RANK_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}')"
[[ -n "$RANK_IP" && "$RANK_IP" != None ]] || die "worker1 Rank service has no ClusterIP"

echo "== Deploy Rank burst wrapper =="
python3 - "$WRAPPER_MANIFEST" "$OUTPUT_DIR/wrapper.yaml" "${RANK_IP}:18211" <<'PY'
import pathlib,sys
text=pathlib.Path(sys.argv[1]).read_text().replace("__RANK_BACKEND_ENDPOINT__",sys.argv[3])
assert "__" not in text
pathlib.Path(sys.argv[2]).write_text(text)
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

echo "== Guaranteed QoS and exclusive CPU gate =="
for pod in "$RANK_POD" "$WRAPPER_POD"; do
  qos="$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.status.qosClass}')"
  echo "pod=$pod qos=$qos" | tee -a "$OUTPUT_DIR/cpu-isolation.txt"
  [[ "$qos" = Guaranteed ]] || die "pod/$pod must have Guaranteed QoS"
done
INFERENCE_POD="$(ready_pod "$INFERENCE_DEPLOYMENT")"
python3 - "$NAMESPACE" "$RANK_POD" "$WRAPPER_POD" "$INFERENCE_POD" \
  "$OUTPUT_DIR/cpu-isolation.json" <<'PY'
import json,subprocess,sys
namespace,rank_pod,wrapper_pod,inference_pod,output=sys.argv[1:]
targets=[(rank_pod,"adapter","rank_adapter"),(rank_pod,"backend","rank_backend"),
         (wrapper_pod,"rank-burst-wrapper","rank_wrapper")]
raw=json.loads(subprocess.check_output(["kubectl","-n",namespace,"get","pod",inference_pod,"-o","json"]))
for item in raw["spec"]["containers"]:
 targets.append((inference_pod,item["name"],"inference_"+item["name"]))
def expand(value):
 out=set()
 for part in value.split(","):
  ends=part.split("-",1); lo=int(ends[0]); hi=int(ends[-1]); out.update(range(lo,hi+1))
 return out
rows={}
for pod,container,name in targets:
 text=subprocess.check_output(["kubectl","-n",namespace,"exec",pod,"-c",container,"--",
                               "sh","-c","grep Cpus_allowed_list /proc/1/status"],text=True)
 value=text.split(":",1)[1].strip(); rows[name]={"cpulist":value,"cpus":sorted(expand(value))}
rank_names=["rank_adapter","rank_backend","rank_wrapper"]
inference_names=[name for name in rows if name.startswith("inference_")]
conflicts=[]
for index,left in enumerate(rank_names):
 for right in rank_names[index+1:]+inference_names:
  overlap=sorted(set(rows[left]["cpus"]) & set(rows[right]["cpus"]))
  if overlap: conflicts.append({"left":left,"right":right,"overlap":overlap})
result={"valid":not conflicts,"components":rows,"conflicts":conflicts}
open(output,"w").write(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
assert not conflicts, "Rank CPU sets overlap with another Rank or inference/KVC container"
PY

echo "== Wrapper binary identity and local Health =="
expected="$(awk '{print $1}' "$OUTPUT_DIR/wrapper-host.sha256")"
mounted="$(kubectl -n "$NAMESPACE" exec "$WRAPPER_POD" -c rank-burst-wrapper -- sha256sum /proc/1/exe | awk '{print $1}')"
[[ "$expected" = "$mounted" ]] || die "running Rank wrapper binary does not match worker host binary"
kubectl -n "$NAMESPACE" exec "$WRAPPER_POD" -c rank-burst-wrapper -- \
  /opt/pairec-brpc/bin/brpc_pipeline_client --server=127.0.0.1:18213 --service=rank --timeout_ms=1000

echo "DEEPFM_RANK_BURST_WORKER1_DEPLOYMENT_OK"
echo "rank_pod=$RANK_POD wrapper_pod=$WRAPPER_POD rank_wrapper_endpoint=$(kubectl -n "$NAMESPACE" get service "$WRAPPER_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}'):18213"
echo "output_dir=$OUTPUT_DIR"
