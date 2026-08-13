#!/usr/bin/env bash
# Collect and classify one full-chain Wrapper rerank failure without changing deployments.
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec-brpc-observed-wrapper}"
WRAPPER_DEPLOYMENT="${WRAPPER_DEPLOYMENT:-brpc-burst-wrapper}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
REQUEST_ID="${REQUEST_ID:-${1:-}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${2:-}}"
LOG_SINCE="${LOG_SINCE:-60m}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/pairec-wrapper-rerank-diagnostic}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d-%H%M%S)}"

die() { echo "ERROR: $*" >&2; exit 1; }
for command in kubectl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
mkdir -p "$OUTPUT_DIR"

if [[ -n "$RUN_OUTPUT_DIR" && ! -d "$RUN_OUTPUT_DIR" ]]; then
  die "RUN_OUTPUT_DIR does not exist: $RUN_OUTPUT_DIR"
fi

if [[ -z "$REQUEST_ID" && -n "$RUN_OUTPUT_DIR" ]]; then
  REQUEST_ID="$(python3 - "$RUN_OUTPUT_DIR" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
failures=[]
for path in root.glob("response-*.json"):
 try:
  data=json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):
  continue
 if data.get("code") != 200 and data.get("request_id"):
  try: index=int(path.stem.rsplit("-",1)[1])
  except ValueError: index=-1
  failures.append((index,path.stat().st_mtime,str(data["request_id"])))
print(max(failures)[2] if failures else "")
PY
)"
fi
[[ -n "$REQUEST_ID" ]] || die "set REQUEST_ID or RUN_OUTPUT_DIR containing a failed response-*.json"

collect_app_logs() {
  local app="$1" preferred_container="$2" output="$3"
  : >"$output"
  mapfile -t pods < <(kubectl -n "$NAMESPACE" get pods -l "app=$app" \
    --sort-by=.metadata.creationTimestamp -o name 2>/dev/null || true)
  for pod_ref in "${pods[@]}"; do
    [[ -n "$pod_ref" ]] || continue
    pod="${pod_ref#pod/}"
    containers="$(kubectl -n "$NAMESPACE" get pod "$pod" \
      -o jsonpath='{.spec.containers[*].name}' 2>/dev/null || true)"
    for container in $containers; do
      [[ -z "$preferred_container" || "$container" = "$preferred_container" ]] || continue
      {
        echo "===== pod=$pod container=$container current ====="
        kubectl -n "$NAMESPACE" logs "$pod" -c "$container" --since="$LOG_SINCE" 2>&1 || true
        restart_count="$(kubectl -n "$NAMESPACE" get pod "$pod" \
          -o "jsonpath={.status.containerStatuses[?(@.name=='$container')].restartCount}" 2>/dev/null || echo 0)"
        if [[ "${restart_count:-0}" != 0 ]]; then
          echo "===== pod=$pod container=$container previous ====="
          kubectl -n "$NAMESPACE" logs "$pod" -c "$container" --previous 2>&1 || true
        fi
      } >>"$output"
    done
  done
}

echo "== Collect full-chain evidence =="
collect_app_logs "$PAIREC_DEPLOYMENT" pairec "$OUTPUT_DIR/pairec.log"
collect_app_logs "$WRAPPER_DEPLOYMENT" brpc-burst-wrapper "$OUTPUT_DIR/wrapper.log"
collect_app_logs "$INFERENCE_DEPLOYMENT" brpc-inference "$OUTPUT_DIR/inference.log"

for source in pairec wrapper inference; do
  grep -F "$REQUEST_ID" "$OUTPUT_DIR/${source}.log" \
    >"$OUTPUT_DIR/${source}-request.log" 2>/dev/null || true
done

if [[ -n "$RUN_OUTPUT_DIR" ]]; then
  find "$RUN_OUTPUT_DIR" -maxdepth 2 -type f \
    \( -name '*.log' -o -name 'response-*.json' -o -name 'requests.tsv' \) \
    -print >"$OUTPUT_DIR/run-files.txt"
  grep -R -F "$REQUEST_ID" "$RUN_OUTPUT_DIR" \
    >"$OUTPUT_DIR/run-request.log" 2>/dev/null || true
else
  : >"$OUTPUT_DIR/run-files.txt"
  : >"$OUTPUT_DIR/run-request.log"
fi

kubectl -n "$NAMESPACE" get deployment "$PAIREC_DEPLOYMENT" \
  "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT" -o json \
  >"$OUTPUT_DIR/deployments.json" 2>"$OUTPUT_DIR/deployments.error" || true
kubectl -n "$NAMESPACE" get pods \
  -l "app in ($PAIREC_DEPLOYMENT,$WRAPPER_DEPLOYMENT,$INFERENCE_DEPLOYMENT)" -o wide \
  >"$OUTPUT_DIR/pods.txt" 2>&1 || true
kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp \
  >"$OUTPUT_DIR/events.txt" 2>&1 || true

python3 - "$REQUEST_ID" "$OUTPUT_DIR" "$RUN_OUTPUT_DIR" <<'PY'
import json,pathlib,re,sys
rid,output_dir,run_dir=sys.argv[1:]
out=pathlib.Path(output_dir)

def text(name):
 return (out/name).read_text(errors="replace") if (out/name).exists() else ""

pairec=text("pairec-request.log")
wrapper=text("wrapper-request.log")
inference=text("inference-request.log")
run=text("run-request.log")
combined="\n".join((pairec,wrapper,inference,run))

events=[]
for source,body in (("pairec",pairec),("wrapper",wrapper),("inference",inference)):
 for line in body.splitlines():
  pos=line.find("{")
  if pos < 0: continue
  try: event=json.loads(line[pos:])
  except json.JSONDecodeError: continue
  if str(event.get("request_id",""))==rid:
   event["_source"]=source
   events.append(event)

by_name={}
for event in events:
 by_name.setdefault(event.get("event",""),[]).append(event)

pipeline=(by_name.get("pipeline_trace_complete") or [None])[-1]
generative=[]
if pipeline:
 generative=[s for s in pipeline.get("spans",[]) if s.get("name")=="generative_recall"]
rerank=by_name.get("source_quota_rerank_complete",[])
burst=by_name.get("pairec_brpc_burst_complete",[])
native=by_name.get("datasystem_request_complete",[])
executor=by_name.get("trt_executor_request_complete",[])

if re.search(r"items size not enough|code=299|server error 2001",combined,re.I):
 classification="INFERENCE_ITEMS_INSUFFICIENT"
elif generative and generative[-1].get("status")=="error":
 classification="GENERATIVE_BRPC_FAILURE"
elif rerank and int(rerank[-1].get("generative_input",-1))==0:
 classification="GENERATIVE_ZERO_CANDIDATES"
elif rerank and rerank[-1].get("status")=="error":
 classification="RERANK_SOURCE_QUOTA_FAIL_CLOSED"
elif burst and (not burst[-1].get("burst_valid",True) or int(burst[-1].get("pressure_errors",0))>0):
 classification="BRPC_PRESSURE_FAILURE"
elif not combined.strip():
 classification="REQUEST_EVIDENCE_NOT_FOUND"
else:
 classification="UNKNOWN_RERANK_FAILURE"

trt_top_k=[]
rollout={}
try:
 deployments=json.loads(text("deployments.json"))
 for deployment in deployments.get("items",[]):
  name=deployment["metadata"]["name"]
  status=deployment.get("status",{})
  rollout[name]={
   "generation":deployment["metadata"].get("generation"),
   "observed_generation":status.get("observedGeneration"),
   "replicas":status.get("replicas",0),
   "updated_replicas":status.get("updatedReplicas",0),
   "available_replicas":status.get("availableReplicas",0),
  }
  for container in deployment["spec"]["template"]["spec"].get("containers",[]):
   if container.get("name")=="brpc-inference":
    trt_top_k.extend(arg for arg in container.get("args",[]) if arg.startswith("--trt_top_k="))
except (json.JSONDecodeError,KeyError):
 pass

result={
 "classification":classification,
 "request_id":rid,
 "run_output_dir":run_dir or None,
 "pairec_evidence":bool(pairec.strip()),
 "wrapper_evidence":bool(wrapper.strip()),
 "inference_evidence":bool(inference.strip()),
 "inference_completion_found":bool(native or executor),
 "trt_top_k":trt_top_k,
 "rollout":rollout,
 "generative_span":generative[-1] if generative else None,
 "rerank_event":rerank[-1] if rerank else None,
 "burst_event":burst[-1] if burst else None,
 "datasystem_event":native[-1] if native else None,
 "executor_event":executor[-1] if executor else None,
}
(out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")

print("== Request evidence ==")
for source,body in (("PaiRec",pairec),("Wrapper",wrapper),("Inference",inference)):
 print(f"-- {source} --")
 lines=body.splitlines()
 print("\n".join(lines[-30:]) if lines else "REQUEST_SPECIFIC_LOG_NOT_FOUND")
print("\n== Diagnosis ==")
print(f"classification={classification}")
print(f"request_id={rid}")
print(f"trt_top_k={','.join(trt_top_k) if trt_top_k else 'UNKNOWN'}")
print(f"inference_completion_found={str(bool(native or executor)).lower()}")
if not inference.strip():
 print("inference_log_note=original inference pod may have been replaced during TRT configuration restore")
if generative:
 print(f"generative_status={generative[-1].get('status')} error={generative[-1].get('attributes',{}).get('error','')}")
if rerank:
 print(f"rerank_status={rerank[-1].get('status')} generative_input={rerank[-1].get('generative_input')} error={rerank[-1].get('error','')}")
print(f"summary_json={out/'summary.json'}")
PY

echo "output_dir=$OUTPUT_DIR"
echo "PAIREC_WRAPPER_RERANK_DIAGNOSTIC_COMPLETE"
