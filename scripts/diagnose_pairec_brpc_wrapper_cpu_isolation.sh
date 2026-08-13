#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
WRAPPER_APP="${WRAPPER_APP:-brpc-burst-wrapper}"
WRAPPER_CONTAINER="${WRAPPER_CONTAINER:-brpc-burst-wrapper}"
INFERENCE_APP="${INFERENCE_APP:-inference-brpc-trtllm}"
INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
DETERMINISTIC_TRT_TOP_K="${DETERMINISTIC_TRT_TOP_K:-1}"
INFERENCE_ROLLOUT_TIMEOUT="${INFERENCE_ROLLOUT_TIMEOUT:-10m}"
INFERENCE_CPU_COUNT="${INFERENCE_CPU_COUNT:-8}"
WRAPPER_CPU_COUNT="${WRAPPER_CPU_COUNT:-32}"
INFERENCE_CPUSET="${INFERENCE_CPUSET:-}"
WRAPPER_CPUSET="${WRAPPER_CPUSET:-}"
REQUESTS="${REQUESTS:-100}"
QUALIFICATION_REQUESTS="${QUALIFICATION_REQUESTS:-10}"
USER_ID="${USER_ID:-6312}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-cpu-isolation/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"

die() { echo "ERROR: $*" >&2; exit 1; }

for value in "$INFERENCE_CPU_COUNT" "$WRAPPER_CPU_COUNT" "$REQUESTS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "CPU counts and REQUESTS must be positive integers"
done
[[ "$DETERMINISTIC_TRT_TOP_K" = 1 ]] \
  || die "DETERMINISTIC_TRT_TOP_K must be 1 so request seeds cannot change sampled output"
for command in kubectl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
mkdir -p "$OUTPUT_DIR"

ready_pod() {
  local app="$1"
  kubectl -n "$NAMESPACE" get pods -l "app=$app" -o json | python3 -c '
import json,sys
pods=[]
for pod in json.load(sys.stdin).get("items",[]):
 status=pod.get("status",{}); containers=status.get("containerStatuses",[])
 if (not pod["metadata"].get("deletionTimestamp") and status.get("phase")=="Running"
     and containers and all(item.get("ready") for item in containers)):
  pods.append((pod["metadata"].get("creationTimestamp",""),pod["metadata"]["name"]))
assert pods,f"no ready pod for {sys.argv[1]}"
print(max(pods)[1])
' "$app"
}

allowed_cpus() {
  kubectl -n "$NAMESPACE" exec "$1" -c "$2" -- \
    sh -c "awk '/^Cpus_allowed_list:/ {print \$2}' /proc/1/status"
}

set_affinity() {
  local pod="$1" container="$2" cpus="$3"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$container" -- \
    taskset -apc "$cpus" 1
}

verify_all_threads_affinity() {
  local pod="$1" container="$2" expected="$3"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$container" -- \
    sh -c 'expected="$1"; failed=0; count=0
      for status in /proc/1/task/*/status; do
        actual=$(awk '\''/^Cpus_allowed_list:/ {print $2}'\'' "$status")
        count=$((count + 1))
        if test "$actual" != "$expected"; then
          echo "thread_affinity_mismatch status=$status actual=$actual expected=$expected" >&2
          failed=1
        fi
      done
      test "$count" -gt 0 && test "$failed" -eq 0
      echo "thread_affinity_ok threads=$count cpuset=$expected"' -- "$expected"
}

DEPLOYMENT_BEFORE="$OUTPUT_DIR/inference-deployment-before.json"
DETERMINISTIC_PATCH="$OUTPUT_DIR/inference-top-k-deterministic-patch.json"
RESTORE_PATCH="$OUTPUT_DIR/inference-top-k-restore-patch.json"
kubectl -n "$NAMESPACE" get deployment "$INFERENCE_DEPLOYMENT" -o json \
  >"$DEPLOYMENT_BEFORE"
python3 - "$DEPLOYMENT_BEFORE" "$INFERENCE_CONTAINER" \
  "$DETERMINISTIC_TRT_TOP_K" "$DETERMINISTIC_PATCH" "$RESTORE_PATCH" <<'PY'
import json,pathlib,sys
source,container_name,top_k,deterministic_path,restore_path=sys.argv[1:]
deployment=json.load(open(source))
containers=deployment["spec"]["template"]["spec"]["containers"]
matches=[(index,item) for index,item in enumerate(containers) if item["name"]==container_name]
assert len(matches)==1,f"expected one container {container_name}, got {len(matches)}"
index,container=matches[0]
original=list(container.get("args",[]))
positions=[i for i,arg in enumerate(original) if arg.startswith("--trt_top_k=")]
assert len(positions)==1,f"expected one --trt_top_k argument, got {positions}"
deterministic=list(original)
deterministic[positions[0]]=f"--trt_top_k={top_k}"
path=f"/spec/template/spec/containers/{index}/args"
deterministic_patch=[] if deterministic==original else [{"op":"replace","path":path,"value":deterministic}]
restore_patch=[] if deterministic==original else [{"op":"replace","path":path,"value":original}]
pathlib.Path(deterministic_path).write_text(json.dumps(deterministic_patch)+"\n")
pathlib.Path(restore_path).write_text(json.dumps(restore_patch)+"\n")
print(f"original_trt_top_k={original[positions[0]].split('=',1)[1]}")
print(f"diagnostic_trt_top_k={top_k}")
print(f"deployment_patch_required={str(bool(deterministic_patch)).lower()}")
PY

DETERMINISTIC_APPLIED=0
AFFINITY_APPLIED=0
WRAPPER_POD=""
INFERENCE_POD=""
WRAPPER_ORIGINAL_CPUSET=""
INFERENCE_ORIGINAL_CPUSET=""
cleanup() {
  local status=$?
  local cleanup_failed=0
  trap - EXIT INT TERM
  if [[ "$AFFINITY_APPLIED" = 1 ]]; then
    echo "== Restore original CPU affinity =="
    if [[ "$(ready_pod "$WRAPPER_APP" 2>/dev/null || true)" = "$WRAPPER_POD" ]]; then
      set_affinity "$WRAPPER_POD" "$WRAPPER_CONTAINER" "$WRAPPER_ORIGINAL_CPUSET" \
        >"$OUTPUT_DIR/wrapper-affinity-restore.log" 2>&1 || cleanup_failed=1
    fi
    if [[ "$(ready_pod "$INFERENCE_APP" 2>/dev/null || true)" = "$INFERENCE_POD" ]]; then
      set_affinity "$INFERENCE_POD" "$INFERENCE_CONTAINER" "$INFERENCE_ORIGINAL_CPUSET" \
        >"$OUTPUT_DIR/inference-affinity-restore.log" 2>&1 || cleanup_failed=1
    fi
  fi
  if [[ "$DETERMINISTIC_APPLIED" = 1 ]]; then
    echo "== Restore original TRT sampling configuration =="
    kubectl -n "$NAMESPACE" patch deployment "$INFERENCE_DEPLOYMENT" \
      --type=json -p "$(cat "$RESTORE_PATCH")" \
      >"$OUTPUT_DIR/inference-config-restore.log" 2>&1 || cleanup_failed=1
    if [[ "$cleanup_failed" = 0 ]]; then
      kubectl -n "$NAMESPACE" rollout status "deployment/$INFERENCE_DEPLOYMENT" \
        --timeout="$INFERENCE_ROLLOUT_TIMEOUT" \
        >>"$OUTPUT_DIR/inference-config-restore.log" 2>&1 || cleanup_failed=1
    fi
  fi
  if [[ "$cleanup_failed" = 1 ]]; then
    echo "ERROR: diagnostic cleanup failed; inspect $OUTPUT_DIR/*restore.log" >&2
    [[ "$status" != 0 ]] || status=1
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$(cat "$DETERMINISTIC_PATCH")" != "[]" ]]; then
  echo "== Apply deterministic TRT sampling for diagnostic =="
  kubectl -n "$NAMESPACE" patch deployment "$INFERENCE_DEPLOYMENT" \
    --type=json -p "$(cat "$DETERMINISTIC_PATCH")"
  DETERMINISTIC_APPLIED=1
  kubectl -n "$NAMESPACE" rollout status "deployment/$INFERENCE_DEPLOYMENT" \
    --timeout="$INFERENCE_ROLLOUT_TIMEOUT"
else
  echo "TRT_DETERMINISTIC_SAMPLING_ALREADY_CONFIGURED top_k=$DETERMINISTIC_TRT_TOP_K"
fi

WRAPPER_POD="$(ready_pod "$WRAPPER_APP")"
INFERENCE_POD="$(ready_pod "$INFERENCE_APP")"
WRAPPER_NODE="$(kubectl -n "$NAMESPACE" get pod "$WRAPPER_POD" -o jsonpath='{.spec.nodeName}')"
INFERENCE_NODE="$(kubectl -n "$NAMESPACE" get pod "$INFERENCE_POD" -o jsonpath='{.spec.nodeName}')"
[[ "$WRAPPER_NODE" = "$INFERENCE_NODE" ]] \
  || die "Wrapper and inference are not colocated: $WRAPPER_NODE != $INFERENCE_NODE"

for tuple in "$WRAPPER_POD:$WRAPPER_CONTAINER" "$INFERENCE_POD:$INFERENCE_CONTAINER"; do
  IFS=: read -r pod container <<<"$tuple"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$container" -- \
    sh -c 'command -v taskset >/dev/null && test -r /proc/1/status' \
    || die "taskset or /proc/1/status unavailable in $pod/$container"
done

WRAPPER_ORIGINAL_CPUSET="$(allowed_cpus "$WRAPPER_POD" "$WRAPPER_CONTAINER")"
INFERENCE_ORIGINAL_CPUSET="$(allowed_cpus "$INFERENCE_POD" "$INFERENCE_CONTAINER")"
[[ -n "$WRAPPER_ORIGINAL_CPUSET" && -n "$INFERENCE_ORIGINAL_CPUSET" ]] \
  || die "failed to read original CPU affinity"

mapfile -t selected_sets < <(python3 - "$WRAPPER_ORIGINAL_CPUSET" \
  "$INFERENCE_ORIGINAL_CPUSET" "$INFERENCE_CPU_COUNT" "$WRAPPER_CPU_COUNT" \
  "$INFERENCE_CPUSET" "$WRAPPER_CPUSET" <<'PY'
import sys
wrapper_allowed,inference_allowed,inference_count,wrapper_count,inference_explicit,wrapper_explicit=sys.argv[1:]
def expand(value):
 out=[]
 for part in value.split(','):
  part=part.strip()
  if not part: continue
  if '-' in part:
   lo,hi=map(int,part.split('-',1)); out.extend(range(lo,hi+1))
  else: out.append(int(part))
 return out
def compact(values):
 values=sorted(set(values)); ranges=[]
 for value in values:
  if not ranges or value!=ranges[-1][1]+1: ranges.append([value,value])
  else: ranges[-1][1]=value
 return ','.join(str(lo) if lo==hi else f'{lo}-{hi}' for lo,hi in ranges)
wa=set(expand(wrapper_allowed)); ia=set(expand(inference_allowed)); common=sorted(wa&ia)
if inference_explicit or wrapper_explicit:
 if not inference_explicit or not wrapper_explicit:
  raise SystemExit('INFERENCE_CPUSET and WRAPPER_CPUSET must be provided together')
 inference=expand(inference_explicit); wrapper=expand(wrapper_explicit)
else:
 ic=int(inference_count); wc=int(wrapper_count)
 if len(common)<ic+wc:
  raise SystemExit(f'need {ic+wc} common allowed CPUs, found {len(common)}')
 # Keep inference on the first compact range and Wrapper on the next range.
 inference=common[:ic]; wrapper=common[ic:ic+wc]
if not set(inference)<=ia: raise SystemExit('inference cpuset outside its current affinity')
if not set(wrapper)<=wa: raise SystemExit('wrapper cpuset outside its current affinity')
if set(inference)&set(wrapper): raise SystemExit('inference and wrapper cpusets overlap')
print(compact(inference)); print(compact(wrapper))
PY
)
[[ "${#selected_sets[@]}" = 2 ]] || die "failed to select isolated CPU sets"
INFERENCE_ISOLATED_CPUSET="${selected_sets[0]}"
WRAPPER_ISOLATED_CPUSET="${selected_sets[1]}"

cat <<EOF
== BRPC Wrapper / TRT CPU isolation ==
node=$WRAPPER_NODE
wrapper_pod=$WRAPPER_POD original=$WRAPPER_ORIGINAL_CPUSET isolated=$WRAPPER_ISOLATED_CPUSET
inference_pod=$INFERENCE_POD original=$INFERENCE_ORIGINAL_CPUSET isolated=$INFERENCE_ISOLATED_CPUSET
requests=$REQUESTS qualification_requests=$QUALIFICATION_REQUESTS user_id=$USER_ID output_dir=$OUTPUT_DIR
diagnostic_trt_top_k=$DETERMINISTIC_TRT_TOP_K comparison=unisolated_vs_isolated_same_runtime
EOF

echo "== Baseline: unisolated CPU affinity =="
REQUESTS="$REQUESTS" \
QUALIFICATION_REQUESTS="$QUALIFICATION_REQUESTS" \
USER_ID="$USER_ID" \
OUTPUT_DIR="$OUTPUT_DIR/unisolated" \
BUILD_PAIREC_IMAGE=0 \
IMPORT_PAIREC_IMAGE=0 \
  bash scripts/diagnose_pairec_brpc_wrapper_runner_interference.sh \
  | tee "$OUTPUT_DIR/unisolated.log"

[[ "$(ready_pod "$WRAPPER_APP")" = "$WRAPPER_POD" ]] || die "Wrapper pod changed during baseline"
[[ "$(ready_pod "$INFERENCE_APP")" = "$INFERENCE_POD" ]] || die "inference pod changed during baseline"

echo "== Apply disjoint CPU affinity to all existing process threads =="
AFFINITY_APPLIED=1
set_affinity "$INFERENCE_POD" "$INFERENCE_CONTAINER" "$INFERENCE_ISOLATED_CPUSET" \
  | tee "$OUTPUT_DIR/inference-taskset.log"
set_affinity "$WRAPPER_POD" "$WRAPPER_CONTAINER" "$WRAPPER_ISOLATED_CPUSET" \
  | tee "$OUTPUT_DIR/wrapper-taskset.log"

actual_inference="$(allowed_cpus "$INFERENCE_POD" "$INFERENCE_CONTAINER")"
actual_wrapper="$(allowed_cpus "$WRAPPER_POD" "$WRAPPER_CONTAINER")"
[[ "$actual_inference" = "$INFERENCE_ISOLATED_CPUSET" ]] \
  || die "inference affinity mismatch: $actual_inference"
[[ "$actual_wrapper" = "$WRAPPER_ISOLATED_CPUSET" ]] \
  || die "wrapper affinity mismatch: $actual_wrapper"
verify_all_threads_affinity "$INFERENCE_POD" "$INFERENCE_CONTAINER" \
  "$INFERENCE_ISOLATED_CPUSET" | tee "$OUTPUT_DIR/inference-thread-affinity.log"
verify_all_threads_affinity "$WRAPPER_POD" "$WRAPPER_CONTAINER" \
  "$WRAPPER_ISOLATED_CPUSET" | tee "$OUTPUT_DIR/wrapper-thread-affinity.log"
echo "BRPC_WRAPPER_TRT_CPU_ISOLATION_OK inference=$actual_inference wrapper=$actual_wrapper"

REQUESTS="$REQUESTS" \
QUALIFICATION_REQUESTS="$QUALIFICATION_REQUESTS" \
USER_ID="$USER_ID" \
OUTPUT_DIR="$OUTPUT_DIR/isolated" \
BUILD_PAIREC_IMAGE=0 \
IMPORT_PAIREC_IMAGE=0 \
  bash scripts/diagnose_pairec_brpc_wrapper_runner_interference.sh \
  | tee "$OUTPUT_DIR/isolated.log"

[[ "$(ready_pod "$WRAPPER_APP")" = "$WRAPPER_POD" ]] || die "Wrapper pod changed during test"
[[ "$(ready_pod "$INFERENCE_APP")" = "$INFERENCE_POD" ]] || die "inference pod changed during test"

python3 - "$OUTPUT_DIR/unisolated/summary.json" "$OUTPUT_DIR/isolated/summary.json" \
  "$OUTPUT_DIR/comparison.json" <<'PY'
import json,pathlib,sys
baseline=json.load(open(sys.argv[1])); isolated=json.load(open(sys.argv[2]))
def total(item): return float(item['runner_p99_total_delta_ms'])
result={
 'classification':'PAIREC_BRPC_WRAPPER_CPU_ISOLATION_COMPARISON',
 'unisolated_runner_p99_total_delta_ms':total(baseline),
 'isolated_runner_p99_total_delta_ms':total(isolated),
 'improvement_ms':total(baseline)-total(isolated),
 'isolated_runner_gate_passed':bool(isolated['runner_gate_passed']),
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
PY

if [[ -n "$BASELINE_SUMMARY" ]]; then
  [[ -f "$BASELINE_SUMMARY" ]] || die "BASELINE_SUMMARY does not exist: $BASELINE_SUMMARY"
  python3 - "$BASELINE_SUMMARY" "$OUTPUT_DIR/unisolated/summary.json" \
    "$OUTPUT_DIR/historical-comparison.json" <<'PY'
import json,pathlib,sys
historical=json.load(open(sys.argv[1])); current=json.load(open(sys.argv[2]))
result={
 'classification':'PAIREC_BRPC_WRAPPER_CPU_ISOLATION_HISTORICAL_CONTEXT',
 'historical_runner_p99_total_delta_ms':float(historical['runner_p99_total_delta_ms']),
 'deterministic_unisolated_runner_p99_total_delta_ms':float(current['runner_p99_total_delta_ms']),
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
PY
fi

echo "PAIREC_BRPC_WRAPPER_CPU_ISOLATION_DIAGNOSIS_COMPLETE"
