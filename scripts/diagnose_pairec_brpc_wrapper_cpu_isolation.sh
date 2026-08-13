#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
WRAPPER_APP="${WRAPPER_APP:-brpc-burst-wrapper}"
WRAPPER_CONTAINER="${WRAPPER_CONTAINER:-brpc-burst-wrapper}"
INFERENCE_APP="${INFERENCE_APP:-inference-brpc-trtllm}"
INFERENCE_CONTAINER="${INFERENCE_CONTAINER:-brpc-inference}"
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

RESTORED=0
restore_affinity() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$RESTORED" = 0 ]]; then
    echo "== Restore original CPU affinity =="
    if [[ "$(ready_pod "$WRAPPER_APP" 2>/dev/null || true)" = "$WRAPPER_POD" ]]; then
      set_affinity "$WRAPPER_POD" "$WRAPPER_CONTAINER" "$WRAPPER_ORIGINAL_CPUSET" \
        >"$OUTPUT_DIR/wrapper-affinity-restore.log" 2>&1 || true
    fi
    if [[ "$(ready_pod "$INFERENCE_APP" 2>/dev/null || true)" = "$INFERENCE_POD" ]]; then
      set_affinity "$INFERENCE_POD" "$INFERENCE_CONTAINER" "$INFERENCE_ORIGINAL_CPUSET" \
        >"$OUTPUT_DIR/inference-affinity-restore.log" 2>&1 || true
    fi
    RESTORED=1
  fi
  exit "$status"
}
trap restore_affinity EXIT INT TERM

cat <<EOF
== BRPC Wrapper / TRT CPU isolation ==
node=$WRAPPER_NODE
wrapper_pod=$WRAPPER_POD original=$WRAPPER_ORIGINAL_CPUSET isolated=$WRAPPER_ISOLATED_CPUSET
inference_pod=$INFERENCE_POD original=$INFERENCE_ORIGINAL_CPUSET isolated=$INFERENCE_ISOLATED_CPUSET
requests=$REQUESTS qualification_requests=$QUALIFICATION_REQUESTS user_id=$USER_ID output_dir=$OUTPUT_DIR
EOF

echo "== Apply disjoint CPU affinity to all existing process threads =="
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

if [[ -n "$BASELINE_SUMMARY" ]]; then
  [[ -f "$BASELINE_SUMMARY" ]] || die "BASELINE_SUMMARY does not exist: $BASELINE_SUMMARY"
  python3 - "$BASELINE_SUMMARY" "$OUTPUT_DIR/isolated/summary.json" \
    "$OUTPUT_DIR/comparison.json" <<'PY'
import json,pathlib,sys
baseline=json.load(open(sys.argv[1])); isolated=json.load(open(sys.argv[2]))
def total(item): return float(item['runner_p99_total_delta_ms'])
result={
 'classification':'PAIREC_BRPC_WRAPPER_CPU_ISOLATION_COMPARISON',
 'baseline_runner_p99_total_delta_ms':total(baseline),
 'isolated_runner_p99_total_delta_ms':total(isolated),
 'improvement_ms':total(baseline)-total(isolated),
 'isolated_runner_gate_passed':bool(isolated['runner_gate_passed']),
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
PY
fi

echo "PAIREC_BRPC_WRAPPER_CPU_ISOLATION_DIAGNOSIS_COMPLETE"
