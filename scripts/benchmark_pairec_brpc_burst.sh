#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec-brpc-wrapper}"
WRAPPER_DEPLOYMENT="${WRAPPER_DEPLOYMENT:-brpc-burst-wrapper}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
EXPECTED_CONCURRENCY="${EXPECTED_CONCURRENCY:-1000}"
REQUESTS="${REQUESTS:-3}"
USER_ID="${USER_ID:-5}"
SIZE="${SIZE:-8}"
COMPLETION_TIMEOUT_SECONDS="${COMPLETION_TIMEOUT_SECONDS:-10}"
OUT_DIR="${OUT_DIR:-/tmp/pairec-brpc-burst/$(date +%Y%m%d-%H%M%S)-c${EXPECTED_CONCURRENCY}-n${REQUESTS}}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$EXPECTED_CONCURRENCY" =~ ^(1|1000)$ ]] || die "EXPECTED_CONCURRENCY must be 1 or 1000"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
for command in kubectl curl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: ${command}"
done
mkdir -p "$OUT_DIR"
STARTED_AT="$(date --iso-8601=seconds)"

pod_state() {
  kubectl -n "$NAMESPACE" get pod -l "app=$1" -o json | python3 -c '
import json, sys
pods=json.load(sys.stdin).get("items", [])
ready=[]
for p in pods:
 s=p.get("status",{}).get("containerStatuses",[])
 if p.get("status",{}).get("phase")=="Running" and s and all(x.get("ready") for x in s):
  ready.append((p,s))
if len(ready)!=1: raise SystemExit(f"expected one ready pod, found {len(ready)}")
p,s=ready[0]
print("pod=" + p["metadata"]["name"])
print("uid=" + p["metadata"]["uid"])
print("restarts=" + str(sum(x.get("restartCount",0) for x in s)))
'
}

for deployment in "$PAIREC_DEPLOYMENT" "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT"; do
  pod_state "$deployment" >"$OUT_DIR/${deployment}.before"
done
PAIREC_POD="$(awk -F= '$1=="pod"{print $2}' "$OUT_DIR/${PAIREC_DEPLOYMENT}.before")"
ready_line="$(kubectl -n "$NAMESPACE" logs "$PAIREC_POD" | grep '"event":"pairec_brpc_burst_ready"' | tail -1 || true)"
test -n "$ready_line" || die "burst ready event missing"
python3 - "$ready_line" "$EXPECTED_CONCURRENCY" <<'PY'
import json, sys
event=json.loads(sys.argv[1][sys.argv[1].find("{"):])
expected=int(sys.argv[2])
assert event["concurrency"]==expected and event["connected_sessions"]==expected, event
PY

SERVICE_IP="$(kubectl -n "$NAMESPACE" get service "$PAIREC_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}')"
test -n "$SERVICE_IP" && test "$SERVICE_IP" != "None" || die "PaiRec service has no ClusterIP"
URL="http://${SERVICE_IP}:18080/api/recommend"
printf 'index\thttp_code\te2e_ms\trequest_id\titems\n' >"$OUT_DIR/requests.tsv"

for index in $(seq 1 "$REQUESTS"); do
  body="$OUT_DIR/response-${index}.json"
  metric="$(curl -sS --max-time 10 "$URL" -H 'Content-Type: application/json' \
    -d "{\"scene_id\":\"home_feed\",\"uid\":\"${USER_ID}\",\"size\":${SIZE}}" \
    -o "$body" -w '%{http_code} %{time_total}')"
  read -r http_code seconds <<<"$metric"
  read -r code request_id items < <(python3 - "$body" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("code"), d.get("request_id", ""), len(d.get("items", [])))
PY
  )
  test "$http_code" = 200 && test "$code" = 200 || die "request ${index} failed: HTTP=${http_code} code=${code}"
  test "$items" = "$SIZE" || die "request ${index} returned ${items}/${SIZE} items"
  test -n "$request_id" || die "request ${index} has no request_id"
  e2e_ms="$(python3 -c 'import sys; print(f"{float(sys.argv[1])*1000:.3f}")' "$seconds")"
  printf '%s\t%s\t%s\t%s\t%s\n' "$index" "$http_code" "$e2e_ms" "$request_id" "$items" | tee -a "$OUT_DIR/requests.tsv"

  deadline=$((SECONDS + COMPLETION_TIMEOUT_SECONDS))
  while ! kubectl -n "$NAMESPACE" logs "$PAIREC_POD" --since-time="$STARTED_AT" 2>/dev/null \
      | grep -F '"event":"pairec_brpc_burst_complete"' \
      | grep -F "\"request_id\":\"${request_id}\"" >/dev/null; do
    (( SECONDS < deadline )) || die "burst completion timed out for request_id=${request_id}"
    sleep 0.1
  done
done

for deployment in "$PAIREC_DEPLOYMENT" "$WRAPPER_DEPLOYMENT" "$INFERENCE_DEPLOYMENT"; do
  pod_state "$deployment" >"$OUT_DIR/${deployment}.after"
  cmp -s "$OUT_DIR/${deployment}.before" "$OUT_DIR/${deployment}.after" \
    || die "deployment/${deployment} pod identity or restart count changed"
  pod="$(awk -F= '$1=="pod"{print $2}' "$OUT_DIR/${deployment}.after")"
  kubectl -n "$NAMESPACE" logs "$pod" --since-time="$STARTED_AT" >"$OUT_DIR/${deployment}.log" 2>&1
done

python3 - "$OUT_DIR" "$EXPECTED_CONCURRENCY" <<'PY'
import json, math, pathlib, statistics, sys
out=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2])
rows=[]
for line in (out/"requests.tsv").read_text().splitlines()[1:]:
    index,http,e2e,rid,items=line.split("\t")
    rows.append({"request_id":rid,"e2e_ms":float(e2e)})
wanted={row["request_id"] for row in rows}
events={rid:{} for rid in wanted}
for line in (out/"pairec-brpc-wrapper.log").read_text(errors="replace").splitlines():
    pos=line.find("{")
    if pos<0: continue
    try: event=json.loads(line[pos:])
    except json.JSONDecodeError: continue
    rid=event.get("request_id")
    if rid in events and event.get("event","").startswith("pairec_brpc_burst_"):
        events[rid][event["event"]]=event

metrics={"e2e_ms":[row["e2e_ms"] for row in rows],"front_brpc_ms":[],"inference_ms":[],"runner_generate_ms":[],"wrapper_total_ms":[]}
for row in rows:
    rid=row["request_id"]; group=events[rid]
    names=("pairec_brpc_burst_start","pairec_brpc_burst_business_complete","pairec_brpc_burst_complete")
    missing=[name for name in names if name not in group]
    if missing: raise SystemExit(f"missing events for {rid}: {missing}")
    start,business,complete=(group[name] for name in names)
    assert start["concurrency"]==expected and start["connected_sessions"]==expected, start
    assert complete["armed_workers"]==expected, complete
    assert complete["pressure_requests"]==expected-1, complete
    assert complete["pressure_success"]==expected-1 and complete["pressure_errors"]==0, complete
    assert business["business_success"] and business["trace_valid"], business
    assert complete["business_success"] and complete["trace_valid"] and complete["burst_valid"], complete
    metrics["front_brpc_ms"].append(business["business_front_brpc_ms"])
    metrics["inference_ms"].append(business["business_inference_ms"])
    metrics["runner_generate_ms"].append(business["business_runner_generate_ms"])
    metrics["wrapper_total_ms"].append(business["wrapper_total_ms"])

def percentile(values, q):
    values=sorted(values)
    if len(values)==1: return values[0]
    rank=(len(values)-1)*q; lo=math.floor(rank); hi=math.ceil(rank)
    return values[lo]+(values[hi]-values[lo])*(rank-lo)

summary={"concurrency":expected,"samples":len(rows),"valid_samples":len(rows),"metrics":{}}
print("\n== PaiRec embedded BRPC burst summary ==")
print("metric count avg p50 p95 p99 max")
for name,values in metrics.items():
    item={"count":len(values),"avg":statistics.fmean(values),"p50":percentile(values,.5),"p95":percentile(values,.95),"p99":percentile(values,.99),"max":max(values)}
    summary["metrics"][name]=item
    print(f"{name:20s} {len(values):5d} {item['avg']:8.3f} {item['p50']:8.3f} {item['p95']:8.3f} {item['p99']:8.3f} {item['max']:8.3f}")
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
PY

for request_id in $(tail -n +2 "$OUT_DIR/requests.tsv" | cut -f4); do
  grep -F "request_id=${request_id}" "$OUT_DIR/${WRAPPER_DEPLOYMENT}.log" >/dev/null \
    || die "Wrapper trace missing request_id=${request_id}"
  grep -F "request_id=${request_id}" "$OUT_DIR/${INFERENCE_DEPLOYMENT}.log" >/dev/null \
    || die "inference trace missing request_id=${request_id}"
done
if grep -Eqi 'from=cache|fallback|brpc request failed|Segmentation|core dumped|Out of memory' \
    "$OUT_DIR/${PAIREC_DEPLOYMENT}.log" "$OUT_DIR/${WRAPPER_DEPLOYMENT}.log" "$OUT_DIR/${INFERENCE_DEPLOYMENT}.log"; then
  die "cache, fallback, RPC failure, or crash marker detected"
fi

echo "PAIREC_BRPC_BURST_BENCHMARK_OK concurrency=${EXPECTED_CONCURRENCY} samples=${REQUESTS}"
echo "output_dir=${OUT_DIR}"
