#!/usr/bin/env bash
set -euo pipefail

REPEATS="${REPEATS:-3}"
CASES="${CASES:-baseline,brpc-c10,brpc-c100,brpc-c1000,kvc-17.5m-c10,kvc-1.75m-c100}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/tmp/brpc-kvc-pressure-matrix/${RUN_ID}}"
ROUND_COOLDOWN_SECONDS="${ROUND_COOLDOWN_SECONDS:-5}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-scripts/benchmark_brpc_kvc_contention.sh}"
MATRIX_STRICT_COUNTS="${MATRIX_STRICT_COUNTS:-0}"

BRPC_PAYLOAD_BYTES="${BRPC_PAYLOAD_BYTES:-102400}"
BRPC_REQUESTS="${BRPC_REQUESTS:-1000000}"
BRPC_ENDPOINT="${BRPC_ENDPOINT:-192.168.100.11:18100}"

KVC_LOAD_HOST="${KVC_LOAD_HOST:-worker1}"
KVC_DSBENCH_CPP="${KVC_DSBENCH_CPP:-}"
KVC_DS_ENDPOINT="${KVC_DS_ENDPOINT:-192.168.100.12:18482}"
KVC_LARGE_KEY_COUNT="${KVC_LARGE_KEY_COUNT:-100}"
KVC_SMALL_KEY_COUNT="${KVC_SMALL_KEY_COUNT:-1000}"
KVC_LOAD_READY_TIMEOUT_SECONDS="${KVC_LOAD_READY_TIMEOUT_SECONDS:-300}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_case() {
  local name="$1"
  local out_dir="${OUT_ROOT}/${name}"
  local -a args=(
    REPEATS="$REPEATS"
    ROUND_COOLDOWN_SECONDS="$ROUND_COOLDOWN_SECONDS"
    STRICT_COUNTS="$MATRIX_STRICT_COUNTS"
    RUN_ID="${RUN_ID}-${name}"
    OUT_DIR="$out_dir"
  )

  case "$name" in
    baseline)
      args+=(MODE=baseline)
      ;;
    brpc-c10|brpc-c100|brpc-c1000)
      local concurrency="${name#brpc-c}"
      args+=(
        MODE=brpc
        BRPC_ENDPOINT="$BRPC_ENDPOINT"
        BRPC_LOAD_PAYLOAD_BYTES="$BRPC_PAYLOAD_BYTES"
        BRPC_LOAD_CONCURRENCY="$concurrency"
        BRPC_LOAD_REQUESTS="$BRPC_REQUESTS"
        BRPC_LOAD_QPS=0
        BRPC_LOAD_REUSE_CONNECTIONS=1
      )
      ;;
    kvc-17.5m-c10)
      args+=(
        MODE=kvc-get
        KVC_LOAD_HOST="$KVC_LOAD_HOST"
        KVC_DS_ENDPOINT="$KVC_DS_ENDPOINT"
        KVC_PRESSURE_ENGINE=dsbench
        KVC_DSBENCH_CPP="$KVC_DSBENCH_CPP"
        KVC_OBJECT_SIZE=17920KB
        KVC_KEY_COUNT="$KVC_LARGE_KEY_COUNT"
        KVC_BATCH_NUM=1
        KVC_THREAD_NUM=1
        KVC_GET_CLIENTS=10
        KVC_LOAD_READY_TIMEOUT_SECONDS="$KVC_LOAD_READY_TIMEOUT_SECONDS"
      )
      ;;
    kvc-1.75m-c100)
      args+=(
        MODE=kvc-get
        KVC_LOAD_HOST="$KVC_LOAD_HOST"
        KVC_DS_ENDPOINT="$KVC_DS_ENDPOINT"
        KVC_PRESSURE_ENGINE=dsbench
        KVC_DSBENCH_CPP="$KVC_DSBENCH_CPP"
        KVC_OBJECT_SIZE=1792KB
        KVC_KEY_COUNT="$KVC_SMALL_KEY_COUNT"
        KVC_BATCH_NUM=1
        KVC_THREAD_NUM=1
        KVC_GET_CLIENTS=100
        KVC_LOAD_READY_TIMEOUT_SECONDS="$KVC_LOAD_READY_TIMEOUT_SECONDS"
      )
      ;;
    *)
      die "unknown pressure case: $name"
      ;;
  esac

  mkdir -p "$out_dir"
  printf '\n== pressure case=%s ==\n' "$name"
  set +e
  env "${args[@]}" bash "$BENCHMARK_SCRIPT" \
    | tee "${out_dir}/console.log"
  local benchmark_code="${PIPESTATUS[0]}"
  set -e
  echo "$benchmark_code" >"${out_dir}/exit_code"
  if [ "$benchmark_code" -ne 0 ]; then
    echo "WARNING: pressure case=${name} failed with status ${benchmark_code}; continue with remaining cases" >&2
  fi
}

mkdir -p "$OUT_ROOT"
[ -f "$BENCHMARK_SCRIPT" ] || die "benchmark script not found: $BENCHMARK_SCRIPT"
case "$MATRIX_STRICT_COUNTS" in
  0|1) ;;
  *) die "MATRIX_STRICT_COUNTS must be 0 or 1" ;;
esac
IFS=',' read -r -a selected_cases <<<"$CASES"
for selected in "${selected_cases[@]}"; do
  selected="${selected//[[:space:]]/}"
  [ -n "$selected" ] || continue
  run_case "$selected"
done

python3 - "$OUT_ROOT" "${selected_cases[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
names = [name.strip() for name in sys.argv[2:] if name.strip()]
print("\n== pressure matrix summary ==")
print("case status valid e2e_avg e2e_p95 kvc_avg kvc_p95 kvc_max brpc_avg")
failed = False
for name in names:
    path = root / name / "result.json"
    if not path.exists():
        print(f"{name} MISSING 0/0 n/a n/a n/a n/a n/a n/a")
        failed = True
        continue
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result.get("metrics", {})

    def metric(stage, field):
        value = (metrics.get(stage) or {}).get(field)
        return "n/a" if value is None else f"{value:.3f}"

    print(
        name,
        result.get("status", "UNKNOWN"),
        f"{result.get('valid_repeats', 0)}/{result.get('expected_repeats', 0)}",
        metric("e2e_ms", "avg"),
        metric("e2e_ms", "p95"),
        metric("kvc_ms", "avg"),
        metric("kvc_ms", "p95"),
        metric("kvc_ms", "max"),
        metric("brpc_ms", "avg"),
    )
    failed = failed or result.get("status") != "PASS"
print(f"output_root={root}")
raise SystemExit(1 if failed else 0)
PY
