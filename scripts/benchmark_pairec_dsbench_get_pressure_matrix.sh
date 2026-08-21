#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/pairec-dsbench-get-matrix/${RUN_ID}}"
REPEATS="${REPEATS:-1}"

mkdir -p "$OUT_DIR"

run_case() {
  local name="$1"
  local clients="$2"
  local object_size="$3"
  local case_code=0

  printf '\n== Case %s: clients=%s object_size=%s ==\n' \
    "$name" "$clients" "$object_size"
  REPEATS="$REPEATS" \
  DSBENCH_CLIENTS="$clients" \
  DSBENCH_THREADS=1 \
  DSBENCH_KEY_COUNT="$clients" \
  DSBENCH_OBJECT_SIZE="$object_size" \
  RESET_INFERENCE_AFTER_PRIME=0 \
  OUT_DIR="$OUT_DIR/$name" \
    bash scripts/benchmark_pairec_dsbench_get_pressure.sh \
    | tee "$OUT_DIR/$name.console.log" || case_code=$?
  echo "$case_code" >"$OUT_DIR/$name.exit_code"
}

# A and B use the same total synthetic pressure-key footprint (224 MiB):
# 64 * 3.5 MiB == 128 * 1.75 MiB. C doubles the footprint to 448 MiB.
run_case c64-size3.5m 64 3584KB
run_case c128-size1.75m 128 1792KB
run_case c128-size3.5m 128 3584KB

python3 - "$OUT_DIR" <<'PY' | tee "$OUT_DIR/summary.txt"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
case_specs = [
    ("c64-size3.5m", 64, 3584),
    ("c128-size1.75m", 128, 1792),
    ("c128-size3.5m", 128, 3584),
]
rows = []
for name, clients, object_kib in case_specs:
    try:
        summary = json.loads((root / name / "summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        summary = {}
    try:
        exit_code = int((root / f"{name}.exit_code").read_text().strip())
    except (FileNotFoundError, ValueError):
        exit_code = 1
    source = summary.get("averages") if summary.get("valid") else summary.get("diagnostic_averages")
    source = source or {}
    rows.append({
        "case": name,
        "clients": clients,
        "object_size_kib": object_kib,
        "pressure_key_footprint_mib": clients * object_kib / 1024.0,
        "valid": bool(summary.get("valid")),
        "exit_code": exit_code,
        "business_get_total_ms": float(source.get("business_get_total_ms", 0.0)),
        "datasystem_get_ms": float(source.get("datasystem_get_ms", 0.0)),
        "runner_ms": float(source.get("runner_ms", 0.0)),
        "dsbench_get_qps": float(source.get("dsbench_get_qps", 0.0)),
        "dsbench_get_gbps": float(source.get("dsbench_get_gbps", 0.0)),
        "dsbench_get_max_inflight": float(source.get("dsbench_get_max_inflight", 0.0)),
        "nic_peak_rx_gbps": float(source.get("nic_peak_rx_gbps", 0.0)),
    })

valid_count = sum(row["valid"] for row in rows)
result = {
    "classification": (
        "PAIREC_DSBENCH_GET_PRESSURE_MATRIX_VALID"
        if valid_count == len(rows)
        else "PAIREC_DSBENCH_GET_PRESSURE_MATRIX_PARTIAL"
    ),
    "valid_cases": valid_count,
    "expected_cases": len(rows),
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

print(result["classification"])
print("case valid footprint_mib business_get_ms ds_get_ms runner_ms qps gbps inflight nic_peak_rx_gbps")
for row in rows:
    print(
        f"{row['case']} {row['valid']} {row['pressure_key_footprint_mib']:.1f} "
        f"{row['business_get_total_ms']:.3f} {row['datasystem_get_ms']:.3f} "
        f"{row['runner_ms']:.3f} {row['dsbench_get_qps']:.3f} "
        f"{row['dsbench_get_gbps']:.3f} {row['dsbench_get_max_inflight']:.0f} "
        f"{row['nic_peak_rx_gbps']:.3f}"
    )
print(f"summary_json={root / 'summary.json'}")
PY
