#!/usr/bin/env python3
"""Summarize complete and partial PaiRec BRPC/KVC matrix artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read_json_lines(path: pathlib.Path, event_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == event_name:
            events.append(value)
    return events


def read_native(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == "datasystem_request_complete":
            return value
    return None


def classify(native: dict[str, Any] | None, burst: dict[str, Any] | None) -> str:
    if native is None:
        return "missing_native_completion"
    if native.get("get_count", 0) == 0:
        return "zero_onboard"
    if burst is None:
        return "missing_kvc_completion"
    if not burst.get("valid", False):
        return f"kvc_invalid_failure_{burst.get('failure', 'unknown')}"
    return "valid"


def scan_phase(phase_dir: pathlib.Path, required_gets: int) -> dict[str, Any]:
    rows = []
    for round_dir in sorted(phase_dir.glob("round-*"), key=lambda p: int(p.name.split("-")[-1])):
        replay = round_dir / "replay"
        native = read_native(replay / "brpc_trtllm.log")
        bursts = read_json_lines(replay / "kvc_burst.log", "kvc_burst_complete")
        burst = bursts[-1] if bursts else None
        row: dict[str, Any] = {
            "round": int(round_dir.name.split("-")[-1]),
            "classification": classify(native, burst),
        }
        if native is not None:
            row.update({
                "request_id": native.get("request_id"),
                "get_count": native.get("get_count", 0),
                "set_count": native.get("set_count", 0),
                "datasystem_get_ms": native.get("get_us", 0) / 1000.0,
                "datasystem_set_ms": native.get("set_us", 0) / 1000.0,
            })
        if burst is not None:
            row.update({
                "kvc_business_get_ms": burst.get("business_get_ms"),
                "kvc_pressure_p99_ms": burst.get("pressure_get_p99_ms"),
                "business_submit_rank": burst.get("business_submit_rank"),
                "pressure_inflight_at_business_start": burst.get(
                    "pressure_inflight_at_business_start"),
                "pressure_key_count": burst.get("pressure_key_count"),
                "object_size_bytes": burst.get("object_size_bytes"),
            })
        row["performance_valid"] = (
            native is not None
            and native.get("get_count", 0) >= required_gets
            and burst is not None
            and burst.get("valid", False)
        )
        rows.append(row)

    valid = [row for row in rows if row["performance_valid"]]
    return {"rows": rows, "valid": valid}


def stats(rows: list[dict[str, Any]], field: str) -> dict[str, float] | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "avg": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=pathlib.Path)
    parser.add_argument("--required-gets", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.required_gets < 1:
        parser.error("--required-gets must be positive")

    result: dict[str, Any] = {
        "run_root": str(args.run_root),
        "required_gets": args.required_gets,
        "phases": {},
    }
    for phase in ("baseline", "combined"):
        scanned = scan_phase(args.run_root / phase / "contention", args.required_gets)
        valid = scanned["valid"]
        result["phases"][phase] = {
            "total_rounds": len(scanned["rows"]),
            "valid_rounds": len(valid),
            "invalid_rounds": [
                {"round": row["round"], "classification": row["classification"]}
                for row in scanned["rows"] if not row["performance_valid"]
            ],
            "config": next(({
                "pressure_key_count": row.get("pressure_key_count"),
                "object_size_bytes": row.get("object_size_bytes"),
            } for row in valid if row.get("pressure_key_count") is not None), {}),
            "metrics": {
                field: stats(valid, field) for field in (
                    "kvc_business_get_ms", "datasystem_get_ms", "datasystem_set_ms",
                    "kvc_pressure_p99_ms", "business_submit_rank",
                    "pressure_inflight_at_business_start",
                )
            },
            "rows": scanned["rows"],
        }

    baseline = result["phases"]["baseline"]["metrics"]
    combined = result["phases"]["combined"]["metrics"]
    comparison = []
    for field in ("kvc_business_get_ms", "datasystem_get_ms", "datasystem_set_ms"):
        left, right = baseline.get(field), combined.get(field)
        if left and right:
            comparison.append({
                "metric": field,
                "baseline_avg_ms": left["avg"],
                "combined_avg_ms": right["avg"],
                "delta_avg_ms": right["avg"] - left["avg"],
                "baseline_p99_ms": left["p99"],
                "combined_p99_ms": right["p99"],
                "delta_p99_ms": right["p99"] - left["p99"],
            })
    result["comparison"] = comparison
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)

    for phase in ("baseline", "combined"):
        item = result["phases"][phase]
        print(f"{phase}: valid={item['valid_rounds']}/{item['total_rounds']}")
        print(f"  invalid={item['invalid_rounds']}")
        config = item["config"]
        if config:
            print(f"  pressure_key_count={config.get('pressure_key_count')} "
                  f"object_size_bytes={config.get('object_size_bytes')}")
        for field, value in item["metrics"].items():
            if value:
                print(f"  {field}: avg={value['avg']:.3f} p99={value['p99']:.3f} "
                      f"count={value['count']}")
    if comparison:
        print("metric baseline_avg_ms combined_avg_ms delta_avg_ms")
        for row in comparison:
            print(f"{row['metric']} {row['baseline_avg_ms']:.3f} "
                  f"{row['combined_avg_ms']:.3f} {row['delta_avg_ms']:+.3f}")
    if args.output:
        print(f"summary_json={args.output}")


if __name__ == "__main__":
    main()
