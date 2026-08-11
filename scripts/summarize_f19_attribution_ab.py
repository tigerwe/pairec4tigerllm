#!/usr/bin/env python3
"""Summarize paired F19 DataSystem attribution A/B runs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def load_run(root: Path, pair: int, mode: str, expected: int) -> dict:
    run_dir = root / f"pair-{pair}-{mode}"
    summary_path = run_dir / "brpc" / "summary.json"
    requests_path = run_dir / "brpc" / "requests.tsv"
    workload_path = run_dir / "brpc" / "workload.json"
    summary = json.loads(summary_path.read_text())
    workload = json.loads(workload_path.read_text())
    with requests_path.open(encoding="utf-8") as stream:
        latencies = [float(row["e2e_ms"]) for row in csv.DictReader(stream, delimiter="\t")]
    if len(latencies) != expected:
        raise ValueError(
            f"{run_dir}: expected {expected} requests, found {len(latencies)}"
        )
    if summary.get("classification") != "PAIREC_BRPC_PIPELINE_TRACE_OK":
        raise ValueError(f"{run_dir}: pipeline trace did not pass")
    if int(summary.get("valid_count", -1)) != expected:
        raise ValueError(f"{run_dir}: valid_count does not match expected requests")
    if int(summary.get("missing_count", -1)) or int(summary.get("invalid_count", -1)):
        raise ValueError(f"{run_dir}: missing or invalid request traces")
    if mode == "enabled" and int(summary.get("datasystem_complete_count", -1)) != expected:
        raise ValueError(f"{run_dir}: native DataSystem completion count is incomplete")

    client = summary["client_e2e"]
    runner = summary.get("service_phases", {}).get(
        "generative_recall.runner_generate_us", {}
    )
    datasystem = summary.get("datasystem", {})
    if int(workload.get("requests", -1)) != expected:
        raise ValueError(f"{run_dir}: workload request count does not match expected")
    elapsed_seconds = float(workload.get("elapsed_seconds", 0.0))
    if elapsed_seconds <= 0:
        raise ValueError(f"{run_dir}: workload elapsed time must be positive")
    return {
        "pair": pair,
        "mode": mode,
        "requests": len(latencies),
        "client_avg_ms": float(client["avg_ms"]),
        "client_p99_ms": float(client["p99_ms"]),
        "throughput_rps": expected / elapsed_seconds,
        "runner_avg_ms": float(runner.get("avg_ms", 0.0)),
        "runner_p99_ms": float(runner.get("p99_ms", 0.0)),
        "datasystem_get_count": int(datasystem.get("get_count", 0)),
        "datasystem_set_count": int(datasystem.get("set_count", 0)),
        "datasystem_get_avg_ms": float(datasystem.get("get", {}).get("avg_ms", 0.0)),
        "datasystem_set_avg_ms": float(datasystem.get("set", {}).get("avg_ms", 0.0)),
        "run_dir": str(run_dir),
    }


def summarize(args: argparse.Namespace) -> dict:
    root = Path(args.input_root)
    runs = []
    comparisons = []
    for pair in range(1, args.pairs + 1):
        disabled = load_run(root, pair, "disabled", args.expected_requests)
        enabled = load_run(root, pair, "enabled", args.expected_requests)
        runs.extend((disabled, enabled))
        comparisons.append(
            {
                "pair": pair,
                "avg_overhead_ms": enabled["client_avg_ms"] - disabled["client_avg_ms"],
                "p99_overhead_ms": enabled["client_p99_ms"] - disabled["client_p99_ms"],
                "throughput_loss_pct": 100.0
                * (disabled["throughput_rps"] - enabled["throughput_rps"])
                / disabled["throughput_rps"],
                "runner_avg_overhead_ms": enabled["runner_avg_ms"]
                - disabled["runner_avg_ms"],
                "runner_p99_overhead_ms": enabled["runner_p99_ms"]
                - disabled["runner_p99_ms"],
            }
        )

    medians = {
        name: statistics.median(item[name] for item in comparisons)
        for name in (
            "avg_overhead_ms",
            "p99_overhead_ms",
            "throughput_loss_pct",
            "runner_avg_overhead_ms",
            "runner_p99_overhead_ms",
        )
    }
    gates = {
        "avg_overhead": medians["avg_overhead_ms"] <= args.max_avg_overhead_ms,
        "p99_overhead": medians["p99_overhead_ms"] <= args.max_p99_overhead_ms,
        "throughput_loss": medians["throughput_loss_pct"] <= args.max_throughput_loss_pct,
    }
    return {
        "classification": (
            "F19_ATTRIBUTION_AB_PASS" if all(gates.values()) else "F19_ATTRIBUTION_AB_FAIL"
        ),
        "pairs": args.pairs,
        "expected_requests_per_run": args.expected_requests,
        "budgets": {
            "max_avg_overhead_ms": args.max_avg_overhead_ms,
            "max_p99_overhead_ms": args.max_p99_overhead_ms,
            "max_throughput_loss_pct": args.max_throughput_loss_pct,
        },
        "runs": runs,
        "comparisons": comparisons,
        "pair_medians": medians,
        "gates": gates,
    }


def print_summary(result: dict) -> None:
    print("pair mode client_avg_ms client_p99_ms throughput_rps runner_avg_ms runner_p99_ms ds_get ds_set")
    for run in result["runs"]:
        print(
            f'{run["pair"]:>4} {run["mode"]:>8} '
            f'{run["client_avg_ms"]:>13.3f} {run["client_p99_ms"]:>13.3f} '
            f'{run["throughput_rps"]:>14.3f} {run["runner_avg_ms"]:>13.3f} '
            f'{run["runner_p99_ms"]:>13.3f} {run["datasystem_get_count"]:>6} '
            f'{run["datasystem_set_count"]:>6}'
        )
    print("pair avg_delta_ms p99_delta_ms throughput_loss_pct runner_avg_delta_ms runner_p99_delta_ms")
    for item in result["comparisons"]:
        print(
            f'{item["pair"]:>4} {item["avg_overhead_ms"]:>12.3f} '
            f'{item["p99_overhead_ms"]:>12.3f} {item["throughput_loss_pct"]:>19.3f} '
            f'{item["runner_avg_overhead_ms"]:>19.3f} '
            f'{item["runner_p99_overhead_ms"]:>19.3f}'
        )
    medians = result["pair_medians"]
    print(
        "median "
        f'avg_delta_ms={medians["avg_overhead_ms"]:.3f} '
        f'p99_delta_ms={medians["p99_overhead_ms"]:.3f} '
        f'throughput_loss_pct={medians["throughput_loss_pct"]:.3f} '
        f'runner_avg_delta_ms={medians["runner_avg_overhead_ms"]:.3f} '
        f'runner_p99_delta_ms={medians["runner_p99_overhead_ms"]:.3f}'
    )
    print(f'classification={result["classification"]}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--expected-requests", type=int, default=1000)
    parser.add_argument("--max-avg-overhead-ms", type=float, default=0.1)
    parser.add_argument("--max-p99-overhead-ms", type=float, default=0.5)
    parser.add_argument("--max-throughput-loss-pct", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.pairs < 3:
        parser.error("--pairs must be at least 3")
    if args.expected_requests < 1:
        parser.error("--expected-requests must be positive")

    result = summarize(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print_summary(result)
    if result["classification"] != "F19_ATTRIBUTION_AB_PASS":
        raise SystemExit(1)
    print("F19_ATTRIBUTION_AB_OK")


if __name__ == "__main__":
    main()
