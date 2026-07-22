#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_RANGES = {
    "e2e_ms": (320.0, 360.0),
    "brpc_ms": (40.0, 50.0),
    "kvc_ms": (200.0, 210.0),
    "server_other_ms": (80.0, 90.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a real-overload BRPC/KVC result against the latency target."
    )
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--stat", choices=("avg", "p50", "p95"), default="avg")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    metrics = result.get("metrics") or {}
    checks = []

    for name, (lower, upper) in DEFAULT_RANGES.items():
        stage = metrics.get(name) or {}
        value = stage.get(args.stat)
        passed = value is not None and lower <= float(value) <= upper
        checks.append(
            {
                "metric": name,
                "stat": args.stat,
                "value": value,
                "lower": lower,
                "upper": upper,
                "passed": passed,
            }
        )

    shape_ok = all(
        row.get("offload_count") == 3 and row.get("onboard_count") == 2
        for row in result.get("rows") or []
    )
    sample_ok = (
        result.get("status") == "PASS"
        and result.get("valid_repeats") == result.get("expected_repeats")
        and int(result.get("valid_repeats") or 0) > 0
    )
    passed = sample_ok and shape_ok and all(check["passed"] for check in checks)
    report = {
        "status": "PASS" if passed else "FAIL",
        "source": str(args.result_json),
        "sample_ok": sample_ok,
        "cache_shape_3_set_2_get": shape_ok,
        "checks": checks,
    }

    print("BRPC/KVC real-overload target evaluation")
    print(f"  source={args.result_json}")
    print(f"  sample_ok={sample_ok} cache_shape_3_set_2_get={shape_ok}")
    for check in checks:
        value = "n/a" if check["value"] is None else f"{float(check['value']):.3f}"
        print(
            f"  {check['metric']}.{check['stat']}={value} "
            f"target=[{check['lower']:.3f},{check['upper']:.3f}] "
            f"passed={check['passed']}"
        )
    print(f"result={report['status']}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
