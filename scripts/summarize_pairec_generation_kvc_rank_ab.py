#!/usr/bin/env python3
import argparse
import json
import pathlib


KEY_METRICS = (
    "client_e2e_ms",
    "pairec_total_ms",
    "generative_recall_ms",
    "front_brpc_ms",
    "runner_ms",
    "kvc_business_get_ms",
    "kvc_business_get_1_ms",
    "kvc_business_get_2_ms",
    "datasystem_get_ms",
    "datasystem_set_ms",
    "native_add_token_ms",
    "rank_business_client_ms",
    "rank_front_brpc_ms",
    "rank_service_ms",
    "rank_inference_to_business_gap_ms",
    "rank_pressure_p95_ms",
    "rank_pressure_max_active",
    "rank_pressure_tail_after_business_ms",
    "rank_pressure_rerank_overlap_ms",
    "rank_pressure_pipeline_overlap_ms",
)


def summarize(control_path, pressure_path):
    control = json.loads(pathlib.Path(control_path).read_text())
    pressure = json.loads(pathlib.Path(pressure_path).read_text())
    for name, case in (("rank_c1", control), ("rank_c1000", pressure)):
        assert case["classification"].endswith("_OK"), (name, case["classification"])
        assert case["wrapper_concurrency"] == 1000, (name, case)
        assert case["brpc_burst_pool_size"] == 10000, (name, case)
        assert case["kvc_concurrency"] == 32, (name, case)
        assert case["rank_burst_enabled"] is True, (name, case)
        assert case["rank_business_payload_bytes"] == 102400, (name, case)
        assert case["rank_pressure_payload_bytes"] == 102400, (name, case)
        assert case["rank_endpoint_source"] == "override", (name, case)
        assert case["rank_endpoint"], (name, case)
    assert control["rank_burst_concurrency"] == 1, control
    assert pressure["rank_burst_concurrency"] == 1000, pressure
    assert control["rank_endpoint"] == pressure["rank_endpoint"], (
        control["rank_endpoint"], pressure["rank_endpoint"])
    assert len(control["samples"]) == len(pressure["samples"]), (
        len(control["samples"]), len(pressure["samples"]))
    metrics = {}
    common = sorted(set(control["metrics"]) & set(pressure["metrics"]))
    for name in common:
        left = control["metrics"][name]
        right = pressure["metrics"][name]
        metrics[name] = {
            "rank_c1_avg": float(left["avg"]),
            "rank_c1000_avg": float(right["avg"]),
            "delta_avg": float(right["avg"]) - float(left["avg"]),
            "rank_c1_p99": float(left["p99"]),
            "rank_c1000_p99": float(right["p99"]),
            "delta_p99": float(right["p99"]) - float(left["p99"]),
        }
    assert pressure["metrics"]["rank_pressure_success"]["avg"] == 999, pressure
    assert pressure["metrics"]["rank_pressure_errors"]["max"] == 0, pressure
    return {
        "classification": "PAIREC_GENERATION_C1000_KVC_C32_RANK_AB_OK",
        "valid": True,
        "requests_per_case": len(control["samples"]),
        "fixed_pressure": {
            "generative_brpc_concurrency": 1000,
            "generative_pressure_requests": 999,
            "generative_preconnected_sessions": 10000,
            "generative_payload_bytes": 102400,
            "kvc_concurrency": 32,
            "kvc_pressure_lanes": 31,
            "kvc_object_size_bytes": control["kvc_object_size_bytes"],
            "kvc_pressure_key_count": control["kvc_pressure_key_count"],
        },
        "treatment": {
            "factor": "rank_brpc_pressure",
            "control_concurrency": 1,
            "pressure_concurrency": 1000,
            "pressure_requests": 999,
            "business_payload_bytes": 102400,
            "pressure_payload_bytes": 102400,
            "rank_endpoint": pressure["rank_endpoint"],
        },
        "rank_c1": control,
        "rank_c1000": pressure,
        "comparison": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-c1", required=True)
    parser.add_argument("--rank-c1000", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.rank_c1, args.rank_c1000)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("metric rank_c1_avg rank_c1000_avg delta_avg rank_c1_p99 rank_c1000_p99 delta_p99")
    for name in KEY_METRICS:
        values = result["comparison"][name]
        print(name, *(f'{values[key]:.3f}' for key in (
            "rank_c1_avg", "rank_c1000_avg", "delta_avg",
            "rank_c1_p99", "rank_c1000_p99", "delta_p99")))
    print(f"summary_json={args.output}")
    print(result["classification"])


if __name__ == "__main__":
    main()
