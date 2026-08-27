#!/usr/bin/env python3
import argparse
import json
import pathlib


KEY_METRICS = (
    "client_e2e_ms",
    "pairec_total_ms",
    "generative_recall_ms",
    "deepfm_rank_ms",
    "rank_service_ms",
    "rank_front_brpc_ms",
    "rank_kvc_business_get_ms",
    "rank_kvc_pressure_p99_ms",
    "rank_kvc_pressure_tail_ms",
    "rank_kvc_rank_brpc_overlap_ms",
)


def summarize(control_path, treatment_path):
    control = json.loads(pathlib.Path(control_path).read_text())
    treatment = json.loads(pathlib.Path(treatment_path).read_text())
    assert control["rank_kvc_enabled"] and treatment["rank_kvc_enabled"]
    assert control["rank_kvc_concurrency"] == 1, control
    assert treatment["rank_kvc_concurrency"] == 32, treatment
    for case in (control, treatment):
        assert case["rank_burst_concurrency"] == 1000, case
        assert case["rank_kvc_object_size_bytes"] == 8388608, case
        assert case["response_semantic_fingerprints"], case
    assert (
        control["response_semantic_fingerprints"]
        == treatment["response_semantic_fingerprints"]
    ), "Rank KVC c1/c32 changed recommendation semantics"
    comparison = {}
    for name in KEY_METRICS:
        left = control["metrics"][name]
        right = treatment["metrics"][name]
        comparison[name] = {
            "control_avg": left["avg"],
            "treatment_avg": right["avg"],
            "delta_avg": right["avg"] - left["avg"],
            "control_p99": left["p99"],
            "treatment_p99": right["p99"],
            "delta_p99": right["p99"] - left["p99"],
        }
    return {
        "classification": "PAIREC_GENERATION_C1000_KVC_C32_RANK_C1000_KVC_AB_OK",
        "control": control,
        "treatment": treatment,
        "semantic_match": True,
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.control, args.treatment)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("metric control_avg treatment_avg delta_avg control_p99 treatment_p99 delta_p99")
    for name, item in result["comparison"].items():
        print(name, *(f"{item[key]:.3f}" for key in (
            "control_avg", "treatment_avg", "delta_avg",
            "control_p99", "treatment_p99", "delta_p99")))
    print("classification=" + result["classification"])
    print("summary_json=" + args.output)


if __name__ == "__main__":
    main()
