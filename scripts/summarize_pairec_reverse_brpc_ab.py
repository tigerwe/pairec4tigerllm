#!/usr/bin/env python3
import argparse
import json
import pathlib
import statistics


KEY_METRICS = (
    "client_e2e_ms",
    "pairec_total_ms",
    "generative_recall_ms",
    "deepfm_rank_ms",
    "generation_reverse_marker_wall_ms",
    "generation_reverse_marker_front_ms",
    "generation_reverse_pressure_p95_ms",
    "generation_reverse_tail_ms",
    "generation_reverse_tail_rank_overlap_ms",
    "rank_reverse_marker_wall_ms",
    "rank_reverse_marker_front_ms",
    "rank_reverse_pressure_p95_ms",
    "rank_reverse_tail_ms",
    "rank_reverse_tail_rerank_overlap_ms",
)


def distribution(values):
    mean = statistics.fmean(values)
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    cv = statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0
    return {"count": len(values), "avg": mean, "median": median, "mad": mad,
            "cv": cv, "min": min(values), "max": max(values)}


def summarize(path):
    source = json.loads(pathlib.Path(path).read_text())
    samples = source["samples"]
    pattern = source["reverse_burst_pattern"].split(",")
    assert len(samples) == len(pattern) == 20, (len(samples), pattern)
    assert pattern == [label for _ in range(5) for label in ("A", "B", "B", "A")]
    assert sum(int(item["reverse_burst_treatment"]) for item in samples) == 10
    assert len(set(source["response_semantic_fingerprints"])) == 1, (
        "reverse BRPC changed recommendation semantics",
        source["response_semantic_fingerprints"],
    )
    metrics = {}
    for name in KEY_METRICS:
        control = [float(item[name]) for item in samples
                   if not int(item["reverse_burst_treatment"])]
        treatment = [float(item[name]) for item in samples
                     if int(item["reverse_burst_treatment"])]
        metrics[name] = {
            "control": distribution(control),
            "treatment": distribution(treatment),
            "delta_avg": statistics.fmean(treatment) - statistics.fmean(control),
        }
    paired = []
    for block in range(5):
        offset = block * 4
        for control_index, treatment_index in ((offset, offset + 1),
                                                (offset + 3, offset + 2)):
            paired.append({
                "block": block + 1,
                "control_request_id": samples[control_index]["request_id"],
                "treatment_request_id": samples[treatment_index]["request_id"],
                "client_e2e_delta_ms": (
                    samples[treatment_index]["client_e2e_ms"] -
                    samples[control_index]["client_e2e_ms"]),
                "pairec_total_delta_ms": (
                    samples[treatment_index]["pairec_total_ms"] -
                    samples[control_index]["pairec_total_ms"]),
            })
    return {
        "classification": "PAIREC_REVERSE_BRPC_ABBA_OK",
        "pattern": pattern,
        "control_samples": 10,
        "treatment_samples": 10,
        "semantic_match": True,
        "metrics": metrics,
        "paired_deltas": paired,
        "source_summary": str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("metric control_avg treatment_avg delta_avg control_cv treatment_cv control_mad treatment_mad")
    for name, item in result["metrics"].items():
        print(name, *(f"{value:.3f}" for value in (
            item["control"]["avg"], item["treatment"]["avg"], item["delta_avg"],
            item["control"]["cv"], item["treatment"]["cv"],
            item["control"]["mad"], item["treatment"]["mad"],
        )))
    print("classification=" + result["classification"])
    print("summary_json=" + args.output)


if __name__ == "__main__":
    main()
