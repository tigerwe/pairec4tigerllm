#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import re
import statistics
from datetime import datetime


def events(path):
    output = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
            prefix = line[:start].strip()
            if prefix:
                try:
                    timestamp = prefix.split()[0]
                    if timestamp.endswith("Z"):
                        timestamp = timestamp[:-1]
                        whole, separator, fraction = timestamp.partition(".")
                        if separator:
                            timestamp = whole + "." + (fraction + "000000")[:6]
                        timestamp += "+00:00"
                    event["_log_epoch_ns"] = int(
                        datetime.fromisoformat(timestamp).timestamp() * 1e9)
                except ValueError:
                    pass
            output.append(event)
        except json.JSONDecodeError:
            pass
    return output


def key_values(line):
    return dict(re.findall(r"([a-zA-Z0-9_]+)=([^ ]+)", line))


def overlap_ms(left_start, left_end, right_start, right_end):
    return max(0, min(left_end, right_end) - max(left_start, right_start)) / 1e6


def summarize(case, expected, business_bytes, pressure_bytes):
    root = pathlib.Path(case)
    with (root / "requests.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    pairec = events(root / "pairec-rank.log")
    inference = events(root / "inference.log")
    wrapper_lines = (root / "rank-wrapper.log").read_text(errors="replace").splitlines()
    samples = []
    for row in rows:
        request_id = row["request_id"]
        own = [item for item in pairec if item.get("request_id") == request_id]

        def one(name):
            values = [item for item in own if item.get("event") == name]
            assert len(values) == 1, (request_id, name, values)
            return values[0]

        start = one("pairec_rank_brpc_burst_start")
        business = one("pairec_rank_brpc_burst_business_complete")
        complete = one("pairec_rank_brpc_burst_complete")
        rank = one("deepfm_rank_complete")
        rerank = one("source_quota_rerank_complete")
        pipeline = one("pipeline_trace_complete")
        executor_values = [item for item in inference
                           if item.get("event") == "trt_executor_request_complete"
                           and item.get("request_id") == request_id]
        assert len(executor_values) == 1, (request_id, executor_values)
        executor = executor_values[0]
        assert start["concurrency"] == expected, start
        assert start["armed_workers"] == expected, start
        assert start["business_payload_bytes"] == business_bytes, start
        assert start["pressure_payload_bytes"] == pressure_bytes, start
        assert business["business_success"] and business["trace_valid"], business
        assert business["business_payload_bytes"] == business_bytes, business
        assert complete["pressure_requests"] == expected - 1, complete
        assert complete["pressure_success"] == expected - 1, complete
        assert complete["pressure_errors"] == 0 and complete["burst_valid"], complete
        if expected > 1:
            assert complete["pressure_overlap_business"] > 0, complete
        assert rank["candidate_count"] == 50 and rank["reordered"] is True, rank
        assert rank["model_version"] and rank["model_role"], rank
        assert pipeline["valid"] is True and pipeline["status"] == "ok", pipeline
        assert rerank["status"] == "ok", rerank
        generative_spans = [span for span in pipeline.get("spans", [])
                            if span.get("name") == "generative_recall"]
        assert len(generative_spans) == 1, (request_id, generative_spans)
        generative_span = generative_spans[0]
        assert generative_span.get("status") == "ok", generative_span
        matching = [line for line in wrapper_lines
                    if "[brpc-rank-burst-wrapper] method=Rank" in line
                    and f"request_id={request_id}" in line]
        assert len(matching) == 1, (request_id, matching)
        wrapper = key_values(matching[0])
        assert int(wrapper["front_payload_bytes"]) == business_bytes, wrapper
        assert int(wrapper["backend_payload_bytes"]) == 0, wrapper
        assert int(wrapper["code"]) == 200, wrapper
        pressure_start = int(complete["rank_pressure_first_start_epoch_ns"] or
                             business["rank_business_start_epoch_ns"])
        pressure_end = int(complete["rank_pressure_last_end_epoch_ns"] or
                           business["rank_business_end_epoch_ns"])
        rank_end = int(business["rank_business_end_epoch_ns"])
        # Pipeline and Rank epochs are both generated by PaiRec on master. The
        # inference log timestamp is generated on worker1 and is diagnostic only
        # because the two nodes are not guaranteed to have synchronized clocks.
        inference_call_end = int(pipeline["start_epoch_ns"]) + 1000 * (
            int(generative_span["start_offset_us"]) +
            int(generative_span["duration_us"]))
        rank_start = int(business["rank_business_start_epoch_ns"])
        assert inference_call_end <= rank_start, (generative_span, business)
        inference_log_epoch = int(executor.get("completion_epoch_ns") or
                                  executor.get("_log_epoch_ns") or 0)
        response_end = int(row["response_end_epoch_ns"])
        sample = {
            "request_id": request_id,
            "client_e2e_ms": float(row["e2e_ms"]),
            "rank_business_client_ms": float(business["business_client_wall_ms"]),
            "rank_front_brpc_ms": float(business["front_brpc_estimate_ms"]),
            "rank_pressure_p95_ms": float(complete["pressure_latency_p95_ms"]),
            "rank_pressure_max_active": int(complete["max_active_workers"]),
            "rank_pressure_start_skew_us": int(complete["start_skew_us"]),
            "rank_pressure_tail_after_business_ms": float(
                complete["pressure_tail_after_business_ms"]),
            "rank_pressure_rerank_overlap_ms": overlap_ms(
                pressure_start, pressure_end, int(rerank["start_epoch_ns"]),
                int(rerank["end_epoch_ns"])),
            "rank_pressure_pipeline_overlap_ms": overlap_ms(
                pressure_start, pressure_end, rank_end, int(pipeline["end_epoch_ns"])),
            "rank_pressure_tail_after_http_ms": max(0, pressure_end - response_end) / 1e6,
            "wrapper_health_calls_during_rank": int(wrapper["health_calls_during_rank"]),
            "wrapper_health_payload_bytes_during_rank": int(
                wrapper["health_payload_bytes_during_rank"]),
            "inference_to_rank_gap_ms": (rank_start - inference_call_end) / 1e6,
            "cross_node_inference_log_delta_ms": (
                (inference_log_epoch - rank_start) / 1e6
                if inference_log_epoch else 0),
            "inference_complete_before_rank": True,
        }
        if expected > 1:
            assert sample["wrapper_health_calls_during_rank"] > 0, wrapper
            assert sample["wrapper_health_payload_bytes_during_rank"] > 0, wrapper
        samples.append(sample)
    metrics = {}
    for name in samples[0]:
        if name == "request_id" or isinstance(samples[0][name], bool):
            continue
        values = [float(sample[name]) for sample in samples]
        assert all(math.isfinite(value) for value in values), (name, values)
        metrics[name] = {"avg": statistics.fmean(values), "min": min(values),
                         "max": max(values)}
    return {"classification": f"PAIREC_RANK_BRPC_BURST_C{expected}_OK",
            "valid": True, "concurrency": expected, "requests": len(samples),
            "business_payload_bytes": business_bytes,
            "pressure_payload_bytes": pressure_bytes, "samples": samples,
            "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--business-bytes", default=102400, type=int)
    parser.add_argument("--pressure-bytes", default=102400, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.case, args.concurrency, args.business_bytes,
                       args.pressure_bytes)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(result["classification"])
    print("metric avg min max")
    for name, values in result["metrics"].items():
        print(name, f'{values["avg"]:.3f}', f'{values["min"]:.3f}',
              f'{values["max"]:.3f}')


if __name__ == "__main__":
    main()
