#!/usr/bin/env python3
"""Validate and summarize PaiRec request-level pipeline traces."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


REQUIRED_SPANS = {
    "recommend_service", "response_build", "controller_overhead",
    "user_feature", "recall", "filter", "general_rank", "feature_load",
    "framework_rank", "pipeline_wait", "pipeline_merge", "sort",
    "generative_recall", "vector_recall", "deepfm_rank", "rerank",
}
REQUIRED_SERVICE_FIELDS = {
    "generative_recall": {
        "rpc_us", "inference_total_us", "runner_generate_us",
        "runner_per_output_token_us", "output_token_count",
    },
    "vector_recall": {"service_total_us", "feature_us", "compute_us"},
    "deepfm_rank": {"service_total_us", "feature_us", "compute_us", "backend_rpc_us"},
}


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def metric(values):
    return {
        "count": len(values),
        "avg_ms": round(sum(values) / len(values) / 1000, 3) if values else 0.0,
        "p50_ms": round(percentile(values, 0.50) / 1000, 3),
        "p95_ms": round(percentile(values, 0.95) / 1000, 3),
        "p99_ms": round(percentile(values, 0.99) / 1000, 3),
        "max_ms": round(max(values) / 1000, 3) if values else 0.0,
    }


def count_metric(values):
    return {
        "count": len(values),
        "avg": round(sum(values) / len(values), 3) if values else 0.0,
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": max(values) if values else 0,
    }


def extract_traces(path):
    traces = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if "pipeline_trace_complete" not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        try:
            trace = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        request_id = trace.get("request_id")
        if request_id:
            traces[request_id] = trace
    return traces


def extract_datasystem_completions(path):
    completions = {}
    duplicates = set()
    if not path:
        return completions, duplicates
    for line in Path(path).read_text(errors="replace").splitlines():
        if '"event":"datasystem_request_complete"' not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        request_id = event.get("request_id")
        if not request_id:
            continue
        if request_id in completions:
            duplicates.add(request_id)
        completions[request_id] = event
    return completions, duplicates


def extract_quota_stats(path):
    fields = (
        "primary_minimum", "primary_input", "secondary_input", "primary_selected",
        "secondary_selected", "duplicate_count", "final_count",
    )
    pattern = re.compile(
        r"request_id=(?P<request_id>[^ ]+).*module=QuotaMultiRecall.*" +
        ".*".join(rf"{field}=(?P<{field}>[0-9]+)" for field in fields) +
        r".*degraded=(?P<degraded>true|false)")
    result = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if "module=QuotaMultiRecall" not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        values = {field: int(match.group(field)) for field in fields}
        values["degraded"] = match.group("degraded") == "true"
        result[match.group("request_id")] = values
    return result


def read_requests(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {row["request_id"]: round(float(row["e2e_ms"]) * 1000)
                for row in rows}


def validate(trace, require_datasystem, require_source_rerank=False):
    reasons = list(trace.get("invalid_reasons") or [])
    if trace.get("valid") is not True:
        reasons.append("trace_marked_invalid")
    if trace.get("contract_version") != "pairec.pipeline_trace.v1":
        reasons.append("contract_version")
    spans = trace.get("spans") or []
    names = {span.get("name") for span in spans}
    reasons.extend(f"missing_span:{name}" for name in sorted(REQUIRED_SPANS - names))
    for span in spans:
        if span.get("duration_us", 0) < 0 or span.get("start_offset_us", 0) < 0:
            reasons.append(f"invalid_duration:{span.get('name')}")
    rerank = next((span for span in spans if span.get("name") == "rerank"), None)
    if require_source_rerank:
        if not rerank or rerank.get("enabled") is not True:
            reasons.append("source_rerank_not_enabled")
        elif rerank.get("status") != "ok" or rerank.get("protocol") != "in_process":
            reasons.append("source_rerank_status")
        else:
            attributes = rerank.get("attributes") or {}
            required = {
                "policy", "placement", "input_count", "output_count",
                "generative_input", "vector_input", "generative_selected",
                "vector_selected", "minimum_generative", "maximum_generative",
                "moved_count",
            }
            for field in sorted(required - set(attributes)):
                reasons.append(f"missing_rerank_field:{field}")
            if required <= set(attributes):
                if attributes["policy"] != "source_quota_tail" or attributes["placement"] != "tail":
                    reasons.append("source_rerank_policy")
                selected = int(attributes["generative_selected"])
                minimum = int(attributes["minimum_generative"])
                maximum = int(attributes["maximum_generative"])
                available = int(attributes["generative_input"])
                output_count = int(attributes["output_count"])
                vector_selected = int(attributes["vector_selected"])
                if selected < minimum or selected > maximum:
                    reasons.append("source_rerank_quota")
                if selected != min(maximum, available, output_count):
                    reasons.append("source_rerank_did_not_preserve_available")
                if selected + vector_selected != output_count:
                    reasons.append("source_rerank_output_count")
    protocols = {span.get("name"): span.get("protocol") for span in spans}
    for name in ("generative_recall", "vector_recall", "deepfm_rank"):
        if protocols.get(name) != "brpc":
            reasons.append(f"not_brpc:{name}")
    spans_by_name = {span.get("name"): span for span in spans}
    for name, required_fields in REQUIRED_SERVICE_FIELDS.items():
        attributes = (spans_by_name.get(name) or {}).get("attributes") or {}
        for field in sorted(required_fields - set(attributes)):
            reasons.append(f"missing_service_field:{name}:{field}")
    generative_attributes = (spans_by_name.get("generative_recall") or {}).get(
        "attributes") or {}
    if int(generative_attributes.get("output_token_count", 0)) <= 0:
        reasons.append("invalid_service_field:generative_recall:output_token_count")
    if int(generative_attributes.get("runner_per_output_token_us", 0)) <= 0:
        reasons.append("invalid_service_field:generative_recall:runner_per_output_token_us")
    total_us = int(trace.get("pairec_total_us", -1))
    accounted_us = int(trace.get("accounted_us", -1))
    closure_error_us = abs(total_us - accounted_us)
    threshold_us = max(3000, round(max(total_us, 0) * 0.05))
    if closure_error_us > threshold_us:
        reasons.append("closure_error")
    if require_datasystem:
        datasystem = trace.get("_datasystem_final")
        if not datasystem:
            reasons.append("datasystem_completion_missing")
        else:
            if datasystem.get("attribution_complete") is not True:
                reasons.append("datasystem_attribution_incomplete")
            for field in ("get_count", "get_us", "set_count", "set_us",
                          "get_failed_count", "set_failed_count",
                          "pending_count", "unknown_count"):
                if not isinstance(datasystem.get(field), int) or datasystem[field] < 0:
                    reasons.append(f"datasystem_invalid_field:{field}")
            for field in ("get_failed_count", "set_failed_count",
                          "pending_count", "unknown_count"):
                if isinstance(datasystem.get(field), int) and datasystem[field] != 0:
                    reasons.append(f"datasystem_nonzero:{field}")
    quota = trace.get("_quota")
    if not quota:
        reasons.append("missing_quota_multi_recall")
    else:
        if quota["primary_minimum"] < 1:
            reasons.append("invalid_primary_minimum")
        if quota["primary_selected"] < quota["primary_minimum"]:
            reasons.append("generative_recall_not_selected")
        if quota["final_count"] != 50:
            reasons.append("recall_final_count")
        if quota["degraded"]:
            reasons.append("quota_multi_recall_degraded")
    return sorted(set(reasons))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--requests-tsv")
    parser.add_argument("--expected", type=int, default=0)
    parser.add_argument("--require-datasystem-attribution", action="store_true")
    parser.add_argument("--datasystem-log")
    parser.add_argument("--require-source-rerank", action="store_true")
    parser.add_argument("--max-rerank-p99-ms", type=float, default=0.0)
    parser.add_argument("--max-client-p99-ms", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    traces = extract_traces(args.log)
    datasystem_completions, datasystem_duplicates = extract_datasystem_completions(
        args.datasystem_log)
    quota_stats = extract_quota_stats(args.log)
    requests = read_requests(args.requests_tsv)
    expected_ids = set(requests) if requests else set(traces)
    missing = sorted(expected_ids - set(traces))
    invalid = {}
    valid = []
    for request_id in sorted(expected_ids & set(traces)):
        traces[request_id]["_quota"] = quota_stats.get(request_id)
        traces[request_id]["_datasystem_final"] = datasystem_completions.get(request_id)
        reasons = validate(traces[request_id], args.require_datasystem_attribution,
                           args.require_source_rerank)
        if request_id in datasystem_duplicates:
            reasons.append("datasystem_completion_duplicate")
        if reasons:
            invalid[request_id] = reasons
        else:
            valid.append(traces[request_id])

    span_values = {}
    service_values = {}
    service_counts = {}
    for trace in valid:
        for span in trace["spans"]:
            if span.get("enabled"):
                span_values.setdefault(span["name"], []).append(int(span["duration_us"]))
                for name, value in (span.get("attributes") or {}).items():
                    if name.endswith("_us") and isinstance(value, (int, float)):
                        service_values.setdefault(f'{span["name"]}.{name}', []).append(int(value))
                    elif name.endswith("_count") and isinstance(value, (int, float)):
                        service_counts.setdefault(f'{span["name"]}.{name}', []).append(int(value))
    summary = {
        "classification": "PAIREC_BRPC_PIPELINE_TRACE_OK",
        "expected": args.expected or len(expected_ids),
        "trace_count": len(traces),
        "valid_count": len(valid),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "valid_ratio": round(len(valid) / max(len(expected_ids), 1), 6),
        "missing_request_ids": missing[:20],
        "invalid_examples": dict(list(invalid.items())[:20]),
        "client_e2e": metric([requests[trace["request_id"]] for trace in valid
                              if trace["request_id"] in requests]),
        "pairec_total": metric([int(trace["pairec_total_us"]) for trace in valid]),
        "spans": {name: metric(values) for name, values in sorted(span_values.items())},
        "service_phases": {
            name: metric(values) for name, values in sorted(service_values.items())},
        "service_counts": {
            name: count_metric(values) for name, values in sorted(service_counts.items())},
        "datasystem_completion_event_count": len(datasystem_completions),
        "datasystem_duplicate_count": len(datasystem_duplicates),
        "datasystem_complete_count": sum(bool(
            (trace.get("_datasystem_final") or trace.get("datasystem") or {}).get(
                "attribution_complete")) for trace in valid),
        "datasystem": {
            "get": metric([int(trace["_datasystem_final"]["get_us"])
                           for trace in valid if trace.get("_datasystem_final")]),
            "set": metric([int(trace["_datasystem_final"]["set_us"])
                           for trace in valid if trace.get("_datasystem_final")]),
            "get_count": sum(int(trace["_datasystem_final"]["get_count"])
                             for trace in valid if trace.get("_datasystem_final")),
            "set_count": sum(int(trace["_datasystem_final"]["set_count"])
                             for trace in valid if trace.get("_datasystem_final")),
        },
        "quota_multi_recall_count": sum(bool(trace.get("_quota")) for trace in valid),
    }
    if args.expected and len(expected_ids) != args.expected:
        summary["classification"] = "PAIREC_BRPC_PIPELINE_TRACE_FAILED"
    if missing or invalid or len(valid) < math.ceil(max(len(expected_ids), 1) * 0.999):
        summary["classification"] = "PAIREC_BRPC_PIPELINE_TRACE_FAILED"
    rerank_p99 = summary["spans"].get("rerank", {}).get("p99_ms", 0.0)
    if args.max_rerank_p99_ms and rerank_p99 > args.max_rerank_p99_ms:
        summary["classification"] = "PAIREC_BRPC_PIPELINE_TRACE_FAILED"
        summary["rerank_p99_gate"] = {
            "actual_ms": rerank_p99, "maximum_ms": args.max_rerank_p99_ms}
    client_p99 = summary["client_e2e"].get("p99_ms", 0.0)
    if args.max_client_p99_ms and client_p99 > args.max_client_p99_ms:
        summary["classification"] = "PAIREC_BRPC_PIPELINE_TRACE_FAILED"
        summary["client_p99_gate"] = {
            "actual_ms": client_p99, "maximum_ms": args.max_client_p99_ms}
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("metric count avg_ms p50_ms p95_ms p99_ms max_ms")
    for name, values in [("client_e2e", summary["client_e2e"]),
                         ("pairec_total", summary["pairec_total"])]:
        print(name, values["count"], values["avg_ms"], values["p50_ms"],
              values["p95_ms"], values["p99_ms"], values["max_ms"])
    for name, values in summary["spans"].items():
        print(name, values["count"], values["avg_ms"], values["p50_ms"],
              values["p95_ms"], values["p99_ms"], values["max_ms"])
    for name, values in summary["service_phases"].items():
        print(name, values["count"], values["avg_ms"], values["p50_ms"],
              values["p95_ms"], values["p99_ms"], values["max_ms"])
    if summary["service_counts"]:
        print("count_metric count avg p50 p95 p99 max")
        for name, values in summary["service_counts"].items():
            print(name, values["count"], values["avg"], values["p50"],
                  values["p95"], values["p99"], values["max"])
    print("valid={}/{} missing={} invalid={} datasystem_complete={}".format(
        len(valid), len(expected_ids), len(missing), len(invalid),
        summary["datasystem_complete_count"]))
    print(f"summary_json={args.output}")
    if summary["classification"].endswith("FAILED"):
        raise SystemExit(1)
    print("PAIREC_BRPC_PIPELINE_TRACE_OK")


if __name__ == "__main__":
    main()
