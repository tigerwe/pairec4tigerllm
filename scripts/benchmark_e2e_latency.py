#!/usr/bin/env python3
"""Benchmark PaiRec end-to-end latency and correlate structured trace logs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


RECOMMEND_TRACE_FIELDS = [
    "total_ms",
    "user_feature_ms",
    "recall_ms",
    "filter_ms",
    "general_rank_ms",
    "feature_ms",
    "rank_ms",
    "pipeline_wait_ms",
    "merge_ms",
    "sort_ms",
]

RECALL_TRACE_FIELDS = [
    "cost",
    "cache_ms",
    "history_ms",
    "convert_ms",
    "http_ms",
    "items_ms",
    "tr_total_ms",
    "tr_prepare_ms",
    "tr_infer_ms",
    "tr_prompt_ms",
    "tr_runner_ms",
    "tr_parse_ms",
    "tr_pad_ms",
    "tr_map_ms",
    "tr_kv_lookup_ms",
    "tr_kv_write_ms",
    "tr_result_cache_lookup_ms",
    "tr_result_cache_ds_lookup_ms",
    "tr_result_cache_write_submit_ms",
    "http_overhead_ms",
]

TRT_TRACE_FIELDS = [
    "total_ms",
    "prepare_ms",
    "kv_lookup_ms",
    "result_cache_lookup_ms",
    "result_cache_ds_lookup_ms",
    "prompt_ms",
    "runner_ms",
    "runner_calls",
    "runner_avg_ms",
    "runner_max_ms",
    "parse_ms",
    "map_ms",
]

DATASYSTEM_METRIC_FIELDS = {
    "offload": ["create_ms", "d2h_ms", "set_ms", "total_ms"],
    "onboard": ["get_ms", "h2d_ms", "total_ms"],
}


@dataclass
class RequestResult:
    index: int
    uid: str
    ok: bool
    code: Any
    latency_ms: float
    request_id: str = ""
    item_count: int = 0
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PaiRec /api/recommend and summarize trace stages."
    )
    parser.add_argument("--url", default="http://localhost:18080/api/recommend")
    parser.add_argument("--uids", default="130,2184,7494",
                        help="comma-separated realtime or fallback user IDs")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--scene-id", default="home_feed")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--pairec-log", default="/tmp/pairec.log")
    parser.add_argument("--trt-log", default="/tmp/server_v4.log")
    parser.add_argument("--json-output", default="")
    return parser.parse_args()


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def request_one(index: int, uid: str, args: argparse.Namespace) -> RequestResult:
    started = time.perf_counter()
    try:
        body = post_json(
            args.url,
            {"uid": uid, "size": args.size, "scene_id": args.scene_id},
            args.timeout,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return RequestResult(
            index=index,
            uid=uid,
            ok=body.get("code") == 200,
            code=body.get("code"),
            latency_ms=latency_ms,
            request_id=str(body.get("request_id", "")),
            item_count=len(body.get("items") or []),
            error=str(body.get("msg", "")) if body.get("code") != 200 else "",
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return RequestResult(
            index=index,
            uid=uid,
            ok=False,
            code="ERR",
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )


def run_requests(
    count: int, uids: List[str], args: argparse.Namespace
) -> List[RequestResult]:
    jobs = [(index, uids[index % len(uids)]) for index in range(count)]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(request_one, index, uid, args) for index, uid in jobs
        ]
        return sorted((future.result() for future in futures), key=lambda item: item.index)


def file_offset(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_appended(path: str, offset: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            return handle.read()
    except OSError:
        return ""


def parse_key_values(line: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    normalized = line.replace("\t", " ")
    for part in normalized.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def parse_trace_logs(
    pairec_log: str, trt_log: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    recommend_traces: Dict[str, Dict[str, str]] = {}
    recall_traces: Dict[str, Dict[str, str]] = {}
    trt_traces: Dict[str, Dict[str, str]] = {}

    for line in pairec_log.splitlines():
        fields = parse_key_values(line)
        request_id = fields.get("requestId", "")
        if not request_id:
            continue
        if fields.get("module") == "RecommendTrace":
            recommend_traces[request_id] = fields
        elif fields.get("module") == "GenerativeRecall" and "tr_backend" in fields:
            recall_traces[request_id] = fields

    for line in trt_log.splitlines():
        if "[TRACE]" not in line:
            continue
        fields = parse_key_values(line)
        request_id = fields.get("request_id", "")
        if request_id:
            trt_traces[request_id] = fields

    return recommend_traces, recall_traces, trt_traces


def percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> Optional[Dict[str, float]]:
    numbers = list(values)
    if not numbers:
        return None
    return {
        "count": len(numbers),
        "avg": statistics.mean(numbers),
        "p50": percentile(numbers, 0.50),
        "p95": percentile(numbers, 0.95),
        "p99": percentile(numbers, 0.99),
        "p9999": percentile(numbers, 0.9999),
        "max": max(numbers),
    }


def numeric_fields(
    traces: Dict[str, Dict[str, str]],
    request_ids: Iterable[str],
    fields: List[str],
) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for field in fields:
        values: List[float] = []
        for request_id in request_ids:
            value = traces.get(request_id, {}).get(field)
            if value is None:
                continue
            try:
                values.append(float(value))
            except ValueError:
                pass
        summary = summarize(values)
        if summary:
            output[field] = summary
    return output


def datasystem_cpp_metrics(trt_log: str) -> Dict[str, Any]:
    events = [
        parse_key_values(line)
        for line in trt_log.splitlines()
        if "[Datasystem][TRACE]" in line
    ]
    report: Dict[str, Any] = {}
    for op, fields in DATASYSTEM_METRIC_FIELDS.items():
        op_events = [event for event in events if event.get("op") == op]
        metrics: Dict[str, Dict[str, float]] = {}
        for field in fields:
            values: List[float] = []
            for event in op_events:
                try:
                    values.append(float(str(event[field]).rstrip(",.")))
                except (KeyError, ValueError):
                    pass
            summary = summarize(values)
            if summary:
                metrics[field] = summary
        report[op] = {
            "count": len(op_events),
            "metrics": metrics,
        }
    return report


def print_metric_table(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print(f"\n== {title} ==")
    if not metrics:
        print("  (no correlated trace lines)")
        return
    print("  metric                              count      avg      p50      p95      p99    p9999      max")
    for name, values in metrics.items():
        print(
            f"  {name:<35} {int(values['count']):>5} "
            f"{values['avg']:>8.1f} {values['p50']:>8.1f} "
            f"{values['p95']:>8.1f} {values['p99']:>8.1f} "
            f"{values['p9999']:>8.1f} {values['max']:>8.1f}"
        )


def print_datasystem_cpp_table(report: Dict[str, Any]) -> None:
    print("\n== DataSystem C++ KV block stages ==")
    print("  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock")
    if not report or all(report[op]["count"] == 0 for op in ("offload", "onboard")):
        print("  (no DataSystem C++ TRACE lines in the benchmark log slice)")
        return
    print("  metric                              count      avg      p50      p95      p99    p9999      max")
    for op in ("offload", "onboard"):
        op_report = report[op]
        print(f"  {op}: events={op_report['count']}")
        for name, values in op_report["metrics"].items():
            metric_name = f"{op}.{name}"
            print(
                f"  {metric_name:<35} {int(values['count']):>5} "
                f"{values['avg']:>8.3f} {values['p50']:>8.3f} "
                f"{values['p95']:>8.3f} {values['p99']:>8.3f} "
                f"{values['p9999']:>8.3f} {values['max']:>8.3f}"
            )


def print_item_summary(results: List[RequestResult], expected_size: int) -> Dict[str, Any]:
    item_counts = [item.item_count for item in results if item.ok]
    full_size = sum(count >= expected_size for count in item_counts)
    summary = summarize(item_counts)
    print("\n== Response items ==")
    if not summary:
        print("  (no successful responses)")
        return {
            "expected_size": expected_size,
            "full_size": 0,
            "count": 0,
        }
    print(
        f"  full_size={full_size}/{len(item_counts)} expected={expected_size} "
        f"min={min(item_counts)} p50={summary['p50']:.1f} "
        f"p95={summary['p95']:.1f} max={int(summary['max'])}"
    )
    return {
        "expected_size": expected_size,
        "full_size": full_size,
        "count": len(item_counts),
        "metrics": summary,
    }


def main() -> int:
    args = parse_args()
    uids = [uid.strip() for uid in args.uids.split(",") if uid.strip()]
    if not uids or args.requests < 1 or args.concurrency < 1:
        raise SystemExit("--uids, --requests and --concurrency must be non-empty and positive")

    print("PaiRec end-to-end latency benchmark")
    print(f"  url={args.url}")
    print(f"  uids={','.join(uids)} requests={args.requests} warmup={args.warmup}")
    print(f"  concurrency={args.concurrency} size={args.size} scene={args.scene_id}")

    if args.warmup:
        print("\n== Warmup ==")
        warmup = run_requests(args.warmup, uids, args)
        print(f"  ok={sum(item.ok for item in warmup)}/{len(warmup)}")

    pairec_offset = file_offset(args.pairec_log)
    trt_offset = file_offset(args.trt_log)

    print("\n== Benchmark ==")
    started = time.perf_counter()
    results = run_requests(args.requests, uids, args)
    elapsed_s = time.perf_counter() - started
    ok = [item for item in results if item.ok]
    failed = [item for item in results if not item.ok]
    print(f"  ok={len(ok)} fail={len(failed)} elapsed={elapsed_s:.1f}s")
    for item in failed[:5]:
        print(f"  failed[{item.index}] uid={item.uid} code={item.code} error={item.error}")

    pairec_log = read_appended(args.pairec_log, pairec_offset)
    trt_log = read_appended(args.trt_log, trt_offset)
    recommend_traces, recall_traces, trt_traces = parse_trace_logs(pairec_log, trt_log)
    request_ids = [item.request_id for item in ok if item.request_id]

    client_summary = summarize(item.latency_ms for item in ok)
    client_metrics = {"client_e2e_ms": client_summary} if client_summary else {}
    recommend_metrics = numeric_fields(recommend_traces, request_ids, RECOMMEND_TRACE_FIELDS)
    recall_metrics = numeric_fields(recall_traces, request_ids, RECALL_TRACE_FIELDS)
    trt_metrics = numeric_fields(trt_traces, request_ids, TRT_TRACE_FIELDS)
    datasystem_metrics = datasystem_cpp_metrics(trt_log)

    print_metric_table("Client", client_metrics)
    item_metrics = print_item_summary(ok, args.size)
    print_metric_table("PaiRec recommend stages", recommend_metrics)
    print_metric_table("GenerativeRecall stages", recall_metrics)
    print_metric_table("TRT service stages", trt_metrics)
    if not trt_metrics and recall_metrics:
        print("\nINFO: TRT service log lines were not correlated by request_id.")
        print("  Use the GenerativeRecall tr_* fields above as the TRT response trace.")
    if not recommend_traces or not recall_traces:
        print("\nWARN: PaiRec structured trace lines were not found.")
        print("  Restart PaiRec with scripts/start_pairec.sh and capture stderr:")
        print("  CONFIG_PATH=./configs/pairec_config.kafka.json bash scripts/start_pairec.sh 2>&1 | tee /tmp/pairec.log")

    grouped: Dict[str, List[str]] = {}
    group_trace_source = "trt_log"
    if trt_traces:
        for request_id in request_ids:
            source = trt_traces.get(request_id, {}).get("cache", "unknown")
            grouped.setdefault(source, []).append(request_id)
        group_traces = trt_traces
        group_fields = ["total_ms", "runner_ms"]
        total_field = "total_ms"
    else:
        group_trace_source = "generative_recall"
        for request_id in request_ids:
            source = recall_traces.get(request_id, {}).get(
                "tr_result_cache_source", "unknown"
            )
            grouped.setdefault(source, []).append(request_id)
        group_traces = recall_traces
        group_fields = ["tr_total_ms", "tr_runner_ms"]
        total_field = "tr_total_ms"

    print(f"\n== TRT result cache groups ({group_trace_source}) ==")
    if not grouped:
        print("  (no correlated TRT trace lines)")
    for source, ids in sorted(grouped.items()):
        metrics = numeric_fields(group_traces, ids, group_fields)
        total = metrics.get(total_field, {})
        print(
            f"  {source:<12} count={len(ids):>4} "
            f"total_p50={total.get('p50', 0):>8.1f}ms "
            f"total_p95={total.get('p95', 0):>8.1f}ms "
            f"total_p99={total.get('p99', 0):>8.1f}ms "
            f"total_p9999={total.get('p9999', 0):>8.1f}ms "
            f"total_max={total.get('max', 0):>8.1f}ms"
        )

    print_datasystem_cpp_table(datasystem_metrics)

    report = {
        "config": vars(args),
        "elapsed_s": elapsed_s,
        "results": [asdict(item) for item in results],
        "client_metrics": client_metrics,
        "item_metrics": item_metrics,
        "recommend_metrics": recommend_metrics,
        "recall_metrics": recall_metrics,
        "trt_metrics": trt_metrics,
        "datasystem_cpp_metrics": datasystem_metrics,
        "cache_group_source": group_trace_source,
        "cache_groups": {source: len(ids) for source, ids in grouped.items()},
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\nWrote report: {args.json_output}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
