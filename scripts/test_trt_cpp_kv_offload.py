#!/usr/bin/env python3
"""Stress the TRT-LLM C++ KV cache offload/onboard path.

This script assumes the inference server is already running. It sends requests
that intentionally avoid the Python result-cache key while reusing prompt
prefixes, then checks the TRT-LLM log for C++ KV transfer events.

Typical use:
  python scripts/test_trt_cpp_kv_offload.py --log /tmp/server_v4.log
  python scripts/test_trt_cpp_kv_offload.py --requests 160 --concurrency 1
  python scripts/test_trt_cpp_kv_offload.py --requests 360 --repeat-requests 128 \
      --replay-source-count 4 --replay-tail-offset 1 --strict-onboard
  python scripts/test_trt_cpp_kv_offload.py --requests 360 --repeat-requests 10000 \
      --replay-source-count 4 --replay-tail-offset 1 --min-onboard-samples 10000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


EVENT_PATTERNS: Dict[str, str] = {
    "scheduler_max": r"Capacity Scheduler Policy:\s*MAX_UTILIZATION",
    "scheduler_no_evict": r"Capacity Scheduler Policy:\s*GUARANTEED_NO_EVICT",
    "reuse_enabled": r"KV cache block reuse is enabled",
    "reuse_disabled": r"KV cache reuse disabled",
    "primary_secondary": r"\.primaryBlocks=(\d+),\s*\.secondayBlocks=(\d+)",
    "copy_block": r"copyBlock entered",
    "offload_copy": r"OffLoad copy",
    "onboard_copy": r"OnBoard copy",
    "create_key": r"Create Key",
    "set_key": r"Set Key",
    "get_key": r"Get Key",
    "datasystem_trace": r"\[Datasystem\]\[TRACE\]",
    "datasystem_offload_trace": r"\[Datasystem\]\[TRACE\]\s+op=offload\b",
    "datasystem_onboard_trace": r"\[Datasystem\]\[TRACE\]\s+op=onboard\b",
    "hbm_kv": r"Kvcache in HBM",
    "matched_full": r"Matched full block",
    "partial_reuse": r"Reused partially|Copied partially",
    "errors": r"ERROR|Traceback|SIGSEGV|segmentation fault|Exception",
}

TAIL_EVENT_RE = re.compile(
    r"copyBlock entered|OffLoad copy|OnBoard copy|Create Key|Set Key|Get Key|"
    r"\[Datasystem\]\[TRACE\]|"
    r"Kvcache in HBM|Matched full block|Reused partially|Copied partially|"
    r"KV cache block reuse is enabled|Capacity Scheduler Policy|"
    r"KV cache reuse disabled|ERROR|Traceback|SIGSEGV",
    re.IGNORECASE,
)

RUN_MARKERS = (
    "[TRTQwen3Backend] Scheduler policy",
    "[TensorRT-LLM] TensorRT LLM version",
    "Starting TensorRT LLM init",
)

DATASYSTEM_METRIC_FIELDS = {
    "offload": ["create_ms", "d2h_ms", "set_ms", "total_ms"],
    "onboard": ["get_ms", "h2d_ms", "total_ms"],
}

DATASYSTEM_METRIC_SCOPE = {
    "offload": {
        "create_ms": "host DataSystem Create API wall-clock",
        "d2h_ms": "host-timed blocking GPU->CPU transfer wall-clock",
        "set_ms": "host DataSystem Set API wall-clock",
        "total_ms": "host offload end-to-end wall-clock",
    },
    "onboard": {
        "get_ms": "host DataSystem Get API wall-clock",
        "h2d_ms": "host-timed blocking CPU->GPU transfer wall-clock",
        "total_ms": "host onboard end-to-end wall-clock",
    },
}

P9999_MIN_SAMPLE_COUNT = 10000
P9999_RECOMMENDED_SAMPLE_COUNT = 100000


@dataclass
class RequestResult:
    phase: str
    index: int
    ok: bool
    code: Any
    status: int
    latency_ms: float
    kv_source: str
    backend: str
    item_count: int
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Stress TRT-LLM C++ KV cache offload/onboard and parse logs."
    )
    parser.add_argument("--url", default=os.environ.get("TRT_SERVER_URL", "http://localhost:18000"))
    parser.add_argument("--log", default=os.environ.get("TRT_SERVER_LOG", "/tmp/server_v4.log"))
    parser.add_argument("--requests", "--request", type=int, default=120,
                        help="pressure wave request count")
    parser.add_argument("--repeat-requests", type=int, default=32,
                        help="replay/onboard wave request count")
    parser.add_argument("--replay-source-count", type=int, default=4,
                        help="number of recent pressure histories cycled during replay")
    parser.add_argument("--replay-tail-offset", type=int, default=0,
                        help="skip this many newest pressure histories when selecting replay sources")
    parser.add_argument("--warmup-requests", type=int, default=4,
                        help="small same-prefix warmup count; 0 disables warmup")
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="client-side HTTP concurrency; keep 1 unless runner.generate is known thread-safe")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--prefix", default=f"cppkv_{int(time.time())}")
    parser.add_argument("--max-phase-failures", type=int, default=5,
                        help="abort a phase after this many HTTP failures; 0 disables fail-fast")
    parser.add_argument("--strict-onboard", action="store_true",
                        help="return non-zero if Get/OnBoard events are not observed")
    parser.add_argument("--min-onboard-samples", type=int, default=0,
                        help="require at least this many structured onboard traces")
    parser.add_argument("--no-fail-on-missing-log", action="store_true",
                        help="only fail on HTTP errors when the log file is absent")
    parser.add_argument("--json-output", default="",
                        help="optional path for the machine-readable latency report")
    return parser.parse_args()


def read_log(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def count_events(log_text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, pattern in EVENT_PATTERNS.items():
        counts[name] = len(re.findall(pattern, log_text, flags=re.IGNORECASE))
    return counts


def latest_event_lines(log_text: str, limit: int = 40) -> List[str]:
    lines = [line for line in log_text.splitlines() if TAIL_EVENT_RE.search(line)]
    return lines[-limit:]


def latest_pattern_lines(log_text: str, pattern: str, limit: int = 20) -> List[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    lines = [line for line in log_text.splitlines() if regex.search(line)]
    return lines[-limit:]


def latest_run_segment(log_text: str) -> str:
    start = max(log_text.rfind(marker) for marker in RUN_MARKERS)
    if start < 0:
        return log_text
    return log_text[start:]


def appended_log(previous: str, current: str) -> str:
    if current.startswith(previous):
        return current[len(previous):]
    return current


def parse_key_values(line: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for part in line.replace("\t", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value.rstrip(".,")
    return values


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


def datasystem_latency_metrics(log_text: str) -> Dict[str, Any]:
    events = [
        parse_key_values(line)
        for line in log_text.splitlines()
        if "[Datasystem][TRACE]" in line
    ]
    output: Dict[str, Any] = {}
    for op, metric_fields in DATASYSTEM_METRIC_FIELDS.items():
        op_events = [event for event in events if event.get("op") == op]
        metrics: Dict[str, Dict[str, float]] = {}
        for field in metric_fields:
            values: List[float] = []
            for event in op_events:
                try:
                    values.append(float(event[field]))
                except (KeyError, ValueError):
                    pass
            summary = summarize(values)
            if summary:
                metrics[field] = summary
        output[op] = {"count": len(op_events), "metrics": metrics}
    return output


def print_latency_summary(name: str, values: Dict[str, float], indent: str = "  ") -> None:
    print(
        f"{indent}{name:<14} count={int(values['count']):>5} "
        f"avg={values['avg']:>9.3f}ms p50={values['p50']:>9.3f}ms "
        f"p95={values['p95']:>9.3f}ms p99={values['p99']:>9.3f}ms "
        f"p9999={values['p9999']:>9.3f}ms max={values['max']:>9.3f}ms"
    )


def print_datasystem_latency_report(phase_logs: Dict[str, str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    print("\n== DataSystem metric scope ==")
    print("  Clock source: host std::chrono::steady_clock")
    print("  d2h_ms/h2d_ms: host wall-clock around blocking cudaMemcpySanitized/cudaMemcpy")
    print("  Device-only CUDA event or kernel time: not collected")
    for op in ("offload", "onboard"):
        print(f"  {op}:")
        for name, scope in DATASYSTEM_METRIC_SCOPE[op].items():
            print(f"    {name:<10} {scope}")
    print("\n== DataSystem C++ latency by phase ==")
    all_phases = {**phase_logs, "all measured phases": "".join(phase_logs.values())}
    for phase, log_text in all_phases.items():
        metrics = datasystem_latency_metrics(log_text)
        report[phase] = metrics
        print(f"\n  [{phase}]")
        for op in ("offload", "onboard"):
            op_metrics = metrics[op]
            print(f"    {op}: count={op_metrics['count']}")
            for name, values in op_metrics["metrics"].items():
                print_latency_summary(name, values, indent="      ")
    overall = report["all measured phases"]
    print("\n== Percentile sample guidance ==")
    for op in ("offload", "onboard"):
        count = overall[op]["count"]
        if count < P9999_MIN_SAMPLE_COUNT:
            readiness = "insufficient"
        elif count < P9999_RECOMMENDED_SAMPLE_COUNT:
            readiness = "minimum tail observation only"
        else:
            readiness = "better supported"
        print(
            f"  {op}: count={count}, p9999={readiness}; "
            f"minimum={P9999_MIN_SAMPLE_COUNT}, recommended>={P9999_RECOMMENDED_SAMPLE_COUNT}"
        )
    return report


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, json.loads(body)


def get_json(url: str, timeout: float) -> Tuple[int, Dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, json.loads(body)


def make_history(seed: int, length: int) -> List[List[int]]:
    # Multiplication by an odd number is reversible modulo 2^32. Unlike the
    # previous seed % 256 pattern, this keeps 100k+ generated histories distinct.
    mixed_seed = (seed * 2654435761) & 0xFFFFFFFF
    seed_bytes = [(mixed_seed >> shift) & 0xFF for shift in (0, 8, 16, 24)]
    history: List[List[int]] = []
    for j in range(length):
        # Keep IDs inside the semantic-id codebook range and avoid all-zero rows.
        s0 = (seed_bytes[0] + j * 13 + 7) % 256
        s1 = (seed_bytes[1] + j * 19 + 11) % 256
        s2 = (seed_bytes[2] + j * 23 + 3) % 256
        s3 = (seed_bytes[3] + j * 31 + 5) % 256
        history.append([s0, s1, s2, s3])
    return history


def request_one(
    phase: str,
    index: int,
    url: str,
    user_id: str,
    history: List[List[int]],
    topk: int,
    timeout: float,
) -> RequestResult:
    payload = {"user_id": user_id, "history": history, "topk": topk}
    started = time.perf_counter()
    try:
        status, body = post_json(f"{url.rstrip('/')}/recommend", payload, timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        trace = body.get("trace") or {}
        return RequestResult(
            phase=phase,
            index=index,
            ok=status == 200 and body.get("code") == 200,
            code=body.get("code"),
            status=status,
            latency_ms=latency_ms,
            kv_source=str(trace.get("kv_source", "")),
            backend=str(trace.get("backend", "")),
            item_count=len(body.get("recommendations") or []),
            error=str(body.get("error", "")),
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return RequestResult(
            phase=phase,
            index=index,
            ok=False,
            code="ERR",
            status=0,
            latency_ms=latency_ms,
            kv_source="",
            backend="",
            item_count=0,
            error=str(exc),
        )


def run_phase(
    phase: str,
    jobs: Iterable[Tuple[str, List[List[int]]]],
    args: argparse.Namespace,
) -> List[RequestResult]:
    job_list = list(jobs)
    results: List[RequestResult] = []
    print(f"\n== {phase}: {len(job_list)} requests, concurrency={args.concurrency} ==")
    started = time.perf_counter()
    fail_count = 0
    submitted = 0
    aborted = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        job_iter = iter(enumerate(job_list, start=1))
        pending: Dict[concurrent.futures.Future[RequestResult], int] = {}

        def submit_next() -> bool:
            nonlocal submitted
            try:
                i, (user_id, history) = next(job_iter)
            except StopIteration:
                return False
            future = executor.submit(
                request_one,
                phase,
                i,
                args.url,
                user_id,
                history,
                args.topk,
                args.timeout,
            )
            pending[future] = i
            submitted += 1
            return True

        for _ in range(min(args.concurrency, len(job_list))):
            submit_next()

        while pending:
            done_set, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done_set:
                pending.pop(future, None)
                result = future.result()
                results.append(result)
                if not result.ok:
                    fail_count += 1
                done = len(results)
                if done == 1 or done == len(job_list) or done % 10 == 0 or not result.ok:
                    ok_count = sum(1 for item in results if item.ok)
                    print(
                        f"  progress {done}/{len(job_list)} submitted={submitted} "
                        f"ok={ok_count} fail={fail_count} last={format_result(result)}"
                    )

            if args.max_phase_failures > 0 and fail_count >= args.max_phase_failures:
                aborted = True
                for future in pending:
                    future.cancel()
                break

            while len(pending) < args.concurrency and submitted < len(job_list):
                if not submit_next():
                    break

    elapsed = time.perf_counter() - started
    if aborted:
        print(
            f"  aborting {phase}: {fail_count} failures reached "
            f"--max-phase-failures={args.max_phase_failures}"
        )
    print_summary(phase, results, elapsed, args.topk)
    return sorted(results, key=lambda item: item.index)


def format_result(result: RequestResult) -> str:
    if result.ok:
        return (
            f"code={result.code} kv={result.kv_source or '-'} "
            f"backend={result.backend or '-'} items={result.item_count} "
            f"ms={result.latency_ms:.0f}"
        )
    return f"FAIL status={result.status} code={result.code} err={result.error[:90]}"


def print_summary(phase: str, results: List[RequestResult], elapsed_s: float, topk: int) -> None:
    ok = [item for item in results if item.ok]
    failed = [item for item in results if not item.ok]
    latencies = [item.latency_ms for item in ok]
    if latencies:
        summary = summarize(latencies)
        assert summary is not None
        print(
            f"  {phase} summary: ok={len(ok)} fail={len(failed)} "
            f"avg={summary['avg']:.0f}ms p50={summary['p50']:.0f}ms "
            f"p95={summary['p95']:.0f}ms p99={summary['p99']:.0f}ms "
            f"p9999={summary['p9999']:.0f}ms max={summary['max']:.0f}ms "
            f"elapsed={elapsed_s:.1f}s"
        )
        item_counts = [item.item_count for item in ok]
        full_topk = sum(item.item_count >= topk for item in ok)
        item_summary = summarize(item_counts)
        assert item_summary is not None
        print(
            f"  {phase} items: full_topk={full_topk}/{len(ok)} "
            f"min={min(item_counts)} "
            f"p50={item_summary['p50']:.1f} p95={item_summary['p95']:.1f} "
            f"max={int(item_summary['max'])}"
        )
    else:
        print(f"  {phase} summary: ok=0 fail={len(failed)} elapsed={elapsed_s:.1f}s")
    if failed:
        for item in failed[:5]:
            print(f"    failed[{item.index}]: {format_result(item)}")


def print_log_report(
    before: Dict[str, int],
    after: Dict[str, int],
    log_text: str,
    min_onboard_samples: int = 0,
) -> int:
    delta = {name: after.get(name, 0) - before.get(name, 0) for name in after}
    run_log = latest_run_segment(log_text)
    run_counts = count_events(run_log)
    pool_matches = re.findall(EVENT_PATTERNS["primary_secondary"], run_log, flags=re.IGNORECASE)
    latest_pool = pool_matches[-1] if pool_matches else None
    scheduler_matches = re.findall(
        r"Capacity Scheduler Policy:\s*([A-Z_]+)",
        run_log,
        flags=re.IGNORECASE,
    )
    latest_scheduler = scheduler_matches[-1].upper() if scheduler_matches else "UNKNOWN"

    print("\n== Log verdict ==")
    if latest_pool:
        print(f"  latest pool: primaryBlocks={latest_pool[0]} secondaryBlocks={latest_pool[1]}")
    else:
        print("  latest pool: not found")
    print(f"  latest scheduler policy:       {latest_scheduler}")
    print(f"  block reuse enabled seen:      {run_counts['reuse_enabled'] > 0}")
    print(f"  reuse disabled warning seen:   {run_counts['reuse_disabled'] > 0}")
    print(f"  new copyBlock entered:         {delta['copy_block']}")
    print(f"  new OffLoad copy:              {delta['offload_copy']}")
    print(f"  new Create/Set Key:            {delta['create_key']}/{delta['set_key']}")
    print(f"  new Get/OnBoard Key:           {delta['get_key']}/{delta['onboard_copy']}")
    print(f"  new DataSystem trace lines:    {delta['datasystem_trace']}")
    print(f"  new trace offload/onboard:     {delta['datasystem_offload_trace']}/{delta['datasystem_onboard_trace']}")
    print(f"  new HBM reuse lines:           {delta['hbm_kv']}")
    print(f"  new matched/reused lines:      {delta['matched_full'] + delta['partial_reuse']}")
    print(f"  new error-like lines:          {delta['errors']}")

    print("\n== Recent DataSystem onboard lines ==")
    onboard_lines = latest_pattern_lines(run_log, r"Get Key|Get KvCache|OnBoard copy|\[Datasystem\]\[TRACE\]\s+op=onboard")
    if onboard_lines:
        for line in onboard_lines:
            print(f"  {line}")
    else:
        print("  (no Get/OnBoard lines)")

    print("\n== Recent matching log lines ==")
    recent = latest_event_lines(log_text)
    if recent:
        for line in recent:
            print(f"  {line}")
    else:
        print("  (no matching lines)")

    if run_counts["reuse_disabled"] > 0:
        print("\nFAIL: TensorRT-LLM disabled KV block reuse.")
        return 2
    if latest_scheduler != "MAX_UTILIZATION":
        print("\nFAIL: MAX_UTILIZATION scheduler was not confirmed in the log.")
        return 2
    structured_offload_seen = delta["datasystem_offload_trace"] > 0
    structured_onboard_seen = delta["datasystem_onboard_trace"] > 0
    if run_counts["reuse_enabled"] == 0 and not structured_offload_seen:
        print("\nFAIL: KV cache block reuse was not confirmed in the log.")
        return 2
    if run_counts["reuse_enabled"] == 0:
        print("\nINFO: DEBUG reuse-enabled log is absent; structured offload trace proves the transfer path ran.")
    if delta["copy_block"] == 0 and delta["offload_copy"] == 0 and delta["set_key"] == 0 and not structured_offload_seen:
        print("\nFAIL: pressure wave did not trigger C++ offload.")
        print("Hint: retry with --requests 180 --history-len 10 --concurrency 1, "
              "or lower max_tokens_in_paged_kv_cache.")
        return 3

    onboard_seen = delta["get_key"] > 0 or delta["onboard_copy"] > 0 or structured_onboard_seen
    structured_onboard_count = delta["datasystem_onboard_trace"]
    if structured_onboard_count < min_onboard_samples:
        print(
            f"\nFAIL: structured onboard samples {structured_onboard_count} "
            f"< --min-onboard-samples={min_onboard_samples}."
        )
        return 5
    if not onboard_seen:
        print("\nWARN: C++ KV offload was observed, but DataSystem onboard was not proven.")
        print("Observed HBM reuse/copy lines do not prove kvClient Get/onBoardCopy.")
        return 4

    print("\nPASS: C++ KV offload and DataSystem onboard were both observed.")
    return 0


def main() -> int:
    args = parse_args()
    args.url = args.url.rstrip("/")
    if args.concurrency < 1:
        print("FAIL: --concurrency must be >= 1")
        return 2
    if args.replay_source_count < 1:
        print("FAIL: --replay-source-count must be >= 1")
        return 2
    if args.replay_tail_offset < 0:
        print("FAIL: --replay-tail-offset must be >= 0")
        return 2
    if args.min_onboard_samples < 0:
        print("FAIL: --min-onboard-samples must be >= 0")
        return 2

    print("TRT-LLM C++ KV offload/onboard stress test")
    print(f"  url={args.url}")
    print(f"  log={args.log}")
    print(f"  requests={args.requests} repeat_requests={args.repeat_requests}")
    print(f"  replay_source_count={args.replay_source_count} replay_tail_offset={args.replay_tail_offset}")
    print(f"  min_onboard_samples={args.min_onboard_samples}")
    print(f"  history_len={args.history_len} concurrency={args.concurrency} topk={args.topk}")

    try:
        status, health = get_json(f"{args.url}/health", timeout=5.0)
        print(f"\n== Health ==\n  status={status} body={json.dumps(health, ensure_ascii=False)}")
    except Exception as exc:
        print(f"\nFAIL: cannot reach {args.url}/health: {exc}")
        return 1

    before_log = read_log(args.log)
    if not before_log:
        msg = f"WARN: log file is empty or missing: {args.log}"
        print(f"\n{msg}")
        if not args.no_fail_on_missing_log:
            print("Use --no-fail-on-missing-log to run HTTP-only smoke testing.")
    before_counts = count_events(before_log)

    warmup_history = make_history(seed=777, length=args.history_len)
    warmup_jobs = [
        (f"{args.prefix}_warmup_{i}", warmup_history)
        for i in range(args.warmup_requests)
    ]

    pressure_histories = [
        make_history(seed=i + 1000, length=args.history_len)
        for i in range(args.requests)
    ]
    pressure_jobs = [
        (f"{args.prefix}_pressure_{i}", history)
        for i, history in enumerate(pressure_histories)
    ]

    replay_end = max(0, len(pressure_histories) - args.replay_tail_offset)
    replay_start = max(0, replay_end - args.replay_source_count)
    replay_sources = pressure_histories[replay_start:replay_end]
    replay_jobs = [
        (f"{args.prefix}_replay_{i}", replay_sources[i % len(replay_sources)])
        for i in range(args.repeat_requests)
    ] if replay_sources else []
    if replay_sources:
        print(
            f"\nReplay sources: pressure indexes [{replay_start}, {replay_end}), "
            f"cycled across {len(replay_jobs)} requests"
        )
    elif args.repeat_requests:
        print("\nWARN: replay wave is empty; reduce --replay-tail-offset or increase --requests")

    all_results: List[RequestResult] = []
    phase_results: Dict[str, List[RequestResult]] = {}
    phase_logs: Dict[str, str] = {}
    previous_log = before_log

    def run_traced_phase(phase: str, jobs: Iterable[Tuple[str, List[List[int]]]]) -> None:
        nonlocal previous_log
        results = run_phase(phase, jobs, args)
        phase_results[phase] = results
        all_results.extend(results)
        current_log = read_log(args.log)
        phase_logs[phase] = appended_log(previous_log, current_log)
        previous_log = current_log

    if warmup_jobs:
        run_traced_phase("same-prefix warmup wave", warmup_jobs)
        time.sleep(0.5)
    run_traced_phase("pressure wave", pressure_jobs)
    time.sleep(0.5)
    run_traced_phase("replay/onboard wave", replay_jobs)

    failed = [item for item in all_results if not item.ok]
    if failed:
        print(f"\nFAIL: {len(failed)} HTTP requests failed.")
        return 1

    after_log = read_log(args.log)
    if not after_log:
        if args.no_fail_on_missing_log:
            print("\nHTTP-only PASS: all requests succeeded, but no log verdict was available.")
            return 0
        print("\nFAIL: cannot parse C++ evidence without the server log.")
        return 2

    after_counts = count_events(after_log)
    datasystem_report = print_datasystem_latency_report(phase_logs)
    verdict = print_log_report(before_counts, after_counts, after_log, args.min_onboard_samples)
    if args.json_output:
        report = {
            "config": vars(args),
            "results": [asdict(item) for item in all_results],
            "phase_http_metrics": {
                phase: summarize(item.latency_ms for item in results if item.ok)
                for phase, results in phase_results.items()
            },
            "datasystem_metrics": datasystem_report,
            "datasystem_metric_scope": DATASYSTEM_METRIC_SCOPE,
            "event_delta": {
                name: after_counts.get(name, 0) - before_counts.get(name, 0)
                for name in after_counts
            },
            "verdict": verdict,
        }
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\nWrote report: {args.json_output}")
    if verdict == 4 and not args.strict_onboard:
        return 0
    return verdict


if __name__ == "__main__":
    sys.exit(main())
