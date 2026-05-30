#!/usr/bin/env python3
"""Stress the TRT-LLM C++ KV cache offload/onboard path.

This script assumes the inference server is already running. It sends requests
that intentionally avoid the Python result-cache key while reusing prompt
prefixes, then checks the TRT-LLM log for C++ KV transfer events.

Typical use:
  python scripts/test_trt_cpp_kv_offload.py --log /tmp/server_v4.log
  python scripts/test_trt_cpp_kv_offload.py --requests 160 --concurrency 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


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
    "hbm_kv": r"Kvcache in HBM",
    "matched_full": r"Matched full block",
    "partial_reuse": r"Reused partially|Copied partially",
    "errors": r"ERROR|Traceback|SIGSEGV|segmentation fault|Exception",
}

TAIL_EVENT_RE = re.compile(
    r"copyBlock entered|OffLoad copy|OnBoard copy|Create Key|Set Key|Get Key|"
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
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress TRT-LLM C++ KV cache offload/onboard and parse logs."
    )
    parser.add_argument("--url", default=os.environ.get("TRT_SERVER_URL", "http://localhost:18000"))
    parser.add_argument("--log", default=os.environ.get("TRT_SERVER_LOG", "/tmp/server_v4.log"))
    parser.add_argument("--requests", type=int, default=120, help="pressure wave request count")
    parser.add_argument("--repeat-requests", type=int, default=32, help="same-history reuse/onboard wave count")
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--prefix", default=f"cppkv_{int(time.time())}")
    parser.add_argument("--strict-onboard", action="store_true",
                        help="return non-zero if Get/OnBoard events are not observed")
    parser.add_argument("--no-fail-on-missing-log", action="store_true",
                        help="only fail on HTTP errors when the log file is absent")
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


def latest_run_segment(log_text: str) -> str:
    start = max(log_text.rfind(marker) for marker in RUN_MARKERS)
    if start < 0:
        return log_text
    return log_text[start:]


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
    history: List[List[int]] = []
    for j in range(length):
        # Keep IDs inside the semantic-id codebook range and avoid all-zero rows.
        s0 = (seed * 17 + j * 13 + 7) % 256
        s1 = (seed * 29 + j * 19 + 11) % 256
        s2 = (seed * 37 + j * 23 + 3) % 256
        s3 = (seed * 43 + j * 31 + 5) % 256
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                request_one,
                phase,
                i,
                args.url,
                user_id,
                history,
                args.topk,
                args.timeout,
            )
            for i, (user_id, history) in enumerate(job_list, start=1)
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if done == 1 or done == len(futures) or done % 10 == 0:
                ok_count = sum(1 for item in results if item.ok)
                print(f"  progress {done}/{len(futures)} ok={ok_count} last={format_result(result)}")

    elapsed = time.perf_counter() - started
    print_summary(phase, results, elapsed)
    return sorted(results, key=lambda item: item.index)


def format_result(result: RequestResult) -> str:
    if result.ok:
        return (
            f"code={result.code} kv={result.kv_source or '-'} "
            f"backend={result.backend or '-'} ms={result.latency_ms:.0f}"
        )
    return f"FAIL status={result.status} code={result.code} err={result.error[:90]}"


def print_summary(phase: str, results: List[RequestResult], elapsed_s: float) -> None:
    ok = [item for item in results if item.ok]
    failed = [item for item in results if not item.ok]
    latencies = [item.latency_ms for item in ok]
    if latencies:
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        avg = statistics.mean(latencies)
        print(
            f"  {phase} summary: ok={len(ok)} fail={len(failed)} "
            f"avg={avg:.0f}ms p50={p50:.0f}ms p95={p95:.0f}ms elapsed={elapsed_s:.1f}s"
        )
    else:
        print(f"  {phase} summary: ok=0 fail={len(failed)} elapsed={elapsed_s:.1f}s")
    if failed:
        for item in failed[:5]:
            print(f"    failed[{item.index}]: {format_result(item)}")


def print_log_report(before: Dict[str, int], after: Dict[str, int], log_text: str) -> int:
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
    print(f"  new HBM reuse lines:           {delta['hbm_kv']}")
    print(f"  new matched/reused lines:      {delta['matched_full'] + delta['partial_reuse']}")
    print(f"  new error-like lines:          {delta['errors']}")

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
    if run_counts["reuse_enabled"] == 0:
        print("\nFAIL: KV cache block reuse was not confirmed in the log.")
        return 2
    if delta["copy_block"] == 0 or (delta["offload_copy"] == 0 and delta["set_key"] == 0):
        print("\nFAIL: pressure wave did not trigger C++ offload.")
        print("Hint: retry with --requests 180 --concurrency 8, or lower max_tokens_in_paged_kv_cache.")
        return 3

    onboard_seen = delta["get_key"] > 0 or delta["onboard_copy"] > 0 or delta["hbm_kv"] > 0
    if not onboard_seen:
        print("\nWARN: offload was observed, but onboard/reuse was not proven by this run.")
        return 4

    print("\nPASS: C++ KV offload was observed; onboard/reuse evidence is present.")
    return 0


def main() -> int:
    args = parse_args()
    args.url = args.url.rstrip("/")

    print("TRT-LLM C++ KV offload/onboard stress test")
    print(f"  url={args.url}")
    print(f"  log={args.log}")
    print(f"  requests={args.requests} repeat_requests={args.repeat_requests}")
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

    shared_history = make_history(seed=777, length=args.history_len)
    reuse_jobs = [
        (f"{args.prefix}_reuse_{i}", shared_history)
        for i in range(args.repeat_requests)
    ]

    pressure_histories = [
        make_history(seed=i + 1000, length=args.history_len)
        for i in range(args.requests)
    ]
    pressure_jobs = [
        (f"{args.prefix}_pressure_{i}", history)
        for i, history in enumerate(pressure_histories)
    ]

    replay_count = min(args.repeat_requests, len(pressure_histories))
    replay_jobs = [
        (f"{args.prefix}_replay_{i}", pressure_histories[i])
        for i in range(replay_count)
    ]

    all_results: List[RequestResult] = []
    all_results.extend(run_phase("same-prefix reuse wave", reuse_jobs, args))
    time.sleep(0.5)
    all_results.extend(run_phase("pressure wave", pressure_jobs, args))
    time.sleep(0.5)
    all_results.extend(run_phase("replay/onboard wave", replay_jobs, args))

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
    verdict = print_log_report(before_counts, after_counts, after_log)
    if verdict == 4 and not args.strict_onboard:
        return 0
    return verdict


if __name__ == "__main__":
    sys.exit(main())
