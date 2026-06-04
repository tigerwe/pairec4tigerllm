#!/usr/bin/env python3
"""Run TRT_NUM_SAMPLES latency/quality A/B tests.

The script can attach to an already running TRT service for one sample count,
or manage the service lifecycle when --server-cmd is provided.

Attached single-sample example:
  python scripts/benchmark_trt_runner_samples.py \
      --samples 4 --log-template /tmp/server_samples{sample}.log

Managed multi-sample example:
  python scripts/benchmark_trt_runner_samples.py \
      --samples 1,2,4,8 \
      --server-cmd 'python -m inference.trt_llm.server --model_path ...'

If an old service is running, stop it before invoking this script. Do not pass
pkill -f 'inference.trt_llm.server' as --stop-command: it can also match this
script's own --server-cmd argument and terminate the benchmark process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUNNER_TRACE_RE = re.compile(
    r"runner_calls=(?P<calls>\d+)\s+"
    r"runner_avg_ms=(?P<avg>[0-9.]+)\s+"
    r"runner_max_ms=(?P<max>[0-9.]+)"
)


@dataclass
class SampleReport:
    sample: int
    health_sample: Optional[int]
    health_ok: bool
    test_exit: int
    request_count: int
    ok_count: int
    fail_count: int
    full_topk: int
    min_items: int
    http_avg_ms: Optional[float]
    http_p50_ms: Optional[float]
    http_p95_ms: Optional[float]
    http_p99_ms: Optional[float]
    http_max_ms: Optional[float]
    runner_trace_count: int
    runner_calls_min: Optional[int]
    runner_calls_max: Optional[int]
    runner_avg_ms: Optional[float]
    runner_max_ms: Optional[float]
    json_report: str
    log_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TRT_NUM_SAMPLES quality/latency tradeoffs."
    )
    parser.add_argument("--samples", default="1,2,4,8",
                        help="comma-separated TRT_NUM_SAMPLES values")
    parser.add_argument("--url", default=os.environ.get("TRT_SERVER_URL", "http://localhost:18000"))
    parser.add_argument("--server-cmd", default="",
                        help="optional TRT server command; TRT_NUM_SAMPLES is injected per run")
    parser.add_argument("--stop-command", default="",
                        help="optional command run before each managed server start; "
                             "avoid pkill -f patterns that can match --server-cmd")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--log-template", default="/tmp/server_samples{sample}.log")
    parser.add_argument("--json-dir", default="/tmp")
    parser.add_argument("--summary-json", default="/tmp/trt_runner_samples_ab.json")
    parser.add_argument("--summary-csv", default="/tmp/trt_runner_samples_ab.csv")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--repeat-requests", type=int, default=0)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--keep-server", action="store_true",
                        help="leave managed server running after the last sample")
    return parser.parse_args()


def parse_samples(raw: str) -> List[int]:
    samples: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("--samples values must be >= 1")
        samples.append(value)
    if not samples:
        raise ValueError("--samples must not be empty")
    return samples


def get_json(url: str, timeout: float = 5.0) -> Tuple[int, Dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, json.loads(body)


def wait_health(url: str, expected_sample: int, timeout_s: float) -> Tuple[bool, Optional[int]]:
    deadline = time.time() + timeout_s
    latest_sample: Optional[int] = None
    while time.time() < deadline:
        try:
            status, body = get_json(f"{url.rstrip('/')}/health", timeout=5.0)
            latest_sample = body.get("trt_num_samples")
            if status == 200 and latest_sample == expected_sample:
                return True, latest_sample
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            pass
        time.sleep(2.0)
    return False, latest_sample


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


def run_command(command: str) -> int:
    if not command:
        return 0
    completed = subprocess.run(command, shell=True)
    return completed.returncode


def dangerous_stop_command(args: argparse.Namespace) -> Optional[str]:
    if not args.stop_command:
        return None
    normalized = " ".join(args.stop_command.split())
    if "pkill" not in normalized or "-f" not in normalized:
        return None
    server_tokens = [
        token for token in re.split(r"\s+", args.server_cmd)
        if token and len(token) >= 8 and not token.startswith("-")
    ]
    matched = [token for token in server_tokens if token in normalized]
    if matched:
        return matched[0]
    return None


def start_server(command: str, sample: int, log_path: str) -> Tuple[subprocess.Popen[str], Any]:
    env = os.environ.copy()
    env["TRT_NUM_SAMPLES"] = str(sample)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return process, handle


def stop_server(process: Optional[subprocess.Popen[str]], handle: Any) -> None:
    if process is None:
        return
    if process.poll() is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
    if handle:
        handle.close()


def run_kv_test(args: argparse.Namespace, sample: int, log_path: str, json_path: str) -> int:
    command = [
        sys.executable,
        "scripts/test_trt_cpp_kv_offload.py",
        "--url", args.url,
        "--log", log_path,
        "--requests", str(args.requests),
        "--warmup-requests", str(args.warmup_requests),
        "--repeat-requests", str(args.repeat_requests),
        "--history-len", str(args.history_len),
        "--concurrency", str(args.concurrency),
        "--topk", str(args.topk),
        "--timeout", str(args.timeout),
        "--json-output", json_path,
    ]
    print(f"\n== TRT_NUM_SAMPLES={sample} ==")
    completed = subprocess.run(command)
    return completed.returncode


def summarize_runner_trace(log_text: str) -> Dict[str, Optional[float]]:
    calls: List[int] = []
    avgs: List[float] = []
    maxes: List[float] = []
    for match in RUNNER_TRACE_RE.finditer(log_text):
        calls.append(int(match.group("calls")))
        avgs.append(float(match.group("avg")))
        maxes.append(float(match.group("max")))
    if not calls:
        return {
            "trace_count": 0,
            "calls_min": None,
            "calls_max": None,
            "avg_ms": None,
            "max_ms": None,
        }
    return {
        "trace_count": len(calls),
        "calls_min": min(calls),
        "calls_max": max(calls),
        "avg_ms": sum(avgs) / len(avgs),
        "max_ms": max(maxes),
    }


def load_test_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_sample_report(
    sample: int,
    health_sample: Optional[int],
    health_ok: bool,
    test_exit: int,
    log_path: str,
    log_offset: int,
    json_path: str,
    topk: int,
) -> SampleReport:
    report = load_test_report(json_path)
    results = report.get("results") or []
    pressure_results = [item for item in results if item.get("phase") == "pressure wave"]
    ok = [item for item in pressure_results if item.get("ok")]
    failed = [item for item in pressure_results if not item.get("ok")]
    item_counts = [int(item.get("item_count") or 0) for item in ok]
    full_topk = sum(count >= topk for count in item_counts)
    pressure_metrics = (report.get("phase_http_metrics") or {}).get("pressure wave") or {}
    runner = summarize_runner_trace(read_appended(log_path, log_offset))
    return SampleReport(
        sample=sample,
        health_sample=health_sample,
        health_ok=health_ok,
        test_exit=test_exit,
        request_count=len(pressure_results),
        ok_count=len(ok),
        fail_count=len(failed),
        full_topk=full_topk,
        min_items=min(item_counts) if item_counts else 0,
        http_avg_ms=pressure_metrics.get("avg"),
        http_p50_ms=pressure_metrics.get("p50"),
        http_p95_ms=pressure_metrics.get("p95"),
        http_p99_ms=pressure_metrics.get("p99"),
        http_max_ms=pressure_metrics.get("max"),
        runner_trace_count=int(runner["trace_count"] or 0),
        runner_calls_min=None if runner["calls_min"] is None else int(runner["calls_min"]),
        runner_calls_max=None if runner["calls_max"] is None else int(runner["calls_max"]),
        runner_avg_ms=runner["avg_ms"],
        runner_max_ms=runner["max_ms"],
        json_report=json_path,
        log_path=log_path,
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def print_summary(reports: Iterable[SampleReport]) -> None:
    print("\n== TRT_NUM_SAMPLES A/B summary ==")
    print(
        "sample health exit calls ok/topk http_p50 http_p99 "
        "runner_avg runner_max json"
    )
    for item in reports:
        calls = "-"
        if item.runner_calls_min is not None:
            calls = str(item.runner_calls_min)
            if item.runner_calls_max != item.runner_calls_min:
                calls = f"{item.runner_calls_min}-{item.runner_calls_max}"
        print(
            f"{item.sample:>6} {str(item.health_ok):>6} {item.test_exit:>4} {calls:>5} "
            f"{item.ok_count}/{item.full_topk:<5} "
            f"{fmt(item.http_p50_ms):>8} {fmt(item.http_p99_ms):>8} "
            f"{fmt(item.runner_avg_ms):>10} {fmt(item.runner_max_ms):>10} "
            f"{item.json_report}"
        )


def write_outputs(reports: List[SampleReport], summary_json: str, summary_csv: str) -> None:
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in reports], handle, ensure_ascii=False, indent=2)
    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(reports[0]).keys()))
        writer.writeheader()
        for report in reports:
            writer.writerow(asdict(report))
    print(f"\nWrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV:  {summary_csv}")


def run_sample(args: argparse.Namespace, sample: int, managed: bool) -> SampleReport:
    log_path = args.log_template.format(sample=sample)
    json_path = str(Path(args.json_dir) / f"trt_runner_samples_{sample}.json")
    process: Optional[subprocess.Popen[str]] = None
    handle: Any = None
    if managed:
        if args.stop_command:
            stop_exit = run_command(args.stop_command)
            if stop_exit != 0:
                print(f"WARN: --stop-command exited with {stop_exit}")
        process, handle = start_server(args.server_cmd, sample, log_path)
        log_offset = 0
    else:
        log_offset = file_offset(log_path)

    try:
        health_ok, health_sample = wait_health(args.url, sample, args.startup_timeout)
        if not health_ok:
            raise RuntimeError(
                f"/health did not report trt_num_samples={sample}; "
                f"latest={health_sample}"
            )
        test_exit = run_kv_test(args, sample, log_path, json_path)
        if not Path(json_path).exists():
            raise RuntimeError(f"test report was not written: {json_path}")
        return build_sample_report(
            sample=sample,
            health_sample=health_sample,
            health_ok=health_ok,
            test_exit=test_exit,
            log_path=log_path,
            log_offset=log_offset,
            json_path=json_path,
            topk=args.topk,
        )
    finally:
        if managed and not args.keep_server:
            stop_server(process, handle)


def main() -> int:
    args = parse_args()
    args.url = args.url.rstrip("/")
    samples = parse_samples(args.samples)
    managed = bool(args.server_cmd)
    if not managed and len(samples) != 1:
        print("FAIL: without --server-cmd, run one --samples value at a time.")
        return 2
    if managed and args.keep_server and len(samples) != 1:
        print("FAIL: --keep-server is only valid for a single managed sample.")
        return 2
    dangerous_token = dangerous_stop_command(args)
    if dangerous_token:
        print(
            "FAIL: --stop-command uses pkill -f with a pattern that also appears "
            f"in --server-cmd ({dangerous_token!r})."
        )
        print(
            "Run the cleanup command once before this script, then omit "
            "--stop-command."
        )
        return 2
    if args.requests < 1 or args.concurrency < 1:
        print("FAIL: --requests and --concurrency must be >= 1")
        return 2

    reports: List[SampleReport] = []
    for sample in samples:
        reports.append(run_sample(args, sample, managed))

    print_summary(reports)
    write_outputs(reports, args.summary_json, args.summary_csv)

    bad = [
        item for item in reports
        if not item.health_ok
        or item.test_exit not in (0, 4)
        or item.fail_count
        or item.full_topk < item.ok_count
        or item.runner_calls_min != item.sample
        or item.runner_calls_max != item.sample
    ]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
