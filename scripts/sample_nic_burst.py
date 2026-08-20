#!/usr/bin/env python3
"""Collect and summarize high-frequency Linux interface byte counters."""

from __future__ import annotations

import argparse
import json
import pathlib
import time


def collect(interface: str, interval_ms: int, stop_file: pathlib.Path) -> None:
    base = pathlib.Path("/sys/class/net") / interface / "statistics"
    rx = base / "rx_bytes"
    tx = base / "tx_bytes"
    if not rx.is_file() or not tx.is_file():
        raise SystemExit(f"interface counters not found: {interface}")
    interval = interval_ms / 1000.0
    while not stop_file.exists():
        print(f"{time.time_ns()} {rx.read_text().strip()} {tx.read_text().strip()}", flush=True)
        time.sleep(interval)
    print(f"{time.time_ns()} {rx.read_text().strip()} {tx.read_text().strip()}", flush=True)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] + (values[high] - values[low]) * fraction


def summarize(path: pathlib.Path, link_bps: float) -> dict:
    samples = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            samples.append(tuple(int(value) for value in fields))
        except ValueError:
            continue
    if len(samples) < 2:
        raise ValueError(
            f"need at least 2 valid NIC samples, found {len(samples)} in {path}"
        )
    rx_bps = []
    tx_bps = []
    for before, after in zip(samples, samples[1:]):
        elapsed = (after[0] - before[0]) / 1e9
        if elapsed <= 0:
            continue
        rx_bps.append(max(0.0, (after[1] - before[1]) * 8 / elapsed))
        tx_bps.append(max(0.0, (after[2] - before[2]) * 8 / elapsed))
    peak_rx = max(rx_bps, default=0.0)
    peak_tx = max(tx_bps, default=0.0)
    return {
        "sample_count": len(samples),
        "interval_count": len(rx_bps),
        "window_ms": (samples[-1][0] - samples[0][0]) / 1e6 if len(samples) >= 2 else 0.0,
        "peak_rx_gbps": peak_rx / 1e9,
        "peak_tx_gbps": peak_tx / 1e9,
        "p95_rx_gbps": percentile(rx_bps, 0.95) / 1e9,
        "p95_tx_gbps": percentile(tx_bps, 0.95) / 1e9,
        "peak_rx_link_pct": peak_rx / link_bps * 100 if link_bps else 0.0,
        "peak_tx_link_pct": peak_tx / link_bps * 100 if link_bps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--interface", required=True)
    collect_parser.add_argument("--interval-ms", type=int, default=20)
    collect_parser.add_argument("--stop-file", type=pathlib.Path, required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--input", type=pathlib.Path, required=True)
    summary_parser.add_argument("--output", type=pathlib.Path, required=True)
    summary_parser.add_argument("--link-bps", type=float, default=25e9)
    args = parser.parse_args()
    if args.command == "collect":
        if not 1 <= args.interval_ms <= 1000:
            raise SystemExit("interval-ms must be between 1 and 1000")
        collect(args.interface, args.interval_ms, args.stop_file)
        return
    result = summarize(args.input, args.link_bps)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"NIC_BURST_SUMMARY_OK output={args.output}")


if __name__ == "__main__":
    main()
