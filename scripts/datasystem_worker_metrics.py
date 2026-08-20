#!/usr/bin/env python3
"""Parse and diff DataSystem worker host metric snapshots.

The companion shell script collect_datasystem_worker_metrics.sh collects a raw
sectioned snapshot from the DataSystem worker pod (hostNetwork, so /proc and
/proc/net reflect the node). This module turns the raw text into a JSON
snapshot and computes per-second deltas between two snapshots. Kept as a
standalone module so the parsing and delta math are unit-testable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

CLK_TCK_DEFAULT = 100


def split_sections(raw: str) -> dict:
    sections: dict[str, list[str]] = {}
    current = None
    for line in raw.splitlines():
        if line.startswith("@"):
            current = line[1:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def parse_key_values(lines: list[str]) -> dict:
    values = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            values[parts[0]] = int(parts[1])
    return values


def parse_psi(lines: list[str]) -> dict:
    result = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        entry = {}
        for token in parts[1:]:
            key, _, value = token.partition("=")
            entry[key] = float(value)
        result[parts[0]] = entry
    return result


def parse_netdev(lines: list[str]) -> dict:
    interfaces = {}
    for line in lines:
        if ":" not in line:
            continue
        name, _, rest = line.partition(":")
        name = name.strip()
        fields = rest.split()
        if len(fields) < 9:
            continue
        interfaces[name] = {"rx_bytes": int(fields[0]), "tx_bytes": int(fields[8])}
    return interfaces


def parse_softirq(lines: list[str]) -> dict:
    totals = {"net_rx": 0, "net_tx": 0}
    for line in lines:
        name, _, rest = line.partition(":")
        name = name.strip()
        if name not in ("NET_RX", "NET_TX"):
            continue
        total = sum(int(token) for token in rest.split())
        totals["net_rx" if name == "NET_RX" else "net_tx"] = total
    return totals


def parse_proc_stat(lines: list[str]) -> dict:
    if not lines:
        return {}
    line = lines[0]
    close = line.rfind(")")
    if close < 0:
        return {}
    fields = line[close + 1 :].split()
    # fields[0] is state (field 3); utime/stime are fields 14/15.
    if len(fields) < 13:
        return {}
    return {"utime_ticks": int(fields[11]), "stime_ticks": int(fields[12])}


def parse_status_ctxt(lines: list[str]) -> dict:
    ctxt = {"voluntary_ctxt_switches": 0, "nonvoluntary_ctxt_switches": 0}
    for line in lines:
        name, _, rest = line.partition(":")
        name = name.strip()
        if name in ctxt:
            ctxt[name] = int(rest.split()[0])
    return ctxt


def parse_snapshot(raw: str, pod: str, node: str, ts_ns: int) -> dict:
    sections = split_sections(raw)
    cpu_stat = parse_key_values(sections.get("cpu_stat", []))
    cpuacct = parse_key_values(sections.get("cpuacct", []))
    loadavg = sections.get("loadavg", [""])[0].split()
    memstat = parse_key_values(sections.get("memstat", []))
    memcurrent_lines = sections.get("memcurrent", [])
    return {
        "pod": pod,
        "node": node,
        "ts_ns": ts_ns,
        "cpu_stat": cpu_stat,
        "cpuacct_usage_ns": cpuacct.get("usage"),
        "loadavg": {
            "load1": float(loadavg[0]) if len(loadavg) > 0 else None,
            "load5": float(loadavg[1]) if len(loadavg) > 1 else None,
            "load15": float(loadavg[2]) if len(loadavg) > 2 else None,
        },
        "ctxt_switches": parse_status_ctxt(sections.get("ctxt", [])),
        "proc_stat": parse_proc_stat(sections.get("procstat", [])),
        "netdev": parse_netdev(sections.get("netdev", [])),
        "softirq": parse_softirq(sections.get("softirq", [])),
        "psi_cpu": parse_psi(sections.get("psi_cpu", [])),
        "psi_memory": parse_psi(sections.get("psi_memory", [])),
        "memory_stat": {k: memstat.get(k) for k in ("pgfault", "pgmajfault")},
        "memory_current_bytes": int(memcurrent_lines[0].strip())
        if memcurrent_lines and memcurrent_lines[0].strip().isdigit()
        else None,
    }


def _rate(before, after, key, elapsed_s):
    if before.get(key) is None or after.get(key) is None or elapsed_s <= 0:
        return None
    return (after[key] - before[key]) / elapsed_s


def compute_delta(
    before: dict, after: dict, clk_tck: int = CLK_TCK_DEFAULT, link_bps: float | None = None
) -> dict:
    elapsed_s = (after["ts_ns"] - before["ts_ns"]) / 1e9
    delta = {
        "pod": after["pod"],
        "node": after["node"],
        "pod_changed": before["pod"] != after["pod"],
        "elapsed_s": round(elapsed_s, 6),
        "window_start_ns": before["ts_ns"],
        "window_end_ns": after["ts_ns"],
        "loadavg_before": before["loadavg"],
        "loadavg_after": after["loadavg"],
    }

    # cgroup CPU usage: v2 usage_usec (us) or v1 cpuacct.usage (ns).
    cpu = {}
    usage_before_ns = None
    usage_after_ns = None
    if "usage_usec" in before["cpu_stat"] and "usage_usec" in after["cpu_stat"]:
        usage_before_ns = before["cpu_stat"]["usage_usec"] * 1000
        usage_after_ns = after["cpu_stat"]["usage_usec"] * 1000
    elif before.get("cpuacct_usage_ns") is not None and after.get("cpuacct_usage_ns") is not None:
        usage_before_ns = before["cpuacct_usage_ns"]
        usage_after_ns = after["cpuacct_usage_ns"]
    if usage_before_ns is not None and elapsed_s > 0:
        cpu["avg_cores"] = round((usage_after_ns - usage_before_ns) / 1e9 / elapsed_s, 6)
    for key in ("nr_throttled", "nr_periods"):
        rate = _rate(before["cpu_stat"], after["cpu_stat"], key, elapsed_s)
        if rate is not None:
            cpu[f"{key}_per_s"] = round(rate, 3)
    proc_before = before["proc_stat"]
    proc_after = after["proc_stat"]
    if proc_before and proc_after and elapsed_s > 0:
        ticks = (proc_after["utime_ticks"] - proc_before["utime_ticks"]) + (
            proc_after["stime_ticks"] - proc_before["stime_ticks"]
        )
        cpu["worker_process_avg_cores"] = round(ticks / clk_tck / elapsed_s, 6)
    delta["cpu"] = cpu

    ctxt = {}
    for key in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
        rate = _rate(before["ctxt_switches"], after["ctxt_switches"], key, elapsed_s)
        if rate is not None:
            ctxt[f"{key}_per_s"] = round(rate, 1)
    delta["ctxt_switches"] = ctxt

    interfaces = {}
    for name, after_counters in after["netdev"].items():
        before_counters = before["netdev"].get(name)
        if not before_counters or elapsed_s <= 0:
            continue
        rx_bps = (after_counters["rx_bytes"] - before_counters["rx_bytes"]) / elapsed_s
        tx_bps = (after_counters["tx_bytes"] - before_counters["tx_bytes"]) / elapsed_s
        if rx_bps == 0 and tx_bps == 0:
            continue
        interfaces[name] = {"rx_Bps": round(rx_bps, 1), "tx_Bps": round(tx_bps, 1)}
    delta["netdev"] = interfaces
    netdev_total = {
        "rx_Bps": round(sum(item["rx_Bps"] for item in interfaces.values()), 1),
        "tx_Bps": round(sum(item["tx_Bps"] for item in interfaces.values()), 1),
    }
    if link_bps:
        # B/s * 8 -> bps, then as a percentage of the link capacity.
        netdev_total["rx_link_pct"] = round(netdev_total["rx_Bps"] * 8 / link_bps * 100, 3)
        netdev_total["tx_link_pct"] = round(netdev_total["tx_Bps"] * 8 / link_bps * 100, 3)
    delta["netdev_total"] = netdev_total

    softirq = {}
    for key in ("net_rx", "net_tx"):
        rate = _rate(before["softirq"], after["softirq"], key, elapsed_s)
        if rate is not None:
            softirq[f"{key}_per_s"] = round(rate, 1)
    delta["softirq"] = softirq

    psi = {}
    for name in ("psi_cpu", "psi_memory"):
        before_some = before[name].get("some", {})
        after_some = after[name].get("some", {})
        entry = {"after_avg10": after_some.get("avg10"), "after_avg60": after_some.get("avg60")}
        if "total" in before_some and "total" in after_some and elapsed_s > 0:
            entry["some_stall_us_per_s"] = round(
                (after_some["total"] - before_some["total"]) / elapsed_s, 1
            )
        psi[name] = entry
    delta["psi"] = psi

    memory = {"current_bytes_after": after["memory_current_bytes"]}
    for key in ("pgfault", "pgmajfault"):
        rate = _rate(before["memory_stat"], after["memory_stat"], key, elapsed_s)
        if rate is not None:
            memory[f"{key}_per_s"] = round(rate, 1)
    delta["memory"] = memory
    return delta


# DataSystem worker "resource" log layout. Lines split on " | ": the first 7
# fields are the standard log header (time|level|filename|pod|pid:tid|trace_id|
# cluster_name); the remaining 22 are the ResMetricName fields in
# res_metrics.def order. Verified against the client log-monitor test which
# uses exactly 7 header fields and asserts object size at absolute index 11.
RESOURCE_HEADER_FIELDS = 7
# 0-based message indices (i.e. offset after the 7 header fields).
RESOURCE_INT_FIELDS = {
    "client_count": 2,   # ACTIVE_CLIENT_COUNT
    "object_count": 3,   # OBJECT_COUNT
    "object_size": 4,    # OBJECT_SIZE
}
# Slash-separated fields kept verbatim (memHit/diskHit/l2Hit/remoteHit/miss).
RESOURCE_STRING_FIELDS = {
    "cache_hit": 21,     # OC_HIT_NUM
}
# Thread pools report "maxRunning/currentTotal/tasksDelta/maxWaiting/usage"
# (thread_pool.h ThreadPoolUsage::ToString(int64_t)). maxWaiting is the peak
# queued-task depth over the interval -- the direct saturation signal.
THREAD_POOL_FIELDS = (
    ("worker_oc_service", 5),          # WORKER_OC_SERVICE_THREAD_POOL (main Get/Set RPC pool)
    ("worker_worker_oc_service", 6),
    ("master_worker_oc_service", 7),
    ("master_oc_service", 8),
    ("master_async_tasks", 12),
)


def _field_int(fields: list[str], index: int) -> int | None:
    if index >= len(fields):
        return None
    value = fields[index].strip()
    try:
        return int(value)
    except ValueError:
        return None


def parse_thread_pool(field: str) -> dict | None:
    parts = field.strip().split("/")
    if len(parts) != 5:
        return None
    try:
        return {
            "max_running": int(parts[0]),
            "current_total": int(parts[1]),
            "tasks_delta": int(parts[2]),
            "max_waiting": int(parts[3]),
            "usage": float(parts[4]),
        }
    except ValueError:
        return None


def parse_resource_timestamp_ns(value: str) -> int | None:
    """Parse DataSystem resource timestamps.

    DataSystem emits naive ISO timestamps in UTC even when the Kubernetes host
    uses CST. Offset-bearing timestamps are respected as written.
    """
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    parsed = parsed.astimezone(dt.timezone.utc)
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    delta = parsed - epoch
    return ((delta.days * 86400 + delta.seconds) * 1_000_000_000
            + delta.microseconds * 1000)


def _thread_pool_peak(lines: list[dict]) -> dict:
    peak: dict = {}
    for name, _ in THREAD_POOL_FIELDS:
        waiting = [e[name]["max_waiting"] for e in lines if name in e]
        usage = [e[name]["usage"] for e in lines if name in e]
        peak[name] = {
            "max_waiting": max(waiting) if waiting else None,
            "max_usage": max(usage) if usage else None,
        }
    return peak


def parse_resource_log(
    raw: str,
    pod: str,
    node: str,
    ts_ns: int,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
) -> dict:
    lines: list[dict] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(" | ")]
        if len(fields) < RESOURCE_HEADER_FIELDS + 1:
            continue
        entry: dict = {"ts": fields[0]}
        resource_ts_ns = parse_resource_timestamp_ns(fields[0])
        if resource_ts_ns is not None:
            entry["resource_ts_ns"] = resource_ts_ns
        for name, msg_idx in RESOURCE_INT_FIELDS.items():
            value = _field_int(fields, RESOURCE_HEADER_FIELDS + msg_idx)
            if value is not None:
                entry[name] = value
        for name, msg_idx in RESOURCE_STRING_FIELDS.items():
            if RESOURCE_HEADER_FIELDS + msg_idx < len(fields):
                entry[name] = fields[RESOURCE_HEADER_FIELDS + msg_idx]
        for name, msg_idx in THREAD_POOL_FIELDS:
            idx = RESOURCE_HEADER_FIELDS + msg_idx
            if idx >= len(fields):
                continue
            pool = parse_thread_pool(fields[idx])
            if pool is not None:
                entry[name] = pool
        lines.append(entry)

    window_lines = lines
    coverage_ts_ns = None
    if window_start_ns is not None and window_end_ns is not None:
        # A resource line summarizes the preceding monitor interval. The first
        # line emitted after measurement end is therefore the interval that
        # contains the tail of the measured burst and is the least diluted
        # server-side saturation sample.
        coverage = [
            entry for entry in lines
            if entry.get("resource_ts_ns", -1) >= window_end_ns
        ]
        if coverage:
            selected = min(coverage, key=lambda entry: entry["resource_ts_ns"])
            coverage_ts_ns = selected["resource_ts_ns"]
            window_lines = [selected]
        else:
            window_lines = []
    return {
        "pod": pod,
        "node": node,
        "ts_ns": ts_ns,
        "line_count": len(lines),
        "resource_timestamp_timezone": "UTC when no offset is present",
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "coverage_resource_ts_ns": coverage_ts_ns,
        "window_line_count": len(window_lines),
        "peak": _thread_pool_peak(window_lines),
        "window_lines": window_lines,
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("parse-snapshot")
    snapshot.add_argument("--raw", type=pathlib.Path, required=True)
    snapshot.add_argument("--pod", required=True)
    snapshot.add_argument("--node", required=True)
    snapshot.add_argument("--ts-ns", type=int, required=True)
    snapshot.add_argument("--out", type=pathlib.Path, required=True)

    delta = sub.add_parser("delta")
    delta.add_argument("--before", type=pathlib.Path, required=True)
    delta.add_argument("--after", type=pathlib.Path, required=True)
    delta.add_argument("--out", type=pathlib.Path, required=True)
    delta.add_argument("--clk-tck", type=int, default=CLK_TCK_DEFAULT)
    delta.add_argument("--link-bps", type=float, default=25e9,
                       help="link capacity in bits/s for saturation percentage (default 25 Gbps)")

    reslog = sub.add_parser("parse-resource-log")
    reslog.add_argument("--raw", type=pathlib.Path, required=True)
    reslog.add_argument("--pod", required=True)
    reslog.add_argument("--node", required=True)
    reslog.add_argument("--ts-ns", type=int, required=True)
    reslog.add_argument("--window-start-ns", type=int)
    reslog.add_argument("--window-end-ns", type=int)
    reslog.add_argument("--out", type=pathlib.Path, required=True)

    args = parser.parse_args()
    if args.command == "parse-snapshot":
        result = parse_snapshot(args.raw.read_text(), args.pod, args.node, args.ts_ns)
    elif args.command == "parse-resource-log":
        result = parse_resource_log(
            args.raw.read_text(), args.pod, args.node, args.ts_ns,
            args.window_start_ns, args.window_end_ns,
        )
    else:
        result = compute_delta(
            json.loads(args.before.read_text()),
            json.loads(args.after.read_text()),
            args.clk_tck,
            args.link_bps,
        )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    marker = {"parse-snapshot": "SNAPSHOT", "parse-resource-log": "RESOURCE_LOG", "delta": "DELTA"}[args.command]
    print(f"DATASYSTEM_WORKER_METRICS_{marker}_OK out={args.out}")


if __name__ == "__main__":
    main()
