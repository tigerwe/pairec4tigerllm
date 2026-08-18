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


def compute_delta(before: dict, after: dict, clk_tck: int = CLK_TCK_DEFAULT) -> dict:
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
    delta["netdev_total"] = {
        "rx_Bps": round(sum(item["rx_Bps"] for item in interfaces.values()), 1),
        "tx_Bps": round(sum(item["tx_Bps"] for item in interfaces.values()), 1),
    }

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

    args = parser.parse_args()
    if args.command == "parse-snapshot":
        result = parse_snapshot(args.raw.read_text(), args.pod, args.node, args.ts_ns)
    else:
        result = compute_delta(
            json.loads(args.before.read_text()),
            json.loads(args.after.read_text()),
            args.clk_tck,
        )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"DATASYSTEM_WORKER_METRICS_{'SNAPSHOT' if args.command == 'parse-snapshot' else 'DELTA'}_OK out={args.out}")


if __name__ == "__main__":
    main()
