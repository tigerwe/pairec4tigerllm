#!/usr/bin/env python3
"""Align header-only tcpdump output with a KVC burst business Get window."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


PACKET_RE = re.compile(
    r"^(?P<ts>\d+\.\d+)\s+IP\s+"
    r"(?P<src_ip>\d+(?:\.\d+){3})\.(?P<src_port>\d+)\s+>\s+"
    r"(?P<dst_ip>\d+(?:\.\d+){3})\.(?P<dst_port>\d+):.*\slength\s+"
    r"(?P<length>\d+)\s*$"
)


def read_burst(path: pathlib.Path) -> dict:
    result = None
    for line in path.read_text(errors="replace").splitlines():
        marker = line.find("{")
        if marker < 0 or '"event":"kvc_burst_complete"' not in line:
            continue
        try:
            event = json.loads(line[marker:])
        except json.JSONDecodeError:
            continue
        if event.get("event") == "kvc_burst_complete":
            result = event
    if result is None:
        raise ValueError(f"no kvc_burst_complete event in {path}")
    for field in ("business_get_start_epoch_ns", "business_get_end_epoch_ns"):
        if not int(result.get(field, 0)):
            raise ValueError(f"KVC event is missing {field}")
    return result


def parse_packet(line: str) -> dict | None:
    match = PACKET_RE.match(line.strip())
    if match is None:
        return None
    row = match.groupdict()
    seconds, fraction = row["ts"].split(".", 1)
    timestamp_ns = int(seconds) * 1_000_000_000 + int(fraction[:9].ljust(9, "0"))
    return {
        "ts_ns": timestamp_ns,
        "src_ip": row["src_ip"],
        "src_port": int(row["src_port"]),
        "dst_ip": row["dst_ip"],
        "dst_port": int(row["dst_port"]),
        "payload_bytes": int(row["length"]),
    }


def summarize(capture: pathlib.Path, burst: dict, endpoint: str, bucket_ms: int) -> dict:
    endpoint_ip, endpoint_port_text = endpoint.rsplit(":", 1)
    endpoint_port = int(endpoint_port_text)
    business_start = int(burst["business_get_start_epoch_ns"])
    business_end = int(burst["business_get_end_epoch_ns"])
    pressure_starts = [int(value) for value in burst.get("pressure_start_epoch_ns", []) if value]
    pressure_ends = [int(value) for value in burst.get("pressure_end_epoch_ns", []) if value]
    pressure_start = min(pressure_starts, default=business_start)
    pressure_end = max(pressure_ends, default=business_end)
    bucket_ns = bucket_ms * 1_000_000
    connections: dict[tuple[str, int], dict] = {}
    parsed_packets = 0

    for line in capture.read_text(errors="replace").splitlines():
        packet = parse_packet(line)
        if packet is None:
            continue
        server_to_client = (
            packet["src_ip"] == endpoint_ip and packet["src_port"] == endpoint_port
        )
        client_to_server = (
            packet["dst_ip"] == endpoint_ip and packet["dst_port"] == endpoint_port
        )
        if not server_to_client and not client_to_server:
            continue
        parsed_packets += 1
        client_ip = packet["dst_ip"] if server_to_client else packet["src_ip"]
        client_port = packet["dst_port"] if server_to_client else packet["src_port"]
        key = (client_ip, client_port)
        row = connections.setdefault(
            key,
            {
                "connection": f"{client_ip}:{client_port}->{endpoint}",
                "client_ip": client_ip,
                "client_port": client_port,
                "client_to_server_payload_bytes": 0,
                "server_to_client_payload_bytes": 0,
                "client_to_server_business_window_bytes": 0,
                "server_to_client_business_window_bytes": 0,
                "client_to_server_pressure_window_bytes": 0,
                "server_to_client_pressure_window_bytes": 0,
                "server_to_client_before_business_bytes": 0,
                "server_to_client_after_business_bytes": 0,
                "packet_count": 0,
                "business_window_packet_count": 0,
                "server_to_client_buckets": {},
            },
        )
        payload = packet["payload_bytes"]
        direction = "server_to_client" if server_to_client else "client_to_server"
        row[f"{direction}_payload_bytes"] += payload
        row["packet_count"] += 1
        if business_start <= packet["ts_ns"] <= business_end:
            row[f"{direction}_business_window_bytes"] += payload
            row["business_window_packet_count"] += 1
        if pressure_start <= packet["ts_ns"] <= pressure_end:
            row[f"{direction}_pressure_window_bytes"] += payload
        if server_to_client:
            if packet["ts_ns"] < business_start:
                row["server_to_client_before_business_bytes"] += payload
            elif packet["ts_ns"] > business_end:
                row["server_to_client_after_business_bytes"] += payload
            bucket = packet["ts_ns"] // bucket_ns
            buckets = row["server_to_client_buckets"]
            buckets[bucket] = buckets.get(bucket, 0) + payload

    ordered = []
    for row in connections.values():
        buckets = row.pop("server_to_client_buckets")
        row["peak_server_to_client_bucket_bytes"] = max(buckets.values(), default=0)
        row["peak_server_to_client_bucket_gbps"] = (
            row["peak_server_to_client_bucket_bytes"] * 8 / (bucket_ms / 1000.0) / 1e9
        )
        ordered.append(row)
    ordered.sort(
        key=lambda row: row["server_to_client_pressure_window_bytes"], reverse=True
    )
    return {
        "request_id": burst.get("request_id", ""),
        "endpoint": endpoint,
        "business_get_start_epoch_ns": business_start,
        "business_get_end_epoch_ns": business_end,
        "business_get_window_ms": (business_end - business_start) / 1e6,
        "pressure_window_start_epoch_ns": pressure_start,
        "pressure_window_end_epoch_ns": pressure_end,
        "pressure_window_ms": (pressure_end - pressure_start) / 1e6,
        "bucket_ms": bucket_ms,
        "parsed_packet_count": parsed_packets,
        "unique_connection_count": len(ordered),
        "connections": ordered,
    }


def print_summary(result: dict) -> None:
    print(
        "request_id business_window_ms pressure_window_ms packets connections"
    )
    print(
        f'{result["request_id"]} {result["business_get_window_ms"]:.3f} '
        f'{result["pressure_window_ms"]:.3f} {result["parsed_packet_count"]} '
        f'{result["unique_connection_count"]}'
    )
    print(
        "connection s2c_total s2c_before_business s2c_during_business "
        "s2c_after_business s2c_pressure_window c2s_during_business peak_bucket_gbps"
    )
    for row in result["connections"]:
        print(
            f'{row["connection"]} {row["server_to_client_payload_bytes"]} '
            f'{row["server_to_client_before_business_bytes"]} '
            f'{row["server_to_client_business_window_bytes"]} '
            f'{row["server_to_client_after_business_bytes"]} '
            f'{row["server_to_client_pressure_window_bytes"]} '
            f'{row["client_to_server_business_window_bytes"]} '
            f'{row["peak_server_to_client_bucket_gbps"]:.3f}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=pathlib.Path, required=True)
    parser.add_argument("--kvc-log", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket-ms", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.bucket_ms <= 1000:
        raise SystemExit("bucket-ms must be between 1 and 1000")
    burst = read_burst(args.kvc_log)
    result = summarize(args.capture, burst, args.endpoint, args.bucket_ms)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_summary(result)
    print(f'KVC_TCP_CAPTURE_SUMMARY_OK output={args.output}')


if __name__ == "__main__":
    main()
