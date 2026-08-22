#!/usr/bin/env python3
"""Sample TCP queue depths for connections to a specific endpoint."""

from __future__ import annotations

import argparse
import ipaddress
import json
import pathlib
import time


def decode_ipv4(value: str) -> str:
    raw = bytes.fromhex(value)
    return str(ipaddress.ip_address(raw[::-1]))


def read_tcp4(endpoint_ip: str, endpoint_port: int, role: str) -> list[dict]:
    rows = []
    path = pathlib.Path("/proc/net/tcp")
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "01":
            continue
        local_hex, local_port_hex = fields[1].split(":")
        remote_hex, remote_port_hex = fields[2].split(":")
        local_ip = decode_ipv4(local_hex)
        remote_ip = decode_ipv4(remote_hex)
        local_port = int(local_port_hex, 16)
        remote_port = int(remote_port_hex, 16)
        if role == "client":
            matches = remote_ip == endpoint_ip and remote_port == endpoint_port
        else:
            matches = local_ip == endpoint_ip and local_port == endpoint_port
        if not matches:
            continue
        tx_hex, rx_hex = fields[4].split(":")
        rows.append(
            {
                "local_ip": local_ip,
                "local_port": local_port,
                "remote_ip": remote_ip,
                "remote_port": remote_port,
                "tx_queue_bytes": int(tx_hex, 16),
                "rx_queue_bytes": int(rx_hex, 16),
                "inode": int(fields[9]),
            }
        )
    return rows


def collect(
    endpoint_ip: str,
    endpoint_port: int,
    role: str,
    interval_ms: int,
    stop_file: pathlib.Path,
) -> None:
    interval = interval_ms / 1000.0
    while not stop_file.exists():
        print(
            json.dumps(
                {
                    "ts_ns": time.time_ns(),
                    "role": role,
                    "connections": read_tcp4(endpoint_ip, endpoint_port, role),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        time.sleep(interval)
    print(
        json.dumps(
            {
                "ts_ns": time.time_ns(),
                "role": role,
                "connections": read_tcp4(endpoint_ip, endpoint_port, role),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def summarize(path: pathlib.Path) -> dict:
    samples = []
    connections: dict[str, dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(sample.get("connections"), list):
            continue
        samples.append(sample)
        for row in sample["connections"]:
            key = (
                f'{row["local_ip"]}:{row["local_port"]}'
                f'->{row["remote_ip"]}:{row["remote_port"]}'
            )
            aggregate = connections.setdefault(
                key,
                {
                    "connection": key,
                    "inode": row.get("inode", 0),
                    "observations": 0,
                    "nonzero_queue_observations": 0,
                    "peak_tx_queue_bytes": 0,
                    "peak_rx_queue_bytes": 0,
                },
            )
            aggregate["observations"] += 1
            tx_queue = int(row.get("tx_queue_bytes", 0))
            rx_queue = int(row.get("rx_queue_bytes", 0))
            if tx_queue or rx_queue:
                aggregate["nonzero_queue_observations"] += 1
            aggregate["peak_tx_queue_bytes"] = max(
                aggregate["peak_tx_queue_bytes"], tx_queue
            )
            aggregate["peak_rx_queue_bytes"] = max(
                aggregate["peak_rx_queue_bytes"], rx_queue
            )
    if not samples:
        raise ValueError(f"no valid TCP queue samples in {path}")
    ordered = sorted(
        connections.values(),
        key=lambda row: max(row["peak_tx_queue_bytes"], row["peak_rx_queue_bytes"]),
        reverse=True,
    )
    return {
        "role": samples[0].get("role", "unknown"),
        "sample_count": len(samples),
        "window_ms": (samples[-1]["ts_ns"] - samples[0]["ts_ns"]) / 1e6,
        "unique_connection_count": len(ordered),
        "queued_connection_count": sum(
            1
            for row in ordered
            if row["peak_tx_queue_bytes"] or row["peak_rx_queue_bytes"]
        ),
        "connections": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--endpoint", required=True)
    collect_parser.add_argument("--role", choices=("client", "server"), required=True)
    collect_parser.add_argument("--interval-ms", type=int, default=5)
    collect_parser.add_argument("--stop-file", type=pathlib.Path, required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--input", type=pathlib.Path, required=True)
    summary_parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.command == "collect":
        if not 1 <= args.interval_ms <= 1000:
            raise SystemExit("interval-ms must be between 1 and 1000")
        endpoint_ip, endpoint_port = args.endpoint.rsplit(":", 1)
        collect(
            endpoint_ip,
            int(endpoint_port),
            args.role,
            args.interval_ms,
            args.stop_file,
        )
        return

    result = summarize(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "role sample_count window_ms unique_connections queued_connections"
    )
    print(
        f'{result["role"]} {result["sample_count"]} {result["window_ms"]:.3f} '
        f'{result["unique_connection_count"]} {result["queued_connection_count"]}'
    )
    print(
        "connection observations queued_observations "
        "peak_tx_queue_bytes peak_rx_queue_bytes"
    )
    for row in result["connections"]:
        print(
            f'{row["connection"]} {row["observations"]} '
            f'{row["nonzero_queue_observations"]} '
            f'{row["peak_tx_queue_bytes"]} {row["peak_rx_queue_bytes"]}'
        )
    print(f'TCP_QUEUE_SUMMARY_OK output={args.output}')


if __name__ == "__main__":
    main()
