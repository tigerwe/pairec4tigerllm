#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys


HOP1_EVENT_PREFIX = "pairec_post_rank_hop1_brpc_burst"
HOP2_EVENT_PREFIX = "pairec_post_rank_hop2_brpc_burst"


def read_events(path):
    events = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        position = line.find("{")
        if position < 0:
            continue
        try:
            event = json.loads(line[position:])
        except json.JSONDecodeError:
            continue
        if event.get("request_id"):
            events.append(event)
    return events


def summarize(pairec_events, hop1_events, request_ids, expected_pressure):
    rows = []
    overall = "valid"
    for request_id in request_ids:
        row = {"request_id": request_id}
        for name, source, prefix in (
            ("hop1", pairec_events, HOP1_EVENT_PREFIX),
            ("hop2", hop1_events, HOP2_EVENT_PREFIX),
        ):
            starts = [
                event for event in source
                if event.get("event") == prefix + "_start"
                and event.get("request_id") == request_id
            ]
            businesses = [
                event for event in source
                if event.get("event") == prefix + "_business_complete"
                and event.get("request_id") == request_id
            ]
            completions = [
                event for event in source
                if event.get("event") == prefix + "_complete"
                and event.get("request_id") == request_id
            ]
            side = {
                "start_count": len(starts),
                "business_count": len(businesses),
                "complete_count": len(completions),
                "state": "pending",
            }
            if businesses:
                business = businesses[-1]
                side.update({
                    "business_success": business.get("business_success"),
                    "pressure_started_at_business_start":
                        business.get("pressure_started_at_business_start"),
                    "marker_wait_ms": business.get("marker_wait_ms"),
                })
            if completions:
                complete = completions[-1]
                valid = (
                    complete.get("burst_valid") is True
                    and complete.get("pressure_success") == expected_pressure
                    and complete.get("pressure_errors") == 0
                )
                side.update({
                    "state": "valid" if valid else "invalid",
                    "burst_valid": complete.get("burst_valid"),
                    "pressure_success": complete.get("pressure_success"),
                    "pressure_errors": complete.get("pressure_errors"),
                    "pressure_overlap_business":
                        complete.get("pressure_overlap_business"),
                    "pressure_error_samples":
                        complete.get("pressure_error_samples", []),
                })
            row[name] = side
            if side["state"] == "invalid":
                overall = "invalid"
            elif side["state"] == "pending" and overall == "valid":
                overall = "pending"
        rows.append(row)
    return {
        "status": overall,
        "expected_pressure": expected_pressure,
        "requests": rows,
    }


def print_report(report):
    print(
        "POST_RANK_COMPLETION_STATUS"
        f" status={report['status']}"
        f" requests={len(report['requests'])}"
        f" expected_pressure={report['expected_pressure']}"
    )
    for row in report["requests"]:
        parts = [f"request_id={row['request_id']}"]
        for name in ("hop1", "hop2"):
            side = row[name]
            parts.extend([
                f"{name}_state={side['state']}",
                f"{name}_start={side['start_count']}",
                f"{name}_business={side['business_count']}",
                f"{name}_complete={side['complete_count']}",
            ])
            if "pressure_success" in side:
                parts.extend([
                    f"{name}_pressure_success={side['pressure_success']}",
                    f"{name}_pressure_errors={side['pressure_errors']}",
                    f"{name}_burst_valid={str(side['burst_valid']).lower()}",
                ])
            if side.get("pressure_error_samples"):
                parts.append(
                    f"{name}_error_samples="
                    + json.dumps(side["pressure_error_samples"], separators=(",", ":"))
                )
        print(" ".join(parts))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairec-log", required=True)
    parser.add_argument("--hop1-log", required=True)
    parser.add_argument("--expected-pressure", type=int, required=True)
    request_source = parser.add_mutually_exclusive_group(required=True)
    request_source.add_argument("--request-id", action="append")
    request_source.add_argument("--summary-json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    request_ids = args.request_id
    if args.summary_json:
        summary = json.loads(pathlib.Path(args.summary_json).read_text())
        request_ids = [sample["request_id"] for sample in summary.get("samples", [])]
        if not request_ids:
            parser.error("summary JSON has no sample request IDs")
    report = summarize(
        read_events(args.pairec_log),
        read_events(args.hop1_log),
        request_ids,
        args.expected_pressure,
    )
    if not args.quiet:
        print_report(report)
    return {"valid": 0, "pending": 1, "invalid": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
