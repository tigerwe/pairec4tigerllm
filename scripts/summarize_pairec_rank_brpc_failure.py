#!/usr/bin/env python3
"""Classify one Rank BRPC burst failure from request-correlated logs."""

import argparse
import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple


def read_text(path: str) -> str:
    if not path:
        return ""
    candidate = pathlib.Path(path)
    return candidate.read_text(errors="replace") if candidate.exists() else ""


def json_events(text: str, request_id: str) -> List[Dict[str, Any]]:
    events = []
    for line in text.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if str(event.get("request_id", "")) == request_id:
            events.append(event)
    return events


def last_event(events: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    matches = [event for event in events if event.get("event") == name]
    return matches[-1] if matches else None


def parse_wrapper_line(
    text: str, request_id: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    lines = [
        line for line in text.splitlines()
        if "[brpc-rank-burst-wrapper] method=Rank" in line
        and f"request_id={request_id}" in line
    ]
    if not lines:
        return None, None
    line = lines[-1]
    fields: Dict[str, Any] = {}
    for name in (
        "code", "wrapper_total_ms", "backend_rpc_ms", "backend_service_total_ms",
        "active_health_at_start", "max_active_health", "max_active_total",
        "health_calls_during_rank", "health_payload_bytes_during_rank",
        "health_calls_during_backend", "health_payload_bytes_during_backend",
        "front_payload_bytes", "backend_payload_bytes",
    ):
        match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
        if not match:
            continue
        raw = match.group(1)
        try:
            fields[name] = float(raw) if "." in raw else int(raw)
        except ValueError:
            fields[name] = raw
    error = re.search(r"(?:^| )error=(.*)$", line)
    fields["error"] = error.group(1).strip() if error else ""
    return line, fields


def contains_timeout(value: str) -> bool:
    return bool(re.search(r"timeout|timed out|deadline|ERPCTIMEDOUT", value, re.I))


def classify(
    request_id: str,
    pairec_text: str,
    wrapper_text: str,
    adapter_text: str,
    backend_text: str,
    artifact_text: str,
) -> Dict[str, Any]:
    pairec_events = json_events("\n".join((pairec_text, artifact_text)), request_id)
    business = last_event(pairec_events, "pairec_rank_brpc_burst_business_complete")
    complete = last_event(pairec_events, "pairec_rank_brpc_burst_complete")
    rank_error = last_event(pairec_events, "deepfm_rank_error")
    pipeline = last_event(pairec_events, "pipeline_trace_complete")
    wrapper_line, wrapper = parse_wrapper_line(
        "\n".join((wrapper_text, artifact_text)), request_id
    )

    request_text = "\n".join(
        part for part in (
            pairec_text, wrapper_text, adapter_text, backend_text, artifact_text
        ) if part
    )
    rank_error_text = str((rank_error or {}).get("error", ""))
    business_error = str((business or {}).get("business_error", ""))
    wrapper_error = str((wrapper or {}).get("error", ""))
    business_wall_ms = float((business or {}).get("business_client_wall_ms", 0) or 0)
    backend_rpc_ms = float((wrapper or {}).get("backend_rpc_ms", 0) or 0)
    wrapper_code = int((wrapper or {}).get("code", 0) or 0)

    classification = "UNKNOWN_RANK_BURST_FAILURE"
    confidence = "low"
    reason = "request evidence exists but does not identify a known failure boundary"
    next_action = "inspect the request-specific raw logs before changing timeouts"

    contract_patterns = (
        r"rank service code=|rank request_id mismatch|rank model_version is empty|"
        r"rank model_role mismatch|rank score_unique_count must be positive|"
        r"rank item count mismatch|unknown item_id|duplicate item_id|non-finite score"
    )
    if not request_text.strip():
        classification = "REQUEST_EVIDENCE_NOT_FOUND"
        confidence = "high"
        reason = "no live or artifact log contains the request"
        next_action = "rerun immediately with a longer LOG_SINCE before the Pods are replaced"
    elif rank_error_text and re.search(contract_patterns, rank_error_text, re.I):
        classification = "RANK_RESPONSE_CONTRACT_FAILURE"
        confidence = "high"
        reason = rank_error_text
        next_action = "fix the response identity/model/item/score contract; do not raise RPC timeouts"
    elif re.search(r"DeepFM scoring failed|Traceback \(most recent call last\)", backend_text, re.I):
        classification = "RANK_PYTHON_BACKEND_FAILURE"
        confidence = "high"
        reason = "Python backend emitted a scoring exception or traceback"
        next_action = "inspect rank-backend-request.log and rank-backend.log for the model exception"
    elif wrapper and wrapper_code != 200 and contains_timeout(wrapper_error):
        if backend_rpc_ms >= 200:
            classification = "RANK_WRAPPER_TO_ADAPTER_TIMEOUT"
            confidence = "high"
            reason = f"Rank Wrapper backend RPC failed after {backend_rpc_ms:.3f}ms near its 250ms timeout"
            next_action = "inspect adapter/backend timing and wrapper CPU scheduling before changing the 250ms timeout"
        elif backend_rpc_ms >= 60:
            classification = "RANK_ADAPTER_TO_BACKEND_TIMEOUT"
            confidence = "medium"
            reason = f"Wrapper received an adapter failure after {backend_rpc_ms:.3f}ms near the adapter 80ms timeout"
            next_action = "confirm the adapter HTTP timeout text and Python /rank completion timing"
        else:
            classification = "RANK_ADAPTER_OR_BACKEND_TIMEOUT"
            confidence = "medium"
            reason = f"Wrapper backend RPC reported timeout after {backend_rpc_ms:.3f}ms"
            next_action = "use adapter/backend logs to identify whether the 80ms backend boundary fired"
    elif business and contains_timeout(business_error) and not wrapper:
        classification = "RANK_PAIREC_TO_WRAPPER_TIMEOUT"
        confidence = "high" if business_wall_ms >= 800 else "medium"
        reason = (
            f"PaiRec business Rank failed after {business_wall_ms:.3f}ms and the Wrapper "
            "never logged method=Rank"
        )
        next_action = "inspect Rank Wrapper admission/BRPC server scheduling under 999 Health requests"
    elif business and contains_timeout(business_error):
        classification = "RANK_FRONT_BRPC_TIMEOUT"
        confidence = "medium"
        reason = f"PaiRec business Rank timed out after {business_wall_ms:.3f}ms"
        next_action = "compare business end time with the Wrapper Rank completion before changing the 1000ms timeout"
    elif wrapper and wrapper_code != 200:
        classification = "RANK_ADAPTER_OR_BACKEND_FAILURE"
        confidence = "high"
        reason = wrapper_error or f"Rank Wrapper returned code={wrapper_code}"
        next_action = "inspect adapter and Python backend logs for the propagated backend error"
    elif business and business.get("business_success") is True and business.get("trace_valid") is False:
        classification = "RANK_TRACE_CONTRACT_FAILURE"
        confidence = "high"
        reason = "business RPC succeeded but coordinator rejected its ServiceTrace"
        next_action = "compare Rank Wrapper and adapter TraceContext/component/status fields"
    elif wrapper and wrapper_code == 200 and rank_error_text:
        classification = "RANK_RESPONSE_CONTRACT_FAILURE"
        confidence = "high"
        reason = rank_error_text
        next_action = "inspect DeepFM response validation fields; transport completed successfully"
    elif complete and int(complete.get("pressure_errors", 0) or 0) > 0:
        classification = "RANK_PRESSURE_RPC_FAILURE"
        confidence = "high"
        reason = (
            f'{complete.get("pressure_errors", 0)}/{complete.get("pressure_requests", 0)} '
            "pressure RPCs failed"
        )
        next_action = "inspect pressure_error_samples before changing concurrency"
    elif rank_error_text:
        classification = "UNKNOWN_DEEPFM_RANK_FAILURE"
        confidence = "medium"
        reason = rank_error_text

    result = {
        "classification": classification,
        "confidence": confidence,
        "request_id": request_id,
        "reason": reason,
        "next_action": next_action,
        "evidence": {
            "pairec_event_count": len(pairec_events),
            "business_event": business,
            "complete_event": complete,
            "deepfm_rank_error": rank_error,
            "pipeline_event_found": pipeline is not None,
            "wrapper_rank_line": wrapper_line,
            "wrapper_fields": wrapper,
            "adapter_request_found": request_id in adapter_text,
            "backend_request_found": request_id in backend_text,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--pairec", default="")
    parser.add_argument("--wrapper", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--artifacts", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = classify(
        args.request_id,
        read_text(args.pairec),
        read_text(args.wrapper),
        read_text(args.adapter),
        read_text(args.backend),
        read_text(args.artifacts),
    )
    output = pathlib.Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n")

    evidence = result["evidence"]
    business = evidence["business_event"] or {}
    complete = evidence["complete_event"] or {}
    wrapper = evidence["wrapper_fields"] or {}
    print("== Rank BRPC burst diagnosis ==")
    print(f'classification={result["classification"]}')
    print(f'confidence={result["confidence"]}')
    print(f'request_id={result["request_id"]}')
    print(f'reason={result["reason"]}')
    print(
        "business "
        f'success={business.get("business_success", "missing")} '
        f'wall_ms={business.get("business_client_wall_ms", "missing")} '
        f'trace_valid={business.get("trace_valid", "missing")} '
        f'error={business.get("business_error", "")}'
    )
    print(
        "wrapper "
        f'found={str(bool(evidence["wrapper_rank_line"])).lower()} '
        f'code={wrapper.get("code", "missing")} '
        f'backend_rpc_ms={wrapper.get("backend_rpc_ms", "missing")} '
        f'error={wrapper.get("error", "")}'
    )
    print(
        "pressure "
        f'requests={complete.get("pressure_requests", "missing")} '
        f'success={complete.get("pressure_success", "missing")} '
        f'errors={complete.get("pressure_errors", "missing")} '
        f'samples={complete.get("pressure_error_samples", [])}'
    )
    print(f'next_action={result["next_action"]}')
    print(f"summary_json={output}")


if __name__ == "__main__":
    main()
