#!/usr/bin/env python3
"""Attribute recommendation latency to front-side BRPC stages."""


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _one(events, name):
    matches = [event for event in events if event.get("event") == name]
    return matches[-1] if matches else None


def _pipeline_span(events, name):
    pipeline = _one(events, "pipeline_trace_complete") or {}
    for span in pipeline.get("spans") or []:
        if span.get("name") == name and span.get("status") == "ok":
            return span
    return None


def _span_front_ms(span):
    if not span or span.get("protocol") != "brpc":
        return None
    attributes = span.get("attributes") or {}
    service_us = _number(attributes.get("service_total_us"), -1.0)
    duration_us = _number(span.get("duration_us"), -1.0)
    if service_us < 0 or duration_us < 0:
        return None
    return max(0.0, duration_us - service_us) / 1000.0


def attribute_brpc_chain(summary):
    """Return transport/coordination overhead for each BRPC pipeline stage.

    Rank uses the full pipeline span minus backend service time. Under c1000 this
    intentionally includes coordinator and pressure-tail waiting that contributes
    to recommendation E2E. The business-only transport portion is reported
    separately.
    """
    events = summary.get("pairec_json_events") or []
    generative_trace = summary.get("pairec_generative_trace") or {}
    server_ms = sum(
        _number(event.get("latency_ms"))
        for event in summary.get("brpc_events") or []
    )

    generation_event = _one(events, "pairec_brpc_burst_business_complete")
    if generation_event is not None and "business_front_brpc_ms" in generation_event:
        generative_ms = max(0.0, _number(generation_event["business_front_brpc_ms"]))
        generative_source = "pairec_brpc_burst_business_complete"
    elif generative_trace.get("protocol") in (None, "", "brpc") and (
            "rpc_ms" in generative_trace or "brpc_ms" in generative_trace):
        rpc_ms = _number(
            generative_trace.get("rpc_ms", generative_trace.get("brpc_ms")),
            server_ms,
        )
        generative_ms = max(0.0, rpc_ms - server_ms)
        generative_source = "generative_rpc_minus_inference_service"
    else:
        generative_ms = 0.0
        generative_source = "missing"

    vector_span = _pipeline_span(events, "vector_recall")
    vector_value = _span_front_ms(vector_span)
    if vector_value is not None:
        vector_ms = vector_value
        vector_source = "pipeline_span_minus_service"
    else:
        vector_trace = summary.get("pairec_vector_trace") or {}
        if vector_trace.get("protocol") == "brpc" and "service_us" in vector_trace:
            vector_ms = max(
                0.0,
                _number(vector_trace.get("cost"))
                - _number(vector_trace.get("service_us")) / 1000.0,
            )
            vector_source = "vector_cost_minus_service"
        else:
            vector_ms = 0.0
            vector_source = "missing"

    rank_span = _pipeline_span(events, "deepfm_rank")
    rank_value = _span_front_ms(rank_span)
    rank_event = _one(events, "pairec_rank_brpc_burst_business_complete")
    rank_service = _one(events, "deepfm_rank_complete") or {}
    if rank_value is not None:
        rank_ms = rank_value
        rank_source = "pipeline_span_minus_service"
    elif "client_total_ms" in rank_service and "service_total_ms" in rank_service:
        rank_ms = max(
            0.0,
            _number(rank_service["client_total_ms"])
            - _number(rank_service["service_total_ms"]),
        )
        rank_source = "rank_client_minus_service"
    else:
        rank_ms = 0.0
        rank_source = "missing"

    if rank_event is not None and "front_brpc_estimate_ms" in rank_event:
        rank_business_ms = max(0.0, _number(rank_event["front_brpc_estimate_ms"]))
        rank_business_source = "pairec_rank_brpc_burst_business_complete"
    else:
        rank_business_ms = rank_ms
        rank_business_source = rank_source
    rank_ms = max(rank_ms, rank_business_ms)
    rank_coordination_ms = max(0.0, rank_ms - rank_business_ms)

    complete = all(source != "missing" for source in (
        generative_source, vector_source, rank_source,
    ))
    total_ms = generative_ms + vector_ms + rank_ms
    return {
        "brpc_ms": total_ms,
        "generative_brpc_ms": generative_ms,
        "vector_brpc_ms": vector_ms,
        "rank_brpc_ms": rank_ms,
        "rank_business_brpc_ms": rank_business_ms,
        "rank_coordination_ms": rank_coordination_ms,
        "complete": complete,
        "sources": {
            "generative": generative_source,
            "vector": vector_source,
            "rank": rank_source,
            "rank_business": rank_business_source,
        },
    }
