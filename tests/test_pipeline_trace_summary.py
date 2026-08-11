import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class PipelineTraceSummaryTest(unittest.TestCase):
    def test_valid_trace(self):
        spans = []
        required = [
            "recommend_service", "response_build", "controller_overhead", "user_feature",
            "recall", "filter", "general_rank", "feature_load", "framework_rank",
            "pipeline_wait", "pipeline_merge", "sort", "generative_recall",
            "vector_recall", "deepfm_rank",
        ]
        for name in required:
            protocol = "brpc" if name in {
                "generative_recall", "vector_recall", "deepfm_rank"} else "in_process"
            attributes = {}
            if name == "generative_recall":
                attributes = {"rpc_us": 90, "inference_total_us": 80,
                              "runner_generate_us": 70,
                              "runner_per_output_token_us": 7,
                              "output_token_count": 10}
            elif name == "vector_recall":
                attributes = {"service_total_us": 50, "feature_us": 10, "compute_us": 30}
            elif name == "deepfm_rank":
                attributes = {"service_total_us": 40, "feature_us": 10,
                              "compute_us": 20, "backend_rpc_us": 35}
            spans.append({"name": name, "enabled": True, "accounted": name == "recommend_service",
                          "duration_us": 100, "start_offset_us": 0, "protocol": protocol,
                          "attributes": attributes})
        spans.append({"name": "rerank", "enabled": False, "accounted": False,
                      "duration_us": 0, "start_offset_us": 0})
        trace = {"event": "pipeline_trace_complete", "contract_version": "pairec.pipeline_trace.v1",
                 "request_id": "r1", "valid": True, "pairec_total_us": 100,
                 "accounted_us": 100, "spans": spans,
                 "datasystem": {"attribution_complete": False}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            log = directory / "trace.log"
            out = directory / "summary.json"
            quota = ("[PAIREC_TRACE] requestId=r1 request_id=r1 module=QuotaMultiRecall "
                     "name=multi_recall_2_48 primary=generative_recall "
                     "secondary=milvus_recall primary_minimum=1 primary_input=8 "
                     "secondary_input=50 primary_selected=2 secondary_selected=48 "
                     "duplicate_count=0 final_count=50 degraded=false cost=100")
            log.write_text(quota + "\n" + json.dumps(trace) + "\n")
            subprocess.run([
                "python3", "scripts/summarize_pairec_pipeline_trace.py",
                "--log", str(log), "--expected", "1", "--output", str(out),
            ], check=True, capture_output=True, text=True)
            summary = json.loads(out.read_text())
            self.assertEqual(summary["valid_count"], 1)
            self.assertEqual(
                summary["service_counts"]["generative_recall.output_token_count"]["avg"],
                10.0)

    def test_missing_quota_selection_is_rejected(self):
        module = __import__(
            "scripts.summarize_pairec_pipeline_trace", fromlist=["validate"])
        reasons = module.validate({"valid": True, "_quota": None}, False)
        self.assertIn("missing_quota_multi_recall", reasons)

    def test_quota_parser_records_generative_selection(self):
        module = __import__(
            "scripts.summarize_pairec_pipeline_trace", fromlist=["extract_quota_stats"])
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "quota.log"
            log.write_text(
                "[PAIREC_TRACE] requestId=r1 request_id=r1 module=QuotaMultiRecall "
                "primary_minimum=1 primary_input=8 secondary_input=50 "
                "primary_selected=2 secondary_selected=48 duplicate_count=0 "
                "final_count=50 degraded=false cost=100\n")
            stats = module.extract_quota_stats(log)["r1"]
            self.assertEqual(stats["primary_selected"], 2)
            self.assertFalse(stats["degraded"])

    def test_trace_marked_invalid_is_rejected(self):
        self.assertIn("trace_marked_invalid", __import__(
            "scripts.summarize_pairec_pipeline_trace", fromlist=["validate"]
        ).validate({"valid": False}, False))

    def test_required_source_rerank_is_validated(self):
        module = __import__(
            "scripts.summarize_pairec_pipeline_trace", fromlist=["validate"])
        trace = {
            "valid": True,
            "contract_version": "pairec.pipeline_trace.v1",
            "pairec_total_us": 100,
            "accounted_us": 100,
            "_quota": {
                "primary_minimum": 1, "primary_selected": 2,
                "final_count": 50, "degraded": False,
            },
            "spans": [{
                "name": "rerank", "enabled": True, "status": "ok",
                "protocol": "in_process", "duration_us": 20, "start_offset_us": 0,
                "attributes": {
                    "policy": "source_quota_tail", "placement": "tail",
                    "input_count": 50, "output_count": 10,
                    "generative_input": 2, "vector_input": 48,
                    "generative_selected": 2, "vector_selected": 8,
                    "minimum_generative": 1, "maximum_generative": 2,
                    "moved_count": 2,
                },
            }],
            "datasystem": {"attribution_complete": False},
        }
        reasons = module.validate(trace, False, True)
        self.assertNotIn("source_rerank_not_enabled", reasons)
        self.assertNotIn("source_rerank_quota", reasons)
        trace["spans"][0]["attributes"]["generative_selected"] = 1
        reasons = module.validate(trace, False, True)
        self.assertIn("source_rerank_did_not_preserve_available", reasons)

    def test_source_rerank_must_be_enabled_when_required(self):
        module = __import__(
            "scripts.summarize_pairec_pipeline_trace", fromlist=["validate"])
        reasons = module.validate({
            "valid": True,
            "contract_version": "pairec.pipeline_trace.v1",
            "pairec_total_us": 1,
            "accounted_us": 1,
            "spans": [{"name": "rerank", "enabled": False}],
            "_quota": {
                "primary_minimum": 1, "primary_selected": 1,
                "final_count": 50, "degraded": False,
            },
        }, False, True)
        self.assertIn("source_rerank_not_enabled", reasons)

    def test_native_datasystem_completion_is_joined_by_request_id(self):
        module = __import__(
            "scripts.summarize_pairec_pipeline_trace",
            fromlist=["extract_datasystem_completions", "validate"])
        event = {
            "event": "datasystem_request_complete", "request_id": "r1",
            "get_count": 2, "get_us": 12000,
            "set_count": 3, "set_us": 19000,
            "get_failed_count": 0, "set_failed_count": 0,
            "pending_count": 0, "unknown_count": 0,
            "executor_queue_us": 10,
            "add_sequence_count": 1, "add_sequence_us": 20,
            "prefill_gap_us": 30,
            "add_token_count": 2, "add_token_us": 40,
            "attribution_lookup_us": 2, "sequence_lookup_us": 3,
            "kv_update_us": 20, "phase_record_us": 1, "decode_gap_us": 50,
            "finalization_gap_us": 60,
            "remove_sequence_count": 1, "remove_sequence_us": 10,
            "native_lifecycle_count": 1, "native_lifecycle_us": 220,
            "native_accounted_us": 220, "native_closure_error_us": 0,
            "phase_unknown_count": 0, "phase_timing_complete": True,
            "attribution_complete": True,
        }
        executor = {
            "event": "trt_executor_request_complete", "request_id": "r1",
            "runner_us": 230, "request_setup_us": 2, "enqueue_call_us": 3,
            "await_final_us": 224, "response_extract_us": 1,
            "gateway_accounted_us": 230, "gateway_closure_error_us": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "inference.log"
            log.write_text(
                "native-prefix " + json.dumps(event, separators=(",", ":")) + "\n"
                + json.dumps(executor, separators=(",", ":")) + "\n")
            completions, duplicates = module.extract_datasystem_completions(log)
            executor_completions = module.extract_executor_completions(log)
        self.assertEqual(completions["r1"]["get_count"], 2)
        self.assertFalse(duplicates)

        trace = {
            "valid": True,
            "_datasystem_final": completions["r1"],
            "_executor_final": executor_completions["r1"],
        }
        reasons = module.validate(trace, True)
        self.assertNotIn("datasystem_completion_missing", reasons)
        self.assertNotIn("datasystem_attribution_incomplete", reasons)
        self.assertNotIn("datasystem_phase_timing_incomplete", reasons)
        self.assertNotIn("executor_phase_completion_missing", reasons)

        completions["r1"]["native_closure_error_us"] = 101
        executor_completions["r1"][0]["gateway_closure_error_us"] = 101
        reasons = module.validate(trace, True)
        self.assertIn("datasystem_phase_closure_error", reasons)
        self.assertIn("executor_phase_closure_error", reasons)

        completions["r1"]["native_closure_error_us"] = 0
        executor_completions["r1"].append(dict(executor_completions["r1"][0]))
        reasons = module.validate(trace, True)
        self.assertIn("executor_native_lifecycle_count_mismatch", reasons)

    def test_native_datasystem_pending_or_failure_is_rejected(self):
        module = __import__(
            "scripts.summarize_pairec_pipeline_trace", fromlist=["validate"])
        event = {
            "get_count": 1, "get_us": 100, "set_count": 1, "set_us": 200,
            "get_failed_count": 1, "set_failed_count": 0,
            "pending_count": 1, "unknown_count": 0,
            "attribution_complete": False,
        }
        reasons = module.validate(
            {"valid": True, "_datasystem_final": event}, True)
        self.assertIn("datasystem_attribution_incomplete", reasons)
        self.assertIn("datasystem_nonzero:get_failed_count", reasons)
        self.assertIn("datasystem_nonzero:pending_count", reasons)


if __name__ == "__main__":
    unittest.main()
