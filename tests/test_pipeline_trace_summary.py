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
                              "runner_generate_us": 70}
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
            self.assertEqual(json.loads(out.read_text())["valid_count"], 1)

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


if __name__ == "__main__":
    unittest.main()
