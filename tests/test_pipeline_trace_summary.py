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
            log.write_text(json.dumps(trace) + "\n")
            subprocess.run([
                "python3", "scripts/summarize_pairec_pipeline_trace.py",
                "--log", str(log), "--expected", "1", "--output", str(out),
            ], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(out.read_text())["valid_count"], 1)

    def test_trace_marked_invalid_is_rejected(self):
        self.assertIn("trace_marked_invalid", __import__(
            "scripts.summarize_pairec_pipeline_trace", fromlist=["validate"]
        ).validate({"valid": False}, False))


if __name__ == "__main__":
    unittest.main()
