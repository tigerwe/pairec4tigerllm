import csv
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "scripts" / "summarize_pairec_rank_brpc_burst.py"
spec = importlib.util.spec_from_file_location("rank_burst_summary", SUMMARY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RankBRPCBurstSummaryTest(unittest.TestCase):
    def test_valid_pressure_case_and_tail_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with (root / "requests.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("index", "e2e_ms", "request_id", "response_end_epoch_ns"),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({"index": 1, "e2e_ms": 12.5, "request_id": "req-1",
                                 "response_end_epoch_ns": 360_000_000})
            pairec = [
                {"event": "pairec_rank_brpc_burst_start", "request_id": "req-1",
                 "concurrency": 4, "armed_workers": 4,
                 "business_payload_bytes": 102400, "pressure_payload_bytes": 102400},
                {"event": "pairec_rank_brpc_burst_business_complete", "request_id": "req-1",
                 "business_success": True, "trace_valid": True,
                 "business_payload_bytes": 102400, "business_client_wall_ms": 10,
                 "front_brpc_estimate_ms": 3, "rank_business_start_epoch_ns": 200_000_000,
                 "rank_business_end_epoch_ns": 300_000_000},
                {"event": "pairec_rank_brpc_burst_complete", "request_id": "req-1",
                 "pressure_requests": 3, "pressure_success": 3, "pressure_errors": 0,
                 "burst_valid": True, "pressure_overlap_business": 3,
                 "rank_pressure_first_start_epoch_ns": 190_000_000,
                 "rank_pressure_last_end_epoch_ns": 400_000_000,
                 "pressure_latency_p95_ms": 20, "max_active_workers": 4,
                 "start_skew_us": 5, "pressure_tail_after_business_ms": 100},
                {"event": "deepfm_rank_complete", "request_id": "req-1",
                 "candidate_count": 50, "reordered": True,
                 "model_version": "v1", "model_role": "engineering"},
                {"event": "source_quota_rerank_complete", "request_id": "req-1",
                 "status": "ok", "start_epoch_ns": 310_000_000,
                 "end_epoch_ns": 320_000_000},
                {"event": "pipeline_trace_complete", "request_id": "req-1",
                 "status": "ok", "valid": True, "end_epoch_ns": 350_000_000},
            ]
            (root / "pairec-rank.log").write_text(
                "\n".join(json.dumps(event) for event in pairec) + "\n")
            (root / "inference.log").write_text(
                '1970-01-01T00:00:00.100000000Z '
                + json.dumps({"event": "trt_executor_request_complete", "request_id": "req-1"})
                + "\n")
            (root / "rank-wrapper.log").write_text(
                "[brpc-rank-burst-wrapper] method=Rank request_id=req-1 code=200 "
                "front_payload_bytes=102400 backend_payload_bytes=0 "
                "health_calls_during_rank=3 health_payload_bytes_during_rank=307200\n")

            result = module.summarize(root, 4, 102400, 102400)
            sample = result["samples"][0]
            self.assertEqual("PAIREC_RANK_BRPC_BURST_C4_OK", result["classification"])
            self.assertEqual(10.0, sample["rank_pressure_rerank_overlap_ms"])
            self.assertEqual(50.0, sample["rank_pressure_pipeline_overlap_ms"])
            self.assertEqual(40.0, sample["rank_pressure_tail_after_http_ms"])


if __name__ == "__main__":
    unittest.main()
