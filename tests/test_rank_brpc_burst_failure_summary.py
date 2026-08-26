import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_pairec_rank_brpc_failure.py"
COLLECTOR = ROOT / "scripts" / "diagnose_pairec_rank_brpc_burst_failure.sh"


class RankBurstFailureSummaryTest(unittest.TestCase):
    def test_collector_is_read_only_and_discovers_failed_warmup(self):
        source = COLLECTOR.read_text()
        self.assertIn('root.rglob("response-*.json")', source)
        self.assertIn('data.get("code") != 200', source)
        self.assertNotIn("kubectl -n \"$NAMESPACE\" exec", source)
        self.assertNotIn("curl ", source)

    def summarize(self, pairec_events, wrapper="", adapter="", backend=""):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            files = {}
            for name, content in {
                "pairec": "\n".join(json.dumps(event) for event in pairec_events),
                "wrapper": wrapper,
                "adapter": adapter,
                "backend": backend,
            }.items():
                files[name] = root / f"{name}.log"
                files[name].write_text(content + "\n")
            output = root / "summary.json"
            subprocess.run(
                [
                    "python3", str(SCRIPT), "--request-id", "rid-1",
                    "--pairec", str(files["pairec"]),
                    "--wrapper", str(files["wrapper"]),
                    "--adapter", str(files["adapter"]),
                    "--backend", str(files["backend"]),
                    "--output", str(output),
                ],
                cwd=ROOT, check=True, text=True, capture_output=True,
            )
            return json.loads(output.read_text())

    def test_classifies_pairec_to_wrapper_timeout(self):
        result = self.summarize([{
            "event": "pairec_rank_brpc_burst_business_complete",
            "request_id": "rid-1", "business_success": False,
            "business_client_wall_ms": 1000.4,
            "business_error": "context deadline exceeded", "trace_valid": False,
        }])
        self.assertEqual(result["classification"], "RANK_PAIREC_TO_WRAPPER_TIMEOUT")
        self.assertEqual(result["confidence"], "high")

    def test_classifies_wrapper_to_adapter_timeout(self):
        wrapper = (
            "[brpc-rank-burst-wrapper] method=Rank request_id=rid-1 code=500 "
            "wrapper_total_ms=250.1 backend_rpc_ms=250.0 error=ERPCTIMEDOUT"
        )
        result = self.summarize([], wrapper=wrapper)
        self.assertEqual(result["classification"], "RANK_WRAPPER_TO_ADAPTER_TIMEOUT")

    def test_classifies_adapter_backend_timeout(self):
        wrapper = (
            "[brpc-rank-burst-wrapper] method=Rank request_id=rid-1 code=500 "
            "wrapper_total_ms=81.0 backend_rpc_ms=80.8 error=backend HTTP timeout"
        )
        result = self.summarize([], wrapper=wrapper)
        self.assertEqual(result["classification"], "RANK_ADAPTER_TO_BACKEND_TIMEOUT")

    def test_classifies_response_contract_failure_after_successful_rpc(self):
        events = [
            {
                "event": "pairec_rank_brpc_burst_business_complete",
                "request_id": "rid-1", "business_success": True,
                "business_client_wall_ms": 20.0, "trace_valid": True,
            },
            {
                "event": "deepfm_rank_error", "request_id": "rid-1",
                "error": "rank model_role mismatch: got engineering want production",
            },
        ]
        wrapper = (
            "[brpc-rank-burst-wrapper] method=Rank request_id=rid-1 code=200 "
            "wrapper_total_ms=8 backend_rpc_ms=7 error=none"
        )
        result = self.summarize(events, wrapper=wrapper)
        self.assertEqual(result["classification"], "RANK_RESPONSE_CONTRACT_FAILURE")
        self.assertEqual(result["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
