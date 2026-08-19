import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_pairec_brpc_wrapper_kvc_partial.py"


class PartialKvcSummaryTest(unittest.TestCase):
    def test_skips_zero_onboard_and_compares_valid_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for phase in ("baseline", "combined"):
                replay = root / phase / "contention" / "round-1" / "replay"
                replay.mkdir(parents=True)
                get_us = 7000 if phase == "baseline" else 10000
                (replay / "brpc_trtllm.log").write_text(json.dumps({
                    "event": "datasystem_request_complete", "request_id": phase,
                    "get_count": 2, "get_us": get_us, "set_count": 3, "set_us": 20000,
                }) + "\n")
                (replay / "kvc_burst.log").write_text(json.dumps({
                    "event": "kvc_burst_complete", "valid": True,
                    "business_get_ms": get_us / 1000,
                    "pressure_get_p99_ms": 80, "pressure_key_count": 16,
                    "object_size_bytes": 1835008,
                    "business_submit_rank": 100,
                    "pressure_inflight_at_business_start": 99,
                }) + "\n")
                zero = root / phase / "contention" / "round-2" / "replay"
                zero.mkdir(parents=True)
                (zero / "brpc_trtllm.log").write_text(json.dumps({
                    "event": "datasystem_request_complete", "get_count": 0,
                    "set_count": 3, "get_us": 0, "set_us": 20000,
                }) + "\n")
            output = root / "summary.json"
            completed = subprocess.run(
                ["python3", str(SCRIPT), str(root), "--output", str(output)],
                check=True, text=True, capture_output=True)
            result = json.loads(output.read_text())
            self.assertIn("baseline: valid=1/2", completed.stdout)
            self.assertIn("combined: valid=1/2", completed.stdout)
            self.assertEqual(result["comparison"][0]["delta_avg_ms"], 3.0)
            self.assertEqual(result["phases"]["combined"]["invalid_rounds"][0]["classification"],
                             "zero_onboard")

    def test_reads_alternate_native_log_and_embedded_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for phase in ("baseline", "combined"):
                replay = root / phase / "contention" / "round-1" / "replay"
                replay.mkdir(parents=True)
                native = {
                    "event": "datasystem_request_complete", "get_count": 2,
                    "get_us": 8000, "set_count": 3, "set_us": 20000,
                }
                if phase == "baseline":
                    (replay / "brpc_inference.log").write_text(json.dumps(native) + "\n")
                else:
                    (replay / "summary.json").write_text(json.dumps({
                        "datasystem_request_complete": native,
                    }))
                (replay / "kvc_burst.log").write_text(json.dumps({
                    "event": "kvc_burst_complete", "valid": True,
                    "business_get_ms": 8, "pressure_get_p99_ms": 80,
                }) + "\n")
            output = root / "summary.json"
            subprocess.run(["python3", str(SCRIPT), str(root), "--output", str(output)], check=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["phases"]["baseline"]["valid_rounds"], 1)
            self.assertEqual(result["phases"]["combined"]["valid_rounds"], 1)


if __name__ == "__main__":
    unittest.main()
