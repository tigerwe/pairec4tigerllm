import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_post_rank_two_hop_completion.py"
SPEC = importlib.util.spec_from_file_location("post_rank_completion", CHECKER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def event(name, request_id, **fields):
    return {"event": name, "request_id": request_id, **fields}


class PostRankTwoHopCompletionTest(unittest.TestCase):
    def test_valid_requires_both_hops(self):
        request_id = "r1"
        complete = {
            "burst_valid": True,
            "pressure_success": 999,
            "pressure_errors": 0,
        }
        report = MODULE.summarize(
            [event(MODULE.HOP1_EVENT_PREFIX + "_complete", request_id, **complete)],
            [event(MODULE.HOP2_EVENT_PREFIX + "_complete", request_id, **complete)],
            [request_id],
            999,
        )
        self.assertEqual("valid", report["status"])

    def test_missing_hop_is_pending(self):
        request_id = "r1"
        report = MODULE.summarize(
            [event(MODULE.HOP1_EVENT_PREFIX + "_start", request_id)],
            [event(MODULE.HOP2_EVENT_PREFIX + "_start", request_id)],
            [request_id],
            999,
        )
        self.assertEqual("pending", report["status"])
        self.assertEqual("pending", report["requests"][0]["hop2"]["state"])

    def test_terminal_pressure_error_is_invalid(self):
        request_id = "r1"
        invalid = {
            "burst_valid": False,
            "pressure_success": 998,
            "pressure_errors": 1,
            "pressure_error_samples": ["context deadline exceeded"],
        }
        valid = {
            "burst_valid": True,
            "pressure_success": 999,
            "pressure_errors": 0,
        }
        report = MODULE.summarize(
            [event(MODULE.HOP1_EVENT_PREFIX + "_complete", request_id, **invalid)],
            [event(MODULE.HOP2_EVENT_PREFIX + "_complete", request_id, **valid)],
            [request_id],
            999,
        )
        self.assertEqual("invalid", report["status"])
        self.assertEqual(1, report["requests"][0]["hop1"]["pressure_errors"])

    def test_cli_reads_request_ids_from_summary(self):
        complete = {
            "burst_valid": True,
            "pressure_success": 999,
            "pressure_errors": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            summary = root / "summary.json"
            pairec = root / "pairec.log"
            hop1 = root / "hop1.log"
            summary.write_text(json.dumps({"samples": [{"request_id": "r1"}]}))
            pairec.write_text(json.dumps(event(
                MODULE.HOP1_EVENT_PREFIX + "_complete", "r1", **complete)) + "\n")
            hop1.write_text(json.dumps(event(
                MODULE.HOP2_EVENT_PREFIX + "_complete", "r1", **complete)) + "\n")
            result = MODULE.main([
                "--pairec-log", str(pairec),
                "--hop1-log", str(hop1),
                "--expected-pressure", "999",
                "--summary-json", str(summary),
                "--quiet",
            ])
        self.assertEqual(0, result)


if __name__ == "__main__":
    unittest.main()
