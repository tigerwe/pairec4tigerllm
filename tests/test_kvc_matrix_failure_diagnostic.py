import json
import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_pairec_brpc_wrapper_kvc_matrix_failure.sh"


class MatrixFailureDiagnosticTest(unittest.TestCase):
    def run_diagnostic(self, root: pathlib.Path):
        output = root / "diagnostic"
        env = os.environ.copy()
        env.update({"RUN_ROOT": str(root), "OUTPUT_DIR": str(output)})
        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed, json.loads((output / "summary.json").read_text())

    @staticmethod
    def write_replay(path: pathlib.Path, request_id: str, gets: int, complete: bool,
                     failure: int = 0):
        path.mkdir(parents=True)
        (path / "client.json").write_text(json.dumps({
            "ok": True, "response_code": 200, "request_id": request_id,
        }))
        native = {
            "event": "datasystem_request_complete",
            "request_id": request_id,
            "get_count": gets,
            "set_count": 3,
            "attribution_complete": True,
        }
        (path / "brpc_trtllm.log").write_text(json.dumps(native) + "\n")
        events = []
        if gets:
            events.append({"event": "kvc_burst_start", "request_id": request_id})
        if complete:
            events.append({
                "event": "kvc_burst_complete",
                "request_id": request_id,
                "sustained_loop_gets": 42,
                "sustained_errors": 0,
                "failure": failure,
            })
        (path / "kvc_burst.log").write_text("".join(json.dumps(event) + "\n" for event in events))

    def test_classifies_zero_get_retry_exhaustion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            contention = root / "combined" / "contention"
            round_dir = contention / "round-3"
            self.write_replay(round_dir / "replay-attempt-1", "rid-1", 0, False)
            self.write_replay(round_dir / "replay", "rid-2", 0, False)
            (round_dir / "replay-target-prime-2").mkdir(parents=True)
            (round_dir / "replay-churn-2").mkdir()
            (round_dir / "replay-preparation-2.txt").write_text(
                "pressure_key_control=refresh,target-prime,churn,verify-and-arm\n"
            )
            (round_dir / "replay.attempts").write_text("2\n")
            (round_dir / "replay.exit_code").write_text("1\n")
            (contention / "result.json").write_text(json.dumps({
                "status": "FAIL", "expected_repeats": 3,
                "valid_repeats": 2, "rows": [],
            }))

            completed, result = self.run_diagnostic(root)

            self.assertEqual(result["classification"], "ZERO_ONBOARD_GET_RETRY_EXHAUSTED")
            self.assertEqual(len(result["rounds"][0]["attempts"]), 2)
            self.assertEqual(len(result["rounds"][0]["target_prime_dirs"]), 1)
            self.assertEqual(len(result["rounds"][0]["churn_dirs"]), 1)
            self.assertIn("ZERO_ONBOARD_GET_RETRY_EXHAUSTED", completed.stdout)
            self.assertIn("target_prime_count=1 churn_count=1", completed.stdout)

    def test_classifies_missing_kvc_completion_after_native_get(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            contention = root / "combined" / "contention"
            round_dir = contention / "round-1"
            self.write_replay(round_dir / "replay", "rid-3", 2, False)
            (round_dir / "replay.attempts").write_text("1\n")
            (round_dir / "replay.exit_code").write_text("1\n")
            (contention / "result.json").write_text(json.dumps({
                "status": "FAIL", "expected_repeats": 1,
                "valid_repeats": 0, "rows": [],
            }))

            _, result = self.run_diagnostic(root)

            self.assertEqual(result["classification"], "KVC_TRIGGER_OR_COMPLETION_MISSING")

    def test_classifies_pressure_first_gate_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            contention = root / "combined" / "contention"
            round_dir = contention / "round-1"
            self.write_replay(round_dir / "replay", "rid-pressure", 2, True, failure=5)
            (round_dir / "replay.attempts").write_text("1\n")
            (round_dir / "replay.exit_code").write_text("1\n")
            (contention / "result.json").write_text(json.dumps({
                "status": "FAIL", "expected_repeats": 1,
                "valid_repeats": 0, "rows": [],
            }))

            _, result = self.run_diagnostic(root)

            self.assertEqual(result["classification"], "PRESSURE_FIRST_GATE_FAILURE")


if __name__ == "__main__":
    unittest.main()
