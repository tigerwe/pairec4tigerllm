import json
import pathlib
import re
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_f19_datasystem_real_io.sh"
BENCHMARK = ROOT / "scripts" / "benchmark_brpc_kvc_contention.sh"
TRACE = ROOT / "scripts" / "trace_single_brpc_datasystem_request.sh"


class F19RealIoScriptTest(unittest.TestCase):
    def run_summary(self, get_count=2, set_count=3):
        text = BENCHMARK.read_text()
        blocks = re.findall(r"<<'PY' \| tee \"\$SUMMARY_TXT\"\n(.*?)\nPY", text, re.DOTALL)
        self.assertEqual(1, len(blocks))
        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        round_dir = root / "round-1"
        replay_dir = round_dir / "replay"
        replay_dir.mkdir(parents=True)
        request_id = "f19-fixture-request"
        summary = {
            "request_id": request_id,
            "client": {"ok": True, "client_e2e_ms": 120.0},
            "response_code": 200,
            "brpc_events": [{"latency_ms": 110}],
            "pairec_generative_trace": {"rpc_ms": 112},
            "kvc_access": {
                "offload_events": [{"total_ms": 2}] * 3,
                "onboard_events": [{"total_ms": 3}] * 2,
            },
            "datasystem_request_completion_count": 1,
            "datasystem_request_complete": {
                "request_id": request_id,
                "get_count": get_count,
                "get_us": 6000,
                "set_count": set_count,
                "set_us": 6000,
                "get_failed_count": 0,
                "set_failed_count": 0,
                "pending_count": 0,
                "unknown_count": 0,
                "attribution_complete": True,
            },
        }
        (replay_dir / "summary.json").write_text(json.dumps(summary))
        (replay_dir / "brpc_trtllm.log").write_text("")
        for name, value in (
            ("inference-restarts.before", "0"),
            ("inference-restarts.after", "0"),
            ("inference-pod.before", "inference-pod"),
            ("inference-pod.after", "inference-pod"),
        ):
            (round_dir / name).write_text(value)
        result = root / "result.json"
        original_argv = sys.argv
        try:
            sys.argv = [
                "embedded-summary", str(root), "baseline", "1", "3", "2",
                "1", str(result), "dsbench", "4", "6", "0", "10", "1",
            ]
            with self.assertRaises(SystemExit) as exit_context:
                exec(compile(blocks[0], "embedded-contention-summary", "exec"), {})
        finally:
            sys.argv = original_argv
        payload = json.loads(result.read_text())
        temporary.cleanup()
        return exit_context.exception.code, payload

    def test_validator_uses_stable_cache_shape_and_exact_gate(self):
        text = VALIDATOR.read_text()
        self.assertIn('PRIME_REQUESTS="${PRIME_REQUESTS:-195}"', text)
        self.assertIn("RESET_INFERENCE_BEFORE_ROUND=1", text)
        self.assertIn("REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1", text)
        self.assertIn('PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec-brpc-observed}"', text)

    def test_contention_gate_checks_exact_completion_contract(self):
        text = BENCHMARK.read_text()
        for field in (
            "exact_completion_count == 1",
            'exact.get("request_id") == raw.get("request_id")',
            'exact.get("attribution_complete") is True',
            'exact.get("set_count")',
            'exact.get("get_count")',
            'exact.get("pending_count")',
            'exact.get("unknown_count")',
        ):
            self.assertIn(field, text)

    def test_trace_summary_preserves_completion_multiplicity(self):
        text = TRACE.read_text()
        self.assertRegex(text, re.compile(r"datasystem_completions\.append\(candidate\)"))
        self.assertIn('"datasystem_request_completion_count": len(datasystem_completions)', text)

    def test_trace_summary_reports_output_token_and_decode_cost(self):
        text = TRACE.read_text()
        for field in (
            '"tr_output_tokens"',
            '"tr_runner_per_token_ms"',
            '"decode_gap_per_interval_us"',
            '"model_execution_lifecycle_pct"',
            '"datasystem_io_count"',
        ):
            self.assertIn(field, text)

    def test_exact_three_set_two_get_fixture_passes(self):
        code, result = self.run_summary()
        self.assertEqual(0, code)
        self.assertEqual("PASS", result["status"])
        self.assertTrue(result["rows"][0]["exact_attribution_ok"])

    def test_legacy_counts_cannot_hide_zero_native_io(self):
        code, result = self.run_summary(get_count=0, set_count=0)
        self.assertEqual(1, code)
        self.assertEqual("FAIL", result["status"])
        self.assertTrue(result["rows"][0]["count_ok"])
        self.assertFalse(result["rows"][0]["exact_attribution_ok"])


if __name__ == "__main__":
    unittest.main()
