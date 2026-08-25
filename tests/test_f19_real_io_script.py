import json
import pathlib
import re
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_f19_datasystem_real_io.sh"
F14_VALIDATOR = ROOT / "scripts" / "validate_f14_kvc_burst_proxy.sh"
BENCHMARK = ROOT / "scripts" / "benchmark_brpc_kvc_contention.sh"
TRACE = ROOT / "scripts" / "trace_single_brpc_datasystem_request.sh"
KVC_BURST_WRAPPER = ROOT / "cpp" / "kvc_burst" / "kvc_burst_wrapper.cpp"


class F19RealIoScriptTest(unittest.TestCase):
    def run_summary(
        self,
        get_count=2,
        set_count=3,
        onboard_events=2,
        onboard_min=2,
        onboard_max=2,
    ):
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
                "onboard_events": [{"total_ms": 3}] * onboard_events,
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
                str(onboard_min), str(onboard_max), "1", str(result),
                "dsbench", "4", "6", "0", "10", "1", "0", "10",
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

    def test_f14_container_runtime_reset_preserves_ready_sidecar(self):
        text = BENCHMARK.read_text()
        validator_text = F14_VALIDATOR.read_text()
        self.assertIn("RESET_INFERENCE_MODE=container-runtime", validator_text)
        self.assertIn('RESET_INFERENCE_MODE="${RESET_INFERENCE_MODE:-rollout}"', text)
        self.assertIn('reset_inference_container_runtime "$round_dir"', text)
        self.assertIn("crictl stop --timeout 0", text)
        self.assertIn("ctr -n k8s.io tasks kill --signal SIGKILL", text)
        self.assertIn("InternalIP", text)
        self.assertIn('ssh "$runtime_host"', text)
        self.assertIn('INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS="${INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS:-120}"', text)
        self.assertIn('[ "$sidecar_ready" = "true" ]', text)

    def test_f14_arms_burst_only_after_prime(self):
        benchmark_text = BENCHMARK.read_text()
        validator_text = F14_VALIDATOR.read_text()
        runtime_heredoc = benchmark_text.split("<<'SH'\n", 1)[1].split("\nSH\n", 1)[0]
        self.assertNotIn("set_kvc_burst_arm()", runtime_heredoc)
        self.assertIn("\nset_kvc_burst_arm() {\n", benchmark_text)
        self.assertIn('"--control_action=${action}"', benchmark_text)
        self.assertIn('"--host=${ds_host}"', benchmark_text)
        self.assertIn('capture_kvc_burst_failure "$round_dir"', benchmark_text)
        self.assertIn("kvc-burst-failure-sidecar.log", benchmark_text)
        self.assertIn("kvc-burst-failure-inference.log", benchmark_text)
        replay = benchmark_text.index('run_replay "$round_dir"')
        replay_failure_capture = benchmark_text.index(
            'capture_kvc_burst_failure "$round_dir"', replay
        )
        disarm_after_replay = benchmark_text.index(
            'set_kvc_burst_arm "$round_dir" disarm', replay
        )
        self.assertLess(replay_failure_capture, disarm_after_replay)
        disarm = benchmark_text.index('set_kvc_burst_arm "$round_dir" disarm')
        prime = benchmark_text.index('run_prime "$round_dir"')
        arm = benchmark_text.index('set_kvc_burst_arm "$round_dir" refresh-and-arm')
        replay = benchmark_text.index('run_replay "$round_dir"')
        self.assertLess(disarm, prime)
        self.assertLess(prime, arm)
        self.assertLess(arm, replay)
        self.assertIn('KVC_BURST_INITIAL_ARMED=0', validator_text)
        self.assertIn('export KVC_BURST_DYNAMIC_ARM="$enabled"', validator_text)
        self.assertIn('KVC_BURST_VERBOSE="$enabled"', validator_text)

    def test_f14_refreshes_pressure_keys_before_arming(self):
        text = KVC_BURST_WRAPPER.read_text()
        start = text.index('auto refreshOnly = config.controlAction == "refresh"')
        end = text.index("\n    return 0;\n}", start)
        action = text[start:end]
        disarm = action.index("Store(&control.trigger_armed, 0U)")
        ready = action.index("State::kReady")
        request = action.index("RefreshState::kRequested")
        refresh = action.index("RefreshState::kSucceeded")
        arm = action.index("Store(&control.trigger_armed, arm ? 1U : 0U)")
        self.assertLess(disarm, ready)
        self.assertLess(ready, request)
        self.assertLess(request, refresh)
        self.assertLess(refresh, arm)
        self.assertIn('config.controlAction == "verify-and-arm"', action)
        self.assertIn("RefreshState::kVerifyRequested", action)
        self.assertIn("RefreshState::kVerifySucceeded", action)
        self.assertIn('generationConfig.prefix += "_g"', text)
        sidecar_refresh = text.index("kvc_burst_keys_refreshed")
        self.assertLess(text.rindex("DeleteKeys", 0, sidecar_refresh), sidecar_refresh)
        self.assertLess(text.rindex("PrefillAndVerify", 0, sidecar_refresh), sidecar_refresh)
        self.assertIn("pressure Get failed generation=", text)
        self.assertIn("detail=\" << status.ToString()", text)
        self.assertIn("pressure_key_count", text)
        self.assertIn("permutation[i] = i % pressureKeyCount", text)
        self.assertIn("kvc_burst_keys_verified", text)
        self.assertIn("VerifyKeys(*controlClient, keys)", text)

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

    def test_onboard_range_is_explicit_and_default_remains_exact(self):
        strict_code, strict_result = self.run_summary(
            get_count=1, onboard_events=1
        )
        self.assertEqual(1, strict_code)
        self.assertEqual("FAIL", strict_result["status"])

        range_code, range_result = self.run_summary(
            get_count=1,
            onboard_events=1,
            onboard_min=1,
            onboard_max=2,
        )
        self.assertEqual(0, range_code)
        self.assertEqual("PASS", range_result["status"])
        self.assertTrue(range_result["rows"][0]["exact_attribution_ok"])


if __name__ == "__main__":
    unittest.main()
