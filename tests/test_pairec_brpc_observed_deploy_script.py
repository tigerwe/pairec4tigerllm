import json
import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_and_validate_pairec_brpc_observed.sh"


class PaiRecBrpcObservedDeployScriptTest(unittest.TestCase):
    def test_embedded_python_is_valid(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), re.DOTALL)
        self.assertGreaterEqual(len(blocks), 1)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"observed-deploy-python-{index}", "exec")

    def test_brpc_output_directory_exists_before_workload_log(self):
        text = SCRIPT.read_text()
        workload = text.index('echo "== Run pure BRPC observed workload: $REQUESTS requests =="')
        mkdir = text.index('mkdir -p "$OUTPUT_DIR/brpc"', workload)
        run_requests = text.index(
            'run_requests "$PAIREC_URL" "$REQUESTS" "$OUTPUT_DIR/brpc" 1',
            workload,
        )
        self.assertLess(mkdir, run_requests)

    def test_pod_selection_requires_latest_running_ready_pod(self):
        text = SCRIPT.read_text()
        self.assertIn("ready_pod_for_app()", text)
        self.assertIn('status.get("phase") != "Running"', text)
        self.assertIn('all(container.get("ready") for container in containers)', text)
        self.assertIn('print(max(candidates)[1])', text)
        self.assertNotIn("jsonpath='{.items[0].metadata.name}'", text)

    def test_pod_selector_chooses_newest_ready_candidate(self):
        text = SCRIPT.read_text()
        block = re.search(
            r'ready_pod_for_app\(\) \{.*?python3 -c \'\n(.*?)\n\' "\$app"',
            text,
            re.DOTALL,
        )
        self.assertIsNotNone(block)
        fixture = {
            "items": [
                {
                    "metadata": {"name": "old-ready", "creationTimestamp": "2026-01-01T00:00:00Z"},
                    "status": {"phase": "Running", "containerStatuses": [{"ready": True}]},
                },
                {
                    "metadata": {"name": "new-not-ready", "creationTimestamp": "2026-01-03T00:00:00Z"},
                    "status": {"phase": "Running", "containerStatuses": [{"ready": False}]},
                },
                {
                    "metadata": {"name": "new-ready", "creationTimestamp": "2026-01-02T00:00:00Z"},
                    "status": {"phase": "Running", "containerStatuses": [{"ready": True}]},
                },
            ]
        }
        result = subprocess.run(
            ["python3", "-c", block.group(1), "inference-brpc-trtllm"],
            input=json.dumps(fixture),
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual("new-ready", result.stdout.strip())

    def test_strict_gate_uses_request_completion_not_rotatable_marker(self):
        text = SCRIPT.read_text()
        self.assertNotIn('datasystem_attribution_ready', text)
        self.assertIn('wait_for_native_completions()', text)
        self.assertIn('len(events) != 1', text)
        self.assertIn('event.get("attribution_complete") is not True', text)
        for field in (
            "get_failed_count", "set_failed_count", "pending_count", "unknown_count"
        ):
            self.assertIn(field, text)

    def test_warmup_completion_is_gated_before_workload(self):
        text = SCRIPT.read_text()
        warmup = text.index(
            'run_requests "$PAIREC_URL" "$WARMUP_REQUESTS" "$OUTPUT_DIR/warmup" 1'
        )
        gate = text.index('"$OUTPUT_DIR/warmup/inference.log" warmup', warmup)
        workload = text.index('echo "== Run pure BRPC observed workload', gate)
        self.assertLess(warmup, gate)
        self.assertLess(gate, workload)

    def test_formal_workload_requires_effective_info_log_level(self):
        text = SCRIPT.read_text()
        self.assertIn("TLLM_LOG_LEVEL=INFO", text)
        debug_gate = text.index("TLLM_LOG_LEVEL=INFO is not effective")
        workload = text.index('echo "== Run pure BRPC observed workload', debug_gate)
        self.assertIn("'[TensorRT-LLM][DEBUG]'", text)
        self.assertLess(debug_gate, workload)

    def test_optional_inference_restart_happens_after_env_update(self):
        text = SCRIPT.read_text()
        env_update = text.index('kubectl -n "$NAMESPACE" set env')
        restart = text.index('kubectl -n "$NAMESPACE" rollout restart', env_update)
        rollout = text.index('kubectl -n "$NAMESPACE" rollout status', restart)
        self.assertIn('FORCE_INFERENCE_RESTART="${FORCE_INFERENCE_RESTART:-0}"', text)
        self.assertLess(env_update, restart)
        self.assertLess(restart, rollout)

    def test_pairec_restart_can_be_skipped_for_workload_only_recovery(self):
        text = SCRIPT.read_text()
        self.assertIn('FORCE_PAIREC_RESTART="${FORCE_PAIREC_RESTART:-1}"', text)
        self.assertIn('if [[ "$FORCE_PAIREC_RESTART" = 1 ]]', text)
        restart = text.index(
            'kubectl -n "$NAMESPACE" rollout restart deployment/pairec-brpc-observed'
        )
        condition = text.rindex(
            'if [[ "$FORCE_PAIREC_RESTART" = 1 ]]', 0, restart
        )
        rollout = text.index(
            'kubectl -n "$NAMESPACE" rollout status deployment/pairec-brpc-observed',
            restart,
        )
        self.assertLess(condition, restart)
        self.assertLess(restart, rollout)

    def test_workload_logs_are_streamed_before_requests(self):
        text = SCRIPT.read_text()
        workload = text.index('echo "== Run pure BRPC observed workload')
        pairec_follow = text.index('--since-time="$since" -f', workload)
        inference_follow = text.index(
            '--since-time="$since" -f >"$OUTPUT_DIR/brpc/inference.log"',
            pairec_follow,
        )
        requests = text.index(
            'run_requests "$PAIREC_URL" "$REQUESTS" "$OUTPUT_DIR/brpc" 1',
            inference_follow,
        )
        completion_gate = text.index(
            '"$OUTPUT_DIR/brpc/inference.log" workload 0', requests
        )
        self.assertLess(pairec_follow, requests)
        self.assertLess(inference_follow, requests)
        self.assertLess(requests, completion_gate)

    def test_workload_throughput_excludes_completion_wait(self):
        text = SCRIPT.read_text()
        start = text.index('workload_start_ns="$(date +%s%N)"')
        requests = text.index(
            'run_requests "$PAIREC_URL" "$REQUESTS" "$OUTPUT_DIR/brpc" 1', start
        )
        end = text.index('workload_end_ns="$(date +%s%N)"', requests)
        completion = text.index('wait_for_native_completions', end)
        self.assertIn('"$OUTPUT_DIR/brpc/workload.json"', text[end:completion])
        self.assertLess(start, requests)
        self.assertLess(requests, end)
        self.assertLess(end, completion)

    def test_log_collectors_are_cleaned_on_exit(self):
        text = SCRIPT.read_text()
        self.assertIn('trap cleanup EXIT', text)
        self.assertIn('stop_log_collector "$PAIREC_LOG_PID"', text)
        self.assertIn('stop_log_collector "$INFERENCE_LOG_PID"', text)


if __name__ == "__main__":
    unittest.main()
