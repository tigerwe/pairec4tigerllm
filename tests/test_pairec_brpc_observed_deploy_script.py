import json
import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_and_validate_pairec_brpc_observed.sh"


class PaiRecBrpcObservedDeployScriptTest(unittest.TestCase):
    def test_brpc_output_directory_exists_before_strict_startup_log(self):
        text = SCRIPT.read_text()
        workload = text.index('echo "== Run pure BRPC observed workload: $REQUESTS requests =="')
        mkdir = text.index('mkdir -p "$OUTPUT_DIR/brpc"', workload)
        startup_log = text.index('>"$OUTPUT_DIR/brpc/inference-startup.log"', workload)
        run_requests = text.index(
            'run_requests "$PAIREC_URL" "$REQUESTS" "$OUTPUT_DIR/brpc" 1',
            workload,
        )
        self.assertLess(mkdir, startup_log)
        self.assertLess(startup_log, run_requests)

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

    def test_attribution_marker_failure_names_selected_pod(self):
        text = SCRIPT.read_text()
        self.assertIn(
            'capability marker is missing from ready pod $INFERENCE_POD',
            text,
        )


if __name__ == "__main__":
    unittest.main()
