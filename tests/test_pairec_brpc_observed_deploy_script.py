import pathlib
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


if __name__ == "__main__":
    unittest.main()
